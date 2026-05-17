"""
sentiments.py — News and Reddit sentiment data collection.

Key fixes vs original:
  - News processing loop bug fixed (items were processed outside the ticker loop)
  - Retry/backoff extracted into _fetch_news_with_retry — was copy-pasted 4×
  - Volume pre-filter removed (agg_data CSV already screens by volume upstream)
  - Robinhood login check simplified to a single try/except on the real call
  - reddit_sentiments_for_tickers kept unchanged (was already clean)
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import robin_stocks.robinhood as rb
import yfinance as yf

from util import run_async

# yfinance logs "Failed to retrieve the news and received faulty response instead."
# at ERROR level for rate-limited or empty-result tickers, but returns [] (no exception).
# Our retry logic handles the empty case; silence these to reduce log noise.
logging.getLogger("yfinance").addFilter(
    type("_NoNewsFilter", (logging.Filter,), {
        "filter": staticmethod(lambda r: "faulty response" not in r.getMessage())
    })()
)


# ---------------------------------------------------------------------------
# Reddit sentiment
# ---------------------------------------------------------------------------

async def _fetch_reddit_date(session: aiohttp.ClientSession, date: str) -> tuple[str, Any]:
    try:
        async with session.get(f"https://api.tradestie.com/v1/apps/reddit?date={date}") as r:
            r.raise_for_status()
            return date, await r.json()
    except Exception as e:
        print(f"Reddit fetch error for {date}: {e}")
        return date, None


async def _get_reddit_sentiments_async(days: int = 7) -> dict:
    dates = [
        (datetime.now() - timedelta(days=i)).strftime("%m-%d-%Y")
        for i in range(min(days, 7))
    ]
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_fetch_reddit_date(session, d) for d in dates])
    return dict(results)


def reddit_sentiments_for_tickers(tickers: list[str], days: int = 7) -> dict:
    """Return {date: [ticker_data]} for the given tickers, deduplicated per day."""
    ticker_set = set(tickers)
    raw = asyncio.run(_get_reddit_sentiments_async(days))
    rv: dict[str, list] = {}
    for day, sentiment_list in raw.items():
        if not sentiment_list:
            continue
        seen: set[str] = set()
        for item in sentiment_list:
            ticker = item.get("ticker")
            if ticker in ticker_set and ticker not in seen:
                seen.add(ticker)
                rv.setdefault(day, []).append(item)
    return rv


# ---------------------------------------------------------------------------
# News — shared helpers
# ---------------------------------------------------------------------------

def _robinhood_news(ticker: str, max_articles: int) -> list[dict]:
    """Fetch news from Robinhood. Returns [] if not logged in or on any error."""
    try:
        items = rb.robinhood.get_news(ticker) or []
        result = []
        for item in items[:max_articles]:
            result.append({
                "title": item.get("title", ""),
                "publisher": item.get("source", {}).get("name", "Robinhood"),
                "link": item.get("url", ""),
                "summary": item.get("summary", item.get("preview_text", "")),
                "pub_date": item.get("published_at", ""),
                "formatted_date": item.get("published_at", ""),
                "api_source": "robinhood",
            })
        return result
    except Exception:
        return []


def _parse_yfinance_item(item: dict) -> dict | None:
    """Convert a raw yfinance news item to our standard format."""
    content = item.get("content") or {}
    if not content:
        return None
    pub_date = content.get("pubDate") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        formatted_date = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%m-%d-%Y")
    except Exception:
        formatted_date = datetime.utcnow().strftime("%m-%d-%Y")
    return {
        "title": content.get("title", "No title"),
        "publisher": (content.get("provider") or {}).get("displayName", "Unknown"),
        "link": (content.get("canonicalUrl") or {}).get("url", ""),
        "summary": content.get("summary", ""),
        "pub_date": pub_date,
        "formatted_date": formatted_date,
    }


def _parse_robinhood_item(item: dict) -> dict:
    """Robinhood items are already in our format — just pass through."""
    return {
        "title": item.get("title", "No title"),
        "publisher": item.get("publisher", "Robinhood"),
        "link": item.get("link", ""),
        "summary": item.get("summary", ""),
        "pub_date": item.get("pub_date", ""),
        "formatted_date": item.get("formatted_date", ""),
    }


_news_logger = logging.getLogger(__name__)


def _fetch_news_with_retry(ticker: str, max_articles: int, max_retries: int = 3) -> list[dict]:
    """
    Fetch news for a single ticker via yfinance with exponential backoff,
    falling back to Robinhood on rate-limit or persistent failure.
    """
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep((2 ** attempt) + random.uniform(0, 1))
        try:
            items = (yf.Ticker(ticker).news or [])[:max_articles]
            if items:
                parsed = [_parse_yfinance_item(i) for i in items]
                return [p for p in parsed if p is not None]
            # Empty with no exception — yfinance ate the error internally (faulty response)
            # or genuinely no news. Either way, retry before giving up.
            continue
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "rate limit" in msg or "too many requests" in msg
            is_last = attempt == max_retries - 1

            if is_rate_limit:
                # Back off before hitting Robinhood to avoid cascading rate limits
                time.sleep(random.uniform(3, 7))

            if is_rate_limit or is_last:
                rb_news = _robinhood_news(ticker, max_articles)
                if rb_news:
                    return [_parse_robinhood_item(i) for i in rb_news]
                if is_last:
                    _news_logger.debug("No news available for %s from either API", ticker)
                    return []
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

MAX_NEWS_CONCURRENT = 3  # keep below yfinance's rate limit threshold


async def _fetch_all_news_async(tickers: list[str], max_articles: int) -> dict[str, list]:
    semaphore = asyncio.Semaphore(MAX_NEWS_CONCURRENT)
    total = len(tickers)
    completed = 0

    async def _one(ticker: str) -> tuple[str, list]:
        nonlocal completed
        async with semaphore:
            await asyncio.sleep(random.uniform(0.1, 0.4))  # stagger within the semaphore window
            articles = await asyncio.to_thread(_fetch_news_with_retry, ticker, max_articles)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"News: {completed}/{total} fetched")
            return ticker, articles

    results = await asyncio.gather(*[_one(t) for t in tickers])
    return dict(results)


def get_news_for_tickers_by_symbol(
    tickers: list[str],
    max_articles: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch news for all tickers concurrently. Returns {symbol: [article, ...]}.

    No volume pre-filtering here — callers should pass an already-screened
    list (source_data.py filters by volume via agg_data fundamentals).
    """
    print(f"Fetching news for {len(tickers)} tickers ({MAX_NEWS_CONCURRENT} concurrent threads)...")
    return run_async(_fetch_all_news_async(tickers, max_articles))