"""
data/news.py — News and Reddit sentiment fetching.

get_news_df() and get_news_for_tickers_by_symbol() are the canonical
pipeline functions (moved from source_data._get_news and sentiments.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd
import robin_stocks.robinhood as rb
import yfinance as yf

from core.utils import run_async
from data.cache import read_data_as_pd, store_data_as_csv

logger = logging.getLogger(__name__)

# yfinance logs at ERROR level for rate-limited/empty tickers but returns [] safely.
# Silence to reduce log noise.
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


# ---------------------------------------------------------------------------
# News — shared helpers
# ---------------------------------------------------------------------------

def _robinhood_news(ticker: str, max_articles: int) -> list[dict]:
    """Fetch news from Robinhood. Returns [] if not logged in or on any error.

    Captures the structured ``related_instruments`` field (instrument UUIDs of other
    securities the article links) and resolves it to ``related_symbols`` via the
    cached data-layer resolver — the source for the co-mention graph.
    """
    try:
        from .instrument_resolver import resolve_symbols
        items = rb.get_news(ticker) or []
        result = []
        for item in items[:max_articles]:
            rel_ids = item.get("related_instruments") or []
            rel_map = resolve_symbols(rel_ids) if rel_ids else {}
            related = sorted({s for s in rel_map.values() if s and s != ticker})
            # `source` may be a plain string (e.g. "MarketWatch") or a dict.
            src = item.get("source")
            publisher = src.get("name", "Robinhood") if isinstance(src, dict) else (src or "Robinhood")
            result.append({
                "title": item.get("title", ""),
                "publisher": publisher,
                "link": item.get("url", ""),
                "summary": item.get("summary", item.get("preview_text", "")),
                "pub_date": item.get("published_at", ""),
                "formatted_date": item.get("published_at", ""),
                "api_source": "robinhood",
                "related_symbols": related,
            })
        return result
    except Exception:
        return []


def _parse_yfinance_item(item: dict) -> dict | None:
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
        "related_symbols": [],  # yfinance payload carries no structured related tickers
    }


def _parse_robinhood_item(item: dict) -> dict:
    return {
        "title": item.get("title", "No title"),
        "publisher": item.get("publisher", "Robinhood"),
        "link": item.get("link", ""),
        "summary": item.get("summary", ""),
        "pub_date": item.get("pub_date", ""),
        "formatted_date": item.get("formatted_date", ""),
        "related_symbols": item.get("related_symbols", []),
    }


def _fetch_news_with_retry(ticker: str, max_articles: int, max_retries: int = 3) -> list[dict]:
    """Fetch news for a single ticker via yfinance with exponential backoff."""
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep((2 ** attempt) + random.uniform(0, 1))
        try:
            items = (yf.Ticker(ticker).news or [])[:max_articles]
            if items:
                parsed = [_parse_yfinance_item(i) for i in items]
                return [p for p in parsed if p is not None]
            continue
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "rate limit" in msg or "too many requests" in msg
            is_last = attempt == max_retries - 1

            if is_rate_limit:
                time.sleep(random.uniform(3, 7))

            if is_rate_limit or is_last:
                rb_news = _robinhood_news(ticker, max_articles)
                if rb_news:
                    return [_parse_robinhood_item(i) for i in rb_news]
                if is_last:
                    logger.debug("No news available for %s from either API", ticker)
                    return []
    return []


# ---------------------------------------------------------------------------
# Async news fetching
# ---------------------------------------------------------------------------

MAX_NEWS_CONCURRENT = 3


async def _fetch_all_news_async(tickers: list[str], max_articles: int) -> dict[str, list]:
    semaphore = asyncio.Semaphore(MAX_NEWS_CONCURRENT)
    total = len(tickers)
    completed = 0

    async def _one(ticker: str) -> tuple[str, list]:
        nonlocal completed
        async with semaphore:
            await asyncio.sleep(random.uniform(0.1, 0.4))
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
    """Fetch news for all tickers concurrently. Returns {symbol: [article, ...]}."""
    print(f"Fetching news for {len(tickers)} tickers ({MAX_NEWS_CONCURRENT} concurrent threads)...")
    return run_async(_fetch_all_news_async(tickers, max_articles))


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def get_news_df(tickers: list[str], force_refresh: bool) -> pd.DataFrame | None:
    if not force_refresh:
        return read_data_as_pd("news")

    rb_data = read_data_as_pd("robinhood_data")
    if rb_data is not None and not rb_data.empty and "volume" in rb_data.columns:
        liquid = rb_data[rb_data["volume"] >= 50_000]["symbol"].tolist()
        print(f"News filter: {len(tickers)} total → {len(liquid)} liquid tickers")
    else:
        liquid = tickers

    news_by_symbol = get_news_for_tickers_by_symbol(liquid, max_articles=3)

    for t in tickers:
        news_by_symbol.setdefault(t, [])

    news_df = pd.DataFrame([
        {"symbol": sym, "news": json.dumps(articles)}
        for sym, articles in news_by_symbol.items()
    ])
    store_data_as_csv("news", ["symbol", "news"], news_df)
    return read_data_as_pd("news")


