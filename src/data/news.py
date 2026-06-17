"""
data/news.py — News and Reddit sentiment fetching.

get_news_df() and get_news_for_tickers_by_symbol() are the canonical
pipeline functions (moved from source_data._get_news and sentiments.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

import pandas as pd
import robin_stocks.robinhood as rb
import yfinance as yf

from core.utils import run_async
from data.cache import read_data_as_pd, read_recent_data_as_pd, store_data_as_csv

logger = logging.getLogger(__name__)

# yfinance logs at ERROR level for rate-limited/empty tickers but returns [] safely.
# Silence to reduce log noise.
logging.getLogger("yfinance").addFilter(
    type("_NoNewsFilter", (logging.Filter,), {
        "filter": staticmethod(lambda r: "faulty response" not in r.getMessage())
    })()
)

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

# Reuse an existing same-symbol news scrape if it is younger than this many hours,
# rather than re-running the (slow) full fetch. Headlines move slowly intraday.
NEWS_MAX_AGE_HOURS = 8.0


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

def _enrich_news_with_social(news_df: pd.DataFrame) -> pd.DataFrame:
    """ALWAYS merge normalized Reddit/X social items into the news frame (same
    ["symbol","news"] schema) so the active-sleeve sentiment substrate carries social
    provenance — independent of options_social.enabled (which only gates the 0DTE report).

    Best-effort and fail-closed: any error (or missing network) returns the frame unchanged,
    so social enrichment can never break the news pipeline. Operators can opt out by setting
    options_social.disable_social_news_enrichment: true."""
    try:
        from util import OPTIONS_SOCIAL_PARAMS
        if OPTIONS_SOCIAL_PARAMS.get("disable_social_news_enrichment", False):
            logger.info("social news enrichment disabled (disable_social_news_enrichment).")
            return news_df
        from data.social_sentiment import enrich_news_with_social
        merged = enrich_news_with_social(news_df=news_df, persist=False)
        logger.info("News enriched with social items (Reddit/X provenance).")
        return merged
    except Exception as exc:
        logger.warning("social news enrichment skipped: %s", exc)
        return news_df


def _is_valid_news_scrape(df: pd.DataFrame | None) -> bool:
    """A reusable scrape has rows and at least one symbol with real articles.

    Guards against reusing a botched same-day run that wrote an all-empty
    ("[]") news frame (e.g. a fully rate-limited fetch)."""
    if df is None or df.empty or "news" not in df.columns:
        return False
    return bool((df["news"].astype(str).str.len() > 2).any())


def _skip_fetch_news() -> bool:
    """True when the operator passed --skip-fetch-news.

    Honored via either the env var set by the `daily-investor` CLI or a raw
    sys.argv scan (mirrors how --skip-data is read by the legacy main.py entry),
    so both invocation paths work."""
    if os.environ.get("SKIP_FETCH_NEWS", "").strip() in ("1", "true", "True"):
        return True
    return "--skip-fetch-news" in sys.argv


def get_news_df(tickers: list[str], force_refresh: bool) -> pd.DataFrame | None:
    if not force_refresh:
        return read_data_as_pd("news")

    # --skip-fetch-news: reuse the most-recent cached scrape regardless of age and
    # skip the fetch entirely, even on a fresh-data run. Falls through to a normal
    # fetch only if no valid cached scrape exists (nothing to reuse).
    if _skip_fetch_news():
        cached = read_data_as_pd("news")
        if _is_valid_news_scrape(cached):
            print(
                f"News: --skip-fetch-news → reusing latest cached scrape "
                f"({len(cached)} symbols), skipping fetch."
            )
            logger.info("--skip-fetch-news: reusing latest cached news (%d symbols).", len(cached))
            return cached
        print("News: --skip-fetch-news set but no valid cached scrape found — fetching.")
        logger.warning("--skip-fetch-news set but no valid cached news exists; fetching.")

    # The full news scrape is by far the slowest stage (thousands of tickers, 3
    # concurrent threads). News changes little intraday, so if a valid scrape
    # exists within the freshness window, reuse it instead of refetching. Set
    # NEWS_FORCE_REFETCH=1 to override (e.g. to pick up breaking headlines).
    # 0DTE options sentiment is fetched separately and is NOT cached this way.
    if os.environ.get("NEWS_FORCE_REFETCH", "").strip() not in ("1", "true", "True"):
        cached = read_recent_data_as_pd("news", NEWS_MAX_AGE_HOURS)
        if _is_valid_news_scrape(cached):
            print(
                f"News: reusing recent cached scrape ({len(cached)} symbols, "
                f"<{NEWS_MAX_AGE_HOURS:g}h old) — skipping fetch. "
                "Set NEWS_FORCE_REFETCH=1 to force."
            )
            logger.info("Reusing recent news scrape (%d symbols).", len(cached))
            return cached

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
    news_df = _enrich_news_with_social(news_df)
    store_data_as_csv("news", ["symbol", "news"], news_df)
    return read_data_as_pd("news")


