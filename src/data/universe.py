"""
data/universe.py — Stock universe generation.

gen_symbols_list() is the canonical implementation. UniverseBuilder wraps it.
"""

from __future__ import annotations

import logging
import re
import time

import requests
import robin_stocks.robinhood as rb
from bs4 import BeautifulSoup

from util import read_data_as_pd, store_data_as_csv

logger = logging.getLogger(__name__)

_INDEX_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/Nasdaq-100",
    "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/Russell_2000_Index",
]

_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

_ROBINHOOD_TAGS = [
    "100-most-popular",
    "upcoming-earnings",
    "new-on-robinhood",
    "technology",
    "finance",
    "banking",
    "insurance",
    "healthcare",
    "energy",
    "oil-and-gas",
    "manufacturing",
    "utilities",
    "real-estate",
    "telecommunications",
    "retail",
    "automotive",
    "aerospace",
    "defense",
    "social-media",
]
_ROBINHOOD_INSTRUMENTS_URL = "https://api.robinhood.com/instruments/"
_ROBINHOOD_ACTIVE_STOCK_TYPES = frozenset({"stock", "adr"})
_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
_TICKER_HEADER_KEYWORDS = frozenset({"symbol", "ticker"})



def _is_valid_ticker(symbol: str) -> bool:
    return bool(symbol and isinstance(symbol, str) and _VALID_TICKER_RE.match(symbol))


def _scrape_wikipedia_tickers(url: str, retries: int = 3, base_delay: float = 5.0) -> set[str]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_WIKI_HEADERS, timeout=20)
            soup = BeautifulSoup(resp.content, "html.parser")
            table = None
            for _t in soup.find_all("table", class_="wikitable"):
                _rows = _t.find_all("tr")
                if not _rows:
                    continue
                _hdr = _rows[0].find(["th", "td"])
                if _hdr and any(kw in _hdr.get_text().strip().lower() for kw in _TICKER_HEADER_KEYWORDS):
                    table = _t
                    break
            if not table:
                return set()
            symbols: set[str] = set()
            for row in table.find_all("tr")[1:]:
                for cell in row.find_all("td"):
                    text = cell.text.strip()
                    if _is_valid_ticker(text):
                        symbols.add(text)
            return symbols
        except Exception as e:
            if attempt < retries:
                wait = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Wikipedia scrape attempt %d/%d failed for %s: %s — retrying in %.0fs",
                    attempt, retries, url, e, wait,
                )
                time.sleep(wait)
            else:
                logger.warning("Wikipedia scrape failed after %d attempts for %s: %s", retries, url, e)
    return set()


def _fetch_tag_with_retry(tag: str, retries: int = 3, base_delay: float = 2.0) -> list | None:
    for attempt in range(1, retries + 1):
        try:
            result = rb.get_all_stocks_from_market_tag(tag)
            return result or None
        except Exception as e:
            err = str(e)
            if attempt < retries:
                wait = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Tag '%s' attempt %d/%d failed: %s — retrying in %.0fs",
                    tag, attempt, retries, err[:60], wait,
                )
                time.sleep(wait)
            else:
                logger.warning("Tag '%s' failed after %d attempts: %s", tag, retries, err[:80])
    return None


def _is_robinhood_active_stock(item: dict) -> bool:
    """Return True for active Robinhood stock-like instruments.

    Robinhood's instruments catalog includes ETFs/ETPs, warrants, units,
    preferreds, inactive listings, test rows, and other wrappers. The active
    stock-picking universe should only be expanded with common stocks and ADRs;
    fund/wrapper exposure remains routed through the dedicated ETF/index sleeves.
    """
    return (
        item.get("state") == "active"
        and item.get("tradeable") is True
        and item.get("rhs_tradability") == "tradable"
        and item.get("type") in _ROBINHOOD_ACTIVE_STOCK_TYPES
        and _is_valid_ticker(str(item.get("symbol", "")))
    )


def _fetch_robinhood_instrument_symbols(
    retries: int = 3,
    base_delay: float = 2.0,
) -> set[str]:
    """Fetch all active tradable stock/ADR symbols from Robinhood instruments.

    This paginated endpoint is the broad Robinhood catalog. It is intentionally
    best-effort: if Robinhood is unavailable, the universe still falls back to
    Wikipedia indexes + Robinhood tags rather than failing the whole ETL run.
    """
    for attempt in range(1, retries + 1):
        try:
            session = requests.Session()
            symbols: set[str] = set()
            url: str | None = _ROBINHOOD_INSTRUMENTS_URL
            pages = 0
            while url:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                pages += 1
                for item in data.get("results", []):
                    if _is_robinhood_active_stock(item):
                        symbols.add(str(item["symbol"]))
                url = data.get("next")
            logger.info(
                "Robinhood instruments catalog: %d active tradable stocks/ADRs from %d pages",
                len(symbols), pages,
            )
            return symbols
        except Exception as e:
            err = str(e)
            if attempt < retries:
                wait = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Robinhood instruments attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt, retries, err[:80], wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Robinhood instruments failed after %d attempts: %s",
                    retries, err[:120],
                )
    return set()


def gen_symbols_list(
    force_refresh: bool = False,
    extra_symbols: set[str] | None = None,
) -> list[str]:
    if not force_refresh:
        cached = read_data_as_pd("stock_tickers")
        if cached is not None and not cached.empty and "symbol" in cached.columns:
            base = set(cached["symbol"].tolist())
            if extra_symbols:
                added = [s for s in sorted(extra_symbols) if _is_valid_ticker(s) and s not in base]
                if added:
                    logger.info(
                        "Supplementing cached universe with %d portfolio holdings: %s",
                        len(added), added,
                    )
                base.update(s for s in extra_symbols if _is_valid_ticker(s))
            return sorted(base)

    all_symbols: set[str] = set()
    for url in _INDEX_URLS:
        print(f"Scraping {url}")
        all_symbols.update(_scrape_wikipedia_tickers(url))

    instrument_symbols = _fetch_robinhood_instrument_symbols()
    if instrument_symbols:
        before = len(all_symbols)
        all_symbols.update(instrument_symbols)
        print(
            "Robinhood instruments: "
            f"{len(instrument_symbols)} active tradable stocks/ADRs "
            f"({len(all_symbols) - before} new)"
        )

    rb_sources: list = []
    for fn, args, label in [
        (rb.get_top_movers_sp500, ("down",), "top_movers_sp500(down)"),
        (rb.get_top_movers,       (),         "top_movers"),
        (rb.get_top_100,          (),         "top_100"),
        (rb.get_top_movers_sp500, ("up",),    "top_movers_sp500(up)"),
    ]:
        try:
            result = fn(*args)
            if result:
                rb_sources.append(result)
        except Exception as e:
            print(f"  {label} failed: {str(e)[:60]}")
        time.sleep(0.5)

    for tag in _ROBINHOOD_TAGS:
        _tag_result = _fetch_tag_with_retry(tag)
        if _tag_result is not None:
            rb_sources.append(_tag_result)
            print(f"  Tag '{tag}': {len(_tag_result)} stocks")
        time.sleep(1.0)

    invalid = 0
    for source in rb_sources:
        for item in (source or []):
            sym = item.get("symbol", "")
            if _is_valid_ticker(sym):
                all_symbols.add(sym)
            else:
                invalid += 1

    if extra_symbols:
        added = [s for s in sorted(extra_symbols) if _is_valid_ticker(s) and s not in all_symbols]
        if added:
            logger.info(
                "Adding %d portfolio holdings to refreshed universe: %s", len(added), added,
            )
        all_symbols.update(s for s in extra_symbols if _is_valid_ticker(s))

    print(f"Universe: {len(all_symbols)} valid tickers ({invalid} invalid skipped)")
    store_data_as_csv("stock_tickers", ["symbol"], [[s] for s in sorted(all_symbols)])
    return sorted(all_symbols)


