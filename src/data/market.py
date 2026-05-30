"""
data/market.py — MarketDataProvider and get_data() public entry point.

get_data() orchestrates: universe → fundamentals → news → merge → persist.
MarketDataProvider handles raw price downloads.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import robin_stocks.robinhood as rb

from util import store_data_as_csv

from .fundamentals import get_fundamentals_df
from .news import get_news_df
from .universe import gen_symbols_list

logger = logging.getLogger(__name__)


def get_data(refresh: bool = False) -> pd.DataFrame:
    """
    Full data pipeline: build universe, fetch fundamentals, fetch news, merge.

    refresh=True forces re-fetch from Robinhood + yfinance.
    refresh=False returns cached CSV data.
    """
    held: set[str] = set()
    if refresh:
        try:
            held = set(rb.build_holdings() or {})
        except Exception as e:
            logger.warning("Could not fetch portfolio holdings for universe expansion: %s", e)

    tickers  = gen_symbols_list(refresh, extra_symbols=held or None)
    metrics  = get_fundamentals_df(tickers, refresh, portfolio_symbols=held or None)
    news_df  = get_news_df(tickers, refresh)

    if metrics is None or metrics.empty:
        print("Warning: No fundamental data available")
        return pd.DataFrame()

    if news_df is not None and not news_df.empty:
        result = metrics.merge(news_df, on="symbol", how="left")
    else:
        result = metrics.copy()

    # Tag each symbol with its Robinhood instrument type (etp/cef/mlp/stock/...)
    # so downstream ETF/fund classification covers the full universe, not just
    # the configured ETF tickers. Best-effort: failures leave the column null.
    if refresh and not result.empty:
        try:
            from .market_structure import load_market_structure
            syms = [str(s) for s in result["symbol"].tolist()]
            ms = load_market_structure(syms, auto_refresh=True)
            result["instrument_type"] = result["symbol"].astype(str).map(
                lambda s: (ms.get(s) or {}).get("instrument_type")
            )
        except Exception as e:
            logger.warning("instrument_type enrichment skipped: %s", e)

    if not result.empty:
        store_data_as_csv("agg_data", "", result)
        time.sleep(1)

    return result


