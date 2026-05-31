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

    # Merge the FULL market-structure enrichment (broker risk ratios, analyst
    # ratings, raw valuation multiples, company age, instrument type) into the
    # final data block so the strategy/portfolio layer reads them as columns —
    # rather than calling load_market_structure() live at decision time. This keeps
    # all derivation in the ETL phase (single enriched block; pure-consumer engine).
    # Best-effort: failures leave the columns null.
    if refresh and not result.empty:
        try:
            from .market_structure import (
                MARKET_STRUCTURE_DF_COLS,
                load_market_structure_df,
            )
            syms = [str(s) for s in result["symbol"].tolist()]
            ms_df = load_market_structure_df(syms, auto_refresh=True)
            # Don't clobber columns fundamentals already provides (e.g. market_cap,
            # pe_ratio) — only add market-structure cols absent from the metrics block.
            add_cols = [c for c in MARKET_STRUCTURE_DF_COLS if c not in result.columns]
            if add_cols and not ms_df.empty:
                result = result.merge(
                    ms_df[["symbol", *add_cols]], on="symbol", how="left"
                )
        except Exception as e:
            logger.warning("market-structure enrichment skipped: %s", e)

    # Build the news co-mention graph artifact from the freshly-fetched news, then
    # derive per-node structural features and merge them into the enriched block as
    # columns (pure-consumer engine reads them; never builds a graph at decision time).
    if refresh and news_df is not None and not news_df.empty:
        result = _enrich_with_graph_features(result, news_df, held)

    if not result.empty:
        store_data_as_csv("agg_data", "", result)
        time.sleep(1)

    return result


def _enrich_with_graph_features(result, news_df, held):
    """Build the co-mention graph artifact + per-node features and left-merge the
    GRAPH_FEATURE_COLS into ``result``. Best-effort: any failure leaves the graph
    columns absent and returns ``result`` unchanged."""
    try:
        from .comention_graph import (
            GRAPH_FEATURE_COLS,
            build_and_persist_features,
            build_comention_graph,
        )
        build_comention_graph(news_df=news_df, held_symbols=held or None)
        feats = build_and_persist_features()
        if not result.empty and feats is not None and not feats.empty:
            add_cols = [c for c in GRAPH_FEATURE_COLS if c not in result.columns]
            if add_cols:
                result = result.merge(
                    feats[["symbol", *add_cols]], on="symbol", how="left"
                )
    except Exception as e:
        logger.warning("co-mention graph build skipped: %s", e)
    return result


