# source_data.py — compatibility shim. Import from data.* instead.
from data.universe import gen_symbols_list, _is_valid_ticker
from data.fundamentals import (
    _position_52w,
    _evaluate_stock,
    _diagnose_stock_filter,
    _compute_reliability_scores,
    _enrich_with_quotes,
    _enrich_with_momentum,
    _get_buy_to_sell_ratio,
    _get_earnings_bonus,
    get_fundamentals_df as _get_robinhood_fundamentals,
    get_momentum_score,
)
from data.news import get_news_df as _get_news
from data.market import get_data

# read_data_as_pd is from util (candidate_diagnostics.py imports it via source_data)
from util import read_data_as_pd, store_data_as_csv

__all__ = [
    "gen_symbols_list",
    "_is_valid_ticker",
    "_position_52w",
    "_evaluate_stock",
    "_diagnose_stock_filter",
    "_compute_reliability_scores",
    "_enrich_with_quotes",
    "_enrich_with_momentum",
    "_get_buy_to_sell_ratio",
    "_get_earnings_bonus",
    "_get_robinhood_fundamentals",
    "get_momentum_score",
    "_get_news",
    "get_data",
    "read_data_as_pd",
    "store_data_as_csv",
]
