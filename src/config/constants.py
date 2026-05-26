"""
config/constants.py — Domain-grouped constants for new code.

Prefer importing from here over importing from util.py directly.
Everything here is a thin re-export of util.py values so existing callers
continue to work while new code uses a stable, domain-labelled interface.

Migration target: once util.py is retired, update these to read directly
from ConfigManager without the legacy flat-constant API.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from util import ROOT_DIR, CFG_DIRECTORY, DATA_DIRECTORY, CONFIG_FILE, RATIOS_FILE

# ---------------------------------------------------------------------------
# Universe / instrument filters
# ---------------------------------------------------------------------------

from util import ETFS, EXCLUDED_STOCK_INDUSTRIES, EXCLUDED_STOCK_SECTORS

# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

from util import SCORE_WEIGHTS

# ---------------------------------------------------------------------------
# Scoring / valuation
# ---------------------------------------------------------------------------

from util import (
    SCORING_PARAMS,
    VALUATION_GUARDRAILS,
    VALUE_V2_PARAMS,
    METRIC_THRESHOLD,
    DIVIDEND_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

from util import (
    RISK_LIMITS,
    INDEX_PCT,
    WEEKLY_INVESTMENT,
    MAX_ITERATIONS,
)

# ---------------------------------------------------------------------------
# Sell rules
# ---------------------------------------------------------------------------

from util import SELL_RULES

# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

from util import CANDIDATE_SELECTION_PARAMS

# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

from util import MOMENTUM_PARAMS, MOMENTUM_V2_PARAMS

# ---------------------------------------------------------------------------
# Backtest / tuning
# ---------------------------------------------------------------------------

from util import BACKTEST_PARAMS, TUNING_PARAMS, STABILITY_PARAMS

# ---------------------------------------------------------------------------
# Other operational params
# ---------------------------------------------------------------------------

from util import (
    HARVEST_PARAMS,
    SNAPSHOT_PARAMS,
    REGIME_PARAMS,
    ETF_RISK_PARAMS,
    BEAR_MARKET_PARAMS,
    ANALYST_PARAMS,
    RELIABILITY_PARAMS,
    DIVIDEND_PARAMS,
    EARNINGS_PARAMS,
    CANDIDATE_ROTATION_PARAMS,
)

# ---------------------------------------------------------------------------
# Data schema helpers
# ---------------------------------------------------------------------------

from util import METRIC_KEYS, AGG_DATA_COLUMNS

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

from util import safe_float, run_async, store_data_as_csv, read_data_as_pd
