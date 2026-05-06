"""
util.py — Configuration, shared constants, and I/O helpers.

Single source of truth for:
  - All YAML-driven config values
  - DATA_DIRECTORY path
  - METRIC_KEYS (canonical schema for agg_data rows)
  - safe_float
  - CSV read/write helpers
  - Finviz valuation updater
"""

import asyncio
import concurrent.futures
import csv
import datetime
import logging
import os
import random
import re

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIRECTORY  = os.path.join(ROOT_DIR, "cfg")
DATA_DIRECTORY = os.path.join(ROOT_DIR, "data")
CONFIG_FILE    = os.path.join(CFG_DIRECTORY, "config.yaml")
RATIOS_FILE    = os.path.join(CFG_DIRECTORY, "ratios.yaml")

os.makedirs(DATA_DIRECTORY, exist_ok=True)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def _load_ratios() -> dict:
    with open(RATIOS_FILE, "r") as f:
        return yaml.safe_load(f)


_app    = _load_config()
_ratios = _load_ratios()

# ---------------------------------------------------------------------------
# Public config constants
# ---------------------------------------------------------------------------

IGNORE_NEGATIVE_PE:   bool  = _app.get("ignore_negative_pe", False)
IGNORE_NEGATIVE_PB:   bool  = _app.get("ignore_negative_pb", False)
DIVIDEND_THRESHOLD:   float = float(_app.get("dividend_threshold", 0.03))
METRIC_THRESHOLD:     float = float(_app.get("metric_threshold", 0.8))
SELLOFF_THRESHOLD:    float = float(_app.get("selloff_threshold", 30))
WEEKLY_INVESTMENT:    float = float(_app.get("weekly_investment", 400))
INDEX_PCT:            float = float(_app.get("index_pct", 0.85))
AUTO_APPROVE:         bool  = _app.get("auto_approve", False)
USE_SENTIMENT_ANALYSIS: bool = _app.get("use_sentiment_analysis", False)
CONFIDENCE_THRESHOLD: float = float(_app.get("confidence_threshold", 70))
ETFS:                 list  = _app.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"])

# ---------------------------------------------------------------------------
# Score weights — weights must sum to 1.0; fall back to defaults otherwise
# ---------------------------------------------------------------------------

_sw = _app.get("score_weights", {})
_sw_v = float(_sw.get("value",    0.45))
_sw_q = float(_sw.get("quality",  0.25))
_sw_i = float(_sw.get("income",   0.15))
_sw_m = float(_sw.get("momentum", 0.15))
if abs((_sw_v + _sw_q + _sw_i + _sw_m) - 1.0) > 0.01:
    logger.warning(
        f"score_weights sum to {_sw_v+_sw_q+_sw_i+_sw_m:.3f} (not 1.0) — using defaults"
    )
    _sw_v, _sw_q, _sw_i, _sw_m = 0.45, 0.25, 0.15, 0.15

SCORE_WEIGHTS: dict = {
    "value":    _sw_v,
    "quality":  _sw_q,
    "income":   _sw_i,
    "momentum": _sw_m,
}

# ---------------------------------------------------------------------------
# Valuation guardrails
# ---------------------------------------------------------------------------

_vg = _app.get("valuation_guardrails", {})
MAX_PE_COMPONENT: float = float(_vg.get("max_pe_component", 5.0))
MAX_PB_COMPONENT: float = float(_vg.get("max_pb_component", 5.0))
MIN_PE_RATIO:     float = float(_vg.get("min_pe_ratio",     1.0))
MIN_PB_RATIO:     float = float(_vg.get("min_pb_ratio",     0.1))

VALUATION_GUARDRAILS: dict = {
    "max_pe_component": MAX_PE_COMPONENT,
    "max_pb_component": MAX_PB_COMPONENT,
    "min_pe_ratio":     MIN_PE_RATIO,
    "min_pb_ratio":     MIN_PB_RATIO,
}

# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

_rl = _app.get("risk", {})
RISK_LIMITS: dict = {
    "max_single_position_pct":              float(_rl.get("max_single_position_pct",              0.05)),
    "max_sector_pct":                       float(_rl.get("max_sector_pct",                       0.25)),
    "max_order_pct_of_cash":                float(_rl.get("max_order_pct_of_cash",                0.10)),
    "min_order_amount":                     float(_rl.get("min_order_amount",                     5.00)),
    "min_liquidity_volume":                 float(_rl.get("min_liquidity_volume",                 500_000)),
    "max_buys_per_rebalance":               int(_rl.get("max_buys_per_rebalance",                 10)),
    "max_sentiment_candidates":             int(_rl.get("max_sentiment_candidates",               20)),
    "allow_whole_share_fallback":           bool(_rl.get("allow_whole_share_fallback",            False)),
    "max_whole_share_buys_per_run":         int(_rl.get("max_whole_share_buys_per_run",           3)),
    "max_whole_share_allocation_multiplier":float(_rl.get("max_whole_share_allocation_multiplier",1.5)),
    "min_index_pct":                        float(_rl.get("min_index_pct",                        0.60)),
}

# ---------------------------------------------------------------------------
# Harvest parameters
# ---------------------------------------------------------------------------

_hv = _app.get("harvest", {})
HARVEST_PARAMS: dict = {
    "min_harvest_amount":           float(_hv.get("min_harvest_amount",           25.0)),
    "max_harvest_pct_of_portfolio": float(_hv.get("max_harvest_pct_of_portfolio",  0.02)),
    "harvest_etfs":                 list(_hv.get("harvest_etfs",                  ["SPY", "VTI"])),
}

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------

_bt = _app.get("backtest", {})
BACKTEST_PARAMS: dict = {
    "default_mode":                 str(_bt.get("default_mode",                 "liquid_universe_sanity_test")),
    "universe_selection":           str(_bt.get("universe_selection",           "liquid_sample")),
    "max_symbols":                  int(_bt.get("max_symbols",                  300)),
    "min_volume":                   float(_bt.get("min_volume",                 500_000)),
    "random_seed":                  int(_bt.get("random_seed",                  42)),
    "slippage_bps":                 float(_bt.get("slippage_bps",               10.0)),
    "commission_per_trade":         float(_bt.get("commission_per_trade",       0.0)),
    "train_pct":                    float(_bt.get("train_pct",                  0.70)),
    "benchmark_symbol":             str(_bt.get("benchmark_symbol",             "SPY")),
    "starting_capital":             float(_bt.get("starting_capital",           5_000.0)),
    "weekly_contribution":          float(_bt.get("weekly_contribution",        400.0)),
    "rebalance_frequency_days":     int(_bt.get("rebalance_frequency_days",     5)),
    "deploy_initial_cash":          bool(_bt.get("deploy_initial_cash",         True)),
    "reinvest_sell_proceeds":       bool(_bt.get("reinvest_sell_proceeds",      True)),
    "use_out_of_sample_validation": bool(_bt.get("use_out_of_sample_validation",True)),
    "auto_apply_if_valid":          bool(_bt.get("auto_apply_if_valid",         False)),
    "min_validation_excess_return": float(_bt.get("min_validation_excess_return",0.0)),
    "max_validation_drawdown":      float(_bt.get("max_validation_drawdown",    -0.20)),
    "min_validation_sharpe":        float(_bt.get("min_validation_sharpe",      0.25)),
    "llm_review_enabled":           bool(_bt.get("llm_review_enabled",         False)),
    "llm_review_top_n":             int(_bt.get("llm_review_top_n",             5)),
    "llm_review_apply":             bool(_bt.get("llm_review_apply",            False)),
    "llm_review_model":             str(_bt.get("llm_review_model",             "claude-sonnet-4-6")),
}

# ---------------------------------------------------------------------------
# Sell rules
# ---------------------------------------------------------------------------

_sr = _app.get("sell_rules", {})
SELL_RULES: dict = {
    "stop_loss_pct":                       float(_sr.get("stop_loss_pct",                       -0.12)),
    "trailing_stop_pct":                   float(_sr.get("trailing_stop_pct",                   -0.15)),
    "take_profit_pct":                     float(_sr.get("take_profit_pct",                      0.35)),
    "take_profit_value_floor_multiplier":  float(_sr.get("take_profit_value_floor_multiplier",   1.20)),
    "sell_weak_value_below":               float(_sr.get("sell_weak_value_below",                0.25)),
    "sell_yield_trap":                     bool(_sr.get("sell_yield_trap",                       True)),
    "sell_low_quality_below":              float(_sr.get("sell_low_quality_below",              -0.25)),
    "min_days_held_before_value_exit":     int(_sr.get("min_days_held_before_value_exit",          7)),
}

# ---------------------------------------------------------------------------
# Bear market regime
# ---------------------------------------------------------------------------

_bm = _app.get("bear_market", {})
BEAR_MARKET_PARAMS: dict = {
    "spy_ma_period": int(_bm.get("spy_ma_period", 200)),
    "vix_threshold": float(_bm.get("vix_threshold", 25.0)),
}

# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

_sc = _app.get("scoring", {})
SCORING_PARAMS: dict = {
    "value_pe_weight":                float(_sc.get("value_pe_weight",                0.6)),
    "value_pb_weight":                float(_sc.get("value_pb_weight",                0.4)),
    "income_score_cap":               float(_sc.get("income_score_cap",               1.5)),
    "yield_trap_threshold":           float(_sc.get("yield_trap_threshold",           0.10)),
    "distress_pe_max":                float(_sc.get("distress_pe_max",                5.0)),
    "quality_volume_high":            float(_sc.get("quality_volume_high",            1_000_000)),
    "quality_volume_low":             float(_sc.get("quality_volume_low",             100_000)),
    "quality_dividend_min":           float(_sc.get("quality_dividend_min",           0.02)),
    "quality_dividend_max":           float(_sc.get("quality_dividend_max",           0.06)),
    "quality_weight_has_positive_pe": float(_sc.get("quality_weight_has_positive_pe", 0.5)),
    "quality_weight_distress_pe":     float(_sc.get("quality_weight_distress_pe",     -0.4)),
    "quality_weight_has_positive_pb": float(_sc.get("quality_weight_has_positive_pb", 0.2)),
    "quality_weight_high_volume":     float(_sc.get("quality_weight_high_volume",     0.3)),
    "quality_weight_low_volume":      float(_sc.get("quality_weight_low_volume",      -0.3)),
    "quality_weight_yield_trap":      float(_sc.get("quality_weight_yield_trap",      -0.6)),
    "quality_weight_healthy_dividend":float(_sc.get("quality_weight_healthy_dividend", 0.2)),
}

# ---------------------------------------------------------------------------
# Momentum parameters
# ---------------------------------------------------------------------------

_mo = _app.get("momentum", {})
MOMENTUM_PARAMS: dict = {
    "position_bin_boundaries":           _mo.get("position_bin_boundaries",           [0.15, 0.35, 0.75, 0.95]),
    "position_bin_scores":               _mo.get("position_bin_scores",               [-0.4, 0.1, 0.3, 0.5, 0.2]),
    "return_1m_low_position_cutoff":     float(_mo.get("return_1m_low_position_cutoff",      0.40)),
    "return_1m_recovery_threshold":      float(_mo.get("return_1m_recovery_threshold",        0.05)),
    "return_1m_falling_knife_threshold": float(_mo.get("return_1m_falling_knife_threshold",  -0.10)),
    "return_1m_recovery_bonus":          float(_mo.get("return_1m_recovery_bonus",           0.15)),
    "return_1m_falling_knife_penalty":   float(_mo.get("return_1m_falling_knife_penalty",    0.20)),
}

# ---------------------------------------------------------------------------
# Analyst ratings parameters
# ---------------------------------------------------------------------------

_ar = _app.get("analyst_ratings", {})
ANALYST_PARAMS: dict = {
    "strong_buy_ratio":      float(_ar.get("strong_buy_ratio",      5.0)),
    "net_sell_ratio":        float(_ar.get("net_sell_ratio",        1.0)),
    "strong_buy_multiplier": float(_ar.get("strong_buy_multiplier", 1.05)),
    "net_sell_multiplier":   float(_ar.get("net_sell_multiplier",   0.95)),
}

MAX_ITERATIONS: int = int(_app.get("max_iterations", 10))

# ---------------------------------------------------------------------------
# Canonical agg_data schema — single definition used by all modules
# ---------------------------------------------------------------------------

METRIC_KEYS: list[str] = [
    "industry",
    "sector",
    "volume",
    "pe_ratio",
    "pb_ratio",
    "dividend_yield",
    "current_price",
    "low_52w",
    "high_52w",
    "position_52w",
    "return_1m",
    "pe_comp",
    "pb_comp",
    "value_score",
    "income_score",
    "quality_score",
    "momentum_score",
    "yield_trap_flag",
    "value_metric",
    "buy_to_sell_ratio",
]

AGG_DATA_COLUMNS: list[str] = ["symbol"] + METRIC_KEYS

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run a coroutine from sync code, handling an already-running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """Convert value to float, returning default on failure or None/NaN input."""
    try:
        if value is None or value == "" or str(value).lower() == "nan":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# CSV / DataFrame I/O
# ---------------------------------------------------------------------------

def _dated_filename(dataset: str) -> str:
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    return os.path.join(DATA_DIRECTORY, f"{dataset}_{date_str}.csv")


def store_data_as_csv(
    dataset: str,
    schema: list[str],
    data: list[list] | pd.DataFrame,
    add_timestamp: bool = True,
) -> None:
    filename = _dated_filename(dataset) if add_timestamp else os.path.join(DATA_DIRECTORY, f"{dataset}.csv")

    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
        logger.info(f"Stored {dataset} → {filename}")
        return

    if not data:
        logger.warning(f"store_data_as_csv called with empty data for {dataset}")
        return

    row_len = len(data[0])
    if len(schema) != row_len:
        raise ValueError(f"Schema length {len(schema)} != data row length {row_len}")
    if not all(len(r) == row_len for r in data):
        raise ValueError("Mismatched row lengths in data")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(schema)
        writer.writerows(data)

    logger.info(f"Stored {dataset} → {filename}")


def read_data_as_pd(dataset: str) -> pd.DataFrame | None:
    """Return the most-recent matching CSV for dataset, or None if not found."""
    try:
        files = sorted(os.listdir(DATA_DIRECTORY))
    except FileNotFoundError:
        return None

    matches = [f for f in files if dataset in f and f.endswith(".csv")]
    if not matches:
        logger.debug(f"No CSV found for dataset '{dataset}' in {DATA_DIRECTORY}")
        return None

    # Date-suffixed filenames sort chronologically — take the latest
    path = os.path.join(DATA_DIRECTORY, matches[-1])
    logger.debug(f"Using {matches[-1]} as {dataset} data")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Sector/industry ratio lookup
# ---------------------------------------------------------------------------

def _split_to_set(s: str) -> set[str]:
    return set(re.split(r"[\/ :&]+", s))


def get_investment_ratios(sector: str, industry: str | None = None) -> list[float]:
    """Return [PE_threshold, PB_threshold] for the given sector/industry."""
    DEFAULT = [15.0, 2.5]

    if not sector or sector not in _ratios:
        return DEFAULT

    sector_cfg = _ratios[sector]
    default = sector_cfg.get("default", DEFAULT)

    def _coerce(ratios: list) -> list[float]:
        return [
            ratios[0] if ratios and len(ratios) > 0 and ratios[0] is not None else default[0],
            ratios[1] if ratios and len(ratios) > 1 and ratios[1] is not None else default[1],
        ]

    if not industry:
        return default

    # Exact match
    if industry in sector_cfg and sector_cfg[industry]:
        return _coerce(sector_cfg[industry])

    # Fuzzy match
    try:
        query = _split_to_set(industry)
        best, best_diff = None, float("inf")
        for key in sector_cfg:
            if key == "default":
                continue
            diff = len(query.difference(_split_to_set(key)))
            if diff == 0 and len(_split_to_set(key)) == len(query):
                return _coerce(sector_cfg[key])
            if diff < best_diff:
                best_diff, best = diff, key
        if best and best_diff < 3:
            return _coerce(sector_cfg[best])
    except Exception as e:
        logger.warning(f"Fuzzy match error for industry '{industry}': {e}")

    return default


# ---------------------------------------------------------------------------
# Finviz valuation updater
# ---------------------------------------------------------------------------

_FINVIZ_HEADERS = {"User-Agent": "Mozilla/5.0"}

_SECTOR_MAP = {
    "Materials": "Basic Materials",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Financials": "Financial",
    "Health Care": "Healthcare",
    "Information Technology": "Technology",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Communication Services": "Communication Services",
    "Consumer Services": "Consumer Cyclical",
    "Technology Services": "Technology",
    "Health Technology": "Healthcare",
    "Communications": "Communication Services",
    "Electronic Technology": "Technology",
    "Retail Trade": "Consumer Cyclical",
    "Consumer Durables": "Consumer Cyclical",
    "Transportation": "Industrials",
}

_INDUSTRY_MAP = {
    "Insurance - Life": "Life Insurance",
    "Insurance - Property & Casualty": "Property & Casualty Insurance",
    "Insurance - Specialty": "Specialty Insurance",
    "Insurance - Diversified": "Diversified Insurance",
    "REIT - Mortgage": "Mortgage REITs",
    "REIT - Diversified": "Diversified REITs",
    "REIT - Retail": "Retail REITs",
    "REIT - Residential": "Residential REITs",
    "REIT - Industrial": "Industrial REITs",
    "REIT - Office": "Office REITs",
    "REIT - Hotel & Motel": "Hotel & Motel REITs",
    "REIT - Healthcare Facilities": "Health Care REITs",
    "REIT - Specialty": "Specialty REITs",
    "Oil & Gas E&P": "Oil & Gas Exploration & Production",
    "Beverages - Brewers": "Brewers",
    "Beverages - Wineries & Distilleries": "Distillers & Vintners",
    "Beverages - Non-Alcoholic": "Non-Alcoholic Beverages",
    "Telecom Services": "Telecommunication Services",
    "Internet Content & Information": "Interactive Media & Services",
    "Software - Application": "Application Software",
    "Software - Infrastructure": "Systems Software",
}


def _fetch_finviz_table(url: str) -> dict[str, dict[str, Optional[float]]]:
    resp = requests.get(url, headers=_FINVIZ_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("div", class_="content").find(
        "table",
        class_="styled-table-new is-medium is-rounded is-tabular-nums w-full groups_table",
    )
    if not table:
        raise ValueError(f"Could not find data table at {url}")

    result = {}
    for row in table.find_all("tr")[1:]:
        link = row.find("a")
        if not link:
            continue
        cells = row.find_all("td")
        pe_text = cells[3].get_text() if len(cells) > 3 else "-"
        pb_text = cells[7].get_text() if len(cells) > 7 else "-"
        result[link.get_text()] = {
            "PE": float(pe_text) if pe_text != "-" else None,
            "PB": float(pb_text) if pb_text != "-" else None,
        }
    return result


def update_industry_valuations(verbose: bool = True) -> None:
    SECTOR_URL = "https://finviz.com/groups.ashx?g=sector&v=120&o=pe"
    INDUSTRY_URL = "https://finviz.com/groups.ashx?g=industry&v=120&o=pe"

    try:
        sector_data = _fetch_finviz_table(SECTOR_URL)
        industry_data = _fetch_finviz_table(INDUSTRY_URL)
    except Exception as e:
        logger.error(f"Failed to fetch Finviz data: {e}")
        raise

    ratios = _load_ratios()
    changes: list[dict] = []

    for sector_yaml, industries in ratios.items():
        if not isinstance(industries, dict):
            continue

        finviz_sector = _SECTOR_MAP.get(sector_yaml)
        if finviz_sector and finviz_sector in sector_data:
            new_pe = sector_data[finviz_sector]["PE"]
            new_pb = sector_data[finviz_sector]["PB"]
            default = industries.get("default")
            if isinstance(default, list) and len(default) >= 2:
                old_pe, old_pb = default[0], default[1]
                updated_pe = new_pe if new_pe is not None else old_pe
                updated_pb = new_pb if new_pb is not None else old_pb
                if updated_pe != old_pe or updated_pb != old_pb:
                    industries["default"] = [updated_pe, updated_pb]
                    changes.append({
                        "type": "sector", "name": sector_yaml,
                        "old": f"PE={old_pe}, PB={old_pb}",
                        "new": f"PE={updated_pe}, PB={updated_pb}",
                    })

        for ind_yaml, metrics in industries.items():
            if ind_yaml == "default" or not isinstance(metrics, list) or len(metrics) < 2:
                continue
            finviz_ind = _INDUSTRY_MAP.get(ind_yaml, ind_yaml)
            if finviz_ind not in industry_data:
                continue
            new_pe = industry_data[finviz_ind]["PE"]
            new_pb = industry_data[finviz_ind]["PB"]
            old_pe, old_pb = metrics[0], metrics[1]
            changed = False
            if new_pe is not None and new_pe != old_pe:
                metrics[0] = new_pe
                changed = True
            if new_pb is not None and new_pb != old_pb:
                metrics[1] = new_pb
                changed = True
            if changed:
                changes.append({
                    "type": "industry", "sector": sector_yaml, "name": ind_yaml,
                    "old": f"PE={old_pe}, PB={old_pb}",
                    "new": f"PE={metrics[0]}, PB={metrics[1]}",
                })

    if changes:
        with open(RATIOS_FILE, "w") as f:
            yaml.dump(ratios, f, default_flow_style=False, sort_keys=False)
        if verbose:
            logger.info(f"Updated {len(changes)} valuations in investments.yaml")
            for c in changes:
                loc = c["name"] if c["type"] == "sector" else f"{c['sector']} / {c['name']}"
                logger.info(f"  {c['type'].upper()} {loc}: {c['old']} → {c['new']}")
    else:
        if verbose:
            logger.info("No valuation changes detected")