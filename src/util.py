"""
util.py — Configuration constants and backward-compat re-exports.

Canonical implementations now live in:
  - core/paths.py     — ROOT_DIR, CFG_DIRECTORY, DATA_DIRECTORY, CONFIG_FILE, RATIOS_FILE
  - core/utils.py     — safe_float, run_async
  - data/cache.py     — store_data_as_csv, read_data_as_pd
  - data/valuation.py — get_investment_ratios, update_industry_valuations

This file re-exports all of the above for backward compat while keeping the
YAML-driven config constants as the canonical single source of truth.
"""

import logging

import yaml

from core.paths import ROOT_DIR, CFG_DIRECTORY, DATA_DIRECTORY, CONFIG_FILE, RATIOS_FILE  # noqa: F401
from core.utils import safe_float, run_async  # noqa: F401
from data.cache import store_data_as_csv, read_data_as_pd  # noqa: F401
from data.valuation import get_investment_ratios, update_industry_valuations  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


_app = _load_config()

# ---------------------------------------------------------------------------
# Public config constants
# ---------------------------------------------------------------------------

IGNORE_NEGATIVE_PE:   bool  = _app.get("ignore_negative_pe", False)
IGNORE_NEGATIVE_PB:   bool  = _app.get("ignore_negative_pb", False)
DIVIDEND_THRESHOLD:   float = float(_app.get("dividend_threshold", 0.03))
METRIC_THRESHOLD:     float = float(_app.get("metric_threshold", 0.8))
WEEKLY_INVESTMENT:    float = float(_app.get("weekly_investment", 400))
INDEX_PCT:            float = float(_app.get("index_pct", 0.85))
AUTO_APPROVE:         bool  = _app.get("auto_approve", False)
USE_SENTIMENT_ANALYSIS: bool = _app.get("use_sentiment_analysis", False)
CONFIDENCE_THRESHOLD: float = float(_app.get("confidence_threshold", 70))
SELL_SENTIMENT_OVERRIDE_CONFIDENCE: float = float(_app.get("sell_sentiment_override_confidence", 85))
ETFS:                 list  = _app.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"])

# Instruments that look like stocks in the universe but behave like funds.
# Excluded from all stock scoring and buy decisions; ETF exposure remains
# intentional via the explicit ETFS list above.
EXCLUDED_STOCK_INDUSTRIES: frozenset[str] = frozenset({
    "Investment Trusts Or Mutual Funds",
    "Investment Trusts/Mutual Funds",
})
EXCLUDED_STOCK_SECTORS: frozenset[str] = frozenset({
    "Miscellaneous",
})

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
    "minimum_hold_days":                    int(_rl.get("minimum_hold_days",                      0)),
    "allow_whole_share_fallback":           bool(_rl.get("allow_whole_share_fallback",            False)),
    "max_whole_share_buys_per_run":         int(_rl.get("max_whole_share_buys_per_run",           3)),
    "max_whole_share_allocation_multiplier":float(_rl.get("max_whole_share_allocation_multiplier",1.5)),
    "min_index_pct":                        float(_rl.get("min_index_pct",                        0.60)),
    "min_candidate_allocation_pct":         float(_rl.get("min_candidate_allocation_pct",         0.10)),
}

# ---------------------------------------------------------------------------
# Harvest parameters
# ---------------------------------------------------------------------------
# Candidate rotation
# ---------------------------------------------------------------------------

_cr = _app.get("candidate_rotation", {})
CANDIDATE_ROTATION_PARAMS: dict = {
    "buy_cooldown_days": int(_cr.get("buy_cooldown_days", 30)),
    "score_jitter_pct":  float(_cr.get("score_jitter_pct", 0.08)),
}

# ---------------------------------------------------------------------------
# Value scoring v2 — sector-relative robust normalization
# ---------------------------------------------------------------------------

_vv2   = _app.get("value_v2", {})
_vv2_d = _vv2.get("distress", {})
_vv2_c = _vv2.get("composite", {})
VALUE_V2_PARAMS: dict = {
    "enabled":          bool(_vv2.get("enabled",         True)),
    "winsorize_pct":    float(_vv2.get("winsorize_pct",  0.05)),
    "sector_relative":  bool(_vv2.get("sector_relative", True)),
    "min_sector_size":  int(_vv2.get("min_sector_size",  5)),
    "clamp_low":        float(_vv2.get("clamp_low",      -1.0)),
    "clamp_high":       float(_vv2.get("clamp_high",      1.5)),
    "distress": {
        "pe_threshold":          float(_vv2_d.get("pe_threshold",          5.0)),
        "pe_penalty":            float(_vv2_d.get("pe_penalty",            0.30)),
        "negative_eps_penalty":  float(_vv2_d.get("negative_eps_penalty",  0.25)),
    },
    "composite": {
        "pe_weight": float(_vv2_c.get("pe_weight", 0.60)),
        "pb_weight": float(_vv2_c.get("pb_weight", 0.40)),
    },
}

# ---------------------------------------------------------------------------
# Historical snapshot store parameters
# ---------------------------------------------------------------------------

_snap = _app.get("snapshots", {})
SNAPSHOT_PARAMS: dict = {
    "enabled":        bool(_snap.get("enabled",        True)),
    "subdir":         str(_snap.get("subdir",          "snapshots")),
    "retention_days": int(_snap.get("retention_days",  365)),
    "compression":    str(_snap.get("compression",     "snappy")),
}

# ---------------------------------------------------------------------------
# Dividend tracking parameters
# ---------------------------------------------------------------------------

_dv = _app.get("dividends", {})
DIVIDEND_PARAMS: dict = {
    "enabled":                    bool(_dv.get("enabled",                    True)),
    "track_income":               bool(_dv.get("track_income",               True)),
    "wash_sale_warning":          bool(_dv.get("wash_sale_warning",          True)),
    "block_rebuy_on_wash_sale_risk": bool(_dv.get("block_rebuy_on_wash_sale_risk", False)),
}

# ---------------------------------------------------------------------------
# Earnings quality bonus parameters
# ---------------------------------------------------------------------------

_ea = _app.get("earnings", {})
EARNINGS_PARAMS: dict = {
    "enabled":                  bool(_ea.get("enabled",                  True)),
    "max_quality_bonus":        float(_ea.get("max_quality_bonus",        0.10)),
    "positive_surprise_bonus":  float(_ea.get("positive_surprise_bonus",  0.03)),
    "negative_surprise_penalty":float(_ea.get("negative_surprise_penalty", 0.05)),
}

# ---------------------------------------------------------------------------

_hv = _app.get("harvest", {})
HARVEST_PARAMS: dict = {
    "min_harvest_amount":           float(_hv.get("min_harvest_amount",           25.0)),
    "max_harvest_pct_of_portfolio": float(_hv.get("max_harvest_pct_of_portfolio",  0.02)),
    "harvest_etfs":                 list(_hv.get("harvest_etfs",                  ["SPY", "VTI"])),
    "harvest_to_etfs_pct":         float(_hv.get("harvest_to_etfs_pct",          1.0)),
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
    # trade frequency / realism
    "max_trades_per_week":          int(_bt.get("max_trades_per_week",          10)),
    "cooldown_days_after_sell":     int(_bt.get("cooldown_days_after_sell",     3)),
    "cooldown_days_after_stopout":  int(_bt.get("cooldown_days_after_stopout",  7)),
    "vol_slippage_scaling":         bool(_bt.get("vol_slippage_scaling",        True)),
    "vol_slippage_multiplier":      float(_bt.get("vol_slippage_multiplier",    2.0)),
    "turnover_penalty_enabled":     bool(_bt.get("turnover_penalty_enabled",    True)),
    "turnover_penalty_trade_count": int(_bt.get("turnover_penalty_trade_count", 80)),
    "turnover_penalty_weight":      float(_bt.get("turnover_penalty_weight",    1.0)),
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
    "minimum_days_before_take_profit":     int(_sr.get("minimum_days_before_take_profit",         0)),
}

# ---------------------------------------------------------------------------
# Bear market regime
# ---------------------------------------------------------------------------

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

MAX_ITERATIONS: int = int(_app.get("max_iterations", 10))

# ---------------------------------------------------------------------------
# Momentum v2 parameters
# ---------------------------------------------------------------------------

_mv2 = _app.get("momentum_v2", {})
_mv2_w = _mv2.get("weights", {})
_mv2_p = _mv2.get("penalties", {})
MOMENTUM_V2_PARAMS: dict = {
    "weights": {
        "rs_3m":          float(_mv2_w.get("rs_3m",          0.25)),
        "rs_6m":          float(_mv2_w.get("rs_6m",          0.25)),
        "risk_adj_3m":    float(_mv2_w.get("risk_adj_3m",    0.20)),
        "trend_structure":float(_mv2_w.get("trend_structure", 0.15)),
        "return_1m":      float(_mv2_w.get("return_1m",      0.10)),
        "return_5d":      float(_mv2_w.get("return_5d",      0.05)),
    },
    "penalties": {
        "falling_knife_3m_threshold": float(_mv2_p.get("falling_knife_3m_threshold", -0.15)),
        "falling_knife_penalty":      float(_mv2_p.get("falling_knife_penalty",       0.25)),
        "overextension_52w_threshold":float(_mv2_p.get("overextension_52w_threshold", 0.97)),
        "overextension_penalty":      float(_mv2_p.get("overextension_penalty",       0.20)),
        "high_vol_annual_threshold":  float(_mv2_p.get("high_vol_annual_threshold",   0.50)),
        "high_vol_penalty":           float(_mv2_p.get("high_vol_penalty",            0.15)),
    },
    "clamp_low":     float(_mv2.get("clamp_low",     -1.0)),
    "clamp_high":    float(_mv2.get("clamp_high",     1.5)),
    "winsorize_pct": float(_mv2.get("winsorize_pct",  0.05)),
}

# ---------------------------------------------------------------------------
# Three-tier regime parameters
# ---------------------------------------------------------------------------

_rg = _app.get("regime", {})
_rg_def = _rg.get("defensive", {})
_rg_neu = _rg.get("neutral", {})
REGIME_PARAMS: dict = {
    "spy_ma_period":          int(_rg.get("spy_ma_period", 200)),
    "vix_defensive_threshold":float(_rg.get("vix_defensive_threshold", 30.0)),
    "vix_neutral_threshold":  float(_rg.get("vix_neutral_threshold",   20.0)),
    "defensive": {
        "index_pct_override": (
            float(_rg_def["index_pct_override"])
            if _rg_def.get("index_pct_override") is not None else None
        ),
        "max_buys_override": (
            int(_rg_def["max_buys_override"])
            if _rg_def.get("max_buys_override") is not None else None
        ),
        "stop_loss_tighten": float(_rg_def.get("stop_loss_tighten", 0.05)),
    },
    "neutral": {
        "index_pct_override": None,
        "max_buys_override":  None,
    },
}

# ---------------------------------------------------------------------------
# ETF core protection parameters
# ---------------------------------------------------------------------------

_er = _app.get("etf_risk", {})
ETF_RISK_PARAMS: dict = {
    "enabled":           bool(_er.get("enabled",           True)),
    "use_ma_filter":     bool(_er.get("use_ma_filter",     True)),
    "ma_period":         int(_er.get("ma_period",          200)),
    "defensive_etf_pct": float(_er.get("defensive_etf_pct", 0.85)),
    "defensive_cash_pct":float(_er.get("defensive_cash_pct", 0.10)),
}

# ---------------------------------------------------------------------------
# Reliability scoring — data/signal quality indicators (NOT alpha)
# ---------------------------------------------------------------------------

_rel = _app.get("reliability", {})
RELIABILITY_PARAMS: dict = {
    "enabled":              bool(_rel.get("enabled",              False)),
    "min_reliability_score":float(_rel.get("min_reliability_score", 0.70)),
}

# ---------------------------------------------------------------------------
# Parameter stability analysis — research/diagnostic only
# ---------------------------------------------------------------------------

_stab = _app.get("stability", {})
STABILITY_PARAMS: dict = {
    "enabled":                   bool(_stab.get("enabled",                   True)),
    "windows":                   list(_stab.get("windows",                   [30, 60, 90, 180, 365])),
    "objectives":                list(_stab.get("objectives",                ["sharpe", "calmar"])),
    "output_dir":                str(_stab.get("output_dir",                 "reports/stability")),
    "unstable_spread_threshold": float(_stab.get("unstable_spread_threshold",0.15)),
    "unstable_cv_threshold":     float(_stab.get("unstable_cv_threshold",    0.30)),
    "max_unstable_params":       int(_stab.get("max_unstable_params",        5)),
    "scan_maxiter":              int(_stab.get("scan_maxiter",               15)),
    "scan_popsize":              int(_stab.get("scan_popsize",               6)),
}

# ---------------------------------------------------------------------------
# Candidate selection — percentile/absolute mode + factor gates
# ---------------------------------------------------------------------------

_cs = _app.get("candidate_selection", {})
CANDIDATE_SELECTION_PARAMS: dict = {
    "mode":                             str(_cs.get("mode",                             "percentile")),
    "top_percentile":                   float(_cs.get("top_percentile",                 0.15)),
    "max_candidates":                   int(_cs.get("max_candidates",                   25)),
    "min_candidates":                   int(_cs.get("min_candidates",                   5)),
    "use_absolute_score_floor":         bool(_cs.get("use_absolute_score_floor",        True)),
    "absolute_score_floor":             float(_cs.get("absolute_score_floor",           0.45)),
    "min_quality_score":                float(_cs.get("min_quality_score",              0.30)),
    "min_momentum_score":               float(_cs.get("min_momentum_score",            -0.10)),
    "min_conditional_momentum_score":   float(_cs.get("min_conditional_momentum_score", 0.00)),
    "allow_income_defensive_exception": bool(_cs.get("allow_income_defensive_exception",False)),
}

# ---------------------------------------------------------------------------
# Optimizer tuning controls — frozen params and tighter bounds
# ---------------------------------------------------------------------------

_tn = _app.get("tuning", {})
TUNING_PARAMS: dict = {
    "frozen_parameters": list(_tn.get("frozen_parameters", [])),
    "parameter_bounds":  dict(_tn.get("parameter_bounds", {})),
}

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
    "missing_value_flag",
    "strategy_bucket",
    # momentum v2 raw features (populated by _enrich_with_momentum)
    "return_5d",
    "return_3m",
    "return_6m",
    "rs_1m",
    "rs_3m",
    "rs_6m",
    "realized_vol_3m",
    "risk_adj_momentum_3m",
    "above_50dma",
    "above_200dma",
    # reliability scores — data/signal quality, NOT alpha (populated by _compute_reliability_scores)
    "data_quality_score",
    "feature_coverage_score",
    "liquidity_reliability_score",
    "signal_stability_score",
    "reliability_score",
    # value v2 diagnostic columns (populated by apply_cross_sectional_value_v2)
    "value_score_raw",        # legacy ratio-based score before normalization
    "sector_value_score",     # same as new value_score, kept for UI referencing
    "relative_pe",            # sector-percentile rank of PE (-1=expensive, +1=cheap)
    "relative_pb",            # sector-percentile rank of PB (-1=expensive, +1=cheap)
]

AGG_DATA_COLUMNS: list[str] = ["symbol"] + METRIC_KEYS

# ---------------------------------------------------------------------------
# Exit-decision / TRIM parameters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rebalance / contribution-routing parameters
# ---------------------------------------------------------------------------

_rb = _app.get("rebalance", {})
REBALANCE_PARAMS: dict = {
    "mode":                         str(_rb.get("mode",                         "contribution_driven")),
    "drift_tolerance_pct":          float(_rb.get("drift_tolerance_pct",        0.03)),
    "proportional_deficit_routing": bool(_rb.get("proportional_deficit_routing", True)),
}

# ---------------------------------------------------------------------------
# Exit-decision / TRIM parameters
# ---------------------------------------------------------------------------

_ed = _app.get("exit_decision", {})
EXIT_DECISION_PARAMS: dict = {
    "trim_enabled":                bool(_ed.get("trim_enabled",                True)),
    "trim_fraction":               float(_ed.get("trim_fraction",              0.33)),
    "trim_min_gain_pct":           float(_ed.get("trim_min_gain_pct",          0.08)),
    "trim_score_delta_threshold":  float(_ed.get("trim_score_delta_threshold", -0.15)),
    "trim_requires_positive_momentum": bool(_ed.get("trim_requires_positive_momentum", True)),
    "trim_to_etfs_pct":            float(_ed.get("trim_to_etfs_pct",           0.85)),
    "trim_profit_threshold":       float(_ed.get("trim_profit_threshold",      0.15)),
    "harvest_profit_threshold":    float(_ed.get("harvest_profit_threshold",   0.25)),
}

