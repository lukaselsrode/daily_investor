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

from core.paths import (
    CFG_DIRECTORY,
    CONFIG_FILE,
    DATA_DIRECTORY,
    RATIOS_FILE,
    ROOT_DIR,
)
from core.utils import run_async, safe_float
from data.cache import read_data_as_pd, store_data_as_csv
from data.valuation import get_investment_ratios, update_industry_valuations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(CONFIG_FILE) as f:
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
# Config-driven model id for the sentiment final-guard (env SENTIMENT_MODEL overrides).
# Empty string → sentiment.py uses its built-in default.
SENTIMENT_MODEL: str = str(_app.get("sentiment_model", "") or "")
CONFIDENCE_THRESHOLD: float = float(_app.get("confidence_threshold", 70))
SELL_SENTIMENT_OVERRIDE_CONFIDENCE: float = float(_app.get("sell_sentiment_override_confidence", 85))
ETFS:                 list  = _app.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"])

# Instruments that look like stocks in the universe but behave like funds.
# Excluded from all stock scoring and buy decisions; ETF exposure remains
# intentional via the explicit ETFS list above.
# Discretionary overlay (config `discretionary:` block) — the human judgment the quant can't encode:
#   excluded_industries/sectors  → hard NEVER-BUY (e.g. structurally-declining industries the
#                                   backward-looking factors score well but have no terminal upside)
#   conviction_holds             → hard NEVER-AUTO-SELL (thesis/turnaround positions whose forward
#                                   catalyst the momentum rules can't see)
_disc = _app.get("discretionary", {}) or {}
EXCLUDED_STOCK_INDUSTRIES: frozenset[str] = frozenset({
    "Investment Trusts Or Mutual Funds",
    "Investment Trusts/Mutual Funds",
} | {str(x) for x in _disc.get("excluded_industries", [])})
EXCLUDED_STOCK_SECTORS: frozenset[str] = frozenset(
    str(x) for x in _disc.get("excluded_sectors", [])
)
# Symbols the sell engine must never auto-exit (discretionary conviction).
CONVICTION_HOLDS: frozenset[str] = frozenset({str(x).upper() for x in _disc.get("conviction_holds", [])})

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
    # Position sizing: False = size ∝ score (legacy); True = size ∝ dollar-volume (cap-proxy).
    # Cap-proxy weighting tilts toward larger/more-liquid names WITHIN the per-name cap — the
    # survivorship-free edge lever (score/equal weighting loses to SPY; cap-proxy beats it).
    "size_by_dollar_volume":                bool(_rl.get("size_by_dollar_volume",                 False)),
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
    "buy_cooldown_days":                    int(_cr.get("buy_cooldown_days", 30)),
    "score_jitter_pct":                     float(_cr.get("score_jitter_pct", 0.08)),
    "allow_add_to_existing_if_score_above": (
        float(_cr["allow_add_to_existing_if_score_above"])
        if "allow_add_to_existing_if_score_above" in _cr else None
    ),
    "cooldown_exempt_if_active_underweight": bool(_cr.get("cooldown_exempt_if_active_underweight", False)),
}

# ---------------------------------------------------------------------------
# Unified scoring engine parameters (consolidated from old `scoring`, `momentum`,
# `momentum_v2`, `value_v2`, `scoring_v3` blocks). Hard-cutover: legacy top-level
# keys are rejected with a clear error pointing at the migrate-scoring CLI.
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when the YAML config uses a legacy/unmigrated shape."""


_LEGACY_TOP_LEVEL_KEYS = ("scoring_v3", "momentum_v2", "value_v2")


def _check_legacy_scoring_shape(app: dict) -> None:
    """Raise if any pre-consolidation scoring block exists at the top level."""
    legacy_present = [k for k in _LEGACY_TOP_LEVEL_KEYS if k in app]
    sc = app.get("scoring", {})
    sc_is_legacy = (
        isinstance(sc, dict)
        and "peer_standardization" not in sc
        and "factors" not in sc
        and "quality_checklist" not in sc
    )
    # Old flat scoring: had value_pe_weight / quality_volume_high at the top level
    if isinstance(sc, dict) and (
        "value_pe_weight" in sc or "quality_volume_high" in sc
    ):
        legacy_present.append("scoring (flat)")
    # Old flat momentum: position_bin_boundaries at top level (not under scoring.momentum_warmup)
    mo = app.get("momentum", {})
    if isinstance(mo, dict) and ("position_bin_boundaries" in mo or "position_bin_scores" in mo):
        legacy_present.append("momentum")
    if legacy_present and sc_is_legacy:
        raise ConfigError(
            "Config uses the pre-consolidation scoring shape. Found legacy keys: "
            f"{', '.join(sorted(set(legacy_present)))}. "
            "Run: `daily-investor config migrate-scoring --in-place` to convert to the unified "
            "`scoring:` block."
        )


_check_legacy_scoring_shape(_app)


def _normalize_blend(blend: dict) -> dict[str, float]:
    """Normalize blend weights to sum to 1.0; emit a warning if user values diverge."""
    ind = float(blend.get("industry_relative", 0.60))
    sec = float(blend.get("sector_relative",   0.25))
    mkt = float(blend.get("market_relative",   0.15))
    total = ind + sec + mkt
    if total <= 0:
        logger.warning("peer_standardization.blend weights non-positive — falling back to defaults")
        return {"industry_relative": 0.60, "sector_relative": 0.25, "market_relative": 0.15}
    if abs(total - 1.0) > 0.01:
        logger.warning(
            f"peer_standardization.blend weights sum to {total:.3f} (not 1.0) — using as-given"
        )
    return {"industry_relative": ind, "sector_relative": sec, "market_relative": mkt}


_sc = _app.get("scoring", {})
_sc_ps = _sc.get("peer_standardization", {})
_sc_fc = _sc.get("factors", {})
_sc_mi = _sc.get("momentum_inputs", {})
_sc_mi_w = _sc_mi.get("weights", {})
_sc_mi_p = _sc_mi.get("penalties", {})
_sc_mw = _sc.get("momentum_warmup", {})
_sc_qc = _sc.get("quality_checklist", {})


def _factor(name: str, defaults: dict) -> dict:
    raw = _sc_fc.get(name, {})
    out = {
        "enabled":                       bool(raw.get("enabled",       defaults.get("enabled", True))),
        "peer_relative":                 bool(raw.get("peer_relative", defaults.get("peer_relative", True))),
        "pe_weight":                     float(raw.get("pe_weight",    defaults.get("pe_weight", 0.70))),
        "pb_weight":                     float(raw.get("pb_weight",    defaults.get("pb_weight", 0.30))),
        "use_legacy_checklist_fallback": bool(raw.get("use_legacy_checklist_fallback",
                                                     defaults.get("use_legacy_checklist_fallback", True))),
        "safety_aware":                  bool(raw.get("safety_aware",  defaults.get("safety_aware", True))),
        # anchor_blend: weight on the cross-sectional anchor (vs pure peer-relative).
        # The legacy `v2_blend` key is migrated to `anchor_blend` by `migrate-scoring`.
        "anchor_blend":                  float(raw.get("anchor_blend",
                                                       defaults.get("anchor_blend", 0.0))),
    }
    if "distress" in raw or name == "value":
        d = raw.get("distress", {})
        out["distress"] = {
            "pe_threshold":         float(d.get("pe_threshold",         5.0)),
            "pe_penalty":           float(d.get("pe_penalty",           0.30)),
            "negative_eps_penalty": float(d.get("negative_eps_penalty", 0.25)),
        }
    return out


SCORING_PARAMS: dict = {
    "enabled": bool(_sc.get("enabled", True)),
    # Price-derived factor blend overlays (param slots 48/49, 0.0 = off). Passed
    # through so config edits reach tuning.constants._current_params — previously
    # the curated dict dropped them and the param vector always seeded 0.0.
    "quality_low_vol_blend":   float(_sc.get("quality_low_vol_blend",   0.0)),
    "momentum_residual_blend": float(_sc.get("momentum_residual_blend", 0.0)),
    "peer_standardization": {
        "group_by":          str(_sc_ps.get("group_by",          "industry")),
        "fallback_group_by": str(_sc_ps.get("fallback_group_by", "sector")),
        "min_group_size":    int(_sc_ps.get("min_group_size",    8)),
        "method":            str(_sc_ps.get("method",            "percentile")),
        "winsorize_pct":     float(_sc_ps.get("winsorize_pct",   0.05)),
        "clamp_low":         float(_sc_ps.get("clamp_low",      -1.0)),
        "clamp_high":        float(_sc_ps.get("clamp_high",      1.5)),
        "blend":             _normalize_blend(_sc_ps.get("blend", {})),
    },
    "factors": {
        "value":             _factor("value",             {"pe_weight": 0.70, "pb_weight": 0.30}),
        "quality":           _factor("quality",           {"use_legacy_checklist_fallback": True}),
        "momentum":          _factor("momentum",          {"enabled": False}),
        "income":            _factor("income",            {"safety_aware": True}),
        "growth_leadership": _factor("growth_leadership", {"enabled": False}),
    },
    "momentum_inputs": {
        "weights": {
            "rs_3m":           float(_sc_mi_w.get("rs_3m",           0.25)),
            "rs_6m":           float(_sc_mi_w.get("rs_6m",           0.25)),
            "risk_adj_3m":     float(_sc_mi_w.get("risk_adj_3m",     0.20)),
            "trend_structure": float(_sc_mi_w.get("trend_structure", 0.15)),
            "return_1m":       float(_sc_mi_w.get("return_1m",       0.10)),
            "return_5d":       float(_sc_mi_w.get("return_5d",       0.05)),
        },
        "penalties": {
            "falling_knife_3m_threshold":  float(_sc_mi_p.get("falling_knife_3m_threshold",  -0.15)),
            "falling_knife_penalty":       float(_sc_mi_p.get("falling_knife_penalty",        0.25)),
            "overextension_52w_threshold": float(_sc_mi_p.get("overextension_52w_threshold",  0.97)),
            "overextension_penalty":       float(_sc_mi_p.get("overextension_penalty",        0.20)),
            "high_vol_annual_threshold":   float(_sc_mi_p.get("high_vol_annual_threshold",    0.50)),
            "high_vol_penalty":            float(_sc_mi_p.get("high_vol_penalty",             0.15)),
        },
        "clamp_low":     float(_sc_mi.get("clamp_low",     -1.0)),
        "clamp_high":    float(_sc_mi.get("clamp_high",     1.5)),
        "winsorize_pct": float(_sc_mi.get("winsorize_pct",  0.05)),
    },
    "momentum_warmup": {
        "position_bin_boundaries":           list(_sc_mw.get("position_bin_boundaries",           [0.15, 0.35, 0.75, 0.95])),
        "position_bin_scores":               list(_sc_mw.get("position_bin_scores",               [-0.4, 0.1, 0.3, 0.5, 0.2])),
        "return_1m_low_position_cutoff":     float(_sc_mw.get("return_1m_low_position_cutoff",      0.40)),
        "return_1m_recovery_threshold":      float(_sc_mw.get("return_1m_recovery_threshold",        0.05)),
        "return_1m_falling_knife_threshold": float(_sc_mw.get("return_1m_falling_knife_threshold",  -0.10)),
        "return_1m_recovery_bonus":          float(_sc_mw.get("return_1m_recovery_bonus",           0.15)),
        "return_1m_falling_knife_penalty":   float(_sc_mw.get("return_1m_falling_knife_penalty",    0.20)),
    },
    "quality_checklist": {
        "income_score_cap":               float(_sc_qc.get("income_score_cap",               1.5)),
        "yield_trap_threshold":           float(_sc_qc.get("yield_trap_threshold",           0.10)),
        "distress_pe_max":                float(_sc_qc.get("distress_pe_max",                5.0)),
        "quality_volume_high":            float(_sc_qc.get("quality_volume_high",            1_000_000)),
        "quality_volume_low":             float(_sc_qc.get("quality_volume_low",             100_000)),
        "quality_dividend_min":           float(_sc_qc.get("quality_dividend_min",           0.02)),
        "quality_dividend_max":           float(_sc_qc.get("quality_dividend_max",           0.06)),
        "quality_weight_has_positive_pe": float(_sc_qc.get("quality_weight_has_positive_pe", 0.5)),
        "quality_weight_distress_pe":     float(_sc_qc.get("quality_weight_distress_pe",     -0.4)),
        "quality_weight_has_positive_pb": float(_sc_qc.get("quality_weight_has_positive_pb", 0.2)),
        "quality_weight_high_volume":     float(_sc_qc.get("quality_weight_high_volume",     0.3)),
        "quality_weight_low_volume":      float(_sc_qc.get("quality_weight_low_volume",      -0.3)),
        "quality_weight_yield_trap":      float(_sc_qc.get("quality_weight_yield_trap",      -0.6)),
        "quality_weight_healthy_dividend":float(_sc_qc.get("quality_weight_healthy_dividend", 0.2)),
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
# Contribution-timing overlay (buy-the-dip weekly contribution sizing)
# ---------------------------------------------------------------------------
# Deep-merged with defaults so a partial config block can't produce KeyErrors.
# enabled:false (the default) preserves flat weekly contributions exactly —
# see portfolio/contribution_timing.py.

_ct = _app.get("contribution_timing", {}) or {}
_ct_dip = _ct.get("dip_signal", {}) or {}
_ct_w   = _ct_dip.get("weights", {}) or {}
_ct_m   = _ct.get("multiplier", {}) or {}
_ct_rc  = _ct.get("regime_controls", {}) or {}
CONTRIBUTION_TIMING_PARAMS: dict = {
    "enabled":                      bool(_ct.get("enabled",                      False)),
    "benchmark_symbol":             str(_ct.get("benchmark_symbol",              "SPY")),
    "base_weekly_contribution":     float(_ct.get("base_weekly_contribution",    400.0)),
    "target_monthly_contribution":  float(_ct.get("target_monthly_contribution", 1600.0)),
    "budget_window_weeks":          int(_ct.get("budget_window_weeks",           4)),
    "min_weekly_contribution":      float(_ct.get("min_weekly_contribution",     100.0)),
    "max_weekly_contribution":      float(_ct.get("max_weekly_contribution",     800.0)),
    "preserve_monthly_budget":      bool(_ct.get("preserve_monthly_budget",      True)),
    "allow_budget_acceleration":    bool(_ct.get("allow_budget_acceleration",    False)),
    "monthly_budget_tolerance_pct": float(_ct.get("monthly_budget_tolerance_pct", 0.15)),
    "carry_forward_unused_budget":  bool(_ct.get("carry_forward_unused_budget",  True)),
    "borrow_from_future_weeks":     bool(_ct.get("borrow_from_future_weeks",     True)),
    "dip_signal": {
        "lookback_1w_days":     int(_ct_dip.get("lookback_1w_days",     5)),
        "lookback_1m_days":     int(_ct_dip.get("lookback_1m_days",     21)),
        "high_lookback_short":  int(_ct_dip.get("high_lookback_short",  20)),
        "high_lookback_medium": int(_ct_dip.get("high_lookback_medium", 60)),
        "ma_short":             int(_ct_dip.get("ma_short",             50)),
        "ma_long":              int(_ct_dip.get("ma_long",              200)),
        "weights": {
            "return_1w":    float(_ct_w.get("return_1w",    0.25)),
            "return_1m":    float(_ct_w.get("return_1m",    0.25)),
            "drawdown_20d": float(_ct_w.get("drawdown_20d", 0.20)),
            "drawdown_60d": float(_ct_w.get("drawdown_60d", 0.15)),
            "ma50_gap":     float(_ct_w.get("ma50_gap",     0.10)),
            "ma200_gap":    float(_ct_w.get("ma200_gap",    0.05)),
        },
    },
    "multiplier": {
        "neutral_dip_score": float(_ct_m.get("neutral_dip_score", 0.35)),
        "dip_sensitivity":   float(_ct_m.get("dip_sensitivity",   1.25)),
        "min_multiplier":    float(_ct_m.get("min_multiplier",    0.50)),
        "max_multiplier":    float(_ct_m.get("max_multiplier",    2.00)),
        "smoothing_alpha":   float(_ct_m.get("smoothing_alpha",   0.50)),
    },
    "regime_controls": {
        "cap_multiplier_in_defensive":   bool(_ct_rc.get("cap_multiplier_in_defensive",   True)),
        "defensive_max_multiplier":      float(_ct_rc.get("defensive_max_multiplier",     1.25)),
        "allow_full_dip_buying_in_bull": bool(_ct_rc.get("allow_full_dip_buying_in_bull", True)),
    },
}

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------

_bt = _app.get("backtest", {})
BACKTEST_PARAMS: dict = {
    "default_mode":                 str(_bt.get("default_mode",                 "liquid_universe_full")),
    "universe_selection":           str(_bt.get("universe_selection",           "liquid_sample")),
    "max_symbols":                  int(_bt.get("max_symbols",                  0)),   # 0 = full universe
    "min_volume":                   float(_bt.get("min_volume",                 500_000)),
    "random_seed":                  int(_bt.get("random_seed",                  42)),
    "slippage_bps":                 float(_bt.get("slippage_bps",               10.0)),
    "commission_per_trade":         float(_bt.get("commission_per_trade",       0.0)),
    "train_pct":                    float(_bt.get("train_pct",                  0.70)),
    "benchmark_symbol":             str(_bt.get("benchmark_symbol",             "SPY")),
    "starting_capital":             float(_bt.get("starting_capital",           5_000.0)),
    # Survivorship-free backtesting: when True, every backtest path (UI/CLI/tuner/engine) loads
    # split-adjusted prices for the current universe PLUS the delisted names from the FMP cache
    # (data/fmp_cache_adj/), removing the ~35% survivorship inflation. Requires that cache to be
    # populated; load_and_precompute falls back to yfinance (with a warning) if it is missing.
    "survivorship_free":            bool(_bt.get("survivorship_free",           False)),
    # Apply the discretionary never-buy industry/sector exclusions to the backtest candidate
    # universe (live/backtest parity). False = full-universe research. See data_loader.
    "apply_discretionary_exclusions": bool(_bt.get("apply_discretionary_exclusions", True)),
    "weekly_contribution":          float(_bt.get("weekly_contribution",        400.0)),
    "rebalance_frequency_days":     int(_bt.get("rebalance_frequency_days",     5)),
    "deploy_initial_cash":          bool(_bt.get("deploy_initial_cash",         True)),
    "reinvest_sell_proceeds":       bool(_bt.get("reinvest_sell_proceeds",      True)),
    "use_out_of_sample_validation": bool(_bt.get("use_out_of_sample_validation",True)),
    "auto_apply_if_valid":          bool(_bt.get("auto_apply_if_valid",         False)),
    "min_validation_excess_return": float(_bt.get("min_validation_excess_return",0.0)),
    "max_validation_drawdown":      float(_bt.get("max_validation_drawdown",    -0.20)),
    "min_validation_sharpe":        float(_bt.get("min_validation_sharpe",      0.25)),
    # Incumbent-relative gates: a tuned candidate must beat the CURRENT config's
    # validation excess-vs-SPY on the same split (+margin) and may not exceed its
    # turnover by more than the multiple. See tuner.validate_tuned_params.
    "min_excess_vs_incumbent":      float(_bt.get("min_excess_vs_incumbent",    0.0)),
    "max_turnover_multiple":        float(_bt.get("max_turnover_multiple",      2.0)),
    # Paired random-window reproducibility gate (tuner.paired_random_window_gate).
    "random_window_gate":           dict(_bt.get("random_window_gate",          {}) or {}),
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
    # Incumbent-relative churn penalty inside the DE objective — steers the
    # optimizer away from turnover regions the validation gate rejects anyway
    # (see tuning/objective.de_turnover_penalty).
    "de_turnover_penalty_enabled":      bool(_bt.get("de_turnover_penalty_enabled",      True)),
    "de_turnover_penalty_vs_incumbent": bool(_bt.get("de_turnover_penalty_vs_incumbent", True)),
    "de_turnover_soft_limit_multiple":  float(_bt.get("de_turnover_soft_limit_multiple", 1.5)),
    "de_turnover_hard_limit_multiple":  float(_bt.get("de_turnover_hard_limit_multiple", 2.5)),
    "de_turnover_penalty_weight":       float(_bt.get("de_turnover_penalty_weight",      1.0)),
    # Final multi-horizon confirmation gate (tuner.multi_horizon_confirm): the
    # selected tournament candidate is compared against the incumbent across
    # trailing windows AFTER the split + random-window gates pass.
    "multi_horizon_confirm":            dict(_bt.get("multi_horizon_confirm",            {}) or {}),
    # Stress-episode falsification gate (tuning.gauntlet.stress_gauntlet): the
    # selected candidate must SURVIVE named historical stress regimes relative
    # to the incumbent (catastrophe floors), not win them. Runs LAST.
    "stress_gauntlet":                  dict(_bt.get("stress_gauntlet",                  {}) or {}),
}

# ---------------------------------------------------------------------------
# Barroso–Santa-Clara active-sleeve volatility-scaling overlay
# ---------------------------------------------------------------------------
# Frozen-by-default risk overlay. When enabled, the active STOCK sleeve's
# exposure is scaled by w_t = clip(target_vol / realized_vol_{t-1}, 0, w_max),
# where realized_vol is the trailing annualized stdev of the active sleeve's
# OWN contribution-adjusted daily returns (lagged, so only past data is used).
# The de-risked fraction (1 - w_t) parks in CASH. Applied at rebalance cadence
# with a deadband to limit turnover.
#
# Validation (2026-05-31, multi-substrate paired control, park=cash pure timing):
#   timed-vs-matched-static-exposure beats in 36/45 windows (80%), sign p<1e-4,
#   survives weekly cadence + small deadband. Edge is drawdown reduction + a
#   small consistent excess, NOT an alpha multiplier. See Obsidian findings note.
#
# enabled=false preserves behavior exactly (no-op). Keep false unless a fresh
# robust multi-substrate validation re-confirms the edge on the live substrate.
_svo = _bt.get("sleeve_vol_overlay", {})
SLEEVE_VOL_OVERLAY: dict = {
    "enabled":      bool(_svo.get("enabled",      False)),
    "target_vol":   float(_svo.get("target_vol",   0.15)),
    "lookback":     int(_svo.get("lookback",       63)),
    "w_max":        float(_svo.get("w_max",        1.0)),
    "deadband":     float(_svo.get("deadband",     0.08)),
    "min_history":  int(_svo.get("min_history",    63)),
    "switch_bps":   float(_svo.get("switch_bps",   20.0)),
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

MAX_ITERATIONS: int = int(_app.get("max_iterations", 10))

# ---------------------------------------------------------------------------
# Archetype management parameters
# ---------------------------------------------------------------------------

ARCHETYPE_PARAMS: dict = _app.get("archetype_management", {"enabled": False})

_cp = _app.get("contrarian_penalty", {})
CONTRARIAN_PENALTY_PARAMS: dict = {
    "enabled":                bool(_cp.get("enabled", True)),
    "score_multiplier":       float(_cp.get("score_multiplier", 0.92)),
    "max_position_multiplier": float(_cp.get("max_position_multiplier", 0.60)),
}

_cl = _app.get("concentration_limits", {})
_cl_apply = _cl.get("apply_to", {}) or {}
_cl_enf = _cl.get("enforcement", {}) or {}
CONCENTRATION_LIMIT_PARAMS: dict = {
    "enabled":            bool(_cl.get("enabled", True)),
    "max_cluster_weight": float(_cl.get("max_cluster_weight", 0.35)),
    "max_sector_weight":  float(_cl.get("max_sector_weight", 0.40)),
    "cluster_method":     str(_cl.get("cluster_method", "pca")),
    "n_clusters":         int(_cl.get("n_clusters", 6)),
    "warn_only":          bool(_cl.get("warn_only", True)),
    "apply_to": {
        "active_sleeve": bool(_cl_apply.get("active_sleeve", True)),
        "etf_sleeve":    bool(_cl_apply.get("etf_sleeve",    False)),
    },
    "enforcement": {
        "block_new_buys":              bool(_cl_enf.get("block_new_buys",              True)),
        "allow_existing_positions":    bool(_cl_enf.get("allow_existing_positions",    True)),
        "allow_trim_only":             bool(_cl_enf.get("allow_trim_only",             True)),
        "allow_sell":                  bool(_cl_enf.get("allow_sell",                  True)),
        "allow_if_underweight":        bool(_cl_enf.get("allow_if_underweight",        True)),
        "downsize_to_fit":             bool(_cl_enf.get("downsize_to_fit",             True)),
        "min_remaining_alloc_multiple": float(_cl_enf.get("min_remaining_alloc_multiple", 1.0)),
    },
}

# ---------------------------------------------------------------------------
# Archetype classifier v2 — config-driven thresholds, opt-in via `enabled`
# ---------------------------------------------------------------------------

_ac = _app.get("archetype_classifier", {}) or {}
_ac_cb = _ac.get("confidence_buckets", {}) or {}
_ac_di = _ac.get("defensive_income", {}) or {}
_ac_qc = _ac.get("quality_compounder", {}) or {}
_ac_lt = _ac.get("legacy_turnaround", {}) or {}
_ac_sm = _ac.get("speculative_momentum", {}) or {}
_ac_vr = _ac.get("value_recovery", {}) or {}

ARCHETYPE_CLASSIFIER_PARAMS: dict = {
    "enabled": bool(_ac.get("enabled", False)),
    "confidence_buckets": {
        "high_min":   float(_ac_cb.get("high_min",   0.65)),
        "medium_min": float(_ac_cb.get("medium_min", 0.45)),
    },
    "defensive_income": {
        "require_yield":             bool(_ac_di.get("require_yield",             False)),
        "min_income_score":          float(_ac_di.get("min_income_score",          0.30)),
        "min_quality_score":         float(_ac_di.get("min_quality_score",         0.40)),
        "min_momentum_score":        float(_ac_di.get("min_momentum_score",       -0.10)),
        "max_volatility_percentile": float(_ac_di.get("max_volatility_percentile", 0.75)),
        "reject_falling_knife":      bool(_ac_di.get("reject_falling_knife",      True)),
        "yield_high":                float(_ac_di.get("yield_high",                0.80)),
        "yield_moderate":            float(_ac_di.get("yield_moderate",            0.50)),
        "yield_minimal":             float(_ac_di.get("yield_minimal",             0.05)),
        "sector_defensive":          list(_ac_di.get("sector_defensive", [
            "Utilities", "Real Estate", "Consumer Non-Durables",
            "Consumer Staples", "Finance",
        ])),
        "industry_defensive":        list(_ac_di.get("industry_defensive", [
            "Electric Utilities", "Gas Utilities", "Multi-Utilities",
            "Water Utilities", "Real Estate Investment Trusts",
            "Real Estate (Operations & Services)",
        ])),
        "quality_min_label":         float(_ac_di.get("quality_min_label",         0.25)),
        "momentum_disqualify_above": float(_ac_di.get("momentum_disqualify_above", 0.50)),
    },
    "quality_compounder": {
        "market_cap_mega":      float(_ac_qc.get("market_cap_mega",      100_000_000_000)),
        "market_cap_large":     float(_ac_qc.get("market_cap_large",      10_000_000_000)),
        "market_cap_small":     float(_ac_qc.get("market_cap_small",         500_000_000)),
        "maintenance_low":      float(_ac_qc.get("maintenance_low",      0.25)),
        "maintenance_high":     float(_ac_qc.get("maintenance_high",     0.27)),
        "maintenance_speculative": float(_ac_qc.get("maintenance_speculative", 1.0)),
        "day_trade_normal_max": float(_ac_qc.get("day_trade_normal_max", 0.25)),
        "analyst_buy_strong":   float(_ac_qc.get("analyst_buy_strong",   0.80)),
        "analyst_buy_moderate": float(_ac_qc.get("analyst_buy_moderate", 0.65)),
        "analyst_buy_weak":     float(_ac_qc.get("analyst_buy_weak",     0.40)),
        "quality_high":         float(_ac_qc.get("quality_high",         0.60)),
        "quality_moderate":     float(_ac_qc.get("quality_moderate",     0.35)),
        "quality_low":          float(_ac_qc.get("quality_low",          0.10)),
        "employees_scaled":     float(_ac_qc.get("employees_scaled",     50_000)),
        "employees_small":      float(_ac_qc.get("employees_small",      2_000)),
    },
    "legacy_turnaround": {
        "maintenance_speculative":  float(_ac_lt.get("maintenance_speculative",  1.0)),
        "maintenance_elevated":     float(_ac_lt.get("maintenance_elevated",     0.40)),
        "maintenance_above_standard": float(_ac_lt.get("maintenance_above_standard", 0.27)),
        "day_trade_elevated":       float(_ac_lt.get("day_trade_elevated",       0.25)),
        "market_cap_mid":           float(_ac_lt.get("market_cap_mid",           2_000_000_000)),
        "market_cap_large":         float(_ac_lt.get("market_cap_large",         10_000_000_000)),
        "market_cap_mega":          float(_ac_lt.get("market_cap_mega",          100_000_000_000)),
        "analyst_buy_weak":         float(_ac_lt.get("analyst_buy_weak",         0.35)),
        "analyst_buy_moderate":     float(_ac_lt.get("analyst_buy_moderate",     0.55)),
        "analyst_buy_strong":       float(_ac_lt.get("analyst_buy_strong",       0.80)),
        "momentum_strong":          float(_ac_lt.get("momentum_strong",          0.30)),
    },
    "speculative_momentum": {
        "momentum_very_strong": float(_ac_sm.get("momentum_very_strong", 0.60)),
        "momentum_strong":      float(_ac_sm.get("momentum_strong",      0.35)),
        "quality_very_low":     float(_ac_sm.get("quality_very_low",     0.10)),
        "quality_low":          float(_ac_sm.get("quality_low",          0.25)),
        "quality_too_high":     float(_ac_sm.get("quality_too_high",     0.60)),
        "maintenance_high":     float(_ac_sm.get("maintenance_high",     1.0)),
        "maintenance_elevated": float(_ac_sm.get("maintenance_elevated", 0.40)),
        "day_trade_high":       float(_ac_sm.get("day_trade_high",       0.40)),
        "market_cap_small":     float(_ac_sm.get("market_cap_small",     500_000_000)),
        "market_cap_mega":      float(_ac_sm.get("market_cap_mega",      100_000_000_000)),
        "income_minimal":       float(_ac_sm.get("income_minimal",       0.05)),
    },
    "value_recovery": {
        "value_undervalued":     float(_ac_vr.get("value_undervalued",     0.60)),
        "value_moderate":        float(_ac_vr.get("value_moderate",        0.30)),
        "momentum_improving_max":float(_ac_vr.get("momentum_improving_max",0.40)),
        "momentum_falling_min":  float(_ac_vr.get("momentum_falling_min",  -0.20)),
        "quality_min":           float(_ac_vr.get("quality_min",           0.15)),
        "quality_max":           float(_ac_vr.get("quality_max",           0.55)),
        "maintenance_distress":  float(_ac_vr.get("maintenance_distress",  1.0)),
    },
}

# ---------------------------------------------------------------------------
# Three-tier regime parameters
# ---------------------------------------------------------------------------

_rg = _app.get("regime", {})
_rg_def = _rg.get("defensive", {})
_rg_neu = _rg.get("neutral", {})
_rg_bull = _rg.get("bullish", {})
REGIME_PARAMS: dict = {
    "spy_ma_period":          int(_rg.get("spy_ma_period", 200)),
    "vix_defensive_threshold":float(_rg.get("vix_defensive_threshold", 30.0)),
    "vix_neutral_threshold":  float(_rg.get("vix_neutral_threshold",   20.0)),
    "bullish": {
        # Momentum-alpha tilt applied to score weights in confirmed-bull regime.
        "momentum_tilt": float(_rg_bull.get("momentum_tilt", 0.0)),
    },
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
        # Backtest regime de-risk overlay (frozen off). frac>0 rotates this fraction
        # of the held stock book into the benchmark on defensive-regime entry and
        # holds it there until the regime clears. switch_bps charges friction on each
        # rotation; lag delays acting on the regime signal by N days. See simulator
        # _do_regime_overlay and .session_tmp/regime_real_pinned.py.
        "backtest_derisk_frac": float(_rg_def.get("backtest_derisk_frac", 0.0)),
        "backtest_derisk_switch_bps": float(_rg_def.get("backtest_derisk_switch_bps", 20.0)),
        "backtest_derisk_lag": int(_rg_def.get("backtest_derisk_lag", 1)),
        # Contrarian mean-reversion blend in fear regimes (0.0 = off).
        "mean_reversion_blend": float(_rg_def.get("mean_reversion_blend", 0.0)),
        # Falling-knife / value-trap guard: multiplicatively penalize the composite of
        # below-200DMA names whose composite ranks in the top `fk_top_frac` of their
        # below-200DMA peers. Research (4h session) found high-composite downtrend names
        # systematically underperform (monotonic pooled deciles, t up to -7). 0.0 = off
        # (behavior-preserving default); 0.5 halves the score of the worst value traps.
        "falling_knife_guard": float(_rg_def.get("falling_knife_guard", 0.0)),
        "falling_knife_top_frac": float(_rg_def.get("falling_knife_top_frac", 0.5)),
    },
    "neutral": {
        "index_pct_override": (
            float(_rg_neu["index_pct_override"])
            if _rg_neu.get("index_pct_override") is not None else None
        ),
        "max_buys_override": (
            int(_rg_neu["max_buys_override"])
            if _rg_neu.get("max_buys_override") is not None else None
        ),
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
    # Live entry gate decoupled from metric_threshold (which anchors the EXIT
    # ladder). None = legacy behavior (ladder starts at metric_threshold).
    "entry_threshold_override":         (float(_cs["entry_threshold_override"])
                                         if _cs.get("entry_threshold_override") is not None else None),
    "fallback_thresholds":              [float(t) for t in _cs.get("fallback_thresholds", [])],
    "min_post_cooldown_candidates":     int(_cs.get("min_post_cooldown_candidates",     1)),
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
    # Peer-relative scoring diagnostic columns (populated by strategy.scoring.composite.compute_metric)
    "value_industry_rank",
    "value_sector_rank",
    "value_market_rank",
    "value_fallback_reason",
    "value_distress_flag",
    "quality_industry_rank",
    "quality_sector_rank",
    "quality_market_rank",
    "quality_fallback_reason",
    "momentum_industry_rank",
    "momentum_sector_rank",
    "momentum_market_rank",
    "momentum_fallback_reason",
    "momentum_penalties_applied",
    "income_industry_rank",
    "income_sector_rank",
    "income_fallback_reason",
    "scoring_model_version",
    # Robinhood instrument type (etp/cef/mlp/stock/adr/reit) — merged onto the
    # universe at build time by data.market.get_data; drives ETF/fund detection.
    "instrument_type",
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
_mt = float(_app.get("metric_threshold", 0.75))
_sw = float(_app.get("sell_rules", {}).get("sell_weak_value_below", 0.45))

def _resolve_trim_score_below(ed: dict, metric_threshold: float) -> float:
    """Return explicit trim_score_below, falling back to legacy delta derivation."""
    if "trim_score_below" in ed:
        return float(ed["trim_score_below"])
    delta = float(ed.get("trim_score_delta_threshold", -0.15))
    return metric_threshold * (1.0 + delta)

EXIT_DECISION_PARAMS: dict = {
    "trim_enabled":                bool(_ed.get("trim_enabled",                True)),
    "trim_fraction":               float(_ed.get("trim_fraction",              0.33)),
    "trim_min_gain_pct":           float(_ed.get("trim_min_gain_pct",          0.08)),
    "trim_score_below":            _resolve_trim_score_below(_ed, _mt),
    "trim_requires_positive_momentum": bool(_ed.get("trim_requires_positive_momentum", False)),
    "trim_to_etfs_pct":            float(_ed.get("trim_to_etfs_pct",           0.85)),
    "trim_profit_threshold":       float(_ed.get("trim_profit_threshold",      0.15)),
    "harvest_profit_threshold":    float(_ed.get("harvest_profit_threshold",   0.25)),
    "harvest_fraction":            float(_ed.get("harvest_fraction",           0.40)),
    "review_score_below":          float(_ed.get("review_score_below",         0.45)),
    "positive_pnl_exit_downgrade": bool(_ed.get("positive_pnl_exit_downgrade", True)),
    # DAE soft-exit floors — consumed by the simulator's faithful DecisionAdjustment
    # Engine soft-exit tree (defaults mirror decision_adjustment_engine.py). Exposed
    # here so config edits to the floors reach the backtest, not just the live loop.
    "hard_exit_score_below":           float(_ed.get("hard_exit_score_below",           0.20)),
    "thesis_intact_hard_exit_below":   float(_ed.get("thesis_intact_hard_exit_below",   0.35)),
    "positive_pnl_review_floor":       float(_ed.get("positive_pnl_review_floor",       0.00)),
    "positive_momentum_review_floor":  float(_ed.get("positive_momentum_review_floor",  0.10)),
    "strong_quality_review_floor":     float(_ed.get("strong_quality_review_floor",     0.70)),
    "thesis_intact_review_floor":      float(_ed.get("thesis_intact_review_floor",      0.60)),
    "positive_momentum_exit_downgrade": bool(_ed.get("positive_momentum_exit_downgrade", True)),
    "strong_quality_exit_downgrade":    bool(_ed.get("strong_quality_exit_downgrade",    True)),
    # Opportunity-cost ("max hold without progress") exit. Curated nested so the
    # simulator (reads EXIT_DECISION_PARAMS) and the live engine (reads the raw
    # exit_decision block) see identical values. enabled stays a config flag; the
    # three thresholds below are tunable via the active_opportunity_cost preset.
    "opportunity_cost": {
        "enabled":                    bool(_ed.get("opportunity_cost", {}).get("enabled",                    False)),
        "stall_max_days":             int(_ed.get("opportunity_cost", {}).get("stall_max_days",              120)),
        "reclaim_band":               float(_ed.get("opportunity_cost", {}).get("reclaim_band",              0.03)),
        "progress_momentum_floor":    float(_ed.get("opportunity_cost", {}).get("progress_momentum_floor",   0.10)),
        "require_stronger_candidate": bool(_ed.get("opportunity_cost", {}).get("require_stronger_candidate", False)),
    },
}

