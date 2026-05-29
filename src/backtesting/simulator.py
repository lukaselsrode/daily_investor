"""
backtesting/simulator.py — Simulation engine: scoring, candidate selection, run_simulation,
run_backtest_report, compare_candidate_selection_modes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.types import TradeRecord
from util import (
    ARCHETYPE_PARAMS,
    BACKTEST_PARAMS,
    CANDIDATE_SELECTION_PARAMS,
    EXIT_DECISION_PARAMS,
    INDEX_PCT,
    METRIC_THRESHOLD,
    MOMENTUM_PARAMS,
    MOMENTUM_V2_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
)

from .types import BacktestReport, CandidatePoolDiagnostics, PrecomputedData, SimResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level sell constants
# ---------------------------------------------------------------------------

_TAKE_PROFIT_FLOOR_MULTIPLIER = 1.2
# Read stop-loss from config so live and backtest always stay in sync
_STOP_LOSS_PCT = float(SELL_RULES.get("stop_loss_pct", -0.20))
_MIN_DAYS_HELD_BEFORE_VALUE_EXIT = SELL_RULES.get("min_days_held_before_value_exit", 21)
_MIN_HOLD_DAYS = RISK_LIMITS.get("minimum_hold_days", 0)
_MIN_DAYS_BEFORE_TAKE_PROFIT = SELL_RULES.get("minimum_days_before_take_profit", 0)

# Harvest partial exit — mirrors live HARVEST action (partial profit-taking)
_HARVEST_PROFIT_THRESHOLD = float(EXIT_DECISION_PARAMS.get("harvest_profit_threshold", 0.30))
_HARVEST_FRACTION         = float(EXIT_DECISION_PARAMS.get("harvest_fraction",          0.40))

# WATCH/REVIEW suppression — if profitable and score above this floor, hold rather than exit
_REVIEW_SCORE_FLOOR    = float(EXIT_DECISION_PARAMS.get("review_score_below",         0.45))
_POSITIVE_PNL_DOWNGRADE = bool(EXIT_DECISION_PARAMS.get("positive_pnl_exit_downgrade", True))


# ---------------------------------------------------------------------------
# Public entry: default params
# ---------------------------------------------------------------------------

def get_default_params() -> np.ndarray:
    """
    Build the 15-element params vector from the current config values.

    Layout mirrors tuner._current_params():
      [0-3]  score_weights (value, quality, income, momentum)
      [4]    index_pct
      [5]    metric_threshold
      [6]    take_profit_pct
      [7]    sell_weak_value_below
      [8]    trailing_stop_pct
      [9]    value_pe_weight
      [10-14] momentum_v2 sub-weights (rs_3m, rs_6m, risk_adj_3m, trend_structure, return_1m)
    """
    sw  = SCORE_WEIGHTS
    v2w = MOMENTUM_V2_PARAMS.get("weights", {})
    return np.array([
        sw["value"], sw["quality"], sw["income"], sw["momentum"],
        INDEX_PCT,
        METRIC_THRESHOLD,
        SELL_RULES["take_profit_pct"],
        SELL_RULES["sell_weak_value_below"],
        SELL_RULES["trailing_stop_pct"],
        SCORING_PARAMS["value_pe_weight"],
        v2w.get("rs_3m",           0.25),
        v2w.get("rs_6m",           0.25),
        v2w.get("risk_adj_3m",     0.20),
        v2w.get("trend_structure", 0.15),
        v2w.get("return_1m",       0.10),
    ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col_arr(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).values.astype(np.float64)
    return np.full(len(df), default, dtype=np.float64)


def _momentum_score_vec(
    bin_indices: np.ndarray,
    has_pos: np.ndarray,
    pos_52w: np.ndarray,
    return_1m: np.ndarray,
    mbin_scores: np.ndarray,
) -> np.ndarray:
    """Vectorized v1 (bucket) momentum scoring — used as fallback when v2 features absent."""
    mp = MOMENTUM_PARAMS
    base = np.where(has_pos, mbin_scores[bin_indices], 0.0)

    has_r1m  = np.isfinite(return_1m)
    low_pos  = has_pos & (pos_52w < mp["return_1m_low_position_cutoff"]) & has_r1m
    recovery = low_pos & (return_1m >= mp["return_1m_recovery_threshold"])
    falling  = low_pos & (return_1m <= mp["return_1m_falling_knife_threshold"])

    base = base + np.where(recovery, mp["return_1m_recovery_bonus"],        0.0)
    base = base - np.where(falling,  mp["return_1m_falling_knife_penalty"], 0.0)
    return base


def _pct_rank_vec(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank scaled to [-1, 1]. NaN → 0.0 (neutral)."""
    out = np.zeros(len(arr))
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return out
    vals  = arr[finite]
    ranks = (vals.argsort().argsort() + 1) / (finite.sum() + 1)
    out[finite] = ranks * 2 - 1
    return out


def _momentum_score_v2_vec(
    day: int,
    precomp: PrecomputedData,
    mom_weights_raw: np.ndarray,
) -> np.ndarray:
    """Vectorized v2 momentum scoring using multi-factor cross-sectional model."""
    cfg = MOMENTUM_V2_PARAMS
    pen = cfg["penalties"]
    n   = precomp.prices.shape[1]
    zeros = np.zeros(n)

    def _get(arr, default_val=0.0):
        if arr is None:
            return np.full(n, default_val)
        row = arr[day]
        return np.where(np.isfinite(row), row, default_val)

    rs3m  = _get(precomp.rs_3m_daily)
    rs6m  = _get(precomp.rs_6m_daily)
    ret3m = _get(precomp.ret_3m_daily)
    vol3m = _get(precomp.vol_3m_daily, 0.20)
    ret1m = precomp.return_1m_daily[day]
    ret1m = np.where(np.isfinite(ret1m), ret1m, 0.0)
    ret5d = _get(precomp.ret_5d_daily)
    pos52 = precomp.position_52w_daily[day]
    pos52 = np.where(np.isfinite(pos52), pos52, 0.5)

    safe_vol = np.clip(vol3m, 0.01, None)
    risk_adj = ret3m / safe_vol

    a50  = (precomp.above_50dma_daily[day]  if precomp.above_50dma_daily  is not None else zeros).astype(bool)
    a200 = (precomp.above_200dma_daily[day] if precomp.above_200dma_daily is not None else zeros).astype(bool)
    trend = np.select([a50 & a200, a50 & ~a200, ~a50 & a200], [0.5, 0.1, -0.1], default=-0.5)

    raw_w = np.abs(mom_weights_raw[:5])
    w_r5d = cfg["weights"].get("return_5d", 0.05)
    total = raw_w.sum() + w_r5d
    if total < 1e-9:
        total = 1.0
    w = raw_w / total

    score = (
        w[0] * _pct_rank_vec(rs3m)     +
        w[1] * _pct_rank_vec(rs6m)     +
        w[2] * _pct_rank_vec(risk_adj) +
        w[3] * trend                    +
        w[4] * _pct_rank_vec(ret1m)    +
        (w_r5d / total) * _pct_rank_vec(ret5d)
    )

    score -= np.where(ret3m < pen["falling_knife_3m_threshold"],  pen["falling_knife_penalty"],  0.0)
    score -= np.where(pos52  > pen["overextension_52w_threshold"], pen["overextension_penalty"],  0.0)
    score -= np.where(vol3m  > pen["high_vol_annual_threshold"],    pen["high_vol_penalty"],       0.0)

    return np.clip(score, cfg["clamp_low"], cfg["clamp_high"])


# ---------------------------------------------------------------------------
# Window split
# ---------------------------------------------------------------------------

def split_price_window(n_days: int, train_pct: float) -> tuple[slice, slice]:
    """Split n_days into train and validation slices."""
    train_end = max(1, int(n_days * train_pct))
    return slice(0, train_end), slice(train_end, n_days)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(daily_values: np.ndarray) -> dict:
    """Return sharpe, calmar, max_drawdown, total_return for a daily-value series."""
    if len(daily_values) < 2 or daily_values[0] <= 0:
        return {"sharpe": 0.0, "calmar": 0.0, "max_drawdown": 0.0, "total_return": 0.0}

    total_return = float(daily_values[-1] / daily_values[0]) - 1.0

    daily_rets = np.diff(daily_values) / daily_values[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    sharpe = 0.0
    if len(daily_rets) > 2 and daily_rets.std() > 0:
        sharpe = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))

    cum      = daily_values / daily_values[0]
    roll_max = np.maximum.accumulate(cum)
    drawdowns = np.where(roll_max > 0, cum / roll_max - 1.0, 0.0)
    max_drawdown = float(drawdowns.min())

    calmar = 0.0
    if max_drawdown < -0.001:
        calmar = float(total_return / abs(max_drawdown))

    return {"sharpe": sharpe, "calmar": calmar, "max_drawdown": max_drawdown, "total_return": total_return}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_stocks(precomp: PrecomputedData, params: np.ndarray) -> np.ndarray:
    """Compute per-stock scores. Uses day-0 snapshot — prefer score_stocks_at_day for simulation."""
    return score_stocks_at_day(precomp, params, 0)


def score_stocks_at_day(precomp: PrecomputedData, params: np.ndarray, day: int) -> np.ndarray:
    """
    Score stocks using day-specific rolling momentum features.

    Routes to v2 continuous scoring when multi-factor arrays are populated,
    otherwise falls back to v1 bucket scoring for backward compatibility.
    """
    raw_sw = params[:4]
    sw = raw_sw / max(raw_sw.sum(), 1e-9)
    value_pe_w  = params[9]
    value_score = value_pe_w * precomp.pe_comp + (1.0 - value_pe_w) * precomp.pb_comp

    if precomp.ret_3m_daily is not None:
        momentum_score = _momentum_score_v2_vec(day, precomp, params[10:15])
    else:
        momentum_score = _momentum_score_vec(
            precomp.bin_indices_daily[day],
            precomp.has_position_52w_daily[day],
            precomp.position_52w_daily[day],
            precomp.return_1m_daily[day],
            params[10:15],
        )

    return (
        sw[0] * value_score
        + sw[1] * precomp.quality_scores
        + sw[2] * precomp.income_scores
        + sw[3] * momentum_score
    )


def _momentum_score_at_day(precomp: PrecomputedData, params: np.ndarray, day: int) -> np.ndarray:
    if precomp.ret_3m_daily is not None:
        return _momentum_score_v2_vec(day, precomp, params[10:15])
    return _momentum_score_vec(
        precomp.bin_indices_daily[day],
        precomp.has_position_52w_daily[day],
        precomp.position_52w_daily[day],
        precomp.return_1m_daily[day],
        params[10:15],
    )


def _detect_regime(precomp: PrecomputedData, day: int) -> str:
    bench = precomp.benchmark_prices
    if day >= 200 and np.isfinite(bench[day]) and bench[day] > 0:
        ma200 = float(np.nanmean(bench[max(0, day - 199): day + 1]))
        if bench[day] < ma200 * 0.95:
            return "defensive"
        if bench[day] < ma200:
            return "neutral"
    return "bullish"


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_candidates(
    day: int,
    composite_scores: np.ndarray,
    precomp: PrecomputedData,
    params: np.ndarray,
    cs_params: dict | None = None,
) -> tuple[np.ndarray, CandidatePoolDiagnostics]:
    """
    Choose the buy-eligible candidate mask.
    Returns (candidate_mask, diagnostics).
    """
    if cs_params is None:
        cs_params = CANDIDATE_SELECTION_PARAMS

    n            = len(composite_scores)
    mode         = cs_params["mode"]
    max_cands    = cs_params["max_candidates"]
    min_qual     = cs_params["min_quality_score"]
    min_mom      = cs_params["min_momentum_score"]
    min_cond_mom = cs_params["min_conditional_momentum_score"]
    allow_def    = cs_params["allow_income_defensive_exception"]
    is_defensive = _detect_regime(precomp, day) == "defensive"

    valid = np.isfinite(composite_scores)
    if mode == "percentile":
        top_pct = cs_params["top_percentile"]
        valid_scores = composite_scores[valid]
        cutoff = float(np.percentile(valid_scores, (1.0 - top_pct) * 100.0)) if len(valid_scores) else float(params[5])
    else:
        cutoff = float(params[5])

    score_mask = composite_scores >= cutoff

    floor_excluded = 0
    if cs_params.get("use_absolute_score_floor", True):
        floor       = cs_params["absolute_score_floor"]
        floor_gate  = composite_scores >= floor
        floor_excluded = int((score_mask & ~floor_gate).sum())
        score_mask  = score_mask & floor_gate

    quality_gate = precomp.quality_scores >= min_qual
    qual_excluded = int((score_mask & ~quality_gate).sum())

    mom_scores  = _momentum_score_at_day(precomp, params, day)
    mom_gate    = mom_scores >= min_mom
    mom_excluded = int((score_mask & quality_gate & ~mom_gate).sum())

    has_income    = precomp.income_scores > 0.0
    cond_mom_weak = mom_scores < min_cond_mom
    income_at_risk = has_income & cond_mom_weak & ~precomp.yield_trap_mask
    income_trap_gate = ~income_at_risk | (allow_def & is_defensive)
    income_trap_excluded = int((score_mask & quality_gate & mom_gate & ~income_trap_gate).sum())

    final_mask = score_mask & quality_gate & mom_gate & income_trap_gate

    n_sel = int(final_mask.sum())
    if n_sel > max_cands:
        top_indices = sorted(np.where(final_mask)[0], key=lambda i: -composite_scores[i])[:max_cands]
        final_mask = np.zeros(n, dtype=bool)
        for i in top_indices:
            final_mask[i] = True

    selected = np.where(final_mask)[0]
    n_sel    = len(selected)

    value_pe_w  = float(params[9])
    value_scores_diag = value_pe_w * precomp.pe_comp + (1.0 - value_pe_w) * precomp.pb_comp

    avg_quality  = float(precomp.quality_scores[selected].mean()) if n_sel else 0.0
    avg_momentum = float(mom_scores[selected].mean())              if n_sel else 0.0
    avg_income   = float(precomp.income_scores[selected].mean())   if n_sel else 0.0
    avg_value    = float(value_scores_diag[selected].mean())       if n_sel else 0.0

    sector_counts: dict = {}
    for i in selected:
        s = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"
        sector_counts[s] = sector_counts.get(s, 0) + 1

    excl_names: list[str] = []
    excl_mask = score_mask & quality_gate & mom_gate & ~income_trap_gate
    for i in np.where(excl_mask)[0][:10]:
        if i < len(precomp.symbols):
            excl_names.append(precomp.symbols[i])

    diag = CandidatePoolDiagnostics(
        n_candidates=n_sel,
        score_cutoff=cutoff,
        avg_quality=avg_quality,
        avg_momentum=avg_momentum,
        avg_income=avg_income,
        avg_value=avg_value,
        sector_counts=sector_counts,
        n_income_trap_excluded=income_trap_excluded,
        n_quality_gate_excluded=qual_excluded,
        n_momentum_gate_excluded=mom_excluded,
        n_floor_excluded=floor_excluded,
        excluded_high_income_low_momentum=excl_names,
    )
    return final_mask, diag


# ---------------------------------------------------------------------------
# Benchmark TWR
# ---------------------------------------------------------------------------

def _bench_twr(
    bench_prices: np.ndarray,
    starting_capital: float,
    weekly_contribution: float,
    rebalance_freq: int,
) -> float:
    """Contribution-adjusted TWR for a buy-and-hold benchmark receiving the same cash schedule."""
    n      = len(bench_prices)
    shares = 0.0
    cash   = starting_capital
    ca     = np.zeros(n)
    for d in range(n):
        p      = bench_prices[d]
        contrib = 0.0
        if d == 0 and p > 0:
            shares = cash / p
            cash   = 0.0
        elif d > 0 and d % rebalance_freq == 0 and p > 0:
            contrib      = weekly_contribution
            prev_shares  = shares          # capture pre-contribution shares for TWR
            shares      += contrib / p
            cash         = 0.0
        else:
            prev_shares = shares
        val = shares * p + cash
        if d == 0:
            ca[0] = val
        else:
            prev   = prev_shares * bench_prices[d - 1]
            factor = (val - contrib) / max(prev, 1e-9)
            ca[d]  = ca[d - 1] * factor
    return compute_performance_metrics(ca)["total_return"]


def _bench_daily_equity(
    bench_prices: np.ndarray,
    starting_capital: float,
    weekly_contribution: float,
    rebalance_freq: int,
) -> np.ndarray:
    """Daily benchmark portfolio values matching the same contribution schedule as the strategy."""
    n      = len(bench_prices)
    shares = 0.0
    vals   = np.zeros(n)
    for d in range(n):
        p = bench_prices[d]
        if not np.isfinite(p) or p <= 0:
            vals[d] = vals[d - 1] if d > 0 else starting_capital
            continue
        if d == 0:
            shares = starting_capital / p
        elif d % rebalance_freq == 0:
            shares += weekly_contribution / p
        vals[d] = shares * p
    return vals


def _bench_daily_ca_equity(
    bench_prices: np.ndarray,
    starting_capital: float,
    weekly_contribution: float,
    rebalance_freq: int,
) -> np.ndarray:
    """Contribution-adjusted daily benchmark series (same TWR basis as _bench_twr)."""
    n      = len(bench_prices)
    shares = 0.0
    cash   = starting_capital
    ca     = np.zeros(n)
    for d in range(n):
        p      = bench_prices[d]
        contrib = 0.0
        if d == 0 and p > 0:
            shares = cash / p
            cash   = 0.0
        elif d > 0 and d % rebalance_freq == 0 and p > 0:
            contrib     = weekly_contribution
            prev_shares = shares
            shares     += contrib / p
            cash        = 0.0
        else:
            prev_shares = shares
        val = shares * p + cash
        if d == 0:
            ca[0] = val
        else:
            prev   = prev_shares * bench_prices[d - 1]
            factor = (val - contrib) / max(prev, 1e-9)
            ca[d]  = ca[d - 1] * factor
    return ca


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def _build_archetype_thresholds(
    precomp: PrecomputedData,
    default_take_profit: float,
    default_trailing_stop: float,
    default_min_hold: int,
    arch_cfg_override: dict | None = None,
) -> dict:
    """
    Classify each stock archetype from precomp quality/income/momentum scores.
    Returns a dict of per-stock arrays:
      tp, stop, harvest, min_hold     — lifecycle thresholds
      enabled, score_mult, pos_mult   — buy-side controls
      max_sleeve, min_score           — sleeve / score gating (NaN/None when unset)
      labels                          — archetype label per stock
    Falls back to defaults when archetype management is disabled or signals missing.

    When arch_cfg_override is provided (from `archetype_cfg_from_params(params)`),
    that takes precedence over ARCHETYPE_PARAMS so the tuner's lifecycle slots
    actually flow into per-position thresholds.
    """
    from portfolio.position_archetypes import classify_archetype_from_scores

    n = precomp.quality_scores.shape[0]
    tp_arr        = np.full(n, default_take_profit,    dtype=np.float64)
    stop_arr      = np.full(n, default_trailing_stop,  dtype=np.float64)
    harvest_arr   = np.full(n, default_take_profit,    dtype=np.float64)
    min_hold_arr  = np.full(n, default_min_hold,       dtype=np.int32)
    enabled_arr   = np.ones(n,                         dtype=bool)
    score_mult_arr= np.ones(n,                         dtype=np.float64)
    pos_mult_arr  = np.ones(n,                         dtype=np.float64)
    max_sleeve_arr= np.full(n, np.nan,                 dtype=np.float64)
    min_score_arr = np.full(n, np.nan,                 dtype=np.float64)
    labels        = [""] * n

    if arch_cfg_override is not None and arch_cfg_override:
        arch_cfg = arch_cfg_override
    else:
        arch_cfg = ARCHETYPE_PARAMS if ARCHETYPE_PARAMS.get("enabled", False) else None
    if arch_cfg is None:
        return {
            "tp": tp_arr, "stop": stop_arr, "harvest": harvest_arr, "min_hold": min_hold_arr,
            "enabled": enabled_arr, "score_mult": score_mult_arr, "pos_mult": pos_mult_arr,
            "max_sleeve": max_sleeve_arr, "min_score": min_score_arr, "labels": labels,
        }

    yt_mask = precomp.yield_trap_mask if precomp.yield_trap_mask is not None else np.zeros(n, dtype=bool)

    for i in range(n):
        try:
            policy = classify_archetype_from_scores(
                quality_score  = float(precomp.quality_scores[i]),
                momentum_score = float(precomp.income_scores[i]),   # proxy at day-0
                income_score   = float(precomp.income_scores[i]),
                yield_trap     = bool(yt_mask[i]),
                archetype_cfg  = arch_cfg,
            )
            tp_arr[i]        = policy.harvest_profit_threshold
            stop_arr[i]      = policy.trailing_stop_pct
            harvest_arr[i]   = policy.harvest_profit_threshold
            min_hold_arr[i]  = int(policy.minimum_hold_days)
            enabled_arr[i]   = bool(policy.enabled)
            score_mult_arr[i]= float(policy.score_multiplier)
            pos_mult_arr[i]  = float(policy.max_position_multiplier)
            if policy.max_active_weight is not None:
                max_sleeve_arr[i] = float(policy.max_active_weight)
            if policy.min_score_to_buy is not None:
                min_score_arr[i] = float(policy.min_score_to_buy)
            labels[i]        = policy.archetype
        except Exception:
            pass

    return {
        "tp": tp_arr, "stop": stop_arr, "harvest": harvest_arr, "min_hold": min_hold_arr,
        "enabled": enabled_arr, "score_mult": score_mult_arr, "pos_mult": pos_mult_arr,
        "max_sleeve": max_sleeve_arr, "min_score": min_score_arr, "labels": labels,
    }


def run_simulation(
    precomp: PrecomputedData,
    params: np.ndarray,
    starting_capital: float = 10_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    weekly_contribution: float = 0.0,
    rebalance_frequency_days: int = 5,
    cs_params: dict | None = None,
    archetype_aware: bool = False,
    cluster_tracking: bool = False,
    scope: str = "overall_strategy",
    accounting_trace: list | None = None,
) -> SimResult:
    """
    Simulate the strategy over the precomputed price history.

    params layout (15 values):
      [0] sw_value  [1] sw_quality  [2] sw_income  [3] sw_momentum
      [4] index_pct  [5] metric_threshold  [6] take_profit_pct
      [7] sell_weak_below  [8] trailing_stop  [9] value_pe_weight
      [10-14] momentum sub-weights (v2) or bin scores (v1 fallback)
    """
    n_days, n_stocks = precomp.prices.shape
    n_etfs           = precomp.etf_prices.shape[1]

    index_pct        = float(params[4])
    metric_threshold = float(params[5])
    take_profit_pct  = float(params[6])
    sell_weak_below  = float(params[7])
    trailing_stop    = float(params[8])

    _trim_cfg            = EXIT_DECISION_PARAMS
    _trim_enabled        = bool(_trim_cfg.get("trim_enabled", True))
    _trim_fraction       = float(_trim_cfg.get("trim_fraction", 0.33))
    _trim_min_gain       = float(_trim_cfg.get("trim_min_gain_pct", 0.08))
    _trim_score_below    = float(_trim_cfg.get("trim_score_below",
                               metric_threshold * (1.0 + float(_trim_cfg.get("trim_score_delta_threshold", -0.15)))))
    _trim_to_etfs_pct    = float(_trim_cfg.get("trim_to_etfs_pct", 0.85))
    if scope == "active_sleeve_compounding":
        _trim_to_etfs_pct = 0.0

    base_slippage   = slippage_bps / 10_000.0
    bp_cfg          = BACKTEST_PARAMS
    use_vol_slip    = bp_cfg.get("vol_slippage_scaling", True)
    vol_slip_mult   = bp_cfg.get("vol_slippage_multiplier", 2.0)
    cooldown_sell   = bp_cfg.get("cooldown_days_after_sell", 3)
    cooldown_stop   = bp_cfg.get("cooldown_days_after_stopout", 7)
    max_trades_week = bp_cfg.get("max_trades_per_week", 10)

    min_order      = RISK_LIMITS["min_order_amount"]
    max_single_pct = RISK_LIMITS["max_single_position_pct"]
    max_sector_pct = RISK_LIMITS["max_sector_pct"]
    max_order_pct  = RISK_LIMITS["max_order_pct_of_cash"]
    max_buys       = RISK_LIMITS["max_buys_per_rebalance"]

    # Per-stock archetype thresholds + controls (used when archetype_aware=True)
    # Build a config override from the params tail (slots 15-38) so the tuner's
    # active_archetype_lifecycle preset takes effect during the sim.
    _arch_cfg_override: dict | None = None
    if len(params) > 15:
        try:
            from tuning.constants import archetype_cfg_from_params
            _arch_cfg_override = archetype_cfg_from_params(params)
        except Exception:
            _arch_cfg_override = None
    if archetype_aware:
        _arch = _build_archetype_thresholds(
            precomp, take_profit_pct, trailing_stop, _MIN_HOLD_DAYS,
            arch_cfg_override=_arch_cfg_override,
        )
        _arch_take_profit_arr   = _arch["tp"]
        _arch_trailing_stop_arr = _arch["stop"]
        _arch_harvest_arr       = _arch["harvest"]
        _arch_min_hold_arr      = _arch["min_hold"]
        _arch_enabled_arr       = _arch["enabled"]
        _arch_score_mult_arr    = _arch["score_mult"]
        _arch_pos_mult_arr      = _arch["pos_mult"]
        _arch_max_sleeve_arr    = _arch["max_sleeve"]
        _arch_min_score_arr     = _arch["min_score"]
        _arch_labels            = _arch["labels"]
    else:
        _arch_take_profit_arr = None
        _arch_trailing_stop_arr = None
        _arch_harvest_arr = None
        _arch_min_hold_arr = None
        _arch_enabled_arr = None
        _arch_score_mult_arr = None
        _arch_pos_mult_arr = None
        _arch_max_sleeve_arr = None
        _arch_min_score_arr = None
        _arch_labels = []
    _arch_pnl: dict = {}
    _arch_trade_counts: dict = {}
    _arch_exit_breakdown: dict = {}
    # Extended rollups
    _arch_buy_archetype: dict[int, str] = {}   # idx → archetype at last buy
    _arch_buy_cost_basis: dict[int, float] = {}  # idx → cost basis at last buy
    _arch_win_count: dict[str, int] = {}
    _arch_completed_sells: dict[str, int] = {}
    _arch_hold_days_sum: dict[str, float] = {}
    _arch_decision_source_counts: dict[str, dict[str, int]] = {}
    _arch_daily_value: dict[str, np.ndarray] = {}
    if archetype_aware:
        from portfolio.position_archetypes import ARCHETYPE_LABELS as _AL
        for _label in _AL:
            _arch_daily_value[_label] = np.zeros(n_days, dtype=np.float64)

    # Cluster concentration tracking (walk-forward, diagnostics only)
    _cluster_snapshots: list = []
    _cluster_labels_by_day: np.ndarray | None = None
    _conc_limits = ARCHETYPE_PARAMS  # fallback; we use concentration_limits separately
    try:
        from util import CONCENTRATION_LIMIT_PARAMS as _CLP
    except Exception:
        _CLP = {}
    _max_cluster_w = float(_CLP.get("max_cluster_weight", 0.35))
    _max_sector_w  = float(_CLP.get("max_sector_weight", 0.40))
    _n_clusters_ct = int(_CLP.get("n_clusters", 6))

    if cluster_tracking:
        try:
            from .cluster_tracker import precompute_cluster_labels
            _rebal_days_for_ct = [d for d in range(0, n_days, rebalance_frequency_days)]
            _cluster_labels_by_day = precompute_cluster_labels(
                precomp, _rebal_days_for_ct, n_clusters=_n_clusters_ct,
            )
            logger.debug("Cluster labels precomputed for %d rebalance days", len(_rebal_days_for_ct))
        except Exception as exc:
            logger.warning("Cluster precomputation failed — tracking disabled: %s", exc)
            cluster_tracking = False

    _cs_params     = cs_params if cs_params is not None else CANDIDATE_SELECTION_PARAMS
    current_scores = score_stocks_at_day(precomp, params, 0)
    candidate_mask, _init_diag = select_candidates(0, current_scores, precomp, params, _cs_params)

    stock_shares    = np.zeros(n_stocks)
    stock_avg_cost  = np.zeros(n_stocks)
    stock_peak      = np.zeros(n_stocks)
    stock_day_bought= np.full(n_stocks, -1, dtype=np.int32)
    stock_day_sold  = np.full(n_stocks, -99, dtype=np.int32)
    stock_stopout   = np.zeros(n_stocks, dtype=bool)
    etf_shares      = np.zeros(n_etfs)

    cash                = float(starting_capital)
    daily_values        = np.zeros(n_days)
    ca_daily_values     = np.zeros(n_days)
    _active_daily       = np.zeros(n_days) if scope == "active_sleeve_compounding" else None
    _ca_active_daily    = np.zeros(n_days) if scope == "active_sleeve_compounding" else None
    total_contributions = float(starting_capital)

    trades_made          = 0
    sells_made           = 0
    trim_count           = 0
    harvest_count        = 0
    skipped_buys         = 0
    cap_reductions       = 0
    cooldown_skips       = 0
    stopout_count        = 0
    trades_this_week     = 0
    week_start_day       = 0
    total_positions_sum  = 0
    max_positions        = 0
    total_cash_pct_sum   = 0.0
    total_friction       = 0.0
    total_traded_notional= 0.0
    regime_days          = {"bullish": 0, "neutral": 0, "defensive": 0}
    trade_log: list      = []

    _cur_day = 0

    def _effective_slippage(stock_idx: int) -> float:
        if not use_vol_slip or precomp.vol_3m_daily is None:
            return base_slippage
        v   = precomp.vol_3m_daily[max(0, min(n_days - 1, _cur_day))]
        vol = float(v[stock_idx]) if np.isfinite(v[stock_idx]) else 0.20
        return base_slippage * (1.0 + vol_slip_mult * vol)

    def _current_portfolio_value(day: int) -> float:
        prices_d = precomp.prices[day]
        valid    = np.isfinite(prices_d) & (prices_d > 0)
        sv = float(np.sum(stock_shares * np.where(valid, prices_d, stock_avg_cost)))
        ev = float(np.sum(etf_shares * precomp.etf_prices[day])) if n_etfs > 0 else 0.0
        return cash + sv + ev

    def _sector_exposures(day: int) -> dict:
        prices_d = precomp.prices[day]
        valid    = np.isfinite(prices_d) & (prices_d > 0)
        exposure: dict = {}
        for i in np.where(stock_shares > 0)[0]:
            s   = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"
            val = float(stock_shares[i] * prices_d[i]) if valid[i] else float(stock_shares[i] * stock_avg_cost[i])
            exposure[s] = exposure.get(s, 0.0) + val
        return exposure

    def _archetype_sleeve_value(label: str, day: int) -> float:
        """Sum stock value across positions tagged to archetype label."""
        if not label or not _arch_labels:
            return 0.0
        prices_d = precomp.prices[day]
        valid    = np.isfinite(prices_d) & (prices_d > 0)
        total = 0.0
        for k in range(len(_arch_labels)):
            if _arch_labels[k] == label and stock_shares[k] > 0:
                p_k = prices_d[k] if valid[k] else stock_avg_cost[k]
                total += float(stock_shares[k] * p_k)
        return total

    def _do_buy(day: int, budget: float) -> float:
        nonlocal cash, trades_made, skipped_buys, cap_reductions, cooldown_skips
        nonlocal total_friction, total_traded_notional, trades_this_week

        if budget < min_order:
            return 0.0
        prices_d = precomp.prices[day]
        eligible = candidate_mask & np.isfinite(prices_d) & (prices_d > 0)

        # Apply per-archetype enabled-flag and min_score_to_buy filter
        if _arch_enabled_arr is not None and _arch_labels:
            eligible = eligible & _arch_enabled_arr
            ms = _arch_min_score_arr
            if ms is not None:
                # NaN means "no min" → always pass; finite values gate on raw score.
                min_gate = np.where(np.isfinite(ms), current_scores >= ms, True)
                eligible = eligible & min_gate

        if not eligible.any():
            return 0.0

        # Effective score (used for both ranking AND budget allocation)
        if _arch_score_mult_arr is not None:
            eff_scores = current_scores * _arch_score_mult_arr
        else:
            eff_scores = current_scores

        total_score = eff_scores[eligible].sum()
        if total_score <= 0:
            return 0.0

        portfolio_value = _current_portfolio_value(day)
        sector_exp      = _sector_exposures(day)
        spent           = 0.0
        buys_this_pass  = 0

        candidate_indices = sorted(np.where(eligible)[0], key=lambda i: -eff_scores[i])

        # Track archetype sleeve consumption within this rebalance pass so we cap
        # against running totals, not the stale day-start value.
        sleeve_consumed: dict[str, float] = {}

        for i in candidate_indices:
            if buys_this_pass >= max_buys or trades_this_week >= max_trades_week:
                skipped_buys += 1
                continue

            days_since_sell = day - int(stock_day_sold[i])
            required_cd     = cooldown_stop if stock_stopout[i] else cooldown_sell
            if days_since_sell < required_cd:
                cooldown_skips += 1
                continue

            alloc = (eff_scores[i] / total_score) * budget

            max_by_cash = cash * max_order_pct
            if alloc > max_by_cash:
                alloc = max_by_cash
                cap_reductions += 1

            # Per-archetype max_position_multiplier cap on max_single_pct
            _pos_mult = float(_arch_pos_mult_arr[i]) if _arch_pos_mult_arr is not None else 1.0
            _eff_max_single = max_single_pct * _pos_mult

            if portfolio_value > 0:
                cur_pos_val = float(stock_shares[i] * prices_d[i])
                room = portfolio_value * _eff_max_single - cur_pos_val
                if room <= 0:
                    skipped_buys += 1
                    continue
                if alloc > room:
                    alloc = room
                    cap_reductions += 1

            if portfolio_value > 0:
                sector     = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"
                cur_sector = sector_exp.get(sector, 0.0)
                sector_room = portfolio_value * max_sector_pct - cur_sector
                if sector_room <= 0:
                    skipped_buys += 1
                    continue
                if alloc > sector_room:
                    alloc = sector_room
                    cap_reductions += 1
            else:
                sector = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"

            # Per-archetype max_active_weight sleeve cap (soft, post-rank)
            _al_i = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
            if (
                _al_i
                and _arch_max_sleeve_arr is not None
                and np.isfinite(_arch_max_sleeve_arr[i])
                and portfolio_value > 0
            ):
                base_sleeve = _archetype_sleeve_value(_al_i, day)
                running     = sleeve_consumed.get(_al_i, 0.0)
                sleeve_cap_room = portfolio_value * float(_arch_max_sleeve_arr[i]) - (base_sleeve + running)
                if sleeve_cap_room <= 0:
                    skipped_buys += 1
                    continue
                if alloc > sleeve_cap_room:
                    alloc = sleeve_cap_room
                    cap_reductions += 1

            if alloc < min_order:
                skipped_buys += 1
                continue

            p               = prices_d[i]
            slip            = _effective_slippage(i)
            effective_price = p * (1.0 + slip)
            shares          = alloc / effective_price
            friction        = alloc * slip + commission_per_trade
            total_friction += friction

            if stock_shares[i] > 0:
                old_cost          = stock_avg_cost[i] * stock_shares[i]
                stock_avg_cost[i] = (old_cost + alloc) / (stock_shares[i] + shares)
            else:
                stock_avg_cost[i]  = effective_price
                stock_day_bought[i]= day
                trades_made       += 1
                trades_this_week  += 1
                sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                _al = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
                _arch_buy_archetype[i] = _al
                _arch_buy_cost_basis[i] = alloc
                trade_log.append(TradeRecord(
                    date=str(day), symbol=sym, side="buy",
                    quantity=shares, price=effective_price, amount=alloc, reason="buy",
                    archetype=_al,
                    archetype_at_entry=_al,
                    archetype_at_exit="",
                    decision_source=("archetype_rule" if _al else "global_rule"),
                ))
            stock_shares[i] += shares
            stock_peak[i]    = max(stock_peak[i], p)
            sector_exp[sector] = sector_exp.get(sector, 0.0) + alloc
            if _al_i:
                sleeve_consumed[_al_i] = sleeve_consumed.get(_al_i, 0.0) + alloc
            spent                 += alloc
            total_traded_notional += alloc
            buys_this_pass        += 1

        cash -= spent
        return spent

    # Day 0: ETF buy + initial stock deployment
    if n_etfs > 0 and index_pct > 0:
        etf_budget = cash * index_pct
        p0_etf     = precomp.etf_prices[0]
        valid_etfs = np.isfinite(p0_etf) & (p0_etf > 0)
        n_valid    = int(valid_etfs.sum())
        if n_valid > 0:
            per_etf = etf_budget / n_valid
            if per_etf >= min_order:
                for j in np.where(valid_etfs)[0]:
                    etf_shares[j] = per_etf / p0_etf[j]
                    cash -= per_etf

    _do_buy(0, cash)

    for d in range(n_days):
        _cur_day = d
        prices   = precomp.prices[d]
        held     = stock_shares > 0

        if d > 0 and (d - week_start_day) >= rebalance_frequency_days:
            trades_this_week = 0
            week_start_day   = d

        valid_price = np.isfinite(prices) & (prices > 0)
        stock_peak  = np.where(held & valid_price, np.maximum(stock_peak, prices), stock_peak)

        with np.errstate(invalid="ignore", divide="ignore"):
            pct_from_avg  = np.where(
                held & (stock_avg_cost > 0) & valid_price,
                prices / stock_avg_cost - 1.0, 0.0,
            )
            pct_from_peak = np.where(
                held & (stock_peak > 0) & valid_price,
                prices / stock_peak - 1.0, 0.0,
            )

        days_held      = np.where(stock_day_bought >= 0, d - stock_day_bought, 0)
        take_profit_ok = current_scores < metric_threshold * _TAKE_PROFIT_FLOOR_MULTIPLIER

        _eff_tp      = _arch_take_profit_arr   if _arch_take_profit_arr   is not None else take_profit_pct
        _eff_stop    = _arch_trailing_stop_arr if _arch_trailing_stop_arr is not None else trailing_stop
        _eff_harvest = _arch_harvest_arr       if _arch_harvest_arr       is not None else _HARVEST_PROFIT_THRESHOLD
        _eff_minhold = _arch_min_hold_arr      if _arch_min_hold_arr      is not None else _MIN_HOLD_DAYS

        stop_loss_mask = held & (pct_from_avg  <= _STOP_LOSS_PCT)
        trail_mask     = held & (pct_from_peak <= _eff_stop)        & (days_held >= _eff_minhold)
        tp_mask        = held & (pct_from_avg  >= _eff_tp)          & take_profit_ok & (days_held >= _MIN_DAYS_BEFORE_TAKE_PROFIT)
        # WATCH/REVIEW logic: live holds positions that are still profitable and
        # not fully collapsed — don't exit on score alone when pnl > 0 and score
        # is above the review floor (mirrors decision_adjustment_engine behaviour).
        _watch_suppressed = (
            _POSITIVE_PNL_DOWNGRADE
            and _REVIEW_SCORE_FLOOR > 0
        )
        if _watch_suppressed:
            _hold_not_exit = (pct_from_avg > 0) & (current_scores >= _REVIEW_SCORE_FLOOR)
        else:
            _hold_not_exit = np.zeros(len(held), dtype=bool)

        weak_val_mask  = (
            held
            & (current_scores < sell_weak_below)
            & (days_held >= _MIN_DAYS_HELD_BEFORE_VALUE_EXIT)
            & ~_hold_not_exit
        )
        sell_mask      = stop_loss_mask | trail_mask | tp_mask | weak_val_mask

        if sell_mask.any():
            sell_prices   = np.where(valid_price, prices, stock_avg_cost)
            sell_notional = float(np.sum(stock_shares[sell_mask] * sell_prices[sell_mask]))
            sell_indices  = np.where(sell_mask)[0]
            for i in sell_indices:
                slip       = _effective_slippage(i)
                proceeds_i = float(stock_shares[i] * sell_prices[i] * (1.0 - slip))
                total_friction += float(stock_shares[i] * sell_prices[i] * slip) + commission_per_trade
                cash += proceeds_i
                is_stopout          = bool(stop_loss_mask[i] or trail_mask[i])
                stock_day_sold[i]   = d
                stock_stopout[i]    = is_stopout
                if is_stopout:
                    stopout_count += 1
                cost_basis = float(stock_avg_cost[i] * stock_shares[i])
                if stop_loss_mask[i]:
                    _exit = "stop_loss"
                    _src  = "global_rule"
                elif trail_mask[i]:
                    _exit = "trailing_stop"
                    _src  = "archetype_rule" if _arch_trailing_stop_arr is not None else "global_rule"
                elif tp_mask[i]:
                    _exit = "take_profit"
                    _src  = "archetype_rule" if _arch_take_profit_arr is not None else "global_rule"
                else:
                    _exit = "weak_value"
                    _src  = "global_rule"
                sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                _al = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
                _al_entry = _arch_buy_archetype.get(int(i), _al)
                _pnl_i = proceeds_i - cost_basis
                trade_log.append(TradeRecord(
                    date=str(d), symbol=sym, side="sell",
                    quantity=float(stock_shares[i]), price=float(sell_prices[i]),
                    amount=proceeds_i, exit_type=_exit,  # type: ignore[arg-type]
                    pnl=_pnl_i, hold_days=int(days_held[i]),
                    archetype=_al,
                    archetype_at_entry=_al_entry,
                    archetype_at_exit=_al,
                    decision_source=_src,
                ))
                if _al:
                    _arch_pnl[_al] = _arch_pnl.get(_al, 0.0) + _pnl_i
                    _arch_trade_counts[_al] = _arch_trade_counts.get(_al, 0) + 1
                    _arch_exit_breakdown.setdefault(_al, {})
                    _arch_exit_breakdown[_al][_exit] = _arch_exit_breakdown[_al].get(_exit, 0) + 1
                    _arch_completed_sells[_al] = _arch_completed_sells.get(_al, 0) + 1
                    _arch_hold_days_sum[_al] = _arch_hold_days_sum.get(_al, 0.0) + float(days_held[i])
                    if _pnl_i > 0:
                        _arch_win_count[_al] = _arch_win_count.get(_al, 0) + 1
                    _arch_decision_source_counts.setdefault(_al, {})
                    _arch_decision_source_counts[_al][_src] = _arch_decision_source_counts[_al].get(_src, 0) + 1
                # Clear per-position entry tracking
                if int(i) in _arch_buy_archetype:
                    del _arch_buy_archetype[int(i)]
                if int(i) in _arch_buy_cost_basis:
                    del _arch_buy_cost_basis[int(i)]
            total_traded_notional += sell_notional
            sells_made            += int(sell_mask.sum())
            stock_shares[sell_mask]    = 0.0
            stock_avg_cost[sell_mask]  = 0.0
            stock_peak[sell_mask]      = 0.0
            stock_day_bought[sell_mask]= -1

        # ── Harvest exits (partial profit-taking — mirrors live HARVEST action) ─
        # Triggered when pct_from_avg >= harvest_profit_threshold.
        # Sells harvest_fraction (default 40%) of the position and routes
        # proceeds to ETFs, matching the live HarvestManager behaviour.
        harvest_mask = (
            held
            & ~sell_mask
            & (pct_from_avg >= _eff_harvest)
            & (days_held >= _eff_minhold)
        )
        if harvest_mask.any():
            harv_prices   = np.where(valid_price, prices, stock_avg_cost)
            harv_indices  = np.where(harvest_mask)[0]
            harv_notional = 0.0
            _harv_src = "archetype_rule" if _arch_harvest_arr is not None else "global_rule"
            for i in harv_indices:
                shares_to_sell = stock_shares[i] * _HARVEST_FRACTION
                if shares_to_sell <= 0:
                    continue
                slip       = _effective_slippage(i)
                proceeds_i = float(shares_to_sell * harv_prices[i] * (1.0 - slip))
                total_friction += float(shares_to_sell * harv_prices[i] * slip) + commission_per_trade
                harv_notional  += proceeds_i
                cash           += proceeds_i
                stock_shares[i] -= shares_to_sell
                sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                cost_basis_i = float(stock_avg_cost[i] * shares_to_sell)
                _al = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
                _al_entry = _arch_buy_archetype.get(int(i), _al)
                _pnl_i = proceeds_i - cost_basis_i
                trade_log.append(TradeRecord(
                    date=str(d), symbol=sym, side="sell",
                    quantity=shares_to_sell, price=float(harv_prices[i]),
                    amount=proceeds_i, exit_type="harvest_exit",
                    pnl=_pnl_i, hold_days=int(days_held[i]), is_partial=True,
                    archetype=_al,
                    archetype_at_entry=_al_entry,
                    archetype_at_exit=_al,
                    decision_source=_harv_src,
                ))
                if _al:
                    _arch_pnl[_al] = _arch_pnl.get(_al, 0.0) + _pnl_i
                    _arch_trade_counts[_al] = _arch_trade_counts.get(_al, 0) + 1
                    _arch_exit_breakdown.setdefault(_al, {})
                    _arch_exit_breakdown[_al]["harvest_exit"] = _arch_exit_breakdown[_al].get("harvest_exit", 0) + 1
                    _arch_decision_source_counts.setdefault(_al, {})
                    _arch_decision_source_counts[_al][_harv_src] = _arch_decision_source_counts[_al].get(_harv_src, 0) + 1
            harvest_count         += int(harvest_mask.sum())
            total_traded_notional += harv_notional
            sells_made            += int(harvest_mask.sum())
            # Route harvest proceeds to ETF sleeve (same as live HarvestManager)
            etf_portion = harv_notional * _trim_to_etfs_pct
            if n_etfs > 0 and etf_portion > 0:
                p_etf     = precomp.etf_prices[d]
                valid_etfs = np.isfinite(p_etf) & (p_etf > 0)
                n_valid_e  = int(valid_etfs.sum())
                if n_valid_e > 0:
                    per_etf = etf_portion / n_valid_e
                    if per_etf >= min_order:
                        for j in np.where(valid_etfs)[0]:
                            etf_shares[j] += per_etf / p_etf[j]
                            cash           -= per_etf

        # ── Trim exits (partial position reduction) ──────────────────────────
        if _trim_enabled:
            trim_mask = (
                held
                & ~sell_mask
                & ~harvest_mask
                & (pct_from_avg >= _trim_min_gain)
                & (current_scores >= sell_weak_below)
                & (current_scores < _trim_score_below)
            )
            if trim_mask.any():
                sell_prices_t = np.where(valid_price, prices, stock_avg_cost)
                trim_indices  = np.where(trim_mask)[0]
                trim_notional = 0.0
                for i in trim_indices:
                    shares_to_sell = stock_shares[i] * _trim_fraction
                    if shares_to_sell <= 0:
                        continue
                    slip        = _effective_slippage(i)
                    proceeds_i  = float(shares_to_sell * sell_prices_t[i] * (1.0 - slip))
                    total_friction += float(shares_to_sell * sell_prices_t[i] * slip) + commission_per_trade
                    trim_notional  += proceeds_i
                    cash           += proceeds_i
                    stock_shares[i] -= shares_to_sell
                    sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                    cost_basis_i = float(stock_avg_cost[i] * shares_to_sell)
                    _al = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
                    _al_entry = _arch_buy_archetype.get(int(i), _al)
                    _pnl_i = proceeds_i - cost_basis_i
                    trade_log.append(TradeRecord(
                        date=str(d), symbol=sym, side="sell",
                        quantity=shares_to_sell, price=float(sell_prices_t[i]),
                        amount=proceeds_i, exit_type="trim_exit",
                        pnl=_pnl_i, hold_days=int(days_held[i]), is_partial=True,
                        archetype=_al,
                        archetype_at_entry=_al_entry,
                        archetype_at_exit=_al,
                        decision_source="global_rule",
                    ))
                    if _al:
                        _arch_pnl[_al] = _arch_pnl.get(_al, 0.0) + _pnl_i
                        _arch_trade_counts[_al] = _arch_trade_counts.get(_al, 0) + 1
                        _arch_exit_breakdown.setdefault(_al, {})
                        _arch_exit_breakdown[_al]["trim_exit"] = _arch_exit_breakdown[_al].get("trim_exit", 0) + 1
                        _arch_decision_source_counts.setdefault(_al, {})
                        _arch_decision_source_counts[_al]["global_rule"] = _arch_decision_source_counts[_al].get("global_rule", 0) + 1
                trim_count            += int(trim_mask.sum())
                total_traded_notional += trim_notional
                sells_made            += int(trim_mask.sum())
                # Route trim proceeds to ETF sleeve
                etf_portion = trim_notional * _trim_to_etfs_pct
                if n_etfs > 0 and etf_portion > 0:
                    p_etf     = precomp.etf_prices[d]
                    valid_etfs = np.isfinite(p_etf) & (p_etf > 0)
                    n_valid_e  = int(valid_etfs.sum())
                    if n_valid_e > 0:
                        per_etf = etf_portion / n_valid_e
                        if per_etf >= min_order:
                            for j in np.where(valid_etfs)[0]:
                                etf_shares[j] += per_etf / p_etf[j]
                                cash           -= per_etf

        is_contrib_day       = d > 0 and d % rebalance_frequency_days == 0
        contrib_today        = weekly_contribution if is_contrib_day else 0.0
        active_contrib_today = (
            weekly_contribution * (1.0 - index_pct)
            if (is_contrib_day and _ca_active_daily is not None)
            else 0.0
        )
        if cluster_tracking and is_contrib_day and _cluster_labels_by_day is not None:
            try:
                from .cluster_tracker import record_cluster_snapshot
                _rebal_idx = d // rebalance_frequency_days
                if _rebal_idx < _cluster_labels_by_day.shape[0]:
                    _cls_labels_d = _cluster_labels_by_day[_rebal_idx]
                    _snap = record_cluster_snapshot(
                        day=d,
                        stock_shares=stock_shares,
                        prices=np.where(valid_price, precomp.prices[d], stock_avg_cost),
                        cluster_labels=_cls_labels_d,
                        sector_labels=precomp.sector_labels,
                        max_cluster_weight_threshold=_max_cluster_w,
                        max_sector_weight_threshold=_max_sector_w,
                    )
                    _cluster_snapshots.append(_snap)
            except Exception as exc:
                logger.debug("Cluster snapshot failed at day %d: %s", d, exc)
        if is_contrib_day:
            current_scores = score_stocks_at_day(precomp, params, d)
            candidate_mask, _ = select_candidates(d, current_scores, precomp, params, _cs_params)
            cash              += weekly_contribution
            total_contributions += weekly_contribution

            if index_pct > 0 and n_etfs > 0:
                etf_contrib = weekly_contribution * index_pct
                p_etf       = precomp.etf_prices[d]
                valid_etfs  = np.isfinite(p_etf) & (p_etf > 0)
                n_valid_e   = int(valid_etfs.sum())
                if n_valid_e > 0:
                    per_etf_c = etf_contrib / n_valid_e
                    if per_etf_c >= min_order:
                        for j in np.where(valid_etfs)[0]:
                            etf_shares[j] += per_etf_c / p_etf[j]
                            cash           -= per_etf_c

            if cash >= min_order:
                _do_buy(d, cash)

        etf_value   = float(np.sum(etf_shares * precomp.etf_prices[d])) if n_etfs > 0 else 0.0
        stock_value = float(np.sum(stock_shares * np.where(valid_price, prices, stock_avg_cost)))
        port_val    = cash + stock_value + etf_value
        daily_values[d] = port_val
        if _active_daily is not None:
            _active_daily[d] = port_val - etf_value

        if accounting_trace is not None:
            accounting_trace.append({
                "d": d,
                "cash": cash,
                "stock_value": stock_value,
                "etf_value": etf_value,
                "port_val": port_val,
                "active_equity": port_val - etf_value,
            })

        if d == 0:
            ca_daily_values[0] = port_val
            if _ca_active_daily is not None:
                _ca_active_daily[0] = port_val - etf_value
        else:
            prev_port          = daily_values[d - 1]
            factor             = (port_val - contrib_today) / max(prev_port, 1e-9)
            ca_daily_values[d] = ca_daily_values[d - 1] * factor
            if _ca_active_daily is not None:
                prev_active        = _active_daily[d - 1]
                active_equity_d    = port_val - etf_value
                a_factor           = (active_equity_d - active_contrib_today) / max(prev_active, 1e-9)
                _ca_active_daily[d] = _ca_active_daily[d - 1] * a_factor

        bench_p = precomp.benchmark_prices
        if d >= 200 and np.isfinite(bench_p[d]) and bench_p[d] > 0:
            ma200 = float(np.nanmean(bench_p[max(0, d - 199): d + 1]))
            if bench_p[d] < ma200 * 0.95:
                regime_days["defensive"] += 1
            elif bench_p[d] < ma200:
                regime_days["neutral"] += 1
            else:
                regime_days["bullish"] += 1
        else:
            regime_days["bullish"] += 1

        n_pos = int((stock_shares > 0).sum())
        total_positions_sum += n_pos
        max_positions        = max(max_positions, n_pos)
        total_cash_pct_sum  += (cash / max(port_val, 1e-9))

        # Per-archetype daily sleeve value (used for max DD + final sleeve weight)
        if archetype_aware and _arch_labels:
            for k in range(len(_arch_labels)):
                _ak = _arch_labels[k]
                if not _ak or stock_shares[k] <= 0:
                    continue
                _p = prices[k] if valid_price[k] else stock_avg_cost[k]
                _arch_daily_value[_ak][d] += float(stock_shares[k] * _p)

    final_value = float(daily_values[-1])
    metrics     = compute_performance_metrics(ca_daily_values)
    avg_port    = float(daily_values[daily_values > 0].mean()) if daily_values.any() else starting_capital
    turnover    = total_traded_notional / max(avg_port, 1.0)
    profit      = final_value - total_contributions

    bench_twr_val  = 0.0
    bench_equity   = np.array([])
    bench_ca_equity = np.array([])
    if np.isfinite(precomp.benchmark_prices).all() and precomp.benchmark_prices[0] > 0:
        bench_twr_val = _bench_twr(
            precomp.benchmark_prices, starting_capital,
            weekly_contribution, rebalance_frequency_days,
        )
        bench_equity = _bench_daily_equity(
            precomp.benchmark_prices, starting_capital,
            weekly_contribution, rebalance_frequency_days,
        )
        bench_ca_equity = _bench_daily_ca_equity(
            precomp.benchmark_prices, starting_capital,
            weekly_contribution, rebalance_frequency_days,
        )

    # ── Per-archetype rollups ───────────────────────────────────────────
    _arch_active_excess: dict[str, float] = {}
    _arch_win_rate: dict[str, float] = {}
    _arch_avg_hold: dict[str, float] = {}
    _arch_max_dd: dict[str, float] = {}
    _arch_sleeve_weight: dict[str, float] = {}
    _arch_realized_pnl: dict[str, float] = dict(_arch_pnl)
    _arch_unrealized_pnl: dict[str, float] = {}

    if archetype_aware and _arch_daily_value:
        last_port_val = float(daily_values[-1]) if n_days > 0 else 0.0
        # Unrealized PnL: open positions valued at last-day prices
        last_prices = precomp.prices[-1] if n_days > 0 else None
        if last_prices is not None:
            valid_last = np.isfinite(last_prices) & (last_prices > 0)
            for i in range(len(_arch_labels)):
                _al = _arch_labels[i]
                if not _al or stock_shares[i] <= 0:
                    continue
                _p = last_prices[i] if valid_last[i] else stock_avg_cost[i]
                _val = float(stock_shares[i] * _p)
                _cost = float(stock_shares[i] * stock_avg_cost[i])
                _arch_unrealized_pnl[_al] = _arch_unrealized_pnl.get(_al, 0.0) + (_val - _cost)
        for _al, _series in _arch_daily_value.items():
            # Sleeve weight at end of sim
            if last_port_val > 0:
                _arch_sleeve_weight[_al] = float(_series[-1] / last_port_val)
            else:
                _arch_sleeve_weight[_al] = 0.0
            # Max drawdown of the sleeve value series (ignoring leading zeros)
            _nz = _series[_series > 0]
            if len(_nz) >= 2:
                _peak = np.maximum.accumulate(_nz)
                _dd = (_nz / _peak) - 1.0
                _arch_max_dd[_al] = float(_dd.min())
            else:
                _arch_max_dd[_al] = 0.0
        # Win rate, avg hold days, active excess
        for _al, _n in _arch_completed_sells.items():
            if _n > 0:
                _arch_win_rate[_al] = float(_arch_win_count.get(_al, 0)) / float(_n)
                _arch_avg_hold[_al] = float(_arch_hold_days_sum.get(_al, 0.0)) / float(_n)
        # Active excess: (sleeve realized + unrealized) / sleeve deployed_capital - bench_twr_val
        # Approximate deployed capital as average non-zero sleeve value.
        for _al, _series in _arch_daily_value.items():
            _nz = _series[_series > 0]
            avg_deployed = float(_nz.mean()) if len(_nz) else 0.0
            total_pnl = _arch_realized_pnl.get(_al, 0.0) + _arch_unrealized_pnl.get(_al, 0.0)
            if avg_deployed > 0:
                sleeve_return = total_pnl / avg_deployed
            else:
                sleeve_return = 0.0
            _arch_active_excess[_al] = float(sleeve_return - bench_twr_val)

    _active_total_return: float | None = None
    _active_sharpe:       float | None = None
    _active_calmar:       float | None = None
    _active_max_drawdown: float | None = None
    _active_excess_return: float | None = None
    _active_equity_curve: np.ndarray | None = None
    if _active_daily is not None and _active_daily[0] > 0:
        _am = compute_performance_metrics(_ca_active_daily)
        _active_equity_curve  = _ca_active_daily.copy()
        _active_total_return  = _am["total_return"]
        _active_sharpe        = _am["sharpe"]
        _active_calmar        = _am["calmar"]
        _active_max_drawdown  = _am["max_drawdown"]
        _active_excess_return = _active_total_return - bench_twr_val

    return SimResult(
        final_value=final_value,
        total_return=metrics["total_return"],
        sharpe=metrics["sharpe"],
        calmar=metrics["calmar"],
        max_drawdown=metrics["max_drawdown"],
        trades_made=trades_made,
        sells_made=sells_made,
        skipped_buys=skipped_buys,
        cap_reductions=cap_reductions,
        average_positions=float(total_positions_sum / max(n_days, 1)),
        max_positions=max_positions,
        average_cash_pct=float(total_cash_pct_sum / max(n_days, 1)),
        turnover_estimate=turnover,
        friction_cost=total_friction,
        net_contributions=total_contributions,
        profit=profit,
        stopout_count=stopout_count,
        trim_count=trim_count,
        harvest_count=harvest_count,
        cooldown_skips=cooldown_skips,
        regime_days=regime_days,
        benchmark_twr=bench_twr_val,
        pool_diagnostics=_init_diag,
        trade_log=trade_log,
        equity_curve=daily_values.copy(),
        benchmark_equity=bench_equity,
        benchmark_ca_equity=bench_ca_equity,
        archetype_pnl=_arch_pnl,
        archetype_trade_counts=_arch_trade_counts,
        archetype_exit_breakdown=_arch_exit_breakdown,
        archetype_active_excess=_arch_active_excess,
        archetype_win_rate=_arch_win_rate,
        archetype_avg_hold_days=_arch_avg_hold,
        archetype_max_drawdown=_arch_max_dd,
        archetype_sleeve_weight=_arch_sleeve_weight,
        archetype_realized_pnl=_arch_realized_pnl,
        archetype_unrealized_pnl=_arch_unrealized_pnl,
        archetype_decision_source_counts=_arch_decision_source_counts,
        cluster_result=_build_cluster_result_if_needed(
            _cluster_snapshots, _n_clusters_ct
        ) if cluster_tracking else None,
        scope=scope,
        active_equity_curve=_active_equity_curve,
        active_total_return=_active_total_return,
        active_sharpe=_active_sharpe,
        active_calmar=_active_calmar,
        active_max_drawdown=_active_max_drawdown,
        active_excess_return=_active_excess_return,
    )


def _build_cluster_result_if_needed(snapshots, n_clusters):
    if not snapshots:
        return None
    try:
        from .cluster_tracker import build_cluster_result
        return build_cluster_result(snapshots, n_clusters=n_clusters)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report orchestration
# ---------------------------------------------------------------------------

def run_backtest_report(
    precomp: PrecomputedData,
    params: np.ndarray | None,
    train_slice: slice,
    val_slice: slice | None,
    cluster_tracking: bool = False,
    scope: str = "overall_strategy",
) -> BacktestReport:
    """
    Run strategy on train window, optionally evaluate on validation window.
    params=None uses current config values via get_default_params().
    """
    if params is None:
        params = get_default_params()
    bp = BACKTEST_PARAMS
    _arch_enabled = bool(ARCHETYPE_PARAMS.get("enabled", False))

    def _slice_precomp(s: slice) -> PrecomputedData:
        def _opt(arr):
            return arr[s] if arr is not None else None
        return precomp._replace(
            prices=precomp.prices[s],
            etf_prices=precomp.etf_prices[s],
            benchmark_prices=precomp.benchmark_prices[s],
            position_52w_daily=precomp.position_52w_daily[s],
            return_1m_daily=precomp.return_1m_daily[s],
            bin_indices_daily=precomp.bin_indices_daily[s],
            has_position_52w_daily=precomp.has_position_52w_daily[s],
            ret_5d_daily=_opt(precomp.ret_5d_daily),
            ret_3m_daily=_opt(precomp.ret_3m_daily),
            ret_6m_daily=_opt(precomp.ret_6m_daily),
            rs_3m_daily=_opt(precomp.rs_3m_daily),
            rs_6m_daily=_opt(precomp.rs_6m_daily),
            vol_3m_daily=_opt(precomp.vol_3m_daily),
            above_50dma_daily=_opt(precomp.above_50dma_daily),
            above_200dma_daily=_opt(precomp.above_200dma_daily),
        )

    train_precomp = _slice_precomp(train_slice)
    train_n       = train_precomp.prices.shape[0]
    _sim_kwargs = dict(
        starting_capital=bp["starting_capital"],
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
        archetype_aware=_arch_enabled,
        cluster_tracking=cluster_tracking,
        scope=scope,
    )
    train_result  = run_simulation(train_precomp, params, **_sim_kwargs)

    val_result: SimResult | None = None
    if val_slice is not None:
        val_precomp = _slice_precomp(val_slice)
        if val_precomp.prices.shape[0] >= 5:
            val_result = run_simulation(val_precomp, params, **_sim_kwargs)

    def _bench_metrics(price_slice: slice) -> dict:
        vals = precomp.benchmark_prices[price_slice]
        if len(vals) >= 2 and np.isfinite(vals).all() and vals[0] > 0:
            arr = vals / vals[0] * bp["starting_capital"]
            return compute_performance_metrics(arr)
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    def _twr_for_slice(s: slice) -> float:
        vals = precomp.benchmark_prices[s]
        if len(vals) >= 2 and np.isfinite(vals).all() and vals[0] > 0:
            return _bench_twr(vals, bp["starting_capital"],
                              bp["weekly_contribution"], bp["rebalance_frequency_days"])
        return 0.0

    train_bench     = _bench_metrics(train_slice)
    val_bench_return = 0.0
    if val_slice is not None:
        val_bench_return = _bench_metrics(val_slice)["total_return"]

    train_bench_twr = _twr_for_slice(train_slice)
    val_bench_twr   = _twr_for_slice(val_slice) if val_slice is not None else 0.0
    excess          = train_result.total_return - train_bench["total_return"]

    notes: list[str] = [f"Lookahead bias: {precomp.lookahead_bias_level}"]
    if precomp.lookahead_bias_level == "HIGH":
        notes.append("WARNING: universe selected by current value_metric — results not predictive")
    if _arch_enabled:
        notes.append("Archetype-aware exit thresholds active")

    import datetime
    import hashlib
    import json
    try:
        from util import _app as _cfg_dict
        _cfg_hash = hashlib.sha256(
            json.dumps(_cfg_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
    except Exception:
        _cfg_hash = ""
    _ts = datetime.datetime.utcnow().isoformat(timespec="seconds")

    return BacktestReport(
        mode=precomp.mode,
        universe_selection=precomp.universe_selection,
        lookahead_bias_level=precomp.lookahead_bias_level,
        n_symbols=len(precomp.symbols),
        n_days=train_n,
        train_result=train_result,
        validation_result=val_result,
        benchmark_return=train_bench["total_return"],
        benchmark_sharpe=train_bench["sharpe"],
        benchmark_max_drawdown=train_bench["max_drawdown"],
        excess_return=excess,
        validation_benchmark_return=val_bench_return,
        notes=notes,
        train_benchmark_twr=train_bench_twr,
        val_benchmark_twr=val_bench_twr,
        trade_log=train_result.trade_log,
        config_hash=_cfg_hash,
        run_timestamp=_ts,
    )


def compare_archetype_modes(
    precomp: PrecomputedData,
    params: np.ndarray | None = None,
) -> dict:
    """
    Run the same precomputed window twice — uniform thresholds vs archetype-aware —
    and return a comparison dict with keys ``uniform`` and ``archetype_aware``,
    each holding a SimResult, plus ``_delta`` sub-keys for the key deltas.

    Intended for: ``daily-investor backtest N --archetype-compare``
    """
    if params is None:
        params = get_default_params()

    bp = BACKTEST_PARAMS
    sim_kwargs = dict(
        starting_capital       = bp["starting_capital"],
        slippage_bps           = bp["slippage_bps"],
        commission_per_trade   = bp["commission_per_trade"],
        weekly_contribution    = bp["weekly_contribution"],
        rebalance_frequency_days = bp["rebalance_frequency_days"],
    )

    n_days = precomp.prices.shape[0]
    train_slice, _ = split_price_window(n_days, bp.get("train_pct", 0.70))

    def _slice(s: slice) -> PrecomputedData:
        def _o(a): return a[s] if a is not None else None
        return precomp._replace(
            prices=precomp.prices[s],
            etf_prices=precomp.etf_prices[s],
            benchmark_prices=precomp.benchmark_prices[s],
            position_52w_daily=precomp.position_52w_daily[s],
            return_1m_daily=precomp.return_1m_daily[s],
            bin_indices_daily=precomp.bin_indices_daily[s],
            has_position_52w_daily=precomp.has_position_52w_daily[s],
            ret_5d_daily=_o(precomp.ret_5d_daily),
            ret_3m_daily=_o(precomp.ret_3m_daily),
            ret_6m_daily=_o(precomp.ret_6m_daily),
            rs_3m_daily=_o(precomp.rs_3m_daily),
            rs_6m_daily=_o(precomp.rs_6m_daily),
            vol_3m_daily=_o(precomp.vol_3m_daily),
            above_50dma_daily=_o(precomp.above_50dma_daily),
            above_200dma_daily=_o(precomp.above_200dma_daily),
        )

    train = _slice(train_slice)
    bench_vals = precomp.benchmark_prices[train_slice]
    bench_ret  = float(bench_vals[-1] / bench_vals[0] - 1.0) if (
        len(bench_vals) >= 2 and np.isfinite(bench_vals).all() and bench_vals[0] > 0
    ) else 0.0

    uniform  = run_simulation(train, params, archetype_aware=False, **sim_kwargs)
    arch     = run_simulation(train, params, archetype_aware=True,  **sim_kwargs)

    def _d(a, b): return round(b - a, 5) if (a is not None and b is not None) else None

    return {
        "uniform":         uniform,
        "archetype_aware": arch,
        "_benchmark_return": bench_ret,
        "_n_days":           train.prices.shape[0],
        "_delta": {
            "total_return": _d(uniform.total_return,   arch.total_return),
            "sharpe":       _d(uniform.sharpe,         arch.sharpe),
            "calmar":       _d(uniform.calmar,         arch.calmar),
            "max_drawdown": _d(uniform.max_drawdown,   arch.max_drawdown),
            "trades_made":  _d(uniform.trades_made,    arch.trades_made),
        },
    }


def compare_candidate_selection_modes(
    precomp: PrecomputedData,
    params: np.ndarray | None = None,
) -> dict:
    """
    Run the same precomputed data through three candidate selection modes and
    return a comparison dict: A_absolute, B_percentile, C_percentile_gates.
    """
    if params is None:
        params = get_default_params()

    bp = BACKTEST_PARAMS
    sim_kwargs = dict(
        starting_capital=bp["starting_capital"],
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )

    cs_base  = CANDIDATE_SELECTION_PARAMS
    _GATE_OFF = -999.0

    mode_configs = {
        "A_absolute": {
            "mode": "absolute",
            "top_percentile": cs_base["top_percentile"],
            "max_candidates": 999,
            "min_candidates": 0,
            "use_absolute_score_floor": False,
            "absolute_score_floor": _GATE_OFF,
            "min_quality_score": _GATE_OFF,
            "min_momentum_score": _GATE_OFF,
            "min_conditional_momentum_score": _GATE_OFF,
            "allow_income_defensive_exception": False,
        },
        "B_percentile": {
            "mode": "percentile",
            "top_percentile": cs_base["top_percentile"],
            "max_candidates": cs_base["max_candidates"],
            "min_candidates": cs_base["min_candidates"],
            "use_absolute_score_floor": cs_base["use_absolute_score_floor"],
            "absolute_score_floor": cs_base["absolute_score_floor"],
            "min_quality_score": _GATE_OFF,
            "min_momentum_score": _GATE_OFF,
            "min_conditional_momentum_score": _GATE_OFF,
            "allow_income_defensive_exception": False,
        },
        "C_percentile_gates": dict(cs_base),
    }

    n_days      = precomp.prices.shape[0]
    train_slice, _ = split_price_window(n_days, bp.get("train_pct", 0.70))

    def _slice(s: slice) -> PrecomputedData:
        def _o(a): return a[s] if a is not None else None
        return precomp._replace(
            prices=precomp.prices[s],
            etf_prices=precomp.etf_prices[s],
            benchmark_prices=precomp.benchmark_prices[s],
            position_52w_daily=precomp.position_52w_daily[s],
            return_1m_daily=precomp.return_1m_daily[s],
            bin_indices_daily=precomp.bin_indices_daily[s],
            has_position_52w_daily=precomp.has_position_52w_daily[s],
            ret_5d_daily=_o(precomp.ret_5d_daily),
            ret_3m_daily=_o(precomp.ret_3m_daily),
            ret_6m_daily=_o(precomp.ret_6m_daily),
            rs_3m_daily=_o(precomp.rs_3m_daily),
            rs_6m_daily=_o(precomp.rs_6m_daily),
            vol_3m_daily=_o(precomp.vol_3m_daily),
            above_50dma_daily=_o(precomp.above_50dma_daily),
            above_200dma_daily=_o(precomp.above_200dma_daily),
        )

    train_precomp = _slice(train_slice)
    bench_vals    = precomp.benchmark_prices[train_slice]
    bench_ret     = float(bench_vals[-1] / bench_vals[0] - 1.0) if (
        len(bench_vals) >= 2 and np.isfinite(bench_vals).all() and bench_vals[0] > 0
    ) else 0.0

    results: dict = {"_benchmark_return": bench_ret, "_n_days": train_precomp.prices.shape[0]}
    for label, cs in mode_configs.items():
        sim      = run_simulation(train_precomp, params, cs_params=cs, **sim_kwargs)
        _, pool_diag = select_candidates(0, score_stocks_at_day(train_precomp, params, 0), train_precomp, params, cs)
        results[label] = {"sim": sim, "pool": pool_diag}

    return results
