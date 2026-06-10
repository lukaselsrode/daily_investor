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
    HARVEST_PARAMS,
    INDEX_PCT,
    METRIC_THRESHOLD,
    REGIME_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
    SLEEVE_VOL_OVERLAY,
)

from .types import BacktestReport, CandidatePoolDiagnostics, PrecomputedData, SimResult

# Local aliases for nested scoring sub-blocks.
MOMENTUM_WARMUP_PARAMS = SCORING_PARAMS["momentum_warmup"]
MOMENTUM_INPUT_PARAMS = SCORING_PARAMS["momentum_inputs"]
# Warm-up bin scores (per 52w-position bin), shared with data_loader's day-0 scoring.
# These — NOT the params[10:16] momentum sub-weights — are what
# _momentum_score_warmup_vec indexes by bin_indices.
_MBIN_SCORES = np.array(MOMENTUM_WARMUP_PARAMS["position_bin_scores"], dtype=np.float64)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level sell constants
# ---------------------------------------------------------------------------

# Read from config so live (sell_engine.py) and backtest can never diverge.
_TAKE_PROFIT_FLOOR_MULTIPLIER = float(SELL_RULES.get("take_profit_value_floor_multiplier", 1.2))
# Read stop-loss from config so live and backtest always stay in sync
_STOP_LOSS_PCT = float(SELL_RULES.get("stop_loss_pct", -0.20))
# Archetype `allow_deeper_drawdown`: widen the catastrophic hard stop for flagged
# archetypes so high-conviction names get room to breathe before a failure exit.
_DEEPER_DD_FACTOR = float(SELL_RULES.get("allow_deeper_drawdown_factor", 1.5))
# Archetype `thesis_exit_requires_confirmation`: a soft weak-value exit must persist
# this many consecutive evaluations before it fires for a flagged archetype.
_THESIS_CONFIRM_EVALS = int(SELL_RULES.get("thesis_exit_confirm_evals", 2))
_MIN_DAYS_HELD_BEFORE_VALUE_EXIT = SELL_RULES.get("min_days_held_before_value_exit", 21)
_MIN_HOLD_DAYS = RISK_LIMITS.get("minimum_hold_days", 0)
_MIN_DAYS_BEFORE_TAKE_PROFIT = SELL_RULES.get("minimum_days_before_take_profit", 0)

# Harvest partial exit — mirrors live HARVEST action (partial profit-taking)
_HARVEST_PROFIT_THRESHOLD = float(EXIT_DECISION_PARAMS.get("harvest_profit_threshold", 0.30))
_HARVEST_FRACTION         = float(EXIT_DECISION_PARAMS.get("harvest_fraction",          0.40))


# ---------------------------------------------------------------------------
# Public entry: default params
# ---------------------------------------------------------------------------

def get_default_params() -> np.ndarray:
    """
    Build the 16-element params vector base from the current config values.

    Layout mirrors tuner._current_params():
      [0-3]  score_weights (value, quality, income, momentum)
      [4]    index_pct
      [5]    metric_threshold
      [6]    take_profit_pct
      [7]    sell_weak_value_below
      [8]    trailing_stop_pct
      [9]    value_pe_weight
      [10-15] momentum input sub-weights (rs_3m, rs_6m, risk_adj_3m, trend_structure, return_1m, return_5d)
    """
    sw  = SCORE_WEIGHTS
    v2w = MOMENTUM_INPUT_PARAMS.get("weights", {})
    return np.array([
        sw["value"], sw["quality"], sw["income"], sw["momentum"],
        INDEX_PCT,
        METRIC_THRESHOLD,
        SELL_RULES["take_profit_pct"],
        SELL_RULES["sell_weak_value_below"],
        SELL_RULES["trailing_stop_pct"],
        SCORING_PARAMS["factors"]["value"]["pe_weight"],
        v2w.get("rs_3m",           0.25),
        v2w.get("rs_6m",           0.25),
        v2w.get("risk_adj_3m",     0.20),
        v2w.get("trend_structure", 0.15),
        v2w.get("return_1m",       0.10),
        v2w.get("return_5d",       0.05),
    ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col_arr(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).values.astype(np.float64)
    return np.full(len(df), default, dtype=np.float64)


def _momentum_score_warmup_vec(
    bin_indices: np.ndarray,
    has_pos: np.ndarray,
    pos_52w: np.ndarray,
    return_1m: np.ndarray,
    mbin_scores: np.ndarray,
) -> np.ndarray:
    """Vectorized warm-up bin momentum scoring — used during the first ~63 trading
    days when rolling momentum-input features (rs_3m, rs_6m, …) aren't yet stable."""
    mp = MOMENTUM_WARMUP_PARAMS
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


def _thesis_intact_vec(
    qual: np.ndarray,
    mom: np.ndarray,
    pnl: np.ndarray,
    rank01: np.ndarray,
) -> np.ndarray:
    """
    Vectorized replica of exit_analysis._thesis_intact_score (0–1).
    Higher = thesis still looks valid despite the score-below-threshold exit signal.
    Components mirror the live scalar formula exactly; only finite components count.
    quality/momentum in [-1, 1]; pnl is fractional P/L; rank01 is the universe
    percentile rank in [0, 1].
    """
    acc   = np.zeros_like(pnl, dtype=np.float64)
    denom = np.zeros_like(pnl, dtype=np.float64)

    q_fin = np.isfinite(qual)
    q_norm = np.clip((qual + 0.5) / 1.5, 0.0, 1.0)
    acc   += np.where(q_fin, q_norm, 0.0)
    denom += q_fin

    m_fin = np.isfinite(mom)
    m_norm = np.where(mom >= 0.10, 1.0,
              np.where(mom >= 0.0, 0.6, np.maximum(0.0, 0.5 + mom)))
    acc   += np.where(m_fin, m_norm, 0.0)
    denom += m_fin

    p_fin = np.isfinite(pnl)
    r_norm = np.where(pnl >= 0.05, 1.0,
              np.where(pnl >= 0.0, 0.7, np.maximum(0.0, 0.5 + pnl * 2.0)))
    acc   += np.where(p_fin, r_norm, 0.0)
    denom += p_fin

    k_fin = np.isfinite(rank01)
    acc   += np.where(k_fin, rank01, 0.0)
    denom += k_fin

    with np.errstate(invalid="ignore", divide="ignore"):
        tis = np.where(denom > 0, acc / denom, 0.5)
    return np.round(tis, 3)


def _progress_vec(
    prices: np.ndarray,
    peak: np.ndarray,
    mom: np.ndarray,
    reclaim_band: float,
    mom_floor: float,
) -> np.ndarray:
    """
    Vectorized replica of exit_analysis.is_progress — did each position make
    PROGRESS today? Progress = fresh high within `reclaim_band` of peak OR momentum
    at/above `mom_floor`. A progressing position resets its stall clock (never
    culled). Cross-checked against the scalar in tests/test_opportunity_cost_fidelity.py.
    NaN price/peak/momentum contribute no progress on that term (match the scalar).
    """
    with np.errstate(invalid="ignore"):
        price_term = (peak > 0.0) & np.isfinite(prices) & (prices >= peak * (1.0 - reclaim_band))
        mom_term   = mom >= mom_floor
    return price_term | mom_term


def _dae_soft_exit_full_exit(
    cand: np.ndarray,
    snw: np.ndarray,
    pnl: np.ndarray,
    mom: np.ndarray,
    qual: np.ndarray,
    tis: np.ndarray,
    floors: dict,
) -> np.ndarray:
    """
    Vectorized replica of DecisionAdjustmentEngine._evaluate_soft_exit, restricted
    to the *full-exit* verdict over the score-below-threshold candidate set `cand`.

    Mirrors the live decision tree precedence exactly:
      1. confirmed_breakdown                          → EXIT  (full)
      2. HARVEST (large gain + momentum/thesis alive) → downgrade (NOT a full exit)
      3. TRIM    (moderate gain + positive momentum)  → downgrade (NOT a full exit)
      4. REVIEW  (any positive signal)                → HOLD   (not exited)
      5. all_negative (mom & qual bad, real loss)     → EXIT  (full)
      6. WATCH   (score not collapsed)                → HOLD   (not exited)
      7. EXIT    (score collapsed, no positives)      → EXIT  (full)

    Returns the boolean full-exit mask. HARVEST/TRIM downgrades and REVIEW/WATCH
    holds are deliberately excluded so they are NOT force-sold here — partial
    profit-taking is handled by the harvest/trim rules downstream. is_premature /
    premature-exit-probability are live-only diagnostics with no backtest analogue
    and are treated as absent (False / 0), which can only make exits *more* likely.
    """
    hard_score_floor = floors["hard_exit_score_below"]
    tis_hard_floor   = floors["thesis_intact_hard_exit_below"]
    harvest_pct      = floors["harvest_profit_threshold"]
    trim_pct         = floors["trim_profit_threshold"]
    pnl_floor        = floors["positive_pnl_review_floor"]
    mom_floor        = floors["positive_momentum_review_floor"]
    qual_floor       = floors["strong_quality_review_floor"]
    tis_floor        = floors["thesis_intact_review_floor"]
    pnl_dg           = floors["positive_pnl_exit_downgrade"]
    mom_dg           = floors["positive_momentum_exit_downgrade"]
    qual_dg          = floors["strong_quality_exit_downgrade"]

    positive_pnl = (pnl >= pnl_floor)   & pnl_dg
    positive_mom = (mom >= mom_floor)   & mom_dg
    strong_qual  = (qual >= qual_floor) & qual_dg
    thesis_ok    = tis >= tis_floor

    mom_bad  = mom  < -0.20
    qual_bad = qual < -0.20
    score_collapsed = snw < hard_score_floor
    thesis_broken   = tis < tis_hard_floor

    confirmed_breakdown = score_collapsed & thesis_broken & mom_bad & qual_bad
    harvest_q = (pnl >= harvest_pct) & pnl_dg & (positive_mom | thesis_ok)
    trim_q    = (pnl >= trim_pct)    & pnl_dg & positive_mom & ~harvest_q
    has_any_positive = positive_pnl | positive_mom | strong_qual | thesis_ok
    all_negative = mom_bad & qual_bad & (pnl < -0.05)

    # Resolve the tree precedence into a single full-exit mask.
    full_exit = confirmed_breakdown.copy()
    remaining = cand & ~confirmed_breakdown
    remaining = remaining & ~harvest_q          # HARVEST downgrade — not a full exit
    remaining = remaining & ~trim_q             # TRIM downgrade — not a full exit
    remaining = remaining & ~has_any_positive   # REVIEW — hold
    full_exit |= remaining & all_negative
    remaining = remaining & ~all_negative
    remaining = remaining & score_collapsed     # WATCH (~collapsed) holds; collapsed falls through
    full_exit |= remaining                       # EXIT — score collapsed, no positives
    return cand & full_exit


def _momentum_score_multifactor_vec(
    day: int,
    precomp: PrecomputedData,
    mom_weights_raw: np.ndarray,
) -> np.ndarray:
    """Vectorized cross-sectional momentum scoring — multi-factor blend of
    rs_3m, rs_6m, risk_adj_3m, trend_structure, return_1m, return_5d."""
    cfg = MOMENTUM_INPUT_PARAMS
    pen = cfg["penalties"]
    n   = precomp.prices.shape[1]
    zeros = np.zeros(n)

    def _get(arr, default_val=0.0):
        # No NaN imputation: NaN flows to _pct_rank_vec, which excludes it from the
        # rank and scores it neutral 0.0 — matching live _pct_rank_series. Imputing
        # rs→0.0 / vol→0.20 here ranked missing-data names mid-pack against real
        # values (top-decile momentum in bear tapes). The default only fills a
        # wholly-absent array (feature not computed for this load).
        if arr is None:
            return np.full(n, default_val)
        return arr[day]

    rs3m  = _get(precomp.rs_3m_daily)
    rs6m  = _get(precomp.rs_6m_daily)
    ret3m = _get(precomp.ret_3m_daily)
    vol3m = _get(precomp.vol_3m_daily, 0.20)
    ret1m = precomp.return_1m_daily[day]
    ret5d = _get(precomp.ret_5d_daily)
    # NaN pos52/vol3m simply never trigger their penalties below (comparisons → False).
    pos52 = precomp.position_52w_daily[day]

    # NaN ret3m or NaN vol3m → NaN risk_adj → neutral 0.0 rank (matching live).
    safe_vol = np.clip(vol3m, 0.01, None)
    with np.errstate(invalid="ignore"):
        risk_adj = ret3m / safe_vol

    a50  = (precomp.above_50dma_daily[day]  if precomp.above_50dma_daily  is not None else zeros).astype(bool)
    a200 = (precomp.above_200dma_daily[day] if precomp.above_200dma_daily is not None else zeros).astype(bool)
    trend = np.select([a50 & a200, a50 & ~a200, ~a50 & a200], [0.5, 0.1, -0.1], default=-0.5)

    raw_w = np.abs(mom_weights_raw[:6])
    total = raw_w.sum()
    if total < 1e-9:
        total = 1.0
    w = raw_w / total

    score = (
        w[0] * _pct_rank_vec(rs3m)     +
        w[1] * _pct_rank_vec(rs6m)     +
        w[2] * _pct_rank_vec(risk_adj) +
        w[3] * trend                    +
        w[4] * _pct_rank_vec(ret1m)    +
        w[5] * _pct_rank_vec(ret5d)
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


def information_ratio(active_values: np.ndarray, bench_values: np.ndarray) -> float:
    """Annualized information ratio: mean(daily active-minus-benchmark) / std of same.

    The honest 'beats SPY' metric — rewards CONSISTENT outperformance of the benchmark,
    not just a higher raw number, and resists the lever-up-beta degenerate solution (a
    concentrated bet that wins a bull also inflates tracking-error variance → no free IR).
    Both inputs must be same-length, same-cadence equity series on the SAME basis
    (here both contribution-adjusted). Returns 0.0 when undefined.
    """
    n = min(len(active_values), len(bench_values))
    if n < 3 or active_values[0] <= 0 or bench_values[0] <= 0:
        return 0.0
    a = np.asarray(active_values[:n], dtype=np.float64)
    b = np.asarray(bench_values[:n], dtype=np.float64)
    a_ret = np.diff(a) / a[:-1]
    b_ret = np.diff(b) / b[:-1]
    excess = a_ret - b_ret
    excess = excess[np.isfinite(excess)]
    if len(excess) < 3 or excess.std() <= 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_stocks(precomp: PrecomputedData, params: np.ndarray) -> np.ndarray:
    """Compute per-stock scores. Uses day-0 snapshot — prefer score_stocks_at_day for simulation."""
    return score_stocks_at_day(precomp, params, 0)


def _oversold_score_at_day(precomp: PrecomputedData, day: int, ma_window: int = 25) -> np.ndarray:
    """Cross-sectional 'oversold' score for mean-reversion (Kotegawa/BNF daily analog).

    Computes each stock's deviation BELOW its trailing `ma_window`-day moving average
    (negative deviation = oversold), then returns the cross-sectional percentile rank
    scaled to [-1, 1] where the MOST oversold names score HIGHEST. Causal: uses only
    prices up to and including `day`. In fear regimes oversold names bounce harder than
    the market (validated: +2-3% forward edge in defensive/neutral regimes); in bull
    regimes they underperform, which is why this is only blended in non-bull regimes.
    """
    n_stocks = precomp.prices.shape[1]
    if day < ma_window:
        return np.zeros(n_stocks)
    window = precomp.prices[day - ma_window:day]
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.nanmean(window, axis=0)
        px = precomp.prices[day]
        dev = np.where((ma > 0) & np.isfinite(px), px / ma - 1.0, np.nan)
    # most oversold (most negative dev) -> highest score: rank ascending then map
    return _pct_rank_vec(-dev)


def _low_vol_score_at_day(precomp: PrecomputedData, day: int) -> np.ndarray:
    """Cross-sectional low-volatility 'quality' score (causal).

    Uses the precomputed daily annualized realized vol (vol_3m_daily) up to and
    including `day`; returns a cross-sectional percentile rank in [-1, 1] where the
    LOWEST-vol names score HIGHEST. Captures the low-volatility anomaly (low-vol
    stocks earn higher risk-adjusted, and on this substrate higher absolute,
    forward returns — full-sample fwd-IC +0.04@21d / +0.067@63d, strongest in bull
    and largely orthogonal to momentum: corr(-vol, rs_6m)≈+0.13). Used as a quality
    blend (slot 48), frozen-by-default. Falls back to zeros when vol is unavailable.
    """
    n_stocks = precomp.prices.shape[1]
    if precomp.vol_3m_daily is None:
        return np.zeros(n_stocks)
    vol = precomp.vol_3m_daily[day]
    if not np.any(np.isfinite(vol)):
        return np.zeros(n_stocks)
    # lowest vol -> highest score
    return _pct_rank_vec(-vol)


def _residmom_score_at_day(precomp: PrecomputedData, day: int,
                           beta_win: int = 252, form_lo: int = 252,
                           skip: int = 21) -> np.ndarray:
    """Cross-sectional RESIDUAL-momentum score (Blitz-Huij-Martens 2011), causal.

    For each stock: estimate beta vs SPY over a trailing `beta_win`-day window, strip
    the market component, cumulate the residual return over the formation window
    [day-form_lo, day-skip], standardize by residual std, then cross-sectionally
    percentile-rank to [-1, 1] (highest residual momentum = highest score). Uses only
    prices up to `day`. Residual momentum strips market beta, so it does NOT load up on
    high-beta winners that crash in sharp reversals — the crash-resistance that makes it
    additive to raw relative-strength momentum (corr≈0.56 on this substrate, not a
    duplicate). Falls back to zeros when SPY prices are unavailable or the window is
    not yet filled.
    """
    n_stocks = precomp.prices.shape[1]
    spy = precomp.spy_prices
    if spy is None or day < beta_win or day < form_lo:
        return np.zeros(n_stocks)
    prices = precomp.prices
    # daily simple returns within the beta window
    lo_b = day - beta_win
    with np.errstate(invalid="ignore", divide="ignore"):
        stk_b = prices[lo_b + 1:day + 1] / prices[lo_b:day] - 1.0      # (beta_win, N)
        spy_full = spy[lo_b + 1:day + 1] / spy[lo_b:day] - 1.0          # (beta_win,)
    bmask = np.isfinite(spy_full)
    spy_b = spy_full[bmask]
    var = np.var(spy_b) if spy_b.size else 0.0
    if var <= 0 or spy_b.size < 60:
        return np.zeros(n_stocks)
    # formation-window market returns (a sub-slice of the beta window)
    f0, f1 = day - form_lo, day - skip
    with np.errstate(invalid="ignore", divide="ignore"):
        spy_form = spy[f0 + 1:f1 + 1] / spy[f0:f1] - 1.0
    fmask = np.isfinite(spy_form)
    out = np.full(n_stocks, np.nan)
    for j in range(n_stocks):
        rj_b = stk_b[:, j]
        if np.isfinite(rj_b[bmask]).sum() < 60:
            continue
        cov = np.cov(rj_b[bmask], spy_b)[0, 1]
        beta = cov / var
        with np.errstate(invalid="ignore", divide="ignore"):
            rj_form = prices[f0 + 1:f1 + 1, j] / prices[f0:f1, j] - 1.0
        resid = rj_form - beta * spy_form
        rr = resid[fmask & np.isfinite(rj_form)]
        if rr.size < 30:
            continue
        sd = np.std(rr)
        out[j] = np.sum(rr) / sd if sd > 0 else np.nan
    return _pct_rank_vec(out)


def _regime_tilted_weights(raw_sw: np.ndarray, params: np.ndarray,
                           precomp: PrecomputedData, day: int) -> np.ndarray:
    """Apply a regime-conditional momentum tilt to the raw score weights.

    Slot 46 (`regime.bullish.momentum_tilt`) is frozen-by-default (0.0 →
    behaviour-preserving). When present and positive AND the day's regime is
    confirmed bullish, shift up to `tilt` of total weight from value/quality/
    income into momentum, then renormalise. This makes the active sleeve more
    aggressive (momentum-led) in confirmed bull tapes while keeping its
    defensive quality/income tilt in neutral/defensive regimes. It only changes
    *which* stocks score high — never cash/share accounting.
    """
    sw = raw_sw / max(raw_sw.sum(), 1e-9)
    if len(params) <= 46:
        return sw
    tilt = float(params[46])
    if tilt <= 0.0:
        return sw
    if _detect_regime(precomp, day) != "bullish":
        return sw
    # Move `tilt` of weight away from value/quality/income (slots 0,1,2)
    # proportionally, and add it to momentum (slot 3). sw already sums to 1.
    non_mom = sw[0] + sw[1] + sw[2]
    move = min(tilt, non_mom)  # never take more than available
    if non_mom <= 1e-9:
        return sw
    scale = (non_mom - move) / non_mom
    tilted = sw.copy()
    tilted[0] *= scale
    tilted[1] *= scale
    tilted[2] *= scale
    tilted[3] += move
    return tilted


def score_stocks_at_day(precomp: PrecomputedData, params: np.ndarray, day: int) -> np.ndarray:
    """
    Score stocks using day-specific rolling momentum features.

    Routes to v2 continuous scoring when multi-factor arrays are populated,
    otherwise falls back to v1 bucket scoring for backward compatibility.
    """
    raw_sw = params[:4]
    sw = _regime_tilted_weights(raw_sw, params, precomp, day)
    value_pe_w  = params[9]
    value_score = value_pe_w * precomp.pe_comp + (1.0 - value_pe_w) * precomp.pb_comp

    if precomp.ret_3m_daily is not None:
        momentum_score = _momentum_score_multifactor_vec(day, precomp, params[10:16])
    else:
        momentum_score = _momentum_score_warmup_vec(
            precomp.bin_indices_daily[day],
            precomp.has_position_52w_daily[day],
            precomp.position_52w_daily[day],
            precomp.return_1m_daily[day],
            _MBIN_SCORES,
        )

    # Residual-momentum blend (slot 49, frozen-by-default = 0.0). Blends a causal
    # cross-sectional residual-momentum score (beta-stripped, crash-resistant) into the
    # momentum factor: 0 = pure configured momentum, 1 = pure residual momentum. Strongest
    # single-factor fwd-IC on this substrate and only ~0.56 corr with rs_6m, so it adds
    # information; its edge is regime-shaped (cushions momentum's weak era). Scoring-only.
    if len(params) > 49:
        rm_blend = float(params[49])
        if rm_blend > 0.0:
            resid_mom = _residmom_score_at_day(precomp, day)
            momentum_score = (1.0 - rm_blend) * momentum_score + rm_blend * resid_mom

    quality_component = precomp.quality_scores
    # Low-vol quality blend (slot 48, frozen-by-default = 0.0). Blends a causal
    # cross-sectional low-volatility score into the quality factor: 0 = pure
    # configured quality, 1 = pure low-vol rank. Low-vol is a documented quality
    # anomaly, orthogonal to momentum, positive full-sample fwd-IC on this substrate.
    if len(params) > 48:
        lv_blend = float(params[48])
        if lv_blend > 0.0:
            low_vol = _low_vol_score_at_day(precomp, day)
            quality_component = (1.0 - lv_blend) * quality_component + lv_blend * low_vol

    composite = (
        sw[0] * value_score
        + sw[1] * quality_component
        + sw[2] * precomp.income_scores
        + sw[3] * momentum_score
    )

    # Regime-conditional mean-reversion blend (slot 47, frozen-by-default = 0.0).
    # In NON-bull regimes (neutral/defensive fear tapes), blend an oversold score
    # into the composite: contrarian selection (buy names most below their 25d MA)
    # generates +2-3% forward edge in fear regimes, the mirror of momentum which
    # only works in bull. blend in [0,1]: 0 = pure composite, 1 = pure oversold.
    if len(params) > 47:
        mr_blend = float(params[47])
        if mr_blend > 0.0 and _detect_regime(precomp, day) != "bullish":
            oversold = _oversold_score_at_day(precomp, day)
            composite = (1.0 - mr_blend) * composite + mr_blend * oversold

    return composite


def _momentum_score_at_day(precomp: PrecomputedData, params: np.ndarray, day: int) -> np.ndarray:
    if precomp.ret_3m_daily is not None:
        return _momentum_score_multifactor_vec(day, precomp, params[10:16])
    return _momentum_score_warmup_vec(
        precomp.bin_indices_daily[day],
        precomp.has_position_52w_daily[day],
        precomp.position_52w_daily[day],
        precomp.return_1m_daily[day],
        _MBIN_SCORES,
    )


def _detect_regime(precomp: PrecomputedData, day: int) -> str:
    labels = getattr(precomp, "regime_labels_daily", None)
    if labels is not None and day < len(labels):
        label = str(labels[day])
        if label in {"bullish", "neutral", "defensive"}:
            return label
    bench = precomp.benchmark_prices
    if day < 200 or not (np.isfinite(bench[day]) and bench[day] > 0):
        return "bullish"
    ma200 = float(np.nanmean(bench[max(0, day - 199): day + 1]))
    # VIX present → the SHARED VIX-primary classifier, so backtest regime == live regime.
    vix_arr = getattr(precomp, "vix_prices", None)
    if vix_arr is not None and day < len(vix_arr) and np.isfinite(vix_arr[day]):
        from strategy.regimes.classifier import classify_regime
        return classify_regime(float(bench[day]), ma200, float(vix_arr[day]))
    # No VIX → legacy SPY-vs-200DMA rule (byte-identical to pre-VIX backtests).
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

    # Phase-2 vector override: when the tuner extends params with candidate-filter
    # slots (see _CS_FILTER_SLOT_OFFSET=40 in tuning/constants.py), those values
    # take precedence over the live config for this run.
    if params is not None and len(params) > 40:
        cs_params = dict(cs_params)
        cs_params["top_percentile"]     = float(params[40])
        cs_params["min_quality_score"]  = float(params[41])
        cs_params["min_momentum_score"] = float(params[42])
        if len(params) > 45:
            cs_params["max_candidates"] = round(float(params[45]))

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

    # Discretionary NEVER-BUY exclusions (industry/sector) — mirrors the live gate in
    # data/fundamentals.py so the backtest evaluates the same investable universe. Applied
    # as a candidate gate: excluded names never get bought (existing holds still managed).
    _disc_mask = getattr(precomp, "excluded_mask", None)
    disc_gate = ~_disc_mask if _disc_mask is not None else np.ones(n, dtype=bool)

    # Per-day tradeability (survivorship-free loads): dead names' prices are ffilled past
    # their delist date so held positions can mark, but they must never be BOUGHT there.
    _alive_daily = getattr(precomp, "tradeable_mask_daily", None)
    alive_gate = _alive_daily[day] if _alive_daily is not None else np.ones(n, dtype=bool)

    final_mask = score_mask & quality_gate & mom_gate & income_trap_gate & disc_gate & alive_gate

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
    from portfolio.position_archetypes import (
        classify_archetype_full_from_scores,
    )

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
    confirm_arr   = np.zeros(n,                        dtype=bool)   # thesis_exit_requires_confirmation
    deepdd_arr    = np.zeros(n,                        dtype=bool)   # allow_deeper_drawdown
    labels        = [""] * n
    buckets       = [""] * n   # confidence bucket per stock (for attribution rollups)

    if arch_cfg_override is not None and arch_cfg_override:
        arch_cfg = arch_cfg_override
    else:
        arch_cfg = ARCHETYPE_PARAMS if ARCHETYPE_PARAMS.get("enabled", False) else None
    if arch_cfg is None:
        return {
            "tp": tp_arr, "stop": stop_arr, "harvest": harvest_arr, "min_hold": min_hold_arr,
            "enabled": enabled_arr, "score_mult": score_mult_arr, "pos_mult": pos_mult_arr,
            "max_sleeve": max_sleeve_arr, "min_score": min_score_arr,
            "confirm": confirm_arr, "deepdd": deepdd_arr, "labels": labels,
            "buckets": buckets,
        }

    yt_mask = precomp.yield_trap_mask if precomp.yield_trap_mask is not None else np.zeros(n, dtype=bool)

    # Augment signals: pass sector/industry/market_cap from precomp so the
    # backtest classifier sees ~6 signals instead of 4 (closes most of the
    # live/backtest label-disagreement gap).
    _ind_labels = precomp.industry_labels if precomp.industry_labels else ()
    _mkt_caps = precomp.market_caps if precomp.market_caps is not None else None
    _sec_labels = precomp.sector_labels if precomp.sector_labels else ()
    _mom_scores = precomp.momentum_scores if precomp.momentum_scores is not None else None
    for i in range(n):
        try:
            _sec = _sec_labels[i] if i < len(_sec_labels) else None
            _ind = _ind_labels[i] if i < len(_ind_labels) else None
            _mc  = float(_mkt_caps[i]) if (_mkt_caps is not None
                                           and i < len(_mkt_caps)
                                           and np.isfinite(_mkt_caps[i])) else None
            _mom = float(_mom_scores[i]) if (_mom_scores is not None
                                             and i < len(_mom_scores)
                                             and np.isfinite(_mom_scores[i])) else 0.0
            # Use the full classifier to also capture confidence_bucket for attribution.
            _full = classify_archetype_full_from_scores(
                quality_score  = float(precomp.quality_scores[i]),
                momentum_score = _mom,
                income_score   = float(precomp.income_scores[i]),
                yield_trap     = bool(yt_mask[i]),
                archetype_cfg  = arch_cfg,
                sector         = _sec,
                industry       = _ind,
                market_cap     = _mc,
            )
            policy = _full.policy
            buckets[i] = _full.confidence_bucket
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
            confirm_arr[i]   = bool(policy.thesis_exit_requires_confirmation)
            deepdd_arr[i]    = bool(policy.allow_deeper_drawdown)
            labels[i]        = policy.archetype
        except Exception:
            pass

    return {
        "tp": tp_arr, "stop": stop_arr, "harvest": harvest_arr, "min_hold": min_hold_arr,
        "enabled": enabled_arr, "score_mult": score_mult_arr, "pos_mult": pos_mult_arr,
        "max_sleeve": max_sleeve_arr, "min_score": min_score_arr,
        "confirm": confirm_arr, "deepdd": deepdd_arr, "labels": labels,
        "buckets": buckets,
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
      [10-14] momentum input sub-weights (rs_3m, rs_6m, risk_adj_3m, trend, return_1m)
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
    _harvest_to_etfs_pct = float(HARVEST_PARAMS.get("harvest_to_etfs_pct", 1.0))
    # ── DAE soft-exit floors (mirror DecisionAdjustmentEngine._evaluate_soft_exit) ─
    # The faithful exit/hold/downgrade tree below is keyed on these floors. They are
    # now load-bearing in the backtest (previously the simulator only consulted
    # review_score_below + positive P/L, so these floors were inert).
    _dae_floors = {
        "hard_exit_score_below":          float(_trim_cfg.get("hard_exit_score_below",          0.20)),
        "thesis_intact_hard_exit_below":  float(_trim_cfg.get("thesis_intact_hard_exit_below",  0.35)),
        "harvest_profit_threshold":       _HARVEST_PROFIT_THRESHOLD,
        "trim_profit_threshold":          float(_trim_cfg.get("trim_profit_threshold",          0.08)),
        "positive_pnl_review_floor":      float(_trim_cfg.get("positive_pnl_review_floor",       0.00)),
        "positive_momentum_review_floor": float(_trim_cfg.get("positive_momentum_review_floor",  0.10)),
        "strong_quality_review_floor":    float(_trim_cfg.get("strong_quality_review_floor",     0.70)),
        "thesis_intact_review_floor":     float(_trim_cfg.get("thesis_intact_review_floor",      0.60)),
        "positive_pnl_exit_downgrade":    bool(_trim_cfg.get("positive_pnl_exit_downgrade",      True)),
        "positive_momentum_exit_downgrade": bool(_trim_cfg.get("positive_momentum_exit_downgrade", True)),
        "strong_quality_exit_downgrade":  bool(_trim_cfg.get("strong_quality_exit_downgrade",    True)),
    }
    # When the tuner appends exit-floor slots (active_exit_floors preset), let the
    # optimizer move the floors; otherwise they stay at the config values above.
    from tuning.constants import (
        exit_floors_cfg_from_params,
        opportunity_cost_cfg_from_params,
        rebalance_cfg_from_params,
    )
    # Rebalance cadence + post-sell/stopout cooldowns (active_rebalance_cooldown preset).
    # Overrides the rebalance_frequency_days arg and the cooldown_sell/stop locals (read
    # below) when the slots are present; otherwise config/arg values are used. NOTE:
    # rebalance_frequency_days also gates contribution timing (is_contrib_day).
    _rebal_override = rebalance_cfg_from_params(params)
    if "rebalance_frequency_days" in _rebal_override:
        rebalance_frequency_days = int(_rebal_override["rebalance_frequency_days"])
    _dae_floors.update(exit_floors_cfg_from_params(params))

    # ── Opportunity-cost ("max hold without progress") exit ───────────────────
    # Culls positions that make NO progress for _oc_stall_max_days, to recycle
    # active-sleeve capacity. The stall clock resets whenever a position makes
    # progress (fresh high within band OR momentum >= floor), so a running winner
    # is never culled. Frozen OFF by default — `_oc_enabled` is strictly config-
    # driven so it can never leak into unrelated preset tunes (every tune passes
    # the full 57-slot vector). The active_opportunity_cost preset tunes the three
    # thresholds below; it takes effect only when opportunity_cost.enabled is true
    # in config (set it for the tuning run).
    _oc_cfg            = _trim_cfg.get("opportunity_cost", {}) or {}
    _oc_enabled        = bool(_oc_cfg.get("enabled", False))
    _oc_stall_max_days = int(_oc_cfg.get("stall_max_days", 120))
    _oc_reclaim_band   = float(_oc_cfg.get("reclaim_band", 0.03))
    _oc_mom_floor      = float(_oc_cfg.get("progress_momentum_floor", 0.10))
    _oc_overrides = opportunity_cost_cfg_from_params(params)
    if _oc_overrides:
        _oc_stall_max_days = _oc_overrides["stall_max_days"]   # already int-rounded
        _oc_reclaim_band   = float(_oc_overrides["reclaim_band"])
        _oc_mom_floor      = float(_oc_overrides["progress_momentum_floor"])
    # Momentum veto (frozen default off => behavior-preserving). When enabled, a
    # winner still trending up is NOT trimmed even if its cross-sectional score
    # decayed — converts "cut winners" into "cut only STALLED winners". Validated
    # marginal: improves rolling-window hit-rate vs SPY (30/45, sign-test p=0.036)
    # but Wilcoxon p=0.50 (gives up right-tail in strong bulls) and no clean
    # market-direction mechanism (non-monotonic). NOT wired live / NOT in best_alpha.
    # Reuses the formerly-dead `trim_requires_positive_momentum` config key.
    # Signal: a `trim_veto_ret1m_above` threshold (spare names whose trailing 1m
    # return >= thr) if set, else coarse price>=50DMA. See .session_tmp/trim_veto*.py.
    _trim_mom_veto       = bool(_trim_cfg.get("trim_requires_positive_momentum", False))
    if scope == "active_sleeve_compounding":
        _trim_to_etfs_pct = 0.0
        _harvest_to_etfs_pct = 0.0

    # ── Barroso–Santa-Clara vol-scaling overlay (frozen by default)
    # Scope-independent: scales the stock book's exposure by clip(tv/realized_vol).
    # No-op unless explicitly enabled in config. See util.SLEEVE_VOL_OVERLAY.
    _svo_cfg     = SLEEVE_VOL_OVERLAY
    _svo_enabled = bool(_svo_cfg.get("enabled", False))
    _svo_tv      = float(_svo_cfg.get("target_vol", 0.15))
    _svo_lb      = int(_svo_cfg.get("lookback", 63))
    _svo_wmax    = float(_svo_cfg.get("w_max", 1.0))
    _svo_band    = float(_svo_cfg.get("deadband", 0.08))
    _svo_minhist = int(_svo_cfg.get("min_history", 63))
    _svo_switch  = float(_svo_cfg.get("switch_bps", 20.0)) / 10_000.0
    _svo_cur_w   = _svo_wmax  # current applied exposure weight (starts fully invested)

    base_slippage   = slippage_bps / 10_000.0
    bp_cfg          = BACKTEST_PARAMS
    use_vol_slip    = bp_cfg.get("vol_slippage_scaling", True)
    vol_slip_mult   = bp_cfg.get("vol_slippage_multiplier", 2.0)
    cooldown_sell   = int(_rebal_override.get("cooldown_days_after_sell",    bp_cfg.get("cooldown_days_after_sell", 3)))
    cooldown_stop   = int(_rebal_override.get("cooldown_days_after_stopout", bp_cfg.get("cooldown_days_after_stopout", 7)))
    max_trades_week = bp_cfg.get("max_trades_per_week", 10)

    min_order      = RISK_LIMITS["min_order_amount"]
    max_single_pct = RISK_LIMITS["max_single_position_pct"]
    max_sector_pct = RISK_LIMITS["max_sector_pct"]
    max_order_pct  = RISK_LIMITS["max_order_pct_of_cash"]
    max_buys       = RISK_LIMITS["max_buys_per_rebalance"]
    # Cap-proxy sizing: when on, budget is allocated ∝ dollar-volume (size) instead of ∝ score.
    # Ranking still by score (momentum-dominant); only the WEIGHTING changes. The per-name/sector/
    # cluster caps still bind, so this is a size-TILT within diversification, not concentration.
    _size_by_dv    = bool(RISK_LIMITS.get("size_by_dollar_volume", False))

    # Position-sizing override: the active_position_sizing preset appends sizing
    # slots 43-45; when present they take precedence over live RISK_LIMITS.
    if len(params) > 45:
        max_single_pct = float(params[43])
        max_buys       = round(float(params[44]))

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
        _arch_confirm_arr       = _arch["confirm"]
        _arch_deepdd_arr        = _arch["deepdd"]
        _arch_labels            = _arch["labels"]
        _arch_buckets           = _arch.get("buckets", [""] * len(_arch_labels))
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
        _arch_confirm_arr = None
        _arch_deepdd_arr = None
        _arch_labels = []
        _arch_buckets = []
    _arch_pnl: dict = {}
    _arch_pnl_by_confidence: dict[str, float] = {}
    _arch_trade_counts_by_confidence: dict[str, int] = {}
    _arch_buy_bucket: dict[int, str] = {}
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
    _cluster_pnl: dict[str, float] = {}
    _cluster_trade_counts: dict[str, int] = {}
    _cluster_win_count: dict[str, int] = {}
    _cluster_hold_days_sum: dict[str, float] = {}
    _cluster_buy_cluster: dict[int, str] = {}   # idx → cluster at last buy
    _cluster_decision_counts: dict[str, int] = {"allowed": 0, "downsized": 0, "blocked": 0}
    _cluster_violations_count: int = 0
    _conc_limits = ARCHETYPE_PARAMS  # fallback; we use concentration_limits separately
    try:
        from util import CONCENTRATION_LIMIT_PARAMS as _CLP
    except Exception:
        _CLP = {}
    _max_cluster_w = float(_CLP.get("max_cluster_weight", 0.35))
    _max_sector_w  = float(_CLP.get("max_sector_weight", 0.40))
    _n_clusters_ct = int(_CLP.get("n_clusters", 6))

    # Cluster cap enforcement (no-op when warn_only=true, the default).
    _cluster_cap_enabled = bool(
        _CLP.get("enabled", False)
        and not _CLP.get("warn_only", True)
        and (_CLP.get("apply_to", {}) or {}).get("active_sleeve", True)
    )
    _cluster_cap_limit = _max_cluster_w
    _cluster_cap_downsize = bool((_CLP.get("enforcement", {}) or {}).get("downsize_to_fit", True))

    # Auto-enable cluster_tracking when enforcement is on (we need the per-day labels).
    if _cluster_cap_enabled and not cluster_tracking:
        cluster_tracking = True

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
    # Per-day momentum, cached on the same cadence as current_scores (recomputed on
    # contribution days). Feeds the faithful DAE soft-exit tree. Quality is static
    # (precomp.quality_scores), mirroring how the live composite consumes quality.
    current_mom    = _momentum_score_at_day(precomp, params, 0)
    candidate_mask, _init_diag = select_candidates(0, current_scores, precomp, params, _cs_params)

    stock_shares    = np.zeros(n_stocks)
    stock_avg_cost  = np.zeros(n_stocks)
    stock_peak      = np.zeros(n_stocks)
    # Last valid traded price per symbol (finite & > 0), updated every day a price
    # prints and seeded from the fill price at buy. Used to mark NaN-priced positions
    # and to fill forced exits — marking/filling at cost basis erased losses and let
    # zero-liquidity names exit at cost. NaN until a symbol has ever printed.
    stock_last_price = np.full(n_stocks, np.nan)
    stock_day_bought= np.full(n_stocks, -1, dtype=np.int32)
    stock_day_sold  = np.full(n_stocks, -99, dtype=np.int32)
    stock_stopout   = np.zeros(n_stocks, dtype=bool)
    # Day index of the last time each position made progress (fresh high or strong
    # momentum). Drives the opportunity-cost stall clock. -1 = not held / no record.
    stock_last_progress_day = np.full(n_stocks, -1, dtype=np.int32)
    # Consecutive evaluations a held position has signalled a soft weak-value exit.
    # Drives archetype `thesis_exit_requires_confirmation`. Resets to 0 whenever the
    # weak signal is absent (incl. when not held), so a sold/rebought name starts fresh.
    stock_weak_streak = np.zeros(n_stocks, dtype=np.int32)
    etf_shares      = np.zeros(n_etfs)

    # ── Regime de-risk overlay (frozen off by default) ────────────────────────
    # On entry into a defensive regime (SPY > 5% below its 200DMA, optionally
    # lagged) rotate `_ro_frac` of the held stock book into the benchmark
    # instrument and hold it in a dedicated overlay bucket until the regime
    # clears, then unwind to cash. The overlay bucket is part of the ACTIVE
    # sleeve (it is added to port_val and is NOT part of etf_value), so the
    # active-equity curve `port_val - etf_value` tracks the benchmark return on
    # the de-risked capital — matching the validated return-space PoC
    # (.session_tmp/regime_real_pinned.py). This closes a live-vs-backtest gap:
    # the LIVE path already de-risks defensively via regime.defensive
    # .index_pct_override, but the backtest historically ignored it.
    _ro_cfg       = REGIME_PARAMS.get("defensive", {}) if REGIME_PARAMS else {}
    _ro_frac      = float(_ro_cfg.get("backtest_derisk_frac", 0.0))
    _ro_switch    = float(_ro_cfg.get("backtest_derisk_switch_bps", 20.0)) / 10000.0
    _ro_lag       = int(_ro_cfg.get("backtest_derisk_lag", 0))
    _ro_enabled   = _ro_frac > 0.0
    _overlay_units = 0.0      # benchmark units held by the overlay
    _ro_active     = False    # currently de-risked?
    _ro_days_active = 0       # telemetry: # days held in de-risked state
    _ro_rotations   = 0       # telemetry: # of enter-defensive rotations
    _ro_switch_cost = 0.0     # telemetry: $ switch cost attributable to overlay
    _ro_max_value   = 0.0     # telemetry: peak overlay bucket mark-to-market

    cash                = float(starting_capital)
    daily_values        = np.zeros(n_days)
    ca_daily_values     = np.zeros(n_days)
    _active_daily       = np.zeros(n_days) if scope == "active_sleeve_compounding" else None
    _ca_active_daily    = np.zeros(n_days) if scope == "active_sleeve_compounding" else None
    # Flow-free daily mark-to-market return of the held stock book, used by the
    # vol-scaling overlay. Scope-independent (works in overall_strategy too).
    _svo_rets           = np.full(n_days, np.nan)
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

    def _mark_fallback() -> np.ndarray:
        """Mark/fill price when today's print is missing: last valid traded price,
        falling back to cost basis only for a held name that never printed."""
        return np.where(np.isfinite(stock_last_price), stock_last_price, stock_avg_cost)

    def _mark_prices(prices_d: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Per-symbol mark price vector: today's print where valid, else last traded."""
        return np.where(valid, prices_d, _mark_fallback())

    def _mark_price_i(i: int, prices_d: np.ndarray, valid: np.ndarray) -> float:
        """Scalar mark price for symbol i — same fallback chain as _mark_prices."""
        if valid[i]:
            return float(prices_d[i])
        lp = stock_last_price[i]
        return float(lp) if np.isfinite(lp) else float(stock_avg_cost[i])

    def _current_portfolio_value(day: int) -> float:
        prices_d = precomp.prices[day]
        valid    = np.isfinite(prices_d) & (prices_d > 0)
        sv = float(np.sum(stock_shares * _mark_prices(prices_d, valid)))
        ev = float(np.sum(etf_shares * precomp.etf_prices[day])) if n_etfs > 0 else 0.0
        return cash + sv + ev

    def _sector_exposures(day: int) -> dict:
        prices_d = precomp.prices[day]
        valid    = np.isfinite(prices_d) & (prices_d > 0)
        exposure: dict = {}
        for i in np.where(stock_shares > 0)[0]:
            s   = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"
            val = float(stock_shares[i]) * _mark_price_i(i, prices_d, valid)
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
                total += float(stock_shares[k]) * _mark_price_i(k, prices_d, valid)
        return total

    def _do_buy(day: int, budget: float) -> float:
        nonlocal cash, trades_made, skipped_buys, cap_reductions, cooldown_skips
        nonlocal total_friction, total_traded_notional, trades_this_week
        nonlocal _cluster_violations_count

        if budget < min_order:
            return 0.0
        prices_d = precomp.prices[day]
        eligible = candidate_mask & np.isfinite(prices_d) & (prices_d > 0)
        # Survivorship-free tradeability guard: delisted names' prices are ffilled past
        # their delist date (so holds can mark/exit) but must never be bought there.
        # candidate_mask is refreshed on rebalance days only, so re-gate per buy day.
        if precomp.tradeable_mask_daily is not None:
            eligible = eligible & precomp.tradeable_mask_daily[day]

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

        # Cap-proxy budget weights: dollar-volume (static volume × causal day price) as a size tilt.
        # Selection/ranking is unchanged (still by score); only how budget is split changes.
        if _size_by_dv:
            if precomp.dollar_volume_daily is not None:
                # CAUSAL trailing 21-day dollar-volume — the honest size proxy (no look-ahead).
                _w0 = max(0, day - 20)
                with np.errstate(invalid="ignore"):
                    _dv = np.nanmean(precomp.dollar_volume_daily[_w0:day + 1], axis=0)
                _dv = np.where(np.isfinite(_dv) & (_dv > 0), _dv, 0.0)
            else:
                # Fallback (default loader has no daily volume): static volume × day price.
                _pd_day = precomp.prices[day]
                _dv = np.where(np.isfinite(_pd_day) & (_pd_day > 0), precomp.volume_arr * _pd_day, 0.0)
            _total_dv = _dv[eligible].sum()
        else:
            _dv = None
            _total_dv = 0.0

        portfolio_value = _current_portfolio_value(day)
        sector_exp      = _sector_exposures(day)
        spent           = 0.0
        commission_paid = 0.0
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

            if _size_by_dv and _total_dv > 0:
                alloc = (_dv[i] / _total_dv) * budget
            else:
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

            # ── Cluster concentration enforcement (config-gated) ──────────────
            # When the user flips warn_only=false, the simulator's `_do_buy()` enforces
            # cluster caps the same way as live `buy_cycle()`. Defaults are a no-op.
            _cls_for_buy: str | None = None
            if (
                _cluster_cap_enabled
                and _cluster_labels_by_day is not None
                and portfolio_value > 0
            ):
                _rebal_idx = day // rebalance_frequency_days
                if _rebal_idx < _cluster_labels_by_day.shape[0]:
                    _cls = str(_cluster_labels_by_day[_rebal_idx][i])
                    _cls_for_buy = _cls
                    _cur_cw = 0.0
                    for _k in np.where(stock_shares > 0)[0]:
                        if _k >= _cluster_labels_by_day.shape[1]:
                            continue
                        if str(_cluster_labels_by_day[_rebal_idx][_k]) == _cls:
                            _p_k = prices_d[_k] if np.isfinite(prices_d[_k]) and prices_d[_k] > 0 else (
                                stock_last_price[_k] if np.isfinite(stock_last_price[_k]) else stock_avg_cost[_k]
                            )
                            _cur_cw += float(stock_shares[_k] * _p_k)
                    _cur_cw = _cur_cw / portfolio_value
                    _new_w = _cur_cw + (alloc / portfolio_value)
                    if _new_w > _cluster_cap_limit:
                        _cluster_violations_count += 1
                        if _cluster_cap_downsize:
                            _headroom = max(0.0, _cluster_cap_limit - _cur_cw)
                            _fit = _headroom * portfolio_value
                            if _fit >= min_order:
                                alloc = _fit
                                cap_reductions += 1
                                _cluster_decision_counts["downsized"] += 1
                            else:
                                _cluster_decision_counts["blocked"] += 1
                                skipped_buys += 1
                                continue
                        else:
                            _cluster_decision_counts["blocked"] += 1
                            skipped_buys += 1
                            continue
                    else:
                        _cluster_decision_counts["allowed"] += 1

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
            commission_paid += commission_per_trade
            stock_last_price[i] = p  # seed/refresh last traded price at the fill

            if stock_shares[i] > 0:
                old_cost          = stock_avg_cost[i] * stock_shares[i]
                stock_avg_cost[i] = (old_cost + alloc) / (stock_shares[i] + shares)
            else:
                stock_avg_cost[i]  = effective_price
                stock_day_bought[i]= day
                # A freshly bought name is "making progress" by definition (just
                # selected as a top candidate) — seed its stall clock at buy.
                stock_last_progress_day[i] = day
                trades_made       += 1
                trades_this_week  += 1
                sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                _al = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
                _arch_buy_archetype[i] = _al
                _arch_buy_cost_basis[i] = alloc
                # Capture confidence bucket at buy for by-confidence attribution
                _buy_bucket = _arch_buckets[i] if _arch_buckets and i < len(_arch_buckets) else ""
                if _buy_bucket:
                    _arch_buy_bucket[i] = _buy_bucket
                # Capture cluster label at buy so sell-side rollups can attribute pnl
                if _cls_for_buy is not None:
                    _cluster_buy_cluster[i] = _cls_for_buy
                trade_log.append(TradeRecord(
                    date=str(day), symbol=sym, side="buy",
                    quantity=shares, price=effective_price, amount=alloc, reason="buy",
                    archetype=_al,
                    archetype_at_entry=_al,
                    archetype_at_exit="",
                    decision_source=("archetype_rule" if _al else "global_rule"),
                    cluster_id=(_cls_for_buy or ""),
                ))
            stock_shares[i] += shares
            stock_peak[i]    = max(stock_peak[i], p)
            sector_exp[sector] = sector_exp.get(sector, 0.0) + alloc
            if _al_i:
                sleeve_consumed[_al_i] = sleeve_consumed.get(_al_i, 0.0) + alloc
            spent                 += alloc
            total_traded_notional += alloc
            buys_this_pass        += 1

        cash -= spent + commission_paid
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

        # Flow-free book return: mark-to-market change of the CURRENTLY held shares
        # from yesterday's close to today's, computed BEFORE any trades today (so no
        # contribution/buy/sell flow contaminates it). Scope-independent vol signal
        # for the BSC overlay. Uses constant shares across the d-1 -> d boundary.
        if _svo_enabled and d > 0:
            _pp = precomp.prices[d - 1]
            _vp = np.isfinite(prices) & (prices > 0) & np.isfinite(_pp) & (_pp > 0) & held
            if _vp.any():
                _v_prev = float(np.sum(stock_shares[_vp] * _pp[_vp]))
                _v_now  = float(np.sum(stock_shares[_vp] * prices[_vp]))
                if _v_prev > 0:
                    _svo_rets[d] = _v_now / _v_prev - 1.0

        if d > 0 and (d - week_start_day) >= rebalance_frequency_days:
            trades_this_week = 0
            week_start_day   = d

        valid_price = np.isfinite(prices) & (prices > 0)
        # Record today's print as the last valid traded price (in-place so closures see it).
        np.copyto(stock_last_price, prices, where=valid_price)
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

        # Opportunity-cost stall clock: a position making progress today (fresh high
        # within band OR momentum >= floor) resets its clock; the rest accrue stall
        # days. _progress_vec mirrors the live exit_analysis.is_progress exactly.
        # stock_peak already includes today's price, so a fresh high counts as progress.
        if _oc_enabled:
            progress_mask = held & _progress_vec(
                prices, stock_peak, current_mom, _oc_reclaim_band, _oc_mom_floor,
            )
            stock_last_progress_day = np.where(progress_mask, d, stock_last_progress_day)
            stall_days = np.where(stock_last_progress_day >= 0, d - stock_last_progress_day, 0)
            oc_mask = (
                held
                & ~progress_mask
                & (stall_days >= _oc_stall_max_days)
                & (days_held >= _eff_minhold)
            )
        else:
            oc_mask = np.zeros(len(held), dtype=bool)

        # Archetype allow_deeper_drawdown: widen the catastrophic hard stop for flagged
        # archetypes so high-conviction names get room before a failure exit.
        _eff_stoploss: np.ndarray | float
        if _arch_deepdd_arr is not None:
            _eff_stoploss = np.where(_arch_deepdd_arr, _STOP_LOSS_PCT * _DEEPER_DD_FACTOR, _STOP_LOSS_PCT)
        else:
            _eff_stoploss = _STOP_LOSS_PCT
        stop_loss_mask = held & (pct_from_avg  <= _eff_stoploss)
        trail_mask     = held & (pct_from_peak <= _eff_stop)        & (days_held >= _eff_minhold)
        tp_mask        = held & (pct_from_avg  >= _eff_tp)          & take_profit_ok & (days_held >= _MIN_DAYS_BEFORE_TAKE_PROFIT)
        # Score-below-threshold soft-exit candidates. The live SellDecisionEngine
        # raises a soft EXIT here; the DecisionAdjustmentEngine then runs its full
        # tree (confirmed-breakdown / harvest / trim / review / watch / exit) before
        # anything is sold. We replicate that tree faithfully so the exit floors
        # (hard_exit_score_below, positive_momentum/strong_quality/thesis_intact
        # review floors) are load-bearing — a weak score alone never forces an exit.
        _weak_cand = (
            held
            & (current_scores < sell_weak_below)
            & (days_held >= _MIN_DAYS_HELD_BEFORE_VALUE_EXIT)
        )
        if _weak_cand.any():
            _rank01 = (_pct_rank_vec(current_scores) + 1.0) / 2.0
            _tis    = _thesis_intact_vec(precomp.quality_scores, current_mom, pct_from_avg, _rank01)
            weak_val_mask = _dae_soft_exit_full_exit(
                _weak_cand,
                snw=current_scores,
                pnl=pct_from_avg,
                mom=current_mom,
                qual=precomp.quality_scores,
                tis=_tis,
                floors=_dae_floors,
            )
        else:
            weak_val_mask = np.zeros(len(held), dtype=bool)
        # Archetype thesis_exit_requires_confirmation: a flagged position must show the
        # soft weak-value exit across _THESIS_CONFIRM_EVALS consecutive rebalance
        # evaluations before it fires — a single weak reading never dumps a compounder.
        # The streak advances once per rebalance cycle (scores are piecewise-constant
        # between them) and resets to 0 whenever the weak signal clears at a checkpoint.
        # Hard sells (stop-loss / trailing) are unaffected.
        if _arch_confirm_arr is not None:
            if d > 0 and d % rebalance_frequency_days == 0:
                stock_weak_streak = np.where(weak_val_mask, stock_weak_streak + 1, 0)
            _confirmed_exit = (~_arch_confirm_arr) | (stock_weak_streak >= _THESIS_CONFIRM_EVALS)
            weak_val_mask = weak_val_mask & _confirmed_exit
        sell_mask      = stop_loss_mask | trail_mask | tp_mask | weak_val_mask | oc_mask

        if sell_mask.any():
            sell_prices   = _mark_prices(prices, valid_price)
            sell_notional = float(np.sum(stock_shares[sell_mask] * sell_prices[sell_mask]))
            sell_indices  = np.where(sell_mask)[0]
            for i in sell_indices:
                slip       = _effective_slippage(i)
                proceeds_i = float(stock_shares[i] * sell_prices[i] * (1.0 - slip))
                total_friction += float(stock_shares[i] * sell_prices[i] * slip) + commission_per_trade
                cash += proceeds_i - commission_per_trade
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
                elif weak_val_mask[i]:
                    _exit = "weak_value"
                    _src  = "global_rule"
                elif oc_mask[i]:
                    # Opportunity-cost cull — stalled, no progress for long enough.
                    _exit = "opportunity_cost"
                    _src  = "global_rule"
                else:
                    _exit = "weak_value"
                    _src  = "global_rule"
                sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                _al = _arch_labels[i] if _arch_labels and i < len(_arch_labels) else ""
                _al_entry = _arch_buy_archetype.get(int(i), _al)
                _cls_entry = _cluster_buy_cluster.get(int(i), "")
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
                    cluster_id=_cls_entry,
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
                if _cls_entry:
                    _cluster_pnl[_cls_entry] = _cluster_pnl.get(_cls_entry, 0.0) + _pnl_i
                    _cluster_trade_counts[_cls_entry] = _cluster_trade_counts.get(_cls_entry, 0) + 1
                    _cluster_hold_days_sum[_cls_entry] = _cluster_hold_days_sum.get(_cls_entry, 0.0) + float(days_held[i])
                    if _pnl_i > 0:
                        _cluster_win_count[_cls_entry] = _cluster_win_count.get(_cls_entry, 0) + 1
                # Per-archetype, per-confidence-bucket attribution
                _bucket = _arch_buy_bucket.pop(int(i), "")
                if _al and _bucket:
                    _key = f"{_al}|{_bucket}"
                    _arch_pnl_by_confidence[_key] = _arch_pnl_by_confidence.get(_key, 0.0) + _pnl_i
                    _arch_trade_counts_by_confidence[_key] = _arch_trade_counts_by_confidence.get(_key, 0) + 1
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
            stock_last_progress_day[sell_mask] = -1

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
            harv_prices   = _mark_prices(prices, valid_price)
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
                cash           += proceeds_i - commission_per_trade
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
            etf_portion = harv_notional * _harvest_to_etfs_pct
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
            # Momentum veto: spare winners still trending up from trimming.
            # Trims only STALLED winners, preserving the run in still-ripping names.
            # Signal configurable: '50dma' (price>=50DMA, coarse) or a return_1m
            # threshold via trim_veto_ret1m_above (spare names whose last-month return
            # exceeds the threshold — only the hottest names are spared).
            if _trim_mom_veto:
                _r1m_thr = _trim_cfg.get("trim_veto_ret1m_above", None)
                if _r1m_thr is not None and precomp.return_1m_daily is not None:
                    _still_up = precomp.return_1m_daily[d] >= float(_r1m_thr)
                    trim_mask = trim_mask & ~_still_up
                elif precomp.above_50dma_daily is not None:
                    _still_up = precomp.above_50dma_daily[d].astype(bool)
                    trim_mask = trim_mask & ~_still_up
            if trim_mask.any():
                sell_prices_t = _mark_prices(prices, valid_price)
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
                    cash           += proceeds_i - commission_per_trade
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

        # ── Regime de-risk overlay: rotate book <-> benchmark on regime edges ──
        # Frozen off unless regime.defensive.backtest_derisk_frac > 0. The overlay
        # bucket (_overlay_units of the benchmark instrument) is part of the ACTIVE
        # sleeve, so its mark-to-market rides the benchmark return while de-risked.
        _bench_px_d = precomp.benchmark_prices[d] if precomp.benchmark_prices is not None else float("nan")
        if _ro_enabled and np.isfinite(_bench_px_d) and _bench_px_d > 0:
            _sig_day = max(0, d - _ro_lag)
            _is_def  = _detect_regime(precomp, _sig_day) == "defensive"
            if _is_def and not _ro_active:
                # ENTER defensive: rotate _ro_frac of the held book into benchmark.
                _rot_notional = 0.0
                _held_idx = np.where(stock_shares > 0)[0]
                for i in _held_idx:
                    _sh = stock_shares[i] * _ro_frac
                    if _sh <= 0:
                        continue
                    _p_i  = _mark_price_i(i, prices, valid_price)
                    _slip = _effective_slippage(i)
                    _proceeds = float(_sh * _p_i * (1.0 - _slip))
                    total_friction   += float(_sh * _p_i * _slip)
                    _rot_notional    += _proceeds
                    stock_shares[i]  -= _sh
                if _rot_notional > 0:
                    _rot_net = _rot_notional * (1.0 - _ro_switch)
                    _overlay_units += _rot_net / _bench_px_d
                    total_friction        += _rot_notional * _ro_switch
                    total_traded_notional += _rot_notional
                    _ro_switch_cost       += _rot_notional * _ro_switch
                _ro_rotations += 1
                _ro_active = True
            elif (not _is_def) and _ro_active:
                # EXIT defensive: unwind overlay back to cash (redeployed by _do_buy).
                _unwind = _overlay_units * _bench_px_d
                if _unwind > 0:
                    _unwind_net = _unwind * (1.0 - _ro_switch)
                    cash                  += _unwind_net
                    total_friction        += _unwind * _ro_switch
                    total_traded_notional += _unwind
                    _ro_switch_cost       += _unwind * _ro_switch
                _overlay_units = 0.0
                _ro_active = False

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
                        prices=_mark_prices(precomp.prices[d], valid_price),
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
            current_mom    = _momentum_score_at_day(precomp, params, d)
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

            # ── BSC vol-scaling overlay: set target exposure, de-risk pro-rata,
            # cap the buy budget so the stock book holds w*(stock+cash) in stock
            # and (1-w) in cash. Scope-independent. No-op unless enabled (frozen).
            _svo_budget_cap = None
            if _svo_enabled and d > _svo_minhist:
                _hist = _svo_rets[max(0, d - _svo_lb):d]
                _hist = _hist[np.isfinite(_hist)]
                if len(_hist) > max(5, _svo_lb // 2):
                    _rv = float(np.std(_hist)) * np.sqrt(252.0)
                    if _rv > 1e-6:
                        _w_new = max(0.0, min(_svo_wmax, _svo_tv / _rv))
                        if abs(_w_new - _svo_cur_w) > _svo_band:
                            _svo_cur_w = _w_new
                _cur_sv = float(np.sum(stock_shares * _mark_prices(prices, valid_price)))
                _active_eq = _cur_sv + cash
                _target_sv = _svo_cur_w * _active_eq
                if _cur_sv > _target_sv and _cur_sv > 0:
                    _f = min(1.0, max(0.0, 1.0 - (_target_sv / _cur_sv)))
                    if _f > 0:
                        _derisk_notional = 0.0
                        for _i in np.where(stock_shares > 0)[0]:
                            _sh = stock_shares[_i] * _f
                            if _sh <= 0:
                                continue
                            _p_i  = _mark_price_i(_i, prices, valid_price)
                            _slip = _effective_slippage(_i)
                            _proceeds = float(_sh * _p_i * (1.0 - _slip))
                            total_friction   += float(_sh * _p_i * _slip)
                            _derisk_notional += _proceeds
                            cash             += _proceeds
                            stock_shares[_i] -= _sh
                        cash                  -= _derisk_notional * _svo_switch
                        total_traded_notional += _derisk_notional
                _cur_sv2 = float(np.sum(stock_shares * _mark_prices(prices, valid_price)))
                _svo_budget_cap = max(0.0, _target_sv - _cur_sv2)

            # While de-risked, sweep free cash (contributions + any sell proceeds)
            # into the overlay so the active sleeve STAYS de-risked instead of
            # re-buying stocks during the downturn. Sweeps _ro_frac of free cash;
            # the remaining (1-frac) is deployable to stocks (partial de-risk).
            if _ro_active and np.isfinite(_bench_px_d) and _bench_px_d > 0:
                _sweep = cash * _ro_frac
                if _sweep >= min_order:
                    _sweep_net = _sweep * (1.0 - _ro_switch)
                    _overlay_units        += _sweep_net / _bench_px_d
                    total_friction        += _sweep * _ro_switch
                    total_traded_notional += _sweep
                    cash                  -= _sweep
                    _ro_switch_cost       += _sweep * _ro_switch

            if cash >= min_order:
                _buy_budget = cash if _svo_budget_cap is None else min(cash, _svo_budget_cap)
                if _buy_budget >= min_order:
                    _do_buy(d, _buy_budget)

        _overlay_value = _overlay_units * _bench_px_d if (_ro_enabled and np.isfinite(_bench_px_d) and _bench_px_d > 0) else 0.0
        if _ro_active:
            _ro_days_active += 1
        if _overlay_value > _ro_max_value:
            _ro_max_value = _overlay_value
        etf_value   = float(np.sum(etf_shares * precomp.etf_prices[d])) if n_etfs > 0 else 0.0
        stock_value = float(np.sum(stock_shares * _mark_prices(prices, valid_price)))
        port_val    = cash + stock_value + etf_value + _overlay_value
        daily_values[d] = port_val
        if _active_daily is not None:
            # Overlay is part of the ACTIVE sleeve (de-risked active capital riding
            # the benchmark), so it stays in the active curve: only the index sleeve
            # (etf_value) is excluded.
            _active_daily[d] = port_val - etf_value

        if accounting_trace is not None:
            accounting_trace.append({
                "d": d,
                "cash": cash,
                "stock_value": stock_value,
                "etf_value": etf_value,
                "overlay_value": _overlay_value,
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
                _p = _mark_price_i(k, prices, valid_price)
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
                _p = _mark_price_i(i, last_prices, valid_last)
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

    # ── Per-cluster rollups (only when cluster_tracking ran) ────────────────
    _cluster_win_rate: dict[str, float] = {}
    _cluster_avg_hold: dict[str, float] = {}
    _cluster_sleeve_weight: dict[str, float] = {}
    _cluster_active_excess: dict[str, float] = {}
    _cluster_dominant_sectors: dict[str, str] = {}
    _cluster_dominant_archetypes: dict[str, str] = {}
    if _cluster_labels_by_day is not None and n_days > 0:
        # End-of-sim cluster sleeve weights + dominant labels
        last_port_val = float(daily_values[-1]) if n_days > 0 else 0.0
        last_rebal_idx = min(
            (n_days - 1) // rebalance_frequency_days,
            _cluster_labels_by_day.shape[0] - 1,
        )
        if last_rebal_idx >= 0 and last_port_val > 0:
            last_labels = _cluster_labels_by_day[last_rebal_idx]
            last_prices = precomp.prices[-1] if n_days > 0 else None
            if last_prices is not None:
                _per_cluster_value: dict[str, float] = {}
                _per_cluster_sectors: dict[str, dict[str, int]] = {}
                _per_cluster_archetypes: dict[str, dict[str, int]] = {}
                _valid_last = np.isfinite(last_prices) & (last_prices > 0)
                for k in np.where(stock_shares > 0)[0]:
                    if k >= last_labels.shape[0]:
                        continue
                    cls = str(last_labels[k])
                    _p = _mark_price_i(k, last_prices, _valid_last)
                    _val = float(stock_shares[k] * _p)
                    _per_cluster_value[cls] = _per_cluster_value.get(cls, 0.0) + _val
                    if k < len(precomp.sector_labels):
                        _sec = str(precomp.sector_labels[k])
                        _per_cluster_sectors.setdefault(cls, {})
                        _per_cluster_sectors[cls][_sec] = _per_cluster_sectors[cls].get(_sec, 0) + 1
                    if _arch_labels and k < len(_arch_labels) and _arch_labels[k]:
                        _arl = _arch_labels[k]
                        _per_cluster_archetypes.setdefault(cls, {})
                        _per_cluster_archetypes[cls][_arl] = _per_cluster_archetypes[cls].get(_arl, 0) + 1
                for cls, v in _per_cluster_value.items():
                    _cluster_sleeve_weight[cls] = float(v / last_port_val)
                for cls, sects in _per_cluster_sectors.items():
                    if sects:
                        _cluster_dominant_sectors[cls] = max(sects, key=sects.get)
                for cls, arcs in _per_cluster_archetypes.items():
                    if arcs:
                        _cluster_dominant_archetypes[cls] = max(arcs, key=arcs.get)

        # Win rate + avg hold (from sells)
        for cls, n in _cluster_trade_counts.items():
            if n > 0:
                _cluster_win_rate[cls] = float(_cluster_win_count.get(cls, 0)) / float(n)
                _cluster_avg_hold[cls] = float(_cluster_hold_days_sum.get(cls, 0.0)) / float(n)
        # Approximate per-cluster active excess: realized pnl / avg deployed - bench_twr
        for cls, pnl in _cluster_pnl.items():
            # Crude avg deployed: end-of-sim sleeve value (close enough for diagnostics)
            avg_deployed = _cluster_sleeve_weight.get(cls, 0.0) * (float(daily_values[-1]) if n_days else 0.0)
            if avg_deployed > 0:
                sleeve_return = pnl / avg_deployed
            else:
                sleeve_return = 0.0
            _cluster_active_excess[cls] = float(sleeve_return - bench_twr_val)

    _active_total_return: float | None = None
    _active_sharpe:       float | None = None
    _active_calmar:       float | None = None
    _active_max_drawdown: float | None = None
    _active_excess_return: float | None = None
    _active_information_ratio: float | None = None
    _active_equity_curve: np.ndarray | None = None
    if _active_daily is not None and _active_daily[0] > 0:
        _am = compute_performance_metrics(_ca_active_daily)
        _active_equity_curve  = _ca_active_daily.copy()
        _active_total_return  = _am["total_return"]
        _active_sharpe        = _am["sharpe"]
        _active_calmar        = _am["calmar"]
        _active_max_drawdown  = _am["max_drawdown"]
        _active_excess_return = _active_total_return - bench_twr_val
        # Information ratio vs the SAME-basis (contribution-adjusted) benchmark series.
        if bench_ca_equity is not None and len(bench_ca_equity) >= 3:
            _active_information_ratio = information_ratio(_ca_active_daily, bench_ca_equity)

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
        overlay_telemetry=({
            "enabled": True,
            "frac": _ro_frac,
            "lag": _ro_lag,
            "days_active": _ro_days_active,
            "rotations": _ro_rotations,
            "switch_cost": float(_ro_switch_cost),
            "max_overlay_value": float(_ro_max_value),
        } if _ro_enabled else None),
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
        cluster_pnl=_cluster_pnl,
        cluster_trade_counts=_cluster_trade_counts,
        cluster_win_rate=_cluster_win_rate,
        cluster_avg_hold_days=_cluster_avg_hold,
        cluster_sleeve_weight=_cluster_sleeve_weight,
        cluster_active_excess=_cluster_active_excess,
        cluster_dominant_sectors=_cluster_dominant_sectors,
        cluster_dominant_archetypes=_cluster_dominant_archetypes,
        cluster_violations_count=_cluster_violations_count,
        cluster_decision_counts=_cluster_decision_counts,
        archetype_pnl_by_confidence=_arch_pnl_by_confidence,
        archetype_trade_counts_by_confidence=_arch_trade_counts_by_confidence,
        scope=scope,
        active_equity_curve=_active_equity_curve,
        active_total_return=_active_total_return,
        active_sharpe=_active_sharpe,
        active_calmar=_active_calmar,
        active_max_drawdown=_active_max_drawdown,
        active_excess_return=_active_excess_return,
        active_information_ratio=_active_information_ratio,
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
    params=None uses the FULL live config (``tuning.constants._current_params``,
    60-slot, archetype-aware) so the report reflects the strategy that would
    actually trade — not the stripped 16-slot ``get_default_params()`` base.
    """
    if params is None:
        from tuning.constants import _current_params
        params = _current_params()
    bp = BACKTEST_PARAMS
    _arch_enabled = bool(ARCHETYPE_PARAMS.get("enabled", False))

    # Canonical window slicer — slices EVERY per-day array (vix/spy/dollar-volume/…)
    # so offset windows (the validation slice!) stay calendar-aligned. Lazy import:
    # regime_scope imports this module at load time.
    from .regime_scope import regime_labels, slice_precomp

    # Attach point-in-time regime labels computed on the FULL load before slicing, so
    # the validation window (offset > 0) keeps its 200DMA context instead of resetting
    # to the day<200 "bullish" fallback at its day 0.
    if precomp.regime_labels_daily is None:
        precomp = precomp._replace(regime_labels_daily=regime_labels(precomp))

    def _slice_precomp(s: slice) -> PrecomputedData:
        return slice_precomp(precomp, s)

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

    from strategy.scoring.composite import SCORING_MODEL_VERSION
    _engine_version = SCORING_MODEL_VERSION
    _peer_cfg = dict(SCORING_PARAMS.get("peer_standardization", {}))

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
        scoring_engine_version=_engine_version,
        peer_config=_peer_cfg,
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

    # Canonical slicer — slices every per-day array (vix/spy/dollar-volume/…).
    from .regime_scope import slice_precomp

    train = slice_precomp(precomp, train_slice)
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

    # Canonical slicer — slices every per-day array (vix/spy/dollar-volume/…).
    from .regime_scope import slice_precomp

    train_precomp = slice_precomp(precomp, train_slice)
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
