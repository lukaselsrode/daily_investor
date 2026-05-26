"""
backtesting/simulator.py — Simulation engine: scoring, candidate selection, run_simulation,
run_backtest_report, compare_candidate_selection_modes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from util import (
    BACKTEST_PARAMS,
    CANDIDATE_SELECTION_PARAMS,
    INDEX_PCT,
    METRIC_THRESHOLD,
    MOMENTUM_PARAMS,
    MOMENTUM_V2_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
)

from core.types import TradeRecord
from .types import BacktestReport, CandidatePoolDiagnostics, PrecomputedData, SimResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level sell constants
# ---------------------------------------------------------------------------

_TAKE_PROFIT_FLOOR_MULTIPLIER = 1.2
_STOP_LOSS_PCT = -0.20
_MIN_DAYS_HELD_BEFORE_VALUE_EXIT = SELL_RULES.get("min_days_held_before_value_exit", 21)
_MIN_HOLD_DAYS = RISK_LIMITS.get("minimum_hold_days", 0)
_MIN_DAYS_BEFORE_TAKE_PROFIT = SELL_RULES.get("minimum_days_before_take_profit", 0)


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
    cs_params: "dict | None" = None,
) -> "tuple[np.ndarray, CandidatePoolDiagnostics]":
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


def print_pool_diagnostics(diag: CandidatePoolDiagnostics, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Candidate pool: n={diag.n_candidates}  cutoff={diag.score_cutoff:.3f}  "
        f"qual={diag.avg_quality:.2f}  mom={diag.avg_momentum:.2f}  "
        f"income={diag.avg_income:.2f}  value={diag.avg_value:.2f}"
    )
    if diag.n_quality_gate_excluded or diag.n_momentum_gate_excluded or diag.n_income_trap_excluded or diag.n_floor_excluded:
        print(
            f"{prefix}  Excluded — floor={diag.n_floor_excluded}  "
            f"quality={diag.n_quality_gate_excluded}  "
            f"momentum={diag.n_momentum_gate_excluded}  "
            f"income_trap={diag.n_income_trap_excluded}"
        )
    if diag.excluded_high_income_low_momentum:
        print(f"{prefix}  High-income/low-momentum excluded: {', '.join(diag.excluded_high_income_low_momentum)}")
    if diag.sector_counts:
        top_sectors = sorted(diag.sector_counts.items(), key=lambda x: -x[1])[:5]
        print(f"{prefix}  Sectors: " + "  ".join(f"{s}={c}" for s, c in top_sectors))


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
            contrib = weekly_contribution
            shares += contrib / p
            cash    = 0.0
        val = shares * p + cash
        if d == 0:
            ca[0] = val
        else:
            prev   = shares * bench_prices[d - 1] + cash
            factor = (val - contrib) / max(prev, 1e-9)
            ca[d]  = ca[d - 1] * factor
    return compute_performance_metrics(ca)["total_return"]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(
    precomp: PrecomputedData,
    params: np.ndarray,
    starting_capital: float = 10_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    weekly_contribution: float = 0.0,
    rebalance_frequency_days: int = 5,
    cs_params: "dict | None" = None,
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
    total_contributions = float(starting_capital)

    trades_made          = 0
    sells_made           = 0
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

    def _do_buy(day: int, budget: float) -> float:
        nonlocal cash, trades_made, skipped_buys, cap_reductions, cooldown_skips
        nonlocal total_friction, total_traded_notional, trades_this_week

        if budget < min_order:
            return 0.0
        prices_d = precomp.prices[day]
        eligible = candidate_mask & np.isfinite(prices_d) & (prices_d > 0)
        if not eligible.any():
            return 0.0
        total_score = current_scores[eligible].sum()
        if total_score <= 0:
            return 0.0

        portfolio_value = _current_portfolio_value(day)
        sector_exp      = _sector_exposures(day)
        spent           = 0.0
        buys_this_pass  = 0

        candidate_indices = sorted(np.where(eligible)[0], key=lambda i: -current_scores[i])

        for i in candidate_indices:
            if buys_this_pass >= max_buys or trades_this_week >= max_trades_week:
                skipped_buys += 1
                continue

            days_since_sell = day - int(stock_day_sold[i])
            required_cd     = cooldown_stop if stock_stopout[i] else cooldown_sell
            if days_since_sell < required_cd:
                cooldown_skips += 1
                continue

            alloc = (current_scores[i] / total_score) * budget

            max_by_cash = cash * max_order_pct
            if alloc > max_by_cash:
                alloc = max_by_cash
                cap_reductions += 1

            if portfolio_value > 0:
                cur_pos_val = float(stock_shares[i] * prices_d[i])
                room = portfolio_value * max_single_pct - cur_pos_val
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
                trade_log.append(TradeRecord(
                    date=str(day), symbol=sym, side="buy",
                    quantity=shares, price=effective_price, amount=alloc, reason="buy",
                ))
            stock_shares[i] += shares
            stock_peak[i]    = max(stock_peak[i], p)
            sector_exp[sector] = sector_exp.get(sector, 0.0) + alloc
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

        stop_loss_mask = held & (pct_from_avg  <= _STOP_LOSS_PCT)
        trail_mask     = held & (pct_from_peak <= trailing_stop) & (days_held >= _MIN_HOLD_DAYS)
        tp_mask        = held & (pct_from_avg  >= take_profit_pct) & take_profit_ok & (days_held >= _MIN_DAYS_BEFORE_TAKE_PROFIT)
        weak_val_mask  = held & (current_scores < sell_weak_below) & (days_held >= _MIN_DAYS_HELD_BEFORE_VALUE_EXIT)
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
                elif trail_mask[i]:
                    _exit = "trailing_stop"
                elif tp_mask[i]:
                    _exit = "take_profit"
                else:
                    _exit = "weak_value"
                sym = precomp.symbols[i] if i < len(precomp.symbols) else str(i)
                trade_log.append(TradeRecord(
                    date=str(d), symbol=sym, side="sell",
                    quantity=float(stock_shares[i]), price=float(sell_prices[i]),
                    amount=proceeds_i, exit_type=_exit,
                    pnl=proceeds_i - cost_basis, hold_days=int(days_held[i]),
                ))
            total_traded_notional += sell_notional
            sells_made            += int(sell_mask.sum())
            stock_shares[sell_mask]    = 0.0
            stock_avg_cost[sell_mask]  = 0.0
            stock_peak[sell_mask]      = 0.0
            stock_day_bought[sell_mask]= -1

        is_contrib_day = d > 0 and d % rebalance_frequency_days == 0
        contrib_today  = weekly_contribution if is_contrib_day else 0.0
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

        if d == 0:
            ca_daily_values[0] = port_val
        else:
            prev_port          = daily_values[d - 1]
            factor             = (port_val - contrib_today) / max(prev_port, 1e-9)
            ca_daily_values[d] = ca_daily_values[d - 1] * factor

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

    final_value = float(daily_values[-1])
    metrics     = compute_performance_metrics(ca_daily_values)
    avg_port    = float(daily_values[daily_values > 0].mean()) if daily_values.any() else starting_capital
    turnover    = total_traded_notional / max(avg_port, 1.0)
    profit      = final_value - total_contributions

    bench_twr_val = 0.0
    if np.isfinite(precomp.benchmark_prices).all() and precomp.benchmark_prices[0] > 0:
        bench_twr_val = _bench_twr(
            precomp.benchmark_prices, starting_capital,
            weekly_contribution, rebalance_frequency_days,
        )

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
        cooldown_skips=cooldown_skips,
        regime_days=regime_days,
        benchmark_twr=bench_twr_val,
        pool_diagnostics=_init_diag,
        trade_log=trade_log,
    )


# ---------------------------------------------------------------------------
# Report orchestration
# ---------------------------------------------------------------------------

def run_backtest_report(
    precomp: PrecomputedData,
    params: "np.ndarray | None",
    train_slice: slice,
    val_slice: "slice | None",
) -> BacktestReport:
    """
    Run strategy on train window, optionally evaluate on validation window.
    params=None uses current config values via get_default_params().
    """
    if params is None:
        params = get_default_params()
    bp = BACKTEST_PARAMS

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
    train_result  = run_simulation(
        train_precomp, params,
        starting_capital=bp["starting_capital"],
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )

    val_result: "SimResult | None" = None
    if val_slice is not None:
        val_precomp = _slice_precomp(val_slice)
        if val_precomp.prices.shape[0] >= 5:
            val_result = run_simulation(
                val_precomp, params,
                starting_capital=bp["starting_capital"],
                slippage_bps=bp["slippage_bps"],
                commission_per_trade=bp["commission_per_trade"],
                weekly_contribution=bp["weekly_contribution"],
                rebalance_frequency_days=bp["rebalance_frequency_days"],
            )

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
    )


def compare_candidate_selection_modes(
    precomp: PrecomputedData,
    params: "np.ndarray | None" = None,
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
