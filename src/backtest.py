"""
backtest.py — Lightweight simulation engine for parameter tuning.

Uses current fundamental data (agg_data.csv) as a proxy for historical
fundamentals. Simulates buy/sell logic over the past N trading days using
real price data from yfinance. No Robinhood API calls, no sentiment analysis.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import yfinance as yf

from util import (
    BACKTEST_PARAMS,
    ETFS,
    MOMENTUM_PARAMS,
    MOMENTUM_V2_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    read_data_as_pd,
)

logger = logging.getLogger(__name__)

_TAKE_PROFIT_FLOOR_MULTIPLIER = 1.2
_MIN_DAYS_HELD_BEFORE_VALUE_EXIT = 21
_STOP_LOSS_PCT = -0.20  # hard-coded; not a tuned param


class PrecomputedData(NamedTuple):
    symbols: list[str]
    prices: np.ndarray            # (n_days, n_stocks) float64
    pe_comp: np.ndarray           # (n_stocks,)
    pb_comp: np.ndarray           # (n_stocks,)
    quality_scores: np.ndarray    # (n_stocks,)
    income_scores: np.ndarray     # (n_stocks,)
    yield_trap_mask: np.ndarray   # (n_stocks,) bool
    bin_indices: np.ndarray       # (n_stocks,) int 0-4
    has_position_52w: np.ndarray  # (n_stocks,) bool
    position_52w_arr: np.ndarray  # (n_stocks,) float, NaN where missing
    return_1m_arr: np.ndarray     # (n_stocks,) float, NaN where missing
    etf_symbols: list[str]
    etf_prices: np.ndarray        # (n_days, n_etfs) float64
    baseline_scores: np.ndarray   # (n_stocks,) scored with current config
    sector_labels: list[str]      # (n_stocks,) sector per stock
    volume_arr: np.ndarray        # (n_stocks,) daily avg volume
    mode: str                     # lookahead bias mode
    universe_selection: str       # selection method used
    lookahead_bias_level: str     # HIGH / MEDIUM / LOW
    benchmark_prices: np.ndarray  # (n_days,) benchmark close prices
    benchmark_symbol: str
    # Daily rolling price-derived features for dynamic re-scoring
    position_52w_daily: np.ndarray      # (n_days, n_stocks) float, NaN until window fills
    return_1m_daily: np.ndarray         # (n_days, n_stocks) float, NaN until 21d available
    bin_indices_daily: np.ndarray       # (n_days, n_stocks) int
    has_position_52w_daily: np.ndarray  # (n_days, n_stocks) bool
    # Momentum v2 daily rolling features — None when not computed (v1 fallback activates)
    ret_5d_daily: "np.ndarray | None" = None    # (n_days, n_stocks) 5-day return
    ret_3m_daily: "np.ndarray | None" = None    # (n_days, n_stocks) 63-day return
    ret_6m_daily: "np.ndarray | None" = None    # (n_days, n_stocks) 126-day return
    rs_3m_daily: "np.ndarray | None" = None     # (n_days, n_stocks) relative strength vs SPY
    rs_6m_daily: "np.ndarray | None" = None     # (n_days, n_stocks) relative strength vs SPY
    vol_3m_daily: "np.ndarray | None" = None    # (n_days, n_stocks) annualized realized vol
    above_50dma_daily: "np.ndarray | None" = None   # (n_days, n_stocks) bool
    above_200dma_daily: "np.ndarray | None" = None  # (n_days, n_stocks) bool
    spy_prices: "np.ndarray | None" = None      # (n_days,) SPY closes for RS computation


@dataclass
class SimResult:
    final_value: float
    total_return: float       # time-weighted return (excludes contributions)
    sharpe: float             # computed from TWR daily series
    calmar: float
    max_drawdown: float
    trades_made: int
    # extended fields — default to 0 for backward compat with existing tuner calls
    sells_made: int = 0
    skipped_buys: int = 0
    cap_reductions: int = 0
    average_positions: float = 0.0
    max_positions: int = 0
    average_cash_pct: float = 0.0
    turnover_estimate: float = 0.0
    friction_cost: float = 0.0
    net_contributions: float = 0.0  # starting_capital + all weekly contributions
    profit: float = 0.0             # final_value - net_contributions
    # attribution & regime diagnostics
    stopout_count: int = 0          # hard stop-loss triggered
    cooldown_skips: int = 0         # buys skipped due to post-sell cooldown
    regime_days: "dict | None" = None  # {"bullish": N, "neutral": N, "defensive": N}
    benchmark_twr: float = 0.0     # contribution-adjusted benchmark TWR for comparison


@dataclass
class BacktestReport:
    mode: str
    universe_selection: str
    lookahead_bias_level: str
    n_symbols: int
    n_days: int
    train_result: SimResult
    validation_result: "SimResult | None"
    benchmark_return: float            # train-window benchmark (simple price return)
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    excess_return: float               # train excess return vs benchmark
    validation_benchmark_return: float # validation-window benchmark (0.0 if no val window)
    notes: list[str]
    # extended reporting
    train_benchmark_twr: float = 0.0   # contribution-adjusted benchmark TWR
    val_benchmark_twr: float = 0.0


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

    has_r1m = np.isfinite(return_1m)
    low_pos = has_pos & (pos_52w < mp["return_1m_low_position_cutoff"]) & has_r1m
    recovery = low_pos & (return_1m >= mp["return_1m_recovery_threshold"])
    falling  = low_pos & (return_1m <= mp["return_1m_falling_knife_threshold"])

    base = base + np.where(recovery, mp["return_1m_recovery_bonus"],        0.0)
    base = base - np.where(falling,  mp["return_1m_falling_knife_penalty"], 0.0)
    return base


def _pct_rank_vec(arr: np.ndarray) -> np.ndarray:
    """
    Cross-sectional percentile rank scaled to [-1, 1].
    NaN → 0.0 (neutral).  Causal: operates on one day's values across stocks.
    """
    out = np.zeros(len(arr))
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return out
    vals = arr[finite]
    ranks = (vals.argsort().argsort() + 1) / (finite.sum() + 1)  # (0, 1)
    out[finite] = ranks * 2 - 1  # (-1, 1)
    return out


def _momentum_score_v2_vec(
    day: int,
    precomp: "PrecomputedData",
    mom_weights_raw: np.ndarray,   # params[10:15]: [rs3m, rs6m, risk_adj, trend, r1m]
) -> np.ndarray:
    """
    Vectorized v2 momentum scoring using multi-factor cross-sectional model.

    All features are percentile-ranked within the current day's cross-section
    before combining — causal, no lookahead across time.
    """
    cfg = MOMENTUM_V2_PARAMS
    pen = cfg["penalties"]
    n   = precomp.prices.shape[1]
    zeros = np.zeros(n)

    def _get(arr, default_val=0.0):
        if arr is None:
            return np.full(n, default_val)
        row = arr[day]
        return np.where(np.isfinite(row), row, default_val)

    rs3m   = _get(precomp.rs_3m_daily)
    rs6m   = _get(precomp.rs_6m_daily)
    ret3m  = _get(precomp.ret_3m_daily)
    vol3m  = _get(precomp.vol_3m_daily, 0.20)
    ret1m  = precomp.return_1m_daily[day]
    ret1m  = np.where(np.isfinite(ret1m), ret1m, 0.0)
    ret5d  = _get(precomp.ret_5d_daily)
    pos52  = precomp.position_52w_daily[day]
    pos52  = np.where(np.isfinite(pos52), pos52, 0.5)

    # Risk-adjusted 3m momentum
    safe_vol = np.clip(vol3m, 0.01, None)
    risk_adj = ret3m / safe_vol

    # Trend structure (deterministic signal, not percentile-ranked)
    a50  = (precomp.above_50dma_daily[day]  if precomp.above_50dma_daily  is not None else zeros).astype(bool)
    a200 = (precomp.above_200dma_daily[day] if precomp.above_200dma_daily is not None else zeros).astype(bool)
    trend = np.select([a50 & a200, a50 & ~a200, ~a50 & a200], [0.5, 0.1, -0.1], default=-0.5)

    # Normalize sub-weights (raw values from params)
    raw_w = np.abs(mom_weights_raw[:5])
    w_r5d = cfg["weights"].get("return_5d", 0.05)
    total  = raw_w.sum() + w_r5d
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

    # Penalties
    score -= np.where(ret3m  < pen["falling_knife_3m_threshold"],  pen["falling_knife_penalty"],  0.0)
    score -= np.where(pos52  > pen["overextension_52w_threshold"],  pen["overextension_penalty"],  0.0)
    score -= np.where(vol3m  > pen["high_vol_annual_threshold"],     pen["high_vol_penalty"],       0.0)

    return np.clip(score, cfg["clamp_low"], cfg["clamp_high"])


def split_price_window(n_days: int, train_pct: float) -> tuple[slice, slice]:
    """Split n_days into train and validation slices."""
    train_end = max(1, int(n_days * train_pct))
    return slice(0, train_end), slice(train_end, n_days)


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

    cum = daily_values / daily_values[0]
    roll_max = np.maximum.accumulate(cum)
    drawdowns = np.where(roll_max > 0, cum / roll_max - 1.0, 0.0)
    max_drawdown = float(drawdowns.min())

    calmar = 0.0
    if max_drawdown < -0.001:
        calmar = float(total_return / abs(max_drawdown))

    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
    }


def select_backtest_universe(
    agg_df: "pd.DataFrame",
    mode: str,
    universe_selection: str,
    max_symbols: int,
    min_volume: float,
    random_seed: int,
) -> "pd.DataFrame":
    """
    Select the universe of symbols for backtesting.

    Modes and their lookahead bias levels:
      current_universe_stress_test  → HIGH   (uses value_metric ranking)
      liquid_universe_sanity_test   → MEDIUM (uses volume / random sample)
      walk_forward_price_only_test  → LOW    (uses volume filter only, no scores)

    universe_selection values:
      top_current_scores   — rank by value_metric descending (HIGH bias)
      liquid_all           — all stocks above min_volume (no score bias)
      liquid_sample        — random sample from liquid universe
      sector_balanced_sample — equal-weight sectors, random within each
    """
    import random as _random

    liquid = agg_df[agg_df["volume"] >= min_volume].copy()
    if liquid.empty:
        logger.warning("No symbols pass min_volume filter — using all available")
        liquid = agg_df.copy()

    if universe_selection == "top_current_scores":
        selected = liquid.sort_values("value_metric", ascending=False).head(max_symbols)
        bias = "HIGH"
        logger.warning(
            "Universe selection=top_current_scores uses current value_metric. "
            "LOOK-AHEAD BIAS: HIGH. Results are not predictive."
        )
    elif universe_selection == "liquid_all":
        selected = liquid.head(max_symbols)
        bias = "MEDIUM"
    elif universe_selection == "sector_balanced_sample":
        rng = _random.Random(random_seed)
        sectors = liquid["sector"].dropna().unique().tolist() if "sector" in liquid.columns else []
        if not sectors:
            selected = liquid.sample(n=min(max_symbols, len(liquid)), random_state=random_seed)
        else:
            per_sector = max(1, max_symbols // len(sectors))
            parts = []
            for s in sectors:
                pool = liquid[liquid["sector"] == s]
                n = min(per_sector, len(pool))
                parts.append(pool.sample(n=n, random_state=random_seed))
            selected = pd.concat(parts).head(max_symbols)
        bias = "MEDIUM"
    else:
        # Default: liquid_sample — random from liquid universe
        n = min(max_symbols, len(liquid))
        selected = liquid.sample(n=n, random_state=random_seed)
        bias = "MEDIUM"

    if mode == "current_universe_stress_test":
        bias = "HIGH"
    elif mode == "walk_forward_price_only_test":
        bias = "LOW"

    sectors = selected["sector"].value_counts().to_dict() if "sector" in selected.columns else {}
    logger.info(
        f"Universe: mode={mode} sel={universe_selection} n={len(selected)} bias={bias} "
        f"sectors={len(sectors)}"
    )
    return selected.reset_index(drop=True), bias


def score_stocks(precomp: PrecomputedData, params: np.ndarray) -> np.ndarray:
    """Compute per-stock scores. Uses day-0 snapshot — prefer score_stocks_at_day for simulation."""
    return score_stocks_at_day(precomp, params, 0)


def score_stocks_at_day(precomp: PrecomputedData, params: np.ndarray, day: int) -> np.ndarray:
    """
    Score stocks using day-specific rolling momentum features.

    Routes to v2 continuous scoring when multi-factor arrays are populated,
    otherwise falls back to v1 bucket scoring for backward compatibility.
    params layout:
      [0-3]  score weights (value, quality, income, momentum)
      [4]    index_pct  [5] metric_threshold  [6] take_profit_pct
      [7]    sell_weak  [8] trailing_stop      [9] value_pe_weight
      [10-14] momentum sub-weights (v2) or bin scores (v1 fallback)
    """
    raw_sw = params[:4]
    sw = raw_sw / max(raw_sw.sum(), 1e-9)
    value_pe_w = params[9]
    value_score = value_pe_w * precomp.pe_comp + (1.0 - value_pe_w) * precomp.pb_comp

    # v2 path: multi-factor relative-strength scoring
    if precomp.ret_3m_daily is not None:
        momentum_score = _momentum_score_v2_vec(day, precomp, params[10:15])
    else:
        # v1 fallback: bucket scoring (used by unit tests with minimal PrecomputedData)
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


def _bench_twr(bench_prices: np.ndarray, starting_capital: float,
               weekly_contribution: float, rebalance_freq: int) -> float:
    """
    Contribution-adjusted TWR for a buy-and-hold benchmark receiving the same cash schedule.
    Invests all cash into the benchmark on contribution days at the day's closing price.
    """
    n = len(bench_prices)
    shares = 0.0
    cash   = starting_capital
    ca = np.zeros(n)
    for d in range(n):
        p = bench_prices[d]
        contrib = 0.0
        if d == 0 and p > 0:
            shares = cash / p
            cash   = 0.0
        elif d > 0 and d % rebalance_freq == 0 and p > 0:
            contrib = weekly_contribution
            shares += contrib / p
            cash   = 0.0
        val = shares * p + cash
        if d == 0:
            ca[0] = val
        else:
            prev = shares * bench_prices[d - 1] + cash
            factor = (val - contrib) / max(prev, 1e-9)
            ca[d] = ca[d - 1] * factor
    m = compute_performance_metrics(ca)
    return m["total_return"]


def run_simulation(
    precomp: PrecomputedData,
    params: np.ndarray,
    starting_capital: float = 10_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    weekly_contribution: float = 0.0,
    rebalance_frequency_days: int = 5,
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
    n_etfs = precomp.etf_prices.shape[1]

    index_pct        = float(params[4])
    metric_threshold = float(params[5])
    take_profit_pct  = float(params[6])
    sell_weak_below  = float(params[7])
    trailing_stop    = float(params[8])   # negative

    base_slippage    = slippage_bps / 10_000.0
    bp_cfg           = BACKTEST_PARAMS
    use_vol_slip     = bp_cfg.get("vol_slippage_scaling", True)
    vol_slip_mult    = bp_cfg.get("vol_slippage_multiplier", 2.0)
    cooldown_sell    = bp_cfg.get("cooldown_days_after_sell", 3)
    cooldown_stop    = bp_cfg.get("cooldown_days_after_stopout", 7)
    max_trades_week  = bp_cfg.get("max_trades_per_week", 10)

    min_order        = RISK_LIMITS["min_order_amount"]
    max_single_pct   = RISK_LIMITS["max_single_position_pct"]
    max_sector_pct   = RISK_LIMITS["max_sector_pct"]
    max_order_pct    = RISK_LIMITS["max_order_pct_of_cash"]
    max_buys         = RISK_LIMITS["max_buys_per_rebalance"]

    current_scores = score_stocks_at_day(precomp, params, 0)
    candidate_mask = current_scores >= metric_threshold

    # Portfolio state
    stock_shares    = np.zeros(n_stocks)
    stock_avg_cost  = np.zeros(n_stocks)
    stock_peak      = np.zeros(n_stocks)
    stock_day_bought= np.full(n_stocks, -1, dtype=np.int32)
    stock_day_sold  = np.full(n_stocks, -99, dtype=np.int32)  # cooldown tracking
    stock_stopout   = np.zeros(n_stocks, dtype=bool)           # True = sold via stop-loss
    etf_shares      = np.zeros(n_etfs)

    cash = float(starting_capital)
    daily_values    = np.zeros(n_days)
    ca_daily_values = np.zeros(n_days)
    total_contributions = float(starting_capital)

    trades_made     = 0
    sells_made      = 0
    skipped_buys    = 0
    cap_reductions  = 0
    cooldown_skips  = 0
    stopout_count   = 0
    trades_this_week= 0
    week_start_day  = 0
    total_positions_sum = 0
    max_positions   = 0
    total_cash_pct_sum = 0.0
    total_friction  = 0.0
    total_traded_notional = 0.0
    regime_days     = {"bullish": 0, "neutral": 0, "defensive": 0}

    def _effective_slippage(stock_idx: int) -> float:
        """Volatility-scaled slippage: higher vol → worse fills."""
        if not use_vol_slip or precomp.vol_3m_daily is None:
            return base_slippage
        v = precomp.vol_3m_daily[max(0, min(n_days - 1, _cur_day))]
        vol = float(v[stock_idx]) if np.isfinite(v[stock_idx]) else 0.20
        return base_slippage * (1.0 + vol_slip_mult * vol)

    _cur_day = 0  # updated in loop closure

    def _current_portfolio_value(day: int) -> float:
        prices_d = precomp.prices[day]
        valid = np.isfinite(prices_d) & (prices_d > 0)
        sv = float(np.sum(stock_shares * np.where(valid, prices_d, stock_avg_cost)))
        ev = float(np.sum(etf_shares * precomp.etf_prices[day])) if n_etfs > 0 else 0.0
        return cash + sv + ev

    def _sector_exposures(day: int) -> dict:
        prices_d = precomp.prices[day]
        valid = np.isfinite(prices_d) & (prices_d > 0)
        exposure: dict = {}
        for i in np.where(stock_shares > 0)[0]:
            s = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"
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
        sector_exp = _sector_exposures(day)
        spent = 0.0
        buys_this_pass = 0

        candidate_indices = sorted(np.where(eligible)[0], key=lambda i: -current_scores[i])

        for i in candidate_indices:
            if buys_this_pass >= max_buys or trades_this_week >= max_trades_week:
                skipped_buys += 1
                continue

            # Cooldown: skip if sold recently (longer cooldown for stopouts)
            days_since_sell = day - int(stock_day_sold[i])
            required_cd = cooldown_stop if stock_stopout[i] else cooldown_sell
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
                sector = precomp.sector_labels[i] if i < len(precomp.sector_labels) else "Unknown"
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

            p = prices_d[i]
            slip = _effective_slippage(i)
            effective_price = p * (1.0 + slip)
            shares = alloc / effective_price
            friction = alloc * slip + commission_per_trade
            total_friction += friction

            if stock_shares[i] > 0:
                old_cost = stock_avg_cost[i] * stock_shares[i]
                stock_avg_cost[i] = (old_cost + alloc) / (stock_shares[i] + shares)
            else:
                stock_avg_cost[i] = effective_price
                stock_day_bought[i] = day
                trades_made += 1
                trades_this_week += 1
            stock_shares[i] += shares
            stock_peak[i] = max(stock_peak[i], p)
            sector_exp[sector] = sector_exp.get(sector, 0.0) + alloc
            spent += alloc
            total_traded_notional += alloc
            buys_this_pass += 1

        cash -= spent
        return spent

    # Day 0: ETF buy + initial stock deployment
    if n_etfs > 0 and index_pct > 0:
        etf_budget = cash * index_pct
        p0_etf = precomp.etf_prices[0]
        valid_etfs = np.isfinite(p0_etf) & (p0_etf > 0)
        n_valid = int(valid_etfs.sum())
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

        # Weekly trade budget reset
        if d > 0 and (d - week_start_day) >= rebalance_frequency_days:
            trades_this_week = 0
            week_start_day   = d

        # Trailing-stop peak update
        valid_price = np.isfinite(prices) & (prices > 0)
        stock_peak  = np.where(held & valid_price, np.maximum(stock_peak, prices), stock_peak)

        # Sell conditions
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

        stop_loss_mask  = held & (pct_from_avg  <= _STOP_LOSS_PCT)
        trail_mask      = held & (pct_from_peak <= trailing_stop)
        tp_mask         = held & (pct_from_avg  >= take_profit_pct) & take_profit_ok
        weak_val_mask   = held & (current_scores < sell_weak_below) & (days_held >= _MIN_DAYS_HELD_BEFORE_VALUE_EXIT)
        sell_mask       = stop_loss_mask | trail_mask | tp_mask | weak_val_mask

        if sell_mask.any():
            sell_prices   = np.where(valid_price, prices, stock_avg_cost)
            sell_notional = float(np.sum(stock_shares[sell_mask] * sell_prices[sell_mask]))
            sell_indices  = np.where(sell_mask)[0]
            for i in sell_indices:
                slip = _effective_slippage(i)
                proceeds_i = float(stock_shares[i] * sell_prices[i] * (1.0 - slip))
                total_friction += float(stock_shares[i] * sell_prices[i] * slip) + commission_per_trade
                cash += proceeds_i
                is_stopout = bool(stop_loss_mask[i] or trail_mask[i])
                stock_day_sold[i]  = d
                stock_stopout[i]   = is_stopout
                if is_stopout:
                    stopout_count += 1
            total_traded_notional += sell_notional
            sells_made += int(sell_mask.sum())
            stock_shares[sell_mask]    = 0.0
            stock_avg_cost[sell_mask]  = 0.0
            stock_peak[sell_mask]      = 0.0
            stock_day_bought[sell_mask]= -1

        # Rebalance + contribution
        is_contrib_day = d > 0 and d % rebalance_frequency_days == 0
        contrib_today  = weekly_contribution if is_contrib_day else 0.0
        if is_contrib_day:
            current_scores = score_stocks_at_day(precomp, params, d)
            candidate_mask = current_scores >= metric_threshold
            cash += weekly_contribution
            total_contributions += weekly_contribution
            if cash >= min_order:
                _do_buy(d, cash)

        etf_value   = float(np.sum(etf_shares * precomp.etf_prices[d])) if n_etfs > 0 else 0.0
        stock_value = float(np.sum(stock_shares * np.where(valid_price, prices, stock_avg_cost)))
        port_val    = cash + stock_value + etf_value
        daily_values[d] = port_val

        if d == 0:
            ca_daily_values[0] = port_val
        else:
            prev_port = daily_values[d - 1]
            factor = (port_val - contrib_today) / max(prev_port, 1e-9)
            ca_daily_values[d] = ca_daily_values[d - 1] * factor

        # Regime classification (simple: uses SPY vs benchmark_prices for proxy)
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
        max_positions = max(max_positions, n_pos)
        total_cash_pct_sum += (cash / max(port_val, 1e-9))

    final_value = float(daily_values[-1])
    metrics     = compute_performance_metrics(ca_daily_values)
    avg_port    = float(daily_values[daily_values > 0].mean()) if daily_values.any() else starting_capital
    turnover    = total_traded_notional / max(avg_port, 1.0)
    profit      = final_value - total_contributions

    # Contribution-adjusted benchmark TWR
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
    )


def _extract_closes(raw: "pd.DataFrame", all_tickers: list[str]) -> "pd.DataFrame":
    """Extract Close prices from a yfinance download result (handles MultiIndex)."""
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": all_tickers[0]})
    return closes.ffill().bfill()


def load_and_precompute(
    n_days: int,
    max_symbols: int = 300,
    mode: str | None = None,
    universe_selection: str | None = None,
    min_volume: float | None = None,
    random_seed: int | None = None,
    benchmark_symbol: str | None = None,
) -> PrecomputedData:
    """
    Load fundamentals and download price history.

    Defaults for mode/universe_selection/min_volume/random_seed/benchmark_symbol
    come from BACKTEST_PARAMS so existing callers (tuner.py) need no changes.
    """
    mode              = mode              or BACKTEST_PARAMS["default_mode"]
    universe_selection= universe_selection or BACKTEST_PARAMS["universe_selection"]
    min_volume        = min_volume        if min_volume is not None else BACKTEST_PARAMS["min_volume"]
    random_seed       = random_seed       if random_seed is not None else BACKTEST_PARAMS["random_seed"]
    benchmark_symbol  = benchmark_symbol  or BACKTEST_PARAMS["benchmark_symbol"]

    agg_df = read_data_as_pd("agg_data")
    if agg_df is None or agg_df.empty:
        raise RuntimeError(
            "No fundamental data in data/agg_data.csv. "
            "Run the strategy without --tune first to generate data."
        )

    for col in ["value_metric", "volume", "pe_comp", "pb_comp",
                "quality_score", "income_score", "position_52w", "return_1m"]:
        if col in agg_df.columns:
            agg_df[col] = pd.to_numeric(agg_df[col], errors="coerce")

    agg_df = agg_df.dropna(subset=["symbol"]).copy()
    agg_df["volume"] = agg_df["volume"].fillna(0)
    agg_df["value_metric"] = agg_df["value_metric"].fillna(0)

    agg_df, lookahead_bias = select_backtest_universe(
        agg_df, mode, universe_selection, max_symbols, min_volume, random_seed
    )

    symbols = agg_df["symbol"].tolist()
    etf_list = [e for e in ETFS if e not in set(symbols)]
    benchmark_tickers = [benchmark_symbol] if benchmark_symbol not in set(symbols + etf_list) else []

    n_cal_days = int(n_days * 1.6) + 30
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=n_cal_days)

    all_tickers = symbols + etf_list + benchmark_tickers
    print(
        f"Downloading price history for {len(all_tickers)} tickers "
        f"({n_days} trading days, mode={mode}, bias={lookahead_bias}) …"
    )

    raw = yf.download(
        all_tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")

    closes = _extract_closes(raw, all_tickers)

    if len(closes) < n_days:
        raise RuntimeError(
            f"Only {len(closes)} trading days available; requested {n_days}. "
            "Reduce --tune window."
        )
    closes = closes.iloc[-n_days:]

    stock_cols = [s for s in symbols if s in closes.columns and closes[s].notna().any()]
    etf_cols   = [e for e in etf_list if e in closes.columns and closes[e].notna().any()]

    if not stock_cols:
        raise RuntimeError("No usable stock price data after download.")

    # Benchmark prices
    bench_prices = np.full(n_days, np.nan)
    if benchmark_symbol in closes.columns and closes[benchmark_symbol].notna().any():
        bench_prices = closes[benchmark_symbol].values.astype(np.float64)
    else:
        logger.warning(f"Benchmark {benchmark_symbol} not available in price data")

    # Re-align fundamentals to available + priced stocks
    agg_df = agg_df[agg_df["symbol"].isin(set(stock_cols))].copy()
    sym_order = [s for s in stock_cols if s in agg_df["symbol"].values]
    agg_df = agg_df.set_index("symbol").loc[sym_order].reset_index()
    stock_cols = agg_df["symbol"].tolist()

    stock_prices = closes[stock_cols].values.astype(np.float64)
    etf_prices_arr = (
        closes[etf_cols].values.astype(np.float64)
        if etf_cols
        else np.zeros((n_days, 0), dtype=np.float64)
    )

    pe_comp       = _col_arr(agg_df, "pe_comp")
    pb_comp       = _col_arr(agg_df, "pb_comp")
    quality_scores= _col_arr(agg_df, "quality_score")
    income_scores = _col_arr(agg_df, "income_score")
    volume_arr    = _col_arr(agg_df, "volume")

    sector_labels = (
        agg_df["sector"].fillna("Unknown").tolist()
        if "sector" in agg_df.columns
        else ["Unknown"] * len(agg_df)
    )

    yield_trap_mask = (
        agg_df["yield_trap_flag"].fillna(False).astype(bool).values
        if "yield_trap_flag" in agg_df.columns
        else np.zeros(len(agg_df), dtype=bool)
    )

    pos_arr = _col_arr(agg_df, "position_52w", default=np.nan)
    pos_arr = np.where(np.isfinite(pos_arr), pos_arr, np.nan)
    ret_arr = _col_arr(agg_df, "return_1m", default=np.nan)
    ret_arr = np.where(np.isfinite(ret_arr), ret_arr, np.nan)
    has_pos = np.isfinite(pos_arr)

    boundaries = np.array(MOMENTUM_PARAMS["position_bin_boundaries"])
    bin_indices = np.searchsorted(
        boundaries, np.where(has_pos, pos_arr, 0.5), side="right"
    ).astype(np.int32)

    # walk_forward_price_only_test: zero fundamental arrays so only momentum drives scores
    if mode == "walk_forward_price_only_test":
        pe_comp       = np.zeros(len(agg_df), dtype=np.float64)
        pb_comp       = np.zeros(len(agg_df), dtype=np.float64)
        quality_scores= np.zeros(len(agg_df), dtype=np.float64)
        income_scores = np.zeros(len(agg_df), dtype=np.float64)
        logger.info("walk_forward_price_only_test: fundamental arrays zeroed — momentum only")

    # Precompute rolling daily price features for dynamic re-scoring in simulation
    n_stocks = len(stock_cols)
    pos_52w_daily    = np.full((n_days, n_stocks), np.nan)
    ret_1m_daily     = np.full((n_days, n_stocks), np.nan)
    bin_indices_daily= np.zeros((n_days, n_stocks), dtype=np.int32)

    # Momentum v2 causal rolling arrays (NaN until window fills)
    ret_5d_daily       = np.full((n_days, n_stocks), np.nan)
    ret_3m_daily       = np.full((n_days, n_stocks), np.nan)
    ret_6m_daily       = np.full((n_days, n_stocks), np.nan)
    rs_3m_daily        = np.full((n_days, n_stocks), np.nan)
    rs_6m_daily        = np.full((n_days, n_stocks), np.nan)
    vol_3m_daily       = np.full((n_days, n_stocks), np.nan)
    above_50dma_daily  = np.zeros((n_days, n_stocks), dtype=bool)
    above_200dma_daily = np.zeros((n_days, n_stocks), dtype=bool)

    for d in range(n_days):
        curr = stock_prices[d]

        # Rolling 52-week (252 trading day) position — only uses prices up to day d
        win_start = max(0, d - 251)
        window = stock_prices[win_start : d + 1]
        with np.errstate(invalid="ignore"):
            lo = np.nanmin(window, axis=0)
            hi = np.nanmax(window, axis=0)
        rng = hi - lo
        valid52 = (rng > 0) & np.isfinite(curr)
        raw_pos = np.where(valid52, (curr - lo) / rng, np.nan)
        pos_52w_daily[d] = np.clip(raw_pos, 0.0, 1.0)

        # Rolling 21-day return
        if d >= 21:
            prev = stock_prices[d - 21]
            valid1m = (prev > 0) & np.isfinite(prev) & np.isfinite(curr)
            ret_1m_daily[d] = np.where(valid1m, curr / prev - 1.0, np.nan)

        # Bin indices from rolling position
        valid_pos_d = np.where(np.isfinite(pos_52w_daily[d]), pos_52w_daily[d], 0.5)
        bin_indices_daily[d] = np.searchsorted(boundaries, valid_pos_d, side="right").astype(np.int32)

        # ── Momentum v2 features (all causal) ──────────────────────────────
        # 5-day return
        if d >= 5:
            p5 = stock_prices[d - 5]
            valid5 = (p5 > 0) & np.isfinite(p5) & np.isfinite(curr)
            ret_5d_daily[d] = np.where(valid5, curr / p5 - 1.0, np.nan)

        # 3-month (63-day) return + realized vol + RS vs benchmark
        if d >= 63:
            p63 = stock_prices[d - 63]
            valid63 = (p63 > 0) & np.isfinite(p63) & np.isfinite(curr)
            ret_3m_daily[d] = np.where(valid63, curr / p63 - 1.0, np.nan)

            # Annualized realized vol over last 63 daily returns
            w63 = stock_prices[d - 63: d + 1]          # (64, n_stocks)
            p_prev, p_next = w63[:-1], w63[1:]
            ok63 = (p_prev > 0) & np.isfinite(p_prev) & np.isfinite(p_next)
            dr63 = np.where(ok63, p_next / p_prev - 1.0, np.nan)
            with np.errstate(invalid="ignore"):
                vol_3m_daily[d] = np.nanstd(dr63, axis=0) * np.sqrt(252)

            # RS 3m vs benchmark
            sp63 = bench_prices[d - 63]
            sp_d = bench_prices[d]
            if np.isfinite(sp63) and sp63 > 0 and np.isfinite(sp_d):
                spy_r3m = sp_d / sp63 - 1.0
                rs_3m_daily[d] = np.where(np.isfinite(ret_3m_daily[d]),
                                           ret_3m_daily[d] - spy_r3m, np.nan)

        # 6-month (126-day) return + RS vs benchmark
        if d >= 126:
            p126 = stock_prices[d - 126]
            valid126 = (p126 > 0) & np.isfinite(p126) & np.isfinite(curr)
            ret_6m_daily[d] = np.where(valid126, curr / p126 - 1.0, np.nan)

            sp126 = bench_prices[d - 126]
            if np.isfinite(sp126) and sp126 > 0 and np.isfinite(bench_prices[d]):
                spy_r6m = bench_prices[d] / sp126 - 1.0
                rs_6m_daily[d] = np.where(np.isfinite(ret_6m_daily[d]),
                                           ret_6m_daily[d] - spy_r6m, np.nan)

        # 50-day MA filter
        if d >= 50:
            w50 = stock_prices[d - 49: d + 1]           # 50 bars
            with np.errstate(invalid="ignore"):
                ma50 = np.nanmean(w50, axis=0)
            above_50dma_daily[d] = np.isfinite(curr) & (curr > 0) & (curr > ma50)

        # 200-day MA filter
        if d >= 200:
            w200 = stock_prices[d - 199: d + 1]          # 200 bars
            with np.errstate(invalid="ignore"):
                ma200 = np.nanmean(w200, axis=0)
            above_200dma_daily[d] = np.isfinite(curr) & (curr > 0) & (curr > ma200)

    has_pos_daily = np.isfinite(pos_52w_daily)

    # Baseline scores using current config (for diff display in tuner)
    cur_mbin = np.array(MOMENTUM_PARAMS["position_bin_scores"])
    cur_value = (
        SCORING_PARAMS["value_pe_weight"] * pe_comp
        + SCORING_PARAMS["value_pb_weight"] * pb_comp
    )
    cur_mom = _momentum_score_vec(bin_indices, has_pos, pos_arr, ret_arr, cur_mbin)
    sw = np.array([SCORE_WEIGHTS["value"], SCORE_WEIGHTS["quality"],
                   SCORE_WEIGHTS["income"], SCORE_WEIGHTS["momentum"]])
    sw = sw / sw.sum()
    baseline_scores = (
        sw[0] * cur_value
        + sw[1] * quality_scores
        + sw[2] * income_scores
        + sw[3] * cur_mom
    )

    print(
        f"Precomputed: {n_stocks} stocks, {len(etf_cols)} ETFs, {n_days} trading days "
        f"(lookahead bias: {lookahead_bias}). Ready."
    )
    return PrecomputedData(
        symbols=stock_cols,
        prices=stock_prices,
        pe_comp=pe_comp,
        pb_comp=pb_comp,
        quality_scores=quality_scores,
        income_scores=income_scores,
        yield_trap_mask=yield_trap_mask,
        bin_indices=bin_indices,
        has_position_52w=has_pos,
        position_52w_arr=pos_arr,
        return_1m_arr=ret_arr,
        etf_symbols=etf_cols,
        etf_prices=etf_prices_arr,
        baseline_scores=baseline_scores,
        sector_labels=sector_labels,
        volume_arr=volume_arr,
        mode=mode,
        universe_selection=universe_selection,
        lookahead_bias_level=lookahead_bias,
        benchmark_prices=bench_prices,
        benchmark_symbol=benchmark_symbol,
        position_52w_daily=pos_52w_daily,
        return_1m_daily=ret_1m_daily,
        bin_indices_daily=bin_indices_daily,
        has_position_52w_daily=has_pos_daily,
        ret_5d_daily=ret_5d_daily,
        ret_3m_daily=ret_3m_daily,
        ret_6m_daily=ret_6m_daily,
        rs_3m_daily=rs_3m_daily,
        rs_6m_daily=rs_6m_daily,
        vol_3m_daily=vol_3m_daily,
        above_50dma_daily=above_50dma_daily,
        above_200dma_daily=above_200dma_daily,
        spy_prices=bench_prices,
    )


def run_backtest_report(
    precomp: PrecomputedData,
    params: np.ndarray,
    train_slice: slice,
    val_slice: "slice | None",
) -> BacktestReport:
    """
    Run strategy on train window, optionally evaluate on validation window,
    and compute benchmark metrics.  Returns a BacktestReport.
    """
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
    train_n = train_precomp.prices.shape[0]
    train_result = run_simulation(
        train_precomp,
        params,
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
                val_precomp,
                params,
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

    train_bench = _bench_metrics(train_slice)
    val_bench_return = 0.0
    if val_slice is not None:
        val_bench_return = _bench_metrics(val_slice)["total_return"]

    # Contribution-adjusted benchmark TWR for apples-to-apples comparison
    def _twr_for_slice(s: slice) -> float:
        vals = precomp.benchmark_prices[s]
        if len(vals) >= 2 and np.isfinite(vals).all() and vals[0] > 0:
            return _bench_twr(
                vals, bp["starting_capital"],
                bp["weekly_contribution"], bp["rebalance_frequency_days"],
            )
        return 0.0

    train_bench_twr = _twr_for_slice(train_slice)
    val_bench_twr   = _twr_for_slice(val_slice) if val_slice is not None else 0.0

    excess = train_result.total_return - train_bench["total_return"]
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
    )


def print_backtest_report(report: BacktestReport) -> None:
    """Print a formatted BacktestReport to stdout."""
    r = report
    tr = r.train_result
    print(f"\n{'=' * 64}")
    print(f"BACKTEST REPORT  [{r.mode}  sel={r.universe_selection}  bias={r.lookahead_bias_level}]")
    print(f"{'=' * 64}")
    print(f"  Universe: {r.n_symbols} symbols, {r.n_days} trading days")
    print(f"\n  TRAIN WINDOW")
    print(f"    Return (TWR):    {tr.total_return:+.2%}  (bench TWR {r.train_benchmark_twr:+.2%})")
    print(f"    Benchmark (buy-hold):  {r.benchmark_return:+.2%}")
    print(f"    Excess return:   {r.excess_return:+.2%}")
    print(f"    Sharpe:          {tr.sharpe:+.3f}  (benchmark {r.benchmark_sharpe:+.3f})")
    print(f"    Calmar:          {tr.calmar:+.3f}")
    print(f"    Max drawdown:    {tr.max_drawdown:.2%}  (benchmark {r.benchmark_max_drawdown:.2%})")
    print(f"    Final value:     ${tr.final_value:,.2f}  contributions=${tr.net_contributions:,.2f}  profit=${tr.profit:,.2f}")
    print(f"    Trades:          {tr.trades_made}  sells={tr.sells_made}  skipped={tr.skipped_buys}")
    print(f"    Stopouts:        {tr.stopout_count}  cooldown skips={tr.cooldown_skips}")
    print(f"    Cap reductions:  {tr.cap_reductions}")
    print(f"    Avg positions:   {tr.average_positions:.1f}  max={tr.max_positions}")
    print(f"    Avg cash %:      {tr.average_cash_pct:.1%}")
    print(f"    Friction cost:   ${tr.friction_cost:.2f}  turnover={tr.turnover_estimate:.4f}")
    if tr.regime_days:
        rd = tr.regime_days
        total_rd = max(sum(rd.values()), 1)
        print(
            f"    Regime days:     bullish={rd['bullish']} ({rd['bullish']/total_rd:.0%})  "
            f"neutral={rd['neutral']} ({rd['neutral']/total_rd:.0%})  "
            f"defensive={rd['defensive']} ({rd['defensive']/total_rd:.0%})"
        )
    if r.validation_result:
        vr = r.validation_result
        print(f"\n  VALIDATION WINDOW")
        print(f"    Return (TWR):    {vr.total_return:+.2%}  (bench TWR {r.val_benchmark_twr:+.2%})")
        print(f"    Benchmark:       {r.validation_benchmark_return:+.2%}")
        print(f"    Sharpe:          {vr.sharpe:+.3f}")
        print(f"    Calmar:          {vr.calmar:+.3f}")
        print(f"    Max drawdown:    {vr.max_drawdown:.2%}")
        print(f"    Stopouts:        {vr.stopout_count}  cooldown skips={vr.cooldown_skips}")
        if vr.regime_days:
            rd = vr.regime_days
            total_rd = max(sum(rd.values()), 1)
            print(
                f"    Regime days:     bullish={rd['bullish']} ({rd['bullish']/total_rd:.0%})  "
                f"neutral={rd['neutral']} ({rd['neutral']/total_rd:.0%})  "
                f"defensive={rd['defensive']} ({rd['defensive']/total_rd:.0%})"
            )
    if r.notes:
        print(f"\n  NOTES")
        for n in r.notes:
            print(f"    • {n}")
    print("=" * 64)
