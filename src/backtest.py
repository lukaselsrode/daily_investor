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
    ETFS,
    MOMENTUM_PARAMS,
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
    prices: np.ndarray           # (n_days, n_stocks) float64
    pe_comp: np.ndarray          # (n_stocks,)
    pb_comp: np.ndarray          # (n_stocks,)
    quality_scores: np.ndarray   # (n_stocks,)
    income_scores: np.ndarray    # (n_stocks,)
    yield_trap_mask: np.ndarray  # (n_stocks,) bool
    bin_indices: np.ndarray      # (n_stocks,) int 0-4
    has_position_52w: np.ndarray # (n_stocks,) bool
    position_52w_arr: np.ndarray # (n_stocks,) float, NaN where missing
    return_1m_arr: np.ndarray    # (n_stocks,) float, NaN where missing
    etf_symbols: list[str]
    etf_prices: np.ndarray       # (n_days, n_etfs) float64
    baseline_scores: np.ndarray  # (n_stocks,) scored with current config


@dataclass
class SimResult:
    final_value: float
    total_return: float
    sharpe: float
    calmar: float
    max_drawdown: float
    trades_made: int


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
    """Vectorized momentum scoring using (potentially trial) bin scores."""
    mp = MOMENTUM_PARAMS
    base = np.where(has_pos, mbin_scores[bin_indices], 0.0)

    has_r1m = np.isfinite(return_1m)
    low_pos = has_pos & (pos_52w < mp["return_1m_low_position_cutoff"]) & has_r1m
    recovery = low_pos & (return_1m >= mp["return_1m_recovery_threshold"])
    falling = low_pos & (return_1m <= mp["return_1m_falling_knife_threshold"])

    base = base + np.where(recovery, mp["return_1m_recovery_bonus"], 0.0)
    base = base - np.where(falling, mp["return_1m_falling_knife_penalty"], 0.0)
    return base


def score_stocks(precomp: PrecomputedData, params: np.ndarray) -> np.ndarray:
    """Compute per-stock scores for a trial parameter vector."""
    raw_sw = params[:4]
    sw = raw_sw / max(raw_sw.sum(), 1e-9)
    value_pe_w = params[9]
    value_pb_w = 1.0 - value_pe_w
    mbin_scores = params[10:15]

    value_score = value_pe_w * precomp.pe_comp + value_pb_w * precomp.pb_comp
    momentum_score = _momentum_score_vec(
        precomp.bin_indices,
        precomp.has_position_52w,
        precomp.position_52w_arr,
        precomp.return_1m_arr,
        mbin_scores,
    )
    return (
        sw[0] * value_score
        + sw[1] * precomp.quality_scores
        + sw[2] * precomp.income_scores
        + sw[3] * momentum_score
    )


def run_simulation(
    precomp: PrecomputedData,
    params: np.ndarray,
    starting_capital: float = 10_000.0,
) -> SimResult:
    """
    Simulate the strategy over the precomputed price history.

    params layout (15 values):
      [0] sw_value  [1] sw_quality  [2] sw_income  [3] sw_momentum
      [4] index_pct  [5] metric_threshold  [6] take_profit_pct
      [7] sell_weak_below  [8] trailing_stop  [9] value_pe_weight
      [10-14] mbin_0..4
    """
    n_days, n_stocks = precomp.prices.shape
    n_etfs = precomp.etf_prices.shape[1]

    index_pct = float(params[4])
    metric_threshold = float(params[5])
    take_profit_pct = float(params[6])
    sell_weak_below = float(params[7])
    trailing_stop = float(params[8])  # negative

    trial_scores = score_stocks(precomp, params)
    candidate_mask = trial_scores >= metric_threshold

    # Portfolio state
    stock_shares = np.zeros(n_stocks)
    stock_avg_cost = np.zeros(n_stocks)
    stock_peak = np.zeros(n_stocks)
    stock_day_bought = np.full(n_stocks, -1, dtype=np.int32)
    etf_shares = np.zeros(n_etfs)

    cash = float(starting_capital)
    daily_values = np.zeros(n_days)
    trades_made = 0

    def _do_buy(day: int, budget: float) -> float:
        nonlocal cash, trades_made
        if budget < 5.0:
            return 0.0
        prices_d = precomp.prices[day]
        eligible = candidate_mask & np.isfinite(prices_d) & (prices_d > 0)
        if not eligible.any():
            return 0.0
        total_score = trial_scores[eligible].sum()
        if total_score <= 0:
            return 0.0
        spent = 0.0
        for i in np.where(eligible)[0]:
            alloc = (trial_scores[i] / total_score) * budget
            if alloc < 5.0:
                continue
            p = prices_d[i]
            shares = alloc / p
            if stock_shares[i] > 0:
                old_cost = stock_avg_cost[i] * stock_shares[i]
                stock_avg_cost[i] = (old_cost + alloc) / (stock_shares[i] + shares)
            else:
                stock_avg_cost[i] = p
                stock_day_bought[i] = day
                trades_made += 1
            stock_shares[i] += shares
            stock_peak[i] = max(stock_peak[i], p)
            spent += alloc
        cash -= spent
        return spent

    # Day 0: ETF buy + initial stock buy
    if n_etfs > 0 and index_pct > 0:
        etf_budget = cash * index_pct
        p0_etf = precomp.etf_prices[0]
        valid_etfs = np.isfinite(p0_etf) & (p0_etf > 0)
        n_valid = int(valid_etfs.sum())
        if n_valid > 0:
            per_etf = etf_budget / n_valid
            for j in np.where(valid_etfs)[0]:
                etf_shares[j] = per_etf / p0_etf[j]
                cash -= per_etf

    stock_budget_day0 = cash
    _do_buy(0, stock_budget_day0)

    for d in range(n_days):
        prices = precomp.prices[d]
        held = stock_shares > 0

        # Update trailing-stop peaks
        valid_price = np.isfinite(prices) & (prices > 0)
        update_peak = held & valid_price
        stock_peak = np.where(update_peak, np.maximum(stock_peak, prices), stock_peak)

        # Sell conditions — only evaluate held positions with valid prices
        with np.errstate(invalid="ignore", divide="ignore"):
            pct_from_avg = np.where(
                held & (stock_avg_cost > 0) & valid_price,
                prices / stock_avg_cost - 1.0,
                0.0,
            )
            pct_from_peak = np.where(
                held & (stock_peak > 0) & valid_price,
                prices / stock_peak - 1.0,
                0.0,
            )

        days_held = np.where(stock_day_bought >= 0, d - stock_day_bought, 0)
        take_profit_ok = trial_scores < metric_threshold * _TAKE_PROFIT_FLOOR_MULTIPLIER

        sell_mask = (
            (held & (pct_from_avg <= _STOP_LOSS_PCT))
            | (held & (pct_from_peak <= trailing_stop))
            | (held & (pct_from_avg >= take_profit_pct) & take_profit_ok)
            | (held & (trial_scores < sell_weak_below) & (days_held >= _MIN_DAYS_HELD_BEFORE_VALUE_EXIT))
        )

        if sell_mask.any():
            proceeds = float(np.sum(stock_shares[sell_mask] * prices[sell_mask]))
            cash += proceeds
            stock_shares[sell_mask] = 0.0
            stock_avg_cost[sell_mask] = 0.0
            stock_peak[sell_mask] = 0.0
            stock_day_bought[sell_mask] = -1

        # Weekly rebalance buy (sell proceeds get reinvested)
        if d > 0 and d % 5 == 0 and cash >= 5.0:
            _do_buy(d, cash)

        etf_value = float(np.sum(etf_shares * precomp.etf_prices[d])) if n_etfs > 0 else 0.0
        stock_value = float(np.sum(stock_shares * np.where(valid_price, prices, stock_avg_cost)))
        daily_values[d] = cash + stock_value + etf_value

    final_value = float(daily_values[-1])
    total_return = (final_value / starting_capital) - 1.0

    valid_vals = daily_values[daily_values > 0]
    if len(valid_vals) > 1:
        daily_returns = np.diff(valid_vals) / valid_vals[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]
    else:
        daily_returns = np.array([])

    sharpe = 0.0
    if len(daily_returns) > 2 and daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))

    cum = daily_values / max(daily_values[0], 1e-9)
    roll_max = np.maximum.accumulate(cum)
    drawdowns = np.where(roll_max > 0, cum / roll_max - 1.0, 0.0)
    max_drawdown = float(drawdowns.min())

    calmar = 0.0
    if max_drawdown < -0.001:
        calmar = float(total_return / abs(max_drawdown))

    return SimResult(
        final_value=final_value,
        total_return=total_return,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=max_drawdown,
        trades_made=trades_made,
    )


def load_and_precompute(n_days: int, max_symbols: int = 300) -> PrecomputedData:
    """Load fundamentals and download price history. Call once before optimization."""
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
    agg_df = agg_df[agg_df["volume"] >= 100_000]
    agg_df["value_metric"] = agg_df["value_metric"].fillna(0)
    agg_df = agg_df.sort_values("value_metric", ascending=False).head(max_symbols).reset_index(drop=True)

    symbols = agg_df["symbol"].tolist()
    etf_list = [e for e in ETFS if e not in set(symbols)]

    n_cal_days = int(n_days * 1.6) + 30
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=n_cal_days)

    all_tickers = symbols + etf_list
    print(f"Downloading price history for {len(all_tickers)} tickers ({n_days} trading days) …")

    raw = yf.download(
        all_tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")

    # Close column extraction — handles both MultiIndex and flat column layouts
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["close"]
    else:
        # Single ticker downloaded (edge case)
        closes = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    closes = closes.ffill().bfill()

    if len(closes) < n_days:
        raise RuntimeError(
            f"Only {len(closes)} trading days available; requested {n_days}. "
            "Reduce --tune window."
        )
    closes = closes.iloc[-n_days:]

    stock_cols = [s for s in symbols if s in closes.columns and closes[s].notna().any()]
    etf_cols = [e for e in etf_list if e in closes.columns and closes[e].notna().any()]

    if not stock_cols:
        raise RuntimeError("No usable stock price data after download.")

    # Re-align fundamentals to available + priced stocks
    agg_df = agg_df[agg_df["symbol"].isin(set(stock_cols))].copy()
    sym_order = [s for s in stock_cols if s in agg_df["symbol"].values]
    agg_df = agg_df.set_index("symbol").loc[sym_order].reset_index()
    stock_cols = agg_df["symbol"].tolist()

    stock_prices = closes[stock_cols].values.astype(np.float64)
    etf_prices = (
        closes[etf_cols].values.astype(np.float64)
        if etf_cols
        else np.zeros((n_days, 0), dtype=np.float64)
    )

    pe_comp = _col_arr(agg_df, "pe_comp")
    pb_comp = _col_arr(agg_df, "pb_comp")
    quality_scores = _col_arr(agg_df, "quality_score")
    income_scores = _col_arr(agg_df, "income_score")

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

    # Baseline scores using current config (for diff display)
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

    n_stocks = len(stock_cols)
    print(
        f"Precomputed: {n_stocks} stocks, {len(etf_cols)} ETFs, {n_days} trading days. "
        f"Ready for optimization."
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
        etf_prices=etf_prices,
        baseline_scores=baseline_scores,
    )
