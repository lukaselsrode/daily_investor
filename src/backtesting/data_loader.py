"""
backtesting/data_loader.py — Data loading and universe selection for backtesting.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from util import (
    BACKTEST_PARAMS,
    ETFS,
    MOMENTUM_PARAMS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    read_data_as_pd,
)

from .simulator import _col_arr, _momentum_score_vec, split_price_window  # noqa: F401 (re-exported)
from .types import PrecomputedData

logger = logging.getLogger(__name__)


def _yf_ticker(sym: str) -> str:
    """
    Normalize a ticker symbol for yfinance.
    yfinance uses hyphens where US exchanges use dots: BRK.B → BRK-B.
    Also strips a leading $ that occasionally slips through scrapers.
    """
    return sym.lstrip("$").replace(".", "-")


def _extract_closes(raw: pd.DataFrame, all_tickers: list[str]) -> pd.DataFrame:
    """Extract Close prices from a yfinance download result (handles MultiIndex)."""
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw["close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": all_tickers[0]})
    return closes.ffill().bfill()


def select_backtest_universe(
    agg_df: pd.DataFrame,
    mode: str,
    universe_selection: str,
    max_symbols: int,
    min_volume: float,
    random_seed: int,
) -> "tuple[pd.DataFrame, str]":
    """
    Select the universe of symbols for backtesting.

    Modes and their lookahead bias levels:
      current_universe_stress_test  → HIGH   (uses value_metric ranking)
      liquid_universe_sanity_test   → MEDIUM (uses volume / random sample)
      walk_forward_price_only_test  → LOW    (uses volume filter only, no scores)
    """
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
        sectors = liquid["sector"].dropna().unique().tolist() if "sector" in liquid.columns else []
        if not sectors:
            selected = liquid.sample(n=min(max_symbols, len(liquid)), random_state=random_seed)
        else:
            per_sector = max(1, max_symbols // len(sectors))
            parts = []
            for s in sectors:
                pool = liquid[liquid["sector"] == s]
                n    = min(per_sector, len(pool))
                parts.append(pool.sample(n=n, random_state=random_seed))
            selected = pd.concat(parts).head(max_symbols)
        bias = "MEDIUM"
    else:
        n        = min(max_symbols, len(liquid))
        selected = liquid.sample(n=n, random_state=random_seed)
        bias     = "MEDIUM"

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
    mode               = mode               or BACKTEST_PARAMS["default_mode"]
    universe_selection = universe_selection or BACKTEST_PARAMS["universe_selection"]
    min_volume         = min_volume         if min_volume is not None else BACKTEST_PARAMS["min_volume"]
    random_seed        = random_seed        if random_seed is not None else BACKTEST_PARAMS["random_seed"]
    benchmark_symbol   = benchmark_symbol   or BACKTEST_PARAMS["benchmark_symbol"]

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
    agg_df["volume"]       = agg_df["volume"].fillna(0)
    agg_df["value_metric"] = agg_df["value_metric"].fillna(0)

    agg_df, lookahead_bias = select_backtest_universe(
        agg_df, mode, universe_selection, max_symbols, min_volume, random_seed
    )

    symbols           = agg_df["symbol"].tolist()
    etf_list          = [e for e in ETFS if e not in set(symbols)]
    benchmark_tickers = [benchmark_symbol] if benchmark_symbol not in set(symbols + etf_list) else []

    n_cal_days = int(n_days * 1.6) + 30
    end_date   = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=n_cal_days)

    all_tickers = symbols + etf_list + benchmark_tickers
    yf_map      = {_yf_ticker(t): t for t in all_tickers}
    yf_tickers  = list(yf_map.keys())

    print(
        f"Downloading price history for {len(yf_tickers)} tickers "
        f"({n_days} trading days, mode={mode}, bias={lookahead_bias}) …"
    )

    raw = yf.download(
        yf_tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")

    closes = _extract_closes(raw, yf_tickers)

    rename_back = {yf_t: orig for yf_t, orig in yf_map.items() if yf_t != orig}
    if rename_back:
        closes.rename(columns=rename_back, inplace=True)

    if len(closes) < n_days:
        raise RuntimeError(
            f"Only {len(closes)} trading days available; requested {n_days}. "
            "Reduce --tune window."
        )
    closes = closes.iloc[-n_days:]

    stock_cols = [s for s in symbols  if s in closes.columns and closes[s].notna().any()]
    etf_cols   = [e for e in etf_list if e in closes.columns and closes[e].notna().any()]

    if not stock_cols:
        raise RuntimeError("No usable stock price data after download.")

    bench_prices = np.full(n_days, np.nan)
    if benchmark_symbol in closes.columns and closes[benchmark_symbol].notna().any():
        bench_prices = closes[benchmark_symbol].values.astype(np.float64)
    else:
        logger.warning(f"Benchmark {benchmark_symbol} not available in price data")

    agg_df   = agg_df[agg_df["symbol"].isin(set(stock_cols))].copy()
    sym_order = [s for s in stock_cols if s in agg_df["symbol"].values]
    agg_df   = agg_df.set_index("symbol").loc[sym_order].reset_index()
    stock_cols = agg_df["symbol"].tolist()

    stock_prices    = closes[stock_cols].values.astype(np.float64)
    etf_prices_arr  = (
        closes[etf_cols].values.astype(np.float64)
        if etf_cols
        else np.zeros((n_days, 0), dtype=np.float64)
    )

    pe_comp        = _col_arr(agg_df, "pe_comp")
    pb_comp        = _col_arr(agg_df, "pb_comp")
    quality_scores = _col_arr(agg_df, "quality_score")
    income_scores  = _col_arr(agg_df, "income_score")
    volume_arr     = _col_arr(agg_df, "volume")

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
    ret_arr = _col_arr(agg_df, "return_1m",    default=np.nan)
    ret_arr = np.where(np.isfinite(ret_arr), ret_arr, np.nan)
    has_pos = np.isfinite(pos_arr)

    boundaries  = np.array(MOMENTUM_PARAMS["position_bin_boundaries"])
    bin_indices = np.searchsorted(
        boundaries, np.where(has_pos, pos_arr, 0.5), side="right"
    ).astype(np.int32)

    if mode == "walk_forward_price_only_test":
        pe_comp        = np.zeros(len(agg_df), dtype=np.float64)
        pb_comp        = np.zeros(len(agg_df), dtype=np.float64)
        quality_scores = np.zeros(len(agg_df), dtype=np.float64)
        income_scores  = np.zeros(len(agg_df), dtype=np.float64)
        logger.info("walk_forward_price_only_test: fundamental arrays zeroed — momentum only")

    # Precompute rolling daily price features for dynamic re-scoring in simulation
    n_stocks         = len(stock_cols)
    pos_52w_daily    = np.full((n_days, n_stocks), np.nan)
    ret_1m_daily     = np.full((n_days, n_stocks), np.nan)
    bin_indices_daily = np.zeros((n_days, n_stocks), dtype=np.int32)

    ret_5d_daily       = np.full((n_days, n_stocks), np.nan)
    ret_3m_daily       = np.full((n_days, n_stocks), np.nan)
    ret_6m_daily       = np.full((n_days, n_stocks), np.nan)
    rs_3m_daily        = np.full((n_days, n_stocks), np.nan)
    rs_6m_daily        = np.full((n_days, n_stocks), np.nan)
    vol_3m_daily       = np.full((n_days, n_stocks), np.nan)
    above_50dma_daily  = np.zeros((n_days, n_stocks), dtype=bool)
    above_200dma_daily = np.zeros((n_days, n_stocks), dtype=bool)

    for d in range(n_days):
        curr      = stock_prices[d]
        win_start = max(0, d - 251)
        window    = stock_prices[win_start: d + 1]
        with np.errstate(invalid="ignore"):
            lo = np.nanmin(window, axis=0)
            hi = np.nanmax(window, axis=0)
        rng    = hi - lo
        valid52 = (rng > 0) & np.isfinite(curr)
        raw_pos = np.where(valid52, (curr - lo) / rng, np.nan)
        pos_52w_daily[d] = np.clip(raw_pos, 0.0, 1.0)

        if d >= 21:
            prev   = stock_prices[d - 21]
            valid1m = (prev > 0) & np.isfinite(prev) & np.isfinite(curr)
            ret_1m_daily[d] = np.where(valid1m, curr / prev - 1.0, np.nan)

        valid_pos_d       = np.where(np.isfinite(pos_52w_daily[d]), pos_52w_daily[d], 0.5)
        bin_indices_daily[d] = np.searchsorted(boundaries, valid_pos_d, side="right").astype(np.int32)

        if d >= 5:
            p5     = stock_prices[d - 5]
            valid5 = (p5 > 0) & np.isfinite(p5) & np.isfinite(curr)
            ret_5d_daily[d] = np.where(valid5, curr / p5 - 1.0, np.nan)

        if d >= 63:
            p63    = stock_prices[d - 63]
            valid63 = (p63 > 0) & np.isfinite(p63) & np.isfinite(curr)
            ret_3m_daily[d] = np.where(valid63, curr / p63 - 1.0, np.nan)

            w63   = stock_prices[d - 63: d + 1]
            p_prev, p_next = w63[:-1], w63[1:]
            ok63  = (p_prev > 0) & np.isfinite(p_prev) & np.isfinite(p_next)
            dr63  = np.where(ok63, p_next / p_prev - 1.0, np.nan)
            with np.errstate(invalid="ignore"):
                vol_3m_daily[d] = np.nanstd(dr63, axis=0) * np.sqrt(252)

            sp63 = bench_prices[d - 63]
            sp_d = bench_prices[d]
            if np.isfinite(sp63) and sp63 > 0 and np.isfinite(sp_d):
                spy_r3m = sp_d / sp63 - 1.0
                rs_3m_daily[d] = np.where(np.isfinite(ret_3m_daily[d]),
                                           ret_3m_daily[d] - spy_r3m, np.nan)

        if d >= 126:
            p126   = stock_prices[d - 126]
            valid126 = (p126 > 0) & np.isfinite(p126) & np.isfinite(curr)
            ret_6m_daily[d] = np.where(valid126, curr / p126 - 1.0, np.nan)

            sp126 = bench_prices[d - 126]
            if np.isfinite(sp126) and sp126 > 0 and np.isfinite(bench_prices[d]):
                spy_r6m = bench_prices[d] / sp126 - 1.0
                rs_6m_daily[d] = np.where(np.isfinite(ret_6m_daily[d]),
                                           ret_6m_daily[d] - spy_r6m, np.nan)

        if d >= 50:
            w50  = stock_prices[d - 49: d + 1]
            with np.errstate(invalid="ignore"):
                ma50 = np.nanmean(w50, axis=0)
            above_50dma_daily[d] = np.isfinite(curr) & (curr > 0) & (curr > ma50)

        if d >= 200:
            w200 = stock_prices[d - 199: d + 1]
            with np.errstate(invalid="ignore"):
                ma200 = np.nanmean(w200, axis=0)
            above_200dma_daily[d] = np.isfinite(curr) & (curr > 0) & (curr > ma200)

    has_pos_daily = np.isfinite(pos_52w_daily)

    cur_mbin  = np.array(MOMENTUM_PARAMS["position_bin_scores"])
    cur_value = (
        SCORING_PARAMS["value_pe_weight"] * pe_comp
        + SCORING_PARAMS["value_pb_weight"] * pb_comp
    )
    cur_mom   = _momentum_score_vec(bin_indices, has_pos, pos_arr, ret_arr, cur_mbin)
    sw        = np.array([SCORE_WEIGHTS["value"], SCORE_WEIGHTS["quality"],
                          SCORE_WEIGHTS["income"], SCORE_WEIGHTS["momentum"]])
    sw        = sw / sw.sum()
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
