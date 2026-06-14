"""
backtesting/pit_precompute.py — Point-in-time factor panels for the survivorship-free path.

Builds per-day 2D arrays (n_days × n_symbols) of pe_comp / pb_comp / quality_score /
income_score scored CAUSALLY: at each rebalance date the value/quality/income factors are
computed from fundamentals filed STRICTLY BEFORE that date (data.pit_fundamentals) and the
as-of price, ranked cross-sectionally over the names tradeable that day using the production
peer scorers (strategy.scoring) — never the current ratios.yaml sector baselines. Results
are forward-filled to every day until the next rebalance.

This replaces the static current-snapshot factor arrays (the dominant active-sleeve
look-ahead). Missing fundamentals neutral-score (0.0 / scorer-neutral) exactly as the live
path treats missing data. Cache-only — no network.

Cost model: O(n_symbols) statement-cache reads (once each), then vectorized PE/PB/dividend
panels across rebalance dates, then one cross-sectional scoring per rebalance date.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_pit_factor_panels(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    sectors: list[str],
    industries: list[str] | tuple[str, ...],
    position_52w_daily: np.ndarray | None,
    volume: np.ndarray | None,
    rebalance_freq: int,
    scoring_cfg: dict,
) -> dict[str, np.ndarray]:
    """Return {pe_comp_daily, pb_comp_daily, quality_scores_daily, income_scores_daily},
    each float64 (n_days, n_symbols). Raises RuntimeError if NO symbol has usable PIT data
    (so the loader can hard-raise rather than silently degrade)."""
    from data.pit_fundamentals import causal_ttm_series, dividend_records
    from strategy.scoring.income import apply_income
    from strategy.scoring.peer import compute_peer_relative
    from strategy.scoring.quality import apply_quality

    n_days, n_sym = prices.shape
    # Coerce the calendar to datetime64 (the loader's index may be object/string dtype) so
    # all searchsorted comparisons against filing/ex dates are datetime-vs-datetime.
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    rebal = list(range(0, n_days, max(1, rebalance_freq)))
    rebal_ts = dates[rebal].to_numpy()  # datetime64 at each rebalance date

    # ── Per-symbol causal step series (one cache read per symbol) ──────────────
    ttm_eps_panel = np.full((len(rebal), n_sym), np.nan)   # TTM EPS as-of each rebalance
    shares_panel  = np.full((len(rebal), n_sym), np.nan)
    book_panel    = np.full((len(rebal), n_sym), np.nan)
    div_panel     = np.zeros((len(rebal), n_sym))          # TTM cash dividends as-of each rebalance
    n_with_fund = 0
    for j, sym in enumerate(symbols):
        ts = causal_ttm_series(sym)
        if ts is not None and len(ts):
            n_with_fund += 1
            fd = pd.to_datetime(ts["_fd"]).to_numpy()  # ensure datetime64 for searchsorted
            # index of the LATEST filing strictly before each rebalance date (side="left")
            pos = np.searchsorted(fd, rebal_ts, side="left") - 1
            valid = pos >= 0
            eps_arr = ts["ttm_eps"].to_numpy()
            sh_arr  = ts["shares"].to_numpy()
            bk_arr  = ts["book"].to_numpy()
            ttm_eps_panel[valid, j] = eps_arr[pos[valid]]
            shares_panel[valid, j]  = sh_arr[pos[valid]]
            book_panel[valid, j]    = bk_arr[pos[valid]]
        dr = dividend_records(sym)
        if dr is not None:
            ddates, damts = dr
            cum = np.concatenate([[0.0], np.cumsum(damts)])
            hi = np.searchsorted(ddates, rebal_ts, side="left")           # ex-date strictly < asof
            lo = np.searchsorted(ddates, rebal_ts - np.timedelta64(365, "D"), side="left")
            div_panel[:, j] = cum[hi] - cum[lo]

    if n_with_fund == 0:
        raise RuntimeError(
            "PIT precompute: no symbol had >=4 quarters of cached statements — cannot build "
            "point-in-time factor panels (check data/fmp_cache_adj/statements coverage)."
        )

    # ── PE / PB / dividend-yield at each rebalance date (daily price × step fundamentals) ──
    px_rebal = prices[rebal, :]                       # (n_rebal, n_sym)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe_rebal = np.where(ttm_eps_panel > 0, px_rebal / ttm_eps_panel, np.nan)
        mcap = px_rebal * shares_panel
        pb_rebal = np.where((shares_panel > 0) & (book_panel > 0), mcap / book_panel, np.nan)
        dy_rebal = np.where(px_rebal > 0, div_panel / px_rebal, 0.0)

    sectors = list(sectors)
    industries = list(industries)
    vol_col = (np.asarray(volume, dtype=np.float64) if volume is not None
               else np.zeros(n_sym))

    # ── Cross-sectional scoring per rebalance date (production peer scorers) ───
    pe_comp_r = np.zeros((len(rebal), n_sym))
    pb_comp_r = np.zeros((len(rebal), n_sym))
    qual_r    = np.zeros((len(rebal), n_sym))
    inc_r     = np.zeros((len(rebal), n_sym))
    for ri, d in enumerate(rebal):
        px = prices[d, :]
        tradeable = np.isfinite(px) & (px > 0)
        if not tradeable.any():
            continue
        idx = np.where(tradeable)[0]
        pos52 = (position_52w_daily[d, idx] if position_52w_daily is not None
                 else np.full(idx.size, np.nan))
        fr = pd.DataFrame({
            "symbol":         [symbols[k] for k in idx],
            "sector":         [sectors[k] for k in idx],
            "industry":       [industries[k] for k in idx],
            "pe_ratio":       pe_rebal[ri, idx],
            "pb_ratio":       pb_rebal[ri, idx],
            "dividend_yield": dy_rebal[ri, idx],
            "volume":         vol_col[idx],
            "position_52w":   pos52,
        })
        # Value sub-scores: peer-relative blend of PE and PB (low = better), NO ratios.yaml.
        pe_in = fr["pe_ratio"].where(fr["pe_ratio"] > 0)
        pb_in = fr["pb_ratio"].where(fr["pb_ratio"] > 0)
        pe_blended, *_ = compute_peer_relative(pe_in, fr, scoring_cfg, higher_is_better=False)
        pb_blended, *_ = compute_peer_relative(pb_in, fr, scoring_cfg, higher_is_better=False)
        apply_quality(fr, scoring_cfg)
        apply_income(fr, scoring_cfg)
        pe_comp_r[ri, idx] = np.nan_to_num(pe_blended.to_numpy(), nan=0.0)
        pb_comp_r[ri, idx] = np.nan_to_num(pb_blended.to_numpy(), nan=0.0)
        qual_r[ri, idx]    = np.nan_to_num(fr["quality_score"].to_numpy(), nan=0.0)
        inc_r[ri, idx]     = np.nan_to_num(fr["income_score"].to_numpy(), nan=0.0)

    # ── Forward-fill rebalance scores to every day ────────────────────────────
    def _ffill_daily(rebal_vals: np.ndarray) -> np.ndarray:
        out = np.zeros((n_days, n_sym))
        for ri, d in enumerate(rebal):
            end = rebal[ri + 1] if ri + 1 < len(rebal) else n_days
            out[d:end, :] = rebal_vals[ri, :]
        return out

    logger.info(
        "PIT precompute: %d/%d symbols with >=4q statements; %d rebalance dates scored",
        n_with_fund, n_sym, len(rebal),
    )
    return {
        "pe_comp_daily":        _ffill_daily(pe_comp_r),
        "pb_comp_daily":        _ffill_daily(pb_comp_r),
        "quality_scores_daily": _ffill_daily(qual_r),
        "income_scores_daily":  _ffill_daily(inc_r),
    }
