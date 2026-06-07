"""Utilities for regime-scoped backtest and tuning data selection.

A regime scope does not change the strategy's live regime logic. It controls which
historical days/windows are eligible for an experiment so configs can be tuned and
validated separately on bullish, neutral, defensive/bearish, or all data.
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

import numpy as np

from .simulator import _detect_regime
from .types import PrecomputedData

logger = logging.getLogger(__name__)

RegimeScope = Literal["all", "bullish", "neutral", "defensive", "bearish"]
_EFFECTIVE_SCOPES = {"all", "bullish", "neutral", "defensive"}
_ALIASES = {"bearish": "defensive", "defensive": "defensive", "bullish": "bullish", "neutral": "neutral", "all": "all"}

# A regime block smaller than this is too small to tune/backtest on without severe
# overfitting — rare regimes (esp. defensive) often span only a few dozen days in a
# multi-year window. Slicing still proceeds, but callers are loudly warned.
MIN_REGIME_DAYS_FOR_TUNING = 90


class RegimeScopeMeta(TypedDict):
    requested: str
    effective: str
    total_days: int
    selected_days: int
    start_day: int | None
    end_day: int | None


def normalize_regime_scope(regime_scope: str | None) -> str:
    """Normalize user-facing regime scope; bearish is an alias for defensive."""
    raw = (regime_scope or "all").strip().lower()
    if raw not in _ALIASES:
        valid = ", ".join(sorted(_ALIASES))
        raise ValueError(f"Unknown regime_scope={regime_scope!r}; expected one of: {valid}")
    return _ALIASES[raw]


def regime_labels(precomp: PrecomputedData) -> np.ndarray:
    """Return the simulator's point-in-time regime label for every day."""
    stored = getattr(precomp, "regime_labels_daily", None)
    if stored is not None:
        return np.asarray(stored, dtype=object)
    return np.array([_detect_regime(precomp, d) for d in range(precomp.prices.shape[0])], dtype=object)


def slice_precomp(precomp: PrecomputedData, s: slice) -> PrecomputedData:
    """Slice all time-indexed arrays in PrecomputedData, preserving static fields."""
    def _opt(arr: np.ndarray | None) -> np.ndarray | None:
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
        dollar_volume_daily=_opt(precomp.dollar_volume_daily),
        regime_labels_daily=_opt(precomp.regime_labels_daily),
        vix_prices=_opt(precomp.vix_prices),
    )


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    best_start = best_end = None
    cur_start = None
    for i, ok in enumerate(mask.tolist() + [False]):
        if ok and cur_start is None:
            cur_start = i
        elif not ok and cur_start is not None:
            if best_start is None or i - cur_start > best_end - best_start:  # type: ignore[operator]
                best_start, best_end = cur_start, i
            cur_start = None
    if best_start is None or best_end is None:
        return None
    return best_start, best_end


def apply_regime_scope(precomp: PrecomputedData, regime_scope: str | None = "all") -> tuple[PrecomputedData, RegimeScopeMeta]:
    """Return precomp restricted to the longest contiguous block for the selected regime.

    Non-contiguous concatenation would fabricate price paths and invalid rebalance
    histories, so single-run/tuner uses the longest contiguous block. Random-window
    backtests use eligible_window_starts() instead and can sample from multiple blocks.
    """
    effective = normalize_regime_scope(regime_scope)
    n_total = int(precomp.prices.shape[0])
    if effective == "all":
        return precomp, {
            "requested": regime_scope or "all",
            "effective": "all",
            "total_days": n_total,
            "selected_days": n_total,
            "start_day": 0 if n_total else None,
            "end_day": n_total if n_total else None,
        }

    labels = regime_labels(precomp)
    precomp = precomp._replace(regime_labels_daily=labels)
    mask = labels == effective
    run = _longest_true_run(mask)
    if run is None:
        raise ValueError(f"No {effective} days available for regime_scope={regime_scope!r}")
    start, end = run
    selected = int(end - start)
    if selected < MIN_REGIME_DAYS_FOR_TUNING:
        logger.warning(
            "Regime scope %r yields only %d contiguous days (%d total %s days in a "
            "%d-day window) — too few to tune/backtest reliably; results are "
            "noise-dominated and will not generalize. Treat as indicative only.",
            effective, selected, int(mask.sum()), effective, n_total,
        )
    return slice_precomp(precomp, slice(start, end)), {
        "requested": regime_scope or "all",
        "effective": effective,
        "total_days": n_total,
        "selected_days": int(end - start),
        "start_day": int(start),
        "end_day": int(end),
    }


def eligible_window_starts(
    precomp: PrecomputedData,
    window_days: int,
    regime_scope: str | None = "all",
) -> tuple[np.ndarray, RegimeScopeMeta]:
    """Return start indices whose entire window belongs to the selected regime."""
    effective = normalize_regime_scope(regime_scope)
    n_total = int(precomp.prices.shape[0])
    max_start = n_total - window_days
    if max_start < 0:
        return np.array([], dtype=int), {
            "requested": regime_scope or "all",
            "effective": effective,
            "total_days": n_total,
            "selected_days": 0,
            "start_day": None,
            "end_day": None,
        }
    if effective == "all":
        starts = np.arange(max_start + 1, dtype=int)
        return starts, {
            "requested": regime_scope or "all",
            "effective": "all",
            "total_days": n_total,
            "selected_days": n_total,
            "start_day": 0 if n_total else None,
            "end_day": n_total if n_total else None,
        }

    labels = regime_labels(precomp)
    wanted = labels == effective
    starts = [s for s in range(max_start + 1) if bool(np.all(wanted[s:s + window_days]))]
    if not starts:
        raise ValueError(
            f"No {window_days}d windows fully inside regime_scope={regime_scope!r} ({effective})"
        )
    arr = np.array(starts, dtype=int)
    return arr, {
        "requested": regime_scope or "all",
        "effective": effective,
        "total_days": n_total,
        "selected_days": int(wanted.sum()),
        "start_day": int(arr[0]),
        "end_day": int(arr[-1] + window_days),
    }
