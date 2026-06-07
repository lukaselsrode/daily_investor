"""Regime-scoped sizing/exposure research helpers.

The evidence so far says per-regime score weights are mostly noise. This module
keeps the useful part: compare regime-conditional exposure/sizing variants on
random windows that are fully inside a selected regime.

Read-only research module: it never writes config or places orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from backtesting.random_walk import RandomWindowSummary, random_window_backtest
from backtesting.types import BacktestScope, PrecomputedData


@dataclass(frozen=True)
class SizingVariant:
    """One candidate exposure policy for windows in a selected regime."""

    name: str
    index_pct: float
    max_buys: int | None = None


@dataclass(frozen=True)
class SizingResult:
    """Random-window result for one sizing variant."""

    variant: SizingVariant
    summary: RandomWindowSummary

    @property
    def median_excess(self) -> float:
        return self.summary.median_excess_return

    @property
    def pct_beating(self) -> float:
        return self.summary.pct_beating_benchmark

    @property
    def robust_score(self) -> float:
        return self.summary.robust_score


def build_sizing_params(base_params: np.ndarray, variant: SizingVariant) -> np.ndarray:
    """Return a copied param vector with index_pct and optional max_buys changed.

    ``run_simulation`` reads index_pct from slot 4. It reads max_buys from slot 44
    when the vector is long enough, which ``tuning.constants._current_params`` is.
    A short vector can still test index_pct; max_buys is ignored unless present.
    """
    params = np.asarray(base_params, dtype=float).copy()
    params[4] = float(variant.index_pct)
    if variant.max_buys is not None and len(params) > 44:
        params[44] = float(variant.max_buys)
    return params


def default_neutral_sizing_grid(
    current_index_pct: float,
    current_max_buys: int,
) -> list[SizingVariant]:
    """Candidate neutral-regime exposure grid, with current config first.

    The grid leans active exposure up gradually in neutral, where the sleeve has
    shown edge. Defensive is intentionally not included here; it should stay flat
    unless a separate OOS study says otherwise.
    """
    seen: set[tuple[float, int]] = set()
    raw = [
        (float(current_index_pct), int(current_max_buys), "current"),
        (0.70, max(int(current_max_buys), 4), "neutral_idx70"),
        (0.65, max(int(current_max_buys), 5), "neutral_idx65"),
        (0.60, max(int(current_max_buys), 6), "neutral_idx60"),
        (0.55, max(int(current_max_buys), 6), "neutral_idx55"),
        (0.50, max(int(current_max_buys), 8), "neutral_idx50"),
    ]
    out: list[SizingVariant] = []
    for index_pct, max_buys, name in raw:
        key = (round(index_pct, 6), max_buys)
        if key in seen:
            continue
        seen.add(key)
        out.append(SizingVariant(name=name, index_pct=index_pct, max_buys=max_buys))
    return out


def run_regime_sizing_grid(
    precomp: PrecomputedData,
    base_params: np.ndarray,
    variants: Iterable[SizingVariant],
    regime_scope: str = "neutral",
    n_windows: int = 40,
    window_days: int = 45,
    seed: int = 42,
    scope: BacktestScope = "overall_strategy",
) -> list[SizingResult]:
    """Evaluate sizing variants using random windows fully inside regime_scope."""
    results: list[SizingResult] = []
    for i, variant in enumerate(variants):
        params = build_sizing_params(base_params, variant)
        summary = random_window_backtest(
            precomp,
            params,
            n_windows=n_windows,
            window_days=window_days,
            seed=seed + i,
            scope=scope,
            regime_scope=regime_scope,
        )
        results.append(SizingResult(variant=variant, summary=summary))
    return results


def result_rows(results: Iterable[SizingResult]) -> list[dict]:
    """Serialize results to plain rows for CLI printing / CSV writing."""
    rows: list[dict] = []
    for result in results:
        s = result.summary
        rows.append({
            "variant": result.variant.name,
            "index_pct": result.variant.index_pct,
            "active_pct": 1.0 - result.variant.index_pct,
            "max_buys": result.variant.max_buys,
            "n_windows": s.n_windows,
            "window_days": s.window_days,
            "median_excess": s.median_excess_return,
            "pct_beating": s.pct_beating_benchmark,
            "median_sharpe": s.median_sharpe,
            "median_drawdown": s.median_drawdown,
            "robust_score": s.robust_score,
            "median_strategy_return": s.median_strategy_return,
            "median_benchmark_return": s.median_benchmark_return,
        })
    return rows
