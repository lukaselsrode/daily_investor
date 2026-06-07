"""
tuning/robust_scan.py — Multi-cell robustness scan orchestrator.

Runs random_window_backtest() once per (horizon, seed) cell and aggregates
results into horizon/seed heatmaps plus a cross-horizon overfit score.

Public API
----------
run_robust_scan(precomp, params, run_matrix, ...) -> RobustScanResult
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScanCell:
    """Single (horizon, seed) evaluation."""
    horizon_days: int
    seed: int
    summary: object  # RandomWindowSummary from random_walk.py


@dataclass
class RobustScanResult:
    """Aggregated result across all (horizon, seed) cells."""
    run_matrix: list[dict]
    cells: list[ScanCell]
    scope: str

    # Flattened aggregates across every window in every cell
    overall_robust_score: float = 0.0
    median_excess_return: float = 0.0
    median_sharpe: float = 0.0
    median_drawdown: float = 0.0
    pct_cells_beating_benchmark: float = 0.0

    def horizon_heatmap_df(self) -> pd.DataFrame:
        """
        Rows = unique horizon lengths, cols = [excess, sharpe, pct_beating, drawdown, robust_score].
        Values are medians over all seeds for that horizon.
        """
        horizons = sorted({c.horizon_days for c in self.cells})
        rows = []
        for h in horizons:
            h_cells = [c for c in self.cells if c.horizon_days == h]
            s = h_cells[0].summary  # use first cell's type helpers
            use_active = (self.scope == "active_sleeve_compounding"
                          and getattr(s, "active_robust_score", None) is not None)

            def _get(attr, active_attr, h_cells=h_cells, use_active=use_active):
                vals = []
                for c in h_cells:
                    sm = c.summary
                    v = getattr(sm, active_attr if use_active else attr, None)
                    if v is not None:
                        vals.append(float(v))
                return float(np.median(vals)) if vals else float("nan")

            rows.append({
                "horizon (days)": h,
                "median excess":  _get("median_excess_return", "median_active_excess_return"),
                "median Sharpe":  _get("median_sharpe", "median_active_sharpe"),
                "% beating":      _get("pct_beating_benchmark", "pct_active_beating_benchmark"),
                "median DD":      _get("median_drawdown", "worst_decile_active_drawdown"),
                "robust score":   _get("robust_score", "active_robust_score"),
            })
        return pd.DataFrame(rows)

    def seed_stability_df(self) -> pd.DataFrame:
        """
        Rows = unique seeds, cols = [seed, median_excess (per horizon), overall].
        """
        seeds    = sorted({c.seed for c in self.cells})
        horizons = sorted({c.horizon_days for c in self.cells})
        use_active = self.scope == "active_sleeve_compounding"

        def _excess(cell) -> float:
            sm = cell.summary
            attr = "median_active_excess_return" if use_active else "median_excess_return"
            v = getattr(sm, attr, None)
            return float(v) if v is not None else float("nan")

        rows = []
        for s in seeds:
            s_cells = [c for c in self.cells if c.seed == s]
            row: dict = {"seed": s}
            for h in horizons:
                hc = [c for c in s_cells if c.horizon_days == h]
                row[f"{h}d"] = float(np.mean([_excess(c) for c in hc])) if hc else float("nan")
            all_exc = [_excess(c) for c in s_cells if not np.isnan(_excess(c))]
            row["overall"] = float(np.mean(all_exc)) if all_exc else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    def overfit_warning_score(self) -> float:
        """
        0 = beating benchmark on every horizon, 1 = only one horizon.
        Computed as 1 - (n_horizons_with_positive_median_excess / total_unique_horizons).
        """
        horizons = sorted({c.horizon_days for c in self.cells})
        if not horizons:
            return 0.0
        use_active = self.scope == "active_sleeve_compounding"
        n_positive = 0
        for h in horizons:
            h_cells = [c for c in self.cells if c.horizon_days == h]
            attr = "median_active_excess_return" if use_active else "median_excess_return"
            vals = [getattr(c.summary, attr, None) for c in h_cells]
            vals = [v for v in vals if v is not None]
            if vals and float(np.median(vals)) > 0:
                n_positive += 1
        return 1.0 - (n_positive / len(horizons))

    def all_window_results(self):
        """Flat list of WindowResult across all cells (for fan chart aggregation)."""
        out = []
        for cell in self.cells:
            out.extend(getattr(cell.summary, "window_results", []))
        return out

    def aggregate_summary(self):
        """
        Return a RandomWindowSummary-like object (the cell with the highest n_windows
        that is closest to the overall median robust_score) for use with existing chart
        helpers that expect a RandomWindowSummary.
        """
        if not self.cells:
            return None
        # Just return the summary with the most windows for fan chart display
        return max(self.cells, key=lambda c: getattr(c.summary, "n_windows", 0)).summary


def run_robust_scan(
    precomp,
    params,
    run_matrix: list[dict],
    scope: str = "overall_strategy",
    regime_scope: str = "all",
    progress_callback: Callable | None = None,
) -> RobustScanResult:
    """
    Run random_window_backtest() once per (horizon, seed) cell in run_matrix.

    Parameters
    ----------
    precomp      : PrecomputedData (full history; each cell slices its own window)
    params       : 15-element params vector (or None → current config defaults)
    run_matrix   : output of profiles.expand_run_matrix()
    scope        : "overall_strategy" or "active_sleeve_compounding"
    progress_callback : optional (current_cell: int, total_cells: int) → None
    """
    from backtesting.random_walk import random_window_backtest
    from backtesting.simulator import get_default_params

    if params is None:
        params = get_default_params()

    total = len(run_matrix)
    cells: list[ScanCell] = []

    for i, cell_cfg in enumerate(run_matrix):
        if progress_callback is not None:
            progress_callback(i, total)

        horizon  = cell_cfg["horizon_days"]
        seed     = cell_cfg["seed"]
        n_windows = cell_cfg["n_windows"]

        try:
            summary = random_window_backtest(
                precomp,
                params=params,
                n_windows=n_windows,
                window_days=horizon,
                seed=seed,
                scope=scope,
                regime_scope=regime_scope,
            )
            cells.append(ScanCell(horizon_days=horizon, seed=seed, summary=summary))
        except Exception as exc:
            logger.warning("Cell (horizon=%d, seed=%d) failed: %s", horizon, seed, exc)

    if progress_callback is not None:
        progress_callback(total, total)

    return _aggregate(run_matrix, cells, scope)


def _aggregate(run_matrix: list[dict], cells: list[ScanCell], scope: str) -> RobustScanResult:
    """Build the aggregated RobustScanResult from completed cells."""
    if not cells:
        return RobustScanResult(run_matrix=run_matrix, cells=cells, scope=scope)

    use_active = scope == "active_sleeve_compounding"

    def _vals(attr, active_attr):
        out = []
        for c in cells:
            sm = c.summary
            v = getattr(sm, active_attr if use_active else attr, None)
            if v is not None:
                out.append(float(v))
        return out

    excesses  = _vals("median_excess_return", "median_active_excess_return")
    sharpes   = _vals("median_sharpe",         "median_active_sharpe")
    dds       = _vals("median_drawdown",        "worst_decile_active_drawdown")
    scores    = _vals("robust_score",           "active_robust_score")

    result = RobustScanResult(
        run_matrix=run_matrix,
        cells=cells,
        scope=scope,
        overall_robust_score=float(np.median(scores))      if scores  else 0.0,
        median_excess_return=float(np.median(excesses))    if excesses else 0.0,
        median_sharpe=float(np.median(sharpes))            if sharpes else 0.0,
        median_drawdown=float(np.median(dds))              if dds     else 0.0,
        pct_cells_beating_benchmark=float(np.mean(
            [1.0 if e > 0 else 0.0 for e in excesses]
        )) if excesses else 0.0,
    )
    return result
