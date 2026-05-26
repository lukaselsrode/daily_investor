"""
reporting/attribution.py — Stability classification and attribution.

RESEARCH / DIAGNOSTIC ONLY — never modifies config.yaml.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.types import BacktestReport, TradeRecord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants (imported by diagnostics.py and plots.py)
# ---------------------------------------------------------------------------

_STABLE            = "STABLE"
_MODERATELY_STABLE = "MODERATELY_STABLE"
_UNSTABLE          = "UNSTABLE"

_STABILITY_PALETTE = {
    _STABLE:            "green",
    _MODERATELY_STABLE: "orange",
    _UNSTABLE:          "red",
}


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _date_str() -> str:
    return datetime.date.today().isoformat()


def _try_matplotlib():
    """Import matplotlib with headless backend. Returns (plt, sns_or_None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            sns = None
        return plt, sns
    except ImportError:
        raise RuntimeError(
            "matplotlib is required for heatmaps. Install: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------

def classify_stability(
    cv: float,
    spread: float,
    cv_threshold: float = 0.30,
    spread_threshold: float = 0.15,
) -> str:
    if cv > cv_threshold or spread > spread_threshold:
        return _UNSTABLE
    if cv > cv_threshold * 0.60 or spread > spread_threshold * 0.60:
        return _MODERATELY_STABLE
    return _STABLE


# ---------------------------------------------------------------------------
# Core stability analysis
# ---------------------------------------------------------------------------

def compute_parameter_stability(
    window_results: list[dict],
    param_names: list[str],
    cv_threshold: float = 0.30,
    spread_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Compute per-parameter stability metrics across all window runs.

    window_results entries must have:
        window (int), params_avg (np.ndarray),
        params_sharpe (np.ndarray), params_calmar (np.ndarray).

    Returns a DataFrame with columns:
        param, mean, stddev, cv, sharpe_calmar_spread,
        convergence_frequency, instability_score, stability.
    """
    rows = []
    n_windows = len(window_results)

    for i, name in enumerate(param_names):
        avg_vals = np.array([
            r["params_avg"][i] for r in window_results
            if r.get("params_avg") is not None
        ], dtype=float)

        if len(avg_vals) == 0:
            continue

        mean   = float(avg_vals.mean())
        std    = float(avg_vals.std())
        cv     = float(std / abs(mean)) if abs(mean) > 1e-9 else 0.0

        sc_spreads = []
        for r in window_results:
            ps = r.get("params_sharpe")
            pc = r.get("params_calmar")
            if ps is not None and pc is not None:
                sc_spreads.append(abs(float(ps[i]) - float(pc[i])))
        sc_spread = float(np.mean(sc_spreads)) if sc_spreads else 0.0

        if std > 1e-9 and n_windows > 1:
            within_1std = float(np.mean(np.abs(avg_vals - mean) <= std))
        else:
            within_1std = 1.0

        norm_cv     = min(cv     / max(cv_threshold,     1e-9), 1.0)
        norm_spread = min(sc_spread / max(spread_threshold, 1e-9), 1.0)
        instability = 0.50 * norm_cv + 0.50 * norm_spread

        stability = classify_stability(cv, sc_spread, cv_threshold, spread_threshold)

        rows.append({
            "param":                 name,
            "mean":                  round(mean,          4),
            "stddev":                round(std,           4),
            "cv":                    round(cv,            4),
            "sharpe_calmar_spread":  round(sc_spread,     4),
            "convergence_frequency": round(within_1std,   3),
            "instability_score":     round(instability,   3),
            "stability":             stability,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AttributionReporter
# ---------------------------------------------------------------------------

class AttributionReporter:
    """Parameter stability attribution and (future) factor attribution."""

    def compute_stability(
        self,
        window_results: list[dict],
        param_names: list[str],
        cv_threshold: float = 0.30,
        spread_threshold: float = 0.15,
    ) -> pd.DataFrame:
        return compute_parameter_stability(
            window_results,
            param_names,
            cv_threshold=cv_threshold,
            spread_threshold=spread_threshold,
        )

    def classify(
        self,
        cv: float,
        spread: float,
        cv_threshold: float = 0.30,
        spread_threshold: float = 0.15,
    ) -> str:
        return classify_stability(cv, spread, cv_threshold, spread_threshold)

    def factor_attribution(self, trades: list[TradeRecord]) -> dict:
        """P&L breakdown by exit type for sell trades (proxy for factor attribution)."""
        from collections import defaultdict
        breakdown: dict = defaultdict(lambda: {"count": 0, "total_pnl": 0.0})
        for t in trades:
            if t.side == "sell":
                key = t.exit_type or "unknown"
                breakdown[key]["count"] += 1
                breakdown[key]["total_pnl"] += t.pnl
        for v in breakdown.values():
            v["avg_pnl"] = v["total_pnl"] / max(v["count"], 1)
        return dict(breakdown)

    def sleeve_attribution(self, report: BacktestReport) -> dict:
        """Split return attribution between ETF sleeve and active stock sleeve."""
        sim = getattr(report, "train", None) or getattr(report, "train_result", None)
        if sim is None:
            return {}
        return {
            "etf": {
                "return": getattr(sim, "etf_return", 0.0),
                "avg_allocation": getattr(sim, "etf_allocation_avg", 0.0),
            },
            "stock": {
                "return": getattr(sim, "stock_return", 0.0),
            },
        }

    def exit_type_breakdown(self, trades: list[TradeRecord]) -> dict:
        """Count and P&L summary grouped by exit type for sell trades."""
        from collections import defaultdict
        breakdown: dict = defaultdict(lambda: {"count": 0, "total_pnl": 0.0, "wins": 0, "losses": 0})
        for t in trades:
            if t.side == "sell":
                key = t.exit_type or "unknown"
                breakdown[key]["count"] += 1
                breakdown[key]["total_pnl"] += t.pnl
                if t.pnl >= 0:
                    breakdown[key]["wins"] += 1
                else:
                    breakdown[key]["losses"] += 1
        for v in breakdown.values():
            c = v["count"]
            v["avg_pnl"] = v["total_pnl"] / max(c, 1)
            v["win_rate"] = v["wins"] / c if c > 0 else 0.0
        return dict(breakdown)
