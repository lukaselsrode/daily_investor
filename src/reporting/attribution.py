"""
reporting/attribution.py — AttributionReporter.

Wraps _reporting_legacy for parameter stability attribution.
Factor/sleeve attribution (from BacktestReport.trade_log) is stubbed pending
trade-log instrumentation in the backtest engine.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import _reporting_legacy as _rl
from core.types import BacktestReport, TradeRecord

if TYPE_CHECKING:
    import pandas as pd


class AttributionReporter:
    """
    Parameter stability attribution and (future) factor attribution.

    Currently wraps _reporting_legacy.compute_parameter_stability and
    classify_stability with a typed interface.  Factor attribution methods
    require BacktestReport.trade_log, which is not yet populated by
    BacktestEngine.
    """

    # ------------------------------------------------------------------
    # Parameter stability attribution
    # ------------------------------------------------------------------

    def compute_stability(
        self,
        window_results: list[dict],
        param_names: list[str],
        cv_threshold: float = 0.30,
        spread_threshold: float = 0.15,
    ) -> "pd.DataFrame":
        """
        Compute per-parameter stability metrics across optimization windows.

        Returns DataFrame with columns:
            param, mean, stddev, cv, sharpe_calmar_spread,
            convergence_frequency, instability_score, stability.
        """
        return _rl.compute_parameter_stability(
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
        """Return STABLE / MODERATELY_STABLE / UNSTABLE for one parameter."""
        return _rl.classify_stability(cv, spread, cv_threshold, spread_threshold)

    # ------------------------------------------------------------------
    # Factor attribution — stubbed; requires trade_log in BacktestReport
    # ------------------------------------------------------------------

    def factor_attribution(self, trades: list[TradeRecord]) -> dict:
        """Break down P&L by factor (value, quality, income, momentum)."""
        raise NotImplementedError

    def sleeve_attribution(self, report: BacktestReport) -> dict:
        """Break down returns between ETF sleeve and active sleeve."""
        raise NotImplementedError

    def exit_type_breakdown(self, trades: list[TradeRecord]) -> dict:
        """P&L and count grouped by exit type."""
        raise NotImplementedError
