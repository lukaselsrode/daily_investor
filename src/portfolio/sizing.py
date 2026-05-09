"""
portfolio/sizing.py — PositionSizer: compute per-stock allocation.

Phase 4 migration target: extract allocation logic from main.make_buys().

Current formula:
  alloc = (stock.value_metric / total_value_metric) * available_stock_cash

Then risk-adjusted by RiskManager.can_buy().
"""

from __future__ import annotations

import pandas as pd


class PositionSizer:
    """Proportional value-metric based position sizing."""

    def compute_allocations(
        self,
        candidates: pd.DataFrame,
        available_cash: float,
    ) -> dict[str, float]:
        """
        Return {symbol: raw_allocation} before risk adjustment.
        Proportional to value_metric — higher-scored stocks get more capital.
        """
        if candidates.empty or available_cash <= 0:
            return {}

        total = candidates["value_metric"].sum()
        if total <= 0:
            return {}

        return {
            row["symbol"]: (row["value_metric"] / total) * available_cash
            for _, row in candidates.iterrows()
        }
