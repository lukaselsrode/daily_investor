"""
portfolio/risk.py — RiskManager.

All risk checks are centralized here:
  - liquidity gate (min_volume)
  - order size cap (max_order_pct_of_cash)
  - single-position cap (max_single_position_pct)
  - sector cap (max_sector_pct)
  - minimum order amount gate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from util import RISK_LIMITS, safe_float

logger = logging.getLogger(__name__)


@dataclass
class BuyDecision:
    approved: bool
    reason: str
    adjusted_allocation: float


def _position_value(symbol: str, holdings: dict) -> float:
    try:
        return float(holdings.get(symbol, {}).get("equity", 0) or 0)
    except Exception:
        return 0.0


class RiskManager:
    """
    Validates buy orders against configured risk limits.
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    def can_buy(
        self,
        symbol: str,
        allocation: float,
        holdings: dict,
        agg_df: Optional[pd.DataFrame],
        portfolio_value: float,
        available_cash: float,
        sector_exposure: Optional[dict] = None,
    ) -> BuyDecision:
        max_single    = RISK_LIMITS["max_single_position_pct"]
        max_sector    = RISK_LIMITS["max_sector_pct"]
        max_order_pct = RISK_LIMITS["max_order_pct_of_cash"]
        min_order     = RISK_LIMITS["min_order_amount"]
        min_volume    = RISK_LIMITS["min_liquidity_volume"]

        # Liquidity gate
        if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
            row = agg_df[agg_df["symbol"] == symbol]
            if not row.empty:
                vol = safe_float(row.iloc[0].get("volume"), 0.0)
                if vol < min_volume:
                    return BuyDecision(
                        approved=False,
                        reason=f"volume {vol:,.0f} < min {min_volume:,.0f}",
                        adjusted_allocation=0.0,
                    )

        # Order size cap (fraction of available cash)
        max_order = available_cash * max_order_pct
        if allocation > max_order:
            logger.info(
                f"{symbol}: order ${allocation:.2f} capped to {max_order_pct:.0%} of cash "
                f"(${available_cash:.2f}) = ${max_order:.2f}"
            )
            allocation = max_order

        # Single-position cap
        if portfolio_value > 0:
            current_pos = _position_value(symbol, holdings)
            max_allowed = portfolio_value * max_single
            room = max_allowed - current_pos
            if room <= 0:
                return BuyDecision(
                    approved=False,
                    reason=f"position cap reached (${current_pos:.2f} / ${max_allowed:.2f})",
                    adjusted_allocation=0.0,
                )
            if allocation > room:
                logger.info(
                    f"{symbol}: buy reduced ${allocation:.2f} → ${room:.2f} "
                    f"(single-position cap {max_single:.0%} of ${portfolio_value:.2f})"
                )
                allocation = room

        # Sector cap
        if portfolio_value > 0 and agg_df is not None and not agg_df.empty:
            row = agg_df[agg_df["symbol"] == symbol]
            sector = str(row.iloc[0].get("sector") or "") if not row.empty else ""
            if sector:
                sector_exp     = sector_exposure if sector_exposure is not None else self.get_sector_exposure(holdings, agg_df)
                current_sector = sector_exp.get(sector, 0.0)
                max_sector_val = portfolio_value * max_sector
                room = max_sector_val - current_sector
                if room <= 0:
                    return BuyDecision(
                        approved=False,
                        reason=(
                            f"sector cap reached for {sector!r} "
                            f"(${current_sector:.2f} / ${max_sector_val:.2f})"
                        ),
                        adjusted_allocation=0.0,
                    )
                if allocation > room:
                    logger.info(
                        f"{symbol}: buy reduced ${allocation:.2f} → ${room:.2f} "
                        f"(sector {sector!r} cap {max_sector:.0%})"
                    )
                    allocation = room

        # Final minimum check
        if allocation < min_order:
            return BuyDecision(
                approved=False,
                reason=f"allocation ${allocation:.2f} below min_order_amount ${min_order:.2f}",
                adjusted_allocation=0.0,
            )

        return BuyDecision(approved=True, reason="ok", adjusted_allocation=allocation)

    def get_sector_exposure(
        self,
        holdings: dict,
        agg_df: Optional[pd.DataFrame],
    ) -> dict[str, float]:
        totals: dict[str, float] = {}
        for symbol, data in holdings.items():
            equity = safe_float(data.get("equity"), 0.0)
            sector = "Unknown"
            if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
                row = agg_df[agg_df["symbol"] == symbol]
                if not row.empty:
                    sector = str(row.iloc[0].get("sector") or "Unknown")
            totals[sector] = totals.get(sector, 0.0) + equity
        return totals
