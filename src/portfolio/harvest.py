"""
portfolio/harvest.py — HarvestManager.

Routes take-profit proceeds into harvest ETFs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution.base import BrokerAdapter

from util import HARVEST_PARAMS, RISK_LIMITS

logger = logging.getLogger(__name__)


class HarvestManager:
    """Routes take-profit proceeds into harvest ETFs."""

    def __init__(self, config=None) -> None:
        self._cfg = config

    def route_proceeds(
        self, amount: float, broker: BrokerAdapter, regime: str = "bullish",
    ) -> None:
        """Reinvest take-profit proceeds into harvest ETFs.

        Only `harvest_to_etfs_pct` of proceeds is deployed to ETFs; the remainder
        stays as cash available for the active sleeve (no extra ETF buy triggered).
        The ETF split honors etf_allocation weights (over the harvest-ETF subset);
        with allocation disabled this is the historical equal weight.
        """
        from portfolio.etf_allocation import etf_target_weights, split_budget

        harvest_etfs = HARVEST_PARAMS["harvest_etfs"]
        if not harvest_etfs:
            return
        to_etfs_pct   = float(HARVEST_PARAMS.get("harvest_to_etfs_pct", 1.0))
        etf_amount    = amount * to_etfs_pct
        active_reserve = amount - etf_amount
        min_order = RISK_LIMITS["min_order_amount"]
        weights = etf_target_weights(regime, list(harvest_etfs))
        alloc = {e: d for e, d in split_budget(weights, etf_amount).items() if d >= min_order}
        if not alloc:
            logger.info(
                f"Harvest ETF allocations below min_order ${min_order:.2f} "
                f"— skipping harvest reinvestment"
            )
            return
        logger.info(
            f"=== HARVEST: ${amount:.2f} → ETFs ${etf_amount:.2f} "
            f"({to_etfs_pct:.0%}) + active reserve ${active_reserve:.2f} "
            f"({1 - to_etfs_pct:.0%}) ==="
        )
        for etf, amt in alloc.items():
            try:
                result = broker.buy_fractional(etf, amt)
                logger.info(f"Harvest → {etf}: {result.state if result.success else result.detail}")
            except Exception as e:
                logger.error(f"Harvest reinvestment failed for {etf}: {e}")
