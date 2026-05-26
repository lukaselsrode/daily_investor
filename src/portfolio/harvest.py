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

    def route_proceeds(self, amount: float, broker: "BrokerAdapter") -> None:
        """Reinvest take-profit proceeds into harvest ETFs."""
        harvest_etfs = HARVEST_PARAMS["harvest_etfs"]
        if not harvest_etfs:
            return
        per_etf = amount / len(harvest_etfs)
        min_order = RISK_LIMITS["min_order_amount"]
        if per_etf < min_order:
            logger.info(
                f"Harvest per-ETF ${per_etf:.2f} < min_order ${min_order:.2f} "
                f"— skipping harvest reinvestment"
            )
            return
        logger.info(f"=== HARVEST: ${amount:.2f} → {harvest_etfs} (${per_etf:.2f} each) ===")
        for etf in harvest_etfs:
            try:
                result = broker.buy_fractional(etf, per_etf)
                logger.info(f"Harvest → {etf}: {result.state if result.success else result.detail}")
            except Exception as e:
                logger.error(f"Harvest reinvestment failed for {etf}: {e}")
