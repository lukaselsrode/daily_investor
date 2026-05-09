"""
portfolio/harvest.py — HarvestManager.

Phase 4 migration target: extract allocate_harvest_proceeds_to_etfs() from main.py.
"""

from __future__ import annotations


class HarvestManager:
    """Routes take-profit proceeds into harvest ETFs."""

    def __init__(self, config=None) -> None:
        self._cfg = config

    def route_proceeds(self, amount: float) -> None:
        from main import allocate_harvest_proceeds_to_etfs  # TODO (Phase 4): migrate inline
        allocate_harvest_proceeds_to_etfs(amount)
