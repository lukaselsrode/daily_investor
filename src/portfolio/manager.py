"""
portfolio/manager.py — PortfolioManager: orchestrates the full rebalance cycle.

Phase 4 migration target: extract run_daily_strat() sell + buy cycles from main.py.

Coordinates:
  - SellDecisionEngine (which positions to exit)
  - RiskManager (which buys are safe)
  - HarvestManager (where to route proceeds)
  - TradeExecutor (how to place orders)

PortfolioManager is UI-agnostic: returns structured results,
never prints or logs directly (uses get_logger).
"""

from __future__ import annotations


class PortfolioManager:
    """
    Orchestrates the sell → buy → sweep rebalance cycle.

    TODO (Phase 4): inject dependencies via constructor:
      __init__(
          self,
          broker: BrokerAdapter,
          risk: RiskManager,
          sell_engine: SellDecisionEngine,
          harvest: HarvestManager,
          sentiment: SentimentProviderBase | None,
          config: ConfigManager,
      )
    """

    def rebalance(self, candidates, regime: str = "bullish"):
        # TODO (Phase 4): migrate from main.run_daily_strat
        raise NotImplementedError(
            "PortfolioManager.rebalance not yet migrated. "
            "Call main.run_daily_strat() directly."
        )
