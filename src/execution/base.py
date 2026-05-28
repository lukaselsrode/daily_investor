"""
execution/base.py — BrokerAdapter protocol.

Every broker implementation (Robinhood, Paper, future Alpaca/IBKR) must satisfy
this interface. PortfolioManager and TradeExecutor depend only on this protocol,
never on robin_stocks directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class OrderResult:
    symbol: str
    side: Literal["buy", "sell"]
    amount: float
    quantity: float
    success: bool
    order_id: str | None
    state: str
    detail: str = ""


class BrokerAdapter(ABC):
    """Minimal broker interface needed by PortfolioManager."""

    is_live: bool = True  # False for PaperBroker — skips side-effects like CSV saves

    @abstractmethod
    def buy_fractional(self, symbol: str, amount: float) -> OrderResult:
        """Place a fractional dollar-amount buy order."""

    @abstractmethod
    def buy_whole(self, symbol: str, quantity: int) -> OrderResult:
        """Place a whole-share market buy order."""

    @abstractmethod
    def sell(self, symbol: str, quantity: float) -> OrderResult:
        """Sell an entire position (fractional or whole shares)."""

    @abstractmethod
    def get_holdings(self) -> dict:
        """Return {symbol: holding_data} for all open positions."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return available cash balance."""

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Return total portfolio equity."""

    @abstractmethod
    def get_open_orders(self) -> list[dict]:
        """Return all open (not filled) orders."""

    def enrich_holdings_created_at(self, holdings: dict) -> None:  # noqa: B027
        """Backfill initiation dates from order history (no-op for non-live brokers)."""

    def clear_orders_cache(self) -> None:  # noqa: B027
        """Clear any cached order data (no-op for non-live brokers)."""
