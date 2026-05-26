"""
execution/paper.py — PaperBroker: dry-run / paper-trading broker.

No real orders. Maintains an in-memory portfolio state.
Useful for:
  - Integration tests without Robinhood credentials
  - Paper trading mode (future --paper-trade flag)
  - Backtest engine final settlement
"""

from __future__ import annotations

import logging

from .base import BrokerAdapter, OrderResult

logger = logging.getLogger(__name__)


class PaperBroker(BrokerAdapter):
    """
    In-memory paper broker.

    Prices must be supplied externally (e.g. from a price lookup function).
    """

    is_live: bool = False  # prevents CSV saves and other live-only side-effects

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        price_lookup: "callable | None" = None,
    ) -> None:
        self._cash = starting_cash
        self._holdings: dict[str, dict] = {}
        self._orders: list[dict] = []
        self._price_lookup = price_lookup or (lambda sym: 0.0)

    def buy_fractional(self, symbol: str, amount: float) -> OrderResult:
        price = self._price_lookup(symbol)
        if price <= 0:
            return OrderResult(symbol, "buy", amount, 0, False, None, "rejected", "no price")
        if self._cash < amount:
            return OrderResult(symbol, "buy", amount, 0, False, None, "rejected", "insufficient cash")
        qty = amount / price
        self._cash -= amount
        pos = self._holdings.setdefault(symbol, {"quantity": 0.0, "average_buy_price": price, "equity": 0.0})
        old_qty = pos["quantity"]
        pos["quantity"] = old_qty + qty
        pos["average_buy_price"] = (old_qty * pos["average_buy_price"] + qty * price) / pos["quantity"]
        pos["equity"] = pos["quantity"] * price
        logger.info(f"[Paper] BUY {symbol} ${amount:.2f} @ ${price:.2f} ({qty:.4f} shares)")
        return OrderResult(symbol, "buy", amount, qty, True, f"paper-{symbol}", "filled")

    def buy_whole(self, symbol: str, quantity: int) -> OrderResult:
        price = self._price_lookup(symbol)
        return self.buy_fractional(symbol, price * quantity)

    def sell(self, symbol: str, quantity: float) -> OrderResult:
        pos = self._holdings.get(symbol)
        if not pos or pos["quantity"] <= 0:
            return OrderResult(symbol, "sell", 0, 0, False, None, "rejected", "no position")
        price = self._price_lookup(symbol)
        qty = min(quantity, pos["quantity"])
        proceeds = qty * price
        self._cash += proceeds
        pos["quantity"] -= qty
        pos["equity"] = pos["quantity"] * price
        if pos["quantity"] <= 1e-6:
            del self._holdings[symbol]
        logger.info(f"[Paper] SELL {symbol} {qty:.4f} @ ${price:.2f} = ${proceeds:.2f}")
        return OrderResult(symbol, "sell", proceeds, qty, True, f"paper-sell-{symbol}", "filled")

    def get_holdings(self) -> dict:
        return {k: v.copy() for k, v in self._holdings.items()}

    def get_cash(self) -> float:
        return self._cash

    def get_portfolio_value(self) -> float:
        equity = sum(v["equity"] for v in self._holdings.values())
        return self._cash + equity

    def get_open_orders(self) -> list[dict]:
        return []

    def deposit(self, amount: float) -> None:
        self._cash += amount

    def clear_orders_cache(self) -> None:
        pass

    def enrich_holdings_created_at(self, holdings: dict) -> None:
        pass

    def add_funds(self, amount: float) -> None:
        self._cash += amount
