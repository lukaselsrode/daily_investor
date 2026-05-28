"""
tests/test_execution.py — BrokerAdapter tests (Phase 5).

PaperBroker: fully testable in-process.
RobinhoodBroker: tested via a lightweight mock of robin_stocks so no credentials needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import types

import pytest

from execution.base import OrderResult
from execution.paper import PaperBroker

# ---------------------------------------------------------------------------
# PaperBroker tests
# ---------------------------------------------------------------------------

class TestPaperBrokerBuyFractional:

    def _broker(self, price: float = 100.0) -> PaperBroker:
        return PaperBroker(starting_cash=1_000.0, price_lookup=lambda sym: price)

    def test_buy_reduces_cash(self):
        b = self._broker()
        b.buy_fractional("AAPL", 200.0)
        assert b.get_cash() == pytest.approx(800.0)

    def test_buy_creates_position(self):
        b = self._broker(price=50.0)
        b.buy_fractional("AAPL", 100.0)
        h = b.get_holdings()
        assert "AAPL" in h
        assert h["AAPL"]["quantity"] == pytest.approx(2.0)

    def test_buy_updates_equity(self):
        b = self._broker(price=100.0)
        b.buy_fractional("AAPL", 300.0)
        h = b.get_holdings()
        assert h["AAPL"]["equity"] == pytest.approx(300.0)

    def test_buy_averages_cost_basis(self):
        b = self._broker(price=100.0)
        b.buy_fractional("AAPL", 100.0)  # 1 share @ 100
        b2 = PaperBroker(starting_cash=1_000.0, price_lookup=lambda sym: 200.0)
        b2._holdings = b.get_holdings()
        b2._cash = b.get_cash()
        b2.buy_fractional("AAPL", 200.0)  # 1 share @ 200 → avg = 150
        h = b2.get_holdings()
        assert h["AAPL"]["average_buy_price"] == pytest.approx(150.0)

    def test_buy_rejects_insufficient_cash(self):
        b = self._broker()
        res = b.buy_fractional("AAPL", 2_000.0)
        assert not res.success
        assert b.get_cash() == pytest.approx(1_000.0)

    def test_buy_rejects_zero_price(self):
        b = PaperBroker(starting_cash=1_000.0, price_lookup=lambda sym: 0.0)
        res = b.buy_fractional("AAPL", 100.0)
        assert not res.success

    def test_buy_returns_order_result(self):
        b = self._broker()
        res = b.buy_fractional("AAPL", 100.0)
        assert isinstance(res, OrderResult)
        assert res.success
        assert res.side == "buy"
        assert res.symbol == "AAPL"


class TestPaperBrokerBuyWhole:

    def test_buy_whole_calculates_cost_from_price(self):
        b = PaperBroker(starting_cash=1_000.0, price_lookup=lambda sym: 50.0)
        res = b.buy_whole("AAPL", 3)
        assert res.success
        assert b.get_cash() == pytest.approx(850.0)


class TestPaperBrokerSell:

    def _broker_with_position(self, qty: float = 5.0, price: float = 100.0) -> PaperBroker:
        b = PaperBroker(starting_cash=0.0, price_lookup=lambda sym: price)
        b._holdings["AAPL"] = {
            "quantity": qty,
            "average_buy_price": price,
            "equity": qty * price,
        }
        return b

    def test_sell_increases_cash(self):
        b = self._broker_with_position(qty=5.0, price=100.0)
        b.sell("AAPL", 5.0)
        assert b.get_cash() == pytest.approx(500.0)

    def test_sell_removes_position(self):
        b = self._broker_with_position(qty=5.0)
        b.sell("AAPL", 5.0)
        assert "AAPL" not in b.get_holdings()

    def test_sell_partial_leaves_remainder(self):
        b = self._broker_with_position(qty=5.0, price=100.0)
        b.sell("AAPL", 3.0)
        h = b.get_holdings()
        assert "AAPL" in h
        assert h["AAPL"]["quantity"] == pytest.approx(2.0)

    def test_sell_no_position_returns_failure(self):
        b = PaperBroker(starting_cash=100.0, price_lookup=lambda sym: 100.0)
        res = b.sell("AAPL", 1.0)
        assert not res.success

    def test_sell_returns_order_result(self):
        b = self._broker_with_position()
        res = b.sell("AAPL", 5.0)
        assert isinstance(res, OrderResult)
        assert res.side == "sell"
        assert res.success


class TestPaperBrokerPortfolio:

    def test_portfolio_value_cash_only(self):
        b = PaperBroker(starting_cash=5_000.0, price_lookup=lambda sym: 100.0)
        assert b.get_portfolio_value() == pytest.approx(5_000.0)

    def test_portfolio_value_includes_equity(self):
        b = PaperBroker(starting_cash=500.0, price_lookup=lambda sym: 100.0)
        b._holdings["AAPL"] = {"quantity": 5.0, "average_buy_price": 100.0, "equity": 500.0}
        assert b.get_portfolio_value() == pytest.approx(1_000.0)

    def test_open_orders_always_empty(self):
        b = PaperBroker()
        assert b.get_open_orders() == []

    def test_deposit_increases_cash(self):
        b = PaperBroker(starting_cash=100.0)
        b.deposit(400.0)
        assert b.get_cash() == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# RobinhoodBroker mock tests — no credentials, no network
# ---------------------------------------------------------------------------

def _make_rb_mock(
    fractional_response=None,
    market_response=None,
    sell_fractional_response=None,
    sell_market_response=None,
    positions=None,
    open_orders=None,
    profile=None,
):
    """Build a minimal mock robin_stocks module."""
    rb = types.SimpleNamespace()

    # orders namespace
    def order_buy_fractional_by_price(symbol, amount):
        return fractional_response

    def order_buy_market(symbol, quantity):
        return market_response

    def order_sell_fractional_by_quantity(symbol, quantity):
        return sell_fractional_response

    def get_all_open_stock_orders():
        return open_orders or []

    rb.orders = types.SimpleNamespace(
        order_buy_fractional_by_price=order_buy_fractional_by_price,
        order_buy_market=order_buy_market,
        order_sell_fractional_by_quantity=order_sell_fractional_by_quantity,
        get_all_open_stock_orders=get_all_open_stock_orders,
    )

    # sell market at top level
    def order_sell_market(symbol, quantity, timeInForce="gfd"):
        return sell_market_response

    rb.order_sell_market = order_sell_market

    # positions / holdings
    def get_all_positions():
        return positions or []

    def build_holdings():
        if positions is None:
            return {}
        return {p["symbol"]: p for p in positions}

    rb.get_all_positions = get_all_positions
    rb.build_holdings = build_holdings

    # account
    def build_user_profile():
        return profile or {"cash": "1000.00", "equity": "5000.00"}

    rb.account = types.SimpleNamespace(build_user_profile=build_user_profile)

    return rb


def _make_broker(rb_mock) -> "RobinhoodBroker":
    from execution.robinhood import RobinhoodBroker
    b = object.__new__(RobinhoodBroker)
    b._rb = rb_mock
    return b


class TestRobinhoodBrokerBuyFractional:

    def test_success_returns_filled_result(self):
        rb = _make_rb_mock(fractional_response={"id": "abc12345-xx", "state": "confirmed"})
        b = _make_broker(rb)
        res = b.buy_fractional("AAPL", 100.0)
        assert res.success
        assert res.order_id.startswith("abc")
        assert res.state == "confirmed"
        assert res.side == "buy"

    def test_none_response_returns_failure(self):
        rb = _make_rb_mock(fractional_response=None)
        b = _make_broker(rb)
        res = b.buy_fractional("AAPL", 100.0)
        assert not res.success

    def test_missing_id_returns_failure(self):
        rb = _make_rb_mock(fractional_response={"detail": "insufficient funds"})
        b = _make_broker(rb)
        res = b.buy_fractional("AAPL", 100.0)
        assert not res.success
        assert "insufficient funds" in res.detail

    def test_exception_returns_error_result(self):

        def boom(symbol, amount):
            raise ConnectionError("network error")

        rb = _make_rb_mock()
        rb.orders.order_buy_fractional_by_price = boom
        b = _make_broker(rb)
        res = b.buy_fractional("AAPL", 100.0)
        assert not res.success
        assert res.state == "error"


class TestRobinhoodBrokerBuyWhole:

    def test_success(self):
        rb = _make_rb_mock(market_response={"id": "def67890-yy", "state": "confirmed"})
        b = _make_broker(rb)
        res = b.buy_whole("AAPL", 2)
        assert res.success
        assert res.quantity == pytest.approx(2.0)

    def test_none_response_returns_failure(self):
        rb = _make_rb_mock(market_response=None)
        b = _make_broker(rb)
        res = b.buy_whole("AAPL", 1)
        assert not res.success

    def test_detail_on_rejection(self):
        rb = _make_rb_mock(market_response={"detail": "not enough shares available"})
        b = _make_broker(rb)
        res = b.buy_whole("AAPL", 1)
        assert not res.success
        assert "not enough shares available" in res.detail


class TestRobinhoodBrokerSell:

    def _positions(self, symbol: str, qty: float):
        return [{"symbol": symbol, "quantity": str(qty)}]

    def test_fractional_sell_success(self):
        rb = _make_rb_mock(
            positions=self._positions("AAPL", 2.5),
            sell_fractional_response={"id": "sell-abc-xx", "state": "confirmed"},
        )
        b = _make_broker(rb)
        res = b.sell("AAPL", 2.5)
        assert res.success
        assert res.side == "sell"
        assert res.quantity == pytest.approx(2.5)

    def test_whole_sell_success(self):
        rb = _make_rb_mock(
            positions=self._positions("MSFT", 3.0),
            sell_market_response={"id": "sell-def-yy", "state": "confirmed"},
        )
        b = _make_broker(rb)
        res = b.sell("MSFT", 3.0)
        assert res.success

    def test_zero_live_qty_skips(self):
        rb = _make_rb_mock(positions=self._positions("AAPL", 0.0))
        b = _make_broker(rb)
        res = b.sell("AAPL", 5.0)
        assert not res.success
        assert "already closed" in res.detail

    def test_no_position_found_skips(self):
        rb = _make_rb_mock(positions=[])
        b = _make_broker(rb)
        res = b.sell("AAPL", 1.0)
        assert not res.success

    def test_rejected_response_returns_failure(self):
        # qty=0.5 → live_qty=0.5 → is_fractional=True → routes to sell_fractional
        rb = _make_rb_mock(
            positions=self._positions("AAPL", 0.5),
            sell_fractional_response={"detail": "insufficient shares"},
        )
        b = _make_broker(rb)
        res = b.sell("AAPL", 0.5)
        assert not res.success
        assert "insufficient shares" in res.detail


class TestRobinhoodBrokerAccountQueries:

    def test_get_holdings_returns_dict(self):
        positions = [{"symbol": "AAPL", "quantity": "5.0", "equity": "500.0"}]
        rb = _make_rb_mock(positions=positions)
        b = _make_broker(rb)
        h = b.get_holdings()
        assert "AAPL" in h

    def test_get_cash_no_pending(self):
        rb = _make_rb_mock(profile={"cash": "750.00", "equity": "5000.00"}, open_orders=[])
        b = _make_broker(rb)
        assert b.get_cash() == pytest.approx(750.0)

    def test_get_cash_subtracts_pending_market_buy(self):
        pending = [{
            "side": "buy",
            "state": "confirmed",
            "type": "market",
            "extended_hours": False,
            "total_notional": {"amount": "200.00"},
        }]
        rb = _make_rb_mock(profile={"cash": "1000.00", "equity": "5000.00"}, open_orders=pending)
        b = _make_broker(rb)
        assert b.get_cash() == pytest.approx(800.0)

    def test_get_cash_ignores_sell_orders(self):
        pending = [{"side": "sell", "state": "confirmed", "type": "market"}]
        rb = _make_rb_mock(profile={"cash": "1000.00"}, open_orders=pending)
        b = _make_broker(rb)
        assert b.get_cash() == pytest.approx(1000.0)

    def test_get_cash_ignores_extended_hours(self):
        pending = [{
            "side": "buy",
            "state": "confirmed",
            "type": "market",
            "extended_hours": True,
            "total_notional": {"amount": "300.00"},
        }]
        rb = _make_rb_mock(profile={"cash": "1000.00"}, open_orders=pending)
        b = _make_broker(rb)
        # extended_hours market order is excluded from committed
        assert b.get_cash() == pytest.approx(1000.0)

    def test_get_portfolio_value(self):
        rb = _make_rb_mock(profile={"cash": "1000.00", "equity": "8000.00"})
        b = _make_broker(rb)
        assert b.get_portfolio_value() == pytest.approx(8000.0)

    def test_get_open_orders_returns_list(self):
        orders = [{"id": "ord1", "side": "buy"}, {"id": "ord2", "side": "sell"}]
        rb = _make_rb_mock(open_orders=orders)
        b = _make_broker(rb)
        result = b.get_open_orders()
        assert len(result) == 2
