"""
tests/test_execution_fixes.py — Regressions for live-execution bugs.

1. RobinhoodBroker.sell() must find positions in robin_stocks raw position dicts,
   which carry an instrument URL but NO "symbol" key (matching on p["symbol"]
   made every live sell a silent "skipped — position already closed" no-op).
2. A trim request must sell only the requested quantity, never the full live
   position (quantity = live_qty turned every trim into a full liquidation).
3. REGIME_PARAMS["neutral"] must reflect cfg/config.yaml overrides (the neutral
   block was hardcoded to None, silently ignoring the configured overrides).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import types

import pytest
import yaml

from core.paths import CONFIG_FILE

AAPL_URL = "https://robinhood.com/instruments/450dfc6d-5510-4d40-abfb-f633b7d9be3e/"


def _make_rb_mock(live_qty: float):
    """Mock robin_stocks whose positions are REALISTIC raw dicts: instrument
    URL, account number, quantities — and no "symbol" key anywhere."""
    rb = types.SimpleNamespace()
    rb.calls = {"sells": [], "get_all_positions": 0}

    def get_instruments_by_symbols(symbols):
        return [{"url": AAPL_URL, "symbol": "AAPL", "tradeable": True}]

    def get_open_stock_positions():
        return [{
            "url": "https://robinhood.com/positions/ACC123/450dfc6d/",
            "instrument": AAPL_URL,
            "account_number": "ACC123",
            "average_buy_price": "150.0000",
            "quantity": str(live_qty),
        }]

    def get_all_positions():
        rb.calls["get_all_positions"] += 1
        return get_open_stock_positions()

    def order_sell_fractional_by_quantity(symbol, quantity):
        rb.calls["sells"].append((symbol, float(quantity)))
        return {"id": "sell-order-id-123", "state": "confirmed"}

    def order_sell_market(symbol, quantity, timeInForce="gfd"):
        rb.calls["sells"].append((symbol, float(quantity)))
        return {"id": "sell-order-id-456", "state": "confirmed"}

    rb.get_instruments_by_symbols = get_instruments_by_symbols
    rb.get_open_stock_positions = get_open_stock_positions
    rb.get_all_positions = get_all_positions
    rb.orders = types.SimpleNamespace(
        order_sell_fractional_by_quantity=order_sell_fractional_by_quantity,
    )
    rb.order_sell_market = order_sell_market
    return rb


def _make_broker(rb_mock):
    from execution.robinhood import RobinhoodBroker
    b = object.__new__(RobinhoodBroker)
    b._rb = rb_mock
    b._orders_cache = None
    return b


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("execution.robinhood.time.sleep", lambda _s: None)


class TestSellMatchesRawPositionDicts:
    """Fix 1: positions are matched by instrument URL, not a nonexistent symbol key."""

    def test_sell_finds_position_and_places_order(self):
        rb = _make_rb_mock(live_qty=10.0)
        res = _make_broker(rb).sell("AAPL", 10.0)
        assert res.success, f"sell must not be skipped: state={res.state} detail={res.detail}"
        assert res.state != "skipped"
        assert rb.calls["sells"] == [("AAPL", 10.0)]

    def test_prefers_open_positions_over_all_positions(self):
        rb = _make_rb_mock(live_qty=10.0)
        _make_broker(rb).sell("AAPL", 10.0)
        assert rb.calls["get_all_positions"] == 0

    def test_fractional_position_sells(self):
        rb = _make_rb_mock(live_qty=2.5)
        res = _make_broker(rb).sell("AAPL", 2.5)
        assert res.success
        assert rb.calls["sells"] == [("AAPL", 2.5)]


class TestSellQuantityClamp:
    """Fix 2: requested quantity is honored (trims) and clamped to the live qty."""

    def test_trim_sells_only_requested_quantity(self):
        rb = _make_rb_mock(live_qty=10.0)
        res = _make_broker(rb).sell("AAPL", 5.0)
        assert res.success
        assert rb.calls["sells"] == [("AAPL", 5.0)], (
            "a half-position trim must not liquidate the full live position"
        )
        assert res.quantity == pytest.approx(5.0)

    def test_request_above_live_qty_clamped(self):
        rb = _make_rb_mock(live_qty=10.0)
        res = _make_broker(rb).sell("AAPL", 25.0)
        assert res.success
        assert rb.calls["sells"] == [("AAPL", 10.0)]
        assert res.quantity == pytest.approx(10.0)


class TestNeutralRegimeOverrides:
    """Fix 4: REGIME_PARAMS["neutral"] is built from cfg/config.yaml, not hardcoded None."""

    @staticmethod
    def _cfg_neutral() -> dict:
        with open(CONFIG_FILE) as f:
            return (yaml.safe_load(f).get("regime") or {}).get("neutral") or {}

    def test_neutral_index_pct_override_matches_config(self):
        from util import REGIME_PARAMS
        cfg = self._cfg_neutral()
        expected = cfg.get("index_pct_override")
        expected = float(expected) if expected is not None else None
        assert REGIME_PARAMS["neutral"]["index_pct_override"] == expected

    def test_neutral_max_buys_override_matches_config(self):
        from util import REGIME_PARAMS
        cfg = self._cfg_neutral()
        expected = cfg.get("max_buys_override")
        expected = int(expected) if expected is not None else None
        assert REGIME_PARAMS["neutral"]["max_buys_override"] == expected
