"""
tests/test_defensive_stop_tighten.py — regime.defensive.stop_loss_tighten is live.

The knob existed in config, the schema, and the UI regime page but had NO
consumer — neither the live SellDecisionEngine nor the simulator read it.
It now tightens the stop-loss floor in defensive regimes in BOTH paths
(e.g. -0.30 → -0.25 with the live tighten of 0.05).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from portfolio.sell_engine import SellDecisionEngine, evaluate_sell_candidate
from util import REGIME_PARAMS, SELL_RULES

STOP = float(SELL_RULES["stop_loss_pct"])
TIGHTEN = float(REGIME_PARAMS["defensive"].get("stop_loss_tighten", 0.0))

pytestmark = pytest.mark.skipif(
    TIGHTEN <= 0, reason="stop_loss_tighten disabled in live config"
)


def _holding(pct_change: float) -> dict:
    # Robinhood reports percent_change as a percentage string.
    return {"percent_change": str(pct_change * 100), "quantity": "10",
            "average_buy_price": "100", "price": str(100 * (1 + pct_change))}


def _between_stops() -> float:
    """A loss deeper than the tightened stop but shallower than the base stop."""
    return STOP + TIGHTEN / 2.0   # e.g. -0.30 + 0.025 = -0.275


class TestLiveEngine:

    def test_defensive_regime_tightens_stop(self):
        eng = SellDecisionEngine()
        pct = _between_stops()
        normal = eng.evaluate("XYZ", _holding(pct), None, regime="bullish")
        defensive = eng.evaluate("XYZ", _holding(pct), None, regime="defensive")
        assert not (normal.should_sell and "stop loss" in normal.reason)
        assert defensive.should_sell and "stop loss" in defensive.reason

    def test_base_stop_unchanged_outside_defensive(self):
        eng = SellDecisionEngine()
        deep = STOP - 0.02   # beyond the base stop — must fire in ANY regime
        for regime in (None, "bullish", "neutral"):
            d = eng.evaluate("XYZ", _holding(deep), None, regime=regime)
            assert d.should_sell and "stop loss" in d.reason

    def test_wrapper_threads_regime(self):
        pct = _between_stops()
        d = evaluate_sell_candidate("XYZ", _holding(pct), None, regime="defensive")
        assert d["should_sell"] and "stop loss" in d["reason"]
        d2 = evaluate_sell_candidate("XYZ", _holding(pct), None, regime="bullish")
        assert not (d2["should_sell"] and "stop loss" in d2["reason"])


class TestSimParity:

    def test_simulator_reads_the_same_knob(self):
        """Live and backtest must tighten by the SAME configured amount."""
        import backtesting.simulator as sim
        assert sim._STOP_TIGHTEN == pytest.approx(TIGHTEN)
        assert sim._STOP_LOSS_PCT == pytest.approx(STOP)
