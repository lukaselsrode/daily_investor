"""
tests/test_sell_engine.py — SellDecisionEngine tests (Phase 4, no main.py dependency).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from core.types import SellDecision
from portfolio.sell_engine import SellDecisionEngine
from util import METRIC_THRESHOLD, SELL_RULES


def _engine() -> SellDecisionEngine:
    return SellDecisionEngine()


def _holding(
    percent_change: float | None = None,
    price: float = 100.0,
    average_buy_price: float = 100.0,
    quantity: float = 1.0,
) -> dict:
    h: dict = {"price": str(price), "average_buy_price": str(average_buy_price), "quantity": str(quantity)}
    if percent_change is not None:
        h["percent_change"] = str(percent_change * 100)
    return h


def _metrics(value_metric: float = 1.0, quality_score: float = 0.5, yield_trap: bool = False) -> pd.Series:
    return pd.Series({
        "value_metric": value_metric,
        "quality_score": quality_score,
        "yield_trap_flag": yield_trap,
    })


class TestHardSells:

    def test_stop_loss_triggers(self):
        stop = SELL_RULES["stop_loss_pct"]
        h = _holding(percent_change=stop - 0.01)
        d = _engine().evaluate("AAPL", h, _metrics())
        assert d.should_sell
        assert d.severity == "hard"
        assert d.exit_type == "failure_exit"
        assert "stop loss" in d.reason

    def test_stop_loss_exactly_at_boundary_triggers(self):
        stop = SELL_RULES["stop_loss_pct"]
        h = _holding(percent_change=stop)
        d = _engine().evaluate("AAPL", h, _metrics())
        assert d.should_sell
        assert d.severity == "hard"

    def test_trailing_stop_triggers(self):
        ts = SELL_RULES["trailing_stop_pct"]
        # price is at peak * (1 + trailing_stop - epsilon) → drawdown exceeds threshold
        peak = 200.0
        current = peak * (1 + ts - 0.01)
        h = _holding(price=current)
        d = _engine().evaluate("AAPL", h, _metrics(), peak_price=peak)
        assert d.should_sell
        assert d.severity == "hard"
        assert "trailing stop" in d.reason

    def test_trailing_stop_no_trigger_above_threshold(self):
        ts = SELL_RULES["trailing_stop_pct"]
        peak = 200.0
        current = peak * (1 + ts + 0.05)  # still well above trailing stop
        h = _holding(price=current)
        d = _engine().evaluate("AAPL", h, _metrics(), peak_price=peak)
        # should NOT sell on trailing stop alone
        assert d.severity != "hard" or "trailing stop" not in d.reason

    def test_quality_floor_triggers(self):
        floor = SELL_RULES["sell_low_quality_below"]
        d = _engine().evaluate("AAPL", _holding(), _metrics(quality_score=floor - 0.1))
        assert d.should_sell
        assert d.severity == "hard"
        assert "quality" in d.reason.lower()

    def test_yield_trap_hard_sell(self):
        sell_weak = SELL_RULES["sell_weak_value_below"]
        if not SELL_RULES["sell_yield_trap"]:
            pytest.skip("sell_yield_trap disabled in config")
        d = _engine().evaluate("AAPL", _holding(), _metrics(value_metric=sell_weak - 0.1, yield_trap=True))
        assert d.should_sell
        assert d.severity == "hard"
        assert "yield trap" in d.reason


class TestSoftSells:

    def test_take_profit_triggers(self):
        tp = SELL_RULES["take_profit_pct"]
        h = _holding(percent_change=tp + 0.05)
        # value_metric below cheap floor → take profit fires
        floor_multiplier = SELL_RULES["take_profit_value_floor_multiplier"]
        cheap_threshold = METRIC_THRESHOLD * floor_multiplier
        d = _engine().evaluate("AAPL", h, _metrics(value_metric=cheap_threshold - 0.1))
        assert d.should_sell
        assert d.severity == "soft"
        assert d.exit_type == "harvest_exit"

    def test_take_profit_held_if_cheap(self):
        tp = SELL_RULES["take_profit_pct"]
        h = _holding(percent_change=tp + 0.05)
        floor_multiplier = SELL_RULES["take_profit_value_floor_multiplier"]
        cheap_threshold = METRIC_THRESHOLD * floor_multiplier
        # value_metric well above floor → holding because fundamentally cheap
        d = _engine().evaluate("AAPL", h, _metrics(value_metric=cheap_threshold + 0.5))
        # Should NOT be a harvest_exit
        assert not (d.should_sell and d.exit_type == "harvest_exit")

    def test_weak_value_triggers_thesis_exit(self):
        sell_weak = SELL_RULES["sell_weak_value_below"]
        min_days = SELL_RULES["min_days_held_before_value_exit"]
        d = _engine().evaluate(
            "AAPL",
            _holding(),
            _metrics(value_metric=sell_weak - 0.1),
            # No created_at → days_held=None → treated as "held long enough"
        )
        # days_held is None, so the min_days gate passes
        assert d.should_sell
        assert d.severity == "soft"
        assert d.exit_type == "thesis_exit"


class TestNoSell:

    def test_no_sell_for_healthy_position(self):
        tp = SELL_RULES["take_profit_pct"]
        sell_weak = SELL_RULES["sell_weak_value_below"]
        h = _holding(percent_change=tp * 0.5)
        m = _metrics(value_metric=sell_weak + 0.5, quality_score=0.8)
        d = _engine().evaluate("AAPL", h, m)
        assert not d.should_sell
        assert d.reason == "no sell condition met"


class TestEvaluateHoldings:

    def test_skips_etf_symbols(self):
        holdings = {"SPY": {"quantity": "10", "price": "400", "equity": "4000"}}
        etfs = {"SPY"}
        hard, soft = _engine().evaluate_holdings(holdings, None, {}, etfs)
        assert "SPY" not in hard
        assert "SPY" not in soft

    def test_skips_zero_quantity(self):
        holdings = {"AAPL": {"quantity": "0", "price": "100", "equity": "0"}}
        hard, soft = _engine().evaluate_holdings(holdings, None, {}, set())
        assert "AAPL" not in hard
        assert "AAPL" not in soft

    def test_hard_sell_appears_in_hard_dict(self):
        stop = SELL_RULES["stop_loss_pct"]
        holdings = {
            "AAPL": {"quantity": "1", "price": "80", "percent_change": str((stop - 0.05) * 100)},
        }
        hard, soft = _engine().evaluate_holdings(holdings, None, {}, set())
        assert "AAPL" in hard
        assert "AAPL" not in soft

    def test_returns_sell_decision_objects(self):
        stop = SELL_RULES["stop_loss_pct"]
        holdings = {"AAPL": {"quantity": "1", "price": "80", "percent_change": str((stop - 0.05) * 100)}}
        hard, _ = _engine().evaluate_holdings(holdings, None, {}, set())
        assert isinstance(hard["AAPL"], SellDecision)
