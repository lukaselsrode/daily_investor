"""
tests/test_scoring.py — Scoring function tests.

Migrated / adapted from src/tests.py. These tests import from source_data
directly until Phase 3 (strategy layer migration).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from data.fundamentals import _position_52w, get_momentum_score
from util import MOMENTUM_PARAMS, SELL_RULES, RISK_LIMITS


class TestPosition52w:

    def test_midpoint(self):
        assert _position_52w(150.0, 100.0, 200.0) == pytest.approx(0.5)

    def test_at_low(self):
        assert _position_52w(100.0, 100.0, 200.0) == pytest.approx(0.0)

    def test_at_high(self):
        assert _position_52w(200.0, 100.0, 200.0) == pytest.approx(1.0)

    def test_above_high_clamped(self):
        assert _position_52w(250.0, 100.0, 200.0) == pytest.approx(1.0)

    def test_below_low_clamped(self):
        assert _position_52w(50.0, 100.0, 200.0) == pytest.approx(0.0)

    def test_equal_high_low_returns_none(self):
        assert _position_52w(100.0, 100.0, 100.0) is None

    def test_none_inputs(self):
        assert _position_52w(None, 100.0, 200.0) is None
        assert _position_52w(100.0, None, 200.0) is None
        assert _position_52w(100.0, 100.0, None) is None


class TestMomentumScore:

    def test_none_position_returns_zero(self):
        assert get_momentum_score(None) == 0.0

    def test_low_position_gets_negative_score(self):
        score = get_momentum_score(0.10)
        assert score < 0

    def test_high_position_gets_positive_score(self):
        score = get_momentum_score(0.85)
        assert score > 0

    def test_uses_config_constants(self):
        bins = MOMENTUM_PARAMS["position_bin_boundaries"]
        scores = MOMENTUM_PARAMS["position_bin_scores"]
        # first bin: position < bins[0]
        score = get_momentum_score(bins[0] - 0.01)
        assert score == pytest.approx(scores[0])


class TestSellRulesFromConfig:

    def test_stop_loss_is_negative(self):
        assert SELL_RULES["stop_loss_pct"] < 0

    def test_trailing_stop_is_negative(self):
        assert SELL_RULES["trailing_stop_pct"] < 0

    def test_take_profit_is_positive(self):
        assert SELL_RULES["take_profit_pct"] > 0

    def test_sell_weak_value_is_positive(self):
        assert SELL_RULES["sell_weak_value_below"] > 0

    def test_minimum_hold_days_is_nonnegative(self):
        assert RISK_LIMITS["minimum_hold_days"] >= 0

    def test_minimum_days_before_take_profit_is_nonnegative(self):
        assert SELL_RULES["minimum_days_before_take_profit"] >= 0
