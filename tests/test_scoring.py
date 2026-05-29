"""
tests/test_scoring.py — Unified scoring engine tests.

Covers position_52w geometry, the warm-up momentum scorer (used by the simulator
during the first ~63 days), sell-rule config invariants, and the new SCORING_PARAMS
nested shape.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from data.fundamentals import _position_52w
from strategy.momentum import compute_warmup_momentum_score
from util import RISK_LIMITS, SCORING_PARAMS, SELL_RULES


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


class TestWarmupMomentumScore:

    def test_none_position_returns_zero(self):
        assert compute_warmup_momentum_score(None) == 0.0

    def test_low_position_gets_negative_score(self):
        assert compute_warmup_momentum_score(0.10) < 0

    def test_high_position_gets_positive_score(self):
        assert compute_warmup_momentum_score(0.85) > 0

    def test_uses_config_constants(self):
        mw = SCORING_PARAMS["momentum_warmup"]
        bins = mw["position_bin_boundaries"]
        scores = mw["position_bin_scores"]
        score = compute_warmup_momentum_score(bins[0] - 0.01)
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


class TestScoringParamsShape:

    def test_top_level_keys(self):
        for key in ("enabled", "peer_standardization", "factors",
                    "momentum_inputs", "momentum_warmup", "quality_checklist"):
            assert key in SCORING_PARAMS, f"missing scoring sub-block: {key!r}"

    def test_all_five_factors_present(self):
        for factor in ("value", "quality", "momentum", "income", "growth_leadership"):
            assert factor in SCORING_PARAMS["factors"], f"missing factor: {factor!r}"

    def test_value_factor_has_distress(self):
        assert "distress" in SCORING_PARAMS["factors"]["value"]
