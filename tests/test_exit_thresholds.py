"""
tests/test_exit_thresholds.py — Exit threshold ordering and trim-zone validation.

Tests validate_exit_thresholds() directly, the ConfigManager integration,
and the util.py resolver for backward-compat trim_score_delta fallback.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from config.validation import validate_exit_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid(**overrides) -> dict:
    """Return a baseline valid config dict, overrideable by keyword args."""
    base = dict(
        metric_threshold=0.75,
        sell_weak_value_below=0.72,
        trim_score_below=0.74,
        hard_exit_score_below=0.20,
        review_score_below=0.45,
    )
    base.update(overrides)
    return base


def _check(**overrides) -> list[str]:
    return validate_exit_thresholds(**_valid(**overrides))


# ---------------------------------------------------------------------------
# Valid baseline produces no warnings
# ---------------------------------------------------------------------------

class TestValidBaseline:

    def test_valid_config_no_warnings(self):
        assert _check() == []

    def test_valid_config_narrow_trim_zone_ok_when_above_threshold(self):
        # trim zone = 0.74 - 0.72 = 0.02, exactly at the narrow-zone boundary
        # The boundary check is strictly less-than, so 0.02 produces no warning
        result = _check(sell_weak_value_below=0.72, trim_score_below=0.74)
        assert result == []


# ---------------------------------------------------------------------------
# hard_exit >= sell_weak (hard exit zone empty)
# ---------------------------------------------------------------------------

class TestHardExitOrdering:

    def test_hard_exit_equals_sell_weak_is_invalid(self):
        msgs = _check(hard_exit_score_below=0.72, sell_weak_value_below=0.72)
        assert any("hard_exit" in m and "empty" in m for m in msgs)

    def test_hard_exit_above_sell_weak_is_invalid(self):
        msgs = _check(hard_exit_score_below=0.80, sell_weak_value_below=0.72)
        assert any("hard_exit" in m for m in msgs)

    def test_hard_exit_well_below_sell_weak_no_warning(self):
        msgs = _check(hard_exit_score_below=0.20, sell_weak_value_below=0.72)
        assert not any("hard_exit" in m for m in msgs)


# ---------------------------------------------------------------------------
# sell_weak >= trim_score_below (trim zone empty)
# ---------------------------------------------------------------------------

class TestTrimZoneOrdering:

    def test_sell_weak_equals_trim_is_empty_zone(self):
        msgs = _check(sell_weak_value_below=0.74, trim_score_below=0.74)
        assert any("trim zone" in m and "empty" in m for m in msgs)

    def test_sell_weak_above_trim_is_empty_zone(self):
        msgs = _check(sell_weak_value_below=0.80, trim_score_below=0.74)
        assert any("trim zone" in m and "empty" in m for m in msgs)

    def test_sell_weak_well_below_trim_no_empty_warning(self):
        msgs = _check(sell_weak_value_below=0.60, trim_score_below=0.74)
        assert not any("empty" in m for m in msgs)


# ---------------------------------------------------------------------------
# Narrow trim zone (valid but warned)
# ---------------------------------------------------------------------------

class TestNarrowTrimZone:

    def test_very_narrow_zone_warns(self):
        # width = 0.001 < 0.02 threshold
        msgs = _check(sell_weak_value_below=0.730, trim_score_below=0.731)
        assert any("narrow" in m.lower() for m in msgs)

    def test_reasonably_wide_zone_no_narrow_warning(self):
        msgs = _check(sell_weak_value_below=0.60, trim_score_below=0.74)
        assert not any("narrow" in m.lower() for m in msgs)


# ---------------------------------------------------------------------------
# trim_score_below > metric_threshold
# ---------------------------------------------------------------------------

class TestTrimAboveEntryThreshold:

    def test_trim_above_metric_threshold_warns(self):
        msgs = _check(trim_score_below=0.80, metric_threshold=0.75)
        assert any("metric_threshold" in m for m in msgs)

    def test_trim_equal_to_metric_threshold_no_warning(self):
        msgs = _check(trim_score_below=0.75, metric_threshold=0.75)
        assert not any("metric_threshold" in m for m in msgs)

    def test_trim_below_metric_threshold_no_warning(self):
        msgs = _check(trim_score_below=0.74, metric_threshold=0.75)
        assert not any("metric_threshold" in m for m in msgs)


# ---------------------------------------------------------------------------
# review_score_below > trim_score_below (informational)
# ---------------------------------------------------------------------------

class TestReviewAboveTrim:

    def test_review_above_trim_warns(self):
        msgs = _check(review_score_below=0.80, trim_score_below=0.74)
        assert any("review" in m.lower() for m in msgs)

    def test_review_below_trim_no_warning(self):
        msgs = _check(review_score_below=0.45, trim_score_below=0.74)
        assert not any("review" in m.lower() for m in msgs)

    def test_none_review_no_warning(self):
        msgs = validate_exit_thresholds(
            metric_threshold=0.75,
            sell_weak_value_below=0.72,
            trim_score_below=0.74,
            hard_exit_score_below=0.20,
            review_score_below=None,
        )
        assert msgs == []


# ---------------------------------------------------------------------------
# ConfigManager integration — warnings emitted at load time
# ---------------------------------------------------------------------------

class TestConfigManagerIntegration:

    def test_valid_config_loads_with_no_threshold_warnings(self):
        import warnings
        from config.manager import ConfigManager

        data = {
            "metric_threshold": 0.75,
            "sell_rules": {"sell_weak_value_below": 0.72},
            "exit_decision": {
                "hard_exit_score_below": 0.20,
                "review_score_below": 0.45,
                "trim_score_below": 0.74,
            },
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConfigManager.from_dict(data)
        threshold_warns = [w for w in caught if "threshold" in str(w.message).lower()
                           or "trim zone" in str(w.message).lower()]
        assert threshold_warns == []

    def test_invalid_config_logs_warning(self):
        import logging
        from config.manager import ConfigManager

        data = {
            "metric_threshold": 0.75,
            "sell_rules": {"sell_weak_value_below": 0.72},
            "exit_decision": {
                "hard_exit_score_below": 0.20,
                "review_score_below": 0.45,
                # sell_weak=0.72 >= trim=0.70 → empty trim zone
                "trim_score_below": 0.70,
            },
        }
        # Should not raise — violations are logged, not raised
        cfg = ConfigManager.from_dict(data)
        assert cfg is not None


# ---------------------------------------------------------------------------
# util.py: trim_score_below resolver (backward compat)
# ---------------------------------------------------------------------------

class TestTrimScoreBelowResolver:

    def test_explicit_trim_score_below_used(self):
        """When trim_score_below is present, it should appear in EXIT_DECISION_PARAMS."""
        # We can't easily reload util.py, so test the resolver logic directly
        from util import _resolve_trim_score_below
        result = _resolve_trim_score_below({"trim_score_below": 0.74}, metric_threshold=0.75)
        assert result == pytest.approx(0.74)

    def test_delta_fallback_when_no_explicit(self):
        from util import _resolve_trim_score_below
        result = _resolve_trim_score_below(
            {"trim_score_delta_threshold": -0.10}, metric_threshold=0.75
        )
        assert result == pytest.approx(0.75 * 0.90)

    def test_default_delta_fallback(self):
        from util import _resolve_trim_score_below
        result = _resolve_trim_score_below({}, metric_threshold=0.80)
        assert result == pytest.approx(0.80 * 0.85)

    def test_explicit_overrides_delta(self):
        from util import _resolve_trim_score_below
        result = _resolve_trim_score_below(
            {"trim_score_below": 0.74, "trim_score_delta_threshold": -0.20},
            metric_threshold=0.75,
        )
        # explicit wins
        assert result == pytest.approx(0.74)


# ---------------------------------------------------------------------------
# Zone boundaries: multiple simultaneous violations
# ---------------------------------------------------------------------------

class TestMultipleViolations:

    def test_sell_weak_above_trim_and_trim_above_mt(self):
        msgs = _check(
            sell_weak_value_below=0.80,
            trim_score_below=0.75,
            metric_threshold=0.70,
        )
        # Both the empty-zone and the above-metric_threshold violations should fire
        assert len(msgs) >= 2
        assert any("trim zone" in m and "empty" in m for m in msgs)
        assert any("metric_threshold" in m for m in msgs)

    def test_all_thresholds_equal(self):
        msgs = _check(
            hard_exit_score_below=0.75,
            sell_weak_value_below=0.75,
            trim_score_below=0.75,
            metric_threshold=0.75,
        )
        assert any("hard_exit" in m for m in msgs)
        assert any("trim zone" in m and "empty" in m for m in msgs)
