"""tests/test_odte_strategy_policy.py — canonical ODTE guardrail policy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.odte_strategy_policy import (
    canonical_guardrails,
    classify_strategy_window,
    window_descriptions,
)


def test_strategy_window_classification_uses_et():
    assert classify_strategy_window("2026-06-30T13:45:00Z") == "high_action_0930_1300"
    assert classify_strategy_window("2026-06-30T17:10:00Z") == "midday_1300_1530"
    assert classify_strategy_window("2026-06-30T19:45:00Z") == "late_1530_close"
    assert classify_strategy_window("2026-06-30T21:00:00Z") == "outside_regular"
    assert classify_strategy_window("not-a-time") == "unknown"


def test_canonical_guardrails_include_operational_lessons():
    keys = {g["key"] for g in canonical_guardrails()}
    assert "broker_truth_first" in keys
    assert "stale_artifact_veto" in keys
    assert "post_loss_cooldown" in keys
    assert "longer_dated_when_better" in keys
    assert "reclaim_retest_observe_only" in keys
    assert "low_bp_lotto_veto" in keys


def test_window_descriptions_capture_high_action_observation():
    desc = window_descriptions()["high_action_0930_1300"]
    assert "09:30" in desc["description"]
    assert "not permission" in desc["posture"]
