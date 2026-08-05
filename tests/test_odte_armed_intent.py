"""tests/test_odte_armed_intent.py — armed-intent schema validation + trigger evaluator.

Pure/offline: no broker, network, LLM, or files. Pins the two-lane contract: fail-closed
reason-coded validation at ARM time (restricted underlyings, executable universe, the same-day
15:30 ET expiry ceiling, well-formed trigger blocks) and the timestamped N-consecutive-checks
counter (too-fast polls never double-count, a definitive False resets, a stale gap resets across
restarts, an indeterminate read neither resets nor advances). Thresholds/universes come from the
live modules — never re-hardcoded.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_armed_intent as ai
from data.odte_config import EXECUTABLE_UNIVERSE

NOW = datetime(2026, 8, 5, 14, 31, 2, tzinfo=timezone.utc)     # 10:31 ET, Wednesday


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def _intent(**over):
    intent = {
        "intent_id": "spy-c-756-20260805-a1b2c3",
        "schema_version": ai.SCHEMA_VERSION,
        "armed_at": _iso(NOW - timedelta(minutes=5)),
        "armed_by": "hermes-controller-v6",
        "expires_at": _iso(NOW + timedelta(hours=2)),          # 12:31 ET < 15:30 ET ceiling
        "status": "armed",
        "symbol": "SPY",
        "direction": "bullish",
        "contract": {"option_id": "opt-756c", "chain_symbol": "SPY",
                     "expiration_date": "2026-08-05", "strike_price": 756.0,
                     "option_type": "call"},
        "trigger": {
            "level_acceptance": {"side": "above", "level": 755.60,
                                 "n_consecutive_checks": 3,
                                 "min_seconds_between_checks": 4},
            "confirmations": {"min_confirmers": 3, "max_dissenters": 0},
            "vixy_condition": "weak",
        },
        "sizing_hints": {"max_debit": 130.0, "tier_floor": "b_plus"},
    }
    intent.update(over)
    return intent


def _market(spy_last=756.0, **over):
    m = {
        "SPY": {"last": spy_last, "above_vwap": True, "orb_state": "above"},
        "QQQ": {"last": 697.0, "above_vwap": True, "orb_state": "above"},
        "IWM": {"last": 296.0, "above_vwap": True, "orb_state": "above"},
        "VIXY": {"above_vwap": False, "change_pct": -1.5},
    }
    m.update(over)
    return m


def _run_consecutive(intent, markets, start=NOW, gap_seconds=5):
    """Feed a sequence of markets through the evaluator, threading state like the daemon does."""
    state, result = {}, None
    for i, market in enumerate(markets):
        result = ai.evaluate_armed_intent(intent, market,
                                          state, start + timedelta(seconds=i * gap_seconds))
        state = result["state"]
    return result


# --- validation ------------------------------------------------------------------------------

def test_valid_intent_has_no_reasons():
    assert ai.validate_armed_intent(_intent()) == []


def test_missing_identity_fields_are_named():
    reasons = ai.validate_armed_intent(_intent(intent_id="", armed_by=""))
    assert "intent_id_missing" in reasons and "armed_by_missing" in reasons


def test_nvda_is_refused_at_arm_time():
    reasons = ai.validate_armed_intent(_intent(symbol="NVDA"))
    assert "symbol_restricted" in reasons
    bad_contract = _intent()
    bad_contract["contract"]["chain_symbol"] = "NVDA"
    assert "contract_symbol_restricted" in ai.validate_armed_intent(bad_contract)


def test_non_executable_universe_refused():
    assert "XSP" not in EXECUTABLE_UNIVERSE                    # scan-only vehicle
    assert "symbol_not_executable" in ai.validate_armed_intent(_intent(symbol="XSP"))


def test_contract_fields_required_and_must_match_symbol():
    intent = _intent()
    del intent["contract"]["option_id"]
    assert "contract_option_id_missing" in ai.validate_armed_intent(intent)
    mismatch = _intent()
    mismatch["contract"]["chain_symbol"] = "QQQ"
    assert "contract_symbol_mismatch" in ai.validate_armed_intent(mismatch)


def test_expiry_ceiling_same_et_day_1530():
    # 20:00 UTC == 16:00 ET — beyond the 15:30 ET ceiling.
    late = _intent(expires_at=_iso(NOW.replace(hour=20, minute=0)))
    assert "expires_at_beyond_session_ceiling" in ai.validate_armed_intent(late)
    tomorrow = _intent(expires_at=_iso(NOW + timedelta(days=1)))
    assert "expires_at_beyond_session_ceiling" in ai.validate_armed_intent(tomorrow)
    backwards = _intent(expires_at=_iso(NOW - timedelta(hours=1, minutes=6)))
    assert "expires_at_not_after_armed_at" in ai.validate_armed_intent(backwards)


def test_trigger_shape_is_validated():
    assert "trigger_empty" in ai.validate_armed_intent(_intent(trigger={}))
    bad = _intent()
    bad["trigger"]["level_acceptance"]["side"] = "sideways"
    assert "level_acceptance_side_invalid" in ai.validate_armed_intent(bad)
    bad_vixy = _intent()
    bad_vixy["trigger"]["vixy_condition"] = "calm"
    assert "vixy_condition_invalid" in ai.validate_armed_intent(bad_vixy)
    bad_tier = _intent()
    bad_tier["sizing_hints"]["tier_floor"] = "a_plus_plus"
    assert "tier_floor_invalid" in ai.validate_armed_intent(bad_tier)
    bad_fence = _intent()
    bad_fence["trigger"]["not_before_et"] = "ten am"
    assert "not_before_et_unparseable" in ai.validate_armed_intent(bad_fence)


# --- consecutive-check counter ---------------------------------------------------------------

def test_two_of_three_does_not_fire_three_straight_fires():
    intent = _intent()
    # accepted / rejected / accepted: counter resets on the definitive False.
    r = _run_consecutive(intent, [_market(756.0), _market(755.0), _market(756.0)])
    assert r["fires"] is False and r["consecutive_count"] == 1
    # three straight acceptances at the poll cadence: fires.
    r = _run_consecutive(intent, [_market(756.0)] * 3)
    assert r["fires"] is True, r["reasons"]
    assert r["consecutive_count"] == 3


def test_too_fast_polls_do_not_double_count():
    intent = _intent()                                         # min_seconds_between_checks = 4
    r1 = ai.evaluate_armed_intent(intent, _market(756.0), {}, NOW)
    r2 = ai.evaluate_armed_intent(intent, _market(756.0), r1["state"],
                                  NOW + timedelta(seconds=1))
    assert r2["consecutive_count"] == 1                        # too soon: not counted, not reset


def test_stale_gap_resets_counter_across_restarts():
    intent = _intent()
    la = intent["trigger"]["level_acceptance"]
    stale_after = ai.STALE_MULTIPLE * la["min_seconds_between_checks"]
    r1 = _run_consecutive(intent, [_market(756.0)] * 2)
    assert r1["consecutive_count"] == 2
    # A restart later than stale_after cannot resume the old count.
    r2 = ai.evaluate_armed_intent(intent, _market(756.0), r1["state"],
                                  NOW + timedelta(seconds=10 + stale_after + 1))
    assert r2["consecutive_count"] == 1
    # A restart INSIDE the staleness horizon keeps the fresh count.
    r3 = ai.evaluate_armed_intent(intent, _market(756.0), r1["state"],
                                  NOW + timedelta(seconds=10 + la["min_seconds_between_checks"]))
    assert r3["consecutive_count"] == 3


def test_indeterminate_tape_neither_resets_nor_advances():
    intent = _intent()
    r1 = _run_consecutive(intent, [_market(756.0)] * 2)
    no_spot = _market()
    no_spot["SPY"] = {"above_vwap": True, "orb_state": "above"}     # no last/price/spot/mark
    r2 = ai.evaluate_armed_intent(intent, no_spot, r1["state"], NOW + timedelta(seconds=15))
    assert r2["consecutive_count"] == 2
    assert r2["checks"]["level_acceptance"]["instant"] is None


def test_replay_is_deterministic():
    intent = _intent()
    state = _run_consecutive(intent, [_market(756.0)] * 2)["state"]
    a = ai.evaluate_armed_intent(intent, _market(756.0), dict(state), NOW + timedelta(seconds=20))
    b = ai.evaluate_armed_intent(intent, _market(756.0), dict(state), NOW + timedelta(seconds=20))
    assert a == b


def test_buffer_defaults_to_wall_shape():
    # Acceptance needs level + buffer, not a bare touch: at the level exactly, not accepted.
    intent = _intent()
    lvl = intent["trigger"]["level_acceptance"]["level"]
    buf = ai.level_buffer(lvl, None)
    assert buf == max(0.03, lvl * 0.0002)
    r = ai.evaluate_armed_intent(intent, _market(lvl), {}, NOW)
    assert r["checks"]["level_acceptance"]["instant"] is False
    r2 = ai.evaluate_armed_intent(intent, _market(lvl + buf), {}, NOW)
    assert r2["checks"]["level_acceptance"]["instant"] is True


# --- composed gates --------------------------------------------------------------------------

def test_dissenter_blocks_when_max_dissenters_zero():
    intent = _intent()
    market = _market(QQQ={"last": 697.0, "above_vwap": False, "orb_state": "below"})
    r = _run_consecutive(intent, [market] * 3)
    assert r["fires"] is False
    assert any(reason.startswith("confirmations_short") for reason in r["reasons"])
    # The level counter kept tracking the tape while confirmations lagged.
    assert r["consecutive_count"] == 3


def test_vixy_condition_gates():
    intent = _intent()
    firming = _market(VIXY={"above_vwap": True, "change_pct": 2.0})
    r = _run_consecutive(intent, [firming] * 3)
    assert r["fires"] is False and "vixy_not_weak" in r["reasons"]
    bear = _intent(direction="bearish")
    bear["trigger"]["vixy_condition"] = "firming"
    bear["trigger"]["level_acceptance"]["side"] = "below"
    bear["trigger"]["confirmations"] = {"min_confirmers": 1, "max_dissenters": 3}
    bear_market = {
        "SPY": {"last": 754.0, "above_vwap": False, "orb_state": "below"},
        "QQQ": {"last": 690.0, "above_vwap": False, "orb_state": "below"},
        "IWM": {"last": 292.0, "above_vwap": False, "orb_state": "below"},
        "VIXY": {"above_vwap": True, "change_pct": 2.0},
    }
    r2 = _run_consecutive(bear, [bear_market] * 3)
    assert r2["fires"] is True, r2["reasons"]


def test_wall_block_uses_level_acceptance_primitive():
    intent = _intent()
    intent["trigger"]["wall"] = {"level": 755.0, "side": "above"}
    r = _run_consecutive(intent, [_market(756.0)] * 3)
    assert r["fires"] is True, r["reasons"]
    intent["trigger"]["wall"] = {"level": 757.0, "side": "above"}
    r2 = _run_consecutive(intent, [_market(756.0)] * 3)
    assert r2["fires"] is False and "wall_not_accepted" in r2["reasons"]


def test_expired_or_disarmed_never_fires():
    expired = _intent(expires_at=_iso(NOW - timedelta(minutes=1)))
    r = ai.evaluate_armed_intent(expired, _market(756.0), {}, NOW)
    assert r["fires"] is False and "intent_expired" in r["reasons"]
    disarmed = _intent(status="disarmed")
    r2 = ai.evaluate_armed_intent(disarmed, _market(756.0), {}, NOW)
    assert r2["fires"] is False
    assert any(reason.startswith("intent_not_armed") for reason in r2["reasons"])


def test_et_time_fences():
    intent = _intent()
    intent["trigger"]["not_before_et"] = "11:00"               # NOW is 10:31 ET
    r = _run_consecutive(intent, [_market(756.0)] * 3)
    assert r["fires"] is False and "before_not_before_et" in r["reasons"]
    intent2 = _intent()
    intent2["trigger"]["not_after_et"] = "10:30"
    r2 = _run_consecutive(intent2, [_market(756.0)] * 3)
    assert r2["fires"] is False and "after_not_after_et" in r2["reasons"]
