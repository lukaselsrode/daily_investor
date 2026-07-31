"""tests/test_odte_candidate_watch.py — pre-entry candidate HAWK lane.

Pure/offline: no broker, network, LLM, or orders. Candidate watch may confirm that a fresh entry gate
should be built, but it must never set execution_allowed=True by itself.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_candidate_watch as cw
import data.odte_execution_policy as xp
import data.odte_loop_status as ls

NOW = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)


def _market(**over):
    m = {
        "day_verdict": "GOOD_DAY",
        "minutes_to_close": 240,
        "SPY": {"last": 741.0, "above_vwap": True, "orb_state": "above"},
        "QQQ": {"last": 724.2, "above_vwap": True, "orb_state": "above"},
        "IWM": {"last": 299.0, "above_vwap": True, "orb_state": "above"},
        "XSP": {"last": 741.0, "above_vwap": True, "orb_state": "above"},
        "VIXY": {"above_vwap": False, "change_pct": -2.0},
    }
    m.update(over)
    return m


def test_bullish_etf_candidate_confirms_from_tape_only():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(), now=NOW
    )
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["execution_allowed"] is False
    assert payload["scan_only"] is True
    assert payload["places_orders"] is False


def test_confirmed_candidate_locks_exact_option_contract_identity() -> None:
    vehicle_score = {
        "verdict": "GOOD_BET",
        "direction": "bullish",
        "contract": {
            "underlying": "QQQ",
            "option_type": "call",
            "option_id": "QQQ260629C00725000",
            "expiration_date": "2026-06-29",
            "strike_price": 725.0,
        },
    }
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"},
        market=_market(),
        vehicle_score=vehicle_score,
        now=NOW,
    )
    candidate = payload["candidate"]
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert candidate["option_id"] == "QQQ260629C00725000"
    assert candidate["expiration_date"] == "2026-06-29"
    assert candidate["strike_price"] == 725.0
    assert candidate["option_type"] == "call"
    assert candidate["selection_timestamp"] == NOW.isoformat()
    assert candidate["candidate_fingerprint"] == xp.candidate_fingerprint(candidate)


def test_contract_switch_starts_new_candidate_cycle() -> None:
    old = {
        "ticker": "QQQ", "direction": "bullish", "selected_vehicle": "QQQ",
        "option_id": "QQQ260629C00724000", "expiration_date": "2026-06-29",
        "strike_price": 724.0, "option_type": "call",
        "selection_timestamp": "2026-06-29T13:50:00+00:00",
    }
    vehicle_score = {
        "verdict": "GOOD_BET", "direction": "bullish",
        "contract": {
            "underlying": "QQQ", "option_type": "call",
            "option_id": "QQQ260629C00725000", "expiration_date": "2026-06-29",
            "strike_price": 725.0,
        },
    }
    payload = cw.evaluate_candidate_watch(old, market=_market(), vehicle_score=vehicle_score, now=NOW)
    candidate = payload["candidate"]
    assert candidate["selection_timestamp"] == NOW.isoformat()
    assert candidate["selection_reason"] == "exact_contract_switch_new_candidate_cycle"
    assert candidate["option_id"] == "QQQ260629C00725000"
    assert candidate["candidate_fingerprint"] == xp.candidate_fingerprint(candidate)


def test_market_only_etf_lane_creates_candidate_without_social():
    payload = cw.evaluate_candidate_watch({}, market=_market(), now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["candidate"]["ticker"] in cw.ETF_UNIVERSE
    assert payload["candidate"]["source"] == "etf_momentum_tape"
    assert payload["execution_allowed"] is False


def test_chop_keeps_watching_unless_a_plus_confirmation():
    m = _market(day_verdict="CHOP")
    m["IWM"] = {"last": 298.0, "above_vwap": False, "orb_state": "inside"}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=m, now=NOW
    )
    assert payload["decision"] == cw.KEEP_WATCHING
    assert any("CHOP" in r for r in payload["reasons"])


def test_pin_wall_blocks_confirmation_until_acceptance_above_wall():
    m = _market(QQQ={"last": 724.97, "above_vwap": True, "orb_state": "above"})
    gamma = {"pin_risk": {"level": "high"}, "call_wall": 725.0, "max_gamma_strike": 725.0}
    cand = {"ticker": "QQQ", "direction": "bullish", "strike": 725.0}
    payload = cw.evaluate_candidate_watch(cand, market=m, gamma_map=gamma, now=NOW)
    assert payload["decision"] == cw.KEEP_WATCHING
    assert any("wall acceptance" in r for r in payload["reasons"])

    m2 = _market(QQQ={"last": 725.25, "above_vwap": True, "orb_state": "above"})
    payload2 = cw.evaluate_candidate_watch(cand, market=m2, gamma_map=gamma, now=NOW)
    assert payload2["decision"] == cw.CONFIRM_ENTRY


def test_degraded_or_expired_candidate_never_authorizes_execution():
    avoid = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=_market(day_verdict="AVOID"), now=NOW
    )
    assert avoid["decision"] == cw.DEGRADED_NO_TRADE
    assert avoid["execution_allowed"] is False

    old = {"ticker": "SPY", "direction": "bullish", "created_at": "2026-06-29T13:00:00+00:00"}
    expired = cw.evaluate_candidate_watch(old, market=_market(), now=NOW, max_watch_minutes=20)
    assert expired["decision"] == cw.EXPIRED_NO_CONFIRMATION
    assert expired["execution_allowed"] is False


def test_broker_blocked_is_loud_and_not_normal_watch():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(),
        broker_health={"execution_lane": "blocked"}, now=NOW,
    )
    assert payload["decision"] == cw.BROKER_BLOCKED
    assert payload["state"] == "BROKER_BLOCKED"
    assert payload["execution_allowed"] is False


def test_loop_status_surfaces_active_candidate_before_stale_gated_gate():
    active_candidate = {"ticker": "QQQ", "direction": "bullish", "state": "WATCHING_CONFIRMATION"}
    candidate_decision = {
        "decision": cw.KEEP_WATCHING,
        "candidate": active_candidate,
        "scan_only": True,
        "execution_allowed": False,
    }
    gate = {
        "event_type": "entry_decision",
        "seq": 5,
        "ts": "2026-06-29T13:30:00+00:00",
        "underlying": "SPY",
        "decision": "observe",
        "scan_only": True,
        "execution_allowed": False,
    }
    r = ls.derive_loop_state(active_candidate=active_candidate, candidate_decision=candidate_decision,
                             journal_events=[gate], now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["next_command"] == "odte-candidate-watch"
    assert r["context"]["candidate_watch"] is True
    assert r["executable"] is False


def test_loop_status_ignores_degraded_candidate_and_falls_through_to_scan():
    r = ls.derive_loop_state(
        active_candidate={"ticker": "QQQ", "direction": "bullish"},
        candidate_decision={"decision": cw.DEGRADED_NO_TRADE, "candidate": {"ticker": "QQQ"}},
        now=NOW,
    )
    assert r["state"] == "SCAN"


def test_loop_status_surfaces_broker_blocked_candidate_as_execution_lane_blocker():
    active_candidate = {"ticker": "QQQ", "direction": "bullish", "state": "BROKER_BLOCKED"}
    candidate_decision = {
        "decision": cw.BROKER_BLOCKED,
        "candidate": active_candidate,
        "scan_only": True,
        "execution_allowed": False,
    }
    r = ls.derive_loop_state(active_candidate=active_candidate, candidate_decision=candidate_decision,
                             now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["next_command"] == "verify-broker-review-lane"
    assert "blocked" in r["next_action"]
    assert r["executable"] is False


# --- VEHICLE LOCK (2026-07-23 remediation): one thesis, one underlying, one fingerprint ---------

def _bearish_market(**over):
    m = {
        "day_verdict": "GOOD_DAY",
        "minutes_to_close": 240,
        "SPY": {"last": 736.4, "above_vwap": False, "orb_state": "below"},
        "QQQ": {"last": 718.6, "above_vwap": False, "orb_state": "below"},
        "IWM": {"last": 296.0, "above_vwap": False, "orb_state": "below"},
        "XSP": {"last": 736.4, "above_vwap": False, "orb_state": "below"},
        "VIXY": {"above_vwap": True, "change_pct": 2.0},
    }
    m.update(over)
    return m


def test_qqq_candidate_with_spy_contract_is_hard_mismatch():
    # The exact failure the lock closes: the controller discusses QQQ but the vehicle score carries
    # a SPY contract. That is a HARD mismatch — degrade, never substitute.
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bearish"}, market=_bearish_market(),
        vehicle_score={"verdict": "GOOD_BET", "direction": "bearish",
                       "contract": {"underlying": "SPY", "option_type": "put", "strike": 737}},
        now=NOW)
    assert payload["decision"] == cw.DEGRADED_NO_TRADE
    assert payload["checks"]["vehicle_underlying"] == "SPY"
    assert any("mismatch" in r for r in payload["reasons"])
    assert payload["execution_allowed"] is False


def test_qqq_candidate_with_qqq_contract_matching_direction_proceeds():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bearish"}, market=_bearish_market(),
        vehicle_score={"verdict": "GOOD_BET", "direction": "bearish",
                       "contract": {"underlying": "QQQ", "option_type": "put", "strike": 718}},
        now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["candidate"]["selected_vehicle"] == "QQQ"
    assert payload["execution_allowed"] is False               # a watch never executes by itself


def test_vehicle_lock_fields_persisted_on_candidate_and_active_file(tmp_path):
    import json
    # No created_at: run_candidate_watch uses the wall clock, so a dated candidate would expire.
    payload = cw.run_candidate_watch(
        candidate_json=json.dumps({"ticker": "QQQ", "direction": "bearish"}),
        market_json=json.dumps(_bearish_market()),
        state_dir=str(tmp_path), write=True)
    cand = payload["candidate"]
    for field in ("selected_vehicle", "selection_reason", "relative_strength_rank",
                  "selection_timestamp", "candidate_fingerprint"):
        assert cand.get(field) is not None, field
    assert cand["selected_vehicle"] == "QQQ"
    active = json.loads((tmp_path / cw.ACTIVE_CANDIDATE_FILENAME).read_text())
    assert active["selected_vehicle"] == "QQQ"
    assert active["candidate_fingerprint"] == cand["candidate_fingerprint"]
    assert active["selection_timestamp"] == cand["selection_timestamp"]


def test_vehicle_switch_mints_new_fingerprint_and_invalidates_old_lease():
    from data.odte_execution_policy import lease_matches_candidate
    qqq = cw.evaluate_candidate_watch({"ticker": "QQQ", "direction": "bearish",
                                       "created_at": NOW.isoformat()},
                                      market=_bearish_market(), now=NOW)
    spy = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bearish",
                                       "created_at": NOW.isoformat()},
                                      market=_bearish_market(), now=NOW)
    fp_qqq = qqq["candidate"]["candidate_fingerprint"]
    fp_spy = spy["candidate"]["candidate_fingerprint"]
    assert fp_qqq and fp_spy and fp_qqq != fp_spy
    # A lease bound to the QQQ cycle can never bind the SPY cycle.
    fake_lease = {"lease_id": "x", "candidate_fingerprint": fp_qqq}
    assert lease_matches_candidate(fake_lease, qqq["candidate"]) is True
    assert lease_matches_candidate(fake_lease, spy["candidate"]) is False


def test_relative_strength_rank_is_deterministic_selection_context():
    # SPY is the only ETF fully aligned bearish -> rank 1; QQQ half-aligned -> rank 2.
    m = _bearish_market(QQQ={"last": 718.6, "above_vwap": False, "orb_state": "inside"},
                        IWM={"last": 296.0, "above_vwap": True, "orb_state": "above"},
                        XSP={})
    spy = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bearish"}, market=m, now=NOW)
    assert spy["candidate"]["relative_strength_rank"] == 1
    qqq = cw.evaluate_candidate_watch({"ticker": "QQQ", "direction": "bearish"}, market=m, now=NOW)
    assert qqq["candidate"]["relative_strength_rank"] == 2
