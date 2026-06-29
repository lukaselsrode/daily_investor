"""tests/test_odte_candidate_watch.py — pre-entry candidate HAWK lane.

Pure/offline: no broker, network, LLM, or orders. Candidate watch may confirm that a fresh entry gate
should be built, but it must never set execution_allowed=True by itself.
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_candidate_watch as cw
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
