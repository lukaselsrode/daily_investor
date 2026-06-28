"""tests/test_odte_loop_status.py — 0DTE loop state machine (pure/offline observability).

No Robinhood, no network, no LLM. derive_loop_state() is pure: the canonical data/odte artifact
payloads in, the current loop state + next command out. It re-derives no gate/decision — it only
summarizes what the other tools wrote. These tests pin the PRIORITY ORDERING that makes the live
loop obvious: a live position/position_decision beats a scan trigger; a scan_only trigger is an
observe-only CANDIDATE (never executable); a promoted/denied entry gate maps to PROMOTED/GATED; and
missing/stale/malformed artifacts degrade or fall back to SCAN without crashing.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_loop_status as ls

NOW = datetime(2026, 6, 26, 18, 0, tzinfo=timezone.utc)


def _ts(minutes_ago=0, hours_ago=0):
    return (NOW - timedelta(minutes=minutes_ago, hours=hours_ago)).isoformat()


def _open_plan(**over):
    plan = {"status": "open", "underlying": "SPY", "option_type": "call", "option_id": "x",
            "entry_price": 1.0, "quantity": 1, "trade_id": "SPY-T1"}
    plan.update(over)
    return plan


def _closed_plan(**over):
    plan = _open_plan(status="closed", quantity=0, entry_price=0.17, exit_price=0.25,
                      gross_pnl=8.0, closed_at=_ts(minutes_ago=30),
                      entry_fill_time=_ts(minutes_ago=45), trade_id="SPY-T1")
    plan.update(over)
    return plan


def _candidate_triggers(**over):
    trig = {"alert": False, "spy_verdict": "OBSERVE",
            "candidate": {"ticker": "SPY", "direction": "bearish"}}
    trig.update(over)
    return trig


def _entry_decision(**over):
    e = {"event_type": "entry_decision", "seq": 5, "ts": _ts(minutes_ago=5),
         "underlying": "SPY", "direction": "call", "decision": "deny",
         "scan_only": False, "execution_allowed": False}
    e.update(over)
    return e


# --- priority: a live position beats a fresh scan trigger ------------------------------------

def test_live_position_beats_trigger():
    # An open plan + a fresh actionable trigger -> MANAGING, never CANDIDATE. Live always wins.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=1)},
                             triggers=_candidate_triggers(alert=True), now=NOW)
    assert r["state"] == "MANAGING"
    assert r["executable"] is True
    assert r["next_command"].startswith("odte-position")


def test_actionable_exit_trigger_is_managing():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "TAKE_PROFIT", "underlying": "SPY",
                                                "pnl_pct": 0.42, "ts": _ts(minutes_ago=1)},
                             now=NOW)
    assert r["state"] == "MANAGING"
    assert r["context"]["decision"] == "TAKE_PROFIT"


def test_plan_open_no_decision_is_entered():
    # Plan open but position_decision still says NO_POSITION (just filled, not yet managed) -> ENTERED.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "NO_POSITION"}, now=NOW)
    assert r["state"] == "ENTERED"
    assert r["executable"] is True


# --- scan_only trigger -> CANDIDATE / observe (never executable) -----------------------------

def test_scan_only_trigger_is_candidate_observe():
    r = ls.derive_loop_state(triggers=_candidate_triggers(), now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["executable"] is False
    assert r["context"]["scan_only"] is True
    assert r["next_command"] == "odte-entry-gate"


def test_restricted_candidate_does_not_promote_to_candidate():
    # A restricted (NVDA-style) candidate must not surface as an actionable CANDIDATE.
    r = ls.derive_loop_state(triggers={"candidate": {"ticker": "NVDA", "direction": "call",
                                                     "restricted": True}}, now=NOW)
    assert r["state"] == "SCAN"


# --- entry gate -> GATED / PROMOTED ----------------------------------------------------------

def test_entry_gate_denied_is_gated():
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="deny")], now=NOW)
    assert r["state"] == "GATED"
    assert r["executable"] is False
    assert r["next_command"].endswith("--promote-to-execution")


def test_entry_gate_executable_is_promoted():
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter",
                                                             execution_allowed=True)], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["executable"] is True
    assert r["next_command"] == "odte-position"


def test_enter_verb_without_execution_allowed_is_gated_not_promoted():
    # PROMOTED requires EXPLICIT permission: execution_allowed is True. A stale/partial record whose
    # decision says "enter" but whose execution_allowed is missing/false must stay GATED, never
    # PROMOTED/executable — the loop only advances on explicit gate permission, not the intent verb.
    missing = _entry_decision(decision="enter")
    missing.pop("execution_allowed")            # execution_allowed absent entirely
    r = ls.derive_loop_state(journal_events=[missing], now=NOW)
    assert r["state"] == "GATED"
    assert r["executable"] is False

    false_flag = _entry_decision(decision="enter", execution_allowed=False)
    r2 = ls.derive_loop_state(journal_events=[false_flag], now=NOW)
    assert r2["state"] == "GATED"
    assert r2["executable"] is False


def test_fresh_gate_outranks_scan_trigger():
    # A current-cycle gate beats a bare scan candidate.
    r = ls.derive_loop_state(triggers=_candidate_triggers(),
                             journal_events=[_entry_decision(decision="deny")], now=NOW)
    assert r["state"] == "GATED"


def test_stale_gate_from_prior_trade_is_ignored():
    # A gate dated BEFORE the closed trade's close is that trade's own (consumed) gate -> not GATED.
    # The closed+reviewed trade falls through to REVIEWED, not a stale PROMOTED.
    gate = _entry_decision(decision="enter", execution_allowed=True, ts=_ts(hours_ago=2))
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=10)}
    r = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[gate, pm], now=NOW)
    assert r["state"] == "REVIEWED"


# --- exit / review lane ----------------------------------------------------------------------

def test_closed_unreviewed_trade_is_exited():
    r = ls.derive_loop_state(active_trade=_closed_plan(), now=NOW)
    assert r["state"] == "EXITED"
    assert r["context"]["trade_id"] == "SPY-T1"
    assert "postmortem" in r["next_command"].lower() or "journal" in r["next_command"].lower()


def test_closed_reviewed_trade_is_reviewed():
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=10)}
    r = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[pm], now=NOW)
    assert r["state"] == "REVIEWED"


def test_exited_outranks_fresh_candidate():
    # Discipline: review the closed trade before chasing a new scan candidate.
    r = ls.derive_loop_state(active_trade=_closed_plan(), triggers=_candidate_triggers(), now=NOW)
    assert r["state"] == "EXITED"


def test_stale_closed_trade_does_not_mask_fresh_candidate():
    # A long-abandoned unreviewed trade stops nagging EXITED so a new scan stays visible.
    old = _closed_plan(closed_at=_ts(hours_ago=72), entry_fill_time=_ts(hours_ago=73))
    r = ls.derive_loop_state(active_trade=old, triggers=_candidate_triggers(), now=NOW)
    assert r["state"] == "CANDIDATE"


# --- degraded / fail-soft --------------------------------------------------------------------

def test_monitoring_degraded_is_degraded():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "MONITORING_DEGRADED",
                                                "underlying": "SPY", "ts": _ts(minutes_ago=1)},
                             now=NOW)
    assert r["state"] == "DEGRADED"
    assert r["executable"] is False


def test_stale_live_decision_is_degraded():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(hours_ago=6)}, now=NOW)
    assert r["state"] == "DEGRADED"
    assert any("stale" in s for s in r["reasons"])


def test_live_scalp_decision_stales_fast_for_hawk_mode():
    # 0DTE scalp management should not allow a hours-old default. If a live contract has not had a
    # fresh management decision for >10 minutes, the loop should surface DEGRADED so Hermes treats it
    # as a risk event rather than continuing to scan.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=11)}, now=NOW)
    assert r["state"] == "DEGRADED"
    assert r["loop_stage"] == "degraded"


def test_malformed_live_artifact_degrades_not_crashes():
    # active_trade present-but-malformed while a plan looks open -> DEGRADED, never an exception.
    r = ls.derive_loop_state(active_trade=_open_plan(), errors={"active_trade"}, now=NOW)
    assert r["state"] == "DEGRADED"


def test_all_missing_is_scan():
    r = ls.derive_loop_state(now=NOW)
    assert r["state"] == "SCAN"
    assert r["executable"] is False
    assert r["places_orders"] is False


def test_never_places_orders_flag():
    for kwargs in ({}, {"active_trade": _open_plan(),
                        "position_decision": {"decision": "HOLD", "ts": _ts(minutes_ago=1)}},
                   {"journal_events": [_entry_decision(decision="enter", execution_allowed=True)]}):
        assert ls.derive_loop_state(now=NOW, **kwargs)["places_orders"] is False


# --- run_loop_status: reads files, status-aware, JSON-clean ----------------------------------

def test_run_loop_status_reads_artifacts(tmp_path):
    (tmp_path / "active_trade.json").write_text(json.dumps(_open_plan()))
    (tmp_path / "position_decision.json").write_text(
        json.dumps({"decision": "HOLD", "underlying": "SPY", "ts": _ts(minutes_ago=1)}))
    payload = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert payload["state"] == "MANAGING"
    assert payload["artifacts"]["active_trade"] == "ok"
    assert payload["artifacts"]["position_decision"] == "ok"
    assert payload["artifacts"]["triggers"] == "missing"
    # stdout contract must be JSON-serializable.
    json.dumps(payload)


def test_run_loop_status_empty_dir_is_scan(tmp_path):
    payload = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert payload["state"] == "SCAN"
    assert payload["artifacts"]["journal_events"] == 0


def test_run_loop_status_malformed_live_file_degrades(tmp_path):
    (tmp_path / "active_trade.json").write_text("{ this is not json")
    payload = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert payload["artifacts"]["active_trade"] == "invalid"
    # A malformed live artifact must not crash; with no other context it falls back to SCAN.
    assert payload["state"] in ("SCAN", "DEGRADED")
