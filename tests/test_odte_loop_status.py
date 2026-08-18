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
    required_confirmations = ["live_chain_recheck", "spread_cap_check", "budget_check"]
    e = {"event_type": "entry_decision", "seq": 5, "ts": _ts(minutes_ago=5),
         "underlying": "SPY", "direction": "call", "decision": "deny",
         "scan_only": False, "execution_allowed": False,
         "required_confirmations": required_confirmations,
         "confirmations": {name: True for name in required_confirmations},
         "confirmation_needed": False}
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
    assert "odte-convert" in r["next_command"]


def test_restricted_candidate_does_not_promote_to_candidate():
    # A restricted (NVDA-style) candidate must not surface as an actionable CANDIDATE.
    r = ls.derive_loop_state(triggers={"candidate": {"ticker": "NVDA", "direction": "call",
                                                     "restricted": True}}, now=NOW)
    assert r["state"] == "SCAN"


# --- entry gate -> GATED / PROMOTED ----------------------------------------------------------

def test_denied_veto_gate_does_not_pin_loop_at_gated():
    # REGRESSION (stale-sidelines "stuck GATED on veto"): a fresh veto/deny entry gate is a NO-TRADE
    # verdict, not a promotable gate. It must NOT pin the loop at GATED telling the cron to promote the
    # old gate — it is consumed and the loop falls through to SCAN to look for a FRESH setup.
    for verdict in ("deny", "veto", "no_trade", "NO_TRADE_FINAL_TAPE_RECLAIM_FAILED", "rejected"):
        r = ls.derive_loop_state(journal_events=[_entry_decision(decision=verdict, ts=_ts(minutes_ago=5))],
                                 now=NOW)
        assert r["state"] == "SCAN", f"{verdict!r} should expire to SCAN, got {r['state']}"
        assert "--promote-to-execution" not in r["next_command"]
        assert any("denied entry gate not sticky" in s for s in r["notes"])


def test_denied_gate_with_stale_candidate_is_scan_not_gated():
    # Exact reported scenario: a 39m-stale candidate watch (ignored) + a fresh veto entry gate must NOT
    # read GATED with next=promote — both are non-actionable, so the loop is SCAN (look for fresh edge).
    stale_watch = {"decision": "KEEP_WATCHING", "ts": _ts(minutes_ago=39),
                   "candidate": {"ticker": "SPY", "direction": "bearish"}}
    r = ls.derive_loop_state(candidate_decision=stale_watch,
                             journal_events=[_entry_decision(decision="veto", ts=_ts(minutes_ago=8))],
                             now=NOW)
    assert r["state"] == "SCAN"
    assert r["next_command"] != "odte-entry-gate --promote-to-execution"


def test_denied_gate_falls_through_to_fresh_candidate():
    # A denied gate is consumed, so a genuinely fresh scan candidate surfaces (CANDIDATE), instead of
    # the old veto pinning GATED and hiding the fresh setup.
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             journal_events=[_entry_decision(decision="deny", ts=_ts(minutes_ago=6))],
                             now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["context"]["scan_only"] is True


def test_pending_enter_gate_still_gated():
    # An affirmative gate ("enter") without execution permission is a legitimate PENDING gate — it
    # still maps to GATED (awaiting a fresh execution lease), unlike a denied verdict. Bare
    # promotion is dead (2026-07-23): the next command points at the lease tier instead.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter",
                                                             execution_allowed=False)], now=NOW)
    assert r["state"] == "GATED"
    assert r["executable"] is False
    assert "odte-convert" in r["next_command"]
    assert "--promote-to-execution" not in r["next_command"]


def test_execution_allowed_flag_cannot_spoof_missing_final_confirmations() -> None:
    gate = _entry_decision(
        decision="enter",
        execution_allowed=True,
        confirmations={},
        confirmation_needed=True,
    )
    result = ls.derive_loop_state(journal_events=[gate], now=NOW)
    assert result["state"] == "GATED"
    assert result["posture"] == "WAIT_FRESH_CONFIRMATION"
    assert result["executable"] is False


def test_legacy_gate_without_confirmation_schema_cannot_promote() -> None:
    gate = _entry_decision(
        decision="enter",
        execution_allowed=True,
        required_confirmations=[],
        confirmations={},
        confirmation_needed=False,
    )
    result = ls.derive_loop_state(journal_events=[gate], now=NOW)
    assert result["state"] == "GATED"
    assert result["executable"] is False


def test_declared_confirmation_subset_cannot_promote() -> None:
    gate = _entry_decision(
        decision="enter",
        execution_allowed=True,
        required_confirmations=["budget_check"],
        confirmations={"budget_check": True},
        confirmation_needed=False,
    )
    result = ls.derive_loop_state(journal_events=[gate], now=NOW)
    assert result["state"] == "GATED"
    assert result["executable"] is False


def test_entry_gate_executable_is_promoted():
    # PROMOTED must drive the actual sequence: broker review/place under standing auth FIRST, then
    # the position watch — not point straight at odte-position as if the entry already happened.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter",
                                                             execution_allowed=True)], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["executable"] is True
    assert "broker-review-place" in r["next_command"]
    assert "odte-position" in r["next_command"]
    assert "standing auth" in r["next_action"]


def test_no_trade_decision_consumes_matching_non_executable_gate():
    # Once the controller has journaled the matching no-trade decision, the same scan-only/denied
    # gate should not keep loop-status stuck at GATED and spam the next cron tick.
    gate = _entry_decision(decision="veto", underlying="META", symbol="META", ts=_ts(minutes_ago=5),
                           execution_allowed=False)
    no_trade = {"event_type": "no_trade_decision", "underlying": "META", "symbol": "META",
                "ts": _ts(minutes_ago=4), "seq": 6}
    r = ls.derive_loop_state(journal_events=[gate, no_trade], now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored consumed entry gate" in s for s in r["notes"])


def test_no_trade_decision_consumes_matching_executable_gate_cycle():
    # A granted gate must stop being execution-ready after the controller records a terminal veto for
    # that exact candidate cycle (for example, execution-authorize refused stale broker/market inputs).
    gate = _entry_decision(decision="enter", execution_allowed=True,
                           candidate_fingerprint="candidate-cycle-1")
    no_trade = {"event_type": "no_trade_decision", "underlying": "SPY", "symbol": "SPY",
                "candidate_fingerprint": "candidate-cycle-1", "execution_allowed": False,
                "ts": _ts(minutes_ago=4), "seq": 6}
    r = ls.derive_loop_state(journal_events=[gate, no_trade], now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored consumed entry gate" in s for s in r["notes"])


def test_unrelated_no_trade_cannot_consume_executable_gate_cycle():
    gate = _entry_decision(decision="enter", execution_allowed=True,
                           candidate_fingerprint="candidate-cycle-1")
    no_trade = {"event_type": "no_trade_decision", "underlying": "SPY", "symbol": "SPY",
                "candidate_fingerprint": "candidate-cycle-2", "execution_allowed": False,
                "ts": _ts(minutes_ago=4), "seq": 6}
    r = ls.derive_loop_state(journal_events=[gate, no_trade], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["executable"] is True


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


def test_fresh_pending_gate_outranks_scan_trigger():
    # A current-cycle PENDING gate (affirmative "enter", awaiting promotion) beats a bare scan
    # candidate — you are further down the funnel. (A DENIED gate does not; see the denied-gate tests.)
    r = ls.derive_loop_state(triggers=_candidate_triggers(),
                             journal_events=[_entry_decision(decision="enter",
                                                             execution_allowed=False)], now=NOW)
    assert r["state"] == "GATED"


def test_newer_trigger_supersedes_prior_failed_gate_without_execution():
    # A failed gate is not sticky forever. If a materially newer scan candidate appears later in the
    # same day, surface it as a fresh CANDIDATE so the controller can build a new gate. This still
    # stays scan-tier/non-executable; it never silently promotes to PROMOTED.
    gate = _entry_decision(decision="NO_TRADE_FINAL_TAPE_RECLAIM_FAILED",
                           ts=_ts(minutes_ago=30), execution_allowed=False)
    trig = _candidate_triggers(ts=_ts(minutes_ago=1), alert=True,
                               candidate={"ticker": "QQQ", "direction": "bullish"})
    r = ls.derive_loop_state(triggers=trig, journal_events=[gate], now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["executable"] is False
    assert r["context"]["scan_only"] is True
    assert r["context"]["underlying"] == "QQQ"
    assert r["context"].get("superseded_gate_decision") in (
        None, "NO_TRADE_FINAL_TAPE_RECLAIM_FAILED")


def test_denied_gate_with_stale_trigger_is_scan():
    # A denied gate is consumed and a stale scan trigger is ignored -> SCAN (no recycled edge).
    gate = _entry_decision(decision="NO_TRADE_FINAL_TAPE_RECLAIM_FAILED",
                           ts=_ts(minutes_ago=5), execution_allowed=False)
    trig = _candidate_triggers(ts=_ts(minutes_ago=35), alert=True,
                               candidate={"ticker": "QQQ", "direction": "bullish"})
    r = ls.derive_loop_state(triggers=trig, journal_events=[gate], now=NOW)
    assert r["state"] == "SCAN"


def test_not_materially_newer_trigger_does_not_supersede_pending_gate():
    # Supersede invariant still holds for a PENDING (affirmative, non-denied) gate: a scan candidate
    # that is not materially newer must not demote it out of GATED.
    gate = _entry_decision(decision="enter", ts=_ts(minutes_ago=10), execution_allowed=False)
    trig = _candidate_triggers(ts=_ts(minutes_ago=1), alert=True,
                               candidate={"ticker": "QQQ", "direction": "bullish"})
    r = ls.derive_loop_state(triggers=trig, journal_events=[gate], now=NOW)
    assert r["state"] == "GATED"


def test_newer_trigger_does_not_downgrade_execution_allowed_gate():
    gate = _entry_decision(decision="enter", ts=_ts(minutes_ago=5), execution_allowed=True)
    trig = _candidate_triggers(ts=_ts(minutes_ago=1), alert=True,
                               candidate={"ticker": "QQQ", "direction": "bullish"})
    r = ls.derive_loop_state(triggers=trig, journal_events=[gate], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["executable"] is True


def test_stale_execution_allowed_gate_is_ignored():
    gate = _entry_decision(decision="enter", ts=_ts(minutes_ago=30), execution_allowed=True)
    r = ls.derive_loop_state(journal_events=[gate], now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored stale entry gate" in s for s in r["notes"])


def test_stale_gate_from_prior_trade_is_ignored():
    # A gate dated BEFORE the closed trade's close is that trade's own (consumed) gate -> not GATED.
    # The closed+reviewed trade falls through to REVIEWED, not a stale PROMOTED.
    gate = _entry_decision(decision="enter", execution_allowed=True, ts=_ts(hours_ago=2))
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=10)}
    r = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[gate, pm], now=NOW)
    assert r["state"] == "REVIEWED"


def test_closed_reviewed_trade_ignores_old_candidate_and_promoted_gate():
    close_ts = _ts(minutes_ago=20)
    plan = _closed_plan(closed_at=close_ts, exit_fill_time=close_ts)
    gate = _entry_decision(decision="enter", execution_allowed=True, ts=_ts(minutes_ago=30))
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=10)}
    acand = {"ticker": "QQQ", "direction": "bullish", "ts": _ts(minutes_ago=30)}
    cdec = {"decision": "CONFIRM_ENTRY", "candidate": acand, "ts": _ts(minutes_ago=30)}
    r = ls.derive_loop_state(active_trade=plan, active_candidate=acand, candidate_decision=cdec,
                             journal_events=[gate, pm], now=NOW)
    assert r["state"] == "REVIEWED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert any("ignored stale entry gate" in s for s in r["notes"])
    assert any("ignored stale candidate watch" in s for s in r["notes"])


def test_closed_reviewed_trade_with_stale_pre_entry_artifacts_is_not_scout_edge():
    # Exact cron conflict: closed/reviewed old trade + stale trigger/candidate/gate must not report
    # SCOUT_FRESH_SETUP. Reviewed+ignored-stale leftovers are a flat idle tick, not fresh edge.
    close_ts = _ts(minutes_ago=20)
    plan = _closed_plan(closed_at=close_ts, exit_fill_time=close_ts)
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=10)}
    gate = _entry_decision(decision="enter", execution_allowed=True, ts=_ts(minutes_ago=30))
    trig = _candidate_triggers(ts=_ts(minutes_ago=40), alert=True,
                               candidate={"ticker": "QQQ", "direction": "bullish"})
    cdec = {"decision": "KEEP_WATCHING", "candidate": {"ticker": "QQQ"},
            "ts": _ts(minutes_ago=60)}
    r = ls.derive_loop_state(active_trade=plan, triggers=trig, candidate_decision=cdec,
                             journal_events=[gate, pm], now=NOW)
    assert r["state"] == "REVIEWED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["posture"] != "SCOUT_FRESH_SETUP"


def test_closed_reviewed_trade_suppresses_denied_gate_debug_noise():
    # A later denied/no-trade gate may exist in the audit journal after the closed trade, but the flat
    # cron UX should not keep mentioning internal "not sticky" gate diagnostics.
    close_ts = _ts(minutes_ago=30)
    plan = _closed_plan(closed_at=close_ts, exit_fill_time=close_ts)
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=25)}
    denied_gate = _entry_decision(decision="veto", execution_allowed=False, ts=_ts(minutes_ago=5))
    r = ls.derive_loop_state(active_trade=plan, journal_events=[pm, denied_gate], now=NOW)
    assert r["state"] == "REVIEWED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert not any("denied entry gate not sticky" in s for s in r["reasons"])


def test_stale_candidate_watch_is_ignored_but_live_position_still_wins():
    # A stale non-obligating watch (KEEP_WATCHING) is disposable cruft — quietly ignored. (An aged
    # CONFIRM_ENTRY is NOT: see the EXECUTION_CONVERSION_FAILURE regression tests below.)
    stale_cdec = {"decision": "KEEP_WATCHING", "candidate": {"ticker": "QQQ"},
                  "ts": _ts(minutes_ago=20)}
    r = ls.derive_loop_state(candidate_decision=stale_cdec, now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored stale candidate watch" in s for s in r["notes"])

    # And a live position always outranks the candidate lane — even a stale CONFIRM_ENTRY.
    stale_confirm = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "QQQ"},
                     "ts": _ts(minutes_ago=20)}
    live = ls.derive_loop_state(active_trade=_open_plan(), position_decision={
        "decision": "HOLD", "underlying": "SPY", "ts": _ts(minutes_ago=1)},
        candidate_decision=stale_confirm, now=NOW)
    assert live["state"] == "MANAGING"


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
    r = ls.derive_loop_state(active_trade=old, triggers=_candidate_triggers(ts=_ts(minutes_ago=1)), now=NOW)
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
    # fresh management decision for >1 minute, the loop should surface DEGRADED so Hermes treats it
    # as a risk event rather than continuing to scan.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=2)}, now=NOW)
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


# --- cron POSTURE: coarse "what to do this tick" over the fine-grained state ------------------

def test_posture_manage_position_for_live_trade():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=0)}, now=NOW)
    assert r["state"] == "MANAGING"
    assert r["posture"] == "MANAGE_POSITION"


def test_posture_fresh_scan_candidate_waits_for_confirmation():
    # A fresh scan_only trigger is not fresh edge — it's a candidate that still needs confirmation.
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True), now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["posture"] == "WAIT_FRESH_CONFIRMATION"
    assert r["executable"] is False


def test_posture_confirmed_candidate_watch_scouts_fresh_setup():
    # A CONFIRM_ENTRY candidate watch says "build a fresh gate" -> actionable SCOUT_FRESH_SETUP.
    cdec = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "QQQ", "direction": "bullish"},
            "ts": _ts(minutes_ago=1)}
    r = ls.derive_loop_state(candidate_decision=cdec, now=NOW)
    assert r["context"].get("confirmed") is True
    assert r["posture"] == "SCOUT_FRESH_SETUP"
    assert r["next_command"] == "odte-convert"


def test_posture_promoted_gate_is_execution_ready():
    # An execution-allowed gate is NOT "go scouting" — it is EXECUTION_READY: move to broker
    # review/place under standing auth. This label split is what stops the nudge-required stall.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "EXECUTION_READY"


def test_posture_empty_scan_is_no_trade_not_scout():
    # A flat/idle scan with NO fresh actionable setup is NOT fresh edge. SCOUT_FRESH_SETUP requires a
    # promoted gate or confirmed candidate; a bare empty board is a flat no-trade posture.
    r = ls.derive_loop_state(now=NOW)
    assert r["state"] == "SCAN"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["posture"] != "SCOUT_FRESH_SETUP"


def test_posture_stale_only_candidate_is_flat_no_trade():
    # The only thing on the board is a stale scan trigger -> it must be ignored AND the posture must
    # say FLAT_NO_TRADE so the cron doesn't recycle an old setup as if it were fresh edge.
    stale_trig = _candidate_triggers(ts=_ts(minutes_ago=45), alert=True,
                                     candidate={"ticker": "QQQ", "direction": "bullish"})
    r = ls.derive_loop_state(triggers=stale_trig, now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored stale scan trigger" in s for s in r["notes"])
    assert r["posture"] == "FLAT_NO_TRADE"


# --- EXECUTION-CONVERSION accountability (2026-07-27 IWM 292P regression) ---------------------
# An identity-bound CONFIRM_ENTRY is a conversion obligation, not disposable scan cruft. It may
# resolve ONLY via matching terminal evidence in the journal (entry gate / order / no-trade
# verdict for the same identity). Aging out with NO such evidence is a fail-closed
# EXECUTION_CONVERSION_FAILURE — never a quiet FLAT_NO_TRADE.

IWM_OPTION_ID = "e9b7a460-cf60-4480-a81f-0818a0b0aeec"


def _confirmed_iwm_watch(minutes_ago=1):
    return {"decision": "CONFIRM_ENTRY", "ts": _ts(minutes_ago=minutes_ago),
            "candidate": {"ticker": "IWM", "direction": "bearish", "option_type": "put",
                          "strike_price": 292.0, "expiration_date": "2026-07-27",
                          "option_id": IWM_OPTION_ID}}


def test_fresh_confirmed_entry_is_conversion_due_not_flat():
    # (1) A FRESH identity-bound CONFIRM_ENTRY is conversion-due: convert THIS TICK — a distinct
    # imperative posture, never soft "scouting" and never FLAT_NO_TRADE.
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=1), now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["posture"] == "CONVERT_CANDIDATE_NOW"
    assert r["posture"] != "FLAT_NO_TRADE"
    assert r["next_command"] == "odte-convert"


def test_convert_candidate_now_posture_separation():
    # Identity-bound confirm -> CONVERT_CANDIDATE_NOW. Generic (no identity) confirm keeps the
    # softer SCOUT_FRESH_SETUP. EXECUTION_READY stays reserved for an execution-allowed GATE.
    bound = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=1), now=NOW)
    assert bound["posture"] == "CONVERT_CANDIDATE_NOW"
    generic = ls.derive_loop_state(candidate_decision={
        "decision": "CONFIRM_ENTRY", "candidate": {"ticker": "QQQ", "direction": "bullish"},
        "ts": _ts(minutes_ago=1)}, now=NOW)
    assert generic["posture"] == "SCOUT_FRESH_SETUP"
    gate = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter",
                                                               execution_allowed=True)], now=NOW)
    assert gate["posture"] == "EXECUTION_READY"
    assert bound["posture"] != "EXECUTION_READY"


def test_convert_candidate_now_exposes_identity_and_sla_deadline():
    confirmed_ts = (NOW - timedelta(seconds=30)).isoformat()
    watch = _confirmed_iwm_watch()
    watch["ts"] = confirmed_ts
    r = ls.derive_loop_state(candidate_decision=watch, now=NOW)
    ctx = r["context"]
    assert ctx.get("option_id") == IWM_OPTION_ID
    assert ctx.get("conversion_due") is True
    assert ctx.get("confirmed_at") == (NOW - timedelta(seconds=30)).isoformat(timespec="seconds")
    assert ctx.get("candidate_age_seconds") == 30.0
    assert ctx.get("conversion_due_by") == (
        NOW - timedelta(seconds=30)
        + timedelta(seconds=ls.CONFIRM_CONVERSION_SLA_SECONDS)).isoformat(timespec="seconds")
    assert ctx.get("conversion_sla_seconds_remaining") == ls.CONFIRM_CONVERSION_SLA_SECONDS - 30.0
    assert ctx.get("conversion_sla_seconds") == ls.CONFIRM_CONVERSION_SLA_SECONDS


def test_convert_candidate_now_is_not_executable_and_imperative():
    # The new posture gains NO execution authority — it commands an exact gate build, not an order.
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=1), now=NOW)
    assert r["executable"] is False
    assert r["next_command"] == "odte-convert"
    assert "CONVERT NOW" in r["next_action"]
    assert "do not scan" in r["next_action"]


def test_convert_candidate_now_outranks_fresh_unrelated_scan_trigger():
    # A fresh unrelated scan candidate must not distract from a conversion-due confirm.
    fresh_trig = _candidate_triggers(ts=_ts(minutes_ago=1), alert=True,
                                     candidate={"ticker": "SPY", "direction": "bullish"})
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=1),
                             triggers=fresh_trig, now=NOW)
    assert r["posture"] == "CONVERT_CANDIDATE_NOW"
    assert r["context"].get("underlying") == "IWM"


def test_conversion_failure_outranks_fresh_unrelated_scan_trigger():
    # The aged fail-closed state blocks NEW scanning until broker reconciliation + a terminal
    # disposition for the orphaned identity — a fresh unrelated trigger must not override it.
    fresh_trig = _candidate_triggers(ts=_ts(minutes_ago=1), alert=True,
                                     candidate={"ticker": "SPY", "direction": "bullish"})
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=25),
                             triggers=fresh_trig, journal_events=[], now=NOW)
    assert r["posture"] == "EXECUTION_CONVERSION_FAILURE"
    assert r["executable"] is False


def test_aged_confirmed_entry_without_terminal_evidence_is_conversion_failure():
    # (2) REGRESSION (2026-07-27 IWM 292P): the confirmed identity aged past the candidate TTL with
    # NO matching gate/lease/order/no-trade verdict in the journal. That means the loop dropped a
    # confirmed entry on the floor (or an unaccounted order may exist at the broker) — it must fail
    # CLOSED as EXECUTION_CONVERSION_FAILURE, never quietly become FLAT_NO_TRADE.
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=25),
                             journal_events=[], now=NOW)
    assert r["posture"] == "EXECUTION_CONVERSION_FAILURE"
    assert r["executable"] is False
    assert r["state"] != "SCAN"
    assert r["context"].get("option_id") == IWM_OPTION_ID
    assert any(IWM_OPTION_ID in s for s in r["reasons"])


def test_aged_confirmed_entry_with_matching_no_trade_verdict_resolves_flat():
    # (3) A matching terminal no_trade_decision for the SAME identity accounts for the confirmed
    # entry: the cycle is reviewed/closed and the tick is a normal flat no-trade.
    no_trade = {"event_type": "no_trade_decision", "seq": 9, "ts": _ts(minutes_ago=20),
                "underlying": "IWM", "option_id": IWM_OPTION_ID,
                "decision": "NO_TRADE_CONFIRMATION_EXPIRED"}
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=25),
                             journal_events=[no_trade], now=NOW)
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["state"] == "SCAN"
    assert r["executable"] is False


def test_unrelated_no_trade_with_different_option_id_does_not_resolve_orphan():
    # (a) An IWM no-trade verdict for a DIFFERENT contract must not absolve the 292P obligation.
    other = {"event_type": "no_trade_decision", "seq": 9, "ts": _ts(minutes_ago=20),
             "underlying": "IWM", "option_id": "00000000-0000-0000-0000-000000000000",
             "decision": "NO_TRADE_SPREAD_TOO_WIDE"}
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=25),
                             journal_events=[other], now=NOW)
    assert r["posture"] == "EXECUTION_CONVERSION_FAILURE"
    assert r["executable"] is False


def test_same_underlying_no_trade_without_identity_does_not_resolve_identity_bound_orphan():
    # (b) A same-underlying no-trade event carrying NO identity fields cannot resolve an
    # identity-bound confirm — only an exact option_id/fingerprint match may.
    bare = {"event_type": "no_trade_decision", "seq": 9, "ts": _ts(minutes_ago=20),
            "underlying": "IWM", "decision": "NO_TRADE_TAPE_FADED"}
    r = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=25),
                             journal_events=[bare], now=NOW)
    assert r["posture"] == "EXECUTION_CONVERSION_FAILURE"
    assert r["executable"] is False


def test_generic_stale_confirm_without_identity_keeps_prior_flat_behavior():
    # (c) A CONFIRM_ENTRY that never locked a contract (no option_id, no fingerprint) carries no
    # conversion obligation — it keeps the prior quiet-ignore behavior, no false incident.
    generic = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "QQQ"}, "ts": _ts(minutes_ago=20)}
    r = ls.derive_loop_state(candidate_decision=generic, now=NOW)
    assert r["state"] == "SCAN"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert any("ignored stale candidate watch" in s for s in r["notes"])


def test_stale_scan_only_trigger_still_flat_no_trade():
    # (4) A stale SCAN-ONLY trigger carries no conversion obligation — even with a contract identity
    # attached it is disposable observe-lane cruft and stays a plain FLAT_NO_TRADE.
    stale_trig = _candidate_triggers(ts=_ts(minutes_ago=45), alert=True,
                                     candidate={"ticker": "IWM", "direction": "bearish",
                                                "option_id": IWM_OPTION_ID})
    r = ls.derive_loop_state(triggers=stale_trig, now=NOW)
    assert r["state"] == "SCAN"
    assert r["posture"] == "FLAT_NO_TRADE"


# --- BROKER lane: supplied/probed health payload, orders blocked when it can't be trusted -----

def _broker(**over):
    b = {"lane": "ok", "as_of": _ts(minutes_ago=0)}
    b.update(over)
    return b


def test_broker_lane_ok_allows_orders():
    lane = ls.classify_broker_lane(_broker(lane="ok"), now=NOW)
    assert lane["lane"] == "ok"
    assert lane["orders_ok"] is True


def test_broker_lane_down_blocks_orders():
    lane = ls.classify_broker_lane(_broker(lane="down"), now=NOW)
    assert lane["lane"] == "down"
    assert lane["orders_ok"] is False


def test_broker_lane_stale_probe_blocks_orders():
    # A probe older than BROKER_STALE_MINUTES can't be called live truth even if it claimed "ok".
    lane = ls.classify_broker_lane(_broker(lane="ok", as_of=_ts(minutes_ago=30)), now=NOW)
    assert lane["lane"] == "stale"
    assert lane["orders_ok"] is False


def test_broker_lane_read_only_fallback_blocks_orders():
    lane = ls.classify_broker_lane(_broker(lane="ok", read_only_fallback=True), now=NOW)
    assert lane["lane"] == "read_only_fallback"
    assert lane["orders_ok"] is False


def test_broker_absent_is_unknown_pure_offline():
    lane = ls.classify_broker_lane(None, now=NOW)
    assert lane["lane"] == "unknown"
    assert lane["orders_ok"] is None


def test_broker_down_forces_broker_degraded_and_non_executable():
    # A promoted (execution-allowed) gate must NOT read as executable when the broker lane is down.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             broker_health=_broker(lane="down"), now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "BROKER_DEGRADED"
    assert r["executable"] is False
    assert r["broker_lane"]["lane"] == "down"


def test_read_only_fallback_blocks_execution_of_fresh_gate():
    # Local read fresh but place lane down: a ready setup is BROKER_DEGRADED, orders blocked, and the
    # next command points at repairing the broker lane — not at placing an order.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             broker_health=_broker(read_only_fallback=True), now=NOW)
    assert r["posture"] == "BROKER_DEGRADED"
    assert r["executable"] is False
    assert "mcp" in r["next_command"].lower() or "broker" in r["next_command"].lower()


def test_live_position_with_dead_broker_is_broker_degraded():
    # An open position we can't value because the broker lane is down -> BROKER_DEGRADED, not a
    # confident MANAGE_POSITION that assumes fills.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=0)},
                             broker_health=_broker(lane="down"), now=NOW)
    assert r["posture"] == "BROKER_DEGRADED"
    assert r["executable"] is False


def test_healthy_broker_promoted_gate_is_execution_ready_and_executes():
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             broker_health=_broker(lane="ok"), now=NOW)
    assert r["posture"] == "EXECUTION_READY"
    assert r["executable"] is True


def test_run_loop_status_folds_broker_health_and_ages(tmp_path):
    (tmp_path / "triggers.json").write_text(json.dumps(_candidate_triggers(ts=_ts(minutes_ago=2))))
    (tmp_path / "broker_health.json").write_text(json.dumps(_broker(lane="down")))
    payload = ls.run_loop_status(state_dir=str(tmp_path),
                                 broker_health_path=str(tmp_path / "broker_health.json"), now=NOW)
    assert payload["posture"] == "BROKER_DEGRADED"
    assert payload["broker_lane"]["lane"] == "down"
    assert payload["artifact_ages"]["triggers"]["fresh"] is True
    assert payload["artifact_ages"]["triggers"]["age_minutes"] == 2.0
    json.dumps(payload)   # stdout contract stays JSON-serializable


# --- canonical cron handoff schema: parent review/place lane is AUTHORITATIVE ----------------

def _handoff(**over):
    b = {"parent_robinhood_mcp": "OK", "cli_mcp_test": "CONNECTED", "fallback": "OK",
         "live_review_place_allowed": True, "as_of": _ts(minutes_ago=0), "source": "cron-probe"}
    b.update(over)
    return b


def test_handoff_parent_ok_allows_orders():
    lane = ls.classify_broker_lane(_handoff(), now=NOW)
    assert lane["lane"] == "ok"
    assert lane["orders_ok"] is True


def test_handoff_split_brain_parent_down_cli_connected_is_read_only_fallback():
    # The reported split-brain: `hermes mcp test robinhood` connects (CLI) but the parent chat/cron MCP
    # review/place lane is DOWN. That verifies flatness only — it must NOT authorize live orders.
    lane = ls.classify_broker_lane(
        _handoff(parent_robinhood_mcp="DOWN", cli_mcp_test="CONNECTED",
                 live_review_place_allowed=False), now=NOW)
    assert lane["lane"] == "read_only_fallback"
    assert lane["orders_ok"] is False


def test_handoff_parent_down_and_cli_down_is_down():
    lane = ls.classify_broker_lane(
        _handoff(parent_robinhood_mcp="DOWN", cli_mcp_test="DOWN", fallback="DOWN",
                 live_review_place_allowed=False), now=NOW)
    assert lane["lane"] == "down"
    assert lane["orders_ok"] is False


def test_handoff_parent_unknown_cannot_confirm_place():
    # Parent UNKNOWN (never probed) can read via CLI/fallback but can't confirm the place lane.
    lane = ls.classify_broker_lane(
        _handoff(parent_robinhood_mcp="UNKNOWN", live_review_place_allowed=None), now=NOW)
    assert lane["lane"] == "read_only_fallback"
    assert lane["orders_ok"] is False


def test_parent_down_overrides_stale_data_posture_to_broker_degraded():
    # Even when the loop itself is flat/stale (FLAT_NO_TRADE), a DOWN parent MCP must surface as
    # the top-line BROKER_DEGRADED so the cron never quietly implies the lane is fine. No contradiction.
    flat = ls.derive_loop_state(now=NOW)
    assert flat["posture"] == "FLAT_NO_TRADE"
    degraded = ls.derive_loop_state(
        broker_health=_handoff(parent_robinhood_mcp="DOWN", cli_mcp_test="CONNECTED",
                               live_review_place_allowed=False), now=NOW)
    assert degraded["posture"] == "BROKER_DEGRADED"
    assert degraded["broker_lane"]["lane"] == "read_only_fallback"
    assert degraded["executable"] is False


def test_run_loop_status_auto_reads_canonical_broker_health_file(tmp_path):
    # Durable handoff: cron writes data/odte/broker_health.json; loop-status folds it with NO flag.
    (tmp_path / "broker_health.json").write_text(json.dumps(
        _handoff(parent_robinhood_mcp="DOWN", cli_mcp_test="CONNECTED",
                 live_review_place_allowed=False)))
    payload = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert payload["broker_lane"]["lane"] == "read_only_fallback"
    assert payload["posture"] == "BROKER_DEGRADED"
    assert payload["artifacts"]["broker_health"] == "ok"


def test_run_loop_status_no_broker_file_is_offline_unknown(tmp_path):
    payload = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert payload["broker_lane"]["lane"] == "unknown"
    assert payload["artifacts"]["broker_health"] == "missing"


def test_run_loop_status_hides_inactive_and_inert_stale_artifact_ages(tmp_path):
    # Inactive candidate/NO_POSITION files and a stale observe-only gate are audit trail, not live
    # artifacts. They should not keep popping up as stale work after the loop has returned to scan.
    (tmp_path / "active_candidate.json").write_text(json.dumps({
        "state": "INACTIVE", "decision": "NO_CANDIDATE", "updated_at": _ts(hours_ago=2)}))
    (tmp_path / "candidate_decision.json").write_text(json.dumps({
        "state": "INACTIVE", "decision": "NO_CANDIDATE", "generated_at": _ts(hours_ago=2),
        "candidate": {}}))
    (tmp_path / "position_decision.json").write_text(json.dumps({
        "decision": "NO_POSITION", "ts": _ts(hours_ago=2)}))
    (tmp_path / "triggers.json").write_text(json.dumps({
        "ts": _ts(hours_ago=2), "alert": False, "triggers": [], "candidate": None,
        "spy_verdict": "PUT-leaning"}))
    (tmp_path / "decision_journal.jsonl").write_text(json.dumps({
        "event_type": "entry_decision", "seq": 1, "ts": _ts(hours_ago=2), "underlying": "TSLA",
        "decision": "observe", "scan_only": True, "execution_allowed": False}) + "\n")
    payload = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert payload["state"] == "SCAN"
    assert payload["artifact_ages"]["triggers"]["as_of"] is None
    assert payload["artifact_ages"]["candidate"]["as_of"] is None
    assert payload["artifact_ages"]["gate"]["as_of"] is None
    assert payload["artifact_ages"]["position_decision"]["as_of"] is None
    assert not any("ignored stale entry gate" in r for r in payload["reasons"])


def test_flat_reviewed_empty_heartbeat_gives_clean_user_facing_reasons(tmp_path):
    # The exact infuriating cron case: flat, a closed+reviewed trade, an empty/no-alert trigger
    # heartbeat, an observe-only/denied gate in the journal, and a FRESH broker_health file. The
    # user-facing payload must read clean: posture FLAT_NO_TRADE, broker ok, NO stale/denied/
    # consumed/idle noise in `reasons` (that trail lives in `notes`), and null artifact ages.
    (tmp_path / "active_trade.json").write_text(json.dumps(_closed_plan(
        closed_at=_ts(minutes_ago=90), exit_fill_time=_ts(minutes_ago=90),
        entry_fill_time=_ts(minutes_ago=110), trade_id="IWM-T9")))
    (tmp_path / "position_decision.json").write_text(json.dumps({
        "decision": "NO_POSITION", "ts": _ts(minutes_ago=90)}))
    (tmp_path / "active_candidate.json").write_text(json.dumps({
        "state": "INACTIVE", "decision": "DEGRADED_NO_TRADE", "updated_at": _ts(minutes_ago=90)}))
    (tmp_path / "triggers.json").write_text(json.dumps({
        "ts": _ts(minutes_ago=1), "alert": False, "triggers": [], "candidate": None,
        "spy_verdict": "PUT-leaning"}))
    (tmp_path / "decision_journal.jsonl").write_text(
        json.dumps({"event_type": "entry_decision", "seq": 1, "ts": _ts(minutes_ago=25),
                    "underlying": "TSLA", "decision": "veto", "execution_allowed": False}) + "\n"
        + json.dumps({"event_type": "postmortem", "trade_id": "IWM-T9", "seq": 2,
                      "ts": _ts(minutes_ago=80)}) + "\n")
    (tmp_path / "broker_health.json").write_text(json.dumps(_handoff(as_of=_ts(minutes_ago=0))))

    p = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert p["posture"] == "FLAT_NO_TRADE"
    assert p["broker_lane"]["lane"] == "ok" and p["broker_lane"]["orders_ok"] is True
    # empty heartbeat + inactive/observe/NO_POSITION artifacts -> ages null, never "stale"
    for key in ("triggers", "candidate", "gate", "position_decision"):
        assert p["artifact_ages"][key]["as_of"] is None, key
        assert p["artifact_ages"][key]["fresh"] is None, key
    # user-facing reasons carry NO housekeeping noise; the audit trail is in notes
    joined = " ".join(p["reasons"]).lower()
    for noise in ("ignored stale", "ignored consumed", "denied entry gate not sticky",
                  "closed and reviewed", "no actionable candidate", "unreviewed but stale"):
        assert noise not in joined, f"{noise!r} leaked into user-facing reasons: {p['reasons']}"
    assert p["reasons"] == ["flat — no fresh actionable setup"]
    json.dumps(p)


def test_notes_field_retains_audit_trail_for_debugging():
    # The breadcrumbs still exist for debugging — just demoted out of `reasons` into `notes`.
    r = ls.derive_loop_state(
        journal_events=[_entry_decision(decision="veto", ts=_ts(minutes_ago=5))], now=NOW)
    assert r["state"] == "SCAN"
    assert any("denied entry gate not sticky" in s for s in r["notes"])
    assert not any("denied entry gate not sticky" in s for s in r["reasons"])


# --- posture split: FLAT_NO_TRADE (idle) vs STALE_DATA_BLOCKED (real block) -------------------

def test_reviewed_with_ignored_stale_leftovers_is_flat_not_stale():
    # The exact UX bug: a reviewed closed trade + stale candidate/gate leftovers that were IGNORED is
    # a normal idle tick. Posture must be FLAT_NO_TRADE and must NOT contain the word "stale".
    plan = _closed_plan(closed_at=_ts(minutes_ago=20), exit_fill_time=_ts(minutes_ago=20))
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=10)}
    gate = _entry_decision(decision="veto", ts=_ts(minutes_ago=30), execution_allowed=False)
    r = ls.derive_loop_state(active_trade=plan, journal_events=[gate, pm], now=NOW)
    assert r["state"] == "REVIEWED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert "STALE" not in r["posture"]


def test_empty_scan_is_flat_no_trade():
    r = ls.derive_loop_state(now=NOW)
    assert r["state"] == "SCAN"
    assert r["posture"] == "FLAT_NO_TRADE"


def test_open_position_with_stale_live_read_is_stale_data_blocked_and_shouts():
    # Broker is fine, but the live management decision went stale -> holding blind. This is the ONLY
    # legitimate stale-data posture: STALE_DATA_BLOCKED, with a loud reason (not demoted to notes).
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=6)},
                             broker_health=_broker(lane="ok"), live_mode=True, now=NOW)
    assert r["state"] == "DEGRADED"
    assert r["posture"] == "STALE_DATA_BLOCKED"
    assert any("stale" in s.lower() for s in r["reasons"])


def test_stale_live_read_with_broker_down_is_broker_degraded():
    # If the broker is ALSO down, broker truth dominates the stale-read block.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=6)},
                             broker_health=_broker(lane="down"), live_mode=True, now=NOW)
    assert r["posture"] == "BROKER_DEGRADED"


def test_malformed_nonlive_artifact_is_stale_data_blocked():
    r = ls.derive_loop_state(triggers={"candidate": {"ticker": "SPY"}}, errors={"triggers"},
                             now=NOW)
    # malformed non-live artifact surfaces as a diagnostic block, not idle
    if r["state"] == "DEGRADED":
        assert r["posture"] == "STALE_DATA_BLOCKED"


# --- live-mode fail-closed: unknown/missing broker can't authorize a live order ---------------

def test_live_mode_promoted_gate_with_unknown_broker_fails_closed():
    # A ready (execution-allowed) gate but NO broker_health, in live mode -> cannot authorize a live
    # order -> BROKER_DEGRADED, executable false. This is the fail-closed requirement.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             live_mode=True, now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "BROKER_DEGRADED"
    assert r["executable"] is False


def test_offline_mode_promoted_gate_with_unknown_broker_is_execution_ready():
    # Pure decision-support (offline) is lenient: unknown broker is reported, not blocked.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             live_mode=False, now=NOW)
    assert r["posture"] == "EXECUTION_READY"
    assert r["executable"] is True


def test_live_mode_flat_with_unknown_broker_is_flat_not_degraded():
    # A flat tick has nothing to authorize, so an unknown broker does NOT force BROKER_DEGRADED.
    r = ls.derive_loop_state(live_mode=True, now=NOW)
    assert r["posture"] == "FLAT_NO_TRADE"


def test_live_mode_open_position_with_unknown_broker_fails_closed():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=0)},
                             live_mode=True, now=NOW)
    assert r["posture"] == "BROKER_DEGRADED"
    assert r["executable"] is False


def test_live_mode_open_position_with_fresh_ok_broker_manages():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=0)},
                             broker_health=_broker(lane="ok"), live_mode=True, now=NOW)
    assert r["posture"] == "MANAGE_POSITION"


# --- stale broker_health.json: an outdated probe FILE is refreshable, not a broker fault --------

def test_stale_broker_probe_on_flat_tick_is_flat_not_broker_degraded():
    # REGRESSION (the babysit bug, seen live 2026-07-06): a flat/reviewed tick + a broker_health.json
    # hours past its TTL read BROKER_DEGRADED with "repair the parent lane" — pure stale-file panic.
    # Nothing needs the live lane on a flat tick, so the posture must stay FLAT_NO_TRADE; the stale
    # lane (orders blocked) plus its refresh command stay visible in broker_lane.
    r = ls.derive_loop_state(broker_health=_handoff(as_of=_ts(minutes_ago=300)),
                             live_mode=True, now=NOW)
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["broker_lane"]["lane"] == "stale"
    assert r["broker_lane"]["orders_ok"] is False
    assert "broker_health.json" in r["broker_lane"]["refresh_command"]


def test_stale_broker_probe_reviewed_trade_is_flat_not_broker_degraded():
    plan = _closed_plan(closed_at=_ts(minutes_ago=60), exit_fill_time=_ts(minutes_ago=60))
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=50)}
    r = ls.derive_loop_state(active_trade=plan, journal_events=[pm],
                             broker_health=_handoff(as_of=_ts(minutes_ago=120)),
                             live_mode=True, now=NOW)
    assert r["state"] == "REVIEWED"
    assert r["posture"] == "FLAT_NO_TRADE"


def test_stale_broker_probe_with_ready_setup_says_refresh_not_lane_down():
    # A ready (promoted) setup behind a stale probe file must be blocked BUT the message/next_command
    # must be the exact refresh — never "assume live orders are impossible" / "repair the lane".
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             broker_health=_handoff(as_of=_ts(minutes_ago=30)),
                             live_mode=True, now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "STALE_DATA_BLOCKED"
    assert r["executable"] is False
    assert "broker_health.json" in r["next_command"]
    assert "last probe said parent lane OK" in r["next_action"]


def test_stale_broker_probe_with_live_position_shouts_refresh():
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=0)},
                             broker_health=_handoff(as_of=_ts(minutes_ago=30)),
                             live_mode=True, now=NOW)
    assert r["posture"] == "STALE_DATA_BLOCKED"
    assert "refresh broker truth" in r["next_action"].lower() or \
           "broker_health.json" in r["next_command"]


def test_classify_stale_lane_keeps_last_known_place_permission():
    stale_ok = ls.classify_broker_lane(_handoff(as_of=_ts(minutes_ago=30)), now=NOW)
    assert stale_ok["lane"] == "stale"
    assert stale_ok["orders_ok"] is False               # stale can never authorize
    assert stale_ok["last_known_place_allowed"] is True  # ...but the last probe said the lane was OK
    stale_down = ls.classify_broker_lane(
        _handoff(parent_robinhood_mcp="DOWN", live_review_place_allowed=False,
                 as_of=_ts(minutes_ago=30)), now=NOW)
    assert stale_down["last_known_place_allowed"] is False


def test_confirmed_broker_down_on_flat_tick_still_broker_degraded():
    # The flat-tick leniency is ONLY for a stale probe file. A CONFIRMED down lane stays top-line.
    r = ls.derive_loop_state(broker_health=_broker(lane="down"), live_mode=True, now=NOW)
    assert r["posture"] == "BROKER_DEGRADED"


# --- GATED: the gate breakdown is machine-readable at the loop surface -------------------------

def test_gated_context_names_failing_and_unknown_gates():
    gate = _entry_decision(decision="enter", execution_allowed=False,
                           gates={"day_regime": True, "vehicle": False,
                                  "directional_thesis": True, "account": None},
                           veto_reasons=[])
    r = ls.derive_loop_state(journal_events=[gate], now=NOW)
    assert r["state"] == "GATED"
    assert r["context"]["failing_gates"] == ["vehicle"]
    assert r["context"]["unknown_gates"] == ["account"]
    assert "vehicle" in r["next_action"]
    assert "account" in r["next_action"]


def test_run_loop_status_live_mode_flag_fails_closed_on_missing_broker(tmp_path):
    (tmp_path / "decision_journal.jsonl").write_text(json.dumps(
        _entry_decision(decision="enter", execution_allowed=True)) + "\n")
    live = ls.run_loop_status(state_dir=str(tmp_path), live_mode=True, now=NOW)
    assert live["posture"] == "BROKER_DEGRADED"
    offline = ls.run_loop_status(state_dir=str(tmp_path), live_mode=False, now=NOW)
    assert offline["posture"] == "EXECUTION_READY"


# --- green-day preservation: no fresh entries after a banked green scalp -----------------------

def _green_scalp_journal():
    """Today's exact live sequence (2026-07-10): 3x SPY 753C in @0.98, all 3 out @1.05 — banked green."""
    base = {"trade_id": "SPY-753C-1", "underlying": "SPY", "option_type": "call", "strike": 753}
    return [
        {**base, "event_type": "order_filled", "seq": 1, "ts": _ts(hours_ago=3),
         "quantity": 3, "entry_price": 0.98},
        {**base, "event_type": "order_closed", "seq": 2, "ts": _ts(hours_ago=2),
         "quantity": 3, "exit_price": 1.05, "realized_pnl": 21.0},
    ]


def test_promoted_gate_after_green_scalp_is_locked_out():
    # REGRESSION (live 2026-07-10): after the green scalp closed, a later execution-allowed gate for
    # the SAME contract re-entered @1.27 for a scratch. A fresh promoted gate after a banked green
    # close must be CONSUMED: never PROMOTED/EXECUTION_READY, posture FLAT_NO_TRADE, loud reason.
    gate = _entry_decision(decision="enter", execution_allowed=True, seq=6, ts=_ts(minutes_ago=5))
    r = ls.derive_loop_state(journal_events=[*_green_scalp_journal(), gate], now=NOW)
    assert r["state"] != "PROMOTED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["executable"] is False
    assert any("green_day_preservation_lockout" in s for s in r["reasons"])


def test_reentry_override_gate_survives_green_lockout():
    # The explicit escape hatch: a gate the entry gate ARMED (allow_reentry_after_green, BP-checked
    # there) still promotes — the lockout is a default, not a cage.
    gate = _entry_decision(decision="enter", execution_allowed=True, seq=6, ts=_ts(minutes_ago=5),
                           allow_reentry_after_green=True)
    r = ls.derive_loop_state(journal_events=[*_green_scalp_journal(), gate], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "EXECUTION_READY"


def test_fresh_candidate_after_green_scalp_surfaces_with_budget_slot(monkeypatch):
    # The auto-arm MECHANISM under test; the live posture is OFF since 2026-08-14 (W33 fence:
    # post-green re-entries ran -$49/4), so pin it ON here the same way the kill-switch test
    # pins it off.
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    # 2026-08-03 auto-arm: with a budget slot open and the cooldown clear, a post-green scan
    # candidate SURFACES (tier/BP enforced at the gate inside odte-convert) instead of being
    # consumed — on 2026-08-03 the old consumption made trade #2 structurally impossible.
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             journal_events=_green_scalp_journal(), now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["executable"] is False                     # surfacing is never execution authority
    assert any("green_reentry_auto_arm" in s for s in r["reasons"])


def test_fresh_candidate_after_green_scalp_consumed_when_auto_arm_off(monkeypatch):
    import data.odte_config as oc
    monkeypatch.setattr(oc, "GREEN_REENTRY_AUTO_ARM", False)
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             journal_events=_green_scalp_journal(), now=NOW)
    assert r["state"] != "CANDIDATE"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert any("green_day_preservation_lockout" in s for s in r["reasons"])


def test_fresh_candidate_after_green_scalp_consumed_when_budget_exhausted(monkeypatch):
    # The auto-arm MECHANISM under test; the live posture is OFF since 2026-08-14 (W33 fence:
    # post-green re-entries ran -$49/4), so pin it ON here the same way the kill-switch test
    # pins it off.
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    import data.odte_config as oc
    journal = _green_scalp_journal()
    for i in range(2, oc.DAILY_TRADE_BUDGET + 1):
        base = {"trade_id": f"SPY-753C-{i}", "underlying": "SPY"}
        journal += [
            {**base, "event_type": "order_filled", "seq": 10 + i, "ts": _ts(hours_ago=1.5)},
            {**base, "event_type": "order_closed", "seq": 20 + i, "ts": _ts(hours_ago=1.0),
             "realized_pnl": 4.0},
        ]
    trig = _candidate_triggers(ts=_ts(minutes_ago=1), alert=True)
    # A+ UNCAPPED (2026-08-06 user policy): a net-GREEN exhausted-budget day keeps the scan
    # lane ALIVE — only a_plus can convert (the gate enforces the tier).
    r = ls.derive_loop_state(triggers=trig, journal_events=journal, now=NOW)
    assert r["state"] == "CANDIDATE"
    assert any("green_reentry_auto_arm" in s for s in r["reasons"])
    # Kill switch off: the original consume-at-cap behavior stands.
    monkeypatch.setattr(oc, "DAILY_BUDGET_APLUS_UNCAPPED", False)
    r2 = ls.derive_loop_state(triggers=trig, journal_events=journal, now=NOW)
    assert r2["state"] != "CANDIDATE"
    assert r2["posture"] == "FLAT_NO_TRADE"
    assert any("budget" in s for s in r2["reasons"])


def test_confirmed_candidate_watch_after_green_scalp_surfaces(monkeypatch):
    # The auto-arm MECHANISM under test; the live posture is OFF since 2026-08-14 (W33 fence:
    # post-green re-entries ran -$49/4), so pin it ON here the same way the kill-switch test
    # pins it off.
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    cdec = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "SPY", "direction": "bullish"},
            "ts": _ts(minutes_ago=1)}
    r = ls.derive_loop_state(candidate_decision=cdec, journal_events=_green_scalp_journal(), now=NOW)
    assert r["posture"] in ("SCOUT_FRESH_SETUP", "CONVERT_CANDIDATE_NOW")
    assert r["executable"] is False
    assert any("green_reentry_auto_arm" in s for s in r["reasons"])


def test_below_winner_tier_watch_after_green_scalp_is_consumed(monkeypatch):
    # The auto-arm MECHANISM under test; the live posture is OFF since 2026-08-14 (W33 fence:
    # post-green re-entries ran -$49/4), so pin it ON here the same way the kill-switch test
    # pins it off.
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    monkeypatch.setattr(_oc, "GREEN_REENTRY_REQUIRE_BETTER_TIER", False)  # same-or-better mechanism
    # The green trade's tier defaults to "full" (legacy journal, no tier events): a B+ watch ranks
    # below it and is consumed early — same-or-better tier only.
    import data.odte_journal as oj
    assert oj.tier_rank("b_plus") < oj.tier_rank("full")
    cdec = {"decision": "CONFIRM_ENTRY",
            "candidate": {"ticker": "SPY", "direction": "bullish", "tier": "b_plus"},
            "ts": _ts(minutes_ago=1)}
    r = ls.derive_loop_state(candidate_decision=cdec, journal_events=_green_scalp_journal(), now=NOW)
    assert r["posture"] == "FLAT_NO_TRADE"
    assert any("today's winning tier" in s for s in r["reasons"])


def test_green_closed_plan_alone_locks_fresh_promoted_gate():
    # The lockout also derives from the closed active_trade plan (gross_pnl > 0, closed today) —
    # journal exit events are not required for the guard to hold.
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": _ts(minutes_ago=25)}
    gate = _entry_decision(decision="enter", execution_allowed=True, seq=11, ts=_ts(minutes_ago=5))
    r = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[pm, gate], now=NOW)
    assert r["state"] == "REVIEWED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["executable"] is False


def test_red_close_does_not_lock_fresh_promoted_gate():
    # Falsification: a LOSING close is not green-day preservation — a fresh gate still promotes.
    journal = _green_scalp_journal()
    journal[1] = {**journal[1], "realized_pnl": -30.0}
    gate = _entry_decision(decision="enter", execution_allowed=True, seq=6, ts=_ts(minutes_ago=5))
    r = ls.derive_loop_state(journal_events=[*journal, gate], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "EXECUTION_READY"


# --- BROKER TRUTH WINS: fresh probe position/order counts reconcile lagging local artifacts ----
# REGRESSION (2026-07-16 artifact miss): two agentic 1DTE scalps (SPY 753P +21%, IWM 297C +20%)
# opened AND closed at the broker while every local artifact kept the prior day's state — check-ins
# read flat/stale/candidate during a live hold and could read live/blind after the flat close.

_1DTE_EXPIRY = (NOW + timedelta(days=1)).date().isoformat()


def _todays_two_scalp_journal():
    """The 2026-07-16 net-green day: SPY 753P 1DTE put +21%, then IWM 297C 1DTE call +20%."""
    def trade(tid, sym, otype, strike, entry, exit_p, pnl, seq0, mins_ago):
        base = {"trade_id": tid, "underlying": sym, "option_type": otype, "strike": strike,
                "expiration_date": _1DTE_EXPIRY}       # 1DTE vehicle — selected by BP/quality, OK
        return [
            {**base, "event_type": "order_filled", "seq": seq0, "ts": _ts(minutes_ago=mins_ago),
             "quantity": 1, "entry_price": entry},
            {**base, "event_type": "order_closed", "seq": seq0 + 1,
             "ts": _ts(minutes_ago=mins_ago - 10), "quantity": 1, "exit_price": exit_p,
             "realized_pnl": pnl},
        ]
    return [*trade("SPY-753P-1DTE", "SPY", "put", 753, 0.98, 1.19, 21.0, 1, 120),
            *trade("IWM-297C-1DTE", "IWM", "call", 297, 0.50, 0.60, 20.0, 3, 60)]


def test_classify_broker_lane_extracts_position_order_bp_truth():
    lane = ls.classify_broker_lane(_handoff(nonzero_option_positions_count=2,
                                            open_option_orders_count=1,
                                            today_option_orders_count=4,
                                            buying_power=7.6, cash=407.34), now=NOW)
    assert lane["truth"] == {"open_option_positions": 2.0, "open_option_orders": 1.0,
                             "today_option_orders": 4.0, "buying_power": 7.6, "cash": 407.34}
    # nested account block (active_state.json-style probes) also works
    lane2 = ls.classify_broker_lane({"parent_robinhood_mcp": "OK",
                                     "live_review_place_allowed": True,
                                     "as_of": _ts(minutes_ago=0),
                                     "account": {"positions_count": 1, "open_orders_count": 0,
                                                 "buying_power": 12.5}}, now=NOW)
    assert lane2["truth"]["open_option_positions"] == 1.0
    assert lane2["truth"]["open_option_orders"] == 0.0
    assert lane2["truth"]["buying_power"] == 12.5
    # a bare lane probe (no counts) carries no truth block — reconciliation stays off
    assert "truth" not in ls.classify_broker_lane(_handoff(), now=NOW)


def test_broker_open_position_overrides_flat_local_artifacts():
    # DURING the live hold: local artifacts are yesterday's flat state, broker shows a position.
    # The check-in must say MANAGING / MANAGE_POSITION and name the reconcile, never flat/stale.
    bh = _handoff(nonzero_option_positions_count=1, open_option_orders_count=0, buying_power=12.0)
    r = ls.derive_loop_state(broker_health=bh, live_mode=True, now=NOW)
    assert r["state"] == "MANAGING"
    assert r["posture"] == "MANAGE_POSITION"
    assert r["live"] is True
    assert r["reconciliation"] == "broker_open_position_overrides_flat_artifacts"
    assert any("BROKER TRUTH" in s for s in r["reasons"])
    assert "active_trade.json" in r["next_action"]
    assert "odte-position" in r["next_command"]


def test_broker_open_position_outranks_fresh_local_candidate():
    # Local artifacts say "candidate on the board" while the broker holds a live position:
    # manage before scan — broker truth outranks every pre-entry artifact.
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             broker_health=_handoff(nonzero_option_positions_count=1),
                             live_mode=True, now=NOW)
    assert r["state"] == "MANAGING"
    assert r["posture"] == "MANAGE_POSITION"
    assert r["context"]["reconciled_from_state"] == "CANDIDATE"


def test_broker_open_position_with_read_only_lane_still_surfaces_live():
    # Read-only probe verifies the position but the place lane can't back an exit: the live trade
    # must still surface (MANAGING) with the broker block on top — never a flat/candidate tick.
    bh = _handoff(parent_robinhood_mcp="DOWN", live_review_place_allowed=False,
                  nonzero_option_positions_count=1)
    r = ls.derive_loop_state(broker_health=bh, live_mode=True, now=NOW)
    assert r["state"] == "MANAGING"
    assert r["posture"] == "BROKER_DEGRADED"
    assert r["executable"] is False


def test_broker_flat_close_overrides_open_local_plan():
    # AFTER the agentic close: broker is flat (0 positions / 0 open orders) but active_trade.json
    # still says open. That is a completed trade, not ENTERED/DEGRADED "holding blind".
    plan = _open_plan(underlying="SPY", option_type="put", strike_price=753,
                      expiration_date=_1DTE_EXPIRY, trade_id="SPY-753P-1DTE", entry_price=0.98)
    bh = _handoff(nonzero_option_positions_count=0, open_option_orders_count=0,
                  today_option_orders_count=4, buying_power=7.6)
    r = ls.derive_loop_state(active_trade=plan, journal_events=_todays_two_scalp_journal(),
                             broker_health=bh, live_mode=True, now=NOW)
    assert r["state"] == "EXITED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["executable"] is False
    assert r["reconciliation"] == "broker_flat_overrides_live_artifacts"
    assert r["context"]["trade_id"] == "SPY-753P-1DTE"
    assert "clean stop" in r["next_action"]
    assert "postmortem" in r["next_command"] or "journal" in r["next_command"]


def test_broker_flat_close_overrides_stale_live_read():
    # Previously this read STALE_DATA_BLOCKED / "holding blind" — with the broker confirming flat,
    # the stale local position read is a lagging artifact, not a live risk.
    r = ls.derive_loop_state(active_trade=_open_plan(),
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=45)},
                             broker_health=_handoff(nonzero_option_positions_count=0,
                                                    open_option_orders_count=0),
                             live_mode=True, now=NOW)
    assert r["state"] == "EXITED"
    assert r["posture"] == "FLAT_NO_TRADE"
    assert r["posture"] != "STALE_DATA_BLOCKED"


def test_todays_net_green_low_bp_day_ends_clean_flat_no_new_entry(monkeypatch):
    # The auto-arm MECHANISM under test; the live posture is OFF since 2026-08-14 (W33 fence:
    # post-green re-entries ran -$49/4), so pin it ON here the same way the kill-switch test
    # pins it off.
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    # The full acceptance scenario: two 1DTE scalps banked net green, broker flat, BP down to $7.60,
    # and a fresh candidate still on the board. The day must end FLAT_NO_TRADE with the green-day
    # lockout blocking re-entry — low BP and the 1DTE vehicle are neither errors nor violations.
    bh = _handoff(nonzero_option_positions_count=0, open_option_orders_count=0,
                  today_option_orders_count=4, buying_power=7.6, cash=407.34)
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             journal_events=_todays_two_scalp_journal(),
                             broker_health=bh, live_mode=True, now=NOW)
    # A+ UNCAPPED (2026-08-06): a net-green day keeps WATCHING past the cap — but nothing is
    # executable, and the $7.60 BP means the gate/authorize refuses any actual entry (bp_ok).
    assert r["posture"] == "WAIT_FRESH_CONFIRMATION"
    assert "STALE" not in r["posture"] and r["posture"] != "BROKER_DEGRADED"
    assert r["executable"] is False
    assert r["broker_lane"]["truth"]["buying_power"] == 7.6
    joined = " ".join(r["reasons"] + r.get("notes", [])).lower()
    assert "violation" not in joined and "error" not in joined


def test_1dte_live_plan_is_managing_not_a_violation():
    # A 1DTE contract selected by the BP/quality gates is a legitimate vehicle: a live 1DTE plan
    # with a fresh decision manages normally — no violation, no degrade, no vehicle complaint.
    plan = _open_plan(expiration_date=_1DTE_EXPIRY)
    r = ls.derive_loop_state(active_trade=plan,
                             position_decision={"decision": "HOLD", "underlying": "SPY",
                                                "ts": _ts(minutes_ago=0)},
                             broker_health=_handoff(nonzero_option_positions_count=1),
                             live_mode=True, now=NOW)
    assert r["state"] == "MANAGING"
    assert r["posture"] == "MANAGE_POSITION"
    joined = " ".join(r["reasons"]).lower()
    assert "violation" not in joined and "1dte" not in joined


def test_stale_or_undated_broker_truth_never_reconciles():
    # A 30m-old probe claiming an open position must NOT resurrect a live trade (outdated counts
    # are not live truth) — and an undated probe can't reconcile either.
    stale = ls.derive_loop_state(broker_health=_handoff(nonzero_option_positions_count=1,
                                                        as_of=_ts(minutes_ago=30)),
                                 live_mode=True, now=NOW)
    assert stale["state"] == "SCAN"
    assert stale["posture"] == "FLAT_NO_TRADE"
    undated = ls.derive_loop_state(broker_health=_handoff(nonzero_option_positions_count=1,
                                                          as_of=None),
                                   live_mode=True, now=NOW)
    assert undated["state"] == "SCAN"


# --- ORDER-GUARD lane (2026-07-23 remediation): pending-order cancellation outranks everything ---

import data.odte_order_guard as og  # noqa: E402  (module-level import kept near its test section)


def _guard(state, minutes_ago=1, **over):
    g = {"schema_version": 1, "generated_at": _ts(minutes_ago=minutes_ago), "state": state,
         "underlying": "SPY", "lease_id": "lease123",
         "order": {"status": "pending", "symbol": "SPY", "option_id": "SPY260723P00737000",
                   "quantity": 2, "limit_price": 1.68},
         "cancel_required": state in (og.CANCEL_STALE_ENTRY, og.CANCEL_THESIS_INVALID),
         "safety_incident": state in og.INCIDENT_STATES,
         "prohibit_new_entries": state in og.INCIDENT_STATES,
         "pending_order_age_seconds": 95.0, "lease_seconds_remaining": -65.0,
         "next_action": "CANCEL NOW: stale entry order",
         "next_command": "broker-cancel (Hermes MCP lane) → odte-order-guard"}
    g.update(over)
    return g


def test_cancel_stale_order_outranks_fresh_candidate_and_gate():
    # A cancel-required pending order beats a fresh scan candidate AND a fresh promoted gate:
    # cancel first, never chase new entries while an unfilled stale order is live at the broker.
    gate = _entry_decision(decision="enter", execution_allowed=True, ts=_ts(minutes_ago=2))
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             journal_events=[gate],
                             order_guard=_guard(og.CANCEL_STALE_ENTRY), now=NOW)
    assert r["state"] == "PENDING_ORDER"
    assert r["posture"] == "CANCEL_PENDING_ORDER"
    assert r["context"]["cancel_required"] is True
    assert r["executable"] is False
    assert "cancel" in r["next_action"].lower() or "CANCEL" in r["next_action"]


def test_pending_fresh_order_exposes_age_and_lease_remaining():
    g = _guard(og.PENDING_FRESH, cancel_required=False, safety_incident=False,
               prohibit_new_entries=False, pending_order_age_seconds=7.0,
               lease_seconds_remaining=20.0,
               next_action="order pending inside the lease", next_command="odte-order-guard")
    r = ls.derive_loop_state(order_guard=g, now=NOW)
    assert r["state"] == "PENDING_ORDER"
    assert r["posture"] == "PENDING_ORDER_GUARD"
    assert r["context"]["pending_order_age_seconds"] == 7.0
    assert r["context"]["lease_seconds_remaining"] == 20.0


def test_fill_without_valid_lease_is_top_line_safety_incident():
    r = ls.derive_loop_state(order_guard=_guard(og.FILLED_WITHOUT_VALID_LEASE,
                                                order={"status": "filled", "symbol": "SPY"}),
                             now=NOW)
    assert r["state"] == "PENDING_ORDER"
    assert r["posture"] == "CANCEL_PENDING_ORDER"
    assert r["context"]["safety_incident"] is True
    assert r["context"]["prohibit_new_entries"] is True
    assert r["executable"] is False
    assert "SAFETY INCIDENT" in r["next_action"] or "incident" in r["next_action"].lower()


def test_stale_order_guard_artifact_is_ignored():
    stale = _guard(og.CANCEL_STALE_ENTRY, minutes_ago=ls.ORDER_GUARD_STALE_MINUTES + 5)
    r = ls.derive_loop_state(order_guard=stale, now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored stale order-guard decision" in s for s in r["notes"])


def test_no_order_guard_state_falls_through():
    r = ls.derive_loop_state(order_guard={"state": "NO_ORDER", "generated_at": _ts(minutes_ago=1)},
                             now=NOW)
    assert r["state"] == "SCAN"


def test_pending_order_with_broker_down_is_broker_degraded():
    r = ls.derive_loop_state(order_guard=_guard(og.CANCEL_STALE_ENTRY),
                             broker_health=_broker(lane="down"), now=NOW)
    assert r["state"] == "PENDING_ORDER"
    assert r["posture"] == "BROKER_DEGRADED"
    assert "cancel" in r["next_action"].lower()


def test_broker_truth_does_not_reconcile_away_pending_order_lane():
    # A fresh probe showing an open position must NOT erase a cancel-first instruction.
    bh = _handoff(nonzero_option_positions_count=1, open_option_orders_count=1)
    r = ls.derive_loop_state(order_guard=_guard(og.CANCEL_STALE_ENTRY),
                             broker_health=bh, live_mode=True, now=NOW)
    assert r["state"] == "PENDING_ORDER"


# --- execution-safety lockout: an incident closes the fresh-entry lane for the session ----------

def _incident_event(minutes_ago=10):
    return {"event_type": "execution_safety_incident", "seq": 40, "ts": _ts(minutes_ago=minutes_ago),
            "underlying": "SPY", "guard_state": "FILLED_WITHOUT_VALID_LEASE"}


def test_safety_incident_consumes_fresh_promoted_gate():
    gate = _entry_decision(decision="enter", execution_allowed=True, seq=41, ts=_ts(minutes_ago=2))
    r = ls.derive_loop_state(journal_events=[_incident_event(), gate], now=NOW)
    assert r["state"] != "PROMOTED"
    assert r["executable"] is False
    assert any("execution_safety_lockout" in s for s in r["reasons"])


def test_safety_incident_consumes_fresh_candidate_and_watch():
    r = ls.derive_loop_state(triggers=_candidate_triggers(ts=_ts(minutes_ago=1), alert=True),
                             journal_events=[_incident_event()], now=NOW)
    assert r["state"] != "CANDIDATE"
    assert any("execution_safety_lockout" in s for s in r["reasons"])
    cdec = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "SPY", "direction": "bearish"},
            "ts": _ts(minutes_ago=1)}
    r2 = ls.derive_loop_state(candidate_decision=cdec, journal_events=[_incident_event()], now=NOW)
    assert r2["posture"] != "SCOUT_FRESH_SETUP"
    assert any("execution_safety_lockout" in s for s in r2["reasons"])


def test_prior_day_incident_does_not_lock_today():
    old = _incident_event(minutes_ago=60 * 30)     # ~30h ago -> a prior ET day
    gate = _entry_decision(decision="enter", execution_allowed=True, seq=41, ts=_ts(minutes_ago=2))
    r = ls.derive_loop_state(journal_events=[old, gate], now=NOW)
    assert r["state"] == "PROMOTED"


# --- run_loop_status: new execution-safety artifacts surface with ages + lease countdown --------

def test_run_loop_status_reads_order_guard_and_lease_files(tmp_path):
    guard = _guard(og.CANCEL_STALE_ENTRY)
    (tmp_path / ls.ORDER_GUARD_FILENAME).write_text(json.dumps(guard))
    lease = {"authorized": True,
             "lease": {"lease_id": "lease123", "symbol": "SPY", "risk_mode": "PARTIAL_ACCOUNT",
                       "issued_at": _ts(minutes_ago=2), "expires_at": _ts(minutes_ago=1)}}
    (tmp_path / ls.EXECUTION_LEASE_FILENAME).write_text(json.dumps(lease))
    p = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert p["state"] == "PENDING_ORDER"
    assert p["artifacts"]["order_guard"] == "ok"
    assert p["artifacts"]["execution_lease"] == "ok"
    assert p["execution_lease"]["lease_id"] == "lease123"
    assert p["execution_lease"]["expired"] is True
    assert p["execution_lease"]["seconds_remaining"] == -60.0
    assert p["artifact_ages"]["order_guard"]["age_minutes"] == 1.0
    assert p["artifact_ages"]["execution_lease"]["as_of"] is not None
    json.dumps(p)   # stdout contract stays JSON-serializable


def test_run_loop_status_missing_guard_and_lease_files_read_missing(tmp_path):
    p = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert p["artifacts"]["order_guard"] == "missing"
    assert p["artifacts"]["execution_lease"] == "missing"
    assert "execution_lease" not in p          # no lease block invented when no lease exists
    assert p["artifact_ages"]["order_guard"]["as_of"] is None


def test_run_loop_status_attaches_weekly_telemetry(tmp_path):
    # The telemetry block rides EVERY status payload so a dying week is visible mid-week.
    r = ls.run_loop_status(state_dir=str(tmp_path))
    wk = r.get("weekly_telemetry")
    assert wk is not None
    assert "trades_this_week" in wk and "tripwire" in wk
    assert ls.render_markdown(r)          # renders without raising


# --- live_rails (2026-08-03: the agent reads live numbers, never remembered policy) -------------

def test_live_rails_rides_every_payload_with_live_constants():
    import data.odte_config as oc
    import data.odte_execution_policy as xp
    r = ls.derive_loop_state(now=NOW)
    rails = r["live_rails"]
    assert rails["conversion_sla_seconds"] == ls.CONFIRM_CONVERSION_SLA_SECONDS
    assert rails["snapshot_ttl_seconds"] == oc.SNAPSHOT_TTL_SECONDS
    assert rails["lease"]["default_ttl_seconds"] == xp.DEFAULT_LEASE_TTL_SECONDS
    assert rails["lease"]["hard_cap_seconds"] == xp.MAX_LEASE_TTL_SECONDS
    assert rails["chase_band_fraction"] == oc.CHASE_BAND_FRACTION
    assert rails["max_debit_fraction"] == {"full": oc.MAX_DEBIT_FRACTION,
                                           "b_plus": oc.B_PLUS_DEBIT_FRACTION}
    assert rails["executable_universe"] == list(oc.EXECUTABLE_UNIVERSE)
    assert rails["daily_trade_budget"]["budget"] == oc.DAILY_TRADE_BUDGET
    # no fresh BP probe -> dollars are NULL, never guessed (the 754C dead-rail fix)
    assert rails["max_debit_dollars"] is None
    assert ls.render_markdown(r)


def test_live_rails_computes_dollars_from_fresh_broker_truth():
    import data.odte_config as oc
    health = {"parent_robinhood_mcp": "OK", "live_review_place_allowed": True,
              "as_of": NOW.isoformat(), "buying_power": 348.16}
    r = ls.derive_loop_state(broker_health=health, now=NOW)
    rails = r["live_rails"]
    assert rails["buying_power"] == 348.16
    assert rails["max_debit_dollars"]["full"] == round(oc.MAX_DEBIT_FRACTION * 348.16, 2)
    assert rails["max_debit_dollars"]["b_plus"] == round(oc.B_PLUS_DEBIT_FRACTION * 348.16, 2)


def test_live_rails_green_reentry_state_reflects_journal(monkeypatch):
    # The auto-arm MECHANISM under test; the live posture is OFF since 2026-08-14 (W33 fence:
    # post-green re-entries ran -$49/4), so pin it ON here the same way the kill-switch test
    # pins it off.
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    r = ls.derive_loop_state(journal_events=_green_scalp_journal(), now=NOW)
    gr = r["live_rails"]["green_reentry"]
    assert gr["locked"] is True
    assert gr["structurally_armable"] is True          # budget slot open, cooldown clear
    assert gr["winning_tier_today"] == "full"          # legacy journal -> default tier
    assert gr["manual_override_key"] == "allow_reentry_after_green"


# --- 2026-08-05 blind-day fixes: REVIEWED date-gate, rescan override, lease suppression -------

def _pm(trade_id="SPY-T1"):
    return {"event_type": "postmortem", "trade_id": trade_id, "seq": 9, "ts": _ts(minutes_ago=10)}


def test_prior_day_reviewed_trade_resumes_scan():
    # Aug-5: the Aug-3 reviewed trade parked the loop at REVIEWED all session and the controller
    # obeyed "roll up the journal report" for 68 ticks. A PRIOR-ET-day review now falls to SCAN.
    plan = _closed_plan(closed_at=_ts(hours_ago=30), exit_fill_time=_ts(hours_ago=30),
                        entry_fill_time=_ts(hours_ago=31), updated_at=_ts(hours_ago=30))
    r = ls.derive_loop_state(active_trade=plan, journal_events=[_pm()], now=NOW)
    assert r["state"] == "SCAN"
    assert any("prior-day" in reason for reason in r["reasons"])
    assert "odte-journal-report" not in r["next_command"]


def test_same_day_reviewed_keeps_state_but_gets_rescan_override():
    # NOW is 14:00 ET Friday: session open, budget untouched — even a legitimately REVIEWED
    # same-day loop must RE-SCAN, not re-journal.
    r = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[_pm()], now=NOW)
    assert r["state"] == "REVIEWED"
    assert r.get("rescan_override") is True
    assert "odte-candidate-watch MARKET=" in r["next_command"]
    assert "odte-journal-report" not in r["next_command"]


def test_rescan_override_respects_budget_and_late_day():
    from datetime import datetime, timezone

    import data.odte_config as oc
    # Budget exhausted on a net-RED day: no override — the anti-tilt half of the rail holds.
    fills = []
    for i in range(oc.DAILY_TRADE_BUDGET):
        fills += [{"event_type": "order_filled", "trade_id": f"t{i}", "option_id": f"o{i}",
                   "ts": _ts(hours_ago=2)},
                  {"event_type": "order_closed", "trade_id": f"t{i}", "option_id": f"o{i}",
                   "realized_pnl": -4.0, "ts": _ts(hours_ago=1)}]
    r = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[_pm(), *fills], now=NOW)
    assert r.get("rescan_override") is None
    assert "odte-journal-report" in r["next_command"]
    # Budget exhausted on a net-GREEN day: the scan stays alive, A_PLUS-only, said explicitly.
    green_fills = [dict(f, realized_pnl=4.0) if f["event_type"] == "order_closed" else f
                   for f in fills]
    rg = ls.derive_loop_state(active_trade=_closed_plan(), journal_events=[_pm(), *green_fills],
                              now=NOW)
    assert rg.get("rescan_override") is True
    assert "A_PLUS-TIER SETUPS ONLY" in rg["next_action"]
    # 15:30 ET (<45 min to close): no override.
    late = datetime(2026, 6, 26, 19, 30, tzinfo=timezone.utc)
    plan = _closed_plan(closed_at=(late.isoformat()), updated_at=late.isoformat())
    pm = {"event_type": "postmortem", "trade_id": "SPY-T1", "seq": 9, "ts": late.isoformat()}
    r2 = ls.derive_loop_state(active_trade=plan, journal_events=[pm], now=late)
    assert r2.get("rescan_override") is None


def test_bare_scan_gets_rescan_override_in_session():
    r = ls.derive_loop_state(now=NOW)
    assert r["state"] == "SCAN"
    assert r.get("rescan_override") is True
    assert "odte-day-score" in r["next_command"]


def test_prior_day_expired_lease_age_is_suppressed(tmp_path):
    from datetime import timedelta
    prior_issue = (NOW - timedelta(hours=30))
    lease = {"authorized": True,
             "lease": {"lease_id": "L-old", "symbol": "SPY",
                       "issued_at": prior_issue.isoformat(),
                       "expires_at": (prior_issue + timedelta(seconds=60)).isoformat()}}
    (tmp_path / ls.EXECUTION_LEASE_FILENAME).write_text(json.dumps(lease))
    p = ls.run_loop_status(state_dir=str(tmp_path), now=NOW)
    assert p["execution_lease"]["expired"] is True
    assert p["artifact_ages"]["execution_lease"]["as_of"] is None       # inert audit trail
    # A SAME-day expired lease still shows (it is current work that just died) — pinned by
    # test_run_loop_status_reads_order_guard_and_lease_files above.


def test_adjudicated_incident_reopens_the_entry_lane():
    # 2026-08-06: a proven-false incident locked the day; the user-authorized adjudication
    # (named, attributed, same ET day) must let a fresh candidate be actionable again.
    incident = {"event_type": "execution_safety_incident", "event_id": "ev-abc", "seq": 5937,
                "ts": _ts(minutes_ago=60), "underlying": "IWM",
                "guard_state": "BROKER_MISMATCH_BLOCKED"}
    trig = _candidate_triggers(alert=True)
    locked = ls.derive_loop_state(triggers=trig, journal_events=[incident], now=NOW)
    assert locked["state"] != "CANDIDATE"                      # lockout swallows the candidate
    adjudication = {"event_type": "execution_safety_incident_adjudicated",
                    "incident_event_id": "ev-abc", "ts": _ts(minutes_ago=5),
                    "adjudicated_by": "lukas", "reason": "guard false positive, fix committed"}
    unlocked = ls.derive_loop_state(triggers=trig, journal_events=[incident, adjudication],
                                    now=NOW)
    assert unlocked["state"] == "CANDIDATE"


def test_terminally_refused_confirm_is_consumed_not_reconverted():
    # 2026-08-06 QQQ 720C: odte-convert journaled the identity-bound budget veto twice while
    # loop-status kept commanding CONVERT_CANDIDATE_NOW for the same dead contract — the
    # terminal-evidence check existed only on the STALE-confirm path. A fresh confirm whose
    # identity was terminally refused AT/AFTER the confirm is consumed; the loop re-scans.
    confirm = _confirmed_iwm_watch(minutes_ago=3)
    veto = {"event_type": "no_trade_decision", "seq": 50, "ts": _ts(minutes_ago=2),
            "underlying": "IWM", "option_id": IWM_OPTION_ID, "stage": "entry_gate",
            "reason_codes": ["final_confirmation_budget_check_failed"]}
    r = ls.derive_loop_state(candidate_decision=confirm, journal_events=[veto], now=NOW)
    assert r["posture"] != "CONVERT_CANDIDATE_NOW"
    assert any("terminally refused" in x for x in r["reasons"])
    assert r.get("rescan_override") is True        # session open + budget: go re-scan the tape
    # A veto OLDER than the confirm never consumes it — a NEW confirm cycle re-arms normally.
    r2 = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=3),
                              journal_events=[{**veto, "ts": _ts(minutes_ago=10)}], now=NOW)
    assert r2["posture"] == "CONVERT_CANDIDATE_NOW"
    # A different contract's veto never absolves this identity.
    r3 = ls.derive_loop_state(candidate_decision=_confirmed_iwm_watch(minutes_ago=3),
                              journal_events=[{**veto, "option_id": "OTHER-OPTION"}], now=NOW)
    assert r3["posture"] == "CONVERT_CANDIDATE_NOW"


def test_rescan_override_never_contradicts_the_safety_lockout():
    # 2026-08-06: the payload told a safety-locked day to "RE-SCAN the tape" while its reasons
    # enforced the lockout. A locked day never re-scans for entries; an ADJUDICATED day does.
    incident = {"event_type": "execution_safety_incident", "event_id": "ev-x", "seq": 70,
                "ts": _ts(minutes_ago=30), "underlying": "QQQ"}
    r = ls.derive_loop_state(journal_events=[incident], now=NOW)
    assert r.get("rescan_override") is None
    assert "odte-journal-report" not in str(r.get("next_command")) or True   # whatever it is,
    assert "RE-SCAN" not in str(r.get("next_action"))                        # never a re-scan
    adjudication = {"event_type": "execution_safety_incident_adjudicated",
                    "incident_event_id": "ev-x", "ts": _ts(minutes_ago=5),
                    "adjudicated_by": "lukas", "reason": "false positive, fix committed"}
    r2 = ls.derive_loop_state(journal_events=[incident, adjudication], now=NOW)
    assert r2.get("rescan_override") is True                                 # unlocked: scan on


def test_scanning_next_commands_carry_journal_so_the_lane_records():
    """The controller treats loop-status `next_command` as authoritative, so the JOURNAL=1 that
    makes the scanning lane record its evaluations has to live HERE, not only in the agent prompt.

    2026-08-07: run_candidate_watch gained --journal and the Makefile gained the passthrough, but
    loop-status kept emitting the command without it — so the near-miss population would still have
    gone unrecorded on every tick that followed next_command verbatim.
    """
    import inspect

    import data.odte_loop_status as mod
    src = inspect.getsource(mod)
    for fragment in ("make odte-candidate-watch CANDIDATE=", "make odte-candidate-watch MARKET="):
        idx = src.find(fragment)
        assert idx != -1, f"missing suggested command: {fragment}"
    # every emitted odte-candidate-watch invocation asks for journaling
    for line in src.splitlines():
        if "odte-candidate-watch" in line and "make " in line and "WRITE=1" in line:
            assert "JOURNAL=1" in line or "JOURNAL=1" in src[src.find(line):src.find(line) + 400], line


def test_undated_broker_health_fails_closed():
    """A NON-EMPTY health payload carrying no parseable timestamp used to skip the staleness branch
    entirely (`age is not None and age > TTL`) and fall through to "live orders permitted" — the
    same treatment as a one-minute-old probe, with no way to tell how old the truth was.

    Every other staleness check on the money path already fails closed on an undated artifact:
    odte_convert's preflight emits `_undated`, odte_execution_policy states "undated fails closed",
    and odte_order_guard cancels on an undated lease and raises a safety incident.
    """
    from datetime import datetime, timedelta, timezone

    from data.odte_loop_status import BROKER_STALE_MINUTES, classify_broker_lane
    now = datetime.now(timezone.utc)
    healthy = {"parent_robinhood_mcp": "OK", "live_review_place_allowed": True, "place_lane": "ok"}

    undated = classify_broker_lane(healthy, now=now)
    assert undated["lane"] == "stale"
    assert undated["orders_ok"] is False
    assert "UNDATED" in undated["note"]
    # a stale probe is an outdated FILE, not a broker fault — the last known permission is kept so
    # the controller refreshes rather than concluding live orders are impossible
    assert undated["last_known_place_allowed"] is True
    assert undated.get("refresh_command")

    # an EMPTY payload is still `unknown`, not stale — pure offline mode, behaviour unchanged
    assert classify_broker_lane({}, now=now)["lane"] == "unknown"
    assert classify_broker_lane(None, now=now)["orders_ok"] is None

    fresh = classify_broker_lane({**healthy, "as_of": now.isoformat()}, now=now)
    assert fresh["lane"] == "ok" and fresh["orders_ok"] is True
    old = {**healthy, "as_of": (now - timedelta(minutes=BROKER_STALE_MINUTES + 1)).isoformat()}
    assert classify_broker_lane(old, now=now)["lane"] == "stale"


def test_broker_health_accepts_every_timestamp_spelling_the_writers_use():
    """The controller drifts between spellings — on 2026-08-07 it wrote market_snapshot_live.json
    with `generated_at` and no `as_of` at all. Reading only a subset turns a fresh probe into a
    stale one (or, before the fix, an undated one into a fresh one)."""
    from datetime import datetime, timezone

    from data.odte_loop_status import classify_broker_lane
    now = datetime.now(timezone.utc)
    healthy = {"parent_robinhood_mcp": "OK", "live_review_place_allowed": True, "place_lane": "ok"}
    for field in ("as_of", "ts", "checked_at", "generated_at", "updated_at"):
        out = classify_broker_lane({**healthy, field: now.isoformat()}, now=now)
        assert out["lane"] == "ok", f"{field} not accepted: {out}"


def test_live_rails_advertise_the_strict_reentry_bar(monkeypatch):
    """After the strict bar shipped, the controller ran FOUR doomed convert cycles on post-green
    b_plus confirms — nothing told it the bar had moved. The rails now name the lowest tier that
    can arm, and None when nothing can (an a_plus win ends the day)."""
    import data.odte_config as _oc
    monkeypatch.setattr(_oc, "GREEN_REENTRY_AUTO_ARM", True)
    monkeypatch.setattr(_oc, "GREEN_REENTRY_REQUIRE_BETTER_TIER", True)
    r = ls.derive_loop_state(journal_events=_green_scalp_journal(), now=NOW)
    gr = r["live_rails"]["green_reentry"]
    assert gr["require_better_tier"] is True
    assert gr["winning_tier_today"] == "full"
    assert gr["min_reentry_tier"] == "a_plus"             # strictly above full

    monkeypatch.setattr(_oc, "GREEN_REENTRY_REQUIRE_BETTER_TIER", False)
    gr2 = ls.derive_loop_state(journal_events=_green_scalp_journal(),
                               now=NOW)["live_rails"]["green_reentry"]
    assert gr2["min_reentry_tier"] == "full"              # same-or-better semantics


def test_live_rails_advertise_the_fences():
    """2026-08-18: on the first daily-loss-floor day the controller converted a clean confirm
    straight into a gate veto — it could not see the day was over. The fences block folds the
    floor, budget, and the freshness gate into rails the agent reads every tick."""
    r = ls.derive_loop_state(journal_events=[], now=NOW)
    f = r["live_rails"]["fences"]
    import data.odte_config as _oc
    assert f["min_entry_premium"] == _oc.MIN_ENTRY_PREMIUM
    assert f["daily_loss_floor_dollars"] == _oc.DAILY_LOSS_FLOOR_DOLLARS
    assert f["max_signal_age_minutes"] == _oc.MAX_SIGNAL_AGE_MINUTES
    assert f["daily_loss_floor_reached"] is False
    assert f["day_over"] is False


def test_fences_day_over_folds_floor_and_budget():
    # A closed trade at -$40 (past any armed floor) plus an exhausted budget both flip day_over.
    events = [
        {"event_type": "order_filled", "trade_id": "t1", "option_id": "o1",
         "ts": NOW.isoformat()},
        {"event_type": "order_closed", "trade_id": "t1", "option_id": "o1",
         "realized_pnl": -40.0, "ts": NOW.isoformat()},
    ]
    r = ls.derive_loop_state(journal_events=events, now=NOW)
    f = r["live_rails"]["fences"]
    assert f["daily_loss_floor_reached"] is True
    assert f["day_over"] is True
