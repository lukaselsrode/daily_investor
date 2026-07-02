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
    # still maps to GATED (awaiting promotion), unlike a denied verdict.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter",
                                                             execution_allowed=False)], now=NOW)
    assert r["state"] == "GATED"
    assert r["executable"] is False
    assert r["next_command"].endswith("--promote-to-execution")


def test_entry_gate_executable_is_promoted():
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter",
                                                             execution_allowed=True)], now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["executable"] is True
    assert r["next_command"] == "odte-position"


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
    stale_cdec = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "QQQ"},
                  "ts": _ts(minutes_ago=20)}
    r = ls.derive_loop_state(candidate_decision=stale_cdec, now=NOW)
    assert r["state"] == "SCAN"
    assert any("ignored stale candidate watch" in s for s in r["notes"])

    live = ls.derive_loop_state(active_trade=_open_plan(), position_decision={
        "decision": "HOLD", "underlying": "SPY", "ts": _ts(minutes_ago=1)},
        candidate_decision=stale_cdec, now=NOW)
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
    assert r["next_command"] == "odte-entry-gate"


def test_posture_promoted_gate_scouts_fresh_setup():
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             now=NOW)
    assert r["state"] == "PROMOTED"
    assert r["posture"] == "SCOUT_FRESH_SETUP"


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


def test_posture_stale_candidate_watch_is_flat_no_trade():
    stale_cdec = {"decision": "CONFIRM_ENTRY", "candidate": {"ticker": "QQQ"}, "ts": _ts(minutes_ago=20)}
    r = ls.derive_loop_state(candidate_decision=stale_cdec, now=NOW)
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


def test_healthy_broker_promoted_gate_scouts_and_executes():
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             broker_health=_broker(lane="ok"), now=NOW)
    assert r["posture"] == "SCOUT_FRESH_SETUP"
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


def test_offline_mode_promoted_gate_with_unknown_broker_is_scout():
    # Pure decision-support (offline) is lenient: unknown broker is reported, not blocked.
    r = ls.derive_loop_state(journal_events=[_entry_decision(decision="enter", execution_allowed=True)],
                             live_mode=False, now=NOW)
    assert r["posture"] == "SCOUT_FRESH_SETUP"
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


def test_run_loop_status_live_mode_flag_fails_closed_on_missing_broker(tmp_path):
    (tmp_path / "decision_journal.jsonl").write_text(json.dumps(
        _entry_decision(decision="enter", execution_allowed=True)) + "\n")
    live = ls.run_loop_status(state_dir=str(tmp_path), live_mode=True, now=NOW)
    assert live["posture"] == "BROKER_DEGRADED"
    offline = ls.run_loop_status(state_dir=str(tmp_path), live_mode=False, now=NOW)
    assert offline["posture"] == "SCOUT_FRESH_SETUP"
