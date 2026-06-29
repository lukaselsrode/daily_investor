"""0DTE loop status — PURE/OFFLINE state machine over the canonical artifacts. NO broker/LLM/orders.

One read-only surface that tells the live controller (Hermes/MCP) WHERE in the loop it is and WHICH
command runs next, so the execution loop reads as one obvious cycle instead of a pile of independent
tools:

    SCAN → CANDIDATE → GATED → PROMOTED → ENTERED → MANAGING → EXITED → REVIEWED   (DEGRADED on a fault)

It does NOT re-run any gate or re-derive any decision — it only SUMMARIZES artifacts the other 0DTE
tools already wrote under ``data/odte/`` and points at the next command. The hard rule mirrors the
rest of the layer: a LIVE position (or a fault on one) always outranks the scan/candidate/gate lane —
managing/exiting an open trade beats chasing a new one. Reads, never writes; places NO orders.

Inputs (all read by ``run_loop_status``; ``derive_loop_state`` is the pure core, given the payloads):
  active_trade.json        the current plan/position  (``odte_position``/controller)
  position_decision.json   the latest live-position decision  (``odte-position``)
  active_candidate.json    a pre-entry setup being watched hawkishly  (``odte-candidate-watch``)
  candidate_decision.json  latest pre-entry watch decision  (``odte-candidate-watch``)
  triggers.json            the latest scan/trigger lane payload  (``odte-watchdog``)
  decision_journal.jsonl   the journal — latest ``entry_decision`` (gate) + any ``postmortem`` (review)

States
  SCAN       nothing actionable — keep scanning (``odte-watchdog``)
  CANDIDATE  a non-restricted candidate is on the board, scan_only/observe — assemble the gate
  GATED      an entry-gate record exists but is NOT execution-allowed — promote only if gates pass
  PROMOTED   an entry-gate record is execution-allowed — the manager may enter, then watch
  ENTERED    a plan is open but no live decision computed yet — start the position watch
  MANAGING   a live position with a current decision (HOLD or an actionable exit trigger)
  EXITED     the last trade closed and has no postmortem yet — record the review
  REVIEWED   the last trade closed and is reviewed — idle; roll up the journal report
  DEGRADED   a live position can't be valued / a live artifact is malformed or stale — re-establish
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_DATA_DIR

SCHEMA_VERSION = 1

DEFAULT_STATE_DIR = ODTE_DATA_DIR
PLAN_FILENAME = "active_trade.json"
POSITION_DECISION_FILENAME = "position_decision.json"
ACTIVE_CANDIDATE_FILENAME = "active_candidate.json"
CANDIDATE_DECISION_FILENAME = "candidate_decision.json"
TRIGGERS_FILENAME = "triggers.json"
JOURNAL_FILENAME = "decision_journal.jsonl"

# Ordered loop states (scan → review), plus the DEGRADED fault state.
LOOP_STATES = ("SCAN", "CANDIDATE", "GATED", "PROMOTED", "ENTERED", "MANAGING",
               "EXITED", "REVIEWED", "DEGRADED")

# A plan whose status is one of these is NOT a live position (mirrors odte_position._INACTIVE_STATUS).
_INACTIVE_STATUS = {"closed", "exited", "flat", "done"}
# position_decision.decision values that mean "no live position to manage".
_NO_POSITION_DECISIONS = {"", "NO_POSITION", "RESTRICTED"}

# A live-position decision older than this (when a wall clock is supplied) is treated as flying blind.
# This must stay tight for 0DTE scalp HAWK mode: the live controller should poll a held contract
# around every 30 seconds inside the holding branch, so a >2-minute-old management decision means
# several checks were missed or the feed/cron is degraded.
STALE_DECISION_MINUTES = 2
# A closed trade older than this stops nagging EXITED so it can't mask a genuinely fresh scan.
STALE_TRADE_HOURS = 36
# A FAILED entry gate (present, not execution-allowed) stops being sticky once a fresh scan candidate
# lands at least this many minutes after it: on a real trend day the tape keeps moving, so an old
# NO_TRADE/deny gate must not pin the loop at GATED forever when a materially newer watchdog candidate
# shows up. Only DENIED gates are superseded — an execution-allowed (PROMOTED) gate is never demoted
# by a scan, and the candidate this falls through to stays scan_only/observe (never executable).
SUPERSEDE_GATE_MINUTES = 15

# state → (human loop stage, whether an execution-tier action is authorized by the artifacts).
# Only a live trade (ENTERED/MANAGING) or an execution-allowed gate (PROMOTED) is "executable".
_STAGE = {
    "SCAN": ("scan", False),
    "CANDIDATE": ("thesis", False),
    "GATED": ("entry", False),
    "PROMOTED": ("entry", True),
    "ENTERED": ("watch", True),
    "MANAGING": ("watch", True),
    "EXITED": ("exit", False),
    "REVIEWED": ("review", False),
    "DEGRADED": ("degraded", False),
}


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _age_minutes(ts: Any, now: datetime | None) -> float | None:
    dt = _parse_ts(ts)
    if dt is None or now is None:
        return None
    return (now - dt).total_seconds() / 60.0


def _plan_present(plan: dict) -> bool:
    return bool(plan) and any(plan.get(k) for k in
                              ("underlying", "trade_id", "option_id", "strike_price", "option_type"))


def _plan_had_entry(plan: dict) -> bool:
    return any(plan.get(k) for k in ("entry_price", "entry_premium", "entry_order_id", "entry_fill_time"))


def _next_for(state: str, *, live: bool) -> tuple[str, str]:
    """(next_action prose, next_command). Decision-support only — never an order."""
    if state == "SCAN":
        return ("keep scanning for a non-restricted candidate", "odte-watchdog")
    if state == "CANDIDATE":
        return ("assemble the thesis→entry gate for the candidate", "odte-entry-gate")
    if state == "GATED":
        return ("gate not execution-allowed — promote only if every gate passes",
                "odte-entry-gate --promote-to-execution")
    if state == "PROMOTED":
        return ("gate execution-allowed — manager may enter, then start the position watch",
                "odte-position")
    if state == "ENTERED":
        return ("plan open, no live decision yet — start the position watch",
                "odte-position --snapshot <live.json>")
    if state == "MANAGING":
        return ("watch the live position; act on the current decision",
                "odte-position --snapshot <live.json>")
    if state == "EXITED":
        return ("trade closed — record the postmortem, then fold the day's artifacts",
                "odte-journal (postmortem) → odte-ingest-artifacts")
    if state == "REVIEWED":
        return ("loop complete — roll up the journal report", "odte-journal-report --write")
    # DEGRADED
    if live:
        return ("can't value the live position — re-establish the snapshot",
                "odte-position --snapshot <live.json>")
    return ("a live artifact is malformed/stale — fold and inspect the journal",
            "odte-ingest-artifacts")


def _latest_event(events: list[dict], event_type: str) -> dict | None:
    """Most recent journal event of a type (by seq, then position). Never raises."""
    best, best_seq = None, -1.0
    for i, ev in enumerate(events or []):
        if not isinstance(ev, dict) or ev.get("event_type") != event_type:
            continue
        seq = ev.get("seq")
        key = float(seq) if isinstance(seq, (int, float)) else float(i)
        if key >= best_seq:
            best, best_seq = ev, key
    return best


def derive_loop_state(active_trade: dict | None = None,
                      position_decision: dict | None = None,
                      active_candidate: dict | None = None,
                      candidate_decision: dict | None = None,
                      triggers: dict | None = None,
                      journal_events: list[dict] | None = None,
                      *, errors: set[str] | None = None,
                      now: datetime | None = None,
                      stale_decision_minutes: int = STALE_DECISION_MINUTES) -> dict:
    """Pure loop-state resolver. Summarizes the artifacts; re-derives no gate/decision. Never raises.

    `errors` names artifacts that were present-but-malformed (so a broken live artifact degrades
    rather than silently reads as missing). `now` (optional) enables staleness/recency checks."""
    errors = set(errors or [])
    events = journal_events or []
    plan = _dict(active_trade)
    pdec = _dict(position_decision)
    acand = _dict(active_candidate)
    cdec = _dict(candidate_decision)
    trig = _dict(triggers)
    reasons: list[str] = []

    status = str(plan.get("status") or ("open" if _plan_present(plan) else "")).strip().lower()
    plan_present = _plan_present(plan)
    plan_open = plan_present and status not in _INACTIVE_STATUS
    pd_decision = str(pdec.get("decision") or "").strip().upper()
    pd_live = pd_decision not in _NO_POSITION_DECISIONS
    position_live = plan_open or pd_live
    live_malformed = "active_trade" in errors or "position_decision" in errors

    # --- Live-position lane: always outranks scan/candidate/gate -----------------------------
    if position_live or (live_malformed and plan_present):
        underlying = pdec.get("underlying") or plan.get("underlying")
        stale = _age_minutes(pdec.get("ts"), now)
        if pd_decision == "MONITORING_DEGRADED":
            reasons.append("position_decision=MONITORING_DEGRADED")
            state = "DEGRADED"
        elif live_malformed:
            reasons.append("live position artifact malformed")
            state = "DEGRADED"
        elif stale is not None and stale > stale_decision_minutes:
            reasons.append(f"position_decision stale ({stale:.0f}m > {stale_decision_minutes}m)")
            state = "DEGRADED"
        elif pd_live:
            reasons.append(f"live position {underlying or '?'} decision={pd_decision}")
            state = "MANAGING"
        else:
            reasons.append(f"plan open ({underlying or '?'}) — no live decision yet")
            state = "ENTERED"
        return _payload(state, reasons, now, live=True, context={
            "underlying": underlying, "decision": pd_decision or None,
            "pnl_pct": pdec.get("pnl_pct"), "mode": pdec.get("mode") or plan.get("mode"),
            "plan_status": status or None})

    # --- Post-trade / scan / gate lane -------------------------------------------------------
    closed_trade = plan_present and status in _INACTIVE_STATUS and _plan_had_entry(plan)
    trade_id = plan.get("trade_id")
    close_ts = plan.get("closed_at") or plan.get("exit_fill_time") or plan.get("updated_at")
    reviewed = bool(closed_trade and trade_id and any(
        isinstance(e, dict) and e.get("event_type") == "postmortem" and e.get("trade_id") == trade_id
        for e in events))
    close_age_h = _age_minutes(close_ts, now)
    recent_close = close_age_h is None or close_age_h <= STALE_TRADE_HOURS * 60

    # Entry-gate (latest journal entry_decision). Only honored if it belongs to the CURRENT cycle —
    # a gate dated at/before the last close is the closed trade's own gate, already consumed.
    gate = _latest_event(events, "entry_decision") or {}
    gate_ts = _parse_ts(gate.get("ts")) if gate else None
    close_dt = _parse_ts(close_ts)
    gate_fresh = bool(gate) and (not closed_trade or close_dt is None or
                                 (gate_ts is not None and gate_ts > close_dt))
    # PROMOTED requires EXPLICIT gate permission: execution_allowed is True. A bare decision=="enter"
    # with execution_allowed missing/false is a stale/partial record and stays GATED — the loop only
    # advances on explicit manager promotion / gate permission, never on the intent verb alone.
    gate_exec = gate_fresh and gate.get("execution_allowed") is True

    candidate = _dict(trig.get("candidate"))
    trig_candidate = bool(candidate.get("ticker")) and not candidate.get("restricted")
    trig_ts = _parse_ts(trig.get("ts") or trig.get("generated_at"))
    newer_candidate_after_gate = False
    if trig_candidate and gate_fresh and not gate_exec and gate_ts is not None and trig_ts is not None:
        newer_candidate_after_gate = (
            (trig_ts - gate_ts).total_seconds() / 60.0 >= SUPERSEDE_GATE_MINUTES
        )

    if closed_trade and not reviewed and recent_close:
        reasons.append(f"trade {trade_id or '?'} closed, no postmortem yet")
        return _payload("EXITED", reasons, now, live=False,
                        context={"trade_id": trade_id, "underlying": plan.get("underlying"),
                                 "realized_pnl": plan.get("gross_pnl") or plan.get("net_pnl_est")})
    if gate_exec:
        reasons.append(f"entry gate execution-allowed ({gate.get('underlying') or '?'})")
        return _payload("PROMOTED", reasons, now, live=False, context=_gate_ctx(gate))

    # Pre-entry candidate HAWK lane. A live/open position and an execution-allowed gate still win,
    # but an actively watched candidate should outrank stale/non-executable gates so the controller
    # keeps checking confirmation/degradation instead of falling back to a slow broad scan. Candidate
    # watch is never executable by itself; CONFIRM_ENTRY means "build/promote a fresh entry gate".
    watch_decision = str(cdec.get("decision") or acand.get("decision") or "").upper()
    watched = _dict(cdec.get("candidate")) or acand
    watch_live = bool(watched.get("ticker") or watched.get("underlying") or watched.get("symbol"))
    inactive_watch = {"DEGRADED_NO_TRADE", "EXPIRED_NO_CONFIRMATION", ""}
    if watch_live and watch_decision not in inactive_watch:
        if watch_decision == "CONFIRM_ENTRY":
            reasons.append("candidate watch confirmed setup; build a fresh entry gate")
            payload = _payload("CANDIDATE", reasons, now, live=False,
                               context=_candidate_watch_ctx(watched, cdec, confirmed=True))
            payload["next_command"] = "odte-entry-gate"
            payload["next_action"] = "candidate confirmed — assemble/promote a fresh entry gate"
            return payload
        if watch_decision == "BROKER_BLOCKED":
            reasons.append("candidate watch blocked by broker/review lane")
            payload = _payload("CANDIDATE", reasons, now, live=False,
                               context=_candidate_watch_ctx(watched, cdec, confirmed=False))
            payload["next_command"] = "verify-broker-review-lane"
            payload["next_action"] = "execution lane blocked — verify/repair broker review before promotion"
            return payload
        reasons.append(f"candidate watch active ({watch_decision or 'KEEP_WATCHING'})")
        payload = _payload("CANDIDATE", reasons, now, live=False,
                           context=_candidate_watch_ctx(watched, cdec, confirmed=False))
        payload["next_command"] = "odte-candidate-watch"
        payload["next_action"] = "candidate HAWK — keep checking confirm/degrade before broad scan"
        return payload
    if newer_candidate_after_gate:
        reasons.append("newer scan candidate supersedes prior non-executable gate")
        reasons.append(f"candidate {candidate.get('ticker')} {candidate.get('direction') or ''} "
                       f"(scan_only/observe)".strip())
        return _payload("CANDIDATE", reasons, now, live=False, context={
            "underlying": candidate.get("ticker"), "direction": candidate.get("direction"),
            "spy_verdict": trig.get("spy_verdict"), "scan_only": True,
            "superseded_gate_decision": gate.get("decision")})
    if gate_fresh:
        reasons.append(f"entry gate present, not execution-allowed (decision={gate.get('decision')})")
        return _payload("GATED", reasons, now, live=False, context=_gate_ctx(gate))
    if trig_candidate:
        reasons.append(f"candidate {candidate.get('ticker')} {candidate.get('direction') or ''} "
                       f"(scan_only/observe)".strip())
        return _payload("CANDIDATE", reasons, now, live=False, context={
            "underlying": candidate.get("ticker"), "direction": candidate.get("direction"),
            "spy_verdict": trig.get("spy_verdict"), "scan_only": True})
    if reviewed:
        reasons.append(f"trade {trade_id or '?'} closed and reviewed — idle")
        return _payload("REVIEWED", reasons, now, live=False,
                        context={"trade_id": trade_id, "underlying": plan.get("underlying")})
    if closed_trade and not reviewed:
        reasons.append(f"prior trade {trade_id or '?'} unreviewed but stale — scanning")
    reasons.append("no actionable candidate, gate, or live position")
    return _payload("SCAN", reasons, now, live=False, context={})


def _gate_ctx(gate: dict) -> dict:
    return {"underlying": gate.get("underlying") or gate.get("symbol"),
            "direction": gate.get("direction"), "decision": gate.get("decision"),
            "scan_only": bool(gate.get("scan_only", False)),
            "execution_allowed": bool(gate.get("execution_allowed", False))}


def _candidate_watch_ctx(candidate: dict, decision: dict, *, confirmed: bool) -> dict:
    return {"underlying": candidate.get("ticker") or candidate.get("underlying") or candidate.get("symbol"),
            "direction": candidate.get("direction"), "candidate_watch": True,
            "candidate_decision": decision.get("decision"), "confirmed": confirmed,
            "scan_only": True, "execution_allowed": False}


def _payload(state: str, reasons: list[str], now: datetime | None, *, live: bool,
             context: dict) -> dict:
    stage, executable = _STAGE[state]
    action, command = _next_for(state, live=live)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "state": state,
        "loop_stage": stage,
        "executable": executable,
        "next_action": action,
        "next_command": command,
        "reasons": reasons,
        "context": {k: v for k, v in context.items() if v is not None},
        "places_orders": False,
    }


def run_loop_status(state_dir: str | None = None, *, now: datetime | None = None) -> dict:
    """Read the canonical data/odte artifacts and resolve the loop state. NO broker/LLM/orders.

    Reuses odte_watchdog._read_json (status-aware: ok|missing|invalid) and odte_journal.read_events
    (skips malformed lines, never raises), so a missing artifact reads as SCAN and a malformed live
    artifact degrades rather than crashing."""
    from data.odte_journal import read_events
    from data.odte_watchdog import _read_json

    now = now or datetime.now(timezone.utc)   # wall clock lives in the IO wrapper; derive stays pure
    base = Path(os.path.expanduser(state_dir or DEFAULT_STATE_DIR))
    plan, plan_status = _read_json(base / PLAN_FILENAME)
    pdec, pdec_status = _read_json(base / POSITION_DECISION_FILENAME)
    acand, acand_status = _read_json(base / ACTIVE_CANDIDATE_FILENAME)
    cdec, cdec_status = _read_json(base / CANDIDATE_DECISION_FILENAME)
    trig, trig_status = _read_json(base / TRIGGERS_FILENAME)
    journal = base / JOURNAL_FILENAME
    events = read_events(str(journal)) if journal.exists() else []

    errors = {name for name, st in (("active_trade", plan_status),
                                    ("position_decision", pdec_status),
                                    ("active_candidate", acand_status),
                                    ("candidate_decision", cdec_status),
                                    ("triggers", trig_status)) if st == "invalid"}
    payload = derive_loop_state(active_trade=plan, position_decision=pdec,
                                active_candidate=acand, candidate_decision=cdec, triggers=trig,
                                journal_events=events, errors=errors, now=now)
    payload["artifacts"] = {
        "active_trade": plan_status,
        "position_decision": pdec_status,
        "active_candidate": acand_status,
        "candidate_decision": cdec_status,
        "triggers": trig_status,
        "journal_events": len(events),
    }
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    ctx = p.get("context") or {}
    lines = [f"# 0DTE loop status: **{p.get('state')}**  ({p.get('loop_stage')})",
             "",
             f"Next: **{p.get('next_command')}** — {p.get('next_action')}  ",
             f"Executable: {p.get('executable')}  ·  places orders: {p.get('places_orders')}  ",
             ""]
    if p.get("reasons"):
        lines += ["## Why", *[f"- {r}" for r in p["reasons"]], ""]
    if ctx:
        lines += ["## Context", *[f"- {k}: {v}" for k, v in ctx.items()], ""]
    arts = p.get("artifacts")
    if arts:
        lines += ["## Artifacts", *[f"- {k}: {v}" for k, v in arts.items()]]
    return "\n".join(lines)
