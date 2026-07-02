"""0DTE loop status — PURE/OFFLINE state machine over the canonical artifacts. NO broker calls/LLM/orders.

One read-only surface that tells the live controller (Hermes/MCP) WHERE in the loop it is and WHICH
command runs next, so the execution loop reads as one obvious cycle instead of a pile of independent
tools:

    SCAN → CANDIDATE → GATED → PROMOTED → ENTERED → MANAGING → EXITED → REVIEWED   (DEGRADED on a fault)

It does NOT re-run any gate or re-derive any decision — it only SUMMARIZES artifacts the other 0DTE
tools already wrote under ``data/odte/`` and points at the next command. The hard rule mirrors the
rest of the layer: a LIVE position (or a fault on one) always outranks the scan/candidate/gate lane —
managing/exiting an open trade beats chasing a new one. Reads, never writes; places NO orders.

For the cron/Telegram lane it also emits, on top of the fine-grained state:
  * ``posture``     — one coarse word for what the controller should DO this tick, drawn from
                      {MANAGE_POSITION, SCOUT_FRESH_SETUP, WAIT_FRESH_CONFIRMATION,
                       FLAT_NO_TRADE, STALE_DATA_BLOCKED, BROKER_DEGRADED}. ``FLAT_NO_TRADE`` is the
                      normal idle state (flat / reviewed / no candidate / empty heartbeat) and does
                      NOT contain the word "stale". ``STALE_DATA_BLOCKED`` is reserved for an actual
                      stale/malformed artifact that is BLOCKING or diagnosing action (e.g. an open
                      position whose live read went stale) — never a leftover ignored after a review.
                      This is what stops a cron run from narrating fresh live edge, and stops a plain
                      idle tick from being mislabelled as a stale-data problem.
  * ``broker_lane`` — a normalization of a SUPPLIED/PROBED broker-health payload (Hermes feeds the
                      MCP/CLI probe result; this module never touches the broker): ok / down / stale /
                      read_only_fallback / unknown, plus whether live orders are permitted. When the
                      place lane is down but a local read-only probe is fresh, the loop says
                      read_only_fallback and BLOCKS live orders until the MCP place/review lane is back.
  * ``artifact_ages`` — as-of timestamp, age, TTL and fresh flag for triggers / candidate / gate /
                      position_decision / active_trade, so staleness is visible, not guessed.

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
# Durable broker-lane handoff: cron/Hermes writes this each tick from its MCP/CLI/read-only probes
# (canonical schema in classify_broker_lane / broker_health.example.json). loop-status auto-folds it
# so the lane truth rides in a file, not ad-hoc chat prose. A stale file fails closed (orders blocked).
BROKER_HEALTH_FILENAME = "broker_health.json"

# Ordered loop states (scan → review), plus the DEGRADED fault state.
LOOP_STATES = ("SCAN", "CANDIDATE", "GATED", "PROMOTED", "ENTERED", "MANAGING",
               "EXITED", "REVIEWED", "DEGRADED")

# A plan whose status is one of these is NOT a live position (mirrors odte_position._INACTIVE_STATUS).
_INACTIVE_STATUS = {"closed", "exited", "flat", "done"}
# position_decision.decision values that mean "no live position to manage".
_NO_POSITION_DECISIONS = {"", "NO_POSITION", "RESTRICTED"}

# A live-position decision older than this (when a wall clock is supplied) is treated as flying blind.
# This must stay tight for 0DTE scalp HAWK mode: the live controller should poll a held contract
# around every 10 seconds inside the holding branch, so a >1-minute-old management decision means
# several checks were missed or the feed/cron is degraded.
STALE_DECISION_MINUTES = 1
# A closed trade older than this stops nagging EXITED so it can't mask a genuinely fresh scan.
STALE_TRADE_HOURS = 36
# A FAILED entry gate (present, not execution-allowed) stops being sticky once a fresh scan candidate
# lands at least this many minutes after it: on a real trend day the tape keeps moving, so an old
# NO_TRADE/deny gate must not pin the loop at GATED forever when a materially newer watchdog candidate
# shows up. Only DENIED gates are superseded — an execution-allowed (PROMOTED) gate is never demoted
# by a scan, and the candidate this falls through to stays scan_only/observe (never executable).
SUPERSEDE_GATE_MINUTES = 15
# Pre-entry artifacts are disposable, not durable authority. Old candidates/gates must never resurrect
# after a trade has closed/reviewed or after the tape has moved on.
STALE_ENTRY_GATE_MINUTES = 15
STALE_CANDIDATE_MINUTES = 10
STALE_TRIGGER_MINUTES = 30
# A broker-health probe older than this can't be called live truth: the chat-lane MCP can report
# "transport down" while a CLI probe from minutes ago read healthy, so an aged payload is treated as
# stale (orders blocked) rather than fresh confirmation that the place lane is up.
BROKER_STALE_MINUTES = 5

# Coarse cron-facing posture — one word for what the controller should DO this tick. Layered on top of
# the fine-grained loop state so a Telegram update reads as an action, not observe-only boilerplate.
POSTURES = ("MANAGE_POSITION", "SCOUT_FRESH_SETUP", "WAIT_FRESH_CONFIRMATION",
            "FLAT_NO_TRADE", "STALE_DATA_BLOCKED", "BROKER_DEGRADED")
# Normalized broker-lane labels for a supplied/probed health payload (this module makes NO broker call).
BROKER_LANES = ("ok", "read_only_fallback", "down", "stale", "unknown")

# Housekeeping/audit breadcrumbs. The resolver appends these when it IGNORES a disposable artifact
# (stale/denied/consumed pre-entry gate, empty scan, closed+reviewed trade). They are the audit trail,
# NOT user-facing "why" — a flat cron tick must not read like it has stale actionable work. These move
# to ``notes``; the loud live-position degrades (position_decision stale/MONITORING_DEGRADED/malformed)
# are deliberately NOT here, so a real risk on an open trade still shouts in ``reasons``.
_REASON_NOISE_MARKERS = (
    "ignored stale", "ignored consumed", "denied entry gate not sticky",
    "unreviewed but stale", "no actionable candidate", "closed and reviewed",
)


def _is_reason_noise(reason: str) -> bool:
    return any(marker in reason for marker in _REASON_NOISE_MARKERS)

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


def _event_ts(payload: dict) -> datetime | None:
    """Best-effort timestamp for disposable pre-entry artifacts."""
    p = _dict(payload)
    for key in ("ts", "generated_at", "updated_at", "created_at"):
        dt = _parse_ts(p.get(key))
        if dt is not None:
            return dt
    return None


def _fresh_after_close(payload: dict, *, now: datetime | None, close_dt: datetime | None,
                       ttl_minutes: int) -> tuple[bool, str | None]:
    """Whether a pre-entry artifact is fresh enough to act on. Missing ts is allowed only when no
    wall-clock/close reference was supplied (old tests/fixtures); otherwise it is treated as stale."""
    if not payload:
        return False, None
    ts = _event_ts(payload)
    if ts is None:
        # Legacy fixtures and some scan artifacts have no timestamp; allow them unless a closed trade
        # creates a concrete cycle boundary. New TTL behavior applies when timestamps are present.
        if close_dt is None:
            return True, None
        return False, "missing timestamp"
    if close_dt is not None and ts <= close_dt:
        return False, "older than closed trade"
    if now is not None:
        age = (now - ts).total_seconds() / 60.0
        if age > ttl_minutes:
            return False, f"stale ({age:.0f}m > {ttl_minutes}m)"
    return True, None


def classify_broker_lane(broker_health: dict | None, now: datetime | None = None) -> dict:
    """Normalize a SUPPLIED/PROBED broker-health payload into a lane label + live-order permission.

    This module never touches the broker. Hermes/MCP (or a local read-only probe) writes the health
    payload and feeds it in; here we only classify it so the loop can't narrate a dead/stale broker
    lane as fresh, and so live orders are blocked whenever the place/review lane isn't provably up.
    An absent/empty payload is ``unknown`` — pure offline mode, behavior unchanged (``orders_ok`` None).

    The PARENT Robinhood review/place lane is AUTHORITATIVE for live orders. The split-brain case is
    explicit: a CLI ``hermes mcp test robinhood`` that connects, or a local read-only probe that reads
    positions/BP/quotes, verifies FLATNESS but does NOT authorize order review/place — so parent DOWN +
    CLI CONNECTED is ``read_only_fallback`` (orders BLOCKED), not ``ok``.

    Canonical cron handoff (see ``data/odte/broker_health.example.json``):
      parent_robinhood_mcp        'OK' | 'DOWN' | 'UNKNOWN'  — the authoritative review/place lane
      cli_mcp_test                'CONNECTED' | 'DOWN'       — `hermes mcp test robinhood` (read-capable)
      fallback                    'OK'|'DOWN' | bool         — local read-only probe status
      live_review_place_allowed   bool                       — master switch: may live orders be placed?
      as_of / source              timestamp / origin         — payload freshness + who wrote it
    Also tolerant of the ad-hoc shapes the chat/CLI lanes emit:
      lane / mcp / status         'ok'|'up'|'healthy'|'live' · 'down'|'unavailable'|'transport_down'
      place_lane / execution_lane 'ok' | 'blocked'
      read_only_fallback / read_only · blocked · live_orders_allowed · mcp_ok / ok / transport
    An absent/empty payload is ``unknown`` — pure offline mode, behavior unchanged (``orders_ok`` None).
    """
    b = _dict(broker_health)
    as_of = b.get("as_of") or b.get("ts") or b.get("checked_at")
    age = _age_minutes(as_of, now)
    source = b.get("source")

    def out(lane: str, orders_ok: bool | None, note: str) -> dict:
        return {"lane": lane, "orders_ok": orders_ok, "as_of": as_of,
                "age_minutes": round(age, 1) if age is not None else None,
                "source": source, "note": note}

    if not b:
        return out("unknown", None, "no broker-health payload supplied")

    # --- canonical handoff signals (authoritative) -------------------------------------------
    parent = str(b.get("parent_robinhood_mcp") or "").strip().lower()
    cli = str(b.get("cli_mcp_test") or "").strip().lower()
    fb = b.get("fallback")
    fallback_ok = (fb is True or str(fb).strip().lower()
                   in ("ok", "connected", "healthy", "up", "read_only", "read-only"))
    review_place = b.get("live_review_place_allowed")   # bool | None (master order switch)
    parent_ok = parent == "ok"
    parent_down = parent in ("down", "unavailable", "offline", "error")
    cli_connected = cli in ("connected", "ok", "up")

    # --- ad-hoc shapes -----------------------------------------------------------------------
    lane_hint = str(b.get("lane") or b.get("mcp") or b.get("status") or "").strip().lower()
    place = str(b.get("place_lane") or b.get("execution_lane") or "").strip().lower()
    transport = str(b.get("transport") or "").strip().lower()
    read_only = (b.get("read_only_fallback") is True or b.get("read_only") is True
                 or lane_hint in ("read_only", "read-only", "read_only_fallback"))
    blocked = (b.get("blocked") is True or place == "blocked"
               or b.get("live_orders_allowed") is False or read_only)
    down = (lane_hint in ("down", "unavailable", "error", "transport_down", "closed")
            or transport in ("down", "closed", "error")
            or b.get("mcp_ok") is False or b.get("ok") is False)
    healthy = (lane_hint in ("ok", "up", "healthy", "live")
               or b.get("mcp_ok") is True or b.get("ok") is True)

    # Can a LIVE order be placed? Parent review/place lane is authoritative; the master switch and any
    # explicit block override everything. Never infer place permission from a CLI/read-only lane.
    if review_place is False or blocked or parent_down:
        place_allowed = False
    elif review_place is True:
        place_allowed = True
    elif parent:                       # parent stated but not "ok" (e.g. UNKNOWN) → can't confirm place
        place_allowed = parent_ok
    else:                              # no canonical parent field → fall back to ad-hoc health
        place_allowed = healthy and not blocked
    # Can we at least READ/verify (flatness, BP, positions, quotes)?
    read_capable = read_only or cli_connected or fallback_ok or healthy or parent_ok

    # A probe older than the TTL is not live truth — block orders regardless of what it claimed.
    if age is not None and age > BROKER_STALE_MINUTES:
        return out("stale", False, f"broker-health probe {age:.0f}m old (> {BROKER_STALE_MINUTES}m)")
    if place_allowed:
        return out("ok", True, "parent review/place lane healthy — live orders permitted")
    if read_capable:
        why = ("parent review/place lane DOWN" if parent_down or parent else
               "place/review lane down")
        return out("read_only_fallback", False,
                   f"{why}; CLI/read-only lane fresh — verify flatness/BP only, live orders BLOCKED "
                   "until parent MCP place lane is healthy")
    if down or parent_down:
        return out("down", False, "broker/MCP transport down — live orders BLOCKED")
    return out("unknown", None, "broker-health payload inconclusive")


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


def _same_underlying(a: dict, b: dict) -> bool:
    """Best-effort symbol match for journal events that use underlying/symbol inconsistently."""
    av = str(a.get("underlying") or a.get("symbol") or "").upper()
    bv = str(b.get("underlying") or b.get("symbol") or "").upper()
    return bool(av and bv and av == bv)


# A gate decision containing any of these is a NO-TRADE verdict, not a promotable gate. Such a gate
# must NEVER pin the loop at GATED telling the cron to "promote" it — a veto/deny already said no, so
# it is consumed for loop progression and the loop falls through to a fresh candidate or SCAN. (An
# execution-allowed gate is still never demoted by this — see the `execution_allowed` guard below.)
# Substring match so verbose real decisions like "NO_TRADE_FINAL_TAPE_RECLAIM_FAILED" are caught.
_DENY_TOKENS = ("deny", "denied", "veto", "no_trade", "no-trade", "notrade", "no_go", "no-go",
                "reject", "block", "abort", "scratch", "invalid")
_INACTIVE_WATCH_DECISIONS = {"", "NO_CANDIDATE", "DEGRADED_NO_TRADE", "EXPIRED_NO_CONFIRMATION"}
_INERT_GATE_DECISIONS = {"", "observe", "skip", "wait", "watch", "hold", "no_candidate"}


def _gate_is_denied(decision: Any) -> bool:
    d = str(decision or "").strip().lower()
    return bool(d) and any(tok in d for tok in _DENY_TOKENS)


def _gate_is_inert_observe(gate: dict) -> bool:
    """True for scan-only/observe gates that are audit trail only, not current loop state."""
    if not gate or gate.get("execution_allowed") is True:
        return False
    decision = str(gate.get("decision") or gate.get("intent") or "").strip().lower()
    return bool(gate.get("scan_only") is True or decision in _INERT_GATE_DECISIONS)


def _watch_is_inactive(watch: dict) -> bool:
    """Inactive candidate-watch files should not keep showing as stale live artifacts."""
    if not watch:
        return True
    decision = str(watch.get("decision") or "").strip().upper()
    state = str(watch.get("state") or "").strip().upper()
    candidate = _dict(watch.get("candidate"))
    has_symbol = bool(watch.get("ticker") or watch.get("underlying") or watch.get("symbol")
                      or candidate.get("ticker") or candidate.get("underlying") or candidate.get("symbol"))
    return state == "INACTIVE" or decision in _INACTIVE_WATCH_DECISIONS or not has_symbol


def _trigger_is_inactive(trigger: dict) -> bool:
    """Empty/no-alert scan snapshots are heartbeat data, not stale actionable artifacts."""
    if not trigger:
        return True
    candidate = _dict(trigger.get("candidate"))
    has_candidate = bool(candidate.get("ticker") or candidate.get("underlying") or candidate.get("symbol"))
    has_trigger_rows = bool(trigger.get("triggers"))
    return not bool(trigger.get("alert")) and not has_candidate and not has_trigger_rows


def _gate_consumed_by_no_trade(gate: dict, events: list[dict]) -> bool:
    """Return True when a same-symbol no-trade event has already acknowledged a non-executable gate.

    Entry gates are disposable pre-entry artifacts. The journal keeps the audit trail, but loop-status
    should not keep surfacing the same scan-only/denied gate every cron tick after the controller has
    already journaled the matching no-trade decision.
    """
    if not gate or gate.get("execution_allowed") is True:
        return False
    gate_ts = _parse_ts(gate.get("ts"))
    gate_seq = gate.get("seq") if isinstance(gate.get("seq"), (int, float)) else None
    for i, ev in enumerate(events or []):
        if not isinstance(ev, dict) or ev.get("event_type") != "no_trade_decision":
            continue
        if not _same_underlying(gate, ev):
            continue
        ev_ts = _parse_ts(ev.get("ts") or ev.get("timestamp") or ev.get("generated_at"))
        if gate_ts is not None and ev_ts is not None:
            if ev_ts >= gate_ts:
                return True
            continue
        ev_seq = ev.get("seq") if isinstance(ev.get("seq"), (int, float)) else float(i)
        if gate_seq is not None and ev_seq >= gate_seq:
            return True
    return False


def _resolve_loop_state(active_trade: dict | None = None,
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
    # a gate dated at/before the last close is the closed trade's own gate, already consumed. Even an
    # execution_allowed gate is disposable after its TTL; never resurrect stale promotion authority.
    gate = _latest_event(events, "entry_decision") or {}
    gate_ts = _parse_ts(gate.get("ts")) if gate else None
    close_dt = _parse_ts(close_ts)
    gate_fresh, gate_stale_reason = _fresh_after_close(
        gate, now=now, close_dt=close_dt if closed_trade else None,
        ttl_minutes=STALE_ENTRY_GATE_MINUTES)
    if gate and not gate_fresh and gate_stale_reason and not _gate_is_inert_observe(gate):
        reasons.append(f"ignored stale entry gate: {gate_stale_reason}")
    if gate_fresh and _gate_consumed_by_no_trade(gate, events):
        gate_fresh = False
        reasons.append("ignored consumed entry gate: matching no-trade decision already journaled")
    # A denied/veto gate is a terminal NO-TRADE verdict for its cycle, NOT a promotable gate. It must
    # not pin the loop at GATED and tell the cron to re-promote an old veto (the stale-sidelines bug).
    # Consume it here so the loop falls through to a genuinely fresh candidate or SCAN. Execution-
    # allowed gates are exempt — a granted gate is never demoted by a stale-verb heuristic.
    if gate_fresh and gate.get("execution_allowed") is not True and _gate_is_denied(gate.get("decision")):
        gate_fresh = False
        # In a flat/reviewed cycle this is audit history, not current work. Keep the noisy diagnostic for
        # active scan/gate resolution tests, but do not leak it into the normal closed/reviewed cron UX.
        if not reviewed:
            reasons.append(f"denied entry gate not sticky (decision={gate.get('decision')}) — "
                           "expired to fresh candidate/scan")
    # PROMOTED requires EXPLICIT gate permission: execution_allowed is True. A bare decision=="enter"
    # with execution_allowed missing/false is a stale/partial record and stays GATED — the loop only
    # advances on explicit manager promotion / gate permission, never on the intent verb alone.
    gate_exec = gate_fresh and gate.get("execution_allowed") is True

    candidate = _dict(trig.get("candidate"))
    trig_fresh, trig_stale_reason = _fresh_after_close(
        trig, now=now, close_dt=close_dt if closed_trade else None,
        ttl_minutes=STALE_TRIGGER_MINUTES)
    trig_candidate = bool(candidate.get("ticker")) and not candidate.get("restricted") and trig_fresh
    if candidate.get("ticker") and not trig_fresh and trig_stale_reason:
        reasons.append(f"ignored stale scan trigger: {trig_stale_reason}")
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
    watch_payload = cdec or acand
    watch_fresh, watch_stale_reason = _fresh_after_close(
        watch_payload, now=now, close_dt=close_dt if closed_trade else None,
        ttl_minutes=STALE_CANDIDATE_MINUTES)
    if watch_live and not watch_fresh and watch_stale_reason:
        reasons.append(f"ignored stale candidate watch: {watch_stale_reason}")
    inactive_watch = {"DEGRADED_NO_TRADE", "EXPIRED_NO_CONFIRMATION", ""}
    if watch_live and watch_fresh and watch_decision not in inactive_watch:
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
        "live": live,
        "next_action": action,
        "next_command": command,
        "reasons": reasons,
        "context": {k: v for k, v in context.items() if v is not None},
        "places_orders": False,
    }


def _posture(payload: dict, broker: dict, *, live_mode: bool = False) -> tuple[str, str | None, str | None]:
    """Coarse cron posture over the resolved loop state + broker lane. Pure; returns
    (posture, next_action override or None, next_command override or None).

    Precedence encodes the 0DTE rules: broker truth first, manage an open position before scanning,
    never advertise fresh live edge on stale/absent data, and never call a plain idle tick "stale".
      1. BROKER can't back a needed live order → BROKER_DEGRADED (top-line). That means either an
         active lane FAULT (down / stale / read-only-fallback → ``orders_ok is False``), OR — in
         ``live_mode`` — a live position / ready setup while the parent lane is NOT a fresh confirmed
         OK (``orders_ok`` not True: fail closed on unknown/missing broker_health). A flat tick with
         an unknown lane is NOT degraded — there is nothing to authorize.
      2. A live position with an authorized lane → MANAGE_POSITION (manage before scan).
      3. A live position whose live read is stale/malformed (broker fine) → STALE_DATA_BLOCKED and
         the reasons SHOUT — the controller is holding blind and must refresh the position read now.
      4. A fresh, confirmed/promoted setup → SCOUT_FRESH_SETUP (the ONLY path to SCOUT).
      5. A candidate/gate awaiting confirmation → WAIT_FRESH_CONFIRMATION.
      6. A non-live malformed/stale live artifact → STALE_DATA_BLOCKED (diagnostic).
      7. SCAN / EXITED / REVIEWED with nothing fresh (empty heartbeat, reviewed trade, ignored stale
         leftovers) → FLAT_NO_TRADE. This is the normal idle state and is NEVER called "stale".
    """
    state = payload.get("state")
    live = bool(payload.get("live"))
    executable = bool(payload.get("executable"))
    ctx = payload.get("context") or {}
    lane = broker.get("lane")
    orders_ok = broker.get("orders_ok")            # True only if parent lane is a fresh confirmed OK
    broker_faulted = lane in ("down", "stale", "read_only_fallback")
    needs_live_lane = live or state == "PROMOTED" or executable or ctx.get("confirmed") is True
    # Fail closed: an active fault always blocks; in live_mode an unconfirmed lane (unknown/missing)
    # cannot authorize a live order either. A flat tick (no live lane needed) is never blocked here.
    broker_blocks = broker_faulted or (live_mode and needs_live_lane and orders_ok is not True)
    verify = "hermes mcp test robinhood  # repair parent review/place lane before any live order"

    # 1. Broker can't back a needed live order — top-line BROKER_DEGRADED.
    if broker_blocks:
        if live and state in ("MANAGING", "ENTERED", "DEGRADED"):
            return ("BROKER_DEGRADED",
                    "OPEN position but broker place/review lane can't back an exit — reconcile via "
                    "read lane; do NOT assume fills, block live orders", verify)
        if state == "PROMOTED" or ctx.get("confirmed") is True:
            return ("BROKER_DEGRADED",
                    f"fresh setup ready but broker lane '{lane}' can't authorize a live order — "
                    "read-only state only, block orders until parent MCP place lane is a fresh OK", verify)
        return ("BROKER_DEGRADED",
                f"broker place/review lane '{lane}' — live orders BLOCKED; verify parent MCP before acting",
                verify)

    # 2. Live position lane: manage before scan.
    if live and state in ("MANAGING", "ENTERED"):
        return ("MANAGE_POSITION", None, None)
    # 3. Live position we can't value — broker is fine, so this is a STALE live read, not a broker fault.
    if state == "DEGRADED" and live:
        return ("STALE_DATA_BLOCKED",
                "OPEN position but its live management read is STALE/malformed — refresh the position "
                "decision NOW before it drifts; do not act on the old read", None)

    # 4. Fresh, actionable setup ready to act on — the ONLY path to SCOUT_FRESH_SETUP.
    if state == "PROMOTED" or ctx.get("confirmed") is True:
        return ("SCOUT_FRESH_SETUP", None, None)

    # 5. Candidate / gate on the board, awaiting confirmation (fresh but not yet actionable).
    if state in ("CANDIDATE", "GATED"):
        return ("WAIT_FRESH_CONFIRMATION", None, None)

    # 6. Non-live malformed/stale live artifact — a real diagnostic block, not idle.
    if state == "DEGRADED":
        return ("STALE_DATA_BLOCKED",
                "a live artifact is malformed/stale — inspect and refresh it before acting", None)

    # 7. SCAN / EXITED / REVIEWED: genuinely flat — empty heartbeat / reviewed trade / no fresh setup.
    # Any stale candidate/gate here was ignored as leftover cruft, NOT blocking action, so this is a
    # plain idle no-trade tick — never mislabelled "stale".
    return ("FLAT_NO_TRADE", None, None)


def derive_loop_state(active_trade: dict | None = None,
                      position_decision: dict | None = None,
                      active_candidate: dict | None = None,
                      candidate_decision: dict | None = None,
                      triggers: dict | None = None,
                      journal_events: list[dict] | None = None,
                      *, errors: set[str] | None = None,
                      broker_health: dict | None = None,
                      live_mode: bool = False,
                      now: datetime | None = None,
                      stale_decision_minutes: int = STALE_DECISION_MINUTES) -> dict:
    """Resolve the loop state and layer the cron-facing posture + broker lane on top. Never raises.

    ``_resolve_loop_state`` is the pure fine-grained resolver; here we classify a SUPPLIED/PROBED
    broker-health payload (no broker call) and reduce state+broker to one ``posture`` word, forcing
    ``executable`` off whenever the broker lane can't back a live order. ``live_mode`` (the cron/live
    controller) fails closed: a live position / ready setup with an unknown or missing broker_health
    reads BROKER_DEGRADED, so no live order is ever authorized without a fresh confirmed parent lane."""
    payload = _resolve_loop_state(active_trade=active_trade, position_decision=position_decision,
                                  active_candidate=active_candidate, candidate_decision=candidate_decision,
                                  triggers=triggers, journal_events=journal_events, errors=errors,
                                  now=now, stale_decision_minutes=stale_decision_minutes)
    broker = classify_broker_lane(broker_health, now)
    posture, action_override, command_override = _posture(payload, broker, live_mode=live_mode)
    payload["posture"] = posture
    payload["broker_lane"] = broker
    if action_override:
        payload["next_action"] = action_override
    if command_override:
        payload["next_command"] = command_override
    # A degraded broker lane must never advertise executable live edge.
    if posture == "BROKER_DEGRADED":
        payload["executable"] = False
    # Split the audit trail out of the user-facing "why": housekeeping breadcrumbs (ignored stale/
    # denied/consumed gates, empty scans, closed+reviewed trades) move to `notes` so a flat cron tick
    # doesn't read like it holds stale actionable work. Loud live-position degrades stay in `reasons`.
    full = payload.get("reasons") or []
    payload["reasons"] = [r for r in full if not _is_reason_noise(r)]
    payload["notes"] = [r for r in full if _is_reason_noise(r)]
    if not payload["reasons"]:
        payload["reasons"] = [_flat_reason(posture)]
    return payload


def _flat_reason(posture: str) -> str:
    """A single clean user-facing line when the only reasons were housekeeping noise."""
    return {
        "FLAT_NO_TRADE": "flat — no fresh actionable setup",
        "STALE_DATA_BLOCKED": "stale/malformed artifact blocking action — refresh before acting",
        "SCOUT_FRESH_SETUP": "fresh setup ready — re-validate live before entry",
        "WAIT_FRESH_CONFIRMATION": "candidate on the board — awaiting confirmation",
        "MANAGE_POSITION": "live position — manage/exit before scanning",
        "BROKER_DEGRADED": "broker lane degraded — live orders blocked",
    }.get(posture, "no fresh actionable setup")


def _age_entry(ts: Any, now: datetime | None, ttl_minutes: int) -> dict:
    """as-of / age / TTL / fresh triple for one artifact, so staleness is visible not guessed."""
    dt = _parse_ts(ts)
    age = _age_minutes(dt, now) if dt is not None else None
    return {"as_of": dt.isoformat(timespec="seconds") if dt is not None else None,
            "age_minutes": round(age, 1) if age is not None else None,
            "ttl_minutes": ttl_minutes,
            "fresh": (age <= ttl_minutes) if age is not None else None}


def run_loop_status(state_dir: str | None = None, *, broker_health: dict | None = None,
                    broker_health_path: str | None = None, live_mode: bool = False,
                    now: datetime | None = None) -> dict:
    """Read the canonical data/odte artifacts and resolve the loop state + posture + broker lane.

    Makes NO broker call: ``broker_health`` (dict) or ``broker_health_path`` (a JSON file Hermes wrote
    from an MCP/CLI probe), else the ``ODTE_BROKER_HEALTH`` env path, else the durable canonical
    ``data/odte/broker_health.json``, supplies the lane truth. ``live_mode`` (the cron/live controller)
    fails closed on unknown/missing broker_health so no live order is authorized without a fresh parent
    OK. Reuses odte_watchdog._read_json (status-aware: ok|missing|invalid) and odte_journal.read_events
    (skips malformed lines, never raises): a missing artifact reads as SCAN and a malformed live
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

    broker_health_status = "supplied" if broker_health is not None else "missing"
    if broker_health is None:
        # Precedence: explicit path/env → durable canonical file. Any wins folds the lane in.
        health_path = broker_health_path or os.environ.get("ODTE_BROKER_HEALTH")
        health_file = Path(os.path.expanduser(health_path)) if health_path else base / BROKER_HEALTH_FILENAME
        bh, bh_status = _read_json(health_file)
        broker_health_status = bh_status
        broker_health = bh if bh_status == "ok" else None

    errors = {name for name, st in (("active_trade", plan_status),
                                    ("position_decision", pdec_status),
                                    ("active_candidate", acand_status),
                                    ("candidate_decision", cdec_status),
                                    ("triggers", trig_status)) if st == "invalid"}
    payload = derive_loop_state(active_trade=plan, position_decision=pdec,
                                active_candidate=acand, candidate_decision=cdec, triggers=trig,
                                journal_events=events, errors=errors,
                                broker_health=broker_health, live_mode=live_mode, now=now)
    payload["artifacts"] = {
        "active_trade": plan_status,
        "position_decision": pdec_status,
        "active_candidate": acand_status,
        "candidate_decision": cdec_status,
        "triggers": trig_status,
        "broker_health": broker_health_status,
        "journal_events": len(events),
    }
    gate = _latest_event(events, "entry_decision") or {}
    trig_d, pdec_d, plan_d = _dict(trig), _dict(pdec), _dict(plan)
    watch = _dict(cdec) or _dict(acand)
    # A gate is "current work" (worth an age) ONLY when it actually drives the resolved state —
    # GATED (pending promotion) or PROMOTED (executable). A denied/consumed/stale/observe gate that
    # the resolver ignored is audit trail, not current work, so its age is nulled (never "stale").
    gate_current = payload.get("state") in ("GATED", "PROMOTED")
    gate_display_ts = gate.get("ts") if (gate_current and not _gate_is_inert_observe(gate)) else None
    trigger_display_ts = None if _trigger_is_inactive(trig_d) else (trig_d.get("ts") or trig_d.get("generated_at"))
    watch_display_ts = None if _watch_is_inactive(watch) else _event_ts(watch)
    pdec_display_ts = None if str(pdec_d.get("decision") or "").strip().upper() in _NO_POSITION_DECISIONS else pdec_d.get("ts")
    payload["artifact_ages"] = {
        "triggers": _age_entry(trigger_display_ts, now, STALE_TRIGGER_MINUTES),
        "candidate": _age_entry(watch_display_ts, now, STALE_CANDIDATE_MINUTES),
        "gate": _age_entry(gate_display_ts, now, STALE_ENTRY_GATE_MINUTES),
        "position_decision": _age_entry(pdec_display_ts, now, STALE_DECISION_MINUTES),
        "active_trade": _age_entry(
            plan_d.get("updated_at") or plan_d.get("closed_at") or plan_d.get("exit_fill_time")
            or plan_d.get("entry_fill_time"), now, STALE_TRADE_HOURS * 60),
    }
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    ctx = p.get("context") or {}
    posture = p.get("posture")
    header = f"# 0DTE loop status: **{posture or p.get('state')}**"
    if posture:
        header += f"  ·  {p.get('state')} ({p.get('loop_stage')})"
    else:
        header += f"  ({p.get('loop_stage')})"
    lines = [header, "",
             f"Next: **{p.get('next_command')}** — {p.get('next_action')}  ",
             f"Executable: {p.get('executable')}  ·  places orders: {p.get('places_orders')}  "]
    broker = p.get("broker_lane") or {}
    if broker.get("lane") and broker.get("lane") != "unknown":
        orders = broker.get("orders_ok")
        flag = "live orders OK" if orders is True else "live orders BLOCKED" if orders is False else "orders unknown"
        lines.append(f"Broker lane: **{broker.get('lane')}** ({flag}) — {broker.get('note')}  ")
    lines.append("")
    if p.get("reasons"):
        lines += ["## Why", *[f"- {r}" for r in p["reasons"]], ""]
    if ctx:
        lines += ["## Context", *[f"- {k}: {v}" for k, v in ctx.items()], ""]
    ages = p.get("artifact_ages")
    if ages:
        lines += ["## Freshness"]
        for k, meta in ages.items():
            if not meta.get("as_of"):
                lines.append(f"- {k}: —")
                continue
            fresh = meta.get("fresh")
            tag = "fresh" if fresh else "STALE" if fresh is False else "?"
            lines.append(f"- {k}: {meta.get('age_minutes')}m old ({tag}, ttl {meta.get('ttl_minutes')}m)")
        lines.append("")
    arts = p.get("artifacts")
    if arts:
        lines += ["## Artifacts", *[f"- {k}: {v}" for k, v in arts.items()]]
    return "\n".join(lines)
