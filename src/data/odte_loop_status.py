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
                      {MANAGE_POSITION, EXECUTION_READY, SCOUT_FRESH_SETUP, WAIT_FRESH_CONFIRMATION,
                       FLAT_NO_TRADE, STALE_DATA_BLOCKED, BROKER_DEGRADED}. ``EXECUTION_READY`` means
                      an execution-allowed gate is on the board: proceed to broker review/place under
                      standing auth (via the Hermes MCP lane — this repo still places NO orders).
                      ``SCOUT_FRESH_SETUP`` is a confirmed candidate: assemble/promote a fresh entry
                      gate now. ``FLAT_NO_TRADE`` is the normal idle state (flat / reviewed / no
                      candidate / empty heartbeat) and does NOT contain the word "stale".
                      ``STALE_DATA_BLOCKED`` is reserved for an actual stale/malformed artifact that
                      is BLOCKING or diagnosing action (an open position whose live read went stale,
                      or a stale broker_health.json probe FILE while a live lane is needed — the fix
                      there is the exact refresh command, not "assume the broker is down") — never a
                      leftover ignored after a review. ``BROKER_DEGRADED`` is a CONFIRMED broker
                      fault (down / read-only fallback / unconfirmable in live mode), not a stale
                      probe file. This is what stops a cron run from narrating fresh live edge, and
                      stops a plain idle tick from being mislabelled as a stale-data problem.
  * ``broker_lane`` — a normalization of a SUPPLIED/PROBED broker-health payload (Hermes feeds the
                      MCP/CLI probe result; this module never touches the broker): ok / down / stale /
                      read_only_fallback / unknown, plus whether live orders are permitted. When the
                      place lane is down but a local read-only probe is fresh, the loop says
                      read_only_fallback and BLOCKS live orders until the MCP place/review lane is back.
                      When the probe carries POSITION/ORDER/BP counts (``truth``), BROKER TRUTH WINS
                      over lagging local artifacts (2026-07-16 artifact miss): a fresh probe with ≥1
                      open option position while the local artifacts read flat/candidate reconciles
                      to MANAGING; a fresh flat probe (0 positions AND 0 open orders) while the local
                      plan still reads live reconciles to EXITED (the trade already closed at the
                      broker — a net-green low-BP stop is CLEAN, never a fault). The reconciled
                      ``next_action`` names the exact files to rewrite; this module itself stays
                      read-only and never invents fills.
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
  GATED      an entry-gate record exists but is NOT execution-allowed — refresh inputs, then lease
  PROMOTED   an entry-gate record is execution-allowed — mint a lease, then broker review, then watch
  PENDING_ORDER  an ENTRY order is live at the broker (or must be cancelled) — the order-guard lane
             outranks scan/candidate/gate work entirely; cancel-first on TTL/thesis/incident
  ENTERED    a plan is open but no live decision computed yet — start the position watch
  MANAGING   a live position with a current decision (HOLD or an actionable exit trigger)
  EXITED     the last trade closed and has no postmortem yet — record the review
  REVIEWED   the last trade closed and is reviewed — idle; roll up the journal report
  DEGRADED   a live position can't be valued / a live artifact is malformed or stale — re-establish
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_DATA_DIR
from data.odte_config import CONFIRM_CONVERSION_SLA_SECONDS as _CONVERT_SLA
from data.odte_strategy_policy import ET

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
# Execution-safety artifacts (2026-07-23 delayed-fill remediation): the single-use lease that
# `odte-execution-authorize` minted, and the latest `odte-order-guard` decision. A live/cancel-
# required pending ENTRY order OUTRANKS scanning and new-entry work — cancel first, then cycle.
EXECUTION_LEASE_FILENAME = "execution_lease.json"
ORDER_GUARD_FILENAME = "order_guard_decision.json"

# Ordered loop states (scan → review), plus the DEGRADED fault state. PENDING_ORDER is the
# order-guard lane: an entry order is live at the broker (or needs cancelling) — it outranks the
# whole scan/candidate/gate funnel.
LOOP_STATES = ("SCAN", "CANDIDATE", "GATED", "PROMOTED", "PENDING_ORDER", "ENTERED", "MANAGING",
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
# CONFIRMED-CANDIDATE CONVERSION SLA: an identity-bound CONFIRM_ENTRY must be converted (fresh
# snapshots + `odte-convert`) within this window. loop-status surfaces the countdown so the
# controller sees a deadline, not a suggestion; the aged/unaccounted case fails closed as
# EXECUTION_CONVERSION_FAILURE. 2026-08-02 retune: config-tunable, default 180s — the old 60s
# same-tick SLA was structurally unmeetable at the controller's ~5-minute tick (the 2026-07-31
# conversion measured 73s candidate→gate); conversion itself is now one atomic command.
CONFIRM_CONVERSION_SLA_SECONDS = _CONVERT_SLA
# A broker-health probe older than this can't be called live truth: the chat-lane MCP can report
# "transport down" while a CLI probe from minutes ago read healthy, so an aged payload is treated as
# stale (orders blocked) rather than fresh confirmation that the place lane is up.
BROKER_STALE_MINUTES = 5
# An order-guard decision older than this is not current work: pending 0DTE entry orders live on a
# seconds scale, so an aged guard artifact is ignored (with a note) rather than resurrected.
ORDER_GUARD_STALE_MINUTES = 5
# Guard states that make the order lane CURRENT work (NO_ORDER / FILLED_FRESH fall through: nothing
# pending, or the fill is the position lane's job).
_ACTIVE_GUARD_STATES = ("PENDING_FRESH", "CANCEL_STALE_ENTRY", "CANCEL_THESIS_INVALID",
                        "FILLED_WITHOUT_VALID_LEASE", "BROKER_MISMATCH_BLOCKED")
# The exact fix for a stale broker_health.json — surfaced verbatim so the controller never has to
# guess whether "stale" means the broker is down (it doesn't; it means the probe FILE is outdated).
BROKER_HEALTH_REFRESH_COMMAND = (
    "refresh broker truth: re-probe the parent Robinhood MCP (read account/BP/positions), rewrite "
    "data/odte/broker_health.json (schema: data/odte/broker_health.example.json), then re-run "
    "`make odte-loop-status JSON=1`")

# Coarse cron-facing posture — one word for what the controller should DO this tick. Layered on top of
# the fine-grained loop state so a Telegram update reads as an action, not observe-only boilerplate.
# CANCEL_PENDING_ORDER = a pending entry order must be cancelled NOW (stale lease / dead thesis /
# safety incident); PENDING_ORDER_GUARD = an order is pending inside its lease — poll the guard.
POSTURES = ("MANAGE_POSITION", "EXECUTION_READY", "SCOUT_FRESH_SETUP", "WAIT_FRESH_CONFIRMATION",
            "FLAT_NO_TRADE", "STALE_DATA_BLOCKED", "BROKER_DEGRADED",
            "PENDING_ORDER_GUARD", "CANCEL_PENDING_ORDER", "EXECUTION_CONVERSION_FAILURE",
            "CONVERT_CANDIDATE_NOW")
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
    "PENDING_ORDER": ("order_guard", False),
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
    """Age of `ts` in minutes, or None only when `ts` itself is absent/unparseable.

    `now` defaults to the wall clock rather than collapsing to None (2026-08-10). It used to
    return None when `now` was omitted, which was harmless until `classify_broker_lane` grew a
    fail-closed `age is None -> stale` branch: from then on any caller that dropped `now` got a
    confident "UNDATED / stale" verdict on a perfectly well-dated payload. Production always
    passes `now`, so the live loop was never wrong — but it made the function lie under manual
    diagnosis, which is precisely when a broker-health verdict is being trusted by a human.
    A missing clock is not evidence about the data."""
    dt = _parse_ts(ts)
    if dt is None:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
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


def _count(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if out == out else None   # NaN guard


def _broker_truth(b: dict) -> dict | None:
    """Position/order/BP counts a broker probe may carry (flat, or under an ``account`` block).

    These are the BROKER-TRUTH reconciliation inputs — extracted verbatim from what the probe
    reported, never inferred. Returns None when the payload carries no such counts (a bare lane
    probe), so reconciliation simply stays off."""
    acct = b.get("account") if isinstance(b.get("account"), dict) else {}

    def pick(*keys: str) -> float | None:
        for container in (b, acct):
            for k in keys:
                v = _count(container.get(k))
                if v is not None:
                    return v
        return None

    truth = {
        "open_option_positions": pick("nonzero_option_positions_count",
                                      "open_option_positions_count", "positions_count",
                                      "open_positions_count"),
        "open_option_orders": pick("open_option_orders_count", "open_orders_count"),
        "today_option_orders": pick("today_option_orders_count"),
        "buying_power": pick("buying_power", "options_buying_power", "account_buying_power"),
        "cash": pick("cash"),
    }
    return truth if any(v is not None for v in truth.values()) else None


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
    # Accept every timestamp spelling the writers actually use. `odte_convert._payload_ts` and
    # `odte_execution_policy._payload_ts` already read this wider set, and the controller
    # demonstrably drifts between them — on 2026-08-07 it wrote market_snapshot_live.json with
    # `generated_at` and no `as_of` at all.
    as_of = (b.get("as_of") or b.get("ts") or b.get("checked_at")
             or b.get("generated_at") or b.get("updated_at"))
    age = _age_minutes(as_of, now)
    source = b.get("source")
    truth = _broker_truth(b)

    def out(lane: str, orders_ok: bool | None, note: str) -> dict:
        result = {"lane": lane, "orders_ok": orders_ok, "as_of": as_of,
                  "age_minutes": round(age, 1) if age is not None else None,
                  "source": source, "note": note}
        if truth is not None:
            result["truth"] = truth
        return result

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
    # BUT a stale probe is an outdated FILE, not a confirmed broker fault: keep what the last probe
    # said (`last_known_place_allowed`) and the exact refresh command, so the controller refreshes
    # the file instead of concluding live orders are impossible.
    # UNDATED FAILS CLOSED (2026-08-07). This used to be `age is not None and age > TTL`, so a
    # NON-EMPTY health payload carrying no parseable timestamp skipped the staleness branch entirely
    # and fell through to "live orders permitted" — identical treatment to a one-minute-old probe,
    # with no way to tell how old the truth actually was. An empty payload is still `unknown`
    # (offline mode, handled above); this only covers a payload that claims health without saying
    # when. Every other staleness check on the money path already fails closed on an undated
    # artifact — odte_convert's preflight emits `_undated`, odte_execution_policy states "undated
    # fails closed", and odte_order_guard cancels on an undated lease and raises a safety incident.
    if age is None or age > BROKER_STALE_MINUTES:
        how_old = (f"{age:.0f}m old (> {BROKER_STALE_MINUTES}m)" if age is not None
                   else "UNDATED (no as_of/ts/checked_at/generated_at/updated_at)")
        stale = out("stale", False,
                    f"broker-health probe {how_old} — outdated "
                    "probe FILE, not a confirmed broker fault; refresh it before any live order")
        stale["last_known_place_allowed"] = bool(place_allowed)
        stale["refresh_command"] = BROKER_HEALTH_REFRESH_COMMAND
        return stale
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
        return ("run the candidate HAWK loop WITH ITS INPUTS — pass the candidate from "
                "triggers.json as CANDIDATE= and a FRESH market snapshot as MARKET= (a bare "
                "invocation scans nothing and returns 'no candidate yet'); on CONFIRM_ENTRY "
                "fetch fresh snapshots and run the atomic conversion",
                "make odte-candidate-watch CANDIDATE=<triggers.json candidate> "
                "MARKET=<fresh market.json> JSON=1 WRITE=1 JOURNAL=1 → odte-convert (on CONFIRM_ENTRY)")
    if state == "GATED":
        return ("gate not execution-allowed — refresh the failing/missing inputs and re-run the "
                "atomic conversion (it re-confirms + re-gates + mints the lease under one clock); "
                "never a bare promotion",
                "odte-convert  # after refreshing the failing inputs")
    if state == "PROMOTED":
        return ("ALL GATES PASSED — consume the odte-convert lease IMMEDIATELY at broker "
                "review/place under standing auth via the Hermes MCP lane (this repo places NO "
                "orders); if the lease expired, re-run odte-convert; run the pending-order guard "
                "until the fill, then start the position watch",
                "broker-review-place (Hermes MCP lane) → odte-order-guard → "
                "odte-position --snapshot <live.json>")
    if state == "PENDING_ORDER":
        return ("a pending ENTRY order is under guard — poll fresh broker order truth and re-run "
                "the order guard every tick; cancel IMMEDIATELY on TTL expiry or thesis "
                "invalidation (a stale order is never extended)",
                "odte-order-guard --order <fresh broker order.json>")
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


def _rescan_override(payload: dict, events: list[dict], now: datetime) -> None:
    """While the session is OPEN with entry budget remaining, an idle flat loop must RE-SCAN
    the tape, not re-journal (2026-08-05: 68 consecutive controller ticks obeyed 'roll up the
    journal report' verbatim while bars were fetched and discarded — zero scorer runs after
    10:19). Mutates next_action/next_command in place; the broker-lane fail-closed overrides
    downstream still outrank this, and late-day/no-budget keeps the original instruction."""
    et_now = now.astimezone(ET)
    if et_now.weekday() >= 5:
        return
    session_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    rescan_cutoff = et_now.replace(hour=15, minute=15, second=0, microsecond=0)  # >45m to close
    if not (session_open <= et_now < rescan_cutoff):
        return
    from data.odte_journal import daily_trade_budget, execution_safety_lockout
    # 2026-08-06: the override once told a SAFETY-LOCKED day to re-scan while the reasons
    # enforced the lockout — contradictory instructions in one payload. A locked day never
    # re-scans for entries. (Green-day preservation deliberately does NOT suppress this: the
    # post-green scan lane stays alive by design, tier/BP enforced at the gate.)
    if execution_safety_lockout(events, now=now).get("locked"):
        return
    budget = daily_trade_budget(events, now=now)
    try:
        remaining = int(budget.get("remaining") or 0)
    except (TypeError, ValueError):
        remaining = 0
    aplus_only = False
    if remaining < 1:
        # A+ UNCAPPED (2026-08-06): a net-green day keeps scanning past the base cap, but ONLY
        # a_plus-tier setups can convert (the gate enforces it) — say so explicitly.
        if not budget.get("aplus_uncapped_active"):
            return
        aplus_only = True
    payload["next_action"] = (
        "session OPEN with entry budget remaining — RE-SCAN the tape, do not re-journal: build "
        "a FRESH market snapshot (fast_path_snapshots.md shapes, INCLUDE minutes_to_close), run "
        "odte-day-score, then odte-candidate-watch WITH MARKET= so the tape lane can "
        "manufacture/confirm a candidate (a bare invocation without MARKET= scans nothing)"
        + (" — BUDGET BASE EXHAUSTED on a net-green day: A_PLUS-TIER SETUPS ONLY (the gate "
           "enforces the tier; a net-red day ends entries)" if aplus_only else ""))
    payload["next_command"] = ("make odte-day-score MARKET=<fresh market.json> JSON=1 JOURNAL=1 → "
                               "make odte-candidate-watch MARKET=<fresh market.json> JSON=1 "
                               "WRITE=1 JOURNAL=1")
    payload["rescan_override"] = True


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
    """Return True when a later no-trade event terminally closes the same gate cycle.

    Non-executable gates are disposable by same-symbol no-trade events. An execution-allowed gate is
    stronger authority, so it is consumed only by a no-trade event carrying the exact same non-empty
    candidate fingerprint. This lets authorization/refusal or thesis-invalidation vetoes close a
    promoted cycle without allowing an unrelated same-symbol no-trade event to cancel live authority.
    """
    if not gate:
        return False
    executable = gate.get("execution_allowed") is True
    gate_fingerprint = str(gate.get("candidate_fingerprint") or "").strip()
    if executable and not gate_fingerprint:
        return False
    gate_ts = _parse_ts(gate.get("ts"))
    gate_seq = gate.get("seq") if isinstance(gate.get("seq"), (int, float)) else None
    for i, ev in enumerate(events or []):
        if not isinstance(ev, dict) or ev.get("event_type") != "no_trade_decision":
            continue
        if not _same_underlying(gate, ev):
            continue
        if executable and str(ev.get("candidate_fingerprint") or "").strip() != gate_fingerprint:
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


# Journal event types that can terminally account for a confirmed candidate identity: the entry
# converted (gate/order lane picked it up) or was terminally refused. Deliberately ONLY identity-
# matchable conversion/refusal events — postmortem/execution_safety_incident are day/cycle-scoped
# (handled by the closed-trade and lockout checks), so they must never resolve a specific contract.
_CONFIRM_TERMINAL_EVENTS = ("entry_decision", "no_trade_decision", "order_submitted", "order_filled")


def _live_rails(events: list[dict] | None, broker: dict, now: datetime) -> dict:
    """PURE: the authoritative LIVE rails, attached to every status payload.

    2026-08-03: the controller vetoed a compliant $125 debit against a retired $1.20 flat rail and
    killed a confirm against a hand-remembered 60s SLA while the live SLA was 180s — remembered
    policy, not live policy. This block puts the current numbers in every tick so the agent reads
    them instead of remembering them. Dollar ceilings are computed ONLY from fresh broker-truth
    buying power (never guessed): a stale/absent probe reads null."""
    from data.odte_config import (
        B_PLUS_DEBIT_FRACTION,
        CHASE_BAND_FRACTION,
        EXECUTABLE_UNIVERSE,
        GREEN_REENTRY_AUTO_ARM,
        GREEN_REENTRY_MIN_BP_MULTIPLE,
        MAX_DEBIT_FRACTION,
        SNAPSHOT_TTL_SECONDS,
    )
    from data.odte_execution_policy import DEFAULT_LEASE_TTL_SECONDS, MAX_LEASE_TTL_SECONDS
    from data.odte_journal import (
        daily_trade_budget,
        green_day_preservation,
        green_day_winning_tier,
    )

    def _reentry_requires_better() -> bool:
        from data.odte_config import GREEN_REENTRY_REQUIRE_BETTER_TIER
        return bool(GREEN_REENTRY_REQUIRE_BETTER_TIER)

    def _min_reentry_tier(winning: str | None) -> str | None:
        """Lowest tier that can arm a post-green re-entry, or None when nothing can (a_plus win,
        or no green banked). Names, not ranks — this is agent-facing guidance."""
        if winning is None:
            return None
        from data.odte_journal import TIER_RANK, tier_rank
        required = tier_rank(winning) + (1 if _reentry_requires_better() else 0)
        by_rank = {r: name for name, r in TIER_RANK.items()}
        return by_rank.get(required)

    broker = _dict(broker)
    truth = _dict(broker.get("truth"))
    lane = broker.get("lane")
    age = broker.get("age_minutes")
    bp = (truth.get("buying_power")
          if lane in _TRUTH_LANES and age is not None and age <= BROKER_STALE_MINUTES else None)
    budget = daily_trade_budget(events, now=now)
    preservation = green_day_preservation(events, now=now)
    locked = bool(preservation.get("locked"))
    winning_tier = green_day_winning_tier(events, now=now).get("winning_tier") if locked else None
    return {
        "conversion_sla_seconds": CONFIRM_CONVERSION_SLA_SECONDS,
        "snapshot_ttl_seconds": SNAPSHOT_TTL_SECONDS,
        "lease": {"default_ttl_seconds": DEFAULT_LEASE_TTL_SECONDS,
                  "hard_cap_seconds": MAX_LEASE_TTL_SECONDS},
        "chase_band_fraction": CHASE_BAND_FRACTION,
        "max_debit_fraction": {"full": MAX_DEBIT_FRACTION, "b_plus": B_PLUS_DEBIT_FRACTION},
        "buying_power": bp,
        "buying_power_lane": lane,
        "max_debit_dollars": ({"full": round(MAX_DEBIT_FRACTION * bp, 2),
                               "b_plus": round(B_PLUS_DEBIT_FRACTION * bp, 2)}
                              if bp is not None else None),
        "daily_trade_budget": budget,
        "executable_universe": list(EXECUTABLE_UNIVERSE),
        "green_reentry": {
            "locked": locked,
            "auto_arm_enabled": GREEN_REENTRY_AUTO_ARM,
            # STRICT BAR ADVERTISED (2026-08-14): after the strict-tier rule shipped, the
            # controller ran FOUR doomed confirm->convert->veto cycles on post-green b_plus
            # confirms because nothing told it the bar had moved. The gate stays the enforcement;
            # `min_reentry_tier` lets the agent skip converting candidates that cannot arm
            # (None = nothing can arm today, e.g. after an a_plus win).
            "require_better_tier": _reentry_requires_better(),
            "min_reentry_tier": _min_reentry_tier(winning_tier),
            "min_bp_multiple": GREEN_REENTRY_MIN_BP_MULTIPLE,
            "winning_tier_today": winning_tier,
            "budget_remaining": budget.get("remaining"),
            "cooldown_active": budget.get("cooldown_active"),
            "cooldown_until": budget.get("cooldown_until"),
            "structurally_armable": bool(GREEN_REENTRY_AUTO_ARM
                                         and budget.get("remaining", 0) >= 1
                                         and not budget.get("cooldown_active")),
            "manual_override_key": "allow_reentry_after_green",
        },
        # FENCES ADVERTISED (2026-08-18): on the first daily-loss-floor day the controller could
        # not see the day was over — it converted a clean confirm straight into a gate veto. The
        # gate stays the enforcement; this block lets the agent stop spending conversion cycles
        # on entries that cannot pass. `day_over` folds the floor and the budget into one bit.
        "fences": _fences_block(budget, events, now),
        "basis": ("authoritative live rails — read these numbers every tick; never act on "
                  "remembered policy"),
    }


def _fences_block(budget: dict, events: list[dict] | None = None,
                  now: datetime | None = None) -> dict:
    from data.odte_config import (
        DAILY_LOSS_FLOOR_DOLLARS,
        MAX_SIGNAL_AGE_MINUTES,
        MIDDAY_FULL_TIER_AFTER_ET_HOUR,
        MIN_ENTRY_PREMIUM,
    )
    # SIGNAL AGES ADVERTISED (2026-08-19): four convert cycles burned in one hour on confirms
    # the freshness gate vetoed every time — the controller could not see a move's age without
    # paying for the convert. Same clock as the gate (first_signal_age_minutes); a move at/past
    # the limit is one the FAST PATH should skip, stating the fence.
    signal_ages: dict[str, dict] = {}
    if events and now is not None and MAX_SIGNAL_AGE_MINUTES > 0:
        from data.odte_journal import SIGNAL_AGE_LOOKBACK_MINUTES, first_signal_age_minutes
        seen: set[tuple[str, str]] = set()
        for e in events:
            if e.get("event_type") != "candidate_evaluation":
                continue
            sym = str(e.get("symbol") or "").upper()
            direc = str(e.get("direction") or "").lower()
            if not sym or not direc or (sym, direc) in seen:
                continue
            seen.add((sym, direc))
        for sym, direc in sorted(seen):
            age = first_signal_age_minutes(events, sym, direc, now=now)
            if age is not None and age <= SIGNAL_AGE_LOOKBACK_MINUTES:
                signal_ages[f"{sym}:{direc}"] = {
                    "age_minutes": age, "stale": age > MAX_SIGNAL_AGE_MINUTES}
    day_net = budget.get("day_net_pnl")
    floor_armed = DAILY_LOSS_FLOOR_DOLLARS > 0
    floor_reached = bool(floor_armed and isinstance(day_net, (int, float))
                         and day_net <= -DAILY_LOSS_FLOOR_DOLLARS)
    return {
        "min_entry_premium": (MIN_ENTRY_PREMIUM if MIN_ENTRY_PREMIUM > 0 else None),
        "midday_full_tier_after_et_hour": (MIDDAY_FULL_TIER_AFTER_ET_HOUR
                                           if MIDDAY_FULL_TIER_AFTER_ET_HOUR > 0 else None),
        "daily_loss_floor_dollars": (DAILY_LOSS_FLOOR_DOLLARS if floor_armed else None),
        "daily_loss_floor_reached": floor_reached,
        "max_signal_age_minutes": (MAX_SIGNAL_AGE_MINUTES
                                   if MAX_SIGNAL_AGE_MINUTES > 0 else None),
        "signal_ages": signal_ages or None,
        # Honest even off the resume posture: an exhausted budget with the a_plus exception
        # still active is NOT a finished day — an a_plus confirm could trade it.
        "day_over": bool(floor_reached or (budget.get("exhausted")
                                           and not budget.get("aplus_uncapped_active"))),
    }


def _green_reentry_scan_allowed(events: list[dict] | None, now: datetime | None,
                                tier) -> tuple[bool, str]:
    """PURE: may the post-green scan/watch lane stay alive for this candidate?

    2026-08-03 auto-arm: nothing at loop-status can place an order, so the scan lane stays open
    when the cheap deterministic checks pass — auto-arm enabled, a daily-budget slot open, the
    post-close cooldown clear, and (when the candidate already carries a tape tier) that tier
    ranking >= the winning trade's tier. A raw candidate with no tier yet passes through: the
    tier/BP enforcement point is the deterministic gate inside odte-convert. Returns
    (allowed, blocking_reason)."""
    from data.odte_config import GREEN_REENTRY_AUTO_ARM
    from data.odte_journal import daily_trade_budget, green_day_winning_tier, tier_rank
    if not GREEN_REENTRY_AUTO_ARM:
        return False, "profitable trade already banked today (auto-arm disabled)"
    budget = daily_trade_budget(events, now=now)
    if budget.get("remaining", 0) < 1:
        # A+ UNCAPPED (2026-08-06): a green day keeps the scan lane alive past the base cap for
        # a_plus-tier work — an untiered candidate passes through (the gate enforces the tier).
        aplus_ok = bool(budget.get("aplus_uncapped_active")
                        and (tier is None or str(tier).lower() == "a_plus"))
        if not aplus_ok:
            return False, "daily trade budget exhausted (a_plus-only exception not available)"
    if budget.get("cooldown_active"):
        return False, f"post-trade cooldown active until {budget.get('cooldown_until')}"
    if tier is not None:
        from data.odte_config import GREEN_REENTRY_REQUIRE_BETTER_TIER
        winning = green_day_winning_tier(events, now=now)
        required = winning.get("winning_rank", 0) + (1 if GREEN_REENTRY_REQUIRE_BETTER_TIER else 0)
        if tier_rank(tier) < required:
            # Mirrors the gate's strict bar — advertising "armable" here for a tier the gate will
            # veto sends the controller into a doomed convert.
            bar = "strictly above" if GREEN_REENTRY_REQUIRE_BETTER_TIER else "at or above"
            return False, (f"candidate tier {tier} not {bar} today's winning tier "
                           f"{winning.get('winning_tier')}")
    return True, ""


def _confirm_identity(watched: dict) -> tuple[str, str]:
    """(option_id, candidate_fingerprint) of a watched candidate, normalized ('' when absent)."""
    return (str(watched.get("option_id") or "").strip(),
            str(watched.get("candidate_fingerprint") or "").strip())


def _confirm_conversion_evidence(watched: dict, watch_payload: dict, events: list[dict]) -> str | None:
    """Terminal evidence that accounts for an aged CONFIRM_ENTRY candidate identity.

    An IDENTITY-BOUND confirm (``option_id`` or non-empty ``candidate_fingerprint``) resolves ONLY
    on an exact identity match: the event's ``option_id`` equals the watched one, or both sides
    carry the SAME non-empty fingerprint. A same-underlying event without matching identity never
    resolves it — an unrelated IWM no-trade must not absolve the specific 292P obligation. The
    same-underlying fallback exists only for a watched candidate with NEITHER identity field.
    Returns the matching event_type, or None when NOTHING in the journal converted or terminally
    refused the confirmed entry (the fail-closed case)."""
    option_id, fingerprint = _confirm_identity(watched)
    confirm_ts = _event_ts(_dict(watch_payload))
    for ev in reversed(events or []):
        if not isinstance(ev, dict) or ev.get("event_type") not in _CONFIRM_TERMINAL_EVENTS:
            continue
        ev_ts = _parse_ts(ev.get("ts") or ev.get("timestamp") or ev.get("generated_at"))
        if confirm_ts is not None and ev_ts is not None and ev_ts < confirm_ts:
            continue
        ev_option = str(ev.get("option_id") or _dict(ev.get("candidate")).get("option_id") or "").strip()
        ev_fp = str(ev.get("candidate_fingerprint") or "").strip()
        if option_id or fingerprint:
            if option_id and ev_option and ev_option == option_id:
                return str(ev.get("event_type"))
            if fingerprint and ev_fp and ev_fp == fingerprint:
                return str(ev.get("event_type"))
            continue
        if _same_underlying(watched, ev):
            return str(ev.get("event_type"))
    return None


def _resolve_loop_state(active_trade: dict | None = None,
                        position_decision: dict | None = None,
                        active_candidate: dict | None = None,
                        candidate_decision: dict | None = None,
                        triggers: dict | None = None,
                        journal_events: list[dict] | None = None,
                        *, errors: set[str] | None = None,
                        order_guard: dict | None = None,
                        now: datetime | None = None,
                        stale_decision_minutes: int = STALE_DECISION_MINUTES) -> dict:
    """Pure loop-state resolver. Summarizes the artifacts; re-derives no gate/decision. Never raises.

    `errors` names artifacts that were present-but-malformed (so a broken live artifact degrades
    rather than silently reads as missing). `now` (optional) enables staleness/recency checks.
    `order_guard` (optional) is the latest `odte-order-guard` decision: a fresh active guard state
    (pending / cancel-required / safety incident) OUTRANKS every other lane — cancel first."""
    errors = set(errors or [])
    events = journal_events or []
    plan = _dict(active_trade)
    pdec = _dict(position_decision)
    acand = _dict(active_candidate)
    cdec = _dict(candidate_decision)
    trig = _dict(triggers)
    reasons: list[str] = []

    # --- ORDER-GUARD lane (2026-07-23 remediation): a live/cancel-required pending ENTRY order
    # outranks EVERYTHING — an unfilled order past its lease must be cancelled before any scan,
    # candidate, gate, or new-entry work; a fill/mismatch outside a valid lease is a top-line
    # safety incident. NO_ORDER / FILLED_FRESH / stale guard artifacts fall through.
    guard = _dict(order_guard)
    guard_state = str(guard.get("state") or "").upper()
    guard_age = _age_minutes(_event_ts(guard), now) if guard else None
    if guard and guard_state in _ACTIVE_GUARD_STATES:
        if guard_age is not None and guard_age > ORDER_GUARD_STALE_MINUTES:
            reasons.append(f"ignored stale order-guard decision "
                           f"({guard_age:.0f}m > {ORDER_GUARD_STALE_MINUTES}m)")
        else:
            order = _dict(guard.get("order"))
            cancel_required = (guard.get("cancel_required") is True
                               or guard_state in ("CANCEL_STALE_ENTRY", "CANCEL_THESIS_INVALID"))
            incident = (guard.get("safety_incident") is True
                        or guard_state in ("FILLED_WITHOUT_VALID_LEASE", "BROKER_MISMATCH_BLOCKED"))
            reasons.append(f"order guard: {guard_state} "
                           f"({guard.get('underlying') or order.get('symbol') or '?'})")
            for r in (guard.get("reasons") or [])[:3]:
                reasons.append(str(r))
            out = _payload("PENDING_ORDER", reasons, now,
                           live=guard_state == "FILLED_WITHOUT_VALID_LEASE",
                           context={
                               "underlying": guard.get("underlying") or order.get("symbol"),
                               "order_state": guard_state,
                               "cancel_required": cancel_required,
                               "safety_incident": incident,
                               "prohibit_new_entries": (guard.get("prohibit_new_entries") is True
                                                        or incident),
                               "pending_order_age_seconds": guard.get("pending_order_age_seconds"),
                               "lease_seconds_remaining": guard.get("lease_seconds_remaining"),
                               "lease_id": guard.get("lease_id"),
                           })
            if guard.get("next_action") and guard.get("next_command"):
                out["next_action"] = guard["next_action"]
                out["next_command"] = guard["next_command"]
            return out

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

    # GREEN-DAY PRESERVATION (post-scalp lockout): once a completed profitable trade is banked for
    # the ET day, the fresh-entry lane is CLOSED — a later candidate/gate must not re-open it (live
    # 2026-07-10 bug: 3x SPY 753C scalped green, then a fresh promoted gate re-entered the same
    # contract for a scratch). Managing a live position is never affected (that lane returned above).
    from data.odte_journal import execution_safety_lockout, green_day_preservation
    preservation = green_day_preservation(events, active_trade=plan, now=now)
    green_locked = bool(preservation.get("locked"))
    # EXECUTION-SAFETY LOCKOUT (2026-07-23 remediation): once an execution_safety_incident is
    # journaled for the ET day, the fresh-entry lane is CLOSED — no gate, candidate, or scan
    # trigger may surface as actionable. There is deliberately NO override flag.
    safety = execution_safety_lockout(events, now=now)
    safety_locked = bool(safety.get("locked"))

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
    # Green-day preservation consumes ANY remaining fresh gate — even an execution-allowed one —
    # unless the gate itself explicitly armed the re-entry override (BP-checked at gate build).
    if green_locked and gate_fresh and gate.get("allow_reentry_after_green") is not True:
        gate_fresh = False
        reasons.append("green_day_preservation_lockout: ignored fresh entry gate — profitable "
                       "trade already banked today; re-entry requires an explicit "
                       "allow_reentry_after_green override")
    # Execution-safety lockout consumes ANY fresh gate unconditionally — no override exists.
    if safety_locked and gate_fresh:
        gate_fresh = False
        reasons.append("execution_safety_lockout: ignored fresh entry gate — an "
                       "execution_safety_incident is journaled today; NO new entries this session")
    # PROMOTED requires EXPLICIT gate permission plus every required final live confirmation. This
    # is an independent fail-closed backstop: even a malformed/legacy journal event carrying
    # execution_allowed=true cannot surface EXECUTION_READY while confirmation_needed is true.
    canonical_confirmations = ("live_chain_recheck", "spread_cap_check", "budget_check")
    raw_required_confirmations = gate.get("required_confirmations")
    declared_confirmations = (tuple(raw_required_confirmations)
                              if isinstance(raw_required_confirmations, (list, tuple)) else ())
    gate_confirmations = _dict(gate.get("confirmations"))
    gate_confirmations_ok = (
        set(canonical_confirmations).issubset(set(declared_confirmations))
        and all(gate_confirmations.get(name) is True for name in canonical_confirmations)
    )
    gate_exec = (gate_fresh and gate.get("execution_allowed") is True
                 and gate.get("confirmation_needed") is not True
                 and gate_confirmations_ok)
    if gate_fresh and gate.get("execution_allowed") is True and not gate_confirmations_ok:
        reasons.append("ignored execution-allowed gate: final confirmations missing/failed")

    candidate = _dict(trig.get("candidate"))
    trig_fresh, trig_stale_reason = _fresh_after_close(
        trig, now=now, close_dt=close_dt if closed_trade else None,
        ttl_minutes=STALE_TRIGGER_MINUTES)
    trig_candidate = bool(candidate.get("ticker")) and not candidate.get("restricted") and trig_fresh
    if candidate.get("ticker") and not trig_fresh and trig_stale_reason:
        reasons.append(f"ignored stale scan trigger: {trig_stale_reason}")
    if green_locked and trig_candidate:
        allowed, block_reason = _green_reentry_scan_allowed(events, now, candidate.get("tier"))
        if allowed:
            reasons.append("green_reentry_auto_arm: post-green scan candidate surfaced (budget "
                           "slot or a_plus-uncapped exception, cooldown clear) — tier/BP enforced at the gate via "
                           "odte-convert")
        else:
            trig_candidate = False
            reasons.append(f"green_day_preservation_lockout: ignored fresh scan candidate — "
                           f"{block_reason}")
    if safety_locked and trig_candidate:
        trig_candidate = False
        reasons.append("execution_safety_lockout: ignored fresh scan candidate — an "
                       "execution_safety_incident is journaled today; NO new entries this session")
    trig_ts = _parse_ts(trig.get("ts") or trig.get("generated_at"))
    newer_candidate_after_gate = False
    if trig_candidate and gate_fresh and not gate_exec and gate_ts is not None and trig_ts is not None:
        newer_candidate_after_gate = (
            (trig_ts - gate_ts).total_seconds() / 60.0 >= SUPERSEDE_GATE_MINUTES
        )

    if closed_trade and not reviewed and recent_close:
        reasons.append(f"trade {trade_id or '?'} closed, no postmortem yet")
        # `gross_pnl or net_pnl_est` had two faults (2026-08-12): it knew only two of the five
        # spellings a close is written under, and `or` is falsy — a SCRATCH plan with
        # `gross_pnl: 0.0` reported `net_pnl_est` instead of zero, and scratches are real
        # (2026-07-06 SCRATCH_FILLED, 2026-08-06 closed at exactly +0.00). `_realized` is the one
        # reader that understands every spelling and distinguishes 0.0 from absent.
        from data.odte_journal import _realized
        return _payload("EXITED", reasons, now, live=False,
                        context={"trade_id": trade_id, "underlying": plan.get("underlying"),
                                 "realized_pnl": _realized(plan)})
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
        # CONVERSION ACCOUNTABILITY (2026-07-27 IWM 292P regression): an identity-bound
        # CONFIRM_ENTRY is a conversion obligation. It may expire quietly ONLY when something
        # terminal accounts for it — a matching gate/order/no-trade verdict in the journal, the
        # trade cycle closing, or a day lockout. Aging out with NO such evidence means the loop
        # dropped a confirmed entry on the floor (an unaccounted order may even exist at the
        # broker), so it fails CLOSED instead of dissolving into a flat no-trade tick.
        # Only an IDENTITY-BOUND confirm (option_id or non-empty fingerprint) carries the
        # obligation — a generic confirm without a locked contract keeps the prior quiet-ignore
        # behavior rather than manufacturing false incidents.
        watch_identity = any(_confirm_identity(watched))
        unaccounted_confirm = (watch_decision == "CONFIRM_ENTRY" and watch_identity
                               and watch_stale_reason != "older than closed trade"
                               and not closed_trade and not green_locked and not safety_locked)
        evidence = (_confirm_conversion_evidence(watched, watch_payload, events)
                    if unaccounted_confirm else None)
        if unaccounted_confirm and evidence is None:
            option_id = watched.get("option_id")
            underlying = (watched.get("ticker") or watched.get("underlying")
                          or watched.get("symbol"))
            reasons.append(
                f"EXECUTION CONVERSION FAILURE: CONFIRM_ENTRY {underlying or '?'} "
                f"(option_id={option_id or '?'}) aged out ({watch_stale_reason}) with NO matching "
                "entry gate, lease, order, or no-trade verdict in the journal — the confirmed "
                "entry was never converted or terminally refused; failing closed")
            payload = _payload("DEGRADED", reasons, now, live=False, context={
                **_candidate_watch_ctx(watched, cdec, confirmed=False),
                "option_id": option_id,
                "execution_conversion_failure": True})
            payload["next_action"] = (
                "verify at the broker that NO order/position exists for the confirmed contract, "
                "then journal the terminal no_trade_decision (same identity) — or the incident — "
                "before any new entry work")
            payload["next_command"] = "odte-journal (no_trade_decision) → odte-loop-status"
            return payload
        if evidence is not None:
            reasons.append(f"ignored stale candidate watch: CONFIRM_ENTRY resolved by journal "
                           f"{evidence} ({watch_stale_reason})")
        else:
            reasons.append(f"ignored stale candidate watch: {watch_stale_reason}")
    inactive_watch = {"DEGRADED_NO_TRADE", "EXPIRED_NO_CONFIRMATION", ""}
    watch_actionable = watch_live and watch_fresh and watch_decision not in inactive_watch
    if green_locked and watch_actionable:
        allowed, block_reason = _green_reentry_scan_allowed(events, now, watched.get("tier"))
        if allowed:
            reasons.append("green_reentry_auto_arm: post-green candidate watch surfaced (budget "
                           "slot or a_plus-uncapped exception, cooldown clear) — tier/BP enforced at the gate via "
                           "odte-convert")
        else:
            watch_actionable = False
            reasons.append(f"green_day_preservation_lockout: ignored candidate watch — "
                           f"{block_reason}")
    if safety_locked and watch_actionable:
        watch_actionable = False
        reasons.append("execution_safety_lockout: ignored candidate watch — an "
                       "execution_safety_incident is journaled today; NO new entries this session")
    if watch_actionable and watch_decision == "CONFIRM_ENTRY":
        # A FRESH confirm whose exact identity was already TERMINALLY REFUSED after the confirm
        # was written is CONSUMED — 2026-08-06: odte-convert journaled the identity-bound budget
        # veto twice while this branch kept re-commanding CONVERT_CANDIDATE_NOW for the same
        # dead contract, pinning the loop (the stale-confirm path had this check; the fresh path
        # did not). A NEW confirm cycle (newer than the refusal) re-arms conversion normally.
        terminal = _confirm_conversion_evidence(watched, watch_payload, events)
        if terminal == "no_trade_decision":
            watch_actionable = False
            reasons.append(
                "confirmed candidate already terminally refused (identity-bound "
                "no_trade_decision at/after the confirm) — conversion consumed; do NOT "
                "re-convert this contract; rotate to a BP-fitting vehicle or rescan the tape")
    if watch_actionable:
        if watch_decision == "CONFIRM_ENTRY":
            option_id, fingerprint = _confirm_identity(watched)
            ctx = _candidate_watch_ctx(watched, cdec, confirmed=True)
            if option_id or fingerprint:
                # IDENTITY-BOUND confirm: conversion is the SOLE pre-position priority this tick
                # (same-tick SLA) — surface the exact contract + deadline, not a scouting hint.
                confirmed_dt = _event_ts(_dict(watch_payload))
                ctx.update({"option_id": option_id or None,
                            "candidate_fingerprint": fingerprint or None,
                            "conversion_due": True,
                            "conversion_sla_seconds": CONFIRM_CONVERSION_SLA_SECONDS})
                if confirmed_dt is not None:
                    due = confirmed_dt + timedelta(seconds=CONFIRM_CONVERSION_SLA_SECONDS)
                    ctx["confirmed_at"] = confirmed_dt.isoformat(timespec="seconds")
                    ctx["conversion_due_by"] = due.isoformat(timespec="seconds")
                    if now is not None:
                        ctx["candidate_age_seconds"] = round((now - confirmed_dt).total_seconds(), 1)
                        ctx["conversion_sla_seconds_remaining"] = round((due - now).total_seconds(), 1)
                reasons.append(f"identity-bound CONFIRM_ENTRY "
                               f"({ctx.get('underlying') or '?'} option_id={option_id or '—'}) — "
                               "convert it NOW with odte-convert")
                payload = _payload("CANDIDATE", reasons, now, live=False, context=ctx)
                payload["next_command"] = "odte-convert"
                payload["next_action"] = (
                    "CONVERT NOW: fetch fresh market/broker/contract snapshots and run "
                    "odte-convert for THIS exact confirmed contract before anything else — one "
                    "atomic command (tape re-check → gate → lease); do not scan another idea, do "
                    "not end the tick; the confirm expires against the conversion SLA")
                return payload
            reasons.append("candidate watch confirmed setup; run the atomic conversion")
            payload = _payload("CANDIDATE", reasons, now, live=False, context=ctx)
            payload["next_command"] = "odte-convert"
            payload["next_action"] = ("candidate confirmed — fetch fresh snapshots and run "
                                      "odte-convert (atomic gate+lease)")
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
        payload = _payload("GATED", reasons, now, live=False, context=_gate_ctx(gate))
        # Name the exact blockers so the controller knows the trigger needed — not just "promote".
        failing = payload["context"].get("failing_gates") or []
        unknown = payload["context"].get("unknown_gates") or []
        if failing or unknown:
            bits = ([f"failing: {', '.join(failing)}"] if failing else []) + \
                   ([f"missing input: {', '.join(unknown)}"] if unknown else [])
            payload["next_action"] = (f"gate not execution-allowed ({'; '.join(bits)}) — refresh "
                                      "those inputs and re-run the gate; promote only when every "
                                      "required gate is explicitly True")
        return payload
    if trig_candidate:
        reasons.append(f"candidate {candidate.get('ticker')} {candidate.get('direction') or ''} "
                       f"(scan_only/observe)".strip())
        return _payload("CANDIDATE", reasons, now, live=False, context={
            "underlying": candidate.get("ticker"), "direction": candidate.get("direction"),
            "spy_verdict": trig.get("spy_verdict"), "scan_only": True})
    if reviewed:
        # REVIEWED DATE-GATE (2026-08-05): only TODAY's reviewed trade parks the loop. A
        # prior-day reviewed trade handed "roll up the journal report" as next_command every
        # morning — and on Aug-5 the controller obeyed it for 68 consecutive ticks while the
        # session tape went unread. Prior-day reviews fall through to SCAN.
        close_dt = _parse_ts(close_ts)
        close_et_date = close_dt.astimezone(ET).date() if close_dt else None
        if close_et_date == now.astimezone(ET).date():
            reasons.append(f"trade {trade_id or '?'} closed and reviewed — idle")
            payload = _payload("REVIEWED", reasons, now, live=False,
                               context={"trade_id": trade_id,
                                        "underlying": plan.get("underlying")})
            _rescan_override(payload, events, now)
            return payload
        reasons.append(f"prior-day trade {trade_id or '?'} reviewed — resuming scan")
    elif closed_trade and not reviewed:
        reasons.append(f"prior trade {trade_id or '?'} unreviewed but stale — scanning")
    reasons.append("no actionable candidate, gate, or live position")
    payload = _payload("SCAN", reasons, now, live=False, context={})
    _rescan_override(payload, events, now)
    return payload


def _gate_ctx(gate: dict) -> dict:
    """Gate context incl. the MACHINE-READABLE gate breakdown: which gates failed and which are
    still unknown (missing input), so the controller sees the exact trigger needed to advance —
    without re-reading/re-deriving the gate artifacts."""
    ctx = {"underlying": gate.get("underlying") or gate.get("symbol"),
           "direction": gate.get("direction"), "decision": gate.get("decision"),
           "scan_only": bool(gate.get("scan_only", False)),
           "execution_allowed": bool(gate.get("execution_allowed", False))}
    gates = gate.get("gates") if isinstance(gate.get("gates"), dict) else {}
    if gates:
        ctx["gates"] = gates
        ctx["failing_gates"] = sorted(k for k, v in gates.items() if v is False)
        ctx["unknown_gates"] = sorted(k for k, v in gates.items() if v is None)
    if gate.get("veto_reasons"):
        ctx["veto_reasons"] = list(gate.get("veto_reasons") or [])
    return ctx


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
      1. A CONFIRMED broker fault (down / read-only-fallback) → BROKER_DEGRADED (top-line, even on a
         flat tick — the cron must never quietly imply the lane is fine). In ``live_mode`` an
         UNCONFIRMED lane (unknown/missing broker_health) while a live position / ready setup needs
         it also fails closed to BROKER_DEGRADED.
      1b. A STALE broker_health.json while a live lane is needed → STALE_DATA_BLOCKED with the exact
         refresh command. A stale probe is an outdated FILE, not a confirmed fault: the fix is
         "refresh the probe", never "assume live orders are impossible". On a tick with nothing to
         authorize a stale probe does NOT block — it falls through to the normal posture (orders
         stay blocked via ``orders_ok`` until the file is fresh again).
      2. A live position with an authorized lane → MANAGE_POSITION (manage before scan).
      3. A live position whose live read is stale/malformed (broker fine) → STALE_DATA_BLOCKED and
         the reasons SHOUT — the controller is holding blind and must refresh the position read now.
      4. An execution-allowed (PROMOTED) gate → EXECUTION_READY: move to broker review/place under
         standing auth. A confirmed candidate watch → SCOUT_FRESH_SETUP: build/promote a fresh gate.
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
    broker_stale = lane == "stale"                 # outdated probe FILE — refreshable, not a fault
    broker_faulted = lane in ("down", "read_only_fallback")
    needs_live_lane = (live or state in ("PROMOTED", "PENDING_ORDER") or executable
                       or ctx.get("confirmed") is True)
    # Fail closed: a confirmed fault always blocks; in live_mode an unconfirmed lane (unknown/missing)
    # cannot authorize a live order either. A flat tick (no live lane needed) is never blocked here.
    # A stale probe file gets its own branch (1b) so the fix reads "refresh", not "lane is down".
    broker_blocks = broker_faulted or (live_mode and needs_live_lane and orders_ok is not True
                                       and not broker_stale)
    verify = "hermes mcp test robinhood  # repair parent review/place lane before any live order"
    refresh = broker.get("refresh_command") or BROKER_HEALTH_REFRESH_COMMAND

    # 1. Broker confirmed-faulted / unconfirmable for a needed live order — top-line BROKER_DEGRADED.
    if broker_blocks:
        if state == "PENDING_ORDER":
            return ("BROKER_DEGRADED",
                    "PENDING entry order but the broker lane can't confirm cancel/fill — verify "
                    "via the read lane and cancel when possible; NEVER assume the order is "
                    "unfilled", verify)
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

    # 1b. Stale broker_health.json while a live lane is needed: blocked until refreshed, but say so
    # precisely — the probe FILE is old; the last probe may well have said the parent lane was OK.
    if broker_stale and needs_live_lane:
        last = broker.get("last_known_place_allowed")
        last_txt = ("last probe said parent lane OK" if last is True
                    else "last probe could not confirm the parent lane" if last is False
                    else "last lane state unknown")
        if live and state in ("MANAGING", "ENTERED", "DEGRADED"):
            return ("STALE_DATA_BLOCKED",
                    f"OPEN position but broker_health.json is STALE ({last_txt}) — refresh broker "
                    "truth NOW; this is an outdated probe file, not a confirmed broker fault", refresh)
        return ("STALE_DATA_BLOCKED",
                f"setup ready but broker_health.json is STALE ({last_txt}) — refresh the probe file "
                "first; if the parent lane reads OK, proceed to broker review/place", refresh)

    # 1c. Order-guard lane: cancel-first beats every scan/new-entry posture. A safety incident or
    # cancel-required guard is the loudest actionable tick there is.
    if state == "PENDING_ORDER":
        if ctx.get("safety_incident") is True:
            return ("CANCEL_PENDING_ORDER",
                    "EXECUTION SAFETY INCIDENT — fill/mismatch outside a valid lease: flatten or "
                    "alert per the reviewed emergency policy, journal execution_safety_incident, "
                    "and PROHIBIT new entries for the session", None)
        if ctx.get("cancel_required") is True:
            return ("CANCEL_PENDING_ORDER",
                    "CANCEL FIRST: the pending entry order is stale/thesis-dead — cancel at the "
                    "broker NOW, verify with fresh order truth, and require a full fresh "
                    "candidate/gate/lease cycle before any new order", None)
        return ("PENDING_ORDER_GUARD", None, None)

    # 1d. Unconverted CONFIRM_ENTRY (fail-closed): a confirmed identity aged out with no matching
    # gate/lease/order/no-trade verdict. Not idle, not a stale-file diagnostic — the loop dropped a
    # conversion obligation and must reconcile it (broker verify + terminal journal entry) first.
    if ctx.get("execution_conversion_failure") is True:
        return ("EXECUTION_CONVERSION_FAILURE", None, None)

    # 2. Live position lane: manage before scan.
    if live and state in ("MANAGING", "ENTERED"):
        return ("MANAGE_POSITION", None, None)
    # 3. Live position we can't value — broker is fine, so this is a STALE live read, not a broker fault.
    if state == "DEGRADED" and live:
        return ("STALE_DATA_BLOCKED",
                "OPEN position but its live management read is STALE/malformed — refresh the position "
                "decision NOW before it drifts; do not act on the old read", None)

    # 4. Execution-allowed gate → EXECUTION_READY (broker review/place under standing auth).
    #    Confirmed candidate watch → SCOUT_FRESH_SETUP (build/promote a fresh gate now).
    if state == "PROMOTED":
        return ("EXECUTION_READY", None, None)
    # An identity-bound confirm is a same-tick conversion DEADLINE, not a scouting hint: the sole
    # pre-position priority is the final refresh + exact entry gate for the confirmed contract.
    # EXECUTION_READY stays reserved for an execution-allowed gate; this posture never places.
    if ctx.get("conversion_due") is True:
        return ("CONVERT_CANDIDATE_NOW", None, None)
    if ctx.get("confirmed") is True:
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


# Lanes whose position/order counts are trustworthy: a fresh parent-OK probe, or a fresh read-only
# probe (the read lane exists precisely to verify flatness/positions). stale/down/unknown never are.
_TRUTH_LANES = ("ok", "read_only_fallback")


def _reconcile_broker_truth(payload: dict, broker: dict, plan: dict,
                            now: datetime | None) -> dict:
    """BROKER TRUTH WINS: reconcile the resolved loop state against a FRESH probe's position/order
    counts, so a check-in never narrates flat/candidate while the broker holds a live option
    position, and never narrates a live/blind position after the broker went flat (the 2026-07-16
    artifact miss: two agentic scalps opened+closed at the broker while every local artifact —
    active_trade/active_state/broker_health/journal — kept the prior day's state).

    Case A — probe shows ≥1 open option position but the local artifacts resolved to a NON-live
    state: the controller is holding a trade the local files lag. Surface MANAGING (manage before
    scan) + the exact reconcile instruction. The probe carries counts, not contracts, so the
    context names no symbol — the position watch identifies it from the live snapshot.
    Case B — probe shows FLAT (0 open option positions AND 0 open option orders, both explicitly
    reported) while the local plan/decision still reads live: the trade already closed at the
    broker (e.g. an agentic harvest). Surface EXITED — journal the close + postmortem — instead of
    ENTERED / DEGRADED "holding blind". A net-green day that ends with low BP is a CLEAN stop,
    never a fault.
    A stale, undated, down, or unknown probe NEVER reconciles: outdated counts are not live truth.
    Pure + read-only — this fixes the REPORTED state and instructions only; the controller owns the
    artifact files and this module never invents fills or places orders."""
    truth = broker.get("truth") or {}
    age = broker.get("age_minutes")
    if (not truth or broker.get("lane") not in _TRUTH_LANES
            or age is None or age > BROKER_STALE_MINUTES):
        return payload
    # The order-guard lane already consumed broker ORDER truth directly — reconciling it away
    # would erase a cancel-first/incident instruction. Leave it top-line.
    if payload.get("state") == "PENDING_ORDER":
        return payload
    positions = truth.get("open_option_positions")
    orders = truth.get("open_option_orders")
    prior = payload.get("state")

    # Case A: live at the broker, flat/candidate/gate locally.
    if positions is not None and positions > 0 and not payload.get("live"):
        out = _payload("MANAGING", [
            f"BROKER TRUTH: {positions:.0f} open option position(s) at the broker but local "
            f"artifacts resolved {prior} — active_trade.json/position_decision.json lag reality",
            *(payload.get("reasons") or []),
        ], now, live=True, context={
            "broker_open_option_positions": positions,
            "reconciled_from_state": prior})
        out["next_action"] = ("RECONCILE NOW: rewrite data/odte/active_trade.json from broker truth "
                              "(the open position + orders), then run the position watch with a "
                              "fresh live snapshot — manage the live trade before any scan")
        out["next_command"] = ("odte-position --snapshot <live.json>  "
                               "# after rewriting active_trade.json from broker truth")
        out["reconciliation"] = "broker_open_position_overrides_flat_artifacts"
        return out

    # Case B: flat at the broker, live locally.
    if (payload.get("live") and positions is not None and positions <= 0
            and orders is not None and orders <= 0):
        out = _payload("EXITED", [
            "BROKER TRUTH: flat (0 open option positions, 0 open option orders) but the local "
            f"plan/decision still read live ({prior}) — the trade already closed at the broker",
            *(payload.get("reasons") or []),
        ], now, live=False, context={
            "trade_id": plan.get("trade_id"), "underlying": plan.get("underlying"),
            "plan_status": (str(plan.get("status") or "") or None),
            "reconciled_from_state": prior})
        out["next_action"] = ("RECONCILE NOW: mark data/odte/active_trade.json closed with the real "
                              "exit fills, journal the close + postmortem, then stand down — a "
                              "closed net-green day with low BP is a clean stop, not a fault")
        out["reconciliation"] = "broker_flat_overrides_live_artifacts"
        return out
    return payload


def derive_loop_state(active_trade: dict | None = None,
                      position_decision: dict | None = None,
                      active_candidate: dict | None = None,
                      candidate_decision: dict | None = None,
                      triggers: dict | None = None,
                      journal_events: list[dict] | None = None,
                      *, errors: set[str] | None = None,
                      broker_health: dict | None = None,
                      order_guard: dict | None = None,
                      live_mode: bool = False,
                      now: datetime | None = None,
                      stale_decision_minutes: int = STALE_DECISION_MINUTES) -> dict:
    """Resolve the loop state and layer the cron-facing posture + broker lane on top. Never raises.

    ``_resolve_loop_state`` is the pure fine-grained resolver; here we classify a SUPPLIED/PROBED
    broker-health payload (no broker call) and reduce state+broker to one ``posture`` word, forcing
    ``executable`` off whenever the broker lane can't back a live order. ``live_mode`` (the cron/live
    controller) fails closed: a live position / ready setup with an unknown or missing broker_health
    reads BROKER_DEGRADED, so no live order is ever authorized without a fresh confirmed parent lane.
    ``order_guard`` (the latest `odte-order-guard` decision) makes pending-order cancellation the
    TOP-priority lane — above scanning and every new-entry posture."""
    payload = _resolve_loop_state(active_trade=active_trade, position_decision=position_decision,
                                  active_candidate=active_candidate, candidate_decision=candidate_decision,
                                  triggers=triggers, journal_events=journal_events, errors=errors,
                                  order_guard=order_guard,
                                  now=now, stale_decision_minutes=stale_decision_minutes)
    broker = classify_broker_lane(broker_health, now)
    # BROKER TRUTH WINS: a fresh probe's position/order counts reconcile a lagging local resolution
    # (open position ⇒ MANAGING; confirmed flat while local reads live ⇒ EXITED) BEFORE the posture
    # is derived, so the check-in narrates reality, not the stale artifacts.
    payload = _reconcile_broker_truth(payload, broker, _dict(active_trade), now)
    posture, action_override, command_override = _posture(payload, broker, live_mode=live_mode)
    payload["posture"] = posture
    payload["broker_lane"] = broker
    if action_override:
        payload["next_action"] = action_override
    if command_override:
        payload["next_command"] = command_override
    # A broker lane that can't back a live order must never advertise executable live edge —
    # covers both a confirmed fault (BROKER_DEGRADED) and a stale probe file (orders_ok False).
    if posture == "BROKER_DEGRADED" or broker.get("orders_ok") is False:
        payload["executable"] = False
    # Split the audit trail out of the user-facing "why": housekeeping breadcrumbs (ignored stale/
    # denied/consumed gates, empty scans, closed+reviewed trades) move to `notes` so a flat cron tick
    # doesn't read like it holds stale actionable work. Loud live-position degrades stay in `reasons`.
    full = payload.get("reasons") or []
    payload["reasons"] = [r for r in full if not _is_reason_noise(r)]
    payload["notes"] = [r for r in full if _is_reason_noise(r)]
    if not payload["reasons"]:
        payload["reasons"] = [_flat_reason(posture)]
    # LIVE RAILS (2026-08-03): the authoritative current numbers ride EVERY payload so the
    # controller reads live policy instead of enforcing remembered (possibly retired) rails.
    payload["live_rails"] = _live_rails(journal_events, broker,
                                        now or datetime.now(timezone.utc))
    return payload


def _flat_reason(posture: str) -> str:
    """A single clean user-facing line when the only reasons were housekeeping noise."""
    return {
        "FLAT_NO_TRADE": "flat — no fresh actionable setup",
        "STALE_DATA_BLOCKED": "stale/malformed artifact blocking action — refresh before acting",
        "EXECUTION_READY": "all gates passed — broker review/place under standing auth, then watch",
        "SCOUT_FRESH_SETUP": "candidate confirmed — assemble/promote a fresh entry gate now",
        "WAIT_FRESH_CONFIRMATION": "candidate on the board — awaiting confirmation",
        "MANAGE_POSITION": "live position — manage/exit before scanning",
        "BROKER_DEGRADED": "broker lane degraded — live orders blocked",
        "PENDING_ORDER_GUARD": "entry order pending inside its lease — poll the order guard",
        "CANCEL_PENDING_ORDER": "pending entry order stale/invalid — cancel at the broker NOW",
        "EXECUTION_CONVERSION_FAILURE": ("confirmed entry aged out unconverted — verify at the "
                                         "broker and journal the terminal no-trade verdict"),
        "CONVERT_CANDIDATE_NOW": ("identity-bound confirm on the board — run odte-convert NOW "
                                  "(atomic gate+lease); conversion is the sole pre-position "
                                  "priority"),
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
    guard, guard_status = _read_json(base / ORDER_GUARD_FILENAME)
    lease_doc, lease_status = _read_json(base / EXECUTION_LEASE_FILENAME)
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
                                broker_health=broker_health, order_guard=guard,
                                live_mode=live_mode, now=now)
    # Weekly trade-rate telemetry (2026-08-02 retune): the conversion funnel and the zero-trade
    # tripwire ride every status payload so a dying week is visible mid-week, not in a postmortem.
    from data.odte_journal import weekly_telemetry
    payload["weekly_telemetry"] = weekly_telemetry(events, now=now)
    payload["artifacts"] = {
        "active_trade": plan_status,
        "position_decision": pdec_status,
        "active_candidate": acand_status,
        "candidate_decision": cdec_status,
        "triggers": trig_status,
        "order_guard": guard_status,
        "execution_lease": lease_status,
        "broker_health": broker_health_status,
        "journal_events": len(events),
    }
    # Execution-lease visibility: lease time remaining is first-class status, never guessed.
    lease = _dict(_dict(lease_doc).get("lease")) or _dict(lease_doc)
    if lease.get("lease_id"):
        exp = _parse_ts(lease.get("expires_at"))
        remaining = round((exp - now).total_seconds(), 1) if exp is not None else None
        payload["execution_lease"] = {
            "lease_id": lease.get("lease_id"), "symbol": lease.get("symbol"),
            "risk_mode": lease.get("risk_mode"), "expires_at": lease.get("expires_at"),
            "seconds_remaining": remaining,
            "expired": bool(remaining is not None and remaining <= 0),
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
    guard_d = _dict(guard)
    guard_display_ts = (guard_d.get("generated_at") or guard_d.get("ts")) \
        if str(guard_d.get("state") or "").upper() in _ACTIVE_GUARD_STATES else None
    lease_display_ts = lease.get("issued_at") if lease.get("lease_id") else None
    # PRIOR-DAY EXPIRED LEASE IS INERT (2026-08-05): every reader fails closed on expiry (hook,
    # guard, fast lane), so a lease that expired on a previous ET day is audit trail, not
    # current work — the only artifact row that lacked an inertness predicate nagged
    # "STALE, ttl 1m" in every recap. Same-day leases (live or freshly expired) still show.
    lease_info = _dict(payload.get("execution_lease"))
    if lease_display_ts and lease_info.get("expired"):
        issued_dt = _parse_ts(lease_display_ts)
        if issued_dt and issued_dt.astimezone(ET).date() < now.astimezone(ET).date():
            lease_display_ts = None
    payload["artifact_ages"] = {
        "triggers": _age_entry(trigger_display_ts, now, STALE_TRIGGER_MINUTES),
        "candidate": _age_entry(watch_display_ts, now, STALE_CANDIDATE_MINUTES),
        "gate": _age_entry(gate_display_ts, now, STALE_ENTRY_GATE_MINUTES),
        "position_decision": _age_entry(pdec_display_ts, now, STALE_DECISION_MINUTES),
        "active_trade": _age_entry(
            plan_d.get("updated_at") or plan_d.get("closed_at") or plan_d.get("exit_fill_time")
            or plan_d.get("entry_fill_time"), now, STALE_TRADE_HOURS * 60),
        "order_guard": _age_entry(guard_display_ts, now, ORDER_GUARD_STALE_MINUTES),
        # Lease TTL is seconds-scale (≤60s), so freshness is judged on a 1-minute window here;
        # the authoritative countdown is payload["execution_lease"]["seconds_remaining"].
        "execution_lease": _age_entry(lease_display_ts, now, 1),
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
        truth = broker.get("truth") or {}
        truth_bits = [f"{k.replace('_', ' ')}: {v:g}" for k, v in truth.items() if v is not None]
        if truth_bits:
            lines.append(f"Broker truth: {' · '.join(truth_bits)}  ")
        if broker.get("refresh_command"):
            lines.append(f"Refresh: {broker.get('refresh_command')}  ")
    rails = p.get("live_rails") or {}
    if rails:
        frac = rails.get("max_debit_fraction") or {}
        dollars = rails.get("max_debit_dollars")
        dollar_txt = (f" (${dollars['full']:g} / ${dollars['b_plus']:g} at BP "
                      f"${rails.get('buying_power'):g})" if dollars else " ($? — no fresh BP probe)")
        gr = rails.get("green_reentry") or {}
        gr_txt = ("locked" if gr.get("locked") else "open")
        if gr.get("locked"):
            gr_txt += (f", re-entry {'ARMABLE' if gr.get('structurally_armable') else 'blocked'}"
                       f" (winning tier {gr.get('winning_tier_today')})")
        lease = rails.get("lease") or {}
        lines.append(
            f"Live rails: convert SLA {rails.get('conversion_sla_seconds'):g}s · snapshots "
            f"{rails.get('snapshot_ttl_seconds'):g}s · lease {lease.get('default_ttl_seconds'):g}s "
            f"(cap {lease.get('hard_cap_seconds'):g}s) · chase band "
            f"{rails.get('chase_band_fraction'):.0%} · max debit {frac.get('full'):.0%}/"
            f"{frac.get('b_plus'):.0%} of BP{dollar_txt} · green day {gr_txt} — read these, never "
            f"remembered policy  ")
    wk = p.get("weekly_telemetry") or {}
    if wk:
        target = wk.get("weekly_target") or ["?", "?"]
        trip = wk.get("tripwire") or {}
        trip_note = (" · **TRIPWIRE FIRED: zero trades this week**" if trip.get("fired")
                     else " · tripwire armed" if trip.get("armed") else "")
        lines.append(
            f"Week {wk.get('iso_week')}: **{wk.get('trades_this_week')} trades** "
            f"(target {target[0]}-{target[1]}) · gates passed {wk.get('gates_passed')} · "
            f"leases {wk.get('leases_issued')}/{wk.get('leases_issued', 0) + wk.get('lease_refusals', 0)} · "
            f"today {wk.get('trades_today')} used, {wk.get('budget_remaining_today')} left"
            f"{trip_note}  ")
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
