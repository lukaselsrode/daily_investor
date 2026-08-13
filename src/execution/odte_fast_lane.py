"""0DTE fast-lane daemon — the deterministic executor of the two-lane architecture.

The slow lane (Hermes) reasons in minutes and ARMS intents; this daemon watches the tape at
seconds cadence and, when an armed trigger fires, runs the ENTIRE existing decision chain
in-process — tape re-check → convert (gate + lease) → ONE review → pre-place guard →
consume-before-place → order-guard poll → exit management — with zero LLM turns anywhere in the
latency window. The 2026-08-04 failure mode (a 60s lease dying under 61.4s of model re-review
turns) is structurally impossible here.

State machine (one `tick()` per poll):

    IDLE ──armed intents──▶ WATCHING ──intent fires──▶ FIRING ──placed──▶ PENDING ──filled──▶
    MANAGING ──exit filled──▶ IDLE          (convert refused → WATCHING; cancel states → IDLE)

Modes (from `data/odte/fast_lane_stage.json`; a constructor/CLI mode may only be MORE
conservative than the stage file):
  * shadow  — full compute + shadow journal; the client is wrapped in ShadowClient, which RAISES
              on any order-lane tool: placement is structurally impossible, not policy.
  * exits   — MANAGING may place real sell-to-close orders; the entry side still only journals
              `shadow_order_intent`.
  * live    — armed entries place for real through lease + guard + consumed ledger.

Safety composition (nothing here weakens the existing rails):
  * every open goes through `run_convert` (budget/green/cooldown/tier/chase band/lease) and then
    `pre_place_check`/`consume_then` — the consumed ledger is SHARED with the Hermes hook;
  * closes always pass (the incident-remediation invariant) but still get the NVDA scan;
  * `FILLED_WITHOUT_VALID_LEASE`/`BROKER_MISMATCH_BLOCKED` journal an execution_safety_incident
    and LOCK entries for the session;
  * the pause file (`data/odte/fast_lane_pause`) halts ALL placements on the next tick while
    monitoring/journaling continues — Hermes keeps its own close/cancel valve regardless.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.paths import ODTE_DATA_DIR, atomic_write_text
from data.odte_armed_intent import (
    ARMED_INTENTS_FILENAME,
    INTENT_STATE_FILENAME,
    evaluate_armed_intent,
    validate_armed_intent,
)
from data.odte_convert import run_convert
from data.odte_execution_policy import LEASE_FILENAME
from data.odte_journal import DEFAULT_JOURNAL_PATH, append_decision_journal, read_events
from data.odte_order_guard import (
    CANCEL_STATES,
    FILLED_FRESH,
    INCIDENT_STATES,
    NO_ORDER,
    evaluate_order_guard,
)
from data.odte_position import DEFAULT_PLAN_FILENAME, evaluate_position
from data.odte_shadow_report import (
    SHADOW_EXIT_INTENT,
    SHADOW_INCIDENT,
    SHADOW_ORDER_INTENT,
)
from execution.odte_pre_place_guard import consume_then, order_for_guard, pre_place_check
from execution.odte_tape import build_tape

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

DEFAULT_STATE_DIR = ODTE_DATA_DIR
STAGE_FILENAME = "fast_lane_stage.json"
PAUSE_FILENAME = "fast_lane_pause"
STATUS_FILENAME = "fast_lane_status.json"
SHADOW_DIRNAME = "shadow"

MODE_SHADOW, MODE_EXITS, MODE_LIVE = "shadow", "exits", "live"
_MODE_RANK = {MODE_SHADOW: 0, MODE_EXITS: 1, MODE_LIVE: 2}
_STAGE_TO_MODE = {"shadow": MODE_SHADOW, "exits_live": MODE_EXITS, "entries_live": MODE_LIVE}

IDLE, WATCHING, FIRING, PENDING, MANAGING = "IDLE", "WATCHING", "FIRING", "PENDING", "MANAGING"

SESSION_START_ET = dtime(9, 25)
SESSION_END_ET = dtime(16, 5)

# A convert refusal parks the intent briefly so a still-true trigger can't hot-loop refires.
REFIRE_COOLDOWN_SECONDS = 60.0
# Exit reprice loop: cancel/replace at the fresh bid when unfilled and faded, bounded.
EXIT_REPRICE_AFTER_SECONDS = 10.0
EXIT_REPRICE_MIN_TICK = 0.01
EXIT_MAX_REPLACEMENTS = 5

# Trigger ACTIONS that demand a close NOW. Mode semantics stay honest: a trend/lotto/runner
# TAKE_PROFIT alert (protect_profit / hold_but_alert / trail_runner) is NOT force-exited.
_EXIT_ACTIONS = {"exit", "exit_all", "harvest_now", "exit_or_let_expire", "flatten"}


class ShadowOrderBlocked(RuntimeError):
    """Raised by ShadowClient on ANY order-lane tool — shadow mode cannot place, structurally."""


class ShadowClient:
    """Read-only wrapper: market data + broker truth pass through; order tools raise."""

    _ORDER_METHODS = ("review_option_order", "place_option_order", "cancel_option_order")

    def __init__(self, client: Any) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        if name in self._ORDER_METHODS:
            raise ShadowOrderBlocked(f"{name} is blocked in shadow mode")
        return getattr(self._client, name)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_mode(stage_path: Path, requested: str | None) -> str:
    """Stage file is authority; an explicit request may only be MORE conservative."""
    stage = str(_read_json(stage_path).get("stage") or "shadow").lower()
    stage_mode = _STAGE_TO_MODE.get(stage, MODE_SHADOW)
    if requested is None:
        return stage_mode
    requested = requested if requested in _MODE_RANK else MODE_SHADOW
    return requested if _MODE_RANK[requested] <= _MODE_RANK[stage_mode] else stage_mode


def in_session(now: datetime) -> bool:
    et = now.astimezone(_ET)
    return et.weekday() < 5 and SESSION_START_ET <= et.time() < SESSION_END_ET


def _option_quote_to_contract(intent: dict, quote: Any, now: datetime) -> dict:
    quote = quote if isinstance(quote, dict) else {}
    inner = quote.get("quote") if isinstance(quote.get("quote"), dict) else quote
    contract = dict(intent.get("contract") or {})
    bid = _num(inner.get("bid_price") or inner.get("bid"))
    ask = _num(inner.get("ask_price") or inner.get("ask"))
    mark = _num(inner.get("mark_price") or inner.get("mark") or inner.get("adjusted_mark_price"))
    if mark is None and bid is not None and ask is not None:
        mark = round((bid + ask) / 2.0, 4)
    out = {
        "as_of": now.isoformat(timespec="seconds"),
        "underlying": intent.get("symbol"),
        "direction": intent.get("direction"),
        **contract,
        "bid": bid, "ask": ask, "mark": mark,
    }
    for key in ("volume", "open_interest", "implied_volatility", "delta", "gamma"):
        n = _num(inner.get(key))
        if n is not None:
            out[key] = n
    return out


def _order_row_to_guard(row: Any, account_symbol: str | None = None) -> dict:
    """Broker order row (get_option_orders shape) → the guard's order dict."""
    row = row if isinstance(row, dict) else {}
    legs = [leg for leg in (row.get("legs") or []) if isinstance(leg, dict)]
    leg = legs[0] if legs else {}
    option_id = (leg.get("option_id") or leg.get("option")
                 or row.get("option_id"))
    if isinstance(option_id, str) and "/" in option_id:        # URL form: take the UUID tail
        option_id = option_id.rstrip("/").rsplit("/", 1)[-1]
    state = str(row.get("state") or row.get("status") or "").lower()
    filled_at = row.get("filled_at")
    if filled_at is None and state == "filled":
        filled_at = row.get("last_transaction_at") or row.get("updated_at")
    out = {
        "status": state,
        "symbol": row.get("chain_symbol") or account_symbol,
        "option_id": option_id,
        "quantity": _num(row.get("quantity")),
        "filled_quantity": _num(row.get("processed_quantity") or row.get("filled_quantity")),
        "limit_price": _num(row.get("price")),
        "submitted_at": row.get("created_at"),
        "filled_at": filled_at,
        "order_id": row.get("id") or row.get("order_id"),
        # Legs ride along so the guard's own normalization derives the FULL option identity
        # (type/strike/expiration) — without them the daemon's identity coverage rested on
        # option_id alone (2026-08-06 collision audit).
        "legs": row.get("legs"),
    }
    avg = _num(row.get("average_price") or row.get("processed_premium"))
    if avg is not None:
        out["average_price"] = avg
    return out


class FastLaneDaemon:
    """One instance per session. `tick(now)` is the whole machine — deterministic given the
    injected client/clock, so tests drive it directly with FakeMcpSession + a fake clock."""

    def __init__(self, client: Any, *, account_number: str,
                 state_dir: str | Path = DEFAULT_STATE_DIR,
                 mode: str | None = None, journal_path: str | None = None) -> None:
        self.base_dir = Path(os.path.expanduser(str(state_dir)))
        self.mode = resolve_mode(self.base_dir / STAGE_FILENAME, mode)
        self.client = ShadowClient(client) if self.mode == MODE_SHADOW else client
        self.account_number = str(account_number)
        self.journal_path = journal_path or (
            DEFAULT_JOURNAL_PATH if Path(DEFAULT_STATE_DIR) == self.base_dir
            else str(self.base_dir / "decision_journal.jsonl"))
        self.shadow_dir = self.base_dir / SHADOW_DIRNAME
        self.shadow_journal_path = str(self.shadow_dir / "decision_journal.jsonl")
        self.ledger_path = str(self.base_dir / "consumed_leases.json")

        self.state = IDLE
        self.entries_locked = False
        self.tape_state: dict = {}
        self.last_tape: dict = {}
        self.intents: list[dict] = []
        self.intents_mtime: float | None = None
        self.intent_states: dict[str, dict] = self._load_intent_states()
        self.lease: dict | None = None
        self.pending_order: dict | None = None                 # {order_id, ref_id, placed_at}
        self.exit_order: dict | None = None                    # {order_id, limit, placed_at, n}
        self.counts: dict[str, int] = {"ticks": 0, "fires": 0, "placed": 0, "exits": 0,
                                       "refusals": 0, "incidents": 0}
        self.last_error: str | None = None
        self._started = False

    # ── journaling ──────────────────────────────────────────────────────────────────────────

    def _journal(self, event: dict, event_type: str, *, shadow: bool, now: datetime) -> None:
        path = self.shadow_journal_path if shadow else self.journal_path
        event = {"ts": now.isoformat(timespec="seconds"), "mode": self.mode, **event}
        append_decision_journal(event, source="fast_lane", event_type=event_type,
                                journal_path=path, now=now)

    def _real_events(self) -> list[dict]:
        try:
            return read_events(self.journal_path)
        except OSError:
            return []

    # ── file state ──────────────────────────────────────────────────────────────────────────

    def _load_intent_states(self) -> dict:
        data = _read_json(Path(self.base_dir) / INTENT_STATE_FILENAME)
        return data if isinstance(data, dict) else {}

    def _persist_intent_states(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.base_dir / INTENT_STATE_FILENAME,
                          json.dumps(self.intent_states, indent=2, default=str))

    def _reload_intents(self) -> None:
        path = self.base_dir / ARMED_INTENTS_FILENAME
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self.intents, self.intents_mtime = [], None
            return
        if mtime == self.intents_mtime:
            return
        doc = _read_json(path)
        intents = [i for i in (doc.get("intents") or []) if isinstance(i, dict)]
        self.intents = [i for i in intents if not validate_armed_intent(i)]
        dropped = len(intents) - len(self.intents)
        if dropped:
            logger.warning("armed_intents.json: %d invalid intent(s) ignored", dropped)
        self.intents_mtime = mtime

    def _paused(self) -> bool:
        return (self.base_dir / PAUSE_FILENAME).exists()

    def _write_heartbeat(self, status: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.base_dir / STATUS_FILENAME,
                          json.dumps(status, indent=2, default=str))

    def _plan_path(self) -> Path:
        """The plan we READ — always the canonical one. Observing the real position is the point."""
        return self.base_dir / DEFAULT_PLAN_FILENAME

    def _plan_write_path(self) -> Path:
        """The plan we WRITE — the shadow copy unless this lane actually owns the position.

        SHADOW MUST NOT MUTATE SHARED STATE (2026-08-07). `MODE_SHADOW` wraps the broker client
        (`ShadowClient`) so no order can escape, but the plan writes were ungated by mode: a shadow
        daemon started while the CONTROLLER held a position would read the controller's
        `active_trade.json`, recompute, and write its own version back — clobbering `thesis` (the
        pre-entry invalidation) and `management.best_seen_bid` (the entire state of the bid-memory
        rail). Wrapping the broker is not enough when two lanes share a state file.

        This mirrors what the daemon already does for its journal: `shadow_journal_path` beside
        `journal_path`, selected per event at `_journal`. Same idea, same directory.
        """
        if self.mode == MODE_LIVE:
            return self._plan_path()
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        return self.shadow_dir / DEFAULT_PLAN_FILENAME

    # ── startup reconciliation ──────────────────────────────────────────────────────────────

    async def _startup_reconcile(self, now: datetime) -> None:
        """Adopt whatever reality left behind: an open plan → MANAGING; a live working order the
        lease covers → PENDING (with its real order id); an unexplained working order →
        incident + entries locked. Reads only — reconciliation never places or cancels."""
        self._started = True
        plan = _read_json(self._plan_path())
        if plan and str(plan.get("status") or "open").lower() == "open" and plan.get("underlying"):
            self.state = MANAGING
            return
        lease_doc = _read_json(self.base_dir / LEASE_FILENAME)
        lease = lease_doc.get("lease") if isinstance(lease_doc.get("lease"), dict) else lease_doc
        rows = await self.client.open_option_orders(self.account_number)
        if rows:
            row = _order_row_to_guard(rows[0])
            covered = (lease and lease.get("lease_id")
                       and str(row.get("option_id") or "") == str(lease.get("option_id") or ""))
            if covered:
                self.lease = dict(lease)
                self.pending_order = {"order_id": row.get("order_id"), "ref_id": None,
                                      "placed_at": now.isoformat(timespec="seconds"),
                                      "limit_price": row.get("limit_price"),
                                      "adopted_at_startup": True}
                self.state = PENDING
                return
            self.entries_locked = True
            self.counts["incidents"] += 1
            self._journal({"reason": "unexplained_open_option_order_at_startup",
                           "order_id": row.get("order_id"),
                           "underlying": row.get("symbol"),
                           "option_id": row.get("option_id")},
                          "execution_safety_incident" if self.mode == MODE_LIVE
                          else SHADOW_INCIDENT,
                          shadow=self.mode != MODE_LIVE, now=now)
        self.state = IDLE

    def _adopt_external_position(self, now: datetime) -> None:
        """Move IDLE/WATCHING -> MANAGING when a plan this lane did not open is live.

        Same adoption rule as `_startup_reconcile` (open status + an underlying), just applied
        every tick instead of once. Read-only: it never places, never cancels, and never writes
        the canonical plan — `_plan_write_path` already redirects non-live modes to the shadow
        copy, so adopting in shadow cannot clobber the controller's `thesis` or
        `management.best_seen_bid`.

        Deliberately does NOT adopt while `entries_locked`: an unexplained working order at
        startup means we do not understand the account, and managing into that is how a
        misunderstanding becomes an order.
        """
        if self.entries_locked or self.pending_order:
            return
        plan = _read_json(self._plan_path())
        if not plan or str(plan.get("status") or "open").lower() != "open":
            return
        if not plan.get("underlying"):
            return
        self.state = MANAGING
        self._journal({"underlying": plan.get("underlying"),
                       "option_id": plan.get("option_id"),
                       "trade_id": plan.get("trade_id"),
                       "reason": "adopted_external_position",
                       "opened_by": "controller"},
                      "shadow_position_adopted", shadow=self.mode != MODE_LIVE, now=now)

    # ── the tick ────────────────────────────────────────────────────────────────────────────

    async def tick(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        self.counts["ticks"] += 1
        status: dict[str, Any] = {"ts": now.isoformat(timespec="seconds"), "mode": self.mode,
                                  "account_masked": "****" + self.account_number[-4:]}
        try:
            if not in_session(now):
                status.update({"state": self.state, "session_open": False})
                self._write_heartbeat(status)
                return status
            if not self._started:
                await self._startup_reconcile(now)
            self._reload_intents()
            paused = self._paused()

            self.last_tape, self.tape_state = await build_tape(self.client, self.tape_state, now)

            # ADOPT A POSITION THIS LANE DID NOT OPEN (2026-08-13). `_startup_reconcile` runs
            # exactly once (`self._started`), so a position the CONTROLLER opens mid-session was
            # invisible forever: from IDLE the tick only ran `_watch`, which looks at armed
            # intents and nothing else. Measured live — the controller held IWM 303P from 11:54:29
            # to 12:07:07 while this daemon sat IDLE for 711 ticks and emitted ZERO exit intents.
            #
            # Two consequences, the second serious. Gate 1 requires `shadow_exit_intent` on "any
            # live Hermes position" and could therefore NEVER have been satisfied. And at
            # `exits_live` Hermes DEFERS exits to this lane — against a position this lane cannot
            # see, which is nobody managing it at all.
            #
            # Adoption is read-only and mode-agnostic: in shadow it produces exit INTENTS, which is
            # exactly the evidence the gate asks for.
            if self.state in (IDLE, WATCHING):
                self._adopt_external_position(now)

            if self.state in (IDLE, WATCHING):
                await self._watch(now, paused)
            elif self.state == PENDING:
                await self._pending(now)
            elif self.state == MANAGING:
                await self._manage(now, paused)

            status.update({"state": self.state, "session_open": True, "paused": paused,
                           "entries_locked": self.entries_locked,
                           "intents_armed": len(self._armed_intents(now)),
                           "counts": dict(self.counts),
                           "tape_generated_at": self.last_tape.get("generated_at")})
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("fast-lane tick error")
            status.update({"state": self.state, "error": self.last_error})
        if self.last_error:
            status["last_error"] = self.last_error
        self._write_heartbeat(status)
        return status

    # ── WATCHING ────────────────────────────────────────────────────────────────────────────

    def _armed_intents(self, now: datetime) -> list[dict]:
        out = []
        for intent in self.intents:
            iid = str(intent.get("intent_id") or "")
            istate = self.intent_states.get(iid) or {}
            if istate.get("fired_at") or str(intent.get("status") or "").lower() != "armed":
                continue
            refused = istate.get("last_refusal_ts")
            if refused:
                try:
                    ref_dt = datetime.fromisoformat(str(refused).replace("Z", "+00:00"))
                    if (now - ref_dt).total_seconds() < REFIRE_COOLDOWN_SECONDS:
                        continue
                except ValueError:
                    pass
            out.append(intent)
        return out

    async def _watch(self, now: datetime, paused: bool) -> None:
        armed = self._armed_intents(now)
        if not armed:
            self.state = IDLE
            return
        self.state = WATCHING
        fired = None
        for intent in armed:
            iid = str(intent.get("intent_id") or "")
            result = evaluate_armed_intent(intent, self.last_tape,
                                           self.intent_states.get(iid, {}).get("counter"),
                                           now)
            entry = self.intent_states.setdefault(iid, {})
            entry["counter"] = result["state"]
            entry["last_result"] = {"fires": result["fires"], "reasons": result["reasons"]}
            if result["fires"] and fired is None:
                fired = (intent, result)
        self._persist_intent_states()
        if fired is None:
            return
        intent, result = fired
        if paused:
            logger.warning("intent %s fired while paused — not firing", intent.get("intent_id"))
            return
        if self.entries_locked:
            logger.warning("intent %s fired with entries locked — not firing",
                           intent.get("intent_id"))
            return
        self.counts["fires"] += 1
        self._journal({"intent_id": intent.get("intent_id"), "checks": result["checks"],
                       "underlying": intent.get("symbol"),
                       "option_id": (intent.get("contract") or {}).get("option_id")},
                      "shadow_trigger_fired", shadow=True, now=now)
        await self._fire(intent, now)

    # ── FIRING ──────────────────────────────────────────────────────────────────────────────

    def _candidate_snapshot(self, intent: dict, now: datetime) -> dict:
        contract = dict(intent.get("contract") or {})
        return {
            "ticker": intent.get("symbol"), "direction": intent.get("direction"),
            "created_at": now.isoformat(timespec="seconds"),
            "option_id": contract.get("option_id"),
            "expiration_date": contract.get("expiration_date"),
            "strike_price": contract.get("strike_price"),
            "option_type": contract.get("option_type"),
            "source": "fast_lane_armed_intent", "intent_id": intent.get("intent_id"),
        }

    async def _fire(self, intent: dict, now: datetime) -> None:
        iid = str(intent.get("intent_id") or "")
        quote, broker = await asyncio.gather(
            self.client.option_quote_by_id((intent.get("contract") or {}).get("option_id")),
            self.client.broker_truth(self.account_number, now=now))
        contract = _option_quote_to_contract(intent, quote, now)
        candidate = self._candidate_snapshot(intent, now)
        live = self.mode == MODE_LIVE
        convert = run_convert(
            candidate_json=candidate, market_json=self.last_tape, broker_json=broker,
            contract_json=contract, journal_path=self.journal_path,
            state_dir=str(self.base_dir), write=live, journal=live,
            journal_events=None if live else self._real_events(), now=now)
        if not live:
            self._journal({"intent_id": iid, "converted": convert.get("converted"),
                           "stage": convert.get("stage"),
                           "reason_codes": convert.get("reason_codes"),
                           "underlying": intent.get("symbol"),
                           "option_id": (intent.get("contract") or {}).get("option_id")},
                          "shadow_convert_result", shadow=True, now=now)
        if not convert.get("converted"):
            self.counts["refusals"] += 1
            entry = self.intent_states.setdefault(iid, {})
            entry["last_refusal_ts"] = now.isoformat(timespec="seconds")
            entry["last_refusal"] = convert.get("reason_codes")
            self._persist_intent_states()
            self.state = WATCHING
            return

        lease = (convert.get("lease") or {})
        limit = _num(lease.get("max_limit_price"))
        contract_ask = _num(contract.get("ask"))
        limits = [x for x in (limit, contract_ask) if x is not None]
        if not limits:                                          # no priceable quote: stand down
            self.counts["refusals"] += 1
            self.state = WATCHING
            return
        # Place at the ask, capped by the lease's chase ceiling — never above either.
        place_limit = min(limits)
        order_args = self.client.build_order_args(
            account_number=self.account_number, option_id=str(lease.get("option_id")),
            quantity=int(_num(lease.get("quantity")) or 1), limit_price=place_limit)

        if not live:
            self._journal({"intent_id": iid, "order_args": order_args,
                           "lease": {k: lease.get(k) for k in
                                     ("lease_id", "option_id", "max_limit_price", "max_debit",
                                      "expires_at", "tier")},
                           "quote": {"bid": contract.get("bid"), "ask": contract.get("ask")},
                           "limit_price": place_limit,
                           "underlying": intent.get("symbol"),
                           "option_id": lease.get("option_id")},
                          SHADOW_ORDER_INTENT, shadow=True, now=now)
            entry = self.intent_states.setdefault(iid, {})
            entry["fired_at"] = now.isoformat(timespec="seconds")
            self._persist_intent_states()
            self.state = WATCHING
            return

        # LIVE: ONE review, then guard-consume-place inside the lease window.
        review = await self.client.review_option_order(order_args)
        guard_order = order_for_guard(
            order_args, symbol=str(intent.get("symbol") or ""),
            option_type=(intent.get("contract") or {}).get("option_type"),
            strike_price=_num((intent.get("contract") or {}).get("strike_price")),
            expiration_date=(intent.get("contract") or {}).get("expiration_date"))
        ref_id = str(uuid.uuid4())

        async def _place():
            return await self.client.place_option_order(order_args, ref_id=ref_id)

        check, placed = await consume_then(guard_order, lease, ledger_path=self.ledger_path,
                                           now=now, place=_place)
        if not check["allowed"]:
            self.counts["refusals"] += 1
            self._journal({"intent_id": iid, "reasons": check["reasons"],
                           "underlying": intent.get("symbol"),
                           "option_id": lease.get("option_id")},
                          "no_trade_decision", shadow=False, now=now)
            self.state = WATCHING
            return
        placed = placed if isinstance(placed, dict) else {}
        placed_row = placed.get("data") if isinstance(placed.get("data"), dict) else placed
        order_id = (placed_row or {}).get("id") or (placed_row or {}).get("order_id")
        self.counts["placed"] += 1
        self.lease = dict(lease)
        self.pending_order = {"order_id": order_id, "ref_id": ref_id,
                              "placed_at": now.isoformat(timespec="seconds"),
                              "limit_price": place_limit, "intent_id": iid,
                              "review_alerts": (review or {}).get("alerts")
                              if isinstance(review, dict) else None}
        entry = self.intent_states.setdefault(iid, {})
        entry["fired_at"] = now.isoformat(timespec="seconds")
        self._persist_intent_states()
        self.state = PENDING

    # ── PENDING ─────────────────────────────────────────────────────────────────────────────

    async def _pending(self, now: datetime) -> None:
        order_id = (self.pending_order or {}).get("order_id")
        row = await self.client.order_state(order_id, self.account_number) if order_id else {}
        order = _order_row_to_guard(row, account_symbol=(self.lease or {}).get("symbol"))
        guard = evaluate_order_guard(order, lease=self.lease, market_snapshot=self.last_tape,
                                     now=now)
        state = guard["state"]
        filled_qty = _num(order.get("filled_quantity")) or 0

        if state == FILLED_FRESH:
            await self._on_entry_fill(order, now)
            return
        if state in CANCEL_STATES:
            if order_id:
                await self.client.cancel_option_order(order_id, self.account_number)
            self._journal({"order_id": order_id, "guard_state": state,
                           "reasons": guard.get("reasons"),
                           "underlying": order.get("symbol"),
                           "option_id": order.get("option_id")},
                          "no_trade_decision", shadow=self.mode != MODE_LIVE, now=now)
            if filled_qty > 0:                                 # partial past lease: manage it
                await self._on_entry_fill(order, now, quantity=int(filled_qty))
                return
            self.pending_order, self.lease = None, None
            self.state = WATCHING
            return
        if state in INCIDENT_STATES:
            self.counts["incidents"] += 1
            self.entries_locked = True
            self._journal({"order_id": order_id, "guard_state": state,
                           "reasons": guard.get("reasons"),
                           "underlying": order.get("symbol"),
                           "option_id": order.get("option_id")},
                          "execution_safety_incident" if self.mode == MODE_LIVE
                          else SHADOW_INCIDENT,
                          shadow=self.mode != MODE_LIVE, now=now)
            if order.get("status") == "filled" or filled_qty > 0:
                await self._on_entry_fill(order, now, incident=True)
            else:
                self.pending_order, self.lease = None, None
                self.state = IDLE
            return
        if state == NO_ORDER:
            # The broker says our order is gone (externally cancelled/rejected) or the adopted
            # order never resolved — stand down cleanly; a new entry needs a fresh fire.
            if order_id:
                self._journal({"order_id": order_id, "reasons": guard.get("reasons"),
                               "underlying": order.get("symbol"),
                               "option_id": order.get("option_id")},
                              "no_trade_decision", shadow=self.mode != MODE_LIVE, now=now)
            self.pending_order, self.lease = None, None
            self.state = WATCHING
            return
        # PENDING_FRESH (incl. a partial inside the lease): keep polling.

    async def _on_entry_fill(self, order: dict, now: datetime, quantity: int | None = None,
                             incident: bool = False) -> None:
        lease = self.lease or {}
        intent = next((i for i in self.intents
                       if str(i.get("intent_id")) == (self.pending_order or {}).get("intent_id")),
                      {})
        fill_price = _num(order.get("average_price")) or _num(order.get("limit_price")) \
            or (self.pending_order or {}).get("limit_price")
        qty = quantity if quantity is not None else int(_num(order.get("quantity")) or 1)
        exit_plan = dict(intent.get("exit_plan") or {})
        trade_id = (f"{str(lease.get('symbol') or 'x').lower()}-"
                    f"{now.astimezone(_ET).strftime('%Y%m%d-%H%M%S')}-fastlane")
        plan = {
            "schema_version": 1, "trade_id": trade_id, "status": "open",
            "opened_at": now.isoformat(timespec="seconds"),
            "account_masked": "****" + self.account_number[-4:],
            "underlying": lease.get("symbol"), "chain_symbol": lease.get("symbol"),
            "option_id": lease.get("option_id"),
            "expiration_date": lease.get("expiration_date"),
            "strike_price": lease.get("strike_price"),
            "option_type": lease.get("option_type"),
            "side": "long", "quantity": qty, "entry_price": fill_price,
            "entry_order_id": order.get("order_id"),
            "execution_lease_id": lease.get("lease_id"),
            "mode": str(exit_plan.get("mode") or "scalp"),
            "opened_by": "fast_lane", "incident_entry": bool(incident),
            **{k: exit_plan[k] for k in ("profit_rules", "thesis", "risk_rules", "time_rules",
                                         "bid_memory") if isinstance(exit_plan.get(k), dict)},
            "management": {"cadence_seconds": 3, "best_seen_bid": None},
        }
        self.base_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._plan_write_path(), json.dumps(plan, indent=2, default=str))
        self._journal({"trade_id": trade_id, "underlying": lease.get("symbol"),
                       "option_id": lease.get("option_id"), "quantity": qty,
                       "fill_price": fill_price, "order_id": order.get("order_id")},
                      "entry_fill", shadow=self.mode != MODE_LIVE, now=now)
        self.pending_order = None
        self.state = MANAGING

    # ── MANAGING ────────────────────────────────────────────────────────────────────────────

    def _position_snapshot(self, plan: dict, quote: Any, now: datetime) -> dict:
        inner = (quote.get("quote") if isinstance(quote, dict)
                 and isinstance(quote.get("quote"), dict) else quote)
        inner = inner if isinstance(inner, dict) else {}
        bid = _num(inner.get("bid_price") or inner.get("bid"))
        ask = _num(inner.get("ask_price") or inner.get("ask"))
        mark = _num(inner.get("mark_price") or inner.get("mark"))
        if mark is None and bid is not None and ask is not None:
            mark = round((bid + ask) / 2.0, 4)
        snap: dict[str, Any] = {"now_et": now.astimezone(_ET).isoformat(timespec="seconds")}
        if bid is not None:
            snap["option_bid"] = bid
        if ask is not None:
            snap["option_ask"] = ask
        if mark is not None:
            snap["option_mark"] = mark
        underlying = str(plan.get("underlying") or "").upper()
        for symbol, key in (("SPY", "spy_last"), ("QQQ", "qqq_last"), ("IWM", "iwm_last"),
                            ("VIXY", "vixy")):
            last = _num((self.last_tape.get(symbol) or {}).get("last"))
            if last is not None:
                snap[key] = last
                if symbol == underlying:
                    snap["underlying_last"] = last
        return snap

    async def _manage(self, now: datetime, paused: bool) -> None:
        plan = _read_json(self._plan_path())
        if not plan or str(plan.get("status") or "open").lower() != "open":
            self.exit_order, self.lease = None, None
            self.state = IDLE
            return
        if self.exit_order:
            await self._poll_exit(plan, now, paused)
            return
        quote = await self.client.option_quote_by_id(str(plan.get("option_id")))
        snapshot = self._position_snapshot(plan, quote, now)
        result = evaluate_position(plan, snapshot, now=now)
        best_seen = result.get("best_seen_bid")
        if best_seen is not None and best_seen != (plan.get("management") or {}).get(
                "best_seen_bid"):
            plan.setdefault("management", {})["best_seen_bid"] = best_seen
            atomic_write_text(self._plan_write_path(), json.dumps(plan, indent=2, default=str))
        decision = result["decision"]
        trigger = next((t for t in result["triggers"] if t["type"] == decision), {})
        exit_needed = str(trigger.get("action") or "").lower() in _EXIT_ACTIONS
        if not exit_needed:
            return
        bid = snapshot.get("option_bid")
        if self.mode == MODE_SHADOW or paused:
            self._journal({"trade_id": plan.get("trade_id"), "trigger_type": decision,
                           "bid": bid, "best_seen_bid": best_seen,
                           "would_limit": bid, "paused": paused,
                           "underlying": plan.get("underlying"),
                           "option_id": plan.get("option_id"),
                           "detail": trigger.get("detail")},
                          SHADOW_EXIT_INTENT, shadow=True, now=now)
            return
        if bid is None:
            logger.warning("exit trigger %s with no live bid — holding for a valued tick",
                           decision)
            return
        await self._place_exit(plan, bid, decision, now)

    async def _place_exit(self, plan: dict, limit: float, trigger_type: str,
                          now: datetime) -> None:
        order_args = self.client.build_order_args(
            account_number=self.account_number, option_id=str(plan.get("option_id")),
            quantity=int(_num(plan.get("quantity")) or 1), limit_price=limit,
            position_effect="close")
        guard_order = order_for_guard(order_args, symbol=str(plan.get("underlying") or ""))
        check = pre_place_check(guard_order, None, ledger_path=self.ledger_path, now=now)
        if not check["allowed"]:                               # NVDA/defense only — closes pass
            self.counts["incidents"] += 1
            self._journal({"reasons": check["reasons"], "trade_id": plan.get("trade_id")},
                          "execution_safety_incident", shadow=False, now=now)
            return
        ref_id = str(uuid.uuid4())
        placed = await self.client.place_option_order(order_args, ref_id=ref_id)
        placed_row = placed.get("data") if isinstance(placed, dict) and isinstance(
            placed.get("data"), dict) else placed
        self.exit_order = {"order_id": (placed_row or {}).get("id")
                           or (placed_row or {}).get("order_id"),
                           "limit": limit, "trigger_type": trigger_type,
                           "placed_at": now.isoformat(timespec="seconds"), "replacements": 0}
        self.counts["exits"] += 1

    async def _poll_exit(self, plan: dict, now: datetime, paused: bool) -> None:
        info = self.exit_order or {}
        row = await self.client.order_state(info.get("order_id"), self.account_number)
        order = _order_row_to_guard(row)
        if order.get("status") == "filled":
            exit_price = _num(order.get("average_price")) or info.get("limit")
            entry = _num(plan.get("entry_price"))
            qty = int(_num(plan.get("quantity")) or 1)
            pnl = (round((exit_price - entry) * 100.0 * qty, 2)
                   if exit_price is not None and entry is not None else None)
            plan.update({"status": "closed", "closed_at": now.isoformat(timespec="seconds"),
                         "exit_price": exit_price, "exit_order_id": info.get("order_id"),
                         "realized_pnl": pnl,
                         "close_reason": f"{info.get('trigger_type')} (fast lane)"})
            atomic_write_text(self._plan_write_path(), json.dumps(plan, indent=2, default=str))
            self._journal({"trade_id": plan.get("trade_id"),
                           "underlying": plan.get("underlying"),
                           "option_id": plan.get("option_id"),
                           "realized_pnl": pnl, "exit_price": exit_price,
                           "trigger_type": info.get("trigger_type")},
                          "order_closed", shadow=self.mode == MODE_SHADOW, now=now)
            self.exit_order, self.lease = None, None
            self.state = IDLE
            return
        if order.get("status") == "gone":                      # cancelled/rejected: re-decide
            self.exit_order = None
            return
        if paused:
            return
        placed_at = datetime.fromisoformat(str(info.get("placed_at")))
        quote = await self.client.option_quote_by_id(str(plan.get("option_id")))
        inner = quote.get("quote") if isinstance(quote, dict) and isinstance(
            quote.get("quote"), dict) else (quote if isinstance(quote, dict) else {})
        bid = _num(inner.get("bid_price") or inner.get("bid"))
        faded = (bid is not None and info.get("limit") is not None
                 and bid <= info["limit"] - EXIT_REPRICE_MIN_TICK)
        if ((now - placed_at).total_seconds() > EXIT_REPRICE_AFTER_SECONDS and faded
                and info.get("replacements", 0) < EXIT_MAX_REPLACEMENTS):
            await self.client.cancel_option_order(info.get("order_id"), self.account_number)
            self.exit_order = None
            await self._place_exit(plan, bid, info.get("trigger_type") or "REPRICE", now)
            if self.exit_order:
                self.exit_order["replacements"] = info.get("replacements", 0) + 1


# ── the loop ────────────────────────────────────────────────────────────────────────────────

TICK_SECONDS = {IDLE: 15.0, WATCHING: 3.0, FIRING: 0.5, PENDING: 2.0, MANAGING: 3.0}
OFF_SESSION_SLEEP_SECONDS = 60.0


async def run_fast_lane(client: Any, *, account_number: str,
                        state_dir: str | Path = DEFAULT_STATE_DIR, mode: str | None = None,
                        once: bool = False, journal_path: str | None = None) -> dict:
    """Session runner: token-gate the mode, connect, tick until 16:05 ET (or `once`), then
    journal the session summary. A stale token (<48h) DOWNGRADES exits/live to shadow with a
    loud journal event rather than dying — shadow evidence still accrues."""
    from execution.odte_mcp_client import TOKEN_MIN_HOURS, McpAuthStale

    requested = mode
    try:
        client.preflight(TOKEN_MIN_HOURS)
    except McpAuthStale as exc:
        logger.error("token preflight failed — forcing shadow mode: %s", exc)
        mode = MODE_SHADOW
    except AttributeError:                                     # bare test doubles: no preflight
        pass
    daemon = FastLaneDaemon(client, account_number=account_number, state_dir=state_dir,
                            mode=mode, journal_path=journal_path)
    if requested and daemon.mode != requested:
        daemon._journal({"requested_mode": requested, "effective_mode": daemon.mode,
                         "reason": "stage_file_or_token_preflight"},
                        "shadow_session_start", shadow=True, now=datetime.now(timezone.utc))
    else:
        daemon._journal({"effective_mode": daemon.mode},
                        "shadow_session_start", shadow=True, now=datetime.now(timezone.utc))
    if hasattr(client, "connect"):
        await client.connect()
    try:
        while True:
            now = datetime.now(timezone.utc)
            status = await daemon.tick(now)
            if once:
                return status
            et = now.astimezone(_ET)
            if et.time() >= SESSION_END_ET and et.weekday() < 5:
                break
            sleep_s = (TICK_SECONDS.get(daemon.state, 3.0) if status.get("session_open")
                       else OFF_SESSION_SLEEP_SECONDS)
            await asyncio.sleep(sleep_s)
    finally:
        end = datetime.now(timezone.utc)
        daemon._journal({"counts": daemon.counts, "entries_locked": daemon.entries_locked,
                         "last_error": daemon.last_error},
                        "shadow_session_end", shadow=True, now=end)
        if hasattr(client, "aclose"):
            await client.aclose()
    return {"state": daemon.state, "counts": daemon.counts}
