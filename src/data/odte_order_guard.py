"""0DTE pending-order guard — cancel-first state machine. PURE/OFFLINE, places NO orders.

Companion to `odte_execution_policy`: the lease says WHETHER an entry may be submitted; this guard
says what to DO with the order the broker actually reports, every tick, until it is filled-fresh or
cancelled. It consumes fresh broker order truth (caller-supplied — this module never calls a
broker), the execution lease, the current market snapshot, and the current time, and classifies:

    NO_ORDER                    nothing open/cancelled — nothing to guard
    PENDING_FRESH               unfilled, lease unexpired, thesis rails still valid — keep polling
    CANCEL_STALE_ENTRY          unfilled past the lease TTL (or no valid lease) — CANCEL NOW
    CANCEL_THESIS_INVALID       unfilled, lease alive, but the thesis rails fired — CANCEL NOW
    FILLED_FRESH                filled at/before lease expiry with exact identity — manage it
    FILLED_WITHOUT_VALID_LEASE  filled after expiry / without a lease — SAFETY INCIDENT
    BROKER_MISMATCH_BLOCKED     broker order identity/size disagrees with the lease — SAFETY INCIDENT

Invariants (2026-07-23 delayed-fill remediation):
  * An unfilled entry order cannot survive beyond its authorization lease — at TTL expiry the
    output is CANCEL_STALE_ENTRY immediately.
  * A stale pending order can never be extended; a new order requires a NEW candidate confirmation
    and a NEW lease.
  * A fill outside a valid lease is never reclassified as the normal path — it is an
    `execution_safety_incident` that PROHIBITS new entries; the external broker lane must flatten
    or alert per the explicitly reviewed emergency policy.
  * Broker truth wins: this guard only classifies what the broker reports; it invents nothing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_DATA_DIR, atomic_write_text
from data.odte_execution_policy import (
    CONSUMED_LEASES_FILENAME,
    LEASE_FILENAME,
    lease_expired,
    lease_seconds_remaining,
    load_consumed_ids,
    record_consumed,
)

SCHEMA_VERSION = 1

DEFAULT_STATE_DIR = ODTE_DATA_DIR
ORDER_GUARD_FILENAME = "order_guard_decision.json"

NO_ORDER = "NO_ORDER"
PENDING_FRESH = "PENDING_FRESH"
CANCEL_STALE_ENTRY = "CANCEL_STALE_ENTRY"
CANCEL_THESIS_INVALID = "CANCEL_THESIS_INVALID"
FILLED_FRESH = "FILLED_FRESH"
FILLED_WITHOUT_VALID_LEASE = "FILLED_WITHOUT_VALID_LEASE"
BROKER_MISMATCH_BLOCKED = "BROKER_MISMATCH_BLOCKED"

GUARD_STATES = (NO_ORDER, PENDING_FRESH, CANCEL_STALE_ENTRY, CANCEL_THESIS_INVALID,
                FILLED_FRESH, FILLED_WITHOUT_VALID_LEASE, BROKER_MISMATCH_BLOCKED)

# States that demand an immediate broker cancel.
CANCEL_STATES = (CANCEL_STALE_ENTRY, CANCEL_THESIS_INVALID)
# States that are execution safety incidents: journal, prohibit new entries, flatten/alert lane.
INCIDENT_STATES = (FILLED_WITHOUT_VALID_LEASE, BROKER_MISMATCH_BLOCKED)

_PENDING_STATUSES = {"pending", "queued", "confirmed", "placed", "open", "live", "unconfirmed",
                     "submitted", "partially_filled", "pending_cancelled"}
_FILLED_STATUSES = {"filled", "executed", "complete", "completed"}
_GONE_STATUSES = {"", "none", "cancelled", "canceled", "rejected", "expired", "failed"}


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_json(path: str | None, raw_json: str | None, default: dict | None = None) -> dict:
    if raw_json:
        obj = json.loads(raw_json)
    elif path:
        obj = json.loads(Path(os.path.expanduser(path)).read_text())
    else:
        return dict(default or {})
    if not isinstance(obj, dict):
        raise ValueError("payload must be a JSON object")
    return obj


def _lease_dict(lease: dict | None) -> dict:
    lease = _dict(lease)
    return _dict(lease.get("lease")) or lease


def _order_status(order: dict) -> str:
    s = str(order.get("status") or order.get("state") or "").strip().lower()
    # A partial fill is a LIVE working order (the remainder is still at the broker) even though
    # the broker stamps filled_at/filled_quantity on it, and a pending cancel is not yet gone —
    # both must stay guarded, never read as a completed fill or as "no live order".
    if s in ("partially_filled", "pending_cancelled"):
        return "pending"
    if s in _FILLED_STATUSES or order.get("filled_at"):
        return "filled"
    if s in _PENDING_STATUSES:
        return "pending"
    return "gone"


def _symbol_block(market: dict, symbol: str | None) -> dict:
    if not symbol:
        return {}
    for key in (symbol.upper(), symbol.lower()):
        block = market.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _identity_mismatches(order: dict, lease: dict) -> list[str]:
    """Exact-identity + maximums check between the broker order and the lease. Any disagreement is
    a hard mismatch — a contract for another underlying is never an equivalent substitute."""
    out: list[str] = []
    o_sym = str(order.get("symbol") or order.get("underlying") or order.get("chain_symbol")
                or "").upper()
    if o_sym and o_sym != str(lease.get("symbol") or "").upper():
        out.append(f"symbol_mismatch:{o_sym}!={lease.get('symbol')}")
    o_id = order.get("option_id") or order.get("occ_symbol")
    if o_id and str(o_id) != str(lease.get("option_id")):
        out.append("option_id_mismatch")
    o_type = str(order.get("option_type") or order.get("type") or "").strip().lower()
    if o_type and o_type != str(lease.get("option_type") or "").lower():
        out.append("option_type_mismatch")
    o_strike = _num(order.get("strike_price") or order.get("strike"))
    l_strike = _num(lease.get("strike_price"))
    if o_strike is not None and l_strike is not None and abs(o_strike - l_strike) > 1e-9:
        out.append("strike_mismatch")
    o_exp = order.get("expiration_date") or order.get("expiration")
    if o_exp and str(o_exp) != str(lease.get("expiration_date")):
        out.append("expiration_mismatch")
    qty = _num(order.get("quantity"))
    l_qty = _num(lease.get("quantity"))
    if qty is not None and l_qty is not None and qty > l_qty:
        out.append("quantity_exceeds_lease")
    limit = _num(order.get("limit_price") or order.get("price"))
    l_limit = _num(lease.get("max_limit_price"))
    if limit is not None and l_limit is not None and limit > l_limit + 1e-9:
        out.append("limit_exceeds_lease")
    debit = _num(order.get("debit"))
    if debit is None and qty is not None and limit is not None:
        debit = qty * limit * 100.0
    l_debit = _num(lease.get("max_debit"))
    if debit is not None and l_debit is not None and debit > l_debit + 1e-6:
        out.append("debit_exceeds_lease")
    return out


def _thesis_invalidated(direction: str | None, symbol: str | None, market: dict) -> str | None:
    """Deterministic thesis rails over the CURRENT market snapshot. Conservative + explicit:
    an explicit `thesis_invalidated: true`, a VWAP flip against the direction, or price at/through
    a supplied `invalidation_level`."""
    if market.get("thesis_invalidated") is True:
        return "explicit thesis_invalidated flag on the market snapshot"
    block = _symbol_block(market, symbol)
    above = block.get("above_vwap")
    last = _num(block.get("last") or block.get("price") or block.get("spot") or block.get("mark"))
    level = _num(market.get("invalidation_level") or block.get("invalidation_level"))
    if direction == "bearish":
        if above is True:
            return f"{symbol} reclaimed VWAP against the bearish thesis"
        if level is not None and last is not None and last >= level:
            return f"{symbol} {last} at/above invalidation level {level}"
    elif direction == "bullish":
        if above is False:
            return f"{symbol} lost VWAP against the bullish thesis"
        if level is not None and last is not None and last <= level:
            return f"{symbol} {last} at/below invalidation level {level}"
    return None


def _next_step(state: str) -> tuple[str, str]:
    if state == NO_ORDER:
        return ("no open entry order — nothing to guard; any new entry needs a fresh candidate "
                "confirmation and a NEW lease", "odte-loop-status")
    if state == PENDING_FRESH:
        return ("order pending inside the lease — poll broker truth and re-run this guard every "
                "tick; at TTL expiry or thesis invalidation cancel IMMEDIATELY",
                "odte-order-guard --order <fresh broker order.json>")
    if state == CANCEL_STALE_ENTRY:
        return ("CANCEL NOW: the entry order outlived its lease — cancel at the broker, verify the "
                "cancel with fresh order truth, and journal it; a new order requires a NEW "
                "candidate confirmation and lease (a stale order is never extended)",
                "broker-cancel (Hermes MCP lane) → odte-order-guard --order <fresh order.json>")
    if state == CANCEL_THESIS_INVALID:
        return ("CANCEL NOW: thesis invalidated before the fill — cancel at the broker, verify, "
                "journal; do not re-enter on the dead thesis",
                "broker-cancel (Hermes MCP lane) → odte-order-guard --order <fresh order.json>")
    if state == FILLED_FRESH:
        return ("filled inside the lease — journal order_filled and start the position watch "
                "immediately", "odte-position --snapshot <live.json>")
    if state == FILLED_WITHOUT_VALID_LEASE:
        return ("EXECUTION SAFETY INCIDENT: fill landed outside a valid lease — journal "
                "execution_safety_incident, PROHIBIT new entries, and flatten or alert per the "
                "explicitly reviewed emergency policy (Hermes MCP lane); never manage this as a "
                "normal A+ position",
                "flatten-or-alert (Hermes MCP lane) → odte-journal --event-json "
                "'{\"event_type\":\"execution_safety_incident\",...}'")
    # BROKER_MISMATCH_BLOCKED
    return ("EXECUTION SAFETY INCIDENT: broker order disagrees with the lease identity/maximums — "
            "block everything, cancel if unfilled, journal execution_safety_incident, and "
            "reconcile broker truth before any further action",
            "broker-cancel (Hermes MCP lane) → odte-journal --event-json "
            "'{\"event_type\":\"execution_safety_incident\",...}'")


def evaluate_order_guard(order: dict | None = None, *, lease: dict | None = None,
                         market_snapshot: dict | None = None,
                         now: datetime | None = None) -> dict:
    """PURE cancel-first state machine over broker order truth + the execution lease. Never raises,
    never calls a broker, places NO orders — it only classifies and instructs."""
    now = now or datetime.now(timezone.utc)
    order = _dict(order)
    ld = _lease_dict(lease)
    market = _dict(market_snapshot)
    reasons: list[str] = []

    status = _order_status(order)
    has_lease = bool(ld.get("lease_id"))
    expired = lease_expired(ld, now) if has_lease else True
    remaining = lease_seconds_remaining(ld, now) if has_lease else None
    submitted = _parse_ts(order.get("submitted_at") or order.get("created_at"))
    pending_age = round((now - submitted).total_seconds(), 3) if submitted else None
    direction = str(ld.get("direction") or "").lower() or None
    symbol = (order.get("symbol") or order.get("underlying") or ld.get("symbol"))

    # A lease can NEVER retroactively cover an order submitted before it was issued — that would be
    # "extending" a stale order with a fresh lease, which the incident remediation forbids.
    issued = _parse_ts(ld.get("issued_at")) if has_lease else None
    predates_lease = bool(submitted is not None and issued is not None and submitted < issued)

    if status == "gone":
        if order:
            reasons.append(f"order status '{order.get('status') or 'none'}' — no live entry order")
        else:
            reasons.append("no broker order supplied — nothing open")
        state = NO_ORDER
    else:
        mismatches = _identity_mismatches(order, ld) if has_lease else []
        if mismatches:
            reasons.append("broker order disagrees with the lease: " + ", ".join(mismatches))
            state = BROKER_MISMATCH_BLOCKED
        elif predates_lease and status == "pending":
            reasons.append("order predates the lease — a lease can never retroactively cover or "
                           "extend an existing order; cancel and run a full fresh cycle")
            state = CANCEL_STALE_ENTRY
        elif predates_lease and status == "filled":
            reasons.append("filled order predates the lease — its submission was never covered by "
                           "a valid authorization")
            state = FILLED_WITHOUT_VALID_LEASE
        elif status == "filled":
            filled_at = _parse_ts(order.get("filled_at"))
            lease_exp = _parse_ts(ld.get("expires_at"))
            if not has_lease:
                reasons.append("fill reported with NO execution lease — unauthorized order")
                state = FILLED_WITHOUT_VALID_LEASE
            elif filled_at is None or lease_exp is None:
                reasons.append("fill/lease timestamp missing/unparseable — freshness cannot be "
                               "proven (fail closed)")
                state = FILLED_WITHOUT_VALID_LEASE
            elif filled_at > lease_exp:
                reasons.append(f"filled at {filled_at.isoformat(timespec='seconds')} AFTER lease "
                               f"expiry {ld.get('expires_at')} — the stale-fill incident path")
                state = FILLED_WITHOUT_VALID_LEASE
            else:
                reasons.append("filled inside the lease window with exact identity")
                state = FILLED_FRESH
        else:  # pending
            if not has_lease:
                reasons.append("pending entry order with NO execution lease — cancel immediately")
                state = CANCEL_STALE_ENTRY
            elif expired:
                reasons.append(f"lease expired {ld.get('expires_at')} with the order still "
                               "pending — cancel immediately; a stale order is never extended")
                state = CANCEL_STALE_ENTRY
            else:
                invalid = _thesis_invalidated(direction, str(symbol or "").upper() or None, market)
                if invalid:
                    reasons.append(f"thesis invalidated before fill: {invalid}")
                    state = CANCEL_THESIS_INVALID
                else:
                    reasons.append("pending inside the lease; thesis rails still valid")
                    state = PENDING_FRESH

    cancel_required = state in CANCEL_STATES or (state == BROKER_MISMATCH_BLOCKED
                                                 and status == "pending")
    incident = state in INCIDENT_STATES
    next_action, next_command = _next_step(state)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "state": state,
        "underlying": (str(symbol).upper() if symbol else None),
        "lease_id": ld.get("lease_id"),
        "order": {k: order.get(k) for k in
                  ("status", "symbol", "option_id", "option_type", "strike_price",
                   "expiration_date", "quantity", "limit_price", "submitted_at", "filled_at",
                   "filled_quantity")
                  if order.get(k) is not None},
        "cancel_required": cancel_required,
        "safety_incident": incident,
        "prohibit_new_entries": incident,
        "pending_order_age_seconds": pending_age,
        "lease_seconds_remaining": remaining,
        "reasons": reasons,
        "next_action": next_action,
        "next_command": next_command,
        "places_orders": False,
    }


def run_order_guard(order_json: str | None = None, order_path: str | None = None,
                    lease_json: str | None = None, lease_path: str | None = None,
                    market_json: str | None = None, market_path: str | None = None,
                    state_dir: str | None = None, write: bool = False,
                    now: datetime | None = None) -> dict:
    """Load broker order truth + lease + market snapshot and run the guard. NO broker calls.

    Defaults the lease to the canonical ``data/odte/execution_lease.json`` when present. On any
    order that exists at the broker (pending/filled/cancel-required), the matching lease is
    recorded as CONSUMED in ``data/odte/consumed_leases.json`` — a submitted order burns its lease,
    so a second submission on the same lease fails closed. ``write=True`` stores the decision at
    ``data/odte/order_guard_decision.json`` atomically; the payload carries
    ``lease_consumed_now=True`` the first time a lease is burned (the CLI journals it)."""
    base = Path(os.path.expanduser(state_dir or DEFAULT_STATE_DIR))
    lease = _load_json(lease_path, lease_json)
    if not lease and (base / LEASE_FILENAME).exists():
        try:
            lease = json.loads((base / LEASE_FILENAME).read_text())
        except Exception:
            lease = {}
    payload = evaluate_order_guard(
        _load_json(order_path, order_json) or None,
        lease=lease or None,
        market_snapshot=_load_json(market_path, market_json) or None,
        now=now,
    )
    # Single-use ledger: an order that exists at the broker means its lease is consumed.
    lease_id = payload.get("lease_id")
    if lease_id and payload["state"] != NO_ORDER:
        ledger = base / CONSUMED_LEASES_FILENAME
        already = lease_id in load_consumed_ids(ledger)
        if not already:
            record_consumed(ledger, str(lease_id))
        payload["lease_consumed_now"] = not already
    if write:
        base.mkdir(parents=True, exist_ok=True)
        path = base / ORDER_GUARD_FILENAME
        atomic_write_text(path, json.dumps(payload, indent=2, default=str))
        payload["artifact"] = str(path)
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    order = p.get("order") or {}
    lines = [f"# 0DTE order guard: **{p.get('state')}**",
             "",
             f"Order: {order.get('symbol') or '?'} {order.get('option_type') or ''} "
             f"{order.get('strike_price') or ''} ×{order.get('quantity') or '?'} "
             f"@{order.get('limit_price') or '?'} ({order.get('status') or 'none'})  ",
             f"Cancel required: **{p.get('cancel_required')}**  ·  safety incident: "
             f"**{p.get('safety_incident')}**  ·  new entries prohibited: "
             f"{p.get('prohibit_new_entries')}  ",
             f"Pending age: {p.get('pending_order_age_seconds')}s  ·  lease remaining: "
             f"{p.get('lease_seconds_remaining')}s (lease {p.get('lease_id') or '—'})  ",
             f"Next: **{p.get('next_command')}** — {p.get('next_action')}", ""]
    if p.get("reasons"):
        lines += ["## Why", *[f"- {r}" for r in p["reasons"]]]
    return "\n".join(lines)
