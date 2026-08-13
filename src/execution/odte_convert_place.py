"""Consume a freshly-minted convert lease IN-PROCESS — the slow lane's missing atomic tail.

WHY THIS EXISTS. A 0DTE lease lives 60 seconds (`MAX_LEASE_TTL_SECONDS`, an incident invariant,
never tunable upward), and that budget was sized for "one MCP place round-trip plus a timeout
retry" — ZERO seconds were budgeted for inference. But `run_convert` mints the lease and then the
PROCESS EXITS; everything after is a model turn. `odte_convert`'s own docstring names the seam:

    -> agent: ONE review_option_order + place_option_order inside the 60s lease

Measured lease->submit on every lease that produced one: 14s, 19.6s, 23.3s, 26.0s, and 37.1s —
that last with 3.7 seconds of margin. Three of fifteen leases never produced an order at all. On
2026-08-13 the lease was minted 11:03:36 and expired 11:04:36 while the tick was still running; the
agent re-ran `day_score`/`vehicle_score` instead of placing. The prompt already says "ONE
review_option_order, then place_option_order — NEVER re-review" and cites the 2026-08-04
miss-by-2.2s. The instruction is correct and was still not followed: prohibitions land, positive
instructions do not.

WHY IT LIVES IN `execution/` AND NOT IN `odte_convert`. `odte_convert` is `data`, and the
import-linter contract "data must not import from higher layers" forbids `data -> execution`. The
broker client and the pre-place guard are both `execution`, so the placement CANNOT live beside the
mint. The CLI orchestrates the two halves instead: `run_convert` decides, this places.

NOTHING HERE REASONS. It reads a lease that a deterministic gate chain already authorized and
transmits exactly what that lease permits. Every rail is reused, not reimplemented:
`consume_then`/`pre_place_check` (the same guard the fast-lane daemon uses) and `OdteMcpClient`.

THE HERMES HOOK DOES NOT COVER THIS PATH. `scripts/hermes/odte_order_guard_hook.py` is a
`pre_tool_call` hook on `mcp__robinhood__place_option_order`; a direct `OdteMcpClient` place never
passes through it — exactly as the fast lane never does. `pre_place_check` inside `consume_then`
is therefore mandatory, not belt-and-braces: it enforces the same identity, expiry, single-use and
qty/price/debit ceilings the hook would have.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from data.odte_journal import append_decision_journal
from execution.odte_mcp_client import OdteMcpClient
from execution.odte_pre_place_guard import consume_then, order_for_guard

logger = logging.getLogger(__name__)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def place_limit_for(lease: dict, contract: dict) -> float | None:
    """Place at the ask, capped by the lease's chase ceiling — never above either.

    Identical rule to `odte_fast_lane._fire`: `min` over whichever of the two exist. With no ask
    this falls back to the lease ceiling, which is safe because that ceiling is exactly what
    `authorize_entry` sized against buying power and the tier's debit fraction. Only when NEITHER
    exists is there nothing to price, and the caller stands down.
    """
    limits = [x for x in (_num(lease.get("max_limit_price")), _num(contract.get("ask")))
              if x is not None]
    return min(limits) if limits else None


async def place_converted(payload: dict, *, account_number: str, ledger_path: str,
                          contract: dict | None = None, journal_path: str | None = None,
                          client: Any = None, now: datetime | None = None) -> dict:
    """Review once, then guard-consume-place the lease `run_convert` just minted.

    `contract` MUST be passed by the caller — the same snapshot it handed `run_convert`. The
    convert payload does NOT carry one (verified 2026-08-13: no "contract" key anywhere in its
    return), and reading `payload["contract"]` therefore silently yielded {}, which makes
    `place_limit_for` fall back to the lease ceiling and pay the top of the chase band on EVERY
    fill instead of the ask. Passing it in also keeps `run_convert`'s payload untouched, so the
    no-`--place` path stays byte-identical.

    Returns a report dict; NEVER raises. A refusal or an error leaves the lease unconsumed to
    expire on its own — missing a fill is acceptable, acquiring a stale 0DTE position is not.
    """
    now = now or datetime.now(timezone.utc)
    report: dict[str, Any] = {"placed": False, "reasons": [], "lease_id": None, "order_id": None}
    try:
        if not payload.get("converted"):
            report["reasons"] = ["not_converted"]
            return report
        lease = _dict(payload.get("lease"))
        contract = _dict(contract)
        report["lease_id"] = lease.get("lease_id")

        # The window belongs to the LEASE, not to convert's entry clock. Re-read it at the moment
        # of placing: a slow preflight can spend the budget before we ever get here.
        expires = lease.get("expires_at")
        remaining = None
        if expires:
            try:
                exp = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
                remaining = (exp - now).total_seconds()
            except ValueError:
                remaining = None
        report["lease_seconds_remaining_at_place"] = remaining
        if remaining is not None and remaining <= 0:
            report["reasons"] = ["lease_expired_before_place"]
            return report

        limit = place_limit_for(lease, contract)
        if limit is None:
            report["reasons"] = ["no_priceable_quote"]
            return report
        report["limit_price"] = limit

        own_client = client is None
        client = client or OdteMcpClient()
        try:
            order_args = client.build_order_args(
                account_number=str(account_number), option_id=str(lease.get("option_id")),
                quantity=int(_num(lease.get("quantity")) or 1), limit_price=limit)

            review = await client.review_option_order(order_args)      # exactly ONE, never re-run
            guard_order = order_for_guard(
                order_args, symbol=str(lease.get("symbol") or contract.get("chain_symbol") or ""),
                option_type=lease.get("option_type") or contract.get("option_type"),
                strike_price=_num(lease.get("strike_price") or contract.get("strike_price")),
                expiration_date=(lease.get("expiration_date")
                                 or contract.get("expiration_date")))
            ref_id = str(uuid.uuid4())

            async def _place():
                # The SAME ref_id on a timeout retry — the broker dedupes, so the worst case is
                # one order, never two.
                return await client.place_option_order(order_args, ref_id=ref_id)

            check, placed = await consume_then(guard_order, lease, ledger_path=ledger_path,
                                               now=now, place=_place)
        finally:
            # `aclose`, NOT `close` — OdteMcpClient has no close() and the AttributeError was being
            # swallowed by this very handler, leaking the MCP session on every call. Same guarded
            # form the fast-lane daemon uses (odte_fast_lane.py:879).
            if own_client and hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception:                                       # teardown is best-effort
                    pass

        report["guard"] = check
        if not check.get("allowed"):
            report["reasons"] = list(check.get("reasons") or ["guard_refused"])
            _journal(journal_path, "no_trade_decision", {
                "underlying": lease.get("symbol"), "option_id": lease.get("option_id"),
                "lease_id": lease.get("lease_id"), "reason_codes": report["reasons"],
                "stage": "convert_place"}, now)
            return report

        row = _dict(placed)
        row = _dict(row.get("data")) or row
        report.update({"placed": True, "order_id": row.get("id") or row.get("order_id"),
                       "ref_id": ref_id,
                       "review_alerts": (review or {}).get("alerts")
                       if isinstance(review, dict) else None})
        _journal(journal_path, "order_placed", {
            "underlying": lease.get("symbol"), "option_id": lease.get("option_id"),
            "lease_id": lease.get("lease_id"), "order_id": report["order_id"],
            "limit_price": limit, "quantity": lease.get("quantity"),
            "lease_seconds_remaining": remaining, "source_lane": "odte_convert_place"}, now)
        return report
    except Exception as exc:                                   # fail CLOSED, never into the tick
        logger.error("place_converted failed — lease left to expire: %s", exc)
        report["reasons"] = [f"place_error:{type(exc).__name__}"]
        return report


def _journal(journal_path: str | None, event_type: str, fields: dict, now: datetime) -> None:
    """Best-effort telemetry. A journal failure must never turn into an order failure."""
    if not journal_path:
        return
    try:
        append_decision_journal({**fields, "event_type": event_type},
                                source="odte_convert_place", event_type=event_type,
                                journal_path=journal_path, now=now)
    except Exception as exc:
        logger.warning("convert-place journal append failed: %s", exc)
