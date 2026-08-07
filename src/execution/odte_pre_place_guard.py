"""Fast-lane pre-place guard — THE safety gate between a decision and a transmitted order.

The daemon never passes through the Hermes pre-order hook, so this module IS its hook — and it
is strictly stronger: `_identity_mismatches` (order-guard) checks symbol/type/strike/expiration
beyond the hook's option_id/qty/price/debit, plus lease expiry, the single-use consumed ledger,
and the NVDA employer restriction. The ledger (`data/odte/consumed_leases.json`) is SHARED with
the still-installed Hermes hook — one arbiter across both lanes, so a lease burned here can
never be replayed there.

The ordering invariant that makes a crash safe: `record_consumed` runs BEFORE the place call is
transmitted. A crash between the two burns the lease with no order at the broker (annoying, a
re-convert away from recovery); the reverse ordering could place an order whose lease is then
replayable — never acceptable. `consume_then()` makes that ordering structural rather than
conventional: callers cannot place through this module without consuming first.

Closing orders ALWAYS pass the lease gates (exits are never blocked — the incident-remediation
invariant) but still get the NVDA scan.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

from data.odte_execution_policy import lease_expired, load_consumed_ids, record_consumed
from data.odte_order_guard import _identity_mismatches

logger = logging.getLogger(__name__)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _lease_dict(lease: Any) -> dict:
    lease = _dict(lease)
    return _dict(lease.get("lease")) or lease


def order_for_guard(order_args: dict, *, symbol: str | None = None,
                    option_type: str | None = None, strike_price: float | None = None,
                    expiration_date: str | None = None) -> dict:
    """Normalize MCP order args (account_number/legs/price/quantity) into the guard-shaped order
    dict `_identity_mismatches` compares against the lease. Contract identity fields come from
    the CALLER's intent/contract — that is the point: a drift between what the daemon believes
    it is trading and what the lease authorized must surface as a mismatch."""
    legs = [leg for leg in (order_args.get("legs") or []) if isinstance(leg, dict)]
    leg = legs[0] if legs else {}
    out: dict[str, Any] = {
        "option_id": leg.get("option_id"),
        "position_effect": leg.get("position_effect"),
        "quantity": order_args.get("quantity"),
        "limit_price": order_args.get("price"),
    }
    if symbol:
        out["symbol"] = symbol
    if option_type:
        out["option_type"] = option_type
    if strike_price is not None:
        out["strike_price"] = strike_price
    if expiration_date:
        out["expiration_date"] = expiration_date
    return out


def _is_closing(order: dict) -> bool:
    """True only when EVERY leg (or the flat order) declares position_effect close."""
    legs = [leg for leg in (order.get("legs") or []) if isinstance(leg, dict)]
    if legs:
        return all(str(leg.get("position_effect") or "").lower() == "close" for leg in legs)
    return str(order.get("position_effect") or "").lower() == "close"


def _restricted_hits(order: dict, lease: dict) -> list[str]:
    from data.social_sentiment import is_restricted_underlying
    hits = []
    seen = set()
    for source in (order, *[leg for leg in (order.get("legs") or []) if isinstance(leg, dict)],
                   lease):
        for key in ("symbol", "underlying", "chain_symbol"):
            value = str(_dict(source).get(key) or "").upper()
            if value and value not in seen:
                seen.add(value)
                if is_restricted_underlying(value):
                    hits.append(value)
    return hits


def pre_place_check(order: dict | None, lease: dict | None, *, ledger_path: str,
                    now: datetime) -> dict:
    """Fail-closed gate chain. Returns {allowed, closing, lease_id, reasons}. Any internal error
    blocks — an order the guard cannot reason about is never transmitted."""
    try:
        order = _dict(order)
        ld = _lease_dict(lease)
        lease_id = str(ld.get("lease_id") or "") or None
        closing = _is_closing(order)
        reasons: list[str] = []

        restricted = _restricted_hits(order, ld)
        if restricted:
            reasons.append(f"restricted_underlying:{','.join(restricted)}")

        if not closing and not reasons:
            if not lease_id:
                reasons.append("no_execution_lease")
            else:
                mismatches = _identity_mismatches(order, ld)
                if mismatches:
                    reasons.append("lease_identity_mismatch:" + ",".join(mismatches))
                if lease_expired(ld, now):
                    reasons.append("lease_expired")
                if lease_id in load_consumed_ids(ledger_path):
                    reasons.append("lease_already_consumed")

        return {"allowed": not reasons, "closing": closing, "lease_id": lease_id,
                "reasons": reasons}
    except Exception as exc:                                   # fail CLOSED, never open
        logger.error("pre_place_check internal error — blocking: %s", exc)
        return {"allowed": False, "closing": False, "lease_id": None,
                "reasons": [f"guard_error:{type(exc).__name__}"]}


async def consume_then(order: dict | None, lease: dict | None, *, ledger_path: str,
                       now: datetime,
                       place: Callable[[], Awaitable[Any]]) -> tuple[dict, Any]:
    """Structural consume-before-place. Runs the guard; on an allowed OPENING order the lease is
    recorded consumed BEFORE `place()` is awaited — a crash in between burns the lease, never
    the reverse. Returns (check, place_result-or-None)."""
    check = pre_place_check(order, lease, ledger_path=ledger_path, now=now)
    if not check["allowed"]:
        return check, None
    if not check["closing"] and check["lease_id"]:
        # ATOMIC TEST-AND-SET, not a bare record (2026-08-07). `pre_place_check` READ the ledger to
        # verify the lease was unconsumed; recording it is a separate WRITE. Between the two, a
        # second lane — the fast-lane daemon shares this exact ledger path — could pass its own
        # check and place against the SAME single-use lease. `record_consumed` now claims the id
        # under a lock and returns False if someone else already had it; losing that race is a
        # refusal, never a place.
        if not record_consumed(ledger_path, check["lease_id"]):
            lost = dict(check)
            lost["allowed"] = False
            lost["reasons"] = [*check.get("reasons", []), "lease_already_consumed"]
            return lost, None
    result = await place()
    return check, result
