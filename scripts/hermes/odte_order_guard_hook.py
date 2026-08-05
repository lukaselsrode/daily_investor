#!/usr/bin/env python3
"""Deterministic pre_tool_call guard for mcp__robinhood__place_option_order — v6 CANONICAL COPY.

This file is the repo-owned source of truth for the Hermes hook; deploying = copying it to
~/.hermes/hooks/odte_order_guard_hook.py (see docs/hermes/v2/hook_patch_v6.md). Fires on EVERY
Hermes session (interactive, cron, delegated) before an option order reaches the broker.
Fail-closed: any parse error, missing lease, or identity mismatch blocks OPENING orders.
Closing/sell orders are always allowed — blocking an exit is more dangerous than allowing it.

v6 (two-lane architecture, 2026-08-05): reads data/odte/fast_lane_stage.json. At stage
`entries_live` the fast-lane daemon is the ONLY lane that may OPEN positions — this hook blocks
EVERY Hermes opening order regardless of lease state (the daemon never passes through this hook;
it runs its own strictly-stronger in-process guard against the SAME consumed-leases ledger).
Closes/cancels still always pass: Hermes keeps its emergency exit valve at every stage.

Contract (Hermes shell hooks): JSON on stdin with tool_name/tool_input.
Print {"decision": "block", "reason": "..."} to block; print nothing to allow.
Born from the 2026-07-23 stale-fill incident (-$52.16): no fresh exact-contract
execution lease => no new position.
"""
import json
import os
import sys
from datetime import datetime, timezone

LEASE_PATH = os.environ.get(
    "ODTE_LEASE_PATH",
    "/Users/lukaselsrode/dev_work/daily_investor/data/odte/execution_lease.json",
)
STAGE_PATH = os.environ.get(
    "ODTE_STAGE_PATH",
    os.path.join(os.path.dirname(LEASE_PATH), "fast_lane_stage.json"),
)
# 2026-08-03 v5 port: the flat $120 A+ cap is DELETED — sizing is BP-proportional and tiered
# inside the lease (max_debit = tier fraction x live BP, chase band anchor x 1.15). The hook
# enforces the LEASE ceilings below; a second flat rail double-blocked compliant orders.
EPS = 0.005


def block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": "ODTE order guard: " + reason}))
    sys.exit(0)


def allow() -> None:
    sys.exit(0)


def parse_ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def first_key(mapping, *names):
    for name in names:
        if isinstance(mapping, dict) and mapping.get(name) is not None:
            return mapping[name]
    return None


def collect_legs(tool_input):
    legs = tool_input.get("legs")
    if isinstance(legs, list) and legs:
        return [leg for leg in legs if isinstance(leg, dict)]
    return [tool_input]


def is_closing_only(tool_input):
    """True only when every leg is unambiguously a close/sell."""
    saw_signal = False
    for leg in collect_legs(tool_input):
        effect = str(first_key(leg, "position_effect", "positionEffect", "effect") or "").lower()
        side = str(first_key(leg, "side", "action", "direction") or "").lower()
        if effect == "open" or side.startswith("buy"):
            return False
        if effect == "close" or side.startswith("sell"):
            saw_signal = True
        else:
            return False  # ambiguous leg -> treat as open, enforce lease
    return saw_signal


def fast_lane_stage():
    try:
        with open(STAGE_PATH, encoding="utf-8") as fh:
            return str(json.load(fh).get("stage") or "").lower()
    except Exception:  # noqa: BLE001 - missing/unreadable stage file = no stage restriction
        return ""


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - fail closed on any parse problem
        block(f"unparseable hook payload ({exc})")
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        block("tool_input is not an object")

    blob = json.dumps(tool_input).upper()
    if "NVDA" in blob:
        block("NVDA is employer-restricted; never tradeable in any form")

    if is_closing_only(tool_input):
        allow()  # exits are risk-reducing; never trap a position

    # --- v6: at entries_live the fast-lane daemon owns ALL opens -------------------------------
    if fast_lane_stage() == "entries_live":
        block(
            "fast lane owns opening orders at stage entries_live — Hermes may only "
            "close/cancel. Arm intents via `make odte-arm`; a lease minted here would expire "
            "unconsumed in <=60s anyway"
        )

    # --- opening order: a fresh exact-contract execution lease is mandatory ---
    if not os.path.exists(LEASE_PATH):
        block(
            f"no execution lease at {LEASE_PATH}. Run `make odte-convert` "
            "with fresh snapshots; bare promotion is dead"
        )
    try:
        with open(LEASE_PATH, encoding="utf-8") as fh:
            lease = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        block(f"lease unreadable ({exc})")
    if isinstance(lease, dict) and "authorized" in lease and lease.get("authorized") is not True:
        reasons = lease.get("reason_codes") or []
        block(f"authorization REFUSED ({', '.join(map(str, reasons)) or 'no lease issued'})")
    if isinstance(lease, dict) and isinstance(lease.get("lease"), dict):
        lease = lease["lease"]  # authorize payload wraps the lease object
    if not isinstance(lease, dict):
        block("lease artifact is not an object")

    status = str(lease.get("status") or "").lower()
    if lease.get("consumed") is True or status in {"consumed", "expired", "revoked"}:
        block(f"lease is not live (status={status or 'consumed'}); a lease is single-use")

    ledger_path = os.path.join(os.path.dirname(LEASE_PATH), "consumed_leases.json")
    lease_id = str(lease.get("lease_id") or "")
    if lease_id and os.path.exists(ledger_path):
        try:
            with open(ledger_path, encoding="utf-8") as fh:
                consumed_ids = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            block(f"consumed-lease ledger unreadable ({exc})")
        if lease_id in (consumed_ids if isinstance(consumed_ids, list) else []):
            block(f"lease {lease_id} already consumed; a submitted order burns its lease")

    expires = parse_ts(first_key(lease, "expires_at", "expires", "expiry"))
    if expires is None:
        block("lease has no parseable expires_at")
    now = datetime.now(timezone.utc)
    if now >= expires:
        block(
            f"lease expired at {expires.isoformat()} (now {now.isoformat()}). "
            "Stale authorization cannot fill; re-run the full gate cycle"
        )

    lease_option = str(first_key(lease, "option_id", "option", "instrument_id") or "")
    order_options = set()
    for leg in collect_legs(tool_input):
        opt = first_key(leg, "option_id", "option", "instrument_id", "option_instrument_id")
        if opt:
            order_options.add(str(opt))
    if not lease_option:
        block("lease carries no option_id; exact-contract binding is mandatory")
    if not order_options:
        block("order carries no recognizable option_id; refusing ambiguous entry")
    if order_options != {lease_option}:
        block(
            f"contract mismatch: lease authorizes {lease_option}, order targets "
            f"{sorted(order_options)}. Vehicle/contract switch requires a new lease"
        )

    try:
        qty = float(first_key(tool_input, "quantity", "qty") or 0)
        price = float(first_key(tool_input, "price", "limit_price", "limitPrice") or 0)
    except (TypeError, ValueError):
        block("order quantity/price not numeric")
    if qty <= 0 or price <= 0:
        block("order missing positive quantity/limit price")

    max_qty = float(lease.get("quantity") or 0)
    max_limit = float(first_key(lease, "max_limit_price", "max_limit") or 0)
    max_debit = float(lease.get("max_debit") or 0)
    if qty > max_qty + EPS:
        block(f"quantity {qty:g} exceeds leased {max_qty:g}")
    if price > max_limit + EPS:
        block(f"limit {price} exceeds leased max {max_limit}")
    debit = qty * price * 100.0
    if debit > max_debit + EPS:
        block(f"debit ${debit:.2f} exceeds leased max ${max_debit:.2f}")
    allow()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - absolute fail-closed backstop
        print(json.dumps({"decision": "block", "reason": f"ODTE order guard crashed: {exc}"}))
        sys.exit(0)
