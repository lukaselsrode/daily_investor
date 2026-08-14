"""0DTE execution-authorization lease — PURE/OFFLINE, NO broker/network/LLM, places NO orders.

This is the ONE tier that mints execution authority, and it mints it as a SHORT-LIVED LEASE bound
to one exact contract — never as a reusable flag. Born from the 2026-07-23 delayed-fill incident:
a `scan_only` watchdog signal was promoted with a bare `--promote-to-execution` boolean, the
2-contract SPY 737P limit sat pending for 2.5 minutes, filled after the momentum extension had
stalled, and closed for a loss. The structural fixes enforced here:

  * Scan/watchdog output can NEVER be promoted by a bare boolean — `odte_entry_gate` now answers
    the deprecated flag with `execution_lease_required`, and `authorize_entry` refuses any gate
    whose `scan_only` is not explicitly False.
  * Authorization is exact-contract, exact-direction, exact-quantity, exact-price, and SHORT-LIVED:
    config default TTL, configurable DOWNWARD only, hard-capped at 60s for 0DTE.
  * PARTIAL_ACCOUNT entries carry a CHASE BAND (2026-08-02): the fill ceiling is the CONFIRM_ENTRY
    anchor quote × (1 + band); sizing is tiered and BP-proportional (full vs B+ half size).
  * The lease carries MAXIMUMS (quantity / limit / debit); downstream can only stay at or below.
  * A lease is SINGLE-USE: `consume_lease` burns it at submission; reuse fails closed.
  * Full-account size is permitted only under an explicit `FULL_ACCOUNT_A_PLUS` lease with the max
    premium loss, exact trigger, named invalidation, live bid/ask, target, scratch rail, and active
    management cadence all recorded. A model-generated "A+" label alone NEVER sets size.

Everything here is decision/record tooling: it reads caller-supplied artifacts, returns a payload,
and never calls a broker. The Hermes/MCP lane owns order review/place; `odte_order_guard` owns the
pending-order cancel-first state machine that keeps a stale lease from turning into a stale fill.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:                                   # POSIX advisory locking; absent on some platforms
    import fcntl
except ImportError:                     # pragma: no cover - non-POSIX
    fcntl = None                        # type: ignore[assignment]

from core.paths import ODTE_DATA_DIR, atomic_write_text
from data.odte_config import (
    B_PLUS_DEBIT_FRACTION,
    B_PLUS_MIN_VEHICLE_SCORE,
    CHASE_BAND_FRACTION,
    LEASE_TTL_SECONDS,
    MAX_DEBIT_FRACTION,
    SNAPSHOT_TTL_SECONDS,
)

SCHEMA_VERSION = 1

DEFAULT_STATE_DIR = ODTE_DATA_DIR
LEASE_FILENAME = "execution_lease.json"
CONSUMED_LEASES_FILENAME = "consumed_leases.json"

# Lease TTL: config-tunable default (60s per the 2026-08-02 retune — covers one MCP place
# round-trip plus a timeout retry); callers may shorten it but can NEVER extend past the 60s hard
# cap for 0DTE. The cap is an incident invariant and is deliberately NOT config-tunable.
# Missing a fill is intentional and acceptable; acquiring a stale 0DTE position is not.
MAX_LEASE_TTL_SECONDS = 60.0
DEFAULT_LEASE_TTL_SECONDS = min(LEASE_TTL_SECONDS, MAX_LEASE_TTL_SECONDS)

# Account-risk modes. PARTIAL_ACCOUNT is the default: one contract, debit capped to a fraction of
# buying power. FULL_ACCOUNT_A_PLUS permits a full-account debit ONLY when the lease explicitly
# records the complete management plan (see FULL_ACCOUNT_PLAN_FIELDS) — size is never inferred
# from an unverified "A+" label.
RISK_MODES = ("PARTIAL_ACCOUNT", "FULL_ACCOUNT_A_PLUS")
DEFAULT_RISK_MODE = "PARTIAL_ACCOUNT"
FULL_ACCOUNT_REQUIRES_FRESH_LEASE = True
FULL_ACCOUNT_REQUIRES_NAMED_INVALIDATION = True
FULL_ACCOUNT_REQUIRES_ACTIVE_MANAGEMENT = True

# Default (PARTIAL_ACCOUNT) sizing policy — the incident's 2-contract / 83.9%-of-BP order fails
# BOTH caps. Per-call policy overrides can only tighten these, never widen them; bigger size must
# go through the explicit FULL_ACCOUNT_A_PLUS path. The debit fraction is BP-proportional and
# tiered (2026-08-02 retune, replacing the flat agent-side $120 rail): full tier 60% of BP, B+
# (CHOP half-size) tier 30%. The 1-contract limit stays hard-coded.
DEFAULT_MAX_CONTRACTS = 1
DEFAULT_MAX_DEBIT_FRACTION = MAX_DEBIT_FRACTION

# Every input artifact must carry a parseable timestamp within its own TTL (seconds). A missing or
# stale timestamp fails CLOSED — an undated artifact is never fresh. Market/broker snapshots get a
# wider bound (2026-08-02 retune: the 60s bound refused the one fully-qualified entry of the
# 2026-07-27..31 zero-trade week 11s after its gate passed; price risk inside the window is bounded
# by the chase band + lease ceilings, which are the honest rails). The three in-process artifacts
# stay at 60s — odte-convert mints them microseconds before authorization.
INPUT_TTLS_SECONDS: dict[str, float] = {
    "entry_gate": 60.0,
    "candidate_decision": 60.0,
    "vehicle_score": 60.0,
    "market_snapshot": SNAPSHOT_TTL_SECONDS,
    "broker_snapshot": SNAPSHOT_TTL_SECONDS,
}
MAX_FUTURE_SKEW_SECONDS = 2.0

# The complete management plan a FULL_ACCOUNT_A_PLUS lease must record (policy.management_plan).
FULL_ACCOUNT_PLAN_FIELDS = ("trigger", "invalidation", "target", "scratch_rail",
                            "management_cadence_seconds", "max_premium_loss")

# Final live confirmations required as EXPLICIT booleans on the gate (strings/missing fail closed).
# Mirrors odte_entry_gate.REQUIRED_CONFIRMATIONS (kept literal here so this module stays leaf-pure).
# premium_floor_check joined 2026-08-14 (W33 fence) — the lease layer re-requires the SAME set the
# entry gate does; a drift between the two lists is how a confirmation stops being enforced.
DEFAULT_REQUIRED_CONFIRMATIONS = ("live_chain_recheck", "spread_cap_check", "budget_check",
                                  "premium_floor_check")

_BULLISH = {"call", "bullish", "long_call", "calls", "up"}
_BEARISH = {"put", "bearish", "long_put", "puts", "down"}
_DIRECTION_OPTION_TYPE = {"bullish": "call", "bearish": "put"}


@dataclass(frozen=True)
class ExecutionLease:
    """Single-use, short-lived, exact-identity execution authorization. Frozen by design —
    downstream can consume it or let it expire, never edit it."""

    lease_id: str
    issued_at: datetime
    expires_at: datetime
    symbol: str
    direction: str
    option_id: str
    expiration_date: str
    strike_price: float
    option_type: str
    quantity: int
    max_limit_price: float
    max_debit: float
    candidate_fingerprint: str
    market_fingerprint: str
    risk_mode: str = DEFAULT_RISK_MODE
    max_premium_loss: float | None = None
    # 2026-08-02 retune: the tape-computed confirmation tier (sizing) and the CONFIRM_ENTRY anchor
    # the max_limit_price chase band was derived from — journaled for auditability.
    tier: str | None = None
    anchor_quote: float | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["issued_at"] = self.issued_at.isoformat(timespec="seconds")
        out["expires_at"] = self.expires_at.isoformat(timespec="seconds")
        return out


# --- small helpers -----------------------------------------------------------------------------

def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None   # NaN guard


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


def _payload_ts(payload: dict) -> datetime | None:
    for key in ("generated_at", "as_of", "ts", "updated_at", "created_at", "selection_timestamp"):
        dt = _parse_ts(payload.get(key))
        if dt is not None:
            return dt
    return None


def _norm_direction(value: Any) -> str | None:
    s = str(value or "").strip().lower()
    if s in _BULLISH:
        return "bullish"
    if s in _BEARISH:
        return "bearish"
    return None


def _norm_symbol(value: Any) -> str | None:
    return str(value).strip().upper() if value else None


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


# --- fingerprints -------------------------------------------------------------------------------

def candidate_fingerprint(candidate: dict | None) -> str:
    """Stable 16-hex identity for one exact contract candidate cycle.

    The hash binds underlying, direction, selected vehicle, exact option id/expiry/strike/type, and
    the selection timestamp. Callers must separately reject missing identity fields; hashing never
    turns an absent field into authority.
    """
    c = _dict(candidate)
    basis = json.dumps({
        "ticker": _norm_symbol(c.get("ticker") or c.get("underlying") or c.get("symbol")),
        "direction": _norm_direction(c.get("direction") or c.get("option_type")),
        "selected_vehicle": _norm_symbol(c.get("selected_vehicle"))
                            or _norm_symbol(c.get("ticker") or c.get("underlying") or c.get("symbol")),
        "option_id": c.get("option_id") or c.get("id") or c.get("occ_symbol"),
        "expiration_date": c.get("expiration_date") or c.get("expiration"),
        "strike_price": _num(c.get("strike_price") or c.get("strike")),
        "option_type": str(c.get("option_type") or c.get("type") or "").strip().lower() or None,
        "selected_at": str(c.get("selection_timestamp") or c.get("created_at") or c.get("ts")
                           or c.get("generated_at") or ""),
    }, sort_keys=True)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def market_fingerprint(market_snapshot: dict | None) -> str:
    """16-hex content hash of the market snapshot the lease was authorized against."""
    basis = json.dumps(_dict(market_snapshot), sort_keys=True, default=str)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


# --- lease lifecycle ----------------------------------------------------------------------------

def _lease_dict(lease: ExecutionLease | dict | None) -> dict:
    if isinstance(lease, ExecutionLease):
        return lease.to_dict()
    lease = _dict(lease)
    # Accept either the bare lease or a full authorize payload with the lease nested.
    return _dict(lease.get("lease")) or lease


def lease_expired(lease: ExecutionLease | dict | None, now: datetime) -> bool:
    """True when the lease is missing, undated, or `now` is at/past `expires_at` (boundary
    inclusive — an exactly-expired lease is expired). Fails CLOSED on unparseable input."""
    exp = _parse_ts(_lease_dict(lease).get("expires_at"))
    if exp is None:
        return True
    return now >= exp


def lease_seconds_remaining(lease: ExecutionLease | dict | None, now: datetime) -> float | None:
    exp = _parse_ts(_lease_dict(lease).get("expires_at"))
    if exp is None:
        return None
    return round((exp - now).total_seconds(), 3)


def lease_matches_candidate(lease: ExecutionLease | dict | None, candidate: dict | None) -> bool:
    """Exact-identity binding: the lease is valid only for the candidate cycle it was minted for.
    A vehicle switch / fresh cycle changes the candidate fingerprint and invalidates the lease."""
    ld = _lease_dict(lease)
    fp = ld.get("candidate_fingerprint")
    return bool(fp) and fp == candidate_fingerprint(candidate)


def consume_lease(lease: ExecutionLease | dict | None, *, consumed_ids: set[str],
                  now: datetime) -> dict:
    """SINGLE-USE consumption: burn the lease at order submission. Returns a status dict —
    'consumed' exactly once; 'already_consumed' / 'expired' / 'invalid' fail closed. Mutates
    `consumed_ids` (the caller owns persistence — see load/record helpers)."""
    ld = _lease_dict(lease)
    lease_id = ld.get("lease_id")
    if not lease_id:
        return {"status": "invalid", "lease_id": None,
                "detail": "no lease_id — not a valid execution lease"}
    if lease_id in consumed_ids:
        return {"status": "already_consumed", "lease_id": lease_id,
                "detail": "lease already consumed — a new order requires a NEW lease"}
    if lease_expired(ld, now):
        return {"status": "expired", "lease_id": lease_id,
                "detail": "lease expired — re-run the full candidate/gate/authorize cycle"}
    consumed_ids.add(lease_id)
    return {"status": "consumed", "lease_id": lease_id,
            "consumed_at": now.isoformat(timespec="seconds")}


def load_consumed_ids(path: str | Path) -> set[str]:
    """Read the consumed-lease ledger (a JSON list of lease ids). Missing/malformed reads empty."""
    p = Path(os.path.expanduser(str(path)))
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return {str(x) for x in data} if isinstance(data, list) else set()
    except Exception:
        return set()


def seed_consumed_ledger(path: str | Path) -> None:
    """Ensure the single-use ledger EXISTS (empty list) so no reader can skip the replay check.

    2026-08-05 audit: the ledger only materialized on the first guard/consume call, and the
    Hermes pre-order hook skips its already-consumed check when the file is absent — Aug-3's
    lease was replayable for its whole TTL. Minting a lease now seeds the ledger first."""
    p = Path(os.path.expanduser(str(path)))
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, "[]")


def record_consumed(path: str | Path, lease_id: str) -> bool:
    """Atomically claim `lease_id` in the single-use ledger. True if WE claimed it, False if it
    was already there.

    The return value is the point (2026-08-07). This was a bare read-modify-write —
    load -> membership check -> add -> atomic_write_text — with no lock, so two consumers could
    both pass their check before either recorded, and both place against ONE single-use lease.
    Callers must treat False as "someone else consumed this lease" and refuse; see `consume_then`.

    The lock is a SIDECAR file, not the ledger itself: `atomic_write_text` finishes with
    `os.replace`, so a lock held on the ledger inode would be released to a stale inode and a
    second process could lock the replacement independently. The sidecar is never replaced.
    Best-effort, mirroring `odte_journal._append_jsonl_locked`: if flock is unavailable the
    behaviour degrades to the previous read-modify-write rather than failing the place.
    """
    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_name(f".{p.name}.lock")
    try:
        lock = open(lock_path, "a+")
    except OSError:
        lock = None
    try:
        if lock is not None and fcntl is not None:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass                      # best-effort; single-writer deployments are unaffected
        ids = load_consumed_ids(p)
        if lease_id in ids:
            return False
        ids.add(lease_id)
        atomic_write_text(p, json.dumps(sorted(ids), indent=2))
        return True
    finally:
        if lock is not None:
            lock.close()


# --- authorization core ---------------------------------------------------------------------

def _resolve_ttl(policy: dict) -> float:
    """TTL from policy, clamped: configurable DOWNWARD, never above MAX_LEASE_TTL_SECONDS."""
    ttl = _num(policy.get("ttl_seconds"))
    if ttl is None or ttl <= 0:
        ttl = DEFAULT_LEASE_TTL_SECONDS
    return min(ttl, MAX_LEASE_TTL_SECONDS)


def _check_freshness(reasons: list[str], name: str, payload: dict, now: datetime) -> None:
    ts = _payload_ts(payload)
    if ts is None:
        reasons.append(f"{name}_undated")
        return
    age = (now - ts).total_seconds()
    if age < -MAX_FUTURE_SKEW_SECONDS:
        reasons.append(f"{name}_from_future")
    elif age > INPUT_TTLS_SECONDS[name]:
        reasons.append(f"{name}_stale")


def _contract_identity(contract: dict) -> dict:
    return {
        "underlying": _norm_symbol(contract.get("underlying") or contract.get("symbol")
                                   or contract.get("ticker") or contract.get("chain_symbol")),
        "option_type": str(contract.get("option_type") or contract.get("type") or "").strip().lower() or None,
        "option_id": contract.get("option_id") or contract.get("id") or contract.get("occ_symbol"),
        "expiration_date": contract.get("expiration_date") or contract.get("expiration"),
        "strike_price": _num(contract.get("strike_price") or contract.get("strike")),
        "bid": _num(contract.get("bid") or contract.get("bid_price")),
        "ask": _num(contract.get("ask") or contract.get("ask_price")),
        "mark": _num(contract.get("mark") or contract.get("mark_price")),
    }


def _broker_count(broker: dict, *keys: str) -> float | None:
    acct = broker.get("account") if isinstance(broker.get("account"), dict) else {}
    for container in (broker, acct):
        for k in keys:
            v = _num(container.get(k))
            if v is not None:
                return v
    return None


def _check_broker(reasons: list[str], broker: dict) -> float | None:
    """Broker-truth prerequisites for ANY lease: fresh BP, position/order/today counts present,
    account not blocked, controller not locked. Missing fields fail CLOSED. Returns buying power."""
    bp = _num(broker.get("buying_power") or broker.get("options_buying_power")
              or broker.get("account_buying_power"))
    if bp is None:
        reasons.append("buying_power_missing")
    elif bp <= 0:
        reasons.append("insufficient_buying_power")
    positions = _broker_count(broker, "nonzero_option_positions_count",
                              "open_option_positions_count", "positions_count",
                              "open_positions_count")
    if positions is None:
        reasons.append("broker_positions_count_missing")
    elif positions > 0:
        reasons.append("position_already_open")
    open_orders = _broker_count(broker, "open_option_orders_count", "open_orders_count")
    if open_orders is None:
        reasons.append("broker_open_orders_count_missing")
    elif open_orders > 0:
        reasons.append("open_order_outstanding")
    if _broker_count(broker, "today_option_orders_count", "today_orders_count") is None:
        reasons.append("broker_today_orders_count_missing")
    if broker.get("blocked") is True or broker.get("trading_blocked") is True:
        reasons.append("account_blocked")
    if broker.get("controller_locked") is True:
        reasons.append("controller_locked")
    dt_left = _num(broker.get("day_trades_left"))
    if dt_left is not None and dt_left <= 0:
        reasons.append("no_day_trades_left")
    return bp


def authorize_entry(*, gate: dict, candidate_decision: dict, vehicle_score: dict,
                    broker_snapshot: dict, market_snapshot: dict,
                    now: datetime, policy: dict) -> dict:
    """PURE: authorize one entry as a short-lived exact-identity ExecutionLease, or fail closed.

    Every deterministic requirement must hold SIMULTANEOUSLY at `now`:
      * every input artifact timestamped within its own TTL (INPUT_TTLS_SECONDS);
      * a fresh `candidate_decision.decision == "CONFIRM_ENTRY"`;
      * an entry gate with EXPLICIT `scan_only=False` and `execution_allowed=True` (the watchdog
        artifact itself remains non-authoritative — bare promotion mints nothing);
      * every required final confirmation an EXPLICIT boolean True on `gate.confirmations`;
      * exact identity agreement: candidate ticker == contract underlying == gate symbol, one
        direction everywhere, option type matching the direction, option id / expiration / strike
        present (and equal wherever both sides state them);
      * broker truth: BP + position/open-order/today-order counts present, flat, not blocked/locked;
      * sizing within the risk mode: PARTIAL_ACCOUNT (default) = max 1 contract and debit ≤ 50% of
        BP (policy can only tighten); FULL_ACCOUNT_A_PLUS = explicit risk mode + complete
        management plan (trigger/invalidation/target/scratch rail/cadence/max premium loss) + live
        bid/ask, with the debit within both the accepted max loss and BP.

    The returned lease carries MAXIMUM quantity/limit/debit — downstream can never increase any
    value — plus the candidate/market fingerprints it is bound to. Never raises; never places or
    reviews an order."""
    gate = _dict(gate)
    cd = _dict(candidate_decision)
    vs = _dict(vehicle_score)
    broker = _dict(broker_snapshot)
    market = _dict(market_snapshot)
    pol = _dict(policy)
    reasons: list[str] = []
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    risk_mode = str(pol.get("risk_mode") or DEFAULT_RISK_MODE)
    if risk_mode not in RISK_MODES:
        reasons.append("risk_mode_invalid")
        risk_mode = DEFAULT_RISK_MODE
    ttl_seconds = _resolve_ttl(pol)

    # 1) freshness: each artifact within its own TTL; undated fails closed.
    for name, payload in (("entry_gate", gate), ("candidate_decision", cd),
                          ("vehicle_score", vs), ("market_snapshot", market),
                          ("broker_snapshot", broker)):
        _check_freshness(reasons, name, payload, now)

    # 2) fresh candidate confirmation.
    if str(cd.get("decision") or "").upper() != "CONFIRM_ENTRY":
        reasons.append("candidate_not_confirmed")
    candidate = _dict(cd.get("candidate"))
    selected_at = (candidate.get("selection_timestamp") or candidate.get("created_at")
                   or candidate.get("ts") or candidate.get("generated_at"))
    if not selected_at:
        reasons.append("candidate_cycle_timestamp_missing")
    candidate_contract = _contract_identity(candidate)
    # Tier + anchor (2026-08-02 retune). TAPE-COMPUTED by candidate-watch and carried on the
    # candidate — never read from a policy label, preserving "a label never sets size".
    tier = str(candidate.get("tier") or "").lower() or None
    gate_tier = str(gate.get("tier") or "").lower() or None
    anchor = _num(candidate.get("anchor_quote"))

    # 3) gate: explicit booleans only. A scan-tier record can never lease (the incident's hole).
    if gate.get("scan_only") is not False:
        reasons.append("gate_scan_only_not_false")
    if gate.get("execution_allowed") is not True:
        reasons.append("gate_not_execution_allowed")
    computed_candidate_fp = candidate_fingerprint(candidate)
    supplied_candidate_fp = candidate.get("candidate_fingerprint")
    if supplied_candidate_fp and supplied_candidate_fp != computed_candidate_fp:
        reasons.append("candidate_fingerprint_invalid")
    gate_candidate_fp = gate.get("candidate_fingerprint")
    if not gate_candidate_fp:
        reasons.append("gate_candidate_fingerprint_missing")
    elif gate_candidate_fp != computed_candidate_fp:
        reasons.append("gate_candidate_fingerprint_mismatch")
    confirmations = _dict(gate.get("confirmations"))
    raw_declared_confirmations = gate.get("required_confirmations")
    declared_confirmations = (tuple(raw_declared_confirmations)
                              if isinstance(raw_declared_confirmations, (list, tuple)) else ())
    if not set(DEFAULT_REQUIRED_CONFIRMATIONS).issubset(set(declared_confirmations)):
        reasons.append("confirmation_schema_incomplete")
    for conf in DEFAULT_REQUIRED_CONFIRMATIONS:
        v = confirmations.get(conf)
        if not isinstance(v, bool):
            reasons.append(f"confirmation_not_boolean:{conf}")
        elif v is not True:
            reasons.append(f"confirmation_failed:{conf}")

    # 4) exact identity agreement.
    contract = _contract_identity(_dict(vs.get("contract")))
    gate_contract = _contract_identity(gate)
    cand_sym = candidate_contract["underlying"]
    gate_sym = _norm_symbol(gate.get("symbol") or gate.get("underlying"))
    syms = {s for s in (cand_sym, gate_sym, contract["underlying"]) if s}
    if not cand_sym or not gate_sym or not contract["underlying"]:
        reasons.append("symbol_missing")
    elif len(syms) != 1:
        reasons.append("symbol_mismatch")
    symbol = cand_sym or contract["underlying"] or gate_sym

    directions = {d for d in (_norm_direction(candidate.get("direction")),
                              _norm_direction(gate.get("direction")),
                              _norm_direction(vs.get("direction"))) if d}
    if not directions:
        reasons.append("direction_missing")
    elif len(directions) != 1:
        reasons.append("direction_mismatch")
    direction = next(iter(directions)) if len(directions) == 1 else None

    if contract["option_type"] not in ("call", "put"):
        reasons.append("option_type_missing")
    elif direction and _DIRECTION_OPTION_TYPE[direction] != contract["option_type"]:
        reasons.append("option_type_direction_mismatch")
    if candidate_contract["option_type"] not in ("call", "put"):
        reasons.append("candidate_option_type_missing")
    elif contract["option_type"] and candidate_contract["option_type"] != contract["option_type"]:
        reasons.append("candidate_option_type_mismatch")

    if not contract["option_id"]:
        reasons.append("option_id_missing")
    if not candidate_contract["option_id"]:
        reasons.append("candidate_option_id_missing")
    elif contract["option_id"] and candidate_contract["option_id"] != contract["option_id"]:
        reasons.append("option_id_mismatch")

    if not contract["expiration_date"]:
        reasons.append("expiration_missing")
    if not candidate_contract["expiration_date"]:
        reasons.append("candidate_expiration_missing")
    elif (contract["expiration_date"]
          and str(candidate_contract["expiration_date"]) != str(contract["expiration_date"])):
        reasons.append("expiration_mismatch")

    if contract["strike_price"] is None:
        reasons.append("strike_missing")
    if candidate_contract["strike_price"] is None:
        reasons.append("candidate_strike_missing")
    elif (contract["strike_price"] is not None
          and abs(candidate_contract["strike_price"] - contract["strike_price"]) > 1e-9):
        reasons.append("strike_mismatch")

    for field in ("option_id", "expiration_date", "strike_price", "option_type"):
        gate_value = gate_contract[field]
        contract_value = contract[field]
        if gate_value is None:
            reasons.append(f"gate_{field}_missing")
        elif contract_value is not None:
            mismatch = (abs(gate_value - contract_value) > 1e-9
                        if field == "strike_price" else str(gate_value) != str(contract_value))
            if mismatch:
                reasons.append(f"gate_{field}_mismatch")

    if symbol:
        from data.social_sentiment import is_restricted_underlying
        if is_restricted_underlying(symbol):
            reasons.append("restricted_underlying")

    vehicle_verdict = str(vs.get("verdict") or "").upper()
    if vehicle_verdict != "GOOD_BET":
        # B+ tier (CHOP half-size) mirrors the entry gate: a WATCH vehicle passes when its raw
        # score clears the floor. Everything else still fails closed.
        vehicle_total = _num(vs.get("score"))
        if not (vehicle_verdict == "WATCH" and tier == "b_plus" and vehicle_total is not None
                and vehicle_total >= B_PLUS_MIN_VEHICLE_SCORE):
            reasons.append("vehicle_not_good_bet")

    # 5) broker truth prerequisites (fail closed on missing counts/BP).
    bp = _check_broker(reasons, broker)

    # 6) sizing within the risk mode. The risk mode is NEVER inferred from candidate/gate labels.
    quantity = int(_num(pol.get("quantity")) or 1)
    if quantity < 1:
        reasons.append("quantity_invalid")
        quantity = 1
    limit_price = _num(pol.get("limit_price")) or contract["ask"] or contract["mark"]
    if limit_price is None or limit_price <= 0:
        reasons.append("limit_price_missing")
    debit = round(quantity * limit_price * 100.0, 2) if limit_price else None

    max_premium_loss: float | None = None
    chase_ceiling: float | None = None
    partial_frac: float | None = None
    tier_max = DEFAULT_MAX_CONTRACTS
    if risk_mode == "FULL_ACCOUNT_A_PLUS":
        plan = _dict(pol.get("management_plan"))
        for field in FULL_ACCOUNT_PLAN_FIELDS:
            v = plan.get(field)
            if v in (None, "") or (field in ("management_cadence_seconds", "max_premium_loss")
                                   and _num(v) is None):
                reasons.append(f"full_account_missing:{field}")
        max_premium_loss = _num(plan.get("max_premium_loss"))
        if contract["bid"] is None or contract["ask"] is None:
            reasons.append("full_account_missing:bid_ask")
        if max_premium_loss is not None and debit is not None and debit > max_premium_loss:
            reasons.append("debit_exceeds_accepted_max_loss")
    else:
        if gate_tier and tier and gate_tier != tier:
            reasons.append("tier_mismatch")
        base_frac = B_PLUS_DEBIT_FRACTION if tier == "b_plus" else DEFAULT_MAX_DEBIT_FRACTION
        frac = _num(pol.get("max_debit_fraction"))
        # Policy can only TIGHTEN the debit cap, never widen it.
        partial_frac = min(frac, base_frac) if frac else base_frac
        # TIER-SCALED SIZING (2026-08-14, operator decision). At 1 contract the debit fractions
        # never bind, so a +20% winner moved the account +3.3% — tier quality changed nothing
        # about size. full/a_plus may now size up to their configured max, WITHIN the fraction
        # and buying power; b_plus (the tier that carried every recent loss) stays at
        # DEFAULT_MAX_CONTRACTS. The policy's `quantity` remains the requested base; sizing only
        # ever raises it when the tier ceiling and affordability both allow, and the lease still
        # carries the exact resulting quantity as a MAXIMUM downstream.
        from data.odte_config import TIER_MAX_CONTRACTS_APLUS, TIER_MAX_CONTRACTS_FULL
        if tier == "full":
            tier_max = max(DEFAULT_MAX_CONTRACTS, TIER_MAX_CONTRACTS_FULL)
        elif tier == "a_plus":
            tier_max = max(DEFAULT_MAX_CONTRACTS, TIER_MAX_CONTRACTS_APLUS)
        if (tier_max > quantity and limit_price and limit_price > 0
                and bp is not None and bp > 0):
            affordable = int((partial_frac * bp) // (limit_price * 100.0))
            if affordable > quantity:
                quantity = min(tier_max, affordable)
                debit = round(quantity * limit_price * 100.0, 2)
        if quantity > tier_max:
            reasons.append("quantity_exceeds_policy")
        # CHASE BAND: the fill ceiling is the CONFIRM_ENTRY anchor plus a bounded band — an entry
        # may pay up to anchor*(1+band), never more. Re-anchoring at confirmation (not precompute)
        # is what stops the rail from vetoing entries whose premium rose BECAUSE the thesis
        # confirmed. A missing anchor fails closed: every fresh confirm stamps one.
        if anchor is None or anchor <= 0:
            reasons.append("anchor_quote_missing")
        else:
            chase_ceiling = round(anchor * (1.0 + CHASE_BAND_FRACTION), 2)
            if limit_price is not None and limit_price > chase_ceiling:
                reasons.append("limit_exceeds_chase_band")
        if debit is not None and bp is not None and debit > partial_frac * bp:
            reasons.append("debit_exceeds_policy")
    if debit is not None and bp is not None and debit > bp:
        reasons.append("debit_exceeds_buying_power")

    resolved_policy = {
        "risk_mode": risk_mode,
        "ttl_seconds": ttl_seconds,
        "max_contracts": (quantity if risk_mode == "FULL_ACCOUNT_A_PLUS" else tier_max),
        "max_debit_fraction": partial_frac,
        "tier": tier,
        "chase_band_fraction": (None if risk_mode == "FULL_ACCOUNT_A_PLUS" else CHASE_BAND_FRACTION),
        "anchor_quote": anchor,
        "max_premium_loss": max_premium_loss,
    }
    base = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "risk_mode": risk_mode,
        "policy": resolved_policy,
        "symbol": symbol,
        "direction": direction,
        "quantity": quantity,
        "debit": debit,
        "buying_power": bp,
        "places_orders": False,
        "basis": ("execution lease authorization: freshness + confirmation + exact identity + "
                  "broker truth + risk-mode sizing; mints a single-use short-lived lease only"),
    }

    # De-dupe reason codes, order-preserving.
    seen: set[str] = set()
    reason_codes = [r for r in reasons if not (r in seen or seen.add(r))]
    if reason_codes:
        return {**base, "authorized": False, "reason_codes": reason_codes, "lease": None,
                "next_action": ("authorization REFUSED (fail closed) — fix/refresh the named "
                                "inputs and re-run the FULL cycle via odte-convert; never widen "
                                "the policy to fit the order"),
                "next_command": "odte-convert  # after refreshing the failing inputs"}

    issued_at = now
    expires_at = now + timedelta(seconds=ttl_seconds)
    cand_fp = candidate_fingerprint(candidate)
    mkt_fp = market_fingerprint(market)
    lease_id = hashlib.sha1(
        f"{symbol}:{contract['option_id']}:{quantity}:{limit_price}:"
        f"{issued_at.isoformat()}:{cand_fp}:{mkt_fp}".encode()
    ).hexdigest()[:16]
    # PARTIAL_ACCOUNT lease maxima extend to the chase-band ceiling (the fill may pay up to
    # anchor*(1+band)), still capped by the tier's BP fraction. FULL_ACCOUNT maxima stay at the
    # exact reviewed limit/debit.
    if chase_ceiling is not None and partial_frac is not None and bp is not None:
        max_limit_val = float(chase_ceiling)
        max_debit_val = round(min(quantity * max_limit_val * 100.0, partial_frac * bp), 2)
        # MUTUAL CONSISTENCY (2026-08-06): when the BP fraction caps the debit below the chase
        # ceiling's worth, the published limit ceiling clamps DOWN (floored to the cent) so an
        # order at (quantity, max_limit_price) can NEVER violate max_debit. The QQQ 722C lease
        # published 0.86 x 100 = $86 against an $84.61 debit cap — internally contradictory
        # ceilings that the hook then (correctly) blocked. The clamp can never fall below the
        # reviewed limit_price: that debit already passed the fraction check above.
        affordable = int(max_debit_val / (quantity * 100.0) * 100.0) / 100.0
        max_limit_val = min(max_limit_val, affordable)
    else:
        max_limit_val = float(limit_price)
        max_debit_val = float(debit)
    lease = ExecutionLease(
        lease_id=lease_id, issued_at=issued_at, expires_at=expires_at,
        symbol=str(symbol), direction=str(direction),
        option_id=str(contract["option_id"]), expiration_date=str(contract["expiration_date"]),
        strike_price=float(contract["strike_price"]), option_type=str(contract["option_type"]),
        quantity=quantity, max_limit_price=max_limit_val, max_debit=max_debit_val,
        candidate_fingerprint=cand_fp, market_fingerprint=mkt_fp,
        risk_mode=risk_mode, max_premium_loss=max_premium_loss,
        tier=tier, anchor_quote=anchor,
    )
    return {**base, "authorized": True, "reason_codes": [], "lease": lease.to_dict(),
            "next_action": (f"lease {lease_id} issued — SINGLE USE, expires "
                            f"{expires_at.isoformat(timespec='seconds')} ({ttl_seconds:.0f}s): "
                            "consume it at broker review/place via the Hermes MCP lane, then run "
                            "the pending-order guard every tick until filled-fresh or cancelled"),
            "next_command": ("broker-review-place (Hermes MCP lane) → "
                             "odte-order-guard --order <broker order.json>")}


# --- IO wrapper + rendering -----------------------------------------------------------------

def run_execution_authorize(gate_json: str | None = None, gate_path: str | None = None,
                            candidate_decision_json: str | None = None,
                            candidate_decision_path: str | None = None,
                            vehicle_score_json: str | None = None,
                            vehicle_score_path: str | None = None,
                            broker_json: str | None = None, broker_path: str | None = None,
                            market_json: str | None = None, market_path: str | None = None,
                            policy_json: str | None = None, policy_path: str | None = None,
                            state_dir: str | None = None, write: bool = False,
                            now: datetime | None = None) -> dict:
    """Load the input artifacts, run `authorize_entry`, optionally persist the lease artifact.

    `write=True` stores the full authorization payload (lease included when authorized) at
    ``data/odte/execution_lease.json`` atomically so `odte-order-guard` / `odte-loop-status` can
    read it. No broker/network/LLM; places NO orders."""
    payload = authorize_entry(
        gate=_load_json(gate_path, gate_json),
        candidate_decision=_load_json(candidate_decision_path, candidate_decision_json),
        vehicle_score=_load_json(vehicle_score_path, vehicle_score_json),
        broker_snapshot=_load_json(broker_path, broker_json),
        market_snapshot=_load_json(market_path, market_json),
        now=now or datetime.now(timezone.utc),
        policy=_load_json(policy_path, policy_json),
    )
    if write:
        base = Path(os.path.expanduser(state_dir or DEFAULT_STATE_DIR))
        base.mkdir(parents=True, exist_ok=True)
        path = base / LEASE_FILENAME
        atomic_write_text(path, json.dumps(payload, indent=2, default=str))
        payload["artifact"] = str(path)
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    lease = p.get("lease") or {}
    lines = [f"# 0DTE execution authorization: "
             f"**{'LEASE ISSUED' if p.get('authorized') else 'REFUSED'}**",
             "",
             f"Symbol: **{p.get('symbol') or '?'}** {p.get('direction') or ''}  ·  "
             f"risk mode: {p.get('risk_mode')}  ·  quantity: {p.get('quantity')}  ·  "
             f"debit: {p.get('debit')}  ",
             f"Next: **{p.get('next_command')}** — {p.get('next_action')}  ",
             f"Places orders: {p.get('places_orders')}", ""]
    if lease:
        lines += ["## Lease (single use)",
                  *[f"- {k}: {lease.get(k)}" for k in
                    ("lease_id", "issued_at", "expires_at", "symbol", "direction", "option_id",
                     "expiration_date", "strike_price", "option_type", "quantity",
                     "max_limit_price", "max_debit", "risk_mode", "max_premium_loss",
                     "candidate_fingerprint", "market_fingerprint")], ""]
    if p.get("reason_codes"):
        lines += ["## Refusal reasons (fail closed)", *[f"- {r}" for r in p["reason_codes"]]]
    return "\n".join(lines)
