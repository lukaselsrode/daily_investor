"""0DTE candidate watch — pre-entry HAWK lane, PURE/OFFLINE, places NO orders.

This module sits between broad scan/watchdog and entry-gate promotion. Once a potential setup is on
the board, it can be watched with faster cadence until it either confirms entry or degrades. It is
intentionally decision-only: NO broker calls, NO network, NO LLM, and ``execution_allowed`` is always
False. The live manager must still build/promote an entry gate and perform broker review before any
order.

VEHICLE LOCK (2026-07-23 delayed-fill remediation): the watched candidate carries an explicit
``selected_vehicle`` + ``selection_reason`` + ``relative_strength_rank`` + ``selection_timestamp``
and a ``candidate_fingerprint`` (persisted into ``active_candidate.json``). A vehicle-score
contract for ANOTHER underlying is a HARD mismatch (``DEGRADED_NO_TRADE``), never an equivalent
substitute — switching QQQ→SPY (or any cross-underlying swap) invalidates the current candidate and
requires a fresh watch/score/gate/lease cycle; the new cycle mints a NEW fingerprint, which
invalidates any lease bound to the old one. Broad-market disagreement stays confidence/risk context
(dissenters in ``checks``); it can never silently select a different vehicle.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_DATA_DIR, atomic_write_text
from data.odte_config import (
    A_PLUS_MIN_CONFIRMATIONS,
    B_PLUS_MAX_DISSENTERS,
    B_PLUS_MIN_CONFIRMATIONS,
    EXECUTABLE_UNIVERSE,
    SCAN_UNIVERSE,
)

SCHEMA_VERSION = 1
DEFAULT_STATE_DIR = ODTE_DATA_DIR
ACTIVE_CANDIDATE_FILENAME = "active_candidate.json"
CANDIDATE_DECISION_FILENAME = "candidate_decision.json"

# Scan universe: tape counted for confirmation/dissent. Executable universe: the
# only symbols a candidate may CONVERT on (XSP tape confirms but never converts —
# broker liquidity for this account is unproven).
ETF_UNIVERSE = SCAN_UNIVERSE
CONFIRM_ENTRY = "CONFIRM_ENTRY"
KEEP_WATCHING = "KEEP_WATCHING"
DEGRADED_NO_TRADE = "DEGRADED_NO_TRADE"
EXPIRED_NO_CONFIRMATION = "EXPIRED_NO_CONFIRMATION"
BROKER_BLOCKED = "BROKER_BLOCKED"

_BULLISH = {"bullish", "call", "calls", "up", "long_call"}
_BEARISH = {"bearish", "put", "puts", "down", "long_put"}


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


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _norm_direction(value: Any) -> str | None:
    s = str(value or "").strip().lower()
    if s in _BULLISH:
        return "bullish"
    if s in _BEARISH:
        return "bearish"
    return None


def _norm_symbol(value: Any) -> str | None:
    if not value:
        return None
    return str(value).strip().upper()


def _symbol_block(market: dict, symbol: str | None) -> dict:
    if not symbol:
        return {}
    sym = symbol.upper()
    for key in (sym, sym.lower(), "underlying"):
        block = market.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _bool_field(block: dict, name: str) -> bool | None:
    value = block.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1", "above"}:
            return True
        if value.lower() in {"false", "no", "0", "below"}:
            return False
    return None


def _orb_state(block: dict) -> str:
    return str(block.get("orb_state") or block.get("opening_range_state") or "").lower()


def _above_vwap(market: dict, symbol: str | None) -> bool | None:
    block = _symbol_block(market, symbol)
    return _bool_field(block, "above_vwap")


def _spot(market: dict, symbol: str | None) -> float | None:
    block = _symbol_block(market, symbol)
    for key in ("last", "price", "spot", "mark"):
        n = _num(block.get(key))
        if n is not None:
            return n
    return None


def _vixy_weak(market: dict) -> bool:
    block = _symbol_block(market, "VIXY") or _symbol_block(market, "VIX")
    if not block:
        return False
    above = _bool_field(block, "above_vwap")
    change = _num(block.get("change_pct") or block.get("pct_change"))
    if above is False:
        return True
    return bool(change is not None and change < 0)


def _vixy_firming(market: dict) -> bool:
    block = _symbol_block(market, "VIXY") or _symbol_block(market, "VIX")
    if not block:
        return False
    above = _bool_field(block, "above_vwap")
    change = _num(block.get("change_pct") or block.get("pct_change"))
    if above is True:
        return True
    return bool(change is not None and change > 0)


def _day_verdict(market: dict, day_score: dict) -> str:
    return str(day_score.get("verdict") or market.get("day_score") or market.get("day_verdict") or "").upper()


def _minutes_to_close(market: dict) -> float | None:
    return _num(market.get("minutes_to_close") or market.get("minutes_to_close_et"))


def _candidate_age_minutes(candidate: dict, now: datetime) -> float | None:
    created = _parse_ts(candidate.get("created_at") or candidate.get("ts") or candidate.get("generated_at"))
    if created is None:
        return None
    return (now - created).total_seconds() / 60.0


def _extract_candidate(candidate: dict, market: dict) -> dict:
    """Return an explicit candidate or a simple ETF tape candidate from market-only inputs."""
    cand = dict(candidate or {})
    sym = _norm_symbol(cand.get("ticker") or cand.get("underlying") or cand.get("symbol"))
    direction = _norm_direction(cand.get("direction") or cand.get("option_type"))
    if sym and direction:
        cand["ticker"] = sym
        cand["direction"] = direction
        return cand

    # Tape-only ETF lane: choose the strongest visible ETF candidate without social dependency.
    # Iterates only the EXECUTABLE universe so it never manufactures a scan-only (XSP) candidate.
    if market.get("candidate") and isinstance(market.get("candidate"), dict):
        return _extract_candidate(market["candidate"], market)
    for etf in EXECUTABLE_UNIVERSE:
        block = _symbol_block(market, etf)
        if not block:
            continue
        above = _above_vwap(market, etf)
        orb = _orb_state(block)
        if above is True and orb == "above" and _vixy_weak(market):
            return {"ticker": etf, "direction": "bullish", "source": "etf_momentum_tape"}
        if above is False and orb == "below" and _vixy_firming(market):
            return {"ticker": etf, "direction": "bearish", "source": "etf_momentum_tape"}
    return cand


def _confirmation_counts(market: dict, direction: str) -> tuple[int, list[str], list[str]]:
    """Count ETF tape confirmers and DISSENTERS for `direction`. A confirmer is above VWAP + above ORB
    (bullish) / below + below (bearish). A dissenter must carry a DEFINITIVE opposite read — its VWAP
    side must be present; missing/partial tape (XSP snapshots routinely lack VWAP/ORB fields) is
    neutral, never a silent veto. A+ confirmation (CHOP / late-day) requires zero dissenters."""
    hits: list[str] = []
    dissents: list[str] = []
    for etf in ETF_UNIVERSE:
        block = _symbol_block(market, etf)
        if not block:
            continue
        above = _above_vwap(market, etf)
        orb = _orb_state(block)
        if direction == "bullish":
            if above is True and orb == "above":
                hits.append(etf)
            elif above is not None and (above is False or orb == "below"):
                dissents.append(etf)
        else:
            if above is False and orb == "below":
                hits.append(etf)
            elif above is not None and (above is True or orb == "above"):
                dissents.append(etf)
    return len(hits), hits, dissents


def _pin_wall(candidate: dict, gamma: dict) -> float | None:
    contract = _dict(candidate.get("contract"))
    strike = _num(candidate.get("strike") or candidate.get("strike_price") or contract.get("strike") or
                  contract.get("strike_price"))
    direction = _norm_direction(candidate.get("direction") or candidate.get("option_type"))
    pin = _dict(gamma.get("pin_risk"))
    level = str(pin.get("level") or gamma.get("pin_risk") or "").lower()
    walls = [gamma.get("max_gamma_strike")]
    if direction == "bullish":
        walls.append(gamma.get("call_wall"))
    elif direction == "bearish":
        walls.append(gamma.get("put_wall"))
    for wall in walls:
        w = _num(wall)
        if w is not None and strike is not None and abs(w - strike) < 0.01 and level in {"high", "medium_high", "elevated"}:
            return w
    return None


def _accepted_wall(candidate: dict, market: dict, wall: float, direction: str) -> bool:
    if candidate.get("accepted_above_wall") is True or candidate.get("retest_hold") is True:
        return True
    sym = _norm_symbol(candidate.get("ticker") or candidate.get("underlying") or candidate.get("symbol"))
    spot = _spot(market, sym)
    if spot is None:
        return False
    buffer = max(0.03, wall * 0.0002)
    if direction == "bullish":
        return spot >= wall + buffer
    return spot <= wall - buffer


def _relative_strength_rank(market: dict, direction: str, symbol: str) -> int | None:
    """1-based tape-alignment rank of `symbol` among the ETF universe for `direction` (VWAP side +
    ORB side, deterministic, ties broken by universe order). Descriptive selection context only —
    never execution authority."""
    scores: list[tuple[str, int]] = []
    for etf in ETF_UNIVERSE:
        block = _symbol_block(market, etf)
        if not block:
            continue
        above = _bool_field(block, "above_vwap")
        orb = _orb_state(block)
        if direction == "bullish":
            score = int(above is True) + int(orb == "above")
        else:
            score = int(above is False) + int(orb == "below")
        scores.append((etf, score))
    names = [name for name, _ in scores]
    if symbol not in names:
        return None
    ranked = sorted(scores, key=lambda t: (-t[1], ETF_UNIVERSE.index(t[0])))
    return 1 + [name for name, _ in ranked].index(symbol)


def _lock_vehicle(cand: dict, symbol: str, direction: str, market: dict, now: datetime,
                  contract: dict | None = None) -> dict:
    """Stamp the exact VEHICLE/CONTRACT LOCK onto the candidate.

    The candidate fingerprint binds one selected underlying AND one exact option id/expiry/strike/type.
    A contract switch requires a new candidate cycle and fingerprint; affordability never permits an
    unjournaled strike substitution after confirmation.
    """
    out = dict(cand)
    out["ticker"] = symbol
    out["direction"] = direction
    out["selected_vehicle"] = symbol
    contract = _dict(contract)
    exact_fields = {
        "option_id": contract.get("option_id") or contract.get("id") or contract.get("occ_symbol"),
        "expiration_date": contract.get("expiration_date") or contract.get("expiration"),
        "strike_price": contract.get("strike_price") or contract.get("strike"),
        "option_type": contract.get("option_type") or contract.get("type"),
    }
    prior_identity = (
        out.get("option_id") or out.get("id") or out.get("occ_symbol"),
        out.get("expiration_date") or out.get("expiration"),
        out.get("strike_price") or out.get("strike"),
        str(out.get("option_type") or out.get("type") or "").lower() or None,
    )
    next_identity = (
        exact_fields["option_id"], exact_fields["expiration_date"], exact_fields["strike_price"],
        str(exact_fields["option_type"] or "").lower() or None,
    )
    if (all(value is not None for value in next_identity)
            and any(value is not None for value in prior_identity)
            and prior_identity != next_identity):
        out["selection_timestamp"] = now.isoformat()
        out["selection_reason"] = "exact_contract_switch_new_candidate_cycle"
    for name, value in exact_fields.items():
        if value is not None and value != "":
            out[name] = value
    out.setdefault("selection_reason",
                   str(out.get("source") or "explicit_candidate"))
    out["relative_strength_rank"] = _relative_strength_rank(market, direction, symbol)
    out.setdefault("selection_timestamp",
                   str(out.get("created_at") or out.get("ts") or out.get("generated_at")
                       or now.isoformat(timespec="seconds")))
    from data.odte_execution_policy import candidate_fingerprint
    out["candidate_fingerprint"] = candidate_fingerprint(out)
    return out


def evaluate_candidate_watch(candidate: dict | None = None, *, market: dict | None = None,
                             day_score: dict | None = None, vehicle_score: dict | None = None,
                             gamma_map: dict | None = None, broker_health: dict | None = None,
                             now: datetime | None = None, max_watch_minutes: int = 20) -> dict:
    """Pure pre-entry decision. Never authorizes execution; only says whether to keep watching."""
    now = now or datetime.now(timezone.utc)
    market = _dict(market)
    day_score = _dict(day_score)
    vehicle_score = _dict(vehicle_score)
    gamma_map = _dict(gamma_map)
    broker = _dict(broker_health)
    cand = _extract_candidate(_dict(candidate), market)

    sym = _norm_symbol(cand.get("ticker") or cand.get("underlying") or cand.get("symbol"))
    direction = _norm_direction(cand.get("direction") or cand.get("option_type"))
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    from data.social_sentiment import is_restricted_underlying
    if not sym or sym not in EXECUTABLE_UNIVERSE:
        decision = DEGRADED_NO_TRADE if sym else KEEP_WATCHING
        reasons.append(f"candidate not in executable universe {'/'.join(EXECUTABLE_UNIVERSE)}"
                       if sym else "no candidate yet")
        return _payload(decision, cand, reasons, checks, now)
    if is_restricted_underlying(sym):
        reasons.append("restricted employer symbol")
        return _payload(DEGRADED_NO_TRADE, cand, reasons, checks, now)
    if not direction:
        reasons.append("candidate direction missing")
        return _payload(KEEP_WATCHING, cand, reasons, checks, now)

    # VEHICLE LOCK: pin the selected vehicle + fingerprint on the candidate the moment identity is
    # known, so every downstream artifact (active_candidate.json, gate, lease) binds to ONE vehicle.
    cand = _lock_vehicle(cand, sym, direction, market, now,
                         contract=_dict(vehicle_score.get("contract")))
    checks["selected_vehicle"] = sym
    checks["candidate_fingerprint"] = cand.get("candidate_fingerprint")

    # HARD MISMATCH: a vehicle-score contract for another underlying is never an equivalent
    # substitute (the QQQ-thesis→SPY-contract failure). Degrade — a switch requires a FRESH
    # watch/score/gate/lease cycle for the new underlying.
    contract = _dict(vehicle_score.get("contract"))
    contract_underlying = _norm_symbol(contract.get("underlying") or contract.get("symbol")
                                       or contract.get("chain_symbol"))
    if contract_underlying and contract_underlying != sym:
        checks["vehicle_underlying"] = contract_underlying
        reasons.append(f"vehicle underlying mismatch: contract {contract_underlying} != candidate "
                       f"{sym} — hard mismatch; a vehicle switch requires a fresh "
                       "watch/score/gate/lease cycle")
        return _payload(DEGRADED_NO_TRADE, cand, reasons, checks, now)

    age = _candidate_age_minutes(cand, now)
    if age is not None and age > max_watch_minutes:
        checks["age_minutes"] = round(age, 1)
        reasons.append(f"candidate expired ({age:.0f}m > {max_watch_minutes}m)")
        return _payload(EXPIRED_NO_CONFIRMATION, cand, reasons, checks, now)

    if broker.get("blocked") is True or broker.get("execution_lane") == "blocked":
        reasons.append("broker/review lane blocked")
        return _payload(BROKER_BLOCKED, cand, reasons, checks, now)

    day = _day_verdict(market, day_score)
    checks["day_verdict"] = day or None
    if day == "AVOID":
        reasons.append("day_score=AVOID")
        return _payload(DEGRADED_NO_TRADE, cand, reasons, checks, now)

    veh = str(vehicle_score.get("verdict") or "").upper()
    if veh == "BAD_BET":
        reasons.append("vehicle_score=BAD_BET")
        return _payload(DEGRADED_NO_TRADE, cand, reasons, checks, now)

    underlying_above = _above_vwap(market, sym)
    underlying_orb = _orb_state(_symbol_block(market, sym))
    confirmations, confirmers, dissenters = _confirmation_counts(market, direction)
    checks.update({"underlying_above_vwap": underlying_above,
                   "underlying_orb_state": underlying_orb,
                   "confirmations": confirmations,
                   "confirmers": confirmers,
                   "dissenters": dissenters,
                   "vixy_weak": _vixy_weak(market),
                   "vixy_firming": _vixy_firming(market),
                   "minutes_to_close": _minutes_to_close(market)})
    # Telemetry (2026-08-05): _vixy_weak short-circuits on above_vwap=False, _vixy_firming on
    # above_vwap=True — an above-VWAP-but-down-on-day VIXY satisfies BOTH, silently handing a
    # free vol confirmation to EITHER direction. Flag it; the mutual-exclusion fix waits for a
    # clean session of this telemetry.
    if checks["vixy_weak"] and checks["vixy_firming"]:
        checks["vixy_conflict"] = True

    if direction == "bullish":
        invalidated = underlying_above is False or underlying_orb == "below"
        confirmed = underlying_above is True and underlying_orb == "above" and confirmations >= 2 and _vixy_weak(market)
    else:
        invalidated = underlying_above is True and underlying_orb == "above"
        confirmed = underlying_above is False and underlying_orb == "below" and confirmations >= 2 and _vixy_firming(market)

    if invalidated:
        reasons.append(f"{sym} tape invalidated {direction} candidate")
        return _payload(DEGRADED_NO_TRADE, cand, reasons, checks, now)

    # Confirmation tiers. A+ = confirmed underlying, >=3 ETF confirmers, ZERO dissenters — demanded
    # in the late-day window, where a marginal read fails. B+ = confirmed with >=2 confirmers and at
    # most 1 definitive dissenter — tradeable on CHOP days at HALF size (the 2026-07-27..31 zero-trade
    # week showed CHOP-with-directional-tape days are the modal setup; A+-or-nothing left them all
    # unconverted). The tier rides the candidate into gate/lease, where it halves the debit fraction.
    a_plus = confirmed and confirmations >= A_PLUS_MIN_CONFIRMATIONS and not dissenters
    b_plus = (confirmed and confirmations >= B_PLUS_MIN_CONFIRMATIONS
              and len(dissenters) <= B_PLUS_MAX_DISSENTERS)
    tier = "a_plus" if a_plus else ("b_plus" if (day == "CHOP" and b_plus) else "full")
    checks["tier"] = tier

    mtc = _minutes_to_close(market)
    if mtc is not None and mtc < 45 and not a_plus:
        reasons.append("late-day window requires A+ confirmation")
        return _payload(KEEP_WATCHING, cand, reasons, checks, now)

    if day == "CHOP" and not a_plus and not b_plus:
        reasons.append("CHOP requires at least B+ ETF confirmation")
        return _payload(KEEP_WATCHING, cand, reasons, checks, now)

    wall = _pin_wall(cand, gamma_map)
    if wall is not None:
        checks["pin_wall"] = wall
        if not _accepted_wall(cand, market, wall, direction):
            reasons.append("pin/call-wall acceptance required before entry confirmation")
            return _payload(KEEP_WATCHING, cand, reasons, checks, now)

    if confirmed:
        # Stamp the sizing tier and the CHASE-BAND ANCHOR at the moment of confirmation. The anchor
        # is the fill-price reference for the lease ceiling (anchor * (1 + chase band)); anchoring
        # here — not at precompute — is what stops the rail from vetoing entries whose premium rose
        # BECAUSE the thesis confirmed. Deliberately NOT part of the identity fingerprint: a
        # re-confirm re-anchors the same contract cycle without invalidating identity.
        cand["tier"] = tier
        anchor = _num(contract.get("ask") or contract.get("ask_price")
                      or contract.get("mark") or contract.get("mark_price"))
        if anchor is not None and anchor > 0:
            cand["anchor_quote"] = anchor
            cand["anchor_ts"] = now.isoformat(timespec="seconds")
            checks["anchor_quote"] = anchor
        reasons.append(f"ETF tape confirmed candidate ({tier}); run odte-convert next")
        return _payload(CONFIRM_ENTRY, cand, reasons, checks, now)
    reasons.append("candidate still forming; keep watching confirmation")
    return _payload(KEEP_WATCHING, cand, reasons, checks, now)


def _payload(decision: str, candidate: dict, reasons: list[str], checks: dict, now: datetime) -> dict:
    cand = dict(candidate or {})
    cand.setdefault("scan_only", True)
    state = "WATCHING_CONFIRMATION"
    if decision == CONFIRM_ENTRY:
        state = "CONFIRMED_ENTRY"
    elif decision == BROKER_BLOCKED:
        state = "BROKER_BLOCKED"
    elif decision in {DEGRADED_NO_TRADE, EXPIRED_NO_CONFIRMATION}:
        state = "INACTIVE"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "state": state,
        "decision": decision,
        "candidate": cand,
        "scan_only": True,
        "execution_allowed": False,
        "places_orders": False,
        "next_action": _next_action(decision),
        "reasons": reasons,
        "checks": {k: v for k, v in checks.items() if v is not None},
    }


def _next_action(decision: str) -> str:
    if decision == CONFIRM_ENTRY:
        return "run odte-convert (atomic gate+lease in one process); still no order until broker review consumes the lease"
    if decision == KEEP_WATCHING:
        return "continue candidate HAWK checks until confirm or degrade"
    if decision == BROKER_BLOCKED:
        return "do not enter; repair/verify broker review lane before promotion"
    if decision == EXPIRED_NO_CONFIRMATION:
        return "expire candidate and resume broad scan"
    return "candidate degraded; resume broad scan"


def run_candidate_watch(candidate_json: str | None = None, candidate_path: str | None = None,
                        market_json: str | None = None, market_path: str | None = None,
                        day_score_json: str | None = None, day_score_path: str | None = None,
                        vehicle_score_json: str | None = None, vehicle_score_path: str | None = None,
                        gamma_json: str | None = None, gamma_path: str | None = None,
                        broker_health_json: str | None = None, broker_health_path: str | None = None,
                        state_dir: str | None = None, write: bool = False) -> dict:
    candidate = _load_json(candidate_path, candidate_json)
    market = _load_json(market_path, market_json)
    day_score = _load_json(day_score_path, day_score_json)
    vehicle = _load_json(vehicle_score_path, vehicle_score_json)
    gamma = _load_json(gamma_path, gamma_json)
    broker = _load_json(broker_health_path, broker_health_json)
    payload = evaluate_candidate_watch(candidate, market=market, day_score=day_score,
                                       vehicle_score=vehicle, gamma_map=gamma,
                                       broker_health=broker)
    if write:
        base = Path(os.path.expanduser(state_dir or DEFAULT_STATE_DIR))
        base.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, default=str)
        atomic_write_text(base / CANDIDATE_DECISION_FILENAME, text)
        # Keep active_candidate as the compact watch plan while the candidate is live.
        if payload["decision"] in {KEEP_WATCHING, CONFIRM_ENTRY, BROKER_BLOCKED}:
            active = dict(payload["candidate"])
            active.update({"state": payload["state"], "updated_at": payload["generated_at"],
                           "decision": payload["decision"], "scan_only": True,
                           "execution_allowed": False})
            atomic_write_text(base / ACTIVE_CANDIDATE_FILENAME,
                              json.dumps(active, indent=2, default=str))
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    cand = p.get("candidate") or {}
    lines = [f"# 0DTE candidate watch: **{p.get('decision')}**",
             "",
             f"Candidate: **{cand.get('ticker') or cand.get('underlying')}** {cand.get('direction')}  ",
             f"Execution allowed: {p.get('execution_allowed')} · places orders: {p.get('places_orders')}  ",
             f"Next: {p.get('next_action')}", ""]
    if p.get("reasons"):
        lines += ["## Why", *[f"- {r}" for r in p["reasons"]], ""]
    if p.get("checks"):
        lines += ["## Checks", *[f"- {k}: {v}" for k, v in p["checks"].items()]]
    return "\n".join(lines)
