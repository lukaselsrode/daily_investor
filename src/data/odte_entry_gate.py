"""0DTE entry-gate decision builder — PURE/OFFLINE, NO broker/network/LLM, places NO orders.

This is the THESIS → ENTRY seam. It sits between the scan/trigger lane (`odte_watchdog`, which is
always `scan_only=True` / `execution_allowed=False`) and the execution manager (Hermes/MCP, which
stays autonomous). It does NOT trade, fetch, or call a broker — it only assembles a *structured,
journalable entry-gate record* the manager can read BEFORE it acts, so the decision is recorded the
same way every time and the manager is kept honest.

It is the ONE tier where `execution_allowed` may be True — and only under strict, conservative
conditions: every *required* gate must be EXPLICITLY True, the record must not be `scan_only`, and
the underlying must not be employer-restricted (NVDA). Any missing input fails CLOSED (a gate whose
input is absent is `None`, which is not True, so execution stays disallowed). Restricted underlyings
are always non-executable.

Inputs (all optional dicts, supplied by the caller from artifacts already collected upstream):
  trigger          a watchdog trigger payload (carries decision_context: thesis/confidence/
                   veto_reasons/observed_market_context/social_context/gamma_context)
  candidate        a candidate dict (ticker/direction/...) — pulled from `trigger` if absent
  day_score        `odte_day_score.score_day` output  {verdict: GOOD_DAY|CHOP|AVOID, ...}
  vehicle_score    `odte_vehicle_score.score_vehicle` output {verdict: GOOD_BET|WATCH|BAD_BET, ...}
  gamma_map        `odte_gamma_map` output (pin_risk{level}, gamma_available, ...)  — informational
  broker_snapshot  caller-supplied account-ish dict {buying_power, day_trades_left, blocked, ...}

The output is consumed by `odte_journal.event_from_entry_gate` → an `entry_decision` journal event;
the journal re-enforces `scan_only/restricted ⇒ execution_allowed=False` as defense in depth.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_REPORT_DIR

SCHEMA_VERSION = 1

# Default required gates — EVERY one must be explicitly True for execution_allowed. A missing input
# leaves its gate None (unknown) → not True → fails closed. Callers may override `required_gates`.
DEFAULT_REQUIRED_GATES = ("day_regime", "vehicle", "directional_thesis", "account")

# Live re-validations the manager must still perform before acting, even when the gates pass. These
# preserve Hermes autonomy (no `human_review` block) while recording the honest pre-trade checks.
REQUIRED_CONFIRMATIONS = ("live_chain_recheck", "spread_cap_check", "budget_check")

_BULLISH = {"call", "bullish", "long_call", "calls", "up"}
_BEARISH = {"put", "bearish", "long_put", "puts", "down"}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
        return out if out == out else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


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


def _norm_direction(value: Any) -> str | None:
    s = str(value or "").strip().lower()
    if s in _BULLISH:
        return "bullish"
    if s in _BEARISH:
        return "bearish"
    return None


def _coalesce_symbol(candidate: dict, vehicle_score: dict, trigger: dict) -> str | None:
    for src in (candidate, vehicle_score.get("contract") if isinstance(vehicle_score.get("contract"), dict) else {},
                vehicle_score, trigger):
        for key in ("ticker", "underlying", "symbol"):
            v = _dict(src).get(key) if isinstance(src, dict) else None
            if v:
                return str(v).upper()
    return None


def _account_gate(broker: dict) -> tuple[bool | None, str | None]:
    """Evaluate the account/buying-power gate. (True | False | None, veto_reason | None).

    True only when buying power is positive, the account is not blocked, and at least one day-trade
    remains. False (hard veto) on insufficient funds / blocked / no day-trades. None when no broker
    snapshot was supplied (fail closed — we will not authorize execution without confirmed funds)."""
    if not broker:
        return None, None
    if broker.get("blocked") is True or broker.get("trading_blocked") is True:
        return False, "account_blocked"
    dt_left = _num(broker.get("day_trades_left"))
    if dt_left is not None and dt_left <= 0:
        return False, "no_day_trades_left"
    bp = _num(broker.get("buying_power"))
    if bp is None:
        bp = _num(broker.get("account_buying_power") or broker.get("options_buying_power"))
    if bp is None:
        return None, None
    if bp <= 0:
        return False, "insufficient_buying_power"
    return True, None


def build_entry_gate_decision(trigger: dict | None = None, candidate: dict | None = None, *,
                              day_score: dict | None = None, vehicle_score: dict | None = None,
                              gamma_map: dict | None = None, broker_snapshot: dict | None = None,
                              required_gates: tuple[str, ...] | None = None,
                              scan_only: bool | None = None,
                              promote_to_execution: bool = False,
                              now: datetime | None = None) -> dict:
    """PURE: assemble a journalable entry-gate decision record. No IO/network/broker/orders.

    Returns a dict with: decision/intent (enter|deny|veto|observe), reason_codes, gates, veto_reasons,
    required_confirmations, thesis, confidence, scan_only, promoted_to_execution, execution_allowed.
    Conservative by construction — execution_allowed is True ONLY when scan_only is False, the symbol
    is not restricted, there are no veto reasons, and EVERY required gate is explicitly True. Restricted
    underlyings are always non-executable; any missing gate input fails closed.

    TIER BOUNDARY: a scan_only candidate must NOT silently become an execution candidate. When
    `scan_only` is not passed explicitly it is INHERITED from `trigger.scan_only`/`candidate.scan_only`
    (the watchdog lane is always scan_only=True). An inherited/explicit scan_only record can only be
    demoted to the execution tier when the manager EXPLICITLY sets `promote_to_execution=True` — this
    is still recording/tooling only, it places no orders."""
    trigger = _dict(trigger)
    dctx = _dict(trigger.get("decision_context"))
    candidate = _dict(candidate) or _dict(trigger.get("candidate"))
    day_score = _dict(day_score)
    vehicle_score = _dict(vehicle_score)
    gamma_map = _dict(gamma_map)
    broker = _dict(broker_snapshot)
    req = tuple(required_gates) if required_gates else DEFAULT_REQUIRED_GATES

    sym = _coalesce_symbol(candidate, vehicle_score, trigger)
    # Direction precedence: explicit candidate → vehicle_score → trigger thesis.
    direction = (_norm_direction(candidate.get("direction") or candidate.get("option_type"))
                 or _norm_direction(vehicle_score.get("direction"))
                 or _norm_direction(_dict(dctx.get("thesis")).get("direction")))

    from data.social_sentiment import is_restricted_underlying
    restricted = bool(sym and is_restricted_underlying(sym))

    # TIER BOUNDARY (see docstring): inherit scan_only from the upstream trigger/candidate unless the
    # caller states it explicitly; a scan_only record only drops to the execution tier on an EXPLICIT
    # promote_to_execution. This keeps a watchlist/scan name from silently becoming executable.
    if scan_only is not None:
        base_scan_only = bool(scan_only)
    else:
        base_scan_only = bool(trigger.get("scan_only")) or bool(candidate.get("scan_only"))
    promoted = bool(base_scan_only and promote_to_execution)
    scan_only = False if promoted else base_scan_only

    day_verdict = str(day_score.get("verdict") or "").upper()
    veh_verdict = str(vehicle_score.get("verdict") or "").upper()
    acct_ok, acct_veto = _account_gate(broker)

    gates: dict[str, bool | None] = {
        "day_regime": True if day_verdict == "GOOD_DAY" else (False if day_verdict == "AVOID" else None),
        "vehicle": True if veh_verdict == "GOOD_BET" else (False if veh_verdict == "BAD_BET" else None),
        "directional_thesis": True if direction else None,
        "account": acct_ok,
    }

    veto_reasons: list[str] = []
    if restricted:
        veto_reasons.append("restricted_employer")
    if day_verdict == "AVOID":
        veto_reasons.append("day_regime_avoid")
    if veh_verdict == "BAD_BET":
        veto_reasons.append("vehicle_bad_bet")
    if acct_veto:
        veto_reasons.append(acct_veto)
    # Carry forward any veto reasons the upstream trigger already recorded (deduped, order-preserving).
    for r in (dctx.get("veto_reasons") or []):
        if r and r not in veto_reasons:
            veto_reasons.append(str(r))

    reason_codes: list[str] = []
    for name in req:
        state = gates.get(name)
        reason_codes.append(f"{name}:{'ok' if state is True else 'fail' if state is False else 'unknown'}")
    if base_scan_only:
        reason_codes.append("scan_only_promoted_to_execution" if promoted else "scan_only_inherited")

    # HARD conservative gate: execution_allowed only when nothing blocks it and ALL required gates
    # are explicitly True. Missing inputs (None) fail closed.
    execution_allowed = (not scan_only and not restricted and not veto_reasons
                         and all(gates.get(g) is True for g in req))

    if restricted or veto_reasons:
        intent = "veto"
    elif scan_only:
        intent = "observe"
    elif execution_allowed:
        intent = "enter"
    elif any(gates.get(g) is None for g in req):
        intent = "observe"     # missing data — keep watching, do not deny outright
    else:
        intent = "deny"        # gates evaluated but not all positive

    confidence = dctx.get("confidence") if dctx.get("confidence") is not None else vehicle_score.get("score")

    thesis = _dict(dctx.get("thesis")) or {}
    basis = list(thesis.get("basis") or [])
    if not basis:
        basis = [str(r) for r in (vehicle_score.get("reasons") or [])[:4]]
    thesis_block = {
        "direction": direction or thesis.get("direction"),
        "basis": basis,
        "day_regime": day_verdict or None,
        "vehicle_verdict": veh_verdict or None,
    }

    gamma_ctx = dctx.get("gamma_context")
    if not gamma_ctx and gamma_map:
        pin = gamma_map.get("pin_risk")
        gamma_ctx = {"available": bool(gamma_map.get("gamma_available", True)),
                     "pin_risk": (pin.get("level") if isinstance(pin, dict) else pin),
                     "basis": "pin_risk_only_not_dealer_gex"}

    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "symbol": sym,
        "direction": direction,
        "decision": intent,
        "intent": intent,
        "reason_codes": reason_codes,
        "required_gates": list(req),
        "gates": gates,
        "veto_reasons": veto_reasons,
        "required_confirmations": list(REQUIRED_CONFIRMATIONS),
        "confirmation_needed": not execution_allowed,
        "thesis": thesis_block,
        "confidence": confidence,
        "observed_market_context": dctx.get("observed_market_context"),
        "social_context": dctx.get("social_context"),
        "gamma_context": gamma_ctx,
        "scan_only": scan_only,
        "promoted_to_execution": promoted,
        "execution_allowed": execution_allowed,
        "places_orders": False,
        "basis": ("offline entry-gate decision: day_regime + vehicle + directional_thesis + account "
                  "gates; records intent only, places NO orders"),
    }


def run_entry_gate(trigger_json: str | None = None, trigger_path: str | None = None,
                   candidate_json: str | None = None, candidate_path: str | None = None,
                   day_score_json: str | None = None, day_score_path: str | None = None,
                   vehicle_score_json: str | None = None, vehicle_score_path: str | None = None,
                   gamma_json: str | None = None, gamma_path: str | None = None,
                   broker_json: str | None = None, broker_path: str | None = None,
                   scan_only: bool | None = None, promote_to_execution: bool = False,
                   out_dir: str | None = None, write: bool = False) -> dict:
    """Load the (optional) input artifacts and build the entry-gate decision. No orders/broker/network.

    `scan_only=None` (the default) INHERITS scan_only from the trigger/candidate; pass True/False to
    state it explicitly. `promote_to_execution=True` is the manager's explicit opt-in to demote an
    (inherited) scan_only record to the execution tier."""
    payload = build_entry_gate_decision(
        trigger=_load_json(trigger_path, trigger_json) or None,
        candidate=_load_json(candidate_path, candidate_json) or None,
        day_score=_load_json(day_score_path, day_score_json) or None,
        vehicle_score=_load_json(vehicle_score_path, vehicle_score_json) or None,
        gamma_map=_load_json(gamma_path, gamma_json) or None,
        broker_snapshot=_load_json(broker_path, broker_json) or None,
        scan_only=scan_only,
        promote_to_execution=promote_to_execution,
    )
    if write:
        out = Path(os.path.expanduser(out_dir or ODTE_REPORT_DIR))
        out.mkdir(parents=True, exist_ok=True)
        sym = str(payload.get("symbol") or "candidate").lower()
        path = out / f"odte_entry_gate_{sym}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        payload["artifact"] = str(path)
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    gates = p.get("gates") or {}
    gate_line = " · ".join(f"{k}={'✅' if v is True else '❌' if v is False else '∅'}"
                           for k, v in gates.items()) or "—"
    lines = [f"# 0DTE entry gate: {p.get('symbol') or 'candidate'} {p.get('direction') or ''}".strip(),
             "",
             f"Decision: **{p.get('decision')}**  ",
             f"Execution allowed: **{p.get('execution_allowed')}**  ·  scan_only: {p.get('scan_only')}  ",
             f"Gates: {gate_line}  ",
             f"Basis: {p.get('basis')}", ""]
    if p.get("veto_reasons"):
        lines += ["## Veto reasons", *[f"- {r}" for r in p["veto_reasons"]], ""]
    lines += ["## Reason codes", *[f"- {r}" for r in (p.get("reason_codes") or [])]]
    if p.get("required_confirmations"):
        lines += ["", "## Required live confirmations (manager must still verify)",
                  *[f"- {c}" for c in p["required_confirmations"]]]
    return "\n".join(lines)
