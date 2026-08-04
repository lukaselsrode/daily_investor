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

An execution-allowed GATE is still not order authority (2026-07-23 delayed-fill remediation): the
manager must additionally mint a fresh, short-lived, exact-identity execution LEASE via
`odte-execution-authorize` (data.odte_execution_policy) before broker review/place, and run the
pending-order guard (`odte-order-guard`) until filled-fresh or cancelled. The deprecated bare
`promote_to_execution` boolean fails CLOSED with reason `execution_lease_required`.

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
from data.odte_config import B_PLUS_MIN_VEHICLE_SCORE, GREEN_REENTRY_MIN_BP_MULTIPLE

SCHEMA_VERSION = 1

# Default required gates — EVERY one must be explicitly True for execution_allowed. A missing input
# leaves its gate None (unknown) → not True → fails closed. Callers may override `required_gates`.
DEFAULT_REQUIRED_GATES = ("day_regime", "vehicle", "directional_thesis", "account")

# Live re-validations the manager must still perform before acting, even when the gates pass. These
# preserve Hermes autonomy (no `human_review` block) while recording the honest pre-trade checks.
REQUIRED_CONFIRMATIONS = ("live_chain_recheck", "spread_cap_check", "budget_check")

_BULLISH = {"call", "bullish", "long_call", "calls", "up"}
_BEARISH = {"put", "bearish", "long_put", "puts", "down"}

# Green-day preservation (post-scalp lockout). Once a completed profitable trade is banked for the
# ET day (per journal events), a fresh entry gate is VETOED — the banked green must not be handed
# back on a re-entry (live 2026-07-10 failure: 3x SPY 753C scalped green, then re-entered 1x for a
# scratch). The ONLY way back in is an EXPLICIT `allow_reentry_after_green: true` on the trigger or
# broker snapshot, and even then buying power must comfortably cover the re-entry at a multiple of
# the contract's own estimated cost (BP-SCALED, 2026-08-02 retune: the old flat $500 floor sat above
# the whole account's buying power, so the override could never arm and any green trade permanently
# ended the day — incompatible with the 2-trades/day budget). Unknown contract cost stays locked.
GREEN_LOCKOUT_VETO = "green_day_preservation_lockout"
GREEN_REENTRY_BP_VETO = "insufficient_bp_for_green_reentry"

# Daily trade cadence (2026-08-02 retune): a hard per-ET-day entry budget plus a cooldown after
# every completed trade. Enforced from the day's journal events (see odte_journal.daily_trade_budget).
DAILY_BUDGET_VETO = "daily_trade_budget_exhausted"
COOLDOWN_VETO = "post_trade_cooldown_active"

# FAIL-CLOSED promotion (2026-07-23 delayed-fill incident): `promote_to_execution=True` is a
# DEPRECATED input. A bare boolean can no longer demote a scan-tier record to the execution tier —
# execution authority is minted ONLY by a fresh, exact-identity, short-lived execution lease
# (`odte-execution-authorize` / data.odte_execution_policy.authorize_entry). A promotion attempt is
# recorded and answered with this reason code so the controller runs the lease path instead.
EXECUTION_LEASE_REQUIRED = "execution_lease_required"

# A confirmed candidate may cross the scan→gate boundary only through a fresh, identity-matched
# candidate-watch decision. The lease tier enforces the same 60s ceiling again: the transition must
# happen in the SAME live-controller tick, not on the next two-minute cron run.
CONFIRMED_CANDIDATE_MAX_AGE_SECONDS = 60.0
CONFIRMED_CANDIDATE_MAX_FUTURE_SKEW_SECONDS = 2.0

# What produces each missing gate input — surfaced verbatim in next_action so the controller
# refreshes the exact input instead of stalling on an "unknown" gate.
_GATE_INPUT_COMMANDS = {
    "day_regime": "make odte-day-score MARKET=<market.json> JSON=1",
    "vehicle": "make odte-vehicle-score CONTRACT=<contract.json> MARKET=<market.json> BP=<bp> JSON=1",
    "account": "supply a fresh broker snapshot (BROKER=<broker.json> from the Hermes MCP read lane)",
    "directional_thesis": "make odte-watchdog JSON=1  # or pass CANDIDATE= with an explicit direction",
}
# Veto reasons that mean the CONTRACT didn't fit, not the day: before declaring no-trade on these,
# scan the other index ETF vehicles for a BP-fit structure — one bullet a day must not die on the
# first contract checked.
_BP_FIT_VETOES = {"insufficient_buying_power", "vehicle_bad_bet"}


def _next_step(intent: str, execution_allowed: bool, gates: dict, veto_reasons: list[str],
               req: tuple[str, ...]) -> tuple[str, str]:
    """(next_action prose, next_command) for the gate record — decision-support only, never an order.

    Encodes the anti-passivity ladder: a passed gate says GO to broker review/place (standing auth);
    a BP/vehicle-fit fail says scan QQQ/SPY/IWM for a fitting vehicle BEFORE declaring no-trade; a
    missing input names the exact command that produces it; a hard veto stands down to the scan lane."""
    if execution_allowed:
        return ("ALL GATES PASSED — mint a fresh execution lease (odte-execution-authorize), do "
                "the final live refresh, then broker review/place under standing auth via the "
                "Hermes MCP lane (this repo places NO orders); run the pending-order guard every "
                "tick until filled-fresh or cancelled, then start the position watch",
                "odte-execution-authorize → broker-review-place (Hermes MCP lane) → "
                "odte-order-guard → odte-position --snapshot <live.json>")
    if GREEN_LOCKOUT_VETO in veto_reasons or GREEN_REENTRY_BP_VETO in veto_reasons:
        return ("green-day preservation lockout — a profitable trade is already banked today; NO "
                "fresh entries this session. Re-entry needs BOTH an explicit "
                "allow_reentry_after_green=true on the trigger/broker snapshot AND buying power "
                "above the re-entry floor. Bank the green day and review.",
                "odte-journal-report --write  # bank the green day; do not re-enter")
    if any(v in _BP_FIT_VETOES for v in veto_reasons):
        return ("vehicle/BP fit failed for THIS contract — before declaring no-trade on BP, scan "
                "the other index ETF vehicles (QQQ/SPY/IWM) for a BP-fit structure; reject only if "
                "every candidate vehicle fails",
                "make odte-vehicle-score CONTRACT=<qqq|spy|iwm contract.json> "
                "MARKET=<market.json> BP=<bp> JSON=1")
    if intent == "veto":
        return (f"hard veto ({', '.join(veto_reasons) or 'restricted'}) — stand down for this "
                "candidate and resume the scan lane",
                "odte-watchdog")
    unknown = [g for g in req if gates.get(g) is None]
    failing = [g for g in req if gates.get(g) is False]
    if unknown:
        cmds = "; ".join(_GATE_INPUT_COMMANDS.get(g, f"supply the {g} input") for g in unknown)
        return (f"gate inputs missing ({', '.join(unknown)}) — produce them and re-run the gate: "
                f"{cmds}",
                "odte-entry-gate  # re-run with the missing inputs supplied")
    if failing:
        return (f"gates failing ({', '.join(failing)}) — keep the candidate HAWK loop until a "
                "materially new read flips the failing gate or the candidate degrades; do not "
                "re-promote on the same read",
                "odte-candidate-watch")
    # scan_only observe record with every gate green: bare promotion is DEAD (2026-07-23 incident).
    # Execution authority is minted only by a fresh execution lease bound to one exact contract.
    return ("all required gates read True but the record is scan-tier — bare promotion is "
            "disabled; execution authority is minted ONLY by a fresh execution lease "
            "(exact contract/direction/quantity/price, ~30s TTL) via odte-execution-authorize",
            "odte-execution-authorize")


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


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _norm_symbol(value: Any) -> str | None:
    return str(value).strip().upper() if value else None


def _confirmed_candidate_transition(candidate_decision: dict, candidate: dict,
                                    now: datetime) -> tuple[bool, list[str]]:
    """Validate the only safe scan→gate transition.

    A bare boolean still cannot promote anything. The caller must supply a fresh candidate-watch
    payload whose decision is CONFIRM_ENTRY and whose locked candidate fingerprint, symbol,
    direction, and selected vehicle match the candidate passed to the entry gate.
    """
    if not candidate_decision:
        return False, []
    reasons: list[str] = []
    if str(candidate_decision.get("decision") or "").upper() != "CONFIRM_ENTRY":
        reasons.append("candidate_decision_not_confirmed")
    generated_at = _parse_ts(candidate_decision.get("generated_at") or candidate_decision.get("ts"))
    if generated_at is None:
        reasons.append("candidate_decision_timestamp_missing")
    else:
        age_seconds = (now - generated_at).total_seconds()
        if age_seconds < -CONFIRMED_CANDIDATE_MAX_FUTURE_SKEW_SECONDS:
            reasons.append("candidate_decision_from_future")
        elif age_seconds > CONFIRMED_CANDIDATE_MAX_AGE_SECONDS:
            reasons.append("candidate_decision_stale_for_transition")

    locked = _dict(candidate_decision.get("candidate"))
    if not locked:
        reasons.append("candidate_decision_candidate_missing")
        return False, reasons

    from data.odte_execution_policy import candidate_fingerprint

    # Never trust a caller-supplied fingerprint as proof of the fields around it: recompute both
    # sides, validate any embedded fingerprints, then compare the recomputed identities.
    locked_fp = candidate_fingerprint(locked)
    candidate_fp = candidate_fingerprint(candidate)
    supplied_locked_fp = locked.get("candidate_fingerprint")
    supplied_candidate_fp = candidate.get("candidate_fingerprint")
    if supplied_locked_fp and supplied_locked_fp != locked_fp:
        reasons.append("candidate_decision_fingerprint_invalid")
    if supplied_candidate_fp and supplied_candidate_fp != candidate_fp:
        reasons.append("candidate_artifact_fingerprint_invalid")
    if locked_fp != candidate_fp:
        reasons.append("candidate_fingerprint_mismatch")

    locked_symbol = _norm_symbol(locked.get("ticker") or locked.get("underlying")
                                 or locked.get("symbol"))
    candidate_symbol = _norm_symbol(candidate.get("ticker") or candidate.get("underlying")
                                    or candidate.get("symbol"))
    if not locked_symbol or not candidate_symbol or locked_symbol != candidate_symbol:
        reasons.append("candidate_symbol_mismatch")

    locked_direction = _norm_direction(locked.get("direction") or locked.get("option_type"))
    candidate_direction = _norm_direction(candidate.get("direction") or candidate.get("option_type"))
    if not locked_direction or not candidate_direction or locked_direction != candidate_direction:
        reasons.append("candidate_direction_mismatch")

    locked_vehicle = _norm_symbol(locked.get("selected_vehicle") or locked_symbol)
    candidate_vehicle = _norm_symbol(candidate.get("selected_vehicle") or candidate_symbol)
    if not locked_vehicle or not candidate_vehicle or locked_vehicle != candidate_vehicle:
        reasons.append("candidate_vehicle_mismatch")

    locked_selected_at = (locked.get("selection_timestamp") or locked.get("created_at")
                          or locked.get("ts") or locked.get("generated_at"))
    candidate_selected_at = (candidate.get("selection_timestamp") or candidate.get("created_at")
                             or candidate.get("ts") or candidate.get("generated_at"))
    if not locked_selected_at or not candidate_selected_at:
        reasons.append("candidate_cycle_timestamp_missing")

    exact_fields = {
        "option_id": lambda c: c.get("option_id") or c.get("id") or c.get("occ_symbol"),
        "expiration_date": lambda c: c.get("expiration_date") or c.get("expiration"),
        "strike_price": lambda c: _num(c.get("strike_price") or c.get("strike")),
        "option_type": lambda c: str(c.get("option_type") or c.get("type") or "").lower() or None,
    }
    for field, getter in exact_fields.items():
        locked_value = getter(locked)
        candidate_value = getter(candidate)
        if locked_value is None or candidate_value is None:
            reasons.append(f"candidate_contract_{field}_missing")
        elif locked_value != candidate_value:
            reasons.append(f"candidate_contract_{field}_mismatch")
    return not reasons, reasons


def _coalesce_symbol(candidate: dict, vehicle_score: dict, trigger: dict) -> str | None:
    for src in (candidate, vehicle_score.get("contract") if isinstance(vehicle_score.get("contract"), dict) else {},
                vehicle_score, trigger):
        for key in ("ticker", "underlying", "symbol"):
            v = _dict(src).get(key) if isinstance(src, dict) else None
            if v:
                return str(v).upper()
    return None


def _buying_power(broker: dict) -> float | None:
    bp = _num(broker.get("buying_power"))
    if bp is None:
        bp = _num(broker.get("account_buying_power") or broker.get("options_buying_power"))
    return bp


def _broker_count(broker: dict, *keys: str) -> float | None:
    account: dict = _dict(broker.get("account"))
    for container in (broker, account):
        for key in keys:
            value = _num(container.get(key))
            if value is not None:
                return value
    return None


def _contract_cost(vehicle_score: dict) -> float | None:
    """Estimated debit for ONE contract from the vehicle-score contract (ask, else mark), ×100."""
    contract = _dict(vehicle_score.get("contract"))
    premium = (_num(contract.get("ask") or contract.get("ask_price"))
               or _num(contract.get("mark") or contract.get("mark_price")))
    return premium * 100.0 if premium else None


def _account_gate(broker: dict) -> tuple[bool | None, str | None]:
    """Evaluate the account/buying-power gate. (True | False | None, veto_reason | None).

    True only when buying power is positive, the account is not blocked/locked, and broker truth
    explicitly reports flat option positions, no open option orders, today's order count, and at
    least one day-trade when that field is supplied. Missing broker prerequisites fail closed so the
    gate can never advertise EXECUTION_READY when the lease tier must refuse."""
    if not broker:
        return None, "broker_snapshot_missing"
    if broker.get("blocked") is True or broker.get("trading_blocked") is True:
        return False, "account_blocked"
    if broker.get("controller_locked") is True:
        return False, "controller_locked"
    dt_left = _num(broker.get("day_trades_left"))
    if dt_left is not None and dt_left <= 0:
        return False, "no_day_trades_left"
    bp = _buying_power(broker)
    if bp is None:
        return None, "buying_power_missing"
    if bp <= 0:
        return False, "insufficient_buying_power"
    positions = _broker_count(broker, "nonzero_option_positions_count",
                              "open_option_positions_count", "positions_count",
                              "open_positions_count")
    if positions is None:
        return None, "broker_positions_count_missing"
    if positions > 0:
        return False, "position_already_open"
    open_orders = _broker_count(broker, "open_option_orders_count", "open_orders_count")
    if open_orders is None:
        return None, "broker_open_orders_count_missing"
    if open_orders > 0:
        return False, "open_order_outstanding"
    if _broker_count(broker, "today_option_orders_count", "today_orders_count") is None:
        return None, "broker_today_orders_count_missing"
    return True, None


def build_entry_gate_decision(trigger: dict | None = None, candidate: dict | None = None, *,
                              candidate_decision: dict | None = None,
                              day_score: dict | None = None, vehicle_score: dict | None = None,
                              gamma_map: dict | None = None, broker_snapshot: dict | None = None,
                              confirmations: dict | None = None,
                              required_gates: tuple[str, ...] | None = None,
                              scan_only: bool | None = None,
                              promote_to_execution: bool = False,
                              journal_events: list[dict] | None = None,
                              now: datetime | None = None) -> dict:
    """PURE: assemble a journalable entry-gate decision record. No IO/network/broker/orders.

    Returns a dict with: decision/intent (enter|deny|veto|observe), reason_codes, gates, veto_reasons,
    required_confirmations, thesis, confidence, scan_only, promoted_to_execution, execution_allowed.
    Conservative by construction — execution_allowed is True ONLY when scan_only is False, the symbol
    is not restricted, there are no veto reasons, and EVERY required gate is explicitly True. Restricted
    underlyings are always non-executable; any missing gate input fails closed.

    TIER BOUNDARY: a scan_only candidate must NOT silently become an execution candidate. When
    `scan_only` is not passed explicitly it is INHERITED from `trigger.scan_only`/`candidate.scan_only`
    (the watchdog lane is always scan_only=True). The only safe transition is a fresh, identity-matched
    `candidate_decision.decision == CONFIRM_ENTRY` from candidate-watch in the same controller tick.
    FAIL-CLOSED (2026-07-23 delayed-fill incident): `promote_to_execution=True` is a DEPRECATED input —
    it no longer demotes a scan-tier record to the execution tier. The attempt is recorded
    (`promotion_requested`) and answered with the `execution_lease_required` reason code; execution
    authority is minted ONLY by a fresh short-lived exact-identity lease
    (`odte-execution-authorize`).

    GREEN-DAY PRESERVATION: when `journal_events` (the day's decision journal) shows a completed
    profitable trade for the current ET day, the gate VETOES (`green_day_preservation_lockout`) —
    re-entry needs an explicit `allow_reentry_after_green: true` on the trigger/broker snapshot AND
    buying power at/above GREEN_REENTRY_MIN_BP_MULTIPLE × the contract's estimated cost, else
    `insufficient_bp_for_green_reentry`. `journal_events` also drives the DAILY TRADE BUDGET: at
    most `odte_config.DAILY_TRADE_BUDGET` entries per ET day (`daily_trade_budget_exhausted`) with a
    `REENTRY_COOLDOWN_MINUTES` gap after every completed trade (`post_trade_cooldown_active`)."""
    trigger = _dict(trigger)
    dctx = _dict(trigger.get("decision_context"))
    candidate = _dict(candidate) or _dict(trigger.get("candidate"))
    candidate_decision = _dict(candidate_decision)
    if not candidate and candidate_decision:
        candidate = _dict(candidate_decision.get("candidate"))
    day_score = _dict(day_score)
    vehicle_score = _dict(vehicle_score)
    gamma_map = _dict(gamma_map)
    broker = _dict(broker_snapshot)
    confirmations = _dict(confirmations)
    req = tuple(required_gates) if required_gates else DEFAULT_REQUIRED_GATES
    current_now = now or datetime.now(timezone.utc)
    if current_now.tzinfo is None:
        current_now = current_now.replace(tzinfo=timezone.utc)
    from data.odte_execution_policy import candidate_fingerprint
    computed_candidate_fingerprint = candidate_fingerprint(candidate) if candidate else None

    sym = _coalesce_symbol(candidate, vehicle_score, trigger)
    # Direction precedence: explicit candidate → vehicle_score → trigger thesis.
    direction = (_norm_direction(candidate.get("direction") or candidate.get("option_type"))
                 or _norm_direction(vehicle_score.get("direction"))
                 or _norm_direction(_dict(dctx.get("thesis")).get("direction")))

    from data.social_sentiment import is_restricted_underlying
    restricted = bool(sym and is_restricted_underlying(sym))

    # TIER BOUNDARY (see docstring): inherit scan_only from the upstream trigger/candidate unless
    # the caller states it explicitly. A fresh, identity-matched CONFIRM_ENTRY decision may cross
    # that boundary in the same controller tick. The deprecated bare promotion flag still cannot.
    transition_reasons: list[str] = []
    confirmed_candidate_transition = False
    if scan_only is not None:
        base_scan_only = bool(scan_only)
    else:
        base_scan_only = bool(trigger.get("scan_only")) or bool(candidate.get("scan_only"))
        if base_scan_only:
            confirmed_candidate_transition, transition_reasons = _confirmed_candidate_transition(
                candidate_decision, candidate, current_now)
    promotion_requested = bool(base_scan_only and promote_to_execution)
    scan_only = bool(base_scan_only and not confirmed_candidate_transition)

    day_verdict = str(day_score.get("verdict") or "").upper()
    veh_verdict = str(vehicle_score.get("verdict") or "").upper()
    acct_ok, acct_veto = _account_gate(broker)

    # Confirmation tier, stamped by candidate-watch at CONFIRM_ENTRY (tape-computed, never a model
    # label). CHOP is no longer an automatic unknown: an A+/B+ tier passes the day gate (B+ at half
    # size — the lease tier halves the debit fraction), and a WATCH vehicle verdict passes for B+
    # when its raw score clears the floor. Bare CHOP/WATCH still fail closed.
    tier = (str(candidate.get("tier")
                or _dict(candidate_decision.get("candidate")).get("tier") or "").lower() or None)
    veh_score_total = _num(vehicle_score.get("score"))

    def _day_regime_gate() -> bool | None:
        if day_verdict == "GOOD_DAY":
            return True
        if day_verdict == "AVOID":
            return False
        if day_verdict == "CHOP" and tier in ("a_plus", "b_plus"):
            return True
        return None

    def _vehicle_gate() -> bool | None:
        if veh_verdict == "GOOD_BET":
            return True
        if veh_verdict == "BAD_BET":
            return False
        if (veh_verdict == "WATCH" and tier == "b_plus" and veh_score_total is not None
                and veh_score_total >= B_PLUS_MIN_VEHICLE_SCORE):
            return True
        return None

    gates: dict[str, bool | None] = {
        "day_regime": _day_regime_gate(),
        "vehicle": _vehicle_gate(),
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
    if acct_veto and acct_ok is False:
        veto_reasons.append(acct_veto)
    # Carry forward any veto reasons the upstream trigger already recorded (deduped, order-preserving).
    for r in (dctx.get("veto_reasons") or []):
        if r and r not in veto_reasons:
            veto_reasons.append(str(r))

    # GREEN-DAY PRESERVATION (post-scalp lockout): once the day's journal shows a completed
    # profitable trade, this gate VETOES — a later "valid setup" must not hand the banked green
    # back (the live failure this guards: green scalp, then a same-day re-entry scratch). The
    # override is only ARMED when it is explicit (`allow_reentry_after_green: true` on the trigger
    # or broker snapshot) AND buying power comfortably covers the re-entry — the contract's
    # estimated cost, floored at GREEN_REENTRY_MIN_BP. No journal context supplied skips the check
    # (the caller is responsible for feeding the day's journal on the live path).
    preservation = None
    if journal_events is not None:
        from data.odte_journal import green_day_preservation
        preservation = green_day_preservation(journal_events, now=current_now)
    green_locked = bool(preservation and preservation.get("locked"))
    reentry_override = (trigger.get("allow_reentry_after_green") is True
                        or broker.get("allow_reentry_after_green") is True)
    bp = _buying_power(broker)
    # BP-scaled arming: the override needs buying power at a multiple of the contract's OWN cost.
    # Unknown cost fails closed — a re-entry whose size cannot be estimated stays locked.
    reentry_cost = _contract_cost(vehicle_score)
    reentry_armed = bool(reentry_override and bp is not None and reentry_cost is not None
                         and bp >= GREEN_REENTRY_MIN_BP_MULTIPLE * reentry_cost)
    if green_locked and not reentry_armed:
        veto_reasons.append(GREEN_REENTRY_BP_VETO if reentry_override else GREEN_LOCKOUT_VETO)

    # DAILY TRADE BUDGET + COOLDOWN (2026-08-02 retune): a hard per-ET-day entry cap and a minimum
    # gap after every completed trade. Deterministic from the same journal the green lockout reads.
    budget = None
    if journal_events is not None:
        from data.odte_journal import daily_trade_budget
        budget = daily_trade_budget(journal_events, now=current_now)
        if budget.get("exhausted"):
            veto_reasons.append(DAILY_BUDGET_VETO)
        elif budget.get("cooldown_active"):
            veto_reasons.append(COOLDOWN_VETO)

    confirmation_states = {name: confirmations.get(name) for name in REQUIRED_CONFIRMATIONS}
    confirmations_ok = all(confirmation_states[name] is True for name in REQUIRED_CONFIRMATIONS)
    core_gates_ready = (not scan_only and not restricted and not veto_reasons
                        and all(gates.get(g) is True for g in req))
    # Final live chain/spread/budget confirmation is part of EVERY otherwise-executable gate—not a
    # later advisory note. Incomplete upstream gates remain observe/deny rather than being mislabeled
    # as a final-confirmation veto.
    if core_gates_ready and not confirmations_ok:
        for name, state in confirmation_states.items():
            if state is not True:
                suffix = "failed" if state is False else "missing"
                veto_reasons.append(f"final_confirmation_{name}_{suffix}")

    reason_codes: list[str] = []
    for name in req:
        state = gates.get(name)
        reason_codes.append(f"{name}:{'ok' if state is True else 'fail' if state is False else 'unknown'}")
    if acct_veto and acct_ok is None:
        reason_codes.append(acct_veto)
    for name, state in confirmation_states.items():
        reason_codes.append(
            f"{name}:{'ok' if state is True else 'fail' if state is False else 'unknown'}"
        )
    if confirmed_candidate_transition:
        reason_codes.append("confirmed_candidate_transition")
    elif base_scan_only:
        reason_codes.append("scan_only_inherited")
        if promotion_requested:
            # Deprecated bare promotion: refused, and the fail-closed reason names the fix.
            reason_codes.append(EXECUTION_LEASE_REQUIRED)
    reason_codes.extend(transition_reasons)
    if green_locked:
        reason_codes.append("green_reentry_override_armed" if reentry_armed
                            else "green_day_preservation_locked")

    # HARD conservative gate: execution_allowed only when nothing blocks it and ALL required gates
    # are explicitly True. Missing inputs (None) fail closed.
    execution_allowed = (not scan_only and not restricted and not veto_reasons
                         and confirmations_ok
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

    next_action, next_command = _next_step(intent, execution_allowed, gates, veto_reasons, req)

    stamp = current_now.isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "symbol": sym,
        "direction": direction,
        "candidate_fingerprint": computed_candidate_fingerprint,
        "option_id": candidate.get("option_id") or candidate.get("id") or candidate.get("occ_symbol"),
        "expiration_date": candidate.get("expiration_date") or candidate.get("expiration"),
        "strike_price": _num(candidate.get("strike_price") or candidate.get("strike")),
        "option_type": str(candidate.get("option_type") or candidate.get("type") or "").lower() or None,
        "decision": intent,
        "intent": intent,
        "reason_codes": reason_codes,
        "required_gates": list(req),
        "gates": gates,
        "failing_gates": [g for g in req if gates.get(g) is False],
        "unknown_gates": [g for g in req if gates.get(g) is None],
        "next_action": next_action,
        "next_command": next_command,
        "veto_reasons": veto_reasons,
        "required_confirmations": list(REQUIRED_CONFIRMATIONS),
        "confirmations": confirmation_states,
        "confirmation_needed": not confirmations_ok,
        "thesis": thesis_block,
        "confidence": confidence,
        "observed_market_context": dctx.get("observed_market_context"),
        "social_context": dctx.get("social_context"),
        "gamma_context": gamma_ctx,
        "scan_only": scan_only,
        # Bare promotion is dead: this key is kept for downstream compatibility but is ALWAYS
        # False now — a scan-tier record can only become executable through an execution lease.
        "promoted_to_execution": False,
        "promotion_requested": promotion_requested,
        "confirmed_candidate_transition": confirmed_candidate_transition,
        "execution_allowed": execution_allowed,
        "green_day_preservation": preservation,
        "allow_reentry_after_green": reentry_armed,
        "tier": tier,
        "sizing_tier": ("half" if tier == "b_plus" else "full"),
        "daily_trade_budget": budget,
        "places_orders": False,
        "basis": ("offline entry-gate decision: day_regime + vehicle + directional_thesis + account "
                  "gates; records intent only, places NO orders"),
    }


def run_entry_gate(trigger_json: str | None = None, trigger_path: str | None = None,
                   candidate_json: str | None = None, candidate_path: str | None = None,
                   candidate_decision_json: str | None = None,
                   candidate_decision_path: str | None = None,
                   day_score_json: str | None = None, day_score_path: str | None = None,
                   vehicle_score_json: str | None = None, vehicle_score_path: str | None = None,
                   gamma_json: str | None = None, gamma_path: str | None = None,
                   broker_json: str | None = None, broker_path: str | None = None,
                   confirmations_json: str | None = None,
                   confirmations_path: str | None = None,
                   scan_only: bool | None = None, promote_to_execution: bool = False,
                   journal_path: str | None = None,
                   out_dir: str | None = None, write: bool = False,
                   now: datetime | None = None) -> dict:
    """Load the (optional) input artifacts and build the entry-gate decision. No orders/broker/network.

    `scan_only=None` (the default) INHERITS scan_only from the trigger/candidate; pass True/False to
    state it explicitly. `promote_to_execution=True` is DEPRECATED and fail-closed: it can no longer
    make the record executable — the response carries the `execution_lease_required` reason code and
    points at `odte-execution-authorize`. `journal_path` feeds the day's decision
    journal into the GREEN-DAY PRESERVATION check (post-scalp lockout) — the CLI passes the
    canonical journal by default; a missing/empty journal simply leaves the lockout disengaged."""
    journal_events = None
    if journal_path:
        from data.odte_journal import read_events
        journal_events = read_events(journal_path)
    payload = build_entry_gate_decision(
        trigger=_load_json(trigger_path, trigger_json) or None,
        candidate=_load_json(candidate_path, candidate_json) or None,
        candidate_decision=_load_json(candidate_decision_path, candidate_decision_json) or None,
        day_score=_load_json(day_score_path, day_score_json) or None,
        vehicle_score=_load_json(vehicle_score_path, vehicle_score_json) or None,
        gamma_map=_load_json(gamma_path, gamma_json) or None,
        broker_snapshot=_load_json(broker_path, broker_json) or None,
        confirmations=_load_json(confirmations_path, confirmations_json) or None,
        scan_only=scan_only,
        promote_to_execution=promote_to_execution,
        journal_events=journal_events,
        now=now,
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
             f"Next: **{p.get('next_command')}** — {p.get('next_action')}  ",
             f"Basis: {p.get('basis')}", ""]
    if p.get("veto_reasons"):
        lines += ["## Veto reasons", *[f"- {r}" for r in p["veto_reasons"]], ""]
    lines += ["## Reason codes", *[f"- {r}" for r in (p.get("reason_codes") or [])]]
    if p.get("required_confirmations"):
        lines += ["", "## Required live confirmations (manager must still verify)",
                  *[f"- {c}" for c in p["required_confirmations"]]]
    return "\n".join(lines)
