"""0DTE armed intents — the slow-lane/fast-lane contract. PURE decision logic, no IO.

The two-lane architecture (2026-08-04): the LLM (slow lane) reasons in minutes and ARMS a
machine-readable intent — exact contract, trigger predicates, sizing hints, exit plan. The
deterministic daemon (fast lane) evaluates armed intents against the live tape every few seconds
and, on fire, runs the full conversion chain (odte-convert internals) in-process. This module owns
the intent schema contract: `validate_armed_intent` (fail-closed, reason-coded — an invalid intent
is refused at ARM time, never silently at fire time) and `evaluate_armed_intent` (the per-tick
trigger evaluator). Neither touches a file, a broker, or a clock it wasn't handed.

An intent is AUTHORIZATION TO EVALUATE, not to trade: every deterministic gate (candidate watch,
entry gate, lease, pre-place guard) still applies at fire time. `sizing_hints` are hints — the
execution lease remains the only authority on quantity/price/debit ceilings.

----------------------------------------------------------------------------------------------------
armed_intents.json  =  {"schema_version": 1, "intents": [intent, ...]}
----------------------------------------------------------------------------------------------------
intent fields (see validate_armed_intent for what is required):
  intent_id:     unique id, e.g. "spy-c-756-20260805-a1b2c3"
  armed_at/by:   ISO-8601 UTC + author label (audit only, never authority)
  expires_at:    required hard ceiling — same ET day as armed_at, at/before 15:30 ET
  status:        armed | fired | expired | disarmed (+ status_history appended by writers)
  symbol:        EXECUTABLE_UNIVERSE only; restricted underlyings refused at ARM time
  direction:     bullish | bearish
  contract:      EXACT vehicle lock: option_id, chain_symbol, expiration_date, strike_price,
                 option_type
  trigger:       ALL PRESENT sub-blocks must pass —
    level_acceptance: {symbol?, side: above|below, level, buffer?, n_consecutive_checks,
                       min_seconds_between_checks, stale_after_seconds?}
                      buffer None => max(0.03, level*2e-4); stale default 3x min_seconds
    confirmations:    {min_confirmers, max_dissenters}   (_confirmation_counts semantics)
    vixy_condition:   weak | firming | any
    wall:             {level, side}                       (level-acceptance primitive, no counter)
    not_before_et / not_after_et: "HH:MM" ET fences
  sizing_hints:  {max_debit?, max_limit_price?, tier_floor?} — HINTS; lease is authoritative
  exit_plan:     copied into active_trade.json on fill (odte_position schema; omit unused stops,
                 never 0.0)
  notes:         free text, never parsed

Per-intent runtime state (armed_intent_state.json, caller-persisted, keyed by intent_id):
  {count, last_true_ts, last_check_ts, last_result, fired_at}
The N-consecutive counter is TIMESTAMPED so a daemon restart neither resets a fresh count nor
resumes a stale one: a gap since last_check_ts beyond stale_after_seconds zeroes the count.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from data.odte_candidate_watch import (
    _confirmation_counts,
    _num,
    _parse_ts,
    _spot,
    _vixy_firming,
    _vixy_weak,
)
from data.odte_config import EXECUTABLE_UNIVERSE

_ET = ZoneInfo("America/New_York")

SCHEMA_VERSION = 1
ARMED_INTENTS_FILENAME = "armed_intents.json"
INTENT_STATE_FILENAME = "armed_intent_state.json"

STATUS_ARMED = "armed"
INTENT_STATUSES = (STATUS_ARMED, "fired", "expired", "disarmed")

# Hard ceiling for expires_at, same ET day: no armed intent survives into the flat_before window.
INTENT_EXPIRY_CEILING_ET = (15, 30)

_DIRECTIONS = ("bullish", "bearish")
_SIDES = ("above", "below")
_VIXY_CONDITIONS = ("weak", "firming", "any")
_CONTRACT_REQUIRED = ("option_id", "chain_symbol", "expiration_date", "strike_price",
                      "option_type")
_TIERS = ("b_plus", "full", "a_plus")

# Consecutive-check defaults (an intent's level_acceptance block may override each).
DEFAULT_CONSECUTIVE_CHECKS = 2
DEFAULT_MIN_SECONDS_BETWEEN_CHECKS = 3.0
STALE_MULTIPLE = 3.0            # stale_after default = STALE_MULTIPLE * min_seconds_between_checks


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _parse_hm(value: Any) -> tuple[int, int] | None:
    try:
        h, m = str(value).split(":")[:2]
        return (int(h), int(m))
    except (TypeError, ValueError, AttributeError):
        return None


def level_buffer(level: float, buffer: float | None) -> float:
    """Acceptance buffer: explicit when supplied, else the `_accepted_wall` shape —
    max(3 cents, 2bp of the level)."""
    if buffer is not None and buffer >= 0:
        return float(buffer)
    return max(0.03, level * 0.0002)


def _level_accepted(market: dict, symbol: str | None, side: str, level: float,
                    buffer: float | None = None) -> bool | None:
    """Instantaneous level acceptance: spot beyond level +/- buffer on the required side.
    None (not False) when the tape has no spot — missing tape is indeterminate, never a pass."""
    spot = _spot(market, symbol)
    if spot is None:
        return None
    buf = level_buffer(level, buffer)
    if side == "above":
        return spot >= level + buf
    return spot <= level - buf


# ── validation (ARM time, fail closed) ──────────────────────────────────────────────────────────

def validate_armed_intent(intent: dict) -> list[str]:
    """Reason-coded validation. Empty list == valid. Every refusal names its field so the slow
    lane can fix and re-arm; an intent that would be ambiguous at fire time is refused HERE."""
    from data.social_sentiment import is_restricted_underlying

    out: list[str] = []
    intent = _dict(intent)
    if not str(intent.get("intent_id") or "").strip():
        out.append("intent_id_missing")
    if intent.get("schema_version") not in (None, SCHEMA_VERSION):
        out.append("schema_version_unsupported")
    status = str(intent.get("status") or STATUS_ARMED).lower()
    if status not in INTENT_STATUSES:
        out.append("status_invalid")

    symbol = str(intent.get("symbol") or "").upper()
    if not symbol:
        out.append("symbol_missing")
    elif symbol not in EXECUTABLE_UNIVERSE:
        out.append("symbol_not_executable")
    if symbol and is_restricted_underlying(symbol):
        out.append("symbol_restricted")
    if str(intent.get("direction") or "").lower() not in _DIRECTIONS:
        out.append("direction_invalid")

    contract = _dict(intent.get("contract"))
    for field in _CONTRACT_REQUIRED:
        if contract.get(field) in (None, ""):
            out.append(f"contract_{field}_missing")
    chain = str(contract.get("chain_symbol") or "").upper()
    if chain and is_restricted_underlying(chain):
        out.append("contract_symbol_restricted")
    if symbol and chain and chain != symbol:
        out.append("contract_symbol_mismatch")

    armed_at = _parse_ts(intent.get("armed_at"))
    if armed_at is None:
        out.append("armed_at_missing_or_unparseable")
    if not str(intent.get("armed_by") or "").strip():
        out.append("armed_by_missing")
    expires_at = _parse_ts(intent.get("expires_at"))
    if expires_at is None:
        out.append("expires_at_missing_or_unparseable")
    elif armed_at is not None:
        if expires_at <= armed_at:
            out.append("expires_at_not_after_armed_at")
        armed_et = armed_at.astimezone(_ET)
        exp_et = expires_at.astimezone(_ET)
        ceiling = armed_et.replace(hour=INTENT_EXPIRY_CEILING_ET[0],
                                   minute=INTENT_EXPIRY_CEILING_ET[1], second=0, microsecond=0)
        if exp_et.date() != armed_et.date() or exp_et > ceiling:
            out.append("expires_at_beyond_session_ceiling")

    trigger = _dict(intent.get("trigger"))
    la = _dict(trigger.get("level_acceptance"))
    conf = _dict(trigger.get("confirmations"))
    vixy = trigger.get("vixy_condition")
    wall = _dict(trigger.get("wall"))
    if not trigger or not (la or conf or vixy or wall):
        out.append("trigger_empty")
    if la:
        if _num(la.get("level")) is None:
            out.append("level_acceptance_level_missing")
        if str(la.get("side") or "").lower() not in _SIDES:
            out.append("level_acceptance_side_invalid")
        n = _num(la.get("n_consecutive_checks"))
        if n is not None and n < 1:
            out.append("level_acceptance_checks_invalid")
        gap = _num(la.get("min_seconds_between_checks"))
        if gap is not None and gap < 0:
            out.append("level_acceptance_gap_invalid")
    if conf and _num(conf.get("min_confirmers")) is None:
        out.append("confirmations_min_confirmers_missing")
    if vixy is not None and str(vixy).lower() not in _VIXY_CONDITIONS:
        out.append("vixy_condition_invalid")
    if wall:
        if _num(wall.get("level")) is None:
            out.append("wall_level_missing")
        if str(wall.get("side") or "").lower() not in _SIDES:
            out.append("wall_side_invalid")
    for fence in ("not_before_et", "not_after_et"):
        if trigger.get(fence) is not None and _parse_hm(trigger.get(fence)) is None:
            out.append(f"{fence}_unparseable")

    hints = _dict(intent.get("sizing_hints"))
    tier_floor = hints.get("tier_floor")
    if tier_floor is not None and str(tier_floor).lower() not in _TIERS:
        out.append("tier_floor_invalid")
    return out


# ── fire-time evaluation (per tick) ─────────────────────────────────────────────────────────────

def _update_counter(state: dict, instant: bool | None, la: dict, now: datetime) -> dict:
    """Timestamped N-consecutive counter. Purely from timestamps, so a restart neither resets a
    fresh count nor resumes a stale one: a gap since last_check_ts beyond stale_after_seconds
    zeroes the count BEFORE this observation is applied; a definitive False zeroes it outright;
    a True only increments when min_seconds_between_checks has elapsed since the last counted
    True (faster polls neither double-count nor reset). An indeterminate read (no spot) keeps
    the count but never advances it."""
    gap = _num(la.get("min_seconds_between_checks"))
    gap = DEFAULT_MIN_SECONDS_BETWEEN_CHECKS if gap is None else float(gap)
    stale_after = _num(la.get("stale_after_seconds"))
    stale_after = (STALE_MULTIPLE * gap if stale_after is None else float(stale_after))

    count = int(_num(state.get("count")) or 0)
    last_true = _parse_ts(state.get("last_true_ts"))
    last_check = _parse_ts(state.get("last_check_ts"))

    if last_check is not None and (now - last_check).total_seconds() > stale_after:
        count, last_true = 0, None
    if instant is False:
        count, last_true = 0, None
    elif instant is True:
        if last_true is None or (now - last_true).total_seconds() >= gap:
            count += 1
            last_true = now

    return {"count": count,
            "last_true_ts": (last_true.isoformat(timespec="seconds") if last_true else None),
            "last_check_ts": now.isoformat(timespec="seconds"),
            "last_result": instant}


def evaluate_armed_intent(intent: dict, market: dict, state: dict | None,
                          now: datetime) -> dict:
    """One tick of trigger evaluation. Returns {fires, checks, consecutive_count, state, reasons}.

    `state` is this intent's persisted counter state (caller-owned; pass {} on first sight) and
    comes back updated in the result for the caller to persist — the evaluator is pure. ALL
    present trigger sub-blocks must pass for `fires`; reasons name every failing check. The level
    counter tracks the tape on every tick regardless of the other checks, so a fire is never
    delayed by a VIXY wobble that resolved before the level finished confirming."""
    intent = _dict(intent)
    market = _dict(market)
    state = dict(state or {})
    trigger = _dict(intent.get("trigger"))
    symbol = str(intent.get("symbol") or "").upper() or None
    direction = str(intent.get("direction") or "").lower()

    checks: dict[str, Any] = {}
    reasons: list[str] = []

    status = str(intent.get("status") or "").lower()
    if status != STATUS_ARMED:
        reasons.append(f"intent_not_armed:{status or 'missing'}")
    expires_at = _parse_ts(intent.get("expires_at"))
    if expires_at is None:
        reasons.append("expires_at_missing")
    elif now >= expires_at:
        reasons.append("intent_expired")

    et_now = now.astimezone(_ET)
    cur = (et_now.hour, et_now.minute)
    before = _parse_hm(trigger.get("not_before_et"))
    after = _parse_hm(trigger.get("not_after_et"))
    if before and cur < before:
        reasons.append("before_not_before_et")
    if after and cur >= after:
        reasons.append("after_not_after_et")
    checks["time_fence"] = not any(r.endswith("_et") for r in reasons)

    la = _dict(trigger.get("level_acceptance"))
    if la:
        level = _num(la.get("level"))
        side = str(la.get("side") or "").lower()
        la_symbol = str(la.get("symbol") or "").upper() or symbol
        instant = (_level_accepted(market, la_symbol, side, level, _num(la.get("buffer")))
                   if level is not None and side in _SIDES else None)
        state = _update_counter(state, instant, la, now)
        need = int(_num(la.get("n_consecutive_checks")) or DEFAULT_CONSECUTIVE_CHECKS)
        level_ok = state["count"] >= need
        checks["level_acceptance"] = {"instant": instant, "count": state["count"],
                                      "needed": need, "ok": level_ok}
        if not level_ok:
            reasons.append(f"level_acceptance_pending:{state['count']}/{need}")
    else:
        state = {**state, "last_check_ts": now.isoformat(timespec="seconds")}

    conf = _dict(trigger.get("confirmations"))
    if conf:
        n, confirmers, dissenters = _confirmation_counts(market, direction)
        need_conf = int(_num(conf.get("min_confirmers")) or 0)
        max_diss = int(_num(conf.get("max_dissenters")) or 0)
        conf_ok = n >= need_conf and len(dissenters) <= max_diss
        checks["confirmations"] = {"confirmers": confirmers, "dissenters": dissenters,
                                   "needed": need_conf, "max_dissenters": max_diss, "ok": conf_ok}
        if not conf_ok:
            reasons.append(f"confirmations_short:{n}/{need_conf}"
                           f"+{len(dissenters)}d/{max_diss}d")

    vixy = str(trigger.get("vixy_condition") or "any").lower()
    if vixy != "any":
        vixy_ok = _vixy_weak(market) if vixy == "weak" else _vixy_firming(market)
        checks["vixy_condition"] = {"required": vixy, "ok": vixy_ok}
        if not vixy_ok:
            reasons.append(f"vixy_not_{vixy}")

    wall = _dict(trigger.get("wall"))
    if wall:
        w_level = _num(wall.get("level"))
        w_side = str(wall.get("side") or "").lower()
        wall_ok = (_level_accepted(market, symbol, w_side, w_level, _num(wall.get("buffer")))
                   if w_level is not None and w_side in _SIDES else None)
        checks["wall"] = {"level": w_level, "side": w_side, "ok": wall_ok}
        if wall_ok is not True:
            reasons.append("wall_not_accepted")

    fires = not reasons
    return {"fires": fires,
            "intent_id": intent.get("intent_id"),
            "checks": checks,
            "consecutive_count": int(_num(state.get("count")) or 0),
            "state": state,
            "reasons": reasons}
