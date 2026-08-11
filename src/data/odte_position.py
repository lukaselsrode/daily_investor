"""0DTE live-position watchdog — BROKER-AWARE inputs, DECISION-ONLY output.

This is the discipline layer for an ALREADY-OPEN 0DTE option. Unlike the social watchdog
(``odte_watchdog.py``, which never touches a position), this module reasons about a live trade —
but it still places NO orders and makes NO broker/LLM calls. It is pure decision logic:

    active trade plan (data/odte/active_trade.json)  +  live snapshot (fed by Hermes from MCP)
        -> structured triggers: TAKE_PROFIT / THESIS_DEAD / BID_FLOOR / TIME_RISK /
           MONITORING_DEGRADED / HOLD / NO_POSITION / RESTRICTED

The snapshot is supplied by the caller (Hermes feeds real broker/market values from its MCP tools);
this module NEVER fabricates live broker data. When the snapshot can't value the position it returns
MONITORING_DEGRADED rather than guessing — flying blind on a live option is itself a risk.

Employer/compliance-restricted underlyings (e.g. NVDA) are refused outright: a restricted plan
returns RESTRICTED and no management triggers (defense in depth — such a trade should never exist).

----------------------------------------------------------------------------------------------------
active_trade.json schema (all fields optional unless noted; unknown keys are ignored)
----------------------------------------------------------------------------------------------------
  status:           "open" (default) | "closed"/"exited"/"flat" -> treated as NO_POSITION
  mode:             "scalp" | "trend" | "lotto" | "runner"
  underlying:       e.g. "SPY"            (required for an active plan)
  option_id:        broker option id (opaque; passed through for Hermes)
  option_type:      "call"/"c" | "put"/"p"
  strike, expiration: passed through (not used by the math)
  entry_price:      premium per share at entry (e.g. 1.00)        [for P/L]
  quantity:         contracts (1 => single-contract scalp semantics)
  entry_time:       ISO (passed through)
  take_profit_pct:  scale take-profit trigger (scalp default 0.20; legacy/non-scalp default 0.35)
  strong_exit_pct:  strong-profit trigger (scalp default 0.25; legacy/non-scalp default 0.60)
                    Both accept FRACTION (0.20) or PERCENT (20) — any threshold > 3 is treated as a
                    percent and divided by 100 (the 2026-08-03 live plan carried 20 meaning +20%;
                    read literally that take-profit could never fire).
  profit_rules:     optional mode-aware overrides (any subset; falls back to the mode profile):
                      take_profit_pct, strong_exit_pct,
                      take_profit_action      (qty-agnostic take-profit action override),
                      single_contract_action, multi_contract_action  (scale stage, by quantity),
                      strong_exit_action      (strong-stage action override)
                    TAKE_PROFIT thresholds are mode-aware (scalp +20%/+25%; non-scalp defaults
                    +35%/+60%); the recommended ACTION also differs by mode (decision-only — never
                    an order):
                      scalp  -> single: exit            / multi: scale_keep_runner  / strong: exit_all
                      trend  -> protect_profit (alert, not forced exit)             / strong: trail_or_exit_on_stall
                      lotto  -> hold_but_alert                                      / strong: harvest_optional
                      runner -> trail_runner                                       / strong: trail_runner
                    Unknown/blank mode uses the scalp profile (original behavior).
  bid_floor:        per-share bid at/under which the option is treated near-worthless; falls back to
                    risk_rules.initial_bid_floor, then the 0.05 default (the 2026-08-03 live plan
                    declared only risk_rules.initial_bid_floor — unmapped, its floor never fired)
  thesis:           { underlying_stop, spy_stop, qqq_stop, iwm_stop, vix_stop, vixy_stop } (subset)
                    A "stop" is the level that KILLS the thesis when crossed AGAINST the position.
                    Direction is inferred from option_type (see _thesis_breaches). A level of 0 is
                    treated as ABSENT, never as a live stop (a vix_stop of 0.0 read literally makes
                    "vix >= 0" instantly true for calls).
  bid_memory:       { arm_gain_pct, giveback_fraction } — giveback harvest overrides (also accepted
                    under profit_rules.bid_memory). Once best-seen bid >= entry*(1+arm_gain_pct),
                    BID_MEMORY_PROTECT fires when the live bid gives back giveback_fraction of the
                    best-seen gain. Caller persists management.best_seen_bid between polls (the
                    evaluator is pure and returns the updated value).
  management:       { best_seen_bid } — highest bid seen while managing (caller-persisted)
  time_rules:       { tighten_after: "HH:MM" ET, flat_before: "HH:MM" ET }

live snapshot schema (caller-supplied; real broker/market values only)
  option_mark:  current premium per share         (P/L vs entry_price)
  pnl_pct:      OR provide P/L fraction directly (0.42 == +42%); wins over option_mark
  option_bid:   current per-share bid              (BID_FLOOR)
  underlying_last, spy_last, qqq_last, iwm_last, vix, vixy:  thesis levels
  now_et:       ISO timestamp (else uses `now`/wall clock) for time rules
  monitoring_ok: explicit False => MONITORING_DEGRADED
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from core.paths import ODTE_DATA_DIR, atomic_write_text

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

DEFAULT_STATE_DIR = ODTE_DATA_DIR
DEFAULT_PLAN_FILENAME = "active_trade.json"
STATE_FILENAME = "position_state.json"
DECISION_FILENAME = "position_decision.json"

STATE_VERSION = 1

DEFAULT_SCALP_TAKE_PROFIT_PCT = 0.20   # +20% — first-harvest scalp trigger
DEFAULT_SCALP_STRONG_EXIT_PCT = 0.25   # +25% — stretch/strong scalp trigger
DEFAULT_TAKE_PROFIT_PCT = 0.35         # +35% — legacy/non-scalp take-profit trigger
DEFAULT_STRONG_EXIT_PCT = 0.60         # +60% — legacy/non-scalp strong-profit trigger
DEFAULT_BID_FLOOR = 0.05         # per-share bid at/under which an option is treated near-worthless

# MAX-LOSS BACKSTOP (2026-08-07). THESIS_DEAD and TIME_RISK only exist when the CONTROLLER wrote a
# `thesis` / `time_rules` block into the plan — `_thesis_breaches` returns [] on an absent thesis and
# `_time_risk` returns None on absent rules. So a thin plan had exactly one unconditional exit,
# BID_FLOOR, which on a $1.00 entry does not fire until -95%. The 2026-08-07 IWM trade exited at -16%
# only because the agent happened to write a thesis; had it written none, the position rides to the
# floor. That is a rail whose existence depends on agent prose, which is the failure this whole class
# keeps producing.
#
# Sized from the record, not from taste: across all 150 `management_check` observations in the
# journal the worst open-position excursion is -33.8% (p5 -27.9%, median -5.6%). A -40% backstop has
# therefore NEVER fired — it cuts no winner and changes no historical trade — while replacing the
# -95% cliff. It is a catastrophe backstop, deliberately far outside the working range, NOT a
# strategy stop; tightening it toward the -20%/-30% band would start cutting live trades (20% / 2.7%
# of observations respectively) and must be measured separately if ever wanted.
# Overridable per plan via `max_loss_pct` / `risk_rules.max_loss_pct`.
DEFAULT_MAX_LOSS_PCT = -0.40

# CLOCK BACKSTOP (2026-08-07). Same defect on the time axis, and worse: `_time_risk` returned None
# on an absent `time_rules`, so a plan declaring none had no clock exit whatsoever. A WINNING 0DTE
# position at 15:59 ET read HOLD and rode into expiry — where an out-of-the-money contract is worth
# zero. The max-loss backstop cannot catch that (it is a winner) and neither can BID_FLOOR (the bid
# is healthy). Measured the same way: no exit in the journal has ever occurred after 15:30 ET
# (latest 14:20, n=12), so a 15:45 flatten has never bound a real trade.
# Overridable per plan via `time_rules.flat_before`.
DEFAULT_FLAT_BEFORE_ET = "15:45"

# Bid-memory / giveback protection (the rule that closed the 2026-08-03 winner, prose-only until
# now): once the best-seen bid prints a meaningful gain over entry, never let it round-trip — fire
# a harvest when the live bid has given back a large share of the best-seen gain.
BID_MEMORY_ARM_GAIN_PCT = 0.15       # arm once best_seen_bid >= entry * (1 + this)
BID_MEMORY_GIVEBACK_FRACTION = 0.40  # fire once bid <= entry + (best_seen - entry) * (1 - this)

# Mode-specific profit semantics. SCALP mode harvests earlier at +20% / +25% because tiny-account
# 0DTE impulse gains can fade quickly; the old +35% / +60% defaults remain for trend/lotto/runner
# alerting unless the plan overrides thresholds. The recommended ACTION is mode-aware so a
# single-contract trend/lotto/runner is NOT force-exited like a scalp. `single_contract_action` /
# `multi_contract_action` drive the scale stage (by quantity); `strong_exit_action` drives the strong
# stage. Plans override any of these via plan["profit_rules"] (see _resolve_profit_config). An
# unknown/blank mode falls back to the scalp profile — preserving the original single=exit /
# multi=scale_keep_runner behavior while using the earlier scalp harvest band.
_MODE_PROFIT_PROFILES: dict[str, dict] = {
    "scalp":  {"take_profit_pct": DEFAULT_SCALP_TAKE_PROFIT_PCT,
               "strong_exit_pct": DEFAULT_SCALP_STRONG_EXIT_PCT,
               "single_contract_action": "exit", "multi_contract_action": "scale_keep_runner",
               "strong_exit_action": "exit_all"},
    "trend":  {"take_profit_pct": DEFAULT_TAKE_PROFIT_PCT, "strong_exit_pct": DEFAULT_STRONG_EXIT_PCT,
               "single_contract_action": "protect_profit", "multi_contract_action": "protect_profit",
               "strong_exit_action": "trail_or_exit_on_stall"},
    "lotto":  {"take_profit_pct": DEFAULT_TAKE_PROFIT_PCT, "strong_exit_pct": DEFAULT_STRONG_EXIT_PCT,
               "single_contract_action": "hold_but_alert", "multi_contract_action": "hold_but_alert",
               "strong_exit_action": "harvest_optional"},
    "runner": {"take_profit_pct": DEFAULT_TAKE_PROFIT_PCT, "strong_exit_pct": DEFAULT_STRONG_EXIT_PCT,
               "single_contract_action": "trail_runner", "multi_contract_action": "trail_runner",
               "strong_exit_action": "trail_runner"},
}
_DEFAULT_PROFIT_PROFILE = _MODE_PROFIT_PROFILES["scalp"]   # backward-compatible default

# Primary-decision priority (most urgent first). HOLD is the implicit default. BID_MEMORY_PROTECT
# outranks TAKE_PROFIT (a fading winner is harvested before a plain profit scale) but yields to the
# death/floor/time triggers.
# MAX_LOSS sits BELOW the named reasons deliberately. It is a catch-all backstop, so whenever a
# specific diagnosis also fired — the tape invalidated the thesis, or the option is near-worthless —
# that is the reason the postmortem should attribute the exit to. The backstop only ever surfaces as
# the primary decision when nothing more specific explains the loss, which is precisely the
# thin-plan case it exists for.
_PRIORITY = ["RESTRICTED", "THESIS_DEAD", "BID_FLOOR", "MAX_LOSS", "TIME_RISK",
             "BID_MEMORY_PROTECT", "TAKE_PROFIT", "MONITORING_DEGRADED"]

_INACTIVE_STATUS = {"closed", "exited", "flat", "done"}


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_pct(v: float | None) -> float | None:
    """Normalize a profit threshold to a FRACTION. Values > 3 are percent (20 -> 0.20): no sane
    0DTE take-profit is +300% as a fraction, but 20-meaning-20% is exactly what the 2026-08-03
    live plan shipped (its own take_profit_price proved percent intent)."""
    if v is None:
        return None
    return v / 100.0 if v > 3 else v


def _norm_type(v) -> str:
    s = str(v or "").strip().lower()
    if s in ("call", "c", "calls"):
        return "call"
    if s in ("put", "p", "puts"):
        return "put"
    return ""


def _parse_hm(s) -> tuple[int, int] | None:
    if not s:
        return None
    try:
        h, m = str(s).split(":")[:2]
        return (int(h), int(m))
    except Exception:
        return None


def _now_et(snapshot: dict, now: datetime | None) -> datetime:
    iso = (snapshot or {}).get("now_et")
    dt: datetime | None = None
    if iso:
        try:
            dt = datetime.fromisoformat(str(iso))
        except Exception:
            dt = None
    if dt is None:
        dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)


def _plan_inactive(plan: dict) -> bool:
    if not plan or not plan.get("underlying"):
        return True
    if str(plan.get("status", "open")).strip().lower() in _INACTIVE_STATUS:
        return True
    return plan.get("active") is False


def _compute_pnl_pct(plan: dict, snapshot: dict) -> float | None:
    """Long-option P/L fraction: explicit snapshot.pnl_pct wins, else option_mark vs entry_price."""
    direct = _num(snapshot.get("pnl_pct"))
    if direct is not None:
        return direct
    mark, entry = _num(snapshot.get("option_mark")), _num(plan.get("entry_price"))
    if mark is not None and entry not in (None, 0.0):
        return mark / entry - 1.0
    return None


def _resolve_profit_config(plan: dict, mode: str, qty: int) -> dict:
    """Resolve mode-aware take-profit thresholds + actions, honoring overrides.

    Precedence (most specific wins): plan["profit_rules"][k] -> legacy top-level take_profit_pct/
    strong_exit_pct -> the mode profile (scalp profile for unknown/blank mode). The scale action is
    qty-aware: single_contract_action when qty<=1, else multi_contract_action; a qty-agnostic
    profit_rules.take_profit_action overrides both. strong_exit_action drives the strong stage."""
    profile = _MODE_PROFIT_PROFILES.get(mode, _DEFAULT_PROFIT_PROFILE)
    pr = plan.get("profit_rules") or {}
    tp = (_norm_pct(_num(pr.get("take_profit_pct"))) or _norm_pct(_num(plan.get("take_profit_pct")))
          or profile["take_profit_pct"])
    strong = (_norm_pct(_num(pr.get("strong_exit_pct")))
              or _norm_pct(_num(plan.get("strong_exit_pct")))
              or profile["strong_exit_pct"])
    if qty <= 1:
        scale_action = (pr.get("single_contract_action") or pr.get("take_profit_action")
                        or profile["single_contract_action"])
    else:
        scale_action = (pr.get("multi_contract_action") or pr.get("take_profit_action")
                        or profile["multi_contract_action"])
    strong_action = pr.get("strong_exit_action") or profile["strong_exit_action"]
    return {"tp": float(tp), "strong": float(strong),
            "scale_action": scale_action, "strong_action": strong_action}


def _thesis_breaches(option_type: str, thesis: dict, snapshot: dict) -> list[str]:
    """Thesis-death reasons. A `*_stop` level kills the thesis when crossed AGAINST the position.

    Long CALL (bullish): price stops are SUPPORTS — dead when value <= stop; a VIX/VIXY stop is a
    CEILING — dead when value >= stop (vol spiking against risk-on). Long PUT (bearish) inverts both:
    price stops are RESISTANCES (dead when value >= stop); vol stops are FLOORS (dead when value <=
    stop, i.e. vol faded and risk-on resumed)."""
    if not thesis:
        return []
    bullish = option_type == "call"
    out: list[str] = []
    # A 0 level is ABSENT, never a live stop: writers emit 0.0 for "unused" (the 2026-08-03 plan
    # shipped vix_stop 0.0 — read literally, "vix >= 0" is always true for a call).
    for key, snap_key, label in (("underlying_stop", "underlying_last", "underlying"),
                                 ("spy_stop", "spy_last", "SPY"),
                                 ("qqq_stop", "qqq_last", "QQQ"),
                                 ("iwm_stop", "iwm_last", "IWM")):
        lvl, val = _num(thesis.get(key)), _num(snapshot.get(snap_key))
        if not lvl or val is None:
            continue
        if bullish and val <= lvl:
            out.append(f"{label} {val:g} <= stop {lvl:g} (support lost)")
        elif not bullish and val >= lvl:
            out.append(f"{label} {val:g} >= stop {lvl:g} (resistance reclaimed)")
    for key, snap_key, label in (("vix_stop", "vix", "VIX"), ("vixy_stop", "vixy", "VIXY")):
        lvl, val = _num(thesis.get(key)), _num(snapshot.get(snap_key))
        if not lvl or val is None:
            continue
        if bullish and val >= lvl:
            out.append(f"{label} {val:g} >= stop {lvl:g} (vol spiked against calls)")
        elif not bullish and val <= lvl:
            out.append(f"{label} {val:g} <= stop {lvl:g} (vol faded against puts)")
    return out


def _time_risk(time_rules: dict | None, snapshot: dict, now: datetime | None) -> dict | None:
    """Clock-based exits. An ABSENT `time_rules` key falls back to DEFAULT_FLAT_BEFORE_ET.

    2026-08-07: this returned None whenever `time_rules` was falsy, so a plan that never mentioned
    the key had NO clock exit at all — a WINNING 0DTE position at 15:59 ET read HOLD and rode into
    expiry, where an out-of-the-money contract is worth zero. Neither the max-loss backstop (it is a
    winner) nor BID_FLOOR (the bid is healthy) covers that. The default is measured, not chosen: no
    exit in the journal has ever occurred after 15:30 ET (latest 14:20, n=12), so a 15:45 flatten
    has never bound a real trade.

    ABSENT and EXPLICITLY-EMPTY are treated differently on purpose. `time_rules` missing means the
    author never considered the clock, and gets the backstop. `time_rules: {}` is a deliberate
    declaration that this position has no clock exit, and is honoured — that is the contract the
    existing suite pins by passing `time_rules={}` to isolate the profit and bid-memory axes.
    """
    if time_rules is None:
        time_rules = {"flat_before": DEFAULT_FLAT_BEFORE_ET}
    if not time_rules:
        return None
    et = _now_et(snapshot, now)
    cur = (et.hour, et.minute)
    flat, tighten = _parse_hm(time_rules.get("flat_before")), _parse_hm(time_rules.get("tighten_after"))
    if flat and cur >= flat:
        return {"type": "TIME_RISK", "stage": "flat", "action": "flatten",
                "detail": f"{cur[0]:02d}:{cur[1]:02d} ET >= flat_before {flat[0]:02d}:{flat[1]:02d}"}
    if tighten and cur >= tighten:
        return {"type": "TIME_RISK", "stage": "tighten", "action": "tighten_stops",
                "detail": f"{cur[0]:02d}:{cur[1]:02d} ET >= tighten_after {tighten[0]:02d}:{tighten[1]:02d}"}
    return None


def _primary_decision(triggers: list[dict]) -> str:
    types = {t["type"] for t in triggers}
    for p in _PRIORITY:
        if p in types:
            return p
    return "HOLD"


def evaluate_position(plan: dict | None, snapshot: dict | None,
                      now: datetime | None = None) -> dict:
    """PURE: map an active trade plan + a live snapshot to a decision + structured triggers.

    No file IO, no network, no broker, no LLM — fully unit-testable. Returns a dict with
    ``decision`` (the primary), ``triggers`` (all that fired), ``pnl_pct``, and context fields."""
    plan = plan or {}
    snapshot = snapshot or {}

    if _plan_inactive(plan):
        return {"decision": "NO_POSITION", "triggers": [], "active": False,
                "pnl_pct": None, "underlying": None, "option_id": None}

    underlying = str(plan.get("underlying") or "").upper()
    option_id = plan.get("option_id")
    option_type = _norm_type(plan.get("option_type"))
    mode = str(plan.get("mode") or "").lower()
    qty = int(_num(plan.get("quantity")) or 0)

    # Defense in depth: a restricted underlying should never be open — refuse to manage it.
    from data.social_sentiment import is_restricted_underlying
    if is_restricted_underlying(underlying):
        return {"decision": "RESTRICTED", "active": True, "pnl_pct": None,
                "underlying": underlying, "option_id": option_id, "mode": mode,
                "option_type": option_type,
                "triggers": [{"type": "RESTRICTED", "action": "no_action", "reason": "employer",
                              "detail": f"{underlying} is employer-restricted — never trade/manage."}]}

    triggers: list[dict] = []
    pnl_pct = _compute_pnl_pct(plan, snapshot)
    bid = _num(snapshot.get("option_bid"))
    can_value = pnl_pct is not None or bid is not None

    # bid_floor falls back to risk_rules.initial_bid_floor — live plans declare the floor there
    # (2026-08-03: initial_bid_floor 0.427 never fired because only top-level bid_floor was read).
    _floor = _num(plan.get("bid_floor"))
    if _floor is None:
        _floor = _num((plan.get("risk_rules") or {}).get("initial_bid_floor"))
    bid_floor = float(_floor) if _floor is not None else DEFAULT_BID_FLOOR

    # 0) MAX-LOSS BACKSTOP — unconditional, so it survives a plan that declared no thesis. Mirrors
    #    the bid_floor fallback above, including reading `risk_rules` where live plans put risk keys.
    _mx = _num(plan.get("max_loss_pct"))
    if _mx is None:
        _mx = _num((plan.get("risk_rules") or {}).get("max_loss_pct"))
    max_loss = float(_mx) if _mx is not None else DEFAULT_MAX_LOSS_PCT
    if max_loss > 0:                      # accept 0.40 or -0.40 from a plan; normalise to a loss
        max_loss = -max_loss
    if pnl_pct is not None and pnl_pct <= max_loss:
        triggers.append({"type": "MAX_LOSS", "action": "exit",
                         "reason": "max_loss_backstop",
                         "detail": f"P/L {pnl_pct:.1%} at/through the {max_loss:.0%} backstop "
                                   f"— unconditional, independent of any plan-declared thesis"})

    # 1) Take-profit. Scalp mode uses the earlier +20% / +25% harvest band; non-scalp modes keep the
    #    legacy +35% / +60% alert/trail bands unless overridden. The recommended ACTION is mode-aware
    #    (see _MODE_PROFIT_PROFILES): a single-contract trend/lotto/runner is alerted (protect_profit
    #    / hold_but_alert / trail_runner), NOT force-exited like a scalp. Plans override thresholds /
    #    actions via plan["profit_rules"].
    if pnl_pct is not None:
        pc = _resolve_profit_config(plan, mode, qty)
        label = mode or "default"
        # Float-safe boundary: option marks like 1.20 / 1.00 can compute as 0.19999999999999996.
        eps = 1e-9
        if pnl_pct + eps >= pc["strong"]:
            triggers.append({"type": "TAKE_PROFIT", "stage": "strong", "action": pc["strong_action"],
                             "pnl_pct": round(pnl_pct, 4),
                             "detail": f"+{pnl_pct:.0%} >= strong {pc['strong']:.0%} ({label})"})
        elif pnl_pct + eps >= pc["tp"]:
            triggers.append({"type": "TAKE_PROFIT", "stage": "scale", "action": pc["scale_action"],
                             "pnl_pct": round(pnl_pct, 4),
                             "detail": f"+{pnl_pct:.0%} >= take-profit {pc['tp']:.0%} ({label})"})

    # 2) Bid-memory giveback protection. Once the best-seen bid arms (a meaningful gain over
    #    entry), a fade through the giveback floor harvests instead of round-tripping. best_seen_bid
    #    is caller-persisted (plan["management"]["best_seen_bid"]) and returned updated — this
    #    evaluator stays pure.
    entry = _num(plan.get("entry_price"))
    bm = plan.get("bid_memory") or (plan.get("profit_rules") or {}).get("bid_memory") or {}
    arm_gain = _norm_pct(_num(bm.get("arm_gain_pct"))) or BID_MEMORY_ARM_GAIN_PCT
    giveback = _num(bm.get("giveback_fraction")) or BID_MEMORY_GIVEBACK_FRACTION
    prev_best = _num((plan.get("management") or {}).get("best_seen_bid"))
    best_seen = prev_best
    if bid is not None:
        best_seen = bid if prev_best is None else max(prev_best, bid)
    if (bid is not None and entry and best_seen is not None
            and best_seen >= entry * (1.0 + arm_gain)):
        giveback_floor = entry + (best_seen - entry) * (1.0 - giveback)
        if bid <= giveback_floor + 1e-9:
            triggers.append({"type": "BID_MEMORY_PROTECT", "action": "harvest_now",
                             "bid": bid, "best_seen_bid": best_seen,
                             "giveback_floor": round(giveback_floor, 4),
                             "detail": f"bid {bid:.2f} <= giveback floor {giveback_floor:.2f} "
                                       f"(best {best_seen:.2f}, entry {entry:.2f})"})

    # 3) Thesis death.
    breaches = _thesis_breaches(option_type, plan.get("thesis") or {}, snapshot)
    if breaches:
        triggers.append({"type": "THESIS_DEAD", "action": "exit",
                         "reasons": breaches, "detail": "; ".join(breaches)})

    # 4) Bid floor — near-worthless / no path.
    if bid is not None and bid <= bid_floor:
        triggers.append({"type": "BID_FLOOR", "action": "exit_or_let_expire", "bid": bid,
                         "detail": f"bid {bid:.2f} <= floor {bid_floor:.2f}"})

    # 5) Time risk.
    t = _time_risk(plan.get("time_rules"), snapshot, now)
    if t:
        triggers.append(t)

    # 6) Monitoring degraded — can't value the live position, or caller flagged the feed bad.
    #
    # NAME THE CAUSE (2026-08-11). This used to answer one generic string for three different
    # problems, so a CALLER ERROR read exactly like a bad feed. On 2026-08-11 the controller ran
    # `make odte-position JSON=1` twice with no SNAPSHOT= while a position was live; both returned
    # MONITORING_DEGRADED, which looks like a legitimate verdict, so nothing corrected it — 2 of 26
    # checks that day (7.7%) against a 1.7% historical rate. A bare invocation cannot value a live
    # option, and the payload should say so in the words that fix it.
    if snapshot.get("monitoring_ok") is False:
        triggers.append({"type": "MONITORING_DEGRADED", "action": "verify_feed",
                         "cause": "monitoring_ok_false",
                         "detail": "caller set monitoring_ok=false — the feed is flagged bad"})
    elif not snapshot:
        triggers.append({"type": "MONITORING_DEGRADED", "action": "supply_snapshot",
                         "cause": "no_snapshot_supplied",
                         "detail": "NO SNAPSHOT SUPPLIED — a bare odte-position call cannot value a "
                                   "live position; re-run with SNAPSHOT=<live.json> (or "
                                   "--snapshot-json). This check monitored nothing."})
    elif not can_value:
        triggers.append({"type": "MONITORING_DEGRADED", "action": "verify_feed",
                         "cause": "snapshot_unpriceable",
                         "detail": "snapshot carries no option_mark / option_bid / pnl_pct — "
                                   "cannot value the position without a price"})

    return {"decision": _primary_decision(triggers), "triggers": triggers, "active": True,
            "pnl_pct": (round(pnl_pct, 4) if pnl_pct is not None else None),
            "best_seen_bid": best_seen,
            "underlying": underlying, "option_id": option_id, "mode": mode,
            "option_type": option_type}


def run_position_watchdog(plan_path: str | None = None, snapshot: dict | None = None,
                          snapshot_path: str | None = None,
                          state_dir: str = DEFAULT_STATE_DIR,
                          now: datetime | None = None) -> dict:
    """Read the active trade plan + a caller-supplied snapshot, evaluate, persist, return payload.

    NO broker and NO LLM calls: the snapshot is provided by the caller (Hermes/MCP) — this function
    only reads it. ``payload['alert']`` is True for any actionable decision (not NO_POSITION/HOLD)."""
    from data.odte_watchdog import _read_json  # shared JSON reader (status-aware)

    now = now or datetime.now(timezone.utc)
    sdir = Path(os.path.expanduser(state_dir))
    sdir.mkdir(parents=True, exist_ok=True)
    ppath = Path(os.path.expanduser(plan_path)) if plan_path else sdir / DEFAULT_PLAN_FILENAME

    plan, plan_status = _read_json(ppath)

    snap = snapshot
    snap_status = "inline" if snapshot is not None else "none"
    if snap is None and snapshot_path:
        snap, snap_status = _read_json(Path(os.path.expanduser(snapshot_path)))
    snap = snap or {}

    result = evaluate_position(plan or {}, snap, now=now)
    decision = result["decision"]
    alert = decision not in ("NO_POSITION", "HOLD")

    payload = {
        "ts": now.isoformat(timespec="seconds"),
        "alert": alert,
        "decision": decision,
        "triggers": result["triggers"],
        "pnl_pct": result["pnl_pct"],
        "best_seen_bid": result.get("best_seen_bid"),
        "underlying": result.get("underlying"),
        "option_id": result.get("option_id"),
        "mode": result.get("mode"),
        "plan_status": plan_status,
        "snapshot_status": snap_status,
    }
    state = {
        "version": STATE_VERSION,
        "updated_at": now.isoformat(timespec="seconds"),
        "decision": decision,
        "active": result.get("active", False),
        "pnl_pct": result["pnl_pct"],
        "underlying": result.get("underlying"),
        "plan_status": plan_status,
        "snapshot_status": snap_status,
    }
    # Atomic writes (tmp + os.replace): a crash mid-write must not leave a truncated state/decision
    # file for the next poll to misread.
    decision_text = json.dumps(payload, indent=2, default=str)
    atomic_write_text(sdir / STATE_FILENAME, json.dumps(state, indent=2, default=str))
    atomic_write_text(sdir / DECISION_FILENAME, decision_text)
    _persist_best_seen_bid(ppath, result.get("best_seen_bid"),
                           active=bool(result.get("active")))
    _journal_position_decision(payload, decision_text, sdir / DECISION_FILENAME)
    return payload


def _persist_best_seen_bid(plan_path: Path, best_seen: float | None, *, active: bool) -> None:
    """Write the updated bid-memory peak back into the plan.

    BID_MEMORY_PROTECT's entire state is ``management.best_seen_bid``, and ``evaluate_position``
    is a pure evaluator that RETURNS the updated peak without storing it — so the runner has to.
    Until 2026-08-10 nothing did, and the docstring's "caller-persisted" meant the LLM controller
    was expected to hand-write it between polls. It did not: on the 14:19 SPY 774C trade the plan
    went unwritten across 7+ management checks while the bid moved. The rail that exists to stop a
    winner round-tripping was reading a peak that only advanced when the agent remembered to
    advance it. A rail whose state the agent maintains is not a rail — same lesson as the
    hand-derived stop (2026-08-07), third occurrence.

    Deliberately narrow, because the controller also owns this file (it writes the post-fill plan
    and the thesis):

    * re-reads immediately before writing and merges ONE key, so a concurrent controller write
      loses at most this field instead of the thesis;
    * refuses to act unless the plan already reads ``ok`` — this never creates or repairs a plan;
    * monotonic, so a stale-low evaluation can never walk the recorded peak back down;
    * fail-safe — any error leaves the plan untouched and never changes the decision.
    """
    if not active or best_seen is None:
        return
    try:
        from data.odte_watchdog import _read_json  # shared JSON reader (status-aware)

        fresh, status = _read_json(plan_path)
        if status != "ok" or not isinstance(fresh, dict):
            return
        management = dict(fresh.get("management") or {})
        prior = management.get("best_seen_bid")
        try:
            if prior is not None and float(prior) >= float(best_seen):
                return                                 # the peak only ever ratchets up
        except (TypeError, ValueError):
            pass                                       # unparseable prior: overwrite with a number
        management["best_seen_bid"] = best_seen
        fresh["management"] = management
        atomic_write_text(plan_path, json.dumps(fresh, indent=2, default=str))
    except Exception as exc:                           # never let persistence affect the decision
        logger.debug("bid-memory persistence skipped (%s)", exc)


def _journal_position_decision(payload: dict, decision_text: str, decision_path: Path) -> None:
    """Best-effort: fold the position decision into the standardized decision journal as a
    `management_check`. FULLY fail-safe — any error here is swallowed so it can NEVER change the
    decision, the stdout contract, or crash the poll. NOT an execution authorization: this records a
    monitoring decision on an EXISTING position; `execution_allowed` stays False."""
    try:
        from data.odte_journal import append_decision_journal, event_from_position_decision
        ev = event_from_position_decision(payload)
        ev["ts"] = payload.get("ts")
        # Provenance + content-addressed idempotency: each poll's distinct payload → distinct id;
        # re-journaling the identical decision file dedupes.
        ev["raw_artifact_path"] = str(decision_path)
        ev["raw_artifact_sha"] = hashlib.sha1(decision_text.encode("utf-8")).hexdigest()[:16]
        ev["execution_allowed"] = False
        # Journal co-located with the state files (so a tmp state_dir in tests/dry-runs never touches
        # the real journal; in production decision_path.parent is data/odte/).
        jp = str(decision_path.parent / "decision_journal.jsonl")
        append_decision_journal(ev, source="position", event_type="management_check", journal_path=jp)
    except Exception as exc:        # never let journaling affect the watchdog
        logger.debug("position decision journaling skipped (%s)", exc)
