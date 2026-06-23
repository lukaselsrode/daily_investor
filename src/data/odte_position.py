"""0DTE live-position watchdog — BROKER-AWARE inputs, DECISION-ONLY output.

This is the discipline layer for an ALREADY-OPEN 0DTE option. Unlike the social watchdog
(``odte_watchdog.py``, which never touches a position), this module reasons about a live trade —
but it still places NO orders and makes NO broker/LLM calls. It is pure decision logic:

    active trade plan (~/0dte/active_trade.json)  +  live snapshot (fed by Hermes from MCP)
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
  take_profit_pct:  scale/exit take-profit trigger (default 0.35; sane range 0.35-0.50)
  strong_exit_pct:  strong/default full-exit trigger (default 0.60)
  bid_floor:        per-share bid at/under which the option is treated near-worthless (default 0.05)
  thesis:           { underlying_stop, spy_stop, qqq_stop, vix_stop, vixy_stop }  (any subset)
                    A "stop" is the level that KILLS the thesis when crossed AGAINST the position.
                    Direction is inferred from option_type (see _thesis_breaches).
  time_rules:       { tighten_after: "HH:MM" ET, flat_before: "HH:MM" ET }

live snapshot schema (caller-supplied; real broker/market values only)
  option_mark:  current premium per share         (P/L vs entry_price)
  pnl_pct:      OR provide P/L fraction directly (0.42 == +42%); wins over option_mark
  option_bid:   current per-share bid              (BID_FLOOR)
  underlying_last, spy_last, qqq_last, vix, vixy:  thesis levels
  now_et:       ISO timestamp (else uses `now`/wall clock) for time rules
  monitoring_ok: explicit False => MONITORING_DEGRADED
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

DEFAULT_STATE_DIR = os.path.expanduser("~/0dte")
DEFAULT_PLAN_FILENAME = "active_trade.json"
STATE_FILENAME = "position_state.json"
DECISION_FILENAME = "position_decision.json"

STATE_VERSION = 1

DEFAULT_TAKE_PROFIT_PCT = 0.35   # +35% — lower bound of the 35-50% scalp take-profit band
DEFAULT_STRONG_EXIT_PCT = 0.60   # +60% — strong / default full-exit trigger
DEFAULT_BID_FLOOR = 0.05         # per-share bid at/under which an option is treated near-worthless

# Primary-decision priority (most urgent first). HOLD is the implicit default.
_PRIORITY = ["RESTRICTED", "THESIS_DEAD", "BID_FLOOR", "TIME_RISK", "TAKE_PROFIT",
             "MONITORING_DEGRADED"]

_INACTIVE_STATUS = {"closed", "exited", "flat", "done"}


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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
    for key, snap_key, label in (("underlying_stop", "underlying_last", "underlying"),
                                 ("spy_stop", "spy_last", "SPY"),
                                 ("qqq_stop", "qqq_last", "QQQ")):
        lvl, val = _num(thesis.get(key)), _num(snapshot.get(snap_key))
        if lvl is None or val is None:
            continue
        if bullish and val <= lvl:
            out.append(f"{label} {val:g} <= stop {lvl:g} (support lost)")
        elif not bullish and val >= lvl:
            out.append(f"{label} {val:g} >= stop {lvl:g} (resistance reclaimed)")
    for key, snap_key, label in (("vix_stop", "vix", "VIX"), ("vixy_stop", "vixy", "VIXY")):
        lvl, val = _num(thesis.get(key)), _num(snapshot.get(snap_key))
        if lvl is None or val is None:
            continue
        if bullish and val >= lvl:
            out.append(f"{label} {val:g} >= stop {lvl:g} (vol spiked against calls)")
        elif not bullish and val <= lvl:
            out.append(f"{label} {val:g} <= stop {lvl:g} (vol faded against puts)")
    return out


def _time_risk(time_rules: dict, snapshot: dict, now: datetime | None) -> dict | None:
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

    tp = float(_num(plan.get("take_profit_pct")) or DEFAULT_TAKE_PROFIT_PCT)
    strong = float(_num(plan.get("strong_exit_pct")) or DEFAULT_STRONG_EXIT_PCT)
    bid_floor = float(_num(plan.get("bid_floor")) if plan.get("bid_floor") is not None
                      else DEFAULT_BID_FLOOR)

    # 1) Take-profit. Strong (+60%) => full exit, all modes. Scale (+35%) => exit a single-contract
    #    scalp, else sell partial and keep a runner.
    if pnl_pct is not None:
        if pnl_pct >= strong:
            triggers.append({"type": "TAKE_PROFIT", "stage": "strong", "action": "exit_all",
                             "pnl_pct": round(pnl_pct, 4),
                             "detail": f"+{pnl_pct:.0%} >= strong exit {strong:.0%}"})
        elif pnl_pct >= tp:
            action = "exit" if (mode == "scalp" or qty <= 1) else "scale_keep_runner"
            triggers.append({"type": "TAKE_PROFIT", "stage": "scale", "action": action,
                             "pnl_pct": round(pnl_pct, 4),
                             "detail": f"+{pnl_pct:.0%} >= take-profit {tp:.0%}"})

    # 2) Thesis death.
    breaches = _thesis_breaches(option_type, plan.get("thesis") or {}, snapshot)
    if breaches:
        triggers.append({"type": "THESIS_DEAD", "action": "exit",
                         "reasons": breaches, "detail": "; ".join(breaches)})

    # 3) Bid floor — near-worthless / no path.
    if bid is not None and bid <= bid_floor:
        triggers.append({"type": "BID_FLOOR", "action": "exit_or_let_expire", "bid": bid,
                         "detail": f"bid {bid:.2f} <= floor {bid_floor:.2f}"})

    # 4) Time risk.
    t = _time_risk(plan.get("time_rules") or {}, snapshot, now)
    if t:
        triggers.append(t)

    # 5) Monitoring degraded — can't value the live position, or caller flagged the feed bad.
    if snapshot.get("monitoring_ok") is False or not can_value:
        triggers.append({"type": "MONITORING_DEGRADED", "action": "verify_feed",
                         "detail": "cannot value live position (no mark/bid/pnl) or monitoring_ok=false"})

    return {"decision": _primary_decision(triggers), "triggers": triggers, "active": True,
            "pnl_pct": (round(pnl_pct, 4) if pnl_pct is not None else None),
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
    (sdir / STATE_FILENAME).write_text(json.dumps(state, indent=2, default=str))
    (sdir / DECISION_FILENAME).write_text(json.dumps(payload, indent=2, default=str))
    return payload
