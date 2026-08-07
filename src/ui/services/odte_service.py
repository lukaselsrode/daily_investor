"""
ui/services/odte_service.py — the single seam between the 0DTE UI and src/data.

Everything the 0DTE pages render comes from here: the live cockpit payload, the conversion funnel,
the trade ledger, the lease/incident timeline, the latency chain, and the day packets. Components
render — they do not join events, coalesce legacy key spellings, or decide what counts as a trade.
That work is done once, here, on top of the aggregators that already exist in data.odte_journal
(summarize, weekly_telemetry, funnel_counts, daily_trade_budget, green_day_preservation,
execution_safety_lockout, green_day_winning_tier) and data.odte_loop_status.run_loop_status.

DECISION-ONLY, like the rest of the 0DTE app: reads the local store under data/odte/, never places
an order, never calls a broker or an LLM.

Caching
-------
`_cached(key, ttl, producer)` memoizes in st.session_state when a Streamlit runtime is present and
in a module dict otherwise, so every function here stays importable and testable headless. Live
views use a short TTL (30 s — the controller ticks every 5 min, but a stale cockpit is worse than a
re-read); derived/heavy views use 300-600 s. The journal cache key carries the file's mtime+size,
so an append invalidates it immediately regardless of TTL.

Eras
----
The data model changed twice in ways no chart may paper over:
  * 2026-07-24 — day packets restart after the 07-03..07-23 gap
  * 2026-08-03 — lease-bound fills, rail-named exits, graded postmortems ("modern")
and day-packet trades.jsonl changed meaning on 2026-08-05 (polls+vetoes → lifecycle only), which
is why nothing here ever charts len(trades.jsonl).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# Era boundaries — components caption charts with these instead of plotting phantom zeros.
ERA_PACKET_RESTART = date(2026, 7, 24)
ERA_MODERN = date(2026, 8, 3)
ERA_MODERN_NOTE = "lease-bound fills, rail-named exits and graded postmortems begin 2026-08-03"

# Preserved from the retired odte_dashboard so every 0DTE page agrees on decision colouring.
DECISION_COLOR = {
    "enter": "#2e7d32", "open": "#2e7d32", "allow": "#2e7d32", "issue": "#2e7d32",
    "hold": "#1565c0", "observe": "#1565c0", "keep_watching": "#1565c0",
    "veto": "#c62828", "deny": "#c62828", "no_trade": "#c62828", "skip": "#ef6c00",
    "exit": "#6a1b9a", "take_profit": "#2e7d32",
}

TIER_COLOR = {"a_plus": "#2e7d32", "full": "#1565c0", "b_plus": "#ef6c00"}

_MEM_CACHE: dict = {}


def _cached(key, ttl: float, producer):
    """Memoize `producer()` under `key` for `ttl` seconds. Streamlit session first, module dict
    otherwise (headless import, tests, CLI). Never raises out of the cache layer itself."""
    import time
    now = time.monotonic()
    store = _MEM_CACHE
    try:
        import streamlit as st
        # Only touch session_state inside a real script run — reading it headless works but logs a
        # "missing ScriptRunContext" warning per access, which would drown a CLI/test run.
        if st.runtime.exists():
            store = st.session_state.setdefault("_odte_service_cache", {})
    except Exception:
        pass
    hit = store.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    value = producer()
    store[key] = (now + ttl, value)
    return value


def _journal_path() -> str:
    from ui.utils import ODTE_DATA_DIR
    return str(ODTE_DATA_DIR / "decision_journal.jsonl")


def _stamp(path: str) -> tuple:
    """(mtime, size) so an appended journal invalidates the cache before its TTL expires."""
    try:
        import os
        st_ = os.stat(path)
        return (st_.st_mtime, st_.st_size)
    except OSError:
        return (0.0, 0)


def _num(v):
    try:
        if v is None or isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(v):
    if not v:
        return None
    try:
        raw = str(v).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _et_date(v):
    from data.odte_strategy_policy import ET
    dt = _parse_ts(v)
    return dt.astimezone(ET).date() if dt else None


# --- raw events ------------------------------------------------------------------------------

def load_events(journal_path: str | None = None) -> list[dict]:
    """Every journal event, newest last. Uses the canonical reader (skips malformed lines,
    never raises). Cached on (path, mtime, size) so an append is picked up immediately."""
    path = journal_path or _journal_path()
    return _cached(("events", path, _stamp(path)), 30.0, lambda: _read(path))


def _read(path: str) -> list[dict]:
    from data.odte_journal import read_events
    try:
        return read_events(journal_path=path)
    except Exception:
        return []


def summary(journal_path: str | None = None) -> dict:
    """Deterministic all-time metrics (summarize): trades, hit rate, realized P/L, MFE capture,
    by-mode / by-time-bucket rollups, process quality, pnl_sequence."""
    path = journal_path or _journal_path()
    from data.odte_journal import summarize
    return _cached(("summary", path, _stamp(path)), 300.0,
                   lambda: summarize(load_events(path)))


# --- live cockpit ----------------------------------------------------------------------------

def cockpit_state(live_mode: bool = False) -> dict:
    """The full loop-status payload: state/posture/loop_stage, artifacts + their ages and TTLs,
    the live execution lease with seconds_remaining, weekly telemetry, and live_rails (tier debit
    ceilings, chase band, lease TTLs, green-reentry block). Fail-soft: returns {"error": ...} rather
    than raising into the page."""
    return _cached(("cockpit", live_mode), 30.0, lambda: _cockpit(live_mode))


def _cockpit(live_mode: bool) -> dict:
    from ui.utils import ODTE_DATA_DIR
    try:
        from data.odte_loop_status import run_loop_status
        return run_loop_status(state_dir=str(ODTE_DATA_DIR), live_mode=live_mode)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def budget_now(now: datetime | None = None) -> dict:
    """Today's trade budget with the A+ uncapped exception, the green-day preservation lock, and
    which tier is currently the day's winner (what a re-entry would have to beat)."""
    path = _journal_path()
    return _cached(("budget", path, _stamp(path), now), 30.0, lambda: _budget(path, now))


def _budget(path: str, now: datetime | None) -> dict:
    from data.odte_journal import (
        daily_trade_budget,
        green_day_preservation,
        green_day_winning_tier,
    )
    events = load_events(path)
    out = dict(daily_trade_budget(events, now=now))
    out["green_day"] = green_day_preservation(events, now=now)
    out["winning_tier"] = green_day_winning_tier(events, now=now)
    return out


def safety_state(now: datetime | None = None) -> dict:
    """Execution-safety lockout: whether new entries are barred, the incidents behind it, and the
    human adjudications that cleared them (with their reasons — an adjudication is only ever
    human-authored, so the prose is the audit trail)."""
    path = _journal_path()
    from data.odte_journal import execution_safety_lockout
    return _cached(("safety", path, _stamp(path), now), 30.0,
                   lambda: execution_safety_lockout(load_events(path), now=now))


# --- conversion funnel -----------------------------------------------------------------------

def funnel(days: int = 10, now: datetime | None = None) -> dict:
    """Per-ET-day conversion funnel plus the aggregate over the window.

    Returns {"rows": [...], "total": {...}, "pareto": [(stage:reason, n)], "by_stage": {...}}.
    Rows only cover ET weekdays that actually have events, so a chart never invents flat zeros for
    a weekend. Counts come from data.odte_journal.funnel_counts — the same tally weekly_telemetry
    uses, so this can never disagree with the cockpit's weekly numbers."""
    path = _journal_path()
    return _cached(("funnel", path, _stamp(path), days, now), 300.0,
                   lambda: _funnel(path, days, now))


def _funnel(path: str, days: int, now: datetime | None) -> dict:
    from collections import Counter

    from data.odte_journal import funnel_counts
    events = load_events(path)
    end = (now or datetime.now(timezone.utc)).astimezone(_ET()).date()
    start = end - timedelta(days=max(1, days) - 1)
    present = {d for d in (_et_date(e.get("ts")) for e in events) if d and start <= d <= end}
    rows = [funnel_counts(events, start=d, end=d) for d in sorted(present)]
    total = funnel_counts(events, start=start, end=end)
    pareto = Counter()
    for r in rows:
        for reason, n in r["top_refusal_reasons"]:
            pareto[reason] += n
    return {"rows": rows, "total": total, "pareto": pareto.most_common(15),
            "by_stage": total["refusals_by_stage"], "start": start, "end": end}


def _ET():
    from data.odte_strategy_policy import ET
    return ET


def orb_near_misses(days: int = 10, now: datetime | None = None) -> dict:
    """Evidence for the deferred ORB-breakout-gate decision.

    The confirmation clause is `above VWAP AND orb_state == "above" AND >=2 confirmers AND VIXY
    agrees` (odte_candidate_watch). A NEAR MISS is an evaluation that satisfied everything EXCEPT
    the ORB clause — i.e. exactly the setups the ORB requirement is costing. Counting them requires
    the per-tick `candidate_evaluation` events (first written 2026-08-06); before that the checks
    were discarded, so an empty result here means "not collected yet", never "none happened".

    Returns {"collecting": bool, "evaluated": n, "confirmed": n, "near_miss": n, "samples": [...],
    "by_day": {date: {...}}}.
    """
    path = _journal_path()
    return _cached(("orb", path, _stamp(path), days, now), 300.0,
                   lambda: _orb(path, days, now))


def _orb(path: str, days: int, now: datetime | None) -> dict:
    end = (now or datetime.now(timezone.utc)).astimezone(_ET()).date()
    start = end - timedelta(days=max(1, days) - 1)
    evals = [e for e in load_events(path)
             if e.get("event_type") == "candidate_evaluation"
             and (_et_date(e.get("ts")) or start) >= start]
    by_day: dict = {}
    samples: list[dict] = []
    evaluated = confirmed = near_miss = 0
    for e in evals:
        checks = e.get("checks") if isinstance(e.get("checks"), dict) else {}
        if not checks:
            continue
        evaluated += 1
        d = _et_date(e.get("ts"))
        slot = by_day.setdefault(d, {"evaluated": 0, "confirmed": 0, "near_miss": 0})
        slot["evaluated"] += 1
        is_confirm = str(e.get("decision") or "").upper() == "CONFIRM_ENTRY"
        if is_confirm:
            confirmed += 1
            slot["confirmed"] += 1
            continue
        # Everything the confirmation needs except the ORB state.
        direction = str(e.get("direction") or "").lower()
        bullish = direction != "bearish"
        vwap_ok = checks.get("underlying_above_vwap") is (True if bullish else False)
        vol_ok = bool(checks.get("vixy_weak") if bullish else checks.get("vixy_firming"))
        # BREADTH-ERA AWARE (2026-08-07). The confirmation clause became a graded breadth score
        # that day; measuring "everything except ORB" against the retired `confirmations >= 2`
        # binary reports near_miss 0 forever, because the very tapes this panel exists to count
        # are the ones carrying ONE full confirmer plus halves. Read the score the gate actually
        # used, with its own required level, and fall back to the old count only for events that
        # predate the field.
        score = _num(checks.get("breadth_score"))
        if score is not None:
            confirmers_ok = score >= (_num(checks.get("breadth_required")) or score + 1)
        else:
            confirmers_ok = (_num(checks.get("confirmations")) or 0) >= 2
        orb = str(checks.get("underlying_orb_state") or "")
        orb_ok = orb == ("above" if bullish else "below")
        if vwap_ok and vol_ok and confirmers_ok and not orb_ok:
            near_miss += 1
            slot["near_miss"] += 1
            if len(samples) < 25:
                samples.append({"ts": e.get("ts"), "underlying": e.get("underlying"),
                                "direction": direction, "orb_state": orb,
                                "confirmations": checks.get("confirmations"),
                                "confirmers": checks.get("confirmers"),
                                "half_confirmers": checks.get("half_confirmers"),
                                "breadth_score": checks.get("breadth_score"),
                                "breadth_required": checks.get("breadth_required"),
                                "dissenters": checks.get("dissenters"),
                                "tier": checks.get("tier"),
                                "minutes_to_close": checks.get("minutes_to_close")})
    return {"collecting": evaluated == 0, "evaluated": evaluated, "confirmed": confirmed,
            "near_miss": near_miss, "samples": samples,
            "by_day": {k: v for k, v in sorted(by_day.items()) if k}}


# --- trade ledger ----------------------------------------------------------------------------

def _fill_types() -> tuple[tuple, tuple]:
    """Live vocabularies from the journal module — never re-listed here, or a new exit verb would
    silently stop showing up in the ledger."""
    from data.odte_journal import ENTRY_FILL_EVENTS, EXIT_FILL_EVENTS
    return tuple(ENTRY_FILL_EVENTS), tuple(EXIT_FILL_EVENTS)


_ENTRY_TYPES, _EXIT_TYPES = _fill_types()


def trade_ledger(journal_path: str | None = None) -> list[dict]:
    """One row per executed trade, fully attributed.

    Joins entry fill ⋈ lease (on execution_lease_id) ⋈ gate decision (on candidate_fingerprint)
    ⋈ exit ⋈ postmortem, keyed on trade_id and falling back to option_id when a lane omitted it
    (the order guard journals order_filled without one). Realized P/L goes through
    odte_journal._realized, which coalesces the six legacy spellings — never re-derived here.
    Newest first."""
    path = journal_path or _journal_path()
    return _cached(("ledger", path, _stamp(path)), 300.0, lambda: _ledger(path))


def _trade_key(e: dict) -> str | None:
    tid = e.get("trade_id")
    if tid:
        return f"tid:{tid}"
    oid = e.get("option_id")
    return f"oid:{oid}" if oid else None


def _ledger(path: str) -> list[dict]:
    from data.odte_journal import _realized
    events = load_events(path)

    # option_id -> trade_id, so a lane that omitted trade_id still lands on the right trade.
    oid_to_tid: dict = {}
    for e in events:
        if e.get("trade_id") and e.get("option_id"):
            oid_to_tid.setdefault(str(e["option_id"]), f"tid:{e['trade_id']}")

    def key(e):
        k = _trade_key(e)
        if k and k.startswith("oid:"):
            return oid_to_tid.get(str(e.get("option_id")), k)
        return k

    leases = {str(e.get("lease_id")): e for e in events
              if e.get("event_type") == "execution_lease_issued" and e.get("lease_id")}
    gates = {str(e.get("candidate_fingerprint")): e for e in events
             if e.get("event_type") == "entry_decision" and e.get("candidate_fingerprint")}

    grouped: dict = {}
    for e in events:
        et = e.get("event_type")
        if et not in (*_ENTRY_TYPES, *_EXIT_TYPES, "postmortem"):
            continue
        k = key(e)
        if not k:
            continue
        grouped.setdefault(k, []).append(e)

    rows: list[dict] = []
    for k, evs in grouped.items():
        evs = sorted(evs, key=lambda e: (_parse_ts(e.get("ts")) or datetime.min.replace(
            tzinfo=timezone.utc)))
        entry = next((e for e in evs if e.get("event_type") in _ENTRY_TYPES), None)
        # Prefer exit_fill: the same exit is also journaled as order_closed, but only exit_fill
        # carries the fill price, the best bid seen, and the net-of-fees estimate.
        exit_ = (next((e for e in reversed(evs) if e.get("event_type") == "exit_fill"), None)
                 or next((e for e in reversed(evs) if e.get("event_type") in _EXIT_TYPES), None))
        pm = next((e for e in reversed(evs) if e.get("event_type") == "postmortem"), None)
        if entry is None and exit_ is None:
            continue
        anchor = entry or exit_ or {}
        lease = leases.get(str((entry or {}).get("execution_lease_id") or "")) or {}
        fp = (entry or {}).get("candidate_fingerprint") or lease.get("candidate_fingerprint")
        gate = gates.get(str(fp or "")) or {}
        pm_entry = pm.get("entry") if isinstance((pm or {}).get("entry"), dict) else {}
        exc = pm.get("excursion") if isinstance((pm or {}).get("excursion"), dict) else {}
        pnl_block = pm.get("pnl") if isinstance((pm or {}).get("pnl"), dict) else {}

        realized = None
        for e in reversed(evs):
            realized = _realized(e)
            if realized is not None:
                break
        entry_ts, exit_ts = _parse_ts((entry or {}).get("ts")), _parse_ts((exit_ or {}).get("ts"))
        rows.append({
            "key": k,
            "trade_id": anchor.get("trade_id"),
            "trade_date": anchor.get("trade_date") or (
                entry_ts.astimezone(_ET()).date().isoformat() if entry_ts else None),
            "underlying": (anchor.get("underlying") or anchor.get("symbol")
                           or lease.get("underlying")),
            "option_id": anchor.get("option_id"),
            "direction": anchor.get("direction") or gate.get("direction"),
            "mode": anchor.get("mode"),
            # tier: the postmortem states it; else the lease minted it; else the gate assigned it
            "tier": (pm_entry.get("tier") or lease.get("tier") or gate.get("tier")),
            "sizing_tier": gate.get("sizing_tier"),
            "lease_id": (entry or {}).get("execution_lease_id") or lease.get("lease_id"),
            "lease_valid_at_fill": (entry or {}).get("lease_valid_at_fill"),
            "entry_ts": (entry or {}).get("ts"),
            "exit_ts": (exit_ or {}).get("ts"),
            "entry_price": _num((entry or {}).get("fill_price")) or _num(pm_entry.get("fill_price")),
            "exit_price": _num((exit_ or {}).get("fill_price") or (exit_ or {}).get("exit_price")),
            "quantity": _num((entry or {}).get("quantity")) or _num(pm_entry.get("quantity")),
            "rail_fired": (exit_ or {}).get("rail_fired") or (
                (pm or {}).get("exit") or {}).get("rail_fired"),
            "realized_pnl": realized,
            "net_pnl": (_num((exit_ or {}).get("estimated_net_pnl"))
                        or _num(pnl_block.get("estimated_net"))),
            "best_seen_bid": _num((exit_ or {}).get("best_seen_bid")),
            "mfe_capture_pct": _num(exc.get("mfe_capture_pct")),
            "mfe_pct": _num(exc.get("mfe_pct")),
            "mae_pct": _num(exc.get("mae_pct")),
            "process_quality": (pm or {}).get("process_quality"),
            "outcome_quality": (pm or {}).get("outcome_quality"),
            "failure_layer": (pm or {}).get("failure_layer"),
            "rule_violations": list(((pm or {}).get("process_review") or {}).get(
                "rule_violations") or []),
            "held_minutes": (round((exit_ts - entry_ts).total_seconds() / 60.0, 1)
                             if entry_ts and exit_ts else None),
            "has_postmortem": pm is not None,
            "modern": bool(exc or (entry or {}).get("execution_lease_id")),
        })
    rows.sort(key=lambda r: (r["entry_ts"] or r["exit_ts"] or ""), reverse=True)
    return rows


def intratrade_series(row: dict, journal_path: str | None = None) -> list[dict]:
    """The management_check pnl_pct series for one ledger row.

    Joined by TIMESTAMP WINDOW, not by key: modern management_check events carry trade_id=None
    (see gap G7). Only one 0DTE position is ever open at a time, so the entry→exit bracket is
    unambiguous — but this is an inference and the UI says so.
    """
    path = journal_path or _journal_path()
    start, end = _parse_ts(row.get("entry_ts")), _parse_ts(row.get("exit_ts"))
    if not start:
        return []
    end = end or (start + timedelta(hours=8))
    out = []
    for e in load_events(path):
        if e.get("event_type") != "management_check":
            continue
        ts = _parse_ts(e.get("ts"))
        if ts is None or not (start <= ts <= end):
            continue
        pct = _num(e.get("pnl_pct"))
        if pct is None:
            continue
        out.append({"ts": e.get("ts"), "minutes": round((ts - start).total_seconds() / 60.0, 2),
                    "pnl_pct": pct, "decision": e.get("decision")})
    return out


def latency_rows(journal_path: str | None = None) -> list[dict]:
    """Per-trade execution latency legs, in seconds, from journal timestamps.

    lease→submit and submit→fill come from the order-guard fields on the fill event; lease→fill is
    always available once a fill carries execution_lease_id (2026-08-03+). This is a per-trade
    series, NOT a percentile distribution — at n=3 a p95 would be fiction.
    """
    path = journal_path or _journal_path()
    return _cached(("latency", path, _stamp(path)), 300.0, lambda: _latency(path))


def _latency(path: str) -> list[dict]:
    events = load_events(path)
    leases = {str(e.get("lease_id")): e for e in events
              if e.get("event_type") == "execution_lease_issued" and e.get("lease_id")}
    consumed = {str(e.get("lease_id")): e for e in events
                if e.get("event_type") == "execution_lease_consumed" and e.get("lease_id")}

    def secs(a, b):
        return round((b - a).total_seconds(), 2) if (a and b) else None

    # ONE ROW PER LEASE. The same entry is journaled by two lanes — the controller's entry_fill
    # (which carries trade_id) and the order guard's order_filled (which carries submitted_at /
    # filled_at). Keyed per event they would plot as two trades; merged per lease they are one
    # trade with a complete leg breakdown.
    merged: dict = {}
    for e in events:
        if e.get("event_type") not in _ENTRY_TYPES:
            continue
        lid = str(e.get("execution_lease_id") or e.get("lease_id") or "")
        lease = leases.get(lid)
        if not lease:
            continue
        t_lease = _parse_ts(lease.get("ts") or lease.get("issued_at"))
        t_fill = _parse_ts(e.get("filled_at") or e.get("ts"))
        t_submit = _parse_ts(e.get("submitted_at"))
        t_consume = _parse_ts((consumed.get(lid) or {}).get("ts"))
        row = merged.setdefault(lid, {
            "trade_id": None, "underlying": None, "trade_date": None, "lease_id": lid,
            "ts": lease.get("ts"), "lease_to_submit": None, "submit_to_fill": None,
            "lease_to_fill": None, "lease_to_consume": None,
        })
        for field, value in (("trade_id", e.get("trade_id")),
                             ("underlying", e.get("underlying") or e.get("symbol")),
                             ("trade_date", e.get("trade_date")),
                             ("lease_to_submit", secs(t_lease, t_submit)),
                             ("submit_to_fill", secs(t_submit, t_fill)),
                             ("lease_to_fill", secs(t_lease, t_fill)),
                             ("lease_to_consume", secs(t_lease, t_consume))):
            if row[field] is None and value is not None:
                row[field] = value
    rows = sorted(merged.values(), key=lambda r: (r["ts"] or ""))
    return rows


def conversion_sla_seconds() -> float:
    """The confirm→conversion SLA the latency chart draws its reference line at (live constant)."""
    from data.odte_config import CONFIRM_CONVERSION_SLA_SECONDS
    return float(CONFIRM_CONVERSION_SLA_SECONDS)


def lease_ttl_seconds() -> tuple[float, float]:
    """(default TTL, hard cap). The hard cap is an incident invariant, not a tunable."""
    from data.odte_execution_policy import DEFAULT_LEASE_TTL_SECONDS, MAX_LEASE_TTL_SECONDS
    return float(DEFAULT_LEASE_TTL_SECONDS), float(MAX_LEASE_TTL_SECONDS)


# --- rails / leases / incidents ---------------------------------------------------------------

def lease_timeline(days: int = 10, now: datetime | None = None) -> dict:
    """Lease lifecycle spans plus the safety events around them.

    Each lease is issue → (consumed | filled | expired). "expired" is INFERRED — the atomic convert
    path did not record expires_at before 2026-08-06, so a lease with no consume and no fill is
    presumed expired rather than read as expired. Also returns rail-fired counts, the pre-order-hook
    blocks (which are refusals, never incidents), and incidents with their adjudications.
    """
    path = _journal_path()
    return _cached(("rails", path, _stamp(path), days, now), 300.0,
                   lambda: _rails(path, days, now))


def _rails(path: str, days: int, now: datetime | None) -> dict:
    from collections import Counter
    events = load_events(path)
    end = (now or datetime.now(timezone.utc)).astimezone(_ET()).date()
    start = end - timedelta(days=max(1, days) - 1)

    def in_window(e):
        d = _et_date(e.get("ts"))
        return d is not None and start <= d <= end

    consumed = {str(e.get("lease_id")): e for e in events
                if e.get("event_type") == "execution_lease_consumed" and e.get("lease_id")}
    filled = {}
    for e in events:
        if e.get("event_type") in _ENTRY_TYPES and e.get("execution_lease_id"):
            filled[str(e["execution_lease_id"])] = e
    incidents = [e for e in events if e.get("event_type") == "execution_safety_incident"]
    adjudications = [e for e in events
                     if e.get("event_type") == "execution_safety_incident_adjudicated"]
    inc_by_lease: dict = {}
    for e in incidents:
        inc_by_lease.setdefault(str(e.get("lease_id") or ""), []).append(e)

    spans = []
    for e in events:
        if e.get("event_type") != "execution_lease_issued" or not in_window(e):
            continue
        lid = str(e.get("lease_id") or "")
        t0 = _parse_ts(e.get("ts") or e.get("issued_at"))
        fill, cons = filled.get(lid), consumed.get(lid)
        t_end = _parse_ts((fill or {}).get("ts")) or _parse_ts((cons or {}).get("ts"))
        if e.get("authorized") is False:
            outcome = "denied"
        elif fill is not None:
            outcome = "filled"
        elif cons is not None:
            outcome = "consumed_no_fill"
        else:
            outcome = "expired_inferred"
        expires = _parse_ts(e.get("expires_at"))
        spans.append({
            "lease_id": lid, "ts": e.get("ts"), "underlying": e.get("underlying") or e.get("symbol"),
            "tier": e.get("tier"), "authorized": e.get("authorized"), "outcome": outcome,
            "seconds": round((t_end - t0).total_seconds(), 1) if (t0 and t_end) else None,
            "ttl_seconds": round((expires - t0).total_seconds(), 1) if (t0 and expires) else None,
            "expires_at": e.get("expires_at"),
            "max_limit_price": _num(e.get("max_limit_price")),
            "max_debit": _num(e.get("max_debit")),
            "reason_codes": list(e.get("reason_codes") or []),
            "incidents": len(inc_by_lease.get(lid, [])),
        })
    spans.sort(key=lambda s: (s["ts"] or ""), reverse=True)

    # Rails counted PER TRADE, not per event: an exit is journaled by both lanes (exit_fill and
    # order_closed), so counting raw events doubles every rail. The ledger is already deduped.
    rails = Counter(str(r["rail_fired"]) for r in trade_ledger(path)
                    if r.get("rail_fired") and r.get("trade_date")
                    and start.isoformat() <= str(r["trade_date"]) <= end.isoformat())
    hook_blocks = [e for e in events
                   if e.get("event_type") == "no_trade_decision"
                   and str(e.get("stage") or "") == "pre_order_hook"]
    adj_by_incident = {}
    for a in adjudications:
        for anchor in (a.get("incident_event_id"), a.get("incident_seq")):
            if anchor is not None:
                adj_by_incident[str(anchor)] = a
    return {
        "spans": spans,
        "rail_counts": rails.most_common(),
        "hook_blocks": hook_blocks,
        "incidents": [{
            "ts": e.get("ts"), "event_id": e.get("event_id"), "seq": e.get("seq"),
            "underlying": e.get("underlying") or e.get("symbol"),
            "guard_state": e.get("guard_state"), "stage": e.get("stage"),
            "reason_codes": list(e.get("reason_codes") or []),
            "adjudication": (adj_by_incident.get(str(e.get("event_id")))
                             or adj_by_incident.get(str(e.get("seq")))),
        } for e in sorted(incidents, key=lambda x: str(x.get("ts") or ""), reverse=True)],
        "start": start, "end": end,
    }


def shadow_state() -> dict | None:
    """Fast-lane shadow-vs-live divergence, or None while the daemon has never run.

    The report contract (both_fired / shadow_only / live_only / exit_divergences / counts / clean)
    is already built and tested; this lights up the day data/odte/shadow/ appears."""
    return _cached(("shadow",), 60.0, _shadow)


def _shadow() -> dict | None:
    from ui.utils import ODTE_DATA_DIR
    shadow_dir = ODTE_DATA_DIR / "shadow"
    shadow_journal = shadow_dir / "decision_journal.jsonl"
    if not shadow_journal.exists():
        return None
    try:
        from data.odte_shadow_report import build_shadow_report
        return build_shadow_report(load_events(), _read(str(shadow_journal)))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def fast_lane_stage() -> dict | None:
    """Contents of fast_lane_stage.json (shadow / exits_live / entries_live), or None."""
    from ui.utils import load_odte_json
    return load_odte_json("fast_lane_stage.json")


# --- day packets -----------------------------------------------------------------------------

def day_index() -> list[str]:
    """Available day-packet dates (data/odte/days/<date>/), newest first."""
    return _cached(("day_index",), 600.0, _day_index)


def _day_index() -> list[str]:
    from ui.utils import ODTE_DATA_DIR
    root = ODTE_DATA_DIR / "days"
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if p.is_dir()), reverse=True)


def day_packet(trade_date: str) -> dict:
    """One day's packet: the five streams, the postmortem, and a tape-density verdict.

    `postmortem` is the HUMAN-authoritative file; when a `.generated.md` twin exists and differs,
    `postmortem_generated_differs` is True so the page can say a human edited it. `tape_density`
    warns when there are too few market snapshots to draw a line — the honest answer is markers on
    a scatter, never interpolation. `archive_snapshots` reports how many higher-resolution
    controller snapshots sit in data/odte/archive/ for that date but are not in the packet (G5).
    """
    return _cached(("packet", trade_date), 600.0, lambda: _packet(trade_date))


_STREAMS = ("market_snapshots", "candidates", "vehicle_scores", "trades", "controller_events")


def _packet(trade_date: str) -> dict:
    import json

    from ui.utils import ODTE_DATA_DIR
    root = ODTE_DATA_DIR / "days" / str(trade_date)
    out: dict = {"trade_date": trade_date, "exists": root.is_dir()}
    if not root.is_dir():
        return out
    for name in _STREAMS:
        rows = []
        p = root / f"{name}.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        out[name] = rows
    pm, gen = root / "postmortem.md", root / "postmortem.generated.md"
    out["postmortem"] = pm.read_text() if pm.exists() else None
    out["postmortem_generated_differs"] = bool(
        gen.exists() and pm.exists() and gen.read_text() != pm.read_text())
    n_snap = len(out.get("market_snapshots") or [])
    out["tape_density"] = ("dense" if n_snap >= 20 else "sparse" if n_snap >= 10 else "markers")
    out["archive_snapshots"] = _archive_snapshot_count(trade_date)
    # trades.jsonl means DIFFERENT things across this date (2026-08-05: lifecycle-only).
    out["trades_semantics"] = ("lifecycle" if str(trade_date) >= "2026-08-05" else "mixed")
    return out


def _archive_snapshot_count(trade_date: str) -> int:
    """Higher-resolution controller snapshots archived for this date but absent from the packet."""
    from ui.utils import ODTE_DATA_DIR
    archive = ODTE_DATA_DIR / "archive"
    if not archive.is_dir():
        return 0
    stamp = str(trade_date).replace("-", "")
    return sum(1 for d in archive.iterdir() if d.is_dir()
               for _ in d.glob(f"controller_market_{stamp}*.json"))


def tape_frames(packet: dict, symbol: str) -> list[dict]:
    """Plottable tape rows for one symbol out of a day packet's market snapshots.

    Reads the per-symbol nested block (SPY/QQQ/IWM/VIXY — the shape written since 2026-07-27) and
    falls back to the older flat `{sym}_vwap` / `{sym}_orb_state` keys so pre-07-27 days still
    render. Rows without a price are dropped rather than zero-filled.
    """
    sym = str(symbol).upper()
    rows = []
    for s in (packet or {}).get("market_snapshots") or []:
        block = s.get(sym) if isinstance(s.get(sym), dict) else {}
        low = sym.lower()
        last = _num(block.get("last")) or _num(s.get(f"{low}_price")) or _num(s.get(f"{low}_last"))
        if last is None:
            continue
        rows.append({
            "ts": s.get("as_of") or s.get("generated_at") or s.get("ts"),
            "last": last,
            "vwap": _num(block.get("vwap")) or _num(s.get(f"{low}_vwap")),
            "orb_high": _num(block.get("orb_high")) or _num(s.get(f"{low}_orb_high")),
            "orb_low": _num(block.get("orb_low")) or _num(s.get(f"{low}_orb_low")),
            "orb_state": block.get("orb_state") or s.get(f"{low}_orb_state"),
            "above_vwap": block.get("above_vwap") if "above_vwap" in block
            else s.get(f"{low}_above_vwap"),
        })
    rows.sort(key=lambda r: str(r["ts"] or ""))
    return rows


def day_markers(trade_date: str) -> dict:
    """Entry/exit/refusal/lease markers for a replay chart, from the journal (not the packet, so a
    day with a thin packet still gets its markers)."""
    path = _journal_path()
    return _cached(("markers", path, _stamp(path), trade_date), 600.0,
                   lambda: _markers(path, trade_date))


def _markers(path: str, trade_date: str) -> dict:
    want = str(trade_date)
    out: dict = {"entries": [], "exits": [], "refusals": [], "leases": [], "evaluations": []}
    for e in load_events(path):
        d = _et_date(e.get("ts"))
        if (e.get("trade_date") or (d.isoformat() if d else None)) != want:
            continue
        et = e.get("event_type")
        base = {"ts": e.get("ts"), "underlying": e.get("underlying") or e.get("symbol")}
        if et in _ENTRY_TYPES:
            out["entries"].append({**base, "price": _num(e.get("fill_price")),
                                   "option_id": e.get("option_id")})
        elif et in _EXIT_TYPES:
            out["exits"].append({**base, "price": _num(e.get("fill_price")
                                                      or e.get("exit_price")),
                                 "option_id": e.get("option_id"),
                                 "rail_fired": e.get("rail_fired")})
        elif et == "no_trade_decision":
            out["refusals"].append({**base, "stage": e.get("stage"),
                                    "reason_codes": list(e.get("reason_codes") or [])})
        elif et == "execution_lease_issued":
            out["leases"].append({**base, "lease_id": e.get("lease_id"),
                                  "authorized": e.get("authorized"), "tier": e.get("tier")})
        elif et == "candidate_evaluation":
            checks = e.get("checks") if isinstance(e.get("checks"), dict) else {}
            out["evaluations"].append({**base, "decision": e.get("decision"),
                                       "orb_state": checks.get("orb_state")
                                       or checks.get("underlying_orb_state")})
    # One marker per fill, not per journaling lane: the guard's order_filled lands seconds after
    # the controller's entry_fill for the same contract and would draw a phantom second trade.
    out["entries"] = _dedupe_fills(out["entries"])
    out["exits"] = _dedupe_fills(out["exits"])
    return out


def _dedupe_fills(rows: list[dict]) -> list[dict]:
    """Collapse fills sharing an option_id inside the journal's same-fill window."""
    from data.odte_journal import _SAME_FILL_WINDOW_MINUTES
    kept: list[dict] = []
    for r in sorted(rows, key=lambda x: str(x.get("ts") or "")):
        ts, oid = _parse_ts(r.get("ts")), r.get("option_id")
        dup = any(
            k.get("option_id") and k["option_id"] == oid and ts and _parse_ts(k.get("ts"))
            and abs((ts - _parse_ts(k["ts"])).total_seconds()) <= _SAME_FILL_WINDOW_MINUTES * 60
            for k in kept)
        if not dup:
            kept.append(r)
    return kept


# --- day-score headroom ------------------------------------------------------------------------

def day_score_series(days: int = 10, now: datetime | None = None) -> list[dict]:
    """Day-score events with their headroom telemetry over the window.

    `max_possible_score` vs the GOOD_DAY threshold answers a question the verdict alone cannot:
    whether a GOOD_DAY was even REACHABLE, or whether a component never arrived. First-party
    journaling started 2026-08-06, so earlier points are sparse EOD-ingest artifacts."""
    path = _journal_path()
    return _cached(("dayscore", path, _stamp(path), days, now), 300.0,
                   lambda: _day_scores(path, days, now))


def _day_scores(path: str, days: int, now: datetime | None) -> list[dict]:
    end = (now or datetime.now(timezone.utc)).astimezone(_ET()).date()
    start = end - timedelta(days=max(1, days) - 1)
    rows = []
    for e in load_events(path):
        if e.get("event_type") != "day_score":
            continue
        d = _et_date(e.get("ts"))
        if d is None or not (start <= d <= end):
            continue
        comps = e.get("components") if isinstance(e.get("components"), dict) else {}
        rows.append({"ts": e.get("ts"), "date": d.isoformat(), "score": _num(e.get("score")),
                     "verdict": e.get("verdict"), "components": comps,
                     "components_supplied": e.get("components_supplied"),
                     "components_missing": list(e.get("components_missing") or []),
                     "max_possible_score": _num(e.get("max_possible_score"))})
    rows.sort(key=lambda r: str(r["ts"] or ""))
    return rows


def good_day_min_score() -> float:
    """The GOOD_DAY threshold the headroom chart draws as a reference line (live constant)."""
    from data.odte_day_score import GOOD_DAY_MIN_SCORE
    return float(GOOD_DAY_MIN_SCORE)
