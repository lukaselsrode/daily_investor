"""Shadow-vs-live divergence report — the fast lane's rollout evidence surface. PURE, no IO
in the core.

While the fast-lane daemon runs in shadow (full compute, structurally unable to place), its
journal (`data/odte/shadow/decision_journal.jsonl`) records what it WOULD have done with
namespaced `shadow_*` event types that no live reader can miscount (`daily_trade_budget` counts
only ENTRY_FILL_EVENTS; shadow uses none of those types). This module joins the two journals
into the adjudication surface the rollout gates require:

  * both_fired      — a live entry and a matching shadow_order_intent: latency delta (negative
                      = the shadow lane was FASTER) + limit-price delta
  * shadow_only     — the shadow lane would have entered where the live lane did not (each one
                      must be human-adjudicated before the next session)
  * live_only       — the live lane entered with NO matching shadow intent (a divergence the
                      shadow lane must explain before it can be trusted with entries)
  * exit_divergences— live closes vs shadow_exit_intent timing/level differences
  * counts          — raw event tallies both sides

Join key: ET day + underlying, preferring exact option_id equality, else a ±5-minute window.

Shadow event vocabulary (written by the daemon, read here):
  shadow_session_start / shadow_session_end / shadow_trigger_fired / shadow_convert_result /
  shadow_order_intent / shadow_exit_intent / shadow_incident
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from data.odte_journal import ENTRY_FILL_EVENTS, EXIT_FILL_EVENTS

_ET = ZoneInfo("America/New_York")

SCHEMA_VERSION = 1
SHADOW_DIRNAME = "shadow"
MATCH_WINDOW_SECONDS = 300.0    # +/- 5 minutes when option_id can't make the join exact

SHADOW_ORDER_INTENT = "shadow_order_intent"
SHADOW_EXIT_INTENT = "shadow_exit_intent"
SHADOW_INCIDENT = "shadow_incident"
SHADOW_SESSION_START = "shadow_session_start"
SHADOW_EVENT_TYPES = (SHADOW_SESSION_START, "shadow_session_end", "shadow_trigger_fired",
                      "shadow_convert_result", SHADOW_ORDER_INTENT, SHADOW_EXIT_INTENT,
                      SHADOW_INCIDENT)

# The live lane's "entered" anchors: the lease issuance is the decision moment; a fill event is
# the executed moment. First one wins per (day, underlying, option_id).
_LIVE_ENTRY_TYPES = ("execution_lease_issued", *ENTRY_FILL_EVENTS)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _parse_ts(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _key_fields(event: dict) -> tuple[str | None, str | None, str | None, datetime | None]:
    ts = _parse_ts(event.get("ts"))
    day = ts.astimezone(_ET).date().isoformat() if ts else None
    underlying = str(event.get("underlying") or event.get("symbol") or "").upper() or None
    option_id = str(event.get("option_id") or "") or None
    return day, underlying, option_id, ts


def _limit_of(event: dict) -> float | None:
    for key in ("limit_price", "price", "max_limit_price", "limit"):
        n = _num(event.get(key))
        if n is not None:
            return n
    return _num((event.get("order") or {}).get("price")) if isinstance(event.get("order"),
                                                                       dict) else None


def _first_per_identity(events: list[dict], types: tuple[str, ...]) -> list[dict]:
    """Earliest event per (day, underlying, option_id) among `types` — one anchor per trade."""
    best: dict[tuple, dict] = {}
    for event in events:
        if event.get("event_type") not in types:
            continue
        day, underlying, option_id, ts = _key_fields(event)
        if ts is None or day is None:
            continue
        key = (day, underlying, option_id)
        if key not in best or ts < _parse_ts(best[key]["ts"]):
            best[key] = event
    return sorted(best.values(), key=lambda e: str(e.get("ts")))


def _match(anchor: dict, candidates: list[dict], used: set[int]) -> dict | None:
    """Best unused candidate: same ET day + underlying; exact option_id wins, else the closest
    inside the +/- window."""
    day_a, und_a, opt_a, ts_a = _key_fields(anchor)
    best_idx, best_score = None, None
    for idx, cand in enumerate(candidates):
        if idx in used:
            continue
        day_c, und_c, opt_c, ts_c = _key_fields(cand)
        if day_c != day_a or und_c != und_a or ts_c is None or ts_a is None:
            continue
        gap = abs((ts_c - ts_a).total_seconds())
        if opt_a and opt_c and opt_a == opt_c:
            score = (0, gap)
        elif gap <= MATCH_WINDOW_SECONDS:
            score = (1, gap)
        else:
            continue
        if best_score is None or score < best_score:
            best_idx, best_score = idx, score
    if best_idx is None:
        return None
    used.add(best_idx)
    return candidates[best_idx]


def build_shadow_report(real_events: list[dict], shadow_events: list[dict],
                        now: datetime | None = None) -> dict:
    """Join the live journal with the shadow journal into the adjudication payload.

    WINDOWED TO THE DAYS THE SHADOW LANE ACTUALLY RAN (2026-08-12). Comparing every live event
    ever journaled against the shadow stream made `clean` unreachable by construction: on the
    daemon's first run this reported 20 `live_only` entries and 17
    `live_exit_without_shadow_intent` divergences going back to 2026-06-24, against 90 seconds of
    shadow data. Those are not divergences — the lane did not exist yet — and they never age out,
    so Gate 1's "`clean: true`" could never have been satisfied.

    A day counts as observed if the shadow journal has ANY event on it (the daemon writes
    `shadow_session_start` at startup, so a session that produced no intent still counts — that is
    the case where a genuine `live_only` IS a real divergence). Live events outside those days are
    excluded and COUNTED in `window.live_events_out_of_window`; a narrowing that is not reported is
    indistinguishable from clean.
    """
    now = now or datetime.now(timezone.utc)
    real_events = [e for e in real_events or [] if isinstance(e, dict)]
    shadow_events = [e for e in shadow_events or [] if isinstance(e, dict)]

    # The window is keyed on `shadow_session_start` specifically, not on the earliest shadow event
    # of any kind. That distinction matters: a live entry with no matching shadow intent is the
    # core divergence signal Gate 1 adjudicates, and windowing on "first intent seen" would delete
    # it whenever the lane fired later in the day than the live lane did. The session marker is the
    # lane's own statement of when it began observing, so it is the only honest boundary.
    #
    # A shadow journal with NO session marker is left unwindowed, preserving the original
    # semantics. Day-level windowing was tried first and was still too coarse — it charged the
    # daemon with 2026-08-12's 11:00 ET trade when it started at 13:50 that afternoon.
    observed_from: dict[str, datetime] = {}
    for e in shadow_events:
        if e.get("event_type") != SHADOW_SESSION_START:
            continue
        day, _, _, ts = _key_fields(e)
        if not day or ts is None:
            continue
        if day not in observed_from or ts < observed_from[day]:
            observed_from[day] = ts
    out_of_window = 0
    if observed_from:
        in_window = []
        for e in real_events:
            day, _, _, ts = _key_fields(e)
            start = observed_from.get(day) if day else None
            if start is not None and ts is not None and ts >= start:
                in_window.append(e)
        out_of_window = len(real_events) - len(in_window)
        real_events = in_window

    live_entries = _first_per_identity(real_events, _LIVE_ENTRY_TYPES)
    shadow_intents = [e for e in shadow_events if e.get("event_type") == SHADOW_ORDER_INTENT]
    live_exits = _first_per_identity(real_events, EXIT_FILL_EVENTS)
    shadow_exits = [e for e in shadow_events if e.get("event_type") == SHADOW_EXIT_INTENT]

    both_fired: list[dict] = []
    live_only: list[dict] = []
    used: set[int] = set()
    for live in live_entries:
        day, underlying, option_id, live_ts = _key_fields(live)
        shadow = _match(live, shadow_intents, used)
        if shadow is None:
            live_only.append({"day": day, "underlying": underlying, "option_id": option_id,
                              "ts": live.get("ts"), "event_type": live.get("event_type")})
            continue
        shadow_ts = _parse_ts(shadow.get("ts"))
        live_limit, shadow_limit = _limit_of(live), _limit_of(shadow)
        both_fired.append({
            "day": day, "underlying": underlying, "option_id": option_id,
            "live_ts": live.get("ts"), "shadow_ts": shadow.get("ts"),
            "latency_delta_s": (round((shadow_ts - live_ts).total_seconds(), 1)
                                if shadow_ts and live_ts else None),
            "limit_delta": (round(shadow_limit - live_limit, 4)
                            if live_limit is not None and shadow_limit is not None else None),
        })
    shadow_only = []
    for idx, intent in enumerate(shadow_intents):
        if idx in used:
            continue
        day, underlying, option_id, _ = _key_fields(intent)
        shadow_only.append({"day": day, "underlying": underlying, "option_id": option_id,
                            "ts": intent.get("ts"),
                            "intent_id": intent.get("intent_id"),
                            "reasons": intent.get("reasons")})

    exit_divergences: list[dict] = []
    used_exits: set[int] = set()
    for live in live_exits:
        day, underlying, option_id, live_ts = _key_fields(live)
        shadow = _match(live, shadow_exits, used_exits)
        entry = {"day": day, "underlying": underlying, "option_id": option_id,
                 "live_ts": live.get("ts"), "live_type": live.get("event_type")}
        if shadow is None:
            entry["divergence"] = "live_exit_without_shadow_intent"
        else:
            shadow_ts = _parse_ts(shadow.get("ts"))
            entry.update({
                "shadow_ts": shadow.get("ts"),
                "shadow_trigger": shadow.get("trigger_type") or shadow.get("trigger"),
                "timing_delta_s": (round((shadow_ts - live_ts).total_seconds(), 1)
                                   if shadow_ts and live_ts else None),
                "divergence": "matched",
            })
        exit_divergences.append(entry)
    for idx, intent in enumerate(shadow_exits):
        if idx in used_exits:
            continue
        day, underlying, option_id, _ = _key_fields(intent)
        exit_divergences.append({"day": day, "underlying": underlying, "option_id": option_id,
                                 "shadow_ts": intent.get("ts"),
                                 "shadow_trigger": (intent.get("trigger_type")
                                                    or intent.get("trigger")),
                                 "divergence": "shadow_exit_without_live_close"})

    incidents = [e for e in shadow_events if e.get("event_type") == SHADOW_INCIDENT]
    counts = {
        "live_entries": len(live_entries),
        "shadow_order_intents": len(shadow_intents),
        "live_exits": len(live_exits),
        "shadow_exit_intents": len(shadow_exits),
        "shadow_incidents": len(incidents),
        "shadow_events_total": len(shadow_events),
        "both_fired": len(both_fired),
        "shadow_only": len(shadow_only),
        "live_only": len(live_only),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        # What was compared, stated outright — the gate reads `clean`, so the window that produced
        # it must be auditable rather than implied.
        "window": {"observed_from": {d: t.isoformat(timespec="seconds")
                                     for d, t in sorted(observed_from.items())},
                   "live_events_out_of_window": out_of_window},
        "both_fired": both_fired,
        "shadow_only": shadow_only,
        "live_only": live_only,
        "exit_divergences": exit_divergences,
        "incidents": incidents,
        "counts": counts,
        # The Gate-1 bar: zero unexplained divergence. Matched pairs are evidence, not
        # divergence; everything else demands a one-line adjudication note before next session.
        "clean": not (shadow_only or live_only or incidents
                      or any(d["divergence"] != "matched" for d in exit_divergences)),
    }


def render_markdown(report: dict) -> str:
    """Terminal/Telegram-friendly summary of a shadow report."""
    counts = report.get("counts") or {}
    lines = [
        "# Shadow vs live — divergence report",
        f"_generated {report.get('generated_at')}_",
        "",
        (f"entries: live {counts.get('live_entries', 0)} / shadow "
         f"{counts.get('shadow_order_intents', 0)}  ·  matched {counts.get('both_fired', 0)}  ·  "
         f"shadow-only {counts.get('shadow_only', 0)}  ·  live-only {counts.get('live_only', 0)}"),
        (f"exits: live {counts.get('live_exits', 0)} / shadow intents "
         f"{counts.get('shadow_exit_intents', 0)}  ·  incidents "
         f"{counts.get('shadow_incidents', 0)}"),
        f"**clean: {report.get('clean')}**",
    ]
    for pair in report.get("both_fired") or []:
        lines.append(f"- MATCH {pair['underlying']} {pair['option_id']}: latency delta "
                     f"{pair['latency_delta_s']}s, limit delta {pair['limit_delta']}")
    for miss in report.get("shadow_only") or []:
        lines.append(f"- SHADOW-ONLY {miss['underlying']} {miss.get('option_id')} at "
                     f"{miss['ts']} — adjudicate before next session")
    for miss in report.get("live_only") or []:
        lines.append(f"- LIVE-ONLY {miss['underlying']} {miss.get('option_id')} at "
                     f"{miss['ts']} — the shadow lane must explain this miss")
    for div in report.get("exit_divergences") or []:
        if div.get("divergence") != "matched":
            lines.append(f"- EXIT {div['underlying']} {div.get('option_id')}: "
                         f"{div['divergence']}")
    return "\n".join(lines)
