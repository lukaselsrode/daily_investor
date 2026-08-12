"""0DTE decision journal — local/offline, NO broker, NO LLM, NO secrets.

A persistent JSONL journal of Hermes's 0DTE decision-making: the pre-trade thesis, the entry/exit
decisions, management checks, postmortems, and next-cycle experiments. It is a *record + analysis*
layer — it never places orders, never calls a broker or model, and never recommends or trades NVDA
(employer-restricted; any restricted event is tagged and kept out of experiments/metrics).

Pipeline:
    append_event(event)   -> one JSON line in data/odte/decision_journal.jsonl
    build_report()        -> deterministic metrics + Markdown/CSV artifacts under data/odte/reports/

----------------------------------------------------------------------------------------------------
event schema (flexible — JSONL; only `event_type` is required, everything else optional)
----------------------------------------------------------------------------------------------------
  event_type:   pre_trade_thesis | entry_decision | order_filled | management_check |
                exit_decision | order_closed | postmortem | experiment | note
  ts:           ISO timestamp (auto-stamped if absent); seq: append index (auto)
  trade_id:     groups all events of one trade
  mode:         scalp | trend | lotto | runner
  underlying, option_id, option_type, strike, expiration, quantity
  account_snapshot: optional dict (caller-supplied; stored verbatim)
  thesis:       direction, catalyst, social_pulse, market_read, key_levels, invalidation,
                profit_plan, time_rules   (any subset; free-form)
  decision:     action, confidence, reasons[], alternatives[], changed_since_prior
  outcome:      entry_price, exit_price, mfe, mae, realized_pnl, rule_violations[], lessons[]
  experiment:   hypothesis, metric, promote_if, kill_if, status
Outcome/decision fields may be given nested (outcome.realized_pnl) OR flat (realized_pnl) — both read.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:                                   # POSIX advisory file lock (mac/linux). Best-effort.
    import fcntl
except ImportError:                    # pragma: no cover - non-POSIX fallback
    fcntl = None

from core.paths import ODTE_DATA_DIR, ODTE_REPORT_DIR
from data.odte_strategy_policy import (
    ET,
    canonical_guardrails,
    classify_strategy_window,
    window_descriptions,
)

logger = logging.getLogger(__name__)

DEFAULT_DIR = ODTE_DATA_DIR
DEFAULT_JOURNAL_PATH = os.path.join(DEFAULT_DIR, "decision_journal.jsonl")
DEFAULT_REPORT_DIR = ODTE_REPORT_DIR

EVENT_TYPES = ("pre_trade_thesis", "entry_decision", "order_filled", "management_check",
               "exit_decision", "order_closed", "postmortem", "experiment", "note",
               # execution-safety layer (2026-07-23 delayed-fill remediation):
               "execution_lease_issued", "execution_lease_consumed", "entry_order_pending",
               "entry_order_cancelled_stale", "execution_safety_incident")

# The execution-safety event family — every lease issuance/consumption and every pending-order
# submit/cancel/fill transition must have a matching journal event (shadow-rollout gate).
EXECUTION_SAFETY_EVENT_TYPES = ("execution_lease_issued", "execution_lease_consumed",
                                "entry_order_pending", "entry_order_cancelled_stale",
                                "execution_safety_incident")

# FILL VOCABULARY (2026-08-04): the live controller journals `entry_fill`/`exit_fill` while the
# repo's original schema said `order_filled`. Every cadence/telemetry counter must accept BOTH —
# on 2026-08-03 the day's real fill was an `entry_fill`, so an `order_filled`-only counter read
# zero trades and would have left the 2/day budget and the zero-trade tripwire non-functional.
ENTRY_FILL_EVENTS = ("order_filled", "entry_fill")


def _decision_word(event: dict) -> str:
    """The event's `decision` as a lowercase word, whatever shape it arrived in.

    Funnel counters compared `decision` case-sensitively against literals ("enter", "issue",
    "allow"), but the field is written by several lanes and has drifted: across the journal it
    holds `enter`/`veto`/`skip` alongside `NO_TRADE`, `WAIT_NO_TRADE`,
    `NO_TRADE_BROKER_DEGRADED`, `NO TRADE` — and in ~40 events a whole DICT
    (`{"action": "HOLD", "reasons": [...]}`) rather than a string.

    Today those all correctly fail the `== "enter"` test because none of them IS an entry, so the
    counters are right BY LUCK rather than by construction — a single uppercase `ENTER` from any
    writer would be silently counted as a non-entry and understate `gates_passed`. Normalising at
    the READER fixes future drift and the existing history in one place, and never touches the
    write path (the append is fail-safe telemetry and must stay that way).
    """
    d = event.get("decision")
    if isinstance(d, dict):                      # some lanes journalled the whole decision object
        d = d.get("action") or d.get("decision")
    return str(d or "").strip().lower()
EXIT_FILL_EVENTS = ("order_closed", "exit_fill", "exit_decision")

_ENTRY_EVENTS = {"entry_decision", *ENTRY_FILL_EVENTS}
_EXIT_EVENTS = {"exit_decision", "order_closed", "exit_fill"}

_BLOCKS = "▁▂▃▄▅▆▇█"

# Honest gamma basis — mirrors odte_gamma_map.GAMMA_REGIME_LABEL. Any gamma snapshot rolled up here
# is ABSOLUTE pin-risk/OI concentration only; it never carries (or lets us infer) dealer net GEX.
_GAMMA_BASIS = "pin_risk_only_not_dealer_gex"


# --- small helpers ---------------------------------------------------------------------------

def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get(event: dict, key: str):
    """Read `key` from the top level, else from a nested outcome/decision/thesis container."""
    if key in event:
        return event[key]
    for container in ("outcome", "decision", "thesis"):
        sub = event.get(container)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return None


def _list(event: dict, key: str) -> list:
    v = _get(event, key)
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _first(events: list[dict], key: str):
    for e in events:
        v = _get(e, key)
        if v not in (None, ""):
            return v
    return None


def _last_num(events: list[dict], key: str) -> float | None:
    out = None
    for e in events:
        v = _num(_get(e, key))
        if v is not None:
            out = v
    return out


# The 2026-08-03 postmortem schema moved the self-eval fields into typed sub-objects
# (`excursion`, `rails`, `process_review`) instead of the flat top-level keys the pre-08-03 events
# used. These read the nested form; every caller falls back to the legacy key so both eras stay
# measurable in one series. `_get` deliberately does NOT know these containers — it walks
# outcome/decision/thesis, which are a different (older) nesting convention.

def _sub(event: dict, container: str, key: str):
    sub = event.get(container)
    return sub.get(key) if isinstance(sub, dict) else None


def _last_sub_num(events: list[dict], container: str, key: str) -> float | None:
    out = None
    for e in events:
        v = _num(_sub(e, container, key))
        if v is not None:
            out = v
    return out


def _first_sub(events: list[dict], container: str, key: str):
    for e in events:
        v = _sub(e, container, key)
        if v not in (None, ""):
            return v
    return None


def _sub_list(event: dict, container: str, key: str) -> list:
    v = _sub(event, container, key)
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        raw = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _time_bucket(ts) -> str:
    """ET time-of-day bucket for ODTE self-eval. This is descriptive, never execution authority."""
    return classify_strategy_window(ts)


def _dedupe_key(*parts) -> tuple:
    cleaned = []
    for part in parts:
        text = re.sub(r"\s+", " ", str(part or "").strip().lower())
        cleaned.append(text)
    return tuple(cleaned)


def _held_minutes(events: list[dict]) -> float | None:
    entry = next((_parse_ts(e.get("ts")) for e in events if e.get("event_type") in _ENTRY_EVENTS), None)
    exits = [_parse_ts(e.get("ts")) for e in events if e.get("event_type") in _EXIT_EVENTS]
    exits = [x for x in exits if x is not None]
    if entry and exits:
        return round((max(exits) - entry).total_seconds() / 60.0, 1)
    return None


def _sparkline(vals: list) -> str:
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    if hi == lo:
        return _BLOCKS[3] * len(nums)
    return "".join(_BLOCKS[int((v - lo) / (hi - lo) * (len(_BLOCKS) - 1))] for v in nums)


def _bar(value: float, max_abs: float, width: int = 18) -> str:
    if not max_abs:
        return "░" * width
    n = min(width, round(abs(value) / max_abs * width))
    return "█" * n + "░" * (width - n)


# --- append / read ---------------------------------------------------------------------------

def normalize_event(event: dict | None, now: datetime | None = None, seq: int | None = None) -> dict:
    """Stamp event_type/ts/seq and tag employer-restricted underlyings (e.g. NVDA). Never raises."""
    from data.social_sentiment import is_restricted_underlying
    e = dict(event or {})
    et = str(e.get("event_type") or e.get("type") or "").strip()
    e["event_type"] = et or "note"
    if not e.get("ts"):
        e["ts"] = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    if seq is not None:
        e["seq"] = seq
    und = str(e.get("underlying") or "").upper()
    if und and is_restricted_underlying(und):
        # Hard restriction is preserved as a record: tag it; the report keeps it out of
        # experiments/metrics and surfaces it loudly as a violation. NVDA must never be traded.
        e["restricted"] = True
        e["restricted_reason"] = "employer"
    return e


def append_event(event: dict, journal_path: str | None = None, now: datetime | None = None) -> dict:
    """Append one normalized event as a JSON line; return the stored event (with seq/ts)."""
    path = Path(os.path.expanduser(journal_path or DEFAULT_JOURNAL_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = sum(1 for _ in path.open()) if path.exists() else 0
    stored = normalize_event(event, now=now, seq=seq)
    with path.open("a") as f:
        f.write(json.dumps(stored, default=str) + "\n")
    return stored


def read_events(journal_path: str | None = None) -> list[dict]:
    """Read the JSONL journal; malformed lines are skipped (never raises)."""
    path = Path(os.path.expanduser(journal_path or DEFAULT_JOURNAL_PATH))
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            logger.debug("journal: skipping malformed line")
    return out


# --- green-day preservation (post-scalp lockout) ---------------------------------------------
# Once the first COMPLETED profitable trade of the ET trading day is banked, the fresh-entry lane
# is locked: `odte_entry_gate` vetoes a later gate and `odte_loop_status` refuses to surface fresh
# candidates / promoted gates, unless the caller EXPLICITLY overrides with
# `allow_reentry_after_green: true` (and passes the BP floor the entry gate enforces). The new-day
# reset is the ET calendar date — nothing carries over past the session.

# Event types that mark a COMPLETED trade (may carry realized P/L). entry/management events never lock.
# `exit_fill` joined this on 2026-08-12. It is the exit lane's own fill record and states the same
# close as `order_closed` (which is agent-authored on the live path) — on 2026-08-12 order_closed
# was never written, so a -$20.00 close left net_day reading $0.00. That is not cosmetic: net_day
# gates `aplus_uncapped_active`, whose policy is that any net-red day de-activates the cap bypass,
# so a red day was presenting itself as flat. Historical values are unchanged — every prior
# exit_fill carries a gross_pnl identical to its order_closed realized_pnl, and the dedupe below
# is per trade_id, so the pair collapses to one row.
_COMPLETED_EVENTS = ("order_closed", "exit_decision", "postmortem", "exit_fill")
_INACTIVE_PLAN_STATUS = {"closed", "exited", "flat", "done"}   # mirrors odte_position._INACTIVE_STATUS


def _realized(container: dict) -> float | None:
    """Realized P/L from an event/plan: canonical `realized_pnl` first, then the flat live-journal
    and plan spellings (`realized_pnl_dollars_gross`, `estimated_net_pnl`, `gross_pnl`,
    `net_pnl_est`). `estimated_net_pnl` is the exit lane's own fill record and is preferred over
    `gross_pnl` because it is net of the fee estimate; "estimated" qualifies the fees, not the
    fill."""
    for key in ("realized_pnl", "realized_pnl_dollars_gross", "estimated_net_pnl", "gross_pnl",
                "net_pnl_est"):
        v = _num(_get(container, key))
        if v is not None:
            return v
    return None


def _et_date(ts) -> str | None:
    dt = _parse_ts(ts)
    return dt.astimezone(ET).date().isoformat() if dt else None


def green_day_preservation(events: list[dict] | None, active_trade: dict | None = None,
                           now: datetime | None = None) -> dict:
    """PURE: has a COMPLETED profitable trade already been banked for the current ET trading day?

    Reads only the supplied journal events + (optionally) the closed active_trade plan — no IO, no
    broker. A trade counts as banked green when a completion event (order_closed / exit_decision /
    postmortem) or a closed plan carries a positive realized P/L dated TODAY in ET. Returns
    ``{locked, trade_date, banked_pnl, green_trades}`` — ``locked`` True means green-day
    preservation is in force: NO fresh entries without an explicit ``allow_reentry_after_green``
    override (see `odte_entry_gate` / `odte_loop_status`)."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ET).date().isoformat()
    # Track one realized P/L per completed trade for the ET day.  Green-day preservation is a
    # *net-day* discipline lockout, not a "last trade was green" lockout: if an earlier loss keeps
    # the session net red, fresh entries remain allowed but still require the stricter post-loss
    # quality gates.  Keep the positive rows for reporting, but gate ``locked`` on net realized P/L.
    realized_by_trade: dict[str, dict] = {}
    for i, e in enumerate(events or []):
        if not isinstance(e, dict) or e.get("event_type") not in _COMPLETED_EVENTS:
            continue
        if str(e.get("source") or "").startswith("ingest:"):
            continue        # an ingested completion is a record copy, never countable P/L
        if _et_date(e.get("ts")) != today:
            continue
        pnl = _realized(e)
        if pnl is None:
            continue
        key = str(e.get("trade_id") or f"event#{e.get('seq', i)}")
        row = {"trade_id": e.get("trade_id"),
               "underlying": (str(e.get("underlying") or e.get("symbol") or "").upper() or None),
               "realized_pnl": pnl, "closed_ts": e.get("ts")}
        prev = realized_by_trade.get(key)
        # Prefer the latest same-trade completion record; duplicate journal entries usually carry
        # the same P/L, while postmortems/corrections may refine it later in the day.
        if prev is None or str(row.get("closed_ts") or "") >= str(prev.get("closed_ts") or ""):
            realized_by_trade[key] = row
    plan = active_trade if isinstance(active_trade, dict) else {}
    if str(plan.get("status") or "").strip().lower() in _INACTIVE_PLAN_STATUS:
        close_ts = plan.get("closed_at") or plan.get("exit_fill_time") or plan.get("updated_at")
        pnl = _realized(plan)
        key = str(plan.get("trade_id") or "active_trade")
        if pnl is not None and _et_date(close_ts) == today and key not in realized_by_trade:
            realized_by_trade[key] = {"trade_id": plan.get("trade_id"),
                                      "underlying": (str(plan.get("underlying") or "").upper() or None),
                                      "realized_pnl": pnl, "closed_ts": close_ts}
    all_rows = sorted(realized_by_trade.values(), key=lambda r: str(r.get("closed_ts") or ""))
    rows = [r for r in all_rows if r["realized_pnl"] > 0]
    net_day_pnl = round(sum(r["realized_pnl"] for r in all_rows), 4)
    return {"locked": bool(rows) and net_day_pnl > 0, "trade_date": today,
            "banked_pnl": round(sum(r["realized_pnl"] for r in rows), 4),
            "net_day_pnl": net_day_pnl, "green_trades": rows,
            "completed_trades": all_rows}


def execution_safety_lockout(events: list[dict] | None, now: datetime | None = None) -> dict:
    """PURE: has an `execution_safety_incident` been journaled for the current ET day?

    2026-07-23 delayed-fill remediation: a fill outside a valid execution lease (or a broker/lease
    mismatch) is a SAFETY INCIDENT — once one is journaled, NO new entries are permitted for the
    session (`odte_loop_status` consumes fresh gates/candidates/triggers). There is no automatic
    override: the lockout clears with the ET calendar day, or through EXACTLY ONE sanctioned
    escape (2026-08-06, user-authorized after a proven-false incident flattened a healthy
    position): a `execution_safety_incident_adjudicated` event that (a) is journaled the SAME ET
    day, (b) NAMES the incident it clears (`incident_event_id`, fallback `incident_seq`), and
    (c) carries `adjudicated_by` + `reason`. Adjudication is a HUMAN act journaled via
    `odte-journal --event-json` after the underlying defect is fixed and test-pinned — the
    controller is told nothing about this event type and must never emit it. An unmatched or
    unattributed adjudication clears nothing. Reads only the supplied events — no IO, no broker."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ET).date().isoformat()
    adjudications = [
        e for e in (events or [])
        if isinstance(e, dict)
        and e.get("event_type") == "execution_safety_incident_adjudicated"
        and _et_date(e.get("ts")) == today
        and str(e.get("adjudicated_by") or "").strip()
        and str(e.get("reason") or "").strip()
    ]

    def _cleared(incident: dict) -> dict | None:
        for adj in adjudications:
            if (incident.get("event_id") and adj.get("incident_event_id")
                    and str(adj["incident_event_id"]) == str(incident["event_id"])):
                return adj
            if (adj.get("incident_seq") is not None and incident.get("seq") is not None
                    and str(adj["incident_seq"]) == str(incident["seq"])):
                return adj
        return None

    incidents, cleared = [], []
    for e in (events or []):
        if not (isinstance(e, dict) and e.get("event_type") == "execution_safety_incident"
                and _et_date(e.get("ts")) == today):
            continue
        row = {"seq": e.get("seq"), "ts": e.get("ts"),
               "underlying": (str(e.get("underlying") or e.get("symbol") or "").upper() or None),
               "reason_codes": list(e.get("reason_codes") or []),
               "guard_state": e.get("guard_state") or e.get("state")}
        adj = _cleared(e)
        if adj is None:
            incidents.append(row)
        else:
            cleared.append({**row, "adjudicated_by": adj.get("adjudicated_by"),
                            "adjudication_reason": adj.get("reason")})
    return {"locked": bool(incidents), "trade_date": today, "incidents": incidents,
            "adjudicated": cleared}


# Confirmation-tier ranking (2026-08-03 green re-entry auto-arm). The tape-computed tiers order as
# a_plus > full > b_plus: an A+ read (>=3 confirmers, zero dissenters) outranks a standard GOOD_DAY
# confirm, which outranks the CHOP half-size B+ tier. Unknown/absent ranks 0 (never auto-arms).
TIER_RANK = {"b_plus": 1, "full": 2, "a_plus": 3}


def tier_rank(tier) -> int:
    return TIER_RANK.get(str(tier or "").strip().lower(), 0)


def green_day_winning_tier(events: list[dict] | None, active_trade: dict | None = None,
                           now: datetime | None = None) -> dict:
    """PURE: the confirmation tier(s) behind today's banked green trade(s).

    2026-08-03 auto-arm rule: a post-green re-entry must be a same-or-better setup than the trade
    that banked the green, judged by the tape-computed tier. Joins each green trade row (from
    `green_day_preservation`) to today's tier-bearing entry-side events — `entry_decision`,
    `execution_lease_issued`, `order_filled` — by trade_id, else option_id, else same underlying
    same ET day. A trade whose entry events carry no tier (pre-retune journals) counts as "full"
    (`tier_source: "default_full"`): the standard GOOD_DAY bar, neither free (b_plus) nor
    prohibitive (a_plus). Reads only the supplied events — no IO, no broker. Returns
    {trade_date, winning_tier, winning_rank, trades}."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ET).date().isoformat()
    preservation = green_day_preservation(events, active_trade=active_trade, now=now)
    entry_events = [
        e for e in (events or [])
        if isinstance(e, dict)
        and e.get("event_type") in ("entry_decision", "execution_lease_issued", *ENTRY_FILL_EVENTS)
        and _et_date(e.get("ts")) == today
    ]

    def _event_tier(e: dict):
        return e.get("tier") or _dict_get(e, "candidate", "tier")

    def _tier_for(row: dict) -> tuple[str, str]:
        trade_id = str(row.get("trade_id") or "")
        underlying = str(row.get("underlying") or "").upper()
        # The preservation row itself carries no option_id — derive it from the trade's own
        # entry-side events (order_filled carries both trade_id and option_id).
        row_opt = str(row.get("option_id") or "")
        if not row_opt and trade_id:
            for e in entry_events:
                if str(e.get("trade_id") or "") == trade_id and e.get("option_id"):
                    row_opt = str(e.get("option_id"))
                    break
        for join in ("trade_id", "option_id", "underlying"):
            for e in reversed(entry_events):
                t = _event_tier(e)
                if not t:
                    continue
                if join == "trade_id":
                    if trade_id and str(e.get("trade_id") or "") == trade_id:
                        return str(t).lower(), join
                elif join == "option_id":
                    if row_opt and str(e.get("option_id") or "") == row_opt:
                        return str(t).lower(), join
                else:
                    ev_sym = str(e.get("underlying") or e.get("symbol") or "").upper()
                    if underlying and ev_sym == underlying:
                        return str(t).lower(), join
        return "full", "default_full"

    trades = []
    for row in preservation.get("green_trades") or []:
        tier, source = _tier_for(row)
        trades.append({"trade_id": row.get("trade_id"), "underlying": row.get("underlying"),
                       "tier": tier, "tier_source": source})
    winning = max(trades, key=lambda t: tier_rank(t["tier"]), default=None)
    return {
        "trade_date": today,
        "winning_tier": winning["tier"] if winning else None,
        "winning_rank": tier_rank(winning["tier"]) if winning else 0,
        "trades": trades,
    }


def _dict_get(e: dict, key: str, sub: str):
    v = e.get(key)
    return v.get(sub) if isinstance(v, dict) else None


# Event types ONLY the live trading lanes may write — ingest refuses them, and the money
# counters ignore any that carry an ingest: source (belt and suspenders, 2026-08-06).
_INGEST_PROTECTED_LIFECYCLE = frozenset({
    *ENTRY_FILL_EVENTS, *EXIT_FILL_EVENTS, "entry_decision", "execution_lease_issued",
    "execution_lease_consumed", "execution_safety_incident",
    "execution_safety_incident_adjudicated", "no_trade_decision", "postmortem",
    # First-party at computation time (odte_convert). An EOD artifact sweep would re-add the same
    # decision under a different dedupe key and double-count it.
    "candidate_evaluation",
    # `vehicle_score` is protected for a DIFFERENT reason than candidate_evaluation, and NOT by
    # analogy to day_score below — its ingest lane has been DRY since the v5 port. The 21 historic
    # events all carry `source: ingest:vehicle_score` from loose top-level artifacts the
    # controller stopped writing on 2026-08-04, and `run_vehicle_score --write` lands in reports/,
    # which the top-level-only ingest glob never reaches. So there is no live drop for the
    # protection to suppress and no blind window to open: unlike day_score, protecting this
    # forfeits nothing.
    "vehicle_score",
})
# `day_score` is deliberately NOT protected (2026-08-06). It is first-party journaled from
# odte_convert, but that call sits AFTER the preflight refusal — so on a session where every
# convert tick dies on a stale snapshot (the 2026-08-05 blind-day shape) there is no first-party
# event at all, and the controller's own top-level day_score drops are the only record. Those are
# exactly the days the headroom telemetry exists to explain. day_score touches no money or volume
# counter (not a fill, not a refusal, not keyed by trade_id), so an overlapping ingest costs a
# duplicate point on a chart, while protecting it costs the whole day.

_SAME_FILL_WINDOW_MINUTES = 10.0


def _entry_fill_keys(fills: list[tuple[dict, int]]) -> set[str]:
    """Canonical distinct-entry keys for fill events. Two events describe the SAME entry when
    they share an option_id within a short window — the order guard journals `order_filled`
    (often without a trade_id) seconds after the controller's `entry_fill` for the same order,
    and the old `trade_id or option_id` key counted them as two trades (2026-08-06: one QQQ
    fill read as trades_today=3 and would burn a phantom budget slot on any future day).
    Legitimate same-contract re-entries are always separated by the post-close cooldown
    (>= REENTRY_COOLDOWN_MINUTES), far outside the join window."""
    keys: set[str] = set()
    seen_opts: list[tuple[str, datetime]] = []
    for e, i in fills:
        if str(e.get("source") or "").startswith("ingest:"):
            continue        # an ingested fill is a record copy, never a countable trade
        opt = str(e.get("option_id") or "").strip()
        ts = _parse_ts(e.get("ts"))
        if opt and ts is not None:
            if any(o == opt and abs((ts - t).total_seconds()) <= _SAME_FILL_WINDOW_MINUTES * 60
                   for o, t in seen_opts):
                continue                        # duplicate journaling of the same fill
            seen_opts.append((opt, ts))
            keys.add(f"{opt}@{ts.astimezone(ET).isoformat(timespec='minutes')}")
            continue
        keys.add(str(e.get("trade_id") or opt or f"event#{e.get('seq', i)}"))
    return keys


def daily_trade_budget(events: list[dict] | None, now: datetime | None = None) -> dict:
    """PURE: today's ET-day entry count vs the daily trade budget, plus the post-close cooldown.

    2026-08-02 retune: the trade-cadence rail is a hard per-day budget (odte_config
    DAILY_TRADE_BUDGET, default 2) instead of the un-armable flat-BP green lockout being the only
    second-trade gate. An entry is a today's-ET `order_filled` event (deduped by trade_id/option_id
    so entry+exit fills of one trade count once); the cooldown window opens at the latest completed
    trade (order_closed / exit_decision) and lasts REENTRY_COOLDOWN_MINUTES. Reads only the supplied
    events — no IO, no broker. Returns {trades_today, budget, remaining, exhausted, last_close_ts,
    cooldown_until, cooldown_active, trade_date}."""
    from data.odte_config import (
        DAILY_BUDGET_APLUS_UNCAPPED,
        DAILY_TRADE_BUDGET,
        REENTRY_COOLDOWN_MINUTES,
    )
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ET).date().isoformat()
    fills: list[tuple[dict, int]] = []
    last_close: datetime | None = None
    for i, e in enumerate(events or []):
        if not isinstance(e, dict) or _et_date(e.get("ts")) != today:
            continue
        et = e.get("event_type")
        if et in ENTRY_FILL_EVENTS:
            fills.append((e, i))
        elif et in _EXIT_EVENTS:
            ts = _parse_ts(e.get("ts"))
            if ts is not None and (last_close is None or ts > last_close):
                last_close = ts
    entry_keys = _entry_fill_keys(fills)
    cooldown_until = (last_close + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)
                      if last_close is not None else None)
    trades_today = len(entry_keys)
    # A+ UNCAPPED (2026-08-06 user policy): while the day's realized P/L is >= 0, an a_plus-tier
    # setup may enter past the base cap (the GATE enforces the tier; this payload carries the
    # eligibility so every consumer reads one source). Any net-red day de-activates it.
    net_day = _num(green_day_preservation(events, now=now).get("net_day_pnl"))
    aplus_uncapped_active = bool(DAILY_BUDGET_APLUS_UNCAPPED
                                 and (net_day is None or net_day >= 0))
    return {
        "trades_today": trades_today,
        "budget": DAILY_TRADE_BUDGET,
        "remaining": max(0, DAILY_TRADE_BUDGET - trades_today),
        "exhausted": trades_today >= DAILY_TRADE_BUDGET,
        "day_net_pnl": net_day,
        "aplus_uncapped_active": aplus_uncapped_active,
        "last_close_ts": last_close.isoformat(timespec="seconds") if last_close else None,
        "cooldown_until": cooldown_until.isoformat(timespec="seconds") if cooldown_until else None,
        "cooldown_active": bool(cooldown_until is not None and now < cooldown_until),
        "trade_date": today,
    }


def _funnel_tally(events: list[dict] | None, keep) -> tuple[Counter, Counter, Counter, list]:
    """PURE: tally the conversion funnel over the events `keep(dt, event)` accepts.

    Shared by `weekly_telemetry` (ISO-week window) and `funnel_counts` (arbitrary window) so the
    two can never disagree about what a gate pass or a refusal is. Returns
    (counts, refusals, stage_counts, fills) — fills as (event, index) pairs for `_entry_fill_keys`.

    Event types not named here (watchdog ticks, snapshots, candidate_evaluation, day_score…) are
    deliberately inert: telemetry must never move a funnel counter.
    """
    counts: Counter = Counter()
    refusals: Counter = Counter()
    stage_counts: Counter = Counter()
    fills: list[tuple[dict, int]] = []
    for i, e in enumerate(events or []):
        if not isinstance(e, dict):
            continue
        dt = _parse_ts(e.get("ts"))
        if dt is None or not keep(dt, e):
            continue
        et = e.get("event_type")
        if et in ENTRY_FILL_EVENTS:
            fills.append((e, i))
        elif et == "entry_decision":
            counts["entry_decisions"] += 1
            if e.get("execution_allowed") is True or _decision_word(e) == "enter":
                counts["gates_passed"] += 1
        elif et == "execution_lease_issued":
            # `decision` reads "issue" on the pre-2026-08-06 convert path and "allow" once both
            # paths moved to the canonical builder; `authorized` is the field that never moved.
            if e.get("authorized") is True or _decision_word(e) in ("issue", "allow"):
                counts["leases_issued"] += 1
            else:
                counts["lease_refusals"] += 1
                for r in (e.get("reason_codes") or []):
                    refusals[f"authorize:{r}"] += 1
        elif et == "no_trade_decision":
            counts["no_trade_decisions"] += 1
            # v5 refusals happen at preflight/candidate_watch/entry_gate — BEFORE any lease event
            # exists. 2026-08-04: an 18-refusal day read `top_refusal_reasons: []` because only
            # lease refusals were tallied. Stage-prefix every terminal reason.
            stage = str(e.get("stage") or "unknown")
            stage_counts[stage] += 1
            for r in (e.get("reason_codes") or []):
                refusals[f"{stage}:{r}"] += 1
        elif et == "candidate_evaluation":
            # Not a funnel STAGE — recorded alongside so a window can report how many evaluations
            # produced how few conversions without a second pass over the events.
            counts["candidate_evaluations"] += 1
            if str(e.get("decision") or "").upper() == "CONFIRM_ENTRY":
                counts["candidate_confirms"] += 1
        elif et == "watchdog_trigger":
            counts["watchdog_ticks"] += 1
            if e.get("alert") is True:
                counts["watchdog_alerts"] += 1
    return counts, refusals, stage_counts, fills


def funnel_counts(events: list[dict] | None, *, start: date, end: date) -> dict:
    """PURE: conversion-funnel counts over an inclusive ET date range [start, end].

    The same tally `weekly_telemetry` runs, but over an arbitrary window, so the UI can chart the
    funnel per day / per week without re-deriving what counts as a gate pass or a refusal.
    Trades are deduped with `_entry_fill_keys` (one fill journaled by two lanes counts once).
    """
    counts, refusals, stage_counts, fills = _funnel_tally(
        events, lambda dt, _e: start <= dt.astimezone(ET).date() <= end)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "watchdog_ticks": counts["watchdog_ticks"],
        "watchdog_alerts": counts["watchdog_alerts"],
        "candidate_evaluations": counts["candidate_evaluations"],
        "candidate_confirms": counts["candidate_confirms"],
        "entry_decisions": counts["entry_decisions"],
        "gates_passed": counts["gates_passed"],
        "leases_issued": counts["leases_issued"],
        "fills": len(_entry_fill_keys(fills)),
        "no_trade_decisions": counts["no_trade_decisions"],
        "lease_refusals": counts["lease_refusals"] + sum(stage_counts.values()),
        "top_refusal_reasons": refusals.most_common(10),
        "refusals_by_stage": dict(stage_counts),
    }


def weekly_telemetry(events: list[dict] | None, now: datetime | None = None) -> dict:
    """PURE: ISO-week (ET) conversion-funnel counts + the zero-trade tripwire.

    2026-08-02 retune: the weekly target is 3-4 trades and a zero-trade week must be VISIBLE while
    it is still fixable, not discovered in a Friday postmortem. Counts this ET ISO-week's journal
    events — trades (deduped order_filled), gates passed, leases issued/refused with the top
    refusal reasons — and arms a tripwire once the week reaches the configured weekday (default
    Wednesday) with zero trades. Reads only the supplied events — no IO, no broker. The tripwire is
    ADVISORY telemetry: it never loosens a gate by itself."""
    from data.odte_config import (
        WEEKLY_TRADE_TARGET_MAX,
        WEEKLY_TRADE_TARGET_MIN,
        ZERO_TRADE_TRIPWIRE_WEEKDAY,
    )
    now = now or datetime.now(timezone.utc)
    year, week, iso_weekday = now.astimezone(ET).isocalendar()
    counts, refusals, stage_counts, week_fills = _funnel_tally(
        events, lambda dt, _e: dt.astimezone(ET).isocalendar()[:2] == (year, week))
    trades = len(_entry_fill_keys(week_fills))
    budget = daily_trade_budget(events, now=now)
    armed = (iso_weekday - 1) >= ZERO_TRADE_TRIPWIRE_WEEKDAY   # isocalendar: 1=Mon
    return {
        "iso_week": f"{year}-W{week:02d}",
        "trades_this_week": trades,
        "weekly_target": [WEEKLY_TRADE_TARGET_MIN, WEEKLY_TRADE_TARGET_MAX],
        "on_pace": trades >= WEEKLY_TRADE_TARGET_MIN or not armed,
        "entry_decisions": counts["entry_decisions"],
        "gates_passed": counts["gates_passed"],
        "leases_issued": counts["leases_issued"],
        # Total refusals across ALL stages (2026-08-05: the scalar counted only lease-stage
        # refusals, reading 0 on an 18-refusal week while refusals_by_stage held the truth).
        "lease_refusals": counts["lease_refusals"] + sum(stage_counts.values()),
        "no_trade_decisions": counts["no_trade_decisions"],
        "top_refusal_reasons": refusals.most_common(5),
        "refusals_by_stage": dict(stage_counts),
        "trades_today": budget["trades_today"],
        "budget_remaining_today": budget["remaining"],
        "tripwire": {"armed": armed, "fired": bool(armed and trades == 0),
                     "weekday_et": iso_weekday - 1},
    }


# --- standardized decision-journal layer ----------------------------------------------------
# A normalized envelope so loose controller/watchdog/veto artifacts can all be folded into the
# SAME journal the post-day self-eval reads. Conservative by construction: nothing here can make
# trading more aggressive — `execution_allowed` defaults False and a `scan_only` event can NEVER
# carry execution_allowed=True (the scan tier must never silently become an execution tier).

DECISION_SCHEMA = "odte_decision_v1"

# Recognized `mode` tiers (free-form is still accepted; these are the canonical ones).
DECISION_MODES = ("scan", "candidate", "watchdog", "execution", "management", "postmortem", "note")
# Recognized `decision` verbs (free-form accepted; canonical set for analysis grouping).
DECISION_VERBS = ("allow", "deny", "veto", "skip", "enter", "exit", "hold", "observe", "wait", "note")

# Fields that define an event's identity for idempotent (de-duplicated) ingestion. A loose file is
# keyed by its artifact path + type (re-ingesting the same file never double-appends); a live
# append with no artifact is keyed by its semantic content + timestamp.
_ID_FIELDS = ("source", "event_type", "trade_date", "symbol", "mode", "decision", "ts")

# Synonyms → canonical decision verbs (free-form still passes through lowercased). Keeps post-day
# grouping/self-eval clean when callers write 'open'/'close'/'no_trade'/'DO NOTHING' etc.
_VERB_SYNONYMS = {
    "open": "enter", "buy": "enter", "entered": "enter", "entry": "enter",
    "close": "exit", "closed": "exit", "sell": "exit", "sold": "exit", "exited": "exit",
    "no_trade": "skip", "notrade": "skip", "pass": "skip", "passed": "skip",
    "do_nothing": "observe", "nothing": "observe", "monitor": "hold", "holding": "hold",
}


def _normalize_verb(v) -> str | None:
    """Map a decision value to a canonical scalar verb (allow/deny/veto/skip/enter/exit/hold/
    observe/wait/note); unknown verbs pass through lowercased (free-form still allowed)."""
    if v in (None, ""):
        return None
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    return _VERB_SYNONYMS.get(s, s)


def _derive_trade_date(ts: str | None) -> str:
    """YYYY-MM-DD trade date from an ISO ts (UTC), or today (UTC) if unparseable."""
    dt = _parse_ts(ts) or datetime.now(timezone.utc)
    return dt.date().isoformat()


def _compute_event_id(e: dict) -> str:
    """Stable 16-hex id for idempotency. For a loose artifact, anchored to its resolved path PLUS a
    content hash (so a CHANGED file re-ingests as a new record instead of being silently skipped);
    for a live append, anchored to the semantic identity fields."""
    raw = e.get("raw_artifact_path")
    if raw:
        sha = e.get("raw_artifact_sha")
        basis = f"{raw}::{sha}" if sha else f"{raw}::{e.get('event_type', '')}"
    else:
        basis = json.dumps({k: e.get(k) for k in _ID_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


# Nested containers a loose artifact may carry a symbol under (parent -> child keys, in order).
_SYMBOL_NESTS = (("contract", ("underlying", "symbol", "ticker")),
                 ("candidate", ("ticker", "underlying", "symbol")),
                 ("vehicle_score", ("underlying", "symbol")),
                 ("thesis", ("underlying", "ticker", "symbol")))


def _coalesce_symbol(event: dict) -> str | None:
    """Pull a ticker from the top level, else from a nested contract/candidate/vehicle_score/thesis
    container (so ingested artifacts that only carry contract.underlying still get tagged)."""
    for k in ("symbol", "ticker", "underlying"):
        v = event.get(k)
        if v:
            return str(v).upper()
    for parent, childs in _SYMBOL_NESTS:
        sub = event.get(parent)
        if isinstance(sub, dict):
            for c in childs:
                if sub.get(c):
                    return str(sub[c]).upper()
    return None


def existing_event_ids(journal_path: str | None = None) -> set[str]:
    """Set of event_id values already in the journal (for idempotent ingestion). Never raises."""
    return {e["event_id"] for e in read_events(journal_path) if e.get("event_id")}


def build_decision_event(event: dict | None, *, source: str, event_type: str,
                         trade_date: str | None = None, now: datetime | None = None) -> dict:
    """Build (but do not append) a normalized standardized decision event. Pure + never raises.

    Wraps an arbitrary `event` dict in the standardized envelope, preserving all original fields
    while overlaying the canonical observability fields. Restriction-tagging (NVDA) is applied via
    `normalize_event`. Conservative guard: `scan_only` forces `execution_allowed=False`.
    """
    base = normalize_event(event, now=now)        # stamps ts/event_type, tags restricted underlyings
    base["event_type"] = str(event_type or base.get("event_type") or "note").strip() or "note"
    base["schema"] = DECISION_SCHEMA
    base["source"] = str(source or "unknown")
    base["trade_date"] = trade_date or _derive_trade_date(base.get("ts"))
    sym = _coalesce_symbol(base)
    if sym:
        base["symbol"] = sym
        # normalize_event() only saw a top-level `underlying`; a nested contract.underlying=NVDA must
        # still be tagged restricted (and forced non-executable below).
        from data.social_sentiment import is_restricted_underlying
        if is_restricted_underlying(sym):
            base["restricted"] = True
            base["restricted_reason"] = "employer"
    base.setdefault("mode", None)
    # `decision` is normalized to a SCALAR verb for clean grouping; a dict decision (action/reasons)
    # is preserved verbatim under `decision_detail` so nothing is lost.
    dec = base.get("decision")
    if isinstance(dec, dict):
        base["decision_detail"] = dec
        base["decision"] = _normalize_verb(dec.get("action") or dec.get("decision"))
    else:
        base["decision"] = _normalize_verb(dec)
    base.setdefault("reason_codes", list(base.get("reason_codes") or []))
    base.setdefault("thesis", base.get("thesis"))
    base.setdefault("confidence", base.get("confidence"))
    base.setdefault("confirmation_needed", base.get("confirmation_needed"))
    # Conservative defaults: observe-only unless the caller explicitly opts a decision into execution.
    scan_only = bool(base.get("scan_only", False))
    base["scan_only"] = scan_only
    exec_allowed = bool(base.get("execution_allowed", False))
    # HARD GUARD: scan-tier events can never be execution-allowed, and a restricted (NVDA) underlying
    # is never execution-allowed regardless of caller input. Execution-safety events (lease/order
    # guard) are RECORDS of the safety layer, never authority — always non-executable on store.
    if scan_only or base.get("restricted") or base["event_type"] in EXECUTION_SAFETY_EVENT_TYPES:
        exec_allowed = False
    base["execution_allowed"] = exec_allowed
    base["event_id"] = _compute_event_id(base)
    return base


def _append_jsonl_locked(path: Path, stored: dict, eid: str, dedupe: bool) -> str:
    """Critical section: dedupe-check + seq-assign + append under an exclusive advisory file lock so
    concurrent writers can't interleave or race the seq counter. Returns 'appended' | 'duplicate'.

    The lock is best-effort (POSIX flock); `seq` is advisory ordering metadata only — `event_id` is
    the authoritative identity used for de-duplication, so a missing lock degrades gracefully.
    """
    with open(path, "a+") as f:
        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass                      # best-effort; event_id remains authoritative
        f.seek(0)
        seq = 0
        existing: set[str] = set()
        for line in f:
            seq += 1
            try:
                o = json.loads(line)
            except Exception:
                continue
            if isinstance(o, dict) and o.get("event_id"):
                existing.add(o["event_id"])
        if dedupe and eid in existing:
            return "duplicate"
        stored["seq"] = seq
        f.write(json.dumps(stored, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())              # durable, and lock released on close
    return "appended"


def append_decision_journal(event: dict, *, source: str, event_type: str,
                            trade_date: str | None = None, journal_path: str | None = None,
                            now: datetime | None = None, dedupe: bool = True) -> dict:
    """Standardized, fail-safe, idempotent JSONL append for the decision journal.

    Returns a result dict ``{"status": appended|duplicate|error, "event_id": ..., "event": ...}``.
    NEVER raises — a journaling failure must not crash the trading/analysis loop (it logs + returns
    status="error"). De-dupe + seq + append happen under one advisory file lock; an event whose
    ``event_id`` already exists is skipped (status="duplicate"), which makes ingestion safe to re-run.
    """
    try:
        stored = build_decision_event(event, source=source, event_type=event_type,
                                      trade_date=trade_date, now=now)
        eid = stored["event_id"]
        path = Path(os.path.expanduser(journal_path or DEFAULT_JOURNAL_PATH))
        path.parent.mkdir(parents=True, exist_ok=True)
        status = _append_jsonl_locked(path, stored, eid, dedupe)
        return {"status": status, "event_id": eid, "event": stored}
    except Exception as exc:              # fail-safe: journaling must never break the caller
        logger.warning("append_decision_journal failed (%s): %s", event_type, exc)
        return {"status": "error", "event_id": None, "event": None, "error": str(exc)}


# --- loose-artifact ingestion --------------------------------------------------------------
# The Hermes/MCP controller drops timestamped loose JSON into data/odte/ (controller_*, event_*,
# candidate_*, market_snapshot_*, *vehicle_score*, *gamma_map*). build_report() only reads the
# JSONL, so those decisions are invisible post-day. This folds them into the journal IDEMPOTENTLY
# (re-runnable; deduped by raw_artifact_path) so a day can be fully reconstructed. Read-only over the
# source files — it never deletes or mutates them, never places orders, never executes anything.

# (lowercased filename glob, source class, default event_type, default mode). First match wins.
_ARTIFACT_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    ("market_snapshot_*.json", "market_snapshot", "market_snapshot", "watchdog"),
    ("candidate_*.json",       "candidate",       "candidate",        "candidate"),
    ("*vehicle_score*.json",   "vehicle_score",   "vehicle_score",    "candidate"),
    ("*gamma_map*.json",       "gamma_map",       "gamma_map",        "candidate"),
    # 2026-08-05: day-score payloads (incl. components_supplied/max_possible_score headroom
    # telemetry) were computed all morning and never journaled — no pattern matched them.
    ("*day_score*.json",       "day_score",       "day_score",        "watchdog"),
    ("controller_*.json",      "controller",      "controller_event", "execution"),
    ("event_*.json",           "event",           "controller_event", "execution"),
)

# Decision verbs we can safely infer from a filename when the payload doesn't state one. Conservative:
# anything order/entry-ish stays None unless the payload says so — ingestion must not invent executions.
_FILENAME_DECISION_HINTS = (("no_trade", "skip"), ("veto", "veto"), ("wait", "wait"),
                            ("skip", "skip"), ("hold", "hold"), ("closed", "exit"), ("exit", "exit"))


def _classify_artifact(name: str):
    import fnmatch
    low = name.lower()
    for glob_pat, klass, ev_type, mode in _ARTIFACT_PATTERNS:
        if fnmatch.fnmatch(low, glob_pat):
            return klass, ev_type, mode
    return None


# YYYYMMDD / YYYY_MM_DD / YYYY-MM-DD anywhere in a filename or field value.
_FILENAME_DATE_RE = re.compile(r"(20\d{2})[_-]?(0[1-9]|1[0-2])[_-]?(0[1-9]|[12]\d|3[01])")


def artifact_trade_date(fp: Path, payload: dict) -> str:
    """Best-effort trade date (YYYY-MM-DD) for a loose artifact: payload ts/date/trade_date first,
    then a date pattern in the FILENAME (so older filename-dated artifacts with no `ts` aren't all
    bucketed to today), then the file mtime as last resort."""
    for k in ("trade_date", "date", "ts"):
        v = payload.get(k)
        d = _parse_ts(v)
        if d:
            return d.date().isoformat()
        if isinstance(v, str):
            m = _FILENAME_DATE_RE.search(v)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _FILENAME_DATE_RE.search(fp.name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    try:
        return datetime.fromtimestamp(fp.stat().st_mtime, timezone.utc).date().isoformat()
    except Exception:
        return _derive_trade_date(None)


def _infer_decision(payload: dict, filename: str):
    """Best-effort decision verb + reason_codes from a loose artifact (payload first, filename hint
    second). Never infers an execution/enter verb from a filename — only the payload may assert that."""
    dec = payload.get("decision")
    if isinstance(dec, dict):
        dec = dec.get("action") or dec.get("decision")
    if isinstance(dec, str) and dec.strip():
        verb = dec.strip()
    else:
        verb = next((v for token, v in _FILENAME_DECISION_HINTS if token in filename.lower()), None)
    reasons = (payload.get("reason_codes") or payload.get("veto_reasons")
               or (payload.get("decision", {}).get("reasons") if isinstance(payload.get("decision"), dict) else None)
               or payload.get("reasons") or [])
    if not isinstance(reasons, list):
        reasons = [reasons]
    return verb, [str(r) for r in reasons if r]


def ingest_loose_artifacts(data_dir: str | None = None, journal_path: str | None = None,
                           trade_date: str | None = None, dry_run: bool = False) -> dict:
    """Fold loose data/odte/*.json controller artifacts into the decision journal, idempotently.

    Scans `data_dir` (default data/odte/) for the known artifact patterns, wraps each in the
    standardized envelope (`raw_artifact_path` set, so re-runs dedupe), and appends new ones. With
    `trade_date` (YYYY-MM-DD) only artifacts for that day are ingested. `dry_run` computes the
    summary without writing. Returns {files_scanned, events_appended, duplicates_skipped, errors,
    by_event_type}. Never raises."""
    summary = {"dry_run": bool(dry_run), "files_scanned": 0, "events_appended": 0,
               "events_would_append": 0, "duplicates_skipped": 0, "errors": 0,
               "by_event_type": {}, "error_files": []}
    try:
        ddir = Path(os.path.expanduser(data_dir or DEFAULT_DIR))
        if not ddir.exists():
            return summary
        seen = existing_event_ids(journal_path)            # idempotency vs the journal-on-disk
        run_seen: set[str] = set()                          # idempotency within this scan
        for fp in sorted(ddir.glob("*.json")):
            cls = _classify_artifact(fp.name)
            if cls is None:
                continue
            summary["files_scanned"] += 1
            _klass, default_type, default_mode = cls
            try:
                raw_bytes = fp.read_bytes()
                payload = json.loads(raw_bytes)
                if not isinstance(payload, dict):
                    raise ValueError("artifact is not a JSON object")
            except Exception as exc:
                summary["errors"] += 1
                summary["error_files"].append(f"{fp.name}: {exc}")
                continue
            ev_type = str(payload.get("event_type") or payload.get("type") or default_type)
            # LIFECYCLE PROTECTION (2026-08-06 EOD incident): a loose artifact whose payload
            # carries a trade-lifecycle event_type is a COPY of something the trading lane
            # already journaled — re-appending it under an ingest: source mints a fresh
            # event_id that dedupe cannot catch, and the money counters then double-count it
            # (the day packet briefly read +$28 on a +$14 day). The trading lanes are the ONLY
            # writers of lifecycle events; ingest refuses them outright.
            if ev_type in _INGEST_PROTECTED_LIFECYCLE:
                summary["lifecycle_skipped"] = summary.get("lifecycle_skipped", 0) + 1
                continue
            tdate = artifact_trade_date(fp, payload)
            if trade_date and tdate != trade_date:
                continue                                    # day filter: skip other days' artifacts
            verb, reasons = _infer_decision(payload, fp.name)
            # idempotency anchor: resolved path + content hash → a CHANGED file re-ingests as new.
            raw_sha = hashlib.sha1(raw_bytes).hexdigest()[:16]
            event = {
                **payload,
                "raw_artifact_path": str(fp.resolve()),
                "raw_artifact_sha": raw_sha,
                "mode": payload.get("mode") or default_mode,
                "decision": verb,
                "reason_codes": reasons,
                # AUDIT path: ingested artifacts are a RECORD of what already happened, NOT fresh
                # authority. execution_allowed is forced False regardless of the payload (the original
                # is preserved under raw_execution_allowed for inspection). scan_only is preserved.
                "raw_execution_allowed": bool(payload.get("execution_allowed", False)),
                "execution_allowed": False,
                "scan_only": bool(payload.get("scan_only", False)),
            }
            stored = build_decision_event(event, source=f"ingest:{_klass}", event_type=ev_type,
                                          trade_date=tdate)
            eid = stored["event_id"]
            if eid in seen or eid in run_seen:
                summary["duplicates_skipped"] += 1
                continue
            run_seen.add(eid)
            if dry_run:
                summary["events_would_append"] += 1
                summary["by_event_type"][ev_type] = summary["by_event_type"].get(ev_type, 0) + 1
                continue
            res = append_decision_journal(event, source=f"ingest:{_klass}", event_type=ev_type,
                                          trade_date=tdate, journal_path=journal_path)
            if res["status"] == "error":
                summary["errors"] += 1
                summary["error_files"].append(f"{fp.name}: append error")
                continue
            if res["status"] == "duplicate":
                summary["duplicates_skipped"] += 1
                continue
            summary["events_appended"] += 1
            summary["by_event_type"][ev_type] = summary["by_event_type"].get(ev_type, 0) + 1
    except Exception as exc:                                # whole-scan fail-safe
        logger.warning("ingest_loose_artifacts failed: %s", exc)
        summary["errors"] += 1
    return summary


# --- additive day packet -------------------------------------------------------------------
# A per-day fan-out of the journal into data/odte/days/YYYY-MM-DD/<stream>.jsonl, so a single day
# can be reviewed stream-by-stream. ADDITIVE + DERIVED: it only ever reads the authoritative journal
# and (re)writes the day folder — it never writes the journal, never touches the loose artifacts,
# and is OFF by default (built only when explicitly requested). Regenerating is idempotent.

DAY_PACKET_DIRNAME = "days"
_DAY_STREAMS = ("market_snapshots", "candidates", "vehicle_scores", "trades", "controller_events")


def _classify_day_stream(e: dict) -> str:
    """Route one journal event to its day-packet stream."""
    et = str(e.get("event_type") or "")
    src = str(e.get("source") or "")
    if et == "market_snapshot" or "market_snapshot" in src:
        return "market_snapshots"
    if et in ("candidate", "candidate_evaluation") or src == "ingest:candidate":
        return "candidates"
    if et == "vehicle_score" or "vehicle_score" in src or e.get("vehicle_score"):
        return "vehicle_scores"
    # TRADES = trade LIFECYCLE only (2026-08-05 fix): fills, exits, and the postmortem. The old
    # classifier also routed `management_check` (monitoring polls) and bare `entry_decision`
    # (gate verdicts, mostly veto/observe) here — Aug-3's packet held 1 real trade event out of
    # 29 rows and Aug-4's "trades" were 13 vetoes. Gate/monitoring telemetry belongs in
    # controller_events; a trades.jsonl row must be usable as P/L evidence.
    if et in ENTRY_FILL_EVENTS or et in EXIT_FILL_EVENTS or et == "postmortem":
        return "trades"
    return "controller_events"


def _day_postmortem_content(trade_date: str, buckets: dict) -> str:
    """Generated day-review draft from journal streams. It is intentionally conservative and only
    summarizes recorded events; no broker/LLM calls and no invented causes."""
    trades = buckets.get("trades", [])
    controllers = buckets.get("controller_events", [])
    postmortems = [e for e in controllers + trades if e.get("event_type") == "postmortem"]
    experiments = [e for e in controllers + trades if e.get("event_type") == "experiment" or _get(e, "hypothesis")]
    no_trades = [e for e in controllers if str(e.get("event_type") or "").endswith("no_trade_decision")]
    closed = [e for e in postmortems + trades if _num(e.get("realized_pnl_dollars_gross")) is not None]
    pnl = sum(_num(e.get("realized_pnl_dollars_gross")) or 0.0 for e in closed)

    lines = [f"# 0DTE postmortem — {trade_date}", "",
             "_Generated from the decision journal; edit freely._", "",
             "## What happened", ""]
    if closed:
        for e in closed:
            tid = e.get("trade_id") or e.get("contract") or e.get("underlying") or "trade"
            entry = e.get("entry_price")
            exit_price = e.get("exit_price") or e.get("close_price")
            rp = _num(e.get("realized_pnl_dollars_gross"))
            pct = _num(e.get("realized_pnl_pct_gross"))
            ret = f", {pct * 100:+.1f}%" if pct is not None else ""
            lines.append(f"- **{tid}**: entry `{entry}`, exit `{exit_price}`, gross **{rp:+.2f}**{ret}.")
        lines.append(f"- Day closed-trade gross P/L from recorded postmortems: **{pnl:+.2f}**.")
    else:
        lines.append("- No closed trades with realized P/L were recorded for this day packet.")
    if no_trades:
        lines.append(f"- No-trade decisions recorded: **{len(no_trades)}**.")

    worked = []
    risks = []
    durable = []
    for e in postmortems:
        worked.extend(str(x) for x in _list(e, "what_worked"))
        risks.extend(str(x) for x in _list(e, "what_failed_or_risked"))
        if e.get("durable_rule"):
            durable.append(str(e["durable_rule"]))
    lines += ["", "## What went well", ""]
    lines += [f"- {x}" for x in dict.fromkeys(worked)] or ["- No explicit `what_worked` notes recorded yet."]
    lines += ["", "## What did not go well / risks", ""]
    lines += [f"- {x}" for x in dict.fromkeys(risks)] or ["- No explicit risk notes recorded yet."]
    lines += ["", "## Experiments for tomorrow", ""]
    seen_exp = set()
    exp_lines = []
    for e in experiments:
        hyp = _get(e, "hypothesis")
        if not hyp:
            continue
        key = _dedupe_key(e.get("trade_id"), hyp, _get(e, "promote_if"), _get(e, "kill_if"))
        if key in seen_exp:
            continue
        seen_exp.add(key)
        exp_lines.append(f"- **{hyp}**")
        if _get(e, "promote_if"):
            exp_lines.append(f"  - Promote if: {_get(e, 'promote_if')}")
        if _get(e, "kill_if"):
            exp_lines.append(f"  - Kill if: {_get(e, 'kill_if')}")
    lines += exp_lines or ["- No experiments queued from journal events."]
    lines += ["", "## Keep / change next session", ""]
    if durable:
        lines += ["**Keep**", *[f"- {x}" for x in dict.fromkeys(durable)], ""]
    lines += ["**Change / inspect**",
              "- Treat 09:30–13:00 ET as the high-action observation window; it is context, not permission to trade.",
              "- Ignore stale candidates/gates after review/TTL; rebuild fresh thesis/gate before any new entry.",
              "", "## Canonical strategy guardrails", ""]
    for guardrail in canonical_guardrails():
        lines.append(f"- **{guardrail['title']}** (`{guardrail['key']}`): {guardrail['rule']}")
    return "\n".join(lines) + "\n"


def _day_postmortem_stub(trade_date: str, buckets: dict) -> str:
    return _day_postmortem_content(trade_date, buckets)


def _is_auto_postmortem(text: str) -> bool:
    return ("_Auto-scaffold from the decision journal" in text or
            "## What happened\n\n## Thesis right or wrong?" in text)


def build_day_packet(trade_date: str | None = None, journal_path: str | None = None,
                     out_root: str | None = None) -> dict:
    """(Re)build the additive day packet for `trade_date` (default: today UTC) from the journal.

    Returns {trade_date, files: {stream: n}, events_written}. Never raises. The stream .jsonl files
    are overwritten each call (they are a derived view); postmortem.md is created only if missing so
    human edits are preserved."""
    td = trade_date or datetime.now(timezone.utc).date().isoformat()
    summary = {"trade_date": td, "files": {}, "events_written": 0}
    try:
        events = [e for e in read_events(journal_path) if e.get("trade_date") == td]
        root = Path(os.path.expanduser(out_root or DEFAULT_DIR)) / DAY_PACKET_DIRNAME / td
        root.mkdir(parents=True, exist_ok=True)
        buckets: dict[str, list] = defaultdict(list)
        for e in events:
            buckets[_classify_day_stream(e)].append(e)
        for stream in _DAY_STREAMS:
            evs = buckets.get(stream, [])
            fp = root / f"{stream}.jsonl"
            with open(fp, "w") as f:
                for e in evs:
                    f.write(json.dumps(e, default=str) + "\n")
            summary["files"][f"{stream}.jsonl"] = len(evs)
            summary["events_written"] += len(evs)
        pm = root / "postmortem.md"
        generated = _day_postmortem_content(td, buckets)
        if not pm.exists():
            pm.write_text(generated)
            summary["files"]["postmortem.md"] = 1
        elif _is_auto_postmortem(pm.read_text()):
            pm.write_text(generated)
            summary["files"]["postmortem.md"] = 1
        else:
            gp = root / "postmortem.generated.md"
            gp.write_text(generated)
            summary["files"]["postmortem.generated.md"] = 1
    except Exception as exc:
        logger.warning("build_day_packet failed (%s): %s", td, exc)
        summary["error"] = str(exc)
    return summary


def event_from_position_decision(position_payload: dict, trade_id: str | None = None,
                                 extra: dict | None = None) -> dict:
    """Convert an `odte-position` decision payload into a `management_check` journal event (a plain
    dict the caller can pass to append_event). Does NOT auto-append — keeps the seam explicit."""
    p = position_payload or {}
    trigs = p.get("triggers") or []
    e = {
        "event_type": "management_check",
        "trade_id": trade_id or p.get("trade_id"),
        "underlying": p.get("underlying"),
        "mode": p.get("mode"),
        "option_id": p.get("option_id"),
        "decision": {"action": p.get("decision"),
                     "reasons": [t.get("detail") for t in trigs if t.get("detail")]},
        "pnl_pct": p.get("pnl_pct"),
        "triggers": [t.get("type") for t in trigs],
    }
    if extra:
        e.update(extra)
    return e


def event_from_entry_gate(gate_decision: dict, trade_id: str | None = None,
                          extra: dict | None = None) -> dict:
    """Convert an `odte_entry_gate.build_entry_gate_decision` record into an `entry_decision` journal
    event (a plain dict the caller passes to append_event/append_decision_journal). Records the
    thesis → entry gate: the intent verb, the gate results, veto/reason codes, and the
    `scan_only`/`execution_allowed` tier flags. Does NOT auto-append — keeps the seam explicit
    (parallel to `event_from_position_decision`/`event_from_vehicle_score`).

    DEFENSE IN DEPTH: even though the gate already computed `execution_allowed` conservatively, the
    flag flows through `build_decision_event`, which re-enforces `scan_only/restricted ⇒
    execution_allowed=False` at write time. The underlying is normalized/restriction-tagged at append.
    """
    g = gate_decision or {}
    e = {
        "event_type": "entry_decision",
        "trade_id": trade_id or g.get("trade_id"),
        "underlying": g.get("symbol") or g.get("underlying"),
        "direction": g.get("direction"),
        "decision": {"action": g.get("decision") or g.get("intent"),
                     "reasons": list(g.get("reason_codes") or [])},
        "thesis": g.get("thesis"),
        "confidence": g.get("confidence"),
        "reason_codes": list(g.get("reason_codes") or []),
        "veto_reasons": list(g.get("veto_reasons") or []),
        "required_confirmations": list(g.get("required_confirmations") or []),
        "confirmations": dict(g.get("confirmations") or {}),
        "confirmation_needed": g.get("confirmation_needed"),
        "candidate_fingerprint": g.get("candidate_fingerprint"),
        "option_id": g.get("option_id"),
        "expiration_date": g.get("expiration_date"),
        "strike_price": g.get("strike_price"),
        "option_type": g.get("option_type"),
        "gates": g.get("gates"),
        "tier": g.get("tier"),
        "sizing_tier": g.get("sizing_tier"),
        "scan_only": bool(g.get("scan_only", False)),
        "execution_allowed": bool(g.get("execution_allowed", False)),
    }
    # Green-day preservation marker: only a gate that EXPLICITLY armed the re-entry override may
    # survive the post-scalp lockout in odte_loop_status — so the flag must ride on the event.
    if g.get("allow_reentry_after_green") is True:
        e["allow_reentry_after_green"] = True
    if g.get("green_day_preservation"):
        e["green_day_preservation"] = g.get("green_day_preservation")
    if extra:
        e.update(extra)
    return e


def event_from_candidate_evaluation(cd_payload: dict, trade_id: str | None = None,
                                    extra: dict | None = None) -> dict:
    """Convert an `evaluate_candidate_watch` payload into a `candidate_evaluation` journal event.

    The candidate watch computes a full `checks` dict on EVERY tick — VWAP side, ORB state,
    confirmer/dissenter counts, the VIXY read, the tier it would have minted, minutes to close —
    and then discards everything except the verdict. Only the last tick of the day survived, in an
    overwritten artifact, so questions of the form "how many setups had the tape but failed one
    clause?" were unanswerable and every gate-loosening decision stayed a judgment call.

    Recording the checks makes the near-miss population countable. It is TELEMETRY: the event is
    `scan_only` with `execution_allowed=False` by construction and never feeds a gate. Does NOT
    auto-append — the seam stays explicit, as with `event_from_entry_gate`."""
    p = cd_payload or {}
    cand = p.get("candidate") if isinstance(p.get("candidate"), dict) else {}
    checks = p.get("checks") if isinstance(p.get("checks"), dict) else {}
    e = {
        "event_type": "candidate_evaluation",
        "trade_id": trade_id or p.get("trade_id"),
        "underlying": cand.get("ticker") or cand.get("underlying") or cand.get("symbol"),
        "direction": cand.get("direction"),
        "state": p.get("state"),
        "decision": p.get("decision"),
        "reasons": list(p.get("reasons") or []),
        # Verbatim — the whole point is that the individual clause results survive.
        "checks": dict(checks),
        "tier": checks.get("tier") or cand.get("tier"),
        "candidate_fingerprint": cand.get("candidate_fingerprint"),
        "option_id": cand.get("option_id") or cand.get("id") or cand.get("occ_symbol"),
        "expiration_date": cand.get("expiration_date"),
        "strike_price": cand.get("strike_price"),
        "option_type": cand.get("option_type"),
        "generated_at": p.get("generated_at"),
        "scan_only": True,
        "execution_allowed": False,
    }
    if extra:
        e.update(extra)
    return e


def event_from_day_score(score_payload: dict, extra: dict | None = None) -> dict:
    """Convert an `odte-day-score` payload into a `day_score` journal event, including the headroom
    telemetry (`components_supplied`/`components_missing`/`max_possible_score`) that says whether a
    GOOD_DAY was even reachable. Does NOT auto-append."""
    p = score_payload or {}
    e = {
        "event_type": "day_score",
        "decision": p.get("verdict"),
        "score": p.get("score"),
        "verdict": p.get("verdict"),
        "components": dict(p.get("components") or {}),
        "components_supplied": p.get("components_supplied"),
        "components_missing": list(p.get("components_missing") or []),
        "max_possible_score": p.get("max_possible_score"),
        "reasons": list(p.get("reasons") or []),
        "basis": p.get("basis"),
        "generated_at": p.get("generated_at"),
        "scan_only": True,
        "execution_allowed": False,
    }
    if extra:
        e.update(extra)
    return e


def event_from_vehicle_score(score_payload: dict, trade_id: str | None = None,
                             extra: dict | None = None) -> dict:
    """Convert a `score_vehicle` payload into a `vehicle_score` journal event (a plain dict the
    caller can pass to append_event). Records the non-sentiment GOOD_BET/WATCH/BAD_BET verdict,
    its components/score, the reasons, and the ADDITIVE `bp_fit` block — the gate's own budget
    arithmetic (tier fraction x BP, max affordable ask, fits) surfaced at SELECTION time. Does
    NOT auto-append — keeps the seam explicit. The underlying is restriction-tagged at append.

    Emits `vehicle_score`, NOT `pre_trade_thesis`: the old label was wrong (that type belongs to
    hand-authored thesis notes), and it meant the scorer's own stream was unqueryable.

    NO `trade_id` unless one is explicitly passed, and the key is OMITTED rather than set None:
    `summarize()` opens a trade row for ANY event whose trade_id `is not None`, so a scan-time
    score carrying one would inflate n_trades and every rate derived from it. Omitting also
    guards the `""` case, which is not None and would open a row keyed on the empty string."""
    p = score_payload or {}
    c = p.get("contract") if isinstance(p.get("contract"), dict) else {}
    e = {
        "event_type": "vehicle_score",
        "underlying": c.get("underlying") or c.get("symbol"),
        "option_type": c.get("option_type") or c.get("type"),
        "strike": c.get("strike") or c.get("strike_price"),
        "option_id": c.get("option_id") or c.get("id"),
        "verdict": p.get("verdict"),
        "score": p.get("score"),
        "decision": {"action": p.get("verdict"),
                     "reasons": list(p.get("reasons") or [])},
        "vehicle_score": {"verdict": p.get("verdict"), "score": p.get("score"),
                          "components": p.get("components"), "direction": p.get("direction")},
        "bp_fit": p.get("bp_fit"),
        "reasons": list(p.get("reasons") or []),
        "generated_at": p.get("generated_at"),
        "scan_only": True,
        "execution_allowed": False,
    }
    tid = trade_id or p.get("trade_id")
    if tid:
        e["trade_id"] = tid
    if extra:
        e.update(extra)
    return e


# --- execution-safety event converters (lease + order guard) ---------------------------------
# Parallel to event_from_entry_gate/event_from_position_decision: PURE converters that build (but
# do not append) journal events from the execution-safety payloads. The journal is a RECORD, never
# authority — `execution_allowed` stays False on these events by construction.

def event_from_execution_lease(auth_payload: dict, trade_id: str | None = None,
                               extra: dict | None = None) -> dict:
    """Convert an `odte-execution-authorize` payload into an `execution_lease_issued` journal
    event (authorized) or a deny record (refused). The policy values and accepted maximum loss are
    serialized into the event, per the execution-safety contract. Does NOT auto-append."""
    p = auth_payload or {}
    lease = p.get("lease") if isinstance(p.get("lease"), dict) else {}
    e = {
        "event_type": "execution_lease_issued",
        "trade_id": trade_id,
        "underlying": lease.get("symbol") or p.get("symbol"),
        "direction": lease.get("direction") or p.get("direction"),
        "option_id": lease.get("option_id"),
        "option_type": lease.get("option_type"),
        "strike": lease.get("strike_price"),
        "expiration": lease.get("expiration_date"),
        "quantity": lease.get("quantity") or p.get("quantity"),
        "lease_id": lease.get("lease_id"),
        "issued_at": lease.get("issued_at"),
        "expires_at": lease.get("expires_at"),
        "max_limit_price": lease.get("max_limit_price"),
        "max_debit": lease.get("max_debit") or p.get("debit"),
        "risk_mode": lease.get("risk_mode") or p.get("risk_mode"),
        "max_premium_loss": lease.get("max_premium_loss"),
        "tier": lease.get("tier"),
        "anchor_quote": lease.get("anchor_quote"),
        "candidate_fingerprint": lease.get("candidate_fingerprint"),
        "market_fingerprint": lease.get("market_fingerprint"),
        "policy": p.get("policy"),
        "authorized": bool(p.get("authorized")),
        "decision": {"action": "allow" if p.get("authorized") else "deny",
                     "reasons": list(p.get("reason_codes") or [])},
        "reason_codes": list(p.get("reason_codes") or []),
        # A journal record of the lease is never itself execution authority.
        "scan_only": False,
        "execution_allowed": False,
    }
    if extra:
        e.update(extra)
    return e


# odte_order_guard state → journal event type. FILLED_FRESH reuses the existing `order_filled`
# type; both cancel flavors journal as `entry_order_cancelled_stale` (the reason names which rail
# fired); every incident state journals as `execution_safety_incident`. NO_ORDER journals nothing.
_GUARD_EVENT_TYPES = {
    "PENDING_FRESH": "entry_order_pending",
    "CANCEL_STALE_ENTRY": "entry_order_cancelled_stale",
    "CANCEL_THESIS_INVALID": "entry_order_cancelled_stale",
    "FILLED_FRESH": "order_filled",
    "FILLED_WITHOUT_VALID_LEASE": "execution_safety_incident",
    "BROKER_MISMATCH_BLOCKED": "execution_safety_incident",
}


def event_from_order_guard(guard_payload: dict, trade_id: str | None = None,
                           extra: dict | None = None) -> dict | None:
    """Convert an `odte-order-guard` decision into its journal event (see _GUARD_EVENT_TYPES).
    Returns None for NO_ORDER (nothing to record). Does NOT auto-append."""
    p = guard_payload or {}
    state = str(p.get("state") or "").upper()
    event_type = _GUARD_EVENT_TYPES.get(state)
    if event_type is None:
        return None
    order = p.get("order") if isinstance(p.get("order"), dict) else {}
    e = {
        "event_type": event_type,
        "trade_id": trade_id,
        "underlying": p.get("underlying") or order.get("symbol"),
        "option_id": order.get("option_id"),
        "option_type": order.get("option_type"),
        "strike": order.get("strike_price"),
        "expiration": order.get("expiration_date"),
        "quantity": order.get("quantity"),
        "guard_state": state,
        "lease_id": p.get("lease_id"),
        "order_status": order.get("status"),
        "limit_price": order.get("limit_price"),
        "submitted_at": order.get("submitted_at"),
        "filled_at": order.get("filled_at"),
        "cancel_required": bool(p.get("cancel_required")),
        "safety_incident": bool(p.get("safety_incident")),
        "prohibit_new_entries": bool(p.get("prohibit_new_entries")),
        "pending_order_age_seconds": p.get("pending_order_age_seconds"),
        "lease_seconds_remaining": p.get("lease_seconds_remaining"),
        "decision": {"action": ("veto" if p.get("safety_incident")
                                else "exit" if p.get("cancel_required") else "hold"),
                     "reasons": list(p.get("reasons") or [])},
        "reason_codes": list(p.get("reasons") or []),
        "scan_only": False,
        "execution_allowed": False,
    }
    if extra:
        e.update(extra)
    return e


# --- sentiment / gamma status rollups --------------------------------------------------------

# Canonical status vocabularies + back-compat aliases. The live Hermes journal logs status either
# as a nested snapshot OR as flat top-level fields (sentiment_status/sentiment_state,
# gamma_status/gamma_pin_state) and as free-text freshness/pulse strings older logs used before the
# nested blocks existed. Normalizing them here keeps today's events from being invisible in the
# rollup while preserving the honest "no dealer GEX" basis below.
_SENTIMENT_STATUS_CANON = {"useful_context", "aligned_confirming", "diverged_warning",
                           "stale_unavailable", "not_used"}
_SENTIMENT_STATUS_ALIASES = {
    "useful": "useful_context", "context": "useful_context", "used": "useful_context",
    "aligned": "aligned_confirming", "confirming": "aligned_confirming",
    "confirm": "aligned_confirming", "agree": "aligned_confirming", "fresh": "useful_context",
    "diverged": "diverged_warning", "divergent": "diverged_warning",
    "diverge": "diverged_warning", "warning": "diverged_warning", "conflict": "diverged_warning",
    "stale": "stale_unavailable", "unavailable": "stale_unavailable",
    "missing": "stale_unavailable", "none": "stale_unavailable", "no_data": "stale_unavailable",
    "not_used": "not_used", "unused": "not_used", "off": "not_used",
    "skip": "not_used", "skipped": "not_used", "disabled": "not_used",
}
_GAMMA_STATUS_CANON = {"available", "unavailable_no_export", "source_limited",
                       "unavailable", "stale", "not_used"}
_GAMMA_STATUS_ALIASES = {
    "available": "available", "ok": "available", "fresh": "available",
    "unknown_no_export_available": "unavailable_no_export",
    "unknown_no_robinhood_export_for_odte_gamma_map": "unavailable_no_export",
    "no_export": "unavailable_no_export", "no_export_available": "unavailable_no_export",
    "no_robinhood_export": "unavailable_no_export",
    "source_limit": "source_limited", "source_limited": "source_limited",
    "limited": "source_limited", "source_capped": "source_limited",
    "unavailable": "unavailable", "missing": "unavailable", "none": "unavailable",
    "no_data": "unavailable",
    "stale": "stale", "quotes_stale": "stale", "quote_stale": "stale",
    "not_used": "not_used", "unused": "not_used", "off": "not_used", "disabled": "not_used",
}


def _norm_key(value) -> str | None:
    if value in (None, ""):
        return None
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return key or None


def _text(value) -> str | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    return s or None


def _norm_sentiment_status(value) -> str | None:
    key = _norm_key(value)
    if key is None:
        return None
    if key in _SENTIMENT_STATUS_CANON:
        return key
    return _SENTIMENT_STATUS_ALIASES.get(key)


def _norm_gamma_status(value) -> str | None:
    key = _norm_key(value)
    if key is None:
        return None
    if key in _GAMMA_STATUS_CANON:
        return key
    if key in _GAMMA_STATUS_ALIASES:
        return _GAMMA_STATUS_ALIASES[key]
    # Substring fallbacks for unmapped variants of the live pin-state strings.
    if "no_export" in key or "no_robinhood" in key:
        return "unavailable_no_export"
    if "source_limit" in key or "source_cap" in key:
        return "source_limited"
    if "stale" in key:
        return "stale"
    if "not_used" in key or "unused" in key:
        return "not_used"
    if "unknown" in key or "unavailable" in key or "missing" in key:
        return "unavailable"
    return None


def _sentiment_status(events: list[dict]) -> dict:
    """Roll up any sentiment readings attached to journal events. Pure: no IO/network.

    A reading is produced from a nested `sentiment` snapshot (an `odte-social-report` reading with
    verdict / direction|intent / confidence / sentiment(score) / mentions) AND/OR the flat
    back-compat fields the live journal writes: `sentiment_status` / `sentiment_state` and the
    free-text `social_freshness` / `thesis.social_pulse` context. So an event carrying only
    `social_freshness` still surfaces a status row instead of being invisible. Restricted-underlying
    readings are tagged and kept out of the latest read + distributions (never a forward bias) but
    surfaced in `restricted_readings`."""
    readings: list[dict] = []
    for e in events:
        snap = e.get("sentiment")
        snap = snap if isinstance(snap, dict) else {}
        status_raw = (e.get("sentiment_status") or e.get("sentiment_state")
                      or snap.get("status") or snap.get("state"))
        freshness = e.get("social_freshness") or snap.get("social_freshness")
        pulse = _get(e, "social_pulse")                 # top-level or thesis.social_pulse
        if not (snap or status_raw or freshness or pulse):
            continue
        status = _norm_sentiment_status(status_raw) or _norm_sentiment_status(freshness)
        direction = snap.get("direction") or snap.get("intent")
        score = _num(snap.get("sentiment"))
        readings.append({
            "seq": e.get("seq"), "ts": e.get("ts"), "trade_id": e.get("trade_id"),
            "underlying": (str(e.get("underlying")).upper() if e.get("underlying") else None),
            "restricted": bool(e.get("restricted")),
            "verdict": snap.get("verdict"),
            "direction": str(direction).lower() if direction else None,
            "confidence": snap.get("confidence"),
            "score": score,
            "mentions": _num(snap.get("mentions")),
            "status": status,
            "context": _text(status_raw) or _text(freshness) or _text(pulse),
        })
    readings.sort(key=lambda r: (r.get("seq") or 0))
    measurable = [r for r in readings if not r["restricted"]]
    scores = [r["score"] for r in measurable if r["score"] is not None]
    return {
        "n_readings": len(readings),
        "latest": measurable[-1] if measurable else None,
        "by_verdict": dict(Counter(r["verdict"] for r in measurable if r["verdict"])),
        "by_direction": dict(Counter(r["direction"] for r in measurable if r["direction"])),
        "by_status": dict(Counter(r["status"] for r in measurable if r["status"])),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
        "restricted_readings": sorted({r["underlying"] for r in readings
                                       if r["restricted"] and r["underlying"]}),
    }


def _gamma_status(events: list[dict]) -> dict:
    """Roll up any gamma readings attached to journal events. Pure: no IO/network.

    A reading is produced from a nested `gamma` snapshot (an `odte-gamma-map` output: pin_risk{level},
    max_gamma_strike, call/put walls, gamma_available, freshness{quote_fresh}) AND/OR the flat
    back-compat fields the live journal writes: `gamma_status` and `gamma_pin_state` strings such as
    `unknown_no_export_available` / `unknown_no_robinhood_export_for_odte_gamma_map` (both normalized
    to `unavailable_no_export`). So a no-export event still surfaces a status row. HONEST by
    construction — this is ABSOLUTE pin-risk concentration only; it never reports (or infers) dealer
    net GEX / gamma flip / sign. `includes_dealer_positioning` is always False for consumers."""
    readings: list[dict] = []
    for e in events:
        snap = e.get("gamma")
        snap = snap if isinstance(snap, dict) else {}
        explicit = e.get("gamma_status") or snap.get("status") or snap.get("gamma_status")
        pin_state = (e.get("gamma_pin_state") or snap.get("gamma_pin_state")
                     or snap.get("pin_state"))
        if not (snap or explicit or pin_state):
            continue
        status = _norm_gamma_status(explicit) or _norm_gamma_status(pin_state)
        pin = snap.get("pin_risk")
        pin_level = pin.get("level") if isinstance(pin, dict) else snap.get("pin_risk_level")
        fresh = snap.get("freshness")
        quote_fresh = fresh.get("quote_fresh") if isinstance(fresh, dict) else snap.get("quote_fresh")
        readings.append({
            "seq": e.get("seq"), "ts": e.get("ts"), "trade_id": e.get("trade_id"),
            "underlying": (str(e.get("underlying")).upper() if e.get("underlying") else None),
            "restricted": bool(e.get("restricted")),
            "regime": snap.get("gamma_regime") or _GAMMA_BASIS,
            "pin_risk": str(pin_level).lower() if pin_level else None,
            "max_gamma_strike": _num(snap.get("max_gamma_strike")),
            "call_wall": _num(snap.get("call_wall")),
            "put_wall": _num(snap.get("put_wall")),
            "gamma_available": snap.get("gamma_available"),
            "quote_fresh": quote_fresh,
            "status": status,
            "context": _text(pin_state) or _text(explicit),
        })
    readings.sort(key=lambda r: (r.get("seq") or 0))
    measurable = [r for r in readings if not r["restricted"]]
    latest = measurable[-1] if measurable else None
    return {
        "n_readings": len(readings),
        "latest": latest,
        "by_pin_risk": dict(Counter(r["pin_risk"] for r in measurable if r["pin_risk"])),
        "by_status": dict(Counter(r["status"] for r in measurable if r["status"])),
        # Preserve the honest label from the source map; never a dealer-positioning number.
        "regime": (latest or {}).get("regime", _GAMMA_BASIS),
        "includes_dealer_positioning": False,
        "restricted_readings": sorted({r["underlying"] for r in readings
                                       if r["restricted"] and r["underlying"]}),
    }


# --- summarize -------------------------------------------------------------------------------

# Loss-cause taxonomy (explicit tags only — we never guess a cause from P/L alone).
LOSS_CATEGORIES = ("execution", "thesis", "timing", "vehicle", "risk", "regime")


def _process_outcome(t: dict) -> str | None:
    """One of the four process×outcome cells for a CLOSED trade. Modern postmortems (2026-08-03+)
    STATE `process_quality`/`outcome_quality` explicitly — use them; older ones only recorded rule
    violations, so derive from rule_violations + win. Either way the answer lands in the same four
    cells, which distinguishes a good process that still lost (variance) from a bad process that
    happened to win (lucky — the dangerous one)."""
    if not t["closed"]:
        return None
    pq, oq = str(t.get("process_quality") or ""), str(t.get("outcome_quality") or "")
    if pq and oq:
        good_process, win = pq.startswith("good"), oq.startswith("good")
    else:
        good_process, win = (not t["rule_violations"]), t["win"]
    if good_process:
        return "good_process_good_outcome" if win else "good_process_bad_outcome"
    return "bad_process_lucky_outcome" if win else "bad_process_bad_outcome"


def _execution_diagnosis(t: dict) -> str:
    """Entry/exit/thesis diagnosis for a closed trade. Prefers an EXPLICIT `diagnosis` field; else
    derives conservatively from MFE (max favorable excursion) vs realized P/L:
      • won and kept ≥half the favorable move      -> clean_win
      • won but captured <half of it                -> good_entry_bad_exit (left money on the table)
      • lost but WAS favorable at some point        -> good_thesis_bad_exit (round-tripped a winner)
      • lost and never went favorable               -> thesis_wrong (the read itself was off)
    Anything without the data to tell -> unclassified (never fabricated)."""
    if t.get("diagnosis"):
        return str(t["diagnosis"])
    realized, mfe = t["realized_pnl"], t["mfe"]
    if realized is None:
        return "unclassified"
    cap = (realized / mfe) if (mfe and mfe > 0) else None
    if realized > 0:
        if cap is not None:
            return "clean_win" if cap >= 0.5 else "good_entry_bad_exit"
        return "clean_win"
    if mfe is not None and mfe > 0:
        return "good_thesis_bad_exit"
    if mfe is not None and mfe <= 0:
        return "thesis_wrong"
    return "unclassified"


def _process_quality(trade_rows: list[dict]) -> dict:
    """Self-eval rollup over measurable (non-restricted) trades: process×outcome cells, entry/exit/
    thesis diagnosis counts, and loss causes (explicit tags; losers without a tag -> 'uncategorized').
    All derived from fields already on the events — no hallucinated causes."""
    measurable = [t for t in trade_rows if not t["restricted"] and t["closed"]]
    process_outcome: Counter = Counter()
    diagnosis: Counter = Counter()
    loss_categories: Counter = Counter()
    failure_layers: Counter = Counter()
    for t in measurable:
        po = _process_outcome(t)
        if po:
            process_outcome[po] += 1
        diagnosis[_execution_diagnosis(t)] += 1
        # `failure_layer` (2026-08-03 postmortems) names WHICH layer failed — a different axis from
        # the entry/exit/thesis diagnosis above, so it is counted separately rather than folded in
        # and mixing two vocabularies in one histogram.
        # "none" and its qualified spellings ("none_normal_execution_slippage") are the author
        # saying nothing failed — counting them would invent failures out of clean trades.
        layer = str(t.get("failure_layer") or "").strip().lower()
        if layer and not layer.startswith("none"):
            failure_layers[layer] += 1
        if t["realized_pnl"] is not None and t["realized_pnl"] < 0:
            lc = t.get("loss_category")
            loss_categories[lc if lc in LOSS_CATEGORIES else "uncategorized"] += 1
    return {
        "n_diagnosed": len(measurable),
        "process_outcome": dict(process_outcome),
        "execution_diagnosis": dict(diagnosis),
        "failure_layers": dict(failure_layers),
        "loss_categories": dict(loss_categories),
    }


def _exit_fill_net(events: list[dict]) -> float | None:
    """Realized P&L as stated by the exit lane's own fill event.

    ``order_closed`` carries ``realized_pnl``; ``exit_fill`` states the same close under
    ``estimated_net_pnl`` (net of the fee estimate) or ``gross_pnl``. The ledger reads only the
    former, and on the live path ``order_closed`` is agent-authored — on 2026-08-12 it was simply
    never written, so a broker-flat-verified -$20.00 close left the ledger reading closed=17 /
    $75.00 while the account was flat. Same failure as the mfe/mae note below: the schema grew a
    second spelling and the reader kept reading one.

    "estimated" qualifies the *fees*, not the fill — these events carry a real ``exit_order_id``
    and ``fill_price``. Explicit ``realized_pnl`` still wins wherever it exists.
    """
    for e in reversed(events or []):
        if str(e.get("event_type")) in EXIT_FILL_EVENTS:
            pnl = _realized(e)
            if pnl is not None:
                return pnl
    return None


_EXCURSION_POLL_EVENTS = ("management_check", "position_check")


def excursion_from_polls(events: list[dict]) -> dict[str, dict]:
    """Measure per-contract excursion from the monitoring polls the tools already emit.

    ``management_check`` carries ``option_id`` and a tool-computed ``pnl_pct`` on every poll, but
    as of 2026-08-12 only 10 of 455 carry ``trade_id`` — so the poll stream joins to a trade
    through the *contract* id, not the trade id. Grouping these by ``trade_id`` silently yields
    zero polls for nearly every trade; grouping by (date, underlying) instead merges two trades
    that share a symbol on one day, which is what 2026-08-11's pair of SPY trades did.

    This exists because the postmortem's own ``excursion`` block is agent-authored. Across the 22
    trades whose polls join, it was absent for 15 and reported a *positive* MAE — impossible — for
    another. Where it is present and polls are plentiful the two agree inside 2.5pp, so this
    measures the same quantity and simply always has it.

    Returns ``{option_id: {"mae_pct", "mfe_pct", "polls"}}`` as fractions of entry (-0.19, not
    -19.0), deliberately NOT the units of the dollar-denominated ``mfe``/``mae`` trade fields.
    ``polls`` is part of the contract: one poll cannot measure an excursion, and a lone poll is
    exactly how 2026-08-06's IWM trade reported a +4.67% "MAE".
    """
    seen: dict[str, list[float]] = defaultdict(list)
    for e in events or []:
        if str(e.get("event_type")) not in _EXCURSION_POLL_EVENTS:
            continue
        oid, pnl = e.get("option_id"), e.get("pnl_pct")
        if not oid or pnl is None:
            continue
        try:
            seen[str(oid)].append(float(pnl))
        except (TypeError, ValueError):
            continue
    return {
        oid: {"mae_pct": min(vals), "mfe_pct": max(vals), "polls": len(vals)}
        for oid, vals in seen.items()
    }


def summarize(events: list[dict]) -> dict:
    """Deterministic metrics over the journal. Pure: no IO, no network."""
    events = sorted(list(events or []), key=lambda e: (e.get("seq", 0)))
    by_type = dict(Counter(str(e.get("event_type", "?")) for e in events))
    measured_excursions = excursion_from_polls(events)

    trades: dict[str, list] = defaultdict(list)
    for e in events:
        tid = e.get("trade_id")
        if tid is not None:
            trades[str(tid)].append(e)

    trade_rows: list[dict] = []
    restricted_flags: list[str] = []
    for tid, evs in trades.items():
        evs = sorted(evs, key=lambda e: (e.get("seq", 0)))
        restricted = any(e.get("restricted") for e in evs)
        underlying = str(_first(evs, "underlying") or "").upper()
        if restricted and underlying:
            restricted_flags.append(underlying)
        realized = _last_num(evs, "realized_pnl")
        if realized is None:
            realized = _exit_fill_net(evs)
        # Modern (2026-08-03+) postmortems carry excursion.{mfe,mae}_dollars + mfe_capture_pct;
        # older ones carry flat mfe/mae. Read modern first, fall back to legacy — reading only the
        # legacy keys is what made avg_mfe_capture report 7.7% on a 100%-capture trade.
        mfe = _last_sub_num(evs, "excursion", "mfe_dollars")
        if mfe is None:
            mfe = _last_num(evs, "mfe")
        mae = _last_sub_num(evs, "excursion", "mae_dollars")
        if mae is None:
            mae = _last_num(evs, "mae")
        violations = [v for e in evs for v in _list(e, "rule_violations")]
        violations += [v for e in evs for v in _sub_list(e, "process_review", "rule_violations")]
        if restricted:
            violations.append(f"RESTRICTED_EMPLOYER: {underlying or 'restricted'} must never be traded")
        entry_ts = next((_parse_ts(e.get("ts")) for e in evs if e.get("event_type") in _ENTRY_EVENTS), None)
        measured = measured_excursions.get(str(_first(evs, "option_id") or "")) or {}
        trade_rows.append({
            "trade_id": tid, "mode": str(_first(evs, "mode") or "unknown"),
            "underlying": underlying, "restricted": restricted,
            "realized_pnl": realized, "mfe": mfe, "mae": mae,
            # Tool-measured excursion as FRACTIONS of entry — deliberately separate fields from
            # the dollar-denominated agent-authored mfe/mae above, because `realized_pnl / mfe`
            # downstream is a dollar ratio and a percentage there would read as a 100x error.
            "mae_pct_measured": measured.get("mae_pct"),
            "mfe_pct_measured": measured.get("mfe_pct"),
            "excursion_polls": measured.get("polls", 0),
            "closed": realized is not None, "win": realized is not None and realized > 0,
            "rule_violations": violations, "held_minutes": _held_minutes(evs),
            "entry_ts": entry_ts.isoformat() if entry_ts else None,
            "time_bucket": _time_bucket(entry_ts.isoformat() if entry_ts else _first(evs, "ts")),
            # explicit self-eval fields (used as-is when present; never fabricated):
            "loss_category": _first(evs, "loss_category"),
            "diagnosis": _first(evs, "diagnosis"),
            # 2026-08-03 schema: the postmortem states capture and its own grades outright.
            "mfe_capture_pct": _last_sub_num(evs, "excursion", "mfe_capture_pct"),
            "process_quality": _first(evs, "process_quality"),
            "outcome_quality": _first(evs, "outcome_quality"),
            "failure_layer": _first(evs, "failure_layer"),
            "rail_fired": _first(evs, "rail_fired"),
            "tier": _first(evs, "tier") or _first_sub(evs, "entry", "tier"),
        })

    # Restricted trades are excluded from performance metrics (they should never exist) but the flag
    # and the synthetic violation above keep them loudly visible.
    measurable = [t for t in trade_rows if not t["restricted"]]
    closed = [t for t in measurable if t["closed"]]
    wins = [t for t in closed if t["win"]]
    # MFE capture: the postmortem's own `excursion.mfe_capture_pct` is authoritative when present
    # (it is computed against the best bid actually seen, not against a P/L ratio); older trades
    # only allow the realized/MFE derivation.
    caps = []
    for t in closed:
        if t["mfe_capture_pct"] is not None:
            caps.append(t["mfe_capture_pct"] / 100.0)
        elif t["mfe"] and t["mfe"] > 0 and t["realized_pnl"] is not None:
            caps.append(t["realized_pnl"] / t["mfe"])
    held = [t["held_minutes"] for t in closed if t["held_minutes"] is not None]

    by_mode: dict[str, dict] = {}
    by_time_bucket: dict[str, dict] = {}
    for t in measurable:
        b = by_time_bucket.setdefault(t["time_bucket"],
                                      {"trades": 0, "closed": 0, "wins": 0, "realized_pnl": 0.0})
        b["trades"] += 1
        if t["closed"]:
            b["closed"] += 1
            b["realized_pnl"] += t["realized_pnl"]
            b["wins"] += int(t["win"])
        d = by_mode.setdefault(t["mode"], {"trades": 0, "closed": 0, "wins": 0, "realized_pnl": 0.0})
        d["trades"] += 1
        if t["closed"]:
            d["closed"] += 1
            d["realized_pnl"] += t["realized_pnl"]
            d["wins"] += int(t["win"])
    for d in list(by_mode.values()) + list(by_time_bucket.values()):
        d["realized_pnl"] = round(d["realized_pnl"], 4)
        d["hit_rate"] = round(d["wins"] / d["closed"], 4) if d["closed"] else None

    violation_counts = Counter(v for t in trade_rows for v in t["rule_violations"])
    for e in events:   # standalone (non-trade) violations
        if e.get("trade_id") is None:
            violation_counts.update(_list(e, "rule_violations"))
            violation_counts.update(_sub_list(e, "process_review", "rule_violations"))

    experiments = []
    seen_experiments: set[tuple] = set()
    for e in events:
        if e.get("restricted"):
            continue   # never surface a restricted symbol as a forward experiment
        if e.get("event_type") == "experiment" or _get(e, "hypothesis"):
            key = _dedupe_key(e.get("trade_id"), _get(e, "hypothesis"), _get(e, "promote_if"),
                              _get(e, "kill_if"))
            if key in seen_experiments:
                continue
            seen_experiments.add(key)
            experiments.append({
                "hypothesis": _get(e, "hypothesis"), "metric": _get(e, "metric"),
                "promote_if": _get(e, "promote_if"), "kill_if": _get(e, "kill_if"),
                "status": _get(e, "status") or "open", "trade_id": e.get("trade_id"),
            })

    lessons = []
    seen_lessons: set[tuple] = set()
    for e in events:
        for x in _list(e, "lessons"):
            key = _dedupe_key(e.get("trade_id"), x)
            if key in seen_lessons:
                continue
            seen_lessons.add(key)
            lessons.append({"trade_id": e.get("trade_id"), "lesson": x})

    return {
        "n_events": len(events),
        "by_type": by_type,
        "n_trades": len(trade_rows),
        "n_closed": len(closed),
        "hit_rate": round(len(wins) / len(closed), 4) if closed else None,
        "total_realized_pnl": round(sum(t["realized_pnl"] for t in closed), 4) if closed else 0.0,
        "avg_realized_pnl": round(sum(t["realized_pnl"] for t in closed) / len(closed), 4) if closed else None,
        "avg_mfe_capture": round(sum(caps) / len(caps), 4) if caps else None,
        "avg_held_minutes": round(sum(held) / len(held), 1) if held else None,
        "n_management_checks": by_type.get("management_check", 0),
        "by_mode": by_mode,
        "by_time_bucket": by_time_bucket,
        "rule_violations": dict(violation_counts),
        "n_rule_violations": int(sum(violation_counts.values())),
        "restricted_flags": sorted(set(restricted_flags)),
        "sentiment_status": _sentiment_status(events),
        "gamma_status": _gamma_status(events),
        "experiments": experiments,
        "lessons": lessons,
        "pnl_sequence": [t["realized_pnl"] for t in closed],
        # Per-trade excursion measured from the monitoring polls, keyed by trade_id. Kept out of
        # the dollar-denominated mfe/mae fields (see excursion_from_polls) and carried here so the
        # MAX_LOSS-threshold question can be asked of every trade that was polled, not only the
        # ones whose agent-authored postmortem happened to state an excursion.
        "measured_excursions": {
            t["trade_id"]: {"mae_pct": t["mae_pct_measured"], "mfe_pct": t["mfe_pct_measured"],
                            "polls": t["excursion_polls"]}
            for t in trade_rows if t.get("excursion_polls")
        },
        "process_quality": _process_quality(trade_rows),
        "canonical_guardrails": canonical_guardrails(),
        "time_window_policy": window_descriptions(),
    }


# --- render ----------------------------------------------------------------------------------

def _render_context_lines(sent: dict, gamma: dict) -> list[str]:
    """Markdown for the rolled-up sentiment / gamma context (omitted entirely when both empty)."""
    sent, gamma = sent or {}, gamma or {}
    if not sent.get("n_readings") and not gamma.get("n_readings"):
        return []
    lines = ["", "## Sentiment & gamma context", ""]
    if sent.get("n_readings"):
        latest = sent.get("latest") or {}
        bias = " · ".join(f"{k}×{v}" for k, v in sorted(sent.get("by_direction", {}).items())) or "—"
        cur = (f"{latest.get('direction') or '?'}/{latest.get('verdict') or '?'} "
               f"(conf {latest.get('confidence') or 'n/a'})") if latest else "n/a"
        lines.append(f"- 🗣️ Sentiment: **{sent['n_readings']}** reading(s) · latest **{cur}** "
                     f"· avg score {sent.get('avg_score') if sent.get('avg_score') is not None else 'n/a'} "
                     f"· bias {bias}")
        by_status = sent.get("by_status") or {}
        if by_status or (latest and (latest.get("status") or latest.get("context"))):
            counts = " · ".join(f"{k}×{v}" for k, v in sorted(by_status.items())) or "—"
            cur_ctx = (f" — {latest.get('context')}" if latest and latest.get("context") else "")
            lines.append(f"  - status: {counts} · latest "
                         f"**{(latest or {}).get('status') or 'n/a'}**{cur_ctx}")
        if sent.get("restricted_readings"):
            lines.append(f"  - ⛔ restricted reads (ignored): {', '.join(sent['restricted_readings'])}")
    if gamma.get("n_readings"):
        latest = gamma.get("latest") or {}
        pins = " · ".join(f"{k}×{v}" for k, v in sorted(gamma.get("by_pin_risk", {}).items())) or "—"
        mgs = latest.get("max_gamma_strike")
        lines.append(f"- 📌 Gamma (**{gamma.get('regime', _GAMMA_BASIS)}**, NOT dealer GEX): "
                     f"**{gamma['n_readings']}** reading(s) · latest pin-risk "
                     f"**{str(latest.get('pin_risk') or 'n/a').upper()}** "
                     f"· max-γ strike {mgs if mgs is not None else 'n/a'} · pin levels {pins}")
        by_status = gamma.get("by_status") or {}
        if by_status or (latest and (latest.get("status") or latest.get("context"))):
            counts = " · ".join(f"{k}×{v}" for k, v in sorted(by_status.items())) or "—"
            cur_ctx = (f" — {latest.get('context')}" if latest and latest.get("context") else "")
            lines.append(f"  - status: {counts} · latest "
                         f"**{(latest or {}).get('status') or 'n/a'}**{cur_ctx}")
        if gamma.get("restricted_readings"):
            lines.append(f"  - ⛔ restricted reads (ignored): {', '.join(gamma['restricted_readings'])}")
    return lines


def render_markdown(summary: dict, now: datetime | None = None) -> str:
    s = summary or {}
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    if not s.get("n_events"):
        return f"# 0DTE Decision Journal\n\n_No journal events yet (as of {stamp})._\n"

    def pct(x):
        return "n/a" if x is None else f"{x * 100:.0f}%"

    def money(x):
        return "n/a" if x is None else f"{x:+.2f}"

    lines = [
        "# 0DTE Decision Journal — Improvement Loop",
        f"_Generated {stamp} · local/offline · no broker/LLM calls_",
        "",
        "## Scorecard",
        f"- Events: **{s['n_events']}** · Trades: **{s['n_trades']}** · Closed: **{s['n_closed']}**",
        f"- Hit rate: **{pct(s['hit_rate'])}** · Total P/L: **{money(s['total_realized_pnl'])}** "
        f"· Avg P/L: **{money(s['avg_realized_pnl'])}**",
        f"- MFE capture: **{pct(s['avg_mfe_capture'])}** "
        f"· Avg hold: **{s['avg_held_minutes'] if s['avg_held_minutes'] is not None else 'n/a'} min** "
        f"· Mgmt checks: **{s['n_management_checks']}**",
        f"- Closed-P/L trend: `{_sparkline(s.get('pnl_sequence', [])) or '—'}`",
    ]
    if s.get("restricted_flags"):
        lines.append(f"- ⛔ **RESTRICTED underlyings present:** {', '.join(s['restricted_flags'])} "
                     "(employer — must never be traded)")

    lines += _render_context_lines(s.get("sentiment_status", {}), s.get("gamma_status", {}))

    lines += ["", "## Trades by mode", "", "| Mode | Trades | Closed | Hit | P/L | |",
              "|------|-------:|-------:|----:|----:|---|"]
    modes = s.get("by_mode", {})
    max_pl = max((abs(d["realized_pnl"]) for d in modes.values()), default=0.0)
    for mode in sorted(modes):
        d = modes[mode]
        hr = "n/a" if d["hit_rate"] is None else f"{d['hit_rate'] * 100:.0f}%"
        lines.append(f"| {mode} | {d['trades']} | {d['closed']} | {hr} | {money(d['realized_pnl'])} "
                     f"| `{_bar(d['realized_pnl'], max_pl)}` |")

    buckets = s.get("by_time_bucket", {})
    if buckets:
        lines += ["", "## Time-of-day buckets", "",
                  "_Descriptive only — time window is not execution permission._", "",
                  "| Bucket | Trades | Closed | Hit | P/L |",
                  "|--------|-------:|-------:|----:|----:|"]
        order = ["high_action_0930_1300", "midday_1300_1530", "late_1530_close",
                 "outside_regular", "unknown"]
        for bucket in [b for b in order if b in buckets] + sorted(set(buckets) - set(order)):
            d = buckets[bucket]
            hr = "n/a" if d["hit_rate"] is None else f"{d['hit_rate'] * 100:.0f}%"
            lines.append(f"| {bucket} | {d['trades']} | {d['closed']} | {hr} | {money(d['realized_pnl'])} |")

    guardrails = s.get("canonical_guardrails", [])
    if guardrails:
        lines += ["", "## Canonical strategy guardrails", ""]
        for g in guardrails:
            lines.append(f"- **{g.get('title')}** (`{g.get('key')}`): {g.get('rule')}")

    pq = s.get("process_quality", {})
    if pq.get("n_diagnosed"):
        lines += ["", "## Process quality & loss diagnosis", "",
                  f"_Self-eval over {pq['n_diagnosed']} measurable closed trade(s) — separates process "
                  f"from outcome, and where a trade actually broke._", ""]
        po = pq.get("process_outcome", {})
        if po:
            lines.append("**Process × outcome** (a bad-process *win* is the dangerous one):")
            for k in ("good_process_good_outcome", "good_process_bad_outcome",
                      "bad_process_lucky_outcome", "bad_process_bad_outcome"):
                if po.get(k):
                    lines.append(f"- {k.replace('_', ' ')}: **{po[k]}**")
        diag = pq.get("execution_diagnosis", {})
        if diag:
            lines += ["", "**Entry / exit / thesis diagnosis:**"]
            lines += [f"- {k.replace('_', ' ')}: **{v}**"
                      for k, v in sorted(diag.items(), key=lambda kv: -kv[1])]
        lc = pq.get("loss_categories", {})
        if lc:
            lines += ["", "**Loss causes** (explicit tags; `uncategorized` = no `loss_category` set):"]
            lines += [f"- {k}: **{v}**" for k, v in sorted(lc.items(), key=lambda kv: -kv[1])]

    viol = s.get("rule_violations", {})
    lines += ["", "## Rule violations", ""]
    if viol:
        lines += [f"- {n}× {name}" for name, n in sorted(viol.items(), key=lambda kv: -kv[1])]
    else:
        lines.append("- _none recorded_ ✅")

    exps = s.get("experiments", [])
    lines += ["", "## Next-cycle experiments", ""]
    if exps:
        lines += ["| Status | Hypothesis | Metric | Promote if | Kill if |",
                  "|--------|-----------|--------|-----------|---------|"]
        for e in exps:
            lines.append(f"| {e.get('status', 'open')} | {e.get('hypothesis') or ''} "
                         f"| {e.get('metric') or ''} | {e.get('promote_if') or ''} "
                         f"| {e.get('kill_if') or ''} |")
    else:
        lines.append("- _no experiments queued_")

    lessons = s.get("lessons", [])
    if lessons:
        lines += ["", "## Lessons", ""]
        lines += [f"- ({x.get('trade_id') or '—'}) {x.get('lesson')}" for x in lessons]
    return "\n".join(lines) + "\n"


def render_csv(summary: dict) -> str:
    """By-mode summary CSV (one row per mode) for later plotting."""
    rows = ["mode,trades,closed,wins,hit_rate,realized_pnl"]
    for mode in sorted((summary or {}).get("by_mode", {})):
        d = summary["by_mode"][mode]
        hr = "" if d["hit_rate"] is None else f"{d['hit_rate']:.4f}"
        rows.append(f"{mode},{d['trades']},{d['closed']},{d['wins']},{hr},{d['realized_pnl']:.4f}")
    return "\n".join(rows) + "\n"


def build_report(journal_path: str | None = None, out_dir: str | None = None,
                 write_artifacts: bool = False, now: datetime | None = None) -> dict:
    """Read the journal, compute metrics, render Markdown+CSV, optionally write artifacts.

    Returns {summary, markdown, csv, artifacts}. No broker/LLM/network anywhere."""
    events = read_events(journal_path)
    summary = summarize(events)
    md = render_markdown(summary, now=now)
    csv = render_csv(summary)
    artifacts: dict[str, str] = {}
    if write_artifacts or out_dir:
        odir = Path(os.path.expanduser(out_dir or DEFAULT_REPORT_DIR))
        odir.mkdir(parents=True, exist_ok=True)
        md_path, csv_path = odir / "odte_journal_report.md", odir / "odte_journal_summary.csv"
        md_path.write_text(md)
        csv_path.write_text(csv)
        artifacts = {"markdown": str(md_path), "csv": str(csv_path)}
    return {"summary": summary, "markdown": md, "csv": csv, "artifacts": artifacts}
