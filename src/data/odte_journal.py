"""0DTE decision journal — local/offline, NO broker, NO LLM, NO secrets.

A persistent JSONL journal of Hermes's 0DTE decision-making: the pre-trade thesis, the entry/exit
decisions, management checks, postmortems, and next-cycle experiments. It is a *record + analysis*
layer — it never places orders, never calls a broker or model, and never recommends or trades NVDA
(employer-restricted; any restricted event is tagged and kept out of experiments/metrics).

Pipeline:
    append_event(event)   -> one JSON line in ~/0dte/decision_journal.jsonl
    build_report()        -> deterministic metrics + Markdown/CSV artifacts under ~/0dte/reports/

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

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DIR = os.path.expanduser("~/0dte")
DEFAULT_JOURNAL_PATH = os.path.join(DEFAULT_DIR, "decision_journal.jsonl")
DEFAULT_REPORT_DIR = os.path.join(DEFAULT_DIR, "reports")

EVENT_TYPES = ("pre_trade_thesis", "entry_decision", "order_filled", "management_check",
               "exit_decision", "order_closed", "postmortem", "experiment", "note")

_ENTRY_EVENTS = {"entry_decision", "order_filled"}
_EXIT_EVENTS = {"exit_decision", "order_closed"}

_BLOCKS = "▁▂▃▄▅▆▇█"


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


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


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


# --- summarize -------------------------------------------------------------------------------

def summarize(events: list[dict]) -> dict:
    """Deterministic metrics over the journal. Pure: no IO, no network."""
    events = sorted(list(events or []), key=lambda e: (e.get("seq", 0)))
    by_type = dict(Counter(str(e.get("event_type", "?")) for e in events))

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
        mfe = _last_num(evs, "mfe")
        violations = [v for e in evs for v in _list(e, "rule_violations")]
        if restricted:
            violations.append(f"RESTRICTED_EMPLOYER: {underlying or 'restricted'} must never be traded")
        trade_rows.append({
            "trade_id": tid, "mode": str(_first(evs, "mode") or "unknown"),
            "underlying": underlying, "restricted": restricted,
            "realized_pnl": realized, "mfe": mfe, "mae": _last_num(evs, "mae"),
            "closed": realized is not None, "win": realized is not None and realized > 0,
            "rule_violations": violations, "held_minutes": _held_minutes(evs),
        })

    # Restricted trades are excluded from performance metrics (they should never exist) but the flag
    # and the synthetic violation above keep them loudly visible.
    measurable = [t for t in trade_rows if not t["restricted"]]
    closed = [t for t in measurable if t["closed"]]
    wins = [t for t in closed if t["win"]]
    caps = [t["realized_pnl"] / t["mfe"] for t in closed
            if t["mfe"] and t["mfe"] > 0 and t["realized_pnl"] is not None]
    held = [t["held_minutes"] for t in closed if t["held_minutes"] is not None]

    by_mode: dict[str, dict] = {}
    for t in measurable:
        d = by_mode.setdefault(t["mode"], {"trades": 0, "closed": 0, "wins": 0, "realized_pnl": 0.0})
        d["trades"] += 1
        if t["closed"]:
            d["closed"] += 1
            d["realized_pnl"] += t["realized_pnl"]
            d["wins"] += int(t["win"])
    for d in by_mode.values():
        d["realized_pnl"] = round(d["realized_pnl"], 4)
        d["hit_rate"] = round(d["wins"] / d["closed"], 4) if d["closed"] else None

    violation_counts = Counter(v for t in trade_rows for v in t["rule_violations"])
    for e in events:   # standalone (non-trade) violations
        if e.get("trade_id") is None:
            violation_counts.update(_list(e, "rule_violations"))

    experiments = []
    for e in events:
        if e.get("restricted"):
            continue   # never surface a restricted symbol as a forward experiment
        if e.get("event_type") == "experiment" or _get(e, "hypothesis"):
            experiments.append({
                "hypothesis": _get(e, "hypothesis"), "metric": _get(e, "metric"),
                "promote_if": _get(e, "promote_if"), "kill_if": _get(e, "kill_if"),
                "status": _get(e, "status") or "open", "trade_id": e.get("trade_id"),
            })

    lessons = [{"trade_id": e.get("trade_id"), "lesson": x}
               for e in events for x in _list(e, "lessons")]

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
        "rule_violations": dict(violation_counts),
        "n_rule_violations": int(sum(violation_counts.values())),
        "restricted_flags": sorted(set(restricted_flags)),
        "experiments": experiments,
        "lessons": lessons,
        "pnl_sequence": [t["realized_pnl"] for t in closed],
    }


# --- render ----------------------------------------------------------------------------------

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

    lines += ["", "## Trades by mode", "", "| Mode | Trades | Closed | Hit | P/L | |",
              "|------|-------:|-------:|----:|----:|---|"]
    modes = s.get("by_mode", {})
    max_pl = max((abs(d["realized_pnl"]) for d in modes.values()), default=0.0)
    for mode in sorted(modes):
        d = modes[mode]
        hr = "n/a" if d["hit_rate"] is None else f"{d['hit_rate'] * 100:.0f}%"
        lines.append(f"| {mode} | {d['trades']} | {d['closed']} | {hr} | {money(d['realized_pnl'])} "
                     f"| `{_bar(d['realized_pnl'], max_pl)}` |")

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
