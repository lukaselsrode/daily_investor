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
from datetime import datetime, timezone
from pathlib import Path

try:                                   # POSIX advisory file lock (mac/linux). Best-effort.
    import fcntl
except ImportError:                    # pragma: no cover - non-POSIX fallback
    fcntl = None

from core.paths import ODTE_DATA_DIR, ODTE_REPORT_DIR

logger = logging.getLogger(__name__)

DEFAULT_DIR = ODTE_DATA_DIR
DEFAULT_JOURNAL_PATH = os.path.join(DEFAULT_DIR, "decision_journal.jsonl")
DEFAULT_REPORT_DIR = ODTE_REPORT_DIR

EVENT_TYPES = ("pre_trade_thesis", "entry_decision", "order_filled", "management_check",
               "exit_decision", "order_closed", "postmortem", "experiment", "note")

_ENTRY_EVENTS = {"entry_decision", "order_filled"}
_EXIT_EVENTS = {"exit_decision", "order_closed"}

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
    # is never execution-allowed regardless of caller input.
    if scan_only or base.get("restricted"):
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
    if et == "candidate" or src == "ingest:candidate":
        return "candidates"
    if et == "vehicle_score" or "vehicle_score" in src or e.get("vehicle_score"):
        return "vehicle_scores"
    if et in _ENTRY_EVENTS or et in _EXIT_EVENTS or et == "management_check":
        return "trades"
    return "controller_events"


def _day_postmortem_stub(trade_date: str, buckets: dict) -> str:
    """A minimal, human-editable postmortem scaffold (only written if absent). The self-eval report
    (build_report) fills the analytical answers; this is the per-day narrative space."""
    lines = [f"# 0DTE postmortem — {trade_date}", "",
             "_Auto-scaffold from the decision journal; edit freely. Counts:_", ""]
    for s in _DAY_STREAMS:
        lines.append(f"- **{s}**: {len(buckets.get(s, []))}")
    lines += ["", "## What happened", "", "## Thesis right or wrong?", "",
              "## Entry vs exit", "", "## Gamma/pin helped or hurt?", "",
              "## Social helped or distracted?", "", "## Keep / change next session", ""]
    return "\n".join(lines)


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
        if not pm.exists():
            pm.write_text(_day_postmortem_stub(td, buckets))
        summary["files"]["postmortem.md"] = 1
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
        "confirmation_needed": g.get("confirmation_needed"),
        "gates": g.get("gates"),
        "scan_only": bool(g.get("scan_only", False)),
        "execution_allowed": bool(g.get("execution_allowed", False)),
    }
    if extra:
        e.update(extra)
    return e


def event_from_vehicle_score(score_payload: dict, trade_id: str | None = None,
                             extra: dict | None = None) -> dict:
    """Convert an `odte-vehicle-score` payload into a `pre_trade_thesis` journal event (a plain dict
    the caller can pass to append_event). Records the non-sentiment GOOD_BET/WATCH/BAD_BET verdict,
    its components/score, and the reasons so the live controller can journal *why* a candidate
    vehicle looked good or bad. Does NOT auto-append — keeps the seam explicit (parallel to
    `event_from_position_decision`). The underlying is normalized/restriction-tagged at append."""
    p = score_payload or {}
    c = p.get("contract") if isinstance(p.get("contract"), dict) else {}
    e = {
        "event_type": "pre_trade_thesis",
        "trade_id": trade_id or p.get("trade_id"),
        "underlying": c.get("underlying") or c.get("symbol"),
        "option_type": c.get("option_type") or c.get("type"),
        "strike": c.get("strike") or c.get("strike_price"),
        "decision": {"action": p.get("verdict"),
                     "reasons": list(p.get("reasons") or [])},
        "vehicle_score": {"verdict": p.get("verdict"), "score": p.get("score"),
                          "components": p.get("components"), "direction": p.get("direction")},
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
    """One of the four process×outcome cells for a CLOSED trade, from rule_violations + win only.
    'Process' = followed the plan (no rule violations). Distinguishes a good process that still lost
    (variance) from a bad process that happened to win (lucky — the dangerous one)."""
    if not t["closed"]:
        return None
    good_process = not t["rule_violations"]
    if good_process:
        return "good_process_good_outcome" if t["win"] else "good_process_bad_outcome"
    return "bad_process_lucky_outcome" if t["win"] else "bad_process_bad_outcome"


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
    for t in measurable:
        po = _process_outcome(t)
        if po:
            process_outcome[po] += 1
        diagnosis[_execution_diagnosis(t)] += 1
        if t["realized_pnl"] is not None and t["realized_pnl"] < 0:
            lc = t.get("loss_category")
            loss_categories[lc if lc in LOSS_CATEGORIES else "uncategorized"] += 1
    return {
        "n_diagnosed": len(measurable),
        "process_outcome": dict(process_outcome),
        "execution_diagnosis": dict(diagnosis),
        "loss_categories": dict(loss_categories),
    }


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
            # explicit self-eval fields (used as-is when present; never fabricated):
            "loss_category": _first(evs, "loss_category"),
            "diagnosis": _first(evs, "diagnosis"),
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
        "sentiment_status": _sentiment_status(events),
        "gamma_status": _gamma_status(events),
        "experiments": experiments,
        "lessons": lessons,
        "pnl_sequence": [t["realized_pnl"] for t in closed],
        "process_quality": _process_quality(trade_rows),
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
