"""0DTE watchdog — script-only, NO LLM, NO Robinhood, places NO orders.

Runs the LOCAL ``build_odte_social_report`` (which makes ZERO LLM/model calls), diffs the
actionable candidate against the previous run, and writes compact state/trigger JSON under
``data/odte/`` so a cron job (``no_agent=True``) can cheaply decide WHEN to wake the controller —
instead of an agent polling a model on a clock (the OpenAI/model-429 avoidance: there is simply no
model call in this path). The controller policy it checks is a SECRET and is read from ``~/0dte/``
(see ``DEFAULT_POLICY_PATH``), kept out of the app's data tree.

Conservative triggers only:
  * a NEW or CHANGED actionable, NON-restricted candidate appears,
  * an unchanged actionable candidate PERSISTS past the re-alert window (2026-08-02 retune —
    change-only alerting let a parked candidate go silent forever), or
  * the controller policy is missing / invalid / unreadable.

Candidates are filtered to the EXECUTABLE universe (SPY/QQQ/IWM): a single-name chatter read can
never park as the candidate key (it demotes to ``single_name_context``), because the downstream
candidate-watch lane hard-rejects non-ETF symbols and a parked unconvertible candidate suppresses
every alert.

Employer/compliance-restricted symbols (e.g. NVDA) are NEVER actionable — they surface in the
state's ``restricted_chatter`` as read-only context. This module never touches Robinhood and never
places, cancels, or sizes orders.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from core.paths import ODTE_DATA_DIR, ODTE_SECRETS_DIR, atomic_write_text
from data.odte_config import EXECUTABLE_UNIVERSE, WATCHDOG_REALERT_MINUTES

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = ODTE_DATA_DIR
STATE_FILENAME = "watchdog_state.json"
TRIGGERS_FILENAME = "triggers.json"
POLICY_FILENAME = "controller_policy.json"
# Controller policy holds account/execution config — a SECRET. It stays in ~/0dte/ (Hermes
# territory), NOT in the app data tree. State/triggers (above) are data and live in data/odte/.
DEFAULT_POLICY_PATH = os.path.join(ODTE_SECRETS_DIR, POLICY_FILENAME)

STATE_VERSION = 1


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _read_json(path: Path) -> tuple[dict | None, str]:
    """Return (parsed_dict | None, status). status is 'ok' | 'missing' | 'invalid'."""
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None, "invalid"
        return data, "ok"
    except Exception as exc:
        logger.warning("watchdog: could not read %s: %s", path.name, exc)
        return None, "invalid"


def _candidate_key(candidate: dict | None) -> str | None:
    """Stable 'TICKER:direction' key for an actionable, NON-restricted candidate (else None)."""
    if not isinstance(candidate, dict):
        return None
    from data.social_sentiment import is_restricted_underlying
    tk = candidate.get("ticker")
    if not tk or is_restricted_underlying(tk):
        return None   # defensive: candidate selection already excludes restricted symbols
    direction = candidate.get("direction") or "?"
    return f"{str(tk).upper()}:{direction}"


def _scorecard_market_candidate(report: dict) -> dict | None:
    """Synthesize a scan-only SPY market candidate when the broad scorecard is directional.

    The social report can produce a real market read (``CALL-leaning`` / ``PUT-leaning``) without a
    single-name chatter candidate. That used to collapse the watchdog to ``candidate=null`` and the
    live loop to ``FLAT_NO_TRADE`` even while SPY/QQQ/VIXY were giving a directional setup — exactly
    the kind of "wait for fresh confirmation" / no-nudge stall the controller is supposed to avoid.

    This synthetic candidate is still scan-only and non-executable; it simply keeps the state machine
    in CANDIDATE / WAIT_FRESH_CONFIRMATION so the controller runs opening HAWK checks, contract scan,
    and entry gate instead of going idle.
    """
    raw_scorecard = report.get("scorecard")
    scorecard: dict = raw_scorecard if isinstance(raw_scorecard, dict) else {}
    verdict = str(scorecard.get("verdict") or "").strip()
    direction = {"CALL-leaning": "bullish", "PUT-leaning": "bearish"}.get(verdict)
    if not direction:
        return None
    raw_trend = report.get("spy_trend")
    raw_social = report.get("social_intent")
    trend: dict = raw_trend if isinstance(raw_trend, dict) else {}
    social: dict = raw_social if isinstance(raw_social, dict) else {}
    return {
        "ticker": "SPY",
        "direction": direction,
        "mentions": social.get("n_docs") or 0,
        "sentiment": social.get("intent"),
        "source": "market_scorecard",
        "market_verdict": verdict,
        "scorecard_confidence": scorecard.get("confidence"),
        "scorecard_reasons": list(scorecard.get("reasons") or [])[:4],
        "observed_market_context": {
            "pct_vs_prev_close": trend.get("pct_vs_prev_close"),
            "above_vwap": trend.get("above_vwap"),
        },
    }


# Confirmations a watchdog candidate ALWAYS requires before it could ever be acted on. The watchdog
# is a decision-support/trigger lane, never an execution lane — so these are non-negotiable.
_REQUIRED_CONFIRMATIONS = ("human_review", "live_chain_recheck", "spread_cap_check", "budget_check")


def _decision_context(report: dict, candidate: dict | None, scorecard: dict,
                      policy_ok: bool, policy_status: str, report_error: str | None) -> dict:
    """Build the enriched, CONSERVATIVE observability block for the trigger payload.

    The watchdog is a scan/trigger lane: `scan_only` is always True and `execution_allowed` always
    False here — nothing this function returns can authorize a trade. Fields are best-effort from the
    local report; missing context is reported as unavailable rather than guessed."""
    cand = candidate if isinstance(candidate, dict) else {}
    trend = report.get("spy_trend") if isinstance(report.get("spy_trend"), dict) else {}
    social = report.get("social_intent") if isinstance(report.get("social_intent"), dict) else {}
    gamma = cand.get("gamma") if isinstance(cand.get("gamma"), dict) else None

    veto_reasons: list[str] = []
    if not policy_ok:
        veto_reasons.append(f"policy_{policy_status}")
    if report_error:
        veto_reasons.append("report_error")
    if (scorecard.get("verdict") or "OBSERVE") == "OBSERVE":
        veto_reasons.append("no_directional_edge")
    if cand.get("restricted"):
        veto_reasons.append("restricted_employer")

    return {
        "thesis": {"direction": cand.get("direction"),
                   "basis": list(scorecard.get("reasons") or [])[:4]} if cand else None,
        "confidence": scorecard.get("confidence"),
        "confirmation_needed": True,                 # a trigger is never self-authorizing
        "required_confirmations": list(_REQUIRED_CONFIRMATIONS),
        # HARD conservative defaults for the scan/trigger lane:
        "scan_only": True,
        "execution_allowed": False,
        "veto_reasons": veto_reasons,
        "risk_notes": ["decision-support only — PAPER/analysis, places NO orders",
                       "0DTE: total-loss risk; re-validate chain/spread/budget live before any action"],
        "observed_market_context": {
            "spy_verdict": scorecard.get("verdict", "OBSERVE"),
            "pct_vs_prev_close": trend.get("pct_vs_prev_close"),
            "above_vwap": trend.get("above_vwap"),
        },
        "social_context": {
            "intent": social.get("intent") or cand.get("direction"),
            "n_docs": social.get("n_docs"),
            "mentions": cand.get("mentions"),
            "sentiment": cand.get("sentiment"),
        },
        "gamma_context": gamma or {"available": False, "basis": "pin_risk_only_not_dealer_gex"},
    }


def run_watchdog(state_dir: str = DEFAULT_STATE_DIR, policy_path: str | None = None,
                 allow_fetch: bool = True, now: datetime | None = None) -> dict:
    """Build the local report, diff vs prior state, persist state + triggers, return the payload.

    NO LLM and NO broker calls anywhere in this path. Returns the trigger payload dict (also
    written to ``triggers.json``); ``payload['alert']`` is True iff a conservative trigger fired.
    """
    now = now or datetime.now(timezone.utc)
    sdir = Path(os.path.expanduser(state_dir))
    sdir.mkdir(parents=True, exist_ok=True)
    # Policy is a secret read from ~/0dte/ by default (NOT the state dir); --policy still overrides.
    ppath = Path(os.path.expanduser(policy_path)) if policy_path else Path(DEFAULT_POLICY_PATH)

    # 1) Controller policy presence/validity (we do NOT echo its contents — it holds account info).
    _policy, policy_status = _read_json(ppath)
    policy_ok = policy_status == "ok"

    # 2) Prior watchdog state (for candidate diffing).
    prev, _ = _read_json(sdir / STATE_FILENAME)
    prev = prev or {}

    # 3) LOCAL report — zero LLM calls. Fail-closed: a build error becomes a conservative trigger.
    report: dict = {}
    report_error: str | None = None
    try:
        from data.social_sentiment import build_odte_social_report
        report = build_odte_social_report(allow_fetch=allow_fetch)
    except Exception as exc:   # pragma: no cover - defensive; report builds fail-closed internally
        report_error = str(exc)
        logger.warning("watchdog: report build failed: %s", exc)

    scorecard = report.get("scorecard") or {}
    spy_verdict = scorecard.get("verdict", "OBSERVE")
    candidate = report.get("candidate") or _scorecard_market_candidate(report)
    # EXECUTABLE-UNIVERSE FILTER (2026-08-02 retune): the candidate-watch lane can only convert
    # SPY/QQQ/IWM, so a single-name chatter candidate (AAPL/TSLA/...) must never PARK as the
    # watchdog's candidate key — last week it sat on AAPL:bearish for days, suppressing every
    # alert while being structurally unconvertible. Demote it to observability context and fall
    # back to the market-scorecard candidate.
    single_name_context: dict | None = None
    cand_ticker = str((candidate or {}).get("ticker") or "").upper()
    if candidate and cand_ticker and cand_ticker not in EXECUTABLE_UNIVERSE:
        single_name_context = candidate
        candidate = _scorecard_market_candidate(report)
    candidate_key = _candidate_key(candidate)
    restricted_chatter = sorted({
        str(c.get("ticker")).upper()
        for c in (report.get("top_chatter") or [])
        if c.get("restricted")
    })

    # 4) Conservative triggers.
    triggers: list[dict] = []
    if not policy_ok:
        triggers.append({"type": f"policy_{policy_status}",
                         "detail": f"controller policy {policy_status} at {ppath.name}"})
    if report_error:
        triggers.append({"type": "report_error", "detail": report_error})
    prev_key = prev.get("candidate_key")
    if candidate_key and candidate_key != prev_key:
        triggers.append({"type": "new_candidate", "candidate": candidate_key,
                         "detail": "new/changed actionable non-restricted candidate"})
    elif candidate_key and candidate_key == prev_key:
        # PERSISTENCE RE-ALERT (2026-08-02 retune): change-only alerting let an unchanged
        # candidate go silent forever. An actionable candidate still on the board re-alerts every
        # WATCHDOG_REALERT_MINUTES so the controller keeps working it instead of forgetting it.
        last_alert = _parse_ts(prev.get("last_alert_utc"))
        if (last_alert is None
                or (now - last_alert).total_seconds() >= WATCHDOG_REALERT_MINUTES * 60):
            triggers.append({"type": "candidate_persisting", "candidate": candidate_key,
                             "detail": (f"candidate unchanged >= {WATCHDOG_REALERT_MINUTES}m "
                                        "since last alert — re-alerting")})

    alert = bool(triggers)
    first_seen = (prev.get("candidate_first_seen_utc")
                  if candidate_key and candidate_key == prev_key else None)
    if candidate_key and not first_seen:
        first_seen = now.isoformat(timespec="seconds")

    state = {
        "version": STATE_VERSION,
        "updated_at": now.isoformat(timespec="seconds"),
        "last_run_utc": now.isoformat(timespec="seconds"),
        "policy_ok": policy_ok,
        "policy_status": policy_status,
        "spy_verdict": spy_verdict,
        "candidate_key": candidate_key,
        "candidate_first_seen_utc": first_seen,
        "last_alert_utc": (now.isoformat(timespec="seconds") if alert
                           else prev.get("last_alert_utc")),
        "restricted_chatter": restricted_chatter,
        "report_ok": report_error is None,
    }
    payload = {
        "ts": now.isoformat(timespec="seconds"),
        "alert": alert,
        "triggers": triggers,
        "spy_verdict": spy_verdict,
        "candidate": candidate if candidate_key else None,   # never a restricted symbol
        # Demoted non-executable single-name read (context only — never the candidate key).
        "single_name_context": single_name_context,
        "restricted_chatter": restricted_chatter,
        "policy_ok": policy_ok,
        # Additive enriched, CONSERVATIVE observability block (old consumers ignore unknown keys).
        # scan_only=True / execution_allowed=False are invariants of this trigger lane.
        "scan_only": True,
        "execution_allowed": False,
        "decision_context": _decision_context(report, candidate if candidate_key else None,
                                               scorecard, policy_ok, policy_status, report_error),
    }

    # Atomic writes (tmp + os.replace): a crash mid-write must not leave a truncated state/trigger
    # file for the next poll to misread (dropped/duplicate trigger).
    triggers_text = json.dumps(payload, indent=2, default=str)
    atomic_write_text(sdir / STATE_FILENAME, json.dumps(state, indent=2, default=str))
    atomic_write_text(sdir / TRIGGERS_FILENAME, triggers_text)
    _journal_watchdog_trigger(payload, triggers_text, sdir / TRIGGERS_FILENAME)
    return payload


def render_pulse(payload: dict) -> str:
    """Compact HUMAN pulse for a fired trigger — what lands in Telegram via the no-agent cron.

    The machine contract is untouched: controllers read data/odte/triggers.json (or --json).
    2026-08-05: the cron delivered the compact-JSON alert line verbatim to Telegram, so every
    trigger read as a wall of JSON on a phone. A pulse names the trigger, the candidate, and
    the hand-off — nothing else."""
    cand = payload.get("candidate") or {}
    triggers = payload.get("triggers") or []
    types = ", ".join(sorted({str(t.get("type") or "trigger") for t in triggers})) or "trigger"
    lines = [f"🔔 0DTE watchdog — {types}"]
    if cand:
        who = " ".join(str(x) for x in (cand.get("ticker"), cand.get("direction")) if x)
        verdict = str(payload.get("spy_verdict") or cand.get("market_verdict") or "").strip()
        conf = str(cand.get("scorecard_confidence") or "").strip()
        detail = ", ".join(x for x in (verdict, f"confidence {conf}" if conf else "") if x)
        lines.append(f"• candidate: {who}" + (f" ({detail})" if detail else ""))
    for trig in triggers[:3]:
        detail = str(trig.get("detail") or "").strip()
        if detail:
            lines.append(f"• {detail}")
    lines.append("→ controller works it on its next tick (odte-candidate-watch → odte-convert)")
    return "\n".join(lines)


def _journal_watchdog_trigger(payload: dict, triggers_text: str, triggers_path: Path) -> None:
    """Best-effort: fold the watchdog trigger payload into the standardized decision journal as a
    scan-tier `watchdog_trigger` event. FULLY fail-safe — any error is swallowed so it can NEVER
    change the trigger payload, the stdout contract, or crash the poll. INVARIANT: this is the scan/
    trigger lane — `scan_only=True`, `execution_allowed=False`; nothing here authorizes a trade."""
    try:
        from data.odte_journal import append_decision_journal
        cand = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        dc = payload.get("decision_context") or {}
        ev = {
            "ts": payload.get("ts"),
            "underlying": cand.get("ticker"),
            "decision": "observe",                      # a trigger is a heads-up, never an execution
            "alert": payload.get("alert"),
            "spy_verdict": payload.get("spy_verdict"),
            "reason_codes": [t.get("type") for t in (payload.get("triggers") or []) if t.get("type")],
            "thesis": dc.get("thesis"),
            "confidence": dc.get("confidence"),
            "confirmation_needed": True,
            "veto_reasons": dc.get("veto_reasons"),
            "restricted_chatter": payload.get("restricted_chatter"),
            # scan/trigger lane invariants — append_decision_journal re-enforces these too:
            "scan_only": True,
            "execution_allowed": False,
            "raw_artifact_path": str(triggers_path),
            "raw_artifact_sha": hashlib.sha1(triggers_text.encode("utf-8")).hexdigest()[:16],
        }
        # Journal co-located with the state files (so a tmp state_dir in tests never touches the real
        # journal; in production triggers_path.parent is data/odte/).
        jp = str(triggers_path.parent / "decision_journal.jsonl")
        append_decision_journal(ev, source="watchdog", event_type="watchdog_trigger", journal_path=jp)
    except Exception as exc:        # never let journaling affect the watchdog
        logger.debug("watchdog trigger journaling skipped (%s)", exc)
