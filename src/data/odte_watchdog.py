"""0DTE watchdog — script-only, NO LLM, NO Robinhood, places NO orders.

Runs the LOCAL ``build_odte_social_report`` (which makes ZERO LLM/model calls), diffs the
actionable candidate against the previous run, and writes compact state/trigger JSON under
``data/odte/`` so a cron job (``no_agent=True``) can cheaply decide WHEN to wake the controller —
instead of an agent polling a model on a clock (the OpenAI/model-429 avoidance: there is simply no
model call in this path). The controller policy it checks is a SECRET and is read from ``~/0dte/``
(see ``DEFAULT_POLICY_PATH``), kept out of the app's data tree.

Conservative triggers only:
  * a NEW or CHANGED actionable, NON-restricted candidate appears, or
  * the controller policy is missing / invalid / unreadable.

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

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = ODTE_DATA_DIR
STATE_FILENAME = "watchdog_state.json"
TRIGGERS_FILENAME = "triggers.json"
POLICY_FILENAME = "controller_policy.json"
# Controller policy holds account/execution config — a SECRET. It stays in ~/0dte/ (Hermes
# territory), NOT in the app data tree. State/triggers (above) are data and live in data/odte/.
DEFAULT_POLICY_PATH = os.path.join(ODTE_SECRETS_DIR, POLICY_FILENAME)

STATE_VERSION = 1


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
    candidate = report.get("candidate")
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
    if candidate_key and candidate_key != prev.get("candidate_key"):
        triggers.append({"type": "new_candidate", "candidate": candidate_key,
                         "detail": "new/changed actionable non-restricted candidate"})

    alert = bool(triggers)

    state = {
        "version": STATE_VERSION,
        "updated_at": now.isoformat(timespec="seconds"),
        "last_run_utc": now.isoformat(timespec="seconds"),
        "policy_ok": policy_ok,
        "policy_status": policy_status,
        "spy_verdict": spy_verdict,
        "candidate_key": candidate_key,
        "restricted_chatter": restricted_chatter,
        "report_ok": report_error is None,
    }
    payload = {
        "ts": now.isoformat(timespec="seconds"),
        "alert": alert,
        "triggers": triggers,
        "spy_verdict": spy_verdict,
        "candidate": candidate if candidate_key else None,   # never a restricted symbol
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
