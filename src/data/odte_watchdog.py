"""0DTE watchdog — script-only, NO LLM, NO Robinhood, places NO orders.

Runs the LOCAL ``build_odte_social_report`` (which makes ZERO LLM/model calls), diffs the
actionable candidate against the previous run, and writes compact state/trigger JSON under
``~/0dte/`` so a cron job (``no_agent=True``) can cheaply decide WHEN to wake the controller —
instead of an agent polling a model on a clock (the OpenAI/model-429 avoidance: there is simply no
model call in this path).

Conservative triggers only:
  * a NEW or CHANGED actionable, NON-restricted candidate appears, or
  * the controller policy is missing / invalid / unreadable.

Employer/compliance-restricted symbols (e.g. NVDA) are NEVER actionable — they surface in the
state's ``restricted_chatter`` as read-only context. This module never touches Robinhood and never
places, cancels, or sizes orders.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = os.path.expanduser("~/0dte")
STATE_FILENAME = "watchdog_state.json"
TRIGGERS_FILENAME = "triggers.json"
POLICY_FILENAME = "controller_policy.json"

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


def run_watchdog(state_dir: str = DEFAULT_STATE_DIR, policy_path: str | None = None,
                 allow_fetch: bool = True, now: datetime | None = None) -> dict:
    """Build the local report, diff vs prior state, persist state + triggers, return the payload.

    NO LLM and NO broker calls anywhere in this path. Returns the trigger payload dict (also
    written to ``triggers.json``); ``payload['alert']`` is True iff a conservative trigger fired.
    """
    now = now or datetime.now(timezone.utc)
    sdir = Path(os.path.expanduser(state_dir))
    sdir.mkdir(parents=True, exist_ok=True)
    ppath = Path(os.path.expanduser(policy_path)) if policy_path else sdir / POLICY_FILENAME

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
    }

    (sdir / STATE_FILENAME).write_text(json.dumps(state, indent=2, default=str))
    (sdir / TRIGGERS_FILENAME).write_text(json.dumps(payload, indent=2, default=str))
    return payload
