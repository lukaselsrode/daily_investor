"""Canonical ODTE strategy policy — local/pure guardrails, no broker/model calls.

This module is the durable home for lessons learned from live play-money options
sessions. It intentionally describes *process constraints* and reporting labels;
it does not place orders, fetch broker state, or turn an observation into trade
permission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class StrategyWindow:
    """Descriptive ET time-of-day bucket for ODTE review and gating context."""

    label: str
    start_minute_et: int | None
    end_minute_et: int | None
    description: str
    posture: str


TIME_WINDOWS: tuple[StrategyWindow, ...] = (
    StrategyWindow(
        label="high_action_0930_1300",
        start_minute_et=9 * 60 + 30,
        end_minute_et=13 * 60,
        description="09:30–13:00 ET high-action/violent-move window",
        posture=(
            "Best opportunity window, but not permission to trade: require fresh broker truth, "
            "fresh artifacts, SPY/QQQ/VIXY confirmation, and clean vehicle quality."
        ),
    ),
    StrategyWindow(
        label="midday_1300_1530",
        start_minute_et=13 * 60,
        end_minute_et=15 * 60 + 30,
        description="13:00–15:30 ET lower-quality midday/chop window",
        posture=(
            "Raise selectivity; prefer no-trade or cleaner 1DTE/weekly exposure if 0DTE theta/pin "
            "risk makes the same thesis structurally bad."
        ),
    ),
    StrategyWindow(
        label="late_1530_close",
        start_minute_et=15 * 60 + 30,
        end_minute_et=16 * 60 + 1,
        description="15:30–close ET late theta/position-risk window",
        posture=(
            "Only exceptional momentum/flush setups; otherwise protect the day and avoid low-BP lottos."
        ),
    ),
)


@dataclass(frozen=True)
class StrategyGuardrail:
    """Canonical strategy/process guardrail surfaced in reports and prompts."""

    key: str
    title: str
    rule: str


CANONICAL_GUARDRAILS: tuple[StrategyGuardrail, ...] = (
    StrategyGuardrail(
        key="broker_truth_first",
        title="Broker/account truth first",
        rule="Verify account, buying power, open positions, and open orders before any entry or exit claim.",
    ),
    StrategyGuardrail(
        key="stale_artifact_veto",
        title="Stale artifacts are never authority",
        rule=(
            "Expired candidates, entry gates, scan triggers, and position decisions must be ignored or "
            "marked DEGRADED; rebuild a fresh thesis/gate for the current cycle."
        ),
    ),
    StrategyGuardrail(
        key="post_loss_cooldown",
        title="Post-loss cooldown raises the bar",
        rule=(
            "After a loss, re-verify flatness/orders/BP, wait for at least one fresh scan cycle, and require "
            "stronger confirmation. A loss is not permission to revenge trade."
        ),
    ),
    StrategyGuardrail(
        key="stricter_continuation",
        title="Continuation trades need stronger proof",
        rule=(
            "Do not chase a rip on speed alone; require reclaim/hold, SPY+QQQ agreement, weak VIXY, "
            "reasonable strike path, and non-hostile pin/gamma context."
        ),
    ),
    StrategyGuardrail(
        key="longer_dated_when_better",
        title="Use 1DTE/weekly only when structurally better",
        rule=(
            "Longer-dated single-leg options are allowed when they reduce theta/pin/chop risk and fit debit, "
            "liquidity, and invalidation; they are not a way to avoid stops or average down."
        ),
    ),
    StrategyGuardrail(
        key="reclaim_retest_observe_only",
        title="Reclaim/retest is observe-only",
        rule=(
            "A 1–2 minute reclaim/retest after a support break may be studied only when bid floor and VIXY "
            "are safe; do not promote it to live authority until enough examples prove it."
        ),
    ),
    StrategyGuardrail(
        key="target_zone_harvest",
        title="Harvest single-contract target-zone wins",
        rule=(
            "With one contract, take declared target-zone profits instead of negotiating for perfection and "
            "risking a round trip."
        ),
    ),
    StrategyGuardrail(
        key="low_bp_lotto_veto",
        title="Low-BP far-OTM lotto veto",
        rule=(
            "If remaining BP only fits far-OTM low-delta contracts without a real volatility/path setup, stay flat."
        ),
    ),
)


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        raw = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def classify_strategy_window(ts) -> str:
    """Return the canonical ET strategy-window label for a timestamp.

    This is descriptive context for self-eval/reporting. It never grants execution authority.
    """
    dt = _parse_ts(ts)
    if dt is None:
        return "unknown"
    et = dt.astimezone(ET)
    minutes = et.hour * 60 + et.minute
    for window in TIME_WINDOWS:
        if (window.start_minute_et is not None and window.end_minute_et is not None
                and window.start_minute_et <= minutes < window.end_minute_et):
            return window.label
    return "outside_regular"


def window_descriptions() -> dict[str, dict[str, str]]:
    """Serializable descriptions for report rendering."""
    out = {
        w.label: {"description": w.description, "posture": w.posture}
        for w in TIME_WINDOWS
    }
    out["outside_regular"] = {
        "description": "Outside regular 09:30–16:00 ET options session",
        "posture": "Do not assume regular-session liquidity/path; require explicit session support.",
    }
    out["unknown"] = {
        "description": "Timestamp missing/unparseable",
        "posture": "Treat as context-poor; do not use as trade authority.",
    }
    return out


def canonical_guardrails() -> list[dict[str, str]]:
    """Serializable canonical guardrails for prompts/reports/tests."""
    return [{"key": g.key, "title": g.title, "rule": g.rule} for g in CANONICAL_GUARDRAILS]


def canonical_guardrail_markdown() -> str:
    """Markdown block suitable for journal reports, day postmortems, and cron prompts."""
    lines = ["## Canonical strategy guardrails", ""]
    for g in CANONICAL_GUARDRAILS:
        lines.append(f"- **{g.title}** (`{g.key}`): {g.rule}")
    return "\n".join(lines) + "\n"
