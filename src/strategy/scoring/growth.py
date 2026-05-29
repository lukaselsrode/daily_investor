"""
strategy/scoring/growth.py — Placeholder for peer-relative growth/leadership.

The current snapshot schema has no earnings-surprise, revenue-growth, estimate-
revisions, or margin-expansion columns. Per the scoring plan: don't fake
fundamentals — only wire clean hooks/config/diagnostics so that future data
can plug in without refactoring the composite.

Output columns: none (no-op).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_growth(df: pd.DataFrame, scoring_cfg: dict | None = None) -> None:
    """No-op stub. Logs a single info line when growth_leadership is configured."""
    from util import SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    factor = cfg.get("factors", {}).get("growth_leadership", {})
    if not factor.get("enabled", False):
        return

    logger.info(
        "growth_leadership enabled in config but no fundamentals data available "
        "(earnings surprise / revenue growth / estimate revisions not in snapshot schema) — "
        "skipping. Hook is here for future data wiring."
    )
