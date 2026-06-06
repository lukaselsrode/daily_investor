"""
portfolio/position_journal.py — Append-only position event journal.

Persists to data/position_journal.csv.
Used to track thesis changes, state transitions, and review events.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_JOURNAL_COLS = [
    "timestamp",
    "symbol",
    "event_type",
    "sleeve",
    "status",
    "price",
    "composite_score",
    "rank_pct",
    "rationale",
]

def _journal_path() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR / "position_journal.csv"


def load_journal(symbol: str | None = None, limit: int = 200) -> pd.DataFrame:
    """
    Load journal entries. If symbol is given, filter to that symbol.
    Returns empty DataFrame (with correct columns) if journal does not exist.
    """
    path = _journal_path()
    if not path.exists():
        return pd.DataFrame(columns=_JOURNAL_COLS)
    try:
        # Tolerate malformed rows (e.g. a foreign writer appended wrong-width
        # lines): skip bad lines and keep only rows matching our schema rather
        # than blanking the whole panel on a single parse error.
        df = pd.read_csv(path, on_bad_lines="skip")
        # Drop any rows whose columns don't match this journal's schema.
        if list(df.columns) != _JOURNAL_COLS:
            keep = [c for c in _JOURNAL_COLS if c in df.columns]
            df = df[keep] if keep else pd.DataFrame(columns=_JOURNAL_COLS)
        if symbol is not None:
            df = df[df["symbol"] == symbol]
        return df.tail(limit)
    except Exception as exc:
        logger.warning("Could not load position_journal.csv: %s", exc)
        return pd.DataFrame(columns=_JOURNAL_COLS)


def log_portfolio_review(positions: list[dict]) -> None:
    """
    Batch-log a HOLD_REVIEW event for every position in the current portfolio review.
    positions: list of dicts with keys matching _JOURNAL_COLS.
    """
    path = _journal_path()
    if not positions:
        return
    rows = pd.DataFrame(positions, columns=_JOURNAL_COLS)
    try:
        if path.exists():
            rows.to_csv(path, mode="a", header=False, index=False)
        else:
            rows.to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Could not batch-write position_journal.csv: %s", exc)
