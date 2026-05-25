"""
portfolio/position_journal.py — Append-only position event journal.

Persists to data/position_journal.csv.
Used to track thesis changes, state transitions, and review events.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

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

_EVENT_TYPES = frozenset([
    "BUY",
    "SELL",
    "HOLD_REVIEW",
    "WATCH",
    "EXIT_SIGNAL",
    "HARVEST",
    "THESIS_CHANGED",
    "SCORE_DETERIORATED",
    "RANK_DETERIORATED",
])


def _journal_path() -> Path:
    from ui.utils import DATA_DIR
    return DATA_DIR / "position_journal.csv"


def load_journal(symbol: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
    """
    Load journal entries. If symbol is given, filter to that symbol.
    Returns empty DataFrame (with correct columns) if journal does not exist.
    """
    path = _journal_path()
    if not path.exists():
        return pd.DataFrame(columns=_JOURNAL_COLS)
    try:
        df = pd.read_csv(path)
        if symbol is not None:
            df = df[df["symbol"] == symbol]
        return df.tail(limit)
    except Exception as exc:
        logger.warning("Could not load position_journal.csv: %s", exc)
        return pd.DataFrame(columns=_JOURNAL_COLS)


def log_event(
    symbol: str,
    event_type: str,
    sleeve: str = "",
    status: str = "",
    price: Optional[float] = None,
    composite_score: Optional[float] = None,
    rank_pct: Optional[float] = None,
    rationale: str = "",
) -> None:
    """Append one event to position_journal.csv."""
    if event_type not in _EVENT_TYPES:
        logger.warning("Unknown journal event type: %s", event_type)

    row = pd.DataFrame([{
        "timestamp":       datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":          symbol,
        "event_type":      event_type,
        "sleeve":          sleeve,
        "status":          status,
        "price":           price,
        "composite_score": composite_score,
        "rank_pct":        rank_pct,
        "rationale":       rationale,
    }])

    path = _journal_path()
    try:
        if path.exists():
            row.to_csv(path, mode="a", header=False, index=False)
        else:
            row.to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Could not write to position_journal.csv: %s", exc)


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
