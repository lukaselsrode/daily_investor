"""
portfolio/progress_tracker.py — persistent "last progress" store for the
opportunity-cost ("max hold without progress") exit.

Records, per held symbol, the most recent date the position made progress (a fresh
high within reclaim_band OR momentum >= floor — see exit_analysis.is_progress). The
stall clock is days-since-last-progress; a progressing position resets it and is
never culled. Mirrors the peak_prices.csv idiom in PortfolioManager.

First-seen symbols (no row yet) are seeded to TODAY by the caller, so the feature
can never retroactively cull a long-standing holding on its first run after deploy.
"""
from __future__ import annotations

import datetime
import os

import pandas as pd

from core.paths import DATA_DIRECTORY

_LAST_PROGRESS_CSV = os.path.join(DATA_DIRECTORY, "last_progress.csv")


def load_last_progress() -> dict[str, datetime.date]:
    """Load {symbol: last_progress_date}. Missing file / parse errors → empty dict."""
    out: dict[str, datetime.date] = {}
    try:
        df = pd.read_csv(_LAST_PROGRESS_CSV)
    except FileNotFoundError:
        return out
    except Exception:
        return out
    if "symbol" not in df.columns or "last_progress_date" not in df.columns:
        return out
    for _, row in df.iterrows():
        try:
            out[str(row["symbol"])] = datetime.date.fromisoformat(str(row["last_progress_date"]))
        except Exception:
            continue
    return out


def save_last_progress(store: dict[str, datetime.date]) -> None:
    """Persist {symbol: last_progress_date} to last_progress.csv (best-effort)."""
    try:
        pd.DataFrame(
            [{"symbol": s, "last_progress_date": d.isoformat()} for s, d in store.items()]
        ).to_csv(_LAST_PROGRESS_CSV, index=False)
    except Exception:
        pass


def stall_days_for(
    symbol: str,
    store: dict[str, datetime.date],
    today: datetime.date,
) -> int | None:
    """Days since the symbol last made progress, or None if it has no record."""
    d = store.get(symbol)
    if d is None:
        return None
    return (today - d).days
