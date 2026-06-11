"""
portfolio/thesis_confirm_tracker.py — persistent "consecutive weak evaluations" store
for the archetype `thesis_exit_requires_confirmation` switch.

Records, per held symbol, how many consecutive sell-scans the position has signalled a
soft thesis-weak exit (value_metric below the weak floor past its minimum hold). A
flagged archetype (e.g. quality_compounder) only exits once the streak reaches the
confirmation count — a single weak reading never dumps a compounder. The streak resets
to 0 the moment the weak signal clears, and the store is rewritten each run from the
symbols actually evaluated, so sold/exited names drop out. Mirrors the peak_prices.csv /
last_progress.csv idiom in PortfolioManager.
"""
from __future__ import annotations

import os

import pandas as pd

from core.paths import DATA_DIRECTORY

_WEAK_STREAK_CSV = os.path.join(DATA_DIRECTORY, "weak_streak.csv")


def load_weak_streak() -> dict[str, int]:
    """Load {symbol: consecutive_weak_evals}. Missing file / parse errors → empty dict."""
    out: dict[str, int] = {}
    try:
        df = pd.read_csv(_WEAK_STREAK_CSV)
    except FileNotFoundError:
        return out
    except Exception:
        return out
    if "symbol" not in df.columns or "weak_streak" not in df.columns:
        return out
    for _, row in df.iterrows():
        try:
            out[str(row["symbol"])] = max(0, int(row["weak_streak"]))
        except Exception:
            continue
    return out


def save_weak_streak(store: dict[str, int]) -> None:
    """Persist {symbol: consecutive_weak_evals} to weak_streak.csv (best-effort).

    Columns are explicit so an EMPTY store writes a header-only CSV — a
    column-less DataFrame writes a single newline, which pandas then refuses
    to parse (EmptyDataError) and which crashed the UI Data Explorer."""
    try:
        pd.DataFrame(
            [{"symbol": s, "weak_streak": int(n)} for s, n in store.items() if n],
            columns=["symbol", "weak_streak"],
        ).to_csv(_WEAK_STREAK_CSV, index=False)
    except Exception:
        pass
