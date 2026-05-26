"""
ui/services/research_service.py — Research data service for UI components.

Wraps IC computation and snapshot loading so research components
don't import from factor_lab directly.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR


@st.cache_data(ttl=300)
def load_all_snapshots() -> list[pd.DataFrame]:
    """Load all snapshot parquet files. Returns [] if none exist."""
    snap_dir = DATA_DIR / "snapshots"
    if not snap_dir.exists():
        return []
    frames = []
    for p in sorted(snap_dir.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            pass
    return frames


@st.cache_data(ttl=300)
def compute_ic(
    factors: tuple[str, ...],
    horizons: tuple[int, ...],
    ic_type: str = "spearman",
) -> Optional[dict]:
    """
    Compute IC data for given factors and horizons.

    Returns dict with keys: ic_df, summary, decay — or None on failure.
    """
    try:
        from ui.components.factor_lab import _compute_ic_data
        return _compute_ic_data(factors, horizons, ic_type)
    except Exception:
        return None


def score_distributions(agg_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return score column distributions for quick display."""
    cols = [c for c in ("value_score", "quality_score", "income_score", "momentum_score", "value_metric") if c in agg_df.columns]
    return {c: agg_df[c].dropna() for c in cols}
