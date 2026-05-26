"""
ui/services/portfolio_service.py — Portfolio data service.

Loads holdings, P&L, decision outcomes, and exposure data.
Components should call these functions rather than hitting data files directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR


@st.cache_data(ttl=30)
def get_holdings() -> Optional[pd.DataFrame]:
    """Load most recent holdings CSV."""
    files = sorted(DATA_DIR.glob("holdings_*.csv"))
    if not files:
        return None
    try:
        return pd.read_csv(files[-1])
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_decision_outcomes(
    record_type: Optional[str] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """
    Load decision_outcomes.parquet, optionally filtered by record_type.
    Returns empty DataFrame if unavailable.
    """
    try:
        from portfolio.outcome_tracker import load_outcomes
        df = load_outcomes()
        if record_type and "record_type" in df.columns:
            df = df[df["record_type"] == record_type]
        return df.tail(limit)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_position_journal() -> Optional[pd.DataFrame]:
    """Load position journal CSV."""
    p = DATA_DIR / "position_journal.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def holding_decision_history(symbol: str) -> pd.DataFrame:
    """Return decision history for a single symbol from outcomes."""
    df = get_decision_outcomes()
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    return df[df["symbol"] == symbol].copy()
