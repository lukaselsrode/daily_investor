"""
ui/services/data_service.py — Data loading service for UI components.

All UI data access should go through here rather than calling util.py or
filesystem helpers directly in component files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR, latest_csv_path, load_latest_csv, load_config_raw


@st.cache_data(ttl=120)
def get_agg_data() -> Optional[pd.DataFrame]:
    return load_latest_csv("agg_data")


@st.cache_data(ttl=120)
def get_agg_data_path() -> Optional[Path]:
    return latest_csv_path("agg_data")


@st.cache_data(ttl=60)
def get_config() -> dict:
    return load_config_raw()


@st.cache_data(ttl=300)
def list_snapshots() -> list[Path]:
    snap_dir = DATA_DIR / "snapshots"
    if not snap_dir.exists():
        return []
    return sorted(snap_dir.glob("*.parquet"))


@st.cache_data(ttl=300)
def list_csv_files() -> dict[str, Path]:
    """All CSVs in data/, keyed by filename."""
    return {p.name: p for p in sorted(DATA_DIR.glob("*.csv"))}


def get_csv(filename: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / filename
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_decision_outcomes() -> Optional[pd.DataFrame]:
    try:
        from portfolio.outcome_tracker import load_outcomes
        df = load_outcomes()
        return df if not df.empty else None
    except Exception:
        return None


def data_freshness() -> dict[str, str]:
    """Return {dataset_name: date_string} for key datasets."""
    result: dict[str, str] = {}
    for prefix in ("agg_data", "order_intents", "order_results"):
        p = latest_csv_path(prefix)
        result[prefix] = p.stem.split("_", 1)[1].replace("_", "-") if p else "—"
    return result
