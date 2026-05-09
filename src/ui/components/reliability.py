"""
ui/components/reliability.py — Data quality and reliability diagnostics.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from ui.utils import data_date, load_latest_csv, no_data_msg


_SCORE_COLS    = ["value_metric", "value_score", "quality_score", "income_score", "momentum_score"]
_FUND_COLS     = ["pe_ratio", "pb_ratio", "dividend_yield", "volume"]
_REQUIRED_COLS = _SCORE_COLS + _FUND_COLS


def render() -> None:
    st.title("🩺 Reliability Diagnostics")
    st.caption("Answers: can I trust today's scoring data?")

    df = load_latest_csv("agg_data")
    if df is None:
        st.warning(no_data_msg("agg_data"))
        return

    st.caption(f"Source: agg_data {data_date('agg_data')} | {len(df)} symbols")

    # ---- NaN coverage -----------------------------------------------------
    st.subheader("Feature coverage")
    nan_summary = []
    for col in _REQUIRED_COLS:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            pct   = n_nan / len(df)
            nan_summary.append({"column": col, "nan_count": n_nan, "nan_pct": pct,
                                 "coverage_pct": 1 - pct})
    if nan_summary:
        nan_df = pd.DataFrame(nan_summary).sort_values("coverage_pct")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(nan_df.set_index("column")["coverage_pct"])
        with c2:
            st.dataframe(nan_df.style.format({"nan_pct": "{:.1%}", "coverage_pct": "{:.1%}"}),
                         use_container_width=True)

    # ---- Yield trap flags -------------------------------------------------
    st.divider()
    st.subheader("Yield trap flags")
    if "yield_trap_flag" in df.columns:
        n_traps = int(df["yield_trap_flag"].astype(bool).sum())
        c1, c2 = st.columns(2)
        c1.metric("Yield trap flags", n_traps)
        c2.metric("Clean (no trap)", len(df) - n_traps)
        if "sector" in df.columns:
            trap_by_sector = df[df["yield_trap_flag"].astype(bool)]["sector"].value_counts()
            if not trap_by_sector.empty:
                st.bar_chart(trap_by_sector)

    # ---- Zero-score indicators --------------------------------------------
    st.divider()
    st.subheader("Zero / missing scores")
    for col in _SCORE_COLS:
        if col in df.columns:
            n_zero = (df[col] == 0.0).sum()
            n_miss = df[col].isna().sum()
            if n_zero + n_miss > 0:
                st.metric(f"{col} — zeros: {n_zero}, NaN: {n_miss}",
                           f"{(n_zero + n_miss)/len(df):.1%} of universe")

    # ---- Volume / liquidity -----------------------------------------------
    st.divider()
    st.subheader("Liquidity distribution")
    if "volume" in df.columns:
        vol = df["volume"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Median volume", f"{vol.median():,.0f}")
        c2.metric("Below 500k", f"{(vol < 500_000).sum()} ({(vol < 500_000).mean():.1%})")
        c3.metric("Below 100k", f"{(vol < 100_000).sum()}")
        st.bar_chart(vol.clip(upper=vol.quantile(0.95)).value_counts(bins=30, sort=False).sort_index())

    # ---- Score distributions ----------------------------------------------
    st.divider()
    st.subheader("Score distributions")
    tab_labels = [c for c in _SCORE_COLS if c in df.columns]
    if tab_labels:
        tabs = st.tabs(tab_labels)
        for tab, col in zip(tabs, tab_labels):
            with tab:
                data = df[col].dropna()
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean", f"{data.mean():.4f}")
                c2.metric("Std", f"{data.std():.4f}")
                c3.metric("Median", f"{data.median():.4f}")
                st.bar_chart(data.value_counts(bins=25, sort=False).sort_index())
