"""
ui/components/decision_journal.py — Decision Journal tab for Portfolio section.

Displays the full history of HOLD/TRIM/HARVEST/EXIT decisions recorded to
decision_outcomes.parquet. Read-only — never feeds back into scoring.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from ui.components.common import empty_state, section, df_download


_DECISION_COLS = [
    "decision_date", "symbol", "record_type", "decision_state", "final_action",
    "executed_bool", "percent_change", "current_value_metric", "holding_days",
    "regime", "sector", "rationale_text",
]

_ACTION_COLOURS = {
    "EXIT":    "#e74c3c",
    "TRIM":    "#e67e22",
    "HARVEST": "#1abc9c",
    "HOLD":    "#2ecc71",
    "REVIEW":  "#9b59b6",
    "BUY":     "#3498db",
    "SKIP":    "#95a5a6",
    "WATCH":   "#f39c12",
}


def _colour_action(val: str) -> str:
    c = _ACTION_COLOURS.get(str(val).upper(), "#7f8c8d")
    return f"color: {c}; font-weight: 600"


def _load_outcomes() -> pd.DataFrame:
    try:
        from portfolio.outcome_tracker import load_outcomes
        return load_outcomes()
    except Exception:
        return pd.DataFrame()


def render() -> None:
    st.subheader("Decision Journal")
    st.caption(
        "Every hold evaluation and candidate decision recorded during live runs. "
        "Read-only — no feedback to factor engine."
    )

    df = _load_outcomes()
    if df.empty:
        empty_state(
            "No decisions recorded yet",
            "Run the bot at least once to populate the decision journal.",
        )
        return

    # ── Filters ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    with c1:
        rec_types = ["All"] + sorted(df["record_type"].dropna().unique().tolist()) if "record_type" in df.columns else ["All"]
        rt_filter = st.selectbox("Record type", rec_types, key="dj_rt")
    with c2:
        actions = ["All"] + sorted(df["final_action"].dropna().unique().tolist()) if "final_action" in df.columns else ["All"]
        act_filter = st.selectbox("Action", actions, key="dj_act")
    with c3:
        syms = ["All"] + sorted(df["symbol"].dropna().unique().tolist()) if "symbol" in df.columns else ["All"]
        sym_filter = st.selectbox("Symbol", syms, key="dj_sym")
    with c4:
        limit = st.number_input("Show last N rows", min_value=50, max_value=5000, value=500, step=50, key="dj_limit")

    # Apply filters
    view = df.copy()
    if rt_filter != "All" and "record_type" in view.columns:
        view = view[view["record_type"] == rt_filter]
    if act_filter != "All" and "final_action" in view.columns:
        view = view[view["final_action"] == act_filter]
    if sym_filter != "All" and "symbol" in view.columns:
        view = view[view["symbol"] == sym_filter]

    view = view.tail(int(limit))
    display_cols = [c for c in _DECISION_COLS if c in view.columns]
    view = view[display_cols]

    st.caption(f"Showing {len(view):,} rows (of {len(df):,} total)")

    # ── Summary metrics ───────────────────────────────────────────────────────
    if "final_action" in df.columns:
        counts = df["final_action"].value_counts()
        c_cols = st.columns(min(len(counts), 6))
        for col, (action, n) in zip(c_cols, counts.items()):
            col.metric(str(action), n)

    st.divider()

    # ── Table ─────────────────────────────────────────────────────────────────
    try:
        styled = view.style.applymap(_colour_action, subset=["final_action"]) if "final_action" in view.columns else view.style
        st.dataframe(styled, use_container_width=True, height=400)
    except Exception:
        st.dataframe(view, use_container_width=True, height=400)

    df_download(view, "decision_journal.csv")

    # ── Outcome rates (if outcomes backfilled) ────────────────────────────────
    section("Outcome Rates", "Populated after running `daily-investor update-outcomes`")

    outcome_cols = [c for c in ("premature_exit", "bad_hold", "good_trim", "good_exit", "outperformed_hold") if c in df.columns]
    if outcome_cols:
        rates = {}
        for col in outcome_cols:
            valid = df[col].dropna()
            if len(valid) > 0:
                rates[col] = f"{valid.mean():.1%} ({valid.sum():.0f}/{len(valid)})"
        if rates:
            rate_df = pd.DataFrame(rates.items(), columns=["Outcome", "Rate"]).set_index("Outcome")
            st.dataframe(rate_df, use_container_width=True)
        else:
            st.info("Outcome columns present but all null — run `daily-investor update-outcomes`.")
    else:
        st.info("Outcome data not yet available. Run `daily-investor update-outcomes` 7+ days after first decision.")
