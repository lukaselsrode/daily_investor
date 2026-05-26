"""
ui/components/decision_quality.py — Decision calibration metrics.

Displays:
  - Premature exit rate (EXIT → stock gained ≥2% after)
  - Trim success rate (TRIM → stock declined or underperformed SPY)
  - Harvest regret rate (HARVEST → stock continued ≥10% after)
  - Bad hold rate (HOLD → stock fell ≥5% while SPY flat/up)
  - Sample sizes for each metric

Data source: data/decision_outcomes.parquet (written by outcome_tracker.py)
SAFE: read-only calibration display. Never writes factor scores or config.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


@st.cache_data(ttl=300)
def _load_calibration_summary() -> dict:
    try:
        from portfolio.outcome_tracker import get_calibration_summary
        return get_calibration_summary()
    except Exception as exc:
        return {"_error": str(exc)}


@st.cache_data(ttl=300)
def _load_outcomes() -> pd.DataFrame:
    try:
        from portfolio.outcome_tracker import load_outcomes
        return load_outcomes()
    except Exception:
        return pd.DataFrame()


def _pct_or_na(v) -> str:
    return f"{v:.0%}" if v is not None else "N/A"


def _delta_vs_baseline(v, baseline: float) -> str | None:
    if v is None:
        return None
    return f"{v - baseline:+.0%} vs {baseline:.0%} baseline"


def _render_summary(summary: dict) -> None:
    if "_error" in summary:
        st.error(f"Could not load calibration data: {summary['_error']}")
        return

    premature = summary.get("premature_exit_rate")
    trim_ok   = summary.get("trim_success_rate")
    harvest_r = summary.get("harvest_regret_rate")
    bad_hold  = summary.get("bad_hold_rate")

    n_exit    = summary.get("n_exit", 0)
    n_trim    = summary.get("n_trim", 0)
    n_harvest = summary.get("n_harvest", 0)
    n_hold    = summary.get("n_hold", 0)

    if all(v is None for v in [premature, trim_ok, harvest_r, bad_hold]):
        st.info(
            "No resolved outcomes yet. Calibration metrics appear after 30 days "
            "of decisions have been recorded and backfilled."
        )
        return

    st.subheader("Exit Decision Quality")
    st.caption(
        "Based on 30-day realized returns after each decision. "
        "Lower premature exit rate = exits were well-timed. "
        "Higher trim success rate = trims were well-timed."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Premature Exit Rate",
        _pct_or_na(premature),
        _delta_vs_baseline(premature, 0.20),
        delta_color="inverse",
        help=(
            f"% of EXIT decisions where stock gained ≥2% in the next 30 days. "
            f"Based on {n_exit} resolved exits. Lower is better."
        ),
    )
    c2.metric(
        "Trim Success Rate",
        _pct_or_na(trim_ok),
        _delta_vs_baseline(trim_ok, 0.50),
        help=(
            f"% of TRIM decisions where stock subsequently declined or "
            f"underperformed SPY within 30 days. Based on {n_trim} resolved trims. "
            f"Higher is better."
        ),
    )
    c3.metric(
        "Harvest Regret Rate",
        _pct_or_na(harvest_r),
        _delta_vs_baseline(harvest_r, 0.20),
        delta_color="inverse",
        help=(
            f"% of HARVEST exits where stock continued to gain ≥10% after the exit. "
            f"Based on {n_harvest} resolved harvests. Lower = better timed."
        ),
    )
    c4.metric(
        "Bad Hold Rate",
        _pct_or_na(bad_hold),
        _delta_vs_baseline(bad_hold, 0.25),
        delta_color="inverse",
        help=(
            f"% of HOLD decisions where stock fell ≥5% while SPY was flat or up. "
            f"Based on {n_hold} resolved holds. Lower is better."
        ),
    )


def _render_recent_outcomes(df: pd.DataFrame) -> None:
    st.subheader("Recent Resolved Outcomes")
    if df.empty:
        st.info("No outcome records found.")
        return

    outcome_cols = [
        "decision_date", "symbol", "decision_state",
        "percent_change", "future_30d_return", "future_30d_vs_spy",
        "premature_exit", "good_trim", "bad_hold", "good_exit",
    ]
    show = df[[c for c in outcome_cols if c in df.columns]].copy()
    resolved = show[show["future_30d_return"].notna()].sort_values(
        "decision_date", ascending=False
    ).head(50)

    if resolved.empty:
        st.info("No decisions with 30-day realized returns yet.")
        return

    for col in ["percent_change", "future_30d_return", "future_30d_vs_spy"]:
        if col in resolved.columns:
            resolved[col] = resolved[col].map(
                lambda v: f"{v:.1%}" if pd.notna(v) else ""
            )

    st.dataframe(resolved, use_container_width=True, hide_index=True)


def render() -> None:
    st.subheader("Decision Quality Calibration")
    st.caption(
        "Compares past exit and hold decisions against realized 30-day returns. "
        "This is for research and transparency only — it does NOT alter any factor "
        "weights or scoring logic."
    )

    summary = _load_calibration_summary()
    _render_summary(summary)

    st.markdown("---")
    df = _load_outcomes()
    _render_recent_outcomes(df)
