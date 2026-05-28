"""
ui/components/snapshot_health.py — Data health and snapshot inventory.

Shows:
  - Data file inventory (CSV count, date range, sizes)
  - Snapshot count and freshness
  - decision_outcomes.parquet row count / fill rate
  - Reports directory inventory
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.common import empty_state, metric_row, section
from ui.utils import DATA_DIR


def _file_size(p: Path) -> str:
    b = p.stat().st_size
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.1f} MB"


def _render_csv_inventory() -> None:
    section("CSV File Inventory", "All dated CSV files in data/")
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        empty_state("No CSV files", "Run the bot to generate data.")
        return

    rows = []
    for p in files:
        rows.append({
            "File": p.name,
            "Size": _file_size(p),
            "Modified": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(f"{len(files)} CSV files, total {sum(p.stat().st_size for p in files) / 1024:.0f} KB")


def _render_snapshot_inventory() -> None:
    section("Snapshot Inventory", "Parquet snapshots in data/snapshots/ — used for IC research")
    snap_dir = DATA_DIR / "snapshots"
    if not snap_dir.exists() or not list(snap_dir.glob("*.parquet")):
        st.info("No snapshot files yet — run the bot on multiple days to build up snapshots.")
        return

    snaps = sorted(snap_dir.glob("*.parquet"), reverse=True)
    st.metric("Total snapshots", len(snaps))

    rows = []
    for p in snaps[:50]:
        rows.append({"File": p.name, "Size": _file_size(p)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if len(snaps) > 50:
        st.caption(f"Showing 50 of {len(snaps)} snapshots")


def _render_outcomes_health() -> None:
    section("Decision Outcomes Health", "data/decision_outcomes.parquet")
    outcomes_path = DATA_DIR / "decision_outcomes.parquet"
    if not outcomes_path.exists():
        st.info("No decision_outcomes.parquet — run the bot to start recording decisions.")
        return

    try:
        from portfolio.outcome_tracker import load_outcomes
        df = load_outcomes()
    except Exception as exc:
        st.error(f"Could not load outcomes: {exc}")
        return

    if df.empty:
        st.info("Outcomes file exists but is empty.")
        return

    # Row counts
    total = len(df)
    by_type = df["record_type"].value_counts() if "record_type" in df.columns else pd.Series(dtype=int)
    holding_count = int(by_type.get("holding", 0))
    candidate_count = int(by_type.get("candidate", 0))

    metric_row([
        ("Total records", total, None),
        ("Holding decisions", holding_count, None),
        ("Candidate decisions", candidate_count, None),
    ])

    # Outcome fill rates
    outcome_cols = [c for c in ("future_30d_return", "future_90d_return", "premature_exit", "bad_hold") if c in df.columns]
    if outcome_cols:
        st.divider()
        st.caption("Outcome fill rates (null = not yet backfilled)")
        fill_rows = []
        for col in outcome_cols:
            non_null = df[col].notna().sum()
            fill_rows.append({"Column": col, "Filled": non_null, "Fill %": f"{non_null/total:.1%}"})
        st.dataframe(pd.DataFrame(fill_rows), use_container_width=True)

    # Date range
    if "decision_date" in df.columns:
        dates = df["decision_date"].dropna()
        if not dates.empty:
            st.caption(f"Date range: {dates.min()} → {dates.max()}")


def _render_reports_inventory() -> None:
    section("Reports Inventory", "Generated reports in reports/")
    root = DATA_DIR.parent
    reports_dir = root / "reports"
    if not reports_dir.exists():
        st.info("No reports/ directory yet.")
        return

    report_files = list(reports_dir.rglob("*.*"))
    if not report_files:
        st.info("No report files generated yet.")
        return

    rows = []
    for p in sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True)[:100]:
        rows.append({
            "File": str(p.relative_to(reports_dir)),
            "Size": _file_size(p),
            "Modified": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render() -> None:
    st.subheader("Snapshot / Data Health")
    st.caption("Data pipeline inventory — file counts, freshness, and outcome fill rates.")

    _render_outcomes_health()
    _render_snapshot_inventory()
    _render_csv_inventory()
    _render_reports_inventory()
