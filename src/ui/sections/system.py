"""ui/sections/system.py — System: config, data health, logs, introspection."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("⚙️ System")
    st.caption("Infrastructure and runtime introspection — config, data health, logs, audit trail.")

    tabs = st.tabs([
        "🛠️ Config",
        "📋 Logs & Audit",
        "🗂️ Data Explorer",
        "🩺 Reliability",
        "📦 Snapshot Health",
    ])

    with tabs[0]:
        st.subheader("Configuration")
        st.caption("Read-only view of `cfg/config.yaml`. Apply changes manually.")
        from ui.components.config_viewer import render as _r
        _r()

    with tabs[1]:
        st.subheader("Logs & Audit Trail")
        st.caption("Application log tail, order history, and audit CSVs.")
        from ui.components.logs import render as _r
        _r()

    with tabs[2]:
        st.subheader("Data Explorer")
        st.caption("Browse any CSV in `data/`, build custom views, and inspect raw data.")
        from ui.components.data_explorer import render as _r
        _r()

    with tabs[3]:
        st.subheader("Reliability Diagnostics")
        st.caption("Data quality checks: NaN coverage, score distributions, liquidity flags.")
        from ui.components.reliability import render as _r
        _r()

    with tabs[4]:
        from ui.components.snapshot_health import render as _r
        _r()
