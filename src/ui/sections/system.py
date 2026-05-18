"""ui/pages/system.py — System: config, logs, runtime introspection."""
from __future__ import annotations
import streamlit as st


def render() -> None:
    st.header("⚙️ System")
    st.caption("Infrastructure and runtime introspection — config, logs, audit trail.")

    tab_cfg, tab_logs = st.tabs([
        "🛠️ Config",
        "📋 Logs & Audit",
    ])

    with tab_cfg:
        st.subheader("Configuration")
        st.caption("Read-only view of `cfg/config.yaml`. Apply changes manually.")
        from ui.components.config_viewer import render as _r
        _r()

    with tab_logs:
        st.subheader("Logs & Audit Trail")
        st.caption("Application log tail, order history, and audit CSVs.")
        from ui.components.logs import render as _r
        _r()
