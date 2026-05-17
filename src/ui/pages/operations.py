"""ui/pages/operations.py — Operations: live trading and execution workflow."""
from __future__ import annotations
import streamlit as st


def render() -> None:
    st.header("⚡ Operations")
    st.caption("Mission control for live trading — run status, order management, execution.")

    tab_dash, tab_run, tab_intents, tab_exec = st.tabs([
        "🏠 Dashboard",
        "🚀 Run Control",
        "🎯 Order Intents",
        "⚡ Execute",
    ])

    with tab_dash:
        from ui.components.home import render as _r
        _r()

    with tab_run:
        from ui.components.run_control import render as _r
        _r()

    with tab_intents:
        from ui.components.intents import render as _r
        _r()

    with tab_exec:
        from ui.components.execution import render as _r
        _r()
