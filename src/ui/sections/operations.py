"""ui/sections/operations.py — Operations: live trading and execution workflow."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("⚡ Operations")
    st.caption("Mission control for live trading — run status, order management, execution, logs.")

    tabs = st.tabs([
        "🏠 Dashboard",
        "🚀 Run Control",
        "🎯 Order Intents",
        "⚡ Execute",
        "📋 Logs",
    ])

    with tabs[0]:
        from ui.components.home import render as _r
        _r()

    with tabs[1]:
        from ui.components.run_control import render as _r
        _r()

    with tabs[2]:
        from ui.components.intents import render as _r
        _r()

    with tabs[3]:
        from ui.components.execution import render as _r
        _r()

    with tabs[4]:
        from ui.components.logs import render as _r
        _r()
