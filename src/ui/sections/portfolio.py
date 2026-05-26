"""ui/sections/portfolio.py — Portfolio: holdings, exposure, decisions, attribution."""
from __future__ import annotations
import streamlit as st


def render() -> None:
    st.header("💼 Portfolio")
    st.caption("Current holdings, factor exposures, decision history, and P&L attribution.")

    tabs = st.tabs([
        "📊 Holdings",
        "⚖️ Exposure",
        "🌡️ Regime",
        "📓 Decision Journal",
        "📈 Attribution",
    ])

    with tabs[0]:
        from ui.components.portfolio import render as _r
        _r()

    with tabs[1]:
        from ui.components.exposure import render as _r
        _r()

    with tabs[2]:
        from ui.components.regime import render as _r
        _r()

    with tabs[3]:
        from ui.components.decision_journal import render as _r
        _r()

    with tabs[4]:
        from ui.components.attribution import render as _r
        _r()
