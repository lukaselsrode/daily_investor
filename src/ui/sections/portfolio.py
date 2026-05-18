"""ui/pages/portfolio.py — Portfolio: current holdings and exposures."""
from __future__ import annotations
import streamlit as st


def render() -> None:
    st.header("💼 Portfolio")
    st.caption("Current holdings, factor exposures, sector concentration, and P&L.")

    tab_hold, tab_exp, tab_reg = st.tabs([
        "📊 Holdings",
        "⚖️ Exposure",
        "🌡️ Regime",
    ])

    with tab_hold:
        from ui.components.portfolio import render as _r
        _r()

    with tab_exp:
        from ui.components.exposure import render as _r
        _r()

    with tab_reg:
        from ui.components.regime import render as _r
        _r()
