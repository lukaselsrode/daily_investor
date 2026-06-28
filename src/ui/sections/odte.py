"""ui/sections/odte.py — 0DTE: same-day options decision-support control surface.

DECISION-ONLY. Nothing here places, sizes, or cancels an order, and nothing calls a broker or an
LLM. The app reads/authors the local 0DTE store under ``data/odte/`` (secrets stay in ``~/0dte/``);
live broker/market values are fed by Hermes/MCP, never fabricated here. NVDA stays employer-blocked.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("🎰 0DTE")
    st.caption(
        "Same-day options decision-support — social candidates, gamma/pin map, live-position "
        "discipline, and the decision journal. **No orders, no broker calls** — all decision-only."
    )

    tabs = st.tabs([
        "🏠 Dashboard",
        "📣 Social & Scrape",
        "🧲 Gamma Map",
        "🎯 Position",
        "📓 Journal",
        "🔎 FMP Context",
    ])

    with tabs[0]:
        from ui.components.odte_dashboard import render as _r
        _r()

    with tabs[1]:
        from ui.components.odte_social import render as _r
        _r()

    with tabs[2]:
        from ui.components.odte_gamma import render as _r
        _r()

    with tabs[3]:
        from ui.components.odte_position import render as _r
        _r()

    with tabs[4]:
        from ui.components.odte_journal import render as _r
        _r()

    with tabs[5]:
        from ui.components.odte_fmp import render as _r
        _r()
