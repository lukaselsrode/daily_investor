"""ui/sections/odte.py — 0DTE: same-day options decision-support control surface.

DECISION-ONLY. Nothing here places, sizes, or cancels an order, and nothing calls a broker or an
LLM. The app reads/authors the local 0DTE store under ``data/odte/`` (secrets stay in ``~/0dte/``);
live broker/market values are fed by Hermes/MCP, never fabricated here. NVDA stays employer-blocked.

Cockpit first: the landing tab is live operations, and the research views sit behind it. All data
access goes through ``ui.services.odte_service`` — components render, they never join events.
"""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("🎰 0DTE")
    st.caption(
        "Same-day options decision-support — live rails and budget, the conversion funnel, the "
        "trade ledger, the execution-safety trail, and session replay. "
        "**No orders, no broker calls** — all decision-only."
    )

    tabs = st.tabs([
        "🎛️ Cockpit",
        "🪜 Funnel",
        "📒 Ledger",
        "🛡️ Rails",
        "🎬 Replay",
        "🌐 Context",
    ])

    # Imports stay inside each tab: the section loads on every rerun, and importing all six
    # components (plotly, pandas, the data layer) up front would pay for tabs nobody opened.
    with tabs[0]:
        from ui.components.odte_cockpit import render as _r
        _r()

    with tabs[1]:
        from ui.components.odte_funnel import render as _r
        _r()

    with tabs[2]:
        from ui.components.odte_ledger import render as _r
        _r()

    with tabs[3]:
        from ui.components.odte_rails import render as _r
        _r()

    with tabs[4]:
        from ui.components.odte_replay import render as _r
        _r()

    with tabs[5]:
        from ui.components.odte_context import render as _r
        _r()
