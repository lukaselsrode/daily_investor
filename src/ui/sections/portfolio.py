"""ui/sections/portfolio.py — Portfolio: holdings, exposure, decisions, attribution."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    tabs = st.tabs(["📊 Portfolio", "📈 Performance", "📓 Decisions"])

    # ── Tab 0: Portfolio ──────────────────────────────────────────────────────
    with tabs[0]:
        inner = st.tabs(["📊 Holdings", "🏦 Allocation", "⚖️ Exposure", "🌡️ Regime"])

        with inner[0]:
            from ui.components.portfolio import render as _r
            _r()

        with inner[1]:
            from ui.components.allocation_diagnostics import render as _r
            _r()

        with inner[2]:
            from ui.components.exposure import render as _r
            _r()

        with inner[3]:
            from ui.components.regime import render as _r
            _r()

    # ── Tab 1: Performance ────────────────────────────────────────────────────
    with tabs[1]:
        inner = st.tabs(["📈 Attribution", "🧬 Archetype Attribution"])

        with inner[0]:
            from ui.components.attribution import render as _r
            _r()

        with inner[1]:
            from ui.components.archetype_attribution import render as _r
            _r()

    # ── Tab 2: Decisions ──────────────────────────────────────────────────────
    with tabs[2]:
        inner = st.tabs(["📓 Decision Journal", "🎯 Decision Quality"])

        with inner[0]:
            from ui.components.decision_journal import render as _r
            _r()

        with inner[1]:
            from ui.components.decision_quality import render as _r
            _r()
