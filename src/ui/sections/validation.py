"""ui/sections/validation.py — Validation: backtests, robustness, stability, reliability, config debug."""
from __future__ import annotations
import streamlit as st


def render() -> None:
    st.header("✅ Validation")
    st.caption("Answer: 'Can I trust this system?' — backtests, stability, config diagnostics, and data quality.")

    tabs = st.tabs([
        "📈 Backtests",
        "🔭 Stability & Robustness",
        "🩺 Reliability",
        "⚙️ Tuning",
        "🔍 Config Diagnostics",
        "📋 Config Compare",
        "🧪 Ablation Runner",
    ])

    with tabs[0]:
        st.subheader("Backtest Results")
        st.caption("Performance curves, drawdowns, trade statistics, and benchmark comparison.")
        from ui.components.backtests import render as _r
        _r()

    with tabs[1]:
        st.subheader("Stability & Robustness")
        st.caption(
            "Parameter sensitivity across time windows and objectives. "
            "Unstable parameters = overfit parameters."
        )
        from ui.components.stability import render as _r
        _r()

    with tabs[2]:
        st.subheader("Reliability Diagnostics")
        st.caption(
            "Data pipeline integrity: NaN rates, zero-score coverage, "
            "liquidity failures, and feature completeness."
        )
        from ui.components.reliability import render as _r
        _r()

    with tabs[3]:
        st.subheader("Parameter Tuning")
        st.caption(
            "Auto-tune factor weights and risk parameters. "
            "Review before applying — tuning suggestions are not auto-applied."
        )
        from ui.components.tuning import render as _r
        _r()

    with tabs[4]:
        from ui.components.config_diagnostics import render as _r
        _r()

    with tabs[5]:
        from ui.components.config_compare import render as _r
        _r()

    with tabs[6]:
        from ui.components.ablation_runner import render as _r
        _r()
