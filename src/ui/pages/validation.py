"""ui/pages/validation.py — Validation: backtests, robustness, stability, reliability."""
from __future__ import annotations
import streamlit as st


def render() -> None:
    st.header("✅ Validation")
    st.caption("Answer: 'Can I trust this system?' — backtests, stability, and data quality.")

    tab_bt, tab_stab, tab_rel, tab_tune = st.tabs([
        "📈 Backtests",
        "🔭 Stability & Robustness",
        "🩺 Reliability",
        "⚙️ Tuning",
    ])

    with tab_bt:
        st.subheader("Backtest Results")
        st.caption("Performance curves, drawdowns, trade statistics, and benchmark comparison.")
        from ui.components.backtests import render as _r
        _r()

    with tab_stab:
        st.subheader("Stability & Robustness")
        st.caption(
            "Parameter sensitivity across time windows and objectives. "
            "Unstable parameters = overfit parameters."
        )
        from ui.components.stability import render as _r
        _r()

    with tab_rel:
        st.subheader("Reliability Diagnostics")
        st.caption(
            "Data pipeline integrity: NaN rates, zero-score coverage, "
            "liquidity failures, and feature completeness."
        )
        from ui.components.reliability import render as _r
        _r()

    with tab_tune:
        st.subheader("Parameter Tuning")
        st.caption(
            "Auto-tune factor weights and risk parameters. "
            "Review before applying — tuning suggestions are not auto-applied."
        )
        from ui.components.tuning import render as _r
        _r()
