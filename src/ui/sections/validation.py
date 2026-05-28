"""
ui/sections/validation.py — Validation: backtests, robustness, tuning, config debug.

Answers the core question: "Can I trust this strategy?"

Tab layout
----------
0  ▶ Run     — single backtest + random windows robustness check
1  🎯 Tune   — manual weights + random search + scipy optimizer + config variants
2  ⚙️ Config — diagnostics, compare, stability scan

Data pipeline reliability diagnostics live in System > Reliability.
"""
from __future__ import annotations

import streamlit as st


def _glossary():
    with st.expander("📖 Glossary"):
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Backtest** | Simulating the strategy over historical data as if you ran it in the past. |
| **Randomized window** | Sampling many random historical periods instead of one fixed period. |
| **Robust score** | Combined metric: median excess + 0.5·Sharpe + 0.25·% beating - penalties for drawdown/turnover/volatility. |
| **Excess return** | Strategy return minus benchmark (SPY) return. Positive = outperformance. |
| **Sharpe ratio** | Return per unit of volatility (annualized). Above 0.5 is good; above 1.0 is excellent. |
| **Calmar ratio** | Return divided by max drawdown depth. Higher = better risk-adjusted return. |
| **Max drawdown** | Worst peak-to-trough decline. |
| **Benchmark** | SPY by default (S&P 500 ETF). The strategy is compared against this. |
| **Config variant** | A saved copy of the config with modified parameters for comparison. |
| **Ablation** | Turning off one strategy component at a time to measure its contribution. |
        """)


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("✅ Validation")
    _glossary()

    tabs = st.tabs(["▶ Run", "🎯 Tune", "⚙️ Config"])

    with tabs[0]:
        from ui.components.backtests import render as _bt
        _bt()
        st.divider()
        from ui.components.random_windows import render as _rw
        _rw()

    with tabs[1]:
        from ui.components.weight_tuner import render as _wt
        _wt()

    with tabs[2]:
        inner = st.tabs(["🔍 Diagnostics", "📋 Compare", "🔭 Stability"])
        with inner[0]:
            from ui.components.config_diagnostics import render as _r
            _r()
        with inner[1]:
            from ui.components.config_compare import render as _r
            _r()
        with inner[2]:
            from ui.components.stability import render as _r
            _r()
