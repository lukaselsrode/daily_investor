"""
ui/components/backtests.py — Backtest runner and results viewer.
Calls BacktestEngine directly (same path as CLI cmd_backtest).
"""

from __future__ import annotations

import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS, load_config_raw


def _run_backtest(n_days: int, mode: str, starting_capital: float):
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    return engine.run(n_days=n_days, mode=mode)


def render() -> None:
    st.title("📈 Backtests")
    st.caption("Run a simulation using the current config. No live orders are placed.")

    cfg = load_config_raw()
    bt_cfg = cfg.get("backtest", {})

    # ---- Controls ---------------------------------------------------------
    st.subheader("Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_days = st.number_input("Look-back days", min_value=30, max_value=1000,
                                  value=90, step=30)
    with c2:
        mode = st.selectbox("Mode", BACKTEST_MODES)
    with c3:
        capital = st.number_input("Starting capital ($)", min_value=1000.0,
                                   value=float(bt_cfg.get("starting_capital", 5000.0)), step=500.0)

    st.caption(f"⚠️ Lookahead bias: {LOOKAHEAD_LABELS[mode]}")
    if mode == "current_universe_stress_test":
        st.error("This mode uses current fundamentals throughout — it is a stress test, NOT a predictive backtest.")

    st.code(f"daily-investor backtest {n_days} --mode {mode}", language="bash")

    # ---- Run button -------------------------------------------------------
    if st.button("▶ Run backtest", type="primary"):
        with st.spinner(f"Running {n_days}-day backtest…"):
            try:
                result = _run_backtest(n_days, mode, capital)
                st.session_state["bt_result"] = result
                st.success("✅ Backtest complete.")
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.exception(exc)
                return

    # ---- Results ----------------------------------------------------------
    result = st.session_state.get("bt_result")
    if result is None:
        st.info("No backtest run yet. Configure settings above and click Run.")
        return

    st.divider()
    st.subheader("Results")

    rpt = result.report
    train = rpt.train_result

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return (TWR)", f"{train.total_return:+.1%}")
    c2.metric("Benchmark return",   f"{rpt.benchmark_return:+.1%}")
    c3.metric("Excess return",      f"{rpt.excess_return:+.1%}")
    c4.metric("Sharpe",             f"{train.sharpe:+.3f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calmar",             f"{train.calmar:+.3f}")
    c2.metric("Max drawdown",       f"{train.max_drawdown:.1%}")
    c3.metric("Trades",             train.trades_made)
    c4.metric("Lookahead bias",     rpt.lookahead_bias_level)

    with st.expander("Full report"):
        st.markdown(f"- **Mode:** {rpt.mode}")
        st.markdown(f"- **Universe selection:** {rpt.universe_selection}")
        st.markdown(f"- **Symbols:** {rpt.n_symbols}")
        st.markdown(f"- **Days:** {rpt.n_days}")
        st.markdown(f"- **Benchmark Sharpe:** {rpt.benchmark_sharpe:+.3f}")
        st.markdown(f"- **Benchmark max drawdown:** {rpt.benchmark_max_drawdown:.1%}")

        if rpt.validation_result:
            val = rpt.validation_result
            st.subheader("Validation (out-of-sample)")
            v1, v2, v3 = st.columns(3)
            v1.metric("Val return", f"{val.total_return:+.1%}")
            v2.metric("Val Sharpe", f"{val.sharpe:+.3f}")
            v3.metric("Val drawdown", f"{val.max_drawdown:.1%}")

        if rpt.notes:
            st.markdown("**Notes:** " + "; ".join(rpt.notes))
