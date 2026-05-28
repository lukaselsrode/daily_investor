"""
ui/components/backtests.py — Backtest runner and results viewer.

Shows equity curve vs benchmark, drawdown chart, metric cards, and
an expandable trade/position log.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS, load_config_raw

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_backtest(n_days: int, mode: str, save_artifacts: bool = False,
                  cluster_tracking: bool = False, scope: str = "overall_strategy"):
    from ui.services.backtest_service import run_single_backtest
    return run_single_backtest(n_days=n_days, mode=mode,
                               save_artifacts=save_artifacts,
                               cluster_tracking=cluster_tracking,
                               scope=scope)


def _equity_chart(train_result, val_result=None, equity_override=None):
    """Render Plotly equity-vs-benchmark chart + drawdown panel."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("plotly not installed — install with `pip install plotly`.")
        return

    eq  = equity_override if equity_override is not None else train_result.equity_curve
    ben = train_result.benchmark_equity

    if len(eq) == 0:
        st.info("No equity curve data available for this result.")
        return

    days = np.arange(len(eq))

    # Index both series to 100 at day 0 for apples-to-apples comparison
    eq_idx  = eq  / max(eq[0],  1e-9) * 100.0
    ben_idx = ben / max(ben[0], 1e-9) * 100.0 if len(ben) == len(eq) else None

    # Drawdown from peak
    cum_max = np.maximum.accumulate(eq_idx)
    dd = np.where(cum_max > 0, eq_idx / cum_max - 1.0, 0.0)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("Portfolio Growth (indexed to 100)", "Drawdown from Peak"),
    )

    fig.add_trace(go.Scatter(
        x=days, y=eq_idx,
        name="Strategy",
        line=dict(color="#4c8ef5", width=2),
        hovertemplate="Day %{x}<br>Strategy: %{y:.1f}<extra></extra>",
    ), row=1, col=1)

    if ben_idx is not None:
        fig.add_trace(go.Scatter(
            x=days, y=ben_idx,
            name="Benchmark",
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
            hovertemplate="Day %{x}<br>Benchmark: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

    # Validation region shading
    if val_result is not None and len(val_result.equity_curve) > 0:
        val_start = len(eq)
        val_eq  = val_result.equity_curve
        val_idx = val_eq / max(val_eq[0], 1e-9) * 100.0
        val_days = np.arange(val_start, val_start + len(val_eq))

        fig.add_trace(go.Scatter(
            x=val_days, y=val_idx,
            name="Strategy (validation)",
            line=dict(color="#f5a623", width=2, dash="dash"),
        ), row=1, col=1)

        if len(val_result.benchmark_equity) == len(val_eq):
            vb_idx = val_result.benchmark_equity / max(val_result.benchmark_equity[0], 1e-9) * 100.0
            fig.add_trace(go.Scatter(
                x=val_days, y=vb_idx,
                name="Benchmark (validation)",
                line=dict(color="#cccccc", width=1, dash="dot"),
            ), row=1, col=1)

        fig.add_vrect(
            x0=val_start, x1=val_start + len(val_eq) - 1,
            fillcolor="orange", opacity=0.05,
            layer="below", line_width=0,
            row=1, col=1,
        )

    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=days, y=dd * 100,
        name="Drawdown %",
        fill="tozeroy",
        fillcolor="rgba(255,80,80,0.20)",
        line=dict(color="rgba(220,50,50,0.7)", width=1),
        hovertemplate="Day %{x}<br>Drawdown: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Index", row=1, col=1)
    fig.update_yaxes(title_text="DD %",  row=2, col=1)
    fig.update_xaxes(title_text="Trading days", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _metric_cards(train, rpt, val=None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy return (TWR)", f"{train.total_return:+.1%}")
    c2.metric("Benchmark (SPY)",       f"{rpt.benchmark_return:+.1%}")
    c3.metric("Excess return",         f"{rpt.excess_return:+.1%}",
              delta=f"{rpt.excess_return:+.1%}", delta_color="normal")
    c4.metric("Sharpe",                f"{train.sharpe:+.3f}")
    if getattr(rpt, "config_hash", "") or getattr(rpt, "run_timestamp", ""):
        _hash = getattr(rpt, "config_hash", "")
        _ts   = getattr(rpt, "run_timestamp", "")[:16]
        st.caption(f"Config hash: `{_hash}`  —  run at {_ts} UTC")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calmar",         f"{train.calmar:+.3f}")
    c2.metric("Max drawdown",   f"{train.max_drawdown:.1%}")
    c3.metric("Trades",         train.trades_made)
    c4.metric("Lookahead bias", rpt.lookahead_bias_level)

    if val is not None:
        st.caption("**Out-of-sample validation**")
        v1, v2, v3, v4 = st.columns(4)
        val_excess = val.total_return - rpt.validation_benchmark_return
        v1.metric("Val return",   f"{val.total_return:+.1%}")
        v2.metric("Val Sharpe",   f"{val.sharpe:+.3f}")
        v3.metric("Val drawdown", f"{val.max_drawdown:.1%}")
        v4.metric("Val excess",   f"{val_excess:+.1%}",
                  delta=f"{val_excess:+.1%}", delta_color="normal")


def _detail_expander(rpt, train):
    with st.expander("Full report details"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"- **Mode:** {rpt.mode}")
            st.markdown(f"- **Universe:** {rpt.universe_selection}")
            st.markdown(f"- **Symbols:** {rpt.n_symbols}")
            st.markdown(f"- **Days:** {rpt.n_days}")
        with c2:
            st.markdown(f"- **Benchmark Sharpe:** {rpt.benchmark_sharpe:+.3f}")
            st.markdown(f"- **Benchmark drawdown:** {rpt.benchmark_max_drawdown:.1%}")
            st.markdown(f"- **Avg positions:** {train.average_positions:.1f}")
            st.markdown(f"- **Avg cash %:** {train.average_cash_pct:.1%}")
            st.markdown(f"- **Friction cost:** ${train.friction_cost:,.2f}")
            st.markdown(f"- **Turnover est.:** {train.turnover_estimate:.2f}×")

        if train.regime_days:
            rd = train.regime_days
            st.markdown(
                f"- **Regime:** {rd.get('bullish', 0)}d bullish / "
                f"{rd.get('neutral', 0)}d neutral / "
                f"{rd.get('defensive', 0)}d defensive"
            )

        if rpt.notes:
            st.markdown("**Notes:** " + "; ".join(rpt.notes))

    # Exit breakdown
    total_sells = train.sells_made
    if total_sells > 0:
        with st.expander("Exit breakdown"):
            cols = st.columns(4)
            cols[0].metric("Stop-outs",    train.stopout_count)
            cols[1].metric("Trim exits",   train.trim_count)
            cols[2].metric("Harvest exits",train.harvest_count)
            cols[3].metric("Cooldown skips", train.cooldown_skips)

    # Candidate pool diagnostics
    if train.pool_diagnostics is not None:
        p = train.pool_diagnostics
        with st.expander("Day-0 candidate pool"):
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Candidates",    p.n_candidates)
            pc2.metric("Score cutoff",  f"{p.score_cutoff:.3f}")
            pc3.metric("Avg quality",   f"{p.avg_quality:.3f}")
            pc1.metric("Avg momentum",  f"{p.avg_momentum:.3f}")
            pc2.metric("Avg income",    f"{p.avg_income:.3f}")
            pc3.metric("Avg value",     f"{p.avg_value:.3f}")
            if p.sector_counts:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(
                        {"sector": list(p.sector_counts.keys()),
                         "candidates": list(p.sector_counts.values())}
                    ).sort_values("candidates", ascending=False),
                    use_container_width=True, hide_index=True,
                )

    # Trade log
    # Archetype breakdown
    if getattr(train, "archetype_pnl", None):
        with st.expander("Archetype breakdown"):
            from ui.components.archetype_diagnostics import render_archetype_breakdown
            render_archetype_breakdown(train)

    # Cluster concentration timeline
    if getattr(train, "cluster_result", None) is not None:
        with st.expander("Cluster concentration timeline"):
            from ui.components.cluster_diagnostics import render_cluster_diagnostics
            render_cluster_diagnostics(train.cluster_result)

    if train.trade_log:
        with st.expander(f"Trade log ({len(train.trade_log)} records)"):
            import pandas as pd
            rows = []
            for t in train.trade_log:
                rows.append({
                    "day":       getattr(t, "date", ""),
                    "symbol":    getattr(t, "symbol", ""),
                    "side":      getattr(t, "side", ""),
                    "qty":       f"{getattr(t, 'quantity', 0):.4f}",
                    "price":     f"${getattr(t, 'price', 0):.2f}",
                    "amount":    f"${getattr(t, 'amount', 0):.2f}",
                    "exit":      getattr(t, "exit_type", ""),
                    "archetype": getattr(t, "archetype", ""),
                    "pnl":       f"${getattr(t, 'pnl', 0):.2f}" if getattr(t, "pnl", None) is not None else "",
                    "hold_d":    getattr(t, "hold_days", ""),
                    "partial":   "✓" if getattr(t, "is_partial", False) else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------

def render() -> None:
    cfg    = load_config_raw()
    bt_cfg = cfg.get("backtest", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        n_days = st.number_input(
            "Look-back days", min_value=30, max_value=1000,
            value=90, step=30, key="bt_n_days",
        )
    with c2:
        mode = st.selectbox("Mode", BACKTEST_MODES, key="bt_mode",
                            help=LOOKAHEAD_LABELS.get(BACKTEST_MODES[0], ""))
    with c3:
        capital = st.number_input(
            "Starting capital ($)", min_value=1000.0,
            value=float(bt_cfg.get("starting_capital", 5000.0)),
            step=500.0, key="bt_capital",
        )

    scope = st.radio(
        "Backtest scope",
        ["overall_strategy", "active_sleeve_compounding"],
        format_func=lambda s: {
            "overall_strategy":          "Overall Strategy — full ETF + active portfolio",
            "active_sleeve_compounding": "Active Sleeve — stock-picking only, proceeds recycled",
        }[s],
        horizontal=True,
        key="bt_scope",
    )

    oc1, oc2 = st.columns(2)
    with oc1:
        cluster_tracking = st.checkbox(
            "Cluster concentration tracking",
            value=False, key="bt_cluster_tracking",
            help="Fit PCA+KMeans walk-forward at each rebalance date to track factor cluster concentration. Adds ~2s.",
        )
    with oc2:
        save_artifacts = st.checkbox(
            "Save artifacts",
            value=False, key="bt_save_artifacts",
            help="Save metrics, equity curve, and trade log to reports/backtests/",
        )

    bias_label = LOOKAHEAD_LABELS[mode]
    if "HIGH" in bias_label:
        st.error(f"⚠️ {bias_label}")

    if st.button("▶ Run backtest", type="primary", key="bt_run"):
        with st.spinner(f"Running {n_days}-day backtest…"):
            try:
                result = _run_backtest(n_days, mode, save_artifacts=save_artifacts,
                                      cluster_tracking=cluster_tracking, scope=scope)
                st.session_state["bt_result"] = result
                st.success("✅ Backtest complete.")
                if save_artifacts:
                    st.info("Artifacts saved to reports/backtests/")
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.exception(exc)
                return

    result = st.session_state.get("bt_result")
    if result is None:
        st.info("No backtest run yet. Configure settings above and click Run.")
        return

    st.divider()
    rpt   = result.report
    train = rpt.train_result
    val   = rpt.validation_result

    # Determine which scope the stored result was run under
    result_scope = getattr(train, "scope", "overall_strategy")

    st.subheader("Results")

    if result_scope == "active_sleeve_compounding":
        st.info(
            "ETF sleeve frozen at initial allocation. Harvest/trim proceeds recycled into "
            "active picks. Metrics below reflect the active sleeve only."
        )
        # Primary: active sleeve metrics
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Active return",   f"{train.active_total_return:+.1%}" if train.active_total_return is not None else "n/a")
        a2.metric("Active Sharpe",   f"{train.active_sharpe:+.3f}"       if train.active_sharpe is not None else "n/a")
        a3.metric("Active drawdown", f"{train.active_max_drawdown:.1%}"  if train.active_max_drawdown is not None else "n/a")
        excess_val = train.active_excess_return
        a4.metric("Active excess vs SPY",
                  f"{excess_val:+.1%}" if excess_val is not None else "n/a",
                  delta=f"{excess_val:+.1%}" if excess_val is not None else None,
                  delta_color="normal")
        if train.active_calmar is not None:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Active Calmar",  f"{train.active_calmar:+.3f}")
            b2.metric("Trades",         train.trades_made)
            b3.metric("Lookahead bias", rpt.lookahead_bias_level)
            b4.metric("Benchmark (SPY)", f"{rpt.benchmark_return:+.1%}")

        # Total portfolio metrics in expander
        with st.expander("Total portfolio (including ETF sleeve)"):
            _metric_cards(train, rpt, val)

        st.subheader("Active Equity Curve vs Benchmark")
        _equity_chart(train, val, equity_override=train.active_equity_curve)
    else:
        _metric_cards(train, rpt, val)
        st.subheader("Equity Curve vs Benchmark")
        _equity_chart(train, val)

    _detail_expander(rpt, train)

    with st.expander("Saved backtest runs"):
        try:
            import pandas as pd

            from backtesting.artifacts import list_saved_runs
            saved = list_saved_runs()
            if saved:
                st.dataframe(pd.DataFrame(saved), use_container_width=True, hide_index=True)
            else:
                st.caption("No saved runs yet. Check 'Save artifacts' before running.")
        except Exception as exc:
            st.caption(f"Could not load saved runs: {exc}")
