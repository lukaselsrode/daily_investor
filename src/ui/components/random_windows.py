"""
ui/components/random_windows.py — Randomized walk-forward backtest UI.

Samples N random M-day windows from historical data and evaluates the current
strategy on each.  Answers: "Does the strategy consistently beat SPY, or
did it just get lucky in one historical period?"
"""
from __future__ import annotations

import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _distribution_chart(df, col: str, label: str, benchmark_col: str | None = None):
    """Histogram + optional benchmark overlay."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.write(df[[col]].describe())
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col] * 100,
        name=label,
        nbinsx=20,
        marker_color="#4c8ef5",
        opacity=0.75,
    ))
    if benchmark_col and benchmark_col in df.columns:
        fig.add_trace(go.Histogram(
            x=df[benchmark_col] * 100,
            name="Benchmark",
            nbinsx=20,
            marker_color="#aaaaaa",
            opacity=0.55,
        ))
    fig.add_vline(x=0, line_dash="dot", line_color="red", opacity=0.6)
    fig.update_layout(
        barmode="overlay",
        xaxis_title=f"{label} (%)",
        yaxis_title="Windows",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _scatter_chart(df):
    """Excess return vs Sharpe scatter."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    beats = df["beats_benchmark"] if "beats_benchmark" in df.columns else [True] * len(df)
    colors = ["#4c8ef5" if b else "#ff6b6b" for b in beats]

    fig = go.Figure(go.Scatter(
        x=df["sharpe"],
        y=df["excess_return"] * 100,
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.8),
        text=df["window_id"].astype(str),
        hovertemplate="Window %{text}<br>Sharpe: %{x:.2f}<br>Excess: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Sharpe",
        yaxis_title="Excess return (%)",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _summary_cards(summary):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median excess return",  f"{summary.median_excess_return:+.1%}")
    c2.metric("Median Sharpe",         f"{summary.median_sharpe:.3f}")
    c3.metric("% Beating benchmark",   f"{summary.pct_beating_benchmark:.0%}")
    c4.metric("Robust score",          f"{summary.robust_score:.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median strategy return", f"{summary.median_strategy_return:+.1%}")
    c2.metric("Median benchmark return",f"{summary.median_benchmark_return:+.1%}")
    c3.metric("Median max drawdown",    f"{summary.median_drawdown:.1%}")
    c4.metric("Worst-decile drawdown",  f"{summary.worst_decile_drawdown:.1%}")


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------

def render() -> None:
    scope = st.radio(
        "Backtest scope",
        ["overall_strategy", "active_sleeve_compounding"],
        format_func=lambda s: {
            "overall_strategy":          "Overall Strategy — full ETF + active portfolio",
            "active_sleeve_compounding": "Active Sleeve — stock-picking only, proceeds recycled",
        }[s],
        horizontal=True,
        key="rw_scope",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_days_load = st.number_input(
            "History (days)", min_value=120, max_value=1500,
            value=365, step=60, key="rw_n_days_load",
            help="Total days of price history to load. Windows are sampled from this range.",
        )
    with c2:
        window_days = st.number_input(
            "Window length (days)", min_value=20, max_value=250,
            value=60, step=10, key="rw_window_days",
            help="Length of each random window in trading days.",
        )
    with c3:
        n_windows = st.number_input(
            "Windows", min_value=5, max_value=100,
            value=20, step=5, key="rw_n_windows",
            help="How many random windows to evaluate. More = slower but more reliable.",
        )
    with c4:
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, key="rw_seed")
    with c5:
        cfg_mode = st.selectbox("Mode", BACKTEST_MODES, key="rw_mode",
                                help=LOOKAHEAD_LABELS.get(BACKTEST_MODES[0], ""))

    mode = st.session_state.get("rw_mode", BACKTEST_MODES[0])

    if window_days * 2 > n_days_load:
        st.warning(
            f"Window length ({window_days}d) is more than half the loaded history "
            f"({n_days_load}d). Consider loading more history for reliable results."
        )

    if st.button("▶ Run Random Window Backtest", type="primary", key="rw_run"):
        from backtesting.simulator import get_default_params
        params = get_default_params()

        progress_bar = st.progress(0, text="Running windows…")

        def _cb(current, total):
            pct = int(current / max(total, 1) * 100)
            progress_bar.progress(pct, text=f"Window {current + 1}/{total}…")

        try:
            from ui.services.backtest_service import run_random_windows
            summary = run_random_windows(
                n_days=n_days_load,
                mode=mode,
                params=params,
                n_windows=n_windows,
                window_days=window_days,
                seed=int(seed),
                progress_callback=_cb,
                scope=scope,
            )
            progress_bar.empty()
            st.session_state["rw_summary"] = summary
            st.success(f"✅ Evaluated {summary.n_windows} windows.")
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Random window backtest failed: {exc}")
            st.exception(exc)
            return

    summary = st.session_state.get("rw_summary")
    if summary is None:
        st.info("No results yet. Configure settings above and click Run.")
        return

    st.divider()
    st.subheader("Results")

    result_scope = getattr(summary, "scope", "overall_strategy")

    if result_scope == "active_sleeve_compounding":
        st.info(
            "Active Sleeve scope: robust_score shown is the active sleeve objective — "
            "optimizer will not improve results by shifting allocation to ETFs."
        )
        # Primary: active sleeve cards
        a1, a2, a3, a4 = st.columns(4)
        a_exc = summary.median_active_excess_return
        a_sh  = summary.median_active_sharpe
        a_beat = summary.pct_active_beating_benchmark
        a_score = summary.active_robust_score
        a1.metric("Active robust score",        f"{a_score:.4f}"  if a_score  is not None else "n/a")
        a2.metric("Median active excess",       f"{a_exc:+.1%}"   if a_exc    is not None else "n/a")
        a3.metric("Median active Sharpe",       f"{a_sh:.3f}"     if a_sh     is not None else "n/a")
        a4.metric("% Active beating benchmark", f"{a_beat:.0%}"   if a_beat   is not None else "n/a")
        # Total portfolio in expander
        with st.expander("Total portfolio summary (including ETF sleeve)"):
            _summary_cards(summary)
    else:
        _summary_cards(summary)

    df = summary.to_dataframe()

    st.subheader("Return distributions")
    left, right = st.columns(2)
    with left:
        st.caption("Excess return distribution")
        _distribution_chart(df, "excess_return", "Excess return", "benchmark_return")
    with right:
        st.caption("Sharpe vs excess return")
        _scatter_chart(df)

    with st.expander("Per-window results table"):
        display_df = df.copy()
        for col in ["strategy_return", "benchmark_return", "excess_return", "max_drawdown", "turnover"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("{:+.2%}".format)
        display_df["sharpe"] = df["sharpe"].map("{:.3f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("Interpreting the robust score"):
        if result_scope == "active_sleeve_compounding" and summary.active_robust_score is not None:
            st.markdown(f"""
| Metric | Value |
|---|---|
| Median active excess return | `{summary.median_active_excess_return:+.2%}` |
| Median active Sharpe | `{summary.median_active_sharpe:.3f}` |
| % Active beating benchmark | `{summary.pct_active_beating_benchmark:.0%}` |
| Worst-decile active drawdown | `{summary.worst_decile_active_drawdown:.1%}` |
| Median turnover | `{summary.median_turnover:.2f}×` |
| **Active robust score** | **`{summary.active_robust_score:.4f}`** |

The **active robust score** measures only the stock-picking engine's compounding ability.
ETF allocation is frozen at day-0 levels so it cannot inflate results.
            """)
        else:
            st.markdown(f"""
| Metric | Value |
|---|---|
| Median excess return | `{summary.median_excess_return:+.2%}` |
| Median Sharpe | `{summary.median_sharpe:.3f}` |
| % Beating benchmark | `{summary.pct_beating_benchmark:.0%}` |
| Worst-decile drawdown | `{summary.worst_decile_drawdown:.1%}` |
| Median turnover | `{summary.median_turnover:.2f}×` |
| Std excess return | `{summary.std_excess_return:.3f}` |
| **Robust score** | **`{summary.robust_score:.4f}`** |

The robust score combines all of these into a single number that rewards consistency,
not just average return.  Compare this number across different configs to see which
one performs better across many market conditions.
            """)
