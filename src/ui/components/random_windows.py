"""
ui/components/random_windows.py — Robust Window Scan UI.

Answers the question: "Does this strategy work across many random market slices,
horizons, and seeds — or did it just get lucky?"

Instead of exposing individual knobs (window_days, seed, n_windows), the user
selects a robustness profile (how thorough) and a horizon profile (which time
lengths to test).  The system expands these into a run matrix and reports results
by horizon and by seed.

Advanced controls are available behind an expander for custom research.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS

# ---------------------------------------------------------------------------
# Chart helpers (unchanged from previous implementation)
# ---------------------------------------------------------------------------

def _fan_chart(summary, use_active: bool = False) -> None:
    """Overlay all window equity curves indexed to 100, with median + IQR band."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("plotly not installed — pip install plotly")
        return

    curves = []
    bench_curves = []
    for w in summary.window_results:
        src = w.active_equity_curve if use_active else w.equity_curve
        if src is None or len(src) < 2:
            continue
        v0 = max(src[0], 1e-9)
        curves.append(src / v0 * 100.0)
        if w.benchmark_equity is not None and len(w.benchmark_equity) == len(src):
            bench_curves.append(w.benchmark_equity * 100.0)

    if not curves:
        st.info("No equity curves available — re-run with current version.")
        return

    max_len = max(len(c) for c in curves)
    days = np.arange(max_len)

    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c

    p25 = np.nanpercentile(padded, 25, axis=0)
    p50 = np.nanpercentile(padded, 50, axis=0)
    p75 = np.nanpercentile(padded, 75, axis=0)

    bench_median = None
    if bench_curves:
        padded_b = np.full((len(bench_curves), max_len), np.nan)
        for i, c in enumerate(bench_curves):
            padded_b[i, : len(c)] = c
        bench_median = np.nanmedian(padded_b, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(76,142,245,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IQR (25–75%)",
        hoverinfo="skip",
        showlegend=True,
    ))
    for i, c in enumerate(curves):
        fig.add_trace(go.Scatter(
            x=np.arange(len(c)), y=c,
            mode="lines",
            line=dict(color="rgba(76,142,245,0.20)", width=1),
            name=f"Window {i}",
            showlegend=False, hoverinfo="skip",
        ))
    if bench_median is not None:
        fig.add_trace(go.Scatter(
            x=days, y=bench_median,
            name="Benchmark median",
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
            hovertemplate="Day %{x}<br>Bench: %{y:.1f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=days, y=p50,
        name="Strategy median",
        line=dict(color="#4c8ef5", width=2.5),
        hovertemplate="Day %{x}<br>Median: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
    title = "Active sleeve" if use_active else "Strategy"
    fig.update_layout(
        title=f"{title} equity across {len(curves)} windows (indexed to 100)",
        xaxis_title="Trading days", yaxis_title="Index (100 = start)",
        height=380, margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _drawdown_fan_chart(summary, use_active: bool = False) -> None:
    """Overlay drawdown paths for all windows."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    dd_curves = []
    for w in summary.window_results:
        src = w.active_equity_curve if use_active else w.equity_curve
        if src is None or len(src) < 2:
            continue
        v0 = max(src[0], 1e-9)
        idx = src / v0 * 100.0
        cum_max = np.maximum.accumulate(idx)
        dd = np.where(cum_max > 0, idx / cum_max - 1.0, 0.0) * 100.0
        dd_curves.append(dd)

    if not dd_curves:
        return

    max_len = max(len(d) for d in dd_curves)
    days = np.arange(max_len)
    padded = np.full((len(dd_curves), max_len), np.nan)
    for i, d in enumerate(dd_curves):
        padded[i, : len(d)] = d

    p10 = np.nanpercentile(padded, 10, axis=0)
    p50 = np.nanpercentile(padded, 50, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([np.zeros(max_len), p10[::-1]]),
        fill="toself", fillcolor="rgba(255,80,80,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Worst-decile DD band", hoverinfo="skip",
    ))
    for d in dd_curves:
        fig.add_trace(go.Scatter(
            x=np.arange(len(d)), y=d,
            mode="lines", line=dict(color="rgba(220,50,50,0.18)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=days, y=p50,
        name="Median drawdown",
        line=dict(color="rgba(220,50,50,0.8)", width=2),
        hovertemplate="Day %{x}<br>DD: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="Drawdown paths across windows",
        xaxis_title="Trading days", yaxis_title="Drawdown (%)",
        height=240, margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _distribution_chart(df, col: str, label: str, benchmark_col: str | None = None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.write(df[[col]].describe())
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[col] * 100, name=label, nbinsx=20,
                               marker_color="#4c8ef5", opacity=0.75))
    if benchmark_col and benchmark_col in df.columns:
        fig.add_trace(go.Histogram(x=df[benchmark_col] * 100, name="Benchmark",
                                   nbinsx=20, marker_color="#aaaaaa", opacity=0.55))
    fig.add_vline(x=0, line_dash="dot", line_color="red", opacity=0.6)
    fig.update_layout(barmode="overlay", xaxis_title=f"{label} (%)",
                      yaxis_title="Windows", height=280,
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def _scatter_chart(df):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    beats = df["beats_benchmark"] if "beats_benchmark" in df.columns else [True] * len(df)
    colors = ["#4c8ef5" if b else "#ff6b6b" for b in beats]
    fig = go.Figure(go.Scatter(
        x=df["sharpe"], y=df["excess_return"] * 100,
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.8),
        text=df["window_id"].astype(str),
        hovertemplate="Window %{text}<br>Sharpe: %{x:.2f}<br>Excess: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(xaxis_title="Sharpe", yaxis_title="Excess return (%)",
                      height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


def _summary_cards(summary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median excess return",   f"{summary.median_excess_return:+.1%}")
    c2.metric("Median Sharpe",          f"{summary.median_sharpe:.3f}")
    c3.metric("% Beating benchmark",    f"{summary.pct_beating_benchmark:.0%}")
    c4.metric("Robust score",           f"{summary.robust_score:.4f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median strategy return", f"{summary.median_strategy_return:+.1%}")
    c2.metric("Median benchmark return",f"{summary.median_benchmark_return:+.1%}")
    c3.metric("Median max drawdown",    f"{summary.median_drawdown:.1%}")
    c4.metric("Worst-decile drawdown",  f"{summary.worst_decile_drawdown:.1%}")


# ---------------------------------------------------------------------------
# New: horizon heatmap + seed heatmap
# ---------------------------------------------------------------------------

def _horizon_heatmap(scan_result) -> None:
    """Colored table: rows = horizon lengths, cols = key metrics."""
    df = scan_result.horizon_heatmap_df()
    if df.empty:
        return
    st.subheader("Horizon heatmap")

    # Format for display
    disp = df.copy()
    for col in ["median excess", "median DD"]:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda v: f"{v:+.1%}" if not (v != v) else "—")
    for col in ["median Sharpe", "robust score"]:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda v: f"{v:.3f}" if not (v != v) else "—")
    if "% beating" in disp.columns:
        disp["% beating"] = disp["% beating"].map(lambda v: f"{v:.0%}" if not (v != v) else "—")
    if "horizon (days)" in disp.columns:
        disp["horizon (days)"] = disp["horizon (days)"].astype(str) + "d"

    overfit = scan_result.overfit_warning_score()
    if overfit > 0.5:
        st.warning(
            f"Overfit warning score: {overfit:.0%} — strategy only beats benchmark on "
            f"{len(df) - round(overfit * len(df))}/{len(df)} horizons. "
            "Results may be luck, not skill."
        )
    elif overfit > 0.2:
        st.info(f"Moderate horizon inconsistency (overfit score: {overfit:.0%}).")

    st.dataframe(disp, use_container_width=True, hide_index=True)


def _seed_heatmap(scan_result) -> None:
    """Table: rows = seeds, cols = median excess return per horizon + overall."""
    df = scan_result.seed_stability_df()
    if df.empty or len(df) < 2:
        return  # only interesting when multiple seeds
    st.subheader("Seed stability")

    disp = df.copy()
    for col in df.columns:
        if col == "seed":
            continue
        disp[col] = disp[col].map(lambda v: f"{v:+.1%}" if not (v != v) else "—")

    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        "Each row is a different random seed. Similar values across rows → "
        "results are seed-stable. Large variance → config is sensitive to luck."
    )


def _scan_kpi_row(scan_result) -> None:
    """Top-level KPIs from the multi-cell aggregate."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall robust score",    f"{scan_result.overall_robust_score:.4f}")
    c2.metric("Median excess return",    f"{scan_result.median_excess_return:+.1%}")
    c3.metric("Median Sharpe",           f"{scan_result.median_sharpe:.3f}")
    c4.metric("% Cells beating SPY",     f"{scan_result.pct_cells_beating_benchmark:.0%}")
    c1, c2, c3, c4 = st.columns(4)
    n_cells    = len(scan_result.cells)
    n_horizons = len({c.horizon_days for c in scan_result.cells})
    n_seeds    = len({c.seed for c in scan_result.cells})
    total_wins = int(scan_result.pct_cells_beating_benchmark * n_cells)
    c1.metric("Cells run",         str(n_cells))
    c2.metric("Horizons tested",   str(n_horizons))
    c3.metric("Seeds tested",      str(n_seeds))
    c4.metric("Cells beating SPY", f"{total_wins}/{n_cells}")


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------

def render() -> None:
    from tuning.profiles import (
        HORIZON_PROFILES,
        ROBUSTNESS_PROFILES,
        effort_caption,
        expand_run_matrix,
        total_simulations,
    )

    # Surface the regime de-risk overlay state (active when frac>0).
    from ui.utils import render_overlay_banner
    render_overlay_banner()

    # ── Scope ──────────────────────────────────────────────────────────────
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

    # ── Profile selectors ─────────────────────────────────────────────────
    col_r, col_h, col_days = st.columns([1, 1, 1])

    with col_r:
        robustness = st.selectbox(
            "Robustness",
            list(ROBUSTNESS_PROFILES),
            index=1,  # default: standard
            format_func=lambda k: {
                "quick":      "Quick — fast sanity check",
                "standard":   "Standard — normal research",
                "deep":       "Deep — stronger robustness",
                "exhaustive": "Exhaustive — overnight",
            }[k],
            key="rw_robustness",
        )

    with col_h:
        horizon = st.selectbox(
            "Horizon profile",
            list(HORIZON_PROFILES),
            index=3,  # default: mixed
            format_func=lambda k: {
                "short":  "Short-term (30–90d)",
                "medium": "Medium-term (90–180d)",
                "long":   "Long-term (180–365d)",
                "mixed":  "Mixed (30–365d)",
            }[k],
            key="rw_horizon",
        )

    with col_days:
        n_days_load = st.slider(
            "History (days)", min_value=365, max_value=1500,
            value=730, step=60, key="rw_n_days_load",
            help="Total price history to load. Must be larger than the longest horizon.",
        )

    # ── Advanced expander ──────────────────────────────────────────────────
    custom_horizons = None
    custom_seeds    = None
    windows_override = None
    mode = BACKTEST_MODES[0]

    with st.expander("Advanced options", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            custom_h_str = st.text_input(
                "Custom horizons (days, comma-sep)", value="",
                key="rw_custom_horizons",
                help="Override horizon profile. Example: 45,90,135",
            )
            if custom_h_str.strip():
                try:
                    custom_horizons = [int(x.strip()) for x in custom_h_str.split(",") if x.strip()]
                except ValueError:
                    st.warning("Invalid horizon list — using profile default.")
                    custom_horizons = None
        with adv_col2:
            custom_s_str = st.text_input(
                "Custom seeds (comma-sep)", value="",
                key="rw_custom_seeds",
                help="Override seeds. Example: 7,42,99",
            )
            if custom_s_str.strip():
                try:
                    custom_seeds = [int(x.strip()) for x in custom_s_str.split(",") if x.strip()]
                except ValueError:
                    st.warning("Invalid seed list — using profile default.")
                    custom_seeds = None
        with adv_col3:
            w_override = st.number_input(
                "Windows per horizon override", min_value=0, max_value=100,
                value=0, step=1, key="rw_windows_override",
                help="0 = use profile default.",
            )
            windows_override = int(w_override) if w_override > 0 else None
        with adv_col4:
            mode = st.selectbox("Mode", BACKTEST_MODES, key="rw_mode",
                                help=LOOKAHEAD_LABELS.get(BACKTEST_MODES[0], ""))

    # ── Effort estimate + validation ───────────────────────────────────────
    try:
        run_matrix = expand_run_matrix(
            robustness, horizon,
            custom_horizons=custom_horizons,
            custom_seeds=custom_seeds,
            windows_override=windows_override,
        )
        caption = effort_caption(
            robustness, horizon,
            custom_horizons=custom_horizons,
            custom_seeds=custom_seeds,
            windows_override=windows_override,
        )
        st.caption(caption)
    except ValueError as exc:
        st.error(str(exc))
        return

    max_horizon = max(c["horizon_days"] for c in run_matrix)
    if max_horizon * 2 > n_days_load:
        st.warning(
            f"Longest horizon ({max_horizon}d) is more than half the loaded history "
            f"({n_days_load}d). Increase history or shorten the horizon profile."
        )

    if mode == "walk_forward_price_only_test" and scope == "active_sleeve_compounding":
        st.warning(
            "⚠️ **walk_forward_price_only_test strips fundamental scores** — quality, income, "
            "and PE/PB are zeroed out to eliminate lookahead bias. The active sleeve needs "
            "those scores to rank stocks, so 0 trades will fire and results are meaningless. "
            "Use **liquid_universe_full** for active-sleeve robustness scans."
        )


    # ── Run ────────────────────────────────────────────────────────────────
    n_sims = total_simulations(run_matrix)
    if st.button(f"▶ Run Robust Window Scan ({n_sims} simulations)", type="primary", key="rw_run"):
        from backtesting.simulator import get_default_params
        params = get_default_params()

        n_cells = len(run_matrix)
        progress_bar = st.progress(0, text="Initialising…")

        def _cb(current, total_cells):
            pct  = int(current / max(total_cells, 1) * 100)
            cell = run_matrix[current] if current < len(run_matrix) else run_matrix[-1]
            progress_bar.progress(
                pct,
                text=f"Cell {current + 1}/{total_cells}: "
                     f"{cell['horizon_days']}d window, seed={cell['seed']}…"
            )

        try:
            from ui.services.backtest_service import run_robust_scan as _svc
            scan = _svc(
                n_days=n_days_load,
                run_matrix=run_matrix,
                mode=mode,
                params=params,
                scope=scope,
                progress_callback=_cb,
            )
            progress_bar.empty()
            st.session_state["rw_scan_result"] = scan
            n_done = len(scan.cells)
            st.success(f"Completed {n_done}/{n_cells} cells — {n_sims} simulations total.")
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Robust Window Scan failed: {exc}")
            st.exception(exc)
            return

    scan = st.session_state.get("rw_scan_result")
    if scan is None:
        st.info("No results yet. Configure settings above and click Run.")
        return

    st.divider()
    st.subheader("Results")

    result_scope = getattr(scan, "scope", "overall_strategy")
    is_active = result_scope == "active_sleeve_compounding"

    # ── Aggregate KPIs ────────────────────────────────────────────────────
    if is_active:
        st.info(
            "Active Sleeve scope: all metrics reflect the stock-picking engine only. "
            "ETF allocation is frozen at day-0 levels."
        )
    _scan_kpi_row(scan)

    # ── Horizon + seed heatmaps ────────────────────────────────────────────
    _horizon_heatmap(scan)
    _seed_heatmap(scan)

    st.divider()

    # ── Fan charts and per-window tables aggregated across all cells ───────
    display_summary = scan.aggregate_summary()

    if display_summary is None:
        return

    tab_equity, tab_dist, tab_table, tab_cells = st.tabs([
        "Equity curves", "Distributions", "Per-window table", "Per-cell table"
    ])

    with tab_equity:
        if is_active:
            st.caption("Active sleeve equity (indexed to 100) — largest-cell summary shown")
            _fan_chart(display_summary, use_active=True)
            with st.expander("Total portfolio equity fan"):
                _fan_chart(display_summary, use_active=False)
        else:
            st.caption("Strategy equity across windows (indexed to 100 at window start)")
            _fan_chart(display_summary, use_active=False)
        _drawdown_fan_chart(display_summary, use_active=is_active)

    with tab_dist:
        df = display_summary.to_dataframe()
        left, right = st.columns(2)
        with left:
            st.caption("Excess return distribution")
            _distribution_chart(df, "excess_return", "Excess return", "benchmark_return")
        with right:
            st.caption("Sharpe vs excess return")
            _scatter_chart(df)

    with tab_table:
        df = display_summary.to_dataframe()
        disp = df.copy()
        for col in ["strategy_return", "benchmark_return", "excess_return", "max_drawdown", "turnover"]:
            if col in disp.columns:
                disp[col] = disp[col].map("{:+.2%}".format)
        if "sharpe" in disp.columns:
            disp["sharpe"] = df["sharpe"].map("{:.3f}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab_cells:
        import pandas as pd
        rows = []
        for cell in scan.cells:
            sm = cell.summary
            use_active_m = is_active and getattr(sm, "active_robust_score", None) is not None
            rows.append({
                "horizon (d)":   cell.horizon_days,
                "seed":          cell.seed,
                "windows":       sm.n_windows,
                "robust score":  f"{(sm.active_robust_score if use_active_m else sm.robust_score):.4f}",
                "median excess": f"{(sm.median_active_excess_return if use_active_m else sm.median_excess_return):+.1%}",
                "median Sharpe": f"{(sm.median_active_sharpe if use_active_m else sm.median_sharpe):.3f}",
                "% beating":     f"{(sm.pct_active_beating_benchmark if use_active_m else sm.pct_beating_benchmark):.0%}",
                "median DD":     f"{sm.median_drawdown:.1%}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Interpreting the robust score"):
        if is_active and display_summary.active_robust_score is not None:
            sm = display_summary
            st.markdown(f"""
| Metric | Value |
|---|---|
| Median active excess return | `{sm.median_active_excess_return:+.2%}` |
| Median active Sharpe | `{sm.median_active_sharpe:.3f}` |
| % Active beating benchmark | `{sm.pct_active_beating_benchmark:.0%}` |
| Worst-decile active drawdown | `{sm.worst_decile_active_drawdown:.1%}` |
| Median turnover | `{sm.median_turnover:.2f}×` |
| **Active robust score** | **`{sm.active_robust_score:.4f}`** |
            """)
        else:
            sm = display_summary
            st.markdown(f"""
| Metric | Value |
|---|---|
| Median excess return | `{sm.median_excess_return:+.2%}` |
| Median Sharpe | `{sm.median_sharpe:.3f}` |
| % Beating benchmark | `{sm.pct_beating_benchmark:.0%}` |
| Worst-decile drawdown | `{sm.worst_decile_drawdown:.1%}` |
| Median turnover | `{sm.median_turnover:.2f}×` |
| Std excess return | `{sm.std_excess_return:.3f}` |
| **Robust score** | **`{sm.robust_score:.4f}`** |
            """)
