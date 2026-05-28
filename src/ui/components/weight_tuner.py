"""
ui/components/weight_tuner.py — Score weight tuner UI.

Three modes:
  Manual        — set value/quality/income/momentum weights with sliders, run preview.
  Random Search — sample N weight combos, rank by robust_score (random windows) or
                  Sharpe (full history).
  scipy         — differential evolution over all 15 params, full history or random
                  windows objective.
"""
from __future__ import annotations

import io

import numpy as np
import streamlit as st

from ui.utils import BACKTEST_MODES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEIGHT_NAMES = ["value", "quality", "income", "momentum"]
_WEIGHT_ICONS = {"value": "📊", "quality": "⭐", "income": "💰", "momentum": "🚀"}


def _normalized_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total < 1e-9:
        return {k: 0.25 for k in raw}
    return {k: v / total for k, v in raw.items()}


def _params_from_weights(weights: dict[str, float]) -> np.ndarray:
    from backtesting.simulator import get_default_params
    params = get_default_params()
    params[0] = weights["value"]
    params[1] = weights["quality"]
    params[2] = weights["income"]
    params[3] = weights["momentum"]
    return params


def _weight_sliders(prefix: str, defaults: dict[str, float]) -> dict[str, float]:
    """Render four weight sliders. Returns raw (unnormalized) values."""
    cols = st.columns(4)
    raw = {}
    for i, name in enumerate(_WEIGHT_NAMES):
        icon = _WEIGHT_ICONS[name]
        with cols[i]:
            raw[name] = st.slider(
                f"{icon} {name.capitalize()}",
                min_value=0.0, max_value=1.0,
                value=float(defaults.get(name, 0.25)),
                step=0.01,
                key=f"{prefix}_weight_{name}",
            )
    return raw


def _show_weight_summary(raw: dict[str, float], norm: dict[str, float]):
    cols = st.columns(4)
    for i, name in enumerate(_WEIGHT_NAMES):
        icon = _WEIGHT_ICONS[name]
        cols[i].metric(
            f"{icon} {name.capitalize()}",
            f"{norm[name]:.1%}",
            delta=f"raw {raw[name]:.2f}",
            delta_color="off",
        )
    total = sum(raw.values())
    if abs(total - 1.0) > 0.01:
        st.caption(f"Raw weights sum to {total:.2f} → normalized to 1.00 before running.")
    else:
        st.caption(f"Weights sum to {total:.2f}.")


def _equity_chart_simple(equity_curve: np.ndarray, bench_equity: np.ndarray):
    """Simple indexed growth chart for weight preview."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    if len(equity_curve) == 0:
        return

    days = np.arange(len(equity_curve))
    eq_idx = equity_curve / max(equity_curve[0], 1e-9) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=eq_idx, name="Strategy",
                             line=dict(color="#4c8ef5", width=2)))
    if len(bench_equity) == len(equity_curve):
        b_idx = bench_equity / max(bench_equity[0], 1e-9) * 100.0
        fig.add_trace(go.Scatter(x=days, y=b_idx, name="Benchmark",
                                 line=dict(color="#aaa", width=1.5, dash="dot")))
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h"), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def _robust_score_bar_chart(df):
    """Horizontal bar of top-N candidates by robust_score."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    top = df.head(20)
    labels = [
        f"v={row['value']:.2f} q={row['quality']:.2f} "
        f"i={row['income']:.2f} m={row['momentum']:.2f}"
        for _, row in top.iterrows()
    ]
    colors = ["#4c8ef5" if i == 0 else "#91b4f5" for i in range(len(top))]

    fig = go.Figure(go.Bar(
        x=top["robust_score"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=top["robust_score"].map("{:.4f}".format),
        textposition="outside",
    ))
    fig.update_layout(
        height=max(350, len(top) * 22),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Robust score",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# scipy results visualization helpers
# ---------------------------------------------------------------------------

_PARAM_GROUP_IDX: dict[str, list[int]] = {
    "Score Weights":    [0, 1, 2, 3],
    "Allocation":       [4],
    "Risk":             [5],
    "Sell Rules":       [6, 7, 8],
    "Value Scoring":    [9],
    "Momentum Weights": [10, 11, 12, 13, 14],
}

_PARAM_DISPLAY: dict[str, str] = {
    "sw_value":         "Value",
    "sw_quality":       "Quality",
    "sw_income":        "Income",
    "sw_momentum":      "Momentum",
    "index_pct":        "Index %",
    "metric_threshold": "Metric threshold",
    "take_profit_pct":  "Take-profit %",
    "sell_weak_below":  "Sell-weak below",
    "trailing_stop":    "Trailing stop",
    "value_pe_weight":  "P/E weight",
    "mom_rs3m":         "RS 3m",
    "mom_rs6m":         "RS 6m",
    "mom_radj":         "Risk-adj 3m",
    "mom_trend":        "Trend",
    "mom_r1m":          "Return 1m",
}


def _build_param_df(
    cur: np.ndarray,
    named_candidates: list[tuple[str, np.ndarray]],
    active_set: set[str],
):
    """Build a comparison DataFrame: Group | Param | Active | Current | [label] | [Δ label] per candidate."""
    import pandas as pd

    from tuning.constants import PARAM_NAMES
    idx_to_group = {i: g for g, idxs in _PARAM_GROUP_IDX.items() for i in idxs}
    rows = []
    for i, pname in enumerate(PARAM_NAMES):
        row: dict = {
            "Group":   idx_to_group.get(i, "Other"),
            "Param":   _PARAM_DISPLAY.get(pname, pname),
            "Status":  "✅ tuned" if pname in active_set else "🔒 frozen",
            "Current": round(float(cur[i]), 4),
        }
        for label, arr in named_candidates:
            row[label] = round(float(arr[i]), 4)
            row[f"Δ {label}"] = round(float(arr[i]) - float(cur[i]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _weight_change_bar(
    cur: np.ndarray,
    named_candidates: list[tuple[str, np.ndarray, str]],
) -> None:
    """Grouped bar chart of score weights (indices 0-3): current vs each candidate."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    from tuning.constants import PARAM_NAMES
    weight_names = [_PARAM_DISPLAY.get(PARAM_NAMES[i], PARAM_NAMES[i]) for i in range(4)]
    default_colors = ["#aaaaaa", "#4c8ef5", "#f5a04c", "#5dbb6b"]
    cur_vals = [float(cur[i]) for i in range(4)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current", x=weight_names, y=cur_vals,
        marker_color=default_colors[0],
        text=[f"{v:.3f}" for v in cur_vals], textposition="outside",
    ))
    for k, (label, arr, color) in enumerate(named_candidates):
        vals = [float(arr[i]) for i in range(4)]
        fig.add_trace(go.Bar(
            name=label, x=weight_names, y=vals,
            marker_color=color or default_colors[min(k + 1, len(default_colors) - 1)],
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Weight", range=[0, 1.0]),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _momentum_weight_bar(
    cur: np.ndarray,
    named_candidates: list[tuple[str, np.ndarray, str]],
) -> None:
    """Grouped bar chart of momentum sub-weights (indices 10-14)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    from tuning.constants import PARAM_NAMES
    mom_idxs = list(range(10, 15))
    mom_names = [_PARAM_DISPLAY.get(PARAM_NAMES[i], PARAM_NAMES[i]) for i in mom_idxs]
    default_colors = ["#aaaaaa", "#4c8ef5", "#f5a04c", "#5dbb6b"]
    cur_vals = [float(cur[i]) for i in mom_idxs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current", x=mom_names, y=cur_vals,
        marker_color=default_colors[0],
        text=[f"{v:.3f}" for v in cur_vals], textposition="outside",
    ))
    for k, (label, arr, color) in enumerate(named_candidates):
        vals = [float(arr[i]) for i in mom_idxs]
        fig.add_trace(go.Bar(
            name=label, x=mom_names, y=vals,
            marker_color=color or default_colors[min(k + 1, len(default_colors) - 1)],
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Sub-weight"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _multi_equity_chart(named_sims: list[tuple[str, object, str]]) -> None:
    """Plot indexed growth curves for multiple SimResult objects."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    fig = go.Figure()
    bench_added = False
    for name, sim, color in named_sims:
        eq = getattr(sim, "equity_curve", np.array([]))
        if len(eq) == 0:
            continue
        eq_idx = eq / max(float(eq[0]), 1e-9) * 100.0
        days = np.arange(len(eq_idx))
        fig.add_trace(go.Scatter(
            x=days, y=eq_idx, name=name,
            line=dict(color=color, width=2),
        ))
        if not bench_added:
            bench = getattr(sim, "benchmark_equity", np.array([]))
            if len(bench) == len(eq):
                b_idx = bench / max(float(bench[0]), 1e-9) * 100.0
                fig.add_trace(go.Scatter(
                    x=days, y=b_idx, name="Benchmark (SPY)",
                    line=dict(color="#aaa", width=1.5, dash="dot"),
                ))
                bench_added = True
    if not fig.data:
        return
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.3)
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Growth (indexed to 100)"),
        legend=dict(orientation="h"), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------

def _render_manual_mode():
    from util import SCORE_WEIGHTS

    st.subheader("Manual Weight Tuner")
    st.caption(
        "Adjust the four score weights, then run a backtest or randomized window "
        "evaluation to see how the modified weights perform."
    )

    defaults = {k: SCORE_WEIGHTS.get(k, 0.25) for k in _WEIGHT_NAMES}

    st.markdown("**Set weights** (will be normalized to sum to 1.0):")
    raw = _weight_sliders("manual", defaults)
    norm = _normalized_weights(raw)

    _show_weight_summary(raw, norm)
    st.divider()

    run_mode = st.radio(
        "Run mode",
        ["Single backtest (one window)", "Randomized windows (N random windows)"],
        horizontal=True, key="manual_run_mode",
    )

    c1, c2 = st.columns(2)
    with c1:
        n_days = st.number_input("History (days)", min_value=30, max_value=1000,
                                  value=90, step=30, key="manual_n_days")
        mode   = st.selectbox("Backtest mode", BACKTEST_MODES, key="manual_bt_mode")
    with c2:
        if run_mode.startswith("Randomized"):
            n_windows   = st.number_input("Windows", min_value=5, max_value=50, value=15, key="manual_n_win")
            window_days = st.number_input("Window length", min_value=20, max_value=180, value=60, key="manual_win_days")
        else:
            n_windows = window_days = None

    if st.button("▶ Run", type="primary", key="manual_run_btn"):
        params = _params_from_weights(norm)

        if run_mode.startswith("Single"):
            with st.spinner("Running backtest…"):
                try:
                    from ui.services.backtest_service import run_single_backtest
                    result = run_single_backtest(n_days=n_days, mode=mode, params=params)
                    st.session_state["manual_report"] = result.report
                    st.session_state["manual_run_type"] = "single"
                    st.success("✅ Backtest complete.")
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    st.exception(exc)
        else:
            bar = st.progress(0, text="Running windows…")

            def _cb(cur, tot):
                bar.progress(int(cur / max(tot, 1) * 100), text=f"Window {cur+1}/{tot}…")

            with st.spinner("Running random windows…"):
                try:
                    from ui.services.backtest_service import run_random_windows
                    summary = run_random_windows(
                        n_days=n_days,
                        mode=mode,
                        params=params,
                        n_windows=int(n_windows),
                        window_days=int(window_days),
                        progress_callback=_cb,
                    )
                    bar.empty()
                    st.session_state["manual_summary"] = summary
                    st.session_state["manual_run_type"] = "random"
                    st.success(f"✅ Evaluated {summary.n_windows} windows.")
                except Exception as exc:
                    bar.empty()
                    st.error(f"Random windows failed: {exc}")
                    st.exception(exc)

    # Show results
    run_type = st.session_state.get("manual_run_type")
    if run_type == "single":
        report = st.session_state.get("manual_report")
        if report:
            train = report.train_result
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy return",  f"{train.total_return:+.1%}")
            c2.metric("Benchmark return", f"{report.benchmark_return:+.1%}")
            c3.metric("Excess return",    f"{report.excess_return:+.1%}")
            c4.metric("Sharpe",           f"{train.sharpe:+.3f}")
            _equity_chart_simple(train.equity_curve, train.benchmark_equity)

    elif run_type == "random":
        summary = st.session_state.get("manual_summary")
        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median excess",     f"{summary.median_excess_return:+.1%}")
            c2.metric("Median Sharpe",     f"{summary.median_sharpe:.3f}")
            c3.metric("% Beating",         f"{summary.pct_beating_benchmark:.0%}")
            c4.metric("Robust score",      f"{summary.robust_score:.4f}")
            c1.metric("Median drawdown",   f"{summary.median_drawdown:.1%}")
            c2.metric("Worst-decile DD",   f"{summary.worst_decile_drawdown:.1%}")
            c3.metric("Windows",           summary.n_windows)


# ---------------------------------------------------------------------------
# Random search mode
# ---------------------------------------------------------------------------

def _run_full_history_weight_search(precomp, n_samples: int, seed: int, respect_bounds: bool, progress_callback, scope: str = "overall_strategy") -> list[dict]:
    """Evaluate N weight combos on full history, rank by Sharpe."""
    from backtesting.simulator import get_default_params, run_simulation
    from tuning.random_tune import _weight_bounds_from_config, sample_weight_simplex

    base = get_default_params()
    bounds = _weight_bounds_from_config() if respect_bounds else None
    samples = sample_weight_simplex(n_samples, seed=seed, bounds=bounds)

    results = []
    for i, weights in enumerate(samples):
        if progress_callback:
            progress_callback(i, n_samples)
        params = base.copy()
        params[0:4] = weights
        try:
            sim = run_simulation(precomp, params, scope=scope)
            sharpe_val = (sim.active_sharpe if scope == "active_sleeve_compounding" and sim.active_sharpe is not None else sim.sharpe)
            results.append({
                "value": float(weights[0]), "quality": float(weights[1]),
                "income": float(weights[2]), "momentum": float(weights[3]),
                "sharpe": sharpe_val, "total_return": sim.total_return,
                "max_drawdown": sim.max_drawdown, "calmar": sim.calmar,
            })
        except Exception:
            pass
    results.sort(key=lambda r: -r["sharpe"])
    return results


def _render_random_search_mode(scope: str = "overall_strategy"):
    st.subheader("Random Search")
    st.caption("Sample N weight combinations and rank by performance.")

    optimize_over = st.radio(
        "Optimize over",
        ["🎲 Random Windows (robust — evaluates each combo across N time windows)",
         "📈 Full History (faster — evaluates each combo on one window)"],
        horizontal=False,
        key="at_optimize_over",
    )
    use_random_windows = optimize_over.startswith("🎲")

    with st.expander("ℹ️ How random search works"):
        st.markdown(
            """
Random Search **does not use gradient descent**.  Instead:

1. Samples N random weight combinations from the 4-weight simplex (Dirichlet).
2. **Random Windows:** evaluates each combo on M random historical windows, ranks
   by **robust_score** (rewards consistent outperformance across many market conditions).
3. **Full History:** evaluates each combo on the full loaded window, ranks by Sharpe.
   Faster but more likely to find a result that only works in that specific period.

Config bounds from `tuning.parameter_bounds` are respected when enabled.
            """
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_samples = st.number_input("Weight samples", min_value=10, max_value=200,
                                     value=40, step=10, key="at_n_samples")
    with c2:
        n_windows = st.number_input("Windows / sample", min_value=5, max_value=50,
                                     value=15, step=5, key="at_n_windows",
                                     disabled=not use_random_windows)
    with c3:
        window_days = st.number_input("Window length (days)", min_value=20, max_value=180,
                                       value=60, step=10, key="at_window_days",
                                       disabled=not use_random_windows)
    with c4:
        n_days_load = st.number_input("History to load (days)", min_value=120, max_value=1500,
                                       value=365, step=60, key="at_n_days")

    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="at_mode")
    with c2:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, key="at_seed")
    with c3:
        respect_bounds = st.checkbox("Respect config weight bounds", value=True, key="at_bounds")

    if use_random_windows:
        total_runs = n_samples * n_windows
        st.caption(f"~{n_samples} × {n_windows} = {total_runs} simulation runs (~{total_runs * 0.05:.0f}s)")
        if window_days * 2 > n_days_load:
            st.warning("Window length > half loaded history. Load more days for better diversity.")
    else:
        st.caption(f"~{n_samples} simulation runs on full {n_days_load}d history")

    if st.button("▶ Run Random Search", type="primary", key="at_run"):
        bar = st.progress(0, text="Loading data…")
        try:
            from backtesting.data_loader import load_and_precompute
            precomp = load_and_precompute(int(n_days_load), mode=mode)
        except Exception as exc:
            bar.empty()
            st.error(f"Failed to load data: {exc}")
            st.exception(exc)
            return

        def _cb(cur, total):
            bar.progress(int(cur / max(total, 1) * 100), text=f"Sample {cur + 1}/{total}…")

        if use_random_windows:
            with st.spinner("Running random window search…"):
                try:
                    from ui.services.tuning_service import run_weight_tune
                    tune_result = run_weight_tune(
                        precomp,
                        n_samples=int(n_samples),
                        n_windows=int(n_windows),
                        window_days=int(window_days),
                        seed=int(seed),
                        respect_config_bounds=respect_bounds,
                        progress_callback=_cb,
                        scope=scope,
                    )
                    bar.empty()
                    st.session_state["at_result"] = tune_result
                    st.session_state["at_result_mode"] = "random_windows"
                    st.success(f"✅ Evaluated {tune_result.n_samples} weight combos.")
                except Exception as exc:
                    bar.empty()
                    st.error(f"Search failed: {exc}")
                    st.exception(exc)
                    return
        else:
            with st.spinner("Running full-history weight search…"):
                try:
                    rows = _run_full_history_weight_search(
                        precomp, int(n_samples), int(seed), respect_bounds, _cb, scope=scope,
                    )
                    bar.empty()
                    st.session_state["at_fh_rows"] = rows
                    st.session_state["at_result_mode"] = "full_history"
                    st.success(f"✅ Evaluated {len(rows)} weight combos.")
                except Exception as exc:
                    bar.empty()
                    st.error(f"Search failed: {exc}")
                    st.exception(exc)
                    return

    result_mode = st.session_state.get("at_result_mode")
    if result_mode is None:
        st.info("No results yet. Configure settings above and click Run.")
        return

    st.divider()

    if result_mode == "full_history":
        rows = st.session_state.get("at_fh_rows", [])
        if not rows:
            return
        import pandas as pd
        st.subheader("Top combos by Sharpe (Full History)")
        df = pd.DataFrame(rows[:20])
        fmt_cols = {
            "value": "{:.3f}", "quality": "{:.3f}", "income": "{:.3f}", "momentum": "{:.3f}",
            "sharpe": "{:+.3f}", "total_return": "{:+.1%}", "max_drawdown": "{:.1%}", "calmar": "{:+.3f}",
        }
        disp = df.copy()
        for col, f in fmt_cols.items():
            if col in disp.columns:
                disp[col] = df[col].map(f.format)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        best = rows[0] if rows else None
        if best:
            yaml_snippet = (
                "score_weights:\n"
                f"  value:    {best['value']:.4f}\n"
                f"  quality:  {best['quality']:.4f}\n"
                f"  income:   {best['income']:.4f}\n"
                f"  momentum: {best['momentum']:.4f}\n"
            )
            with st.expander("Best weights YAML"):
                st.code(yaml_snippet, language="yaml")
                st.download_button("⬇️ Download", data=yaml_snippet,
                                   file_name="best_weights_fh.yaml", mime="text/yaml")
        return

    # random_windows result
    result = st.session_state.get("at_result")
    if result is None:
        return

    for w in result.warnings:
        st.warning(w)

    st.subheader("Best vs Current Config")
    cmp_df = result.best_vs_current_df()
    if cmp_df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        with col2:
            if result.best_candidate:
                bw = result.best_candidate.weights_dict()
                cw = {n: float(result.current_weights[i]) for i, n in enumerate(_WEIGHT_NAMES)} \
                     if result.current_weights is not None else {}
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Best config",
                                     x=[n.capitalize() for n in _WEIGHT_NAMES],
                                     y=[bw[n] for n in _WEIGHT_NAMES],
                                     marker_color="#4c8ef5"))
                if cw:
                    fig.add_trace(go.Bar(name="Current config",
                                         x=[n.capitalize() for n in _WEIGHT_NAMES],
                                         y=[cw.get(n, 0) for n in _WEIGHT_NAMES],
                                         marker_color="#aaaaaa"))
                fig.update_layout(barmode="group", height=260, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title="Weight", yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top candidates by robust score")
    df = result.to_dataframe()
    if not df.empty:
        _robust_score_bar_chart(df)
        with st.expander("Full ranked table"):
            fmt = {
                "value": "{:.3f}", "quality": "{:.3f}", "income": "{:.3f}", "momentum": "{:.3f}",
                "robust_score": "{:.4f}", "median_excess": "{:+.2%}", "median_sharpe": "{:.3f}",
                "median_drawdown": "{:.1%}", "pct_beating": "{:.0%}", "worst_decile_dd": "{:.1%}",
            }
            disp = df.copy()
            for col, f in fmt.items():
                if col in disp.columns:
                    disp[col] = df[col].map(f.format)
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.subheader("Export")
    yaml_snippet = result.best_weights_yaml()
    if yaml_snippet:
        c1, c2 = st.columns(2)
        with c1:
            st.code(yaml_snippet, language="yaml")
        with c2:
            st.download_button("⬇️ Download best weights YAML", data=yaml_snippet,
                               file_name="best_score_weights.yaml", mime="text/yaml")
            if not df.empty:
                csv_buf = io.StringIO()
                result.to_dataframe().to_csv(csv_buf, index=False)
                st.download_button("⬇️ Download full results CSV", data=csv_buf.getvalue(),
                                   file_name="weight_tune_results.csv", mime="text/csv")
        st.warning("⚠️ Results are not applied automatically. Review and verify before updating any config.")
        with st.expander("Apply to research-safe config (requires confirmation)"):
            st.markdown("Overwrites `score_weights` in `config_research_safe.yaml`. Production `config.yaml` is NOT changed.")
            confirm = st.text_input("Type CONFIRM to proceed:", key="at_confirm_apply")
            if st.button("Apply to config_research_safe.yaml", key="at_apply_btn"):
                if confirm.strip().upper() != "CONFIRM":
                    st.error("Type CONFIRM in the field above to proceed.")
                else:
                    try:
                        import os

                        import yaml

                        from core.paths import CFG_DIRECTORY
                        safe_path = os.path.join(CFG_DIRECTORY, "config_research_safe.yaml")
                        with open(safe_path) as f:
                            cfg = yaml.safe_load(f)
                        bw = result.best_candidate.weights_dict()
                        cfg["score_weights"] = {k: round(bw[k], 4) for k in _WEIGHT_NAMES}
                        with open(safe_path, "w") as f:
                            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                        st.success(f"✅ Wrote best weights to {os.path.basename(safe_path)}")
                    except Exception as exc:
                        st.error(f"Failed to write config: {exc}")


# ---------------------------------------------------------------------------
# scipy optimizer mode
# ---------------------------------------------------------------------------

def _render_scipy_mode(scope: str = "overall_strategy"):
    st.subheader("scipy Optimizer")
    st.caption("Differential evolution over all 15 parameters. Slower but explores the full param space.")

    optimize_over = st.radio(
        "Optimize over",
        ["📈 Full History (Sharpe + Calmar, with validation gates)",
         "🎲 Random Windows (robust_score objective — slower but more generalizable)"],
        horizontal=False,
        key="sp_optimize_over",
    )
    use_random_windows = optimize_over.startswith("🎲")

    from ui.utils import LOOKAHEAD_LABELS
    c1, c2 = st.columns(2)
    with c1:
        n_days = st.number_input("Look-back days", min_value=30, max_value=1000, value=90, step=30, key="sp_n_days")
    with c2:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="sp_mode")
        st.caption(LOOKAHEAD_LABELS[mode])

    if use_random_windows:
        c1, c2 = st.columns(2)
        with c1:
            sp_n_windows = st.number_input("Windows / evaluation", 3, 20, 8, 1, key="sp_n_windows",
                                           help="More windows = more reliable but much slower")
        with c2:
            sp_window_days = st.number_input("Window length (days)", 20, 120, 45, 5, key="sp_window_days")
        st.caption(
            f"Each function evaluation runs {sp_n_windows} backtests. "
            "Differential evolution typically needs 200–600 evaluations. "
            "Expect 5–15 minutes."
        )
    else:
        sp_n_windows = sp_window_days = None
        from ui.utils import ui_config
        ui_cfg = ui_config()
        allow_write  = ui_cfg.get("allow_config_writes", False)
        allow_force  = ui_cfg.get("allow_force_apply", False)
        apply_cfg    = st.checkbox("Apply if validation passes", disabled=not allow_write, key="sp_apply")
        force_apply  = st.checkbox("Force apply ⚠️", disabled=not allow_force, key="sp_force")
        llm_review   = st.checkbox("LLM second-opinion review", key="sp_llm")

    if st.button("▶ Run scipy", type="primary", key="sp_run"):
        if use_random_windows:
            bar = st.progress(0, text="Loading data…")
            try:
                from backtesting.data_loader import load_and_precompute
                precomp = load_and_precompute(int(n_days), mode=mode)
            except Exception as exc:
                bar.empty()
                st.error(f"Failed to load data: {exc}")
                return

            with st.spinner("Running scipy (random windows objective)…"):
                try:
                    from scipy.optimize import differential_evolution

                    from backtesting.random_walk import random_window_backtest
                    from tuning.constants import _effective_bounds

                    bounds = _effective_bounds(scope=scope)
                    call_count = [0]

                    def _obj(params_arr):
                        call_count[0] += 1
                        if call_count[0] % 10 == 0:
                            bar.progress(min(call_count[0], 99), text=f"Evaluation {call_count[0]}…")
                        try:
                            summary = random_window_backtest(
                                precomp,
                                params=np.array(params_arr),
                                n_windows=int(sp_n_windows),
                                window_days=int(sp_window_days),
                                seed=42,
                                scope=scope,
                            )
                            score = (
                                summary.active_robust_score
                                if scope == "active_sleeve_compounding" and summary.active_robust_score is not None
                                else summary.robust_score
                            )
                            return -score
                        except Exception:
                            return 0.0

                    opt = differential_evolution(_obj, bounds, maxiter=12, popsize=8,
                                                 seed=42, workers=1, tol=0.005)
                    bar.empty()
                    from tuning.constants import PARAM_NAMES, _get_active_indices
                    active_idxs = _get_active_indices(scope=scope)
                    st.session_state["sp_result"] = {
                        "params": opt.x, "score": -opt.fun,
                        "n_evals": call_count[0], "mode": "random_windows",
                        "active_params": [PARAM_NAMES[i] for i in active_idxs],
                        "converged": opt.success,
                        "message": opt.message,
                    }
                    st.success(f"✅ Optimized in {call_count[0]} evaluations. Robust score: {-opt.fun:.4f}")
                except Exception as exc:
                    bar.empty()
                    st.error(f"Optimization failed: {exc}")
                    st.exception(exc)
        else:
            with st.spinner(f"Running {n_days}-day auto-tune…"):
                try:
                    from tuning.tuner import ParameterTuner
                    result = ParameterTuner().auto_tune(
                        n_days=n_days, mode=mode,
                        apply=apply_cfg, force_apply=force_apply,
                        llm_review=llm_review,
                        scope=scope,
                    )
                    st.session_state["sp_tuner_result"] = result
                    st.session_state["sp_result"] = {"mode": "full_history"}
                    st.success("✅ Auto-tune complete.")
                except Exception as exc:
                    st.error(f"Tune failed: {exc}")
                    st.exception(exc)

    sp = st.session_state.get("sp_result")
    if sp is None:
        st.info("No results yet. Configure settings above and click Run.")
        return

    st.divider()

    if sp.get("mode") == "random_windows":
        import pandas as pd

        from backtesting.simulator import get_default_params
        params = sp["params"]
        cur = get_default_params()
        active_set = set(sp.get("active_params", []))

        # --- Summary metrics ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Robust score", f"{sp['score']:.4f}")
        c2.metric("Function evaluations", sp["n_evals"])
        c3.metric("Converged", "✅ Yes" if sp.get("converged") else "⚠️ No")
        c4.metric("Active params", len(active_set))
        if sp.get("message"):
            st.caption(f"Optimizer: {sp['message']}")

        # --- Score weight change chart ---
        st.subheader("Score Weight Changes")
        _weight_change_bar(cur, [("Optimized", params, "#4c8ef5")])

        # --- Momentum sub-weight chart ---
        st.subheader("Momentum Sub-weight Changes")
        _momentum_weight_bar(cur, [("Optimized", params, "#4c8ef5")])

        # --- Full parameter table ---
        st.subheader("All Parameter Changes")
        df = _build_param_df(cur, [("Optimized", params)], active_set)
        disp = df.copy()
        disp["Current"]        = df["Current"].map("{:.4f}".format)
        disp["Optimized"]      = df["Optimized"].map("{:.4f}".format)
        disp["Δ Optimized"]    = df["Δ Optimized"].map("{:+.4f}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    else:
        result = st.session_state.get("sp_tuner_result")
        if result is None:
            return

        # --- Validation badge ---
        if result.validation_passed:
            st.success("✅ Validation PASSED" + (" — config written" if result.config_written else " (not written)"))
        else:
            st.error(f"❌ Validation FAILED: {'; '.join(result.validation_reasons)}")

        avg = result.avg_result
        shr = result.sharpe_result
        cal = result.calmar_result

        # --- Performance summary ---
        st.subheader("Performance Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Return (avg)",      f"{avg.total_return:+.1%}")
        c2.metric("Sharpe (avg)",      f"{avg.sharpe:+.3f}")
        c3.metric("Calmar (avg)",      f"{avg.calmar:+.3f}")
        c4.metric("Max drawdown (avg)", f"{avg.max_drawdown:.1%}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe-opt return", f"{shr.total_return:+.1%}",
                  delta=f"{shr.total_return - avg.total_return:+.1%}", delta_color="off")
        c2.metric("Calmar-opt return", f"{cal.total_return:+.1%}",
                  delta=f"{cal.total_return - avg.total_return:+.1%}", delta_color="off")
        c3.metric("Trades (avg)",  avg.trades_made)
        turnover_str = f"{avg.turnover_estimate:.1%}" if avg.turnover_estimate else "N/A"
        c4.metric("Turnover (avg)", turnover_str)

        # --- Equity curves (when available) ---
        has_curves = any(len(getattr(s, "equity_curve", [])) > 0 for s in [shr, cal, avg])
        if has_curves:
            st.subheader("Equity Curves (train window)")
            _multi_equity_chart([
                ("Sharpe-opt", shr, "#4c8ef5"),
                ("Calmar-opt", cal, "#f5a04c"),
                ("Averaged",   avg, "#5dbb6b"),
            ])

        # --- Weight change charts ---
        from backtesting.simulator import get_default_params
        cur = get_default_params()
        active_set = set(result.active_params)
        candidates = [
            ("Sharpe-opt", result.sharpe_params, "#4c8ef5"),
            ("Calmar-opt", result.calmar_params, "#f5a04c"),
            ("Averaged",   result.avg_params,    "#5dbb6b"),
        ]

        st.subheader("Score Weight Changes")
        _weight_change_bar(cur, candidates)

        st.subheader("Momentum Sub-weight Changes")
        _momentum_weight_bar(cur, candidates)

        # --- Full parameter table ---
        st.subheader("All Parameter Changes")
        import pandas as pd
        df = _build_param_df(cur, [(lbl, arr) for lbl, arr, _ in candidates], active_set)
        disp = df.copy()
        num_cols = ["Current", "Sharpe-opt", "Calmar-opt", "Averaged"]
        delta_cols = ["Δ Sharpe-opt", "Δ Calmar-opt", "Δ Averaged"]
        for col in num_cols:
            if col in disp.columns:
                disp[col] = df[col].map("{:.4f}".format)
        for col in delta_cols:
            if col in disp.columns:
                disp[col] = df[col].map("{:+.4f}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # --- Stability analysis ---
        with st.expander("Parameter stability (Sharpe vs Calmar spread)"):
            spread = result.param_spread
            sp_df = pd.DataFrame({"parameter": list(spread.keys()), "spread": list(spread.values())})
            sp_df["flag"] = sp_df["spread"].apply(lambda x: "⚠️ unstable" if x > 0.05 else "✅")
            st.dataframe(sp_df.sort_values("spread", ascending=False), use_container_width=True, hide_index=True)


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
        key="wt_scope",
    )
    if scope == "active_sleeve_compounding":
        st.info("index_pct is frozen in Active Sleeve scope — optimizer tunes stock-picking params only.")

    mode = st.radio(
        "Mode",
        ["🎛️ Manual — set weights and preview",
         "🎲 Random Search — sample weight combos",
         "⚙️ scipy — differential evolution (all 15 params)"],
        horizontal=False,
        key="wt_mode",
    )

    st.divider()

    if mode.startswith("🎛️"):
        _render_manual_mode()
    elif mode.startswith("🎲"):
        _render_random_search_mode(scope=scope)
    else:
        _render_scipy_mode(scope=scope)

    st.divider()
    from ui.components.config_variants import render_config_variants
    render_config_variants()
