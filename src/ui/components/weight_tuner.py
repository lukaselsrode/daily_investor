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


def _robust_score_bar_chart(df, active_param_names: list[str] | None = None):
    """Horizontal bar of top-N candidates by robust_score."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    # Determine which columns to show in labels dynamically
    _param_cols = active_param_names or [n for n in _WEIGHT_NAMES if n in df.columns]
    # Fall back to whatever non-metric columns exist if nothing matched
    if not _param_cols:
        _skip = {"rank", "robust_score", "median_excess", "median_sharpe",
                 "median_drawdown", "pct_beating", "worst_decile_dd", "std_excess", "n_windows"}
        _param_cols = [c for c in df.columns if c not in _skip]

    top = df.head(20)
    labels = []
    for _, row in top.iterrows():
        parts = []
        for col in _param_cols[:6]:  # cap at 6 to avoid label overflow
            if col in row:
                short = col[:4]
                parts.append(f"{short}={float(row[col]):.2f}")
        labels.append(" ".join(parts))
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

def _run_full_history_weight_search(precomp, n_samples: int, seed: int, respect_bounds: bool, progress_callback, scope: str = "overall_strategy", preset: str | None = None) -> list[dict]:
    """Evaluate N parameter combos on full history, rank by Sharpe. Respects preset active indices."""
    from backtesting.simulator import get_default_params, run_simulation
    from tuning.constants import PARAM_NAMES, _effective_bounds, _get_active_indices
    from tuning.random_tune import _weight_bounds_from_config, sample_weight_simplex

    base        = get_default_params()
    active_idxs = _get_active_indices(scope=scope, preset=preset)
    eff_bounds  = _effective_bounds(scope=scope, preset=preset)

    _SCORE_WEIGHT_IDXS = {0, 1, 2, 3}
    use_simplex = (set(active_idxs) == _SCORE_WEIGHT_IDXS)

    if use_simplex:
        simplex_bounds = _weight_bounds_from_config() if respect_bounds else None
        raw_samples = sample_weight_simplex(n_samples, seed=seed, bounds=simplex_bounds)
    else:
        import numpy as _np
        rng = _np.random.default_rng(seed)
        raw_samples = _np.zeros((n_samples, len(active_idxs)))
        for j, aidx in enumerate(active_idxs):
            lo, hi = eff_bounds[aidx]
            raw_samples[:, j] = rng.uniform(lo, hi, n_samples)

    active_names = [PARAM_NAMES[i] for i in active_idxs]

    results = []
    for i, active_vals in enumerate(raw_samples):
        if progress_callback:
            progress_callback(i, n_samples)
        params = base.copy()
        for j, aidx in enumerate(active_idxs):
            params[aidx] = float(active_vals[j])
        try:
            sim = run_simulation(precomp, params, scope=scope)
            sharpe_val = (sim.active_sharpe if scope == "active_sleeve_compounding" and sim.active_sharpe is not None else sim.sharpe)
            row: dict = {n: float(active_vals[j]) for j, n in enumerate(active_names)}
            row.update({
                "sharpe": sharpe_val, "total_return": sim.total_return,
                "max_drawdown": sim.max_drawdown, "calmar": sim.calmar,
            })
            results.append(row)
        except Exception:
            pass
    results.sort(key=lambda r: -r["sharpe"])
    return results


def _classify_signals(result) -> dict[str, str]:
    """
    Classify each tuned parameter delta as strong / weak / do_not_apply.
    Returns {param_name: "strong" | "weak" | "do_not_apply"}.
    """
    if result is None or result.best_candidate is None or result.current_weights is None:
        return {}
    best_score   = result.best_candidate.robust_score
    curr_summary = result.current_summary
    curr_score   = curr_summary.robust_score if curr_summary is not None else 0.0
    delta        = best_score - curr_score
    signals: dict[str, str] = {}
    for name in (result.active_param_names or []):
        bval = result.best_candidate.weights_dict().get(name, 0.0)
        cidx = (result.active_param_names or []).index(name)
        cval = float(result.current_weights[cidx]) if result.current_weights is not None and cidx < len(result.current_weights) else 0.0
        changed = abs(bval - cval) > 1e-4
        if not changed:
            signals[name] = "do_not_apply"
        elif delta > 0.05 and best_score > 0.0:
            signals[name] = "strong"
        elif delta > 0.0:
            signals[name] = "weak"
        else:
            signals[name] = "do_not_apply"
    return signals


_SIGNAL_CHIP: dict[str, str] = {
    "strong":       "🟢 Strong signal",
    "weak":         "🟡 Weak signal",
    "do_not_apply": "🔴 Do not apply",
}


def _render_random_search_mode(scope: str = "overall_strategy", preset: str | None = None):
    from tuning.profiles import (
        HORIZON_PROFILES,
        ROBUSTNESS_PROFILES,
        effort_caption,
        expand_run_matrix,
    )

    st.subheader("Random Search")
    st.caption("Sample N parameter combinations and rank by robustness across multiple horizons and seeds.")

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

1. Samples N random parameter combinations (Dirichlet simplex for score weights; uniform for other presets).
2. **Random Windows:** evaluates each combo across multiple horizons and seeds, ranks
   by **robust_score** (rewards consistent outperformance).
3. **Full History:** evaluates each combo on the full loaded window, ranks by Sharpe.
   Faster but more likely to find a result that only works in that specific period.

Config bounds from `tuning.parameter_bounds` are respected when enabled.
            """
        )

    # ── Profile selectors (primary inputs) ────────────────────────────────
    if use_random_windows:
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            robustness = st.selectbox(
                "Robustness",
                list(ROBUSTNESS_PROFILES),
                index=1,
                format_func=lambda k: {
                    "quick":      "Quick — fast sanity check",
                    "standard":   "Standard — normal research",
                    "deep":       "Deep — stronger robustness",
                    "exhaustive": "Exhaustive — overnight",
                }[k],
                key="at_robustness",
            )
        with pcol2:
            horizon = st.selectbox(
                "Horizon profile",
                list(HORIZON_PROFILES),
                index=3,
                format_func=lambda k: {
                    "short":  "Short-term (30–90d)",
                    "medium": "Medium-term (90–180d)",
                    "long":   "Long-term (180–365d)",
                    "mixed":  "Mixed (30–365d)",
                }[k],
                key="at_horizon",
            )

    # ── Advanced expander (manual overrides) ──────────────────────────────
    adv_n_samples    = None
    adv_n_windows    = None
    adv_window_days  = None
    adv_seed         = None
    n_days_load      = 730
    mode             = BACKTEST_MODES[0]
    respect_bounds   = True

    with st.expander("Advanced options", expanded=False):
        adv_c1, adv_c2, adv_c3, adv_c4 = st.columns(4)
        with adv_c1:
            _ns = st.number_input("Weight samples override (0=profile)", min_value=0, max_value=500,
                                  value=0, step=10, key="at_adv_n_samples")
            adv_n_samples = int(_ns) if _ns > 0 else None
        with adv_c2:
            _nw = st.number_input("Windows/sample override (0=profile)", min_value=0, max_value=100,
                                  value=0, step=5, key="at_adv_n_windows",
                                  disabled=not use_random_windows)
            adv_n_windows = int(_nw) if _nw > 0 else None
        with adv_c3:
            _wd = st.number_input("Window length override (days, 0=profile)", min_value=0, max_value=365,
                                  value=0, step=10, key="at_adv_window_days",
                                  disabled=not use_random_windows)
            adv_window_days = int(_wd) if _wd > 0 else None
        with adv_c4:
            _sd = st.number_input("Seed override (0=profile)", min_value=0, max_value=9999,
                                  value=0, key="at_adv_seed")
            adv_seed = int(_sd) if _sd > 0 else None

        adv_c5, adv_c6, adv_c7 = st.columns(3)
        with adv_c5:
            n_days_load = st.number_input("History to load (days)", min_value=120, max_value=1500,
                                          value=730, step=60, key="at_n_days")
        with adv_c6:
            mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="at_mode")
        with adv_c7:
            respect_bounds = st.checkbox("Respect config weight bounds", value=True, key="at_bounds")

    # ── Resolve effective params from profile + overrides ─────────────────
    if use_random_windows:
        rp = ROBUSTNESS_PROFILES[robustness]
        hp = HORIZON_PROFILES[horizon]
        # dominant horizon = longest in profile
        _dominant_horizon = max(hp)
        effective_window_days = adv_window_days if adv_window_days is not None else _dominant_horizon
        effective_n_windows   = adv_n_windows   if adv_n_windows   is not None else rp["windows_per_horizon"] * len(hp)
        effective_n_samples   = adv_n_samples   if adv_n_samples   is not None else rp["weight_samples"]
        effective_seed        = adv_seed         if adv_seed         is not None else rp["seeds"][0]

        total_evals = effective_n_samples * effective_n_windows
        cap = effort_caption(robustness, horizon) if adv_window_days is None and adv_n_windows is None else (
            f"Advanced override: {effective_n_samples} samples × {effective_n_windows} windows = "
            f"**{total_evals} simulations**"
        )
        st.caption(cap)
        if effective_window_days * 2 > n_days_load:
            st.warning("Dominant horizon > half loaded history. Increase history days in Advanced.")
    else:
        effective_window_days = 90
        effective_n_windows   = 15
        effective_n_samples   = adv_n_samples or 40
        effective_seed        = adv_seed or 42
        st.caption(f"~{effective_n_samples} simulation runs on full {n_days_load}d history")

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
                        n_samples=int(effective_n_samples),
                        n_windows=int(effective_n_windows),
                        window_days=int(effective_window_days),
                        seed=int(effective_seed),
                        respect_config_bounds=respect_bounds,
                        progress_callback=_cb,
                        scope=scope,
                        preset=preset,
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
                        precomp, int(effective_n_samples), int(effective_seed),
                        respect_bounds, _cb, scope=scope, preset=preset,
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
        _metric_cols = {"sharpe", "total_return", "max_drawdown", "calmar"}
        _param_display_cols = [c for c in df.columns if c not in _metric_cols]
        fmt_cols: dict[str, str] = {
            "sharpe": "{:+.3f}", "total_return": "{:+.1%}",
            "max_drawdown": "{:.1%}", "calmar": "{:+.3f}",
        }
        for c in _param_display_cols:
            fmt_cols[c] = "{:.3f}"
        disp = df.copy()
        for col, f in fmt_cols.items():
            if col in disp.columns:
                disp[col] = df[col].map(f.format)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        best = rows[0] if rows else None
        if best:
            import yaml as _yaml
            # Build nested config YAML from active param values
            from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX, PARAM_NAMES
            _path_inv = {v: k for k, v in _CONFIG_PATH_TO_PARAM_IDX.items()}
            _name_to_path = {n: _path_inv.get(n, n) for i, n in enumerate(PARAM_NAMES) for _path_inv_k in [_CONFIG_PATH_TO_PARAM_IDX] if True}
            # Simpler: directly map param name to config path
            _n2p = {
                "sw_value": "score_weights.value", "sw_quality": "score_weights.quality",
                "sw_income": "score_weights.income", "sw_momentum": "score_weights.momentum",
                "index_pct": "index_pct", "metric_threshold": "metric_threshold",
                "take_profit_pct": "sell_rules.take_profit_pct",
                "sell_weak_below": "sell_rules.sell_weak_value_below",
                "trailing_stop": "sell_rules.trailing_stop_pct",
                "value_pe_weight": "scoring.value_pe_weight",
            }
            nested: dict = {}
            for pname in _param_display_cols:
                if pname not in best:
                    continue
                path = _n2p.get(pname, pname)
                parts = path.split(".")
                cur = nested
                for p in parts[:-1]:
                    cur = cur.setdefault(p, {})
                cur[parts[-1]] = round(float(best[pname]), 4)
            yaml_snippet = _yaml.dump(nested, default_flow_style=False, sort_keys=False)
            with st.expander("Best params YAML"):
                st.code(yaml_snippet, language="yaml")
                st.download_button("⬇️ Download", data=yaml_snippet,
                                   file_name="best_params_fh.yaml", mime="text/yaml")
        return

    # random_windows result
    result = st.session_state.get("at_result")
    if result is None:
        return

    for w in result.warnings:
        st.warning(w)

    st.subheader("Best vs Current Config")
    signals = _classify_signals(result)
    if signals:
        sig_cols = st.columns(len(signals))
        for idx, (pname, sig) in enumerate(signals.items()):
            sig_cols[idx].caption(
                f"**{_PARAM_DISPLAY.get(pname, pname)}**  \n{_SIGNAL_CHIP.get(sig, sig)}"
            )

    cmp_df = result.best_vs_current_df()
    if cmp_df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        with col2:
            if result.best_candidate:
                import plotly.graph_objects as go
                bw   = result.best_candidate.weights_dict()
                anames = result.active_param_names or _WEIGHT_NAMES
                labels = [_PARAM_DISPLAY.get(n, n) for n in anames]
                best_vals = [bw.get(n, 0.0) for n in anames]
                curr_vals = (
                    [float(result.current_weights[j]) for j in range(len(anames))]
                    if result.current_weights is not None else []
                )
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Best config", x=labels, y=best_vals,
                                     marker_color="#4c8ef5",
                                     text=[f"{v:.3f}" for v in best_vals],
                                     textposition="outside"))
                if curr_vals:
                    fig.add_trace(go.Bar(name="Current config", x=labels, y=curr_vals,
                                         marker_color="#aaaaaa",
                                         text=[f"{v:.3f}" for v in curr_vals],
                                         textposition="outside"))
                fig.update_layout(barmode="group", height=260, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title="Value", legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top candidates by robust score")
    df = result.to_dataframe()
    if not df.empty:
        _robust_score_bar_chart(df, active_param_names=result.active_param_names)
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

def _render_scipy_mode(scope: str = "overall_strategy", preset: str | None = None):
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

                    bounds = _effective_bounds(scope=scope, preset=preset)
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
                    active_idxs = _get_active_indices(scope=scope, preset=preset)
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
                        preset=preset,
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

    preset: str | None = None
    if scope == "active_sleeve_compounding":
        from tuning.presets import list_presets
        _preset_names = [name for name, _ in list_presets() if not name.endswith("_filters")
                         and not name.endswith("_cooldown") and not name.endswith("_sizing")]
        _preset_opts = ["(none — use config frozen_parameters)"] + _preset_names
        _sel = st.selectbox(
            "Tuning preset",
            _preset_opts,
            key="wt_preset",
            help="Presets override which parameters are tunable for this run. "
                 "Phase 2 presets (candidate filters, cooldown, sizing) are not yet available.",
        )
        if _sel != _preset_opts[0]:
            preset = _sel
            from tuning.presets import _PRESETS
            st.caption(f"**{preset}** — {_PRESETS[preset]['description']}")

        st.info(
            "Active Sleeve: `index_pct` and ETF routing params are always frozen. "
            "Optimizer ranks by active sleeve metrics (active Sharpe / Calmar)."
        )

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
        _render_random_search_mode(scope=scope, preset=preset)
    else:
        _render_scipy_mode(scope=scope, preset=preset)

    st.divider()
    from ui.components.config_variants import render_config_variants
    render_config_variants()
