"""
ui/components/conditional_features.py — Conditional Factor Interactions research page.

Evaluates whether momentum is more predictive when conditioned on quality, income, or value.
All engineered features are research instruments; nothing here alters live scoring.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Cached compute helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _load_summary(horizon: int, ic_type: str) -> pd.DataFrame:
    from research.ic_engine import FactorResearchEngine
    return FactorResearchEngine().compute_conditional_ic(horizon_days=horizon, ic_type=ic_type)


@st.cache_data(ttl=600)
def _load_timeseries(horizon: int, ic_type: str) -> pd.DataFrame:
    from research.ic_engine import FactorResearchEngine
    return FactorResearchEngine().compute_conditional_ic_timeseries(horizon_days=horizon, ic_type=ic_type)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_GROUP_COLOR: dict[str, str] = {
    "quality_conditioned": "#2ecc71",
    "income_conditioned":  "#3498db",
    "blended":             "#9b59b6",
    "value_conditioned":   "#e67e22",
    "baseline":            "#95a5a6",
}

_GROUP_LABEL: dict[str, str] = {
    "quality_conditioned": "Quality-Conditioned",
    "income_conditioned":  "Income-Conditioned",
    "blended":             "Blended",
    "value_conditioned":   "Value-Conditioned (Exp.)",
    "baseline":            "Baseline",
}

_METRIC_FMT: dict[str, str] = {
    "mean_ic":         "{:+.4f}",
    "icir":            "{:+.3f}",
    "hit_rate":        "{:.1%}",
    "t_stat":          "{:+.3f}",
    "tail_ic":         "{:+.4f}",
    "stability_score": "{:.1%}",
}


def _ic_color(v: float) -> str:
    if v > 0.04:
        return "#2ecc71"
    if v > 0.0:
        return "#f1c40f"
    return "#e74c3c"


def _verdict(row: pd.Series, baseline_mean_ic: float) -> str:
    delta = row["mean_ic"] - baseline_mean_ic
    if row["n_periods"] < 3:
        return "⚪ Insufficient data"
    if delta > 0.01 and row["hit_rate"] >= 0.55:
        return "🟢 Better than baseline"
    if delta > 0.005:
        return "🟡 Marginally better"
    if delta < -0.01:
        return "🔴 Worse than baseline"
    return "⚪ No improvement"


# ---------------------------------------------------------------------------
# Leaderboard table
# ---------------------------------------------------------------------------

def _render_leaderboard(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data — need ≥ 2 snapshots with the required base score columns.")
        return

    baseline_row = df[df["feature"] == "momentum_score"]
    baseline_ic  = float(baseline_row["mean_ic"].iloc[0]) if not baseline_row.empty else 0.0

    st.caption(
        f"**Baseline momentum_score mean IC: {baseline_ic:+.4f}** — "
        "features above the baseline row beat raw momentum on this horizon."
    )

    display = df.copy()
    display["Group"]          = display["group"].map(_GROUP_LABEL).fillna(display["group"])
    display["Feature"]        = display["label"]
    display["Mean IC"]        = display["mean_ic"].map("{:+.4f}".format)
    display["ICIR"]           = display["icir"].map("{:+.3f}".format)
    display["Hit Rate"]       = display["hit_rate"].map("{:.1%}".format)
    display["t-stat"]         = display["t_stat"].map("{:+.3f}".format)
    display["Tail IC (p25)"]  = display["tail_ic"].map("{:+.4f}".format)
    display["Stability"]      = display["stability_score"].map("{:.1%}".format)
    display["n"]              = display["n_periods"]
    display["Verdict"]        = display.apply(lambda r: _verdict(r, baseline_ic), axis=1)

    cols_show = ["Feature", "Group", "Mean IC", "ICIR", "Hit Rate",
                 "t-stat", "Tail IC (p25)", "Stability", "n", "Verdict"]
    st.dataframe(
        display[cols_show].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Rolling IC comparison chart
# ---------------------------------------------------------------------------

def _render_timeseries(ts: pd.DataFrame, selected_features: list[str]) -> None:
    if ts.empty:
        st.info("No time-series data available.")
        return

    fig = go.Figure()

    for feat in selected_features:
        grp   = ts[ts["feature"] == feat].sort_values("date")
        if grp.empty:
            continue
        is_baseline = feat == "momentum_score"
        from strategy.factor_interactions import _FEAT_META
        meta  = _FEAT_META.get(feat, {})
        label = meta.get("label", "Baseline Momentum") if not is_baseline else "Baseline Momentum"
        group = meta.get("group", "baseline") if not is_baseline else "baseline"
        color = _GROUP_COLOR.get(group, "#aaa")
        width = 3 if is_baseline else 1.5
        dash  = "dash" if is_baseline else "solid"

        fig.add_trace(go.Scatter(
            x=grp["date"],
            y=grp["ic"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=width, dash=dash),
            marker=dict(size=5),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
    fig.add_hrect(y0=0.02, y1=0.06,  fillcolor="green", opacity=0.04, line_width=0)
    fig.add_hrect(y0=-0.06, y1=-0.02, fillcolor="red",   opacity=0.04, line_width=0)

    fig.update_layout(
        xaxis_title="Snapshot Date",
        yaxis_title="IC (Spearman rank correlation)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=30, b=40),
        height=400,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#cdd6f4"),
        xaxis=dict(gridcolor="#2d3436"),
        yaxis=dict(gridcolor="#2d3436", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Decile bar chart
# ---------------------------------------------------------------------------

def _render_decile_comparison(best_feature: str, horizon: int) -> None:
    from research.ic_engine import FactorResearchEngine

    @st.cache_data(ttl=600)
    def _decile(feat: str, h: int) -> pd.DataFrame:
        # Decile spread requires adding interaction features to each snapshot first —
        # use a thin wrapper that sets features on snapshot copies.
        from strategy.factor_interactions import add_interaction_features
        engine = FactorResearchEngine()
        dates_map = engine._load_dates_map()
        sorted_dates = sorted(dates_map.keys())
        min_days = max(1, int(h * (1 - engine.max_horizon_slop_pct)))
        max_days = int(h * (1 + engine.max_horizon_slop_pct))
        n_deciles = 5

        pooled: list[pd.DataFrame] = []
        for i, t_date in enumerate(sorted_dates):
            fwd_date = None
            best_diff = 9999
            for fwd in sorted_dates[i + 1:]:
                diff = (fwd - t_date).days
                if diff < min_days:
                    continue
                if diff > max_days:
                    break
                if abs(diff - h) < best_diff:
                    fwd_date = fwd
                    best_diff = abs(diff - h)
            if fwd_date is None:
                continue
            df_t   = dates_map[t_date].copy()
            df_fwd = dates_map[fwd_date]
            fr = engine._forward_returns(df_t, df_fwd)
            if fr.empty:
                continue
            if feat not in engine._load_dates_map()[t_date].columns:
                add_interaction_features(df_t)
            if "symbol" not in df_t.columns or feat not in df_t.columns:
                continue
            fv = pd.to_numeric(df_t.set_index("symbol")[feat], errors="coerce")
            merged = pd.DataFrame({"factor_val": fv, "forward_return": fr}).dropna()
            if len(merged) >= n_deciles:
                pooled.append(merged)

        if not pooled:
            return pd.DataFrame()
        all_data = pd.concat(pooled)
        try:
            all_data["decile"] = pd.qcut(all_data["factor_val"], n_deciles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.DataFrame()
        return (
            all_data.groupby("decile")["forward_return"]
            .agg(mean_forward_return="mean", n_stocks="count")
            .reset_index()
        )

    col_a, col_b = st.columns(2)
    for col, feat, title in [
        (col_a, best_feature, "Best Engineered Feature"),
        (col_b, "momentum_score", "Baseline Momentum"),
    ]:
        df_d = _decile(feat, horizon)
        with col:
            st.caption(title)
            if df_d.empty:
                st.info("Insufficient data for decile analysis.")
                continue
            fig = go.Figure(go.Bar(
                x=df_d["decile"].astype(str),
                y=(df_d["mean_forward_return"] * 100).round(3),
                marker_color=[
                    "#2ecc71" if v > 0 else "#e74c3c"
                    for v in df_d["mean_forward_return"]
                ],
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)
            fig.update_layout(
                xaxis_title="Quintile",
                yaxis_title="Mean Fwd Return (%)",
                margin=dict(l=30, r=10, t=20, b=30),
                height=280,
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#cdd6f4"),
                xaxis=dict(gridcolor="#2d3436"),
                yaxis=dict(gridcolor="#2d3436", zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Research conclusion
# ---------------------------------------------------------------------------

def _render_conclusion(df: pd.DataFrame, horizon: int) -> None:
    if df.empty:
        return

    baseline_row = df[df["feature"] == "momentum_score"]
    engineered   = df[df["feature"] != "momentum_score"]

    if baseline_row.empty or engineered.empty:
        return

    baseline_ic   = float(baseline_row["mean_ic"].iloc[0])
    best          = engineered.iloc[0]
    best_delta    = best["mean_ic"] - baseline_ic
    n_better      = int((engineered["mean_ic"] > baseline_ic).sum())
    n_total       = len(engineered)
    min_periods   = int(df["n_periods"].min())

    # Confidence is low unless we have enough observations
    if min_periods < 5:
        confidence = "LOW"
        conf_note  = f"Only {min_periods} snapshot pairs — treat all findings as exploratory."
    elif min_periods < 15:
        confidence = "MEDIUM"
        conf_note  = f"{min_periods} snapshot pairs — directional signal, not statistically definitive."
    else:
        confidence = "HIGH"
        conf_note  = f"{min_periods} snapshot pairs — results are more reliable."

    _CONF_ICON = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}

    with st.container(border=True):
        st.markdown(f"**Research Conclusion — {horizon}-day horizon**")
        st.markdown(
            f"{_CONF_ICON[confidence]} **Confidence: {confidence}** — {conf_note}"
        )
        st.divider()

        if best_delta > 0.005 and best["hit_rate"] >= 0.5:
            st.success(
                f"**Yes — conditioning momentum on {_GROUP_LABEL.get(best['group'], best['group'])} "
                f"improves IC.**\n\n"
                f"Best feature: **{best['label']}** — "
                f"Mean IC {best['mean_ic']:+.4f} vs baseline {baseline_ic:+.4f} "
                f"(Δ {best_delta:+.4f}).  "
                f"{n_better}/{n_total} engineered features beat the baseline."
            )
            if best["n_periods"] >= 5 and abs(best["t_stat"]) > 1.5:
                st.markdown(
                    f"t-stat = {best['t_stat']:+.2f} and hit rate = {best['hit_rate']:.0%} — "
                    f"the improvement is directionally consistent."
                )
        elif best_delta > 0:
            st.warning(
                f"**Marginal improvement.** Best feature: **{best['label']}** "
                f"(Δ IC {best_delta:+.4f} over baseline). "
                f"Improvement is small — do not change production weights based on this alone."
            )
        else:
            st.info(
                f"**No improvement observed.** Raw momentum_score (IC {baseline_ic:+.4f}) "
                f"outperforms all {n_total} engineered variants at the {horizon}-day horizon. "
                f"Current linear weighting appears adequate."
            )

        st.caption(
            "⚠️ These results are based on snapshot data only. "
            "IC magnitude < 0.05 is common in equity factor research and does not imply the factor is useless. "
            "**Do not modify config.yaml score_weights based solely on this tab** — "
            "use at least 20 snapshots and consult the IC Analysis tab for corroboration."
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Conditional Factor Interactions")
    st.caption(
        "Research question: **Does momentum work better when filtered through quality, income, or value?** "
        "All variants are computed on-the-fly from snapshot data. Nothing here affects live scoring."
    )

    from strategy.factor_interactions import INTERACTION_FEATURES

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        horizon = st.selectbox(
            "Horizon (days)", [5, 20, 60, 120], index=1,
            key="cfi_horizon",
        )
    with c2:
        ic_type = st.selectbox(
            "IC type", ["spearman", "pearson"], index=0,
            key="cfi_ic_type",
        )
    with c3:
        show_groups = st.multiselect(
            "Show groups",
            ["quality_conditioned", "income_conditioned", "blended", "value_conditioned"],
            default=["quality_conditioned", "income_conditioned", "blended"],
            format_func=lambda g: _GROUP_LABEL.get(g, g),
            key="cfi_groups",
        )

    st.divider()

    with st.spinner("Computing conditional IC across snapshots…"):
        summary = _load_summary(horizon, ic_type)
        ts      = _load_timeseries(horizon, ic_type)

    if summary.empty:
        st.warning(
            "Not enough snapshot data. "
            "Need ≥ 2 snapshots in `data/snapshots/` with `momentum_score`, "
            "`quality_score`, `income_score`, `value_score` columns."
        )
        return

    # Filter to selected groups + baseline
    visible_features = (
        summary[
            (summary["group"].isin(show_groups)) | (summary["feature"] == "momentum_score")
        ]
    )

    # ── Leaderboard ──────────────────────────────────────────────────────────
    st.markdown("#### Leaderboard")
    _render_leaderboard(visible_features)

    st.divider()

    # ── Rolling IC comparison ────────────────────────────────────────────────
    st.markdown("#### IC Over Time — Engineered vs Baseline")
    st.caption(
        "Each point is one snapshot-pair's IC. Dashed line = baseline momentum_score. "
        "Select which features to overlay."
    )

    feat_options = [f["name"] for f in INTERACTION_FEATURES if f["group"] in show_groups]
    feat_labels  = {f["name"]: f["label"] for f in INTERACTION_FEATURES}

    selected_ts = st.multiselect(
        "Features to plot",
        feat_options,
        default=feat_options[:4] if len(feat_options) >= 4 else feat_options,
        format_func=lambda n: feat_labels.get(n, n),
        key="cfi_ts_select",
    )

    _render_timeseries(ts, ["momentum_score"] + selected_ts)

    st.divider()

    # ── Decile spread ────────────────────────────────────────────────────────
    best_row = summary[summary["feature"] != "momentum_score"]
    if not best_row.empty:
        best_feature = str(best_row.iloc[0]["feature"])
        best_label   = str(best_row.iloc[0]["label"])
        st.markdown("#### Quintile Spread — Best vs Baseline")
        st.caption(
            f"Pooled quintiles across all snapshot pairs. "
            f"Best engineered feature: **{best_label}**."
        )
        _render_decile_comparison(best_feature, horizon)

    st.divider()

    # ── Research conclusion ──────────────────────────────────────────────────
    st.markdown("#### Research Conclusion")
    _render_conclusion(summary, horizon)

    # ── Feature descriptions ─────────────────────────────────────────────────
    with st.expander("Feature descriptions"):
        for feat in INTERACTION_FEATURES:
            if feat["group"] in show_groups:
                st.markdown(
                    f"**`{feat['name']}`** ({_GROUP_LABEL.get(feat['group'], feat['group'])})  \n"
                    f"{feat['description']}"
                )
