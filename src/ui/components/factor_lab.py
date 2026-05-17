"""
ui/components/factor_lab.py — Factor Lab: multi-horizon IC, decay, and monotonicity.

Research-grade factor validation:
  1. Research Summary     — synthesized conclusions from IC statistics
  2. Weight Recommender   — IC-derived factor weight suggestions
  3. IC summary table     — all factor × horizon pairs
  4. Factor decay curves
  5. Cumulative IC over time
  6. Decile spread / monotonicity
  7. Rolling ICIR
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600)
def _compute_ic_data(factors: tuple, horizons: tuple, ic_type: str) -> dict:
    from strategy.research.ic_engine import FactorResearchEngine

    engine  = FactorResearchEngine(factors=list(factors), horizons=list(horizons))
    ic_df   = engine.compute_multi_horizon_ic(ic_type=ic_type)
    summary = engine.compute_ic_summary(ic_df)
    decay   = engine.compute_factor_decay(ic_type=ic_type)
    return {"ic_df": ic_df, "summary": summary, "decay": decay}


@st.cache_data(ttl=600)
def _compute_decile_spread(factor: str, horizon: int, n_deciles: int) -> pd.DataFrame:
    from strategy.research.ic_engine import FactorResearchEngine
    return FactorResearchEngine().compute_decile_spread(factor, horizon_days=horizon, n_deciles=n_deciles)


@st.cache_data(ttl=600)
def _compute_rolling_icir(factor: str, horizon: int, window: int) -> pd.DataFrame:
    from strategy.research.ic_engine import FactorResearchEngine
    return FactorResearchEngine().compute_rolling_icir(factor, horizon_days=horizon, window=window)


@st.cache_data(ttl=600)
def _compute_cumulative_ic(factors: tuple, horizon: int, ic_type: str) -> pd.DataFrame:
    from strategy.research.ic_engine import FactorResearchEngine
    return FactorResearchEngine(factors=list(factors)).compute_cumulative_ic(
        list(factors), horizon_days=horizon, ic_type=ic_type,
    )


@st.cache_data(ttl=3600)
def _compute_regime_ic(factors: tuple, horizon: int, ic_type: str) -> pd.DataFrame:
    from strategy.research.ic_engine import FactorResearchEngine
    return FactorResearchEngine(factors=list(factors)).compute_regime_conditioned_ic(
        factors=list(factors), horizon_days=horizon, ic_type=ic_type,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRIENDLY: dict[str, str] = {
    "value_score":    "Value",
    "quality_score":  "Quality",
    "income_score":   "Income",
    "momentum_score": "Momentum",
    "value_metric":   "Composite",
}

_FACTOR_TO_CONFIG: dict[str, str] = {
    "value_score":    "value",
    "quality_score":  "quality",
    "income_score":   "income",
    "momentum_score": "momentum",
}

_CONF_ICON  = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
_QUAL_ICON  = {"Strong": "🟢", "Moderate": "🟡", "Weak": "🔴"}

_REGIME_ORDER  = ["bull", "sideways", "bear", "high_vol"]
_REGIME_LABEL  = {"bull": "Bull", "sideways": "Sideways", "bear": "Bear", "high_vol": "High Vol"}


# ---------------------------------------------------------------------------
# Research synthesis  (pure functions — no Streamlit)
# ---------------------------------------------------------------------------


def _research_cfg() -> dict:
    """Load research thresholds from config, with safe fallbacks."""
    try:
        from config.manager import ConfigManager
        rc = ConfigManager.get().research
        return {
            "min_snapshots_for_weight_recommendations": rc.min_snapshots_for_weight_recommendations,
            "min_snapshots_for_high_confidence":        rc.min_snapshots_for_high_confidence,
        }
    except Exception:
        return {
            "min_snapshots_for_weight_recommendations": 20,
            "min_snapshots_for_high_confidence":        60,
        }


def _synthesize_conclusions(summary: pd.DataFrame, n_dates: int) -> dict:
    """Derive qualitative research state from IC summary statistics."""
    if summary.empty:
        return {"insufficient_data": True}

    rcfg = _research_cfg()
    min_snap_weights = rcfg["min_snapshots_for_weight_recommendations"]
    min_snap_high    = rcfg["min_snapshots_for_high_confidence"]

    # Per-factor aggregates across all horizons
    bf = (
        summary.groupby("factor")
        .agg(
            mean_ic=("mean_ic", "mean"),
            mean_icir=("icir", "mean"),
            mean_abs_tstat=("t_stat", lambda x: x.abs().mean()),
            n_periods=("n_periods", "mean"),
        )
    )

    # Short-horizon IC (≤ 20 days) — most actionable signal
    short_df = summary[summary["horizon_days"] <= 20]
    if short_df.empty:
        short_df = summary
    short_ic = short_df.groupby("factor")["mean_ic"].mean()

    strongest_key   = short_ic.idxmax()       if not short_ic.empty else None
    most_stable_key = bf["mean_icir"].idxmax() if not bf.empty else None
    weakest_key     = bf["mean_ic"].idxmin()   if not bf.empty else None

    # Regime-sensitive = biggest short→long IC decay (works only in certain conditions)
    regime_sens_key = None
    long_df = summary[summary["horizon_days"] >= 60]
    if not long_df.empty:
        long_ic = long_df.groupby("factor")["mean_ic"].mean()
        common  = short_ic.index.intersection(long_ic.index)
        if len(common):
            regime_sens_key = (short_ic[common] - long_ic[common]).abs().idxmax()

    # Forward predictive breadth: factors with |mean_ic| > 0.05 AND |t_stat| > 1.65
    useful   = (bf["mean_ic"].abs() > 0.05) & (bf["mean_abs_tstat"] > 1.65)
    n_useful = int(useful.sum())
    n_total  = len(bf)

    # Validation confidence — thresholds from config
    avg_tstat = float(bf["mean_abs_tstat"].mean())
    if n_dates >= min_snap_high and avg_tstat >= 1.5:
        confidence = "HIGH"
    elif n_dates >= min_snap_weights and avg_tstat >= 0.8:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Composite quality
    avg_icir = float(bf["mean_icir"].mean())
    if n_useful >= 3 and avg_icir > 0.5:
        composite = "Strong"
    elif n_useful >= 2 or avg_icir > 0.2:
        composite = "Moderate"
    else:
        composite = "Weak"

    # Most concerning issue (priority-ordered)
    concern = None
    val_rows = short_df[short_df["factor"] == "value_score"]["mean_ic"]
    val_short = float(val_rows.mean()) if not val_rows.empty else float("nan")

    if not pd.isna(val_short) and val_short < -0.03:
        concern = f"Value factor negative IC ({val_short:+.3f}) across short horizons"
    elif confidence == "LOW":
        concern = f"Insufficient history ({n_dates} snapshot dates) — results unreliable"
    elif n_useful < 2:
        concern = f"Only {n_useful}/{n_total} factors show statistically useful signal"
    elif avg_icir < 0.15:
        concern = "All factors show low ICIR — signal is highly inconsistent across periods"

    return {
        "insufficient_data": False,
        "strongest":         strongest_key,
        "most_stable":       most_stable_key,
        "weakest":           weakest_key,
        "regime_sensitive":  regime_sens_key,
        "composite":         composite,
        "confidence":        confidence,
        "n_useful":          n_useful,
        "n_total":           n_total,
        "concern":           concern,
    }


def _synthesize_weights(summary: pd.DataFrame) -> dict:
    """
    Derive suggested factor weights from IC statistics.

    Weight ∝ max(0, mean_ic) × (1 + max(0, icir − 0.3) × 0.5)
    Factors with negative mean IC receive zero weight.
    Focuses on 20–60 day horizons as most actionable.
    """
    med = summary[summary["horizon_days"].isin([20, 60])]
    if med.empty:
        med = summary

    by_factor = med.groupby("factor").agg(
        mean_ic=("mean_ic", "mean"),
        icir=("icir", "mean"),
    ).reset_index()

    scores:  dict[str, float] = {}
    reasons: dict[str, str]   = {}
    core = ["value", "quality", "income", "momentum"]

    for _, row in by_factor.iterrows():
        cfg_key = _FACTOR_TO_CONFIG.get(row["factor"])
        if cfg_key is None:
            continue
        mean_ic = float(row["mean_ic"])
        icir    = float(row["icir"])
        score   = max(0.0, mean_ic) * (1.0 + max(0.0, icir - 0.3) * 0.5)
        scores[cfg_key] = score

        if mean_ic < -0.02:
            reasons[cfg_key] = f"IC = {mean_ic:+.3f} (negative — downweighted to zero)"
        elif icir >= 0.5:
            reasons[cfg_key] = f"IC = {mean_ic:+.3f}, ICIR = {icir:.2f} (strong consistency)"
        elif icir >= 0.2:
            reasons[cfg_key] = f"IC = {mean_ic:+.3f}, ICIR = {icir:.2f} (moderate)"
        else:
            reasons[cfg_key] = f"IC = {mean_ic:+.3f}, ICIR = {icir:.2f} (low consistency)"

    # Floor missing factors
    for k in core:
        if k not in scores:
            scores[k]  = 0.01
            reasons[k] = "No IC data available — assigned minimum floor"

    total = sum(scores.values())
    if total < 1e-9:
        weights = {k: 0.25 for k in core}
        reasons = {k: v + " (equal-weight fallback)" for k, v in reasons.items()}
    else:
        weights = {k: round(scores[k] / total, 3) for k in core}
        # Correct rounding error on the largest bucket
        diff = round(1.0 - sum(weights.values()), 3)
        if diff:
            largest = max(weights, key=weights.get)
            weights[largest] = round(weights[largest] + diff, 3)

    return {
        "weights": dict(sorted(weights.items(), key=lambda x: -x[1])),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_research_summary(conc: dict, n_dates: int) -> None:
    """Render the 'Current Research State' health card."""
    fn = _FRIENDLY

    with st.container(border=True):
        st.markdown("#### CURRENT RESEARCH STATE")

        if conc.get("insufficient_data"):
            st.warning(
                "Not enough snapshot data to synthesize conclusions. "
                "Need ≥ 2 parquet files in `data/snapshots/`."
            )
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Strongest factor",        fn.get(conc["strongest"],        conc["strongest"]        or "—"))
        c2.metric("Most stable factor",      fn.get(conc["most_stable"],      conc["most_stable"]      or "—"))
        c3.metric("Weakest factor",          fn.get(conc["weakest"],          conc["weakest"]          or "—"))
        c4.metric("Regime-sensitive factor", fn.get(conc["regime_sensitive"], conc["regime_sensitive"] or "—"))

        r1, r2, r3 = st.columns(3)
        q_icon  = _QUAL_ICON.get(conc["composite"], "⚪")
        cf_icon = _CONF_ICON.get(conc["confidence"], "⚪")
        r1.metric("Composite quality",     f"{q_icon} {conc['composite']}")
        r2.metric("Validation confidence", f"{cf_icon} {conc['confidence']}")
        r3.metric("Predictive breadth",    f"{conc['n_useful']} / {conc['n_total']} factors")

        if conc["concern"]:
            st.warning(f"⚠️  **Most concerning issue:** {conc['concern']}")
        else:
            st.success("✅  No major concerns detected in current IC statistics.")


def _render_regime_ic(regime_df: pd.DataFrame, sel_factors: list[str]) -> None:
    """Render the regime-conditioned IC heatmap and per-factor regime bar chart."""
    st.subheader("IC by Market Regime")
    st.caption(
        "Same factor — very different signal across market states. "
        "Regimes: **Bull** (SPY > 200DMA, VIX < 25) · **Sideways** (corrective) · "
        "**Bear** (SPY < 200DMA) · **High Vol** (VIX ≥ 25)."
    )

    if regime_df.empty:
        st.info(
            "Regime-conditioned IC requires downloading SPY + VIX history. "
            "Check your internet connection, or build more snapshot history."
        )
        return

    # ── Heatmap pivot ────────────────────────────────────────────────────────
    pivot = (
        regime_df.pivot_table(index="factor", columns="regime", values="mean_ic", aggfunc="mean")
        .reindex(columns=[r for r in _REGIME_ORDER if r in regime_df["regime"].unique()])
        .rename(columns=_REGIME_LABEL)
    )
    pivot.index = [_FRIENDLY.get(f, f) for f in pivot.index]

    # Counts pivot for annotations
    counts = (
        regime_df.pivot_table(index="factor", columns="regime", values="n_periods", aggfunc="sum")
        .reindex(columns=[r for r in _REGIME_ORDER if r in regime_df["regime"].unique()])
        .rename(columns=_REGIME_LABEL)
    )
    counts.index = [_FRIENDLY.get(f, f) for f in counts.index]

    # Build annotation text: "IC\n(n=N)"
    annot = pivot.copy().astype(object)
    for col in pivot.columns:
        for idx in pivot.index:
            ic_val = pivot.loc[idx, col]
            n_val  = int(counts.loc[idx, col]) if (idx in counts.index and col in counts.columns) else 0
            if pd.isna(ic_val):
                annot.loc[idx, col] = "—"
            else:
                annot.loc[idx, col] = f"{ic_val:+.3f}\n(n={n_val})"

    # Plotly heatmap
    z_vals = pivot.values.tolist()
    fig_heat = go.Figure(go.Heatmap(
        z=z_vals,
        x=list(pivot.columns),
        y=list(pivot.index),
        text=[[annot.loc[r, c] for c in pivot.columns] for r in pivot.index],
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale="RdYlGn",
        zmid=0,
        zmin=-0.20,
        zmax=0.20,
        colorbar=dict(title="Mean IC", tickformat=".2f"),
    ))
    fig_heat.update_layout(
        height=max(220, len(pivot) * 70 + 60),
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Key insight callouts ─────────────────────────────────────────────────
    if not pivot.empty:
        insights: list[str] = []
        for factor_label in pivot.index:
            row = pivot.loc[factor_label].dropna()
            if len(row) < 2:
                continue
            best_regime  = row.idxmax()
            worst_regime = row.idxmin()
            spread       = float(row.max() - row.min())
            if spread > 0.10:
                insights.append(
                    f"**{factor_label}** is highly regime-sensitive "
                    f"(spread {spread:.2f}): strongest in **{best_regime}**, "
                    f"weakest in **{worst_regime}**."
                )
            elif float(row.max()) < 0.02:
                insights.append(
                    f"**{factor_label}** shows near-zero or negative IC across all regimes — "
                    "consider reducing weight."
                )
        if insights:
            with st.expander("Key regime insights", expanded=True):
                for ins in insights:
                    st.markdown(f"• {ins}")

    # ── Per-factor regime bar chart ───────────────────────────────────────────
    st.caption("Drill into a single factor to see its regime IC profile.")
    core_factors = [f for f in sel_factors if f in _FACTOR_TO_CONFIG]
    drill_factor = st.selectbox(
        "Factor (regime drill-down)",
        core_factors,
        format_func=lambda x: _FRIENDLY.get(x, x),
        key="fl_regime_drill",
    )
    factor_rows = regime_df[regime_df["factor"] == drill_factor].copy()

    if not factor_rows.empty:
        factor_rows["regime_label"] = factor_rows["regime"].map(
            lambda r: _REGIME_LABEL.get(r, r)
        )
        factor_rows = factor_rows.sort_values(
            "regime", key=lambda s: s.map({r: i for i, r in enumerate(_REGIME_ORDER)})
        )

        bar_colors = [
            "#4CAF50" if ic > 0.05 else "#F44336" if ic < -0.02 else "#9E9E9E"
            for ic in factor_rows["mean_ic"]
        ]
        fig_bar = go.Figure(go.Bar(
            x=factor_rows["regime_label"],
            y=factor_rows["mean_ic"],
            marker_color=bar_colors,
            text=[f"{v:+.3f}<br>(n={n})" for v, n in zip(factor_rows["mean_ic"], factor_rows["n_periods"])],
            textposition="outside",
        ))
        fig_bar.add_hline(y=0.05,  line_dash="dash", line_color="green", line_width=1)
        fig_bar.add_hline(y=0,     line_dash="solid", line_color="gray",  line_width=0.5)
        fig_bar.add_hline(y=-0.05, line_dash="dash", line_color="red",   line_width=1)
        fig_bar.update_layout(
            yaxis_title="Mean IC",
            yaxis=dict(tickformat=".3f", range=[
                min(-0.12, float(factor_rows["mean_ic"].min()) * 1.4),
                max(0.12,  float(factor_rows["mean_ic"].max()) * 1.4),
            ]),
            height=300,
            margin=dict(t=10, b=10),
            title_text=f"{_FRIENDLY.get(drill_factor, drill_factor)} — IC by Regime",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ICIR table for this factor
        disp = factor_rows[["regime_label", "mean_ic", "icir", "hit_rate", "t_stat", "n_periods"]].copy()
        disp.columns = ["Regime", "Mean IC", "ICIR", "Hit Rate", "t-stat", "N"]
        disp = disp.set_index("Regime")

        def _regime_ic_color(val):
            if not isinstance(val, float):
                return ""
            if val > 0.05:
                return "color: #00b300; font-weight: bold"
            if val < -0.02:
                return "color: #cc0000; font-weight: bold"
            return "color: #888888"

        st.dataframe(
            disp.style.format("{:.3f}", subset=["Mean IC", "ICIR", "Hit Rate", "t-stat"])
            .map(_regime_ic_color, subset=["Mean IC"]),
            use_container_width=True,
        )
    else:
        st.info(f"No regime IC data for {_FRIENDLY.get(drill_factor, drill_factor)}.")


def _render_weight_recommendations(recs: dict, n_dates: int = 0) -> None:
    """Render the Factor Weight Recommendation card."""
    from util import SCORE_WEIGHTS

    rcfg     = _research_cfg()
    min_snap = rcfg["min_snapshots_for_weight_recommendations"]

    weights  = recs["weights"]
    reasons  = recs["reasons"]
    friendly = {"value": "Value", "quality": "Quality", "income": "Income", "momentum": "Momentum"}
    max_w    = max(weights.values()) if weights else 1.0

    with st.container(border=True):
        st.markdown("#### FACTOR WEIGHT RECOMMENDATIONS")
        if n_dates < min_snap:
            st.warning(
                f"⚠️ Only {n_dates} snapshot dates available — "
                f"recommendations require ≥ {min_snap}. "
                "Treat these weights as tentative."
            )
        st.caption(
            "Derived from IC strength and consistency at 20–60 day horizons. "
            "**Read-only** — apply manually via `config.yaml`."
        )

        # Horizontal bar chart
        bar_colors = [
            "#4CAF50" if weights[k] >= 0.30 else
            "#FF9800" if weights[k] >= 0.10 else
            "#9E9E9E"
            for k in weights
        ]
        fig = go.Figure(go.Bar(
            x=[weights[k] for k in weights],
            y=[friendly.get(k, k) for k in weights],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{weights[k]:.1%}" for k in weights],
            textposition="outside",
        ))
        fig.add_vline(
            x=0.25, line_dash="dot", line_color="#888", line_width=1,
            annotation_text="Equal weight", annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(tickformat=".0%", range=[0, max_w * 1.40]),
            height=190,
            margin=dict(t=8, b=8, l=0, r=80),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics with delta vs current config
        cols = st.columns(len(weights))
        for col, (k, w) in zip(cols, weights.items()):
            current = SCORE_WEIGHTS.get(k, 0.0)
            col.metric(
                friendly.get(k, k),
                f"{w:.1%}",
                delta=f"{w - current:+.1%} vs config",
                delta_color="normal",
            )

        # Reasoning
        st.divider()
        st.caption("**Reasoning**")
        for k in weights:
            st.caption(f"• **{friendly.get(k, k)}:** {reasons.get(k, '')}")

        # YAML copy block
        with st.expander("Config YAML snippet"):
            yaml_body = "score_weights:\n" + "".join(
                f"  {k}: {weights[k]:.3f}\n" for k in weights
            )
            st.code(yaml_body, language="yaml")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render() -> None:
    st.title("🧪 Factor Lab")
    st.caption("Multi-horizon IC, factor decay, and monotonicity diagnostics.")

    # ── Controls ─────────────────────────────────────────────────────────────
    all_factors  = ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]
    all_horizons = [5, 20, 60, 120, 252]

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        sel_factors = st.multiselect(
            "Factors", all_factors,
            default=all_factors,
            format_func=lambda x: _FRIENDLY.get(x, x),
            key="fl_factors",
        )
    with c2:
        sel_horizons = st.multiselect(
            "Horizons (days)", all_horizons, default=[5, 20, 60, 120], key="fl_horizons"
        )
    with c3:
        ic_type = st.selectbox("IC type", ["spearman", "pearson"], key="fl_ic_type")

    if not sel_factors or not sel_horizons:
        st.warning("Select at least one factor and one horizon.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Computing multi-horizon IC…"):
        try:
            data = _compute_ic_data(
                tuple(sel_factors), tuple(sorted(sel_horizons)), ic_type
            )
        except Exception as exc:
            st.error(f"IC computation failed: {exc}")
            st.exception(exc)
            return

    ic_df   = data["ic_df"]
    summary = data["summary"]
    decay   = data["decay"]

    if ic_df.empty:
        st.warning(
            "Not enough snapshots for IC computation — need at least 2 dated parquet files "
            "in `data/snapshots/`. Run the bot on at least two separate days."
        )
        return

    n_dates  = int(ic_df["date"].nunique())
    n_stocks = int(ic_df["n_stocks"].median())

    # ── Research Summary card ─────────────────────────────────────────────────
    conc = _synthesize_conclusions(summary, n_dates)
    _render_research_summary(conc, n_dates)

    st.divider()

    # ── Weight Recommendations card ───────────────────────────────────────────
    recs = _synthesize_weights(summary)
    _render_weight_recommendations(recs, n_dates=n_dates)

    st.divider()

    # ── Regime-conditioned IC ─────────────────────────────────────────────────
    regime_horizon = st.selectbox(
        "Horizon for regime IC", sorted(sel_horizons), key="fl_regime_horizon",
        help="Which forward-return horizon to use when computing IC per regime.",
    )
    with st.spinner("Fetching regime history and computing regime-conditioned IC…"):
        try:
            regime_df = _compute_regime_ic(
                tuple(f for f in sel_factors if f != "value_metric"),
                regime_horizon,
                ic_type,
            )
        except Exception as exc:
            regime_df = pd.DataFrame()
            st.caption(f"Regime IC unavailable: {exc}")

    _render_regime_ic(regime_df, sel_factors)

    st.divider()

    # ── Snapshot overview metrics ─────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Snapshot dates used",       n_dates)
    m2.metric("Median universe size",      n_stocks)
    m3.metric("Factor × horizon pairs",    len(summary) if not summary.empty else 0)

    st.divider()

    # ── 1. IC Summary Table ───────────────────────────────────────────────────
    st.subheader("IC Summary")

    if not summary.empty:
        disp = summary.copy()
        disp["factor"] = disp["factor"].map(lambda x: _FRIENDLY.get(x, x))
        disp.columns   = [c.replace("_", " ").title() for c in disp.columns]
        num_cols       = disp.select_dtypes("number").columns.tolist()

        def _color_ic(val: object):
            if not isinstance(val, float):
                return ""
            if val > 0.05:
                return "color: #00b300; font-weight: bold"
            if val < -0.05:
                return "color: #cc0000; font-weight: bold"
            return "color: #888888"

        styled   = disp.style.format({c: "{:.4f}" for c in num_cols})
        mean_col = next((c for c in disp.columns if "Mean" in c and "Ic" in c), None)
        if mean_col:
            styled = styled.map(_color_ic, subset=[mean_col])

        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No IC summary data — snapshots may not span sufficient time.")

    st.divider()

    # ── 2. Factor Decay Curves ────────────────────────────────────────────────
    st.subheader("Factor Decay (IC by Horizon)")
    st.caption("How predictive power fades as forecast horizon increases.")

    if not decay.empty:
        fig_decay = go.Figure()
        for factor in decay["factor"].unique():
            grp = decay[decay["factor"] == factor].sort_values("horizon_days")
            fig_decay.add_trace(go.Scatter(
                x=grp["horizon_days"],
                y=grp["mean_ic"],
                mode="lines+markers",
                name=_FRIENDLY.get(factor, factor),
                marker=dict(size=7),
            ))
        fig_decay.add_hline(y=0.05,  line_dash="dash", line_color="green", line_width=1, annotation_text="IC = 0.05")
        fig_decay.add_hline(y=-0.05, line_dash="dash", line_color="red",   line_width=1, annotation_text="IC = −0.05")
        fig_decay.add_hline(y=0,     line_dash="solid", line_color="gray",  line_width=0.5)
        fig_decay.update_layout(
            xaxis_title="Horizon (days)",
            yaxis_title="Mean IC",
            height=380,
            margin=dict(t=10),
            legend_title="Factor",
        )
        st.plotly_chart(fig_decay, use_container_width=True)
    else:
        st.info("Decay data unavailable.")

    st.divider()

    # ── 3. Cumulative IC ──────────────────────────────────────────────────────
    st.subheader("Cumulative IC")
    cum_horizon = st.selectbox(
        "Horizon for cumulative IC", sorted(sel_horizons), key="fl_cum_horizon"
    )

    with st.spinner("Computing cumulative IC…"):
        try:
            cum_df = _compute_cumulative_ic(tuple(sel_factors), cum_horizon, ic_type)
        except Exception as exc:
            cum_df = pd.DataFrame()
            st.caption(f"Cumulative IC unavailable: {exc}")

    if not cum_df.empty:
        fig_cum = go.Figure()
        for factor in cum_df["factor"].unique():
            grp = cum_df[cum_df["factor"] == factor].sort_values("date")
            fig_cum.add_trace(go.Scatter(
                x=grp["date"],
                y=grp["cumulative_ic"],
                mode="lines",
                name=_FRIENDLY.get(factor, factor),
            ))
        fig_cum.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
        fig_cum.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative IC",
            height=320,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info("Not enough data for cumulative IC at this horizon.")

    st.divider()

    # ── 4. Decile Spread ──────────────────────────────────────────────────────
    st.subheader("Monotonicity / Decile Spread")
    st.caption(
        "Mean forward return by factor score decile. "
        "A monotonically increasing pattern signals genuine factor predictive power."
    )

    da, db, dc = st.columns(3)
    dec_factor  = da.selectbox("Factor",   sel_factors, key="fl_dec_factor",
                               format_func=lambda x: _FRIENDLY.get(x, x))
    dec_horizon = db.selectbox("Horizon",  sorted(sel_horizons), key="fl_dec_horizon")
    n_deciles   = dc.selectbox("Deciles",  [5, 10], index=1, key="fl_n_deciles")

    with st.spinner("Computing decile spread…"):
        try:
            decile_df = _compute_decile_spread(dec_factor, dec_horizon, n_deciles)
        except Exception as exc:
            decile_df = pd.DataFrame()
            st.caption(f"Decile spread error: {exc}")

    if not decile_df.empty:
        colors = [
            "#00b300" if r >= 0 else "#cc0000"
            for r in decile_df["mean_forward_return"]
        ]
        fig_dec = go.Figure(go.Bar(
            x=decile_df["decile"],
            y=(decile_df["mean_forward_return"] * 100).round(2),
            marker_color=colors,
            text=[f"{r:.1%}" for r in decile_df["mean_forward_return"]],
            textposition="outside",
        ))
        fig_dec.add_hline(y=0, line_dash="solid", line_color="gray")
        fig_dec.update_layout(
            xaxis_title=f"Decile (1 = lowest {_FRIENDLY.get(dec_factor, dec_factor)}, {n_deciles} = highest)",
            yaxis_title="Mean Forward Return (%)",
            height=320,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_dec, use_container_width=True)

        if len(decile_df) >= 2:
            top_ret = decile_df.loc[decile_df["decile"].idxmax(), "mean_forward_return"]
            bot_ret = decile_df.loc[decile_df["decile"].idxmin(), "mean_forward_return"]
            st.caption(f"Top-minus-bottom spread: **{top_ret - bot_ret:.2%}**")
    else:
        st.info("Not enough data for decile analysis. Build more snapshot history.")

    st.divider()

    # ── 5. Rolling ICIR ───────────────────────────────────────────────────────
    st.subheader("Rolling ICIR")
    st.caption("ICIR = mean(IC) / std(IC) — measures consistency of predictive signal.")

    re, rf = st.columns(2)
    roll_factor  = re.selectbox("Factor",  sel_factors, key="fl_roll_factor",
                                format_func=lambda x: _FRIENDLY.get(x, x))
    roll_horizon = rf.selectbox("Horizon", sorted(sel_horizons), key="fl_roll_horizon")
    roll_window  = st.slider("Rolling window (periods)", 3, 24, 8, key="fl_roll_win")

    with st.spinner("Computing rolling ICIR…"):
        try:
            roll_df = _compute_rolling_icir(roll_factor, roll_horizon, roll_window)
        except Exception as exc:
            roll_df = pd.DataFrame()
            st.caption(f"Rolling ICIR error: {exc}")

    if not roll_df.empty and "rolling_icir" in roll_df.columns:
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Bar(
            x=roll_df["date"],
            y=roll_df["ic"],
            name="IC (period)",
            opacity=0.40,
            marker_color="steelblue",
        ))
        fig_roll.add_trace(go.Scatter(
            x=roll_df["date"],
            y=roll_df["rolling_icir"],
            name=f"Rolling ICIR ({roll_window})",
            yaxis="y2",
            line=dict(width=2, color="orange"),
        ))
        fig_roll.add_hline(y=0.5,  line_dash="dash", line_color="green",
                           annotation_text="ICIR = 0.5 (actionable)")
        fig_roll.add_hline(y=-0.5, line_dash="dash", line_color="red")
        fig_roll.update_layout(
            yaxis=dict(title="IC"),
            yaxis2=dict(title="Rolling ICIR", overlaying="y", side="right"),
            height=340,
            margin=dict(t=10),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("Not enough IC periods for rolling ICIR at this configuration.")

    # ── Methodology ───────────────────────────────────────────────────────────
    with st.expander("Methodology & interpretation"):
        st.markdown("""
**Information Coefficient (IC)** — Spearman (or Pearson) rank correlation between
a factor score at date T and the realized forward return at date T + horizon.

| IC range | Interpretation |
|---|---|
| > 0.10 | Strong signal |
| 0.05 – 0.10 | Moderate |
| 0.00 – 0.05 | Weak / noise |
| < 0.00 | Negative (contrarian) |

**ICIR** = `mean(IC) / std(IC)`. ICIR > 0.5 is generally considered actionable.
**Hit rate** — fraction of periods with IC > 0.
**t-stat** — `mean_IC / (std_IC / √n)`. |t| > 2 suggests non-random signal.

**Monotonicity** — A genuine factor shows monotonically increasing mean forward
return from decile 1 (lowest score) to decile 10 (highest).

**Weight recommendations** — Derived from IC and ICIR at 20–60 day horizons.
Negative-IC factors are zero-weighted. High-ICIR factors get a consistency bonus.
These are research suggestions only — apply manually via `config.yaml`.

**Survivorship bias** — The universe contains only currently-listed stocks.
Delisted companies are excluded, which flatters IC for value factors.
Interpret with caution for horizons > 60 days.
        """)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw IC data"):
        if not ic_df.empty:
            st.dataframe(
                ic_df.sort_values(["horizon_days", "factor", "date"]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download IC CSV",
                ic_df.to_csv(index=False),
                "factor_ic.csv",
                "text/csv",
            )


# ---------------------------------------------------------------------------
# Public sub-section renderers — called by pages/research.py
# Each covers one Research tab. Keys use unique prefixes to avoid collision
# with the existing render() function's "fl_*" session-state keys.
# ---------------------------------------------------------------------------


def render_overview_tab(summary: pd.DataFrame, n_dates: int, n_stocks: int) -> None:
    """Research Summary + Weight Recommendations for Research → Overview."""
    conc = _synthesize_conclusions(summary, n_dates)
    _render_research_summary(conc, n_dates)
    st.divider()
    recs = _synthesize_weights(summary)
    _render_weight_recommendations(recs, n_dates=n_dates)


def render_ic_analysis_tab(
    ic_df: pd.DataFrame,
    summary: pd.DataFrame,
    decay: pd.DataFrame,
    sel_factors: list[str],
    sel_horizons: list[int],
    ic_type: str,
) -> None:
    """IC Summary + Decay + Cumulative IC + Rolling ICIR for Research → IC Analysis."""
    n_d = int(ic_df["date"].nunique())
    n_s = int(ic_df["n_stocks"].median())
    m1, m2, m3 = st.columns(3)
    m1.metric("Snapshot dates",         n_d)
    m2.metric("Median universe size",   n_s)
    m3.metric("Factor × horizon pairs", len(summary) if not summary.empty else 0)

    st.divider()

    # IC Summary Table
    st.subheader("IC Summary")
    if not summary.empty:
        disp = summary.copy()
        disp["factor"] = disp["factor"].map(lambda x: _FRIENDLY.get(x, x))
        disp.columns   = [c.replace("_", " ").title() for c in disp.columns]
        num_cols       = disp.select_dtypes("number").columns.tolist()

        def _color_ic(val: object):
            if not isinstance(val, float):
                return ""
            if val > 0.05:
                return "color: #00b300; font-weight: bold"
            if val < -0.05:
                return "color: #cc0000; font-weight: bold"
            return "color: #888888"

        styled   = disp.style.format({c: "{:.4f}" for c in num_cols})
        mean_col = next((c for c in disp.columns if "Mean" in c and "Ic" in c), None)
        if mean_col:
            styled = styled.map(_color_ic, subset=[mean_col])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No IC summary — build more snapshot history.")

    st.divider()

    # Factor Decay Curves
    st.subheader("Factor Decay (IC by Horizon)")
    st.caption("How predictive power fades as forecast horizon increases.")
    if not decay.empty:
        fig_d = go.Figure()
        for factor in decay["factor"].unique():
            grp = decay[decay["factor"] == factor].sort_values("horizon_days")
            fig_d.add_trace(go.Scatter(
                x=grp["horizon_days"], y=grp["mean_ic"],
                mode="lines+markers",
                name=_FRIENDLY.get(factor, factor),
                marker=dict(size=7),
            ))
        fig_d.add_hline(y=0.05,  line_dash="dash", line_color="green", line_width=1)
        fig_d.add_hline(y=-0.05, line_dash="dash", line_color="red",   line_width=1)
        fig_d.add_hline(y=0,     line_dash="solid", line_color="gray",  line_width=0.5)
        fig_d.update_layout(
            xaxis_title="Horizon (days)", yaxis_title="Mean IC",
            height=360, margin=dict(t=10), legend_title="Factor",
        )
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("Decay data unavailable.")

    st.divider()

    # Cumulative IC
    st.subheader("Cumulative IC")
    cum_horizon = st.selectbox(
        "Horizon", sorted(sel_horizons), key="ria_cum_horizon",
    )
    with st.spinner("Computing cumulative IC…"):
        try:
            cum_df = _compute_cumulative_ic(tuple(sel_factors), cum_horizon, ic_type)
        except Exception as exc:
            cum_df = pd.DataFrame()
            st.caption(f"Unavailable: {exc}")
    if not cum_df.empty:
        fig_c = go.Figure()
        for factor in cum_df["factor"].unique():
            grp = cum_df[cum_df["factor"] == factor].sort_values("date")
            fig_c.add_trace(go.Scatter(
                x=grp["date"], y=grp["cumulative_ic"],
                mode="lines", name=_FRIENDLY.get(factor, factor),
            ))
        fig_c.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
        fig_c.update_layout(xaxis_title="Date", yaxis_title="Cumulative IC", height=300, margin=dict(t=10))
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Not enough data for cumulative IC at this horizon.")

    st.divider()

    # Rolling ICIR
    st.subheader("Rolling ICIR")
    st.caption("ICIR = mean(IC) / std(IC) — measures consistency of predictive signal.")
    ra, rb_ = st.columns(2)
    roll_factor  = ra.selectbox("Factor",  sel_factors, key="ria_roll_factor",
                                format_func=lambda x: _FRIENDLY.get(x, x))
    roll_horizon = rb_.selectbox("Horizon", sorted(sel_horizons), key="ria_roll_horizon")
    roll_window  = st.slider("Rolling window (periods)", 3, 24, 8, key="ria_roll_win")
    with st.spinner("Computing rolling ICIR…"):
        try:
            roll_df = _compute_rolling_icir(roll_factor, roll_horizon, roll_window)
        except Exception as exc:
            roll_df = pd.DataFrame()
            st.caption(f"Error: {exc}")
    if not roll_df.empty and "rolling_icir" in roll_df.columns:
        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(
            x=roll_df["date"], y=roll_df["ic"],
            name="IC (period)", opacity=0.40, marker_color="steelblue",
        ))
        fig_r.add_trace(go.Scatter(
            x=roll_df["date"], y=roll_df["rolling_icir"],
            name=f"Rolling ICIR ({roll_window})",
            yaxis="y2", line=dict(width=2, color="orange"),
        ))
        fig_r.add_hline(y=0.5,  line_dash="dash", line_color="green",
                        annotation_text="ICIR = 0.5 (actionable)")
        fig_r.add_hline(y=-0.5, line_dash="dash", line_color="red")
        fig_r.update_layout(
            yaxis=dict(title="IC"),
            yaxis2=dict(title="Rolling ICIR", overlaying="y", side="right"),
            height=320, margin=dict(t=10),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.info("Not enough IC periods for rolling ICIR at this configuration.")

    with st.expander("Raw IC data"):
        if not ic_df.empty:
            st.dataframe(
                ic_df.sort_values(["horizon_days", "factor", "date"]),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Download IC CSV", ic_df.to_csv(index=False), "factor_ic.csv", "text/csv",
            )


def render_decile_tab(sel_factors: list[str], sel_horizons: list[int]) -> None:
    """Decile spread / monotonicity for Research → Rank & Deciles."""
    da, db, dc = st.columns(3)
    dec_factor  = da.selectbox("Factor",  sel_factors,          key="rdt_dec_factor",
                               format_func=lambda x: _FRIENDLY.get(x, x))
    dec_horizon = db.selectbox("Horizon", sorted(sel_horizons), key="rdt_dec_horizon")
    n_deciles   = dc.selectbox("Deciles", [5, 10], index=1,    key="rdt_n_deciles")

    with st.spinner("Computing decile spread…"):
        try:
            decile_df = _compute_decile_spread(dec_factor, dec_horizon, n_deciles)
        except Exception as exc:
            decile_df = pd.DataFrame()
            st.caption(f"Error: {exc}")

    if not decile_df.empty:
        colors = ["#00b300" if r >= 0 else "#cc0000" for r in decile_df["mean_forward_return"]]
        fig_dec = go.Figure(go.Bar(
            x=decile_df["decile"],
            y=(decile_df["mean_forward_return"] * 100).round(2),
            marker_color=colors,
            text=[f"{r:.1%}" for r in decile_df["mean_forward_return"]],
            textposition="outside",
        ))
        fig_dec.add_hline(y=0, line_dash="solid", line_color="gray")
        fig_dec.update_layout(
            xaxis_title=f"Decile (1 = lowest {_FRIENDLY.get(dec_factor, dec_factor)}, {n_deciles} = highest)",
            yaxis_title="Mean Forward Return (%)",
            height=320, margin=dict(t=10),
        )
        st.plotly_chart(fig_dec, use_container_width=True)
        if len(decile_df) >= 2:
            top_r = decile_df.loc[decile_df["decile"].idxmax(), "mean_forward_return"]
            bot_r = decile_df.loc[decile_df["decile"].idxmin(), "mean_forward_return"]
            st.caption(f"Top-minus-bottom spread: **{top_r - bot_r:.2%}**")
    else:
        st.info("Not enough data for decile analysis — build more snapshot history.")


def render_regime_tab(sel_factors: list[str], sel_horizons: list[int], ic_type: str) -> None:
    """Regime-conditioned IC heatmap for Research → Regime Analysis."""
    regime_horizon = st.selectbox(
        "Horizon for regime IC", sorted(sel_horizons), key="rrt_regime_horizon",
        help="Forward-return horizon for regime IC computation.",
    )
    with st.spinner("Fetching regime history and computing regime-conditioned IC…"):
        try:
            regime_df = _compute_regime_ic(
                tuple(f for f in sel_factors if f != "value_metric"),
                regime_horizon, ic_type,
            )
        except Exception as exc:
            regime_df = pd.DataFrame()
            st.caption(f"Regime IC unavailable: {exc}")
    _render_regime_ic(regime_df, sel_factors)
