"""
ui/components/distribution_intelligence.py — Distribution Intelligence workspace.

Investigates whether the bimodal score distribution contains predictive information
and whether portfolio construction should become threshold-based rather than rank-based.

Subtabs:
  1. Distribution Shape — bimodality test, histogram, stats
  2. Tail Analysis     — bucket returns, monotonicity, alpha concentration
  3. Local IC          — nonlinear predictive power along the score axis
  4. Regime Clusters   — GMM/k-means cluster analysis
  5. Conditional Alpha — factor interactions and conditional IC
  6. Threshold Sim     — compare rank-based vs threshold-gated selection
  7. Confidence Engine — dynamic factor confidence from IC history
  8. Evolution         — distribution shape drift across snapshots
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from ui.utils import data_date, load_latest_csv, no_data_msg, fmt_bin_index

_SCORE_COLS = ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]
_RETURN_COLS = ["return_1m", "return_3m", "return_6m"]
_FRIENDLY = {
    "value_score":    "Value",
    "quality_score":  "Quality",
    "income_score":   "Income",
    "momentum_score": "Momentum",
    "value_metric":   "Composite",
    "return_1m":      "1m return",
    "return_3m":      "3m return",
    "return_6m":      "6m return",
}


# ---------------------------------------------------------------------------
# Cached backend helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _get_analyzer(_df_hash, df_json: str):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df)


@st.cache_data(ttl=600)
def _cached_bimodality(df_json: str, score_col: str):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).test_bimodality(score_col)


@st.cache_data(ttl=600)
def _cached_tail_buckets(df_json: str, score_col: str, return_col: str):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    ana = DistributionAnalyzer(df)
    buckets = ana.compute_tail_buckets(score_col, return_col)
    return ana.buckets_to_df(buckets)


@st.cache_data(ttl=600)
def _cached_monotonicity(df_json: str, score_col: str, return_col: str, n_deciles: int):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).compute_monotonicity(score_col, return_col, n_deciles)


@st.cache_data(ttl=600)
def _cached_local_ic(df_json: str, score_col: str, return_col: str, window_pct: float, step_pct: float):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).compute_local_ic(score_col, return_col, window_pct, step_pct)


@st.cache_data(ttl=600)
def _cached_clusters(df_json: str, features_key: str, features: list, n_clusters: int,
                     method: str, return_col: str):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).compute_clusters(features, n_clusters, method, return_col)


@st.cache_data(ttl=600)
def _cached_conditional_ic(df_json: str, primary: str, cond: str, return_col: str, n_q: int):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).compute_conditional_ic(primary, cond, return_col, n_q)


@st.cache_data(ttl=600)
def _cached_interaction_matrix(df_json: str, score_cols_key: str, score_cols: list, return_col: str):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).compute_interaction_matrix(score_cols, return_col)


@st.cache_data(ttl=600)
def _cached_threshold_sim(df_json: str, score_col: str, return_col: str):
    import io
    from research.distribution_regime_analysis import DistributionAnalyzer
    df = pd.read_json(io.StringIO(df_json))
    return DistributionAnalyzer(df).simulate_threshold_modes(score_col, return_col)


@st.cache_data(ttl=3600)
def _cached_ic_history(factors_key: str, factors: tuple, horizon: int) -> pd.DataFrame:
    from strategy.research.ic_engine import FactorResearchEngine
    engine = FactorResearchEngine(factors=list(factors))
    return engine.compute_multi_horizon_ic(factors=list(factors), horizons=[horizon])


@st.cache_data(ttl=3600)
def _cached_distribution_evolution(score_col: str) -> pd.DataFrame:
    from research.distribution_regime_analysis import DistributionAnalyzer
    return DistributionAnalyzer.compute_distribution_evolution(score_col)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _try_plotly_bar(df: pd.DataFrame, x: str, y: str, title: str,
                    color: str | None = None, color_scale: str = "RdYlGn") -> None:
    try:
        import plotly.express as px
        kwargs = dict(x=x, y=y, title=title, labels={x: x, y: y})
        if color:
            kwargs["color"] = color
            kwargs["color_continuous_scale"] = color_scale
        fig = px.bar(df, **kwargs)
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df.set_index(x)[y])


def _try_plotly_line(df: pd.DataFrame, x: str, y: str | list, title: str) -> None:
    try:
        import plotly.express as px
        if isinstance(y, list):
            fig = px.line(df, x=x, y=y, title=title)
        else:
            fig = px.line(df, x=x, y=y, title=title)
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        cols = y if isinstance(y, list) else [y]
        st.line_chart(df.set_index(x)[cols])


def _try_plotly_scatter(df: pd.DataFrame, x: str, y: str, title: str,
                        color: str | None = None) -> None:
    try:
        import plotly.express as px
        kwargs = dict(x=x, y=y, title=title)
        if color:
            kwargs["color"] = color
        fig = px.scatter(df, **kwargs)
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(df.set_index(x)[[y]])


def _try_plotly_heatmap(matrix_df: pd.DataFrame, title: str) -> None:
    try:
        import plotly.express as px
        fig = px.imshow(
            matrix_df.astype(float),
            text_auto=".3f",
            color_continuous_scale="RdBu_r",
            zmin=-0.15, zmax=0.15,
            title=title,
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.dataframe(matrix_df.style.background_gradient(cmap="RdBu", vmin=-0.15, vmax=0.15),
                     use_container_width=True)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _tab_distribution(df: pd.DataFrame, df_json: str, score_col: str, return_col: str) -> None:
    st.subheader("Score distribution shape")
    st.caption(
        "Tests whether the composite score distribution is bimodal — "
        "indicating the universe may separate into structurally attractive vs. unattractive regimes."
    )

    from research.distribution_regime_analysis import DistributionAnalyzer
    ana = DistributionAnalyzer(df)

    s = pd.to_numeric(df[score_col], errors="coerce").dropna()
    if s.empty:
        st.warning(f"No data for `{score_col}`.")
        return

    # ── Distribution stats ───────────────────────────────────────────────────
    stats_dict = ana.distribution_stats(score_col)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("N", stats_dict.get("n", "—"))
    c2.metric("Mean", f"{stats_dict.get('mean', 0):.4f}")
    c3.metric("Std",  f"{stats_dict.get('std', 0):.4f}")
    c4.metric("Skew", f"{stats_dict.get('skew', 0):.3f}")
    c5.metric("Kurt", f"{stats_dict.get('kurt', 0):.3f}")

    # ── Histogram ───────────────────────────────────────────────────────────
    st.caption(f"{score_col} histogram (30 bins)")
    try:
        import plotly.express as px
        fig = px.histogram(s, nbins=40, title=f"{score_col} distribution",
                           labels={"value": score_col, "count": "count"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(fmt_bin_index(s.value_counts(bins=30, sort=False).sort_index()))

    st.divider()

    # ── Bimodality test ──────────────────────────────────────────────────────
    st.subheader("Bimodality test")
    with st.spinner("Running bimodality test…"):
        try:
            bm = _cached_bimodality(df_json, score_col)
        except Exception as exc:
            st.error(f"Bimodality test failed: {exc}")
            return

    col_a, col_b = st.columns(2)
    with col_a:
        bc = bm.bimodality_coeff
        st.metric(
            "Bimodality coefficient",
            f"{bc:.4f}",
            delta="BIMODAL" if bm.is_bimodal else "unimodal",
            delta_color="inverse" if bm.is_bimodal else "off",
            help="BC > 0.555 suggests bimodal. BC = (skew² + 1) / (kurtosis + 3).",
        )
        st.metric("Skewness", f"{bm.skewness:.4f}")
        st.metric("Excess kurtosis", f"{bm.excess_kurtosis:.4f}")

    with col_b:
        if bm.gmm_bic_k1 < float("inf"):
            st.metric(
                "GMM: BIC k=1 vs k=2",
                f"{bm.gmm_bic_k1:.0f} vs {bm.gmm_bic_k2:.0f}",
                delta="k=2 preferred (bimodal)" if bm.gmm_favors_bimodal else "k=1 preferred (unimodal)",
                delta_color="inverse" if bm.gmm_favors_bimodal else "off",
                help="Lower BIC = better fit. k=2 preferred → GMM evidence of bimodal.",
            )
            if bm.gmm_means:
                st.metric("Cluster separation", f"{bm.separation_score:.3f}",
                          help="|mean1 − mean2| / pooled std. > 1.0 = well-separated clusters.")
                st.caption(
                    f"Cluster means: {bm.gmm_means[0]:.4f} (weight {bm.gmm_weights[0]:.2%}) | "
                    f"{bm.gmm_means[1]:.4f} (weight {bm.gmm_weights[1]:.2%})"
                )
        else:
            st.info("GMM test unavailable (install scikit-learn to enable).")

    # Interpretation
    if bm.is_bimodal or bm.gmm_favors_bimodal:
        st.warning(
            "⚠️ Distribution shows bimodal evidence. "
            "The universe may be separating into structurally attractive vs. unattractive regimes. "
            "Investigate Tail Analysis and Threshold Simulation tabs."
        )
    else:
        st.success("Distribution appears approximately unimodal. Continuous ranking is appropriate.")


def _tab_tail_analysis(df: pd.DataFrame, df_json: str, score_col: str, return_col: str) -> None:
    st.subheader("Tail vs. middle alpha concentration")
    st.caption(
        "If alpha concentrates in the tails and the middle is noise, "
        "the ranking should become threshold-based rather than smooth percentile ranking."
    )

    # ── Bucket stats ─────────────────────────────────────────────────────────
    with st.spinner("Computing tail buckets…"):
        bucket_df = _cached_tail_buckets(df_json, score_col, return_col)

    if bucket_df.empty:
        st.info("Not enough data for tail bucket analysis (need ≥ 30 rows with score + return data).")
        return

    st.subheader("Bucket return statistics")
    _try_plotly_bar(
        bucket_df, x="bucket", y="mean_return",
        title=f"Mean {return_col} by score bucket",
        color="mean_return",
    )
    _try_plotly_bar(
        bucket_df, x="bucket", y="hit_rate",
        title="Hit rate by score bucket",
        color="hit_rate",
    )
    st.dataframe(
        bucket_df.style.background_gradient(
            subset=["mean_return", "hit_rate", "sharpe_proxy"], cmap="RdYlGn"
        ),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Monotonicity ─────────────────────────────────────────────────────────
    n_deciles = st.slider("Deciles", min_value=5, max_value=20, value=10, step=1, key="di_deciles")
    st.subheader("Monotonicity — decile return spread")
    st.caption("Perfect predictive power = monotonically increasing returns from D1 → D10.")

    with st.spinner("Computing decile monotonicity…"):
        mono_df = _cached_monotonicity(df_json, score_col, return_col, n_deciles)

    if mono_df.empty:
        st.info("Not enough data for decile analysis.")
        return

    tau = mono_df.attrs.get("kendall_tau", None)
    tau_p = mono_df.attrs.get("kendall_p", None)
    if tau is not None:
        c1, c2 = st.columns(2)
        c1.metric(
            "Kendall τ (monotonicity)",
            f"{tau:.4f}",
            help="τ = 1: perfectly monotone. τ = 0: no rank correlation. τ = -1: perfectly inverse.",
        )
        if tau_p is not None:
            c2.metric("τ p-value", f"{tau_p:.4f}",
                      delta="significant" if tau_p < 0.05 else "not significant",
                      delta_color="off" if tau_p >= 0.05 else "normal")

    _try_plotly_bar(mono_df, x="decile", y="mean_return",
                    title=f"Mean {return_col} by {score_col} decile",
                    color="mean_return")
    st.dataframe(
        mono_df.style.background_gradient(
            subset=["mean_return", "hit_rate", "sharpe_proxy"], cmap="RdYlGn"
        ),
        use_container_width=True, hide_index=True,
    )


def _tab_local_ic(df: pd.DataFrame, df_json: str, score_col: str, return_col: str) -> None:
    st.subheader("Local IC — nonlinear predictive power")
    st.caption(
        "IC computed in sliding windows along the sorted score distribution. "
        "High IC in tails + near-zero IC in center = threshold-based alpha structure."
    )

    c1, c2 = st.columns(2)
    with c1:
        window_pct = st.slider("Window size (% of universe)", 10, 40, 20, 5, key="di_window_pct") / 100
    with c2:
        step_pct = st.slider("Step size (% of universe)", 2, 15, 5, 1, key="di_step_pct") / 100

    with st.spinner("Computing local IC…"):
        local_ic_df = _cached_local_ic(df_json, score_col, return_col, window_pct, step_pct)

    if local_ic_df.empty:
        st.info("Not enough data for local IC (need ≥ 40 rows).")
        return

    # Main chart: local IC by center percentile
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=local_ic_df["center_pct"],
            y=local_ic_df["local_ic"],
            mode="lines+markers",
            name="Local IC",
            line=dict(color="#4A9EDB", width=2),
            marker=dict(size=5),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        # Highlight significant windows
        sig_df = local_ic_df[local_ic_df["p_value"] < 0.05]
        if not sig_df.empty:
            fig.add_trace(go.Scatter(
                x=sig_df["center_pct"],
                y=sig_df["local_ic"],
                mode="markers",
                name="Significant (p<0.05)",
                marker=dict(size=9, color="#F5A623", symbol="star"),
            ))
        fig.update_layout(
            title=f"Local Spearman IC: {score_col} → {return_col}",
            xaxis_title="Score percentile (0=bottom, 1=top)",
            yaxis_title="Local IC",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        _try_plotly_line(local_ic_df, x="center_pct", y="local_ic",
                         title="Local IC by score percentile")

    # Interpretation
    tail_ic = float(local_ic_df[local_ic_df["center_pct"] > 0.85]["local_ic"].mean()) if not local_ic_df.empty else 0
    mid_ic  = float(local_ic_df[(local_ic_df["center_pct"] >= 0.4) & (local_ic_df["center_pct"] <= 0.6)]["local_ic"].mean()) if not local_ic_df.empty else 0
    if not np.isnan(tail_ic) and not np.isnan(mid_ic):
        c1, c2 = st.columns(2)
        c1.metric("Avg IC — top 15% tail", f"{tail_ic:.4f}")
        c2.metric("Avg IC — middle 40-60%", f"{mid_ic:.4f}")
        if abs(tail_ic) > 2 * abs(mid_ic) + 0.01:
            st.warning(
                "⚠️ Tail IC is substantially stronger than mid-range IC. "
                "Alpha appears to concentrate in the extremes — threshold-based selection may outperform."
            )

    with st.expander("Raw local IC data"):
        st.dataframe(local_ic_df, use_container_width=True, hide_index=True)


def _tab_regime_clusters(df: pd.DataFrame, df_json: str, return_col: str) -> None:
    st.subheader("Regime clustering")
    st.caption(
        "Cluster stocks by factor scores using GMM or k-means. "
        "If clusters predict different forward returns, the universe has structural regimes."
    )

    available_features = [c for c in _SCORE_COLS[:-1] if c in df.columns]  # exclude composite
    if not available_features:
        st.info("No factor score columns found in dataset.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        features = st.multiselect("Clustering features", available_features,
                                  default=available_features[:3], key="di_cluster_features")
    with c2:
        n_clusters = st.slider("Number of clusters (k)", 2, 5, 2, key="di_n_clusters")
    with c3:
        method = st.selectbox("Method", ["gmm", "kmeans"], key="di_cluster_method")

    if not features:
        st.info("Select at least one feature.")
        return

    with st.spinner(f"Running {method} clustering with k={n_clusters}…"):
        cluster_summary = _cached_clusters(
            df_json, ",".join(features), features, n_clusters, method, return_col
        )

    if cluster_summary.empty:
        st.info("Clustering failed — not enough data or sklearn unavailable.")
        return

    st.subheader("Cluster summary")
    _try_plotly_bar(cluster_summary.rename(columns={"cluster": "Cluster"}),
                    x="Cluster", y="mean_return",
                    title=f"Mean {return_col} by cluster",
                    color="mean_return")
    st.dataframe(
        cluster_summary.style.background_gradient(
            subset=[c for c in ["mean_return", "hit_rate", "sharpe_proxy"] if c in cluster_summary.columns],
            cmap="RdYlGn",
        ),
        use_container_width=True, hide_index=True,
    )

    # Cluster interpretation
    if len(cluster_summary) == 2 and "mean_return" in cluster_summary.columns:
        r_diff = float(cluster_summary["mean_return"].max() - cluster_summary["mean_return"].min())
        if r_diff > 0.005:
            st.info(
                f"The two clusters differ in mean {return_col} by {r_diff:.2%}. "
                "If this persists across snapshots, cluster membership may be a useful participation filter."
            )


def _tab_conditional_alpha(df: pd.DataFrame, df_json: str, return_col: str) -> None:
    st.subheader("Conditional alpha — factor interactions")
    st.caption(
        "IC of a primary factor within each quartile of a conditioning factor. "
        "Reveals hidden conditional alpha: value may only work inside high-quality names."
    )

    available = [c for c in _SCORE_COLS[:-1] if c in df.columns]
    if len(available) < 2:
        st.info("Need at least two factor score columns.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        primary = st.selectbox("Primary factor", available,
                               index=0, format_func=lambda x: _FRIENDLY.get(x, x),
                               key="di_primary")
    with c2:
        cond_options = [c for c in available if c != primary]
        cond = st.selectbox("Conditioning factor", cond_options,
                            format_func=lambda x: _FRIENDLY.get(x, x),
                            key="di_cond")
    with c3:
        n_q = st.slider("Quartiles", 2, 6, 4, key="di_nq")

    st.caption(f"IC of **{_FRIENDLY.get(primary, primary)}** within each quartile of **{_FRIENDLY.get(cond, cond)}**")

    with st.spinner("Computing conditional IC…"):
        cic_df = _cached_conditional_ic(df_json, primary, cond, return_col, n_q)

    if cic_df.empty:
        st.info("Not enough data for conditional IC (need ≥ 15 stocks per quartile).")
        return

    _try_plotly_bar(cic_df, x="cond_label", y="ic",
                    title=f"IC of {_FRIENDLY.get(primary, primary)} by {_FRIENDLY.get(cond, cond)} quartile",
                    color="ic")
    st.dataframe(
        cic_df.style.background_gradient(subset=["ic"], cmap="RdYlGn", vmin=-0.2, vmax=0.2),
        use_container_width=True, hide_index=True,
    )

    # Summary insight
    sig_q = cic_df[cic_df["significant"]]
    if not sig_q.empty:
        best = cic_df.loc[cic_df["ic"].abs().idxmax()]
        st.info(
            f"Strongest conditional IC: **{_FRIENDLY.get(primary, primary)}** in "
            f"**{best['cond_label']}** of {_FRIENDLY.get(cond, cond)} → IC = {best['ic']:.4f}"
        )

    st.divider()

    # ── Interaction matrix ────────────────────────────────────────────────────
    st.subheader("Interaction matrix — top-quartile conditional IC")
    st.caption(
        "Entry [row, col] = IC of row factor when col factor is in its top quartile. "
        "Large off-diagonal values = strong conditional factor interaction."
    )

    with st.spinner("Building interaction matrix…"):
        matrix_df = _cached_interaction_matrix(df_json, ",".join(available), available, return_col)

    if not matrix_df.empty:
        _try_plotly_heatmap(matrix_df, "Conditional IC matrix (primary × conditioning factor)")
        with st.expander("Raw interaction matrix"):
            st.dataframe(matrix_df, use_container_width=True)


def _tab_threshold_simulation(df: pd.DataFrame, df_json: str,
                              score_col: str, return_col: str) -> None:
    st.subheader("Threshold simulation — rank-based vs threshold-gated")
    st.caption(
        "Compare portfolio statistics when restricting to names above a score threshold "
        "vs. using the full ranking. If threshold-gating improves hit rate and Sharpe without "
        "dramatically reducing coverage, threshold-based construction may be superior."
    )

    with st.spinner("Simulating threshold modes…"):
        sim_df = _cached_threshold_sim(df_json, score_col, return_col)

    if sim_df.empty:
        st.info("Not enough data for threshold simulation.")
        return

    st.subheader("Selection metrics by threshold")
    _try_plotly_bar(sim_df, x="mode", y="mean_return",
                    title=f"Mean {return_col} by threshold mode", color="mean_return")
    _try_plotly_bar(sim_df, x="mode", y="hit_rate",
                    title="Hit rate by threshold mode", color="hit_rate")
    _try_plotly_bar(sim_df, x="mode", y="pct_universe",
                    title="% of universe selected by threshold", color="pct_universe",
                    color_scale="Blues")

    display_df = sim_df.copy()
    st.dataframe(
        display_df.style.background_gradient(
            subset=["mean_return", "hit_rate", "sharpe_proxy"], cmap="RdYlGn"
        ),
        use_container_width=True, hide_index=True,
    )

    # Summary: find best threshold by hit rate with >5% universe coverage
    viable = sim_df[(sim_df["threshold"].notna()) & (sim_df["pct_universe"] >= 0.05)]
    if not viable.empty:
        best = viable.loc[viable["sharpe_proxy"].idxmax()]
        st.info(
            f"Best viable threshold by Sharpe proxy: **{best['mode']}** — "
            f"selects {best['pct_universe']:.1%} of universe, "
            f"mean return {best['mean_return']:.2%}, "
            f"hit rate {best['hit_rate']:.1%}"
        )

    with st.expander("Methodology note"):
        st.markdown(
            "**Threshold simulation** uses the current snapshot's historical returns (`return_1m` etc.) "
            "as a proxy for forward returns. This is NOT forward-looking — it describes how past returns "
            "were distributed across the current score spectrum. True forward-return validation requires "
            "multiple sequential snapshots (see IC Analysis tab in the main Research page)."
        )


def _tab_confidence_engine(return_col: str) -> None:
    st.subheader("Factor confidence engine")
    st.caption(
        "Dynamic factor confidence derived from historical IC data (snapshot store). "
        "Higher confidence = factor is consistently predictive. "
        "Use to identify which factors deserve more weight vs. which are degrading."
    )

    all_factors = ["value_score", "quality_score", "income_score", "momentum_score"]
    c1, c2 = st.columns(2)
    with c1:
        factors = st.multiselect("Factors", all_factors, default=all_factors, key="di_conf_factors")
    with c2:
        horizon = st.selectbox("IC horizon (days)", [5, 20, 60, 120], index=1, key="di_conf_horizon")

    if not factors:
        st.info("Select at least one factor.")
        return

    with st.spinner("Loading historical IC data…"):
        try:
            ic_df = _cached_ic_history(",".join(sorted(factors)), tuple(sorted(factors)), horizon)
        except Exception as exc:
            st.error(f"Could not load IC history: {exc}")
            return

    if ic_df.empty:
        st.info(
            "No IC history available. "
            "Need ≥ 2 snapshot files in `data/snapshots/`. Run the bot on multiple days."
        )
        return

    from research.distribution_regime_analysis import DistributionAnalyzer
    conf_df = DistributionAnalyzer.compute_factor_confidence(ic_df)

    if conf_df.empty:
        st.info("Not enough IC history to compute confidence.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        _try_plotly_bar(conf_df, x="factor", y="confidence",
                        title="Factor confidence score (0=low, 1=high)",
                        color="confidence", color_scale="RdYlGn")
    with c2:
        st.dataframe(conf_df, use_container_width=True, hide_index=True)

    # Suggested weight adjustments
    st.subheader("Suggested weight adjustments")
    for _, row in conf_df.iterrows():
        adj = float(row["weight_adj"])
        direction = "increase" if adj > 0 else ("decrease" if adj < 0 else "maintain")
        st.markdown(
            f"**{_FRIENDLY.get(row['factor'], row['factor'])}** — "
            f"confidence {row['confidence']:.3f} | "
            f"ICIR {row['icir']:.3f} | "
            f"hit rate {row['hit_rate']:.1%} | "
            f"→ **{direction}** weight ({adj:+.1%})"
        )

    with st.expander("Methodology"):
        st.markdown(
            "Confidence = 0.40 × ICIR_score + 0.35 × hit_rate + 0.25 × direction_score. "
            "ICIR_score normalizes ICIR to [0,1] around zero. "
            "Weight adjustments are suggestions only — verify before changing config."
        )


def _tab_evolution(score_col: str) -> None:
    st.subheader("Distribution evolution over time")
    st.caption(
        "Track bimodality coefficient, skew, and kurtosis across snapshot history. "
        "Rising bimodality coefficient = the universe is separating into increasingly distinct regimes."
    )

    with st.spinner("Loading snapshot history…"):
        evo_df = _cached_distribution_evolution(score_col)

    if evo_df.empty:
        st.info(
            "No multi-snapshot history available. "
            "Need ≥ 2 snapshot files in `data/snapshots/`."
        )
        return

    evo_df["date"] = pd.to_datetime(evo_df["date"])

    _try_plotly_line(evo_df, x="date",
                     y=["bimodality_coeff"],
                     title=f"Bimodality coefficient over time ({score_col})")
    st.caption("Dashed line at 0.555 = bimodality threshold.")

    _try_plotly_line(evo_df, x="date", y=["mean", "std"],
                     title="Distribution mean and std over time")
    _try_plotly_line(evo_df, x="date", y=["skew", "excess_kurtosis"],
                     title="Skewness and kurtosis over time")
    _try_plotly_line(evo_df, x="date", y=["tail_spread"],
                     title="Tail spread (p90 − p10) over time")

    with st.expander("Raw data"):
        st.dataframe(evo_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("🧬 Distribution Intelligence")
    st.caption(
        "Investigates whether the bimodal score distribution contains predictive information "
        "and whether portfolio construction should become threshold-based. "
        "Research only — no config writes, no orders."
    )

    df = load_latest_csv("agg_data")
    if df is None:
        st.warning(no_data_msg("agg_data"))
        return

    # Coerce numerics
    for col in _SCORE_COLS + _RETURN_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_scores = [c for c in _SCORE_COLS if c in df.columns and df[c].notna().sum() >= 10]
    available_returns = [c for c in _RETURN_COLS if c in df.columns and df[c].notna().sum() >= 10]

    if not available_scores:
        st.warning("No score columns found in the latest agg_data snapshot.")
        return

    st.caption(f"Source: agg_data {data_date('agg_data')} | {len(df)} symbols")

    # ── Global controls ──────────────────────────────────────────────────────
    with st.expander("Controls", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            score_col = st.selectbox(
                "Score column",
                available_scores,
                index=available_scores.index("value_metric") if "value_metric" in available_scores else 0,
                format_func=lambda x: _FRIENDLY.get(x, x),
                key="di_score_col",
            )
        with c2:
            return_col = st.selectbox(
                "Return column (proxy for forward returns)",
                available_returns,
                format_func=lambda x: _FRIENDLY.get(x, x),
                key="di_return_col",
            ) if available_returns else None

    if return_col is None:
        st.warning("No return columns available. Cannot compute predictive analytics.")
        return

    # Serialize df for cache keys (use JSON for hashability)
    try:
        df_json = df[available_scores + available_returns + (["sector"] if "sector" in df.columns else [])].to_json()
    except Exception:
        df_json = df.to_json()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Distribution",
        "🎯 Tail Analysis",
        "📡 Local IC",
        "🔵 Regime Clusters",
        "🔗 Conditional Alpha",
        "⚖️ Threshold Sim",
        "💡 Confidence Engine",
        "📈 Evolution",
    ])

    with tabs[0]:
        _tab_distribution(df, df_json, score_col, return_col)

    with tabs[1]:
        _tab_tail_analysis(df, df_json, score_col, return_col)

    with tabs[2]:
        _tab_local_ic(df, df_json, score_col, return_col)

    with tabs[3]:
        _tab_regime_clusters(df, df_json, return_col)

    with tabs[4]:
        _tab_conditional_alpha(df, df_json, return_col)

    with tabs[5]:
        _tab_threshold_simulation(df, df_json, score_col, return_col)

    with tabs[6]:
        _tab_confidence_engine(return_col)

    with tabs[7]:
        _tab_evolution(score_col)
