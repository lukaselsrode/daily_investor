"""
ui/components/cluster_diagnostics.py — Walk-forward cluster concentration charts.

Renders timeline of max cluster weight, stacked cluster composition, and
violation summary from a ClusterTrackingResult.
"""
from __future__ import annotations

import streamlit as st


def render_cluster_diagnostics(cluster_result) -> None:
    """
    Render cluster concentration diagnostics from a ClusterTrackingResult.
    """
    if cluster_result is None or not cluster_result.snapshots:
        st.info("Cluster concentration data not available. Run backtest with cluster tracking enabled.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("plotly not installed — install with `pip install plotly`.")
        _text_fallback(cluster_result)
        return

    snapshots = cluster_result.snapshots
    days      = [s.day for s in snapshots]
    max_wts   = [s.max_cluster_weight for s in snapshots]
    n_held    = [s.n_held for s in snapshots]
    n_clusters = cluster_result.n_clusters

    # ── Summary cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg max cluster weight", f"{cluster_result.avg_max_cluster_weight:.1%}")
    c2.metric("Worst max cluster weight", f"{cluster_result.worst_max_cluster_weight:.1%}")
    c3.metric("Violation days", cluster_result.n_violation_days)
    c4.metric("Snapshots", len(snapshots))

    # ── Max cluster weight timeline ───────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=[w * 100 for w in max_wts],
        mode="lines", name="Max cluster weight %",
        line=dict(color="#4c8ef5", width=2),
        hovertemplate="Day %{x}<br>Max cluster: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=35.0, line_dash="dot", line_color="orange",
        annotation_text="35% threshold", annotation_position="right",
    )
    # Shade violations
    violation_days = [s.day for s in snapshots if s.violation]
    if violation_days:
        for vd in violation_days:
            fig.add_vrect(
                x0=vd - 2, x1=vd + 2,
                fillcolor="red", opacity=0.10,
                layer="below", line_width=0,
            )
    fig.update_layout(
        title="Max Cluster Weight Over Time",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Max cluster weight (%)",
        xaxis_title="Trading day",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Stacked cluster weight composition ────────────────────────────────
    if cluster_result.cluster_timeline is not None:
        timeline = cluster_result.cluster_timeline
        cluster_colors = [
            "#4c8ef5", "#f5a623", "#7ed321", "#d0021b", "#9b59b6", "#aaaaaa",
            "#1abc9c", "#e67e22", "#34495e", "#f39c12",
        ]
        fig2 = go.Figure()
        for cid in range(min(n_clusters, timeline.shape[1])):
            fig2.add_trace(go.Bar(
                x=days,
                y=[timeline[ri, cid] * 100 for ri in range(len(snapshots))],
                name=f"Cluster {cid}",
                marker_color=cluster_colors[cid % len(cluster_colors)],
                hovertemplate=f"Cluster {cid}<br>Day %{{x}}<br>Weight: %{{y:.1f}}%<extra></extra>",
            ))
        fig2.add_hline(y=35.0, line_dash="dot", line_color="orange", opacity=0.5)
        fig2.update_layout(
            barmode="stack",
            title="Cluster Composition of Active Sleeve",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Weight (%)",
            xaxis_title="Trading day",
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Violation table ────────────────────────────────────────────────────
    if cluster_result.n_violation_days > 0:
        import pandas as pd
        rows = []
        for s in snapshots:
            if s.violation:
                top_cluster = max(s.cluster_weights, key=lambda k: s.cluster_weights[k], default=-1)
                rows.append({
                    "Day":              s.day,
                    "Max cluster weight": f"{s.max_cluster_weight:.1%}",
                    "Top cluster ID":   top_cluster,
                    "Top cluster wt":   f"{s.cluster_weights.get(top_cluster, 0):.1%}",
                    "Positions held":   s.n_held,
                })
        with st.expander(f"Violation details ({cluster_result.n_violation_days} days)"):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _text_fallback(cluster_result) -> None:
    st.write(f"Avg max cluster weight: {cluster_result.avg_max_cluster_weight:.1%}")
    st.write(f"Worst max cluster weight: {cluster_result.worst_max_cluster_weight:.1%}")
    st.write(f"Violation days: {cluster_result.n_violation_days}")
