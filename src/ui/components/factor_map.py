"""
ui/components/factor_map.py — Portfolio vs Universe factor lens.

Two tabs:
  Portfolio Lens — answers the three questions fast:
    1. How do owned stocks compare to the universe?
    2. How do candidates compare to the universe?
    3. What are their factor characteristics?
  Factor Map 3D  — PCA/UMAP 3D scatter for cluster analysis.

SAFE: read-only.  Never modifies config, factor scores, or portfolio state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _load_universe() -> tuple[pd.DataFrame, str]:
    try:
        from portfolio.visualization.factor_map import load_universe_with_holdings
        df = load_universe_with_holdings()
        n_owned = int(df["owned"].astype(bool).sum()) if "owned" in df.columns else 0
        return df, f"{len(df):,} symbols  ·  {n_owned} owned"
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def _add_diversification_scores(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """
    Compute diversification metrics in raw (non-reduced) factor space.
    No PCA required — uses all available factor columns simultaneously.

    Adds: dist_from_centroid, norm_dist_from_centroid, diversification_score,
          nearest_owned.
    """
    df = df.copy()
    factor_cols = [f for f in factors if f in df.columns]
    if not factor_cols or "_role" not in df.columns:
        return df

    feat = df[factor_cols].apply(pd.to_numeric, errors="coerce")
    feat = feat.fillna(feat.median())
    mu, sd = feat.mean(), feat.std().replace(0, 1.0)
    feat_std = ((feat - mu) / sd).values  # (n_total, n_factors)

    owned_mask = (df["_role"] == "owned").values
    owned_feat = feat_std[owned_mask]
    if len(owned_feat) == 0:
        return df

    owned_centroid = owned_feat.mean(axis=0)  # (n_factors,)

    dists = np.linalg.norm(feat_std - owned_centroid, axis=1)
    df["dist_from_centroid"] = dists

    d_min, d_max = dists.min(), dists.max()
    norm_dist = (dists - d_min) / (d_max - d_min + 1e-9)
    df["norm_dist_from_centroid"] = norm_dist

    if "value_metric" in df.columns:
        vm = pd.to_numeric(df["value_metric"], errors="coerce").fillna(0.0).values
        df["diversification_score"] = vm * (0.70 + 0.30 * norm_dist)

    # Nearest owned symbol — vectorised broadcast
    if "symbol" in df.columns and len(owned_feat) > 0:
        owned_syms = df.loc[df["_role"] == "owned", "symbol"].values
        diff = feat_std[:, np.newaxis, :] - owned_feat[np.newaxis, :, :]
        dists_to_owned = np.linalg.norm(diff, axis=2)      # (n_total, n_owned)
        nearest_idx = dists_to_owned.argmin(axis=1)
        nearest_syms = owned_syms[nearest_idx].astype(str)
        nearest_syms[owned_mask] = "—"
        df["nearest_owned"] = nearest_syms

    return df


def _classify(df: pd.DataFrame, metric_threshold: float) -> pd.DataFrame:
    df = df.copy()
    owned_mask = df["owned"].astype(bool) if "owned" in df.columns else pd.Series(False, index=df.index)

    cand_mask = pd.Series(False, index=df.index)
    if "strategy_bucket" in df.columns and "value_metric" in df.columns:
        vm = pd.to_numeric(df["value_metric"], errors="coerce").fillna(-999)
        cand_mask = (
            (df["strategy_bucket"] == "core_candidate")
            & (vm >= metric_threshold)
            & ~owned_mask
        )

    df["_role"] = "universe"
    df.loc[cand_mask,  "_role"] = "candidate"
    df.loc[owned_mask, "_role"] = "owned"
    return df


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_FACTOR_COLS = [
    "value_score", "quality_score", "momentum_score",
    "income_score", "value_metric", "reliability_score",
]
_FACTOR_LABELS = {
    "value_score":       "Value",
    "quality_score":     "Quality",
    "momentum_score":    "Momentum",
    "income_score":      "Income",
    "value_metric":      "Composite",
    "reliability_score": "Reliability",
    "relative_pe":       "Rel PE",
    "relative_pb":       "Rel PB",
    "rs_3m":             "RS 3m",
    "rs_6m":             "RS 6m",
    "realized_vol_3m":   "Vol 3m",
    "return_1m":         "Return 1m",
    "return_3m":         "Return 3m",
    "position_52w":      "52w Position",
}

_ROLE_COLOR = {
    "universe":  "rgba(110,120,130,0.18)",
    "candidate": "#2ecc71",
    "owned":     "#3498db",
}
_ROLE_SYMBOL = {
    "universe":  "circle",
    "candidate": "triangle-up",
    "owned":     "diamond",
}
_ROLE_SIZE = {"universe": 4, "candidate": 11, "owned": 11}


# ---------------------------------------------------------------------------
# Fast vectorised hover text
# ---------------------------------------------------------------------------

def _hover_col(df: pd.DataFrame, fields: list[str]) -> list[str]:
    """Build hover strings without iterrows — vectorised concat."""
    parts = ["<b>" + df["symbol"].fillna("").astype(str) + "</b>"]
    if "sector" in df.columns:
        parts.append(df["sector"].fillna("").astype(str))
    for f in fields:
        if f not in df.columns:
            continue
        num = pd.to_numeric(df[f], errors="coerce")
        lbl = _FACTOR_LABELS.get(f, f)
        parts.append(
            lbl + ": " + num.map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        )
    if "equity" in df.columns:
        eq = pd.to_numeric(df["equity"], errors="coerce")
        parts.append("Equity: $" + eq.map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—"))
    return ["<br>".join(row) for row in zip(*parts)]


# ---------------------------------------------------------------------------
# 1. Factor profile — the primary "how do I compare?" chart
# ---------------------------------------------------------------------------

def _factor_profile(df: pd.DataFrame, factors: list[str]):
    import plotly.graph_objects as go

    groups = [
        ("universe",  "Universe avg",   "rgba(110,120,130,0.55)"),
        ("candidate", "Candidates avg", "#2ecc71"),
        ("owned",     "Owned avg",      "#3498db"),
    ]

    fig = go.Figure()
    labels = [_FACTOR_LABELS.get(f, f) for f in factors]

    for role, gname, color in groups:
        sub = df[df["_role"] == role]
        if sub.empty:
            continue
        vals = [
            round(float(pd.to_numeric(sub[f], errors="coerce").mean()), 4)
            if f in sub.columns else 0.0
            for f in factors
        ]
        fig.add_trace(go.Bar(
            name=gname, x=labels, y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=9),
        ))

    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=20, r=20, t=10, b=30),
        yaxis=dict(gridcolor="#2d3436", title="avg score"),
        xaxis=dict(gridcolor="#2d3436"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#cdd6f4", size=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Percentile rank cards
# ---------------------------------------------------------------------------

def _percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.5
    return float((s < value).mean())


def _render_percentile_cards(df: pd.DataFrame, factors: list[str]) -> None:
    owned = df[df["_role"] == "owned"]
    cands = df[df["_role"] == "candidate"]
    univ  = df[df["_role"].isin(["universe", "owned", "candidate"])]

    for label, sub, color in [("Owned", owned, "#3498db"), ("Candidates", cands, "#2ecc71")]:
        if sub.empty:
            continue
        cols = st.columns(len(factors))
        for i, f in enumerate(factors):
            if f not in sub.columns or f not in univ.columns:
                continue
            avg_val = pd.to_numeric(sub[f], errors="coerce").mean()
            if pd.isna(avg_val):
                continue
            pct = _percentile_rank(univ[f], avg_val)
            delta = f"{pct:.0%} univ. pct"
            cols[i].metric(
                f"{label} avg {_FACTOR_LABELS.get(f, f)}",
                f"{avg_val:.3f}",
                delta,
                delta_color="normal",
            )


# ---------------------------------------------------------------------------
# 3. 2D scatter — universe as sampled backdrop, owned+candidates labelled
# ---------------------------------------------------------------------------

def _scatter_2d(df: pd.DataFrame, x_col: str, y_col: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    hover_fields = [x_col, y_col, "value_metric", "quality_score", "momentum_score",
                    "dist_from_centroid", "diversification_score", "nearest_owned"]

    univ = df[df["_role"] == "universe"].copy()
    n_sample = min(500, len(univ))
    univ_sample = univ.sample(n=n_sample, random_state=42) if len(univ) > n_sample else univ

    for frame, role in [(univ_sample, "universe"),
                        (df[df["_role"] == "candidate"], "candidate"),
                        (df[df["_role"] == "owned"], "owned")]:
        if frame.empty:
            continue
        xs = pd.to_numeric(frame[x_col], errors="coerce")
        ys = pd.to_numeric(frame[y_col], errors="coerce")
        hover = _hover_col(frame, hover_fields)
        show_labels = role != "universe"

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                color=_ROLE_COLOR[role],
                size=_ROLE_SIZE[role],
                symbol=_ROLE_SYMBOL[role],
                line=dict(width=0.8 if role != "universe" else 0, color="#111"),
            ),
            text=frame["symbol"].tolist() if show_labels else None,
            textposition="top center",
            textfont=dict(size=8, color="#cdd6f4"),
            name=role.title(),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    # Owned centroid cross-hair
    owned = df[df["_role"] == "owned"]
    if not owned.empty:
        cx = pd.to_numeric(owned[x_col], errors="coerce").mean()
        cy = pd.to_numeric(owned[y_col], errors="coerce").mean()
        if pd.notna(cx) and pd.notna(cy):
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(size=18, color="#3498db", symbol="star",
                            line=dict(width=2, color="#fff"), opacity=1.0),
                text=["Owned centroid"],
                textposition="bottom right",
                textfont=dict(size=9, color="#3498db"),
                name="Owned centroid",
                hovertemplate=f"Owned centroid<br>{x_col}: {cx:.3f}<br>{y_col}: {cy:.3f}<extra></extra>",
            ))

    # Candidate centroid
    cands = df[df["_role"] == "candidate"]
    if not cands.empty:
        cx = pd.to_numeric(cands[x_col], errors="coerce").mean()
        cy = pd.to_numeric(cands[y_col], errors="coerce").mean()
        if pd.notna(cx) and pd.notna(cy):
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(size=18, color="#2ecc71", symbol="star",
                            line=dict(width=2, color="#fff"), opacity=1.0),
                text=["Candidate centroid"],
                textposition="bottom right",
                textfont=dict(size=9, color="#2ecc71"),
                name="Candidate centroid",
                hovertemplate=f"Candidate centroid<br>{x_col}: {cx:.3f}<br>{y_col}: {cy:.3f}<extra></extra>",
            ))

    xl = _FACTOR_LABELS.get(x_col, x_col)
    yl = _FACTOR_LABELS.get(y_col, y_col)
    fig.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#444", line_width=1)

    fig.update_layout(
        height=460,
        margin=dict(l=50, r=20, t=30, b=50),
        xaxis=dict(title=xl, gridcolor="#2d3436", zeroline=False),
        yaxis=dict(title=yl, gridcolor="#2d3436", zeroline=False),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#cdd6f4", size=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)", bordercolor="#444", borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Portfolio Lens
# ---------------------------------------------------------------------------

def _render_portfolio_lens(df: pd.DataFrame, metric_threshold: float) -> None:
    df = _classify(df, metric_threshold)

    factors = [f for f in _FACTOR_COLS if f in df.columns]
    if not factors:
        st.warning("No factor score columns found in the universe data.")
        return

    df = _add_diversification_scores(df, factors)

    n_owned = int((df["_role"] == "owned").sum())
    n_cand  = int((df["_role"] == "candidate").sum())
    n_univ  = int((df["_role"] == "universe").sum())

    st.caption(
        f"**Universe** {n_univ:,}  ·  "
        f"**Candidates** {n_cand} ▲ (score ≥ {metric_threshold}, unowned)  ·  "
        f"**Owned** {n_owned} ◆"
    )

    # ── 1. Factor profile (top — fast) ───────────────────────────────────────
    st.markdown("#### How do owned positions and candidates compare to the universe?")
    st.caption(
        "Each bar is the **average** score for that group.  "
        "Blue = your owned positions.  Green = current buy candidates.  "
        "Gray = full universe."
    )
    st.plotly_chart(_factor_profile(df, factors), use_container_width=True, key="lens_profile")

    # ── 2. Percentile rank cards ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### Percentile rank vs universe")
    st.caption("Where does the average owned / candidate position sit in the universe distribution?")
    _render_percentile_cards(df, factors[:5])

    # ── 3. 2D scatter ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Factor space — owned ◆ and candidates ▲ on universe backdrop")
    st.caption(
        "Gray = 500 random universe symbols.  "
        "Universe reference lines at 0.  Hover any point for details."
    )

    avail = [f for f in [
        "quality_score", "momentum_score", "value_score", "income_score",
        "value_metric", "rs_3m", "rs_6m", "realized_vol_3m",
        "return_1m", "return_3m", "relative_pe", "relative_pb",
        "reliability_score", "position_52w",
    ] if f in df.columns]

    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox(
            "X axis", avail,
            index=avail.index("momentum_score") if "momentum_score" in avail else 0,
            format_func=lambda f: _FACTOR_LABELS.get(f, f),
            key="lens_x",
        )
    with c2:
        y_col = st.selectbox(
            "Y axis", avail,
            index=avail.index("quality_score") if "quality_score" in avail else 1,
            format_func=lambda f: _FACTOR_LABELS.get(f, f),
            key="lens_y",
        )

    st.plotly_chart(_scatter_2d(df, x_col, y_col), use_container_width=True, key="lens_scatter")

    # ── 4. Tables ─────────────────────────────────────────────────────────────
    st.divider()
    t1, t2 = st.tabs(["📋 Candidates", "📋 Owned positions"])

    with t1:
        cands = df[df["_role"] == "candidate"].copy()
        if cands.empty:
            st.info(f"No unowned candidates above score threshold ({metric_threshold}).")
        else:
            show = [c for c in [
                "symbol", "sector",
                "diversification_score", "value_metric",
                "dist_from_centroid", "nearest_owned",
                "quality_score", "momentum_score",
                "income_score", "value_score", "reliability_score",
            ] if c in cands.columns]
            fmt = {c: "{:.3f}" for c in show
                   if c not in ("symbol", "sector", "nearest_owned")}
            sort_col = "diversification_score" if "diversification_score" in cands.columns else "value_metric"
            st.caption(
                "**diversification_score** = model score × distance from owned centroid.  "
                "Rewards candidates that expand factor coverage, not just highest raw score."
            )
            st.dataframe(
                cands[show].sort_values(sort_col, ascending=False)
                    .style.format(fmt, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

    with t2:
        owned = df[df["_role"] == "owned"].copy()
        if not owned.empty:
            show = [c for c in ["symbol", "sector", "equity",
                                  "value_metric", "quality_score", "momentum_score",
                                  "income_score", "value_score", "reliability_score"]
                    if c in owned.columns]
            fmt = {c: "{:.3f}" for c in show if c not in ("symbol", "sector", "equity")}
            if "equity" in fmt:
                del fmt["equity"]
            st.dataframe(
                owned[show].sort_values(
                    "value_metric" if "value_metric" in owned.columns else "symbol",
                    ascending=False,
                ).style.format({**fmt, "equity": "${:,.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Component interpretation rendering
# ---------------------------------------------------------------------------

def _render_loadings_heatmap(df: pd.DataFrame, title: str, key_suffix: str) -> None:
    import plotly.graph_objects as go

    max_abs = df.abs().max(axis=1)
    df_sorted = df.loc[max_abs.sort_values(ascending=True).index]
    y_labels = [_FACTOR_LABELS.get(f, f) for f in df_sorted.index]

    fig = go.Figure(go.Heatmap(
        z=df_sorted.values,
        x=list(df_sorted.columns),
        y=y_labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in df_sorted.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="Feature: %{y}<br>%{x}: %{z:.3f}<extra></extra>",
    ))
    height = max(220, len(df_sorted) * 24 + 70)
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#cdd6f4")),
        height=height,
        margin=dict(l=120, r=20, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#cdd6f4", size=9),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"loadings_{key_suffix}")


def _render_component_report(diags: dict) -> None:
    report = diags.get("component_report")
    if not report:
        return

    method = report["method"]
    comp_names   = report.get("component_names", [])
    labels       = report.get("component_labels", [])
    themes       = report.get("component_themes", [])
    interps      = report.get("component_interpretations", [])

    st.divider()
    st.markdown("#### How to read this map")

    if method == "pca":
        exp_var  = report.get("explained_variance_ratio", [])
        cum_var  = report.get("explained_variance_cumulative", [])
        loadings = report.get("loadings")

        cum_pct = f"{cum_var[-1]:.0%}" if cum_var else "—"
        var_str = " + ".join(f"{v:.0%}" for v in exp_var)
        st.caption(
            f"PCA reduces {len(report.get('feature_names', []))} features to 3 axes.  "
            f"Variance captured: {var_str} = **{cum_pct}** cumulative."
        )

        cols = st.columns(len(comp_names))
        for i, name in enumerate(comp_names):
            pct  = exp_var[i] if i < len(exp_var) else 0.0
            with cols[i]:
                st.markdown(f"**{name}** &nbsp; `{pct:.0%} variance`")
                st.markdown(f"_{themes[i] if i < len(themes) else ''}_")
                st.caption(labels[i] if i < len(labels) else "")
                if i < len(interps):
                    st.caption(interps[i])

        if loadings is not None and not loadings.empty:
            st.markdown("**Component loadings** — red = positive weight, blue = negative")
            _render_loadings_heatmap(loadings, "PCA Loadings (feature weight per axis)", "pca")

    elif method == "umap":
        corr_df = report.get("correlations")

        st.caption(
            "UMAP is a non-linear embedding — axes have no fixed linear meaning. "
            "The table below shows **Spearman correlation** between each feature "
            "and each UMAP dimension, giving the closest interpretable analogue."
        )

        cols = st.columns(len(comp_names))
        for i, name in enumerate(comp_names):
            with cols[i]:
                st.markdown(f"**{name}**")
                st.markdown(f"_{themes[i] if i < len(themes) else ''}_")
                st.caption(labels[i] if i < len(labels) else "")
                if i < len(interps):
                    st.caption(interps[i])

        if corr_df is not None and not corr_df.empty:
            st.markdown("**Feature correlations with UMAP axes** (Spearman ρ)")
            _render_loadings_heatmap(corr_df, "UMAP Feature Correlations", "umap")


# ---------------------------------------------------------------------------
# Tab 2: Factor Map 3D
# ---------------------------------------------------------------------------

def _render_3d_map(df: pd.DataFrame, metric_threshold: float) -> None:
    st.caption("Best used with Owned + Candidates scope (readable). Full universe = abstract blob.")

    with st.expander("⚙️ Map settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            method = st.selectbox("Method", ["pca", "umap"], key="fm3d_method")
        with c2:
            scope = st.selectbox(
                "Scope", ["Owned + Candidates", "Owned only", "Full universe"],
                key="fm3d_scope",
            )
        with c3:
            color_opts = ["(auto)"] + [
                c for c in ["strategy_bucket", "sector", "_role", "cluster",
                             "value_metric", "quality_score", "momentum_score"]
                if c in df.columns or c in ("cluster", "_role")
            ]
            color_choice = st.selectbox("Colour by", color_opts, key="fm3d_color")
            color_by = None if color_choice == "(auto)" else color_choice
        with c4:
            clusters = st.slider("KMeans clusters", 0, 10, 0, key="fm3d_clusters")
            kmeans_clusters = clusters if clusters >= 2 else None
            if kmeans_clusters:
                color_by = "cluster"

    df_scope = _classify(df, metric_threshold)
    if scope == "Owned + Candidates":
        df_scope = df_scope[df_scope["_role"].isin(["owned", "candidate"])]
    elif scope == "Owned only":
        df_scope = df_scope[df_scope["_role"] == "owned"]

    if df_scope.empty:
        st.info("No data in selected scope.")
        return

    with st.spinner(f"Building {method.upper()} ({len(df_scope)} pts)…"):
        try:
            from portfolio.visualization.factor_map import build_factor_map
            fig, df_out, diags = build_factor_map(
                df_scope, method=method, color_by=color_by,
                kmeans_clusters=kmeans_clusters, output_html=None, show=False,
            )
        except ImportError as exc:
            st.error(str(exc)); return
        except ValueError as exc:
            st.warning(str(exc)); return
        except Exception as exc:
            st.error(f"Factor map failed: {exc}"); st.exception(exc); return

    st.plotly_chart(fig, use_container_width=True)

    _render_component_report(diags)

    if "cluster_summary" in diags:
        cs = diags["cluster_summary"]
        st.markdown("**Cluster summary**")

        # Equity bar chart when equity data is present
        if "equity_$" in cs.columns and cs["equity_$"].notna().any():
            import plotly.graph_objects as _go
            _cs_eq = cs.dropna(subset=["equity_$"]).sort_values("equity_$", ascending=True)
            _fig_eq = _go.Figure(_go.Bar(
                x=_cs_eq["equity_$"],
                y=_cs_eq["cluster"].astype(str).radd("Cluster "),
                orientation="h",
                marker_color="#3498db",
                text=_cs_eq["equity_$"].map(lambda v: f"${v:,.0f}"),
                textposition="outside",
            ))
            _fig_eq.update_layout(
                height=max(180, len(_cs_eq) * 40 + 50),
                margin=dict(l=80, r=60, t=10, b=30),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#cdd6f4", size=10),
                xaxis=dict(title="Active equity ($)", gridcolor="#2d3436"),
                yaxis=dict(gridcolor="#2d3436"),
            )
            st.plotly_chart(_fig_eq, use_container_width=True, key="cluster_equity_bar")

        show_cols = [c for c in [
            "cluster", "owned_count", "equity_$", "equity_weight",
            "avg_value_metric", "avg_quality_score", "avg_momentum_score", "top_symbols",
        ] if c in cs.columns]
        fmt = {c: "{:.3f}" for c in ["avg_value_metric", "avg_quality_score", "avg_momentum_score"] if c in cs.columns}
        if "equity_weight" in cs.columns:
            fmt["equity_weight"] = "{:.1%}"
        if "equity_$" in cs.columns:
            fmt["equity_$"] = "${:,.0f}"
        st.dataframe(cs[show_cols].style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Portfolio-page entry point (no 3D tab, no settings overhead)
# ---------------------------------------------------------------------------

def render_portfolio_lens() -> None:
    """
    Lightweight embed for the Portfolio page.
    Shows only the Portfolio Lens (factor profile + scatter).
    No 3D map / no settings expander.
    """
    df, status_msg = _load_universe()
    if df.empty:
        st.warning(f"Factor lens unavailable: {status_msg}")
        st.caption("Run `daily-investor fetch-data` to build agg_data, then reload.")
        return

    st.caption(status_msg)

    from ui.utils import load_config_raw
    cfg = load_config_raw()
    metric_threshold = float(cfg.get("metric_threshold", 0.75))
    _render_portfolio_lens(df, metric_threshold)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Factor Map — Portfolio vs Universe")

    df, status_msg = _load_universe()
    if df.empty:
        st.error(f"Could not load universe: {status_msg}")
        st.info("Run `daily-investor fetch-data` to build agg_data, then reload.")
        return

    st.caption(status_msg)

    from ui.utils import load_config_raw
    cfg = load_config_raw()
    metric_threshold = float(cfg.get("metric_threshold", 0.75))

    tab1, tab2 = st.tabs(["📊 Portfolio Lens", "🗺️ Factor Map 3D"])

    with tab1:
        _render_portfolio_lens(df, metric_threshold)

    with tab2:
        _render_3d_map(df, metric_threshold)
