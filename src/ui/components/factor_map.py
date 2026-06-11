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
    """
    Tag each row with `_role` ∈ {owned, candidate, universe}, matching the LIVE buy
    gate. A candidate is an UNOWNED name in the top `top_percentile` of value_metric
    that clears the quality + momentum floors and is not on the contrarian watchlist
    (CANDIDATE_SELECTION_PARAMS — the same gate select_candidates() uses).

    NB: value_metric is peer-relative, normalized to roughly [-1, 1].
    `metric_threshold` (1.15) is the EXIT-ladder anchor and unreachable as an
    entry bar on this scale (the old code gated on it and produced ZERO
    candidates); the LIVE entry gate is candidate_selection.entry_threshold_override
    (0.75) with the percentile/floor gates below — the same gate select_candidates()
    and the live buy ladder use.
    """
    df = df.copy()
    owned_mask = df["owned"].astype(bool) if "owned" in df.columns else pd.Series(False, index=df.index)

    cand_mask = pd.Series(False, index=df.index)
    if "value_metric" in df.columns:
        from util import CANDIDATE_SELECTION_PARAMS as _cs
        vm = pd.to_numeric(df["value_metric"], errors="coerce")
        unowned_vm = vm[~owned_mask].dropna()
        if not unowned_vm.empty:
            top_pct = float(_cs.get("top_percentile", 0.15))
            cutoff = float(unowned_vm.quantile(max(0.0, min(1.0, 1.0 - top_pct))))
            min_q = float(_cs.get("min_quality_score", 0.30))
            min_m = float(_cs.get("min_momentum_score", -0.10))
            qual = (pd.to_numeric(df["quality_score"], errors="coerce")
                    if "quality_score" in df.columns else pd.Series(min_q, index=df.index))
            mom = (pd.to_numeric(df["momentum_score"], errors="coerce")
                   if "momentum_score" in df.columns else pd.Series(min_m, index=df.index))
            cand_mask = (
                (vm >= cutoff)
                & (qual.fillna(min_q - 1.0) >= min_q)
                & (mom.fillna(min_m - 1.0) >= min_m)
                & ~owned_mask
            )
            # Exclude contrarian/downtrend names — the core candidate set is the
            # momentum-confirmed buys, not the whole top-percentile cheap cohort.
            if "strategy_bucket" in df.columns:
                cand_mask &= (df["strategy_bucket"] == "core_candidate")

    df["_role"] = "universe"
    df.loc[cand_mask,  "_role"] = "candidate"
    df.loc[owned_mask, "_role"] = "owned"
    return df


# ---------------------------------------------------------------------------
# Scope selection — filter the universe BEFORE embedding / clustering so ETFs
# do not influence the geometry when stocks-only analysis is requested.
# ---------------------------------------------------------------------------

SCOPE_OPTIONS = [
    "Stocks only",
    "Full universe",
    "ETFs only",
    "Owned only",
    "Candidates only",
    "Owned + Candidates",
    "Active sleeve only",
]


def apply_scope(
    df: pd.DataFrame,
    scope: str,
    config: dict | None,
    metric_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """Classify owned/candidate roles, tag ETFs, then filter to ``scope``.

    Returns ``(scoped_df, meta)`` where ``meta`` carries the counts used in the
    chart subtitle. All filtering happens here, before any embedding / KMeans /
    diagnostics, so excluded rows never influence the geometry.
    """
    from portfolio.visualization.factor_map import tag_etf

    work = _classify(df, metric_threshold)
    work = tag_etf(work, config)

    etf_mask = work["is_etf"].astype(bool)
    role = work["_role"]

    if scope == "Stocks only":
        out = work[~etf_mask]
    elif scope == "ETFs only":
        out = work[etf_mask]
    elif scope == "Owned only":
        out = work[role == "owned"]
    elif scope == "Candidates only":
        out = work[role == "candidate"]
    elif scope == "Owned + Candidates":
        out = work[role.isin(["owned", "candidate"])]
    elif scope == "Active sleeve only":
        out = work[role.isin(["owned", "candidate"]) & ~etf_mask]
    else:  # "Full universe" (and any unknown value — fail open)
        out = work

    out = out.copy()
    out_role = out["_role"] if "_role" in out.columns else pd.Series(dtype=str)
    meta = {
        "total_universe": len(work),
        "etf_total": int(etf_mask.sum()),
        "in_scope": len(out),
        "owned_in_scope": int((out_role == "owned").sum()),
        "candidates_in_scope": int((out_role == "candidate").sum()),
        "etf_in_scope": int(out["is_etf"].astype(bool).sum()) if "is_etf" in out.columns else 0,
    }
    return out, meta


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
        f"**Candidates** {n_cand} ▲ (top-percentile value, momentum+quality floors, unowned, non-contrarian)  ·  "
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
            st.info("No unowned candidates clear the live buy gate (top-percentile value + momentum/quality floors, non-contrarian).")
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

# Minimum rows needed for a meaningful 3-D embedding / KMeans pass.
_MIN_EMBED_ROWS = 5


# Stable archetype colours (mirrors ui/components/archetype_diagnostics.py).
_ARCHETYPE_COLORS = {
    "quality_compounder":   "#4c8ef5",
    "value_recovery":       "#f5a623",
    "defensive_income":     "#7ed321",
    "speculative_momentum": "#d0021b",
    "legacy_turnaround":    "#9b59b6",
    "core_default":         "#aaaaaa",
}
_GROUP_SHELL_COLORS = {"owned": "#3498db", "candidate": "#2ecc71", "universe": "#8895a7"}


def _coord_cols_for(method: str) -> list[str]:
    return ["umap_1", "umap_2", "umap_3"] if method == "umap" else ["pca_1", "pca_2", "pca_3"]


def _num(v) -> float:
    x = pd.to_numeric(v, errors="coerce")
    return 0.0 if pd.isna(x) else float(x)


@st.cache_data(ttl=300, show_spinner=False)
def _compute_archetypes(df: pd.DataFrame) -> pd.Series:
    """Score-only archetype label per row (no Robinhood / market-structure calls).

    A consistent universe-wide approximation of the live archetype — uses only
    agg_data columns so it can colour the whole map.
    """
    from portfolio.position_archetypes import classify_archetype_full_from_scores
    from ui.utils import load_config_raw

    cfg = load_config_raw().get("archetype_management", {})

    def _one(r: pd.Series) -> str:
        try:
            return classify_archetype_full_from_scores(
                quality_score=_num(r.get("quality_score")),
                momentum_score=_num(r.get("momentum_score")),
                income_score=_num(r.get("income_score")),
                yield_trap=bool(r.get("yield_trap_flag", False)),
                archetype_cfg=cfg,
                sector=r.get("sector"),
                industry=r.get("industry"),
                value_score=_num(r.get("value_score")),
            ).archetype
        except Exception:
            return "core_default"

    return df.apply(_one, axis=1)


def _ellipsoid_mesh(points: np.ndarray, color: str, name: str, nsig: float = 2.0):
    """2σ confidence ellipsoid (solid Mesh3d) for a group of 3-D points."""
    import plotly.graph_objects as go

    if points.shape[0] < 4:
        return None
    mu = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    if np.any(vals <= 1e-9):
        return None  # degenerate / collinear group
    radii = nsig * np.sqrt(vals)
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 18)
    sphere = np.stack([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones_like(u), np.cos(v)).ravel(),
    ], axis=1)
    ell = (sphere * radii) @ vecs.T + mu
    return go.Mesh3d(
        x=ell[:, 0], y=ell[:, 1], z=ell[:, 2],
        alphahull=0, opacity=0.12, color=color,
        name=name, showlegend=True, hoverinfo="name",
    )


def _add_group_ellipsoids(fig, df_out: pd.DataFrame, coord_cols: list[str]) -> None:
    """Overlay 2σ ellipsoids for owned / candidate / universe groups."""
    if not all(c in df_out.columns for c in coord_cols):
        return
    coords = df_out[coord_cols].to_numpy(dtype=float)
    groups: list[tuple[str, np.ndarray]] = [("universe", coords)]
    if "_role" in df_out.columns:
        for role in ("owned", "candidate"):
            mask = (df_out["_role"] == role).to_numpy()
            if mask.sum() >= 4:
                groups.append((role, coords[mask]))
    for name, pts in groups:
        mesh = _ellipsoid_mesh(pts, _GROUP_SHELL_COLORS.get(name, "#888"), f"{name} 2σ")
        if mesh is not None:
            fig.add_trace(mesh)


def _nearest_table(df_out: pd.DataFrame, centroid: np.ndarray, coord_cols: list[str], n: int = 10) -> pd.DataFrame:
    coords = df_out[coord_cols].to_numpy(dtype=float)
    dist = np.linalg.norm(coords - np.asarray(centroid, dtype=float), axis=1)
    out = df_out.assign(distance=dist).nsmallest(n, "distance")
    cols = [c for c in ["symbol", "sector", "_role", "distance",
                        "value_metric", "quality_score", "momentum_score"]
            if c in out.columns]
    return out[cols]


def _render_nearest_tables(fig, df_out, centroids, coord_cols, event) -> None:
    specs = []
    if isinstance(centroids, dict):
        if centroids.get("owned") is not None:
            specs.append(("owned", "◆ Nearest to owned centroid", centroids["owned"]))
        if centroids.get("candidate") is not None:
            specs.append(("candidate", "▲ Nearest to candidate centroid", centroids["candidate"]))
    if not specs:
        return

    # Best-effort: did the user click a centroid? Map selection → trace name.
    focused = None
    try:
        pts = []
        try:
            pts = event.selection["points"]
        except Exception:
            pts = (event or {}).get("selection", {}).get("points", [])
        for p in pts:
            cn = p.get("curve_number", p.get("curveNumber"))
            nm = fig.data[cn].name if (cn is not None and cn < len(fig.data)) else ""
            if "owned centroid" in str(nm).lower():
                focused = "owned"
            elif "candidate centroid" in str(nm).lower():
                focused = "candidate"
    except Exception:
        focused = None
    if focused:
        specs.sort(key=lambda s: s[0] != focused)

    st.markdown("**Nearest stocks to centroids**")
    if focused:
        st.caption(f"Focused on the **{focused}** centroid (clicked). Click a centroid marker to switch.")
    else:
        st.caption("Closest names in embedding space to each centroid. Click a centroid marker to focus.")
    fmt = {c: "{:.3f}" for c in ("distance", "value_metric", "quality_score", "momentum_score")}
    cols = st.columns(len(specs))
    for col, (key, label, centroid) in zip(cols, specs):
        with col:
            st.caption(label)
            tbl = _nearest_table(df_out, centroid, coord_cols)
            st.dataframe(
                tbl.style.format({k: v for k, v in fmt.items() if k in tbl.columns}, na_rep="—"),
                hide_index=True, use_container_width=True,
            )


def _render_3d_map(df: pd.DataFrame, metric_threshold: float, config: dict | None = None) -> None:
    st.caption(
        "Stocks-only excludes ETFs so the embedding reflects the active stock universe. "
        "Full universe shows ETFs vs stocks (the two dominant groups)."
    )

    with st.expander("⚙️ Map settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            method = st.selectbox("Method", ["pca", "umap"], key="fm3d_method")
        with c2:
            scope = st.selectbox(
                "Scope", SCOPE_OPTIONS, index=0, key="fm3d_scope",
            )
        with c3:
            # Color options: only columns actually present. (Legacy *_v3 column
            # fallbacks removed — all on-disk snapshots are converged to peer-1
            # canonical names via `snapshots rescore`.)
            color_candidates = [
                "_role", "cluster", "archetype", "sector", "industry",
                "strategy_bucket",
                "value_score", "quality_score",
                "momentum_score", "income_score",
                "value_metric", "final_score",
            ]
            seen: set[str] = set()
            color_opts = ["(auto)"]
            for c in color_candidates:
                if c in seen:
                    continue
                if c in df.columns or c in ("cluster", "_role", "archetype"):
                    color_opts.append(c)
                    seen.add(c)
            color_choice = st.selectbox("Colour by", color_opts, key="fm3d_color")
            color_by = None if color_choice == "(auto)" else color_choice
        with c4:
            clusters = st.slider("KMeans clusters", 0, 10, 0, key="fm3d_clusters")
            kmeans_clusters = clusters if clusters >= 2 else None
            if kmeans_clusters:
                color_by = "cluster"

        oc1, oc2 = st.columns(2)
        with oc1:
            exclude_outliers = st.checkbox(
                "Exclude outliers", value=False, key="fm3d_outliers",
                help="Drop feature-space oddballs (robust MAD z > 5) BEFORE embedding "
                     "so they don't warp the UMAP/PCA geometry.",
            )
        with oc2:
            show_shells = st.checkbox(
                "Group shells (2σ)", value=False, key="fm3d_shells",
                help="Overlay 2σ confidence ellipsoids for owned / candidate / universe groups.",
            )

    df_scope, meta = apply_scope(df, scope, config, metric_threshold)

    # ── Scope title + counts ──────────────────────────────────────────────────
    st.markdown(f"#### Factor Map — {scope}")
    subtitle = (
        f"{meta['in_scope']:,} in scope · {meta['owned_in_scope']} owned · "
        f"{meta['candidates_in_scope']} candidates"
    )
    if scope in ("Stocks only", "Active sleeve only") and meta["etf_total"]:
        subtitle += f" · {meta['etf_total']} ETFs excluded"
    elif scope == "ETFs only":
        subtitle += f" · {meta['etf_in_scope']} ETFs"
    st.caption(subtitle)

    if df_scope.empty:
        if scope == "ETFs only" and meta["etf_total"] == 0:
            st.info("No ETFs found in the universe for this scope.")
        elif scope == "Stocks only" and meta["in_scope"] == 0:
            st.info("No stocks found in the universe for this scope.")
        else:
            st.info("No data in selected scope.")
        return

    if len(df_scope) < _MIN_EMBED_ROWS:
        st.warning(
            f"Only {len(df_scope)} symbol(s) in scope — need at least "
            f"{_MIN_EMBED_ROWS} for a meaningful embedding. Widen the scope."
        )
        return

    # Guard KMeans against asking for more clusters than points.
    if kmeans_clusters is not None and kmeans_clusters >= len(df_scope):
        st.info(
            f"Reduced KMeans clusters to fit {len(df_scope)} points in scope."
        )
        kmeans_clusters = max(2, len(df_scope) - 1)

    # Archetype is computed on demand (score-only) since it's not in agg_data.
    color_map = None
    if color_by == "archetype":
        with st.spinner("Classifying archetypes…"):
            df_scope = df_scope.copy()
            df_scope["archetype"] = _compute_archetypes(df_scope)
        color_map = _ARCHETYPE_COLORS

    with st.spinner(f"Building {method.upper()} ({len(df_scope)} pts)…"):
        try:
            from portfolio.visualization.factor_map import build_factor_map
            fig, df_out, diags = build_factor_map(
                df_scope, method=method, color_by=color_by,
                kmeans_clusters=kmeans_clusters, output_html=None, show=False,
                color_map=color_map, exclude_outliers=exclude_outliers,
            )
        except ImportError as exc:
            st.error(str(exc)); return
        except ValueError as exc:
            st.warning(str(exc)); return
        except Exception as exc:
            st.error(f"Factor map failed: {exc}"); st.exception(exc); return

    coord_cols = _coord_cols_for(method)
    if show_shells:
        _add_group_ellipsoids(fig, df_out, coord_cols)

    try:
        event = st.plotly_chart(
            fig, use_container_width=True, key="fm3d_chart", on_select="rerun",
        )
    except TypeError:
        # Older Streamlit without on_select — fall back to a static chart.
        st.plotly_chart(fig, use_container_width=True)
        event = None

    if exclude_outliers:
        dropped = diags.get("outliers_excluded") or []
        if dropped:
            with st.expander(f"⚠️ {len(dropped)} feature-space outlier(s) removed before embedding"):
                st.caption("Dropped so they don't distort the UMAP/PCA geometry.")
                st.write(", ".join(map(str, dropped)))
        else:
            st.caption("No outliers exceeded the threshold — embedding used all in-scope points.")

    _render_nearest_tables(fig, df_out, diags.get("centroids", {}), coord_cols, event)

    _render_component_report(diags)

    if "cluster_summary" in diags:
        cs = diags["cluster_summary"]
        st.markdown("**Cluster summary**")
        st.caption(f"Cluster weights, counts & composition computed within scope: **{scope}**.")

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
        _render_3d_map(df, metric_threshold, config=cfg)
