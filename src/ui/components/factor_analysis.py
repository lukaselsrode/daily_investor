"""
ui/components/factor_analysis.py — Factor orthogonalization and overlap diagnostics.

Answers: are value, momentum, quality, and income measuring independent things,
or are they partially double-counting the same signal?

Sections
--------
1. Pairwise correlation — Pearson + Spearman, with significance stars
2. Scatter matrix — bivariate plots for every factor pair, regression line overlaid
3. OLS residualization — for each pair, regress one factor out of the other and show
   how much explanatory power is shared
4. Variance Inflation Factors — multicollinearity summary; VIF > 5 signals redundancy
5. Residualized score distributions — compare raw vs orthogonalized value / momentum
6. Portfolio rank impact — how much does top-20 candidate list change when using
   residualized scores instead of raw scores?
7. Interpretation guide
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui.utils import data_date, fmt_bin_index, load_config_raw, load_latest_csv, no_data_msg

_FACTOR_COLS = ["value_score", "momentum_score", "quality_score", "income_score"]
_FACTOR_LABELS = {
    "value_score":    "Value",
    "momentum_score": "Momentum",
    "quality_score":  "Quality",
    "income_score":   "Income",
}


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------

def _ols_residualize(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Regress y on x (with intercept).  Return (residuals, R², slope).
    NaN-safe: only rows where both are finite are used.
    Residuals at NaN positions are NaN.
    """
    mask = y.notna() & x.notna()
    n = mask.sum()
    if n < 10:
        return y.copy(), float("nan"), float("nan")

    xv = x[mask].values
    yv = y[mask].values
    X = np.column_stack([np.ones(n), xv])
    try:
        b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    except np.linalg.LinAlgError:
        return y.copy(), float("nan"), float("nan")

    fitted   = X @ b
    resid_v  = yv - fitted
    ss_res   = float((resid_v ** 2).sum())
    ss_tot   = float(((yv - yv.mean()) ** 2).sum())
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    resid = pd.Series(float("nan"), index=y.index)
    resid[mask] = resid_v
    return resid, round(r2, 4), round(float(b[1]), 4)


def _vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each column using OLS.
    VIF_j = 1 / (1 - R²_j), where R²_j is from regressing col_j on all others.
    """
    rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        y  = df[col].dropna()
        xs = df[others].loc[y.index].dropna()
        common = y.index.intersection(xs.index)
        yv = y.loc[common].values
        xv = xs.loc[common].values
        if len(common) < 10 or xv.shape[1] == 0:
            rows.append({"factor": col, "VIF": float("nan"), "R²_vs_others": float("nan")})
            continue
        X = np.column_stack([np.ones(len(common)), xv])
        try:
            b, *_ = np.linalg.lstsq(X, yv, rcond=None)
            fitted = X @ b
            ss_res = ((yv - fitted) ** 2).sum()
            ss_tot = ((yv - yv.mean()) ** 2).sum()
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif    = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except np.linalg.LinAlgError:
            r2, vif = float("nan"), float("nan")
        rows.append({"factor": col, "VIF": round(vif, 3), "R²_vs_others": round(r2, 4)})
    return pd.DataFrame(rows).set_index("factor")


def _spearman_ic(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return float("nan")
    return float(a[mask].corr(b[mask], method="spearman"))


def _pearson_p(a: pd.Series, b: pd.Series) -> float:
    """Return two-tailed p-value for Pearson r."""
    from scipy import stats as _stats
    mask = a.notna() & b.notna()
    n = mask.sum()
    if n < 5:
        return float("nan")
    r, p = _stats.pearsonr(a[mask].values, b[mask].values)
    return float(p)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("🔗 Factor Orthogonalization")
    st.caption(
        "Are value, momentum, quality, and income measuring independent things, "
        "or partially double-counting the same signal?"
    )

    df = load_latest_csv("agg_data")
    if df is None:
        st.warning(no_data_msg("agg_data"))
        return

    for col in _FACTOR_COLS + ["value_metric", "pe_ratio", "pb_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    avail = [c for c in _FACTOR_COLS if c in df.columns and df[c].notna().sum() >= 10]
    if len(avail) < 2:
        st.warning("Need at least 2 factor columns with data.")
        return

    st.caption(f"Source: agg_data {data_date('agg_data')} | {len(df)} symbols | factors: {avail}")

    # ── 1. Pairwise correlation matrix ───────────────────────────────────────
    st.subheader("1 · Pairwise factor correlations")

    corr_method = st.radio("Method", ["spearman", "pearson"], horizontal=True, key="corr_method_fa")
    corr_df = df[avail].corr(method=corr_method).round(3)

    # Significance stars (uses scipy if available, else skips)
    try:
        from scipy import stats as _stats
        star_df = pd.DataFrame("", index=corr_df.index, columns=corr_df.columns)
        for i, c1 in enumerate(avail):
            for j, c2 in enumerate(avail):
                if i >= j:
                    continue
                p = _pearson_p(df[c1], df[c2])
                if not np.isnan(p):
                    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                    star_df.loc[c1, c2] = star
                    star_df.loc[c2, c1] = star
        has_scipy = True
    except ImportError:
        has_scipy = False

    try:
        import plotly.express as px
        fig = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Factor correlation matrix ({corr_method})",
            aspect="equal",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.dataframe(corr_df.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                     use_container_width=True)

    # Highlight the most important pair
    off_diag = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    if not off_diag.stack().empty:
        most_corr = off_diag.stack().abs().idxmax()
        r_val     = corr_df.loc[most_corr]
        verdict   = "⚠️ High overlap" if abs(r_val) > 0.4 else ("mild overlap" if abs(r_val) > 0.2 else "✅ Low overlap")
        st.info(
            f"Strongest pair: **{most_corr[0]}** ↔ **{most_corr[1]}**  "
            f"r = {r_val:.3f}  ({verdict})"
        )

    # ── 2. Scatter matrix ────────────────────────────────────────────────────
    st.divider()
    st.subheader("2 · Bivariate scatter plots")

    pairs = [(a, b) for i, a in enumerate(avail) for b in avail[i + 1:]]
    pair_labels = [f"{_FACTOR_LABELS.get(a, a)} vs {_FACTOR_LABELS.get(b, b)}" for a, b in pairs]
    chosen_pair_label = st.selectbox("Factor pair", pair_labels, key="pair_select")
    fa, fb = pairs[pair_labels.index(chosen_pair_label)]

    scatter_df = df[["symbol", fa, fb]].dropna()
    if not scatter_df.empty:
        try:
            import plotly.express as px

            fig = px.scatter(
                scatter_df, x=fa, y=fb,
                hover_name="symbol" if "symbol" in scatter_df.columns else None,
                opacity=0.5,
                title=f"{_FACTOR_LABELS.get(fa, fa)} vs {_FACTOR_LABELS.get(fb, fb)}",
                trendline="ols",
                trendline_color_override="red",
            )
            r, _ = _stats.pearsonr(scatter_df[fa].values, scatter_df[fb].values) if has_scipy else (corr_df.loc[fa, fb], None)
            ic   = _spearman_ic(scatter_df[fa], scatter_df[fb])
            fig.update_layout(
                annotations=[dict(
                    x=0.02, y=0.97, xref="paper", yref="paper",
                    text=f"Pearson r = {corr_df.loc[fa, fb]:.3f}   Spearman ρ = {ic:.3f}",
                    showarrow=False, bgcolor="rgba(255,255,255,0.7)",
                )],
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.scatter_chart(scatter_df[[fa, fb]], x=fa, y=fb)

        r_val  = corr_df.loc[fa, fb]
        ic_val = _spearman_ic(df[fa], df[fb])
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(f"Pearson r ({fa[:5]}↔{fb[:5]})", f"{r_val:.3f}")
        mc2.metric("Spearman ρ", f"{ic_val:.3f}")
        mc3.metric("Shared variance (r²)", f"{r_val**2:.1%}")

    # ── 3. OLS residualization ───────────────────────────────────────────────
    st.divider()
    st.subheader("3 · OLS residualization")
    st.caption(
        "For each factor pair, how much of Factor A is explained by Factor B? "
        "R² = shared variance. Residuals = the part of A that's independent of B."
    )

    resid_table = []
    for fa2, fb2 in pairs:
        resid, r2, slope = _ols_residualize(df[fa2], df[fb2])
        resid_table.append({
            "y (target)":   _FACTOR_LABELS.get(fa2, fa2),
            "x (regressed out)": _FACTOR_LABELS.get(fb2, fb2),
            "slope":    slope,
            "R²":       r2,
            "shared variance": f"{r2:.1%}" if not np.isnan(r2) else "—",
            "residual std": round(float(resid.dropna().std()), 4) if resid.dropna().std() > 0 else float("nan"),
        })
    resid_df = pd.DataFrame(resid_table).sort_values("R²", ascending=False)
    st.dataframe(resid_df, use_container_width=True, hide_index=True)

    st.caption(
        "R² > 0.25 → factors share >25% variance — consider whether the overlap is intentional. "
        "R² > 0.50 → strong redundancy — residualize before combining."
    )

    # Focus residualization: pick any pair and show before/after distribution
    st.markdown("**Residualize one factor vs another:**")
    rc1, rc2 = st.columns(2)
    with rc1:
        resid_y = st.selectbox("Factor to clean", avail, key="resid_y",
                               format_func=lambda c: _FACTOR_LABELS.get(c, c))
    with rc2:
        resid_x_opts = [c for c in avail if c != resid_y]
        resid_x = st.selectbox("Remove influence of", resid_x_opts, key="resid_x",
                               format_func=lambda c: _FACTOR_LABELS.get(c, c))

    resid_series, r2_show, slope_show = _ols_residualize(df[resid_y], df[resid_x])

    if resid_series.dropna().empty:
        st.warning("Not enough data to residualize.")
    else:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.caption(f"Raw {_FACTOR_LABELS.get(resid_y, resid_y)}")
            raw_s = df[resid_y].dropna()
            st.bar_chart(fmt_bin_index(raw_s.value_counts(bins=25, sort=False).sort_index()))
            st.metric("std", f"{raw_s.std():.4f}")
        with rc2:
            st.caption(f"{_FACTOR_LABELS.get(resid_y, resid_y)} ⊥ {_FACTOR_LABELS.get(resid_x, resid_x)} (residual)")
            clean_s = resid_series.dropna()
            st.bar_chart(fmt_bin_index(clean_s.value_counts(bins=25, sort=False).sort_index()))
            st.metric("std", f"{clean_s.std():.4f}")

        st.metric(
            f"Variance removed ({_FACTOR_LABELS.get(resid_x, resid_x)} → {_FACTOR_LABELS.get(resid_y, resid_y)})",
            f"{r2_show:.1%}" if not np.isnan(r2_show) else "—",
            help="Fraction of variance in the target factor explained by the regressed-out factor."
        )

    # ── 4. VIF ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("4 · Variance Inflation Factors")
    st.caption(
        "VIF measures how much of a factor's variance is explained by all other factors combined. "
        "VIF < 2 = independent. VIF 2–5 = moderate overlap. VIF > 5 = high multicollinearity."
    )

    vif_df = _vif(df, avail)
    vif_df["status"] = vif_df["VIF"].apply(
        lambda v: "✅ Independent" if np.isnan(v) or v < 2 else
                  ("⚠️ Moderate"   if v < 5   else "🔴 High overlap")
    )

    try:
        import plotly.express as px
        fig = px.bar(
            vif_df.reset_index(),
            x="factor", y="VIF",
            color="VIF",
            color_continuous_scale=["green", "yellow", "red"],
            range_color=[1, 5],
            title="Variance Inflation Factor by factor",
        )
        fig.add_hline(y=2, line_dash="dot", annotation_text="VIF=2 (mild)", line_color="orange")
        fig.add_hline(y=5, line_dash="dot", annotation_text="VIF=5 (high)", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(vif_df["VIF"])

    st.dataframe(vif_df, use_container_width=True)

    # ── 5. Portfolio rank impact ─────────────────────────────────────────────
    st.divider()
    st.subheader("5 · Candidate list impact")
    st.caption(
        "How much does the top-N buy candidate list change when you substitute a "
        "residualized value_score for the raw one?"
    )

    if "value_score" in df.columns and "momentum_score" in df.columns and "value_metric" in df.columns:
        cfg = load_config_raw()
        sw      = cfg.get("score_weights", {})
        w_val   = float(sw.get("value", 0.08))
        w_qual  = float(sw.get("quality", 0.50))
        w_inc   = float(sw.get("income", 0.08))
        w_mom   = float(sw.get("momentum", 0.34))

        top_n = st.slider("Candidate list size (N)", 5, 50, 20, key="rank_top_n")

        # Compute residualized value_score (value ⊥ momentum)
        resid_val, r2_vm, _ = _ols_residualize(df["value_score"], df["momentum_score"])

        # Recompute value_metric using residualized value
        df["value_score_orth"] = resid_val
        for col in ["quality_score", "income_score", "momentum_score"]:
            df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)

        df["value_metric_orth"] = (
            w_val  * df["value_score_orth"].fillna(0.0)
            + w_qual * df["quality_score"]
            + w_inc  * df["income_score"]
            + w_mom  * df["momentum_score"]
        ).round(4)

        # Top-N by original vs orthogonalized metric
        def _top_symbols(metric_col: str, n: int) -> set[str]:
            col_data = df[["symbol", metric_col]].dropna()
            return set(col_data.nlargest(n, metric_col)["symbol"].tolist())

        top_orig  = _top_symbols("value_metric", top_n)
        top_orth  = _top_symbols("value_metric_orth", top_n)
        overlap   = top_orig & top_orth
        added     = top_orth - top_orig
        removed   = top_orig - top_orth
        stability = len(overlap) / top_n

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Overlap",     f"{len(overlap)}/{top_n}")
        sc2.metric("Stability",   f"{stability:.0%}")
        sc3.metric("New entries", len(added))
        sc4.metric("Dropped",     len(removed))

        st.metric(
            "Value⊥Momentum shared variance (R²)",
            f"{r2_vm:.1%}" if not np.isnan(r2_vm) else "—",
            help="How much of value_score is explained by momentum_score. "
                 "High R² means the residualization changes rankings significantly."
        )

        if added or removed:
            ci1, ci2 = st.columns(2)
            with ci1:
                st.markdown("**Entered top list after orthogonalization**")
                if added:
                    added_rows = df[df["symbol"].isin(added)][
                        ["symbol", "value_score", "momentum_score", "value_metric", "value_metric_orth"]
                    ].sort_values("value_metric_orth", ascending=False)
                    st.dataframe(added_rows.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("None")
            with ci2:
                st.markdown("**Dropped from top list after orthogonalization**")
                if removed:
                    rem_rows = df[df["symbol"].isin(removed)][
                        ["symbol", "value_score", "momentum_score", "value_metric", "value_metric_orth"]
                    ].sort_values("value_metric", ascending=False)
                    st.dataframe(rem_rows.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("None")

        if stability < 0.7:
            st.warning(
                f"Only {stability:.0%} of the top-{top_n} list is stable under residualization. "
                "Value and momentum are significantly overlapping — consider residualizing in the live scorer."
            )
        elif stability < 0.9:
            st.info(f"{stability:.0%} list stability — moderate value/momentum overlap.")
        else:
            st.success(f"{stability:.0%} list stability — factors are largely orthogonal.")
    else:
        st.info("Requires value_score, momentum_score, and value_metric columns in agg_data.")

    # ── 6. Interpretation guide ──────────────────────────────────────────────
    st.divider()
    with st.expander("Interpretation guide"):
        st.markdown("""
**What orthogonalization tells you**

If value and momentum are negatively correlated (cheap stocks tend to have weak momentum),
your optimizer sees a conflict: raising value_score weight hurts via momentum drag, and
lowering it weakens value signal. This can cause the optimizer to oscillate.

Residualizing removes the shared component so each factor measures something genuinely
independent.  This is standard practice in multi-factor quant models (Fama-French, AQR).

---

**When to residualize in the live scorer**

Orthogonalize live scoring when:
- VIF > 3 for any pair
- R² between value ↔ momentum > 25%
- Candidate list stability < 80%

Currently the live scorer uses raw (non-residualized) scores. If the diagnostics above
flag high overlap, the next step is adding an `orthogonalize_value: true` flag to
`scoring.factors.value` config that applies OLS residualization before final
value_metric computation.

---

**Pearson vs Spearman**

Pearson measures linear correlation; Spearman measures rank-order correlation and is more
robust to outliers. For factor overlap diagnostics, Spearman ρ is the more relevant metric
because portfolio construction is rank-based, not magnitude-based.

---

**VIF interpretation**

| VIF   | Meaning                          |
|-------|----------------------------------|
| 1.0   | Completely independent           |
| 1–2   | Low overlap — no action needed   |
| 2–5   | Moderate — monitor               |
| > 5   | High — consider residualization  |
| > 10  | Severe multicollinearity         |
        """)
