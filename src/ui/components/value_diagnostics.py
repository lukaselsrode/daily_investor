"""
ui/components/value_diagnostics.py — Value factor quality and predictive diagnostics.

Sections
--------
1. Distribution — value_score (v2) vs value_score_raw (legacy), tail reduction
2. Sector boxplots — sector-relative spread and bias
3. Outlier table — top / bottom 1% by value_score
4. Decile analysis — value decile vs mean recent returns (cross-sectional IC)
5. Factor correlation — value vs momentum, quality, income, volatility
6. Methodology note
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.utils import data_date, fmt_bin_index, load_latest_csv, no_data_msg

_SCORE_COLS = ["value_score", "quality_score", "income_score", "momentum_score"]
_RETURN_COLS = ["return_1m", "return_3m", "return_6m"]
_DIAG_COLS = ["value_score_raw", "sector_value_score", "relative_pe", "relative_pb"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _percentile_table(s: pd.Series, pcts=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> pd.DataFrame:
    rows = []
    for p in pcts:
        rows.append({"percentile": f"p{p}", "value": round(s.quantile(p / 100), 4)})
    rows.append({"percentile": "mean", "value": round(s.mean(), 4)})
    rows.append({"percentile": "std",  "value": round(s.std(),  4)})
    rows.append({"percentile": "skew", "value": round(s.skew(), 4)})
    rows.append({"percentile": "kurt", "value": round(float(s.kurtosis()), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("📐 Value Factor Diagnostics")
    st.caption(
        "Inspect value_score distribution, sector spread, predictive power, and factor correlations. "
        "Read-only — no orders, no config writes."
    )

    df = load_latest_csv("agg_data")
    if df is None:
        st.warning(no_data_msg("agg_data"))
        return

    # Coerce numeric columns
    for col in _SCORE_COLS + _RETURN_COLS + _DIAG_COLS + ["pe_ratio", "pb_ratio", "volume", "realized_vol_3m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    st.caption(f"Source: agg_data {data_date('agg_data')} | {len(df)} symbols")

    has_sect = "sector" in df.columns

    # ── 1. Distribution ──────────────────────────────────────────────────────
    st.subheader("1 · Score distribution")

    s = _num(df, "value_score").dropna()
    if s.empty:
        st.info("No data for value_score.")
    else:
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Mean",  f"{s.mean():.4f}")
        pc2.metric("Std",   f"{s.std():.4f}")
        pc3.metric("Skew",  f"{s.skew():.3f}")
        pc4.metric("Kurt",  f"{float(s.kurtosis()):.3f}")

        c_hist, c_pct = st.columns([2, 1])
        with c_hist:
            st.caption("value_score histogram")
            st.bar_chart(fmt_bin_index(s.value_counts(bins=30, sort=False).sort_index()))
        with c_pct:
            st.caption("Percentile table")
            st.dataframe(_percentile_table(s), use_container_width=True, hide_index=True)

    # (Legacy v2/raw side-by-side comparison removed — value_score_raw is no longer
    # written by any scorer.)

    # ── 2. Sector boxplots ───────────────────────────────────────────────────
    if has_sect:
        st.divider()
        st.subheader("2 · Sector distribution")
        score_col = st.selectbox("Score column", ["value_score"] + ([c for c in _DIAG_COLS if c in df.columns]), key="sect_score")
        sector_group = (
            df[["sector", score_col]].dropna()
            .groupby("sector")[score_col]
        )
        sector_stats = sector_group.agg(["mean", "std", "median", "count"]).round(4)
        sector_stats.columns = ["mean", "std", "median", "n"]
        sector_stats = sector_stats.sort_values("median", ascending=False)

        try:
            import plotly.express as px
            fig = px.box(
                df[["sector", score_col]].dropna(),
                x="sector", y=score_col,
                title=f"{score_col} by sector",
                color="sector",
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(sector_stats["median"])

        st.dataframe(sector_stats, use_container_width=True)

    # ── 3. Outlier table ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("3 · Top / bottom outliers")
    outlier_pct = st.slider("Outlier threshold (%)", 1, 10, 5, key="outlier_pct")

    vs = _num(df, "value_score").dropna()
    if not vs.empty and "symbol" in df.columns:
        hi_cut = vs.quantile(1 - outlier_pct / 100)
        lo_cut = vs.quantile(outlier_pct / 100)
        display_cols = [c for c in ["symbol", "sector", "pe_ratio", "pb_ratio",
                                     "value_score", "value_score_raw",
                                     "relative_pe", "relative_pb"] if c in df.columns]
        co1, co2 = st.columns(2)
        with co1:
            st.markdown(f"**Top {outlier_pct}% (highest value_score)** — genuinely cheap or distress?")
            top_df = df[df["value_score"] >= hi_cut][display_cols].sort_values("value_score", ascending=False)
            st.dataframe(top_df.reset_index(drop=True), use_container_width=True)
        with co2:
            st.markdown(f"**Bottom {outlier_pct}% (lowest value_score)** — expensive or no-data penalty?")
            bot_df = df[df["value_score"] <= lo_cut][display_cols].sort_values("value_score")
            st.dataframe(bot_df.reset_index(drop=True), use_container_width=True)

    # ── 4. Decile analysis ───────────────────────────────────────────────────
    st.divider()
    st.subheader("4 · Decile analysis")
    st.caption(
        "Decile 1 = cheapest 10% by value_score. "
        "Return columns are historical (not forward-looking). "
        "Monotonic return improvement from D1→D10 indicates value conflicts with recent momentum."
    )

    ret_col = st.selectbox("Return column", [c for c in _RETURN_COLS if c in df.columns], key="ret_col")
    if ret_col and "value_score" in df.columns:
        decile_df = df[["symbol", "value_score", ret_col] + (["sector"] if has_sect else [])].dropna(subset=["value_score", ret_col]).copy()
        if len(decile_df) >= 20:
            decile_df["decile"] = pd.qcut(decile_df["value_score"], q=10, labels=list(range(1, 11)), duplicates="drop")
            decile_df["decile"] = pd.to_numeric(decile_df["decile"], errors="coerce")
            decile_agg = (
                decile_df.groupby("decile")[ret_col]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "avg_return", "std": "return_std", "count": "n"})
                .round(4)
            )
            decile_agg["sharpe_proxy"] = (
                decile_agg["avg_return"] / decile_agg["return_std"].replace(0, float("nan"))
            ).round(3)
            decile_agg["hit_rate"] = (
                decile_df.groupby("decile")[ret_col].apply(lambda x: (x > 0).mean())
            ).round(3)

            try:
                import plotly.express as px
                fig = px.bar(
                    decile_agg.reset_index(),
                    x="decile", y="avg_return",
                    color="avg_return",
                    color_continuous_scale="RdYlGn",
                    title=f"Mean {ret_col} by value_score decile",
                    labels={"decile": "Value decile (1=cheapest)", "avg_return": f"Avg {ret_col}"},
                )
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(decile_agg["avg_return"])

            st.dataframe(decile_agg, use_container_width=True)

            # IC
            ic = decile_df[["value_score", ret_col]].corr(method="spearman").iloc[0, 1]
            st.metric(
                f"Spearman IC (value_score ↔ {ret_col})",
                f"{ic:.4f}",
                help="IC > 0: cheap stocks had better recent returns (momentum-value alignment). "
                     "IC < 0: cheap stocks underperformed recently (contrarian — typical).",
            )
        else:
            st.info("Not enough data for decile analysis (need ≥ 20 stocks with return data).")

    # ── 5. Factor correlation ────────────────────────────────────────────────
    st.divider()
    st.subheader("5 · Factor correlation")
    corr_cols = [c for c in _SCORE_COLS + _DIAG_COLS + _RETURN_COLS + ["pe_ratio", "pb_ratio", "realized_vol_3m"]
                 if c in df.columns]
    if len(corr_cols) >= 2:
        corr_method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
        corr = df[corr_cols].corr(method=corr_method).round(3)
        try:
            import plotly.express as px
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"Factor correlation matrix ({corr_method})",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                         use_container_width=True)

        # Highlight value_score correlations explicitly
        if "value_score" in corr.columns:
            st.caption("value_score correlations (sorted by |r|):")
            vc = (
                corr["value_score"]
                .drop("value_score")
                .sort_values(key=abs, ascending=False)
                .reset_index()
                .rename(columns={"index": "factor", "value_score": "correlation"})
            )
            vc["interpretation"] = vc["correlation"].apply(
                lambda r: "positive (aligned)" if r > 0.1 else ("negative (opposed)" if r < -0.1 else "weak / neutral")
            )
            st.dataframe(vc, use_container_width=True, hide_index=True)

    # ── 6. Peer-relative diagnostics (fallback coverage) ──────────────────────
    _peer_cols = [c for c in (
        "value_fallback_reason", "quality_fallback_reason",
        "momentum_fallback_reason", "income_fallback_reason",
    ) if c in df.columns]
    if _peer_cols:
        st.divider()
        st.subheader("6 · Peer-relative diagnostics")
        st.caption(
            "Industry → sector → market blended ranks. These tabs surface peer leaders "
            "and how often a finer peer group was unavailable."
        )

        peer_tab_leaders, peer_tab_fb = st.tabs(["Peer leaders", "Fallback coverage"])

        with peer_tab_leaders:
            if "value_metric" in df.columns and "symbol" in df.columns:
                lead_cols = [c for c in (
                    "symbol", "sector", "industry",
                    "value_metric", "quality_score", "momentum_score",
                    "value_score", "income_score",
                ) if c in df.columns]
                st.markdown("**Top 20 by composite (peer leaders)**")
                top = df.sort_values("value_metric", ascending=False).head(20)
                st.dataframe(top[lead_cols].reset_index(drop=True), use_container_width=True)

        with peer_tab_fb:
            fb_rows = []
            for fb_col in ("value_fallback_reason", "quality_fallback_reason",
                           "momentum_fallback_reason", "income_fallback_reason"):
                if fb_col in df.columns:
                    for reason, n in df[fb_col].value_counts(dropna=False).items():
                        fb_rows.append({"factor": fb_col, "fallback": str(reason), "n": int(n)})
            if fb_rows:
                st.dataframe(pd.DataFrame(fb_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "industry = primary group had ≥ min_group_size peers; "
                    "sector / market indicate fallback was used; legacy_checklist = "
                    "quality fell back to the raw checklist."
                )
            else:
                st.info("No fallback diagnostics available.")

    # ── 7. Methodology note ──────────────────────────────────────────────────
    st.divider()
    with st.expander("Methodology — peer-relative value scoring"):
        st.markdown("""
**Legacy (value_score_raw)**

`value_score = 0.60 × min(pe_threshold/pe_ratio, 5.0) + 0.40 × min(pb_threshold/pb_ratio, 5.0)`

Problems: ratio grows hyperbolically as PE→0; outlier PEs dominate cross-sectional ranking;
sector thresholds are static (from `ratios.yaml`) not data-driven.

---

**v2 (value_score — current)**

1. **Winsorize** PE and PB within each sector at the 5th/95th percentile — removes extreme tails.
2. **Sector-relative percentile rank** — stocks ranked against their own sector, not the full universe.
   Low PE within sector → high value rank.  Banks and semiconductors are never directly compared.
3. **Composite** = 0.60 × PE_rank + 0.40 × PB_rank, with availability-weighted fallback.
4. **Distress penalties**: PE ≤ 5 → −0.30 (ultra-cheap PE is often cyclical peak or distress);
   Negative PE → −0.25 (loss-making companies penalised regardless of PB).
5. **Clamped** to [−1.0, 1.5].

---

**Backtest note**

The backtest engine recomputes `value_score` internally from `pe_comp`/`pb_comp` stored in
`agg_data` and does not apply cross-sectional sector normalization.  Backtest results therefore
reflect the legacy ratio-based value signal.  Improving backtests to use v2 would require
historical fundamental data by period.
        """)
