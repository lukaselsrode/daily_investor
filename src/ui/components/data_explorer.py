"""
ui/components/data_explorer.py — Generic CSV explorer with charts.
Read-only. Never places orders or mutates config.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR, fmt_bin_index, list_csv_files

_PRESET_VIEWS = [
    "— custom —",
    "Top composite score by sector",
    "Score distributions (all factors)",
    "Momentum vs quality scatter",
    "Value vs momentum scatter",
    "Dividend yield vs yield_trap_flag",
    "Volume distribution",
    "Missing valuation by sector",
    "Sector-level mean scores",
    "Correlation matrix (factor scores)",
]

_SCORE_COLS = ["value_metric", "value_score", "quality_score", "income_score", "momentum_score"]


@st.cache_data(ttl=120)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def render() -> None:
    st.title("🗂️ Data Explorer")
    st.caption("Inspect raw and processed data. Read-only — no orders placed, no config written.")

    csv_files = list_csv_files()
    if not csv_files:
        st.warning(f"No CSV files found in `{DATA_DIR}`. Run the bot first.")
        return

    # ---- Dataset selector -------------------------------------------------
    c1, c2 = st.columns([3, 1])
    with c1:
        chosen_name = st.selectbox("Dataset", list(csv_files.keys()))
    with c2:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()

    df = _read_csv(str(csv_files[chosen_name]))
    num_cols = df.select_dtypes("number").columns.tolist()

    st.caption(f"{len(df)} rows × {len(df.columns)} columns | NaN cells: {df.isna().sum().sum()}")

    # ---- Preset views -----------------------------------------------------
    preset = st.selectbox("Quick view", _PRESET_VIEWS)

    if preset != "— custom —":
        _render_preset(df, preset, num_cols)
        st.divider()

    # ---- Filters ----------------------------------------------------------
    with st.expander("Filter & sort", expanded=(preset == "— custom —")):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sort_col = st.selectbox("Sort by", df.columns.tolist(), index=0)
            ascending = st.checkbox("Ascending", value=False)
        with fc2:
            filter_col = st.selectbox("Filter column", ["(none)"] + df.columns.tolist())
            if filter_col != "(none)" and filter_col in df.columns:
                if df[filter_col].dtype == object:
                    unique_vals = df[filter_col].dropna().unique().tolist()
                    filter_val = st.multiselect("Values", unique_vals, default=[])
                else:
                    lo = float(df[filter_col].min())
                    hi = float(df[filter_col].max())
                    filter_val = st.slider(f"{filter_col} range", lo, hi, (lo, hi))
            else:
                filter_val = None
        with fc3:
            col_select = st.multiselect("Show columns", df.columns.tolist(),
                                         default=df.columns.tolist()[:12])

    # Apply filter
    view = df.copy()
    if filter_col != "(none)" and filter_val is not None:
        col = filter_col
        if isinstance(filter_val, list) and filter_val:
            view = view[view[col].isin(filter_val)]
        elif isinstance(filter_val, tuple):
            view = view[view[col].between(*filter_val)]

    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending)

    show_cols = [c for c in col_select if c in view.columns] or view.columns.tolist()
    st.dataframe(view[show_cols], use_container_width=True, height=380)
    st.download_button("⬇ Download filtered CSV", view[show_cols].to_csv(index=False),
                       file_name="export.csv", mime="text/csv")

    # ---- Charts -----------------------------------------------------------
    st.divider()
    st.subheader("Visualizations")
    chart_type = st.selectbox("Chart type", ["Histogram", "Scatter", "Bar (grouped)", "Correlation heatmap"])

    if chart_type == "Histogram" and num_cols:
        hcol = st.selectbox("Column", num_cols)
        data = view[hcol].dropna()
        st.bar_chart(fmt_bin_index(data.value_counts(bins=30, sort=False).sort_index()))

    elif chart_type == "Scatter" and len(num_cols) >= 2:
        sc1, sc2, sc3 = st.columns(3)
        x = sc1.selectbox("X axis", num_cols, index=0)
        y = sc2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1))
        color_by = sc3.selectbox("Color by", ["(none)"] + df.select_dtypes(object).columns.tolist())
        scatter_df = view[[x, y]].dropna()
        try:
            import plotly.express as px
            color_col = color_by if color_by != "(none)" and color_by in view.columns else None
            sdf = view[[x, y] + ([color_col] if color_col else [])].dropna()
            fig = px.scatter(sdf, x=x, y=y, color=color_col, opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.scatter_chart(scatter_df, x=x, y=y)

    elif chart_type == "Bar (grouped)" and "sector" in view.columns and num_cols:
        bcol = st.selectbox("Value column", [c for c in _SCORE_COLS if c in view.columns] or num_cols)
        grouped = view.groupby("sector")[bcol].mean().sort_values(ascending=False)
        st.bar_chart(grouped)

    elif chart_type == "Correlation heatmap":
        score_cols = [c for c in _SCORE_COLS if c in view.columns]
        if len(score_cols) >= 2:
            corr = view[score_cols].corr()
            try:
                import plotly.express as px
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                title="Factor score correlations")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1))
        else:
            st.info("Need at least 2 score columns for correlation heatmap.")


def _render_preset(df: pd.DataFrame, preset: str, num_cols: list[str]) -> None:
    score_cols = [c for c in _SCORE_COLS if c in df.columns]

    # Presets below depend on factor-score columns; many CSVs (e.g. stock_tickers,
    # holdings) don't have them. Guard with a friendly message instead of crashing.
    _needs_scores = {
        "Score distributions (all factors)",
        "Sector-level mean scores",
        "Correlation matrix (factor scores)",
    }
    if preset in _needs_scores and not score_cols:
        st.info("This view needs factor-score columns (value_metric, *_score). "
                "Pick a scored dataset like `agg_data` to use it.")
        return

    if preset == "Top composite score by sector" and "sector" in df.columns and "value_metric" in df.columns:
        grouped = df.groupby("sector")["value_metric"].mean().sort_values(ascending=False)
        st.subheader("Mean composite score by sector")
        st.bar_chart(grouped)

    elif preset == "Score distributions (all factors)":
        tabs = st.tabs(score_cols)
        for tab, col in zip(tabs, score_cols):
            with tab:
                st.bar_chart(fmt_bin_index(df[col].dropna().value_counts(bins=25, sort=False).sort_index()))

    elif preset == "Momentum vs quality scatter" and "momentum_score" in df.columns and "quality_score" in df.columns:
        try:
            import plotly.express as px
            fig = px.scatter(df, x="quality_score", y="momentum_score",
                             color="sector" if "sector" in df.columns else None,
                             hover_data=["symbol"] if "symbol" in df.columns else None,
                             opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.scatter_chart(df[["quality_score", "momentum_score"]].dropna(),
                             x="quality_score", y="momentum_score")

    elif preset == "Dividend yield vs yield_trap_flag" and all(c in df.columns for c in ["dividend_yield", "yield_trap_flag"]):
        traps    = df[df["yield_trap_flag"].astype(bool)]["dividend_yield"].dropna()
        no_traps = df[~df["yield_trap_flag"].astype(bool)]["dividend_yield"].dropna()
        c1, c2 = st.columns(2)
        c1.metric("Yield traps — median yield", f"{traps.median():.2%}" if len(traps) else "—")
        c2.metric("Non-traps — median yield", f"{no_traps.median():.2%}" if len(no_traps) else "—")

    elif preset == "Sector-level mean scores" and "sector" in df.columns and score_cols:
        tbl = df.groupby("sector")[score_cols].mean().reset_index()
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in score_cols}), use_container_width=True)

    elif preset == "Correlation matrix (factor scores)" and len(score_cols) >= 2:
        corr = df[score_cols].corr()
        try:
            import plotly.express as px
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(corr)

    elif preset == "Volume distribution" and "volume" in df.columns:
        vol = df["volume"].dropna()
        st.bar_chart(fmt_bin_index(vol.clip(upper=vol.quantile(0.95)).value_counts(bins=30, sort=False).sort_index()))

    else:
        # Preset selected but this dataset lacks the columns it needs.
        st.info(f"'{preset}' isn't available for this dataset — it needs columns "
                "this CSV doesn't have (e.g. sector / specific factor scores / volume). "
                "Try the `agg_data` dataset or the — custom — view.")
