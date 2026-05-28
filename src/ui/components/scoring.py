"""
ui/components/scoring.py — Scored universe explorer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.utils import data_date, fmt_bin_index, load_config_raw, load_latest_csv, no_data_msg

_SCORE_COLS = ["value_metric", "value_score", "quality_score", "income_score", "momentum_score"]
_META_COLS  = ["symbol", "owned", "sector", "industry", "pe_ratio", "pb_ratio",
               "dividend_yield", "volume", "current_price", "yield_trap_flag"]
_FRIENDLY   = {
    "value_metric":   "Composite Score",
    "value_score":    "Value",
    "quality_score":  "Quality",
    "income_score":   "Income",
    "momentum_score": "Momentum",
}


def render() -> None:
    st.title("🔬 Scoring Explorer")
    st.caption("Browse and filter the latest scored universe. Read-only — no orders placed.")

    df = load_latest_csv("agg_data")
    if df is None:
        st.warning(no_data_msg("agg_data"))
        return

    cfg = load_config_raw()
    thresh = cfg.get("metric_threshold", 0.0)

    # Cross-reference holdings CSV to tag owned symbols
    holdings_df = load_latest_csv("holdings")
    owned_symbols: set[str] = set()
    if holdings_df is not None and "symbol" in holdings_df.columns:
        owned_symbols = set(holdings_df["symbol"].dropna().tolist())
    if "symbol" in df.columns:
        df["owned"] = df["symbol"].isin(owned_symbols)

    st.caption(f"Source: agg_data {data_date('agg_data')} | {len(df)} symbols | composite score threshold = {thresh} | owned: {len(owned_symbols)}")

    # ---- Filters ----------------------------------------------------------
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sym_filter = st.text_input("Symbol search").upper().strip()
        with c2:
            sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else ["All"]
            sector_filter = st.selectbox("Sector", sectors)
        with c3:
            min_metric = st.slider("Min composite score", -1.0, 2.0, float(thresh), 0.05)
        with c4:
            hide_yield_traps = st.checkbox("Hide yield traps", value=False)
            candidates_only  = st.checkbox("Buy candidates only", value=False)
            owned_only       = st.checkbox("Owned positions only", value=False)

    mask = pd.Series([True] * len(df), index=df.index)
    if sym_filter:
        mask &= df["symbol"].str.contains(sym_filter, case=False, na=False)
    if sector_filter != "All" and "sector" in df.columns:
        mask &= df["sector"] == sector_filter
    if "value_metric" in df.columns:
        mask &= df["value_metric"] >= min_metric
    if hide_yield_traps and "yield_trap_flag" in df.columns:
        mask &= ~df["yield_trap_flag"].astype(bool)
    if candidates_only and "value_metric" in df.columns:
        mask &= df["value_metric"] >= thresh
    if owned_only and "owned" in df.columns:
        mask &= df["owned"]

    view = df[mask].copy()
    st.write(f"**{len(view)}** symbols match filters")

    # ---- Score distribution -----------------------------------------------
    if "value_metric" in view.columns and len(view):
        st.subheader("Composite score distribution")
        st.bar_chart(fmt_bin_index(view["value_metric"].dropna().value_counts(bins=20, sort=False).sort_index()))

    # ---- Main table -------------------------------------------------------
    display_cols = [c for c in _META_COLS + _SCORE_COLS if c in view.columns]
    st.subheader("Universe table")
    _sort_opts = [c for c in _SCORE_COLS if c in view.columns] or display_cols
    sort_col = st.selectbox(
        "Sort by",
        _sort_opts,
        format_func=lambda c: _FRIENDLY.get(c, c),
        index=0,
    )
    view_sorted = view[display_cols].sort_values(sort_col, ascending=False) if sort_col in view.columns else view[display_cols]
    st.dataframe(view_sorted, use_container_width=True, height=400)

    # ---- Symbol drill-down ------------------------------------------------
    st.divider()
    st.subheader("Symbol detail")
    symbols = view["symbol"].dropna().unique().tolist() if "symbol" in view.columns else []
    if symbols:
        chosen = st.selectbox("Select symbol", sorted(symbols))
        row = view[view["symbol"] == chosen]
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Factor scores**")
                for col in _SCORE_COLS:
                    if col in r:
                        v = r[col]
                        st.metric(_FRIENDLY.get(col, col), f"{v:.4f}" if pd.notna(v) else "—")
            with c2:
                st.markdown("**Fundamentals**")
                for col in ["pe_ratio", "pb_ratio", "dividend_yield", "volume"]:
                    if col in r:
                        st.metric(col, r[col] if pd.notna(r[col]) else "—")
            with c3:
                st.markdown("**Flags**")
                if "yield_trap_flag" in r:
                    st.metric("yield_trap", "⚠️ YES" if r["yield_trap_flag"] else "✅ no")
                if "buy_to_sell_ratio" in r:
                    st.metric("buy_to_sell_ratio", r["buy_to_sell_ratio"] if pd.notna(r["buy_to_sell_ratio"]) else "—")
                if "current_price" in r:
                    st.metric("current_price", f"${r['current_price']:.2f}" if pd.notna(r["current_price"]) else "—")
                if "owned" in r:
                    st.metric("owned", "✅ Yes" if r["owned"] else "No")

        # Holdings detail for owned symbols
        if "owned" in r and r["owned"] and holdings_df is not None:
            h = holdings_df[holdings_df["symbol"] == chosen]
            if not h.empty:
                st.divider()
                st.markdown("**Holdings detail**")
                hr = h.iloc[0]
                hc1, hc2, hc3, hc4 = st.columns(4)
                for col in ["equity", "percent_change", "equity_change", "average_buy_price"]:
                    if col in hr:
                        hr[col] = pd.to_numeric(hr[col], errors="coerce")
                hc1.metric("Equity", f"${float(hr['equity']):,.2f}" if pd.notna(hr.get("equity")) else "—")
                hc2.metric("Qty", f"{float(hr['quantity']):.4f}" if pd.notna(hr.get("quantity")) else "—")
                hc3.metric("Avg buy price", f"${float(hr['average_buy_price']):,.2f}" if pd.notna(hr.get("average_buy_price")) else "—")
                hc4.metric("% change", f"{float(hr['percent_change']):+.2f}%" if pd.notna(hr.get("percent_change")) else "—")
    else:
        st.info("No symbols match current filters.")
