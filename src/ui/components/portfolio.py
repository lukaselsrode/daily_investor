"""
ui/components/portfolio.py — Portfolio holdings and allocation viewer.
Loads from robinhood_data CSV (always available without credentials).
Live broker data requires login and is fetched on demand.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from ui.utils import data_date, load_config_raw, load_latest_csv, no_data_msg


def render() -> None:
    st.title("💼 Portfolio")
    st.caption("Holdings and allocation. Loaded from cached robinhood_data CSV by default.")

    cfg = load_config_raw()
    etfs = cfg.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"])
    index_pct = cfg.get("index_pct", 0.65)

    df = load_latest_csv("robinhood_data")
    if df is None:
        st.warning(no_data_msg("robinhood_data"))
        return

    st.caption(f"Source: robinhood_data {data_date('robinhood_data')} | {len(df)} positions")

    # ---- Sleeve classification --------------------------------------------
    if "symbol" in df.columns:
        df["sleeve"] = df["symbol"].apply(lambda s: "ETF/core" if s in etfs else "active")
    else:
        df["sleeve"] = "unknown"

    # ---- Summary cards ----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    if "value_metric" in df.columns:
        c1.metric("Universe rows", len(df))
    if "sector" in df.columns:
        c2.metric("Sectors", df["sector"].nunique())
    etf_rows   = df[df["sleeve"] == "ETF/core"] if "sleeve" in df.columns else df.head(0)
    active_rows = df[df["sleeve"] == "active"]  if "sleeve" in df.columns else df.head(0)
    c3.metric("ETF sleeve rows", len(etf_rows))
    c4.metric("Active sleeve rows", len(active_rows))

    # ---- Sector exposure (from agg_data if available) ---------------------
    agg = load_latest_csv("agg_data")
    if agg is not None and "sector" in agg.columns and "value_metric" in agg.columns:
        st.subheader("Sector distribution (scored universe)")
        sector_counts = agg.groupby("sector")["value_metric"].agg(["count", "mean"]).reset_index()
        sector_counts.columns = ["sector", "count", "avg_value_metric"]
        sector_counts = sector_counts.sort_values("count", ascending=False)
        st.bar_chart(sector_counts.set_index("sector")["count"])
        with st.expander("Sector detail table"):
            st.dataframe(sector_counts, use_container_width=True)

    # ---- Holdings table ---------------------------------------------------
    st.subheader("Holdings (from robinhood_data CSV)")
    display_cols = [c for c in [
        "symbol", "sleeve", "sector", "industry", "current_price",
        "value_metric", "quality_score", "momentum_score", "volume",
    ] if c in df.columns]
    sort_col = st.selectbox("Sort by", display_cols, index=0)
    st.dataframe(df[display_cols].sort_values(sort_col, ascending=False), use_container_width=True, height=400)

    # ---- Live data (optional) ---------------------------------------------
    st.divider()
    st.subheader("Live broker data")
    live = st.session_state.get("live_enabled", False)
    if not live:
        st.info("🔒 Live execution is OFF. Enable it in the sidebar to fetch live holdings from Robinhood.")
        return

    if st.button("Fetch live holdings from Robinhood"):
        with st.spinner("Connecting to Robinhood…"):
            try:
                from main import login, get_current_positions, get_available_cash, get_portfolio_value
                login()
                positions = get_current_positions()
                cash = get_available_cash()
                port_val = get_portfolio_value()
                st.session_state["live_positions"] = positions
                st.session_state["live_cash"] = cash
                st.session_state["live_port_val"] = port_val
                st.success("✅ Live data fetched.")
            except Exception as exc:
                st.error(f"Failed to fetch live data: {exc}")

    if "live_positions" in st.session_state:
        pos = st.session_state["live_positions"]
        cash = st.session_state["live_cash"]
        port_val = st.session_state["live_port_val"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio value", f"${port_val:,.2f}")
        c2.metric("Available cash", f"${cash:,.2f}")
        c3.metric("Positions", len(pos))
        if pos:
            pos_df = pd.DataFrame([
                {"symbol": sym, **data} for sym, data in pos.items()
            ] if isinstance(pos, dict) else pos)
            st.dataframe(pos_df, use_container_width=True)
