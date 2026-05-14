"""
ui/components/portfolio.py — Actual Robinhood holdings viewer.

Primary source: holdings_YYYY_MM_DD.csv written by save_holdings_csv()
  (columns: symbol, name, quantity, average_buy_price, equity,
            percent_change, equity_change, percentage, current_price, type, pe_ratio, id)

Sector / score enrichment: cross-referenced from latest agg_data CSV.
Live fetch: available when live execution is enabled.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from ui.utils import DATA_DIR, data_date, load_config_raw, load_latest_csv, no_data_msg


_FLOAT_COLS = [
    "quantity", "average_buy_price", "equity",
    "percent_change", "equity_change", "percentage", "current_price", "pe_ratio",
]


def _coerce_floats(df: pd.DataFrame) -> pd.DataFrame:
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render() -> None:
    st.title("💼 Portfolio")
    st.caption("Your actual Robinhood holdings. Refreshed each time the bot runs.")

    cfg = load_config_raw()
    etfs = cfg.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"])

    df = load_latest_csv("holdings")
    if df is None:
        st.warning(
            no_data_msg("holdings")
            + "  \nHoldings are saved automatically when the bot runs (`daily-investor run`)."
        )
        _live_section(etfs)
        return

    df = _coerce_floats(df)

    # ---- Sleeve tag ----------------------------------------------------------
    if "symbol" in df.columns:
        df["sleeve"] = df["symbol"].apply(lambda s: "ETF/core" if s in etfs else "active")

    # ---- Sector enrichment from agg_data ------------------------------------
    agg = load_latest_csv("agg_data")
    if agg is not None and "symbol" in agg.columns and "sector" in agg.columns:
        sector_map = agg.set_index("symbol")["sector"].to_dict()
        df["sector"] = df["symbol"].map(sector_map)

    st.caption(f"Source: holdings {data_date('holdings')} | {len(df)} positions")

    # ---- Summary metrics -----------------------------------------------------
    total_equity   = df["equity"].sum()       if "equity"      in df.columns else None
    total_pct_port = df["percentage"].sum()   if "percentage"  in df.columns else None
    n_etf          = (df["sleeve"] == "ETF/core").sum() if "sleeve" in df.columns else 0
    n_active       = (df["sleeve"] == "active").sum()   if "sleeve" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positions",        len(df))
    c2.metric("Total equity",     f"${total_equity:,.2f}"  if total_equity is not None else "—")
    c3.metric("ETF / core",       n_etf)
    c4.metric("Active positions", n_active)

    # ---- P&L overview --------------------------------------------------------
    if "percent_change" in df.columns:
        winners = (df["percent_change"] > 0).sum()
        losers  = (df["percent_change"] < 0).sum()
        med_chg = df["percent_change"].median()
        st.divider()
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Winners / losers", f"{winners} / {losers}")
        pc2.metric("Median % change", f"{med_chg:+.2f}%")
        if "equity_change" in df.columns:
            total_gain = df["equity_change"].sum()
            pc3.metric("Total unrealised P&L", f"${total_gain:+,.2f}")

    # ---- Holdings table ------------------------------------------------------
    st.divider()
    st.subheader("Holdings detail")
    display_cols = [c for c in [
        "symbol", "name", "sleeve", "sector",
        "quantity", "current_price", "average_buy_price",
        "equity", "percent_change", "equity_change", "percentage",
    ] if c in df.columns]

    sort_col = st.selectbox("Sort by", display_cols, index=display_cols.index("equity") if "equity" in display_cols else 0)
    ascending = st.checkbox("Ascending", value=False)

    view = df[display_cols].sort_values(sort_col, ascending=ascending)

    # Colour percent_change column
    def _colour_pct(val):
        if pd.isna(val):
            return ""
        return "color: green" if val > 0 else ("color: red" if val < 0 else "")

    fmt = {}
    for col in ["current_price", "average_buy_price", "equity", "equity_change"]:
        if col in view.columns:
            fmt[col] = "${:,.2f}"
    for col in ["percent_change", "percentage"]:
        if col in view.columns:
            fmt[col] = "{:.2f}%"
    if "quantity" in view.columns:
        fmt["quantity"] = "{:.4f}"

    styled = view.style.format(fmt)
    if "percent_change" in view.columns:
        styled = styled.map(_colour_pct, subset=["percent_change"])

    st.dataframe(styled, use_container_width=True, height=420)
    st.download_button(
        "⬇ Download holdings CSV",
        data=view.to_csv(index=False),
        file_name="holdings_export.csv",
        mime="text/csv",
    )

    # ---- Sector allocation (from holdings) -----------------------------------
    if "sector" in df.columns and "equity" in df.columns:
        st.divider()
        st.subheader("Sector allocation")
        sector_equity = (
            df.dropna(subset=["sector"])
            .groupby("sector")["equity"]
            .sum()
            .sort_values(ascending=False)
        )
        if not sector_equity.empty:
            try:
                import plotly.express as px
                fig = px.pie(
                    sector_equity.reset_index(),
                    names="sector", values="equity",
                    title="Holdings by sector (equity $)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(sector_equity)

    # ---- Sleeve split --------------------------------------------------------
    if "sleeve" in df.columns and "equity" in df.columns:
        st.divider()
        st.subheader("ETF vs active split")
        sleeve_equity = df.groupby("sleeve")["equity"].sum()
        sc1, sc2 = st.columns(2)
        for i, (sleeve, val) in enumerate(sleeve_equity.items()):
            [sc1, sc2][i % 2].metric(sleeve, f"${val:,.2f}")

    # ---- Dividend income -----------------------------------------------------
    _dividend_section()

    _live_section(etfs)


def _dividend_section() -> None:
    div_df = load_latest_csv("dividend_history")
    if div_df is None:
        return

    st.divider()
    st.subheader("Dividend income")

    div_df["amount"] = pd.to_numeric(div_df.get("amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    paid = div_df[div_df.get("state", pd.Series()).eq("paid")] if "state" in div_df.columns else div_df
    total_paid = paid["amount"].sum()

    d1, d2, d3 = st.columns(3)
    d1.metric("Total dividends received", f"${total_paid:,.2f}")
    d2.metric("Dividend payments", len(paid))
    if "symbol" in div_df.columns:
        top_sym = (
            paid.groupby("symbol")["amount"].sum().sort_values(ascending=False).head(1)
        )
        if not top_sym.empty:
            d3.metric("Top payer", f"{top_sym.index[0]}  ${top_sym.iloc[0]:.2f}")

    if "symbol" in paid.columns and not paid.empty:
        by_symbol = paid.groupby("symbol")["amount"].sum().sort_values(ascending=False).head(15)
        st.bar_chart(by_symbol)


def _live_section(etfs: list[str]) -> None:
    st.divider()
    st.subheader("Live broker data")
    live = st.session_state.get("live_enabled", False)
    if not live:
        st.info("🔒 Live execution is OFF. Enable it in the sidebar to fetch live holdings from Robinhood.")
        return

    if st.button("Fetch live holdings from Robinhood"):
        with st.spinner("Connecting to Robinhood…"):
            try:
                from main import login, get_current_positions, get_available_cash, get_portfolio_value, save_holdings_csv
                login()
                holdings = get_current_positions()
                save_holdings_csv(holdings)
                cash      = get_available_cash()
                port_val  = get_portfolio_value()
                st.session_state["live_holdings"]  = holdings
                st.session_state["live_cash"]      = cash
                st.session_state["live_port_val"]  = port_val
                st.success("✅ Live data fetched and saved to holdings CSV.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to fetch live data: {exc}")

    if "live_holdings" in st.session_state:
        holdings = st.session_state["live_holdings"]
        cash     = st.session_state["live_cash"]
        port_val = st.session_state["live_port_val"]
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Portfolio value",  f"${port_val:,.2f}")
        lc2.metric("Available cash",   f"${cash:,.2f}")
        lc3.metric("Positions (live)", len(holdings))
        if holdings:
            pos_df = pd.DataFrame([
                {"symbol": sym, **{k: v for k, v in data.items()}}
                for sym, data in holdings.items()
            ])
            pos_df = _coerce_floats(pos_df)
            etfs_list = etfs
            if "symbol" in pos_df.columns:
                pos_df["sleeve"] = pos_df["symbol"].apply(lambda s: "ETF/core" if s in etfs_list else "active")
            st.dataframe(pos_df, use_container_width=True)
