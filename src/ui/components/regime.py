"""
ui/components/regime.py — Market regime classification and effective config.
"""

from __future__ import annotations

import streamlit as st

from ui.utils import load_config_raw


def render() -> None:
    st.title("🌡️ Regime & Risk")
    st.caption("Effective config and risk parameters by market regime.")

    cfg = load_config_raw()
    regime_cfg = cfg.get("regime", {})
    risk_cfg   = cfg.get("risk", {})
    base_index = cfg.get("index_pct", 0.65)

    # ---- Current regime selector (manual, for inspection) -----------------
    st.subheader("Regime inspector")
    st.info("The bot classifies regime live from SPY/VIX data. Use this panel to inspect what the effective config looks like under each regime.")

    regime = st.radio("Inspect regime", ["Bullish", "Neutral", "Defensive"], horizontal=True)

    # Compute effective values
    def_cfg  = regime_cfg.get("defensive", {})
    neut_cfg = regime_cfg.get("neutral", {})

    if regime == "Bullish":
        eff_index = base_index
        max_buys  = risk_cfg.get("max_buys_per_rebalance", 10)
        stop_adj  = 0.0
        etf_filter = False
    elif regime == "Neutral":
        eff_index = neut_cfg.get("index_pct_override") or base_index
        max_buys  = neut_cfg.get("max_buys_override") or risk_cfg.get("max_buys_per_rebalance", 10)
        stop_adj  = 0.0
        etf_filter = False
    else:  # Defensive
        eff_index = def_cfg.get("index_pct_override", 0.85)
        max_buys  = def_cfg.get("max_buys_override", 3)
        stop_adj  = def_cfg.get("stop_loss_tighten", 0.05)
        etf_filter = True

    base_stop = cfg.get("sell_rules", {}).get("stop_loss_pct", -0.20)
    eff_stop  = base_stop + stop_adj

    st.subheader(f"Effective config — {regime}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ETF allocation", f"{eff_index:.0%}", f"{eff_index - base_index:+.0%} vs base" if eff_index != base_index else "base")
    c2.metric("Max active buys", max_buys)
    c3.metric("Stop-loss pct", f"{eff_stop:.0%}", f"{stop_adj:+.0%} vs base" if stop_adj else "base")
    c4.metric("ETF MA filter", "ACTIVE" if etf_filter else "off")

    # ---- Thresholds -------------------------------------------------------
    st.divider()
    st.subheader("Regime thresholds (from config)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("| Threshold | Value |")
        st.markdown("|---|---|")
        st.markdown(f"| SPY MA period | {regime_cfg.get('spy_ma_period', 200)} days |")
        st.markdown(f"| VIX neutral | {regime_cfg.get('vix_neutral_threshold', 20)} |")
        st.markdown(f"| VIX defensive | {regime_cfg.get('vix_defensive_threshold', 30)} |")
    with c2:
        st.markdown("| Condition | Regime |")
        st.markdown("|---|---|")
        st.markdown("| SPY > 200DMA AND VIX < 20 | **Bullish** |")
        st.markdown("| VIX 20–30 or SPY below 200DMA | **Neutral** |")
        st.markdown("| VIX ≥ 30 | **Defensive** |")

    # ---- Risk limits -------------------------------------------------------
    st.divider()
    st.subheader("Risk limits (current config)")
    c1, c2 = st.columns(2)
    with c1:
        for k, v in {
            "max_single_position_pct": f"{risk_cfg.get('max_single_position_pct', '—'):.0%}" if isinstance(risk_cfg.get("max_single_position_pct"), float) else "—",
            "max_sector_pct":          f"{risk_cfg.get('max_sector_pct', '—'):.0%}" if isinstance(risk_cfg.get("max_sector_pct"), float) else "—",
            "max_order_pct_of_cash":   f"{risk_cfg.get('max_order_pct_of_cash', '—'):.0%}" if isinstance(risk_cfg.get("max_order_pct_of_cash"), float) else "—",
        }.items():
            st.metric(k, v)
    with c2:
        for k in ["min_order_amount", "min_liquidity_volume", "max_buys_per_rebalance"]:
            st.metric(k, risk_cfg.get(k, "—"))
