"""ui/components/odte_fmp.py — 0DTE FMP single-name squeeze/meme SANITY context.

Read-only FMP fundamentals for a meme/squeeze sanity check — NOT an entry signal, NO orders, NO
options/gamma (Robinhood stays the gamma source). Fail-closed without FMP_KEY (never printed).
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300)
def _fmp_context(symbol: str, allow_fetch: bool) -> dict:
    from data.odte_fmp_context import run_fmp_context
    return run_fmp_context(symbol, allow_fetch=allow_fetch)


def render() -> None:
    st.subheader("FMP Context")
    st.caption("Meme/squeeze **sanity** only — not an entry signal, no orders, no options/gamma.")

    c1, c2 = st.columns([2, 1])
    symbol = c1.text_input("Symbol", value="").strip().upper()
    allow_fetch = c2.toggle("Fetch live", value=False, help="Off = offline (no FMP call).")

    if not st.button("🔎 Fetch context", type="primary"):
        return
    if not symbol:
        st.error("Enter a symbol.")
        return

    from data.social_sentiment import is_restricted_underlying
    if is_restricted_underlying(symbol):
        st.warning(f"🚫 {symbol} is employer-restricted — context only, never tradeable.")

    with st.spinner(f"Fetching FMP context for {symbol}…"):
        try:
            ctx = _fmp_context(symbol, allow_fetch)
        except Exception as exc:
            st.error(f"FMP context failed: {exc}")
            return

    m1, m2, m3 = st.columns(3)
    m1.metric("Squeeze profile", str(ctx.get("squeeze_profile", "—")))
    m2.metric("Price", ctx.get("price", "—"))
    m3.metric("Rel. volume", ctx.get("relative_volume", "—"))
    if ctx.get("trade_implication"):
        st.info(ctx["trade_implication"])
    st.caption(f"FMP options available: {ctx.get('fmp_options_available', False)} "
               "(Robinhood remains the gamma source).")

    for w in ctx.get("warnings", []) or []:
        st.caption(f"⚠️ {w}")

    with st.expander("Full context JSON"):
        st.json(ctx)
