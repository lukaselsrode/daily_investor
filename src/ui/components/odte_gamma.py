"""ui/components/odte_gamma.py — 0DTE gamma / pin concentration map.

Paste the Robinhood option-quote rows (or the separate quotes + instruments arrays) Hermes exported
and build the ABSOLUTE gamma/OI concentration map. PURE/OFFLINE — no broker, no network. Honest by
construction: this is pin-risk concentration, NOT dealer net GEX.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_SAMPLE = (
    '{"underlying": "SPY", "spot": 505.0, "expiration": "2026-06-25", "rows": [\n'
    '  {"strike": 503, "option_type": "call", "open_interest": 1200, "gamma": 0.05, "volume": 800},\n'
    '  {"strike": 505, "option_type": "call", "open_interest": 5400, "gamma": 0.08, "volume": 4200},\n'
    '  {"strike": 505, "option_type": "put",  "open_interest": 4800, "gamma": 0.07, "volume": 3900},\n'
    '  {"strike": 507, "option_type": "put",  "open_interest": 2100, "gamma": 0.04, "volume": 1100}\n'
    ']}'
)


def _plot_by_strike(gmap: dict) -> go.Figure | None:
    by_strike = gmap.get("by_strike") or []
    if not by_strike:
        return None
    df = pd.DataFrame(by_strike).sort_values("strike")
    has_gamma = gmap.get("gamma_available") and "total_gamma_notional_1pct" in df.columns
    fig = go.Figure()
    if has_gamma:
        fig.add_bar(x=df["strike"], y=df.get("call_gamma_notional_1pct", 0), name="call γ-notional",
                    marker_color="#2ecc71")
        fig.add_bar(x=df["strike"], y=df.get("put_gamma_notional_1pct", 0), name="put γ-notional",
                    marker_color="#e74c3c")
        ytitle = "gamma notional (1%)"
    else:
        fig.add_bar(x=df["strike"], y=df.get("call_oi", 0), name="call OI", marker_color="#2ecc71")
        fig.add_bar(x=df["strike"], y=df.get("put_oi", 0), name="put OI", marker_color="#e74c3c")
        ytitle = "open interest"

    spot = gmap.get("spot")
    if isinstance(spot, (int, float)):
        fig.add_vline(x=spot, line_dash="dot", line_color="#3498db", annotation_text="spot")
    for key, color, label in (("call_wall", "#27ae60", "call wall"),
                              ("put_wall", "#c0392b", "put wall"),
                              ("max_gamma_strike", "#f39c12", "max-γ")):
        v = gmap.get(key)
        if isinstance(v, (int, float)):
            fig.add_vline(x=v, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(barmode="group", height=380, yaxis_title=ytitle, xaxis_title="strike",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render() -> None:
    st.subheader("Gamma / Pin Map")
    st.caption("ABSOLUTE gamma/OI concentration — **not** dealer net GEX. Robinhood doesn't expose "
               "dealer positioning, so this is a pin-risk heuristic only.")

    mode = st.radio("Input", ["Combined JSON (rows + meta)", "Separate quotes + instruments"],
                    horizontal=True)

    rows_obj = None
    if mode.startswith("Combined"):
        txt = st.text_area("Paste combined JSON", value=_SAMPLE, height=220)
        raw_input = txt
    else:
        q = st.text_area("Quotes JSON (array)", height=150, placeholder='[{"instrument": "...", ...}]')
        i = st.text_area("Instruments JSON (array)", height=120,
                         placeholder='[{"id": "...", "strike_price": "505", "type": "call", ...}]')
        raw_input = None
        rows_obj = (q, i)

    c1, c2, c3 = st.columns(3)
    spot = c1.number_input("Spot override", min_value=0.0, value=0.0, step=0.5,
                           help="0 = infer from input")
    underlying = c2.text_input("Underlying override", value="")
    expiration = c3.text_input("Expiration override", value="")

    if not st.button("🧲 Build gamma map", type="primary"):
        return

    try:
        from data.odte_gamma_map import build_gamma_map, rh_rows_from_quotes, run_gamma_map
        if mode.startswith("Combined"):
            gmap = run_gamma_map(
                input_json=raw_input,
                spot=spot or None,
                underlying=underlying or None,
                expiration=expiration or None,
            )
        else:
            q, i = rows_obj
            quotes = json.loads(q) if q.strip() else []
            instruments = json.loads(i) if i.strip() else None
            rows = rh_rows_from_quotes(quotes, instruments)
            gmap = build_gamma_map(
                rows, spot=spot or None,
                underlying=underlying or None,
                expiration=expiration or None,
                instruments=instruments,
            )
    except Exception as exc:
        st.error(f"Could not build map: {exc}")
        return

    st.info(f"🏷️ `gamma_regime`: {gmap.get('gamma_regime')} — {gmap.get('disclaimer', '')}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot", gmap.get("spot", "—"))
    m2.metric("Call wall", gmap.get("call_wall", "—"))
    m3.metric("Put wall", gmap.get("put_wall", "—"))
    m4.metric("Max-γ strike", gmap.get("max_gamma_strike", "—"))

    pin = gmap.get("pin_risk") or {}
    em = gmap.get("expected_move") or {}
    p1, p2 = st.columns(2)
    p1.metric("Pin risk", str(pin.get("level", "—")), help=str(pin.get("reason", "")))
    if em.get("available"):
        p2.metric("Expected move", f"{em.get('lower','?')} – {em.get('upper','?')}")
    else:
        p2.metric("Expected move", "—", help=str(em.get("reason", "")))

    fresh = gmap.get("freshness") or {}
    if fresh and not fresh.get("quote_fresh", True):
        st.warning(f"⏳ Quotes stale: {fresh.get('reason', '')} (age {fresh.get('age_minutes','?')}m)")

    fig = _plot_by_strike(gmap)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full map JSON"):
        st.json(gmap)
