"""ui/components/odte_position.py — 0DTE live-position monitor + plan authoring.

Three things, all DECISION-ONLY (no orders, no broker calls):
  1. Show the last Hermes-fed decision (data/odte/position_decision.json).
  2. Manual snapshot form → evaluate_position() live in-app (the repo can't fetch broker marks).
  3. Plan editor → author/edit data/odte/active_trade.json. NVDA is hard-blocked here.
"""
from __future__ import annotations

import json

import streamlit as st

from ui.utils import ODTE_DATA_DIR, load_odte_json

_MODES = ["scalp", "trend", "lotto", "runner"]
_PLAN_PATH = ODTE_DATA_DIR / "active_trade.json"


def _f(v) -> float | None:
    try:
        return float(v) if v not in (None, "", 0) else None
    except (TypeError, ValueError):
        return None


def _render_last_decision() -> None:
    st.markdown("### 📟 Last decision (Hermes-fed)")
    decision = load_odte_json("position_decision.json")
    if not decision:
        st.caption("No `position_decision.json` yet — Hermes writes it via `odte-position`.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", decision.get("decision", "—"))
    pnl = decision.get("pnl_pct")
    c2.metric("P/L", f"{float(pnl):+.0%}" if isinstance(pnl, (int, float)) else "—")
    c3.metric("Underlying", decision.get("underlying", "—"))
    st.caption(f"As of {decision.get('ts', '—')} · snapshot: {decision.get('snapshot_status','?')}")
    with st.expander("Decision payload"):
        st.json(decision)


def _render_manual_eval(plan: dict) -> None:
    st.markdown("### 🧪 Manual snapshot → live evaluate")
    st.caption("Type current broker/market values to re-evaluate the plan in-app (pure, no write).")
    if not plan:
        st.info("No active plan — author one below first.")
        return
    with st.form("odte_snapshot"):
        c1, c2, c3 = st.columns(3)
        mark = c1.number_input("Option mark", min_value=0.0, value=0.0, step=0.01)
        bid = c2.number_input("Option bid", min_value=0.0, value=0.0, step=0.01)
        und = c3.number_input("Underlying last", min_value=0.0, value=0.0, step=0.1)
        c4, c5, c6 = st.columns(3)
        spy = c4.number_input("SPY last", min_value=0.0, value=0.0, step=0.1)
        qqq = c5.number_input("QQQ last", min_value=0.0, value=0.0, step=0.1)
        vix = c6.number_input("VIX", min_value=0.0, value=0.0, step=0.1)
        c7, c8 = st.columns(2)
        vixy = c7.number_input("VIXY", min_value=0.0, value=0.0, step=0.1)
        monitoring_ok = c8.checkbox("Monitoring OK", value=True)
        submitted = st.form_submit_button("Evaluate", type="primary")

    if not submitted:
        return
    snapshot = {
        "option_mark": _f(mark), "option_bid": _f(bid), "underlying_last": _f(und),
        "spy_last": _f(spy), "qqq_last": _f(qqq), "vix": _f(vix), "vixy": _f(vixy),
        "monitoring_ok": monitoring_ok,
    }
    snapshot = {k: v for k, v in snapshot.items() if v is not None or k == "monitoring_ok"}
    try:
        from data.odte_position import evaluate_position
        result = evaluate_position(plan, snapshot)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return
    dec = result.get("decision", "—")
    st.metric("Decision", dec)
    if result.get("triggers"):
        st.json(result["triggers"])
    else:
        st.success("HOLD — no triggers fired")


def _render_plan_editor(plan: dict) -> None:
    st.markdown("### ✏️ Active trade plan")
    st.caption(f"Authors `{_PLAN_PATH}`. Decision-only — never places an order. NVDA is blocked.")
    p = plan or {}
    thesis = p.get("thesis") or {}
    time_rules = p.get("time_rules") or {}
    profit_rules = p.get("profit_rules") or {}

    with st.form("odte_plan"):
        c1, c2, c3 = st.columns(3)
        status = c1.selectbox("Status", ["open", "closed", "flat"],
                              index=["open", "closed", "flat"].index(p.get("status", "open"))
                              if p.get("status", "open") in ("open", "closed", "flat") else 0)
        mode = c2.selectbox("Mode", _MODES,
                            index=_MODES.index(p.get("mode")) if p.get("mode") in _MODES else 0)
        underlying = c3.text_input("Underlying", value=str(p.get("underlying", "")))

        c4, c5, c6 = st.columns(3)
        option_type = c4.selectbox("Option type", ["call", "put"],
                                   index=0 if str(p.get("option_type", "call")) == "call" else 1)
        strike = c5.number_input("Strike", min_value=0.0, value=float(p.get("strike") or 0.0), step=0.5)
        expiration = c6.text_input("Expiration", value=str(p.get("expiration", "")))

        c7, c8, c9 = st.columns(3)
        entry_price = c7.number_input("Entry price", min_value=0.0,
                                      value=float(p.get("entry_price") or 0.0), step=0.01)
        quantity = c8.number_input("Quantity", min_value=0, value=int(p.get("quantity") or 1), step=1)
        bid_floor = c9.number_input("Bid floor", min_value=0.0,
                                    value=float(p.get("bid_floor") or 0.05), step=0.01)

        st.markdown("**Profit rules** (blank = mode default)")
        c10, c11 = st.columns(2)
        tp = c10.number_input("Take-profit %", min_value=0.0, max_value=5.0,
                              value=float(profit_rules.get("take_profit_pct") or 0.0), step=0.05)
        se = c11.number_input("Strong-exit %", min_value=0.0, max_value=5.0,
                              value=float(profit_rules.get("strong_exit_pct") or 0.0), step=0.05)

        st.markdown("**Thesis stops** (0 = unset)")
        c12, c13, c14 = st.columns(3)
        underlying_stop = c12.number_input("Underlying stop", value=float(thesis.get("underlying_stop") or 0.0), step=0.5)
        spy_stop = c13.number_input("SPY stop", value=float(thesis.get("spy_stop") or 0.0), step=0.5)
        vix_stop = c14.number_input("VIX stop", value=float(thesis.get("vix_stop") or 0.0), step=0.5)

        st.markdown("**Time rules** (HH:MM ET; blank = unset)")
        c15, c16 = st.columns(2)
        tighten_after = c15.text_input("Tighten after", value=str(time_rules.get("tighten_after", "")))
        flat_before = c16.text_input("Flat before", value=str(time_rules.get("flat_before", "")))

        saved = st.form_submit_button("💾 Save plan", type="primary")

    if not saved:
        return

    from data.social_sentiment import is_restricted_underlying
    if underlying and is_restricted_underlying(underlying):
        st.error(f"🚫 {underlying.upper()} is employer-restricted and cannot be planned/traded.")
        return
    if not underlying:
        st.error("Underlying is required.")
        return

    new_plan: dict = {
        "status": status, "mode": mode, "underlying": underlying.upper(),
        "option_type": option_type, "quantity": int(quantity), "bid_floor": float(bid_floor),
    }
    if strike:
        new_plan["strike"] = float(strike)
    if expiration:
        new_plan["expiration"] = expiration
    if entry_price:
        new_plan["entry_price"] = float(entry_price)
    pr = {}
    if tp:
        pr["take_profit_pct"] = float(tp)
    if se:
        pr["strong_exit_pct"] = float(se)
    if pr:
        new_plan["profit_rules"] = pr
    th = {k: float(v) for k, v in (("underlying_stop", underlying_stop), ("spy_stop", spy_stop),
                                   ("vix_stop", vix_stop)) if v}
    if th:
        new_plan["thesis"] = th
    tr = {k: v for k, v in (("tighten_after", tighten_after.strip()),
                            ("flat_before", flat_before.strip())) if v}
    if tr:
        new_plan["time_rules"] = tr

    try:
        ODTE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _PLAN_PATH.write_text(json.dumps(new_plan, indent=2))
    except Exception as exc:
        st.error(f"Could not save plan: {exc}")
        return
    st.success(f"Saved plan → {_PLAN_PATH}")
    st.json(new_plan)


def render() -> None:
    st.subheader("Position")
    st.caption("Broker-AWARE inputs, **decision-only** output — no orders, no broker/LLM calls.")
    plan = load_odte_json("active_trade.json") or {}
    _render_last_decision()
    st.divider()
    _render_manual_eval(plan)
    st.divider()
    _render_plan_editor(plan)
