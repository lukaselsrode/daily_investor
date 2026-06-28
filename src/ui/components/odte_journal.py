"""ui/components/odte_journal.py — 0DTE decision journal scorecard + append.

Reads data/odte/decision_journal.jsonl, renders deterministic metrics (by-mode table, P/L sequence),
and lets the user append a journal event (thesis / decision / outcome / experiment / note). NVDA is
tagged restricted on store by the journal module itself; the form blocks it up front too.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

_EVENT_TYPES = [
    "pre_trade_thesis", "entry_decision", "order_filled", "management_check",
    "exit_decision", "order_closed", "postmortem", "experiment", "note",
]
_MODES = ["scalp", "trend", "lotto", "runner"]


@st.cache_data(ttl=20)
def _report() -> dict:
    from data.odte_journal import build_report
    return build_report(write_artifacts=False)


def _render_scorecard() -> None:
    rep = _report()
    summary = rep.get("summary", {}) or {}
    n_trades = summary.get("n_trades", 0)
    if not n_trades:
        st.info("No journaled trades yet — append an event below to start the record.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", n_trades)
        hr = summary.get("hit_rate")
        m2.metric("Hit rate", f"{float(hr):.0%}" if isinstance(hr, (int, float)) else "—")
        pnl = summary.get("total_realized_pnl")
        m3.metric("Realized P/L", f"${float(pnl):,.0f}" if isinstance(pnl, (int, float)) else "—")
        cap = summary.get("avg_mfe_capture")
        m4.metric("MFE capture", f"{float(cap):.0%}" if isinstance(cap, (int, float)) else "—")

        by_mode = summary.get("by_mode") or {}
        if by_mode:
            st.markdown("**By mode**")
            st.dataframe(pd.DataFrame(by_mode).T, use_container_width=True)

        seq = summary.get("pnl_sequence") or []
        if seq:
            st.markdown("**Realized P/L sequence**")
            st.bar_chart(pd.Series(seq, name="realized_pnl"))

        flags = summary.get("restricted_flags") or []
        if flags:
            st.error(f"🚫 Restricted underlyings present in journal (kept out of metrics): {', '.join(flags)}")

    with st.expander("Full report (Markdown)"):
        st.markdown(rep.get("markdown", "_(empty)_"))


def _render_append_form() -> None:
    st.markdown("### ➕ Append event")
    with st.form("odte_journal_event"):
        c1, c2, c3 = st.columns(3)
        event_type = c1.selectbox("Event type", _EVENT_TYPES)
        mode = c2.selectbox("Mode", ["", *_MODES])
        trade_id = c3.text_input("Trade ID", value="")
        underlying = st.text_input("Underlying", value="")

        st.caption("Fill only what's relevant for this event type. Blank fields are dropped.")
        cols = st.columns(2)
        direction = cols[0].text_input("Thesis: direction", value="")
        catalyst = cols[1].text_input("Thesis: catalyst", value="")
        action = cols[0].text_input("Decision: action", value="")
        confidence = cols[1].text_input("Decision: confidence", value="")
        realized_pnl = cols[0].text_input("Outcome: realized P/L ($)", value="")
        mfe = cols[1].text_input("Outcome: MFE ($)", value="")
        hypothesis = st.text_input("Experiment: hypothesis", value="")
        note = st.text_area("Free-form note", value="", height=80)

        submitted = st.form_submit_button("Append", type="primary")

    if not submitted:
        return

    if underlying:
        from data.social_sentiment import is_restricted_underlying
        if is_restricted_underlying(underlying):
            st.warning(f"⚠️ {underlying.upper()} is employer-restricted — it will be tagged "
                       "`restricted` and excluded from metrics.")

    event: dict = {"event_type": event_type}
    if mode:
        event["mode"] = mode
    if trade_id:
        event["trade_id"] = trade_id
    if underlying:
        event["underlying"] = underlying.upper()

    thesis = {k: v for k, v in (("direction", direction), ("catalyst", catalyst)) if v}
    if thesis:
        event["thesis"] = thesis
    decision = {k: v for k, v in (("action", action), ("confidence", confidence)) if v}
    if decision:
        event["decision"] = decision
    outcome: dict = {}
    for key, raw in (("realized_pnl", realized_pnl), ("mfe", mfe)):
        if raw.strip():
            try:
                outcome[key] = float(raw)
            except ValueError:
                st.error(f"{key}: not a number ({raw!r})")
                return
    if outcome:
        event["outcome"] = outcome
    if hypothesis:
        event["hypothesis"] = hypothesis
    if note:
        event["note"] = note

    try:
        from data.odte_journal import append_event
        stored = append_event(event)
        _report.clear()  # refresh the cached scorecard
    except Exception as exc:
        st.error(f"Append failed: {exc}")
        return
    st.success(f"Appended event #{stored.get('seq', '?')} ({stored.get('event_type')})")
    st.json(stored)


def render() -> None:
    st.subheader("Decision Journal")
    st.caption("Local/offline record — no broker, no LLM, no secrets.")
    _render_scorecard()
    st.divider()
    _render_append_form()
