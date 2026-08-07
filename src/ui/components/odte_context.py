"""ui/components/odte_context.py — 0DTE supporting context: social, scrape history, FMP sanity.

Merges what were separate Social and FMP tabs. Everything here is CONTEXT, never a signal: the
social lane produces candidates the rails still have to gate, and the FMP block is a meme/squeeze
sanity check only. Restricted underlyings (NVDA) are surfaced as context and never actionable.

Runs the LOCAL social report / watchdog on demand — no LLM, no broker, no orders.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.utils import list_scrape_snapshots, load_odte_json


@st.cache_data(ttl=300)
def _run_social_report(allow_fetch: bool) -> tuple[str, dict]:
    """Build the local social report → (markdown, raw_report). Fail-soft."""
    from data.social_sentiment import build_odte_social_report, format_report
    rep = build_odte_social_report(allow_fetch=allow_fetch)
    return format_report(rep), rep


@st.cache_data(ttl=300)
def _run_watchdog(allow_fetch: bool) -> dict:
    from data.odte_watchdog import run_watchdog
    return run_watchdog(allow_fetch=allow_fetch)


@st.cache_data(ttl=300)
def _fmp_context(symbol: str, allow_fetch: bool) -> dict:
    from data.odte_fmp_context import run_fmp_context
    return run_fmp_context(symbol, allow_fetch=allow_fetch)


@st.cache_data(ttl=60)
def _scrape_counts(kind: str) -> pd.DataFrame:
    """Per-snapshot document counts (lines minus the header) for a small timeline chart."""
    rows = []
    for p in list_scrape_snapshots(kind):
        try:
            lines = p.read_text().splitlines()
        except OSError:
            continue
        docs = max(0, len([ln for ln in lines if ln.strip()]) - 1)  # first line is the '# …' header
        rows.append({"snapshot": p.stem.replace(f"{kind}_text_", ""), "docs": docs})
    return pd.DataFrame(rows)


# --- sections ---------------------------------------------------------------------------------

def _render_latest_watchdog() -> None:
    st.markdown("#### Latest watchdog state")
    triggers = load_odte_json("triggers.json") or {}
    if not triggers:
        st.caption("No `triggers.json` yet.")
        return
    c1, c2 = st.columns(2)
    cand = triggers.get("candidate") or {}
    with c1:
        if cand:
            st.metric(f"{cand.get('ticker', '?')} · {cand.get('direction', '?')}",
                      f"conf {cand.get('confidence', '?')}")
        else:
            st.caption("No actionable non-restricted candidate.")
        if triggers.get("spy_verdict"):
            st.caption(f"SPY verdict: **{triggers['spy_verdict']}**")
    with c2:
        if triggers.get("alert"):
            st.warning("Alert active")
        else:
            st.success("Quiet — nothing actionable")
        restricted = triggers.get("restricted_chatter") or []
        if restricted:
            st.caption(f"🚫 Restricted chatter (context only): {', '.join(restricted)}")


def _render_social() -> None:
    st.markdown("#### Social report")
    st.caption("LOCAL report — no model calls. Restricted underlyings are never actionable.")
    allow_fetch = st.toggle(
        "Fetch live (off = offline/cache-only)", value=False, key="odte_ctx_fetch",
        help="Off keeps it fully offline. Reddit/X auth comes from ~/0dte/config.json.")

    cols = st.columns(2)
    if cols[0].button("📣 Run social report", type="primary", key="odte_ctx_social_btn"):
        with st.spinner("Building local social report…"):
            try:
                md, rep = _run_social_report(allow_fetch)
                st.session_state["_odte_social_md"] = md
                st.session_state["_odte_social_rep"] = rep
            except Exception as exc:
                st.error(f"Report failed: {exc}")
    if cols[1].button("🐶 Run watchdog", key="odte_ctx_watchdog_btn"):
        with st.spinner("Running watchdog…"):
            try:
                st.session_state["_odte_watchdog"] = _run_watchdog(allow_fetch)
            except Exception as exc:
                st.error(f"Watchdog failed: {exc}")

    if "_odte_watchdog" in st.session_state:
        wd = st.session_state["_odte_watchdog"]
        (st.warning if wd.get("alert") else st.success)(
            f"{'⚠️ Watchdog alert' if wd.get('alert') else 'Quiet'} · "
            f"SPY: {wd.get('spy_verdict', '?')}")
        with st.expander("Watchdog payload"):
            st.json(wd)

    if "_odte_social_md" in st.session_state:
        st.markdown(st.session_state["_odte_social_md"])
        with st.expander("Raw report JSON"):
            st.json(st.session_state.get("_odte_social_rep", {}))


def _render_scrape_history() -> None:
    st.markdown("#### Scrape history")
    st.caption("Timestamped analyzed-text snapshots accumulate in `data/odte/scrape/`.")
    kind = st.radio("Source", ["reddit", "x"], horizontal=True, key="_odte_scrape_kind")
    snaps = list_scrape_snapshots(kind)
    if not snaps:
        st.info(f"No `{kind}` scrape snapshots yet — run the social report with fetch on.")
        return
    counts = _scrape_counts(kind)
    if not counts.empty:
        st.line_chart(counts.set_index("snapshot")["docs"], height=180)
    labels = [p.name for p in snaps]
    pick = st.selectbox("Snapshot", list(reversed(labels)), key="odte_ctx_snapshot")
    chosen = next((p for p in snaps if p.name == pick), None)
    if chosen is not None:
        try:
            st.text_area("Analyzed text", chosen.read_text(), height=320)
        except OSError as exc:
            st.error(f"Could not read snapshot: {exc}")


def _render_fmp() -> None:
    st.markdown("#### FMP single-name sanity")
    st.caption("Meme/squeeze **sanity** only — not an entry signal, no orders, no options/gamma "
               "(Robinhood remains the gamma source).")
    c1, c2 = st.columns([2, 1])
    symbol = c1.text_input("Symbol", value="", key="odte_ctx_fmp_sym").strip().upper()
    allow_fetch = c2.toggle("Fetch live", value=False, key="odte_ctx_fmp_fetch",
                            help="Off = offline (no FMP call).")

    if not st.button("🔎 Fetch context", key="odte_ctx_fmp_btn"):
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
    for w in ctx.get("warnings", []) or []:
        st.caption(f"⚠️ {w}")
    with st.expander("Full context JSON"):
        st.json(ctx)


def _render_artifacts() -> None:
    st.markdown("#### Latest scoring artifacts")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Day score**")
        day = load_odte_json("reports/odte_day_score.json")
        if day:
            st.metric(str(day.get("verdict", "—")), day.get("score", "—"))
            missing = day.get("components_missing") or []
            st.caption(f"components supplied {day.get('components_supplied', '—')}"
                       + (f" · missing {', '.join(missing)}" if missing else ""))
        else:
            st.caption("—")
    with c2:
        st.caption("**Gamma / pin map**")
        gamma = (load_odte_json("reports/odte_gamma_map_spy.json")
                 or load_odte_json("reports/odte_gamma_map_qqq.json"))
        if gamma:
            st.metric(str(gamma.get("underlying", "—")), str(gamma.get("pin_risk", "—")))
            st.caption(f"basis: {gamma.get('basis', '—')}")
        else:
            st.caption("— (run `odte-gamma-map`; the map is journaled as `gamma_map` events)")


def render() -> None:
    st.subheader("Context")
    st.caption("Supporting context for the 0DTE lane — social candidates, scrape history, and "
               "single-name sanity. None of it is an entry signal on its own.")

    _render_latest_watchdog()
    st.divider()
    _render_social()
    st.divider()
    _render_scrape_history()
    st.divider()
    _render_fmp()
    st.divider()
    _render_artifacts()
