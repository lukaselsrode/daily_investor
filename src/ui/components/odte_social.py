"""ui/components/odte_social.py — 0DTE social candidates, watchdog, and scrape history.

Runs the LOCAL social report / watchdog on demand (no LLM, no broker) and browses the timestamped
analyzed-text snapshots under data/odte/scrape/ so scraped social text can be reviewed over time.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.utils import list_scrape_snapshots


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


@st.cache_data(ttl=60)
def _scrape_counts(kind: str) -> pd.DataFrame:
    """Per-snapshot document counts (lines minus the header) for a small timeline chart."""
    rows = []
    for p in list_scrape_snapshots(kind):
        try:
            lines = p.read_text().splitlines()
        except Exception:
            continue
        docs = max(0, len([ln for ln in lines if ln.strip()]) - 1)  # first line is the '# …' header
        # filename: {kind}_text_YYYY_MM_DD_HH_MM.txt
        stamp = p.stem.replace(f"{kind}_text_", "")
        rows.append({"snapshot": stamp, "docs": docs})
    return pd.DataFrame(rows)


def render() -> None:
    st.subheader("Social & Scrape")
    st.caption("LOCAL report — no model calls. Restricted underlyings (NVDA) are never actionable.")

    allow_fetch = st.toggle("Fetch live (off = offline/cache-only)", value=False,
                            help="Off keeps it fully offline. Reddit/X auth comes from ~/0dte/config.json.")

    cols = st.columns(2)
    if cols[0].button("📣 Run social report", type="primary"):
        with st.spinner("Building local social report…"):
            try:
                md, rep = _run_social_report(allow_fetch)
                st.session_state["_odte_social_md"] = md
                st.session_state["_odte_social_rep"] = rep
            except Exception as exc:
                st.error(f"Report failed: {exc}")
    if cols[1].button("🐶 Run watchdog"):
        with st.spinner("Running watchdog…"):
            try:
                st.session_state["_odte_watchdog"] = _run_watchdog(allow_fetch)
            except Exception as exc:
                st.error(f"Watchdog failed: {exc}")

    if "_odte_watchdog" in st.session_state:
        wd = st.session_state["_odte_watchdog"]
        if wd.get("alert"):
            st.warning(f"⚠️ Watchdog alert · SPY: {wd.get('spy_verdict','?')}")
        else:
            st.success(f"Quiet — nothing actionable · SPY: {wd.get('spy_verdict','?')}")
        with st.expander("Watchdog payload"):
            st.json(wd)

    if "_odte_social_md" in st.session_state:
        st.markdown(st.session_state["_odte_social_md"])
        with st.expander("Raw report JSON"):
            st.json(st.session_state.get("_odte_social_rep", {}))

    st.divider()
    st.markdown("### 📜 Scrape history")
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
    pick = st.selectbox("Snapshot", list(reversed(labels)))  # newest first
    chosen = next((p for p in snaps if p.name == pick), None)
    if chosen is not None:
        try:
            st.text_area("Analyzed text", chosen.read_text(), height=320)
        except Exception as exc:
            st.error(f"Could not read snapshot: {exc}")
