"""ui/components/odte_dashboard.py — 0DTE overview.

Read-only snapshot of the local 0DTE store (data/odte/): latest social candidate + watchdog
triggers, the active-position decision banner, today's journal scorecard headline, and scrape
freshness. No writes, no broker, no LLM.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui.utils import (
    ODTE_DATA_DIR,
    latest_scrape_snapshot,
    load_odte_json,
)

# Decision → color, mirroring the portfolio cockpit's state palette.
_DECISION_COLOR: dict[str, str] = {
    "TAKE_PROFIT":         "#2ecc71",
    "HOLD":                "#3498db",
    "NO_POSITION":         "#7f8c8d",
    "TIME_RISK":           "#f39c12",
    "MONITORING_DEGRADED": "#9b59b6",
    "BID_FLOOR":           "#e67e22",
    "THESIS_DEAD":         "#e74c3c",
    "RESTRICTED":          "#e74c3c",
}


def _file_age_hours(path: Path) -> float | None:
    if not path or not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


@st.cache_data(ttl=30)
def _journal_headline() -> dict:
    from data.odte_journal import build_report
    rep = build_report(write_artifacts=False)
    return rep.get("summary", {}) or {}


def render() -> None:
    st.subheader("0DTE Dashboard")
    st.caption(f"Store: `{ODTE_DATA_DIR}` · secrets remain in `~/0dte/`")

    # ---- Active position decision banner ---------------------------------
    decision = load_odte_json("position_decision.json")
    if decision:
        dec = str(decision.get("decision", "—"))
        color = _DECISION_COLOR.get(dec, "#3498db")
        pnl = decision.get("pnl_pct")
        pnl_str = f" · P/L {float(pnl):+.0%}" if isinstance(pnl, (int, float)) else ""
        und = decision.get("underlying") or "—"
        st.markdown(
            f"<div style='padding:0.6rem 0.9rem;border-radius:0.4rem;background:{color}22;"
            f"border-left:4px solid {color}'><b>Position:</b> {dec} "
            f"<span style='opacity:0.8'>({und}, {decision.get('mode','?')}{pnl_str})</span></div>",
            unsafe_allow_html=True,
        )
        if decision.get("triggers"):
            with st.expander("Triggers"):
                st.json(decision["triggers"])
    else:
        st.info("No `position_decision.json` yet — run `odte-position` (or the Position tab).")

    st.divider()

    # ---- Latest social candidate / watchdog triggers ---------------------
    c1, c2 = st.columns(2)
    triggers = load_odte_json("triggers.json")
    with c1:
        st.markdown("**Latest social candidate**")
        cand = (triggers or {}).get("candidate")
        if cand:
            st.metric(
                f"{cand.get('ticker','?')} · {cand.get('direction','?')}",
                f"conf {cand.get('confidence','?')}",
            )
            if (triggers or {}).get("spy_verdict"):
                st.caption(f"SPY verdict: {triggers['spy_verdict']}")
        elif triggers:
            st.caption("No actionable non-restricted candidate.")
        else:
            st.caption("No `triggers.json` yet — run the watchdog (Social tab).")
        restricted = (triggers or {}).get("restricted_chatter") or []
        if restricted:
            st.caption(f"🚫 Restricted chatter (context only): {', '.join(restricted)}")

    with c2:
        st.markdown("**Watchdog alert**")
        if triggers and triggers.get("alert"):
            st.warning("Alert active")
            st.json(triggers.get("triggers", []))
        elif triggers:
            st.success("Quiet — nothing actionable")
        else:
            st.caption("—")

    st.divider()

    # ---- Journal scorecard headline --------------------------------------
    st.markdown("**Journal scorecard**")
    summary = _journal_headline()
    n_trades = summary.get("n_trades", summary.get("trades", 0)) or 0
    if not n_trades:
        st.caption("No journaled trades yet (Journal tab).")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", n_trades)
        hr = summary.get("hit_rate")
        m2.metric("Hit rate", f"{float(hr):.0%}" if isinstance(hr, (int, float)) else "—")
        pnl = summary.get("total_realized_pnl")
        m3.metric("Realized P/L", f"${float(pnl):,.0f}" if isinstance(pnl, (int, float)) else "—")
        cap = summary.get("avg_mfe_capture")
        m4.metric("MFE capture", f"{float(cap):.0%}" if isinstance(cap, (int, float)) else "—")

    st.divider()

    # ---- Scrape freshness -------------------------------------------------
    st.markdown("**Scrape freshness**")
    f1, f2 = st.columns(2)
    for col, kind in ((f1, "reddit"), (f2, "x")):
        latest = latest_scrape_snapshot(kind)
        age = _file_age_hours(latest)
        with col:
            if latest is None:
                st.caption(f"{kind}: no snapshots")
            else:
                icon = "✅" if (age is not None and age < 25) else "⚠️"
                st.caption(f"{icon} {kind}: {latest.name} ({age:.1f}h ago)" if age is not None
                           else f"{kind}: {latest.name}")
