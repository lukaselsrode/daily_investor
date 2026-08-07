"""ui/components/odte_funnel.py — 0DTE conversion funnel and refusal analysis.

Answers the volume question: of everything the watchdog saw, how much reached a fill, and where
did the rest die? The stage-prefixed refusal Pareto names the actual clause, so "we're not trading
enough" becomes "N refusals at entry_gate:budget_check:fail".

Also hosts the ORB near-miss panel — the evidence surface for the deferred decision on whether the
opening-range-breakout clause is costing tradeable setups.

Read-only; all counts come from ui.services.odte_service.
"""
from __future__ import annotations

import streamlit as st

from ui.services import odte_service as svc

_STAGES = [
    ("watchdog_ticks", "Watchdog ticks"),
    ("watchdog_alerts", "Alerts"),
    ("candidate_evaluations", "Candidate evaluations"),
    ("candidate_confirms", "Confirmed"),
    ("entry_decisions", "Gate decisions"),
    ("gates_passed", "Gates passed"),
    ("leases_issued", "Leases issued"),
    ("fills", "Fills"),
]


def _render_funnel(total: dict) -> None:
    st.markdown("#### Conversion funnel")
    stages = [(label, int(total.get(key) or 0)) for key, label in _STAGES]
    # Drop leading stages that were never collected so the chart doesn't imply a zero.
    if total.get("candidate_evaluations", 0) == 0:
        stages = [(lbl, n) for lbl, n in stages
                  if lbl not in ("Candidate evaluations", "Confirmed")]

    import plotly.graph_objects as go
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    fig = go.Figure(go.Funnel(y=labels, x=values, textinfo="value+percent previous"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", key="odte_funnel_total")

    survivors = [(lbl, n) for lbl, n in stages if n]
    if len(survivors) >= 2:
        widest, narrowest = survivors[0], survivors[-1]
        st.caption(f"{widest[1]:,} {widest[0].lower()} → {narrowest[1]:,} "
                   f"{narrowest[0].lower()} over this window.")
    if total.get("candidate_evaluations", 0) == 0:
        st.caption("Candidate-evaluation stages are omitted: per-tick evaluation events were only "
                   "added 2026-08-06, so an empty row here would mean 'not collected', not 'none'.")


def _render_daily(rows: list[dict]) -> None:
    st.markdown("#### Per-day")
    if not rows:
        st.info("No days with events in this window.")
        return
    import pandas as pd
    df = pd.DataFrame([{
        "date": r["start"], "ticks": r["watchdog_ticks"], "alerts": r["watchdog_alerts"],
        "gate decisions": r["entry_decisions"], "passed": r["gates_passed"],
        "leases": r["leases_issued"], "fills": r["fills"], "refusals": r["lease_refusals"],
    } for r in rows])
    st.dataframe(df, width="stretch", hide_index=True)

    import plotly.graph_objects as go
    fig = go.Figure()
    for col, color in (("gate decisions", "#7f8c8d"), ("passed", "#3498db"),
                       ("leases", "#f39c12"), ("fills", "#2ecc71")):
        fig.add_trace(go.Bar(name=col, x=df["date"], y=df[col], marker_color=color))
    fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch", key="odte_funnel_daily")
    st.caption("Fills are deduped: the controller and the order guard both journal an entry, and "
               "counting each would read one trade as two.")


def _render_pareto(pareto: list, by_stage: dict) -> None:
    st.markdown("#### Why conversions died")
    if not pareto:
        st.info("No refusals recorded in this window.")
        return
    import plotly.graph_objects as go
    labels = [p[0] for p in pareto][::-1]
    values = [p[1] for p in pareto][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#e74c3c"))
    fig.update_layout(height=max(260, 26 * len(labels)),
                      margin=dict(l=10, r=10, t=10, b=10), xaxis_title="refusals")
    st.plotly_chart(fig, width="stretch", key="odte_funnel_pareto")

    if by_stage:
        cols = st.columns(len(by_stage))
        for col, (stage, n) in zip(cols, sorted(by_stage.items(), key=lambda kv: -kv[1])):
            col.metric(stage, n)
    st.caption("Reasons are stage-prefixed. `unknown` is pre-2026-08-04, before refusals recorded "
               "which stage they died at.")


def _render_orb(orb: dict) -> None:
    st.markdown("#### ORB near-misses")
    st.caption(
        "A candidate confirms on: correct VWAP side **and** its own ORB broken **and** a breadth "
        "score at or above the required level **and** the VIXY read agreeing. Breadth grades each "
        "index 2 for VWAP-side *and* ORB-side, 1 for VWAP side alone — so a leader-led tape with "
        "rangebound laggards can reach the bar. A *near miss* satisfied everything **except** the "
        "candidate's own ORB clause — these are the setups the ORB requirement is costing. "
        "Evaluations before 2026-08-07 are scored by the ≥2-full-confirmer rule that applied then.")

    if orb.get("collecting"):
        st.info("**Collecting.** Per-tick candidate evaluations began 2026-08-06 (convert lane) "
                "and 2026-08-07 (scanning lane); before that the "
                "clause-level checks were computed and discarded. An empty result here means the "
                "data is not in yet — never that no near-misses occurred.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Evaluations", orb.get("evaluated", 0))
    c2.metric("Confirmed", orb.get("confirmed", 0))
    c3.metric("Near miss (ORB only)", orb.get("near_miss", 0))

    by_day = orb.get("by_day") or {}
    if by_day:
        import pandas as pd
        import plotly.graph_objects as go
        df = pd.DataFrame([{"date": str(d), **v} for d, v in by_day.items()])
        fig = go.Figure()
        fig.add_trace(go.Bar(name="confirmed", x=df["date"], y=df["confirmed"],
                             marker_color="#2ecc71"))
        fig.add_trace(go.Bar(name="near miss", x=df["date"], y=df["near_miss"],
                             marker_color="#f39c12"))
        fig.update_layout(barmode="stack", height=280, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.2))
        st.plotly_chart(fig, width="stretch", key="odte_orb_by_day")

    samples = orb.get("samples") or []
    if samples:
        import pandas as pd
        with st.expander(f"Near-miss detail ({len(samples)} shown)"):
            st.dataframe(pd.DataFrame(samples), width="stretch", hide_index=True)
    if orb.get("near_miss", 0) and orb.get("confirmed", 0) is not None:
        st.caption("Read this as the counterfactual population only. Whether those setups would "
                   "have WON is a separate question — nothing follows a refused contract forward "
                   "yet.")


def render() -> None:
    st.subheader("Conversion funnel")
    st.caption("Where volume dies, and why. Counts come from the same tally the cockpit's weekly "
               "telemetry uses, so the two can never disagree.")

    days = st.slider("Window (days)", min_value=3, max_value=60, value=10, step=1,
                     key="odte_funnel_days")
    data = svc.funnel(days=days)
    st.caption(f"{data['start']} → {data['end']}")

    _render_funnel(data["total"])
    st.divider()
    _render_daily(data["rows"])
    st.divider()
    _render_pareto(data["pareto"], data["by_stage"])
    st.divider()
    _render_orb(svc.orb_near_misses(days=days))
