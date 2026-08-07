"""ui/components/odte_rails.py — 0DTE execution-safety audit trail.

The layer between a decision and an order: every lease minted, whether it was consumed, filled or
left to expire; which exit rails fired; which orders the pre-order hook blocked; and every safety
incident with the human adjudication that cleared it.

Also hosts the (dormant) fast-lane shadow comparison, which lights up when the deterministic daemon
starts writing a shadow journal.

Read-only; no orders, no broker.
"""
from __future__ import annotations

import streamlit as st

from ui.services import odte_service as svc

_OUTCOME_COLOR = {
    "filled": "#2ecc71", "consumed_no_fill": "#f39c12",
    "expired_inferred": "#7f8c8d", "denied": "#e74c3c",
}


def _render_leases(spans: list[dict]) -> None:
    st.markdown("#### Lease lifecycle")
    if not spans:
        st.info("No leases minted in this window.")
        return

    default_ttl, hard_cap = svc.lease_ttl_seconds()
    import plotly.graph_objects as go
    fig = go.Figure()
    for i, s in enumerate(reversed(spans)):
        label = f"{s.get('lease_id', '')[:8]} · {s.get('underlying') or '?'}"
        seconds = s.get("seconds")
        fig.add_trace(go.Bar(
            y=[label], x=[seconds if seconds is not None else default_ttl], orientation="h",
            marker_color=_OUTCOME_COLOR.get(s["outcome"], "#3498db"),
            name=s["outcome"], showlegend=False,
            hovertext=(f"{s.get('ts')}<br>tier {s.get('tier') or '—'}<br>{s['outcome']}"
                       f"<br>{'' if seconds is None else f'{seconds:.1f}s to resolution'}"),
            opacity=1.0 if seconds is not None else 0.45,
        ))
        _ = i
    fig.add_vline(x=hard_cap, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"{hard_cap:.0f}s hard cap")
    fig.update_layout(height=max(240, 34 * len(spans)), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="seconds from issue to resolution")
    st.plotly_chart(fig, width="stretch", key="odte_rails_leases")

    import pandas as pd
    st.dataframe(pd.DataFrame([{
        "issued": s.get("ts"), "lease": s.get("lease_id"), "sym": s.get("underlying"),
        "tier": s.get("tier") or "—", "outcome": s["outcome"],
        "resolved in (s)": s.get("seconds"), "ttl (s)": s.get("ttl_seconds"),
        "max limit": s.get("max_limit_price"), "max debit": s.get("max_debit"),
        "incidents": s.get("incidents"),
    } for s in spans]), width="stretch", hide_index=True)

    faded = [s for s in spans if s["outcome"] == "expired_inferred"]
    st.caption(
        f"The {hard_cap:.0f}s hard cap is an incident invariant — deliberately not configurable. "
        + (f"{len(faded)} lease(s) show as *expired (inferred)*: they were never consumed and "
           "never filled, and the atomic conversion path did not record `expires_at` before "
           "2026-08-06, so expiry is deduced rather than read." if faded else ""))


def _render_rails_fired(rail_counts: list) -> None:
    st.markdown("#### Exit rails fired")
    if not rail_counts:
        st.caption("No rail-named exits in this window.")
        return
    import plotly.graph_objects as go
    labels = [r[0] for r in rail_counts][::-1]
    values = [r[1] for r in rail_counts][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#6a1b9a"))
    fig.update_layout(height=max(200, 40 * len(labels)), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="exits")
    st.plotly_chart(fig, width="stretch", key="odte_rails_fired")
    st.caption("Counted per trade, not per event — an exit is journaled by both the controller and "
               "the order guard, so per-event counting doubles every rail.")


def _render_incidents(incidents: list[dict], hook_blocks: list[dict]) -> None:
    st.markdown("#### Incidents & adjudications")
    if not incidents:
        st.success("No execution-safety incidents in this window.")
    for inc in incidents:
        adj = inc.get("adjudication")
        head = (f"{'✅' if adj else '⚠️'} {inc.get('ts')} · {inc.get('underlying') or '?'} · "
                f"{inc.get('guard_state') or inc.get('stage') or 'incident'}")
        with st.expander(head, expanded=not adj):
            st.markdown("**Reasons:** " + ", ".join(f"`{r}`" for r in inc.get("reason_codes") or [])
                        or "—")
            if adj:
                st.markdown(f"**Adjudicated by:** {adj.get('adjudicated_by', '—')}")
                st.markdown(f"**Reason:** {adj.get('adjudication_reason') or adj.get('reason')}")
            else:
                st.warning("Unadjudicated — this incident is still holding the entry lockout. "
                           "Only a human adjudication naming the incident clears it.")

    st.markdown("#### Pre-order hook blocks")
    if not hook_blocks:
        st.caption("No hook blocks in this window.")
    else:
        st.caption(f"{len(hook_blocks)} order(s) refused at the pre-order hook. A hook block is a "
                   "**refusal**, never an incident — the rail did its job.")
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "ts": b.get("ts"), "sym": b.get("underlying") or b.get("symbol"),
            "reasons": ", ".join(b.get("reason_codes") or []),
        } for b in hook_blocks]), width="stretch", hide_index=True)


def _render_shadow() -> None:
    st.markdown("#### Fast-lane shadow comparison")
    stage = svc.fast_lane_stage() or {}
    report = svc.shadow_state()
    if report is None:
        st.info(f"Dormant — the deterministic fast lane is staged **{stage.get('stage', 'unknown')}** "
                "but has not written a shadow journal yet. This panel fills in automatically once "
                "the daemon runs: agreements, shadow-only and live-only decisions, exit "
                "divergences and any incidents.")
        return
    if report.get("error"):
        st.error(f"Shadow report failed: {report['error']}")
        return
    counts = report.get("counts") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Both fired", counts.get("both_fired", 0))
    c2.metric("Shadow only", counts.get("shadow_only", 0))
    c3.metric("Live only", counts.get("live_only", 0))
    c4.metric("Exit divergences", counts.get("exit_divergences", 0))
    if report.get("clean"):
        st.success("Clean — no divergences.")
    else:
        st.warning("Divergences present; review before advancing the rollout stage.")
    with st.expander("Full shadow report"):
        st.json(report)


def render() -> None:
    st.subheader("Rails & safety")
    st.caption("The layer between a decision and an order: leases, exit rails, hook blocks, "
               "incidents and their adjudications.")

    days = st.slider("Window (days)", min_value=3, max_value=60, value=14, step=1,
                     key="odte_rails_days")
    data = svc.lease_timeline(days=days)
    st.caption(f"{data['start']} → {data['end']}")

    _render_leases(data["spans"])
    st.divider()
    _render_rails_fired(data["rail_counts"])
    st.divider()
    _render_incidents(data["incidents"], data["hook_blocks"])
    st.divider()
    _render_shadow()
