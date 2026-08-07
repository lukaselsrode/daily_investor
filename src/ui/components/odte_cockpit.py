"""ui/components/odte_cockpit.py — 0DTE live operations cockpit.

The landing view: where the loop is, what the rails currently permit, what today's budget allows
(including the A+ uncapped exception), whether the safety layer is holding entries, which artifacts
have gone stale, and this week's pace against the 3-4 trade target.

Read-only. Every number comes from ui.services.odte_service, which reads the local store under
data/odte/. No orders, no broker calls, no LLM.
"""
from __future__ import annotations

import streamlit as st

from ui.services import odte_service as svc

_STATE_COLOR = {
    "REVIEWED": "#7f8c8d", "WATCHING": "#3498db", "CANDIDATE": "#f39c12",
    "CONFIRMED": "#2ecc71", "IN_TRADE": "#2ecc71", "BLOCKED": "#e74c3c",
    "DEGRADED": "#9b59b6", "FLAT_NO_TRADE": "#7f8c8d",
}


def _color_for(state: str) -> str:
    up = str(state or "").upper()
    for key, color in _STATE_COLOR.items():
        if key in up:
            return color
    return "#3498db"


def _banner(text: str, color: str, sub: str = "") -> None:
    st.markdown(
        f"<div style='padding:0.6rem 0.9rem;border-radius:0.4rem;background:{color}22;"
        f"border-left:4px solid {color}'><b>{text}</b>"
        + (f"<br><span style='opacity:0.8;font-size:0.9em'>{sub}</span>" if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def _money(v, digits: int = 2) -> str:
    return f"${float(v):,.{digits}f}" if isinstance(v, (int, float)) else "—"


# --- sections ---------------------------------------------------------------------------------

def _render_state(state: dict) -> None:
    loop_state = state.get("state") or "—"
    posture = state.get("posture") or "—"
    _banner(f"{loop_state} · {posture}",
            _color_for(f"{loop_state} {posture}"),
            f"stage <code>{state.get('loop_stage') or '—'}</code> · "
            f"broker lane <code>{state.get('broker_lane') or '—'}</code>")
    if state.get("next_action"):
        st.caption(f"Next: {state['next_action']}")
    if state.get("next_command"):
        st.code(state["next_command"], language="bash")
    reasons = state.get("reasons") or []
    if reasons:
        with st.expander(f"Why this state ({len(reasons)})"):
            for r in reasons:
                st.markdown(f"- {r}")


def _render_budget(budget: dict, rails: dict) -> None:
    st.markdown("#### Today's budget")
    used, cap = budget.get("trades_today", 0), budget.get("budget", 0)
    net = budget.get("day_net_pnl")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades today", f"{used} / {cap}")
    c2.metric("Remaining", budget.get("remaining", 0))
    c3.metric("Day net P/L", _money(net) if net is not None else "—")
    c4.metric("Winning tier", str((budget.get("winning_tier") or {}).get("winning_tier") or "—"))

    if budget.get("aplus_uncapped_active"):
        st.success(
            "**A+ uncapped active** — the day is not red, so an A+ setup may enter past the "
            f"{cap}-trade cap if buying power permits. Lower tiers stay capped.")
    elif budget.get("exhausted"):
        st.warning(f"Budget exhausted ({used}/{cap}) and the A+ exception is not active.")

    if budget.get("cooldown_active"):
        st.info(f"Re-entry cooldown until {budget.get('cooldown_until')}")

    green = budget.get("green_day") or {}
    if green.get("locked"):
        st.info(f"🟢 Green-day preservation locked · banked {_money(green.get('banked_pnl'))} · "
                f"net {_money(green.get('net_day_pnl'))}")

    reentry = (rails or {}).get("green_reentry") or {}
    if reentry:
        armable = reentry.get("structurally_armable")
        st.caption(
            f"Green re-entry: {'structurally armable' if armable else 'not armable'} · "
            f"winning tier today {reentry.get('winning_tier_today') or '—'} · "
            f"budget remaining {reentry.get('budget_remaining')} · "
            f"min BP multiple {reentry.get('min_bp_multiple')}")


def _render_position(state: dict) -> None:
    st.markdown("#### Position & lease")
    from ui.utils import load_odte_json
    trade = load_odte_json("active_trade.json") or {}
    decision = load_odte_json("position_decision.json") or {}
    lease = state.get("execution_lease") or {}

    c1, c2 = st.columns(2)
    with c1:
        if trade.get("active"):
            st.metric(f"{trade.get('underlying', '?')} {trade.get('option_type', '')}".strip(),
                      _money(trade.get("entry_price")),
                      help="Active position from active_trade.json")
            st.caption(f"qty {trade.get('quantity', '?')} · "
                       f"strike {trade.get('strike_price', '?')} · "
                       f"mode {trade.get('mode', '?')}")
        else:
            st.caption("Flat — no active position.")
        if decision:
            pnl = decision.get("pnl_pct")
            pnl_str = f" · P/L {float(pnl):+.1f}%" if isinstance(pnl, (int, float)) else ""
            st.caption(f"Last position decision: **{decision.get('decision', '—')}**{pnl_str}")

    with c2:
        if lease.get("lease_id"):
            remaining = lease.get("seconds_remaining")
            if lease.get("expired"):
                st.caption(f"Lease `{lease['lease_id']}` — **expired** "
                           f"({lease.get('symbol', '?')}, {lease.get('risk_mode', '?')})")
            else:
                st.success(f"Live lease `{lease['lease_id']}` · {lease.get('symbol', '?')} · "
                           f"{float(remaining):.0f}s remaining"
                           if isinstance(remaining, (int, float))
                           else f"Live lease `{lease['lease_id']}`")
        else:
            st.caption("No lease on file.")


def _render_rails(rails: dict) -> None:
    st.markdown("#### Rails in force")
    if not rails:
        st.caption("No live rails payload.")
        return
    lease = rails.get("lease") or {}
    debit = rails.get("max_debit_dollars") or {}
    frac = rails.get("max_debit_fraction") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buying power", _money(rails.get("buying_power")),
              help=f"lane: {rails.get('buying_power_lane')}")
    c2.metric("Max debit · full", _money(debit.get("full")),
              help=f"{frac.get('full')} of buying power")
    c3.metric("Max debit · b_plus", _money(debit.get("b_plus")),
              help=f"{frac.get('b_plus')} of buying power (half size)")
    c4.metric("Chase band", f"{float(rails['chase_band_fraction']):.0%}"
              if isinstance(rails.get("chase_band_fraction"), (int, float)) else "—",
              help="anchor × (1 + band) is the highest limit a lease will authorize")

    st.caption(
        f"Lease TTL {lease.get('default_ttl_seconds')}s (hard cap {lease.get('hard_cap_seconds')}s "
        "— an incident invariant, not a tunable) · conversion SLA "
        f"{rails.get('conversion_sla_seconds')}s · snapshot TTL {rails.get('snapshot_ttl_seconds')}s "
        f"· universe {', '.join(rails.get('executable_universe') or []) or '—'}")


def _render_safety(safety: dict) -> None:
    st.markdown("#### Execution safety")
    incidents = safety.get("incidents") or []
    adjudicated = safety.get("adjudicated") or []
    if safety.get("locked"):
        st.error(f"🔒 **Entries locked** — {len(incidents)} unadjudicated incident(s). "
                 "A lockout clears only by a human adjudication naming the incident.")
    else:
        st.success(f"Unlocked · {len(adjudicated)} adjudicated incident(s) on record.")
    for a in adjudicated:
        with st.expander(f"Adjudicated · {a.get('ts', '')} · {a.get('underlying') or ''}"):
            st.markdown(f"**By:** {a.get('adjudicated_by', '—')}")
            st.markdown(f"**Reason:** {a.get('adjudication_reason') or a.get('reason') or '—'}")
    for i in incidents:
        with st.expander(f"⚠️ Open incident · {i.get('ts', '')}"):
            st.json(i)


def _render_pace(telemetry: dict) -> None:
    st.markdown("#### Weekly pace")
    if not telemetry:
        st.caption("No weekly telemetry.")
        return
    target = telemetry.get("weekly_target") or [None, None]
    trades = telemetry.get("trades_this_week", 0)
    tripwire = telemetry.get("tripwire") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(telemetry.get("iso_week", "week"), trades,
              help=f"target {target[0]}-{target[1]} trades/week")
    c2.metric("Gates passed", telemetry.get("gates_passed", 0))
    c3.metric("Leases issued", telemetry.get("leases_issued", 0))
    c4.metric("Refusals", telemetry.get("lease_refusals", 0))

    if tripwire.get("fired"):
        st.error("🚨 Zero-trade tripwire FIRED — the week has reached its tripwire weekday with "
                 "no trades. Advisory only; it never loosens a gate by itself.")
    elif tripwire.get("armed"):
        st.caption(f"Tripwire armed (weekday {tripwire.get('weekday_et')}), not fired — "
                   f"{trades} trade(s) on the board.")
    else:
        st.caption("Tripwire not yet armed (early week).")

    top = telemetry.get("top_refusal_reasons") or []
    if top:
        st.caption("Top refusals this week: " + " · ".join(f"`{r}` ×{n}" for r, n in top[:3]))


def _render_freshness(state: dict) -> None:
    with st.expander("Artifact freshness & status"):
        artifacts = state.get("artifacts") or {}
        ages = state.get("artifact_ages") or {}
        rows = []
        for name, entry in ages.items():
            if not isinstance(entry, dict):
                continue
            age, ttl, fresh = (entry.get("age_minutes"), entry.get("ttl_minutes"),
                               entry.get("fresh"))
            icon = "✅" if fresh else ("⚠️" if fresh is False else "·")
            rows.append({
                "artifact": name, "": icon,
                "age (min)": round(age, 1) if isinstance(age, (int, float)) else None,
                "ttl (min)": ttl,
                "as_of": entry.get("as_of"),
            })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        bad = {k: v for k, v in artifacts.items() if v not in ("ok",) and not isinstance(v, int)}
        if bad:
            st.warning("Artifacts not OK: " + ", ".join(f"`{k}`={v}" for k, v in bad.items()))
        st.caption(f"Journal events: {artifacts.get('journal_events', '—')} · "
                   f"generated {state.get('generated_at', '—')}")


# --- entry point -------------------------------------------------------------------------------

def render() -> None:
    st.subheader("Cockpit")
    st.caption("Live 0DTE operations — loop state, rails, budget, safety. Decision-only: this page "
               "places no orders and calls no broker.")

    state = svc.cockpit_state()
    if state.get("error"):
        st.error(f"Loop status unavailable: {state['error']}")
        return

    _render_state(state)
    st.divider()
    _render_budget(svc.budget_now(), state.get("live_rails") or {})
    st.divider()
    _render_position(state)
    st.divider()
    _render_rails(state.get("live_rails") or {})
    st.divider()
    _render_safety(svc.safety_state())
    st.divider()
    _render_pace(state.get("weekly_telemetry") or {})
    _render_freshness(state)

    stage = svc.fast_lane_stage() or {}
    if stage:
        st.caption(f"Fast lane stage: **{stage.get('stage', '?')}** (set {stage.get('set_at', '?')})")
