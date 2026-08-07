"""ui/components/odte_ledger.py — 0DTE trade ledger.

Every executed trade with its tier, lease binding, exit rail, MFE capture and process grade, plus
the P/L waterfall, the process×outcome matrix, the intra-trade P/L path, and the execution latency
chain. This is the highest value-per-row surface in the app: one line per trade, fully attributed.

Read-only; all joins happen in ui.services.odte_service.
"""
from __future__ import annotations

import streamlit as st

from ui.services import odte_service as svc


def _pct(v) -> str:
    return f"{float(v):.0f}%" if isinstance(v, (int, float)) else "—"


def _money(v) -> str:
    return f"${float(v):,.2f}" if isinstance(v, (int, float)) else "—"


def _render_headline(rows: list[dict], summary: dict) -> None:
    closed = [r for r in rows if r.get("realized_pnl") is not None]
    wins = [r for r in closed if (r.get("realized_pnl") or 0) > 0]
    captures = [r["mfe_capture_pct"] for r in closed if r.get("mfe_capture_pct") is not None]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closed trades", len(closed))
    c2.metric("Hit rate", f"{len(wins) / len(closed):.0%}" if closed else "—")
    c3.metric("Realized P/L", _money(sum(r["realized_pnl"] for r in closed)) if closed else "—")
    c4.metric("MFE capture", f"{sum(captures) / len(captures):.0f}%" if captures else "—",
              help="From the postmortem's own excursion measurement where present "
                   "(measured against the best bid actually seen), else realized/MFE.")
    modern = [r for r in rows if r.get("modern")]
    st.caption(f"{len(modern)} of {len(rows)} trades are fully instrumented — "
               f"{svc.ERA_MODERN_NOTE}. Earlier rows show what their era recorded and nothing more.")
    if summary.get("avg_held_minutes"):
        st.caption(f"Average hold {summary['avg_held_minutes']} min.")


def _render_table(rows: list[dict]) -> None:
    st.markdown("#### Ledger")
    if not rows:
        st.info("No trades journaled yet.")
        return
    import pandas as pd
    df = pd.DataFrame([{
        "date": r.get("trade_date"),
        "sym": r.get("underlying"),
        "tier": r.get("tier") or "—",
        "lease ✓": ("✓" if r.get("lease_valid_at_fill") else
                    ("—" if r.get("lease_id") is None else "?")),
        "entry": r.get("entry_price"),
        "exit": r.get("exit_price"),
        "rail": r.get("rail_fired") or "—",
        "P/L": r.get("realized_pnl"),
        "net": r.get("net_pnl"),
        "MFE capture %": r.get("mfe_capture_pct"),
        "held (min)": r.get("held_minutes"),
        "process": r.get("process_quality") or "—",
        "outcome": r.get("outcome_quality") or "—",
    } for r in rows])
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "MFE capture %": st.column_config.ProgressColumn(
                "MFE capture", min_value=0, max_value=100, format="%.0f%%"),
            "P/L": st.column_config.NumberColumn("P/L", format="$%.2f"),
            "net": st.column_config.NumberColumn("net", format="$%.2f"),
        })
    violations = [(r, v) for r in rows for v in (r.get("rule_violations") or [])]
    if violations:
        with st.expander(f"Rule violations ({len(violations)})"):
            for r, v in violations:
                st.markdown(f"- **{r.get('trade_date')} {r.get('underlying')}** — {v}")


def _render_waterfall(summary: dict) -> None:
    st.markdown("#### P/L sequence")
    seq = summary.get("pnl_sequence") or []
    if not seq:
        st.caption("No closed trades.")
        return
    import plotly.graph_objects as go
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(seq),
        x=[f"#{i + 1}" for i in range(len(seq))],
        y=seq,
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="realized P/L ($)")
    st.plotly_chart(fig, width="stretch", key="odte_ledger_waterfall")
    st.caption(f"Cumulative {_money(sum(seq))} across {len(seq)} closed trades, oldest first.")


def _render_process_matrix(summary: dict) -> None:
    st.markdown("#### Process × outcome")
    pq = summary.get("process_quality") or {}
    cells = pq.get("process_outcome") or {}
    if not cells:
        st.caption("No graded trades yet.")
        return
    c1, c2 = st.columns(2)
    c1.metric("Good process · good outcome", cells.get("good_process_good_outcome", 0))
    c1.metric("Good process · bad outcome", cells.get("good_process_bad_outcome", 0),
              help="Variance. The process is what you control.")
    c2.metric("Bad process · lucky outcome", cells.get("bad_process_lucky_outcome", 0),
              help="The dangerous cell — a win that rewards a broken process.")
    c2.metric("Bad process · bad outcome", cells.get("bad_process_bad_outcome", 0))

    layers = pq.get("failure_layers") or {}
    if layers:
        st.caption("Failure layers: " + " · ".join(f"`{k}` ×{v}" for k, v in layers.items()))
    diag = pq.get("execution_diagnosis") or {}
    if diag:
        st.caption("Diagnosis: " + " · ".join(f"{k} ×{v}" for k, v in diag.items()))


def _render_intratrade(rows: list[dict]) -> None:
    st.markdown("#### Intra-trade path")
    tradeable = [r for r in rows if r.get("entry_ts")]
    if not tradeable:
        st.caption("No trades with an entry timestamp.")
        return
    labels = [f"{r.get('trade_date')} · {r.get('underlying')} · {r.get('rail_fired') or 'open'}"
              for r in tradeable]
    pick = st.selectbox("Trade", labels, key="odte_ledger_trade_pick")
    row = tradeable[labels.index(pick)]
    series = svc.intratrade_series(row)
    if not series:
        st.caption("No management checks recorded inside this trade's bracket.")
        return

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["minutes"] for p in series], y=[p["pnl_pct"] for p in series],
                             mode="lines+markers", line_shape="hv", name="P/L %",
                             line=dict(color="#3498db")))
    if row.get("mfe_pct") is not None:
        fig.add_hline(y=row["mfe_pct"], line_dash="dot", line_color="#2ecc71",
                      annotation_text=f"MFE {row['mfe_pct']:.1f}%")
    if row.get("mae_pct") is not None:
        fig.add_hline(y=row["mae_pct"], line_dash="dot", line_color="#e74c3c",
                      annotation_text=f"MAE {row['mae_pct']:.1f}%")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="minutes held", yaxis_title="P/L %")
    st.plotly_chart(fig, width="stretch", key="odte_ledger_intratrade")

    last = series[-1]
    st.caption(f"Exited on **{row.get('rail_fired') or '—'}** at {last['pnl_pct']:.2f}% "
               f"(last check: {last.get('decision')}) · capture "
               f"{_pct(row.get('mfe_capture_pct'))} of the favourable excursion.")
    st.caption("⚠️ Joined by timestamp bracket: modern management checks carry no trade_id. Only "
               "one 0DTE position is ever open, so the bracket is unambiguous — but it is an "
               "inference, not a key.")


def _render_latency(rows: list[dict]) -> None:
    st.markdown("#### Execution latency")
    lat = svc.latency_rows()
    if not lat:
        st.info("No lease-bound fills yet — latency legs need the lease binding added 2026-08-03.")
        return

    import plotly.graph_objects as go
    labels = [f"{r.get('trade_date')} {r.get('underlying')}" for r in lat]
    fig = go.Figure()
    for name, key, color in (("lease → submit", "lease_to_submit", "#f39c12"),
                             ("submit → fill", "submit_to_fill", "#2ecc71")):
        fig.add_trace(go.Bar(name=name, y=labels, x=[r.get(key) or 0 for r in lat],
                             orientation="h", marker_color=color))
    fig.add_trace(go.Scatter(name="lease → fill", y=labels,
                             x=[r.get("lease_to_fill") for r in lat],
                             mode="markers", marker=dict(color="#3498db", size=11, symbol="diamond")))
    fig.update_layout(barmode="stack", height=max(240, 60 * len(lat)),
                      margin=dict(l=10, r=10, t=10, b=10), xaxis_title="seconds",
                      legend=dict(orientation="h", y=1.25))
    st.plotly_chart(fig, width="stretch", key="odte_ledger_latency")

    sla = svc.conversion_sla_seconds()
    fills = [r["lease_to_fill"] for r in lat if r.get("lease_to_fill") is not None]
    if fills:
        st.caption(f"Lease→fill: {' · '.join(f'{v:.1f}s' for v in fills)} — all inside the "
                   f"{sla:.0f}s conversion SLA.")
    st.caption(f"**n={len(lat)}.** These are individual trades, not a distribution — a percentile "
               "chart at this sample size would be fiction. It becomes one at roughly n≥20.")


def render() -> None:
    st.subheader("Trade ledger")
    st.caption("Every executed trade, fully attributed: tier, lease binding, exit rail, excursion "
               "capture and process grade.")

    rows = svc.trade_ledger()
    summary = svc.summary()
    _render_headline(rows, summary)
    st.divider()
    _render_table(rows)
    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        _render_waterfall(summary)
    with c2:
        _render_process_matrix(summary)
    st.divider()
    _render_intratrade(rows)
    st.divider()
    _render_latency(rows)
