"""
ui/components/archetype_diagnostics.py — Per-archetype backtest performance breakdown.

Reads archetype_pnl / archetype_trade_counts / archetype_exit_breakdown from a
SimResult and renders Plotly charts + summary tables.
"""
from __future__ import annotations

import streamlit as st

_ARCHETYPE_COLORS = {
    "quality_compounder":  "#4c8ef5",
    "value_recovery":      "#f5a623",
    "defensive_income":    "#7ed321",
    "speculative_momentum":"#d0021b",
    "legacy_turnaround":   "#9b59b6",
    "core_default":        "#aaaaaa",
}

_ARCHETYPE_LABELS = {
    "quality_compounder":  "Quality Compounder",
    "value_recovery":      "Value Recovery",
    "defensive_income":    "Defensive Income",
    "speculative_momentum":"Speculative Momentum",
    "legacy_turnaround":   "Legacy Turnaround",
    "core_default":        "Core Default",
}


def render_archetype_breakdown(sim_result) -> None:
    """
    Render per-archetype PnL, trade counts, and exit type breakdown
    from a SimResult that was run with archetype_aware=True.
    """
    pnl    = sim_result.archetype_pnl
    counts = sim_result.archetype_trade_counts
    exits  = sim_result.archetype_exit_breakdown

    if not pnl and not counts:
        st.info("Archetype breakdown not available — archetype management may be disabled in config.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("plotly not installed — install with `pip install plotly`.")
        _table_fallback(pnl, counts, exits)
        return

    all_archetypes = sorted(set(list(pnl.keys()) + list(counts.keys())))

    # ── PnL bar chart ─────────────────────────────────────────────────────
    labels  = [_ARCHETYPE_LABELS.get(a, a) for a in all_archetypes]
    pnl_vals = [pnl.get(a, 0.0) for a in all_archetypes]
    colors   = [_ARCHETYPE_COLORS.get(a, "#888888") for a in all_archetypes]
    bar_colors = [
        "#4c8ef5" if v >= 0 else "#ff6b6b"
        for v in pnl_vals
    ]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=labels, y=pnl_vals,
            marker_color=bar_colors,
            hovertemplate="%{x}<br>PnL: $%{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="PnL by Archetype ($)",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="PnL ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        count_vals = [counts.get(a, 0) for a in all_archetypes]
        fig2 = go.Figure(go.Bar(
            x=labels, y=count_vals,
            marker_color=colors,
            hovertemplate="%{x}<br>Sells: %{y}<extra></extra>",
        ))
        fig2.update_layout(
            title="Sell Trades by Archetype",
            height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Sells",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Summary table ──────────────────────────────────────────────────────
    import pandas as pd
    excess  = getattr(sim_result, "archetype_active_excess",   {}) or {}
    winrate = getattr(sim_result, "archetype_win_rate",        {}) or {}
    avghold = getattr(sim_result, "archetype_avg_hold_days",   {}) or {}
    maxdd   = getattr(sim_result, "archetype_max_drawdown",    {}) or {}
    sleeve  = getattr(sim_result, "archetype_sleeve_weight",   {}) or {}
    unreal  = getattr(sim_result, "archetype_unrealized_pnl",  {}) or {}

    def _beats_spy_badge(label: str) -> str:
        v = excess.get(label)
        if v is None:
            return "—"
        if v >  0.05: return "✅ Beats"
        if v >= 0.0:  return "🟡 ~Even"
        return "❌ Lags"

    rows = []
    for a in all_archetypes:
        total_pnl = pnl.get(a, 0.0)
        n_sells   = counts.get(a, 0)
        avg_pnl   = total_pnl / n_sells if n_sells > 0 else 0.0
        ex        = exits.get(a, {})
        rows.append({
            "Archetype":     _ARCHETYPE_LABELS.get(a, a),
            "Sells":         n_sells,
            "Win rate":      f"{winrate.get(a, 0.0):.0%}" if a in winrate else "—",
            "Total PnL":     f"${total_pnl:+,.2f}",
            "Unrealized":    f"${unreal.get(a, 0.0):+,.2f}",
            "Avg PnL/trade": f"${avg_pnl:+,.2f}",
            "Avg hold (d)":  f"{avghold.get(a, 0.0):.0f}" if a in avghold else "—",
            "Max DD":        f"{maxdd.get(a, 0.0):.1%}" if a in maxdd else "—",
            "Sleeve weight": f"{sleeve.get(a, 0.0):.1%}" if a in sleeve else "—",
            "Excess vs SPY": f"{excess.get(a, 0.0):+.1%}" if a in excess else "—",
            "Beats SPY":     _beats_spy_badge(a),
            "Stop-outs":     ex.get("stop_loss", 0) + ex.get("trailing_stop", 0),
            "Take-profits":  ex.get("take_profit", 0),
            "Weak-value":    ex.get("weak_value", 0),
            "Harvests":      ex.get("harvest_exit", 0),
            "Trims":         ex.get("trim_exit", 0),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _table_fallback(pnl, counts, exits) -> None:
    import pandas as pd
    rows = []
    for a in sorted(set(list(pnl.keys()) + list(counts.keys()))):
        rows.append({
            "Archetype": _ARCHETYPE_LABELS.get(a, a),
            "Sells": counts.get(a, 0),
            "Total PnL": f"${pnl.get(a, 0.0):+,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
