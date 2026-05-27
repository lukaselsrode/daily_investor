"""
ui/components/archetype_attribution.py — Archetype performance attribution.

Shows per-archetype calibration metrics derived from decision_outcomes.parquet:
  - Trim success rate   (did TRIM avoid further losses?)
  - Premature exit rate (did EXIT miss subsequent gains?)
  - Bad hold rate       (did HOLD ride a loss while SPY was flat?)
  - Harvest regret rate (did HARVEST leave >10% on the table?)
  - Average 30d return after decision
  - Average holding period

SAFE: read-only. Never writes factor scores, weights, or config.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

_ARCHETYPE_LABELS = {
    "quality_compounder":   ("Quality Compounder",   "#1e88e5"),
    "legacy_turnaround":    ("Legacy Turnaround",     "#e53935"),
    "speculative_momentum": ("Speculative Momentum",  "#fb8c00"),
    "value_recovery":       ("Value Recovery",        "#43a047"),
    "defensive_income":     ("Defensive Income",      "#8e24aa"),
    "core_default":         ("Core Default",          "#757575"),
    "etf_core":             ("ETF / Core",            "#00acc1"),
}


@st.cache_data(ttl=300)
def _load_summary():
    try:
        from portfolio.outcome_tracker import get_archetype_calibration_summary
        return get_archetype_calibration_summary(), None
    except Exception as exc:
        return {}, str(exc)


@st.cache_data(ttl=300)
def _load_raw():
    try:
        from portfolio.outcome_tracker import load_outcomes
        df = load_outcomes()
        if "archetype" not in df.columns:
            return pd.DataFrame(), None
        return df[df["archetype"].notna()], None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def _pct(v) -> str:
    return f"{v:.1%}" if v is not None else "—"


def _f1(v) -> str:
    return f"{v:.1f}" if v is not None else "—"


def render() -> None:
    st.subheader("Archetype Performance Attribution")
    st.caption(
        "Calibration metrics broken down by archetype label. "
        "Data source: `decision_outcomes.parquet`. "
        "Only archetypes with at least one resolved decision are shown."
    )

    summary, err = _load_summary()
    if err:
        st.error(f"Could not load archetype summary: {err}")
        return

    if not summary:
        st.info(
            "No archetype-labelled decisions yet. "
            "Archetypes are recorded from the next live run once `archetype_management.enabled: true` "
            "is set in config.yaml."
        )
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for arch, m in sorted(summary.items(), key=lambda x: -(x[1].get("n_total") or 0)):
        label, _ = _ARCHETYPE_LABELS.get(arch, (arch, "#999"))
        rows.append({
            "Archetype":          label,
            "n":                  m.get("n_total", 0),
            "Trim success":       _pct(m.get("trim_success_rate")),
            "Premature exit":     _pct(m.get("premature_exit_rate")),
            "Bad hold":           _pct(m.get("bad_hold_rate")),
            "Harvest regret":     _pct(m.get("harvest_regret_rate")),
            "Avg 30d return":     _pct(m.get("avg_30d_return")),
            "Avg hold (days)":    _f1(m.get("avg_holding_days")),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Per-archetype detail cards ────────────────────────────────────────────
    st.divider()
    st.markdown("**Decision counts by archetype**")

    cols = st.columns(min(len(summary), 4))
    for i, (arch, m) in enumerate(sorted(summary.items(), key=lambda x: -(x[1].get("n_total") or 0))):
        label, color = _ARCHETYPE_LABELS.get(arch, (arch, "#999"))
        col = cols[i % len(cols)]
        with col:
            st.markdown(
                f"<div style='border-left:4px solid {color}; padding:8px 12px; margin-bottom:8px'>"
                f"<b>{label}</b><br>"
                f"<span style='font-size:0.85em; color:#888'>"
                f"Trim: {_pct(m.get('trim_success_rate'))} ✓ &nbsp;"
                f"Exit: {_pct(m.get('premature_exit_rate'))} early &nbsp;"
                f"n={m.get('n_total',0)}"
                f"</span></div>",
                unsafe_allow_html=True,
            )

    # ── Raw decision history filtered by archetype ────────────────────────────
    st.divider()
    st.markdown("**Drill-down: decisions for a specific archetype**")

    raw_df, raw_err = _load_raw()
    if raw_err:
        st.warning(f"Could not load raw decisions: {raw_err}")
        return
    if raw_df.empty:
        return

    arch_options = sorted(raw_df["archetype"].unique())
    arch_labels  = [_ARCHETYPE_LABELS.get(a, (a, ""))[0] for a in arch_options]
    selected_label = st.selectbox("Select archetype", arch_labels, key="arch_attr_select")
    selected_arch  = arch_options[arch_labels.index(selected_label)]

    sub = raw_df[raw_df["archetype"] == selected_arch].copy()

    display_cols = [c for c in [
        "decision_date", "symbol", "decision_state", "percent_change",
        "archetype_confidence", "future_30d_return", "good_trim",
        "premature_exit", "bad_hold", "holding_days", "sector",
    ] if c in sub.columns]

    st.dataframe(
        sub[display_cols].sort_values("decision_date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
