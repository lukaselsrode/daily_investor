"""
ui/components/rolling_ic.py — Rolling factor IC (Information Coefficient) analysis.

Answers: does each factor actually predict future returns, and is the predictive
power stable over time or just noise from a short backtest window?

Sections
--------
1. Snapshot coverage — how many dated snapshots exist; backfill prompt
2. IC summary table — mean IC, std, ICIR, hit rate, t-stat per factor
3. IC time series — per-factor IC plotted over dates
4. Rolling ICIR — smoothed information ratio (trailing N-period window)
5. IC distribution — histogram per factor; zero-line reference
6. Methodology note
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import streamlit as st



# ---------------------------------------------------------------------------
# Data helpers (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _list_snaps() -> list[tuple[datetime.date, str]]:
    try:
        from strategy.snapshots import list_snapshots
        return [(d, str(p)) for d, p in list_snapshots()]
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _run_ic(horizon_days: int, factor_cols: tuple[str, ...]) -> pd.DataFrame:
    from strategy.snapshots import compute_forward_ic
    return compute_forward_ic(horizon_days=horizon_days, factor_cols=list(factor_cols))


def _backfill() -> int:
    from strategy.snapshots import backfill_from_csvs
    return backfill_from_csvs()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_FACTOR_LABELS = {
    "value_score":    "Value",
    "momentum_score": "Momentum",
    "quality_score":  "Quality",
    "income_score":   "Income",
}

_HORIZON_OPTIONS = {
    "1 week  (7 d)":  7,
    "2 weeks (14 d)": 14,
    "1 month (21 d)": 21,
    "3 months (63 d)":63,
}

_IC_VERDICT = [
    (0.05, "strong positive"),
    (0.02, "weak positive"),
    (-0.02, "noise"),
    (-0.05, "weak negative"),
    (float("-inf"), "negative"),
]

def _ic_label(ic: float) -> str:
    for threshold, label in _IC_VERDICT:
        if ic >= threshold:
            return label
    return "negative"


def _ic_color(ic: float) -> str:
    if ic >= 0.05:
        return "green"
    if ic >= 0.02:
        return "limegreen"
    if ic >= -0.02:
        return "gray"
    if ic >= -0.05:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("📡 Rolling IC Analysis")
    st.caption(
        "Measures whether each factor score actually predicts future stock returns. "
        "IC (Information Coefficient) = Spearman rank correlation between factor score today "
        "and forward return at the chosen horizon. Read-only — no orders, no config writes."
    )

    # ── 1. Snapshot coverage ─────────────────────────────────────────────────
    snaps = _list_snaps()
    n_snaps = len(snaps)

    if n_snaps == 0:
        st.warning(
            "No snapshots found in `data/snapshots/`. "
            "Snapshots are saved automatically each time the scoring pipeline runs. "
            "Use the button below to import existing daily agg_data CSVs."
        )
        if st.button("Backfill snapshots from existing CSVs"):
            with st.spinner("Converting CSVs to Parquet snapshots…"):
                _list_snaps.clear()
                n = _backfill()
            st.success(f"Wrote {n} new snapshots. Reload the page to continue.")
        return

    first_date, last_date = snaps[0][0], snaps[-1][0]
    span_days = (last_date - first_date).days

    cov1, cov2, cov3 = st.columns(3)
    cov1.metric("Snapshots",  n_snaps)
    cov2.metric("Date range", f"{first_date} → {last_date}")
    cov3.metric("Span", f"{span_days} days")

    if n_snaps < 3:
        st.info(
            f"Only {n_snaps} snapshot(s) found. IC requires at least 2 snapshots separated "
            "by the chosen horizon. Keep running the bot daily — snapshots accumulate automatically."
        )
        if st.button("Backfill from existing CSVs"):
            with st.spinner("Importing…"):
                _list_snaps.clear()
                n = _backfill()
            st.success(f"Added {n} snapshot(s).")
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    st.divider()
    cc1, cc2, cc3 = st.columns([1, 2, 1])

    with cc1:
        horizon_label = st.selectbox(
            "Forward horizon",
            list(_HORIZON_OPTIONS.keys()),
            index=2,
            key="ic_horizon",
        )
        horizon_days = _HORIZON_OPTIONS[horizon_label]

    with cc2:
        all_factors = ["value_score", "momentum_score", "quality_score", "income_score"]
        chosen_factors = st.multiselect(
            "Factors",
            all_factors,
            default=all_factors,
            format_func=lambda c: _FACTOR_LABELS.get(c, c),
            key="ic_factors",
        )

    with cc3:
        rolling_window = st.number_input(
            "Rolling ICIR window (periods)",
            min_value=2, max_value=20, value=3, step=1, key="ic_roll_win",
        )

    if not chosen_factors:
        st.info("Select at least one factor.")
        return

    # Warn if horizon exceeds half the span
    if horizon_days > span_days * 0.5:
        st.warning(
            f"Horizon ({horizon_days}d) is more than half the snapshot span ({span_days}d). "
            "Few or no IC observations may be produced."
        )

    # ── Compute IC ───────────────────────────────────────────────────────────
    with st.spinner(f"Computing {horizon_label} IC across {n_snaps} snapshots…"):
        ic_df = _run_ic(horizon_days, tuple(chosen_factors))

    if ic_df.empty:
        st.warning(
            f"No IC observations produced for the {horizon_label} horizon. "
            "The snapshots may not be far enough apart yet — try a shorter horizon or wait for more data."
        )
        return

    # Pivot: rows = date, cols = factor
    ic_pivot = ic_df.pivot(index="date", columns="factor", values="ic").sort_index()

    # ── 2. IC summary table ───────────────────────────────────────────────────
    st.divider()
    st.subheader("1 · Factor IC summary")

    summary_rows = []
    for factor in chosen_factors:
        if factor not in ic_pivot.columns:
            continue
        s = ic_pivot[factor].dropna()
        if s.empty:
            continue
        mean_ic = float(s.mean())
        std_ic  = float(s.std()) if len(s) > 1 else float("nan")
        icir    = mean_ic / std_ic if std_ic and std_ic > 0 else float("nan")
        t_stat  = mean_ic / (std_ic / np.sqrt(len(s))) if std_ic and std_ic > 0 else float("nan")
        hit     = float((s > 0).mean())
        summary_rows.append({
            "Factor":    _FACTOR_LABELS.get(factor, factor),
            "Mean IC":   round(mean_ic, 4),
            "Std IC":    round(std_ic,  4),
            "ICIR":      round(icir,    3) if not np.isnan(icir) else None,
            "t-stat":    round(t_stat,  2) if not np.isnan(t_stat) else None,
            "Hit rate":  f"{hit:.0%}",
            "Periods":   len(s),
            "Signal":    _ic_label(mean_ic),
        })

    if not summary_rows:
        st.info("Not enough data to build summary table.")
        return

    sum_df = pd.DataFrame(summary_rows)

    # Highlight mean IC column
    def _style_ic(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return ""
        color = _ic_color(val)
        return f"color: {color}; font-weight: bold"

    styled = sum_df.style.map(_style_ic, subset=["Mean IC"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption(
        "ICIR = Mean IC / Std IC (higher = more consistent signal). "
        "t-stat > 2 suggests the mean IC is statistically different from zero. "
        "|IC| > 0.05 is considered a useful factor in practice."
    )

    # ── 3. IC time series ────────────────────────────────────────────────────
    st.divider()
    st.subheader("2 · IC over time")

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, factor in enumerate(chosen_factors):
            if factor not in ic_pivot.columns:
                continue
            s = ic_pivot[factor].dropna()
            color = colors[i % len(colors)]
            label = _FACTOR_LABELS.get(factor, factor)

            fig.add_trace(go.Scatter(
                x=s.index, y=s.values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"{label}<br>Date: %{{x}}<br>IC: %{{y:.4f}}<extra></extra>",
            ))

            # Zero line reference
        fig.add_hline(y=0,    line_dash="solid", line_color="black",  line_width=1)
        fig.add_hline(y=0.05, line_dash="dot",   line_color="green",  line_width=1,
                      annotation_text="IC=0.05 (useful)", annotation_position="right")
        fig.add_hline(y=-0.05,line_dash="dot",   line_color="red",    line_width=1,
                      annotation_text="IC=−0.05", annotation_position="right")

        fig.update_layout(
            title=f"Factor IC — {horizon_label} forward horizon",
            xaxis_title="Snapshot date",
            yaxis_title="Spearman IC",
            legend=dict(orientation="h", y=-0.15),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.line_chart(ic_pivot[chosen_factors].dropna(how="all"))

    # ── 4. Rolling ICIR ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("3 · Rolling ICIR")
    st.caption(
        f"IC information ratio over a trailing {rolling_window}-period window. "
        "Positive and stable ICIR = consistent predictive power."
    )

    roll_rows: dict[str, pd.Series] = {}
    for factor in chosen_factors:
        if factor not in ic_pivot.columns:
            continue
        s = ic_pivot[factor].dropna()
        if len(s) < rolling_window:
            continue
        roll_mean = s.rolling(rolling_window).mean()
        roll_std  = s.rolling(rolling_window).std()
        roll_icir = roll_mean / roll_std.replace(0, float("nan"))
        roll_rows[_FACTOR_LABELS.get(factor, factor)] = roll_icir

    if roll_rows:
        roll_df = pd.DataFrame(roll_rows)
        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            for col in roll_df.columns:
                fig2.add_trace(go.Scatter(
                    x=roll_df.index, y=roll_df[col],
                    mode="lines", name=col, line=dict(width=2),
                ))
            fig2.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig2.add_hline(y=0.5,  line_dash="dot", line_color="green", line_width=1,
                           annotation_text="ICIR=0.5", annotation_position="right")
            fig2.add_hline(y=-0.5, line_dash="dot", line_color="red",   line_width=1,
                           annotation_text="ICIR=−0.5", annotation_position="right")
            fig2.update_layout(
                title=f"Rolling {rolling_window}-period ICIR",
                xaxis_title="Date", yaxis_title="ICIR",
                legend=dict(orientation="h", y=-0.15),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.line_chart(roll_df)
    else:
        st.info(
            f"Not enough IC observations for a {rolling_window}-period rolling window. "
            "Try reducing the window or waiting for more snapshots."
        )

    # ── 5. IC distribution ───────────────────────────────────────────────────
    st.divider()
    st.subheader("4 · IC distribution")
    st.caption(
        "Distribution of IC values across all observation periods per factor. "
        "A distribution centred clearly above 0 indicates persistent positive predictive power."
    )

    dist_cols = [f for f in chosen_factors if f in ic_pivot.columns]
    if dist_cols:
        try:
            import plotly.figure_factory as ff
            import plotly.graph_objects as go

            hist_data = []
            group_labels = []
            for factor in dist_cols:
                s = ic_pivot[factor].dropna()
                if len(s) < 3:
                    continue
                hist_data.append(s.values)
                group_labels.append(_FACTOR_LABELS.get(factor, factor))

            if hist_data:
                fig3 = ff.create_distplot(hist_data, group_labels, bin_size=0.02, show_rug=True)
                fig3.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
                fig3.add_vline(x=0.05, line_dash="dot", line_color="green")
                fig3.add_vline(x=-0.05, line_dash="dot", line_color="red")
                fig3.update_layout(
                    title="IC distribution (KDE + rug)",
                    xaxis_title="IC",
                    height=380,
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig3, use_container_width=True)
        except (ImportError, Exception):
            # Fallback to simple bar chart per factor
            for factor in dist_cols:
                s = ic_pivot[factor].dropna()
                if s.empty:
                    continue
                label = _FACTOR_LABELS.get(factor, factor)
                st.caption(f"{label} IC distribution")
                counts = s.value_counts(bins=15, sort=False).sort_index()
                counts.index = [f"{i.mid:.3f}" for i in counts.index]
                st.bar_chart(counts)

    # ── 6. Raw IC table ───────────────────────────────────────────────────────
    with st.expander("Raw IC data"):
        show_df = ic_df.copy()
        show_df["factor"] = show_df["factor"].map(lambda c: _FACTOR_LABELS.get(c, c))
        show_df = show_df.sort_values(["date", "factor"]).reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        try:
            csv_bytes = show_df.to_csv(index=False).encode()
            st.download_button(
                "Download IC data as CSV",
                data=csv_bytes,
                file_name=f"rolling_ic_{horizon_days}d.csv",
                mime="text/csv",
            )
        except Exception:
            pass

    # ── 7. Methodology ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Methodology"):
        st.markdown(f"""
**How IC is computed**

1. For each snapshot at date *T*, find the nearest snapshot at *T + {horizon_days} days*
   (within ±50% of horizon).
2. Compute forward return for each symbol:
   - Primary: `current_price[T+h] / current_price[T] − 1`
   - Fallback: `return_1m` from the forward snapshot (when prices are sparse)
3. Spearman rank correlation between `factor_score[T]` and `forward_return`.

**IC interpretation**

| IC range | Interpretation |
|----------|---------------|
| > 0.10   | Strong positive — factor reliably predicts returns |
| 0.05–0.10 | Useful — worth including |
| 0.02–0.05 | Weak — marginal predictive value |
| −0.02–0.02 | Noise — no meaningful signal |
| < −0.05  | Negative — factor predicts *poor* returns (contrarian) |

**ICIR (IC Information Ratio)**

`ICIR = mean(IC) / std(IC)`

Measures consistency. A factor with IC = 0.05 every period is more useful than one
alternating between +0.15 and −0.05. ICIR > 0.5 is considered actionable.

**Data requirements**

Each IC observation requires two snapshots separated by approximately the chosen horizon.
With {n_snaps} snapshots spanning {span_days} days, you have roughly
`{n_snaps} - 1` potential IC observations at the current horizon.
Longer horizons (63d) need snapshots to span at least 2× the horizon to produce reliable estimates.

**Bias warnings**

- Snapshots are point-in-time scored universes, not survivorship-adjusted.
- `return_1m` in a snapshot is a *backward-looking* 21-day return — the price-based
  forward return computation avoids this look-ahead by using current_price from two different dates.
- Short time series (< 20 periods) produce unreliable IC estimates — t-stat < 2 means
  the signal may not be statistically distinguishable from zero.
        """)
