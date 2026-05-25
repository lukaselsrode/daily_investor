"""
ui/components/model_calibration.py — Model Calibration research tab.

Displays:
  - EXIT / WATCH / REVIEW / HOLD accuracy from realized outcomes
  - Premature exit rate and false exit rate
  - Decision confusion matrix
  - Historical confidence calibration (confidence vs realized accuracy)
  - Calibration state (current threshold suggestions)

Data source: data/decision_outcomes.parquet (written by outcome_tracker.py)
Reads calibration_state.json for current threshold settings.

SAFE: this module is read-only — it never writes factor scores or weights.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Cache wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_calibration():
    try:
        from research.decision_calibration import compute_calibration
        return compute_calibration()
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=300)
def _load_outcomes():
    try:
        from portfolio.outcome_tracker import load_outcomes
        return load_outcomes()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_summary_metrics(result) -> None:
    st.subheader("Decision Accuracy Overview")
    st.caption(
        f"Based on {result.n_with_outcomes} decisions with realized 30-day returns "
        f"(out of {result.n_total} total recorded)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "EXIT Accuracy",
        f"{result.exit_accuracy:.0%}",
        help="% of EXIT decisions where stock declined within 30 days",
    )
    c2.metric(
        "Premature Exit Rate",
        f"{result.premature_exit_rate:.0%}",
        delta=f"{result.premature_exit_rate - 0.20:+.0%} vs 20% baseline",
        delta_color="inverse",
        help="% of EXITs that gained >2% afterward — missed gains",
    )
    c3.metric(
        "WATCH Recovery",
        f"{result.watch_recovery_rate:.0%}",
        help="% of WATCH positions that recovered (+1%) within 30 days",
    )
    c4.metric(
        "HOLD vs SPY",
        f"{result.hold_outperformance_rate:.0%}",
        help="% of HOLDs that outperformed SPY in 30 days",
    )

    # Second row
    c5, c6, _, _ = st.columns(4)
    c5.metric(
        "False Exit Rate",
        f"{result.false_exit_rate:.0%}",
        delta=f"{result.false_exit_rate - 0.15:+.0%} vs 15% baseline",
        delta_color="inverse",
        help="% of EXITs that subsequently outperformed SPY",
    )
    c6.metric(
        "REVIEW Precision",
        f"{result.review_precision:.0%}",
        help="% of REVIEW flags where the price moved decisively (>5%) within 30d",
    )


def _render_by_state(result) -> None:
    st.subheader("Outcomes by Decision State")
    if result.by_state is None or result.by_state.empty:
        st.info("Not enough data yet. Outcomes are recorded each time the bot runs.")
        return

    df = result.by_state.copy()
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    if "Mean 30D Return" in df.columns:
        df["Mean 30D Return"] = df["Mean 30D Return"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    if "Pct Positive" in df.columns:
        df["Pct Positive"] = df["Pct Positive"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    if "Pct Negative" in df.columns:
        df["Pct Negative"] = df["Pct Negative"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    st.dataframe(df, use_container_width=True)


def _render_confusion_matrix(result) -> None:
    st.subheader("Decision Confusion Matrix")
    st.caption("Predicted state vs actual 30-day outcome (up = positive return)")

    if result.confusion_matrix is None:
        st.info("Not enough outcomes to compute confusion matrix.")
        return

    st.dataframe(result.confusion_matrix, use_container_width=True)


def _render_calibration_curve(result) -> None:
    st.subheader("Confidence Calibration")
    st.caption(
        "Are HIGH confidence decisions actually more accurate than LOW confidence ones? "
        "A well-calibrated system shows: HIGH > MEDIUM > LOW accuracy."
    )

    if result.calibration_curve is None or result.calibration_curve.empty:
        st.info("Not enough data to compute calibration curve.")
        return

    import plotly.graph_objects as go

    df = result.calibration_curve
    conf_order = ["HIGH", "MEDIUM", "LOW"]
    df = df.set_index("confidence").reindex(conf_order).reset_index().dropna(subset=["n"])

    colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
    bar_colors = [colors.get(c, "#7f8c8d") for c in df["confidence"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["confidence"],
        y=df["mean_30d_return"] * 100 if "mean_30d_return" in df.columns else [],
        name="Mean 30d Return (%)",
        marker_color=bar_colors,
        text=[f"n={int(n)}" for n in df["n"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Mean 30-Day Return by Decision Confidence",
        yaxis_title="Mean 30d Return (%)",
        xaxis_title="Confidence Level",
        showlegend=False,
        height=350,
        margin=dict(t=40, b=40),
        yaxis=dict(tickformat=".1f", ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True, key="calib_curve_bar")


def _render_calibration_state() -> None:
    st.subheader("Current Calibration State")
    st.caption(
        "Threshold suggestions computed from realized outcomes. "
        "These influence WHEN the system escalates exits to REVIEW — "
        "never the factor scores or composite formula."
    )

    try:
        from research.decision_calibration import load_calibration_state
        state = load_calibration_state()
    except Exception:
        state = {}

    if not state:
        st.info("No calibration state computed yet. Run the bot multiple times to accumulate outcomes.")
        return

    cols = st.columns(3)
    cols[0].metric(
        "Premature Exit Threshold",
        f"{state.get('premature_exit_threshold', 0.45):.0%}",
        help="PEP above this → escalate EXIT to REVIEW",
    )
    cols[1].metric(
        "Review Confidence Gate",
        f"{state.get('review_confidence_threshold', 0.50):.0%}",
        help="Thesis intact score above this + LOW exit confidence → REVIEW",
    )
    cols[2].metric(
        "Premature Exit Rate",
        f"{state.get('premature_exit_rate', 0.0):.0%}",
        help="Empirical rate from realized outcomes",
    )

    last = state.get("last_updated", "Never")
    n    = state.get("n_decisions", 0)
    st.caption(f"Last updated: {last} | Based on {n} decisions")


def _render_raw_outcomes() -> None:
    st.subheader("Raw Decision Log")
    df = _load_outcomes()
    if df.empty:
        st.info("No decisions recorded yet.")
        return

    display_cols = [
        "ticker", "decision_date", "decision_state", "decision_confidence",
        "thesis_intact_score", "premature_exit_probability",
        "composite_score", "holding_days", "portfolio_pnl",
        "future_7d_return", "future_30d_return", "future_90d_return",
        "outperformed_spy_30d",
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[show_cols].tail(200), use_container_width=True)


def _render_apply_calibration(result) -> None:
    st.subheader("Apply Calibration")
    st.caption(
        "Save the calibration engine's threshold suggestions to "
        "data/calibration_state.json so the Decision Adjustment Engine "
        "uses them on the next run."
    )

    if not result.is_reliable:
        st.warning(
            f"Calibration requires at least 10 resolved outcomes ({result.n_with_outcomes} available). "
            "Run the bot more to accumulate data."
        )
        return

    pet = result.suggested_premature_exit_threshold
    rct = result.suggested_review_confidence_threshold

    c1, c2 = st.columns(2)
    c1.metric("Suggested Premature Exit Threshold", f"{pet:.0%}")
    c2.metric("Suggested Review Confidence Gate",   f"{rct:.0%}")

    if st.button("Save Calibration", key="save_calibration_btn", type="primary"):
        try:
            from research.decision_calibration import save_calibration_state
            save_calibration_state(result)
            st.success("Calibration saved. Thresholds will take effect on the next bot run.")
            _load_calibration.clear()
        except Exception as exc:
            st.error(f"Could not save calibration: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Model Calibration")
    st.caption(
        "Track decision quality using realized outcomes. "
        "Calibration adjusts decision thresholds only — "
        "factor scores and weights are never modified here."
    )

    result = _load_calibration()
    if result is None:
        st.error("Could not load calibration data.")
        return

    if result.n_total == 0:
        st.info(
            "No decision outcomes recorded yet. "
            "Decision outcomes are written each time the bot runs. "
            "Come back after a few runs to see calibration metrics."
        )
        return

    tabs = st.tabs([
        "📊 Accuracy Summary",
        "📋 By State",
        "🔀 Confusion Matrix",
        "📈 Confidence Calibration",
        "⚙️ Calibration State",
        "🗂️ Raw Log",
    ])

    with tabs[0]:
        _render_summary_metrics(result)
        st.divider()
        _render_apply_calibration(result)

    with tabs[1]:
        _render_by_state(result)

    with tabs[2]:
        _render_confusion_matrix(result)

    with tabs[3]:
        _render_calibration_curve(result)

    with tabs[4]:
        _render_calibration_state()

    with tabs[5]:
        _render_raw_outcomes()
