"""
ui/components/attribution.py — Portfolio attribution for Portfolio section.

Two sub-sections:
  1. Parameter Stability Attribution — uses reporting/attribution.py (live)
  2. Factor Attribution — stub pending BacktestReport.trade_log instrumentation
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from ui.components.common import empty_state, section, df_download, warn_banner


# ---------------------------------------------------------------------------
# Parameter stability attribution
# ---------------------------------------------------------------------------

def _render_stability_attribution() -> None:
    st.subheader("Parameter Stability Attribution")
    st.caption(
        "How stable are tuned parameters across windows? "
        "High CV (>0.30) or spread (>0.15) indicates overfitting risk."
    )

    reports_dir = None
    try:
        from pathlib import Path
        import sys
        src = Path(__file__).resolve().parents[2]
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from ui.utils import DATA_DIR
        reports_dir = DATA_DIR.parent / "reports" / "stability"
    except Exception:
        pass

    if reports_dir is None or not reports_dir.exists():
        warn_banner(
            "No stability reports found in `reports/stability/`. "
            "Run `daily-investor stability-scan` to generate them."
        )
        return

    import json
    scan_files = sorted(reports_dir.glob("*.json"))
    if not scan_files:
        empty_state("No stability scan results", "Run `daily-investor stability-scan` first.")
        return

    # Load latest scan
    chosen = st.selectbox("Scan file", [f.name for f in reversed(scan_files)], key="attr_scan")
    path = reports_dir / chosen
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        st.error(f"Could not load {chosen}: {exc}")
        return

    window_results = data.get("window_results", [])
    param_names    = data.get("param_names", [])

    if not window_results or not param_names:
        st.warning("Scan file has no window_results or param_names.")
        return

    try:
        from reporting.attribution import AttributionReporter
        reporter = AttributionReporter()
        stab_df = reporter.compute_stability(window_results, param_names)
        if stab_df.empty:
            st.info("Stability computation returned empty result.")
            return

        # Colour unstable rows
        def _colour_stability(val):
            return "color:#e74c3c;font-weight:600" if val == "UNSTABLE" else "color:#27ae60"

        display = stab_df.copy()
        styled = display.style
        if "stability" in display.columns:
            styled = styled.applymap(_colour_stability, subset=["stability"])
        st.dataframe(styled, use_container_width=True)
        df_download(stab_df, "stability_attribution.csv")

    except Exception as exc:
        st.error(f"Attribution failed: {exc}")


# ---------------------------------------------------------------------------
# Factor attribution (stub)
# ---------------------------------------------------------------------------

def _render_factor_attribution() -> None:
    st.subheader("Factor Attribution")
    st.caption(
        "Attribute P&L to individual factors (value, quality, income, momentum) "
        "based on backtest trade logs."
    )

    st.info(
        "Factor attribution requires BacktestReport.trade_log to be populated. "
        "This is currently a stub — full attribution will be available once "
        "the backtest engine instruments per-trade factor contributions.",
        icon="🔧",
    )

    # Show whatever we can from decision_outcomes
    try:
        from portfolio.outcome_tracker import load_outcomes
        df = load_outcomes()
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        empty_state("No decision data", "Run the bot to start collecting decisions.")
        return

    factor_cols = [c for c in ("value_score", "quality_score", "income_score", "momentum_score") if c in df.columns]
    if not factor_cols:
        st.info("Factor score columns not found in decision data.")
        return

    outcome_col = "future_30d_return"
    if outcome_col not in df.columns or df[outcome_col].isna().all():
        st.info(
            "30d outcomes not yet populated — run `daily-investor update-outcomes` "
            "7+ days after recording decisions to see factor attribution."
        )
        return

    section("Factor Score vs 30d Forward Return", "Holdings only")
    holdings = df[df["record_type"] == "holding"].dropna(subset=[outcome_col]) if "record_type" in df.columns else df.dropna(subset=[outcome_col])

    if holdings.empty:
        st.info("No resolved holding decisions yet.")
        return

    for col in factor_cols:
        if col not in holdings.columns:
            continue
        with st.expander(col):
            try:
                plot_df = holdings[[col, outcome_col]].dropna()
                plot_df.columns = ["factor_score", "fwd_return_30d"]
                plot_df = plot_df.sort_values("factor_score")
                st.scatter_chart(plot_df, x="factor_score", y="fwd_return_30d")
                corr = plot_df["factor_score"].corr(plot_df["fwd_return_30d"])
                st.caption(f"Spearman r ≈ {corr:.3f}  (N={len(plot_df)})")
            except Exception as exc:
                st.caption(f"Plot failed: {exc}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    sub = st.tabs(["📊 Parameter Stability", "🎯 Factor Attribution"])
    with sub[0]:
        _render_stability_attribution()
    with sub[1]:
        _render_factor_attribution()
