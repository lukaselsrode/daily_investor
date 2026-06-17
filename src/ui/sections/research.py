"""
ui/sections/research.py — Research: alpha discovery and factor experimentation.

Tabs:
  Factor Lab · Universe · Analytics
"""
from __future__ import annotations

import streamlit as st

_FRIENDLY: dict[str, str] = {
    "value_score":    "Value",
    "quality_score":  "Quality",
    "income_score":   "Income",
    "momentum_score": "Momentum",
    "value_metric":   "Composite",
}
_ALL_FACTORS  = ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]
_ALL_HORIZONS = [5, 20, 60, 120, 252]
_DEFAULT_FACTORS = ["value_score", "quality_score", "income_score", "momentum_score"]


def _research_controls() -> tuple[list[str], list[int], str]:
    """Shared controls that feed IC-dependent tabs (Overview, IC, Deciles, Regime)."""
    c1, c2, c3 = st.columns([3, 3, 1])
    with c1:
        sel_factors = st.multiselect(
            "Factors", _ALL_FACTORS,
            default=_DEFAULT_FACTORS,
            format_func=lambda x: _FRIENDLY.get(x, x),
            key="rp_factors",
        )
    with c2:
        sel_horizons = st.multiselect(
            "Horizons (days)", _ALL_HORIZONS, default=[5, 20, 60, 120],
            key="rp_horizons",
        )
    with c3:
        ic_type = st.selectbox("IC type", ["spearman", "pearson"], key="rp_ic_type")
    return sel_factors, sel_horizons, ic_type


def render() -> None:
    tabs = st.tabs(["📊 Factor Lab", "🔍 Universe", "🧬 Analytics"])

    # ── Tab 0: Factor Lab ─────────────────────────────────────────────────────
    with tabs[0]:
        with st.expander("📐 Research controls", expanded=True):
            sel_factors, sel_horizons, ic_type = _research_controls()

        if not sel_factors or not sel_horizons:
            st.warning("Select at least one factor and one horizon to begin.")
        else:
            ic_data_ok = False
            ic_df = summary = decay = None
            n_dates = n_stocks = 0

            with st.spinner("Loading factor IC…"):
                try:
                    from ui.components.factor_lab import _compute_ic_data
                    data    = _compute_ic_data(tuple(sel_factors), tuple(sorted(sel_horizons)), ic_type)
                    ic_df   = data["ic_df"]
                    summary = data["summary"]
                    decay   = data["decay"]
                    if not ic_df.empty:
                        ic_data_ok = True
                        n_dates  = int(ic_df["date"].nunique())
                        n_stocks = int(ic_df["n_stocks"].median())
                except Exception as exc:
                    st.caption(f"IC unavailable: {exc}")

            inner = st.tabs(["📊 Overview", "📡 IC Analysis", "📊 Rank & Deciles", "🌡️ Regime"])

            with inner[0]:
                st.subheader("Research State Overview")
                st.caption(
                    "Synthesized conclusions from IC statistics. "
                    "Build more snapshots (run the bot on different days) for higher confidence."
                )
                if ic_data_ok:
                    from ui.components.factor_lab import render_overview_tab
                    render_overview_tab(summary, n_dates, n_stocks)
                else:
                    st.info(
                        "Need ≥ 2 snapshot files in `data/snapshots/` for research conclusions. "
                        "Run the bot on at least two separate days."
                    )

            with inner[1]:
                st.subheader("Information Coefficient Analysis")
                st.caption("Forward predictive power of each factor, decay curves, cumulative IC, and ICIR.")
                if ic_data_ok:
                    from ui.components.factor_lab import render_ic_analysis_tab
                    render_ic_analysis_tab(ic_df, summary, decay, sel_factors, sel_horizons, ic_type)
                else:
                    st.info("Snapshot data needed — see controls above.")
                st.divider()
                st.subheader("Rolling IC Time Series")
                st.caption("Per-snapshot IC over time — track whether signal is strengthening or degrading.")
                from ui.components.rolling_ic import render as _r
                _r()

            with inner[2]:
                st.subheader("Rank & Decile Analysis")
                st.caption(
                    "Does a higher factor score actually predict better forward returns? "
                    "Monotonically increasing deciles = genuine predictive power."
                )
                if ic_data_ok:
                    from ui.components.factor_lab import render_decile_tab
                    render_decile_tab(sel_factors, sel_horizons)
                else:
                    st.info("Snapshot data needed — run the bot on multiple days.")

            with inner[3]:
                st.subheader("Regime-Conditioned IC")
                st.caption(
                    "Same factor, very different behavior across bull / bear / high-vol / sideways markets. "
                    "Reveals which factors are regime-dependent vs. regime-agnostic."
                )
                if ic_data_ok:
                    from ui.components.factor_lab import render_regime_tab
                    render_regime_tab(sel_factors, sel_horizons, ic_type)
                else:
                    st.info("Snapshot data needed — run the bot on multiple days.")

    # ── Tab 1: Universe ───────────────────────────────────────────────────────
    with tabs[1]:
        inner = st.tabs(["📈 Scoring", "💎 Value", "🧬 Distribution", "🎯 Candidate Pool", "🗺️ Factor Map", "🕸️ News Graph", "🔎 Single Stock"])

        with inner[0]:
            from ui.components.scoring import render as _r
            _r()

        with inner[1]:
            from ui.components.value_diagnostics import render as _r
            _r()

        with inner[2]:
            from ui.components.distribution_intelligence import render as _r
            _r()

        with inner[3]:
            st.subheader("Candidate Pool")
            st.caption(
                "Analyze how the candidate_selection config filters the score distribution "
                "into a buy-eligible pool. Includes threshold sensitivity, factor distributions, "
                "income trap detection, and A/B/C selection mode comparison."
            )
            from ui.components.candidate_diagnostics import render as _r
            _r()

        with inner[4]:
            from ui.components.factor_map import render as _r
            _r()

        with inner[5]:
            from ui.components.news_graph import render as _r
            _r()

        with inner[6]:
            from ui.components.single_stock_analyzer import render as _r
            _r()

    # ── Tab 2: Analytics ──────────────────────────────────────────────────────
    with tabs[2]:
        inner = st.tabs(["🔗 Correlations", "🔀 Conditional", "🎯 Calibration", "🧪 Experimental"])

        with inner[0]:
            st.subheader("Factor Correlations & Orthogonalization")
            st.caption("Are your factors redundant? Pairwise IC, VIF, and OLS residualization.")
            from ui.components.factor_analysis import render as _r
            _r()

        with inner[1]:
            from ui.components.conditional_features import render as _r
            _r()

        with inner[2]:
            from ui.components.model_calibration import render as _r
            _r()

        with inner[3]:
            st.subheader("Experimental Workspace")
            st.caption(
                "Raw data explorer — browse any CSV, build custom views, "
                "and prototype new analyses without polluting the main research tabs."
            )
            from ui.components.data_explorer import render as _r
            _r()
