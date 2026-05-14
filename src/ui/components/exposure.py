"""
ui/components/exposure.py — Exposure Dashboard: factor tilts, sector weights, concentration.

Requires:
  - A live Robinhood session (read-only: build_holdings)
  - A scored universe CSV or snapshot parquet (latest available)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _load_universe() -> pd.DataFrame:
    """Load the most recent scored universe (CSV or snapshot parquet)."""
    try:
        from util import DATA_DIRECTORY
        data_dir = Path(DATA_DIRECTORY)
    except ImportError:
        return pd.DataFrame()

    csvs = sorted(data_dir.glob("agg_data_*.csv"), reverse=True)
    if csvs:
        try:
            return pd.read_csv(csvs[0])
        except Exception:
            pass

    snap_dir = data_dir / "snapshots"
    if snap_dir.exists():
        parquets = sorted(snap_dir.glob("*.parquet"), reverse=True)
        if parquets:
            try:
                return pd.read_parquet(parquets[0])
            except Exception:
                pass

    return pd.DataFrame()


def _load_portfolio() -> tuple[dict, float, float]:
    """
    Load live portfolio from Robinhood.

    Returns (portfolio_dict, total_equity, available_cash).
    portfolio_dict: {symbol: {equity, quantity, sector, is_etf}}
    """
    try:
        import robin_stocks.robinhood as rb
        from util import ETFS

        holdings  = rb.account.build_holdings() or {}
        profile   = rb.account.build_user_profile() or {}
        total_eq  = float(profile.get("equity", 0) or 0)
        cash      = float(profile.get("cash", 0) or 0)
        etf_set   = set(ETFS)

        portfolio = {}
        for sym, pos in holdings.items():
            portfolio[sym] = {
                "equity":        float(pos.get("equity") or 0),
                "quantity":      float(pos.get("quantity") or 0),
                "avg_buy_price": float(pos.get("average_buy_price") or 0),
                "sector":        pos.get("equity_fundamentals", {}).get("sector") or "Unknown",
                "is_etf":        sym in etf_set,
            }
        return portfolio, total_eq, cash

    except Exception:
        return {}, 0.0, 0.0


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render() -> None:
    st.title("⚖️ Exposure Dashboard")
    st.caption("Portfolio factor tilts, sector concentration, and rolling exposure drift.")

    # ── Load data ────────────────────────────────────────────────────────────
    with st.spinner("Loading portfolio and universe data…"):
        universe_df              = _load_universe()
        portfolio, total_eq, cash = _load_portfolio()

    if not portfolio:
        st.warning(
            "No live portfolio data. Robinhood authentication may be required, "
            "or the portfolio is empty."
        )
        if not universe_df.empty:
            _render_universe_distribution(universe_df)
        return

    # ── Compute exposure ─────────────────────────────────────────────────────
    try:
        from portfolio.exposure.analyzer import ExposureAnalyzer
        report = ExposureAnalyzer().analyze(
            portfolio, universe_df, total_equity=total_eq, cash=cash
        )
    except Exception as exc:
        st.error(f"Exposure computation failed: {exc}")
        st.exception(exc)
        return

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.subheader("Portfolio snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Positions",   report.n_positions)
    c2.metric("Total equity", f"${report.total_equity:,.0f}")
    c3.metric("ETF %",        f"{report.etf_pct:.1%}")
    c4.metric("Active %",     f"{report.stock_pct:.1%}")
    c5.metric("Cash %",       f"{report.cash_pct:.1%}")

    st.divider()

    # ── Factor tilts ─────────────────────────────────────────────────────────
    st.subheader("Factor tilts (vs. universe median)")
    st.caption(
        "Positive = portfolio holds stocks with above-median scores on this factor. "
        "Units: standard deviations of the universe distribution."
    )

    tilt_names = ["Value", "Quality", "Income", "Momentum", "Composite"]
    tilt_vals  = [
        report.value_tilt,
        report.quality_tilt,
        report.income_tilt,
        report.momentum_tilt,
        report.composite_tilt,
    ]
    tilt_colors = [
        "#00b300" if v > 0.1 else "#cc0000" if v < -0.1 else "#888888"
        for v in tilt_vals
    ]

    fig_tilt = go.Figure(go.Bar(
        x=tilt_names,
        y=tilt_vals,
        marker_color=tilt_colors,
        text=[f"{v:+.2f}σ" for v in tilt_vals],
        textposition="outside",
    ))
    fig_tilt.add_hline(y=0,    line_dash="solid", line_color="gray",  line_width=0.5)
    fig_tilt.add_hline(y=0.5,  line_dash="dash",  line_color="green", line_width=1,
                       annotation_text="+0.5σ")
    fig_tilt.add_hline(y=-0.5, line_dash="dash",  line_color="red",   line_width=1,
                       annotation_text="-0.5σ")
    fig_tilt.update_layout(
        yaxis_title="Tilt (standard deviations)",
        height=300,
        margin=dict(t=10),
    )
    st.plotly_chart(fig_tilt, use_container_width=True)

    st.divider()

    # ── Sector weights + concentration ───────────────────────────────────────
    col_sec, col_conc = st.columns([3, 1])

    with col_sec:
        st.subheader("Sector allocation")
        if report.sector_weights:
            sw = pd.DataFrame(
                [(s, w) for s, w in report.sector_weights.items()],
                columns=["Sector", "Weight"],
            ).sort_values("Weight")

            fig_sec = go.Figure(go.Bar(
                x=sw["Weight"] * 100,
                y=sw["Sector"],
                orientation="h",
                text=[f"{w:.1%}" for w in sw["Weight"]],
                textposition="outside",
                marker_color="steelblue",
            ))
            fig_sec.update_layout(
                xaxis_title="Weight (%)",
                height=max(200, len(report.sector_weights) * 28),
                margin=dict(t=10),
            )
            st.plotly_chart(fig_sec, use_container_width=True)

    with col_conc:
        st.subheader("Concentration")
        hhi_label = (
            "Low" if report.hhi < 0.05 else
            "Moderate" if report.hhi < 0.15 else
            "High"
        )
        st.metric("HHI",           f"{report.hhi:.3f}",  help="0=diversified, 1=concentrated")
        st.metric("Top-5 weight",  f"{report.top5_pct:.1%}")
        st.caption(f"Concentration level: **{hhi_label}**")

        if report.beta_spy is not None:
            st.metric("Beta (SPY)", f"{report.beta_spy:.2f}")

    st.divider()

    # ── Positions table ──────────────────────────────────────────────────────
    st.subheader("Holdings with factor scores")
    if report.positions:
        rows = []
        for p in sorted(report.positions, key=lambda x: x.equity, reverse=True):
            rows.append({
                "Symbol":    p.symbol,
                "Sector":    p.sector,
                "Weight":    f"{p.weight:.1%}",
                "Value":     p.value_score,
                "Quality":   p.quality_score,
                "Income":    p.income_score,
                "Momentum":  p.momentum_score,
                "Composite": p.value_metric,
                "ETF":       "✓" if p.is_etf else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Rolling drift ────────────────────────────────────────────────────────
    st.subheader("Exposure drift over time")
    drift_days = st.slider("Look-back (days)", 30, 365, 90, key="exp_drift_days")

    with st.spinner("Computing exposure drift from snapshots…"):
        try:
            from portfolio.exposure.analyzer import ExposureAnalyzer
            drift_df = ExposureAnalyzer().compute_rolling_drift(portfolio, days=drift_days)
        except Exception:
            drift_df = pd.DataFrame()

    if not drift_df.empty:
        fig_drift = go.Figure()
        color_map = {
            "value_tilt":    "royalblue",
            "quality_tilt":  "green",
            "income_tilt":   "orange",
            "momentum_tilt": "red",
        }
        label_map = {
            "value_tilt":    "Value",
            "quality_tilt":  "Quality",
            "income_tilt":   "Income",
            "momentum_tilt": "Momentum",
        }
        for col, label in label_map.items():
            if col in drift_df.columns:
                fig_drift.add_trace(go.Scatter(
                    x=drift_df["date"],
                    y=drift_df[col],
                    name=label,
                    mode="lines",
                    line=dict(color=color_map[col], width=2),
                ))
        fig_drift.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
        fig_drift.update_layout(
            yaxis_title="Tilt (σ vs. universe)",
            height=300,
            margin=dict(t=10),
        )
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info(
            "Not enough snapshot history for drift analysis. "
            "Build more history by running the bot daily — each run creates a new snapshot."
        )


# ---------------------------------------------------------------------------
# Universe distribution helper (shown when portfolio is unavailable)
# ---------------------------------------------------------------------------


def _render_universe_distribution(df: pd.DataFrame) -> None:
    """Show factor score distribution across the scored universe."""
    st.subheader("Universe factor distribution")
    factor_cols = [
        c for c in ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]
        if c in df.columns
    ]
    if not factor_cols:
        st.info("No factor columns found in universe data.")
        return

    fig = go.Figure()
    friendly = {
        "value_score": "Value", "quality_score": "Quality",
        "income_score": "Income", "momentum_score": "Momentum",
        "value_metric": "Composite",
    }
    for col in factor_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        fig.add_trace(go.Violin(
            y=s, name=friendly.get(col, col),
            box_visible=True, meanline_visible=True,
        ))
    fig.update_layout(yaxis_title="Score", height=360, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
