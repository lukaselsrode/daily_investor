"""
ui/components/portfolio.py — Portfolio Intelligence Cockpit.

Answers: "Why do I own this, what is it doing, and should I still own it?"

Tabs:
  Overview            — summary cards, status counts, portfolio character
  Holdings Intel      — full position intelligence table
  Active Sleeve       — active positions with expandable rationale + charts
  ETF Sleeve          — ETF holdings with role descriptions
  Attribution         — factor tilt vs universe, sleeve P/L attribution
  Journal             — position event log
"""

from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR, data_date, load_config_raw, load_latest_csv, no_data_msg


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FLOAT_COLS = [
    "quantity", "average_buy_price", "equity",
    "percent_change", "equity_change", "percentage", "current_price", "pe_ratio",
]

_STATE_COLOR: dict[str, str] = {
    "BUY":     "#2ecc71",
    "HOLD":    "#3498db",
    "WATCH":   "#f39c12",
    "REVIEW":  "#9b59b6",
    "TRIM":    "#e67e22",
    "HARVEST": "#1abc9c",
    "EXIT":    "#e74c3c",
}

_STATE_ICON: dict[str, str] = {
    "BUY":     "🟢",
    "HOLD":    "🔵",
    "WATCH":   "🟡",
    "REVIEW":  "🟣",
    "TRIM":    "🔶",
    "HARVEST": "💰",
    "EXIT":    "🔴",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def _load_holdings() -> Optional[pd.DataFrame]:
    df = load_latest_csv("holdings")
    if df is None:
        return None
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=120)
def _load_agg() -> Optional[pd.DataFrame]:
    return load_latest_csv("agg_data")


@st.cache_data(ttl=300)
def _load_buy_context() -> pd.DataFrame:
    from portfolio.buy_context import backfill_buy_context
    return backfill_buy_context()


@st.cache_data(ttl=3600)
def _load_score_history(symbol: str) -> pd.DataFrame:
    """Load per-symbol score history from snapshots."""
    try:
        from strategy.snapshots import load_snapshots
        all_snaps = load_snapshots()
        if all_snaps.empty or "symbol" not in all_snaps.columns:
            return pd.DataFrame()
        sym_df = all_snaps[all_snaps["symbol"] == symbol].copy()
        if "snapshot_date" in sym_df.columns:
            sym_df = sym_df.sort_values("snapshot_date")
        return sym_df
    except Exception:
        return pd.DataFrame()


def _load_peak_prices() -> dict[str, float]:
    path = DATA_DIR / "peak_prices.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        return df.set_index("symbol")["peak_price"].to_dict()
    except Exception:
        return {}


def _load_journal() -> pd.DataFrame:
    from portfolio.position_journal import load_journal
    return load_journal()


# ---------------------------------------------------------------------------
# Universe rank computation
# ---------------------------------------------------------------------------

def _compute_universe_ranks(agg: pd.DataFrame) -> dict[str, float]:
    """Percentile rank of value_metric across today's universe."""
    if agg is None or "value_metric" not in agg.columns or "symbol" not in agg.columns:
        return {}
    vm = pd.to_numeric(agg["value_metric"], errors="coerce")
    universe_size = vm.notna().sum()
    if universe_size == 0:
        return {}
    ranks: dict[str, float] = {}
    for _, row in agg.iterrows():
        sym = row.get("symbol")
        v   = pd.to_numeric(row.get("value_metric"), errors="coerce")
        if sym and not pd.isna(v):
            ranks[str(sym)] = float((vm < v).sum() / universe_size)
    return ranks


# ---------------------------------------------------------------------------
# Build enriched holdings frame
# ---------------------------------------------------------------------------

def _enrich_holdings(
    holdings: pd.DataFrame,
    agg: Optional[pd.DataFrame],
    buy_ctx_df: pd.DataFrame,
    peak_prices: dict[str, float],
    etfs: list[str],
) -> pd.DataFrame:
    """
    Merge holdings with agg_data, buy_context, and position rationale.
    Returns enriched DataFrame with state, rationale, factor scores, etc.
    """
    from portfolio.position_rationale import build_position_rationale, etf_role

    df = holdings.copy()
    df["sleeve"] = df["symbol"].apply(lambda s: "ETF/core" if s in etfs else "active")

    if agg is not None and "symbol" in agg.columns:
        score_cols = [
            "sector", "industry", "value_metric", "quality_score", "momentum_score",
            "income_score", "value_score", "yield_trap_flag", "reliability_score",
            "strategy_bucket", "above_200dma", "above_50dma",
        ]
        merge_cols = ["symbol"] + [c for c in score_cols if c in agg.columns]
        df = df.merge(agg[merge_cols], on="symbol", how="left")

    # Universe rank percentiles
    ranks = _compute_universe_ranks(agg)

    # Buy context lookup
    ctx_by_sym: dict[str, dict] = {}
    if not buy_ctx_df.empty and "symbol" in buy_ctx_df.columns:
        for _, r in buy_ctx_df.iterrows():
            s = str(r.get("symbol", "")).strip()
            if s:
                ctx_by_sym[s] = r.to_dict()

    # Holding days from buy_context
    def _holding_days(sym: str) -> Optional[int]:
        ctx = ctx_by_sym.get(sym, {})
        bd_str = str(ctx.get("buy_date", "")).strip()
        try:
            bd = datetime.date.fromisoformat(bd_str)
            return (datetime.date.today() - bd).days
        except Exception:
            return None

    # Build rationale for each row
    state_col = []
    state_reason_col = []
    rationale_col = []
    top_pos_col = []
    top_neg_col = []
    risk_flags_col = []
    next_action_col = []
    score_at_buy_col = []
    score_delta_col = []
    rank_pct_now_col = []
    rank_pct_buy_col = []
    holding_days_col = []
    etf_role_col = []
    thesis_intact_col = []
    exit_analysis_col = []
    decision_output_col = []

    for _, row in df.iterrows():
        sym     = str(row.get("symbol", ""))
        sleeve  = str(row.get("sleeve", "active"))
        holding = row.to_dict()
        ctx     = ctx_by_sym.get(sym)
        peak    = peak_prices.get(sym)
        rank    = ranks.get(sym)

        if sleeve == "ETF/core":
            etf_role_col.append(etf_role(sym))
            state_col.append("HOLD")
            state_reason_col.append("ETF core position")
            rationale_col.append(etf_role(sym))
            top_pos_col.append("—")
            top_neg_col.append("—")
            risk_flags_col.append([])
            next_action_col.append("Hold")
            score_at_buy_col.append(None)
            score_delta_col.append(None)
            rank_pct_now_col.append(None)
            rank_pct_buy_col.append(None)
            holding_days_col.append(_holding_days(sym))
            thesis_intact_col.append(True)
            exit_analysis_col.append(None)
            decision_output_col.append(None)
        else:
            metrics = None
            if agg is not None and "symbol" in agg.columns:
                r = agg[agg["symbol"] == sym]
                if not r.empty:
                    metrics = r.iloc[0]

            pr = build_position_rationale(
                symbol=sym,
                sleeve=sleeve,
                holding=holding,
                metrics=metrics,
                buy_context=ctx,
                peak_price=peak,
                universe_rank_pct=rank,
            )
            etf_role_col.append("")
            state_col.append(pr.state)
            state_reason_col.append(pr.state_reason)
            rationale_col.append(pr.rationale)
            top_pos_col.append(pr.top_positive_factor)
            top_neg_col.append(pr.top_negative_factor)
            risk_flags_col.append(pr.risk_flags)
            next_action_col.append(pr.next_action)
            score_at_buy_col.append(pr.score_at_buy)
            score_delta_col.append(pr.score_delta)
            rank_pct_now_col.append(pr.rank_pct_now)
            rank_pct_buy_col.append(pr.rank_pct_at_buy)
            holding_days_col.append(_holding_days(sym))
            thesis_intact_col.append(pr.thesis_intact)
            exit_analysis_col.append(pr.exit_analysis)
            decision_output_col.append(pr.decision_output)

    df["state"]           = state_col
    df["state_reason"]    = state_reason_col
    df["rationale"]       = rationale_col
    df["top_positive"]    = top_pos_col
    df["top_negative"]    = top_neg_col
    df["risk_flags"]      = risk_flags_col
    df["next_action"]     = next_action_col
    df["score_at_buy"]    = score_at_buy_col
    df["score_delta"]     = score_delta_col
    df["rank_pct_now"]    = rank_pct_now_col
    df["rank_pct_at_buy"] = rank_pct_buy_col
    df["holding_days"]    = holding_days_col
    df["etf_role"]        = etf_role_col
    df["thesis_intact"]   = thesis_intact_col
    df["exit_analysis"]   = exit_analysis_col
    df["decision_output"] = decision_output_col

    return df


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def _tab_overview(df: pd.DataFrame, etfs: list[str]) -> None:
    active = df[df["sleeve"] == "active"]
    etf_df = df[df["sleeve"] == "ETF/core"]

    def _sum(frame, col):
        if col in frame.columns:
            v = pd.to_numeric(frame[col], errors="coerce")
            return v.sum() if not v.empty else None
        return None

    total_equity  = _sum(df, "equity")
    active_equity = _sum(active, "equity")
    etf_equity    = _sum(etf_df, "equity")
    active_pnl    = _sum(active, "equity_change")
    etf_pnl       = _sum(etf_df, "equity_change")

    # Regime (from RegimeDetector if available)
    regime_label = "—"
    try:
        from strategy.regimes import RegimeDetector
        r = RegimeDetector().current_regime()
        regime_label = r.regime if r else "—"
    except Exception:
        pass

    # Status counts
    state_counts = df[df["sleeve"] == "active"]["state"].value_counts().to_dict() if "state" in df.columns else {}

    # ── Summary cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equity", f"${total_equity:,.0f}" if total_equity else "—")
    c2.metric("Active Sleeve", f"${active_equity:,.0f}" if active_equity else "—")
    c3.metric("ETF / Core", f"${etf_equity:,.0f}" if etf_equity else "—")
    c4.metric("Market Regime", regime_label.title())

    st.divider()

    r1, r2, r3, r4 = st.columns(4)
    if active_pnl is not None:
        r1.metric("Active P&L", f"${active_pnl:+,.2f}", delta_color="normal")
    if etf_pnl is not None:
        r2.metric("ETF P&L", f"${etf_pnl:+,.2f}", delta_color="normal")
    if total_equity and total_equity > 0 and active_equity is not None:
        r3.metric("Active weight", f"{active_equity/total_equity:.1%}")
    if "state" in df.columns and not active.empty:
        vm_mean = pd.to_numeric(active.get("value_metric", pd.Series()), errors="coerce").mean()
        if not pd.isna(vm_mean):
            r4.metric("Avg active score", f"{vm_mean:.3f}")

    st.divider()

    # ── Status counts ────────────────────────────────────────────────────────
    st.markdown("**Active position status**")
    all_states = ["BUY", "HOLD", "WATCH", "REVIEW", "TRIM", "HARVEST", "EXIT"]
    all_icons  = ["🟢",  "🔵",   "🟡",    "🟣",     "🔶",   "💰",      "🔴"]
    active_states = [(s, ic) for s, ic in zip(all_states, all_icons) if state_counts.get(s, 0) > 0 or s in ("HOLD", "EXIT")]
    sc = st.columns(len(active_states))
    for i, (state, icon) in enumerate(active_states):
        n = state_counts.get(state, 0)
        sc[i].metric(f"{icon} {state}", n)

    # ── Portfolio character ──────────────────────────────────────────────────
    # ── Suspicious exit summary ───────────────────────────────────────────────
    if "exit_analysis" in df.columns:
        from portfolio.exit_analysis import compute_premature_exit_rate
        exit_rows = df[(df["sleeve"] == "active") & (df["state"].isin(["EXIT", "WATCH"]))]
        eas = [ea for ea in exit_rows["exit_analysis"].tolist() if ea is not None]
        if eas:
            premature = [ea for ea in eas if ea.is_premature]
            rate      = compute_premature_exit_rate(eas)
            if premature:
                syms = ", ".join(ea.symbol for ea in premature)
                st.warning(
                    f"⚠️ **Potential premature exits detected: {syms}**  \n"
                    f"Premature exit rate: **{rate:.0%}** of current EXIT/WATCH signals.  \n"
                    f"These positions still show intact thesis signals. Review before acting."
                )

    # REVIEW banner
    if "state" in df.columns:
        reviews = df[(df["sleeve"] == "active") & (df["state"] == "REVIEW")]
        if not reviews.empty:
            syms = ", ".join(reviews["symbol"].tolist())
            st.info(
                f"🟣 **REVIEW signals: {syms}**  \n"
                f"Exit signal fired but evidence suggests it may be premature. "
                f"Human review recommended before acting."
            )

    if "sleeve" in df.columns and total_equity and total_equity > 0 and not active.empty:
        st.divider()
        active_wt  = (active_equity or 0) / total_equity
        n_pos      = len(df)
        top5_wt    = pd.to_numeric(df["equity"], errors="coerce").nlargest(5).sum() / total_equity
        characters = []
        if active_wt >= 0.50:
            characters.append("Active-heavy")
        elif active_wt <= 0.25:
            characters.append("ETF-heavy")
        if n_pos >= 20:
            characters.append("Diversified")
        elif n_pos <= 8:
            characters.append("Concentrated")
        if top5_wt >= 0.60:
            characters.append(f"Top-5 = {top5_wt:.0%} of portfolio")
        exits   = state_counts.get("EXIT",   0)
        watches = state_counts.get("WATCH",  0)
        reviews = state_counts.get("REVIEW", 0)
        if exits + watches + reviews >= 3:
            characters.append("Deteriorating (several WATCH/REVIEW/EXIT signals)")

        if "quality_score" in active.columns:
            avg_q = pd.to_numeric(active["quality_score"], errors="coerce").mean()
            if not pd.isna(avg_q):
                characters.append("Quality-heavy" if avg_q > 0.7 else ("Quality-light" if avg_q < 0.2 else "Balanced quality"))

        st.markdown("**Portfolio character:** " + " · ".join(characters) if characters else "—")


# ---------------------------------------------------------------------------
# Tab: Holdings Intelligence Table
# ---------------------------------------------------------------------------

def _tab_holdings_intel(df: pd.DataFrame) -> None:
    st.caption("Position intelligence — state, thesis health, factor drivers, and next action.")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sleeve_filter = st.multiselect(
            "Sleeve", ["active", "ETF/core"], default=["active", "ETF/core"], key="pi_sleeve"
        )
    with fc2:
        state_filter = st.multiselect(
            "Status", ["BUY", "HOLD", "WATCH", "REVIEW", "TRIM", "HARVEST", "EXIT"],
            default=["BUY", "HOLD", "WATCH", "REVIEW", "TRIM", "HARVEST", "EXIT"], key="pi_state"
        )
    with fc3:
        if "sector" in df.columns:
            sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
            sector_choice = st.selectbox("Sector", sectors, key="pi_sector")
        else:
            sector_choice = "All"
    with fc4:
        sort_by = st.selectbox(
            "Sort by",
            ["equity", "percent_change", "value_metric", "quality_score", "score_delta", "holding_days"],
            key="pi_sort",
        )

    view = df[
        (df["sleeve"].isin(sleeve_filter)) &
        (df["state"].isin(state_filter))
    ].copy()
    if sector_choice != "All" and "sector" in view.columns:
        view = view[view["sector"] == sector_choice]

    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=(sort_by not in ["equity"]), na_position="last")

    # Display columns
    display_cols = [c for c in [
        "symbol", "name", "sleeve", "state",
        "equity", "percent_change", "equity_change",
        "holding_days",
        "value_metric", "quality_score", "momentum_score",
        "score_at_buy", "score_delta",
        "top_positive", "top_negative",
        "next_action",
    ] if c in view.columns]

    def _fmt_state(v):
        icon = _STATE_ICON.get(str(v), "")
        return f"{icon} {v}"

    view_display = view[display_cols].copy()
    if "state" in view_display.columns:
        view_display["state"] = view_display["state"].map(_fmt_state)

    fmt: dict = {}
    for col in ["equity", "equity_change"]:
        if col in view_display.columns:
            fmt[col] = "${:,.2f}"
    for col in ["value_metric", "quality_score", "momentum_score", "score_at_buy", "score_delta"]:
        if col in view_display.columns:
            fmt[col] = "{:.3f}"
    if "percent_change" in view_display.columns:
        fmt["percent_change"] = "{:+.2f}%"

    styled = view_display.style.format(fmt, na_rep="—")

    def _color_pct(v):
        try:
            return "color: #2ecc71" if float(str(v).replace("%", "").replace("+", "")) > 0 else "color: #e74c3c"
        except Exception:
            return ""

    if "percent_change" in view_display.columns:
        styled = styled.map(_color_pct, subset=["percent_change"])

    st.dataframe(styled, use_container_width=True, height=480, hide_index=True)

    # Risk alerts
    exits   = df[(df["state"] == "EXIT") & (df["sleeve"] == "active")]
    watches = df[(df["state"] == "WATCH") & (df["sleeve"] == "active")]
    if not exits.empty:
        st.error(f"🔴 EXIT signals: **{', '.join(exits['symbol'].tolist())}** — review immediately.")
    if not watches.empty:
        st.warning(f"🟡 WATCH: **{', '.join(watches['symbol'].tolist())}** — thesis weakening.")


# ---------------------------------------------------------------------------
# Tab: Active Sleeve (expandable per-position detail)
# ---------------------------------------------------------------------------

def _position_score_chart(history: pd.DataFrame, key_prefix: str = "") -> None:
    if history.empty:
        st.caption("No snapshot history available for this symbol.")
        return
    import plotly.graph_objects as go
    score_cols = [c for c in ["value_metric", "quality_score", "momentum_score"] if c in history.columns]
    date_col   = "snapshot_date" if "snapshot_date" in history.columns else None
    if not score_cols or date_col is None:
        return

    fig = go.Figure()
    colors = {"value_metric": "#3498db", "quality_score": "#2ecc71", "momentum_score": "#f39c12"}
    labels = {"value_metric": "Composite", "quality_score": "Quality", "momentum_score": "Momentum"}
    for col in score_cols:
        fig.add_trace(go.Scatter(
            x=history[date_col],
            y=pd.to_numeric(history[col], errors="coerce"),
            mode="lines+markers",
            name=labels.get(col, col),
            line=dict(color=colors.get(col, "#aaa"), width=2),
        ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=10, t=15, b=20),
        legend=dict(orientation="h", y=1.15),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#cdd6f4"),
        xaxis=dict(gridcolor="#2d3436"),
        yaxis=dict(gridcolor="#2d3436", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"score_hist_{key_prefix}")


def _render_exit_analysis(ea, key_prefix: str = "") -> None:
    """Render the exit root-cause analysis block for one EXIT/WATCH position."""
    import plotly.graph_objects as go
    from portfolio.exit_analysis import DRIVER_LABELS, DRIVER_ORDER

    if ea is None:
        return

    _CONF_COLOR = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#3498db"}
    conf_color  = _CONF_COLOR.get(ea.confidence, "#aaa")

    st.markdown("**Exit Root-Cause Analysis**")

    header_cols = st.columns(3)
    header_cols[0].markdown(
        f"**Primary driver**  \n"
        f'<span style="font-size:1.05em;color:#e74c3c">{ea.primary_label}</span>',
        unsafe_allow_html=True,
    )
    header_cols[1].markdown(
        f"**Secondary driver**  \n"
        f'<span style="font-size:1.05em;color:#f39c12">{ea.secondary_label}</span>',
        unsafe_allow_html=True,
    )
    header_cols[2].markdown(
        f"**Confidence**  \n"
        f'<span style="font-size:1.05em;color:{conf_color}">{ea.confidence}</span>',
        unsafe_allow_html=True,
    )

    # Thesis-intact score
    tis = ea.thesis_intact_score
    tis_color = "#2ecc71" if tis >= 0.70 else ("#f39c12" if tis >= 0.45 else "#e74c3c")
    st.markdown(
        f"Thesis-intact score: "
        f'<span style="font-weight:700;color:{tis_color}">{tis:.2f}</span>'
        f" / 1.0  (higher = thesis still looks good despite exit signal)",
        unsafe_allow_html=True,
    )

    # Premature exit flag
    if ea.is_premature:
        st.warning(
            f"⚠️ **POTENTIAL PREMATURE EXIT** — {ea.premature_reason}  \n"
            f"Override recommendation: **{ea.override_recommendation or 'WATCH'}** "
            f"(this is a flag for review, not an automatic override)"
        )

    # Weight waterfall bar chart
    ordered_weights = {k: ea.reason_weights.get(k, 0.0) for k in DRIVER_ORDER}
    non_zero = {DRIVER_LABELS[k]: v for k, v in ordered_weights.items() if v > 0.005}
    if non_zero:
        labels = list(non_zero.keys())
        values = list(non_zero.values())
        bar_colors = []
        for k, v in zip(ordered_weights, ordered_weights.values()):
            if k in ("stop_loss", "trailing_stop"):
                bar_colors.append("#e74c3c")
            elif k in ("take_profit", "harvest"):
                bar_colors.append("#2ecc71")
            elif k in ("momentum_deterioration", "score_decay", "rank_deterioration"):
                bar_colors.append("#f39c12")
            else:
                bar_colors.append("#3498db")
        bar_colors = [bar_colors[i] for i, (k, v) in enumerate(ordered_weights.items()) if v > 0.005]

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(120, len(non_zero) * 28),
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(tickformat=".0%", range=[0, 1.05], gridcolor="#2d3436"),
            yaxis=dict(gridcolor="#2d3436"),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#cdd6f4", size=11),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"exit_weights_{key_prefix}")

        # Raw weight table in expander
        with st.expander("Exit reason weights (full breakdown)", key=f"exit_weights_exp_{key_prefix}"):
            weight_df = pd.DataFrame([
                {"Driver": DRIVER_LABELS.get(k, k), "Weight": f"{v:.1%}"}
                for k, v in ordered_weights.items()
            ])
            st.dataframe(weight_df, use_container_width=True, hide_index=True)


def _factor_decomp_chart(contribs: dict[str, float], key_prefix: str = "") -> None:
    if not contribs:
        return
    import plotly.graph_objects as go
    labels = list(contribs.keys())
    values = list(contribs.values())
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#555")
    fig.update_layout(
        height=160, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#cdd6f4"),
        xaxis=dict(gridcolor="#2d3436"),
        yaxis=dict(gridcolor="#2d3436"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"factor_decomp_{key_prefix}")


def _tab_active_sleeve(df: pd.DataFrame) -> None:
    from portfolio.position_rationale import factor_contributions

    active = df[df["sleeve"] == "active"].sort_values(
        "equity", ascending=False, na_position="last"
    )

    if active.empty:
        st.info("No active positions found.")
        return

    agg = _load_agg()

    for _, row in active.iterrows():
        sym    = str(row["symbol"])
        state  = str(row.get("state", "HOLD"))
        icon   = _STATE_ICON.get(state, "")
        equity = row.get("equity")
        pct    = row.get("percent_change")
        eq_str = f"${equity:,.2f}" if pd.notna(equity) else "—"
        pc_str = f"{pct:+.2f}%" if pd.notna(pct) else "—"

        ea  = row.get("exit_analysis")
        do  = row.get("decision_output")

        # Expander header suffix: surface REVIEW and premature exit badges
        badges = list(getattr(do, "badges", [])) if do is not None else []
        _badge_map = {
            "REVIEW NEEDED": "🟣 REVIEW", "RISKY EXIT": "⚠️ RISKY EXIT",
            "SAFE EXIT": "✓ EXIT", "TRIM": "🔶 TRIM", "HARVEST": "💰 HARVEST",
        }
        badge_str = "  " + "  ".join(_badge_map.get(b, b) for b in badges) if badges else ""
        with st.expander(
            f"{icon} **{sym}** — {eq_str}  ({pc_str})  |  {state}{badge_str}  |  {row.get('name', '')}",
            expanded=(state in ("EXIT", "WATCH", "REVIEW", "TRIM", "HARVEST")),
        ):
            col_a, col_b = st.columns([2, 1])

            with col_a:
                st.markdown(f"**State:** {icon} {state}")
                st.markdown(f"**Reason:** {row.get('state_reason', '—')}")
                st.markdown(f"**Rationale:** {row.get('rationale', '—')}")

                if row.get("risk_flags"):
                    for flag in row["risk_flags"]:
                        st.markdown(flag)

                st.markdown(f"**Next action:** {row.get('next_action', '—')}")

            with col_b:
                # Score snapshot
                vm = row.get("value_metric")
                qs = row.get("quality_score")
                ms = row.get("momentum_score")
                hd = row.get("holding_days")
                st.markdown(f"Composite: **{vm:.3f}**" if pd.notna(vm) else "Composite: —")
                st.markdown(f"Quality: **{qs:.3f}**" if pd.notna(qs) else "Quality: —")
                st.markdown(f"Momentum: **{ms:.3f}**" if pd.notna(ms) else "Momentum: —")
                st.markdown(f"Holding: **{hd}d**" if hd is not None else "Holding: —")
                sat = row.get("score_at_buy")
                sn  = row.get("value_metric")
                if pd.notna(sat) and pd.notna(sn):
                    delta = sn - sat
                    color = "#2ecc71" if delta >= 0 else "#e74c3c"
                    st.markdown(
                        f"Score Δ since buy: "
                        f'<span style="color:{color}">{delta:+.3f}</span>',
                        unsafe_allow_html=True,
                    )

            # Decision diagnostics (all non-HOLD/BUY states)
            if state in ("EXIT", "WATCH", "REVIEW", "TRIM", "HARVEST"):
                st.divider()
                if do is not None:
                    from ui.components.decision_diagnostics import render_decision_diagnostics
                    render_decision_diagnostics(do, key_prefix=sym)
                if ea is not None and state in ("EXIT", "WATCH", "REVIEW"):
                    _render_exit_analysis(ea, key_prefix=sym)
                st.divider()

            # Factor decomposition
            metrics = None
            if agg is not None and "symbol" in agg.columns:
                r = agg[agg["symbol"] == sym]
                if not r.empty:
                    metrics = r.iloc[0]

            contribs = factor_contributions(metrics)
            if contribs:
                st.caption("Factor decomposition (contribution to composite score)")
                _factor_decomp_chart(contribs, key_prefix=sym)

            # Score history chart
            st.caption("Score history (from snapshots)")
            with st.spinner("Loading history…"):
                history = _load_score_history(sym)
            _position_score_chart(history, key_prefix=sym)

            # Journal entries for this symbol
            journal = _load_journal()
            if not journal.empty and "symbol" in journal.columns:
                sym_journal = journal[journal["symbol"] == sym].tail(5)
                if not sym_journal.empty:
                    st.caption("Recent journal events")
                    st.dataframe(
                        sym_journal[["timestamp", "event_type", "status", "rationale"]],
                        use_container_width=True, hide_index=True,
                    )


# ---------------------------------------------------------------------------
# Tab: ETF Sleeve
# ---------------------------------------------------------------------------

def _tab_etf_sleeve(df: pd.DataFrame) -> None:
    from portfolio.position_rationale import etf_role

    etf_df = df[df["sleeve"] == "ETF/core"].sort_values("equity", ascending=False, na_position="last")

    if etf_df.empty:
        st.info("No ETF / core positions found.")
        return

    total_etf_equity = pd.to_numeric(etf_df["equity"], errors="coerce").sum()
    total_etf_pnl    = pd.to_numeric(etf_df.get("equity_change", pd.Series()), errors="coerce").sum()

    e1, e2 = st.columns(2)
    e1.metric("ETF Sleeve Value", f"${total_etf_equity:,.2f}")
    e2.metric("ETF Sleeve P&L", f"${total_etf_pnl:+,.2f}")

    st.divider()

    for _, row in etf_df.iterrows():
        sym    = str(row["symbol"])
        equity = row.get("equity")
        pct    = row.get("percent_change")
        eq_str = f"${equity:,.2f}" if pd.notna(equity) else "—"
        pc_str = f"{pct:+.2f}%" if pd.notna(pct) else "—"
        role   = etf_role(sym)

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 2])
            c1.markdown(f"### {sym}")
            c2.markdown(f"**{eq_str}** ({pc_str})")
            c3.markdown(f"*{role}*")


# ---------------------------------------------------------------------------
# Tab: Attribution (factor tilt + sleeve P/L)
# ---------------------------------------------------------------------------

def _tab_attribution(df: pd.DataFrame, agg: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go

    active = df[df["sleeve"] == "active"]
    etf_df = df[df["sleeve"] == "ETF/core"]

    # ── Sleeve attribution ────────────────────────────────────────────────────
    st.markdown("#### Sleeve P&L Attribution")
    active_pnl = pd.to_numeric(active.get("equity_change", pd.Series()), errors="coerce").sum()
    etf_pnl    = pd.to_numeric(etf_df.get("equity_change", pd.Series()), errors="coerce").sum()
    total_pnl  = active_pnl + etf_pnl

    a1, a2, a3 = st.columns(3)
    a1.metric("Active contribution", f"${active_pnl:+,.2f}")
    a2.metric("ETF contribution",    f"${etf_pnl:+,.2f}")
    a3.metric("Total P&L",           f"${total_pnl:+,.2f}")

    if not active.empty:
        winners = (pd.to_numeric(active["percent_change"], errors="coerce") > 0).sum()
        losers  = len(active) - winners
        st.caption(f"Active hit rate: {winners}/{len(active)} positions positive ({winners/max(len(active),1):.0%})")

    st.divider()

    # ── Factor tilt vs universe ───────────────────────────────────────────────
    st.markdown("#### Portfolio Factor Tilt vs Universe")
    score_cols = ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]

    def _weighted_avg(frame, col, weight_col="equity"):
        if col not in frame.columns or weight_col not in frame.columns:
            return None
        vals = pd.to_numeric(frame[col], errors="coerce")
        wts  = pd.to_numeric(frame[weight_col], errors="coerce")
        valid = vals.notna() & wts.notna() & (wts > 0)
        if valid.sum() == 0:
            return None
        return float((vals[valid] * wts[valid]).sum() / wts[valid].sum())

    def _univ_avg(col):
        if agg is None or col not in agg.columns:
            return None
        return float(pd.to_numeric(agg[col], errors="coerce").mean())

    labels = {
        "value_score": "Value", "quality_score": "Quality",
        "income_score": "Income", "momentum_score": "Momentum", "value_metric": "Composite",
    }

    tilt_rows: list[dict] = []
    for col in score_cols:
        port_wa  = _weighted_avg(active, col)
        univ_avg = _univ_avg(col)
        tilt_rows.append({
            "Factor":    labels.get(col, col),
            "Portfolio": round(port_wa, 3) if port_wa is not None else None,
            "Universe":  round(univ_avg, 3) if univ_avg is not None else None,
            "Δ vs Univ": round(port_wa - univ_avg, 3) if port_wa is not None and univ_avg is not None else None,
        })

    tilt_df = pd.DataFrame(tilt_rows)
    if not tilt_df.empty:
        st.dataframe(
            tilt_df.style.format({
                "Portfolio": "{:.3f}", "Universe": "{:.3f}", "Δ vs Univ": "{:+.3f}"
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # Bar chart of Δ
        delta_df = tilt_df.dropna(subset=["Δ vs Univ"])
        if not delta_df.empty:
            fig = go.Figure(go.Bar(
                x=delta_df["Factor"],
                y=delta_df["Δ vs Univ"],
                marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in delta_df["Δ vs Univ"]],
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#555")
            fig.update_layout(
                title="Active sleeve tilt vs universe average",
                height=280,
                margin=dict(l=20, r=10, t=40, b=20),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#cdd6f4"),
                xaxis=dict(gridcolor="#2d3436"),
                yaxis=dict(gridcolor="#2d3436", zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Best / worst contributors ─────────────────────────────────────────────
    if "equity_change" in active.columns:
        st.markdown("#### Top & Bottom Active Contributors")
        contrib_df = active[["symbol", "equity_change", "percent_change", "quality_score", "momentum_score"]].copy()
        contrib_df["equity_change"] = pd.to_numeric(contrib_df["equity_change"], errors="coerce")
        contrib_df = contrib_df.sort_values("equity_change", ascending=False)
        bc1, bc2 = st.columns(2)
        with bc1:
            st.caption("Best contributors")
            st.dataframe(
                contrib_df.head(5).style.format({"equity_change": "${:+,.2f}", "percent_change": "{:+.2f}%"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
        with bc2:
            st.caption("Worst contributors")
            st.dataframe(
                contrib_df.tail(5).style.format({"equity_change": "${:+,.2f}", "percent_change": "{:+.2f}%"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Tab: Journal
# ---------------------------------------------------------------------------

def _tab_journal(df: pd.DataFrame) -> None:
    from portfolio.position_journal import load_journal, log_portfolio_review
    import datetime

    journal = load_journal()

    if journal.empty:
        st.info("No journal entries yet. The journal logs position state transitions automatically.")
    else:
        st.caption(f"{len(journal)} total entries")
        st.dataframe(journal.sort_values("timestamp", ascending=False).head(50), use_container_width=True, hide_index=True)

    st.divider()
    if st.button("📝 Log today's portfolio review", key="pi_log_review"):
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        entries = []
        for _, row in df[df["sleeve"] == "active"].iterrows():
            entries.append({
                "timestamp":       now,
                "symbol":          row.get("symbol", ""),
                "event_type":      "HOLD_REVIEW",
                "sleeve":          row.get("sleeve", "active"),
                "status":          row.get("state", ""),
                "price":           row.get("current_price"),
                "composite_score": row.get("value_metric"),
                "rank_pct":        row.get("rank_pct_now"),
                "rationale":       row.get("state_reason", ""),
            })
        log_portfolio_review(entries)
        st.success(f"Logged review for {len(entries)} active positions.")
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Dividend section (preserved from original)
# ---------------------------------------------------------------------------

def _dividend_section() -> None:
    div_df = load_latest_csv("dividend_history")
    if div_df is None:
        return

    st.divider()
    st.subheader("Dividend income")
    div_df["amount"] = pd.to_numeric(div_df.get("amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    paid = div_df[div_df.get("state", pd.Series()).eq("paid")] if "state" in div_df.columns else div_df
    total_paid = paid["amount"].sum()

    d1, d2, d3 = st.columns(3)
    d1.metric("Total dividends received", f"${total_paid:,.2f}")
    d2.metric("Dividend payments", len(paid))
    if "symbol" in div_df.columns:
        top_sym = paid.groupby("symbol")["amount"].sum().sort_values(ascending=False).head(1)
        if not top_sym.empty:
            d3.metric("Top payer", f"{top_sym.index[0]}  ${top_sym.iloc[0]:.2f}")
    if "symbol" in paid.columns and not paid.empty:
        by_symbol = paid.groupby("symbol")["amount"].sum().sort_values(ascending=False).head(15)
        st.bar_chart(by_symbol)


# ---------------------------------------------------------------------------
# Live broker section (preserved from original)
# ---------------------------------------------------------------------------

def _live_section(etfs: list[str]) -> None:
    st.divider()
    st.subheader("Live broker data")
    live = st.session_state.get("live_enabled", False)
    if not live:
        st.info("🔒 Live execution is OFF. Enable it in the sidebar to fetch live holdings from Robinhood.")
        return

    if st.button("Fetch live holdings from Robinhood"):
        with st.spinner("Connecting to Robinhood…"):
            try:
                from main import login, get_current_positions, get_available_cash, get_portfolio_value, save_holdings_csv
                login()
                holdings = get_current_positions()
                save_holdings_csv(holdings)
                cash     = get_available_cash()
                port_val = get_portfolio_value()
                st.session_state["live_holdings"] = holdings
                st.session_state["live_cash"]     = cash
                st.session_state["live_port_val"] = port_val
                st.success("✅ Live data fetched and saved to holdings CSV.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to fetch live data: {exc}")


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("💼 Portfolio Intelligence")
    st.caption(
        "Why do I own this, what is it doing, and should I still own it?  "
        f"Data: holdings {data_date('holdings')} | universe {data_date('agg_data')}"
    )

    cfg  = load_config_raw()
    etfs = cfg.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"])

    holdings = _load_holdings()
    if holdings is None:
        st.warning(
            no_data_msg("holdings")
            + "  \nHoldings are saved automatically when the bot runs (`daily-investor run`)."
        )
        _live_section(etfs)
        return

    agg         = _load_agg()
    buy_ctx     = _load_buy_context()
    peak_prices = _load_peak_prices()

    with st.spinner("Building portfolio intelligence…"):
        df = _enrich_holdings(holdings, agg, buy_ctx, peak_prices, etfs)

    tabs = st.tabs([
        "📊 Overview",
        "🧠 Holdings Intel",
        "📈 Active Sleeve",
        "🏦 ETF Sleeve",
        "📐 Attribution",
        "📔 Journal",
    ])

    with tabs[0]:
        _tab_overview(df, etfs)
        _dividend_section()

    with tabs[1]:
        _tab_holdings_intel(df)

    with tabs[2]:
        _tab_active_sleeve(df)

    with tabs[3]:
        _tab_etf_sleeve(df)

    with tabs[4]:
        _tab_attribution(df, agg)

    with tabs[5]:
        _tab_journal(df)

    _live_section(etfs)
