"""
ui/components/allocation_diagnostics.py — Sleeve allocation diagnostics.

Displays:
  - ETF sleeve vs active sleeve vs cash: target vs actual vs drift
  - Migration state: velocity, convergence timeline, capital flow direction
  - Capital source breakdown (from sleeve_events.parquet)
  - Drift warning banner when ETF sleeve deviates >5% from target

Data source: holdings CSV + config.yaml + data/sleeve_events.parquet
SAFE: read-only display only. Never writes scores or config.
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR, load_config_raw, load_latest_csv

_DRIFT_WARN_PCT = 0.05  # warn if ETF drift exceeds ±5%


@st.cache_data(ttl=60)
def _load_holdings_for_allocation() -> tuple[pd.DataFrame | None, list[str], float]:
    """
    Load holdings CSV and config. Returns (holdings_df, etf_list, index_pct_target).
    """
    try:
        cfg  = load_config_raw()
        etfs = cfg.get("etfs", [])
        idx  = float(cfg.get("index_pct", 0.70))
    except Exception:
        etfs = []
        idx  = 0.70

    df = load_latest_csv("holdings")
    return df, etfs, idx


@st.cache_data(ttl=60)
def _load_sleeve_events() -> pd.DataFrame:
    try:
        from portfolio.sleeve_tracker import load_sleeve_events
        return load_sleeve_events()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_allocation_history(etfs: tuple[str, ...]) -> list[dict]:
    """
    Load all holdings snapshots and return a list of allocation state dicts
    sorted chronologically. Used for migration velocity computation.
    """
    pattern = os.path.join(DATA_DIR, "holdings_*.csv")
    records = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        date_str = fname.replace("holdings_", "").replace(".csv", "")
        try:
            date = pd.to_datetime(date_str, format="%Y_%m_%d")
        except Exception:
            continue
        try:
            df = pd.read_csv(path)
            df["equity"]   = pd.to_numeric(df["equity"],   errors="coerce").fillna(0.0)
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
            df = df[df["quantity"] > 0]
            etf_eq  = float(df[df["symbol"].isin(etfs)]["equity"].sum())
            act_eq  = float(df[~df["symbol"].isin(etfs)]["equity"].sum())
            total   = etf_eq + act_eq
            if total <= 0:
                continue
            records.append({
                "date":       date,
                "etf_equity": etf_eq,
                "act_equity": act_eq,
                "total":      total,
                "etf_pct":    etf_eq / total,
            })
        except Exception:
            continue
    return records


def _compute_migration_state(
    history: list[dict],
    current_etf_pct: float,
    index_pct_target: float,
    weekly_contribution: float = 400.0,
    harvest_to_etf_pct: float = 0.85,
) -> dict:
    """
    Derive migration velocity and convergence estimate from historical snapshots.
    Returns a dict with keys: velocity_pct_per_month, direction, months_to_target,
    rebalance_mode, equilibrium_pct.
    """
    if len(history) < 2:
        return {}

    # Velocity: linear regression of etf_pct over time
    dates   = [(r["date"] - history[0]["date"]).days for r in history]
    pcts    = [r["etf_pct"] for r in history]
    if len(set(dates)) < 2:
        return {}

    try:
        import numpy as np
        x = np.array(dates, dtype=float)
        y = np.array(pcts, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        slope_per_day = float(coeffs[0])           # fraction/day (can be negative)
        velocity_pct_per_month = slope_per_day * 30.0 * 100.0  # percent/month
    except Exception:
        return {}

    # Direction label
    if abs(velocity_pct_per_month) < 0.05:
        direction = "stable"
    elif velocity_pct_per_month > 0:
        direction = "drifting_higher"   # ETF % growing (moving away from 70% target)
    else:
        direction = "converging"        # ETF % falling (moving toward 70% target)

    # Convergence timeline toward target
    gap = current_etf_pct - index_pct_target
    if slope_per_day < -1e-6:
        # Converging — estimate months to close gap
        months_to_target = round(abs(gap / velocity_pct_per_month * 100), 1)
    else:
        months_to_target = None  # diverging or flat

    # Theoretical equilibrium under current routing policy:
    # Monthly active inflow = weekly_contribution * 4.33 * (1 - index_pct_target)
    # Monthly ETF inflow    = weekly_contribution * 4.33 * index_pct_target
    #                       + exit_proceeds * harvest_to_etf_pct
    # At equilibrium, active fraction stabilises when active inflow = active outflow.
    # With current exit rate draining active, equilibrium approaches ~100% ETF.
    # We estimate it from the observed trend.
    if slope_per_day > 1e-6:
        # Diverging — equilibrium is where active hits near-zero
        total_now = history[-1]["total"]
        act_now   = history[-1]["act_equity"]
        if act_now > 0 and slope_per_day > 0:
            days_to_zero = act_now / max(history[-1]["total"] * slope_per_day, 1e-9)
            equilibrium_pct = min(0.999, current_etf_pct + slope_per_day * days_to_zero)
        else:
            equilibrium_pct = current_etf_pct
    else:
        equilibrium_pct = index_pct_target  # converging toward target

    return {
        "velocity_pct_per_month": velocity_pct_per_month,
        "direction":              direction,
        "months_to_target":       months_to_target,
        "equilibrium_pct":        equilibrium_pct,
        "slope_per_day":          slope_per_day,
        "rebalance_mode":         "contribution_driven",
        "history":                history,
    }


def _render_migration_state(
    state: dict,
    migration: dict,
    etfs: list[str],
    index_pct_target: float,
) -> None:
    """Render the migration state panel."""
    st.subheader("Migration State")
    st.caption(
        "Shows whether the portfolio is converging toward its target allocation, "
        "and at what velocity. "
        "This portfolio evolved from legacy discretionary holdings — drift from target "
        "is expected and normal during the transition period."
    )

    if not migration:
        st.info("Need at least 2 historical holdings snapshots to compute migration velocity.")
        return

    vel   = migration.get("velocity_pct_per_month", 0.0)
    dirn  = migration.get("direction", "stable")
    mtt   = migration.get("months_to_target")
    eq    = migration.get("equilibrium_pct", index_pct_target)
    hist  = migration.get("history", [])

    # Direction indicator
    if dirn == "converging":
        dir_label = "✅ Converging toward target"
        dir_color = "success"
    elif dirn == "drifting_higher":
        dir_label = "⚠️ Drifting away from target (exits routing to ETF)"
        dir_color = "warning"
    else:
        dir_label = "— Stable"
        dir_color = "info"

    getattr(st, dir_color)(dir_label)

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Observed Start",
        f"{hist[0]['etf_pct']:.1%}" if hist else "—",
        f"{hist[0]['date'].strftime('%b %d') if hist else ''}",
        delta_color="off",
    )
    c2.metric(
        "Current ETF %",
        f"{state.get('etf_pct', 0):.1%}",
        f"{state.get('etf_drift_pct', 0):+.1%} vs target",
        delta_color="inverse",
    )
    c3.metric(
        "Velocity",
        f"{vel:+.2f}% / mo",
        "toward target" if vel < 0 else "away from target",
        delta_color="normal" if vel < 0 else "inverse",
    )
    c4.metric(
        "Est. Equilibrium",
        f"{eq:.1%}",
        f"target: {index_pct_target:.0%}",
        delta_color="off",
    )

    st.markdown("")

    # Convergence timeline or divergence warning
    if dirn == "converging" and mtt is not None:
        st.info(
            f"At current velocity ({vel:+.2f}%/mo), the ETF sleeve reaches the "
            f"{index_pct_target:.0%} target in approximately **{mtt:.0f} months**."
        )
    elif dirn == "drifting_higher":
        st.warning(
            f"The ETF sleeve is growing ({vel:+.2f}%/mo). "
            f"This is caused by legacy active positions being exited with 85% of proceeds "
            f"routed back to ETFs. The portfolio is completing a migration **into** an ETF-first "
            f"strategy, not toward a 70/30 split. "
            f"Current trajectory equilibrium: **{eq:.1%} ETF**."
        )

    # Rebalance policy explanation
    with st.expander("Rebalance policy"):
        st.markdown(
            f"""
**Mode: contribution-driven (no force rebalance)**

New capital is split at the `index_pct` target ({index_pct_target:.0%} ETF / {1-index_pct_target:.0%} active)
each run. There is no mechanism to sell ETFs to fund the active sleeve.

**Effect on a legacy ETF-heavy portfolio:**
- Weekly contribution of $400 → $280 to ETF, $120 to active
- The iteration-1 ETF buy uses `cash × index_pct` without checking whether the ETF sleeve
  is already overweight. The end-of-run sweep correctly skips (deficit is negative),
  but the contribution-split has already routed 70% to ETFs regardless.
- Exit proceeds from legacy active positions route **{migration.get('harvest_to_etf_pct', 0.85):.0%}**
  back to ETF via harvest/trim routing.

**Policy options (not changed automatically):**
| Mode | Behaviour |
|------|-----------|
| **Current (contribution-driven)** | Split inflows at target ratio — maintains existing imbalance |
| **Deficit-aware contributions** | Route all new cash to active until ETF sleeve is back at target |
| **Hard rebalance** | Sell ETF lots, redeploy to active — triggers taxes, not recommended |
"""
        )

    # Historical trajectory chart
    if len(hist) >= 2:
        st.markdown("**ETF % Trajectory**")
        df_hist = pd.DataFrame({
            "Date":  [r["date"] for r in hist],
            "ETF %": [r["etf_pct"] * 100 for r in hist],
        })
        df_hist = df_hist.set_index("Date")
        # Add target line
        df_hist["Target %"] = index_pct_target * 100
        st.line_chart(df_hist, use_container_width=True)

    # Capital flow diagram
    st.markdown("**Active sleeve funding sources (per run)**")
    st.markdown(
        f"""
| Source | Active sleeve | ETF sleeve |
|--------|--------------|------------|
| Weekly contribution ($400) | ${400*(1-index_pct_target):.0f} ({(1-index_pct_target):.0%}) | ${400*index_pct_target:.0f} ({index_pct_target:.0%}) |
| Trim proceeds | 15% | 85% |
| Harvest proceeds | 15% | 85% |
| Exit (full sell) | 15% stays as cash | 85% routed to ETF |
| ETF end-of-run sweep | 0 (deficit-aware) | Fills gap only |
"""
    )


def _compute_allocation_state(
    holdings: pd.DataFrame,
    etfs: list[str],
    index_pct_target: float,
) -> dict:
    """
    Compute ETF vs active vs cash split from the holdings CSV.
    Holdings CSV has columns: symbol, equity, ...
    There is no 'cash' row — cash is implicitly what's not invested.
    """
    if holdings is None or holdings.empty:
        return {}

    equity_col = "equity"
    if equity_col not in holdings.columns:
        return {}

    holdings = holdings.copy()
    holdings[equity_col] = pd.to_numeric(holdings[equity_col], errors="coerce").fillna(0.0)

    etf_mask    = holdings["symbol"].isin(etfs)
    etf_equity  = float(holdings.loc[etf_mask, equity_col].sum())
    active_equity = float(holdings.loc[~etf_mask, equity_col].sum())

    # quantity > 0 filter to exclude zero-quantity rows
    if "quantity" in holdings.columns:
        holdings["quantity"] = pd.to_numeric(holdings["quantity"], errors="coerce").fillna(0.0)
        etf_equity    = float(holdings.loc[etf_mask    & (holdings["quantity"] > 0), equity_col].sum())
        active_equity = float(holdings.loc[(~etf_mask) & (holdings["quantity"] > 0), equity_col].sum())

    total_invested = etf_equity + active_equity
    denom          = max(total_invested, 1e-9)

    return {
        "total_invested":  total_invested,
        "etf_equity":      etf_equity,
        "active_equity":   active_equity,
        "etf_pct":         etf_equity    / denom,
        "active_pct":      active_equity / denom,
        "target_etf_pct":  index_pct_target,
        "etf_drift_pct":   (etf_equity / denom) - index_pct_target,
        "n_etf_positions":    int(etf_mask.sum()),
        "n_active_positions": int((~etf_mask).sum()),
    }


def _render_drift_banner(state: dict) -> None:
    drift = state.get("etf_drift_pct")
    if drift is None:
        return
    if abs(drift) > _DRIFT_WARN_PCT:
        direction = "overweight" if drift > 0 else "underweight"
        st.warning(
            f"⚠️ ETF sleeve is **{direction}** by {abs(drift):.1%} "
            f"(target {state['target_etf_pct']:.0%}, actual {state['etf_pct']:.0%}). "
            f"See Migration State below for convergence analysis."
        )
    else:
        st.success(
            f"✅ ETF sleeve within tolerance — actual {state['etf_pct']:.0%} "
            f"vs target {state['target_etf_pct']:.0%} (drift {drift:+.1%})"
        )


def _render_allocation_metrics(state: dict) -> None:
    st.subheader("Current Sleeve Allocation")
    if not state:
        st.info("No holdings data found. Run the bot once to populate holdings.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "ETF Sleeve",
        f"${state.get('etf_equity', 0):,.0f}",
        f"{state['n_etf_positions']} position(s)",
    )
    c2.metric(
        "Active Sleeve",
        f"${state.get('active_equity', 0):,.0f}",
        f"{state['n_active_positions']} position(s)",
    )
    c3.metric(
        "Total Invested",
        f"${state.get('total_invested', 0):,.0f}",
    )

    st.markdown("---")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Target ETF %",  f"{state['target_etf_pct']:.1%}")
    cc2.metric("Actual ETF %",  f"{state['etf_pct']:.1%}")
    cc3.metric("Drift",         f"{state['etf_drift_pct']:+.1%}",
               delta_color="inverse")


def _render_position_breakdown(holdings: pd.DataFrame, etfs: list[str]) -> None:
    st.subheader("Positions by Sleeve")
    if holdings is None or holdings.empty:
        return

    equity_col = "equity"
    if equity_col not in holdings.columns:
        return

    df = holdings.copy()
    df[equity_col] = pd.to_numeric(df[equity_col], errors="coerce").fillna(0.0)
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
        df = df[df["quantity"] > 0]

    df["sleeve"] = df["symbol"].apply(lambda s: "ETF/core" if s in etfs else "active")

    etf_df    = df[df["sleeve"] == "ETF/core"][["symbol", equity_col]].sort_values(equity_col, ascending=False)
    active_df = df[df["sleeve"] == "active"][["symbol", equity_col]].sort_values(equity_col, ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**ETF Sleeve** ({len(etf_df)} positions)")
        if not etf_df.empty:
            etf_df[equity_col] = etf_df[equity_col].map("${:,.0f}".format)
            st.dataframe(etf_df.rename(columns={equity_col: "Equity"}),
                         use_container_width=True, hide_index=True)
    with col2:
        st.markdown(f"**Active Sleeve** ({len(active_df)} positions)")
        if not active_df.empty:
            top = active_df.head(20)
            top[equity_col] = top[equity_col].map("${:,.0f}".format)
            st.dataframe(top.rename(columns={equity_col: "Equity"}),
                         use_container_width=True, hide_index=True)
            if len(active_df) > 20:
                st.caption(f"… and {len(active_df) - 20} more positions")


def _render_capital_sources(events: pd.DataFrame) -> None:
    st.subheader("Capital Source Breakdown (Last 90 Days)")
    if events.empty:
        st.info(
            "No capital source events recorded yet. "
            "Events are logged automatically as the bot executes trades "
            "(trim, harvest, contribution, sweep)."
        )
        return

    try:
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        recent = events[events["event_date"] >= cutoff].copy()
    except Exception:
        recent = events.copy()

    if recent.empty:
        st.info("No sleeve events in the last 90 days.")
        return

    summary = (
        recent.groupby("event_type")["amount"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Total ($)", "count": "Events"})
        .reset_index()
        .rename(columns={"event_type": "Event Type"})
        .sort_values("Total ($)", ascending=False)
    )
    summary["Total ($)"] = summary["Total ($)"].map("${:,.0f}".format)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("View all events"):
        display_cols = ["event_date", "event_type", "source_symbol", "destination",
                        "amount", "etf_pct_routed", "notes"]
        show = recent[[c for c in display_cols if c in recent.columns]].sort_values(
            "event_date", ascending=False
        )
        st.dataframe(show, use_container_width=True, hide_index=True)


@st.cache_data(ttl=300, show_spinner=False)
def _load_concentration_report() -> tuple[object | None, str]:
    """Run concentration check and return (report, error_msg)."""
    try:
        from portfolio.exposure.cluster_concentration import run_concentration_check
        report = run_concentration_check()
        return report, ""
    except Exception as exc:
        return None, str(exc)


def _render_concentration_diagnostics() -> None:
    import plotly.graph_objects as go

    st.subheader("Factor-Cluster Concentration")
    st.caption(
        "Checks whether the active sleeve is over-concentrated in any PCA/KMeans "
        "factor-space cluster or sector.  Thresholds from `concentration_limits` in config.yaml."
    )

    cfg = load_config_raw()
    params = cfg.get("concentration_limits", {})
    if not params.get("enabled", True):
        st.info("Concentration diagnostics disabled (`concentration_limits.enabled: false`).")
        return

    max_cluster = float(params.get("max_cluster_weight", 0.35))
    max_sector  = float(params.get("max_sector_weight",  0.40))
    n_clusters  = int(params.get("n_clusters", 6))
    method      = str(params.get("cluster_method", "pca"))

    with st.spinner(f"Running {method.upper()} + KMeans({n_clusters}) concentration check…"):
        report, err = _load_concentration_report()

    if err:
        st.warning(f"Concentration check unavailable: {err}")
        return
    if report is None:
        st.info("Concentration diagnostics are disabled in config.")
        return

    # ── Violation banner ──────────────────────────────────────────────────────
    if report.has_violations:
        lines = report.summary_lines()
        st.error(
            "⚠️ **Active sleeve concentration violations detected**\n\n"
            + "\n\n".join(lines)
        )
    else:
        st.success(
            f"✅ No concentration violations  "
            f"(cluster limit {max_cluster:.0%}, sector limit {max_sector:.0%})"
        )

    st.caption(
        f"Active sleeve: **{report.n_active_positions}** positions, "
        f"${report.total_active_equity:,.0f} equity  |  "
        f"method: {report.method.upper()}  |  clusters: {report.n_clusters}"
    )

    # ── Charts side by side ───────────────────────────────────────────────────
    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown("**Cluster concentration** (active sleeve)")
        if report.cluster_weights:
            labels  = [f"Cluster {k}" for k in report.cluster_weights]
            weights = list(report.cluster_weights.values())
            colors  = [
                "#e74c3c" if w > max_cluster else "#3498db"
                for w in weights
            ]
            fig = go.Figure(go.Bar(
                x=labels, y=[w * 100 for w in weights],
                marker_color=colors,
                text=[f"{w:.1%}" for w in weights],
                textposition="outside",
            ))
            fig.add_hline(
                y=max_cluster * 100, line_dash="dash",
                line_color="#f39c12",
                annotation_text=f"limit {max_cluster:.0%}",
                annotation_position="top right",
            )
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=40),
                yaxis=dict(title="%", gridcolor="#2d3436"),
                xaxis=dict(gridcolor="#2d3436"),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#cdd6f4", size=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="conc_cluster_bar")
        else:
            st.caption("No cluster data available.")

    with cc2:
        st.markdown("**Sector concentration** (active sleeve)")
        if report.sector_weights:
            # Top 10 sectors only to keep chart readable
            top_sec = dict(list(report.sector_weights.items())[:10])
            s_labels  = list(top_sec.keys())
            s_weights = list(top_sec.values())
            s_colors  = [
                "#e74c3c" if w > max_sector else "#2ecc71"
                for w in s_weights
            ]
            fig2 = go.Figure(go.Bar(
                x=[w * 100 for w in s_weights], y=s_labels,
                orientation="h",
                marker_color=s_colors,
                text=[f"{w:.1%}" for w in s_weights],
                textposition="outside",
            ))
            fig2.add_vline(
                x=max_sector * 100, line_dash="dash",
                line_color="#f39c12",
                annotation_text=f"limit {max_sector:.0%}",
                annotation_position="top right",
            )
            fig2.update_layout(
                height=300, margin=dict(l=10, r=60, t=20, b=20),
                xaxis=dict(title="%", gridcolor="#2d3436", showticklabels=False),
                yaxis=dict(gridcolor="#2d3436"),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#cdd6f4", size=10),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, key="conc_sector_bar")
        else:
            st.caption("No sector data available.")

    # ── Drill-down: which symbols are in each violated cluster ────────────────
    cluster_violations = [v for v in report.violations if v.kind == "cluster"]
    if cluster_violations:
        st.divider()
        st.markdown("**Cluster violation detail**")
        for v in cluster_violations:
            with st.expander(
                f"Cluster {v.label} — {v.weight:.1%} of active sleeve "
                f"(${v.equity:,.0f}, {len(v.symbols)} positions)",
                expanded=True,
            ):
                st.markdown(
                    f"**{v.weight:.1%}** of active equity in one cluster "
                    f"(limit **{v.threshold:.0%}**)  \n"
                    f"Positions: `{', '.join(v.symbols)}`"
                )

    if report.unmatched_symbols and not report.unmatched_symbols[0].startswith("ERROR"):
        st.caption(
            f"ℹ️ {len(report.unmatched_symbols)} owned symbol(s) not found in universe map "
            f"(delisted / recently added): {', '.join(report.unmatched_symbols[:10])}"
        )


def render() -> None:
    st.subheader("Sleeve Allocation Diagnostics")
    st.caption(
        "Tracks ETF vs active sleeve split from the most recent holdings snapshot. "
        "Compares actual allocation against the target index_pct from config."
    )

    holdings, etfs, index_pct = _load_holdings_for_allocation()
    state = _compute_allocation_state(holdings, etfs, index_pct)

    if state:
        _render_drift_banner(state)
        st.markdown("")
    _render_allocation_metrics(state)

    st.markdown("---")

    try:
        cfg = load_config_raw()
        weekly_contrib = float((cfg.get("harvest") or {}).get("weekly_contribution", 400.0))
        harvest_to_etf = float((cfg.get("harvest") or {}).get("harvest_to_etfs_pct", 0.85))
    except Exception:
        weekly_contrib = 400.0
        harvest_to_etf = 0.85

    history = _load_allocation_history(tuple(etfs))
    migration = _compute_migration_state(
        history,
        current_etf_pct=state.get("etf_pct", 0.0),
        index_pct_target=index_pct,
        weekly_contribution=weekly_contrib,
        harvest_to_etf_pct=harvest_to_etf,
    )
    if migration:
        migration["harvest_to_etf_pct"] = harvest_to_etf
    _render_migration_state(state, migration, etfs, index_pct)

    st.markdown("---")
    _render_position_breakdown(holdings, etfs)

    st.markdown("---")
    events = _load_sleeve_events()
    _render_capital_sources(events)

    st.markdown("---")
    _render_concentration_diagnostics()
