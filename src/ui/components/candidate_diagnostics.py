"""
ui/components/candidate_diagnostics.py — Candidate pool diagnostics.

Shows how candidate selection behaves under the current config, with
sensitivity analysis and A/B/C mode comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _load_agg() -> "pd.DataFrame | None":
    try:
        from data.cache import read_data_as_pd
        df = read_data_as_pd("agg_data")
        if df is None or df.empty:
            return None
        for col in ["value_score", "quality_score", "income_score", "momentum_score", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except Exception:
        return None


def _compute_pool(
    df: pd.DataFrame,
    sw: dict,
    cs: dict,
) -> pd.DataFrame:
    """Compute composite score and candidate mask for every stock."""
    raw_sw = np.array([sw["value"], sw["quality"], sw["income"], sw["momentum"]])
    norm = raw_sw / max(raw_sw.sum(), 1e-9)

    df = df.copy()
    df["_composite"] = (
        norm[0] * df.get("value_score",    pd.Series(0.0, index=df.index)).fillna(0)
        + norm[1] * df.get("quality_score", pd.Series(0.0, index=df.index)).fillna(0)
        + norm[2] * df.get("income_score",  pd.Series(0.0, index=df.index)).fillna(0)
        + norm[3] * df.get("momentum_score",pd.Series(0.0, index=df.index)).fillna(0)
    )

    mode = cs["mode"]
    top_pct = cs["top_percentile"]

    if mode == "percentile":
        cutoff = float(df["_composite"].quantile(1.0 - top_pct))
    else:
        cutoff = float(cs.get("absolute_score_floor", 0.45))

    mask = df["_composite"] >= cutoff

    if cs.get("use_absolute_score_floor", True):
        mask = mask & (df["_composite"] >= cs["absolute_score_floor"])

    mask = mask & (df.get("quality_score", 0.0) >= cs["min_quality_score"])
    mask = mask & (df.get("momentum_score", 0.0) >= cs["min_momentum_score"])

    # Income trap: income > 0 AND momentum < min_conditional_momentum_score → exclude
    min_cond_mom = cs.get("min_conditional_momentum_score", 0.00)
    has_income = df.get("income_score", 0.0) > 0
    cond_weak  = df.get("momentum_score", 0.0) < min_cond_mom
    income_trap = has_income & cond_weak
    if not cs.get("allow_income_defensive_exception", False):
        mask = mask & ~income_trap

    max_cands = cs["max_candidates"]
    if mask.sum() > max_cands:
        top_idx = df.loc[mask, "_composite"].nlargest(max_cands).index
        final = pd.Series(False, index=df.index)
        final.loc[top_idx] = True
        mask = final

    df["_selected"] = mask
    df["_cutoff"] = cutoff
    return df


def _sensitivity_data(df: pd.DataFrame, sw: dict) -> pd.DataFrame:
    """How many candidates are selected at each top-percentile cutoff."""
    raw_sw = np.array([sw["value"], sw["quality"], sw["income"], sw["momentum"]])
    norm = raw_sw / max(raw_sw.sum(), 1e-9)
    composite = (
        norm[0] * df.get("value_score",    pd.Series(0.0, index=df.index)).fillna(0)
        + norm[1] * df.get("quality_score", pd.Series(0.0, index=df.index)).fillna(0)
        + norm[2] * df.get("income_score",  pd.Series(0.0, index=df.index)).fillna(0)
        + norm[3] * df.get("momentum_score",pd.Series(0.0, index=df.index)).fillna(0)
    )
    rows = []
    for pct in np.arange(0.05, 0.51, 0.05):
        cutoff = float(composite.quantile(1.0 - pct))
        n = int((composite >= cutoff).sum())
        rows.append({"top_percentile": round(float(pct), 2), "n_candidates": n, "score_cutoff": round(cutoff, 3)})
    return pd.DataFrame(rows)


def render() -> None:
    st.subheader("Candidate Pool Diagnostics")
    st.caption(
        "Shows how the candidate_selection config transforms the score distribution "
        "into a buy-eligible pool — independent of which specific stocks pass."
    )

    df = _load_agg()
    if df is None:
        st.warning("No agg_data available. Run the bot first to generate fundamental data.")
        return

    from util import CANDIDATE_SELECTION_PARAMS, SCORE_WEIGHTS
    cs = CANDIDATE_SELECTION_PARAMS.copy()
    sw = SCORE_WEIGHTS.copy()

    # ── 1. Pool composition ───────────────────────────────────────────────
    result = _compute_pool(df, sw, cs)
    selected = result[result["_selected"]]
    universe = result

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", len(selected))
    c2.metric("Universe", len(universe))
    c3.metric("Score cutoff", f"{result['_cutoff'].iloc[0]:.3f}")
    c4.metric("Selection %", f"{100 * len(selected) / max(len(universe), 1):.1f}%")

    if len(selected):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg quality",   f"{selected['quality_score'].mean():.3f}")
        c2.metric("Avg momentum",  f"{selected['momentum_score'].mean():.3f}")
        c3.metric("Avg income",    f"{selected['income_score'].mean():.3f}")
        c4.metric("Avg composite", f"{selected['_composite'].mean():.3f}")

    # ── 2. Factor distributions ───────────────────────────────────────────
    with st.expander("Factor distributions: selected vs universe", expanded=True):
        try:
            import altair as alt

            factor_cols = ["quality_score", "momentum_score", "income_score", "_composite"]
            friendly = {
                "quality_score": "Quality", "momentum_score": "Momentum",
                "income_score": "Income", "_composite": "Composite",
            }
            sel_f = st.selectbox("Factor", factor_cols, format_func=lambda x: friendly.get(x, x),
                                 key="cdiag_factor")

            sel_vals = selected[sel_f].dropna()
            all_vals = universe[sel_f].dropna()

            comp_df = pd.concat([
                pd.DataFrame({"score": all_vals, "group": "Universe"}),
                pd.DataFrame({"score": sel_vals, "group": "Selected"}),
            ])
            chart = (
                alt.Chart(comp_df)
                .mark_bar(opacity=0.6)
                .encode(
                    alt.X("score:Q", bin=alt.Bin(step=0.05), title=friendly.get(sel_f, sel_f)),
                    alt.Y("count()", title="Count"),
                    alt.Color("group:N"),
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            st.caption("Install altair for distribution charts.")

    # ── 3. Sector distribution ────────────────────────────────────────────
    if "sector" in selected.columns:
        with st.expander("Sector distribution of selected candidates"):
            sec = selected["sector"].fillna("Unknown").value_counts().reset_index()
            sec.columns = ["Sector", "Count"]
            st.dataframe(sec, use_container_width=True, hide_index=True)

    # ── 4. High-income / low-momentum exposure ────────────────────────────
    with st.expander("High-income / low-momentum analysis"):
        min_cond = cs.get("min_conditional_momentum_score", 0.0)
        at_risk = result[
            (result["income_score"] > 0)
            & (result["momentum_score"] < min_cond)
            & result["_selected"]
        ]
        excluded = result[
            (result["income_score"] > 0)
            & (result["momentum_score"] < min_cond)
            & ~result["_selected"]
        ]
        st.markdown(
            f"**Selected** with income>0 & momentum<{min_cond}: **{len(at_risk)}**  "
            f"| **Excluded** by income trap gate: **{len(excluded)}**"
        )
        if len(excluded) and "symbol" in excluded.columns:
            show_cols = ["symbol", "income_score", "momentum_score", "quality_score", "_composite"]
            show_cols = [c for c in show_cols if c in excluded.columns]
            st.dataframe(
                excluded[show_cols].sort_values("income_score", ascending=False).head(15),
                use_container_width=True, hide_index=True,
            )

    # ── 5. Threshold sensitivity ──────────────────────────────────────────
    with st.expander("Threshold sensitivity"):
        sens = _sensitivity_data(df, sw)
        try:
            import altair as alt
            chart = (
                alt.Chart(sens)
                .mark_line(point=True)
                .encode(
                    alt.X("top_percentile:Q", title="Top percentile cutoff"),
                    alt.Y("n_candidates:Q", title="Candidates selected"),
                    tooltip=["top_percentile", "n_candidates", "score_cutoff"],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            st.dataframe(sens, use_container_width=True, hide_index=True)

    # ── 6. A/B/C static pool comparison ──────────────────────────────────
    with st.expander("A / B / C pool comparison (static, no backtest)"):
        _GATE_OFF = -999.0
        modes = {
            "A — absolute threshold": {
                **cs,
                "mode": "absolute",
                "max_candidates": 9999,
                "use_absolute_score_floor": False,
                "absolute_score_floor": _GATE_OFF,
                "min_quality_score": _GATE_OFF,
                "min_momentum_score": _GATE_OFF,
                "min_conditional_momentum_score": _GATE_OFF,
            },
            "B — percentile, no gates": {
                **cs,
                "mode": "percentile",
                "min_quality_score": _GATE_OFF,
                "min_momentum_score": _GATE_OFF,
                "min_conditional_momentum_score": _GATE_OFF,
            },
            "C — percentile + gates": cs,
        }
        rows = []
        for label, mode_cs in modes.items():
            r = _compute_pool(df, sw, mode_cs)
            sel = r[r["_selected"]]
            rows.append({
                "Mode": label,
                "Candidates": len(sel),
                "Cutoff": round(float(r["_cutoff"].iloc[0]), 3),
                "Avg Quality": round(float(sel["quality_score"].mean()), 3) if len(sel) else 0,
                "Avg Momentum": round(float(sel["momentum_score"].mean()), 3) if len(sel) else 0,
                "Avg Income": round(float(sel["income_score"].mean()), 3) if len(sel) else 0,
                "Income>0 count": int((sel["income_score"] > 0).sum()) if len(sel) else 0,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 7. Backtest comparison (slow, on demand) ──────────────────────────
    st.divider()
    st.markdown("**Full A/B/C backtest comparison** — downloads price data, takes ~60s")
    c1, c2 = st.columns([1, 3])
    with c1:
        n_days_bt = st.number_input("Days", 60, 365, 180, step=30, key="cdiag_bt_days")
    with c2:
        bt_mode = st.selectbox(
            "Mode", ["liquid_universe_sanity_test", "walk_forward_price_only_test"],
            key="cdiag_bt_mode",
        )
    if st.button("▶ Run A/B/C backtest comparison", key="cdiag_bt_run"):
        with st.spinner("Running three simulations…"):
            try:
                from backtesting.data_loader import load_and_precompute
                from backtesting.simulator import compare_candidate_selection_modes
                precomp = load_and_precompute(n_days_bt, mode=bt_mode)
                cmp = compare_candidate_selection_modes(precomp)
                st.session_state["cdiag_comparison"] = cmp
                st.success("Done.")
            except Exception as exc:
                st.error(f"Failed: {exc}")

    cmp = st.session_state.get("cdiag_comparison")
    if cmp:
        bench = cmp.get("_benchmark_return", 0.0)
        rows_bt = []
        label_map = {
            "A_absolute": "A absolute",
            "B_percentile": "B percentile",
            "C_percentile_gates": "C +factor gates",
        }
        for key, name in label_map.items():
            if key not in cmp:
                continue
            s = cmp[key]["sim"]
            p = cmp[key]["pool"]
            rows_bt.append({
                "Mode": name,
                "Return": f"{s.total_return:+.2%}",
                "Sharpe": f"{s.sharpe:+.3f}",
                "Max DD": f"{s.max_drawdown:.2%}",
                "Trades": s.trades_made,
                "Candidates (day0)": p.n_candidates,
                "Avg Momentum": f"{p.avg_momentum:+.2f}",
                "Income trap excl.": p.n_income_trap_excluded,
            })
        st.markdown(f"**Benchmark return:** {bench:+.2%}")
        st.dataframe(pd.DataFrame(rows_bt), use_container_width=True, hide_index=True)
