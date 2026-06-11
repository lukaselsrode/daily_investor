"""
ui/components/ablation_runner.py — Config ablation backtest runner.

Runs BacktestEngine with a parameter vector built from a selected config file
(not from the live util.py constants) so you can compare strategies without
touching config.yaml. Also contains:

  - Candidate Drift Diagnostics (pool composition diff between two configs)
  - Exit/Harvest Diagnostics (threshold analysis and live vs sim gap)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT    = Path(__file__).resolve().parents[3]
CFG_DIR = ROOT / "cfg"

_NAMED_CONFIGS = [
    ("Current (config.yaml)",     "config.yaml"),
    ("Baseline snapshot",         "config_baseline_current.yaml"),
    ("Research safe",             "config_research_safe.yaml"),
    ("Momentum anchor",           "config_momentum_anchor.yaml"),
    ("Quality anchor",            "config_quality_anchor.yaml"),
]

from ui.utils import BACKTEST_MODES
from ui.utils import LOOKAHEAD_LEVELS as LOOKAHEAD

# ---------------------------------------------------------------------------
# Config loading + param extraction
# ---------------------------------------------------------------------------

def _load_cfg(filename: str) -> dict | None:
    p = CFG_DIR / filename
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _params_from_cfg(cfg: dict) -> np.ndarray:
    """Build the 15-element CORE params vector from a config dict for ablation
    comparison — deliberately NOT the full 80-slot tuner vector (_current_params),
    only the base score/exit/momentum slots ablations sweep."""
    sw   = cfg.get("score_weights", {})
    mv2w = cfg.get("scoring", {}).get("momentum_inputs", {}).get("weights", {})
    sr   = cfg.get("sell_rules", {})
    sc   = cfg.get("scoring", {})
    return np.array([
        float(sw.get("value",    0.05)),
        float(sw.get("quality",  0.45)),
        float(sw.get("income",   0.20)),
        float(sw.get("momentum", 0.30)),
        float(cfg.get("index_pct",          0.85)),
        float(cfg.get("metric_threshold",   0.75)),
        float(sr.get("take_profit_pct",     0.60)),
        float(sr.get("sell_weak_value_below", 0.45)),
        float(sr.get("trailing_stop_pct",   -0.08)),
        float(sc.get("value_pe_weight",     0.60)),
        float(mv2w.get("rs_3m",             0.25)),
        float(mv2w.get("rs_6m",             0.25)),
        float(mv2w.get("risk_adj_3m",       0.20)),
        float(mv2w.get("trend_structure",   0.15)),
        float(mv2w.get("return_1m",         0.10)),
    ])


def _cs_params_from_cfg(cfg: dict) -> dict:
    cs = cfg.get("candidate_selection", {})
    return {
        "mode":                             cs.get("mode", "percentile"),
        "top_percentile":                   float(cs.get("top_percentile", 0.15)),
        "max_candidates":                   int(cs.get("max_candidates", 25)),
        "min_candidates":                   int(cs.get("min_candidates", 5)),
        "use_absolute_score_floor":         bool(cs.get("use_absolute_score_floor", True)),
        "absolute_score_floor":             float(cs.get("absolute_score_floor", 0.45)),
        "min_quality_score":                float(cs.get("min_quality_score", 0.30)),
        "min_momentum_score":               float(cs.get("min_momentum_score", -0.10)),
        "min_conditional_momentum_score":   float(cs.get("min_conditional_momentum_score", 0.00)),
        "allow_income_defensive_exception": bool(cs.get("allow_income_defensive_exception", False)),
    }


def _backtest_kwargs(cfg: dict) -> dict:
    bt = cfg.get("backtest", {})
    return {
        "starting_capital":          float(bt.get("starting_capital", 5000.0)),
        "slippage_bps":              float(bt.get("slippage_bps", 10.0)),
        "commission_per_trade":      float(bt.get("commission_per_trade", 0.0)),
        "weekly_contribution":       float(bt.get("weekly_contribution", 400.0)),
        "rebalance_frequency_days":  int(bt.get("rebalance_frequency_days", 5)),
    }


# ---------------------------------------------------------------------------
# Run one ablation leg
# ---------------------------------------------------------------------------

def _run_ablation(cfg: dict, n_days: int, mode: str) -> dict:
    """Load price data and simulate. Returns result dict."""
    from backtesting.data_loader import load_and_precompute
    from backtesting.simulator import run_simulation, split_price_window

    params    = _params_from_cfg(cfg)
    cs_params = _cs_params_from_cfg(cfg)
    bt_kw     = _backtest_kwargs(cfg)
    precomp   = load_and_precompute(n_days, mode=mode)
    actual_n  = precomp.prices.shape[0]

    train_slice, val_slice = split_price_window(actual_n, cfg.get("backtest", {}).get("train_pct", 0.70))
    train_pc = precomp._replace(
        prices=precomp.prices[train_slice],
        etf_prices=precomp.etf_prices[train_slice],
        benchmark_prices=precomp.benchmark_prices[train_slice],
        position_52w_daily=precomp.position_52w_daily[train_slice],
        return_1m_daily=precomp.return_1m_daily[train_slice],
        bin_indices_daily=precomp.bin_indices_daily[train_slice],
        has_position_52w_daily=precomp.has_position_52w_daily[train_slice],
        **{k: (getattr(precomp, k)[train_slice] if getattr(precomp, k) is not None else None)
           for k in ["ret_5d_daily", "ret_3m_daily", "ret_6m_daily",
                     "rs_3m_daily", "rs_6m_daily", "vol_3m_daily",
                     "above_50dma_daily", "above_200dma_daily"]},
    )

    sim = run_simulation(train_pc, params, cs_params=cs_params, **bt_kw)

    # Benchmark return (simple price return)
    bp = precomp.benchmark_prices[train_slice]
    bench_return = float(bp[-1] / bp[0] - 1.0) if np.isfinite(bp).all() and bp[0] > 0 else 0.0
    excess       = sim.total_return - bench_return

    diag = sim.pool_diagnostics
    return {
        "total_return":    sim.total_return,
        "benchmark_return": bench_return,
        "excess_return":   excess,
        "sharpe":          sim.sharpe,
        "calmar":          sim.calmar,
        "max_drawdown":    sim.max_drawdown,
        "trades":          sim.trades_made,
        "sells":           sim.sells_made,
        "stopouts":        sim.stopout_count,
        "avg_positions":   sim.average_positions,
        "avg_cash_pct":    sim.average_cash_pct,
        "n_candidates":    diag.n_candidates if diag else None,
        "avg_quality":     diag.avg_quality  if diag else None,
        "avg_momentum":    diag.avg_momentum if diag else None,
        "avg_income":      diag.avg_income   if diag else None,
        "avg_value":       diag.avg_value    if diag else None,
        "sector_counts":   diag.sector_counts if diag else {},
        "excl_hi_inc_lo_mom": diag.excluded_high_income_low_momentum if diag else [],
        "n_floor_excl":    diag.n_floor_excluded if diag else 0,
        "n_inc_trap_excl": diag.n_income_trap_excluded if diag else 0,
        "n_qual_excl":     diag.n_quality_gate_excluded if diag else 0,
        "n_mom_excl":      diag.n_momentum_gate_excluded if diag else 0,
        "lookahead":       precomp.lookahead_bias_level,
        "n_symbols":       len(precomp.symbols),
        "n_days_actual":   actual_n,
    }


# ---------------------------------------------------------------------------
# Ablation Runner tab
# ---------------------------------------------------------------------------

def render_ablation() -> None:
    st.subheader("Ablation Runner")
    st.caption(
        "Run a backtest for any config file without modifying config.yaml. "
        "Parameters are extracted from the selected config and passed directly to the simulation engine."
    )

    # ── Controls ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        _label_to_file = {lbl: f for lbl, f in _NAMED_CONFIGS}
        available_labels = [lbl for lbl, f in _NAMED_CONFIGS if (CFG_DIR / f).exists()]
        cfg_label = st.selectbox("Config file", available_labels, key="abl_cfg")
        cfg_file = _label_to_file.get(cfg_label, "config.yaml")
    with c2:
        n_days = st.selectbox("Window (days)", [30, 60, 90, 180, 365], index=2, key="abl_days")
    with c3:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="abl_mode")

    bias_level, bias_icon = LOOKAHEAD.get(mode, ("MEDIUM", "🟡"))
    st.caption(f"{bias_icon} Lookahead bias: {bias_level}")
    if bias_level == "HIGH":
        st.error("HIGH bias mode uses current fundamentals throughout. Results are NOT predictive.")

    cfg = _load_cfg(cfg_file)
    if cfg is None:
        st.warning(f"`{cfg_file}` not found in cfg/. Create it first via the Config Compare page.")
        return

    # ── Config preview ────────────────────────────────────────────────────
    with st.expander("Selected config preview"):
        sw = cfg.get("score_weights", {})
        cs = cfg.get("candidate_selection", {})
        st.markdown(
            f"**Score weights:** value={sw.get('value', '?'):.2f}  "
            f"quality={sw.get('quality', '?'):.2f}  "
            f"income={sw.get('income', '?'):.2f}  "
            f"momentum={sw.get('momentum', '?'):.2f}"
        )
        st.markdown(
            f"**index_pct:** {float(cfg.get('index_pct', 0.85)):.0%}  |  "
            f"**candidate mode:** {cs.get('mode', 'percentile')}  |  "
            f"**absolute floor:** {'on' if cs.get('use_absolute_score_floor') else 'off'}"
        )

    # ── Run ───────────────────────────────────────────────────────────────
    if st.button("▶ Run ablation backtest", type="primary", key="abl_run"):
        t0 = time.time()
        with st.spinner(f"Running {n_days}-day backtest for '{cfg_label}' …"):
            try:
                result = _run_ablation(cfg, n_days, mode)
                st.session_state["abl_result"] = result
                st.session_state["abl_cfg_label"] = cfg_label
                st.success(f"✅ Done in {time.time()-t0:.1f}s")
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.exception(exc)
                return

    result = st.session_state.get("abl_result")
    if result is None:
        st.info("Configure settings above and click Run.")
        return

    label = st.session_state.get("abl_cfg_label", "")
    st.divider()
    st.subheader(f"Results — {label}")

    # ── Metric tiles ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return (TWR)", f"{result['total_return']:+.1%}")
    c2.metric("Benchmark return",   f"{result['benchmark_return']:+.1%}")
    c3.metric("Excess return",      f"{result['excess_return']:+.1%}",
              delta_color="normal" if result['excess_return'] >= 0 else "inverse")
    c4.metric("Sharpe",             f"{result['sharpe']:+.3f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calmar",             f"{result['calmar']:+.3f}")
    c2.metric("Max drawdown",       f"{result['max_drawdown']:.1%}")
    c3.metric("Trades",             result["trades"])
    c4.metric("Avg open positions", f"{result['avg_positions']:.1f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg cash %",        f"{result['avg_cash_pct']:.1%}")
    c2.metric("Lookahead bias",    result["lookahead"])
    c3.metric("Stopouts",          result["stopouts"])
    c4.metric("Universe symbols",  result["n_symbols"])

    # ── Candidate pool ────────────────────────────────────────────────────
    if result.get("n_candidates") is not None:
        st.subheader("Day-0 Candidate Pool")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates",    result["n_candidates"])
        c2.metric("Avg quality",   f"{result['avg_quality']:.3f}")
        c3.metric("Avg momentum",  f"{result['avg_momentum']:.3f}")
        c4.metric("Avg income",    f"{result['avg_income']:.3f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Floor excluded",    result["n_floor_excl"])
        c2.metric("Income trap excl.", result["n_inc_trap_excl"])
        c3.metric("Quality gate excl.",result["n_qual_excl"])
        c4.metric("Momentum gate excl.",result["n_mom_excl"])

        if result["excl_hi_inc_lo_mom"]:
            st.caption(f"High-income/low-momentum excluded: {', '.join(result['excl_hi_inc_lo_mom'])}")

        if result["sector_counts"]:
            sec_df = (
                pd.DataFrame({"Sector": list(result["sector_counts"].keys()),
                               "Count":  list(result["sector_counts"].values())})
                .sort_values("Count", ascending=False)
            )
            st.dataframe(sec_df, use_container_width=True, hide_index=True)

    if result["lookahead"] == "HIGH":
        st.error("HIGH bias — do not use these results for parameter selection.")
    elif result["lookahead"] == "MEDIUM":
        st.warning("MEDIUM bias — compare results across configs, don't trust absolute return numbers.")


# ---------------------------------------------------------------------------
# Candidate Drift Diagnostics tab
# ---------------------------------------------------------------------------

def render_candidate_drift() -> None:
    st.subheader("Candidate Drift Diagnostics")
    st.caption(
        "Compare candidate pools between two configs. "
        "Loads day-0 candidates from agg_data.csv without running a full backtest."
    )

    c1, c2 = st.columns(2)
    with c1:
        label_a = st.selectbox("Config A", [lbl for lbl, _ in _NAMED_CONFIGS], index=0, key="drift_a")
    with c2:
        label_b = st.selectbox("Config B", [lbl for lbl, _ in _NAMED_CONFIGS], index=2, key="drift_b")

    if label_a == label_b:
        st.info("Select two different configs to compare.")
        return

    _lmap = {lbl: f for lbl, f in _NAMED_CONFIGS}
    cfg_a = _load_cfg(_lmap[label_a])
    cfg_b = _load_cfg(_lmap[label_b])

    if cfg_a is None:
        st.warning(f"{_lmap[label_a]} not found.")
        return
    if cfg_b is None:
        st.warning(f"{_lmap[label_b]} not found.")
        return

    if st.button("▶ Compute candidate drift", type="primary", key="drift_run"):
        with st.spinner("Scoring candidates for both configs…"):
            try:
                from backtesting.data_loader import load_and_precompute as _lp
                from backtesting.simulator import score_stocks, select_candidates
                from util import read_data_as_pd

                agg_df = read_data_as_pd("agg_data")
                if agg_df is None or agg_df.empty:
                    st.error("No agg_data.csv found. Run the bot first to generate data.")
                    return

                params_a  = _params_from_cfg(cfg_a)
                params_b  = _params_from_cfg(cfg_b)
                cs_a      = _cs_params_from_cfg(cfg_a)
                cs_b      = _cs_params_from_cfg(cfg_b)

                precomp = _lp(30, mode="liquid_universe_full")
                scores_a = score_stocks(precomp, params_a)
                scores_b = score_stocks(precomp, params_b)
                mask_a, diag_a = select_candidates(0, scores_a, precomp, params_a, cs_a)
                mask_b, diag_b = select_candidates(0, scores_b, precomp, params_b, cs_b)

                syms   = np.array(precomp.symbols)
                set_a  = set(syms[mask_a])
                set_b  = set(syms[mask_b])

                overlap     = set_a & set_b
                only_a      = set_a - set_b
                only_b      = set_b - set_a

                st.session_state["drift_result"] = {
                    "overlap": sorted(overlap),
                    "only_a":  sorted(only_a),
                    "only_b":  sorted(only_b),
                    "diag_a":  diag_a,
                    "diag_b":  diag_b,
                    "sectors_a": diag_a.sector_counts,
                    "sectors_b": diag_b.sector_counts,
                    "label_a":  label_a,
                    "label_b":  label_b,
                    "hi_inc_lo_mom_a": diag_a.excluded_high_income_low_momentum,
                    "hi_inc_lo_mom_b": diag_b.excluded_high_income_low_momentum,
                }
                st.success("Done.")
            except Exception as exc:
                st.error(f"Drift analysis failed: {exc}")
                st.exception(exc)

    dr = st.session_state.get("drift_result")
    if dr is None:
        return

    st.divider()
    la, lb = dr["label_a"], dr["label_b"]
    da, db = dr["diag_a"], dr["diag_b"]

    # ── Summary ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{la} candidates",     da.n_candidates)
    c2.metric(f"{lb} candidates",     db.n_candidates)
    c3.metric("Overlap",              len(dr["overlap"]))
    c4.metric("Jaccard similarity",   f"{len(dr['overlap'])/max(len(dr['overlap']|set(dr['only_a'])|set(dr['only_b'])),1):.1%}")

    # ── Factor averages comparison ────────────────────────────────────────
    st.subheader("Factor Composition")
    factor_df = pd.DataFrame({
        "Factor":       ["Quality", "Momentum", "Income", "Value"],
        label_a:        [f"{da.avg_quality:.3f}", f"{da.avg_momentum:.3f}", f"{da.avg_income:.3f}", f"{da.avg_value:.3f}"],
        label_b:        [f"{db.avg_quality:.3f}", f"{db.avg_momentum:.3f}", f"{db.avg_income:.3f}", f"{db.avg_value:.3f}"],
    })
    st.dataframe(factor_df.set_index("Factor"), use_container_width=True)

    # ── Symbol sets ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Overlap ({len(dr['overlap'])})**")
        st.caption(", ".join(dr["overlap"][:20]) + ("…" if len(dr["overlap"]) > 20 else ""))
    with c2:
        st.markdown(f"**Only in {la} ({len(dr['only_a'])})**")
        st.caption(", ".join(dr["only_a"][:20]) + ("…" if len(dr["only_a"]) > 20 else ""))
    with c3:
        st.markdown(f"**Only in {lb} ({len(dr['only_b'])})**")
        st.caption(", ".join(dr["only_b"][:20]) + ("…" if len(dr["only_b"]) > 20 else ""))

    # ── Sector drift ──────────────────────────────────────────────────────
    st.subheader("Sector Distribution")
    all_sectors = sorted(set(dr["sectors_a"]) | set(dr["sectors_b"]))
    if all_sectors:
        sec_data = {
            "Sector": all_sectors,
            label_a:  [dr["sectors_a"].get(s, 0) for s in all_sectors],
            label_b:  [dr["sectors_b"].get(s, 0) for s in all_sectors],
        }
        sec_df = pd.DataFrame(sec_data).set_index("Sector").sort_values(label_a, ascending=False)
        st.dataframe(sec_df, use_container_width=True)

    # ── High-income / low-momentum flags ─────────────────────────────────
    st.subheader("High-Income / Low-Momentum Flags")
    st.caption("Names excluded by income trap protection (income > 0, conditional_momentum < threshold)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{la} excluded:**")
        excl_a = dr["hi_inc_lo_mom_a"]
        st.caption(", ".join(excl_a) if excl_a else "None")
    with c2:
        st.markdown(f"**{lb} excluded:**")
        excl_b = dr["hi_inc_lo_mom_b"]
        st.caption(", ".join(excl_b) if excl_b else "None")


# ---------------------------------------------------------------------------
# Exit / Harvest Diagnostics tab
# ---------------------------------------------------------------------------

def render_exit_diagnostics() -> None:
    st.subheader("Exit / Harvest Diagnostics")
    st.caption(
        "Threshold analysis and live vs backtest simulation gap. "
        "Shows which exit rules are active and where the strategies diverge."
    )

    _lmap = {lbl: f for lbl, f in _NAMED_CONFIGS}
    available_labels = [lbl for lbl, f in _NAMED_CONFIGS if (CFG_DIR / f).exists()]
    cfg_label = st.selectbox("Config", available_labels, key="exit_diag_cfg")
    cfg = _load_cfg(_lmap.get(cfg_label, "config.yaml"))
    if cfg is None:
        st.warning("Config not found.")
        return

    exit_d = cfg.get("exit_decision", {})
    sell   = cfg.get("sell_rules", {})
    hv     = cfg.get("harvest", {})

    trim        = float(exit_d.get("trim_profit_threshold", 0.08))
    harvest_t   = float(exit_d.get("harvest_profit_threshold", 0.15))
    take_profit = float(sell.get("take_profit_pct", 0.60))
    trail_stop  = float(sell.get("trailing_stop_pct", -0.08))
    stop_loss   = float(sell.get("stop_loss_pct", -0.20))
    sell_weak   = float(sell.get("sell_weak_value_below", 0.45))
    harv_pct    = float(hv.get("profit_harvest_pct", 0.40))
    min_hold    = int(sell.get("minimum_days_before_take_profit", 10))
    min_value_d = int(sell.get("min_days_held_before_value_exit", 21))

    # ── Live exit ladder ──────────────────────────────────────────────────
    st.subheader("Live Exit Ladder")
    ladder = [
        {"Trigger": "Trailing stop",      "Condition": f"Down {abs(trail_stop):.0%} from peak",         "Action": "Full exit"},
        {"Trigger": "Stop-loss (hard)",   "Condition": f"Down {abs(stop_loss):.0%} from cost",           "Action": "Full exit"},
        {"Trigger": "Trim",               "Condition": f"Up {trim:.0%} from cost (≥ {min_hold}d hold)", "Action": f"Trim {harv_pct:.0%}"},
        {"Trigger": "Harvest",            "Condition": f"Up {harvest_t:.0%} from cost",                  "Action": f"Harvest {harv_pct:.0%}"},
        {"Trigger": "Take-profit",        "Condition": f"Up {take_profit:.0%} from cost",                "Action": "Full exit (unless value intact)"},
        {"Trigger": "Weak value exit",    "Condition": f"Score < {sell_weak:.2f} (after {min_value_d}d)", "Action": "Full exit"},
        {"Trigger": "Score gate (REVIEW)","Condition": f"Score < {exit_d.get('review_score_below', 0.45):.2f}", "Action": "REVIEW flag"},
        {"Trigger": "Hard exit gate",     "Condition": f"Score < {exit_d.get('hard_exit_score_below', 0.20):.2f}", "Action": "Forced exit"},
    ]
    st.dataframe(pd.DataFrame(ladder), use_container_width=True, hide_index=True)

    # ── Backtest simulation ladder ────────────────────────────────────────
    st.subheader("Backtest Simulated Exits")
    bt_ladder = [
        {"Simulated": "Trailing stop",   "Condition": f"Down {abs(trail_stop):.0%} from peak",  "Action": "Full exit ✅"},
        {"Simulated": "Stop-loss",       "Condition": "Down 20% from cost (hardcoded)",          "Action": "Full exit ✅"},
        {"Simulated": "Take-profit",     "Condition": f"Up {take_profit:.0%} from cost",         "Action": "Full exit ✅"},
        {"Simulated": "Weak value exit", "Condition": f"Score < {sell_weak:.2f}",                "Action": "Full exit ✅"},
        {"Simulated": "Trim",            "Condition": "NOT simulated",                            "Action": "❌ Missing"},
        {"Simulated": "Harvest",         "Condition": "NOT simulated",                            "Action": "❌ Missing"},
        {"Simulated": "REVIEW / WATCH",  "Condition": "NOT simulated",                            "Action": "❌ Missing"},
    ]
    st.dataframe(pd.DataFrame(bt_ladder), use_container_width=True, hide_index=True)

    # ── Gap analysis ──────────────────────────────────────────────────────
    st.subheader("Live vs Backtest Gap Analysis")

    gap_col1, gap_col2 = st.columns(2)

    with gap_col1:
        st.markdown("**Winner path (stock rises)**")
        if trim < 0.15:
            st.warning(
                f"Trim fires at +{trim:.0%}. Backtest holds to +{take_profit:.0%}. "
                f"Winners stopped {(take_profit - trim):.0%} early. "
                "In bull markets this gap can cost 10–20% per winner."
            )
        else:
            st.success(f"Trim threshold {trim:.0%} is reasonably close to backtest take-profit {take_profit:.0%}.")

        st.markdown(f"""
| Gain level | Live action | Backtest action |
|---|---|---|
| +{trim:.0%} | Trim {harv_pct:.0%} of position | Hold |
| +{harvest_t:.0%} | Harvest {harv_pct:.0%} more | Hold |
| +{take_profit:.0%} | Full exit (if reached) | Full exit |
        """)

    with gap_col2:
        st.markdown("**Loser path (stock falls)**")
        st.success(
            f"Stop-loss and trailing stop are simulated in backtest. "
            f"Trailing stop at {trail_stop:.0%} from peak is the most active rule."
        )

        bt_hardcoded_stop = -0.20
        if abs(stop_loss - bt_hardcoded_stop) > 0.005:
            st.warning(
                f"Config stop_loss={stop_loss:.0%} but backtest hardcodes {bt_hardcoded_stop:.0%}. "
                "Change stop-loss in config won't affect backtest."
            )

    # ── Contribution split check ──────────────────────────────────────────
    st.divider()
    st.subheader("Backtest Contribution Split")

    index_pct = float(cfg.get("index_pct", 0.85))
    weekly    = float(cfg.get("backtest", {}).get("weekly_contribution", 400.0))
    etf_contrib   = weekly * index_pct
    stock_contrib = weekly - etf_contrib

    c1, c2, c3 = st.columns(3)
    c1.metric("Weekly contribution", f"${weekly:.0f}")
    c2.metric(f"ETF portion ({index_pct:.0%})", f"${etf_contrib:.0f}")
    c3.metric(f"Stock portion ({1-index_pct:.0%})", f"${stock_contrib:.0f}")

    st.success(
        "✅ Backtest now splits weekly contributions by index_pct (fix applied 2026-05-25). "
        "Live bot and backtest use the same allocation schedule."
    )

    # ── Sensitivity estimate ──────────────────────────────────────────────
    with st.expander("Pre-fix vs post-fix allocation estimate (365d)"):
        n_weeks   = 52
        old_stock = weekly * n_weeks        # was 100% to stocks
        new_stock = stock_contrib * n_weeks
        old_etf   = 0.0
        new_etf   = etf_contrib * n_weeks
        st.markdown(f"""
| | Old (pre-fix) | New (post-fix) |
|---|---|---|
| Contributions to stocks | ${old_stock:,.0f} | ${new_stock:,.0f} |
| Contributions to ETFs   | ${old_etf:,.0f}   | ${new_etf:,.0f} |
| Stock over-allocation   | +${old_stock - new_stock:,.0f} vs live | 0 |
        """)
        st.caption(
            f"At index_pct={index_pct:.0%}, the pre-fix backtest deployed "
            f"${old_stock - new_stock:,.0f} more to stocks over 365 days than the live bot does. "
            "Post-fix, the schedules match."
        )


# ---------------------------------------------------------------------------
# Top-level render (dispatches to tabs)
# ---------------------------------------------------------------------------

def render() -> None:
    tabs = st.tabs([
        "▶ Ablation Runner",
        "🔀 Candidate Drift",
        "📉 Exit / Harvest",
    ])
    with tabs[0]:
        render_ablation()
    with tabs[1]:
        render_candidate_drift()
    with tabs[2]:
        render_exit_diagnostics()
