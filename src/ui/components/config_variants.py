"""
ui/components/config_variants.py — In-UI config editor, save-as-variant, and visual comparison.

Edit all simulated parameters directly in the browser, save as a named config variant
(written to the ephemeral, gitignored VARIANTS_DIR — not tracked cfg/), then compare any
set of variants with overlaid equity curves and a metrics bar chart. Comparison also
discovers the code-referenced anchors that live in cfg/ (config_*.yaml).

Production config.yaml is never written from this component — only variant files.

Fully isolated per variant (passed explicitly to run_simulation):
  - score_weights, scoring.momentum_inputs, index_pct, metric_threshold
  - sell_rules (take_profit, trailing_stop, sell_weak), value_pe_weight
  - candidate_selection params

Use current config for all variants (read from globals at call time):
  - exit_decision trim/harvest thresholds
  - risk limits
  - backtest mechanics (cooldowns, slippage)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

ROOT    = Path(__file__).resolve().parents[3]
CFG_DIR = ROOT / "cfg"

# User-saved variants are transient — keep them OUT of the tracked cfg/ directory
# so they don't accumulate as tracked files. Env-overridable; defaults to a
# gitignored dir under the repo.
VARIANTS_DIR = Path(
    os.environ.get("DAILY_INVESTOR_VARIANTS_DIR", str(ROOT / ".variants"))
)

from ui.utils import BACKTEST_MODES

# ---------------------------------------------------------------------------
# Config I/O helpers
# ---------------------------------------------------------------------------

def _load_cfg(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _list_variants() -> list[tuple[str, Path]]:
    """Return (display_name, path) for config.yaml, the cfg/ anchors, and any
    user-saved variants in the ephemeral VARIANTS_DIR.

    Ephemeral user variants take precedence over a same-named cfg/ file.
    """
    result = []
    base = CFG_DIR / "config.yaml"
    if base.exists():
        result.append(("Current (config.yaml)", base))
    seen: set[str] = set()
    for d in (VARIANTS_DIR, CFG_DIR):
        if not d.exists():
            continue
        for p in sorted(d.glob("config_*.yaml")):
            name = p.stem.removeprefix("config_")
            if name in seen:
                continue
            seen.add(name)
            result.append((name, p))
    return result


def _params_from_cfg(cfg: dict) -> np.ndarray:
    sw   = cfg.get("score_weights", {})
    mv2w = cfg.get("scoring", {}).get("momentum_inputs", {}).get("weights", {})
    sr   = cfg.get("sell_rules", {})
    sc   = cfg.get("scoring", {})
    return np.array([
        float(sw.get("value",               0.05)),
        float(sw.get("quality",             0.45)),
        float(sw.get("income",              0.20)),
        float(sw.get("momentum",            0.30)),
        float(cfg.get("index_pct",          0.85)),
        float(cfg.get("metric_threshold",   0.75)),
        float(sr.get("take_profit_pct",     0.60)),
        float(sr.get("sell_weak_value_below", 0.45)),
        float(sr.get("trailing_stop_pct",  -0.08)),
        float(sc.get("value_pe_weight",     0.60)),
        float(mv2w.get("rs_3m",             0.25)),
        float(mv2w.get("rs_6m",             0.25)),
        float(mv2w.get("risk_adj_3m",       0.20)),
        float(mv2w.get("trend_structure",   0.15)),
        float(mv2w.get("return_1m",         0.10)),
        float(mv2w.get("return_5d",         0.05)),
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
        "starting_capital":         float(bt.get("starting_capital", 5000.0)),
        "slippage_bps":             float(bt.get("slippage_bps", 10.0)),
        "commission_per_trade":     float(bt.get("commission_per_trade", 0.0)),
        "weekly_contribution":      float(bt.get("weekly_contribution", 400.0)),
        "rebalance_frequency_days": int(bt.get("rebalance_frequency_days", 5)),
    }


def _save_variant(cfg: dict, name: str) -> Path:
    name = name.strip().lower().replace(" ", "_")
    if not name:
        raise ValueError("Variant name cannot be empty.")
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    path = VARIANTS_DIR / f"config_{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Per-variant backtest runner
# ---------------------------------------------------------------------------

def _run_variant(cfg: dict, n_days: int, mode: str) -> dict:
    try:
        from backtesting.simulator import run_simulation, split_price_window
        from ui.services.backtest_service import load_precomp

        params    = _params_from_cfg(cfg)
        cs_params = _cs_params_from_cfg(cfg)
        bt_kw     = _backtest_kwargs(cfg)
        precomp   = load_precomp(n_days, mode=mode)
        actual_n  = precomp.prices.shape[0]

        train_pct   = float(cfg.get("backtest", {}).get("train_pct", 0.70))
        train_slice, _ = split_price_window(actual_n, train_pct)
        s = train_slice

        daily_keys = [
            "position_52w_daily", "return_1m_daily", "bin_indices_daily",
            "has_position_52w_daily", "ret_5d_daily", "ret_3m_daily",
            "ret_6m_daily", "rs_3m_daily", "rs_6m_daily", "vol_3m_daily",
            "above_50dma_daily", "above_200dma_daily",
        ]
        kw = {k: (getattr(precomp, k)[s] if getattr(precomp, k) is not None else None)
              for k in daily_keys}
        train_pc = precomp._replace(
            prices=precomp.prices[s],
            etf_prices=precomp.etf_prices[s],
            benchmark_prices=precomp.benchmark_prices[s],
            **kw,
        )

        sim = run_simulation(train_pc, params, cs_params=cs_params, **bt_kw)

        bp = precomp.benchmark_prices[s]
        bench_return = float(bp[-1] / bp[0] - 1.0) if len(bp) > 1 and np.isfinite(bp[-1]) and bp[0] > 0 else 0.0

        return {
            "total_return":     sim.total_return,
            "benchmark_return": bench_return,
            "excess_return":    sim.total_return - bench_return,
            "sharpe":           sim.sharpe,
            "calmar":           sim.calmar,
            "max_drawdown":     sim.max_drawdown,
            "trades":           sim.trades_made,
            "equity_curve":     sim.equity_curve,
            "benchmark_equity": sim.benchmark_equity,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Param editor — renders all simulated params as editable controls
# ---------------------------------------------------------------------------

def _param_editor(cfg: dict, key_prefix: str = "cv") -> dict:
    import copy
    out = copy.deepcopy(cfg)

    with st.expander("📊 Score Weights", expanded=True):
        sw = out.setdefault("score_weights", {})
        sc = out.setdefault("scoring", {})
        c1, c2, c3, c4 = st.columns(4)
        sw["value"]    = c1.slider("Value",    0.0, 1.0, float(sw.get("value",    0.05)), 0.01, key=f"{key_prefix}_sw_val")
        sw["quality"]  = c2.slider("Quality",  0.0, 1.0, float(sw.get("quality",  0.45)), 0.01, key=f"{key_prefix}_sw_qua")
        sw["income"]   = c3.slider("Income",   0.0, 1.0, float(sw.get("income",   0.20)), 0.01, key=f"{key_prefix}_sw_inc")
        sw["momentum"] = c4.slider("Momentum", 0.0, 1.0, float(sw.get("momentum", 0.30)), 0.01, key=f"{key_prefix}_sw_mom")
        total = sum(sw.values())
        st.caption(f"Raw sum: {total:.2f} — normalized to 1.0 during simulation.")
        sc["value_pe_weight"] = st.slider(
            "Value: PE weight (vs PB)", 0.0, 1.0, float(sc.get("value_pe_weight", 0.60)), 0.01,
            key=f"{key_prefix}_pe_w",
            help="1.0 = pure PE scoring, 0.0 = pure PB, 0.5 = equal blend",
        )
        sc["value_pb_weight"] = round(1.0 - sc["value_pe_weight"], 4)

    with st.expander("🚀 Momentum v2 Sub-Weights"):
        mv2 = out.setdefault("scoring", {}).setdefault("momentum_inputs", {}).setdefault("weights", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        mv2["rs_3m"]           = c1.slider("RS 3m",       0.0, 1.0, float(mv2.get("rs_3m",           0.25)), 0.01, key=f"{key_prefix}_m_rs3m")
        mv2["rs_6m"]           = c2.slider("RS 6m",       0.0, 1.0, float(mv2.get("rs_6m",           0.25)), 0.01, key=f"{key_prefix}_m_rs6m")
        mv2["risk_adj_3m"]     = c3.slider("Risk-adj 3m", 0.0, 1.0, float(mv2.get("risk_adj_3m",     0.20)), 0.01, key=f"{key_prefix}_m_ra3m")
        mv2["trend_structure"] = c4.slider("Trend",       0.0, 1.0, float(mv2.get("trend_structure", 0.15)), 0.01, key=f"{key_prefix}_m_trend")
        mv2["return_1m"]       = c5.slider("Return 1m",   0.0, 1.0, float(mv2.get("return_1m",       0.10)), 0.01, key=f"{key_prefix}_m_r1m")
        mv2["return_5d"]       = c6.slider("Return 5d",   0.0, 0.30, float(mv2.get("return_5d",      0.05)), 0.01, key=f"{key_prefix}_m_r5d")
        st.caption("Sub-weights are normalized internally before use.")

    with st.expander("🏦 Allocation"):
        c1, c2 = st.columns(2)
        out["index_pct"]        = c1.slider("index_pct",        0.40, 1.00, float(out.get("index_pct",        0.85)), 0.01, key=f"{key_prefix}_idx")
        out["metric_threshold"] = c2.slider("metric_threshold (exit anchor)", 0.30, 1.50, float(out.get("metric_threshold", 1.15)), 0.01, key=f"{key_prefix}_mt")

    with st.expander("🎯 Candidate Selection"):
        cs = out.setdefault("candidate_selection", {})
        c1, c2, c3 = st.columns(3)
        cs["mode"] = c1.selectbox(
            "Mode", ["percentile", "absolute"],
            index=0 if cs.get("mode", "percentile") == "percentile" else 1,
            key=f"{key_prefix}_cs_mode",
        )
        cs["top_percentile"] = c2.slider("Top percentile", 0.05, 0.50, float(cs.get("top_percentile",  0.15)), 0.01, key=f"{key_prefix}_cs_tp")
        cs["max_candidates"] = int(c3.number_input("Max candidates", 5, 50, int(cs.get("max_candidates", 25)), 1, key=f"{key_prefix}_cs_mc"))
        c1, c2, c3 = st.columns(3)
        cs["absolute_score_floor"]             = c1.slider("Absolute floor",  0.0, 0.8,  float(cs.get("absolute_score_floor",             0.45)), 0.01, key=f"{key_prefix}_cs_af")
        cs["min_quality_score"]                = c2.slider("Min quality",     0.0, 0.6,  float(cs.get("min_quality_score",                0.30)), 0.01, key=f"{key_prefix}_cs_mq")
        cs["min_momentum_score"]               = c3.slider("Min momentum",   -0.5, 0.5,  float(cs.get("min_momentum_score",               -0.10)), 0.01, key=f"{key_prefix}_cs_mm")
        c1, c2, c3 = st.columns(3)
        cs["entry_threshold_override"]         = c1.slider("Entry gate (live)", 0.20, 1.00, float(cs.get("entry_threshold_override") or 0.75), 0.01, key=f"{key_prefix}_cs_eto")
        c1, c2, c3 = st.columns(3)
        cs["min_conditional_momentum_score"]   = c1.slider("Min cond mom",   -0.5, 0.5,  float(cs.get("min_conditional_momentum_score",   0.00)), 0.01, key=f"{key_prefix}_cs_cm")
        cs["use_absolute_score_floor"]         = c2.checkbox("Use absolute floor",          bool(cs.get("use_absolute_score_floor",         True)),  key=f"{key_prefix}_cs_uaf")
        cs["allow_income_defensive_exception"] = c3.checkbox("Allow income def. exception", bool(cs.get("allow_income_defensive_exception", False)), key=f"{key_prefix}_cs_aide")

    with st.expander("📉 Sell Rules"):
        sr = out.setdefault("sell_rules", {})
        c1, c2, c3, c4 = st.columns(4)
        sr["take_profit_pct"]       = c1.slider("Take profit",    0.10, 2.00,  float(sr.get("take_profit_pct",        0.60)), 0.01, key=f"{key_prefix}_sr_tp")
        sr["trailing_stop_pct"]     = c2.slider("Trailing stop", -0.45, -0.02, float(sr.get("trailing_stop_pct",      -0.39)), 0.01, key=f"{key_prefix}_sr_ts")
        sr["stop_loss_pct"]         = c3.slider("Stop loss",     -0.50, -0.05, float(sr.get("stop_loss_pct",          -0.20)), 0.01, key=f"{key_prefix}_sr_sl")
        sr["sell_weak_value_below"] = c4.slider("Sell weak value", -0.45, 0.90, float(sr.get("sell_weak_value_below", -0.18)), 0.01, key=f"{key_prefix}_sr_sw")
        c1, c2 = st.columns(2)
        sr["min_days_held_before_value_exit"]    = int(c1.number_input("Min days before value exit",    0, 90, int(sr.get("min_days_held_before_value_exit",    21)), 1, key=f"{key_prefix}_sr_mdv"))
        sr["minimum_days_before_take_profit"]    = int(c2.number_input("Min days before take-profit",   0, 90, int(sr.get("minimum_days_before_take_profit",    0)),  1, key=f"{key_prefix}_sr_mdtp"))

    with st.expander("✂️ Exit / Trim / Harvest"):
        st.caption("These thresholds are saved to YAML but use the current config during comparison runs (global read limitation).")
        ex = out.setdefault("exit_decision", {})
        c1, c2, c3, c4 = st.columns(4)
        ex["trim_enabled"]               = c1.checkbox("Trim enabled",             bool(ex.get("trim_enabled",               True)),  key=f"{key_prefix}_ex_te")
        ex["trim_fraction"]              = c2.slider("Trim fraction",      0.10, 0.70, float(ex.get("trim_fraction",              0.33)), 0.01, key=f"{key_prefix}_ex_tf")
        ex["trim_min_gain_pct"]          = c3.slider("Trim min gain",      0.02, 0.30, float(ex.get("trim_min_gain_pct",          0.08)), 0.01, key=f"{key_prefix}_ex_tmg")
        ex["trim_score_delta_threshold"] = c4.slider("Trim score delta",  -0.50, 0.00, float(ex.get("trim_score_delta_threshold",-0.15)), 0.01, key=f"{key_prefix}_ex_tsd")
        c1, c2, c3, c4 = st.columns(4)
        ex["harvest_profit_threshold"]   = c1.slider("Harvest threshold",  0.05, 0.50, float(ex.get("harvest_profit_threshold",  0.15)), 0.01, key=f"{key_prefix}_ex_hpt")
        ex["harvest_fraction"]           = c2.slider("Harvest fraction",   0.10, 0.70, float(ex.get("harvest_fraction",          0.40)), 0.01, key=f"{key_prefix}_ex_hf")
        ex["review_score_below"]         = c3.slider("Review floor",       0.10, 0.70, float(ex.get("review_score_below",        0.45)), 0.01, key=f"{key_prefix}_ex_rsb")
        ex["hard_exit_score_below"]      = c4.slider("Hard exit floor",   -0.60, 0.40, float(ex.get("hard_exit_score_below",    -0.35)), 0.01, key=f"{key_prefix}_ex_hesb")
        ex["positive_pnl_exit_downgrade"] = st.checkbox(
            "Positive PNL suppresses downgrade exits",
            bool(ex.get("positive_pnl_exit_downgrade", True)),
            key=f"{key_prefix}_ex_ppd",
        )

    with st.expander("🛡️ Risk Limits"):
        st.caption("Saved to YAML; use current config during comparison runs (global read limitation).")
        risk = out.setdefault("risk", {})
        c1, c2, c3 = st.columns(3)
        risk["max_single_position_pct"] = c1.slider("Max single position", 0.02, 0.20, float(risk.get("max_single_position_pct", 0.08)), 0.01, key=f"{key_prefix}_rl_msp")
        risk["max_sector_pct"]          = c2.slider("Max sector",          0.10, 0.60, float(risk.get("max_sector_pct",          0.30)), 0.01, key=f"{key_prefix}_rl_msc")
        risk["max_order_pct_of_cash"]   = c3.slider("Max order of cash",   0.10, 1.00, float(risk.get("max_order_pct_of_cash",   0.50)), 0.01, key=f"{key_prefix}_rl_moc")
        c1, c2 = st.columns(2)
        risk["max_buys_per_rebalance"]  = int(c1.number_input("Max buys / rebalance", 1, 20, int(risk.get("max_buys_per_rebalance", 4)), 1, key=f"{key_prefix}_rl_mbr"))
        risk["min_order_amount"]        = float(c2.number_input("Min order ($)",      1.0, 500.0, float(risk.get("min_order_amount", 5.0)), 1.0, key=f"{key_prefix}_rl_moa"))

    with st.expander("⚙️ Backtest Mechanics"):
        st.caption("Saved to YAML; use current config during comparison runs (global read limitation).")
        bt = out.setdefault("backtest", {})
        c1, c2, c3 = st.columns(3)
        bt["cooldown_days_after_sell"]    = int(c1.number_input("Cooldown after sell (days)",    0, 20, int(bt.get("cooldown_days_after_sell",    3)), 1, key=f"{key_prefix}_bt_cas"))
        bt["cooldown_days_after_stopout"] = int(c2.number_input("Cooldown after stopout (days)", 0, 30, int(bt.get("cooldown_days_after_stopout", 7)), 1, key=f"{key_prefix}_bt_cso"))
        bt["max_trades_per_week"]         = int(c3.number_input("Max trades / week",             1, 30, int(bt.get("max_trades_per_week",        10)), 1, key=f"{key_prefix}_bt_mtw"))
        c1, c2, c3 = st.columns(3)
        bt["slippage_bps"]                = float(c1.number_input("Slippage (bps)",           0.0, 100.0, float(bt.get("slippage_bps",              10.0)), 1.0, key=f"{key_prefix}_bt_slip"))
        bt["vol_slippage_scaling"]        = c2.checkbox("Volume slippage scaling",            bool(bt.get("vol_slippage_scaling",  True)),  key=f"{key_prefix}_bt_vss")
        bt["vol_slippage_multiplier"]     = float(c3.number_input("Vol slippage multiplier", 0.0, 5.0, float(bt.get("vol_slippage_multiplier", 2.0)), 0.1, key=f"{key_prefix}_bt_vsm"))

    return out


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------

def _comparison_charts(results: dict[str, dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("plotly not installed.")
        return

    colors = ["#4c8ef5", "#f5a623", "#4CAF50", "#e91e63", "#9c27b0", "#00bcd4"]

    # ── Overlaid equity curves ────────────────────────────────────────────
    fig_eq = go.Figure()
    bench_added = False
    for i, (name, r) in enumerate(results.items()):
        if "error" in r:
            continue
        eq = r.get("equity_curve", np.array([]))
        if len(eq) == 0:
            continue
        eq_idx = eq / max(float(eq[0]), 1e-9) * 100.0
        fig_eq.add_trace(go.Scatter(
            x=np.arange(len(eq_idx)), y=eq_idx,
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>",
        ))
        if not bench_added:
            be = r.get("benchmark_equity", np.array([]))
            if len(be) > 0 and len(be) == len(eq):
                be_idx = be / max(float(be[0]), 1e-9) * 100.0
                fig_eq.add_trace(go.Scatter(
                    x=np.arange(len(be_idx)), y=be_idx,
                    name="Benchmark",
                    line=dict(color="#aaaaaa", width=1.5, dash="dot"),
                ))
                bench_added = True

    fig_eq.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
    fig_eq.update_layout(
        title="Equity Curves (indexed to 100 at start)",
        height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Metrics bar chart ─────────────────────────────────────────────────
    good_names = [n for n, r in results.items() if "error" not in r]
    if not good_names:
        return

    metric_pairs = [
        ("total_return",  "Total Return"),
        ("excess_return", "Excess Return"),
        ("sharpe",        "Sharpe"),
        ("max_drawdown",  "Max Drawdown"),
    ]
    fig_bar = make_subplots(
        rows=1, cols=len(metric_pairs),
        subplot_titles=[lbl for _, lbl in metric_pairs],
    )
    for j, (metric, _) in enumerate(metric_pairs):
        vals = [results[n].get(metric, 0.0) for n in good_names]
        fig_bar.add_trace(go.Bar(
            x=good_names, y=vals,
            marker_color=colors[:len(good_names)],
            showlegend=False,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ), row=1, col=j + 1)
    fig_bar.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=40, b=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for name, r in results.items():
        if "error" in r:
            rows.append({"Config": name, "Return": "—", "Excess": "—",
                         "Sharpe": "—", "Calmar": "—", "Max DD": "—",
                         "Trades": "—", "Error": r["error"]})
        else:
            rows.append({
                "Config":  name,
                "Return":  f"{r['total_return']:+.1%}",
                "Excess":  f"{r['excess_return']:+.1%}",
                "Sharpe":  f"{r['sharpe']:+.3f}",
                "Calmar":  f"{r['calmar']:+.3f}",
                "Max DD":  f"{r['max_drawdown']:.1%}",
                "Trades":  r.get("trades", "—"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------

def render_config_variants() -> None:
    st.subheader("Config Variants")
    st.caption(
        "Edit any simulated parameter, save as a named variant, then compare variants "
        "with overlaid equity curves. **config.yaml is never modified here.**"
    )

    base_cfg = _load_cfg(CFG_DIR / "config.yaml")
    if not base_cfg:
        st.warning("config.yaml not found in cfg/. Cannot load baseline.")
        return

    # ── Param editor ──────────────────────────────────────────────────────
    edited_cfg = _param_editor(base_cfg, key_prefix="cv_edit")

    # ── Save as variant ───────────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        variant_name = st.text_input(
            "Variant name",
            placeholder="e.g. high_quality, low_index, momentum_heavy",
            key="cv_variant_name",
        )
    with c2:
        st.write("")
        st.write("")
        if st.button("💾 Save variant", key="cv_save_btn", use_container_width=True):
            if not variant_name.strip():
                st.error("Enter a name.")
            else:
                try:
                    path = _save_variant(edited_cfg, variant_name)
                    st.success(f"Saved: {path.name}")
                except Exception as exc:
                    st.error(f"Save failed: {exc}")
    with c3:
        st.write("")
        st.write("")
        existing = _list_variants()
        variant_files = [p for _, p in existing if p.stem.startswith("config_")]
        if variant_files and st.button("🗑️ Delete variant", key="cv_del_btn", use_container_width=True):
            st.session_state["cv_show_delete"] = True
    if st.session_state.get("cv_show_delete"):
        del_map = {n: p for n, p in _list_variants() if p.stem.startswith("config_")}
        to_del = st.selectbox("Select variant to delete", list(del_map), key="cv_del_sel")
        if st.button("Confirm delete", key="cv_del_confirm"):
            try:
                path = del_map[to_del]
                path.unlink(missing_ok=True)
                st.success(f"Deleted {path.name}")
                st.session_state.pop("cv_show_delete", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Delete failed: {exc}")

    # ── Compare variants ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Compare Variants")

    variants = _list_variants()
    if len(variants) < 2:
        st.info("Save at least one variant above to enable comparison.")
        return

    all_names = [name for name, _ in variants]
    selected = st.multiselect(
        "Variants to compare",
        all_names,
        default=all_names[:min(4, len(all_names))],
        key="cv_compare_sel",
    )
    if not selected:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        n_days = int(c1.number_input("History (days)", 30, 1000, 90, 30, key="cv_n_days"))
    with c2:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="cv_mode")
    with c3:
        st.write("")
        st.write("")
        run = st.button("▶ Compare", type="primary", key="cv_compare_run", use_container_width=True)

    if run:
        name_to_path = {name: path for name, path in variants}
        results: dict[str, dict] = {}
        progress = st.progress(0, text="Starting…")
        for i, name in enumerate(selected):
            progress.progress(int((i + 0.5) / len(selected) * 100), text=f"Running {name}…")
            cfg = _load_cfg(name_to_path[name])
            r = _run_variant(cfg, n_days, mode)
            results[name] = r
            if "error" in r:
                st.warning(f"{name}: {r['error']}")
        progress.empty()
        st.session_state["cv_results"] = results
        good = sum(1 for r in results.values() if "error" not in r)
        st.success(f"Ran {good}/{len(results)} variants successfully.")

    results = st.session_state.get("cv_results")
    if results:
        _comparison_charts(results)
