"""
ui/components/config_compare.py — Side-by-side comparison of named config files.

Shows score weights, index_pct, candidate selection, exit/trim/harvest settings,
and turnover penalty settings for each config. Highlights differences from current.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

ROOT    = Path(__file__).resolve().parents[3]
CFG_DIR = ROOT / "cfg"

# Named configs to compare (file must exist; missing ones are skipped with a note)
_NAMED_CONFIGS = [
    ("Current (config.yaml)",        "config.yaml"),
    ("Baseline snapshot",            "config_baseline_current.yaml"),
    ("Research safe",                "config_research_safe.yaml"),
    ("Momentum anchor",              "config_momentum_anchor.yaml"),
    ("Quality anchor",               "config_quality_anchor.yaml"),
]


def _load(filename: str) -> dict | None:
    p = CFG_DIR / filename
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _nested_get(d: dict, *keys, default=None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _fmt(v: Any, as_pct: bool = False) -> str:
    if v is None:
        return "—"
    if as_pct and isinstance(v, (int, float)):
        return f"{float(v):.0%}"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _diff_marker(current_val: Any, other_val: Any) -> str:
    """Return ▲/▼/= or empty to show direction of change from current."""
    if current_val is None or other_val is None:
        return ""
    try:
        c, o = float(current_val), float(other_val)
        if abs(c - o) < 1e-6:
            return ""
        return "▲" if o > c else "▼"
    except (TypeError, ValueError):
        return "" if current_val == other_val else "≠"


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------

def _score_weights(cfg: dict) -> dict:
    sw = cfg.get("score_weights", {})
    return {
        "value":    float(sw.get("value",    0.05)),
        "quality":  float(sw.get("quality",  0.45)),
        "income":   float(sw.get("income",   0.20)),
        "momentum": float(sw.get("momentum", 0.30)),
    }


def _allocation(cfg: dict) -> dict:
    risk = cfg.get("risk", {})
    return {
        "index_pct":      float(cfg.get("index_pct", 0.85)),
        "min_index_pct":  float(risk.get("min_index_pct", 0.60)),
        "min_candidate_allocation_pct": float(risk.get("min_candidate_allocation_pct", 0.10)),
        "max_order_pct_of_cash": float(risk.get("max_order_pct_of_cash", 0.50)),
        "max_buys_per_rebalance": int(risk.get("max_buys_per_rebalance", 4)),
    }


def _candidate_selection(cfg: dict) -> dict:
    cs = cfg.get("candidate_selection", {})
    return {
        "mode":                           cs.get("mode", "percentile"),
        "top_percentile":                 float(cs.get("top_percentile", 0.15)),
        "max_candidates":                 int(cs.get("max_candidates", 25)),
        "use_absolute_score_floor":       bool(cs.get("use_absolute_score_floor", True)),
        "absolute_score_floor":           float(cs.get("absolute_score_floor", 0.45)),
        "min_quality_score":              float(cs.get("min_quality_score", 0.30)),
        "min_momentum_score":             float(cs.get("min_momentum_score", -0.10)),
        "min_conditional_momentum_score": float(cs.get("min_conditional_momentum_score", 0.00)),
        "allow_income_defensive_exception": bool(cs.get("allow_income_defensive_exception", False)),
    }


def _exit_harvest(cfg: dict) -> dict:
    exit_d = cfg.get("exit_decision", {})
    sell   = cfg.get("sell_rules", {})
    hv     = cfg.get("harvest", {})
    return {
        "trim_profit_threshold":    float(exit_d.get("trim_profit_threshold", 0.08)),
        "harvest_profit_threshold": float(exit_d.get("harvest_profit_threshold", 0.15)),
        "take_profit_pct (backtest)": float(sell.get("take_profit_pct", 0.60)),
        "trailing_stop_pct":        float(sell.get("trailing_stop_pct", -0.08)),
        "stop_loss_pct":            float(sell.get("stop_loss_pct", -0.20)),
        "sell_weak_value_below":    float(sell.get("sell_weak_value_below", 0.45)),
        "profit_harvest_pct":       float(hv.get("profit_harvest_pct", 0.40)),
        "hard_exit_score_below":    float(exit_d.get("hard_exit_score_below", 0.20)),
        "review_score_below":       float(exit_d.get("review_score_below", 0.45)),
    }


def _turnover_penalty(cfg: dict) -> dict:
    bt = cfg.get("backtest", {})
    return {
        "turnover_penalty_enabled":      bool(bt.get("turnover_penalty_enabled", True)),
        "turnover_penalty_trade_count":  int(bt.get("turnover_penalty_trade_count", 80)),
        "turnover_penalty_weight":       float(bt.get("turnover_penalty_weight", 1.0)),
        "max_trades_per_week":           int(bt.get("max_trades_per_week", 10)),
        "cooldown_days_after_sell":      int(bt.get("cooldown_days_after_sell", 3)),
        "cooldown_days_after_stopout":   int(bt.get("cooldown_days_after_stopout", 7)),
    }


def _frozen_params(cfg: dict) -> str:
    tn = cfg.get("tuning", {})
    frozen = tn.get("frozen_parameters", [])
    if not frozen:
        return "none"
    short = [p.split(".")[-1] for p in frozen]
    return ", ".join(short[:6]) + ("…" if len(short) > 6 else "")


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def _build_section_df(
    section_name: str,
    configs: list[tuple[str, dict]],
    extractor,
    as_pct_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame with rows = fields, columns = config names."""
    as_pct_keys = as_pct_keys or set()
    rows = []
    current_vals = extractor(configs[0][1]) if configs else {}

    all_keys = list(extractor(configs[0][1]).keys()) if configs else []

    for key in all_keys:
        row: dict = {"Field": key}
        current_val = current_vals.get(key)
        for label, cfg in configs:
            vals = extractor(cfg)
            v = vals.get(key)
            formatted = _fmt(v, as_pct=key in as_pct_keys)
            if label != configs[0][0]:
                marker = _diff_marker(current_val, v)
                row[label] = f"{formatted} {marker}".strip() if marker else formatted
            else:
                row[label] = formatted
        rows.append(row)

    return pd.DataFrame(rows).set_index("Field")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Config Compare")
    st.caption(
        "Side-by-side comparison of all named config files. "
        "▲/▼ markers show direction of change from the current config."
    )

    # Load all configs
    loaded: list[tuple[str, dict]] = []
    for label, fname in _NAMED_CONFIGS:
        cfg = _load(fname)
        if cfg is None:
            st.caption(f"⚠️ {fname} not found — skipped")
        else:
            loaded.append((label, cfg))

    if not loaded:
        st.error("No config files found in cfg/.")
        return

    # Allow the user to pick which configs to compare
    all_labels = [l for l, _ in loaded]
    selected = st.multiselect(
        "Configs to show",
        all_labels,
        default=all_labels,
        key="compare_selected",
    )
    visible = [(l, c) for l, c in loaded if l in selected]
    if not visible:
        st.info("Select at least one config above.")
        return

    # ── Score weights ─────────────────────────────────────────────────────
    st.subheader("Score Weights")
    df = _build_section_df("weights", visible, _score_weights)
    st.dataframe(df, use_container_width=True)

    # ── Allocation ────────────────────────────────────────────────────────
    st.subheader("Allocation & ETF/Stock Split")
    df = _build_section_df(
        "alloc", visible, _allocation,
        as_pct_keys={"index_pct", "min_index_pct", "min_candidate_allocation_pct", "max_order_pct_of_cash"},
    )
    st.dataframe(df, use_container_width=True)

    st.caption(
        "⚠️ index_pct below min_index_pct means the live bot runs at an allocation "
        "the tuner has never explored."
    )

    # ── Candidate selection ───────────────────────────────────────────────
    st.subheader("Candidate Selection")
    df = _build_section_df(
        "cs", visible, _candidate_selection,
        as_pct_keys={"top_percentile", "absolute_score_floor", "min_quality_score",
                     "min_momentum_score", "min_conditional_momentum_score"},
    )
    st.dataframe(df, use_container_width=True)

    # ── Exit / Trim / Harvest ─────────────────────────────────────────────
    st.subheader("Exit / Trim / Harvest Thresholds")
    df = _build_section_df(
        "exit", visible, _exit_harvest,
        as_pct_keys={"trim_profit_threshold", "harvest_profit_threshold",
                     "take_profit_pct (backtest)", "trailing_stop_pct",
                     "stop_loss_pct", "sell_weak_value_below", "profit_harvest_pct",
                     "hard_exit_score_below", "review_score_below"},
    )
    st.dataframe(df, use_container_width=True)

    st.warning(
        "Backtest simulates a full exit at `take_profit_pct`. "
        "Live bot trims at `trim_profit_threshold` and harvests at `harvest_profit_threshold`. "
        "Large gaps between these values make backtest vs live comparisons unreliable."
    )

    # ── Turnover penalty ──────────────────────────────────────────────────
    st.subheader("Backtest / Turnover Penalty")
    df = _build_section_df("tp", visible, _turnover_penalty)
    st.dataframe(df, use_container_width=True)

    # ── Frozen parameters summary ─────────────────────────────────────────
    st.subheader("Frozen Parameters")
    rows = [{"Config": l, "Frozen params": _frozen_params(c)} for l, c in visible]
    st.dataframe(pd.DataFrame(rows).set_index("Config"), use_container_width=True)

    # ── Major diffs from current ──────────────────────────────────────────
    if len(visible) < 2:
        return

    st.divider()
    st.subheader("Key Differences from Current Config")

    current_label, current_cfg = visible[0]
    for label, cfg in visible[1:]:
        diffs = []

        # Score weights
        sw_curr = _score_weights(current_cfg)
        sw_other = _score_weights(cfg)
        for k, cv in sw_curr.items():
            ov = sw_other.get(k, cv)
            if abs(cv - ov) > 0.005:
                diffs.append(f"score_weights.{k}: {cv:.2f} → {ov:.2f}")

        # index_pct
        ci = float(current_cfg.get("index_pct", 0.5))
        oi = float(cfg.get("index_pct", 0.5))
        if abs(ci - oi) > 0.005:
            diffs.append(f"index_pct: {ci:.2f} → {oi:.2f}")

        # exit thresholds
        eh_curr  = _exit_harvest(current_cfg)
        eh_other = _exit_harvest(cfg)
        for k in ("trim_profit_threshold", "harvest_profit_threshold"):
            cv, ov = eh_curr.get(k, 0), eh_other.get(k, 0)
            if abs(float(cv) - float(ov)) > 0.005:
                diffs.append(f"exit_decision.{k}: {float(cv):.0%} → {float(ov):.0%}")

        # candidate selection
        cs_curr  = _candidate_selection(current_cfg)
        cs_other = _candidate_selection(cfg)
        for k in ("use_absolute_score_floor", "min_conditional_momentum_score", "allow_income_defensive_exception"):
            cv, ov = cs_curr.get(k), cs_other.get(k)
            if cv != ov:
                diffs.append(f"candidate_selection.{k}: {cv} → {ov}")

        # turnover
        tp_curr  = _turnover_penalty(current_cfg)
        tp_other = _turnover_penalty(cfg)
        for k in ("turnover_penalty_trade_count", "turnover_penalty_weight"):
            cv, ov = tp_curr.get(k, 0), tp_other.get(k, 0)
            if abs(float(cv) - float(ov)) > 0.005:
                diffs.append(f"backtest.{k}: {cv} → {ov}")

        with st.expander(f"Current vs {label} ({len(diffs)} changes)", expanded=bool(diffs)):
            if not diffs:
                st.success("No significant differences.")
            else:
                for d in diffs:
                    st.code(d, language="yaml")
