"""
tuning/reports.py — Display helpers, config writer, and LLM review utilities.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import yaml

from backtesting.types import SimResult
from util import CONFIG_FILE, RISK_LIMITS

from .constants import (
    _CONFIG_PATH_TO_PARAM_IDX,
    _current_params,
    PARAM_NAMES,
)

logger = logging.getLogger(__name__)

_LLM_ALLOWED_PARAMS = frozenset([
    "score_weights", "metric_threshold", "index_pct",
    "take_profit_pct", "trailing_stop_pct", "sell_weak_value_below",
    "value_pe_weight", "momentum_v2_weights",
])
_LLM_FORBIDDEN_PARAMS = frozenset([
    "max_single_position_pct", "max_sector_pct", "max_order_pct_of_cash",
    "min_order_amount", "min_liquidity_volume", "allow_whole_share_fallback",
    "max_whole_share_buys_per_run", "max_whole_share_allocation_multiplier",
    "stop_loss_pct", "weekly_investment",
])


def apply_config_params(params: np.ndarray) -> None:
    """Write tuned parameters back to config.yaml, preserving all other keys."""
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    raw_sw = params[:4]
    sw = raw_sw / max(raw_sw.sum(), 1e-9)

    min_idx = RISK_LIMITS["min_index_pct"]
    cfg["index_pct"] = round(max(float(params[4]), min_idx), 4)
    cfg["metric_threshold"] = round(float(params[5]), 4)

    cfg.setdefault("score_weights", {})
    cfg["score_weights"]["value"]    = round(float(sw[0]), 4)
    cfg["score_weights"]["quality"]  = round(float(sw[1]), 4)
    cfg["score_weights"]["income"]   = round(float(sw[2]), 4)
    cfg["score_weights"]["momentum"] = round(float(sw[3]), 4)

    cfg.setdefault("sell_rules", {})
    cfg["sell_rules"]["take_profit_pct"]       = round(float(params[6]), 4)
    cfg["sell_rules"]["sell_weak_value_below"] = round(float(params[7]), 4)
    cfg["sell_rules"]["trailing_stop_pct"]     = round(float(params[8]), 4)

    cfg.setdefault("scoring", {})
    cfg["scoring"]["value_pe_weight"] = round(float(params[9]), 4)
    cfg["scoring"]["value_pb_weight"] = round(float(1.0 - params[9]), 4)

    v2_raw = np.abs(params[10:15])
    v2_total = max(float(v2_raw.sum()), 1e-9)
    v2_norm = v2_raw / v2_total
    v2_keys = ["rs_3m", "rs_6m", "risk_adj_3m", "trend_structure", "return_1m"]
    cfg.setdefault("momentum_v2", {}).setdefault("weights", {})
    for k, v in zip(v2_keys, v2_norm):
        cfg["momentum_v2"]["weights"][k] = round(float(v), 4)

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nconfig.yaml updated: {CONFIG_FILE}")


def _diff_table(
    best_params: np.ndarray,
    label: str = "",
    sharpe_ref: "SimResult | None" = None,
    calmar_ref: "SimResult | None" = None,
    sharpe_params: "np.ndarray | None" = None,
    calmar_params: "np.ndarray | None" = None,
) -> None:
    cur = _current_params()
    raw_sw = best_params[:4]
    norm_sw = raw_sw / max(raw_sw.sum(), 1e-9)
    cur_sw_norm = cur[:4] / max(cur[:4].sum(), 1e-9)

    cur_v2_raw = np.abs(cur[10:15])
    cur_v2_norm = cur_v2_raw / max(cur_v2_raw.sum(), 1e-9)
    v2_raw = np.abs(best_params[10:15])
    v2_norm = v2_raw / max(v2_raw.sum(), 1e-9)

    header = f"AVERAGED CONFIG ({label})" if label else "SUGGESTED CONFIG"
    print(f"\n{'=' * 64}")
    print(header)
    print("=" * 64)

    if sharpe_ref:
        print(
            f"  Sharpe run:  ret={sharpe_ref.total_return:+.1%}  "
            f"sharpe={sharpe_ref.sharpe:+.3f}  trades={sharpe_ref.trades_made}"
        )
    if calmar_ref:
        print(
            f"  Calmar run:  ret={calmar_ref.total_return:+.1%}  "
            f"calmar={calmar_ref.calmar:+.3f}  trades={calmar_ref.trades_made}"
        )
    print()

    v2_keys = ["rs_3m", "rs_6m", "risk_adj_3m", "trend_structure", "return_1m"]
    rows = [
        ("score_weights.value",              cur_sw_norm[0],  norm_sw[0]),
        ("score_weights.quality",            cur_sw_norm[1],  norm_sw[1]),
        ("score_weights.income",             cur_sw_norm[2],  norm_sw[2]),
        ("score_weights.momentum",           cur_sw_norm[3],  norm_sw[3]),
        ("index_pct",                        cur[4],           best_params[4]),
        ("metric_threshold",                 cur[5],           best_params[5]),
        ("sell_rules.take_profit",           cur[6],           best_params[6]),
        ("sell_rules.sell_weak",             cur[7],           best_params[7]),
        ("sell_rules.trailing_stop",         cur[8],           best_params[8]),
        ("scoring.value_pe_weight",          cur[9],           best_params[9]),
        ("momentum_v2.weights.rs_3m",        cur_v2_norm[0],   v2_norm[0]),
        ("momentum_v2.weights.rs_6m",        cur_v2_norm[1],   v2_norm[1]),
        ("momentum_v2.weights.risk_adj_3m",  cur_v2_norm[2],   v2_norm[2]),
        ("momentum_v2.weights.trend",        cur_v2_norm[3],   v2_norm[3]),
        ("momentum_v2.weights.return_1m",    cur_v2_norm[4],   v2_norm[4]),
    ]

    print("CHANGES  (> 1% relative)")
    print("-" * 64)
    any_change = False
    for lbl, old, new in rows:
        rel = abs(new - old) / max(abs(old), 1e-9)
        if rel > 0.01:
            arrow = "▲" if new > old else "▼"
            print(f"  {lbl:<42}  {old:+.4f}  →  {new:+.4f}  {arrow}")
            any_change = True
    if not any_change:
        print("  (no meaningful changes)")

    if sharpe_params is not None and calmar_params is not None:
        print("\nPARAMETER STABILITY  (|sharpe_opt - calmar_opt|)")
        print("-" * 64)
        for i, name in enumerate(PARAM_NAMES):
            spread = abs(float(sharpe_params[i]) - float(calmar_params[i]))
            if spread > 0.05:
                print(f"  {name:<36}  spread={spread:.4f}  ⚠ unstable")

    print("\nconfig.yaml SNIPPET")
    print("-" * 64)
    print("score_weights:")
    for key, val in zip(["value", "quality", "income", "momentum"], norm_sw):
        print(f"  {key}: {val:.4f}")
    print(f"index_pct: {best_params[4]:.4f}")
    print(f"metric_threshold: {best_params[5]:.4f}")
    print("sell_rules:")
    print(f"  take_profit_pct: {best_params[6]:.4f}")
    print(f"  sell_weak_value_below: {best_params[7]:.4f}")
    print(f"  trailing_stop_pct: {best_params[8]:.4f}")
    print("scoring:")
    print(f"  value_pe_weight: {best_params[9]:.4f}")
    print(f"  value_pb_weight: {1.0 - best_params[9]:.4f}")
    print("momentum_v2:")
    print("  weights:")
    for k, v in zip(v2_keys, v2_norm):
        print(f"    {k}: {v:.4f}")
    print("=" * 64)


def print_config_diff(best_params: np.ndarray, best_result: SimResult) -> None:
    """Display diff for a single-objective tune run."""
    print(f"\n{'=' * 64}")
    print("TUNER RESULTS")
    print("=" * 64)
    print(
        f"  Sharpe:      {best_result.sharpe:+.3f}\n"
        f"  Calmar:      {best_result.calmar:+.3f}\n"
        f"  Total return:{best_result.total_return:+.1%}\n"
        f"  Max drawdown:{best_result.max_drawdown:.1%}\n"
        f"  Trades:      {best_result.trades_made}\n"
    )
    _diff_table(best_params)


def build_llm_review_payload(
    candidates: list[dict],
    mode: str,
    universe_selection: str,
    benchmark_symbol: str,
    validation_cfg: dict,
) -> dict:
    safe_candidates = []
    for c in candidates:
        safe = {
            "candidate_id": c.get("candidate_id", ""),
            "alpha_params": {k: v for k, v in c.get("alpha_params", {}).items()
                             if k in _LLM_ALLOWED_PARAMS},
            "train": {
                "total_return": c.get("train_return"),
                "sharpe": c.get("train_sharpe"),
                "calmar": c.get("train_calmar"),
                "max_drawdown": c.get("train_max_drawdown"),
                "trades": c.get("train_trades"),
                "avg_positions": c.get("train_avg_positions"),
                "max_positions": c.get("train_max_positions"),
                "avg_cash_pct": c.get("train_avg_cash_pct"),
                "turnover": c.get("train_turnover"),
                "friction_cost": c.get("train_friction_cost"),
            },
            "validation": {
                "total_return": c.get("val_return"),
                "sharpe": c.get("val_sharpe"),
                "max_drawdown": c.get("val_max_drawdown"),
            },
            "benchmark": {
                "symbol": benchmark_symbol,
                "total_return": c.get("bench_return"),
                "sharpe": c.get("bench_sharpe"),
                "max_drawdown": c.get("bench_max_drawdown"),
            },
            "excess_return": c.get("excess_return"),
            "lookahead_bias_level": c.get("lookahead_bias_level"),
            "notes": c.get("notes", []),
        }
        safe_candidates.append(safe)

    return {
        "task": "review_auto_tune_candidates",
        "mode": mode,
        "universe_selection": universe_selection,
        "n_candidates": len(safe_candidates),
        "validation_gates": {
            "min_validation_excess_return": validation_cfg.get("min_validation_excess_return"),
            "max_validation_drawdown": validation_cfg.get("max_validation_drawdown"),
            "min_validation_sharpe": validation_cfg.get("min_validation_sharpe"),
        },
        "candidates": safe_candidates,
        "instructions": (
            "You are reviewing parameter optimization candidates for a core-satellite "
            "investment strategy. Recommend the best candidate or propose minor adjustments "
            "to alpha parameters only. Safety parameters are off-limits. "
            "Respond with valid JSON matching the specified schema exactly."
        ),
    }


def request_llm_tune_review(payload: dict) -> dict:
    import json
    import os

    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package required. Install: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    from util import BACKTEST_PARAMS
    model = BACKTEST_PARAMS.get("llm_review_model", "claude-sonnet-4-6")

    schema = (
        '{"recommended_candidate_id": "candidate_N", '
        '"apply_candidate_as_is": true, '
        '"proposed_adjustments": {}, '
        '"rationale": "...", '
        '"risk_warnings": [], '
        '"confidence": 0.0}'
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{payload['instructions']}\n\n"
                    f"Candidate data (JSON):\n{json.dumps(payload, indent=2)}\n\n"
                    f"Respond with valid JSON matching this schema:\n{schema}"
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")


def validate_llm_review_response(
    response: dict,
    candidates: list[dict],
) -> tuple[bool, list[str]]:
    errors: list[str] = []

    required = ["recommended_candidate_id", "apply_candidate_as_is",
                 "proposed_adjustments", "rationale", "risk_warnings", "confidence"]
    for key in required:
        if key not in response:
            errors.append(f"Missing required key: {key}")

    if errors:
        return False, errors

    candidate_ids = {c.get("candidate_id") for c in candidates}
    rec_id = response.get("recommended_candidate_id")
    if rec_id not in candidate_ids:
        errors.append(f"recommended_candidate_id '{rec_id}' not in candidate list")

    adjustments = response.get("proposed_adjustments", {})
    if not isinstance(adjustments, dict):
        errors.append("proposed_adjustments must be a dict")
    else:
        for k in adjustments:
            if k in _LLM_FORBIDDEN_PARAMS:
                errors.append(f"LLM proposed forbidden safety param: {k}")
            elif k not in _LLM_ALLOWED_PARAMS:
                errors.append(f"LLM proposed unknown param: {k}")

    conf = response.get("confidence", -1)
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        errors.append(f"confidence must be float in [0, 1], got {conf!r}")

    return len(errors) == 0, errors


def merge_llm_recommendation_with_config(
    base_config: dict,
    llm_response: dict,
) -> dict:
    cfg = copy.deepcopy(base_config)
    adjustments = llm_response.get("proposed_adjustments", {})

    for key, value in adjustments.items():
        if key in _LLM_FORBIDDEN_PARAMS:
            logger.warning("LLM merge: skipping forbidden param %s", key)
            continue
        if key == "score_weights" and isinstance(value, dict):
            cfg.setdefault("score_weights", {}).update(
                {k: round(float(v), 4) for k, v in value.items()}
            )
        elif key == "momentum_v2_weights" and isinstance(value, dict):
            cfg.setdefault("momentum_v2", {}).setdefault("weights", {}).update(
                {k: round(float(v), 4) for k, v in value.items()}
            )
        elif key == "value_pe_weight":
            cfg.setdefault("scoring", {})["value_pe_weight"] = round(float(value), 4)
            cfg["scoring"]["value_pb_weight"] = round(1.0 - float(value), 4)
        elif key in ("take_profit_pct", "trailing_stop_pct", "sell_weak_value_below"):
            cfg.setdefault("sell_rules", {})[key] = round(float(value), 4)
        elif key in ("metric_threshold", "index_pct"):
            cfg[key] = round(float(value), 4)

    return cfg
