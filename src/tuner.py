"""
tuner.py — Parameter optimizer for config.yaml using historical simulation.

Uses scipy.optimize.differential_evolution to maximize Sharpe (default)
or Calmar ratio over a back-simulation window.

--tune:       prints suggested diff, no file changes
--auto-tune:  runs both Sharpe and Calmar, averages the result, writes config.yaml
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import yaml

from backtest import (
    BacktestReport,
    PrecomputedData,
    SimResult,
    load_and_precompute,
    run_backtest_report,
    run_simulation,
    split_price_window,
)
from util import (
    BACKTEST_PARAMS,
    CONFIG_FILE,
    INDEX_PCT,
    METRIC_THRESHOLD,
    MOMENTUM_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
)

logger = logging.getLogger(__name__)

# Minimum diversification: below MIN_TRADES_HARD the run is rejected outright;
# between MIN_TRADES_HARD and MIN_TRADES_SOFT a graduated penalty is applied.
# This prevents the optimizer from cherry-picking 2-9 lucky stocks.
_MIN_TRADES_HARD = 20
_MIN_TRADES_SOFT = 40

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "sw_value",          # score_weights.value        (raw, normalized internally)
    "sw_quality",        # score_weights.quality
    "sw_income",         # score_weights.income
    "sw_momentum",       # score_weights.momentum
    "index_pct",         # index_pct
    "metric_threshold",  # metric_threshold
    "take_profit_pct",   # sell_rules.take_profit_pct
    "sell_weak_below",   # sell_rules.sell_weak_value_below
    "trailing_stop",     # sell_rules.trailing_stop_pct
    "value_pe_weight",   # scoring.value_pe_weight
    "mbin_0",            # momentum.position_bin_scores[0]
    "mbin_1",            # momentum.position_bin_scores[1]
    "mbin_2",            # momentum.position_bin_scores[2]
    "mbin_3",            # momentum.position_bin_scores[3]
    "mbin_4",            # momentum.position_bin_scores[4]
]

BOUNDS: list[tuple[float, float]] = [
    (0.05, 0.80),   # sw_value
    (0.05, 0.60),   # sw_quality
    (0.00, 0.40),   # sw_income
    (0.00, 0.40),   # sw_momentum
    (RISK_LIMITS["min_index_pct"], 0.95),   # index_pct — floor protects ETF core
    (0.30, 3.00),   # metric_threshold
    (0.15, 1.00),   # take_profit_pct
    (0.10, 0.90),   # sell_weak_below
    (-0.30, -0.05), # trailing_stop
    (0.30, 0.90),   # value_pe_weight
    (-1.0,  0.5),   # mbin_0
    (-0.5,  0.8),   # mbin_1
    (-0.2,  1.0),   # mbin_2
    ( 0.0,  1.2),   # mbin_3
    (-0.5,  0.8),   # mbin_4
]


def _current_params() -> np.ndarray:
    mbin = list(MOMENTUM_PARAMS["position_bin_scores"])
    while len(mbin) < 5:
        mbin.append(0.0)
    sw = SCORE_WEIGHTS
    return np.array([
        sw["value"], sw["quality"], sw["income"], sw["momentum"],
        INDEX_PCT,
        METRIC_THRESHOLD,
        SELL_RULES["take_profit_pct"],
        SELL_RULES["sell_weak_value_below"],
        SELL_RULES["trailing_stop_pct"],
        SCORING_PARAMS["value_pe_weight"],
        *mbin[:5],
    ])


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(
    precomp: PrecomputedData,
    objective: Literal["sharpe", "calmar"] = "sharpe",
    starting_capital: float = 10_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    weekly_contribution: float = 0.0,
    rebalance_frequency_days: int = 5,
) -> callable:
    """Return the function scipy minimizes (−metric + diversification penalty)."""
    call_count = [0]

    def _obj(params: np.ndarray) -> float:
        call_count[0] += 1
        result = run_simulation(
            precomp, params, starting_capital,
            slippage_bps=slippage_bps,
            commission_per_trade=commission_per_trade,
            weekly_contribution=weekly_contribution,
            rebalance_frequency_days=rebalance_frequency_days,
        )

        if result.total_return < -0.95:
            return 10.0

        score = result.sharpe if objective == "sharpe" else result.calmar
        if not np.isfinite(score):
            return 10.0

        # Hard reject: too few trades = optimizer cherry-picked lucky stocks
        if result.trades_made < _MIN_TRADES_HARD:
            return 10.0

        # Graduated penalty between hard and soft floor
        penalty = 0.0
        if result.trades_made < _MIN_TRADES_SOFT:
            shortfall = _MIN_TRADES_SOFT - result.trades_made
            penalty = shortfall / _MIN_TRADES_SOFT * 2.0

        if call_count[0] % 50 == 0:
            print(
                f"  [{call_count[0]} evals] {objective}={score:.3f} "
                f"ret={result.total_return:.1%} trades={result.trades_made}"
            )
        return -score + penalty

    return _obj


# ---------------------------------------------------------------------------
# Single-objective run (internal helper)
# ---------------------------------------------------------------------------

def _run_single(
    precomp: PrecomputedData,
    objective: Literal["sharpe", "calmar"],
    starting_capital: float,
    maxiter: int,
    popsize: int,
) -> tuple[np.ndarray, SimResult]:
    from scipy.optimize import differential_evolution

    bp = BACKTEST_PARAMS
    obj_fn = make_objective(
        precomp, objective, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )
    result = differential_evolution(
        obj_fn,
        bounds=BOUNDS,
        maxiter=maxiter,
        popsize=popsize,
        tol=0.02,
        seed=42,
        workers=1,
        disp=False,
        polish=True,
    )
    best_result = run_simulation(
        precomp, result.x, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )
    return result.x, best_result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_tuned_params(
    report: BacktestReport,
    backtest_cfg: dict,
) -> tuple[bool, list[str]]:
    """
    Check whether tuned params pass validation gates.
    Returns (passed, list_of_failure_reasons).
    """
    if report.validation_result is None:
        return False, ["No validation window available — cannot validate"]

    vr = report.validation_result
    reasons: list[str] = []

    min_exc = backtest_cfg.get("min_validation_excess_return", 0.0)
    val_excess = vr.total_return - report.validation_benchmark_return
    if val_excess < min_exc:
        reasons.append(f"Validation excess return {val_excess:+.2%} < {min_exc:+.2%}")

    max_dd = backtest_cfg.get("max_validation_drawdown", -0.20)
    if vr.max_drawdown < max_dd:
        reasons.append(f"Validation max drawdown {vr.max_drawdown:.2%} < {max_dd:.2%}")

    min_sh = backtest_cfg.get("min_validation_sharpe", 0.25)
    if vr.sharpe < min_sh:
        reasons.append(f"Validation Sharpe {vr.sharpe:.3f} < {min_sh:.3f}")

    return len(reasons) == 0, reasons


def should_apply_tuned_config(
    apply_flag: bool,
    validation_passed: bool,
    backtest_cfg: dict,
    force_apply: bool = False,
) -> bool:
    """
    Return True if config should be written to disk.

    Normal --apply still requires validation to pass.
    --force-apply skips that requirement (for debugging/manual override).
    """
    if force_apply:
        return True
    return validation_passed and (apply_flag or backtest_cfg.get("auto_apply_if_valid", False))


def run_tuner(
    n_days: int = 90,
    objective: Literal["sharpe", "calmar"] = "sharpe",
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
    mode: str | None = None,
) -> tuple[np.ndarray, SimResult]:
    """Optimize a single objective. Returns (best_params, SimResult)."""
    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError:
        raise RuntimeError("scipy is required. Install: pip install scipy")

    precomp = load_and_precompute(n_days, mode=mode)
    print(
        f"\nOptimizing {len(PARAM_NAMES)} parameters over {n_days} trading days "
        f"(objective: {objective}, min_trades={_MIN_TRADES_HARD}, mode={precomp.mode})."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")
    print("This may take several minutes …\n")
    return _run_single(precomp, objective, starting_capital, maxiter, popsize)


def run_auto_tune(
    n_days: int = 90,
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
    mode: str | None = None,
    apply: bool = False,
    force_apply: bool = False,
) -> tuple[np.ndarray, SimResult, SimResult, SimResult]:
    """
    Run Sharpe + Calmar optimizations, average the results.
    Validates on held-out window and only writes config.yaml when gates pass
    and apply=True or auto_apply_if_valid=True.
    Returns (avg_params, sharpe_result, calmar_result, avg_result).
    """
    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError:
        raise RuntimeError("scipy is required. Install: pip install scipy")

    bp = BACKTEST_PARAMS
    use_val = bp.get("use_out_of_sample_validation", True)
    train_pct = bp.get("train_pct", 0.70)

    precomp = load_and_precompute(n_days, mode=mode)

    # Split window for tune (train) / validate
    train_sl, val_sl = split_price_window(n_days, train_pct)
    tune_precomp = precomp._replace(
        prices=precomp.prices[train_sl],
        etf_prices=precomp.etf_prices[train_sl],
        benchmark_prices=precomp.benchmark_prices[train_sl],
    )
    train_days = tune_precomp.prices.shape[0]

    print(
        f"\nAuto-tune: {len(PARAM_NAMES)} params, {n_days} days "
        f"({train_days} train / {n_days - train_days} val), "
        f"mode={precomp.mode}, bias={precomp.lookahead_bias_level}."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")

    print("\n[1/2] Optimizing for Sharpe …\n")
    sharpe_params, sharpe_result = _run_single(tune_precomp, "sharpe", starting_capital, maxiter, popsize)

    print("\n[2/2] Optimizing for Calmar …\n")
    calmar_params, calmar_result = _run_single(tune_precomp, "calmar", starting_capital, maxiter, popsize)

    avg_params = (sharpe_params + calmar_params) / 2.0
    avg_result = run_simulation(
        tune_precomp, avg_params, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )

    # Build full report with validation
    val_slice = val_sl if use_val else None
    train_report = run_backtest_report(precomp, avg_params, train_sl, val_slice)

    validation_passed, reasons = validate_tuned_params(train_report, bp)

    if reasons:
        print("\n⚠  Validation gates FAILED:")
        for r in reasons:
            print(f"   • {r}")
    else:
        print("\n✓  Validation gates passed.")

    if should_apply_tuned_config(apply, validation_passed, bp, force_apply=force_apply):
        apply_config_params(avg_params)
    elif force_apply:
        pass  # should_apply returned True already
    elif apply and not validation_passed:
        print("Config NOT written: validation gates failed. Use --force-apply to override.")
    else:
        print("Config NOT written (--apply requires validation to pass; use --force-apply to override).")

    return avg_params, sharpe_result, calmar_result, avg_result


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------

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

    cfg.setdefault("momentum", {})
    cfg["momentum"]["position_bin_scores"] = [round(float(v), 4) for v in params[10:15]]

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nconfig.yaml updated: {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _diff_table(
    best_params: np.ndarray,
    label: str = "",
    sharpe_ref: SimResult | None = None,
    calmar_ref: SimResult | None = None,
) -> None:
    cur = _current_params()
    raw_sw = best_params[:4]
    norm_sw = raw_sw / max(raw_sw.sum(), 1e-9)
    cur_sw_norm = cur[:4] / max(cur[:4].sum(), 1e-9)

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

    rows = [
        ("score_weights.value",       cur_sw_norm[0], norm_sw[0]),
        ("score_weights.quality",     cur_sw_norm[1], norm_sw[1]),
        ("score_weights.income",      cur_sw_norm[2], norm_sw[2]),
        ("score_weights.momentum",    cur_sw_norm[3], norm_sw[3]),
        ("index_pct",                 cur[4],          best_params[4]),
        ("metric_threshold",          cur[5],          best_params[5]),
        ("sell_rules.take_profit",    cur[6],          best_params[6]),
        ("sell_rules.sell_weak",      cur[7],          best_params[7]),
        ("sell_rules.trailing_stop",  cur[8],          best_params[8]),
        ("scoring.value_pe_weight",   cur[9],          best_params[9]),
        ("momentum.bin_scores[0]",    cur[10],         best_params[10]),
        ("momentum.bin_scores[1]",    cur[11],         best_params[11]),
        ("momentum.bin_scores[2]",    cur[12],         best_params[12]),
        ("momentum.bin_scores[3]",    cur[13],         best_params[13]),
        ("momentum.bin_scores[4]",    cur[14],         best_params[14]),
    ]

    print("CHANGES  (> 1% relative)")
    print("-" * 64)
    any_change = False
    for lbl, old, new in rows:
        rel = abs(new - old) / max(abs(old), 1e-9)
        if rel > 0.01:
            arrow = "▲" if new > old else "▼"
            print(f"  {lbl:<36}  {old:+.4f}  →  {new:+.4f}  {arrow}")
            any_change = True
    if not any_change:
        print("  (no meaningful changes)")

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
    print("momentum:")
    print(f"  position_bin_scores: {[round(float(v), 4) for v in best_params[10:15]]}")
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


# ---------------------------------------------------------------------------
# LLM review helpers (Phase 12)
# ---------------------------------------------------------------------------

# Only alpha params may be proposed — never safety controls
_LLM_ALLOWED_PARAMS = frozenset([
    "score_weights", "metric_threshold", "index_pct",
    "take_profit_pct", "trailing_stop_pct", "sell_weak_value_below",
    "value_pe_weight", "position_bin_scores",
])
_LLM_FORBIDDEN_PARAMS = frozenset([
    "max_single_position_pct", "max_sector_pct", "max_order_pct_of_cash",
    "min_order_amount", "min_liquidity_volume", "allow_whole_share_fallback",
    "max_whole_share_buys_per_run", "max_whole_share_allocation_multiplier",
    "stop_loss_pct", "weekly_investment",
])


def build_llm_review_payload(
    candidates: list[dict],
    mode: str,
    universe_selection: str,
    benchmark_symbol: str,
    validation_cfg: dict,
) -> dict:
    """
    Build a sanitized payload for LLM review.

    Never includes secrets, API keys, account IDs, live balances, or PII.
    Only sends metrics and alpha parameter candidates.
    """
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
    """
    Send the review payload to the Anthropic API and return the parsed JSON response.
    Raises RuntimeError on API failure or invalid response.
    """
    import json
    import os

    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package required. Install: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    bp = BACKTEST_PARAMS
    model = bp.get("llm_review_model", "claude-sonnet-4-6")

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
    # Strip markdown code fences if present
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
    """
    Validate LLM response structure and safety constraints.
    Returns (valid, list_of_errors).
    """
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
    """
    Merge LLM-proposed alpha param adjustments into a config dict.
    Safety params are never modified regardless of LLM response.
    Returns a new config dict (does not mutate base_config).
    """
    import copy
    cfg = copy.deepcopy(base_config)
    adjustments = llm_response.get("proposed_adjustments", {})

    for key, value in adjustments.items():
        if key in _LLM_FORBIDDEN_PARAMS:
            logger.warning(f"LLM merge: skipping forbidden param {key}")
            continue
        if key == "score_weights" and isinstance(value, dict):
            cfg.setdefault("score_weights", {}).update(
                {k: round(float(v), 4) for k, v in value.items()}
            )
        elif key == "position_bin_scores" and isinstance(value, list):
            cfg.setdefault("momentum", {})["position_bin_scores"] = [
                round(float(v), 4) for v in value
            ]
        elif key == "value_pe_weight":
            cfg.setdefault("scoring", {})["value_pe_weight"] = round(float(value), 4)
            cfg["scoring"]["value_pb_weight"] = round(1.0 - float(value), 4)
        elif key in ("take_profit_pct", "trailing_stop_pct", "sell_weak_value_below"):
            cfg.setdefault("sell_rules", {})[key] = round(float(value), 4)
        elif key in ("metric_threshold", "index_pct"):
            cfg[key] = round(float(value), 4)

    return cfg
