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

from backtest import PrecomputedData, SimResult, load_and_precompute, run_simulation
from util import (
    CONFIG_FILE,
    INDEX_PCT,
    METRIC_THRESHOLD,
    MOMENTUM_PARAMS,
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
    (0.30, 0.95),   # index_pct
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
) -> callable:
    """Return the function scipy minimizes (−metric + diversification penalty)."""
    call_count = [0]

    def _obj(params: np.ndarray) -> float:
        call_count[0] += 1
        result = run_simulation(precomp, params, starting_capital)

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

    obj_fn = make_objective(precomp, objective, starting_capital)
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
    best_result = run_simulation(precomp, result.x, starting_capital)
    return result.x, best_result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tuner(
    n_days: int = 90,
    objective: Literal["sharpe", "calmar"] = "sharpe",
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
) -> tuple[np.ndarray, SimResult]:
    """Optimize a single objective. Returns (best_params, SimResult)."""
    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError:
        raise RuntimeError("scipy is required. Install: pip install scipy")

    precomp = load_and_precompute(n_days)
    print(
        f"\nOptimizing {len(PARAM_NAMES)} parameters over {n_days} trading days "
        f"(objective: {objective}, min_trades={_MIN_TRADES_HARD})."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")
    print("This may take several minutes …\n")
    return _run_single(precomp, objective, starting_capital, maxiter, popsize)


def run_auto_tune(
    n_days: int = 90,
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
) -> tuple[np.ndarray, SimResult, SimResult, SimResult]:
    """
    Run Sharpe + Calmar optimizations, average the results, write config.yaml.
    Returns (avg_params, sharpe_result, calmar_result, avg_result).
    """
    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError:
        raise RuntimeError("scipy is required. Install: pip install scipy")

    precomp = load_and_precompute(n_days)

    print(
        f"\nAuto-tune: {len(PARAM_NAMES)} params, {n_days} trading days, "
        f"min_trades={_MIN_TRADES_HARD}."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")

    print("\n[1/2] Optimizing for Sharpe …\n")
    sharpe_params, sharpe_result = _run_single(precomp, "sharpe", starting_capital, maxiter, popsize)

    print("\n[2/2] Optimizing for Calmar …\n")
    calmar_params, calmar_result = _run_single(precomp, "calmar", starting_capital, maxiter, popsize)

    avg_params = (sharpe_params + calmar_params) / 2.0
    avg_result = run_simulation(precomp, avg_params, starting_capital)
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

    cfg["index_pct"] = round(float(params[4]), 4)
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
