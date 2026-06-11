"""
tuning/objective.py — Objective function factory and single-run optimizer.

make_objective():  builds the scipy-minimizable closure
_run_single():     runs differential_evolution for one objective
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from backtesting.simulator import run_simulation
from backtesting.types import BacktestScope, PrecomputedData, SimResult
from util import BACKTEST_PARAMS

from .constants import (
    _MIN_TRADES_HARD,
    _MIN_TRADES_SOFT,
    _MIN_TRADES_SOFT_ACTIVE,
    _current_params,
    _effective_bounds,
    _expand_params,
    _get_active_indices,
)


def de_turnover_penalty(turnover: float, incumbent_turnover: float | None, bp: dict) -> float:
    """
    Incumbent-relative churn penalty for the DE objective.

    The validation gates reject any candidate whose turnover exceeds
    max_turnover_multiple x incumbent — but the optimizer used to spend most of
    its evaluations in exactly that region (the first tournament run produced
    5.5x-churn optima across the board). This penalty makes those regions
    unattractive DURING the search instead of after it:

      multiple <= soft limit            → 0 (low-turnover improvements unaffected)
      soft < multiple <= hard           → linear ramp 0 → weight
      multiple > hard                   → weight x (1 + overshoot)  (steep)

    Returns 0 when disabled, when incumbent-relative mode is off, or when the
    incumbent turnover is unavailable (the legacy trade-count penalty still
    applies in the objective either way).
    """
    if not bp.get("de_turnover_penalty_enabled", True):
        return 0.0
    if not bp.get("de_turnover_penalty_vs_incumbent", True):
        return 0.0
    if incumbent_turnover is None or incumbent_turnover <= 0:
        return 0.0
    soft = float(bp.get("de_turnover_soft_limit_multiple", 1.5))
    hard = float(bp.get("de_turnover_hard_limit_multiple", 2.5))
    weight = float(bp.get("de_turnover_penalty_weight", 1.0))
    multiple = float(turnover) / float(incumbent_turnover)
    if multiple <= soft:
        return 0.0
    if multiple <= hard:
        return weight * (multiple - soft) / max(hard - soft, 1e-9)
    return weight * (1.0 + (multiple - hard))


def make_objective(
    precomp: PrecomputedData,
    objective: Literal["sharpe", "calmar", "info_ratio"] = "sharpe",
    starting_capital: float = 10_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    weekly_contribution: float = 0.0,
    rebalance_frequency_days: int = 5,
    scope: BacktestScope = "overall_strategy",
    incumbent_turnover: float | None = None,
) -> Callable[[np.ndarray], float]:
    """Return the function scipy minimizes (−metric + diversification penalty)."""
    call_count = [0]
    _min_soft = _MIN_TRADES_SOFT_ACTIVE if scope == "active_sleeve_compounding" else _MIN_TRADES_SOFT

    def _obj(params: np.ndarray) -> float:
        call_count[0] += 1
        result = run_simulation(
            precomp, params, starting_capital,
            slippage_bps=slippage_bps,
            commission_per_trade=commission_per_trade,
            weekly_contribution=weekly_contribution,
            rebalance_frequency_days=rebalance_frequency_days,
            scope=scope,
        )

        if result.total_return < -0.95:
            return 10.0

        if scope == "active_sleeve_compounding":
            if objective == "info_ratio":
                score_val = result.active_information_ratio
            elif objective == "calmar":
                score_val = result.active_calmar
            else:
                score_val = result.active_sharpe
            score = float(score_val) if score_val is not None else 0.0
        else:
            # info_ratio (excess vs SPY) is only meaningful for the active stock sleeve;
            # at overall scope the index allocation makes it near-benchmark by construction,
            # so fall back to sharpe. Use --scope active_sleeve_compounding with info_ratio.
            if objective == "info_ratio":
                score = result.sharpe
            else:
                score = result.sharpe if objective == "sharpe" else result.calmar

        if not np.isfinite(score):
            return 10.0

        if result.trades_made < _MIN_TRADES_HARD:
            return 10.0

        penalty = 0.0
        if result.trades_made < _min_soft:
            shortfall = _min_soft - result.trades_made
            penalty = shortfall / _min_soft * 2.0

        bp = BACKTEST_PARAMS
        tp_threshold = bp.get("turnover_penalty_trade_count", 80)
        tp_weight = bp.get("turnover_penalty_weight", 1.0) if bp.get("turnover_penalty_enabled", True) else 0.0
        turnover_penalty = max(0.0, result.trades_made - tp_threshold) / max(tp_threshold, 1) * tp_weight

        # Incumbent-relative churn penalty — steer DE away from regions the
        # turnover gate will reject anyway (see de_turnover_penalty).
        churn_penalty = de_turnover_penalty(
            float(result.turnover_estimate), incumbent_turnover, bp,
        )

        diversity_penalty = max(0.0, 5.0 - result.average_positions) * 0.4

        if call_count[0] % 50 == 0:
            _display = "active_" + objective if scope == "active_sleeve_compounding" else objective
            print(
                f"  [{call_count[0]} evals] {_display}={score:.3f} "
                f"ret={result.total_return:.1%} trades={result.trades_made} "
                f"avg_pos={result.average_positions:.1f} turn={result.turnover_estimate:.2f}"
            )
        return -score + penalty + turnover_penalty + churn_penalty + diversity_penalty

    return _obj


def _run_single(
    precomp: PrecomputedData,
    objective: Literal["sharpe", "calmar", "info_ratio"],
    starting_capital: float,
    maxiter: int,
    popsize: int,
    scope: BacktestScope = "overall_strategy",
    preset: str | None = None,
) -> tuple[np.ndarray, SimResult]:
    from scipy.optimize import differential_evolution

    bp = BACKTEST_PARAMS
    active = _get_active_indices(scope, preset=preset)
    frozen_vals = _current_params()
    eff_bounds = _effective_bounds(scope, preset=preset)
    active_bounds = [eff_bounds[i] for i in active]

    # Incumbent turnover anchor for the incumbent-relative churn penalty —
    # computed ONCE per optimization (one extra sim), not per evaluation.
    incumbent_turnover: float | None = None
    if bp.get("de_turnover_penalty_enabled", True) and bp.get("de_turnover_penalty_vs_incumbent", True):
        try:
            _inc_res = run_simulation(
                precomp, frozen_vals, starting_capital,
                slippage_bps=bp["slippage_bps"],
                commission_per_trade=bp["commission_per_trade"],
                weekly_contribution=bp["weekly_contribution"],
                rebalance_frequency_days=bp["rebalance_frequency_days"],
                scope=scope,
            )
            incumbent_turnover = float(_inc_res.turnover_estimate) or None
            if incumbent_turnover:
                print(
                    f"  churn penalty anchor: incumbent turnover {incumbent_turnover:.2f} "
                    f"(soft {bp.get('de_turnover_soft_limit_multiple', 1.5)}x, "
                    f"hard {bp.get('de_turnover_hard_limit_multiple', 2.5)}x)"
                )
        except Exception as _exc:
            print(f"  churn penalty anchor unavailable ({_exc}) — incumbent-relative penalty off")

    obj_fn_full = make_objective(
        precomp, objective, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
        scope=scope,
        incumbent_turnover=incumbent_turnover,
    )

    def _obj(reduced: np.ndarray) -> float:
        return obj_fn_full(_expand_params(reduced, active, frozen_vals))

    n_active = len(active)
    if n_active == 0:
        best_result = run_simulation(
            precomp, frozen_vals, starting_capital,
            slippage_bps=bp["slippage_bps"],
            commission_per_trade=bp["commission_per_trade"],
            weekly_contribution=bp["weekly_contribution"],
            rebalance_frequency_days=bp["rebalance_frequency_days"],
            scope=scope,
        )
        return frozen_vals, best_result

    # DOF advisory: more active params on the same window = higher overfit risk.
    # Rule of thumb: aim for >= ~10 days of history per tuned parameter. The robust
    # multi-window objective + OOS validation gates are the real guard; this only
    # flags risk (e.g. when composing several presets into a large joint surface).
    _hist_days = int(precomp.prices.shape[0])
    _dof_ratio = _hist_days / max(n_active, 1)
    if n_active >= 12 and _dof_ratio < 10:
        print(
            f"⚠ DOF advisory: {n_active} active params over {_hist_days}d "
            f"(~{_dof_ratio:.0f} d/param). Overfit risk is high — prefer the robust "
            f"multi-window objective + OOS validation, or co-tune fewer params."
        )
    # Scale DE iterations mildly with the search dimension (population already scales
    # as popsize × n_active); aids convergence on larger composed surfaces.
    eff_maxiter = maxiter if n_active <= 8 else min(maxiter + (n_active - 8), maxiter * 2)

    result = differential_evolution(
        _obj,
        bounds=active_bounds,
        maxiter=eff_maxiter,
        popsize=popsize,
        tol=0.02,
        seed=42,
        workers=1,
        disp=False,
        polish=True,
    )
    best_full = _expand_params(result.x, active, frozen_vals)
    best_result = run_simulation(
        precomp, best_full, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
        scope=scope,
    )
    return best_full, best_result
