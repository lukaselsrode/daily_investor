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


def make_objective(
    precomp: PrecomputedData,
    objective: Literal["sharpe", "calmar"] = "sharpe",
    starting_capital: float = 10_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    weekly_contribution: float = 0.0,
    rebalance_frequency_days: int = 5,
    scope: BacktestScope = "overall_strategy",
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
            score_val = result.active_sharpe if objective == "sharpe" else result.active_calmar
            score = float(score_val) if score_val is not None else 0.0
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

        diversity_penalty = max(0.0, 5.0 - result.average_positions) * 0.4

        if call_count[0] % 50 == 0:
            _display = "active_" + objective if scope == "active_sleeve_compounding" else objective
            print(
                f"  [{call_count[0]} evals] {_display}={score:.3f} "
                f"ret={result.total_return:.1%} trades={result.trades_made} "
                f"avg_pos={result.average_positions:.1f}"
            )
        return -score + penalty + turnover_penalty + diversity_penalty

    return _obj


def _run_single(
    precomp: PrecomputedData,
    objective: Literal["sharpe", "calmar"],
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

    obj_fn_full = make_objective(
        precomp, objective, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
        scope=scope,
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

    result = differential_evolution(
        _obj,
        bounds=active_bounds,
        maxiter=maxiter,
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
