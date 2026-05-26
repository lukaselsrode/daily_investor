"""
tuning/tuner.py — ParameterTuner and module-level run_tuner / run_auto_tune.

Module-level functions (run_tuner, run_auto_tune, validate_tuned_params,
should_apply_tuned_config) are the canonical implementations. ParameterTuner
is a thin typed wrapper over them.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from backtesting.data_loader import load_and_precompute
from backtesting.simulator import run_backtest_report, run_simulation
from backtesting.types import BacktestReport, PrecomputedData, SimResult
from util import BACKTEST_PARAMS

from .constants import (
    PARAM_NAMES,
    _current_params,
    _effective_bounds,
    _get_active_indices,
)
from .objective import _run_single, make_objective
from .reports import (
    _diff_table,
    apply_config_params,
    build_llm_review_payload,
    merge_llm_recommendation_with_config,
    print_config_diff,
    request_llm_tune_review,
    validate_llm_review_response,
)
from .results import AutoTuneResult, TuneResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def validate_tuned_params(
    report: BacktestReport,
    backtest_cfg: dict,
) -> tuple[bool, list[str]]:
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
    if force_apply:
        return True
    return validation_passed and (apply_flag or backtest_cfg.get("auto_apply_if_valid", False))


# ---------------------------------------------------------------------------
# Single-objective public entry point
# ---------------------------------------------------------------------------

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
        f"(objective: {objective}, mode={precomp.mode})."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")
    print("This may take several minutes …\n")
    return _run_single(precomp, objective, starting_capital, maxiter, popsize)


# ---------------------------------------------------------------------------
# Dual-objective auto-tune
# ---------------------------------------------------------------------------

def run_auto_tune(
    n_days: int = 90,
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
    mode: str | None = None,
    apply: bool = False,
    force_apply: bool = False,
    llm_review: bool = False,
) -> tuple:
    """
    Run Sharpe + Calmar optimizations, average the results.
    Validates on held-out window; writes config.yaml only when gates pass
    and apply=True or auto_apply_if_valid=True.

    Returns (avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params).
    """
    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError:
        raise RuntimeError("scipy is required. Install: pip install scipy")

    bp = BACKTEST_PARAMS
    use_val  = bp.get("use_out_of_sample_validation", True)
    train_pct = bp.get("train_pct", 0.70)

    precomp = load_and_precompute(n_days, mode=mode)

    from backtesting.simulator import split_price_window
    train_sl, val_sl = split_price_window(n_days, train_pct)

    def _opt_sl(arr):
        return arr[train_sl] if arr is not None else None

    tune_precomp = precomp._replace(
        prices=precomp.prices[train_sl],
        etf_prices=precomp.etf_prices[train_sl],
        benchmark_prices=precomp.benchmark_prices[train_sl],
        position_52w_daily=precomp.position_52w_daily[train_sl],
        return_1m_daily=precomp.return_1m_daily[train_sl],
        bin_indices_daily=precomp.bin_indices_daily[train_sl],
        has_position_52w_daily=precomp.has_position_52w_daily[train_sl],
        ret_5d_daily=_opt_sl(precomp.ret_5d_daily),
        ret_3m_daily=_opt_sl(precomp.ret_3m_daily),
        ret_6m_daily=_opt_sl(precomp.ret_6m_daily),
        rs_3m_daily=_opt_sl(precomp.rs_3m_daily),
        rs_6m_daily=_opt_sl(precomp.rs_6m_daily),
        vol_3m_daily=_opt_sl(precomp.vol_3m_daily),
        above_50dma_daily=_opt_sl(precomp.above_50dma_daily),
        above_200dma_daily=_opt_sl(precomp.above_200dma_daily),
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

    val_slice = val_sl if use_val else None
    train_report = run_backtest_report(precomp, avg_params, train_sl, val_slice)

    validation_passed, reasons = validate_tuned_params(train_report, bp)

    if reasons:
        print("\n⚠  Validation gates FAILED:")
        for r in reasons:
            print(f"   • {r}")
    else:
        print("\n✓  Validation gates passed.")

    use_llm = llm_review or bp.get("llm_review_enabled", False)
    llm_apply = bp.get("llm_review_apply", False)
    final_params = avg_params

    if use_llm:
        print(f"\n[LLM review] Sending candidates to {bp.get('llm_review_model', 'claude-sonnet-4-6')} …")
        try:
            tr = train_report.train_result
            vr = train_report.validation_result
            candidates = [
                {
                    "candidate_id": "sharpe_opt",
                    "alpha_params": dict(zip(PARAM_NAMES, sharpe_params.tolist())),
                    "train_return": sharpe_result.total_return,
                    "train_sharpe": sharpe_result.sharpe,
                    "train_calmar": sharpe_result.calmar,
                    "train_max_drawdown": sharpe_result.max_drawdown,
                    "train_trades": sharpe_result.trades_made,
                    "train_avg_positions": sharpe_result.average_positions,
                    "train_max_positions": sharpe_result.max_positions,
                    "train_avg_cash_pct": sharpe_result.average_cash_pct,
                    "train_turnover": sharpe_result.turnover_estimate,
                    "train_friction_cost": sharpe_result.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                },
                {
                    "candidate_id": "calmar_opt",
                    "alpha_params": dict(zip(PARAM_NAMES, calmar_params.tolist())),
                    "train_return": calmar_result.total_return,
                    "train_sharpe": calmar_result.sharpe,
                    "train_calmar": calmar_result.calmar,
                    "train_max_drawdown": calmar_result.max_drawdown,
                    "train_trades": calmar_result.trades_made,
                    "train_avg_positions": calmar_result.average_positions,
                    "train_max_positions": calmar_result.max_positions,
                    "train_avg_cash_pct": calmar_result.average_cash_pct,
                    "train_turnover": calmar_result.turnover_estimate,
                    "train_friction_cost": calmar_result.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                },
                {
                    "candidate_id": "avg",
                    "alpha_params": dict(zip(PARAM_NAMES, avg_params.tolist())),
                    "train_return": tr.total_return,
                    "train_sharpe": tr.sharpe,
                    "train_calmar": tr.calmar,
                    "train_max_drawdown": tr.max_drawdown,
                    "train_trades": tr.trades_made,
                    "train_avg_positions": tr.average_positions,
                    "train_max_positions": tr.max_positions,
                    "train_avg_cash_pct": tr.average_cash_pct,
                    "train_turnover": tr.turnover_estimate,
                    "train_friction_cost": tr.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                },
            ]
            payload = build_llm_review_payload(
                candidates,
                mode=precomp.mode,
                universe_selection=precomp.universe_selection,
                benchmark_symbol=precomp.benchmark_symbol,
                validation_cfg=bp,
            )
            response = request_llm_tune_review(payload)
            valid, errors = validate_llm_review_response(response, candidates)
            if not valid:
                print(f"[LLM review] Response invalid — ignoring: {errors}")
            else:
                print(
                    f"[LLM review] Recommended: {response['recommended_candidate_id']}  "
                    f"confidence={response.get('confidence', '?'):.0%}\n"
                    f"  Rationale: {response.get('rationale', '')}\n"
                )
                if response.get("risk_warnings"):
                    for w in response["risk_warnings"]:
                        print(f"  ⚠ {w}")
                if llm_apply and response.get("apply_candidate_as_is") and response.get("proposed_adjustments"):
                    import yaml as _yaml
                    from util import CONFIG_FILE
                    with open(CONFIG_FILE) as f:
                        cfg = _yaml.safe_load(f)
                    merged = merge_llm_recommendation_with_config(cfg, response)
                    with open(CONFIG_FILE, "w") as f:
                        _yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
                    print("[LLM review] Adjustments applied to config.yaml")
        except Exception as e:
            print(f"[LLM review] Failed ({e}) — continuing without LLM input")

    if should_apply_tuned_config(apply, validation_passed, bp, force_apply=force_apply):
        apply_config_params(final_params)
    elif apply and not validation_passed:
        print("Config NOT written: validation gates failed. Use --force-apply to override.")
    else:
        print("Config NOT written (--apply requires validation to pass; use --force-apply to override).")

    return avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params


# ---------------------------------------------------------------------------
# ParameterTuner — typed wrapper
# ---------------------------------------------------------------------------

class ParameterTuner:
    """
    Conservative parameter optimizer backed by scipy differential_evolution.

    Runs in a reduced parameter space defined by config.tuning.frozen_parameters.
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    @property
    def param_names(self) -> list[str]:
        return list(PARAM_NAMES)

    @property
    def active_params(self) -> list[str]:
        active_idx = set(_get_active_indices())
        return [PARAM_NAMES[i] for i in sorted(active_idx)]

    @property
    def frozen_params(self) -> list[str]:
        active_idx = set(_get_active_indices())
        return [name for i, name in enumerate(PARAM_NAMES) if i not in active_idx]

    @property
    def effective_bounds(self) -> list[tuple[float, float]]:
        return _effective_bounds()

    def current_params(self) -> np.ndarray:
        return _current_params()

    def tune(
        self,
        n_days: int,
        objective: str = "sharpe",
        starting_capital: float = 10_000.0,
        mode: Optional[str] = None,
    ) -> TuneResult:
        params, sim = run_tuner(
            n_days=n_days,
            objective=objective,
            starting_capital=starting_capital,
            mode=mode,
        )
        return TuneResult(
            params=params,
            sim=sim,
            objective=objective,
            n_days=n_days,
            active_params=self.active_params,
        )

    def auto_tune(
        self,
        n_days: int,
        apply: bool = False,
        force_apply: bool = False,
        mode: Optional[str] = None,
        llm_review: bool = False,
        starting_capital: float = 10_000.0,
    ) -> AutoTuneResult:
        raw = run_auto_tune(
            n_days=n_days,
            starting_capital=starting_capital,
            mode=mode,
            apply=apply,
            force_apply=force_apply,
            llm_review=llm_review,
        )
        avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params = raw

        bp = BACKTEST_PARAMS
        use_val = bp.get("use_out_of_sample_validation", True)

        validation_passed = False
        validation_reasons: list[str] = []

        if use_val:
            try:
                from backtesting.simulator import split_price_window
                precomp = load_and_precompute(n_days, mode=mode)
                actual_n = precomp.prices.shape[0]
                train_pct = bp.get("train_pct", 0.70)
                train_sl, val_sl = split_price_window(actual_n, train_pct)
                report = run_backtest_report(precomp, avg_params, train_sl, val_sl)
                from backtesting.validator import WalkForwardValidator
                validation_passed, validation_reasons = WalkForwardValidator().validate_report(report, bp)
            except Exception as e:
                logger.warning("Could not re-run validation for AutoTuneResult: %s", e)
                validation_passed = False
                validation_reasons = [str(e)]
        else:
            validation_passed = True

        config_written = apply and validation_passed or force_apply

        return AutoTuneResult(
            avg_params=avg_params,
            sharpe_params=sharpe_params,
            calmar_params=calmar_params,
            sharpe_result=sharpe_result,
            calmar_result=calmar_result,
            avg_result=avg_result,
            n_days=n_days,
            validation_passed=validation_passed,
            validation_reasons=validation_reasons,
            config_written=config_written,
            active_params=self.active_params,
        )

    def apply_params(self, params: np.ndarray) -> None:
        apply_config_params(params)
