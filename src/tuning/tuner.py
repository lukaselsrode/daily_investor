"""
tuning/tuner.py — ParameterTuner and module-level run_tuner / run_auto_tune.

Module-level functions (run_tuner, run_auto_tune, validate_tuned_params,
should_apply_tuned_config) are the canonical implementations. ParameterTuner
is a thin typed wrapper over them.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from backtesting.data_loader import load_and_precompute
from backtesting.simulator import run_backtest_report, run_simulation
from backtesting.types import BacktestReport, SimResult
from util import BACKTEST_PARAMS

from .constants import (
    PARAM_NAMES,
    _current_params,
    _effective_bounds,
    _get_active_indices,
)
from .objective import _run_single
from .reports import (
    apply_config_params,
    build_llm_review_payload,
    merge_llm_recommendation_with_config,
    request_llm_tune_review,
    validate_llm_review_response,
)
from .results import AutoTuneResult, TuneResult

logger = logging.getLogger(__name__)

# Gate outcome of the most recent run_auto_tune() call — (passed, reasons).
# ParameterTuner.auto_tune() consumes this so AutoTuneResult.validation_passed /
# config_written mirror the EXACT predicate the write used (the gates now include
# incumbent-relative and random-window checks that a recompute would have to
# duplicate, including a second multi-day data load).
_LAST_GATE_OUTCOME: tuple[bool, list[str]] | None = None


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def validate_tuned_params(
    report: BacktestReport,
    backtest_cfg: dict,
    incumbent_report: BacktestReport | None = None,
) -> tuple[bool, list[str]]:
    """
    Gate a tuned candidate on its held-out validation window.

    Absolute gates (min excess / max drawdown / min Sharpe) catch outright failures.
    When *incumbent_report* is given (the CURRENT config evaluated on the SAME
    train/val split), two relative gates are added — these exist because a tuned
    config once passed the absolute gates with +0.34% validation excess while the
    incumbent sat at +8.87% with 5x less turnover (overfit signature):

      • tuned validation excess-vs-SPY must beat the incumbent's by at least
        `min_excess_vs_incumbent` (default 0.0 — never apply a config that is
        worse out-of-sample than what we already run);
      • tuned validation turnover must not exceed the incumbent's by more than
        `max_turnover_multiple` (default 2.0 — churn blowups are how overfit
        configs harvest in-sample noise).
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

    if incumbent_report is not None and incumbent_report.validation_result is not None:
        ivr = incumbent_report.validation_result
        inc_excess = ivr.total_return - incumbent_report.validation_benchmark_return

        margin = backtest_cfg.get("min_excess_vs_incumbent", 0.0)
        if val_excess < inc_excess + margin:
            reasons.append(
                f"Validation excess {val_excess:+.2%} does not beat incumbent config's "
                f"{inc_excess:+.2%} (+{margin:.2%} margin) on the same split"
            )

        turn_mult = backtest_cfg.get("max_turnover_multiple", 2.0)
        inc_turn = max(float(ivr.turnover_estimate), 1e-9)
        if float(vr.turnover_estimate) > inc_turn * turn_mult:
            reasons.append(
                f"Validation turnover {vr.turnover_estimate:.2f} > {turn_mult:.1f}x "
                f"incumbent's {ivr.turnover_estimate:.2f}"
            )

    return len(reasons) == 0, reasons


def paired_random_window_gate(
    tuned_params: np.ndarray,
    incumbent_params: np.ndarray,
    backtest_cfg: dict,
    mode: str | None = None,
    scope: str = "overall_strategy",
    precomp=None,
) -> tuple[bool, list[str], dict]:
    """
    Reproducibility gate: tuned vs incumbent params on the SAME random sub-windows
    of a longer history than the tune saw (paired comparison — shared seed ⇒ shared
    windows). A single train/val split can be won by luck; a paired multi-window
    sweep on unseen history cannot.

    Pass requires (config under backtest.random_window_gate, defaults shown):
      • enabled: true
      • paired per-window win rate ≥ `min_win_rate` (0.5)
      • tuned median excess-vs-SPY > incumbent median excess  (always on)
      • tuned robust_score > incumbent robust_score           (always on)
        (robust_score is excess-dominant and already penalizes turnover/std/drawdown)

    Returns (passed, reasons, stats). `precomp` may be injected for tests; otherwise
    `history_days` (default 730) of data are loaded fresh so windows cover history
    the optimizer never trained on.
    """
    gw = backtest_cfg.get("random_window_gate", {}) or {}
    if not gw.get("enabled", True):
        return True, [], {"skipped": True}

    from backtesting.random_walk import random_window_backtest

    history_days = int(gw.get("history_days", 730))
    n_windows    = int(gw.get("n_windows", 12))
    window_days  = int(gw.get("window_days", 120))
    seed         = int(gw.get("seed", 42))
    min_win_rate = float(gw.get("min_win_rate", 0.5))

    if precomp is None:
        precomp = load_and_precompute(history_days, mode=mode)

    common = dict(
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        slippage_bps=backtest_cfg.get("slippage_bps", 10.0),
        commission_per_trade=backtest_cfg.get("commission_per_trade", 0.0),
        rebalance_frequency_days=backtest_cfg.get("rebalance_frequency_days", 5),
        scope=scope,
    )
    tuned_sum     = random_window_backtest(precomp, tuned_params,     **common)
    incumbent_sum = random_window_backtest(precomp, incumbent_params, **common)

    # Shared seed on the same precomp ⇒ identical windows; pair by window_id.
    inc_by_id = {w.window_id: w for w in incumbent_sum.window_results}
    paired = [
        (t, inc_by_id[t.window_id])
        for t in tuned_sum.window_results
        if t.window_id in inc_by_id
    ]
    wins = sum(1 for t, i in paired if t.excess_return > i.excess_return)
    win_rate = wins / len(paired) if paired else 0.0

    reasons: list[str] = []
    if win_rate < min_win_rate:
        reasons.append(
            f"Paired win rate {win_rate:.0%} ({wins}/{len(paired)} windows) < {min_win_rate:.0%}"
        )
    if tuned_sum.median_excess_return <= incumbent_sum.median_excess_return:
        reasons.append(
            f"Median excess-vs-SPY {tuned_sum.median_excess_return:+.2%} <= "
            f"incumbent's {incumbent_sum.median_excess_return:+.2%}"
        )
    if tuned_sum.robust_score <= incumbent_sum.robust_score:
        reasons.append(
            f"Robust score {tuned_sum.robust_score:.4f} <= incumbent's {incumbent_sum.robust_score:.4f}"
        )

    stats = {
        "win_rate": win_rate,
        "wins": wins,
        "n_paired": len(paired),
        "tuned_median_excess": tuned_sum.median_excess_return,
        "incumbent_median_excess": incumbent_sum.median_excess_return,
        "tuned_robust_score": tuned_sum.robust_score,
        "incumbent_robust_score": incumbent_sum.robust_score,
        "tuned_median_turnover": tuned_sum.median_turnover,
        "incumbent_median_turnover": incumbent_sum.median_turnover,
    }
    return len(reasons) == 0, reasons, stats


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
    objective: Literal["sharpe", "calmar", "info_ratio"] = "sharpe",
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
    mode: str | None = None,
    scope: str = "overall_strategy",
    preset: str | None = None,
    regime_scope: str = "all",
) -> tuple[np.ndarray, SimResult]:
    """Optimize a single objective. Returns (best_params, SimResult)."""
    if preset is not None:
        from .presets import validate_preset
        validate_preset(preset)

    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scipy is required. Install: pip install scipy") from exc

    precomp = load_and_precompute(n_days, mode=mode)
    if regime_scope != "all":
        from backtesting.regime_scope import apply_regime_scope
        precomp, regime_meta = apply_regime_scope(precomp, regime_scope)
        n_days = precomp.prices.shape[0]
        print(
            f"Regime scope: {regime_meta['requested']} → {regime_meta['effective']} "
            f"({regime_meta['selected_days']}/{regime_meta['total_days']} days, "
            f"slice={regime_meta['start_day']}:{regime_meta['end_day']})"
        )
    _preset_label = f", preset={preset}" if preset else ""
    print(
        f"\nOptimizing {len(PARAM_NAMES)} parameters over {n_days} trading days "
        f"(objective: {objective}, mode={precomp.mode}, scope={scope}{_preset_label})."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")
    print("This may take several minutes …\n")
    return _run_single(precomp, objective, starting_capital, maxiter, popsize, scope=scope, preset=preset)


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
    scope: str = "overall_strategy",
    preset: str | None = None,
    regime_scope: str = "all",
) -> tuple:
    """
    Run Sharpe + Calmar optimizations, average the results.
    Validates on held-out window; writes config.yaml only when gates pass
    and apply=True or auto_apply_if_valid=True.

    Returns (avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params).
    """
    global _LAST_GATE_OUTCOME
    _LAST_GATE_OUTCOME = None

    if preset is not None:
        from .presets import validate_preset
        validate_preset(preset)

    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scipy is required. Install: pip install scipy") from exc

    bp = BACKTEST_PARAMS
    use_val  = bp.get("use_out_of_sample_validation", True)
    train_pct = bp.get("train_pct", 0.70)

    precomp = load_and_precompute(n_days, mode=mode)
    if regime_scope != "all":
        from backtesting.regime_scope import apply_regime_scope
        precomp, regime_meta = apply_regime_scope(precomp, regime_scope)
        n_days = precomp.prices.shape[0]
        print(
            f"Regime scope: {regime_meta['requested']} → {regime_meta['effective']} "
            f"({regime_meta['selected_days']}/{regime_meta['total_days']} days, "
            f"slice={regime_meta['start_day']}:{regime_meta['end_day']})"
        )

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
    _preset_label = f", preset={preset}" if preset else ""

    print(f"\n[1/2] Optimizing for Sharpe (scope={scope}{_preset_label}) …\n")
    sharpe_params, sharpe_result = _run_single(tune_precomp, "sharpe", starting_capital, maxiter, popsize, scope=scope, preset=preset)

    print(f"\n[2/2] Optimizing for Calmar (scope={scope}{_preset_label}) …\n")
    calmar_params, calmar_result = _run_single(tune_precomp, "calmar", starting_capital, maxiter, popsize, scope=scope, preset=preset)

    avg_params = (sharpe_params + calmar_params) / 2.0
    avg_result = run_simulation(
        tune_precomp, avg_params, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )

    val_slice = val_sl if use_val else None
    # Gate at the SAME scope the parameters were optimized at: an active-sleeve
    # preset tune gated at "overall_strategy" is ~77% benchmark (index_pct frozen),
    # so the tuned sleeve's effect is diluted ~4x and the gates pass/fail on index noise.
    train_report = run_backtest_report(precomp, avg_params, train_sl, val_slice, scope=scope)

    # Incumbent = the config we're currently running, on the SAME split/scope.
    # The tuned candidate must beat it out-of-sample, not merely clear absolute floors.
    incumbent_params = _current_params()
    incumbent_report = run_backtest_report(precomp, incumbent_params, train_sl, val_slice, scope=scope)

    if use_val and train_report.validation_result is not None and incumbent_report.validation_result is not None:
        _t, _i = train_report, incumbent_report
        _t_exc = _t.validation_result.total_return - _t.validation_benchmark_return
        _i_exc = _i.validation_result.total_return - _i.validation_benchmark_return
        print(
            f"\nValidation window (excess vs SPY):  tuned {_t_exc:+.2%}  vs  incumbent {_i_exc:+.2%}"
            f"   |  turnover: tuned {_t.validation_result.turnover_estimate:.2f}"
            f" vs incumbent {_i.validation_result.turnover_estimate:.2f}"
        )

    validation_passed, reasons = validate_tuned_params(train_report, bp, incumbent_report)

    # Reproducibility gate: paired random windows on longer, mostly-unseen history.
    # Only worth the extra load when the split gates passed.
    if validation_passed:
        gw_cfg = bp.get("random_window_gate", {}) or {}
        if gw_cfg.get("enabled", True):
            print(
                f"\nRandom-window gate: {gw_cfg.get('n_windows', 12)}x"
                f"{gw_cfg.get('window_days', 120)}d paired windows over "
                f"{gw_cfg.get('history_days', 730)}d history …"
            )
            rw_passed, rw_reasons, rw_stats = paired_random_window_gate(
                avg_params, incumbent_params, bp, mode=mode, scope=scope,
            )
            if rw_stats and not rw_stats.get("skipped"):
                print(
                    f"  paired win rate {rw_stats['win_rate']:.0%} "
                    f"({rw_stats['wins']}/{rw_stats['n_paired']})  |  median excess "
                    f"tuned {rw_stats['tuned_median_excess']:+.2%} vs incumbent "
                    f"{rw_stats['incumbent_median_excess']:+.2%}  |  robust score "
                    f"tuned {rw_stats['tuned_robust_score']:.4f} vs incumbent "
                    f"{rw_stats['incumbent_robust_score']:.4f}"
                )
            validation_passed = rw_passed
            reasons.extend(rw_reasons)

    if reasons:
        print("\n⚠  Validation gates FAILED:")
        for r in reasons:
            print(f"   • {r}")
    else:
        print("\n✓  Validation gates passed.")

    _LAST_GATE_OUTCOME = (validation_passed, list(reasons))

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

    @staticmethod
    def _active_param_names(scope: str = "overall_strategy", preset: str | None = None) -> list[str]:
        """Names of the slots actually tunable for the given scope/preset."""
        active_idx = set(_get_active_indices(scope, preset=preset))
        return [PARAM_NAMES[i] for i in sorted(active_idx)]

    @property
    def active_params(self) -> list[str]:
        return self._active_param_names()

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
        mode: str | None = None,
        scope: str = "overall_strategy",
        preset: str | None = None,
        regime_scope: str = "all",
    ) -> TuneResult:
        params, sim = run_tuner(
            n_days=n_days,
            objective=objective,  # type: ignore[arg-type]
            starting_capital=starting_capital,
            mode=mode,
            scope=scope,
            preset=preset,
            regime_scope=regime_scope,
        )
        return TuneResult(
            params=params,
            sim=sim,
            objective=objective,
            n_days=n_days,
            active_params=self._active_param_names(scope, preset),
        )

    def auto_tune(
        self,
        n_days: int,
        apply: bool = False,
        force_apply: bool = False,
        mode: str | None = None,
        llm_review: bool = False,
        starting_capital: float = 10_000.0,
        scope: str = "overall_strategy",
        preset: str | None = None,
        regime_scope: str = "all",
    ) -> AutoTuneResult:
        # Clear any stale stash so we only consume the outcome of THIS call
        # (run_auto_tune may be replaced in tests; it resets the stash itself
        # when it actually runs).
        global _LAST_GATE_OUTCOME
        _LAST_GATE_OUTCOME = None
        raw = run_auto_tune(
            n_days=n_days,
            starting_capital=starting_capital,
            mode=mode,
            apply=apply,
            force_apply=force_apply,
            llm_review=llm_review,
            scope=scope,
            preset=preset,
            regime_scope=regime_scope,
        )
        avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params = raw

        bp = BACKTEST_PARAMS
        use_val = bp.get("use_out_of_sample_validation", True)

        validation_passed = False
        validation_reasons: list[str] = []

        if _LAST_GATE_OUTCOME is not None:
            # run_auto_tune just executed and stashed the exact gate outcome it
            # used for the write decision (absolute + incumbent-relative +
            # random-window gates). Consume it instead of recomputing a weaker
            # approximation that could disagree with what was actually written.
            validation_passed, validation_reasons = _LAST_GATE_OUTCOME
        elif use_val:
            try:
                from backtesting.simulator import split_price_window
                precomp = load_and_precompute(n_days, mode=mode)
                if regime_scope != "all":
                    from backtesting.regime_scope import apply_regime_scope
                    precomp, _ = apply_regime_scope(precomp, regime_scope)
                actual_n = precomp.prices.shape[0]
                train_pct = bp.get("train_pct", 0.70)
                train_sl, val_sl = split_price_window(actual_n, train_pct)
                # Re-validate at the scope the run was tuned at (see run_auto_tune).
                report = run_backtest_report(precomp, avg_params, train_sl, val_sl, scope=scope)
                from backtesting.validator import WalkForwardValidator
                validation_passed, validation_reasons = WalkForwardValidator().validate_report(report, bp)
            except Exception as e:
                logger.warning("Could not re-run validation for AutoTuneResult: %s", e)
                validation_passed = False
                validation_reasons = [str(e)]
        else:
            validation_passed = True

        # Mirror the EXACT predicate run_auto_tune uses before writing (it also
        # writes on auto_apply_if_valid, which the old flag ignored).
        config_written = should_apply_tuned_config(apply, validation_passed, bp, force_apply=force_apply)

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
            active_params=self._active_param_names(scope, preset),
        )

    def apply_params(self, params: np.ndarray) -> None:
        apply_config_params(params)
