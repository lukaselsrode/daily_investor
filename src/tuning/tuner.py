"""
tuning/tuner.py — ParameterTuner.

Wraps tuner.py (the scipy optimizer) with a typed, class-based interface.
All heavy optimization runs delegate to the module-level functions in tuner.py
via `import tuner as _t` so tests can mock them cleanly with patch.object.

Philosophy:
  - Optimizer is narrow: only active (non-frozen) params are varied
  - Frozen parameters are respected from config.tuning.frozen_parameters
  - Bounds come from config.tuning.parameter_bounds
  - Turnover penalty reduces overfitting to high-churn strategies
  - Auto-tune runs both Sharpe + Calmar, averages, validates before writing
"""

from __future__ import annotations

import logging
from typing import Optional

import tuner as _t

from .results import AutoTuneResult, TuneResult

logger = logging.getLogger(__name__)


class ParameterTuner:
    """
    Conservative parameter optimizer backed by scipy differential_evolution.

    Runs in a reduced parameter space defined by config.tuning.frozen_parameters.
    Active parameters: those not listed in frozen_parameters (default: sw_quality,
    sw_momentum, index_pct — 3 degrees of freedom).

    Usage:
        tuner = ParameterTuner()
        result = tuner.tune(n_days=90, objective="sharpe")
        result = tuner.auto_tune(n_days=90, apply=True)
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Read-only parameter space introspection
    # ------------------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        """Full list of parameter names (frozen + active)."""
        return list(_t.PARAM_NAMES)

    @property
    def active_params(self) -> list[str]:
        """Parameter names that are NOT frozen — these are varied by the optimizer."""
        active_idx = set(_t._get_active_indices())
        return [_t.PARAM_NAMES[i] for i in sorted(active_idx)]

    @property
    def frozen_params(self) -> list[str]:
        """Parameter names that are frozen — optimizer holds them at current config values."""
        active_idx = set(_t._get_active_indices())
        return [name for i, name in enumerate(_t.PARAM_NAMES) if i not in active_idx]

    @property
    def effective_bounds(self) -> list[tuple[float, float]]:
        """Return (min, max) bounds for all parameters, with config overrides applied."""
        return _t._effective_bounds()

    def current_params(self):
        """Return the current config as a parameter vector (np.ndarray)."""
        return _t._current_params()

    # ------------------------------------------------------------------
    # Single-objective tune
    # ------------------------------------------------------------------

    def tune(
        self,
        n_days: int,
        objective: str = "sharpe",
        starting_capital: float = 10_000.0,
        mode: Optional[str] = None,
    ) -> TuneResult:
        """
        Run a single-objective optimization over n_days of history.

        Returns TuneResult with best params, simulation result, and metadata.
        Does NOT write config.yaml.
        """
        params, sim = _t.run_tuner(
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

    # ------------------------------------------------------------------
    # Dual-objective auto-tune with validation
    # ------------------------------------------------------------------

    def auto_tune(
        self,
        n_days: int,
        apply: bool = False,
        force_apply: bool = False,
        mode: Optional[str] = None,
        llm_review: bool = False,
        starting_capital: float = 10_000.0,
    ) -> AutoTuneResult:
        """
        Run Sharpe + Calmar optimizations, average, validate, optionally write config.

        apply=True writes config.yaml only when validation gates pass.
        force_apply=True writes regardless of validation (debugging only).

        Returns AutoTuneResult with full attribution and validation status.
        """
        raw = _t.run_auto_tune(
            n_days=n_days,
            starting_capital=starting_capital,
            mode=mode,
            apply=apply,
            force_apply=force_apply,
            llm_review=llm_review,
        )
        # run_auto_tune returns (avg_params, sharpe_result, calmar_result,
        #                        avg_result, sharpe_params, calmar_params)
        avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params = raw

        # Determine whether config was actually written by checking validation state
        from util import BACKTEST_PARAMS
        bp = BACKTEST_PARAMS
        use_val = bp.get("use_out_of_sample_validation", True)

        # Re-run validation check to populate typed fields
        # (tuner already ran it — we replicate the gate logic to get typed output)
        from backtesting.validator import WalkForwardValidator
        from backtest import load_and_precompute, split_price_window
        import numpy as np

        validation_passed = False
        validation_reasons: list[str] = []
        config_written = False

        if use_val:
            try:
                precomp = _t.load_and_precompute(n_days, mode=mode)
                actual_n = precomp.prices.shape[0]
                train_pct = bp.get("train_pct", 0.70)
                train_sl, val_sl = split_price_window(actual_n, train_pct)
                from backtest import run_backtest_report
                report = run_backtest_report(precomp, avg_params, train_sl, val_sl)
                validator = WalkForwardValidator()
                validation_passed, validation_reasons = validator.validate_report(report, bp)
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

    # ------------------------------------------------------------------
    # Config write helper
    # ------------------------------------------------------------------

    def apply_params(self, params) -> None:
        """
        Write a parameter vector to config.yaml.

        Normalizes score_weights and momentum sub-weights before writing.
        Use with caution — this mutates the live config file.
        """
        _t.apply_config_params(params)
