"""
backtesting/validator.py — WalkForwardValidator.

Splits price history into train / validation windows, runs the simulation on
both, and checks tuned parameters against configurable gates before any config
write is allowed.

Responsibilities:
  - Split price window into train (default 70%) / validation (30%)
  - Run run_backtest_report() on the full precomp with both slices
  - Apply validation gates (excess return, max drawdown, Sharpe)
  - Return a typed ValidationResult — caller decides whether to write config
"""

from __future__ import annotations

import logging

import numpy as np

from util import BACKTEST_PARAMS

from .results import ValidationResult
from .simulator import run_backtest_report, split_price_window
from .types import BacktestReport, PrecomputedData

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validation.

    Prevents backtest overfitting by evaluating tuned params on a held-out
    window the optimizer never saw.

    Usage:
        import numpy as np
        from backtesting.data_loader import load_and_precompute

        precomp = load_and_precompute(n_days=90)
        params  = np.array([...])          # tuned param vector
        result  = WalkForwardValidator().split_and_validate(precomp, params, n_days=90)
        if result.passed:
            write_config(result.report)
    """

    def __init__(self, config: dict | None = None) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def split(self, n_days: int, train_pct: float | None = None) -> tuple[slice, slice]:
        """Return (train_slice, val_slice) for a window of n_days."""
        pct = train_pct if train_pct is not None else BACKTEST_PARAMS.get("train_pct", 0.70)
        return split_price_window(n_days, pct)

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def validate_report(
        self,
        report: BacktestReport,
        backtest_cfg: dict | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check a BacktestReport against validation gates.
        Returns (passed, failure_reasons).
        Mirrors tuner.validate_tuned_params() — no dependency on tuner.
        """
        bp = backtest_cfg if backtest_cfg is not None else BACKTEST_PARAMS

        if report.validation_result is None:
            return False, ["No validation window available — cannot validate"]

        vr = report.validation_result
        reasons: list[str] = []

        min_exc = bp.get("min_validation_excess_return", 0.0)
        val_excess = vr.total_return - report.validation_benchmark_return
        if val_excess < min_exc:
            reasons.append(
                f"Validation excess return {val_excess:+.2%} < {min_exc:+.2%}"
            )

        max_dd = bp.get("max_validation_drawdown", -0.20)
        if vr.max_drawdown < max_dd:
            reasons.append(
                f"Validation max drawdown {vr.max_drawdown:.2%} < {max_dd:.2%}"
            )

        min_sh = bp.get("min_validation_sharpe", 0.25)
        if vr.sharpe < min_sh:
            reasons.append(
                f"Validation Sharpe {vr.sharpe:.3f} < {min_sh:.3f}"
            )

        return len(reasons) == 0, reasons

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def split_and_validate(
        self,
        precomp: PrecomputedData,
        params: np.ndarray,
        n_days: int,
        train_pct: float | None = None,
        backtest_cfg: dict | None = None,
        use_validation: bool = True,
    ) -> ValidationResult:
        """
        Split n_days, run train + validation simulations, apply gates.

        Args:
            precomp:        Precomputed price / feature arrays (full window).
            params:         Parameter vector from the optimizer.
            n_days:         Total days in the price window.
            train_pct:      Fraction used for training (default from config).
            backtest_cfg:   Override gate thresholds (default from BACKTEST_PARAMS).
            use_validation: Set False to skip the val window (single-window mode).

        Returns:
            ValidationResult with passed/reasons/report/slices.
        """
        train_slice, val_slice = self.split(n_days, train_pct)
        effective_val = val_slice if use_validation else None

        report = run_backtest_report(precomp, params, train_slice, effective_val)

        passed, reasons = self.validate_report(report, backtest_cfg)

        if reasons:
            logger.warning("Validation gates FAILED: %s", "; ".join(reasons))
        else:
            logger.info("Validation gates passed (val excess=%.2f%%)", (
                (report.validation_result.total_return - report.validation_benchmark_return) * 100
                if report.validation_result else 0.0
            ))

        return ValidationResult(
            passed=passed,
            reasons=reasons,
            report=report,
            train_slice=train_slice,
            val_slice=effective_val,
        )

    def should_apply(
        self,
        result: ValidationResult,
        apply_flag: bool = False,
        force_apply: bool = False,
        backtest_cfg: dict | None = None,
    ) -> bool:
        """
        Return True if validated params should be written to config.

        force_apply bypasses the gate (for manual override / debugging).
        Normal path: validation must pass AND (apply_flag OR auto_apply_if_valid).
        """
        if force_apply:
            return True
        bp = backtest_cfg if backtest_cfg is not None else BACKTEST_PARAMS
        auto = bp.get("auto_apply_if_valid", False)
        return result.passed and (apply_flag or auto)
