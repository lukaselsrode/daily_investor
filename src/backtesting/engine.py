"""
backtesting/engine.py — BacktestEngine.

High-level entry point for all simulation runs. Orchestrates:
  load_and_precompute  →  split_price_window  →  run_backtest_report

BacktestEngine is intentionally thin — all heavy numerics live in backtest.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .data_loader import load_and_precompute, split_price_window
from .results import BacktestResult, ValidationResult
from .simulator import run_backtest_report, run_simulation
from .types import BacktestReport, PrecomputedData, SimResult
from .validator import WalkForwardValidator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Orchestrates data loading, window splitting, simulation, and validation.

    Quick single-run:
        engine = BacktestEngine()
        result = engine.run(n_days=90)

    Walk-forward with gates:
        result = engine.run_walk_forward(n_days=90, params=my_params)
        if result.passed:
            write_config(result.report)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config
        self._validator = WalkForwardValidator(config)

    # ------------------------------------------------------------------
    # Low-level: single simulation over a precomputed dataset
    # ------------------------------------------------------------------

    def simulate(
        self,
        precomp: PrecomputedData,
        params: Optional["np.ndarray"] = None,
        **kwargs,
    ) -> SimResult:
        """Run one simulation pass. Returns a raw SimResult."""
        return run_simulation(precomp, params, **kwargs)

    # ------------------------------------------------------------------
    # Mid-level: train + optional validation report
    # ------------------------------------------------------------------

    def run_report(
        self,
        precomp: PrecomputedData,
        params: "np.ndarray",
        train_slice: slice,
        val_slice: Optional[slice] = None,
    ) -> BacktestReport:
        """
        Run train (and optionally validation) simulation on an existing
        precomp and return a BacktestReport.
        """
        return run_backtest_report(precomp, params, train_slice, val_slice)

    # ------------------------------------------------------------------
    # High-level: load data + run report in one call
    # ------------------------------------------------------------------

    def run(
        self,
        n_days: int = 90,
        params: Optional["np.ndarray"] = None,
        mode: Optional[str] = None,
        train_pct: float = 0.70,
        with_validation: bool = True,
    ) -> BacktestResult:
        """
        Load data, split window, run train + optional validation, return BacktestResult.

        Args:
            n_days:          Total trading days to simulate.
            params:          Parameter vector. None → use current config defaults.
            mode:            Universe selection mode (see backtest.select_backtest_universe).
            train_pct:       Fraction of window used for training (rest = validation).
            with_validation: Whether to run the out-of-sample validation pass.
        """
        precomp  = load_and_precompute(n_days, mode=mode)
        actual_n = precomp.prices.shape[0]

        train_slice, val_slice = split_price_window(actual_n, train_pct)
        effective_val = val_slice if with_validation else None

        report = run_backtest_report(precomp, params, train_slice, effective_val)

        return BacktestResult(
            report=report,
            n_days=actual_n,
            mode=precomp.mode,
        )

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        n_days: int = 90,
        params: Optional["np.ndarray"] = None,
        mode: Optional[str] = None,
        train_pct: float = 0.70,
        backtest_cfg: Optional[dict] = None,
        apply_flag: bool = False,
        force_apply: bool = False,
    ) -> ValidationResult:
        """
        Load data, run walk-forward split, evaluate gates.

        Returns ValidationResult — caller inspects .passed before writing config.
        Use .should_apply() to get the write-or-not decision including the
        apply_flag / force_apply / auto_apply_if_valid logic.
        """
        precomp  = load_and_precompute(n_days, mode=mode)
        actual_n = precomp.prices.shape[0]

        return self._validator.split_and_validate(
            precomp=precomp,
            params=params,
            n_days=actual_n,
            train_pct=train_pct,
            backtest_cfg=backtest_cfg,
        )
