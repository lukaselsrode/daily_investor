"""
tuning/results.py — Typed result containers for ParameterTuner and StabilityAnalyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backtesting.types import SimResult


@dataclass
class TuneResult:
    """Output of ParameterTuner.tune() — single-objective run."""
    params: np.ndarray
    sim: SimResult
    objective: str
    n_days: int
    active_params: list[str]

    @property
    def score(self) -> float:
        return self.sim.sharpe if self.objective == "sharpe" else self.sim.calmar

    def summary(self) -> str:
        return (
            f"TuneResult({self.objective}, {self.n_days}d): "
            f"{self.objective}={self.score:+.3f} "
            f"ret={self.sim.total_return:+.1%} "
            f"dd={self.sim.max_drawdown:.1%} "
            f"trades={self.sim.trades_made} "
            f"active={self.active_params}"
        )


@dataclass
class AutoTuneResult:
    """Output of ParameterTuner.auto_tune() — dual-objective averaged run."""
    avg_params: np.ndarray
    sharpe_params: np.ndarray
    calmar_params: np.ndarray
    sharpe_result: SimResult
    calmar_result: SimResult
    avg_result: SimResult
    n_days: int
    validation_passed: bool
    validation_reasons: list[str]
    config_written: bool
    active_params: list[str]

    @property
    def param_spread(self) -> dict[str, float]:
        """Per-parameter absolute difference between Sharpe and Calmar optimized values."""
        from tuning.constants import PARAM_NAMES
        return {
            name: abs(float(self.sharpe_params[i]) - float(self.calmar_params[i]))
            for i, name in enumerate(PARAM_NAMES)
        }

    @property
    def unstable_params(self, threshold: float = 0.05) -> list[str]:
        """Parameter names whose Sharpe/Calmar spread exceeds threshold."""
        return [name for name, spread in self.param_spread.items() if spread > threshold]

    def summary(self) -> str:
        status = "WRITTEN" if self.config_written else ("PASSED" if self.validation_passed else "FAILED")
        return (
            f"AutoTuneResult({self.n_days}d) val={status}: "
            f"avg ret={self.avg_result.total_return:+.1%} "
            f"sharpe={self.avg_result.sharpe:+.3f} "
            f"dd={self.avg_result.max_drawdown:.1%} "
            f"trades={self.avg_result.trades_made}"
        )


@dataclass
class StabilityReport:
    """Output of StabilityAnalyzer.scan()."""
    window_results: list[dict]
    output_paths: dict[str, str]
    stability_df: object = None  # pd.DataFrame when available

    @property
    def n_windows(self) -> int:
        return len(self.window_results)

    @property
    def unstable_params(self) -> list[str]:
        """Return params flagged as unstable across windows (if stability_df available)."""
        if self.stability_df is None:
            return []
        try:
            import pandas as pd
            df = self.stability_df
            if isinstance(df, pd.DataFrame) and "stability_label" in df.columns:
                return list(df[df["stability_label"] == "UNSTABLE"].index)
        except Exception:
            pass
        return []

    def summary(self) -> str:
        unstable = self.unstable_params
        return (
            f"StabilityReport: {self.n_windows} windows, "
            f"{len(unstable)} unstable params"
            + (f": {unstable}" if unstable else "")
        )
