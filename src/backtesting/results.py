"""
backtesting/results.py — BacktestResult and ValidationResult.

BacktestResult: rich report wrapping backtest.BacktestReport for the new layer.
ValidationResult: output of WalkForwardValidator.split_and_validate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backtest import BacktestReport, SimResult


@dataclass
class BacktestResult:
    """Extended result container returned by BacktestEngine."""
    report: BacktestReport
    n_days: int
    mode: str

    @property
    def total_return(self) -> float:
        return self.report.train_result.total_return

    @property
    def excess_return(self) -> float:
        return self.report.excess_return

    @property
    def sharpe(self) -> float:
        return self.report.train_result.sharpe

    @property
    def max_drawdown(self) -> float:
        return self.report.train_result.max_drawdown

    @property
    def passes_basic_validation(self) -> bool:
        return self.sharpe >= 0.25 and self.max_drawdown >= -0.20

    @property
    def universe_size(self) -> int:
        return self.report.n_symbols

    @property
    def benchmark_return(self) -> float:
        return self.report.benchmark_return

    def summary_dict(self) -> dict:
        return {
            "n_days": self.n_days,
            "mode": self.mode,
            "total_return": f"{self.total_return:+.1%}",
            "benchmark_return": f"{self.benchmark_return:+.1%}",
            "excess_return": f"{self.excess_return:+.1%}",
            "sharpe": f"{self.sharpe:.3f}",
            "calmar": f"{self.report.train_result.calmar:.3f}",
            "max_drawdown": f"{self.max_drawdown:.1%}",
            "trades": self.report.train_result.trades_made,
            "universe_size": self.universe_size,
        }

    def __str__(self) -> str:
        d = self.summary_dict()
        return (
            f"BacktestResult({self.n_days}d {self.mode}): "
            f"ret={d['total_return']} vs {d['benchmark_return']} "
            f"sharpe={d['sharpe']} calmar={d['calmar']} dd={d['max_drawdown']} "
            f"trades={d['trades']}"
        )


@dataclass
class ValidationResult:
    """Output of WalkForwardValidator.split_and_validate()."""
    passed: bool
    reasons: list[str]
    report: BacktestReport
    train_slice: slice
    val_slice: Optional[slice]

    @property
    def val_result(self) -> Optional[SimResult]:
        return self.report.validation_result

    @property
    def val_excess_return(self) -> float:
        if self.report.validation_result is None:
            return 0.0
        return self.report.validation_result.total_return - self.report.validation_benchmark_return

    def summary(self) -> str:
        if self.passed:
            return f"PASSED (val excess={self.val_excess_return:+.2%})"
        return f"FAILED: {'; '.join(self.reasons)}"
