"""
ui/services/validation_service.py — Backtest / validation service for UI.

Wraps BacktestEngine so Streamlit components don't call backtest.py directly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def run_backtest(
    n_days: int = 90,
    mode: Optional[str] = None,
    params: Optional[np.ndarray] = None,
    with_validation: bool = False,
) -> dict:
    """
    Run a backtest and return a plain dict of metrics suitable for display.

    Returns {} if backtest fails (caller should show error message).
    """
    try:
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        result = engine.run(
            n_days=n_days,
            params=params,
            mode=mode,
            with_validation=with_validation,
        )
        rpt = result.report
        train = rpt.train
        return {
            "total_return":     getattr(train, "total_return", None),
            "benchmark_return": getattr(train, "benchmark_return", None),
            "excess_return":    getattr(train, "excess_return", None),
            "sharpe":           getattr(train, "sharpe", None),
            "calmar":           getattr(train, "calmar", None),
            "max_drawdown":     getattr(train, "max_drawdown", None),
            "trades":           getattr(train, "n_trades", None),
            "n_days":           result.n_days,
            "mode":             result.mode,
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_walk_forward(
    n_days: int = 90,
    mode: Optional[str] = None,
    params: Optional[np.ndarray] = None,
) -> dict:
    """
    Run walk-forward validation and return gate pass/fail result dict.
    """
    try:
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        result = engine.run_walk_forward(n_days=n_days, params=params, mode=mode)
        return {
            "passed":  result.passed,
            "gates":   result.gate_results if hasattr(result, "gate_results") else {},
            "reasons": result.reasons if hasattr(result, "reasons") else [],
        }
    except Exception as exc:
        return {"error": str(exc)}
