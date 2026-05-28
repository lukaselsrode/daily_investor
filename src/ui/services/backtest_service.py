"""
ui/services/backtest_service.py — Thin orchestration wrapper for backtest execution.

UI components and CLI should call these functions instead of importing
BacktestEngine or random_window_backtest directly.  This keeps the call sites
consistent and makes it easy to add caching or logging in one place.
"""
from __future__ import annotations


def run_single_backtest(
    n_days: int,
    mode: str | None = None,
    params=None,
    save_artifacts: bool = False,
    cluster_tracking: bool = False,
    scope: str = "overall_strategy",
):
    """Run one historical backtest window. Returns BacktestResult."""
    from backtesting.engine import BacktestEngine
    return BacktestEngine().run(
        n_days=n_days,
        mode=mode,
        params=params,
        save_artifacts=save_artifacts,
        cluster_tracking=cluster_tracking,
        scope=scope,
    )


def run_random_windows(
    n_days: int,
    n_windows: int,
    window_days: int,
    mode: str | None = None,
    params=None,
    seed: int = 42,
    progress_callback=None,
    scope: str = "overall_strategy",
):
    """Sample N random windows and aggregate results. Returns RandomWindowSummary."""
    from backtesting.data_loader import load_and_precompute
    from backtesting.random_walk import random_window_backtest
    precomp = load_and_precompute(n_days, mode=mode)
    return random_window_backtest(
        precomp,
        params=params,
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        progress_callback=progress_callback,
        scope=scope,
    )


def list_saved_runs():
    """Return metadata list of saved backtest artifact runs."""
    from backtesting.artifacts import list_saved_runs as _list
    return _list()
