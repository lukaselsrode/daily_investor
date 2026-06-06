"""
ui/services/backtest_service.py — Thin orchestration wrapper for backtest execution.

UI components and CLI should call these functions instead of importing
BacktestEngine or random_window_backtest directly.  This keeps the call sites
consistent and makes it easy to add caching or logging in one place.

Caching
-------
load_precomp() caches the last PrecomputedData in st.session_state keyed by
(n_days, mode).  Downloading 308 tickers from yfinance takes ~35 s; this avoids
re-downloading on every button click within the same Streamlit session as long as
the parameters haven't changed.  The cache is intentionally per-session (not
cross-session) because the underlying data files change daily.
"""
from __future__ import annotations


def load_precomp(n_days: int, mode: str | None = None):
    """Load (or return cached) PrecomputedData for the given parameters.

    Caches in st.session_state under key '_precomp_cache' so that subsequent
    runs with the same (n_days, mode) within the same Streamlit session skip the
    yfinance download. NOTE: backtests now span the FULL liquid universe
    (~2700+ symbols, max_symbols=0) so the first load is multi-minute, not seconds —
    the cache makes repeat runs instant. Call this instead of load_and_precompute()
    directly from UI code.
    """
    try:
        import streamlit as st

        from util import BACKTEST_PARAMS
        cache = st.session_state.setdefault("_precomp_cache", {})
        # Include the survivorship-free flag in the cache key so the UI toggle actually takes effect —
        # otherwise flipping it would return the stale precomp loaded under the previous setting.
        key = (n_days, mode, bool(BACKTEST_PARAMS.get("survivorship_free", False)))
        if key not in cache:
            from backtesting.data_loader import load_and_precompute
            cache[key] = load_and_precompute(n_days, mode=mode)
        return cache[key]
    except Exception:
        # Outside Streamlit context (tests, CLI): fall through to direct load.
        from backtesting.data_loader import load_and_precompute
        return load_and_precompute(n_days, mode=mode)


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
    from backtesting.random_walk import random_window_backtest
    precomp = load_precomp(n_days, mode=mode)
    return random_window_backtest(
        precomp,
        params=params,
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        progress_callback=progress_callback,
        scope=scope,
    )


def run_robust_scan(
    n_days: int,
    run_matrix: list[dict],
    mode: str | None = None,
    params=None,
    scope: str = "overall_strategy",
    progress_callback=None,
):
    """Load precomp for n_days then run the multi-cell robust scan. Returns RobustScanResult."""
    from tuning.robust_scan import run_robust_scan as _run
    precomp = load_precomp(n_days, mode=mode)
    return _run(precomp, params=params, run_matrix=run_matrix, scope=scope,
                progress_callback=progress_callback)


def list_saved_runs():
    """Return metadata list of saved backtest artifact runs."""
    from backtesting.artifacts import list_saved_runs as _list
    return _list()
