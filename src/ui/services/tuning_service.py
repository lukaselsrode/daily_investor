"""
ui/services/tuning_service.py — Thin orchestration wrapper for tuning workflows.
"""
from __future__ import annotations


def run_weight_tune(
    precomp,
    n_samples: int = 40,
    n_windows: int = 15,
    window_days: int = 60,
    seed: int = 42,
    respect_config_bounds: bool = True,
    progress_callback=None,
    scope: str = "overall_strategy",
    preset: str | None = None,
    run_matrix: list[dict] | None = None,
    regime_scope: str = "all",
):
    """Sample random parameter combinations on a pre-loaded precomp, rank by robust_score.

    When run_matrix is provided, each candidate is evaluated across all (horizon, seed) cells
    via tuning.robust_scan.run_robust_scan rather than a single random_window_backtest call.
    """
    from tuning.random_tune import run_random_weight_tune
    return run_random_weight_tune(
        precomp,
        n_samples=n_samples,
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        respect_config_bounds=respect_config_bounds,
        progress_callback=progress_callback,
        scope=scope,
        preset=preset,
        run_matrix=run_matrix,
        regime_scope=regime_scope,
    )


def run_stability_scan(mode: str | None = None, output_dir: str | None = None):
    """Run parameter stability scan across multiple time windows."""
    from tuning.stability import StabilityAnalyzer
    return StabilityAnalyzer().scan(mode=mode, output_dir=output_dir)
