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
):
    """Sample random score-weight combinations on a pre-loaded precomp, rank by robust_score."""
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
    )


def run_stability_scan(mode: str | None = None, output_dir: str | None = None):
    """Run parameter stability scan across multiple time windows."""
    from tuning.stability import StabilityAnalyzer
    return StabilityAnalyzer().scan(mode=mode, output_dir=output_dir)
