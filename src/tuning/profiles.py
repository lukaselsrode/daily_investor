"""
tuning/profiles.py — Robustness and horizon profiles for the Robust Window Scan.

Rather than exposing individual knobs (window_days, n_windows, seed), the UI
presents two high-level selectors that expand into a concrete run matrix:

  robustness profile  — how thorough the scan is (seeds, windows/horizon, samples)
  horizon profile     — which window lengths to test (days)

  expand_run_matrix(robustness, horizon) → list[dict]
  Each dict: {horizon_days, seed, n_windows, weight_samples}
"""
from __future__ import annotations

ROBUSTNESS_PROFILES: dict[str, dict] = {
    "quick": {
        "seeds": [42],
        "windows_per_horizon": 5,
        "weight_samples": 20,
        "description": "Fast sanity check (~seconds per horizon)",
    },
    "standard": {
        "seeds": [7, 42, 99],
        "windows_per_horizon": 10,
        "weight_samples": 40,
        "description": "Normal research (a few minutes)",
    },
    "deep": {
        "seeds": [1, 7, 21, 42, 99],
        "windows_per_horizon": 15,
        "weight_samples": 80,
        "description": "Stronger robustness check (~10 min per run)",
    },
    "exhaustive": {
        "seeds": [1, 3, 7, 21, 42, 69, 99, 123],
        "windows_per_horizon": 25,
        "weight_samples": 150,
        "description": "Overnight research",
    },
}

HORIZON_PROFILES: dict[str, list[int]] = {
    "short":  [30, 60, 90],
    "medium": [90, 120, 180],
    "long":   [180, 252, 365],
    "mixed":  [30, 60, 90, 120, 180, 365],
}

_ROBUSTNESS_LABELS: dict[str, str] = {
    "quick":      "Quick",
    "standard":   "Standard",
    "deep":       "Deep",
    "exhaustive": "Exhaustive",
}

_HORIZON_LABELS: dict[str, str] = {
    "short":  "Short-term",
    "medium": "Medium-term",
    "long":   "Long-term",
    "mixed":  "Mixed",
}


def expand_run_matrix(
    robustness: str,
    horizon: str,
    *,
    custom_horizons: list[int] | None = None,
    custom_seeds: list[int] | None = None,
    windows_override: int | None = None,
) -> list[dict]:
    """
    Expand a (robustness, horizon) profile pair into a list of run-cell dicts.

    Each cell: {horizon_days, seed, n_windows, weight_samples}

    custom_horizons / custom_seeds / windows_override are for the advanced expander —
    they take precedence over the profile defaults when provided.
    """
    if robustness not in ROBUSTNESS_PROFILES:
        raise ValueError(f"Unknown robustness profile: {robustness!r}. "
                         f"Valid: {list(ROBUSTNESS_PROFILES)}")
    if horizon not in HORIZON_PROFILES and custom_horizons is None:
        raise ValueError(f"Unknown horizon profile: {horizon!r}. "
                         f"Valid: {list(HORIZON_PROFILES)}")

    rp = ROBUSTNESS_PROFILES[robustness]
    horizons = custom_horizons if custom_horizons is not None else HORIZON_PROFILES[horizon]
    seeds    = custom_seeds    if custom_seeds    is not None else rp["seeds"]
    n_windows = windows_override if windows_override is not None else rp["windows_per_horizon"]

    return [
        {
            "horizon_days":   h,
            "seed":           s,
            "n_windows":      n_windows,
            "weight_samples": rp["weight_samples"],
        }
        for h in horizons
        for s in seeds
    ]


def total_simulations(run_matrix: list[dict]) -> int:
    """Total number of individual window simulations across all cells."""
    return sum(cell["n_windows"] for cell in run_matrix)


def sims_per_objective_call(run_matrix: list[dict]) -> int:
    """Simulation runs behind ONE objective evaluation (one robust-scan).

    The cost driver is n_windows × weight_samples per cell, NOT the window count
    alone — `total_simulations` undercounts by the weight-sample factor (40× on the
    standard profile), which is how a 2.4M-sim-per-cluster job got launched as if it
    were an overnight run.
    """
    return sum(
        int(cell.get("n_windows", 1)) * max(1, int(cell.get("weight_samples", 1)))
        for cell in run_matrix
    )


def projected_tune_cost(
    run_matrix: list[dict],
    *,
    maxiter: int,
    popsize: int,
    n_params: int = 5,
    n_stages: int = 1,
) -> dict:
    """Rough cost projection for a differential-evolution tune.

    scipy's DE evaluates roughly (maxiter + 1) × popsize × n_params candidates, each
    costing one full robust-scan. Returns per-stage and total sim-run counts so a
    caller can announce the price BEFORE loading data and committing hours.
    """
    per_call = sims_per_objective_call(run_matrix)
    evals = (int(maxiter) + 1) * int(popsize) * max(1, int(n_params))
    per_stage = per_call * evals
    return {
        "sims_per_objective_call": per_call,
        "evals_per_stage": evals,
        "sims_per_stage": per_stage,
        "stages": int(n_stages),
        "sims_total": per_stage * int(n_stages),
    }


def effort_caption(
    robustness: str,
    horizon: str,
    *,
    custom_horizons: list[int] | None = None,
    custom_seeds: list[int] | None = None,
    windows_override: int | None = None,
) -> str:
    """One-line description shown below the profile selectors."""
    try:
        matrix = expand_run_matrix(
            robustness, horizon,
            custom_horizons=custom_horizons,
            custom_seeds=custom_seeds,
            windows_override=windows_override,
        )
    except ValueError:
        return ""
    rp = ROBUSTNESS_PROFILES[robustness]
    n_cells = len(matrix)
    n_sims  = total_simulations(matrix)
    n_seeds = len(set(c["seed"] for c in matrix))
    n_horizons = len(set(c["horizon_days"] for c in matrix))
    return (
        f"{_ROBUSTNESS_LABELS.get(robustness, robustness)} + "
        f"{_HORIZON_LABELS.get(horizon, horizon)}: "
        f"{n_horizons} horizons × {n_seeds} seeds = "
        f"{n_cells} cells × {matrix[0]['n_windows']} windows = "
        f"**{n_sims} simulations** "
        f"({rp['weight_samples']} weight samples each for tuning)"
    )
