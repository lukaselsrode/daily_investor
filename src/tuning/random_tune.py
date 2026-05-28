"""
tuning/random_tune.py — Random score-weight sampling + ranking by robustness.

Samples N random combinations of (value, quality, income, momentum) score weights
uniformly from the 4-simplex, runs randomized walk-forward backtests for each
combination, and ranks candidates by robust_score.

Public API
----------
sample_weight_simplex(n_samples, n_dims, seed) -> np.ndarray  shape (n_samples, n_dims)
run_random_weight_tune(precomp, base_params, ...) -> RandomTuneResult

Design notes
------------
- Weights are sampled via Dirichlet(alpha=1), which is the uniform distribution
  on the simplex.  Each row sums to 1.0.
- Only the four top-level score weights (params[0:4]) are varied; all other
  parameters stay at base_params values (current config defaults).
- Existing config bounds are respected: samples outside [lo, hi] for a weight
  are clipped and renormalized before evaluation.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.random_walk import RandomWindowSummary, random_window_backtest
from backtesting.types import PrecomputedData

logger = logging.getLogger(__name__)

# Score-weight positions in the 15-element params vector
_WEIGHT_SLICE = slice(0, 4)  # [value, quality, income, momentum]
_WEIGHT_NAMES = ["value", "quality", "income", "momentum"]


# ---------------------------------------------------------------------------
# Simplex sampler
# ---------------------------------------------------------------------------

def sample_weight_simplex(
    n_samples: int,
    n_dims: int = 4,
    seed: int = 42,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Sample n_samples points uniformly from the (n_dims-1)-simplex.

    Uses the Dirichlet(1,...,1) distribution, which is uniform over the simplex.
    When bounds are provided, samples outside [lo, hi] for a dimension are
    clipped and the row is renormalized.  This introduces a small bias toward
    the interior of the feasible region, which is acceptable for tuning.

    Returns:
        Array of shape (n_samples, n_dims) where each row sums to 1.0.
    """
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, (n_samples, n_dims))
    samples = raw / raw.sum(axis=1, keepdims=True)

    if bounds is not None:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        samples = np.clip(samples, lo, hi)
        row_sums = samples.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        samples = samples / row_sums

    return samples


def _weight_bounds_from_config() -> list[tuple[float, float]]:
    """Read score-weight bounds from the tuning section of config."""
    try:
        from util import TUNING_PARAMS
        pb = TUNING_PARAMS.get("parameter_bounds", {})
        defaults = [(0.01, 0.90)] * 4
        paths = [
            "score_weights.value",
            "score_weights.quality",
            "score_weights.income",
            "score_weights.momentum",
        ]
        bounds = []
        for i, path in enumerate(paths):
            b = pb.get(path)
            if b:
                bounds.append((float(b.get("min", defaults[i][0])), float(b.get("max", defaults[i][1]))))
            else:
                bounds.append(defaults[i])
        return bounds
    except Exception:
        return [(0.01, 0.90)] * 4


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WeightCandidate:
    """A single sampled weight combination and its robustness evaluation."""
    sample_id: int
    weights: np.ndarray               # shape (4,) — [value, quality, income, momentum]
    summary: RandomWindowSummary
    robust_score: float
    rank: int = 0

    def weights_dict(self) -> dict[str, float]:
        return {n: float(self.weights[i]) for i, n in enumerate(_WEIGHT_NAMES)}

    def to_row(self) -> dict:
        d = self.weights_dict()
        d.update({
            "rank":             self.rank,
            "robust_score":     round(self.robust_score, 4),
            "median_excess":    round(self.summary.median_excess_return, 4),
            "median_sharpe":    round(self.summary.median_sharpe, 3),
            "median_drawdown":  round(self.summary.median_drawdown, 4),
            "pct_beating":      round(self.summary.pct_beating_benchmark, 3),
            "worst_decile_dd":  round(self.summary.worst_decile_drawdown, 4),
            "std_excess":       round(self.summary.std_excess_return, 4),
            "n_windows":        self.summary.n_windows,
        })
        return d


@dataclass
class RandomTuneResult:
    """Output of run_random_weight_tune — ranked candidates + current-config comparison."""
    n_samples: int
    n_windows: int
    window_days: int
    seed: int
    candidates: list[WeightCandidate] = field(default_factory=list)
    best_candidate: WeightCandidate | None = None
    current_weights: np.ndarray | None = None
    current_summary: RandomWindowSummary | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def best_weights_dict(self) -> dict | None:
        if self.best_candidate is None:
            return None
        return self.best_candidate.weights_dict()

    def to_dataframe(self) -> pd.DataFrame:
        if not self.candidates:
            return pd.DataFrame()
        return pd.DataFrame([c.to_row() for c in self.candidates])

    def best_vs_current_df(self) -> pd.DataFrame | None:
        """Side-by-side comparison of best candidate vs current config."""
        if self.best_candidate is None:
            return None

        rows: list[dict] = []
        best_w   = self.best_candidate.weights_dict()
        curr_w   = {n: float(self.current_weights[i]) for i, n in enumerate(_WEIGHT_NAMES)} \
                   if self.current_weights is not None else {}
        curr_sum = self.current_summary

        metrics = [
            ("value weight",       best_w.get("value", 0),         curr_w.get("value", 0),        "{:.3f}"),
            ("quality weight",     best_w.get("quality", 0),       curr_w.get("quality", 0),      "{:.3f}"),
            ("income weight",      best_w.get("income", 0),        curr_w.get("income", 0),       "{:.3f}"),
            ("momentum weight",    best_w.get("momentum", 0),      curr_w.get("momentum", 0),     "{:.3f}"),
            ("robust_score",       self.best_candidate.robust_score,
                                   curr_sum.robust_score if curr_sum else float("nan"),              "{:+.4f}"),
            ("median excess",      self.best_candidate.summary.median_excess_return,
                                   curr_sum.median_excess_return if curr_sum else float("nan"),      "{:+.1%}"),
            ("median Sharpe",      self.best_candidate.summary.median_sharpe,
                                   curr_sum.median_sharpe if curr_sum else float("nan"),             "{:.3f}"),
            ("median drawdown",    self.best_candidate.summary.median_drawdown,
                                   curr_sum.median_drawdown if curr_sum else float("nan"),           "{:.1%}"),
            ("% beating benchmark",self.best_candidate.summary.pct_beating_benchmark,
                                   curr_sum.pct_beating_benchmark if curr_sum else float("nan"),     "{:.0%}"),
            ("worst-decile DD",    self.best_candidate.summary.worst_decile_drawdown,
                                   curr_sum.worst_decile_drawdown if curr_sum else float("nan"),     "{:.1%}"),
        ]

        rows = []
        for label, best_val, curr_val, fmt in metrics:
            try:
                b_str = fmt.format(best_val)
                c_str = fmt.format(curr_val) if not (isinstance(curr_val, float) and np.isnan(curr_val)) else "—"
            except Exception:
                b_str = str(best_val)
                c_str = str(curr_val)
            rows.append({"metric": label, "best_config": b_str, "current_config": c_str})

        return pd.DataFrame(rows)

    def best_weights_yaml(self) -> str:
        """Formatted YAML snippet for the best score_weights section."""
        if self.best_candidate is None:
            return ""
        w = self.best_candidate.weights_dict()
        return (
            "score_weights:\n"
            f"  value:    {w['value']:.4f}\n"
            f"  quality:  {w['quality']:.4f}\n"
            f"  income:   {w['income']:.4f}\n"
            f"  momentum: {w['momentum']:.4f}\n"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_random_weight_tune(
    precomp: PrecomputedData,
    base_params: np.ndarray | None = None,
    n_samples: int = 40,
    n_windows: int = 15,
    window_days: int = 60,
    seed: int = 42,
    starting_capital: float = 10_000.0,
    weekly_contribution: float = 0.0,
    slippage_bps: float = 10.0,
    rebalance_frequency_days: int = 5,
    respect_config_bounds: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    scope: str = "overall_strategy",
) -> RandomTuneResult:
    """
    Sample n_samples random score-weight combinations, evaluate each on n_windows
    random windows of window_days, and return candidates ranked by robust_score.

    Only params[0:4] (value/quality/income/momentum) are varied.  All other
    parameters remain at base_params values (current config defaults).

    Args:
        precomp:               Precomputed dataset (should have ≥ window_days * 2 rows
                               to allow meaningful window sampling).
        base_params:           Base parameter vector (15 elements). None = current config.
        n_samples:             Number of random weight combos to evaluate.
        n_windows:             Random windows per weight combo.
        window_days:           Trading days per window.
        seed:                  Master random seed.
        starting_capital:      Capital per window.
        weekly_contribution:   Cash per rebalance cycle.
        slippage_bps:          Slippage per trade in bps.
        rebalance_frequency_days: Days between rebalance cycles.
        respect_config_bounds: If True, clip weight samples to bounds from config.
        progress_callback:     Optional callable(sample_id, n_samples) for UI.

    Returns:
        RandomTuneResult with candidates sorted by robust_score (best first).
    """
    from backtesting.simulator import get_default_params

    if base_params is None:
        base_params = get_default_params()

    bounds = _weight_bounds_from_config() if respect_config_bounds else None
    weight_samples = sample_weight_simplex(n_samples, n_dims=4, seed=seed, bounds=bounds)

    # Current config weights (also evaluated as a baseline)
    current_w = base_params[:4].copy()
    s = current_w.sum()
    if s > 1e-9:
        current_w = current_w / s

    warnings: list[str] = []
    n_total = precomp.prices.shape[0]
    if n_total < window_days * 2:
        warnings.append(
            f"Only {n_total} days of data for window_days={window_days}. "
            "Results may be unreliable — consider loading more history."
        )

    candidates: list[WeightCandidate] = []

    for idx, w in enumerate(weight_samples):
        if progress_callback is not None:
            progress_callback(idx, n_samples)

        params_i = base_params.copy()
        params_i[0] = float(w[0])  # value
        params_i[1] = float(w[1])  # quality
        params_i[2] = float(w[2])  # income
        params_i[3] = float(w[3])  # momentum

        try:
            summary = random_window_backtest(
                precomp, params_i,
                n_windows=n_windows,
                window_days=window_days,
                seed=seed + idx + 1,
                starting_capital=starting_capital,
                weekly_contribution=weekly_contribution,
                slippage_bps=slippage_bps,
                rebalance_frequency_days=rebalance_frequency_days,
                scope=scope,
            )
            rank_score = (
                summary.active_robust_score
                if scope == "active_sleeve_compounding" and summary.active_robust_score is not None
                else summary.robust_score
            )
            candidates.append(WeightCandidate(
                sample_id=idx,
                weights=w.copy(),
                summary=summary,
                robust_score=rank_score,
            ))
        except Exception as exc:
            logger.warning("Sample %d failed: %s", idx, exc)
            warnings.append(f"Sample {idx} failed: {exc}")

    # Sort descending by robust_score and assign ranks
    candidates.sort(key=lambda c: c.robust_score, reverse=True)
    for rank, c in enumerate(candidates, 1):
        c.rank = rank

    if not candidates:
        warnings.append("All weight samples failed. Check data and window_days.")

    # Evaluate current config weights as baseline
    current_summary: RandomWindowSummary | None = None
    try:
        current_params = base_params.copy()
        current_params[:4] = current_w
        current_summary = random_window_backtest(
            precomp, current_params,
            n_windows=n_windows,
            window_days=window_days,
            seed=seed + n_samples + 99,
            starting_capital=starting_capital,
            weekly_contribution=weekly_contribution,
            slippage_bps=slippage_bps,
            rebalance_frequency_days=rebalance_frequency_days,
            scope=scope,
        )
    except Exception as exc:
        logger.warning("Current-weights baseline evaluation failed: %s", exc)
        warnings.append(f"Current config evaluation failed: {exc}")

    best = candidates[0] if candidates else None

    return RandomTuneResult(
        n_samples=len(candidates),
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        candidates=candidates,
        best_candidate=best,
        current_weights=current_w,
        current_summary=current_summary,
        warnings=warnings,
    )
