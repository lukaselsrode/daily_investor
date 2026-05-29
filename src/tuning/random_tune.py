"""
tuning/random_tune.py — Random parameter sampling + ranking by robustness.

Default behaviour (no preset): samples N random (value/quality/income/momentum)
score-weight combinations from the 4-simplex.

Preset behaviour: determines which params are active via _get_active_indices(), then:
  - If active set == {0,1,2,3} (score weights only): Dirichlet simplex sampling.
  - Otherwise: independent uniform sampling within each param's effective bounds.

This means active_exits, active_factor_internals, etc. correctly vary their own
parameters rather than always varying score weights.

Public API
----------
sample_weight_simplex(n_samples, n_dims, seed) -> np.ndarray  shape (n_samples, n_dims)
run_random_weight_tune(precomp, base_params, ..., preset=None) -> RandomTuneResult
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.random_walk import RandomWindowSummary, random_window_backtest
from backtesting.types import PrecomputedData
from tuning.robust_scan import RobustScanResult

logger = logging.getLogger(__name__)

# Score-weight positions in the 15-element params vector
_WEIGHT_SLICE = slice(0, 4)  # [value, quality, income, momentum]
_WEIGHT_NAMES = ["value", "quality", "income", "momentum"]

# Inverted config-path → param-name display labels (used for YAML export)
_IDX_TO_CONFIG_PATH: dict[int, str] = {
    0:  "score_weights.value",
    1:  "score_weights.quality",
    2:  "score_weights.income",
    3:  "score_weights.momentum",
    4:  "index_pct",
    5:  "metric_threshold",
    6:  "sell_rules.take_profit_pct",
    7:  "sell_rules.sell_weak_value_below",
    8:  "sell_rules.trailing_stop_pct",
    9:  "scoring.value_pe_weight",
    10: "momentum_v2.weights.rs_3m",
    11: "momentum_v2.weights.rs_6m",
    12: "momentum_v2.weights.risk_adj_3m",
    13: "momentum_v2.weights.trend_structure",
    14: "momentum_v2.weights.return_1m",
}
# Append archetype lifecycle slots 15-38 — derived from the same layout as constants.py
try:
    from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX as _C2I
    for _path, _idx in _C2I.items():
        if _path.startswith("archetype_management.") and _idx not in _IDX_TO_CONFIG_PATH:
            _IDX_TO_CONFIG_PATH[_idx] = _path
except Exception:
    pass

_SCORE_WEIGHT_IDXS: frozenset[int] = frozenset({0, 1, 2, 3})


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
    """A single sampled parameter combination and its robustness evaluation."""
    sample_id: int
    active_values: np.ndarray       # shape (n_active,) — values for active params
    active_names: list[str]         # PARAM_NAMES for each active value
    full_params: np.ndarray         # shape (15,) — complete params vector
    summary: RandomWindowSummary    # representative summary for display
    robust_score: float
    rank: int = 0
    scan_result: "RobustScanResult | None" = None  # populated when run_matrix used

    # Backward-compat alias
    @property
    def weights(self) -> np.ndarray:
        return self.active_values

    def weights_dict(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.active_names, self.active_values)}

    def to_row(self) -> dict:
        d = self.weights_dict()
        d.update({
            "rank":            self.rank,
            "robust_score":    round(self.robust_score, 4),
            "median_excess":   round(self.summary.median_excess_return, 4),
            "median_sharpe":   round(self.summary.median_sharpe, 3),
            "median_drawdown": round(self.summary.median_drawdown, 4),
            "pct_beating":     round(self.summary.pct_beating_benchmark, 3),
            "worst_decile_dd": round(self.summary.worst_decile_drawdown, 4),
            "std_excess":      round(self.summary.std_excess_return, 4),
            "n_windows":       self.summary.n_windows,
        })
        return d


@dataclass
class RandomTuneResult:
    """Output of run_random_weight_tune — ranked candidates + current-config comparison."""
    n_samples: int
    n_windows: int
    window_days: int
    seed: int
    active_param_names: list[str] = field(default_factory=list)
    candidates: list[WeightCandidate] = field(default_factory=list)
    best_candidate: WeightCandidate | None = None
    current_weights: np.ndarray | None = None   # active param values for current config
    current_summary: RandomWindowSummary | None = None
    current_scan_result: "RobustScanResult | None" = None  # populated when run_matrix used
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
        """Side-by-side comparison of best candidate vs current config (active params only)."""
        if self.best_candidate is None:
            return None

        _PARAM_DISPLAY: dict[str, str] = {
            "sw_value": "Value", "sw_quality": "Quality",
            "sw_income": "Income", "sw_momentum": "Momentum",
            "index_pct": "Index %", "metric_threshold": "Metric threshold",
            "take_profit_pct": "Take-profit %", "sell_weak_below": "Sell-weak below",
            "trailing_stop": "Trailing stop", "value_pe_weight": "P/E weight",
            "mom_rs3m": "RS 3m", "mom_rs6m": "RS 6m",
            "mom_radj": "Risk-adj 3m", "mom_trend": "Trend", "mom_r1m": "Return 1m",
        }

        best_d   = self.best_candidate.weights_dict()
        curr_arr = self.current_weights
        curr_sum = self.current_summary

        rows = []
        for i, name in enumerate(self.active_param_names):
            label = _PARAM_DISPLAY.get(name, name)
            bval  = best_d.get(name, float("nan"))
            cval  = float(curr_arr[i]) if curr_arr is not None and i < len(curr_arr) else float("nan")

            # Format: percentage for most, small decimal for sub-weights
            if "pct" in name or "stop" in name or "weak" in name:
                fmt = "{:+.1%}"
            elif "weight" in name or name.startswith("sw_") or name.startswith("mom_"):
                fmt = "{:.3f}"
            else:
                fmt = "{:.4f}"

            try:
                b_str = fmt.format(bval)
                c_str = fmt.format(cval) if not (isinstance(cval, float) and np.isnan(cval)) else "—"
            except Exception:
                b_str, c_str = str(bval), str(cval)

            rows.append({"metric": label, "best_config": b_str, "current_config": c_str})

        # Append robustness metrics
        best_sum = self.best_candidate.summary
        robustness_rows = [
            ("robust_score",        f"{self.best_candidate.robust_score:+.4f}",
             f"{curr_sum.robust_score:+.4f}" if curr_sum else "—"),
            ("median excess",       f"{best_sum.median_excess_return:+.1%}",
             f"{curr_sum.median_excess_return:+.1%}" if curr_sum else "—"),
            ("median Sharpe",       f"{best_sum.median_sharpe:.3f}",
             f"{curr_sum.median_sharpe:.3f}" if curr_sum else "—"),
            ("median drawdown",     f"{best_sum.median_drawdown:.1%}",
             f"{curr_sum.median_drawdown:.1%}" if curr_sum else "—"),
            ("% beating benchmark", f"{best_sum.pct_beating_benchmark:.0%}",
             f"{curr_sum.pct_beating_benchmark:.0%}" if curr_sum else "—"),
            ("worst-decile DD",     f"{best_sum.worst_decile_drawdown:.1%}",
             f"{curr_sum.worst_decile_drawdown:.1%}" if curr_sum else "—"),
        ]
        rows.extend({"metric": m, "best_config": b, "current_config": c}
                    for m, b, c in robustness_rows)

        return pd.DataFrame(rows)

    def best_weights_yaml(self) -> str:
        """YAML snippet for the best active parameters, structured as nested config keys."""
        if self.best_candidate is None:
            return ""

        from tuning.constants import PARAM_NAMES
        d = self.best_candidate.weights_dict()

        # Map PARAM_NAMES → config path via IDX_TO_CONFIG_PATH
        name_to_path = {n: _IDX_TO_CONFIG_PATH.get(i, n) for i, n in enumerate(PARAM_NAMES)}

        nested: dict = {}
        for name, val in d.items():
            path  = name_to_path.get(name, name)
            parts = path.split(".")
            cur   = nested
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = round(float(val), 4)

        import yaml
        return yaml.dump(nested, default_flow_style=False, sort_keys=False)


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
    preset: str | None = None,
    run_matrix: list[dict] | None = None,
) -> RandomTuneResult:
    """
    Sample n_samples random parameter combinations, evaluate each on either:
      - a single (n_windows, window_days) call to random_window_backtest  [legacy]
      - the full multi-cell run_matrix via run_robust_scan                 [new]

    When run_matrix is provided, each candidate is scored by
    RobustScanResult.overall_robust_score, and the best_candidate carries
    the full RobustScanResult for downstream heatmap display.

    When preset is None or active set is exactly {0,1,2,3}: Dirichlet simplex.
    Otherwise: independent uniform within effective bounds.
    """
    from backtesting.simulator import get_default_params
    from tuning.constants import PARAM_NAMES, _effective_bounds, _get_active_indices
    from tuning.robust_scan import run_robust_scan

    if base_params is None:
        base_params = get_default_params()

    active_idxs  = _get_active_indices(scope=scope, preset=preset)
    eff_bounds   = _effective_bounds(scope=scope, preset=preset)
    active_names = [PARAM_NAMES[i] for i in active_idxs]
    active_bnds  = [eff_bounds[i] for i in active_idxs]

    use_simplex = (set(active_idxs) == _SCORE_WEIGHT_IDXS)

    if use_simplex:
        simplex_bounds = _weight_bounds_from_config() if respect_config_bounds else None
        raw_samples = sample_weight_simplex(n_samples, n_dims=4, seed=seed, bounds=simplex_bounds)
    else:
        rng = np.random.default_rng(seed)
        raw_samples = np.zeros((n_samples, len(active_idxs)), dtype=float)
        for j, (lo, hi) in enumerate(active_bnds):
            raw_samples[:, j] = rng.uniform(lo, hi, n_samples)

    current_active = base_params[np.array(active_idxs)].copy()

    warnings: list[str] = []
    n_total = precomp.prices.shape[0]
    _max_horizon = max(c["horizon_days"] for c in run_matrix) if run_matrix else window_days
    if n_total < _max_horizon * 2:
        warnings.append(
            f"Only {n_total} days of data vs longest horizon {_max_horizon}d. "
            "Results may be unreliable — consider loading more history."
        )

    candidates: list[WeightCandidate] = []

    def _evaluate(params_i, eval_seed):
        """Evaluate params_i via run_matrix scan or single window_backtest."""
        if run_matrix is not None:
            scan = run_robust_scan(precomp, params_i, run_matrix, scope=scope)
            rep_summary = scan.aggregate_summary()
            if rep_summary is None:
                return None, None
            rank_score = scan.overall_robust_score
            return rep_summary, rank_score, scan
        else:
            summary = random_window_backtest(
                precomp, params_i,
                n_windows=n_windows,
                window_days=window_days,
                seed=eval_seed,
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
            return summary, rank_score, None

    for idx, active_vals in enumerate(raw_samples):
        if progress_callback is not None:
            progress_callback(idx, n_samples)

        params_i = base_params.copy()
        for j, aidx in enumerate(active_idxs):
            params_i[aidx] = float(active_vals[j])

        try:
            result = _evaluate(params_i, seed + idx + 1)
            if result[0] is None:
                continue
            summary, rank_score, scan = result
            candidates.append(WeightCandidate(
                sample_id=idx,
                active_values=active_vals.copy(),
                active_names=active_names,
                full_params=params_i.copy(),
                summary=summary,
                robust_score=rank_score,
                scan_result=scan,
            ))
        except Exception as exc:
            logger.warning("Sample %d failed: %s", idx, exc)
            warnings.append(f"Sample {idx} failed: {exc}")

    candidates.sort(key=lambda c: c.robust_score, reverse=True)
    for rank, c in enumerate(candidates, 1):
        c.rank = rank

    if not candidates:
        warnings.append("All parameter samples failed. Check data and window_days.")

    # Evaluate current config as baseline
    current_summary: RandomWindowSummary | None = None
    current_scan: RobustScanResult | None = None
    try:
        result = _evaluate(base_params.copy(), seed + n_samples + 99)
        if result[0] is not None:
            current_summary, _, current_scan = result
    except Exception as exc:
        logger.warning("Current-config baseline evaluation failed: %s", exc)
        warnings.append(f"Current config evaluation failed: {exc}")

    best = candidates[0] if candidates else None

    return RandomTuneResult(
        n_samples=len(candidates),
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        active_param_names=active_names,
        candidates=candidates,
        best_candidate=best,
        current_weights=current_active,
        current_summary=current_summary,
        current_scan_result=current_scan,
        warnings=warnings,
    )
