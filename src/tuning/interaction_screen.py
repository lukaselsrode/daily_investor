"""
tuning/interaction_screen.py — empirical parameter-interaction screener.

Measures, on the full universe with the robust (multi-window) objective, where
JOINT tuning of two preset clusters SYNERGIZES vs CLASHES. For groups A and B:

  marginal(A) = best robust_score tuning A's slots alone
  marginal(B) = best robust_score tuning B's slots alone
  joint(A,B)  = best robust_score tuning the composed `A+B` surface
  interaction = joint − max(marginal(A), marginal(B))
      > 0  synergy → co-tune (the join beats either alone)
      ≤ 0  clash   → the optimizer cannot beat the best marginal; they pull apart

Plus a param-displacement signal: how far each group's joint optimum moved from its
solo optimum (a direct "they compromise" measure, even when the net score is flat).

Reuses the robust objective (robust_scan.run_robust_scan), preset composition
(constants._get_active_indices parses `A+B`), and the profile run-matrix expansion.
This is a research diagnostic (like stability-scan) — the full 5-cluster screen is an
overnight job; use the `quick` profile for a smoke run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CLUSTERS: tuple[str, ...] = (
    "active_buy_gate",
    "active_momentum_engine",
    "active_exit_ladder",
    "active_breadth_turnover",
    "active_quality_stack",
)

# Interaction-score thresholds for the verdict label (robust_score units).
_SYNERGY_EPS = 0.02
_CLASH_EPS = -0.02
_HIGH_DISPLACEMENT = 0.15  # mean normalized move ≥ 15% of bound width = strong compromise


@dataclass
class MarginalResult:
    name: str
    score: float
    params: np.ndarray
    active: list[int]


@dataclass
class PairResult:
    a: str
    b: str
    score_a: float
    score_b: float
    score_joint: float
    interaction: float
    displacement: float
    verdict: str


@dataclass
class InteractionResult:
    marginals: dict[str, MarginalResult] = field(default_factory=dict)
    pairs: list[PairResult] = field(default_factory=list)
    scope: str = "active_sleeve_compounding"

    def pairs_df(self):
        import pandas as pd
        rows = [
            {
                "pair": f"{p.a} × {p.b}",
                "marginal A": round(p.score_a, 4),
                "marginal B": round(p.score_b, 4),
                "joint": round(p.score_joint, 4),
                "interaction": round(p.interaction, 4),
                "param shift": round(p.displacement, 3),
                "verdict": p.verdict,
            }
            for p in self.pairs
        ]
        df = pd.DataFrame(rows)
        return df.sort_values("interaction", ascending=False) if not df.empty else df

    def matrix_df(self):
        """Symmetric cluster×cluster interaction-score matrix (diagonal = marginal)."""
        import pandas as pd
        names = list(self.marginals)
        mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
        for n in names:
            mat.loc[n, n] = round(self.marginals[n].score, 4)
        for p in self.pairs:
            mat.loc[p.a, p.b] = round(p.interaction, 4)
            mat.loc[p.b, p.a] = round(p.interaction, 4)
        return mat


def _verdict(interaction: float, displacement: float) -> str:
    if interaction >= _SYNERGY_EPS:
        return "🟢 synergy"
    if interaction <= _CLASH_EPS:
        return "🔴 clash"
    if displacement >= _HIGH_DISPLACEMENT:
        return "↔ compromise"  # params move a lot but net score is flat
    return "⚪ ~independent"


def _tune_subset(precomp, preset, run_matrix, scope, maxiter, popsize, seed=42, baseline=None,
                 regime_scope: str = "all", checkpoint_cb=None, x0=None, max_seconds=None):
    """
    Optimize a preset's active subset over the robust_scan objective, holding all
    other slots at `baseline` (defaults to current config). Returns a MarginalResult,
    or None if the preset has no active slots. `baseline` lets a caller tune a group
    ON TOP of an evolving best vector (used by the staged coordinate-ascent driver).

    Long-run controls (a standard-profile cluster can exceed a day on its own):
      checkpoint_cb(generation, best_x, best_fun) — called after each DE generation so
        the caller can persist best-so-far. Failures are swallowed: a checkpoint write
        must never kill a tune that is otherwise progressing.
      x0        — warm start (REDUCED vector over `active`), e.g. a resumed best-so-far.
                  Restores the best point found, NOT DE's population/RNG state, so a
                  resumed stage explores differently than an uninterrupted one.
      max_seconds — wall-clock budget. On expiry the callback stops DE cleanly and the
                  best-so-far is returned, rather than the operator needing kill -9
                  (which is what lost 39 hours of work on 2026-08-06).
    """
    import time

    from scipy.optimize import differential_evolution

    from .constants import (
        _current_params,
        _effective_bounds,
        _expand_params,
        _get_active_indices,
    )
    from .robust_scan import run_robust_scan

    active = _get_active_indices(scope, preset=preset)
    if not active:
        return None
    frozen = _current_params() if baseline is None else np.asarray(baseline, dtype=float).copy()
    bounds = _effective_bounds(scope, preset=preset)
    active_bounds = [bounds[i] for i in active]

    def _obj(reduced: np.ndarray) -> float:
        full = _expand_params(reduced, active, frozen)
        try:
            scan = run_robust_scan(precomp, params=full, run_matrix=run_matrix, scope=scope,
                                   regime_scope=regime_scope)
            return -float(scan.overall_robust_score)
        except Exception:
            # differential_evolution MINIMIZES this objective and robust scores are
            # routinely negative (so valid configs score > 0 here). Returning 0.0 made
            # a crashing config look better than any valid negative-score config —
            # return a large penalty so crashes always rank last.
            return 1e6

    de_kwargs: dict = {}
    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=float).ravel()
        if x0_arr.size == len(active_bounds):
            lo = np.array([b[0] for b in active_bounds], dtype=float)
            hi = np.array([b[1] for b in active_bounds], dtype=float)
            de_kwargs["x0"] = np.clip(x0_arr, lo, hi)
        else:
            logger.warning(
                "%s: ignoring warm-start x0 of length %d (expected %d active slots)",
                preset, x0_arr.size, len(active_bounds),
            )

    if checkpoint_cb is not None or max_seconds is not None:
        started = time.time()
        gen = {"n": 0}

        # Parameter NAMES matter: scipy >= 1.14 inspects the signature and uses the
        # modern `intermediate_result` form when that name is present, falling back to
        # the legacy `(xk, convergence=...)` call otherwise. Declaring both keeps this
        # working across versions.
        def _de_callback(intermediate_result=None, convergence=None):
            gen["n"] += 1
            best_x, best_fun = None, None
            if intermediate_result is not None:
                if hasattr(intermediate_result, "x"):          # modern OptimizeResult
                    best_x = np.asarray(intermediate_result.x, dtype=float)
                    best_fun = float(getattr(intermediate_result, "fun", np.nan))
                else:                                          # legacy positional xk
                    best_x = np.asarray(intermediate_result, dtype=float)
            if checkpoint_cb is not None and best_x is not None:
                try:
                    checkpoint_cb(gen["n"], best_x, best_fun)
                except Exception as exc:
                    logger.warning("%s: checkpoint callback failed: %s", preset, exc)
            if max_seconds is not None and (time.time() - started) >= float(max_seconds):
                logger.warning(
                    "%s: wall-clock budget %.0fs reached after %d generations — "
                    "stopping DE cleanly with best-so-far.", preset, float(max_seconds), gen["n"],
                )
                return True   # scipy stops the run and returns the incumbent best
            return False

        de_kwargs["callback"] = _de_callback

    res = differential_evolution(
        _obj, active_bounds, maxiter=maxiter, popsize=popsize,
        tol=0.01, seed=seed, workers=1, polish=True, **de_kwargs,
    )
    return MarginalResult(
        name=preset, score=-float(res.fun),
        params=_expand_params(res.x, active, frozen), active=active,
    )


def _displacement(joint: MarginalResult, marg_a: MarginalResult, marg_b: MarginalResult,
                  scope: str) -> float:
    """Mean normalized move of each group's joint optimum from its solo optimum."""
    from .constants import _effective_bounds
    bounds = _effective_bounds(scope)
    moves: list[float] = []
    for marg in (marg_a, marg_b):
        for i in marg.active:
            lo, hi = bounds[i]
            width = max(hi - lo, 1e-9)
            moves.append(abs(float(joint.params[i]) - float(marg.params[i])) / width)
    return float(np.mean(moves)) if moves else 0.0


def screen_interactions(
    precomp,
    run_matrix: list[dict],
    cluster_names: tuple[str, ...] | list[str] = DEFAULT_CLUSTERS,
    scope: str = "active_sleeve_compounding",
    maxiter: int = 8,
    popsize: int = 6,
    progress_callback=None,
    regime_scope: str = "all",
) -> InteractionResult:
    """Screen all cluster pairs for interaction. progress_callback(done, total)."""
    names = list(cluster_names)
    pairs = list(combinations(names, 2))
    total = len(names) + len(pairs)
    done = 0

    out = InteractionResult(scope=scope)
    for n in names:
        m = _tune_subset(precomp, n, run_matrix, scope, maxiter, popsize,
                         regime_scope=regime_scope)
        if m is not None:
            out.marginals[n] = m
        done += 1
        if progress_callback:
            progress_callback(done, total)

    for a, b in pairs:
        ma, mb = out.marginals.get(a), out.marginals.get(b)
        if ma is None or mb is None:
            done += 1
            if progress_callback:
                progress_callback(done, total)
            continue
        joint = _tune_subset(precomp, f"{a}+{b}", run_matrix, scope, maxiter, popsize,
                             regime_scope=regime_scope)
        if joint is not None:
            interaction = joint.score - max(ma.score, mb.score)
            disp = _displacement(joint, ma, mb, scope)
            out.pairs.append(PairResult(
                a=a, b=b, score_a=ma.score, score_b=mb.score, score_joint=joint.score,
                interaction=interaction, displacement=disp,
                verdict=_verdict(interaction, disp),
            ))
        done += 1
        if progress_callback:
            progress_callback(done, total)

    return out
