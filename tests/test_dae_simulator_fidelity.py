"""
tests/test_dae_simulator_fidelity.py — the backtest simulator's vectorized
soft-exit tree must stay decision-equivalent to the live DecisionAdjustmentEngine.

The simulator replicates DecisionAdjustmentEngine._evaluate_soft_exit so that the
exit floors (hard_exit_score_below, positive_momentum / strong_quality /
thesis_intact review floors) are load-bearing in backtests. These tests pin that
equivalence against the live scalar engine so the two implementations cannot drift.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from backtesting.simulator import _dae_soft_exit_full_exit, _thesis_intact_vec
from portfolio.decision_adjustment_engine import DecisionAdjustmentEngine, DecisionInput
from portfolio.exit_analysis import _thesis_intact_score


def _live_floors() -> dict:
    """Pull the floors straight from the live engine's exit config — no hardcoding."""
    ec = DecisionAdjustmentEngine()._ecfg
    return {
        "hard_exit_score_below":          float(ec.get("hard_exit_score_below", 0.20)),
        "thesis_intact_hard_exit_below":  float(ec.get("thesis_intact_hard_exit_below", 0.35)),
        "harvest_profit_threshold":       float(ec.get("harvest_profit_threshold", 0.15)),
        "trim_profit_threshold":          float(ec.get("trim_profit_threshold", 0.08)),
        "positive_pnl_review_floor":      float(ec.get("positive_pnl_review_floor", 0.0)),
        "positive_momentum_review_floor": float(ec.get("positive_momentum_review_floor", 0.10)),
        "strong_quality_review_floor":    float(ec.get("strong_quality_review_floor", 0.70)),
        "thesis_intact_review_floor":     float(ec.get("thesis_intact_review_floor", 0.60)),
        "positive_pnl_exit_downgrade":    bool(ec.get("positive_pnl_exit_downgrade", True)),
        "positive_momentum_exit_downgrade": bool(ec.get("positive_momentum_exit_downgrade", True)),
        "strong_quality_exit_downgrade":  bool(ec.get("strong_quality_exit_downgrade", True)),
    }


_SNW  = [-0.3, 0.0, 0.15, 0.19, 0.25, 0.5, 0.8]
_PNL  = [-0.30, -0.10, -0.06, -0.02, 0.0, 0.05, 0.10, 0.25, 0.60]
_MOM  = [-0.40, -0.25, -0.05, 0.0, 0.10, 0.30]
_QUAL = [-0.40, -0.25, 0.0, 0.40, 0.70, 0.85]
_RANK = [0.1, 0.5, 0.9]


def _grid():
    rows = list(product(_SNW, _PNL, _MOM, _QUAL, _RANK))
    snw  = np.array([r[0] for r in rows], float)
    pnl  = np.array([r[1] for r in rows], float)
    mom  = np.array([r[2] for r in rows], float)
    qual = np.array([r[3] for r in rows], float)
    rank = np.array([r[4] for r in rows], float)
    return rows, snw, pnl, mom, qual, rank


def test_thesis_intact_vec_matches_scalar():
    """Vectorized thesis-intact score matches the live scalar within rounding noise."""
    rows, _snw, pnl, mom, qual, rank = _grid()
    tis_vec = _thesis_intact_vec(qual, mom, pnl, rank)
    for i, (_s, p, m, q, k) in enumerate(rows):
        tis_s = _thesis_intact_score(q, m, p, k)
        # ±0.001 tolerance: both round to 3 dp; float summation order can flip the
        # last digit on exact .xxx5 midpoints. Decision equivalence is asserted below.
        assert abs(tis_s - tis_vec[i]) <= 0.001 + 1e-9


def test_full_exit_decision_matches_live_engine():
    """The simulator's full-exit mask equals (live DAE action == 'EXIT') everywhere."""
    floors = _live_floors()
    rows, snw, pnl, mom, qual, rank = _grid()
    tis_vec = _thesis_intact_vec(qual, mom, pnl, rank)
    cand = np.ones(len(rows), dtype=bool)
    fx_vec = _dae_soft_exit_full_exit(
        cand, snw=snw, pnl=pnl, mom=mom, qual=qual, tis=tis_vec, floors=floors,
    )

    dae = DecisionAdjustmentEngine()
    mismatches = []
    for i, (s, p, m, q, k) in enumerate(rows):
        di = DecisionInput(
            raw_action="EXIT", raw_reason="score below exit threshold",
            exit_type="soft_thesis", pct_change=p, momentum_score=m,
            quality_score=q, score_now=s,
            thesis_intact_score=_thesis_intact_score(q, m, p, k),
            is_premature=False,
        )
        want_full_exit = dae.adjust(di).action == "EXIT"
        if want_full_exit != bool(fx_vec[i]):
            mismatches.append((s, p, m, q, k))
    assert not mismatches, f"{len(mismatches)} full-exit mismatches, e.g. {mismatches[:5]}"


def test_non_candidates_never_full_exit():
    """Stocks outside the candidate set are never force-exited by the soft tree."""
    floors = _live_floors()
    _rows, snw, pnl, mom, qual, rank = _grid()
    tis_vec = _thesis_intact_vec(qual, mom, pnl, rank)
    cand = np.zeros(len(snw), dtype=bool)
    fx = _dae_soft_exit_full_exit(
        cand, snw=snw, pnl=pnl, mom=mom, qual=qual, tis=tis_vec, floors=floors,
    )
    assert not fx.any()
