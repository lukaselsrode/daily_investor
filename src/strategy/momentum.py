"""
strategy/momentum.py — Warm-up momentum scorer (bin-based).

The peer-relative momentum engine in strategy/scoring/ needs ~63 days of price
history to compute the rolling rs_3m / rs_6m / return_1m features. Inside the
simulator's warm-up window — before those features stabilize — fall back to
this bin-based scorer.

Live scoring never uses this module; it's only consumed by
backtesting/simulator.py:_momentum_score_warmup_vec when the rolling multi-factor
momentum inputs are not yet available.
"""

from __future__ import annotations


def compute_warmup_momentum_score(
    position_52w: float | None,
    return_1m: float | None = None,
) -> float:
    """Map 52-week position and 1-month return to a bin-based momentum score.

    Reads bin boundaries / scores / recovery+knife thresholds from
    SCORING_PARAMS["momentum_warmup"].
    """
    from util import SCORING_PARAMS

    if position_52w is None:
        return 0.0

    mw = SCORING_PARAMS["momentum_warmup"]
    bins = mw["position_bin_boundaries"]
    scores = mw["position_bin_scores"]

    base = scores[-1]
    for i, boundary in enumerate(bins):
        if position_52w < boundary:
            base = scores[i]
            break

    cutoff = mw["return_1m_low_position_cutoff"]
    if return_1m is not None and position_52w < cutoff:
        if return_1m >= mw["return_1m_recovery_threshold"]:
            base += mw["return_1m_recovery_bonus"]
        elif return_1m <= mw["return_1m_falling_knife_threshold"]:
            base -= mw["return_1m_falling_knife_penalty"]

    return round(base, 3)
