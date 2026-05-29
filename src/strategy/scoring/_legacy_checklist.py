"""
strategy/scoring/_legacy_checklist.py — Private legacy checklist scorers.

Used ONLY as small-peer-group fallback by the peer-relative engines in
strategy/scoring/{quality,income}.py. The original public functions
(`compute_quality_score`, `compute_income_score`) and their dependencies
(`QualityScorer`, `IncomeScorer`) lived in strategy/{quality,income}.py before
the v3 consolidation — those modules were deleted.

These helpers are private (underscore-prefixed) and are NOT a public API.
"""
from __future__ import annotations


def _checklist_quality(
    pe_ratio: float | None,
    pb_ratio: float | None,
    volume: float,
    dividend_yield: float,
) -> float:
    """Return checklist-style quality score from fundamental signals.

    Components are weighted via SCORING_PARAMS["quality_checklist"]:
      + has_positive_pe   if pe_ratio > 0
      + distress_pe       if 0 < pe_ratio < distress_pe_max  (negative weight)
      + has_positive_pb   if pb_ratio > 0
      + high_volume       if volume >= quality_volume_high
      + low_volume        if volume < quality_volume_low     (negative weight)
      + yield_trap        if yield >= yield_trap_threshold   (negative weight)
      + healthy_dividend  if quality_dividend_min <= yield <= quality_dividend_max
    """
    from util import SCORING_PARAMS

    qc = SCORING_PARAMS["quality_checklist"]
    score = 0.0

    if pe_ratio is not None and pe_ratio > 0:
        score += qc["quality_weight_has_positive_pe"]
    if pe_ratio is not None and 0 < pe_ratio < qc["distress_pe_max"]:
        score += qc["quality_weight_distress_pe"]
    if pb_ratio is not None and pb_ratio > 0:
        score += qc["quality_weight_has_positive_pb"]

    if volume >= qc["quality_volume_high"]:
        score += qc["quality_weight_high_volume"]
    elif volume < qc["quality_volume_low"]:
        score += qc["quality_weight_low_volume"]

    if dividend_yield >= qc["yield_trap_threshold"]:
        score += qc["quality_weight_yield_trap"]
    elif qc["quality_dividend_min"] <= dividend_yield <= qc["quality_dividend_max"]:
        score += qc["quality_weight_healthy_dividend"]

    return round(score, 3)


def _checklist_income(dividend_yield: float) -> tuple[float, bool]:
    """Return (income_score, yield_trap_flag).

    yield_trap_flag=True if yield >= yield_trap_threshold (suspiciously high).
    income_score=0       for no dividend or yield trap.
    income_score=scaled  min(yield / dividend_threshold, income_score_cap)
    """
    from util import DIVIDEND_THRESHOLD, SCORING_PARAMS

    qc = SCORING_PARAMS["quality_checklist"]
    if not dividend_yield or dividend_yield <= 0:
        return 0.0, False
    if dividend_yield >= qc["yield_trap_threshold"]:
        return 0.0, True
    if dividend_yield >= DIVIDEND_THRESHOLD:
        return min(dividend_yield / DIVIDEND_THRESHOLD, qc["income_score_cap"]), False
    return 0.0, False
