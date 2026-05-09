"""
strategy/income.py — IncomeScorer: dividend income vs yield-trap filter.

Migrated from source_data._dividend_income_score.
"""

from __future__ import annotations

from .base import ScoreBreakdown, ScorerBase


def compute_income_score(dividend_yield: float) -> tuple[float, bool]:
    """
    Return (income_score, yield_trap_flag).

    yield_trap_flag=True  if yield >= yield_trap_threshold  (suspiciously high)
    income_score=0        for no dividend, or yield trap
    income_score=scaled   min(yield / dividend_threshold, income_score_cap)
    """
    from util import DIVIDEND_THRESHOLD, SCORING_PARAMS

    sp = SCORING_PARAMS
    if not dividend_yield or dividend_yield <= 0:
        return 0.0, False
    if dividend_yield >= sp["yield_trap_threshold"]:
        return 0.0, True
    if dividend_yield >= DIVIDEND_THRESHOLD:
        return min(dividend_yield / DIVIDEND_THRESHOLD, sp["income_score_cap"]), False
    return 0.0, False


class IncomeScorer(ScorerBase):
    """
    Scores a stock on dividend income. Traps are penalized via the quality scorer instead.
    """

    def score(self, features: dict) -> float:
        score, _ = compute_income_score(float(features.get("dividend_yield") or 0))
        return score

    def score_with_trap(self, features: dict) -> tuple[float, bool]:
        """Return (score, yield_trap_flag)."""
        return compute_income_score(float(features.get("dividend_yield") or 0))

    def breakdown(self, symbol: str, features: dict) -> ScoreBreakdown:
        from util import DIVIDEND_THRESHOLD, SCORING_PARAMS

        dy = float(features.get("dividend_yield") or 0)
        score, yield_trap = compute_income_score(dy)
        sp = SCORING_PARAMS
        return ScoreBreakdown(
            symbol=symbol,
            score=score,
            components={"dividend_yield": dy},
            flags={
                "yield_trap": yield_trap,
                "has_dividend": dy >= DIVIDEND_THRESHOLD,
                "above_cap": dy / max(DIVIDEND_THRESHOLD, 1e-9) > sp["income_score_cap"],
            },
        )
