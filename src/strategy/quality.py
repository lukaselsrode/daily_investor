"""
strategy/quality.py — QualityScorer: earnings, balance-sheet, and liquidity signals.

Migrated from source_data._quality_score.
"""

from __future__ import annotations

from .base import ScoreBreakdown, ScorerBase


def compute_quality_score(
    pe_ratio: float | None,
    pb_ratio: float | None,
    volume: float,
    dividend_yield: float,
) -> float:
    """
    Return quality score from fundamental signals.

    Components (from SCORING_PARAMS):
      +quality_weight_has_positive_pe   if pe_ratio > 0
      +quality_weight_distress_pe       if 0 < pe_ratio < distress_pe_max  (negative weight → penalty)
      +quality_weight_has_positive_pb   if pb_ratio > 0
      +quality_weight_high_volume       if volume >= quality_volume_high
      +quality_weight_low_volume        if volume < quality_volume_low    (negative weight → penalty)
      +quality_weight_yield_trap        if yield >= yield_trap_threshold  (negative weight → penalty)
      +quality_weight_healthy_dividend  if quality_dividend_min <= yield <= quality_dividend_max
    """
    from util import SCORING_PARAMS

    sp = SCORING_PARAMS
    score = 0.0

    if pe_ratio is not None and pe_ratio > 0:
        score += sp["quality_weight_has_positive_pe"]
    if pe_ratio is not None and 0 < pe_ratio < sp["distress_pe_max"]:
        score += sp["quality_weight_distress_pe"]
    if pb_ratio is not None and pb_ratio > 0:
        score += sp["quality_weight_has_positive_pb"]

    if volume >= sp["quality_volume_high"]:
        score += sp["quality_weight_high_volume"]
    elif volume < sp["quality_volume_low"]:
        score += sp["quality_weight_low_volume"]

    if dividend_yield >= sp["yield_trap_threshold"]:
        score += sp["quality_weight_yield_trap"]
    elif sp["quality_dividend_min"] <= dividend_yield <= sp["quality_dividend_max"]:
        score += sp["quality_weight_healthy_dividend"]

    return round(score, 3)


class QualityScorer(ScorerBase):
    """
    Scores a stock on earnings quality, balance sheet, liquidity, and dividend health.

    Score is bounded approximately [-0.7, +1.2] depending on config weights.
    """

    def score(self, features: dict) -> float:
        return compute_quality_score(
            features.get("pe_ratio"),
            features.get("pb_ratio"),
            float(features.get("volume") or 0),
            float(features.get("dividend_yield") or 0),
        )

    def breakdown(self, symbol: str, features: dict) -> ScoreBreakdown:
        from util import SCORING_PARAMS

        pe = features.get("pe_ratio")
        pb = features.get("pb_ratio")
        vol = float(features.get("volume") or 0)
        dy = float(features.get("dividend_yield") or 0)
        sp = SCORING_PARAMS
        score = compute_quality_score(pe, pb, vol, dy)

        components = {}
        if pe is not None and pe > 0:
            components["has_positive_pe"] = sp["quality_weight_has_positive_pe"]
        if pe is not None and 0 < pe < sp["distress_pe_max"]:
            components["distress_pe"] = sp["quality_weight_distress_pe"]
        if pb is not None and pb > 0:
            components["has_positive_pb"] = sp["quality_weight_has_positive_pb"]
        if vol >= sp["quality_volume_high"]:
            components["high_volume"] = sp["quality_weight_high_volume"]
        elif vol < sp["quality_volume_low"]:
            components["low_volume"] = sp["quality_weight_low_volume"]
        if dy >= sp["yield_trap_threshold"]:
            components["yield_trap"] = sp["quality_weight_yield_trap"]
        elif sp["quality_dividend_min"] <= dy <= sp["quality_dividend_max"]:
            components["healthy_dividend"] = sp["quality_weight_healthy_dividend"]

        return ScoreBreakdown(
            symbol=symbol,
            score=score,
            components=components,
            flags={
                "yield_trap": dy >= sp["yield_trap_threshold"],
                "distress_pe": pe is not None and 0 < pe < sp["distress_pe_max"],
                "low_volume": vol < sp["quality_volume_low"],
            },
        )
