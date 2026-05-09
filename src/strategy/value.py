"""
strategy/value.py — ValueScorer: PE/PB relative cheapness vs sector thresholds.

Migrated from source_data._evaluate_stock (PE/PB block).
source_data.py imports compute_value_components() from here instead of computing inline.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import ScoreBreakdown, ScorerBase

logger = logging.getLogger(__name__)


def compute_value_components(
    pe_ratio: Optional[float],
    pb_ratio: Optional[float],
    sector: str,
    industry: str,
) -> tuple[float, float, float, bool]:
    """
    Compute (pe_comp, pb_comp, value_score, missing_value_flag).

    PE component:  min(pe_threshold / pe_ratio, MAX_PE_COMPONENT)   if valid
    PB component:  min(pb_threshold / pb_ratio, MAX_PB_COMPONENT)   if valid
    value_score:   value_pe_weight * pe_comp + value_pb_weight * pb_comp
                   (or -0.25 if both PE and PB are missing)
    """
    from util import (
        MAX_PE_COMPONENT,
        MAX_PB_COMPONENT,
        MIN_PE_RATIO,
        MIN_PB_RATIO,
        SCORING_PARAMS,
        get_investment_ratios,
    )

    pe_threshold, pb_threshold = get_investment_ratios(sector, industry)

    pe_comp_raw = 0.0
    if pe_ratio is not None and MIN_PE_RATIO <= pe_ratio < pe_threshold:
        pe_comp_raw = pe_threshold / pe_ratio
    pe_comp = min(pe_comp_raw, MAX_PE_COMPONENT)

    pb_comp_raw = 0.0
    if pb_ratio is not None and MIN_PB_RATIO <= pb_ratio < pb_threshold:
        pb_comp_raw = pb_threshold / pb_ratio
    pb_comp = min(pb_comp_raw, MAX_PB_COMPONENT)

    if (pe_comp_raw > MAX_PE_COMPONENT or pb_comp_raw > MAX_PB_COMPONENT) and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "capped PE/PB component: pe_raw=%.3f pe_capped=%.3f pb_raw=%.3f pb_capped=%.3f",
            pe_comp_raw, pe_comp, pb_comp_raw, pb_comp,
        )

    missing_value_flag = pe_ratio is None and pb_ratio is None
    if missing_value_flag:
        value_score = -0.25
    else:
        value_score = round(
            SCORING_PARAMS["value_pe_weight"] * pe_comp
            + SCORING_PARAMS["value_pb_weight"] * pb_comp,
            3,
        )

    return pe_comp, pb_comp, value_score, missing_value_flag


class ValueScorer(ScorerBase):
    """
    Scores a stock on relative PE/PB cheapness vs sector/industry thresholds.

    Higher value_score = cheaper vs sector norms.
    -0.25 = no valuation data (penalized but not excluded).
    """

    def score(self, features: dict) -> float:
        _, _, value_score, _ = compute_value_components(
            features.get("pe_ratio"),
            features.get("pb_ratio"),
            features.get("sector") or "",
            features.get("industry") or "",
        )
        return value_score

    def breakdown(self, symbol: str, features: dict) -> ScoreBreakdown:
        pe_comp, pb_comp, value_score, missing = compute_value_components(
            features.get("pe_ratio"),
            features.get("pb_ratio"),
            features.get("sector") or "",
            features.get("industry") or "",
        )
        return ScoreBreakdown(
            symbol=symbol,
            score=value_score,
            components={"pe_comp": pe_comp, "pb_comp": pb_comp},
            flags={"missing_value_flag": missing},
        )
