"""
strategy/composite.py — CompositeScorer: applies SCORE_WEIGHTS to sub-scores.

Computes:
  value_metric = (
      sw.value    * value_score
    + sw.quality  * quality_score
    + sw.income   * income_score
    + sw.momentum * momentum_score
  )

Also provides per-factor attribution breakdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .base import ScorerBase
from .income import IncomeScorer, compute_income_score
from .momentum import MomentumEngine, apply_cross_sectional_momentum_v2, compute_momentum_score_v1
from .quality import QualityScorer, compute_quality_score
from .value import ValueScorer, compute_value_components

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    """Final composite score for one stock."""
    symbol: str
    value_metric: float
    value_score: float
    quality_score: float
    income_score: float
    momentum_score: float
    pe_comp: float = 0.0
    pb_comp: float = 0.0
    yield_trap_flag: bool = False
    missing_value_flag: bool = False
    strategy_bucket: str = "core_candidate"
    value_contribution: float = 0.0
    quality_contribution: float = 0.0
    income_contribution: float = 0.0
    momentum_contribution: float = 0.0


def compute_composite_score(features: dict, score_weights: Optional[dict] = None) -> CompositeScore:
    """
    Compute the full composite score for one stock's feature dict.

    score_weights: override dict with keys value/quality/income/momentum.
                   Defaults to SCORE_WEIGHTS from util.
    """
    from util import SCORE_WEIGHTS

    sw = score_weights if score_weights is not None else SCORE_WEIGHTS

    pe_comp, pb_comp, value_score, missing_value_flag = compute_value_components(
        features.get("pe_ratio"),
        features.get("pb_ratio"),
        features.get("sector") or "",
        features.get("industry") or "",
    )
    income_score, yield_trap_flag = compute_income_score(
        float(features.get("dividend_yield") or 0)
    )
    quality_score = compute_quality_score(
        features.get("pe_ratio"),
        features.get("pb_ratio"),
        float(features.get("volume") or 0),
        float(features.get("dividend_yield") or 0),
    )
    momentum_score = compute_momentum_score_v1(
        features.get("position_52w"),
        features.get("return_1m"),
    )

    value_metric = round(
        sw["value"]    * value_score
        + sw["quality"]  * quality_score
        + sw["income"]   * income_score
        + sw["momentum"] * momentum_score,
        3,
    )

    # Contrarian watchlist: high-quality at deep-value entry but negative momentum
    if quality_score >= 1.0 and momentum_score < 0 and (features.get("position_52w") or 1.0) < 0.25:
        bucket = "contrarian_watchlist"
    else:
        bucket = "core_candidate"

    return CompositeScore(
        symbol=str(features.get("symbol", "")),
        value_metric=value_metric,
        value_score=value_score,
        quality_score=quality_score,
        income_score=income_score,
        momentum_score=momentum_score,
        pe_comp=pe_comp,
        pb_comp=pb_comp,
        yield_trap_flag=yield_trap_flag,
        missing_value_flag=missing_value_flag,
        strategy_bucket=bucket,
        value_contribution=sw["value"]    * value_score,
        quality_contribution=sw["quality"]  * quality_score,
        income_contribution=sw["income"]   * income_score,
        momentum_contribution=sw["momentum"] * momentum_score,
    )


def recompute_dataframe_metrics(df: pd.DataFrame, score_weights: Optional[dict] = None) -> None:
    """
    Recompute value_metric for an existing scored DataFrame after score_weights change.
    Called by backtest and tuner after parameter updates.
    Mutates df in-place.
    """
    from util import SCORE_WEIGHTS

    sw = score_weights if score_weights is not None else SCORE_WEIGHTS

    for col in ["value_score", "income_score", "quality_score", "momentum_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["value_metric"] = (
        sw["value"]    * df.get("value_score",    0)
        + sw["quality"]  * df.get("quality_score",  0)
        + sw["income"]   * df.get("income_score",   0)
        + sw["momentum"] * df.get("momentum_score", 0)
    ).round(3)


class CompositeScorer:
    """
    Combines sub-scorer outputs using SCORE_WEIGHTS.

    Usage:
        scorer = CompositeScorer()
        result = scorer.score_features(features_dict)
        scorer.score_dataframe(df)   # full universe, includes cross-sectional momentum v2
    """

    def __init__(self) -> None:
        self._value    = ValueScorer()
        self._quality  = QualityScorer()
        self._income   = IncomeScorer()
        self._momentum = MomentumEngine()

    def score_features(
        self,
        features: dict,
        score_weights: Optional[dict] = None,
    ) -> CompositeScore:
        """Compute composite score for a single stock's feature dict."""
        return compute_composite_score(features, score_weights)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        score_weights: Optional[dict] = None,
        apply_cross_sectional: bool = True,
    ) -> None:
        """
        Score an entire universe DataFrame in-place.

        Steps:
          1. Compute per-stock component scores
          2. Apply cross-sectional momentum v2 normalization
          3. Recompute value_metric with updated momentum_score
        """
        from util import SCORE_WEIGHTS

        sw = score_weights if score_weights is not None else SCORE_WEIGHTS

        # Per-stock component scores (v1 momentum placeholder — overwritten in step 2)
        for col in ["value_score", "income_score", "quality_score", "momentum_score"]:
            if col not in df.columns:
                df[col] = 0.0

        # Cross-sectional momentum v2 (replaces per-stock v1)
        if apply_cross_sectional:
            apply_cross_sectional_momentum_v2(df)

        # Recompute value_metric with final momentum_score
        recompute_dataframe_metrics(df, sw)
