"""
strategy/composite.py — CompositeScorer: thin shell around strategy/scoring/.

Single-stock scoring (CompositeScore + compute_composite_score) is preserved as
a 1-row shim around compute_metric. DataFrame scoring delegates to compute_metric.

This module exists only for backward-compat with the (still-used) CompositeScore
dataclass; the actual scoring math lives in strategy/scoring/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .scoring.composite import compute_metric

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


def compute_composite_score(features: dict, score_weights: dict | None = None) -> CompositeScore:
    """Compute the full composite score for one stock's feature dict.

    Builds a 1-row DataFrame and runs compute_metric against it. Peer-relative
    scoring degrades to legacy-checklist fallback automatically when n=1.
    """
    from util import SCORE_WEIGHTS

    sw = score_weights if score_weights is not None else SCORE_WEIGHTS

    row = dict(features)
    df = pd.DataFrame([row])
    compute_metric(df, score_weights=sw)
    r = df.iloc[0]

    quality_score = float(r.get("quality_score", 0.0))
    momentum_score = float(r.get("momentum_score", 0.0))
    pos52 = features.get("position_52w") or 1.0
    bucket = (
        "contrarian_watchlist"
        if quality_score >= 1.0 and momentum_score < 0 and pos52 < 0.25
        else "core_candidate"
    )

    value_score = float(r.get("value_score", 0.0))
    income_score = float(r.get("income_score", 0.0))
    value_metric = float(r.get("value_metric", 0.0))

    return CompositeScore(
        symbol=str(features.get("symbol", "")),
        value_metric=value_metric,
        value_score=value_score,
        quality_score=quality_score,
        income_score=income_score,
        momentum_score=momentum_score,
        pe_comp=0.0,
        pb_comp=0.0,
        yield_trap_flag=bool(r.get("yield_trap_flag", False)),
        missing_value_flag=False,
        strategy_bucket=bucket,
        value_contribution=sw["value"]    * value_score,
        quality_contribution=sw["quality"]  * quality_score,
        income_contribution=sw["income"]   * income_score,
        momentum_contribution=sw["momentum"] * momentum_score,
    )


def recompute_dataframe_metrics(df: pd.DataFrame, score_weights: dict | None = None) -> None:
    """Recompute value_metric for an existing scored DataFrame after score_weights change.

    Does NOT re-run peer ranking — only re-applies score weights to existing component
    columns. Cheap weight-reweighting hook the tuner depends on.
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
    """Thin wrapper preserving the (very old) instance-based API.

    Usage:
        scorer = CompositeScorer()
        result = scorer.score_features(features_dict)
        scorer.score_dataframe(df)   # full universe
    """

    def __init__(self) -> None:
        pass

    def score_features(
        self,
        features: dict,
        score_weights: dict | None = None,
    ) -> CompositeScore:
        return compute_composite_score(features, score_weights)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        score_weights: dict | None = None,
    ) -> None:
        """Score an entire universe DataFrame in-place via the unified peer engine."""
        compute_metric(df, score_weights=score_weights)
