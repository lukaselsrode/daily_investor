"""
strategy/factors/engine.py — FactorEngine: central scoring orchestrator.

Wraps all sub-scorers and provides a unified interface for:
  - Single-stock scoring           → score_single()
  - Full-universe scoring          → score_universe()  (with cross-sectional normalization)
  - Factor exposure profiling      → factor_exposures()
  - Factor correlation diagnostics → factor_correlation_matrix()

All legacy imports (strategy.composite, etc.) continue to work unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from strategy.base import ScoreBreakdown
from strategy.composite import (
    CompositeScore,
    CompositeScorer,
    recompute_dataframe_metrics,
)
from strategy.income import IncomeScorer
from strategy.momentum import MomentumEngine
from strategy.quality import QualityScorer
from strategy.value import ValueScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FactorScoreResult:
    """Full per-stock scoring output from FactorEngine.score_single()."""

    symbol: str
    value_metric: float
    value_score: float
    quality_score: float
    income_score: float
    momentum_score: float
    value_contribution: float
    quality_contribution: float
    income_contribution: float
    momentum_contribution: float
    yield_trap_flag: bool = False
    missing_value_flag: bool = False
    strategy_bucket: str = "core_candidate"
    breakdowns: dict[str, ScoreBreakdown] = field(default_factory=dict)


@dataclass
class UniverseScoreResult:
    """Result of scoring a full universe DataFrame."""

    df: pd.DataFrame
    n_stocks: int
    factor_stats: dict[str, dict]  # {factor: {mean, std, median, ...}}
    regime: str = "bullish"


# ---------------------------------------------------------------------------
# FactorEngine
# ---------------------------------------------------------------------------


class FactorEngine:
    """
    Central scoring orchestrator.

    Usage:
        engine = FactorEngine()
        result = engine.score_single("AAPL", features_dict)
        df_out = engine.score_universe(universe_df)
        stats  = engine.factor_exposures(df_out)
        corr   = engine.factor_correlation_matrix(df_out)
    """

    def __init__(self, config=None) -> None:
        # Defer ConfigManager import to avoid circular dependency at module load time
        if config is None:
            from config.manager import ConfigManager
            config = ConfigManager.get()
        self._cfg = config
        self._composite = CompositeScorer()
        self._value     = ValueScorer()
        self._quality   = QualityScorer()
        self._income    = IncomeScorer()
        self._momentum  = MomentumEngine()

    # ── Single-stock scoring ────────────────────────────────────────────────

    def score_single(
        self,
        symbol: str,
        features: dict,
        score_weights: Optional[dict] = None,
    ) -> FactorScoreResult:
        """Compute composite score for one stock's feature dict."""
        cs: CompositeScore = self._composite.score_features(features, score_weights)
        return FactorScoreResult(
            symbol=symbol,
            value_metric=cs.value_metric,
            value_score=cs.value_score,
            quality_score=cs.quality_score,
            income_score=cs.income_score,
            momentum_score=cs.momentum_score,
            value_contribution=cs.value_contribution,
            quality_contribution=cs.quality_contribution,
            income_contribution=cs.income_contribution,
            momentum_contribution=cs.momentum_contribution,
            yield_trap_flag=cs.yield_trap_flag,
            missing_value_flag=cs.missing_value_flag,
            strategy_bucket=cs.strategy_bucket,
        )

    # ── Universe scoring ────────────────────────────────────────────────────

    def score_universe(
        self,
        df: pd.DataFrame,
        score_weights: Optional[dict] = None,
        apply_cross_sectional: bool = True,
        apply_value_v2: bool = True,
    ) -> pd.DataFrame:
        """
        Score an entire universe DataFrame in-place and return it.

        Steps:
          1. Cross-sectional momentum v2 normalization
          2. Cross-sectional value v2 normalization (if config-enabled)
          3. Recompute value_metric with final scores
        """
        self._composite.score_dataframe(df, score_weights, apply_cross_sectional)

        value_v2_enabled = self._cfg.raw.get("value_v2", {}).get("enabled", True)
        if apply_value_v2 and value_v2_enabled and "pe_ratio" in df.columns:
            try:
                from strategy.value_v2 import apply_cross_sectional_value_v2
                apply_cross_sectional_value_v2(df)
                recompute_dataframe_metrics(df, score_weights)
            except Exception as exc:
                logger.warning("value_v2 cross-sectional pass failed: %s", exc)

        return df

    # ── Factor exposure profile ─────────────────────────────────────────────

    def factor_exposures(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Compute cross-sectional distribution statistics for each factor.

        Returns:
            {factor_name: {mean, std, median, p25, p75, skew}}
        """
        factors = [
            "value_score", "quality_score", "income_score",
            "momentum_score", "value_metric",
        ]
        result: dict[str, dict] = {}
        for f in factors:
            if f not in df.columns:
                continue
            s = pd.to_numeric(df[f], errors="coerce").dropna()
            if len(s) < 2:
                continue
            result[f] = {
                "mean":   round(float(s.mean()),           4),
                "std":    round(float(s.std()),            4),
                "median": round(float(s.median()),         4),
                "p25":    round(float(s.quantile(0.25)),   4),
                "p75":    round(float(s.quantile(0.75)),   4),
                "skew":   round(float(s.skew()),           4),
                "n":      int(len(s)),
            }
        return result

    # ── Correlation matrix ──────────────────────────────────────────────────

    def factor_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Spearman rank correlation matrix across the four factors and composite score.
        """
        cols = [
            c for c in [
                "value_score", "quality_score", "income_score",
                "momentum_score", "value_metric",
            ]
            if c in df.columns
        ]
        if not cols:
            return pd.DataFrame()
        return df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").round(3)
