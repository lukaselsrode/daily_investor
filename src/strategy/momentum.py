"""
strategy/momentum.py — MomentumEngine: v1 bin-based + v2 cross-sectional.

Migrated from:
  - source_data.get_momentum_score          → MomentumEngine.score_v1 / compute_momentum_score_v1
  - source_data._apply_cross_sectional_*    → MomentumEngine.apply_cross_sectional
  - source_data._pct_rank_series            → _pct_rank_series (module-level, used by backtest too)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ScoreBreakdown, ScorerBase


def compute_momentum_score_v1(
    position_52w: float | None,
    return_1m: float | None = None,
) -> float:
    """
    Map 52-week position and 1-month return to a bin-based momentum score.
    Migrated from source_data.get_momentum_score().
    """
    from util import MOMENTUM_PARAMS

    if position_52w is None:
        return 0.0

    mp = MOMENTUM_PARAMS
    bins = mp["position_bin_boundaries"]
    scores = mp["position_bin_scores"]

    base = scores[-1]
    for i, boundary in enumerate(bins):
        if position_52w < boundary:
            base = scores[i]
            break

    cutoff = mp["return_1m_low_position_cutoff"]
    if return_1m is not None and position_52w < cutoff:
        if return_1m >= mp["return_1m_recovery_threshold"]:
            base += mp["return_1m_recovery_bonus"]
        elif return_1m <= mp["return_1m_falling_knife_threshold"]:
            base -= mp["return_1m_falling_knife_penalty"]

    return round(base, 3)


def _pct_rank_series(s: pd.Series, winsorize_pct: float = 0.05) -> pd.Series:
    """
    Cross-sectional percentile rank, winsorized and scaled to [-1, 1].
    Missing values → 0.0 (neutral mid-rank).

    Not a lookahead bias: ranks contemporaneous values across all stocks.
    Migrated from source_data._pct_rank_series.
    """
    finite = s.notna()
    if finite.sum() < 2:
        return pd.Series(0.0, index=s.index)
    vals = s[finite].copy()
    if winsorize_pct > 0:
        lo = vals.quantile(winsorize_pct)
        hi = vals.quantile(1.0 - winsorize_pct)
        vals = vals.clip(lo, hi)
    ranks = vals.rank(method="average") / (len(vals) + 1)  # (0, 1)
    result = pd.Series(0.0, index=s.index)
    result[finite] = ranks * 2 - 1  # scale to (-1, 1)
    return result


def apply_cross_sectional_momentum_v2(df: pd.DataFrame) -> None:
    """
    Replace momentum_score column with v2 continuous cross-sectional formula.

    Called once after all stocks are evaluated so ranking is over the full
    daily universe — no lookahead into future dates.
    Migrated from source_data._apply_cross_sectional_momentum_scores.

    Mutates df in-place.
    """
    from util import MOMENTUM_V2_PARAMS

    cfg = MOMENTUM_V2_PARAMS
    wp = cfg["weights"]
    pen = cfg["penalties"]
    wp_total = sum(wp.values())
    if wp_total < 1e-9:
        return

    w_rs3m  = wp["rs_3m"]           / wp_total
    w_rs6m  = wp["rs_6m"]           / wp_total
    w_radj  = wp["risk_adj_3m"]     / wp_total
    w_trend = wp["trend_structure"]  / wp_total
    w_r1m   = wp["return_1m"]       / wp_total
    w_r5d   = wp["return_5d"]       / wp_total

    wz = cfg["winsorize_pct"]

    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(float("nan"), index=df.index)

    n_rs3m = _pct_rank_series(_col("rs_3m"),                wz)
    n_rs6m = _pct_rank_series(_col("rs_6m"),                wz)
    n_radj = _pct_rank_series(_col("risk_adj_momentum_3m"), wz)
    n_r1m  = _pct_rank_series(_col("return_1m"),            wz)
    n_r5d  = _pct_rank_series(_col("return_5d"),            wz)

    above50  = df["above_50dma"].astype(bool)  if "above_50dma"  in df.columns else pd.Series(False, index=df.index)
    above200 = df["above_200dma"].astype(bool) if "above_200dma" in df.columns else pd.Series(False, index=df.index)
    trend = pd.Series(np.select(
        [above50 & above200, above50 & ~above200, ~above50 & above200],
        [0.5,                 0.1,                -0.1],
        default=-0.5,
    ), index=df.index)

    score = (
        w_rs3m  * n_rs3m  +
        w_rs6m  * n_rs6m  +
        w_radj  * n_radj  +
        w_trend * trend   +
        w_r1m   * n_r1m   +
        w_r5d   * n_r5d
    )

    ret3m = _col("return_3m")
    vol3m = _col("realized_vol_3m")
    pos52 = _col("position_52w")

    falling_knife = ret3m.fillna(0.0) < pen["falling_knife_3m_threshold"]
    overextended  = pos52.fillna(0.0) > pen["overextension_52w_threshold"]
    high_vol      = vol3m.fillna(0.0) > pen["high_vol_annual_threshold"]

    score = score - falling_knife.astype(float) * pen["falling_knife_penalty"]
    score = score - overextended.astype(float)  * pen["overextension_penalty"]
    score = score - high_vol.astype(float)      * pen["high_vol_penalty"]

    score = score.clip(cfg["clamp_low"], cfg["clamp_high"]).round(3)
    df["momentum_score"] = score


class MomentumEngine(ScorerBase):
    """
    Dual-mode momentum scorer.

    score()                → v1 bin-based (single stock, no cross-sectional context)
    apply_cross_sectional() → v2 cross-sectional normalization over a full universe DataFrame
    """

    def score(self, features: dict) -> float:
        return compute_momentum_score_v1(
            features.get("position_52w"),
            features.get("return_1m"),
        )

    def apply_cross_sectional(self, df: pd.DataFrame) -> None:
        """Apply v2 cross-sectional momentum scoring to the universe DataFrame in-place."""
        apply_cross_sectional_momentum_v2(df)

    def breakdown(self, symbol: str, features: dict) -> ScoreBreakdown:
        score = self.score(features)
        return ScoreBreakdown(
            symbol=symbol,
            score=score,
            components={
                "position_52w": float(features["position_52w"]) if features.get("position_52w") is not None else 0.0,
                "return_1m": float(features["return_1m"]) if features.get("return_1m") is not None else 0.0,
            },
            flags={
                "has_52w_position": features.get("position_52w") is not None,
            },
        )
