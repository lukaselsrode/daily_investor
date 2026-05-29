"""
strategy/scoring/momentum.py — Peer-relative momentum scoring.

Re-ranks momentum inputs (rs_3m, rs_6m, risk_adj_momentum_3m, trend_structure,
return_1m, return_5d) within industry/sector/market peers, weighted-combines
using `scoring.momentum_inputs.weights`, and applies penalties (falling knife /
overextension / high vol).

When per-factor `enabled: false`, falls back to a pure cross-sectional anchor
(weighted-sum of momentum-input ranks without peer grouping).

Output columns:
  momentum_score
  momentum_industry_rank, momentum_sector_rank, momentum_market_rank
  momentum_fallback_reason
  momentum_penalties_applied   (int — count of penalties hit)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .peer import _pct_rank_series, blend_with_anchor, compute_peer_relative, safe_col

logger = logging.getLogger(__name__)


def _trend_structure_series(df: pd.DataFrame) -> pd.Series:
    above50 = df["above_50dma"].astype(bool) if "above_50dma" in df.columns else pd.Series(False, index=df.index)
    above200 = df["above_200dma"].astype(bool) if "above_200dma" in df.columns else pd.Series(False, index=df.index)
    return pd.Series(np.select(
        [above50 & above200, above50 & ~above200, ~above50 & above200],
        [0.5, 0.1, -0.1],
        default=-0.5,
    ), index=df.index)


def _compute_anchor(df: pd.DataFrame, weights: dict, penalties: dict, clamp: tuple[float, float]) -> pd.Series:
    """Cross-sectional weighted-sum of momentum input ranks (no peer grouping)."""
    raw_inputs = {
        "rs_3m":           safe_col(df, "rs_3m"),
        "rs_6m":           safe_col(df, "rs_6m"),
        "risk_adj_3m":     safe_col(df, "risk_adj_momentum_3m"),
        "trend_structure": _trend_structure_series(df),
        "return_1m":       safe_col(df, "return_1m"),
        "return_5d":       safe_col(df, "return_5d"),
    }
    score = pd.Series(0.0, index=df.index)
    w_total = sum(weights.values())
    if w_total < 1e-9:
        return score
    for name, vals in raw_inputs.items():
        w = weights.get(name, 0.0) / w_total
        if w <= 0:
            continue
        if name == "trend_structure":
            score = score + w * vals
        else:
            score = score + w * _pct_rank_series(vals)
    ret3m = safe_col(df, "return_3m").fillna(0.0)
    vol3m = safe_col(df, "realized_vol_3m").fillna(0.0)
    pos52 = safe_col(df, "position_52w").fillna(0.0)
    score -= (ret3m < penalties["falling_knife_3m_threshold"]).astype(float) * penalties["falling_knife_penalty"]
    score -= (pos52 > penalties["overextension_52w_threshold"]).astype(float) * penalties["overextension_penalty"]
    score -= (vol3m > penalties["high_vol_annual_threshold"]).astype(float) * penalties["high_vol_penalty"]
    return score.clip(clamp[0], clamp[1])


def apply_momentum(df: pd.DataFrame, scoring_cfg: dict | None = None) -> None:
    """Add momentum_score + diagnostic columns to df in-place."""
    from util import SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    factor = cfg.get("factors", {}).get("momentum", {})

    mi = cfg["momentum_inputs"]
    wp = mi["weights"]
    pen = mi["penalties"]
    mi_clamp = (float(mi.get("clamp_low", -1.0)), float(mi.get("clamp_high", 1.5)))

    if not factor.get("enabled", True):
        df["momentum_score"] = _compute_anchor(df, wp, pen, mi_clamp).round(3)
        return

    ps = cfg["peer_standardization"]
    clamp_lo = float(ps.get("clamp_low", -1.0))
    clamp_hi = float(ps.get("clamp_high", 1.5))

    wp_total = sum(wp.values())
    if wp_total < 1e-9:
        df["momentum_score"] = 0.0
        return

    weights = {k: v / wp_total for k, v in wp.items()}
    inputs = {
        "rs_3m":           safe_col(df, "rs_3m"),
        "rs_6m":           safe_col(df, "rs_6m"),
        "risk_adj_3m":     safe_col(df, "risk_adj_momentum_3m"),
        "trend_structure": _trend_structure_series(df),
        "return_1m":       safe_col(df, "return_1m"),
        "return_5d":       safe_col(df, "return_5d"),
    }

    ind_total = pd.Series(0.0, index=df.index)
    sec_total = pd.Series(0.0, index=df.index)
    mkt_total = pd.Series(0.0, index=df.index)
    blended_total = pd.Series(0.0, index=df.index)
    fallback_reasons: list[pd.Series] = []

    for name, vals in inputs.items():
        w = weights[name]
        blended, ind, sec, mkt, reason = compute_peer_relative(
            vals, df, cfg, higher_is_better=True,
        )
        blended_total = blended_total + w * blended
        ind_total = ind_total + w * ind.fillna(0.0)
        sec_total = sec_total + w * sec.fillna(0.0)
        mkt_total = mkt_total + w * mkt.fillna(0.0)
        fallback_reasons.append(reason)

    ret3m = safe_col(df, "return_3m").fillna(0.0)
    vol3m = safe_col(df, "realized_vol_3m").fillna(0.0)
    pos52 = safe_col(df, "position_52w").fillna(0.0)

    falling_knife = ret3m < pen["falling_knife_3m_threshold"]
    overextended  = pos52 > pen["overextension_52w_threshold"]
    high_vol      = vol3m > pen["high_vol_annual_threshold"]

    blended_total -= falling_knife.astype(float) * pen["falling_knife_penalty"]
    blended_total -= overextended.astype(float)  * pen["overextension_penalty"]
    blended_total -= high_vol.astype(float)      * pen["high_vol_penalty"]

    score = blended_total.clip(clamp_lo, clamp_hi).round(3)

    rank_order = {"industry": 0, "sector": 1, "market": 2, "missing": 3}
    inv = {v: k for k, v in rank_order.items()}
    coded = pd.concat([r.map(rank_order).fillna(3) for r in fallback_reasons], axis=1)
    worst = coded.max(axis=1).map(inv)

    # Cross-sectional anchor: same weighted-sum but without peer grouping.
    anchor_blend = float(factor.get("anchor_blend", 0.0))
    if anchor_blend > 0.0:
        anchor = _compute_anchor(df, wp, pen, mi_clamp)
        score = blend_with_anchor(score, anchor, anchor_blend, clamp=(clamp_lo, clamp_hi))

    df["momentum_score"] = score
    df["momentum_industry_rank"] = ind_total.round(4)
    df["momentum_sector_rank"]   = sec_total.round(4)
    df["momentum_market_rank"]   = mkt_total.round(4)
    df["momentum_fallback_reason"] = worst
    df["momentum_penalties_applied"] = (
        falling_knife.astype(int) + overextended.astype(int) + high_vol.astype(int)
    )

    logger.info(
        "momentum: n=%d | mean=%.3f std=%.3f | industry-rank: %d | "
        "penalties (knife/over/vol): %d / %d / %d",
        len(score), float(score.mean()), float(score.std()),
        int((df["momentum_fallback_reason"] == "industry").sum()),
        int(falling_knife.sum()), int(overextended.sum()), int(high_vol.sum()),
    )
