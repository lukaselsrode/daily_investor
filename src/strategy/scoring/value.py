"""
strategy/scoring/value.py — Peer-relative value scoring.

Ranks PE and PB within industry (→ sector → market fallback), blends, applies
distress penalties, clamps.

Mutates df in-place. Output columns:
  value_score
  value_industry_rank, value_sector_rank, value_market_rank
  value_fallback_reason
  value_distress_flag
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .peer import blend_with_anchor, compute_peer_relative, safe_col

logger = logging.getLogger(__name__)


def apply_value(df: pd.DataFrame, scoring_cfg: dict | None = None) -> None:
    """Add peer-relative value_score + diagnostic columns to df in-place."""
    from util import SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    factor = cfg.get("factors", {}).get("value", {})
    if not factor.get("enabled", True):
        df["value_score"] = 0.0
        return

    ps = cfg["peer_standardization"]
    clamp_lo = float(ps.get("clamp_low", -1.0))
    clamp_hi = float(ps.get("clamp_high", 1.5))
    pe_w = float(factor.get("pe_weight", 0.70))
    pb_w = float(factor.get("pb_weight", 0.30))

    dist = factor.get("distress", {})
    dist_thr = float(dist.get("pe_threshold", 5.0))
    dist_pen = float(dist.get("pe_penalty", 0.30))
    neg_pen = float(dist.get("negative_eps_penalty", 0.25))

    pe_raw = safe_col(df, "pe_ratio")
    pb_raw = safe_col(df, "pb_ratio")
    pe = pe_raw.copy()
    pe[pe <= 0] = np.nan
    pb = pb_raw.copy()
    pb[pb <= 0] = np.nan

    # Peer-relative rank for PE and PB separately (low PE = good → invert).
    pe_blended, pe_ind, pe_sec, pe_mkt, pe_reason = compute_peer_relative(
        pe, df, cfg, higher_is_better=False,
    )
    pb_blended, pb_ind, pb_sec, pb_mkt, pb_reason = compute_peer_relative(
        pb, df, cfg, higher_is_better=False,
    )

    # Composite: weighted blend across PE+PB ranks, adjusting for missing inputs.
    has_pe = pe.notna()
    has_pb = pb.notna()
    both = has_pe & has_pb
    pe_only = has_pe & ~has_pb
    pb_only = ~has_pe & has_pb
    neither = ~has_pe & ~has_pb

    composite = pd.Series(0.0, index=df.index)
    composite[both]    = pe_w * pe_blended[both]    + pb_w * pb_blended[both]
    composite[pe_only] = pe_blended[pe_only]
    composite[pb_only] = pb_blended[pb_only]
    composite[neither] = -0.25  # missing both PE and PB: explicit floor

    distress_mask = pe_raw.notna() & (pe_raw > 0) & (pe_raw <= dist_thr)
    composite[distress_mask] -= dist_pen
    neg_eps_mask = pe_raw.notna() & (pe_raw < 0)
    composite[neg_eps_mask] -= neg_pen
    distress_flag = distress_mask | neg_eps_mask

    score = composite.clip(clamp_lo, clamp_hi).round(3)

    # Cross-sectional anchor: market-wide rank of the PE/PB composite. A high
    # anchor_blend (e.g. 0.5) keeps absolute-valuation signal alive when pure
    # peer ranking would otherwise destroy it.
    anchor_blend = float(factor.get("anchor_blend", 0.0))
    if anchor_blend > 0.0:
        from .peer import _pct_rank_series
        # Lower PE/PB = better → invert sign on the rank so high anchor = cheap.
        pe_anchor = -_pct_rank_series(pe_raw.where(pe_raw > 0))
        pb_anchor = -_pct_rank_series(pb_raw.where(pb_raw > 0))
        anchor = pe_w * pe_anchor.fillna(0.0) + pb_w * pb_anchor.fillna(0.0)
        # Distress penalties applied to the anchor too
        anchor[distress_mask] -= dist_pen
        anchor[neg_eps_mask] -= neg_pen
        score = blend_with_anchor(score, anchor, anchor_blend, clamp=(clamp_lo, clamp_hi))

    df["value_score"] = score
    df["value_industry_rank"] = pe_w * pe_ind.fillna(0.0) + pb_w * pb_ind.fillna(0.0)
    df["value_sector_rank"]   = pe_w * pe_sec.fillna(0.0) + pb_w * pb_sec.fillna(0.0)
    df["value_market_rank"]   = pe_w * pe_mkt.fillna(0.0) + pb_w * pb_mkt.fillna(0.0)
    rank_order = {"industry": 0, "sector": 1, "market": 2, "missing": 3}
    pe_rank = pe_reason.map(rank_order).fillna(3)
    pb_rank = pb_reason.map(rank_order).fillna(3)
    inv = {v: k for k, v in rank_order.items()}
    df["value_fallback_reason"] = (
        pd.concat([pe_rank, pb_rank], axis=1).max(axis=1).map(inv)
    )
    df["value_distress_flag"] = distress_flag

    logger.info(
        "value: n=%d | mean=%.3f std=%.3f | distress: %d | industry-rank: %d",
        len(score), float(score.mean()), float(score.std()),
        int(distress_flag.sum()), int((df["value_fallback_reason"] == "industry").sum()),
    )
