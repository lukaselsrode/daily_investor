"""
strategy/scoring/quality.py — Peer-relative quality scoring.

The snapshot schema has no real fundamentals (no ROE/ROA/margins/FCF/debt), so
we peer-rank the existing checklist components separately within
industry/sector/market and recombine.

Components ranked:
  volume                  (liquidity)               higher=better
  has_positive_pe         (clean profitability)     higher=better
  has_positive_pb         (clean book value)        higher=better
  distress_pe_flag        (0 < PE < distress_max)   lower=better (inverted)
  yield_in_healthy_band   (2-6% dividend band)      higher=better
  position_52w            (where in 52w range)      higher=better (small weight)

Yield-trap penalty stays in income — quality doesn't double-penalize sectors
that don't typically pay dividends.

Fallback: when group sizes too small at every level AND
use_legacy_checklist_fallback=True, fall back to _checklist_quality.

Output columns:
  quality_score
  quality_industry_rank, quality_sector_rank, quality_market_rank
  quality_fallback_reason
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._legacy_checklist import _checklist_quality
from .peer import blend_with_anchor, compute_peer_relative, safe_col

logger = logging.getLogger(__name__)


# Defaults when scoring.quality_components is absent from config. dollar_volume +
# volume_consistency together carry the 0.20 mass the raw share-ADV component held
# in peer-1, so sparse frames (which drop the new columns) score like a liquidity swap.
_COMPONENT_WEIGHTS = {
    "dollar_volume":      0.12,
    "volume_consistency": 0.08,
    "has_positive_pe":    0.20,
    "has_positive_pb":    0.10,
    "no_distress_pe":     0.20,
    "healthy_yield":      0.10,
    "position_52w":       0.05,
    "market_cap":         0.05,
    "analyst_conviction": 0.05,
}

_DEFAULT_HORIZON_WEIGHTS = {"dv_5d": 0.20, "dv_21d": 0.30, "dv_63d": 0.50}
_DEFAULT_MIN_COVERAGE = 0.30
_DEFAULT_MIN_NUM_RATINGS = 5


def _dollar_volume_series(df: pd.DataFrame, hw: dict) -> pd.Series:
    """log1p of the horizon-weighted mean dollar volume, NaN-renormalized per row."""
    parts = {
        "dv_5d":  safe_col(df, "dollar_vol_5d"),
        "dv_21d": safe_col(df, "dollar_vol_21d"),
        "dv_63d": safe_col(df, "dollar_vol_63d"),
    }
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for key, vals in parts.items():
        w = float(hw.get(key, _DEFAULT_HORIZON_WEIGHTS[key]))
        num = num + vals.fillna(0.0).clip(lower=0.0) * w
        den = den + vals.notna().astype(float) * w
    combined = num.where(den > 0) / den.where(den > 0)
    return np.log1p(combined)


def _component_series(df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    qc = cfg["quality_checklist"]
    ql = cfg.get("quality_liquidity", {})
    qa = cfg.get("quality_analyst", {})
    pe = safe_col(df, "pe_ratio")
    pb = safe_col(df, "pb_ratio")
    dy = safe_col(df, "dividend_yield").fillna(0.0)
    pos52 = safe_col(df, "position_52w")

    ratings = safe_col(df, "analyst_num_ratings")
    buy_pct = safe_col(df, "analyst_buy_pct")
    min_ratings = float(qa.get("min_num_ratings", _DEFAULT_MIN_NUM_RATINGS))

    return {
        "dollar_volume":      _dollar_volume_series(df, ql.get("horizon_weights", {})),
        # Low variability of daily dollar volume = stable institutional participation;
        # negated so the shared higher-is-better ranking orders it correctly.
        "volume_consistency": -safe_col(df, "dollar_vol_cv_63d"),
        "has_positive_pe":    (pe > 0).astype(float),
        "has_positive_pb":    (pb > 0).astype(float),
        "no_distress_pe":     (~((pe > 0) & (pe < qc["distress_pe_max"]))).astype(float),
        "healthy_yield":      (
            (dy >= qc["quality_dividend_min"]) & (dy <= qc["quality_dividend_max"])
        ).astype(float),
        "position_52w":       pos52,
        "market_cap":         np.log1p(safe_col(df, "market_cap").clip(lower=0.0)),
        "analyst_conviction": buy_pct.where(ratings >= min_ratings),
    }


def apply_quality(df: pd.DataFrame, scoring_cfg: dict | None = None) -> None:
    """Add quality_score + diagnostic columns to df in-place."""
    from util import SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    factor = cfg.get("factors", {}).get("quality", {})
    if not factor.get("enabled", True):
        # Per-factor disable: leave quality_score at 0.0 (composite weights handle the rest)
        df["quality_score"] = 0.0
        return

    ps = cfg["peer_standardization"]
    clamp_lo = float(ps.get("clamp_low", -1.0))
    clamp_hi = float(ps.get("clamp_high", 1.5))
    legacy_fallback = bool(factor.get("use_legacy_checklist_fallback", True))

    if "quality_checklist" not in cfg:
        cfg = {**cfg, "quality_checklist": SCORING_PARAMS["quality_checklist"]}
    components = _component_series(df, cfg)

    # A component whose input column is (near-)absent in this frame — old snapshot
    # vintages, stripped agg_data — is dropped and the remaining weights renormalize,
    # instead of injecting a uniform mid-rank for every row.
    weights_cfg = {
        k: float(v)
        for k, v in (cfg.get("quality_components") or _COMPONENT_WEIGHTS).items()
        if k in components and float(v) > 0.0
    }
    min_coverage = float(cfg.get("quality_liquidity", {}).get("min_coverage", _DEFAULT_MIN_COVERAGE))
    active = {
        name: w for name, w in weights_cfg.items()
        if float(components[name].notna().mean()) >= min_coverage
    }
    dropped = sorted(set(weights_cfg) - set(active))
    if dropped:
        logger.info("quality: dropped low-coverage components: %s", ", ".join(dropped))
    if not active:
        active = {"position_52w": 1.0}

    ind_total = pd.Series(0.0, index=df.index)
    sec_total = pd.Series(0.0, index=df.index)
    mkt_total = pd.Series(0.0, index=df.index)
    blended_total = pd.Series(0.0, index=df.index)
    fallback_reasons: list[pd.Series] = []

    w_sum = sum(active.values())
    for comp_name, w_raw in active.items():
        values = components[comp_name]
        w = w_raw / w_sum
        blended, ind, sec, mkt, reason = compute_peer_relative(
            values, df, cfg, higher_is_better=True,
        )
        blended_total = blended_total + w * blended
        ind_total = ind_total + w * ind.fillna(0.0)
        sec_total = sec_total + w * sec.fillna(0.0)
        mkt_total = mkt_total + w * mkt.fillna(0.0)
        fallback_reasons.append(reason)

    score = blended_total.clip(clamp_lo, clamp_hi).round(3)

    rank_order = {"industry": 0, "sector": 1, "market": 2, "missing": 3}
    inv = {v: k for k, v in rank_order.items()}
    coded = pd.concat([r.map(rank_order).fillna(3) for r in fallback_reasons], axis=1)
    worst = coded.max(axis=1).map(inv)

    if legacy_fallback:
        all_missing = (coded == 3).all(axis=1)
        if all_missing.any():
            for i in df.index[all_missing]:
                score.loc[i] = _checklist_quality(
                    pe_ratio=df.at[i, "pe_ratio"] if "pe_ratio" in df.columns else None,
                    pb_ratio=df.at[i, "pb_ratio"] if "pb_ratio" in df.columns else None,
                    volume=float(df.at[i, "volume"]) if "volume" in df.columns and pd.notna(df.at[i, "volume"]) else 0.0,
                    dividend_yield=(
                        float(df.at[i, "dividend_yield"])
                        if "dividend_yield" in df.columns and pd.notna(df.at[i, "dividend_yield"])
                        else 0.0
                    ),
                )
            worst.loc[all_missing] = "legacy_checklist"

    # Cross-sectional anchor: row-wise checklist score. A high anchor_blend
    # keeps absolute quality signal alive when pure peer ranking would flatten it.
    anchor_blend = float(factor.get("anchor_blend", 0.0))
    if anchor_blend > 0.0:
        anchor = pd.Series(0.0, index=df.index)
        for i in df.index:
            anchor.loc[i] = _checklist_quality(
                pe_ratio=df.at[i, "pe_ratio"] if "pe_ratio" in df.columns else None,
                pb_ratio=df.at[i, "pb_ratio"] if "pb_ratio" in df.columns else None,
                volume=float(df.at[i, "volume"]) if "volume" in df.columns and pd.notna(df.at[i, "volume"]) else 0.0,
                dividend_yield=(
                    float(df.at[i, "dividend_yield"])
                    if "dividend_yield" in df.columns and pd.notna(df.at[i, "dividend_yield"])
                    else 0.0
                ),
            )
        score = blend_with_anchor(score, anchor, anchor_blend, clamp=(clamp_lo, clamp_hi))

    df["quality_score"] = score.round(3)
    df["quality_industry_rank"] = ind_total.round(4)
    df["quality_sector_rank"]   = sec_total.round(4)
    df["quality_market_rank"]   = mkt_total.round(4)
    df["quality_fallback_reason"] = worst

    logger.info(
        "quality: n=%d | mean=%.3f std=%.3f | industry-rank: %d | legacy-fallback: %d",
        len(score), float(score.mean()), float(score.std()),
        int((df["quality_fallback_reason"] == "industry").sum()),
        int((df["quality_fallback_reason"] == "legacy_checklist").sum()),
    )
