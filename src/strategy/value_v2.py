"""
strategy/value_v2.py — Cross-sectional, sector-relative value scoring engine.

Replaces the raw PE/PB ratio transform with:
  1. Per-sector winsorization at configurable percentile cutoffs
  2. Sector-relative percentile ranking (low PE in sector → high value rank)
  3. Robust composite (PE + PB ranks weighted by data availability)
  4. Configurable distress penalties for ultra-low or negative PE

Called once per run after the full universe DataFrame is built — analogous to
apply_cross_sectional_momentum_v2 in strategy/momentum.py.  Mutates df in-place.

Diagnostic columns added:
  value_score_raw   — old ratio-based value_score (pre-normalization)
  sector_value_score — identical to the new value_score, kept for UI referencing
  relative_pe       — sector-percentile rank of PE, (-1, 1), high = cheap
  relative_pb       — sector-percentile rank of PB, (-1, 1), high = cheap

Backward compatibility:
  pe_comp / pb_comp are never touched — the backtest engine reads them directly.
  value_score is replaced in-place so all downstream consumers see the improved signal.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import ScoreBreakdown, ScorerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateless normalization helpers (importable from diagnostics / tests)
# ---------------------------------------------------------------------------

def winsorize_series(s: pd.Series, lo_pct: float = 0.05, hi_pct: float = 0.95) -> pd.Series:
    """Clip series at lo_pct and hi_pct quantiles. NaN values are preserved."""
    finite = s.dropna()
    if len(finite) < 2:
        return s
    lo = float(finite.quantile(lo_pct))
    hi = float(finite.quantile(hi_pct))
    return s.clip(lo, hi)


def robust_zscore(s: pd.Series) -> pd.Series:
    """
    Robust z-score using median and MAD instead of mean / std.

    Scale factor 1.4826 makes the result consistent with standard normal
    z-scores when the underlying distribution is Gaussian.
    NaN → 0.0 (neutral).
    """
    finite = s.dropna()
    if len(finite) < 2:
        return pd.Series(0.0, index=s.index)
    med = float(finite.median())
    mad = float((finite - med).abs().median())
    if mad < 1e-9:
        mad = float(finite.std()) or 1.0
    result = (s - med) / (1.4826 * mad)
    return result.fillna(0.0)


def sector_relative_percentile(
    values: pd.Series,
    sectors: pd.Series,
    min_sector_size: int = 5,
    invert: bool = False,
) -> pd.Series:
    """
    Percentile rank within sector, scaled to (-1, 1).

    Stocks whose sector has fewer than min_sector_size valid observations fall
    back to the global cross-sectional rank.

    Parameters
    ----------
    values : numeric Series (NaN = missing, excluded from ranking)
    sectors : string Series aligned with values
    min_sector_size : minimum sector sample needed to use sector-level ranking
    invert : True → lower value gets higher rank (use for PE: cheap = good)

    Returns
    -------
    Series in (-1, 1); NaN inputs → 0.0 (neutral mid-rank)
    """
    ascending = not invert  # rank ascending when invert=False (higher raw → higher rank)
    result = pd.Series(0.0, index=values.index)

    # Global rank as the default for all stocks with data
    global_vals = values.dropna()
    if len(global_vals) >= 2:
        global_ranks = global_vals.rank(method="average", ascending=ascending) / (len(global_vals) + 1)
        result[global_vals.index] = global_ranks * 2.0 - 1.0  # → (-1, 1)

    if sectors is None:
        return result

    # Override with sector rank where sector is large enough
    for sector in sectors.dropna().unique():
        sector_idx = sectors[sectors == sector].index
        sector_vals = values.loc[sector_idx].dropna()
        if len(sector_vals) < min_sector_size:
            continue  # keep global rank for this sector
        sector_ranks = (
            sector_vals.rank(method="average", ascending=ascending) / (len(sector_vals) + 1)
        )
        result.loc[sector_vals.index] = sector_ranks * 2.0 - 1.0

    return result


# ---------------------------------------------------------------------------
# Cross-sectional value engine (DataFrame-level, called once per run)
# ---------------------------------------------------------------------------

def apply_cross_sectional_value_v2(df: pd.DataFrame) -> None:
    """
    Replace value_score with a sector-relative robust score.  Mutates df in-place.

    Algorithm
    ---------
    1. Save ratio-based value_score as value_score_raw (for diagnostics).
    2. Extract and validate PE/PB (positive values only).
    3. Winsorize within each sector at configured percentiles.
    4. Compute sector-relative percentile rank for PE and PB separately
       (invert=True so low PE → high value rank).
    5. Composite: weighted average of PE rank and PB rank; adjust weights
       when only one valuation is available.
    6. Apply configurable distress penalties (ultra-low PE, negative PE).
    7. Clamp to [clamp_low, clamp_high] and round to 3 dp.
    8. Write sector_value_score (=new value_score) and diagnostic columns.
    """
    from util import VALUE_V2_PARAMS

    cfg = VALUE_V2_PARAMS
    if not cfg.get("enabled", True):
        logger.info("value_v2 disabled — keeping legacy ratio-based value_score")
        return

    wz       = cfg.get("winsorize_pct", 0.05)
    min_n    = cfg.get("min_sector_size", 5)
    pe_w     = cfg["composite"]["pe_weight"]
    pb_w     = cfg["composite"]["pb_weight"]
    dist_thr = cfg["distress"]["pe_threshold"]
    dist_pen = cfg["distress"]["pe_penalty"]
    neg_pen  = cfg["distress"]["negative_eps_penalty"]
    clamp_lo = cfg.get("clamp_low", -1.0)
    clamp_hi = cfg.get("clamp_high", 1.5)

    # -- 1. Save legacy score for diagnostics ---------------------------------
    if "value_score" in df.columns:
        df["value_score_raw"] = pd.to_numeric(df["value_score"], errors="coerce")
    else:
        df["value_score_raw"] = float("nan")

    # -- 2. Extract valid PE/PB -----------------------------------------------
    pe_raw = pd.to_numeric(df.get("pe_ratio", pd.Series(float("nan"), index=df.index)), errors="coerce")
    pb_raw = pd.to_numeric(df.get("pb_ratio", pd.Series(float("nan"), index=df.index)), errors="coerce")

    pe = pe_raw.copy()
    pe[pe <= 0] = float("nan")   # negative / zero PE treated as missing for value ranking
    pb = pb_raw.copy()
    pb[pb <= 0] = float("nan")

    sectors = df.get("sector", None)
    if sectors is not None:
        sectors = df["sector"].copy()

    # -- 3. Per-sector winsorization ------------------------------------------
    pe_w_ser = pe.copy()
    pb_w_ser = pb.copy()

    if cfg.get("sector_relative", True) and sectors is not None:
        for sector in sectors.dropna().unique():
            idx = sectors[sectors == sector].index
            if len(idx) >= 2:
                pe_w_ser.loc[idx] = winsorize_series(pe_w_ser.loc[idx], wz, 1.0 - wz)
                pb_w_ser.loc[idx] = winsorize_series(pb_w_ser.loc[idx], wz, 1.0 - wz)
    else:
        pe_w_ser = winsorize_series(pe_w_ser, wz, 1.0 - wz)
        pb_w_ser = winsorize_series(pb_w_ser, wz, 1.0 - wz)

    # -- 4. Sector-relative percentile ranks ----------------------------------
    rel_pe = sector_relative_percentile(pe_w_ser, sectors, min_n, invert=True)
    rel_pb = sector_relative_percentile(pb_w_ser, sectors, min_n, invert=True)

    df["relative_pe"] = rel_pe.round(3)
    df["relative_pb"] = rel_pb.round(3)

    # -- 5. Composite ---------------------------------------------------------
    has_pe = pe_w_ser.notna()
    has_pb = pb_w_ser.notna()
    both    = has_pe & has_pb
    pe_only = has_pe & ~has_pb
    pb_only = ~has_pe & has_pb
    neither = ~has_pe & ~has_pb

    composite = pd.Series(0.0, index=df.index)
    composite[both]    = pe_w * rel_pe[both]    + pb_w * rel_pb[both]
    composite[pe_only] = rel_pe[pe_only]
    composite[pb_only] = rel_pb[pb_only]
    composite[neither] = -0.25  # penalty for complete absence of valuation data

    # -- 6. Distress penalties ------------------------------------------------
    # Ultra-low PE: could be cyclical peak earnings or distress — penalise scepticism
    distress_mask = pe_raw.notna() & (pe_raw > 0) & (pe_raw <= dist_thr)
    composite[distress_mask] -= dist_pen

    # Negative PE: company is loss-making — override any PB-only value
    neg_eps_mask = pe_raw.notna() & (pe_raw < 0)
    composite[neg_eps_mask] -= neg_pen

    # -- 7. Clamp and assign --------------------------------------------------
    score = composite.clip(clamp_lo, clamp_hi).round(3)
    df["sector_value_score"] = score
    df["value_score"] = score  # replace legacy ratio-based score

    n_distress  = int(distress_mask.sum())
    n_neg_eps   = int(neg_eps_mask.sum())
    n_no_data   = int(neither.sum())

    logger.info(
        "Value v2: n=%d | mean=%.3f std=%.3f | "
        "distress_pe≤%.0f: %d | neg_eps: %d | no_valuation: %d",
        len(score), float(score.mean()), float(score.std()),
        dist_thr, n_distress, n_neg_eps, n_no_data,
    )


# ---------------------------------------------------------------------------
# Scorer class (single-stock interface for diagnostic / testing use)
# ---------------------------------------------------------------------------

class ValueEngineV2(ScorerBase):
    """
    Sector-relative value scorer.

    Single-stock score() uses the legacy ratio-based approach (sector context
    is unavailable for a single row).  Full cross-sectional normalization is
    only available through apply_cross_sectional() which requires the whole
    universe DataFrame.
    """

    def score(self, features: dict) -> float:
        from strategy.value import compute_value_components
        _, _, value_score, _ = compute_value_components(
            features.get("pe_ratio"),
            features.get("pb_ratio"),
            features.get("sector") or "",
            features.get("industry") or "",
        )
        return value_score

    def apply_cross_sectional(self, df: pd.DataFrame) -> None:
        apply_cross_sectional_value_v2(df)

    def breakdown(self, symbol: str, features: dict) -> ScoreBreakdown:
        score = self.score(features)
        pe = features.get("pe_ratio")
        pb = features.get("pb_ratio")
        return ScoreBreakdown(
            symbol=symbol,
            score=score,
            components={
                "relative_pe": features.get("relative_pe", float("nan")),
                "relative_pb": features.get("relative_pb", float("nan")),
            },
            flags={
                "missing_valuation": pe is None and pb is None,
                "distress_pe": pe is not None and pe > 0 and pe <= 5.0,
                "negative_eps": pe is not None and pe < 0,
            },
            notes=["Single-stock score uses legacy ratios; sector-relative score requires full universe."],
        )
