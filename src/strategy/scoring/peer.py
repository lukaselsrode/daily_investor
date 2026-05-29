"""
strategy/scoring/peer.py — Shared peer-relative scoring utilities.

Functions
---------
peer_percentile(values, groups, *, higher_is_better=True, min_group_size=8,
                winsorize_pct=0.05, clamp=(-1.0, 1.5))
    → (ranks, fallback_reason)
    Per-group percentile rank scaled to (clamp_low, clamp_high). Tickers whose
    group has < min_group_size observations get rank=NaN; caller blends them
    with a coarser grouping via blend_relative.

robust_z(values, groups, *, min_group_size=8, winsorize_pct=0.05,
         clamp=(-1.0, 1.5))
    → (z, fallback_reason)
    Robust z-score (median / MAD) within group, clamped.

blend_relative(industry_rank, sector_rank, market_rank, *, weights, group_by,
               fallback_group_by)
    → (blended_score, fallback_reason)
    Combine industry / sector / market ranks per ticker. Falls back to coarser
    grouping when a ticker's preferred group rank is missing.

The blended fallback_reason for each row is one of:
  "industry"  — primary group had enough peers
  "sector"    — fell back to sector
  "market"    — fell back to cross-sectional market rank
  "missing"   — no group had enough peers AND the underlying value was NaN

Lookahead-safe: ranks are computed only from the DataFrame passed in. In a
backtest, that DataFrame is the per-date universe, so peer ranks at date D use
only data available at date D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_series(s: pd.Series, lo_pct: float = 0.05, hi_pct: float = 0.95) -> pd.Series:
    """Clip series at lo_pct and hi_pct quantiles. NaN values are preserved."""
    finite = s.dropna()
    if len(finite) < 2:
        return s
    lo = float(finite.quantile(lo_pct))
    hi = float(finite.quantile(hi_pct))
    return s.clip(lo, hi)


def _pct_rank_series(s: pd.Series, winsorize_pct: float = 0.05) -> pd.Series:
    """Cross-sectional percentile rank, winsorized and scaled to [-1, 1]. Missing → 0.0."""
    finite = s.notna()
    if finite.sum() < 2:
        return pd.Series(0.0, index=s.index)
    vals = s[finite].copy()
    if winsorize_pct > 0:
        lo = vals.quantile(winsorize_pct)
        hi = vals.quantile(1.0 - winsorize_pct)
        vals = vals.clip(lo, hi)
    ranks = vals.rank(method="average") / (len(vals) + 1)
    result = pd.Series(0.0, index=s.index)
    result[finite] = ranks * 2 - 1
    return result


_FALLBACK_INDUSTRY = "industry"
_FALLBACK_SECTOR = "sector"
_FALLBACK_MARKET = "market"
_FALLBACK_MISSING = "missing"


def safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] as a numeric Series, or a NaN-Series aligned with df.index when missing.

    Robust to snapshots that pre-date later schema additions. Without this,
    calls like ``pd.to_numeric(df.get(name), errors="coerce")`` collapse to a
    numpy scalar when the column is absent and break downstream Series ops.
    """
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def blend_with_anchor(
    peer_score: pd.Series,
    anchor_score: pd.Series,
    blend_weight: float,
    *,
    clamp: tuple[float, float] = (-1.0, 1.5),
) -> pd.Series:
    """Blend a peer-relative score with a cross-sectional anchor score.

    final = blend_weight * anchor_score + (1 - blend_weight) * peer_score

    blend_weight=0 returns pure peer-relative; blend_weight=1 returns pure anchor.
    The blend exists because pure peer-relative scoring can destroy absolute
    market-leader signal — anchoring partially to a non-peer cross-sectional
    rank preserves it where empirically useful.
    """
    if blend_weight <= 0.0:
        return peer_score
    a = float(blend_weight)
    blended = a * pd.to_numeric(anchor_score, errors="coerce").fillna(0.0) + (1.0 - a) * peer_score
    return blended.clip(clamp[0], clamp[1]).round(3)


def peer_percentile(
    values: pd.Series,
    groups: pd.Series | None,
    *,
    higher_is_better: bool = True,
    min_group_size: int = 8,
    winsorize_pct: float = 0.05,
    clamp: tuple[float, float] = (-1.0, 1.5),
) -> tuple[pd.Series, pd.Series]:
    """Group-relative percentile rank scaled to clamp range.

    Returns (ranks, has_group_rank):
      ranks   — pd.Series aligned with values; NaN where group too small
                (caller decides fallback)
      has_group_rank — bool Series; True where the ticker received a real
                       within-group rank
    """
    idx = values.index
    ranks = pd.Series(np.nan, index=idx, dtype="float64")
    has_group_rank = pd.Series(False, index=idx)

    if groups is None:
        return ranks, has_group_rank

    values_num = pd.to_numeric(values, errors="coerce")
    ascending = bool(higher_is_better)
    clamp_lo, clamp_hi = clamp

    for group_name in groups.dropna().unique():
        group_idx = groups.index[groups == group_name]
        group_vals = values_num.loc[group_idx].dropna()
        if len(group_vals) < min_group_size:
            continue
        if winsorize_pct > 0 and len(group_vals) >= 2:
            group_vals = winsorize_series(group_vals, winsorize_pct, 1.0 - winsorize_pct)
        n = len(group_vals)
        if n < 2:
            continue
        pct = group_vals.rank(method="average", ascending=ascending) / (n + 1)
        scaled = pct * (clamp_hi - clamp_lo) + clamp_lo
        ranks.loc[group_vals.index] = scaled.values
        has_group_rank.loc[group_vals.index] = True

    return ranks.round(4), has_group_rank


def robust_z(
    values: pd.Series,
    groups: pd.Series | None,
    *,
    higher_is_better: bool = True,
    min_group_size: int = 8,
    winsorize_pct: float = 0.05,
    clamp: tuple[float, float] = (-1.0, 1.5),
) -> tuple[pd.Series, pd.Series]:
    """Robust within-group z-score (median / MAD), clamped.

    higher_is_better=False inverts the sign so lower raw values produce higher
    scores (used for PE / PB).
    """
    idx = values.index
    z = pd.Series(np.nan, index=idx, dtype="float64")
    has_group_rank = pd.Series(False, index=idx)

    if groups is None:
        return z, has_group_rank

    values_num = pd.to_numeric(values, errors="coerce")
    if not higher_is_better:
        values_num = -values_num
    clamp_lo, clamp_hi = clamp

    for group_name in groups.dropna().unique():
        group_idx = groups.index[groups == group_name]
        group_vals = values_num.loc[group_idx].dropna()
        if len(group_vals) < min_group_size:
            continue
        if winsorize_pct > 0 and len(group_vals) >= 2:
            group_vals = winsorize_series(group_vals, winsorize_pct, 1.0 - winsorize_pct)
        if len(group_vals) < 2:
            continue
        med = float(group_vals.median())
        mad = float((group_vals - med).abs().median())
        if mad <= 1e-9:
            scaled = pd.Series(0.0, index=group_vals.index)
        else:
            scaled = (group_vals - med) / (1.4826 * mad)
            scaled = scaled.clip(clamp_lo, clamp_hi)
        z.loc[scaled.index] = scaled.values
        has_group_rank.loc[scaled.index] = True

    return z.round(4), has_group_rank


def market_rank(
    values: pd.Series,
    *,
    higher_is_better: bool = True,
    winsorize_pct: float = 0.05,
    clamp: tuple[float, float] = (-1.0, 1.5),
) -> pd.Series:
    """Cross-sectional rank over the full DataFrame, scaled to clamp range."""
    values_num = pd.to_numeric(values, errors="coerce")
    base = _pct_rank_series(values_num, winsorize_pct=winsorize_pct)
    if not higher_is_better:
        base = -base
    clamp_lo, clamp_hi = clamp
    scaled = (base + 1.0) / 2.0 * (clamp_hi - clamp_lo) + clamp_lo
    return scaled.round(4)


def blend_relative(
    industry_rank: pd.Series,
    sector_rank: pd.Series,
    market_rank_series: pd.Series,
    *,
    industry_has: pd.Series | None = None,
    sector_has: pd.Series | None = None,
    weights: dict[str, float] | None = None,
    group_by: str = "industry",
    fallback_group_by: str = "sector",
) -> tuple[pd.Series, pd.Series]:
    """Blend industry/sector/market ranks per ticker with explicit fallback.

    Behavior
    --------
    Per ticker, in priority order:
      1. If the primary group (industry) had ≥ min_group_size peers → reason="industry",
         score = weighted blend of (industry, sector, market) ranks.
      2. Else if fallback group (sector) had enough peers → reason="sector",
         score = weighted blend of (sector, market) only, weights renormalized.
      3. Else → reason="market", score = market_rank.
      4. If market_rank is also NaN → reason="missing", score = 0.0.

    Returns (blended_score, fallback_reason)
    """
    if weights is None:
        weights = {"industry_relative": 0.60, "sector_relative": 0.25, "market_relative": 0.15}

    w_ind = float(weights.get("industry_relative", 0.60))
    w_sec = float(weights.get("sector_relative",   0.25))
    w_mkt = float(weights.get("market_relative",   0.15))

    idx = market_rank_series.index
    if industry_has is None:
        industry_has = pd.Series(industry_rank.notna(), index=idx)
    if sector_has is None:
        sector_has = pd.Series(sector_rank.notna(), index=idx)

    blended = pd.Series(np.nan, index=idx, dtype="float64")
    reason = pd.Series(_FALLBACK_MISSING, index=idx, dtype="object")

    ind_vals = pd.to_numeric(industry_rank, errors="coerce").reindex(idx)
    sec_vals = pd.to_numeric(sector_rank,   errors="coerce").reindex(idx)
    mkt_vals = pd.to_numeric(market_rank_series, errors="coerce").reindex(idx)

    if group_by == "industry":
        primary_has = industry_has.fillna(False).astype(bool)
        primary_vals = ind_vals
        primary_label = _FALLBACK_INDUSTRY
    elif group_by == "sector":
        primary_has = sector_has.fillna(False).astype(bool)
        primary_vals = sec_vals
        primary_label = _FALLBACK_SECTOR
    else:
        primary_has = pd.Series(False, index=idx)
        primary_vals = pd.Series(np.nan, index=idx)
        primary_label = _FALLBACK_MARKET

    if fallback_group_by == "sector":
        fallback_has = sector_has.fillna(False).astype(bool) & ~primary_has
        fallback_vals = sec_vals
        fallback_label = _FALLBACK_SECTOR
    elif fallback_group_by == "industry":
        fallback_has = industry_has.fillna(False).astype(bool) & ~primary_has
        fallback_vals = ind_vals
        fallback_label = _FALLBACK_INDUSTRY
    else:
        fallback_has = pd.Series(False, index=idx)
        fallback_vals = pd.Series(np.nan, index=idx)
        fallback_label = _FALLBACK_MARKET

    # Tier 1: primary group present (use full blend, falling back to market for missing tiers)
    if primary_has.any():
        ind_part = primary_vals.where(primary_has).fillna(0.0)
        sec_part = sec_vals.where(primary_has).fillna(0.0)
        mkt_part = mkt_vals.where(primary_has).fillna(0.0)
        # If sector rank is missing, redistribute its weight to market+industry
        sec_present = sec_vals.notna() & primary_has
        mkt_present = mkt_vals.notna() & primary_has
        denom = (
            w_ind * primary_has.astype(float)
            + w_sec * sec_present.astype(float)
            + w_mkt * mkt_present.astype(float)
        )
        numer = w_ind * ind_part + w_sec * sec_part * sec_present.astype(float) + w_mkt * mkt_part * mkt_present.astype(float)
        blended.loc[primary_has] = (numer[primary_has] / denom[primary_has]).values
        reason.loc[primary_has] = primary_label

    # Tier 2: fallback group present (renormalize fallback + market only)
    tier2 = fallback_has & ~primary_has
    if tier2.any():
        sec_part = fallback_vals.where(tier2).fillna(0.0)
        mkt_part = mkt_vals.where(tier2).fillna(0.0)
        mkt_present = mkt_vals.notna() & tier2
        denom = (w_sec + w_ind) * tier2.astype(float) + w_mkt * mkt_present.astype(float)
        numer = (w_sec + w_ind) * sec_part + w_mkt * mkt_part * mkt_present.astype(float)
        blended.loc[tier2] = (numer[tier2] / denom[tier2]).values
        reason.loc[tier2] = fallback_label

    # Tier 3: market rank only
    tier3 = ~primary_has & ~fallback_has & mkt_vals.notna()
    if tier3.any():
        blended.loc[tier3] = mkt_vals[tier3].values
        reason.loc[tier3] = _FALLBACK_MARKET

    # Tier 4: missing — explicit 0.0 neutral fallback so downstream sums stay finite
    missing = blended.isna()
    blended.loc[missing] = 0.0
    reason.loc[missing] = _FALLBACK_MISSING

    return blended.round(4), reason


def compute_peer_relative(
    values: pd.Series,
    df: pd.DataFrame,
    cfg: dict,
    *,
    higher_is_better: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """High-level helper: rank a value column at industry/sector/market levels and blend.

    Returns:
      blended_score, industry_rank, sector_rank, market_rank_series, fallback_reason
    """
    ps = cfg["peer_standardization"]
    min_n = int(ps.get("min_group_size", 8))
    wz = float(ps.get("winsorize_pct", 0.05))
    clamp = (float(ps.get("clamp_low", -1.0)), float(ps.get("clamp_high", 1.5)))
    weights = ps.get("blend", {"industry_relative": 0.60, "sector_relative": 0.25, "market_relative": 0.15})
    method = str(ps.get("method", "percentile"))

    industry = df.get("industry")
    sector = df.get("sector")

    rank_fn = peer_percentile if method == "percentile" else robust_z

    if industry is not None:
        ind_rank, ind_has = rank_fn(
            values, industry, higher_is_better=higher_is_better,
            min_group_size=min_n, winsorize_pct=wz, clamp=clamp,
        )
    else:
        ind_rank = pd.Series(np.nan, index=values.index)
        ind_has = pd.Series(False, index=values.index)

    if sector is not None:
        sec_rank, sec_has = rank_fn(
            values, sector, higher_is_better=higher_is_better,
            min_group_size=min_n, winsorize_pct=wz, clamp=clamp,
        )
    else:
        sec_rank = pd.Series(np.nan, index=values.index)
        sec_has = pd.Series(False, index=values.index)

    mkt_rank = market_rank(values, higher_is_better=higher_is_better, winsorize_pct=wz, clamp=clamp)

    blended, reason = blend_relative(
        ind_rank, sec_rank, mkt_rank,
        industry_has=ind_has, sector_has=sec_has,
        weights=weights,
        group_by=str(ps.get("group_by", "industry")),
        fallback_group_by=str(ps.get("fallback_group_by", "sector")),
    )

    return blended, ind_rank, sec_rank, mkt_rank, reason
