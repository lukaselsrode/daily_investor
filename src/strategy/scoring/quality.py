"""
strategy/scoring/quality.py — Peer-relative FUNDAMENTALS quality scoring (peer-3).

Quality is built from real statement fundamentals (data.fundamental_features, PIT
from the FMP cache): profitability, cash generation, accrual discipline, margins
and their trend, leverage, and share-count discipline. Each component is
peer-ranked separately within industry/sector/market and weight-combined;
components with insufficient frame coverage are dropped and the remaining
weights renormalize (old snapshot vintages).

Components ranked:
  roe_ttm                 (TTM net income / equity)          higher=better
  fcf_to_assets           (TTM FCF / total assets)           higher=better
  neg_accruals            (-(NI - CFO)/assets)               higher=better
  gross_margin_ttm        (TTM gross profit / revenue)       higher=better
  low_leverage            (-totalDebt/totalAssets)           higher=better
  share_count_shrink_yoy  (-YoY diluted share growth)        higher=better
  gm_trend_yoy            (TTM gross margin vs 4q ago)       higher=better

peer-3 removed (2026-08-05): dividend terms (income owns dividends), PE/PB terms
(value owns valuation + distress), liquidity/size terms (the reliability layer
and hard volume gates own tradability), position_52w (momentum owns price
geometry), analyst_conviction (never cleared the coverage gate), and the legacy
checklist anchor.

Rows with no fundamental inputs at all (symbol absent from the FMP cache) score
neutral 0.0 with quality_fallback_reason="no_fundamentals".

Output columns:
  quality_score
  quality_industry_rank, quality_sector_rank, quality_market_rank
  quality_fallback_reason
"""

from __future__ import annotations

import logging

import pandas as pd

from .peer import _pct_rank_series, compute_peer_relative, safe_col

logger = logging.getLogger(__name__)


# Defaults when scoring.quality_components is absent (or carries only pre-peer-3
# component names). Config-fixed, NOT tuner slots — correlated DOF on a small
# snapshot substrate is pure overfit surface.
_COMPONENT_WEIGHTS = {
    "roe_ttm":                0.22,
    "fcf_to_assets":          0.20,
    "neg_accruals":           0.15,
    "gross_margin_ttm":       0.12,
    "low_leverage":           0.12,
    "share_count_shrink_yoy": 0.10,
    "gm_trend_yoy":           0.09,
}

_DEFAULT_MIN_COVERAGE = 0.30

_NO_FUNDAMENTALS = "no_fundamentals"


def _component_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "roe_ttm":                safe_col(df, "roe_ttm"),
        "fcf_to_assets":          safe_col(df, "fcf_to_assets"),
        "neg_accruals":           safe_col(df, "neg_accruals"),
        "gross_margin_ttm":       safe_col(df, "gross_margin_ttm"),
        # Negated so the shared higher-is-better ranking orders low leverage first.
        "low_leverage":           -safe_col(df, "debt_to_assets"),
        "share_count_shrink_yoy": safe_col(df, "share_count_shrink_yoy"),
        "gm_trend_yoy":           safe_col(df, "gm_trend_yoy"),
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

    components = _component_series(df)

    cfg_weights = {
        str(k): float(v) for k, v in (cfg.get("quality_components") or {}).items()
    }
    weights_cfg = {k: v for k, v in cfg_weights.items() if k in components and v > 0.0}
    if not weights_cfg:
        if cfg_weights:
            logger.warning(
                "quality: config quality_components carries no peer-3 component names "
                "(%s) — using code defaults; run `daily-investor config migrate-scoring`",
                ", ".join(sorted(cfg_weights)),
            )
        weights_cfg = dict(_COMPONENT_WEIGHTS)

    # A component whose input column is (near-)absent in this frame — old snapshot
    # vintages, unenriched frames — is dropped and the remaining weights renormalize,
    # instead of injecting a uniform mid-rank for every row.
    min_coverage = float(
        cfg.get("quality_fundamentals", {}).get("min_coverage", _DEFAULT_MIN_COVERAGE)
    )
    active = {
        name: w for name, w in weights_cfg.items()
        if float(components[name].notna().mean()) >= min_coverage
    }
    dropped = sorted(set(weights_cfg) - set(active))
    if dropped:
        logger.info("quality: dropped low-coverage components: %s", ", ".join(dropped))

    if not active:
        logger.warning(
            "quality: no fundamental component clears min_coverage=%.2f — "
            "neutral-scoring the whole frame", min_coverage,
        )
        df["quality_score"] = 0.0
        df["quality_industry_rank"] = 0.0
        df["quality_sector_rank"] = 0.0
        df["quality_market_rank"] = 0.0
        df["quality_fallback_reason"] = _NO_FUNDAMENTALS
        return

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

    score = blended_total.clip(clamp_lo, clamp_hi)

    rank_order = {"industry": 0, "sector": 1, "market": 2, "missing": 3}
    inv = {v: k for k, v in rank_order.items()}
    coded = pd.concat([r.map(rank_order).fillna(3) for r in fallback_reasons], axis=1)
    worst = coded.max(axis=1).map(inv)

    # Symbols absent from the fundamentals cache: every active input NaN → hard
    # neutral, explicitly labeled so coverage is auditable per snapshot.
    no_fund = pd.concat(
        [components[name].isna() for name in active], axis=1
    ).all(axis=1)
    if no_fund.any():
        score.loc[no_fund] = 0.0
        worst.loc[no_fund] = _NO_FUNDAMENTALS

    # Low-vol quality blend (param slot 48, frozen-by-default = 0.0): blends the
    # cross-sectional low-volatility rank into quality — mirrors the simulator's
    # _low_vol_score_at_day so a tuned nonzero value behaves identically live.
    qlv = float(cfg.get("quality_low_vol_blend", 0.0))
    if qlv > 0.0:
        low_vol = _pct_rank_series(-safe_col(df, "realized_vol_3m"))
        score = ((1.0 - qlv) * score + qlv * low_vol).clip(clamp_lo, clamp_hi)

    df["quality_score"] = score.round(3)
    df["quality_industry_rank"] = ind_total.round(4)
    df["quality_sector_rank"]   = sec_total.round(4)
    df["quality_market_rank"]   = mkt_total.round(4)
    df["quality_fallback_reason"] = worst

    logger.info(
        "quality: n=%d | mean=%.3f std=%.3f | industry-rank: %d | no-fundamentals: %d | "
        "active: %s",
        len(score), float(score.mean()), float(score.std()),
        int((df["quality_fallback_reason"] == "industry").sum()),
        int((df["quality_fallback_reason"] == _NO_FUNDAMENTALS).sum()),
        ",".join(sorted(active)),
    )
