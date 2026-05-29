"""
strategy/scoring/income.py — Peer-relative, safety-aware income scoring.

  - Peer-relative: within industry/sector/market, dividend yield is ranked,
    so a 2% yield in tech ranks higher than a 2% yield in utilities.
  - Zero-dividend stocks in non-income sectors score 0 (neutral) — they're
    not punished for being growth companies.
  - Yield-trap detection preserved.

Output columns:
  income_score
  income_industry_rank, income_sector_rank
  yield_trap_flag
  income_fallback_reason
"""

from __future__ import annotations

import logging

import pandas as pd

from .peer import blend_with_anchor, compute_peer_relative, safe_col

logger = logging.getLogger(__name__)


def apply_income(df: pd.DataFrame, scoring_cfg: dict | None = None) -> None:
    """Add income_score + diagnostic columns to df in-place."""
    from util import DIVIDEND_THRESHOLD, SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    factor = cfg.get("factors", {}).get("income", {})
    if not factor.get("enabled", True):
        df["income_score"] = 0.0
        return

    ps = cfg["peer_standardization"]
    clamp_lo = float(ps.get("clamp_low", -1.0))
    clamp_hi = float(ps.get("clamp_high", 1.5))

    qc = cfg["quality_checklist"]
    yield_trap_threshold = float(qc["yield_trap_threshold"])
    income_score_cap = float(qc["income_score_cap"])

    dy = safe_col(df, "dividend_yield").fillna(0.0)
    yield_trap_flag = dy >= yield_trap_threshold

    # For ranking purposes mask out yield traps so they don't dominate the top
    # of the peer distribution — they get a forced 0 below.
    rankable = dy.where(~yield_trap_flag)
    # Zero-dividend tickers also receive NaN so they get a neutral 0 below
    # rather than being ranked at the floor.
    rankable = rankable.where(dy > 0)

    blended, ind, sec, mkt, reason = compute_peer_relative(
        rankable, df, cfg, higher_is_better=True,
    )

    # Apply income cap from existing scoring config so behavior parity with v1
    # for "high but not trap" yields.
    over_cap = dy > 0
    if over_cap.any():
        capped = (dy / max(DIVIDEND_THRESHOLD, 1e-9)).clip(upper=income_score_cap)
        # When the raw yield is well above DIVIDEND_THRESHOLD, prefer the
        # higher of (peer-rank blended) and (cap-scaled v1 style) to keep
        # high-yield, peer-leading dividend stocks competitive.
        blended = blended.where(blended.notna(), 0.0)
        blended = pd.concat([blended, capped.where(over_cap, 0.0)], axis=1).max(axis=1)

    blended[yield_trap_flag] = 0.0
    blended[dy <= 0] = 0.0

    score = blended.clip(clamp_lo, clamp_hi).round(3)

    # Cross-sectional anchor: capped yield/threshold ratio.
    anchor_blend = float(factor.get("anchor_blend", 0.0))
    if anchor_blend > 0.0:
        anchor = (dy / max(DIVIDEND_THRESHOLD, 1e-9)).clip(0.0, income_score_cap)
        anchor[yield_trap_flag] = 0.0
        score = blend_with_anchor(score, anchor, anchor_blend, clamp=(clamp_lo, clamp_hi))

    df["income_score"] = score
    df["income_industry_rank"] = ind.fillna(0.0).round(4)
    df["income_sector_rank"]   = sec.fillna(0.0).round(4)
    df["yield_trap_flag"] = yield_trap_flag
    df["income_fallback_reason"] = reason.where(dy > 0, "no_dividend")

    logger.info(
        "income: n=%d | mean=%.3f | dividend-payers: %d | yield_traps: %d",
        len(score), float(score.mean()),
        int((dy > 0).sum()), int(yield_trap_flag.sum()),
    )
