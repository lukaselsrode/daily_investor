"""
strategy/scoring/income.py — Peer-relative, sustainability-aware income scoring (peer-3).

  - Peer-relative: within industry/sector/market, each component is ranked among
    DIVIDEND PAYERS only, so a 2% yield in tech ranks higher than a 2% yield in
    utilities and non-payers never drag the distribution.
  - Sustainability-aware: yield is weighted with FMP-derived payout coverage,
    dividend growth, and payment-streak consistency (data.fundamental_features).
    Missing sustainability inputs renormalize per-row toward yield-only.
  - Zero-dividend stocks score 0 (neutral) — they're not punished for being
    growth companies. Yield traps are masked before ranking and forced to 0.

Components (weights from scoring.income_inputs.weights):
  dividend_yield     0.50   TTM yield
  div_fcf_coverage   0.20   TTM FCF / TTM dividends paid   (div_fcf_coverage_ttm)
  div_growth         0.15   TTM DPS growth vs prior TTM    (div_growth_1y)
  div_streak         0.15   consecutive payments, cap 20   (div_streak_quarters)

peer-3 removed the max(peer_rank, dy/DIVIDEND_THRESHOLD) saturation branch that
pinned every yield ≥ 4.5% at the clamp ceiling (305 names at exactly 1.500) and
made income a near-binary payer indicator.

Output columns:
  income_score
  income_industry_rank, income_sector_rank
  yield_trap_flag
  income_fallback_reason
"""

from __future__ import annotations

import logging

import pandas as pd

from .peer import compute_peer_relative, safe_col

logger = logging.getLogger(__name__)


_COMPONENT_COLUMNS = {
    "dividend_yield":   "dividend_yield",
    "div_fcf_coverage": "div_fcf_coverage_ttm",
    "div_growth":       "div_growth_1y",
    "div_streak":       "div_streak_quarters",
}

_DEFAULT_WEIGHTS = {
    "dividend_yield":   0.50,
    "div_fcf_coverage": 0.20,
    "div_growth":       0.15,
    "div_streak":       0.15,
}

_DEFAULT_MIN_COVERAGE = 0.30


def apply_income(df: pd.DataFrame, scoring_cfg: dict | None = None) -> None:
    """Add income_score + diagnostic columns to df in-place."""
    from util import SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    factor = cfg.get("factors", {}).get("income", {})
    if not factor.get("enabled", True):
        df["income_score"] = 0.0
        return

    ps = cfg["peer_standardization"]
    clamp_lo = float(ps.get("clamp_low", -1.0))
    clamp_hi = float(ps.get("clamp_high", 1.5))

    inputs = cfg.get("income_inputs", {})
    yield_trap_threshold = float(inputs.get("yield_trap_threshold", 0.10))
    weights = {
        k: float(v)
        for k, v in (inputs.get("weights") or _DEFAULT_WEIGHTS).items()
        if k in _COMPONENT_COLUMNS and float(v) > 0.0
    } or dict(_DEFAULT_WEIGHTS)

    dy = safe_col(df, "dividend_yield").fillna(0.0)
    yield_trap_flag = dy >= yield_trap_threshold
    payer = dy > 0
    rankable_rows = payer & ~yield_trap_flag

    # Components with near-zero coverage AMONG PAYERS (sustainability columns on
    # old snapshot vintages) drop entirely; the rest renormalize.
    min_coverage = float(inputs.get("min_coverage", _DEFAULT_MIN_COVERAGE))
    series: dict[str, pd.Series] = {}
    for name in weights:
        vals = dy if name == "dividend_yield" else safe_col(df, _COMPONENT_COLUMNS[name])
        series[name] = vals.where(rankable_rows)
    n_payers = int(rankable_rows.sum())
    active = {
        name: w for name, w in weights.items()
        if n_payers > 0
        and float(series[name].notna().sum()) / max(n_payers, 1) >= min_coverage
    }
    if not active:
        active = {"dividend_yield": 1.0}
    dropped = sorted(set(weights) - set(active))
    if dropped:
        logger.info("income: dropped low-coverage components: %s", ", ".join(dropped))

    # Per-row weight renormalization: a payer missing e.g. div_growth_1y scores on
    # its remaining components instead of being pulled toward 0 by a phantom zero.
    numer = pd.Series(0.0, index=df.index)
    denom = pd.Series(0.0, index=df.index)
    ind_total = pd.Series(0.0, index=df.index)
    sec_total = pd.Series(0.0, index=df.index)
    yield_reason: pd.Series | None = None

    for name, w in active.items():
        vals = series[name]
        blended, ind, sec, _mkt, reason = compute_peer_relative(
            vals, df, cfg, higher_is_better=True,
        )
        valid = vals.notna().astype(float)
        numer = numer + w * blended * valid
        denom = denom + w * valid
        ind_total = ind_total + w * ind.fillna(0.0)
        sec_total = sec_total + w * sec.fillna(0.0)
        if name == "dividend_yield":
            yield_reason = reason

    blended = (numer / denom.where(denom > 0)).fillna(0.0)
    blended[yield_trap_flag] = 0.0
    blended[~payer] = 0.0

    score = blended.clip(clamp_lo, clamp_hi).round(3)

    if yield_reason is None:
        yield_reason = pd.Series("missing", index=df.index)

    df["income_score"] = score
    df["income_industry_rank"] = ind_total.round(4)
    df["income_sector_rank"]   = sec_total.round(4)
    df["yield_trap_flag"] = yield_trap_flag
    df["income_fallback_reason"] = yield_reason.where(payer, "no_dividend")

    logger.info(
        "income: n=%d | mean=%.3f | dividend-payers: %d | yield_traps: %d | active: %s",
        len(score), float(score.mean()),
        int(payer.sum()), int(yield_trap_flag.sum()), ",".join(sorted(active)),
    )
