"""
strategy/scoring/composite.py — Unified scoring engine entry point.

Single source of truth for scoring a universe DataFrame. Calls each per-factor
peer-relative scorer, then combines via SCORE_WEIGHTS into `value_metric`.

There is no longer an "overlay" or "fallback"; this IS the scoring engine.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from .growth import apply_growth
from .income import apply_income
from .momentum import apply_momentum
from .quality import apply_quality
from .value import apply_value

logger = logging.getLogger(__name__)


def _regime_tilt_weights(sw: dict, regime: str | None) -> dict:
    """Apply the regime-conditional momentum tilt to score weights (live mirror of
    backtesting.simulator._regime_tilted_weights). In confirmed-bull regime, shift
    `regime.bullish.momentum_tilt` of total weight from value/quality/income into
    momentum. No-op when regime is not bullish, tilt is 0, or regime is unknown.
    Returns a NEW normalized dict; never mutates the input.
    """
    from util import REGIME_PARAMS

    total = sw.get("value", 0.0) + sw.get("quality", 0.0) + sw.get("income", 0.0) + sw.get("momentum", 0.0)
    base = {k: (sw.get(k, 0.0) / total if total > 0 else 0.0)
            for k in ("value", "quality", "income", "momentum")}
    if regime != "bullish":
        return base
    tilt = float((REGIME_PARAMS or {}).get("bullish", {}).get("momentum_tilt", 0.0))
    if tilt <= 0.0:
        return base
    non_mom = base["value"] + base["quality"] + base["income"]
    if non_mom <= 1e-9:
        return base
    move = min(tilt, non_mom)
    scale = (non_mom - move) / non_mom
    return {
        "value":    base["value"] * scale,
        "quality":  base["quality"] * scale,
        "income":   base["income"] * scale,
        "momentum": base["momentum"] + move,
    }


# Snapshot stamp written into every scored DataFrame so loaders know the
# engine revision used. Bump when peer-relative math changes meaningfully.
SCORING_MODEL_VERSION = "peer-1"


def scoring_config_hash(scoring_cfg: dict | None = None) -> str:
    """Stable short hash of the active scoring config — used for snapshot metadata."""
    from util import SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def compute_metric(
    df: pd.DataFrame,
    score_weights: dict | None = None,
    scoring_cfg: dict | None = None,
    regime: str | None = None,
) -> pd.DataFrame:
    """Score a universe DataFrame in-place. Writes:

      value_score, quality_score, income_score, momentum_score   (per-factor)
      *_industry_rank, *_sector_rank, *_market_rank              (diagnostics)
      *_fallback_reason                                          (diagnostics)
      value_distress_flag, yield_trap_flag                        (per-factor flags)
      momentum_penalties_applied                                  (penalty count)
      value_metric                                                (composite)
      scoring_model_version                                       ("peer-1")

    DataFrame must include `industry` / `sector` columns and the momentum input
    columns (rs_3m, rs_6m, risk_adj_momentum_3m, return_1m, return_5d,
    return_3m, realized_vol_3m, position_52w, above_50dma, above_200dma).
    """
    from util import SCORE_WEIGHTS, SCORING_PARAMS

    cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
    sw = score_weights if score_weights is not None else SCORE_WEIGHTS

    apply_value(df, cfg)
    apply_quality(df, cfg)
    apply_momentum(df, cfg)
    apply_income(df, cfg)
    apply_growth(df, cfg)

    # Ensure every score column exists (per-factor enabled=false above sets to 0.0)
    for col in ("value_score", "quality_score", "income_score", "momentum_score"):
        if col not in df.columns:
            df[col] = 0.0

    # Regime-conditional momentum tilt: in confirmed-bull regime, shift weight toward
    # momentum (alpha engine). No-op when regime is None / not bullish / tilt == 0.
    ew = _regime_tilt_weights(sw, regime)
    if regime == "bullish" and abs(ew["momentum"] - (sw.get("momentum", 0.0))) > 1e-9:
        logger.info(
            "regime=bullish momentum tilt applied: momentum %.3f -> %.3f",
            sw.get("momentum", 0.0), ew["momentum"],
        )

    df["value_metric"] = (
        ew["value"]    * df["value_score"]
        + ew["quality"]  * df["quality_score"]
        + ew["income"]   * df["income_score"]
        + ew["momentum"] * df["momentum_score"]
    ).round(3)

    df["scoring_model_version"] = SCORING_MODEL_VERSION

    logger.info(
        "scoring: n=%d | value_metric mean=%.3f std=%.3f",
        len(df), float(df["value_metric"].mean()), float(df["value_metric"].std()),
    )

    return df
