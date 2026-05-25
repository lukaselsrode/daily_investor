"""
strategy/factor_interactions.py — Conditional factor feature engineering.

Generates engineered momentum variants conditioned on quality, income, or value.
These are research instruments — they are NOT used in live scoring.

All outputs:
  - are cross-sectionally z-score normalized then clipped to [-3, 3]
  - preserve rank ordering within each snapshot
  - are parameterized; no variant is assumed best

Usage:
    from strategy.factor_interactions import add_interaction_features, INTERACTION_FEATURES
    add_interaction_features(df)   # mutates df in-place, returns df
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

INTERACTION_FEATURES: list[dict] = [
    # ── Quality-conditioned momentum ──────────────────────────────────────────
    {
        "name":        "qm_multiply",
        "label":       "Quality × Momentum",
        "group":       "quality_conditioned",
        "description": "momentum * max(quality, 0) — quality gates whether momentum counts",
    },
    {
        "name":        "qm_shift",
        "label":       "Quality-Shifted Momentum",
        "group":       "quality_conditioned",
        "description": "momentum * (0.5 + quality) — quality shifts the momentum multiplier",
    },
    {
        "name":        "qm_sigmoid",
        "label":       "Quality-Sigmoid Momentum",
        "group":       "quality_conditioned",
        "description": "momentum * sigmoid(quality) — smooth quality gating via logistic function",
    },
    {
        "name":        "qm_threshold",
        "label":       "Quality-Threshold Momentum",
        "group":       "quality_conditioned",
        "description": "momentum if quality > 0 else momentum * 0.5 — hard quality threshold penalty",
    },
    # ── Income-conditioned momentum ──────────────────────────────────────────
    {
        "name":        "im_multiply",
        "label":       "Income × Momentum",
        "group":       "income_conditioned",
        "description": "momentum * max(income, 0) — income gates momentum amplitude",
    },
    {
        "name":        "im_shift",
        "label":       "Income-Shifted Momentum",
        "group":       "income_conditioned",
        "description": "momentum * (0.5 + income) — income shifts the momentum multiplier",
    },
    # ── Blended conditioning ──────────────────────────────────────────────────
    {
        "name":        "blend_mean",
        "label":       "Blend-Mean Momentum",
        "group":       "blended",
        "description": "momentum * mean(quality, income) — average of quality and income gates momentum",
    },
    {
        "name":        "blend_weighted",
        "label":       "Blend-Weighted Momentum",
        "group":       "blended",
        "description": "momentum * (0.7 * quality + 0.3 * income) — quality-biased blend gates momentum",
    },
    # ── Value-conditioned momentum (experimental) ────────────────────────────
    {
        "name":        "vm_multiply",
        "label":       "Value × Momentum",
        "group":       "value_conditioned",
        "description": "momentum * max(value, 0) — experimental: does cheap-stock momentum outperform?",
    },
]

INTERACTION_FEATURE_NAMES: list[str] = [f["name"] for f in INTERACTION_FEATURES]

_FEAT_META: dict[str, dict] = {f["name"]: f for f in INTERACTION_FEATURES}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


def _clip_normalize(arr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Cross-sectional z-score then clip to ±clip. Returns rank-stable values."""
    std = np.nanstd(arr)
    if std < 1e-9:
        return np.zeros_like(arr, dtype=float)
    z = (arr - np.nanmean(arr)) / std
    return np.clip(z, -clip, clip)


def _compute_one(
    name: str,
    mom: np.ndarray,
    qual: np.ndarray,
    inc: np.ndarray,
    val: np.ndarray,
) -> np.ndarray:
    if name == "qm_multiply":
        raw = mom * np.maximum(qual, 0.0)
    elif name == "qm_shift":
        raw = mom * (0.5 + qual)
    elif name == "qm_sigmoid":
        raw = mom * _sigmoid(qual)
    elif name == "qm_threshold":
        raw = np.where(qual > 0.0, mom, mom * 0.5)
    elif name == "im_multiply":
        raw = mom * np.maximum(inc, 0.0)
    elif name == "im_shift":
        raw = mom * (0.5 + inc)
    elif name == "blend_mean":
        raw = mom * ((qual + inc) / 2.0)
    elif name == "blend_weighted":
        raw = mom * (0.7 * qual + 0.3 * inc)
    elif name == "vm_multiply":
        raw = mom * np.maximum(val, 0.0)
    else:
        return np.zeros_like(mom, dtype=float)
    return _clip_normalize(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all engineered conditional-momentum columns to df in-place.
    Requires columns: momentum_score, quality_score, income_score, value_score.
    Missing columns are treated as zero rather than raising.
    Returns df.
    """
    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0).values
        return np.zeros(len(df), dtype=float)

    mom  = _col("momentum_score")
    qual = _col("quality_score")
    inc  = _col("income_score")
    val  = _col("value_score")

    for feat in INTERACTION_FEATURES:
        df[feat["name"]] = _compute_one(feat["name"], mom, qual, inc, val)

    return df
