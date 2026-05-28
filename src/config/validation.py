"""
config/validation.py — Exit-threshold ordering validation.

Checks that the four score thresholds that govern when to exit a position are
consistent and create non-empty, logically ordered decision zones:

    hard_exit_score_below  <  sell_weak_value_below  <=  trim_score_below  <=  metric_threshold

Zone semantics
--------------
  score < hard_exit_score_below          → confirmed collapse, forced EXIT
  hard_exit ≤ score < sell_weak          → thesis exit zone (held long enough → sell)
  sell_weak ≤ score < trim_score_below   → trim zone (profitable + weakening → partial exit)
  trim_score_below ≤ score < metric_thr  → hold (score decayed but not enough to trim yet)
  score ≥ metric_threshold               → strong hold / candidate for new buys

Public API
----------
validate_exit_thresholds(...)  → list[str]   (warnings, empty = valid)
warn_if_invalid(...)           → None        (logs/warns; call at config load time)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_NARROW_ZONE_WARN = 0.02  # warn when any ordered zone is narrower than this


def validate_exit_thresholds(
    metric_threshold: float,
    sell_weak_value_below: float,
    trim_score_below: float,
    hard_exit_score_below: float,
    review_score_below: float | None = None,
) -> list[str]:
    """
    Return a list of human-readable warning strings describing any ordering
    violations or empty zones.  An empty list means the config is valid.

    Parameters
    ----------
    metric_threshold      : entry/candidate gate (e.g. 0.75)
    sell_weak_value_below : hard thesis exit (e.g. 0.72)
    trim_score_below      : explicit trim trigger (e.g. 0.74)
    hard_exit_score_below : forced score-collapse exit (e.g. 0.20)
    review_score_below    : hold-not-exit protection floor — not in the main
                            exit chain, so only lightly validated (optional)
    """
    warnings: list[str] = []

    # ── Primary ordering: hard_exit < sell_weak ──────────────────────────────
    if hard_exit_score_below >= sell_weak_value_below:
        warnings.append(
            f"hard_exit_score_below ({hard_exit_score_below:.3f}) >= "
            f"sell_weak_value_below ({sell_weak_value_below:.3f}): "
            "the forced-collapse exit fires at the same score or higher than the "
            "thesis-exit — hard exit zone is empty."
        )

    # ── Primary ordering: sell_weak < trim_score_below ───────────────────────
    if sell_weak_value_below >= trim_score_below:
        warnings.append(
            f"sell_weak_value_below ({sell_weak_value_below:.3f}) >= "
            f"trim_score_below ({trim_score_below:.3f}): "
            "trim zone [sell_weak, trim_score_below) is empty — trim will never fire. "
            "Raise trim_score_below above sell_weak_value_below."
        )
    elif trim_score_below - sell_weak_value_below < _NARROW_ZONE_WARN:
        warnings.append(
            f"Trim zone is very narrow: "
            f"sell_weak={sell_weak_value_below:.3f}, trim_score_below={trim_score_below:.3f} "
            f"(width={trim_score_below - sell_weak_value_below:.4f} < {_NARROW_ZONE_WARN}). "
            "Very few positions will be partially trimmed."
        )

    # ── trim_score_below should not exceed metric_threshold ──────────────────
    if trim_score_below > metric_threshold:
        warnings.append(
            f"trim_score_below ({trim_score_below:.3f}) > "
            f"metric_threshold ({metric_threshold:.3f}): "
            "trim fires even when the score is above the entry/buy threshold. "
            "Consider setting trim_score_below <= metric_threshold "
            f"(currently {metric_threshold:.3f})."
        )

    # ── review_score_below: informational only ────────────────────────────────
    if review_score_below is not None and review_score_below > trim_score_below:
        warnings.append(
            f"review_score_below ({review_score_below:.3f}) > "
            f"trim_score_below ({trim_score_below:.3f}): "
            "the hold-not-exit protection floor is above the trim threshold, which "
            "means profitable positions in the trim zone will be protected from exit. "
            "This may suppress intended trims."
        )

    return warnings


def warn_if_invalid(
    metric_threshold: float,
    sell_weak_value_below: float,
    trim_score_below: float,
    hard_exit_score_below: float,
    review_score_below: float | None = None,
) -> None:
    """
    Log a WARNING for each threshold ordering violation found.
    Called at config load time — never raises, only warns.
    """
    issues = validate_exit_thresholds(
        metric_threshold=metric_threshold,
        sell_weak_value_below=sell_weak_value_below,
        trim_score_below=trim_score_below,
        hard_exit_score_below=hard_exit_score_below,
        review_score_below=review_score_below,
    )
    for msg in issues:
        logger.warning("Exit threshold config issue: %s", msg)
