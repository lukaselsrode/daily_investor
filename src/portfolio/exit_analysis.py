"""
portfolio/exit_analysis.py — Transparent exit root-cause analysis.

For every EXIT or WATCH position, produces:
  - primary_driver / secondary_driver (named, human-readable)
  - reason_weights: normalized dict summing to 1.0
  - PrematureExitFlag when a high-quality name with positive return/momentum is being exited
  - override_recommendation: "WATCH" when the flag fires (not a hard override — review only)

This module is purely analytical. It never places orders or modifies the sell engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Driver labels (display + weight key)
# ---------------------------------------------------------------------------

DRIVER_LABELS: dict[str, str] = {
    "stop_loss":              "Stop loss triggered",
    "trailing_stop":          "Trailing stop triggered",
    "take_profit":            "Take-profit triggered",
    "yield_trap":             "Yield trap detected",
    "quality_floor":          "Quality floor breached",
    "momentum_deterioration": "Momentum deterioration",
    "score_decay":            "Composite score decay",
    "rank_deterioration":     "Universe rank deterioration",
    "confidence_loss":        "Reliability / confidence collapse",
    "sentiment_override":     "Sentiment override",
    "harvest":                "Harvest / profit-taking logic",
}

# Canonical ordering for display
DRIVER_ORDER: list[str] = list(DRIVER_LABELS.keys())


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExitAnalysis:
    symbol: str
    primary_driver: str            # key from DRIVER_LABELS
    secondary_driver: str          # key from DRIVER_LABELS
    reason_weights: dict[str, float]   # normalized, sums to 1.0
    primary_label: str             # human-readable primary driver
    secondary_label: str           # human-readable secondary driver
    thesis_intact_score: float     # 0–1 (higher = thesis still looks good)
    is_premature: bool
    premature_reason: str
    override_recommendation: str   # "WATCH" or "" — never "EXIT" override
    confidence: str                # "HIGH" / "MEDIUM" / "LOW"


@dataclass
class PrematureExitFlag:
    symbol: str
    thesis_intact_score: float
    quality_score: float | None
    momentum_score: float | None
    pct_change: float | None
    primary_exit_driver: str
    reason: str


# ---------------------------------------------------------------------------
# Config / sell-rule helpers
# ---------------------------------------------------------------------------

def _sell_rules() -> dict:
    try:
        from util import SELL_RULES
        return SELL_RULES
    except Exception:
        return {
            "stop_loss_pct": -0.20,
            "trailing_stop_pct": -0.08,
            "take_profit_pct": 0.60,
            "sell_weak_value_below": 0.45,
            "sell_low_quality_below": -0.25,
            "sell_yield_trap": True,
        }


def _pi_config() -> dict:
    try:
        from core.utils import load_config_raw
        return load_config_raw().get("portfolio_intelligence", {})
    except Exception:
        return {}


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _compute_raw_weights(
    pct_change: float | None,
    current_price: float | None,
    peak_price: float | None,
    momentum_score: float | None,
    quality_score: float | None,
    value_metric: float | None,
    score_at_buy: float | None,
    rank_pct_now: float | None,
    rank_pct_at_buy: float | None,
    reliability_score: float | None,
    yield_trap: bool,
    sr: dict,
) -> dict[str, float]:
    """
    Compute un-normalized signal strengths (0–1) for every exit driver.
    """
    raw: dict[str, float] = {k: 0.0 for k in DRIVER_LABELS}

    # ── Hard mechanical triggers ──────────────────────────────────────────────

    if pct_change is not None:
        stop = sr["stop_loss_pct"]
        if pct_change <= stop:
            raw["stop_loss"] = 1.0
        elif pct_change < 0:
            # Proximity: linear scale from 0 at pct=0 to 1 at pct=stop
            raw["stop_loss"] = max(0.0, min(0.6, -pct_change / -stop))

    if current_price is not None and peak_price is not None and peak_price > 0:
        drawdown = (current_price / peak_price) - 1.0
        ts = sr["trailing_stop_pct"]
        if drawdown <= ts:
            raw["trailing_stop"] = 1.0
        elif drawdown < 0:
            raw["trailing_stop"] = max(0.0, min(0.7, drawdown / ts))

    if pct_change is not None:
        tp = sr["take_profit_pct"]
        if pct_change >= tp:
            raw["take_profit"] = 1.0
        elif pct_change > 0.25:
            raw["take_profit"] = (pct_change - 0.25) / max(tp - 0.25, 0.01)

    if yield_trap and sr.get("sell_yield_trap"):
        raw["yield_trap"] = 0.65

    if quality_score is not None:
        floor = sr["sell_low_quality_below"]
        if quality_score <= floor:
            raw["quality_floor"] = 1.0
        elif quality_score < 0:
            raw["quality_floor"] = max(0.0, min(0.7, (-quality_score) / max(-floor, 0.01)))

    # ── Continuous factor signals ─────────────────────────────────────────────

    if momentum_score is not None:
        if momentum_score < -0.40:
            raw["momentum_deterioration"] = 1.0
        elif momentum_score < 0:
            raw["momentum_deterioration"] = -momentum_score / 0.40
        elif momentum_score < 0.10:
            raw["momentum_deterioration"] = max(0.0, (0.10 - momentum_score) / 0.10) * 0.25

    if score_at_buy is not None and value_metric is not None and score_at_buy > 0:
        drop_pct = (score_at_buy - value_metric) / score_at_buy
        raw["score_decay"] = max(0.0, min(1.0, drop_pct / 0.45))

    if rank_pct_at_buy is not None and rank_pct_now is not None:
        drop = rank_pct_at_buy - rank_pct_now
        raw["rank_deterioration"] = max(0.0, min(1.0, drop / 0.45))

    if reliability_score is not None and reliability_score < 0.85:
        # Capped at 0.45: confidence collapse signals uncertainty, not a strong directional exit.
        # Allowing it to reach 1.0 would starve all factor signals during normalization.
        raw["confidence_loss"] = max(0.0, min(0.45, (0.85 - reliability_score) / 0.45))

    return raw


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total < 1e-9:
        # No signal detected — attribute equally to score_decay and momentum
        return {**{k: 0.0 for k in DRIVER_LABELS}, "score_decay": 0.5, "momentum_deterioration": 0.5}
    return {k: round(v / total, 4) for k, v in raw.items()}


def _top_two(weights: dict[str, float]) -> tuple[str, str]:
    ordered = sorted(weights.items(), key=lambda x: -x[1])
    primary   = ordered[0][0] if ordered else "score_decay"
    secondary = ordered[1][0] if len(ordered) > 1 else primary
    return primary, secondary


# ---------------------------------------------------------------------------
# Thesis-intact score
# ---------------------------------------------------------------------------

def _thesis_intact_score(
    quality_score: float | None,
    momentum_score: float | None,
    pct_change: float | None,
    rank_pct_now: float | None,
) -> float:
    """
    0–1 score measuring how intact the original thesis looks despite the exit signal.
    Higher = thesis still looks valid = more suspicious that an exit was triggered.
    """
    components: list[float] = []

    if quality_score is not None:
        # quality in [-1, 1] → normalize to [0, 1]
        q_norm = max(0.0, min(1.0, (quality_score + 0.5) / 1.5))
        components.append(q_norm)

    if momentum_score is not None:
        # positive momentum = thesis supports holding
        m_norm = 1.0 if momentum_score >= 0.10 else (0.6 if momentum_score >= 0 else max(0.0, 0.5 + momentum_score))
        components.append(m_norm)

    if pct_change is not None:
        # positive return = thesis working
        r_norm = 1.0 if pct_change >= 0.05 else (0.7 if pct_change >= 0 else max(0.0, 0.5 + pct_change * 2))
        components.append(r_norm)

    if rank_pct_now is not None:
        components.append(rank_pct_now)

    return round(sum(components) / len(components), 3) if components else 0.5


# ---------------------------------------------------------------------------
# Premature-exit detection
# ---------------------------------------------------------------------------

def _detect_premature(
    symbol: str,
    thesis_score: float,
    weights: dict[str, float],
    quality_score: float | None,
    momentum_score: float | None,
    pct_change: float | None,
    primary_driver: str,
) -> tuple[bool, str]:
    """
    Flag as premature if thesis still looks intact but an exit was triggered.
    Hard mechanical stops (stop_loss / trailing_stop) are never flagged —
    those are safety rules that should not be overridden.
    """
    cfg = _pi_config()
    min_thesis_score  = float(cfg.get("premature_exit_min_thesis_score", 0.65))
    min_quality       = float(cfg.get("premature_exit_min_quality",      0.30))
    min_momentum      = float(cfg.get("premature_exit_min_momentum",     0.00))

    # Safety stops are never premature — they protect capital
    hard_stop_weight = weights.get("stop_loss", 0.0) + weights.get("trailing_stop", 0.0)
    if hard_stop_weight >= 0.55:
        return False, ""

    # Path A: confidence collapse is the primary driver but thesis looks intact.
    # Low reliability ≠ bad stock; it means we're uncertain, not that the thesis broke.
    conf_collapse_weight = weights.get("confidence_loss", 0.0)
    if conf_collapse_weight >= 0.40 and thesis_score >= 0.55:
        quality_ok_loose  = quality_score is None or quality_score >= 0.10
        momentum_ok_loose = momentum_score is None or momentum_score >= -0.10
        if quality_ok_loose and momentum_ok_loose:
            parts: list[str] = ["data reliability is low"]
            if quality_score is not None:
                parts.append(f"quality={quality_score:.3f}")
            if momentum_score is not None:
                parts.append(f"momentum={momentum_score:+.3f}")
            if pct_change is not None:
                parts.append(f"return={pct_change:+.1%}")
            return True, (
                f"Confidence collapse on intact thesis — {', '.join(parts)}. "
                f"Low reliability signals uncertainty, not deterioration."
            )

    if thesis_score < min_thesis_score:
        return False, ""

    # Path B: standard premature detection — thesis clearly intact, exit trigger is soft
    quality_ok   = quality_score is not None and quality_score >= min_quality
    momentum_ok  = momentum_score is not None and momentum_score >= min_momentum
    return_ok    = pct_change is None or pct_change >= 0.0

    if quality_ok and momentum_ok and return_ok:
        reasons: list[str] = []
        if quality_score is not None:
            reasons.append(f"quality_score={quality_score:.3f}")
        if momentum_score is not None:
            reasons.append(f"momentum_score={momentum_score:+.3f}")
        if pct_change is not None:
            reasons.append(f"return={pct_change:+.1%}")
        return True, f"Thesis still intact — {', '.join(reasons)} — primary trigger was {DRIVER_LABELS.get(primary_driver, primary_driver)}"

    return False, ""


# ---------------------------------------------------------------------------
# Confidence assessment
# ---------------------------------------------------------------------------

def _confidence(
    weights: dict[str, float],
    is_premature: bool,
    n_signals: int,
) -> str:
    """
    Confidence in the exit recommendation.

    Rules (in priority order):
      1. Hard mechanical stops (stop_loss / trailing_stop dominant) → HIGH — never doubt a stop.
      2. Premature-exit flag → LOW — thesis is intact; exit is questionable.
      3. Confidence collapse dominant → LOW — data unreliability means uncertainty, not certainty.
      4. Strong factor-driven signal → HIGH.
      5. Multiple signals → MEDIUM.
      6. Weak single signal → LOW.
    """
    top_weight          = max(weights.values()) if weights else 0.0
    hard_stop_weight    = weights.get("stop_loss", 0.0) + weights.get("trailing_stop", 0.0)
    conf_collapse_weight = weights.get("confidence_loss", 0.0)

    # Hard stops are always executed with high confidence — they are safety rules
    if hard_stop_weight >= 0.55:
        return "HIGH"

    # Premature exit = questionable exit
    if is_premature:
        return "LOW"

    # Confidence collapse = uncertainty, not a directional signal
    # Even if it's the largest weight, it means we're unsure — cap at LOW
    if conf_collapse_weight >= 0.35:
        return "LOW"

    # Strong factor-driven exit
    if top_weight >= 0.65:
        return "HIGH"

    if n_signals >= 2 or top_weight >= 0.40:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_exit_analysis(
    symbol: str,
    holding: dict,
    metrics: pd.Series | None,
    buy_context: dict | None,
    peak_price: float | None,
    universe_rank_pct: float | None,
) -> ExitAnalysis:
    """
    Compute full exit attribution for one position.

    Call this for any position where state ∈ {EXIT, WATCH}.
    Safe to call for HOLD positions too — weights will reflect signal magnitudes.
    """
    sr = _sell_rules()

    pct_raw = _safe_float(holding.get("percent_change"))
    pct_change = pct_raw / 100.0 if pct_raw is not None else None
    if pct_change is None:
        avg = _safe_float(holding.get("average_buy_price"))
        cur = _safe_float(holding.get("current_price"))
        if avg and avg > 0 and cur:
            pct_change = (cur / avg) - 1.0

    current_price  = _safe_float(holding.get("current_price"))
    value_metric   = _safe_float(metrics.get("value_metric"))   if metrics is not None else None
    quality_score  = _safe_float(metrics.get("quality_score"))  if metrics is not None else None
    momentum_score = _safe_float(metrics.get("momentum_score")) if metrics is not None else None
    yield_trap     = bool(metrics.get("yield_trap_flag", False)) if metrics is not None else False
    reliability    = _safe_float(metrics.get("reliability_score")) if metrics is not None else None

    score_at_buy  = _safe_float(buy_context.get("composite_score_at_buy"))       if buy_context else None
    rank_at_buy   = _safe_float(buy_context.get("universe_rank_pct_at_buy"))     if buy_context else None

    raw     = _compute_raw_weights(
        pct_change, current_price, peak_price,
        momentum_score, quality_score, value_metric,
        score_at_buy, universe_rank_pct, rank_at_buy,
        reliability, yield_trap, sr,
    )
    weights = _normalize(raw)

    primary, secondary = _top_two(weights)
    n_signals = sum(1 for v in raw.values() if v > 0.05)

    thesis_score = _thesis_intact_score(quality_score, momentum_score, pct_change, universe_rank_pct)
    is_premature, premature_reason = _detect_premature(
        symbol, thesis_score, weights, quality_score, momentum_score, pct_change, primary
    )
    confidence = _confidence(weights, is_premature, n_signals)

    override = "WATCH" if is_premature else ""

    return ExitAnalysis(
        symbol=symbol,
        primary_driver=primary,
        secondary_driver=secondary,
        reason_weights=weights,
        primary_label=DRIVER_LABELS.get(primary, primary),
        secondary_label=DRIVER_LABELS.get(secondary, secondary),
        thesis_intact_score=thesis_score,
        is_premature=is_premature,
        premature_reason=premature_reason,
        override_recommendation=override,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Portfolio-level premature exit rate
# ---------------------------------------------------------------------------

def compute_premature_exit_rate(analyses: list[ExitAnalysis]) -> float:
    """Fraction of exits flagged as potentially premature."""
    if not analyses:
        return 0.0
    return round(sum(1 for a in analyses if a.is_premature) / len(analyses), 3)
