"""
portfolio/position_rationale.py — Deterministic position rationale engine.

Produces:
  - PositionState: BUY / HOLD / WATCH / EXIT
  - PositionRationale: structured data + human-readable strings
  - ETF role classification
  - Factor decomposition per holding

All logic is deterministic — no LLM, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# ETF role map
# ---------------------------------------------------------------------------

ETF_ROLES: dict[str, str] = {
    "SPY":  "Core broad-market ETF (S&P 500)",
    "VOO":  "Core broad-market ETF — Vanguard S&P 500",
    "VTI":  "Core total US stock market",
    "IVV":  "Core broad-market ETF (iShares S&P 500)",
    "QQQ":  "Growth / technology tilt (Nasdaq-100)",
    "SCHD": "Dividend-quality stabilizer",
    "SMH":  "Semiconductor / AI infrastructure satellite",
    "SOXX": "Semiconductor sector satellite",
    "VXUS": "International diversification",
    "EFA":  "Developed-market international exposure",
    "VEA":  "Developed-market international — Vanguard",
    "VWO":  "Emerging-market diversification",
    "VNQ":  "Real estate / rate-sensitive diversifier",
    "SCHH": "Real estate satellite (REIT)",
    "IWM":  "Small-cap / risk-on exposure",
    "IJR":  "Small-cap exposure (iShares S&P 600)",
    "BND":  "Investment-grade bond allocation",
    "AGG":  "Core bond / duration exposure",
    "TLT":  "Long-duration Treasury / rate hedge",
    "GLD":  "Gold / inflation hedge",
    "IAU":  "Gold allocation (iShares)",
    "DIA":  "Large-cap Dow Jones blend",
    "XLK":  "Technology sector tilt",
    "XLV":  "Healthcare sector tilt",
    "XLE":  "Energy sector tilt",
    "XLF":  "Financial sector tilt",
}


def etf_role(symbol: str) -> str:
    return ETF_ROLES.get(symbol.upper(), f"ETF / index exposure ({symbol})")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pct(val: float | None) -> str:
    return f"{val:+.1%}" if val is not None else "—"


def _score(val: float | None) -> str:
    return f"{val:.3f}" if val is not None else "—"


# ---------------------------------------------------------------------------
# Dataclass outputs
# ---------------------------------------------------------------------------

@dataclass
class PositionRationale:
    symbol: str
    sleeve: str
    state: str                               # BUY / HOLD / WATCH / REVIEW / EXIT
    state_reason: str                        # one-liner explanation
    rationale: str                           # full prose
    top_positive_factor: str
    top_negative_factor: str
    risk_flags: list[str]
    next_action: str
    exit_reason: str                         # non-empty for EXIT/REVIEW
    factor_contributions: dict[str, float]   # {factor_name: contribution}
    thesis_intact: bool
    score_at_buy: float | None
    score_now: float | None
    score_delta: float | None
    rank_pct_at_buy: float | None
    rank_pct_now: float | None
    exit_analysis: object | None = None   # ExitAnalysis (populated for EXIT/WATCH/REVIEW)
    decision_output: object | None = None # DecisionOutput from DecisionAdjustmentEngine


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------

def _pi_config() -> dict:
    try:
        from config.manager import ConfigManager
        raw = ConfigManager.get()._raw
        return raw.get("portfolio_intelligence", {})
    except Exception:
        pass
    try:
        from core.utils import load_config_raw
        return load_config_raw().get("portfolio_intelligence", {})
    except Exception:
        return {}


def _sell_rules() -> dict:
    try:
        from util import SELL_RULES
        return SELL_RULES
    except Exception:
        return {
            "stop_loss_pct": -0.20,
            "trailing_stop_pct": -0.08,
            "sell_weak_value_below": 0.45,
            "sell_low_quality_below": -0.25,
            "sell_yield_trap": True,
            "min_days_held_before_value_exit": 21,
        }


def _metric_threshold() -> float:
    try:
        from util import METRIC_THRESHOLD
        return METRIC_THRESHOLD
    except Exception:
        return 0.8


def _should_downgrade_to_watch(
    reliability: float | None,
    quality_score: float | None,
    momentum_score: float | None,
    pct_change: float | None,
) -> bool:
    """
    Return True if a soft (thesis-based) exit should be downgraded to WATCH.

    Rationale: low confidence ≠ bad stock.  Low data reliability means we are
    uncertain about the exit trigger, not that the thesis has collapsed.
    Only applies when fundamental signals are not clearly deteriorating.
    Hard mechanical exits (stop_loss, trailing_stop) are never touched here.
    """
    cfg = _pi_config()
    reliability_threshold = float(cfg.get("confidence_collapse_reliability_threshold", 0.60))

    # Reliability is acceptable — no downgrade needed
    if reliability is None or reliability >= reliability_threshold:
        return False

    # Thesis fundamentals must look at least neutral to justify downgrade
    quality_ok  = quality_score is None or quality_score >= 0.10
    momentum_ok = momentum_score is None or momentum_score >= -0.15
    return_ok   = pct_change is None or pct_change >= -0.08

    return quality_ok and momentum_ok and return_ok


# ---------------------------------------------------------------------------
# State classifier
# ---------------------------------------------------------------------------

def classify_state(
    symbol: str,
    holding: dict,
    metrics: pd.Series | None,
    buy_context: dict | None,
    peak_price: float | None,
    universe_rank_pct: float | None = None,
) -> tuple[str, str]:
    """
    Returns (state, reason) where state ∈ {BUY, HOLD, WATCH, EXIT}.
    Deterministic — no side effects.
    """
    cfg  = _pi_config()
    sr   = _sell_rules()
    mth  = _metric_threshold()

    watch_score_drop = float(cfg.get("watch_score_drop_pct",    0.25))
    watch_rank_drop  = float(cfg.get("watch_rank_drop_pct",     0.30))
    exit_score_below = float(cfg.get("exit_score_below",        sr["sell_weak_value_below"]))
    exit_rank_below  = float(cfg.get("exit_rank_below_pct",     0.40))
    min_watch_days   = int(cfg.get("min_holding_days_before_watch", 5))

    pct_raw = _safe_float(holding.get("percent_change"))
    pct_change = pct_raw / 100.0 if pct_raw is not None else None
    if pct_change is None:
        avg = _safe_float(holding.get("average_buy_price"))
        cur = _safe_float(holding.get("current_price"))
        if avg and avg > 0 and cur:
            pct_change = (cur / avg) - 1.0

    value_metric   = _safe_float(metrics.get("value_metric"))   if metrics is not None else None
    quality_score  = _safe_float(metrics.get("quality_score"))  if metrics is not None else None
    momentum_score = _safe_float(metrics.get("momentum_score")) if metrics is not None else None
    yield_trap     = bool(metrics.get("yield_trap_flag", False)) if metrics is not None else False
    reliability    = _safe_float(metrics.get("reliability_score")) if metrics is not None else None

    # Holding days
    import datetime
    days_held: int | None = None
    if buy_context:
        buy_date_str = str(buy_context.get("buy_date", "")).strip()
        try:
            bd = datetime.date.fromisoformat(buy_date_str)
            days_held = (datetime.date.today() - bd).days
        except Exception:
            pass

    score_at_buy = _safe_float(buy_context.get("composite_score_at_buy")) if buy_context else None
    rank_at_buy  = _safe_float(buy_context.get("universe_rank_pct_at_buy")) if buy_context else None

    # ── EXIT checks (mirrors SellDecisionEngine) ─────────────────────────────

    if pct_change is not None and pct_change <= sr["stop_loss_pct"]:
        return "EXIT", f"Stop loss: {pct_change:+.1%}"

    if peak_price is not None and peak_price > 0:
        cur = _safe_float(holding.get("current_price"))
        if cur is not None:
            drawdown = (cur / peak_price) - 1.0
            if drawdown <= sr["trailing_stop_pct"]:
                return "EXIT", f"Trailing stop: {drawdown:+.1%} from peak ${peak_price:.2f}"

    if sr.get("sell_yield_trap") and yield_trap and value_metric is not None and value_metric < exit_score_below:
        return "EXIT", f"Yield trap + weak score ({value_metric:.3f})"

    if quality_score is not None and quality_score < sr["sell_low_quality_below"]:
        return "EXIT", f"Quality floor breached ({quality_score:.3f})"

    if value_metric is not None and value_metric < exit_score_below:
        min_days = sr["min_days_held_before_value_exit"]
        if days_held is None or days_held >= min_days:
            if _should_downgrade_to_watch(reliability, quality_score, momentum_score, pct_change):
                return "WATCH", (
                    f"Soft exit moderated — reliability {reliability:.2f} is low; "
                    f"score {value_metric:.3f} is below threshold but quality and momentum look OK. "
                    f"Treat as uncertainty, not deterioration."
                )
            return "EXIT", f"Score below exit threshold ({value_metric:.3f} < {exit_score_below})"

    if universe_rank_pct is not None and universe_rank_pct < exit_rank_below:
        if days_held is None or days_held >= min_watch_days:
            if _should_downgrade_to_watch(reliability, quality_score, momentum_score, pct_change):
                return "WATCH", (
                    f"Soft rank exit moderated — reliability {reliability:.2f} is low; "
                    f"rank {100*universe_rank_pct:.0f}th percentile but fundamentals appear intact."
                )
            return "EXIT", f"Rank in bottom {100*(1-universe_rank_pct):.0f}% of universe"

    # ── WATCH checks ─────────────────────────────────────────────────────────

    watch_reasons: list[str] = []

    if score_at_buy is not None and value_metric is not None:
        if score_at_buy > 0:
            drop = (value_metric - score_at_buy) / abs(score_at_buy)
            if drop < -watch_score_drop:
                watch_reasons.append(f"Score dropped {drop:+.1%} since buy")

    if rank_at_buy is not None and universe_rank_pct is not None:
        drop = rank_at_buy - universe_rank_pct
        if drop > watch_rank_drop:
            watch_reasons.append(f"Rank dropped {drop:.0%} since buy")

    if momentum_score is not None and momentum_score < -0.15:
        watch_reasons.append("Momentum weakening")

    if reliability is not None and reliability < 0.6:
        watch_reasons.append(f"Low data reliability ({reliability:.2f})")

    if days_held is not None and days_held >= min_watch_days and watch_reasons:
        return "WATCH", "; ".join(watch_reasons)

    # ── BUY check ────────────────────────────────────────────────────────────

    if value_metric is not None and value_metric >= mth:
        if universe_rank_pct is None or universe_rank_pct >= 0.70:
            return "BUY", f"Model would re-buy today (score {value_metric:.3f})"

    # ── Default: HOLD ─────────────────────────────────────────────────────────

    return "HOLD", "Thesis intact — no buy or sell signal"


# ---------------------------------------------------------------------------
# Factor decomposition
# ---------------------------------------------------------------------------

def factor_contributions(metrics: pd.Series | None) -> dict[str, float]:
    """Return {factor_label: contribution} from current agg_data row."""
    if metrics is None:
        return {}
    try:
        from util import SCORE_WEIGHTS
    except Exception:
        SCORE_WEIGHTS = {"value": 0.05, "quality": 0.45, "income": 0.05, "momentum": 0.45}

    result: dict[str, float] = {}
    mapping = [
        ("Quality",  "quality_score",  SCORE_WEIGHTS.get("quality",  0.45)),
        ("Momentum", "momentum_score", SCORE_WEIGHTS.get("momentum", 0.45)),
        ("Value",    "value_score",    SCORE_WEIGHTS.get("value",    0.05)),
        ("Income",   "income_score",   SCORE_WEIGHTS.get("income",   0.05)),
    ]
    for label, col, weight in mapping:
        v = _safe_float(metrics.get(col))
        if v is not None:
            result[label] = round(weight * v, 4)
    return result


# ---------------------------------------------------------------------------
# Top factor helpers
# ---------------------------------------------------------------------------

def _top_positive(contribs: dict[str, float]) -> str:
    pos = {k: v for k, v in contribs.items() if v > 0}
    return max(pos, key=pos.__getitem__) if pos else "—"


def _top_negative(contribs: dict[str, float]) -> str:
    neg = {k: v for k, v in contribs.items() if v < 0}
    return min(neg, key=neg.__getitem__) if neg else "—"


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

def build_risk_flags(
    metrics: pd.Series | None,
    holding: dict,
    pct_change: float | None,
    peak_price: float | None,
) -> list[str]:
    flags: list[str] = []
    sr = _sell_rules()

    if metrics is not None:
        if bool(metrics.get("yield_trap_flag", False)):
            flags.append("⚠️ Yield trap")
        qs = _safe_float(metrics.get("quality_score"))
        if qs is not None and qs < 0:
            flags.append("⚠️ Negative quality")
        ms = _safe_float(metrics.get("momentum_score"))
        if ms is not None and ms < -0.3:
            flags.append("⚠️ Strong negative momentum")
        rel = _safe_float(metrics.get("reliability_score"))
        if rel is not None and rel < 0.6:
            flags.append(f"⚠️ Low reliability ({rel:.2f})")
        vm = _safe_float(metrics.get("value_metric"))
        if vm is not None:
            gap = sr["sell_weak_value_below"] - vm
            if 0 < gap < 0.1:
                flags.append(f"⚠️ Near sell threshold (score {vm:.3f})")

    if pct_change is not None:
        if pct_change < sr["stop_loss_pct"] * 0.7:
            flags.append(f"⚠️ Approaching stop loss ({pct_change:+.1%})")
        elif pct_change > 0.40:
            flags.append(f"ℹ️ Large gain — monitor take-profit ({pct_change:+.1%})")

    if peak_price is not None and peak_price > 0:
        cur = _safe_float(holding.get("current_price"))
        if cur is not None:
            dd = (cur / peak_price) - 1.0
            if dd < -0.05:
                flags.append(f"⚠️ {dd:+.1%} from peak ${peak_price:.2f}")

    return flags


# ---------------------------------------------------------------------------
# Prose rationale builder
# ---------------------------------------------------------------------------

def _build_prose(
    symbol: str,
    state: str,
    metrics: pd.Series | None,
    buy_context: dict | None,
    rank_pct: float | None,
    contribs: dict[str, float],
    risk_flags: list[str],
) -> str:
    parts: list[str] = []

    vm = _safe_float(metrics.get("value_metric")) if metrics is not None else None
    qs = _safe_float(metrics.get("quality_score")) if metrics is not None else None
    ms = _safe_float(metrics.get("momentum_score")) if metrics is not None else None

    if buy_context:
        bd = buy_context.get("buy_date", "")
        buy_vm = _safe_float(buy_context.get("composite_score_at_buy"))
        rank_b = _safe_float(buy_context.get("universe_rank_pct_at_buy"))
        if bd:
            parts.append(f"Purchased {bd}")
        if buy_vm is not None:
            parts.append(f"with a composite score of {buy_vm:.3f} at entry")
        if rank_b is not None:
            parts.append(f"(top {100*(1-rank_b):.0f}% of universe at buy)")
    else:
        parts.append("Purchase context unavailable")

    parts.append(".")

    if vm is not None:
        parts.append(f"Current score: {vm:.3f}")
    if rank_pct is not None:
        parts.append(f"(top {100*(1-rank_pct):.0f}% of universe today)")
    if qs is not None:
        q_desc = "strong" if qs >= 0.8 else ("moderate" if qs >= 0 else "weak")
        parts.append(f"Quality is {q_desc} ({qs:.3f})")
    if ms is not None:
        m_desc = "positive" if ms >= 0 else "negative"
        parts.append(f"Momentum is {m_desc} ({ms:.3f})")

    if contribs:
        top = max(contribs, key=lambda k: abs(contribs[k]))
        parts.append(f"Largest factor driver: {top} ({contribs[top]:+.3f} contribution)")

    if risk_flags:
        parts.append("Flags: " + ", ".join(f.replace("⚠️ ", "").replace("ℹ️ ", "") for f in risk_flags))

    return ". ".join(p.strip().rstrip(".") for p in parts if p.strip() not in (".", "")) + "."


# ---------------------------------------------------------------------------
# Next action strings
# ---------------------------------------------------------------------------

_NEXT_ACTION: dict[str, str] = {
    "EXIT":    "Sell — exit condition met",
    "REVIEW":  "Human review required — conflicting exit evidence",
    "TRIM":    "Consider trimming — score weakening but gain + momentum intact",
    "HARVEST": "Harvest profits — large gain with weakening score",
    "WATCH":   "Monitor closely — thesis weakening",
    "HOLD":    "Hold — thesis intact",
    "BUY":     "Could add — model would buy today",
}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_position_rationale(
    symbol: str,
    sleeve: str,
    holding: dict,
    metrics: pd.Series | None,
    buy_context: dict | None,
    peak_price: float | None,
    universe_rank_pct: float | None,
) -> PositionRationale:
    """
    Build a complete PositionRationale for one holding.

    Parameters
    ----------
    symbol          : ticker symbol
    sleeve          : "active" or "ETF/core"
    holding         : row dict from holdings CSV
    metrics         : row from latest agg_data CSV (or None)
    buy_context     : row dict from buy_context.csv (or None)
    peak_price      : from peak_prices.csv (or None)
    universe_rank_pct: 0–1 percentile of value_metric in today's universe
    all_scores      : full agg_data DataFrame (for universe comparisons)
    """
    pct_raw = _safe_float(holding.get("percent_change"))
    pct_change = pct_raw / 100.0 if pct_raw is not None else None
    if pct_change is None:
        avg = _safe_float(holding.get("average_buy_price"))
        cur = _safe_float(holding.get("current_price"))
        if avg and avg > 0 and cur:
            pct_change = (cur / avg) - 1.0

    state, state_reason = classify_state(
        symbol, holding, metrics, buy_context, peak_price, universe_rank_pct
    )

    contribs = factor_contributions(metrics)
    risk_flags = build_risk_flags(metrics, holding, pct_change, peak_price)

    score_now    = _safe_float(metrics.get("value_metric")) if metrics is not None else None
    score_at_buy = _safe_float(buy_context.get("composite_score_at_buy")) if buy_context else None
    score_delta  = (score_now - score_at_buy) if score_now is not None and score_at_buy is not None else None

    rank_at_buy  = _safe_float(buy_context.get("universe_rank_pct_at_buy")) if buy_context else None

    prose = _build_prose(symbol, state, metrics, buy_context, universe_rank_pct, contribs, risk_flags)

    # Exit analysis — computed for EXIT and WATCH; feeds the Decision Adjustment Engine
    exit_analysis = None
    if state in ("EXIT", "WATCH"):
        try:
            from portfolio.exit_analysis import compute_exit_analysis
            exit_analysis = compute_exit_analysis(
                symbol=symbol,
                holding=holding,
                metrics=metrics,
                buy_context=buy_context,
                peak_price=peak_price,
                universe_rank_pct=universe_rank_pct,
            )
        except Exception:
            pass

    # Decision Adjustment Engine — may upgrade WATCH→REVIEW or EXIT→REVIEW.
    # It reads diagnostic features from exit_analysis but NEVER writes factor scores.
    decision_output = None
    final_state = state
    final_reason = state_reason
    try:
        from portfolio.decision_adjustment_engine import (
            DecisionAdjustmentEngine,
            build_decision_input,
        )
        exit_severity = None
        exit_type = None
        if state == "EXIT":
            # Infer exit hardness from the reason string
            reason_lower = state_reason.lower()
            if reason_lower.startswith("stop loss"):
                exit_severity, exit_type = "hard", "stop_loss"
            elif reason_lower.startswith("trailing stop"):
                exit_severity, exit_type = "hard", "trailing_stop"
            elif "quality floor" in reason_lower:
                exit_severity, exit_type = "hard", "quality_floor"
            elif "yield trap" in reason_lower:
                exit_severity, exit_type = "hard", "yield_trap"
            else:
                exit_severity, exit_type = "soft", "soft_thesis"

        di = build_decision_input(
            raw_action=state,
            raw_reason=state_reason,
            holding=holding,
            metrics=metrics,
            buy_context=buy_context,
            exit_analysis=exit_analysis,
            exit_severity=exit_severity,
            exit_type=exit_type,
        )
        di.rank_pct_now = universe_rank_pct

        dae = DecisionAdjustmentEngine()
        decision_output = dae.adjust(di)
        final_state  = decision_output.action
        final_reason = decision_output.adjustment_reason if decision_output.adjustment_applied else state_reason
    except Exception:
        pass

    return PositionRationale(
        symbol=symbol,
        sleeve=sleeve,
        state=final_state,
        state_reason=final_reason,
        rationale=prose,
        top_positive_factor=_top_positive(contribs),
        top_negative_factor=_top_negative(contribs),
        risk_flags=risk_flags,
        next_action=_NEXT_ACTION.get(final_state, "Review"),
        exit_reason=final_reason if final_state in ("EXIT", "REVIEW") else "",
        factor_contributions=contribs,
        thesis_intact=(final_state in ("BUY", "HOLD")),
        score_at_buy=score_at_buy,
        score_now=score_now,
        score_delta=score_delta,
        rank_pct_at_buy=rank_at_buy,
        rank_pct_now=universe_rank_pct,
        exit_analysis=exit_analysis,
        decision_output=decision_output,
    )
