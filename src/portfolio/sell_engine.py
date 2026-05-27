"""
portfolio/sell_engine.py — SellDecisionEngine.

Decision hierarchy:
  HARD sells (execute immediately, sentiment cannot override):
    1. stop_loss_pct breached
    2. trailing_stop from peak
    3. yield trap + weak value
    4. quality floor breached

  SOFT sells (sentiment can hold for one cycle):
    5. take_profit_pct reached (unless fundamentally cheap)
    6. value_metric below sell_weak_value_below (after min_days_held)
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import pandas as pd

from core.types import SellDecision
from portfolio.position_archetypes import ArchetypePolicy
from util import ARCHETYPE_PARAMS, EXIT_DECISION_PARAMS, METRIC_THRESHOLD, SELL_RULES, safe_float

logger = logging.getLogger(__name__)


class SellDecisionEngine:
    """
    Evaluates each holding against sell rules and returns SellDecision objects.
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    def evaluate(
        self,
        symbol: str,
        holding: dict,
        metrics_row: Optional[pd.Series],
        peak_price: Optional[float] = None,
        archetype_policy: Optional[ArchetypePolicy] = None,
    ) -> SellDecision:
        # Derive percent_change — Robinhood returns it as a percentage string e.g. "-15.3"
        percent_change: float | None = None
        pct_raw = safe_float(holding.get("percent_change"))
        if pct_raw is not None:
            percent_change = pct_raw / 100.0

        if percent_change is None:
            avg   = safe_float(holding.get("average_buy_price"))
            price = safe_float(holding.get("price"))
            if avg and avg > 0 and price:
                percent_change = (price / avg) - 1.0

        value_metric:    float | None = None
        quality_score:   float | None = None
        yield_trap_flag: bool  | None = None

        if metrics_row is not None:
            value_metric  = safe_float(metrics_row.get("value_metric"))
            quality_score = safe_float(metrics_row.get("quality_score"))
            yt = metrics_row.get("yield_trap_flag")
            if yt is not None:
                try:
                    yield_trap_flag = bool(yt) if not pd.isna(yt) else None
                except Exception:
                    yield_trap_flag = None

        days_held: int | None = None
        try:
            created = holding.get("created_at") or holding.get("initiation_date")
            if created:
                created_dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                days_held = (datetime.datetime.now(datetime.timezone.utc) - created_dt).days
        except Exception:
            pass

        stop_loss   = SELL_RULES["stop_loss_pct"]
        take_profit = SELL_RULES["take_profit_pct"]
        sell_weak   = SELL_RULES["sell_weak_value_below"]
        sell_yt     = SELL_RULES["sell_yield_trap"]
        sell_lq     = SELL_RULES["sell_low_quality_below"]
        min_days    = SELL_RULES["min_days_held_before_value_exit"]

        # Archetype-aware overrides (position management only — hard sells are unchanged)
        _arch_enabled = ARCHETYPE_PARAMS.get("enabled", False)
        if _arch_enabled and archetype_policy is not None:
            take_profit = archetype_policy.harvest_profit_threshold
            min_days    = max(min_days, archetype_policy.minimum_hold_days)

        base = dict(
            percent_change=percent_change,
            value_metric=value_metric,
            quality_score=quality_score,
            yield_trap_flag=yield_trap_flag,
        )

        # ── Hard sells ──────────────────────────────────────────────────────

        if percent_change is not None and percent_change <= stop_loss:
            return SellDecision(
                symbol=symbol,
                should_sell=True,
                reason=f"stop loss breached ({percent_change:.1%} ≤ {stop_loss:.1%})",
                severity="hard",
                exit_type="failure_exit",
                **base,
            )

        trailing_stop = SELL_RULES["trailing_stop_pct"]
        if _arch_enabled and archetype_policy is not None:
            trailing_stop = archetype_policy.trailing_stop_pct
        if peak_price is not None and peak_price > 0:
            current_p = safe_float(holding.get("price"))
            if current_p is not None:
                drawdown = (current_p / peak_price) - 1.0
                if drawdown <= trailing_stop:
                    return SellDecision(
                        symbol=symbol,
                        should_sell=True,
                        reason=f"trailing stop: {drawdown:.1%} from peak ${peak_price:.2f}",
                        severity="hard",
                        exit_type="failure_exit",
                        **base,
                    )

        if sell_yt and yield_trap_flag and value_metric is not None and value_metric < sell_weak:
            return SellDecision(
                symbol=symbol,
                should_sell=True,
                reason=f"yield trap with weak value_metric={value_metric:.3f} < {sell_weak}",
                severity="hard",
                exit_type="failure_exit",
                **base,
            )

        if quality_score is not None and quality_score < sell_lq:
            return SellDecision(
                symbol=symbol,
                should_sell=True,
                reason=f"quality_score {quality_score:.3f} below floor {sell_lq}",
                severity="hard",
                exit_type="failure_exit",
                **base,
            )

        # ── Soft sells ──────────────────────────────────────────────────────

        if percent_change is not None and percent_change >= take_profit:
            floor = SELL_RULES["take_profit_value_floor_multiplier"]
            if value_metric is not None and value_metric >= METRIC_THRESHOLD * floor:
                logger.info(
                    f"{symbol}: take-profit threshold hit ({percent_change:.1%}) "
                    f"but still fundamentally cheap (value_metric={value_metric:.3f}) — holding"
                )
            else:
                return SellDecision(
                    symbol=symbol,
                    should_sell=True,
                    reason=f"take profit triggered ({percent_change:.1%} ≥ {take_profit:.1%})",
                    severity="soft",
                    exit_type="harvest_exit",
                    **base,
                )

        # ── Trim (partial exit) ─────────────────────────────────────────────
        # Fires when: profitable + thesis weakening (score below buy threshold)
        # but not yet collapsed to thesis_exit territory.  Sells only trim_fraction.
        _trim = EXIT_DECISION_PARAMS
        if _trim.get("trim_enabled") and percent_change is not None:
            _trim_min_gain = (
                archetype_policy.trim_profit_threshold
                if _arch_enabled and archetype_policy is not None
                else _trim["trim_min_gain_pct"]
            )
            _trim_fraction  = _trim["trim_fraction"]
            _trim_delta_thr = _trim["trim_score_delta_threshold"]  # e.g. -0.15

            _weakening_threshold = METRIC_THRESHOLD * (1.0 + _trim_delta_thr)  # e.g. 0.8 * 0.85 = 0.68
            _profitable           = percent_change >= _trim_min_gain
            _thesis_weakening     = (
                value_metric is not None
                and value_metric < METRIC_THRESHOLD
                and value_metric >= sell_weak  # not yet thesis_exit territory
                and value_metric < _weakening_threshold
            )

            if _profitable and _thesis_weakening:
                return SellDecision(
                    symbol=symbol,
                    should_sell=True,
                    reason=(
                        f"trim: profit {percent_change:.1%} ≥ {_trim_min_gain:.0%}, "
                        f"value_metric={value_metric:.3f} below buy threshold "
                        f"({_weakening_threshold:.2f}) — partial exit"
                    ),
                    severity="soft",
                    exit_type="trim_exit",
                    trim_fraction=_trim_fraction,
                    **base,
                )

        if value_metric is not None and value_metric < sell_weak:
            if days_held is None or days_held >= min_days:
                days_str = f"{days_held}d" if days_held is not None else "unknown days"
                return SellDecision(
                    symbol=symbol,
                    should_sell=True,
                    reason=f"value_metric={value_metric:.3f} < {sell_weak} (held {days_str})",
                    severity="soft",
                    exit_type="thesis_exit",
                    **base,
                )

        return SellDecision(
            symbol=symbol,
            should_sell=False,
            reason="no sell condition met",
            severity=None,
            exit_type=None,
            **base,
        )

    def evaluate_holdings(
        self,
        holdings: dict,
        agg_df: Optional[pd.DataFrame],
        peak_prices: dict[str, float],
        etfs: set[str],
    ) -> tuple[dict[str, SellDecision], dict[str, SellDecision]]:
        """
        Evaluate all holdings. Return (hard_sells, soft_sells).
        ETF positions are excluded — handled by ETF MA filter separately.
        """
        hard: dict[str, SellDecision] = {}
        soft: dict[str, SellDecision] = {}

        for symbol, data in holdings.items():
            if symbol in etfs:
                continue
            if float(data.get("quantity", 0)) <= 0:
                continue

            metrics_row = None
            if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
                row = agg_df[agg_df["symbol"] == symbol]
                if not row.empty:
                    metrics_row = row.iloc[0]

            decision = self.evaluate(symbol, data, metrics_row, peak_prices.get(symbol))
            if not decision.should_sell:
                continue
            if decision.severity == "hard":
                hard[symbol] = decision
            else:
                soft[symbol] = decision

        return hard, soft


# ---------------------------------------------------------------------------
# Convenience wrapper — backward-compatible with main.py's dict-returning API
# ---------------------------------------------------------------------------

_engine = SellDecisionEngine()


def evaluate_sell_candidate(
    symbol: str,
    holding: dict,
    metrics_row: "Optional[pd.Series]",
    peak_price: "Optional[float]" = None,
    archetype_policy: "Optional[ArchetypePolicy]" = None,
) -> dict:
    """
    Module-level wrapper around SellDecisionEngine.evaluate() that returns a
    plain dict. Keeps main.py callers working while the class-based API is
    the canonical interface going forward.
    """
    d = _engine.evaluate(symbol, holding, metrics_row, peak_price, archetype_policy)
    return {
        "should_sell":    d.should_sell,
        "reason":         d.reason,
        "severity":       d.severity,
        "exit_type":      d.exit_type,
        "trim_fraction":  d.trim_fraction,
        "percent_change": d.percent_change,
        "value_metric":   d.value_metric,
        "quality_score":  d.quality_score,
        "yield_trap_flag": d.yield_trap_flag,
    }
