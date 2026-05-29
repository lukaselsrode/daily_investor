"""
portfolio/decision_logger.py — High-level decision logging adapter.

Called from main.py (trading loop) to assemble full decision records
and persist them to decision_outcomes.parquet via outcome_tracker.

ARCHITECTURE CONTRACT
─────────────────────
• Reads from factor engine outputs (read-only)
• Calls DAE to determine final action
• Writes to outcome_tracker — NEVER modifies factor scores or weights
• Never imported by UI, scoring, or backtest layers
"""

from __future__ import annotations

import datetime
import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


def _sf(v) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _get_dae_action(
    raw_action: str,
    raw_reason: str,
    holding: dict,
    metrics_row,
    buy_context_row: dict | None,
    exit_severity: str | None,
    exit_type: str | None,
) -> tuple[str, object]:
    """
    Call the Decision Adjustment Engine and return (final_action, decision_output).
    Falls back to (raw_action, None) on any error so logging never crashes the bot.
    """
    try:
        from portfolio.decision_adjustment_engine import (
            DecisionAdjustmentEngine,
            build_decision_input,
        )
        inp = build_decision_input(
            raw_action=raw_action,
            raw_reason=raw_reason,
            holding=holding,
            metrics=metrics_row,
            buy_context=buy_context_row,
            exit_analysis=None,
            exit_severity=exit_severity,
            exit_type=exit_type,
        )
        out = DecisionAdjustmentEngine().adjust(inp)
        return out.action, out
    except Exception as exc:
        logger.debug("DAE call failed in decision_logger: %s", exc)
        return raw_action, None


def _universe_rank_pct(symbol: str, agg_df) -> float | None:
    """Compute symbol's value_metric percentile in today's universe."""
    if agg_df is None or agg_df.empty or "value_metric" not in agg_df.columns:
        return None
    try:
        sym_row = agg_df[agg_df["symbol"] == symbol]
        if sym_row.empty:
            return None
        vm = _sf(sym_row.iloc[0].get("value_metric"))
        if vm is None:
            return None
        all_vm = pd.to_numeric(agg_df["value_metric"], errors="coerce").dropna()
        return round(float((all_vm < vm).mean()), 4) if len(all_vm) > 1 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Holding decision logger
# ---------------------------------------------------------------------------

def log_holding_decision(
    symbol: str,
    holding: dict,
    metrics_row,
    raw_decision: dict,
    executed: bool,
    order_id: str | None,
    regime: str,
    buy_context_row: dict | None,
    agg_df=None,
    soft_sell_held: bool = False,
    archetype_result=None,
) -> None:
    """
    Assemble and persist a holding evaluation record.

    Parameters
    ----------
    symbol          : ticker
    holding         : Robinhood holding dict
    metrics_row     : pd.Series from agg_data (or None) — read-only
    raw_decision    : dict from evaluate_sell_candidate()
    executed        : was the sell order actually placed?
    order_id        : Robinhood order ID if available
    regime          : current market regime string
    buy_context_row : row dict from buy_context.csv for this symbol (or None)
    agg_df          : full agg_data DataFrame for rank computation
    soft_sell_held  : True if a soft-sell was blocked by sentiment
    """
    try:
        _log_holding_inner(
            symbol, holding, metrics_row, raw_decision, executed,
            order_id, regime, buy_context_row, agg_df, soft_sell_held,
            archetype_result=archetype_result,
        )
    except Exception as exc:
        logger.warning("log_holding_decision failed for %s: %s", symbol, exc)


def _log_holding_inner(
    symbol, holding, metrics_row, raw_decision, executed,
    order_id, regime, buy_context_row, agg_df, soft_sell_held,
    archetype_result=None,
) -> None:
    from portfolio.outcome_tracker import record_decision_holding

    should_sell  = raw_decision.get("should_sell", False)
    raw_reason   = raw_decision.get("reason", "")
    exit_sev     = raw_decision.get("severity")
    exit_type    = raw_decision.get("exit_type")
    pct_change   = raw_decision.get("percent_change")

    # ── Determine raw and final actions ──────────────────────────────────────
    if not should_sell:
        raw_action = "HOLD"
    else:
        raw_action = "EXIT"

    if soft_sell_held:
        raw_action = "EXIT"   # original signal was exit

    final_action, decision_output = _get_dae_action(
        raw_action=raw_action,
        raw_reason=raw_reason,
        holding=holding,
        metrics_row=metrics_row,
        buy_context_row=buy_context_row,
        exit_severity=exit_sev,
        exit_type=exit_type,
    )

    if soft_sell_held and final_action == "EXIT":
        final_action = "REVIEW"

    # ── Factor snapshot ───────────────────────────────────────────────────────
    def _m(key):
        return _sf(metrics_row.get(key)) if metrics_row is not None else None

    current_vm    = _m("value_metric")
    score_at_buy  = _sf(buy_context_row.get("composite_score_at_buy"))  if buy_context_row else None
    rank_at_buy   = _sf(buy_context_row.get("universe_rank_pct_at_buy")) if buy_context_row else None
    rank_now      = _universe_rank_pct(symbol, agg_df)
    score_delta   = (current_vm - score_at_buy) if (current_vm is not None and score_at_buy is not None) else None
    rank_delta    = (rank_at_buy - rank_now) if (rank_at_buy is not None and rank_now is not None) else None

    # ── Holding context ───────────────────────────────────────────────────────
    holding_days: int | None = None
    buy_date_str: str | None = None
    try:
        created = holding.get("created_at") or holding.get("initiation_date")
        if created:
            cd = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
            holding_days = (datetime.datetime.now(datetime.timezone.utc) - cd).days
    except Exception:
        pass

    if buy_context_row:
        buy_date_str = str(buy_context_row.get("buy_date", "") or "").strip() or None
        if holding_days is None and buy_date_str:
            try:
                bd = datetime.date.fromisoformat(buy_date_str)
                holding_days = (datetime.date.today() - bd).days
            except Exception:
                pass

    # ── DAE diagnostics ───────────────────────────────────────────────────────
    tis  = getattr(decision_output, "thesis_intact_score",        None) if decision_output else None
    pep  = getattr(decision_output, "premature_exit_probability",  None) if decision_output else None
    conf = getattr(decision_output, "confidence",                  None) if decision_output else None

    # ── Sector / industry from metrics ────────────────────────────────────────
    sector   = str(metrics_row.get("sector",   "") or "") if metrics_row is not None else None
    industry = str(metrics_row.get("industry", "") or "") if metrics_row is not None else None

    record_decision_holding(
        symbol          = symbol,
        decision_state  = final_action,
        raw_signal      = raw_action,
        final_action    = final_action,
        executed_bool   = executed,
        order_id        = order_id,
        price           = _sf(holding.get("price") or holding.get("current_price")),
        equity          = _sf(holding.get("equity")),
        percent_change  = pct_change,
        holding_days    = holding_days,
        buy_date        = buy_date_str,
        entry_price     = _sf(holding.get("average_buy_price")),
        current_value_metric        = current_vm,
        score_at_buy                = score_at_buy,
        score_delta                 = score_delta,
        value_score                 = _m("value_score"),
        quality_score               = _m("quality_score"),
        income_score                = _m("income_score"),
        momentum_score              = _m("momentum_score"),
        conditional_momentum_score  = _m("conditional_momentum_score"),
        rank_percentile             = rank_now,
        rank_at_buy                 = rank_at_buy,
        rank_delta                  = rank_delta,
        thesis_intact_score         = tis,
        exit_confidence             = conf,
        premature_exit_probability  = pep,
        primary_exit_driver         = raw_reason or None,
        regime          = regime,
        sector          = sector or None,
        industry        = industry or None,
        reliability_score = _m("reliability_score"),
        yield_trap_flag   = raw_decision.get("yield_trap_flag"),
        archetype          = getattr(archetype_result, "archetype",   None) if archetype_result else None,
        archetype_confidence = getattr(archetype_result, "confidence", None) if archetype_result else None,
        archetype_drivers  = getattr(archetype_result, "drivers",     None) if archetype_result else None,
        archetype_at_entry = (
            (buy_context_row or {}).get("archetype_at_buy")
            or (buy_context_row or {}).get("archetype")
        ),
        archetype_at_exit  = getattr(archetype_result, "archetype",   None) if archetype_result else None,
        decision_source    = raw_decision.get("decision_source") or None,
    )


# ---------------------------------------------------------------------------
# Candidate decision logger
# ---------------------------------------------------------------------------

def log_candidate_decision(
    symbol: str,
    row: pd.Series,
    decision_state: str,
    selected_bool: bool,
    skipped_bool: bool,
    skip_reason: str,
    sentiment_result_dict: dict | None,
    risk_check_passed: bool,
    risk_check_fail_reason: str,
    proposed_allocation: float,
    final_allocation: float,
    regime: str,
    candidate_rank: int,
    agg_df=None,
) -> None:
    """
    Assemble and persist a buy-candidate evaluation record.

    Parameters match the candidate evaluation state at the point of decision.
    Never raises — exceptions are caught and logged.
    """
    try:
        _log_candidate_inner(
            symbol, row, decision_state, selected_bool, skipped_bool, skip_reason,
            sentiment_result_dict, risk_check_passed, risk_check_fail_reason,
            proposed_allocation, final_allocation, regime, candidate_rank, agg_df,
        )
    except Exception as exc:
        logger.warning("log_candidate_decision failed for %s: %s", symbol, exc)


def _log_candidate_inner(
    symbol, row, decision_state, selected_bool, skipped_bool, skip_reason,
    sentiment_result_dict, risk_check_passed, risk_check_fail_reason,
    proposed_allocation, final_allocation, regime, candidate_rank, agg_df,
) -> None:
    from portfolio.outcome_tracker import record_decision_candidate

    def _r(key):
        v = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
        return _sf(v)

    sent_action  = sentiment_result_dict.get("action")  if sentiment_result_dict else None
    sent_conf    = _sf(sentiment_result_dict.get("confidence")) if sentiment_result_dict else None
    rank_pct     = _universe_rank_pct(symbol, agg_df)

    record_decision_candidate(
        symbol          = symbol,
        decision_state  = decision_state,
        selected_bool   = selected_bool,
        skipped_bool    = skipped_bool,
        skip_reason     = skip_reason or None,
        current_value_metric    = _r("value_metric"),
        value_score             = _r("value_score"),
        quality_score           = _r("quality_score"),
        income_score            = _r("income_score"),
        momentum_score          = _r("momentum_score"),
        rank_percentile         = rank_pct,
        candidate_rank          = candidate_rank,
        sentiment_result        = sent_action,
        sentiment_confidence    = sent_conf,
        risk_check_passed       = risk_check_passed,
        risk_check_fail_reason  = risk_check_fail_reason or None,
        proposed_allocation     = _sf(proposed_allocation),
        final_allocation        = _sf(final_allocation),
        regime                  = regime,
        reliability_score       = _r("reliability_score"),
    )
