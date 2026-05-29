"""
portfolio/outcome_tracker.py — Decision Outcome Tracker.

Records every portfolio decision (holding evaluations + buy candidate
evaluations) to data/decision_outcomes.parquet.

On subsequent runs, fill_future_returns() backfills realized outcomes for
past decisions whose horizon has elapsed.

ARCHITECTURE CONTRACT
─────────────────────
This file is CALIBRATION / RESEARCH data storage only.
It is NEVER read back into factor scoring, composite formula, or alpha signals.

Schema
──────
Two record types share one parquet file, distinguished by `record_type`:
  "holding"   — every active position evaluated each run
  "candidate" — every buy candidate evaluated each run

Future outcome columns (backfilled by fill_future_returns):
  future_7d_return / future_30d_return / future_90d_return
  future_7d_vs_spy / future_30d_vs_spy / future_90d_vs_spy
  outperformed_hold / premature_exit / bad_hold / good_trim / good_exit
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA: list[str] = [
    # ── Identity + decision ──────────────────────────────────────────────────
    "record_type",              # "holding" | "candidate"
    "timestamp",                # ISO 8601 datetime (precise)
    "decision_date",            # YYYY-MM-DD
    "symbol",
    "company_name",
    "sleeve",                   # "active" | "ETF/core" (holding only)
    "decision_state",           # BUY/HOLD/WATCH/REVIEW/TRIM/HARVEST/EXIT/SKIP
    "raw_signal",               # raw action before DAE adjustment
    "final_action",             # action after DAE (same as decision_state for simple cases)
    "executed_bool",            # was trade actually placed?
    "order_id",
    # ── Price / P&L (holding) ────────────────────────────────────────────────
    "price",
    "equity",
    "percent_change",
    "equity_change",
    "holding_days",
    "buy_date",
    "entry_price",
    # ── Factor snapshot (both) ───────────────────────────────────────────────
    "current_value_metric",
    "score_at_buy",
    "score_delta",
    "value_score",
    "quality_score",
    "income_score",
    "momentum_score",
    "conditional_momentum_score",
    "rank_percentile",
    "rank_at_buy",
    "rank_delta",
    # ── Diagnostics (holding) ────────────────────────────────────────────────
    "thesis_intact_score",
    "exit_confidence",
    "premature_exit_probability",
    "primary_exit_driver",
    "secondary_exit_driver",
    "exit_reason_weights",      # JSON string
    "rationale_text",
    # ── Context ──────────────────────────────────────────────────────────────
    "regime",
    "sector",
    "industry",
    "reliability_score",
    "yield_trap_flag",
    # ── Candidate-only ───────────────────────────────────────────────────────
    "selected_bool",
    "skipped_bool",
    "skip_reason",
    "candidate_rank",
    "sentiment_result",
    "sentiment_confidence",
    "risk_check_passed",
    "risk_check_fail_reason",
    "proposed_allocation",
    "final_allocation",
    # ── Partial exit details (TRIM decisions) ───────────────────────────────
    "trim_fraction",             # fraction of position sold (e.g. 0.33)
    "quantity_sold",             # shares sold in a trim
    "quantity_remaining",        # shares kept after trim
    # ── Cluster / regime context ─────────────────────────────────────────────
    "cluster",                   # optional market cluster label (string)
    # ── Archetype classification ─────────────────────────────────────────────
    "archetype",                 # classified archetype label at decision time
    "archetype_confidence",      # float 0–1 confidence in the classification
    "archetype_drivers",         # JSON string list of driver strings
    "archetype_at_entry",        # archetype label captured at original buy
    "archetype_at_exit",         # archetype label at the exit decision
    "decision_source",           # "global_rule" | "archetype_rule" | "both" | ""
    # ── Post-decision price path (backfilled later) ──────────────────────────
    "max_drawdown_after_decision",  # worst drawdown within 90d of decision
    "max_runup_after_decision",     # best runup within 90d of decision
    # ── Outcomes (backfilled later) ──────────────────────────────────────────
    "future_7d_return",
    "future_30d_return",
    "future_90d_return",
    "future_7d_vs_spy",
    "future_30d_vs_spy",
    "future_90d_vs_spy",
    "outperformed_hold",        # True if stock outperformed SPY 30d
    "premature_exit",           # True if EXIT → stock gained ≥2% after
    "bad_hold",                 # True if HOLD → stock lost ≥5% while SPY flat/up
    "good_trim",                # True if TRIM → stock declined or underperformed SPY
    "good_exit",                # True if EXIT → stock declined and underperformed SPY
]

# Columns that hold SPY reference prices for VS-SPY computation
_SPY_HORIZON_COLS = {
    7:  ("future_7d_return",  "future_7d_vs_spy"),
    30: ("future_30d_return", "future_30d_vs_spy"),
    90: ("future_90d_return", "future_90d_vs_spy"),
}

_JOURNAL_COLS = [
    "timestamp", "decision_date", "symbol", "record_type", "decision_state",
    "final_action", "executed_bool", "percent_change", "current_value_metric",
    "holding_days", "regime", "rationale_text",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    try:
        from core.paths import DATA_DIR
        return DATA_DIR
    except Exception:
        return Path(__file__).parent.parent.parent / "data"


def _outcomes_path() -> Path:
    return _data_dir() / "decision_outcomes.parquet"


def _journal_path() -> Path:
    return _data_dir() / "position_journal.csv"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_outcomes() -> pd.DataFrame:
    """Load all recorded decision outcomes. Returns empty DataFrame if none exist."""
    path = _outcomes_path()
    if not path.exists():
        return pd.DataFrame(columns=_SCHEMA)
    try:
        df = pd.read_parquet(path)
        for col in _SCHEMA:
            if col not in df.columns:
                df[col] = None
        return df[_SCHEMA]
    except Exception as exc:
        logger.warning("Could not load decision_outcomes.parquet: %s", exc)
        return pd.DataFrame(columns=_SCHEMA)


def _save_outcomes(df: pd.DataFrame) -> None:
    path = _outcomes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except Exception as exc:
        logger.warning("Could not write decision_outcomes.parquet: %s", exc)


def _sf(v) -> float | None:
    import math
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Record a holding decision
# ---------------------------------------------------------------------------

def record_decision_holding(
    symbol: str,
    decision_state: str,
    raw_signal: str,
    final_action: str,
    executed_bool: bool,
    *,
    timestamp: str | None   = None,
    company_name: str | None = None,
    sleeve: str | None       = None,
    order_id: str | None     = None,
    price: float | None      = None,
    equity: float | None     = None,
    percent_change: float | None = None,
    equity_change: float | None  = None,
    holding_days: int | None     = None,
    buy_date: str | None         = None,
    entry_price: float | None    = None,
    current_value_metric: float | None = None,
    score_at_buy: float | None   = None,
    score_delta: float | None    = None,
    value_score: float | None    = None,
    quality_score: float | None  = None,
    income_score: float | None   = None,
    momentum_score: float | None = None,
    conditional_momentum_score: float | None = None,
    rank_percentile: float | None = None,
    rank_at_buy: float | None    = None,
    rank_delta: float | None     = None,
    thesis_intact_score: float | None       = None,
    exit_confidence: str | None             = None,
    premature_exit_probability: float | None = None,
    primary_exit_driver: str | None         = None,
    secondary_exit_driver: str | None       = None,
    exit_reason_weights: dict | None        = None,
    rationale_text: str | None              = None,
    regime: str | None     = None,
    sector: str | None     = None,
    industry: str | None   = None,
    reliability_score: float | None = None,
    yield_trap_flag: bool | None    = None,
    trim_fraction: float | None     = None,
    quantity_sold: float | None     = None,
    quantity_remaining: float | None = None,
    cluster: str | None             = None,
    archetype: str | None           = None,
    archetype_confidence: float | None = None,
    archetype_drivers: list | None  = None,
    archetype_at_entry: str | None  = None,
    archetype_at_exit: str | None   = None,
    decision_source: str | None     = None,
) -> None:
    """Append one holding-evaluation record to decision_outcomes.parquet."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ts  = timestamp or now.isoformat()
    date_str = now.strftime("%Y-%m-%d")

    row: dict = {col: None for col in _SCHEMA}
    row.update({
        "record_type":    "holding",
        "timestamp":      ts,
        "decision_date":  date_str,
        "symbol":         symbol,
        "company_name":   company_name,
        "sleeve":         sleeve,
        "decision_state": decision_state,
        "raw_signal":     raw_signal,
        "final_action":   final_action,
        "executed_bool":  executed_bool,
        "order_id":       order_id,
        "price":          price,
        "equity":         equity,
        "percent_change": percent_change,
        "equity_change":  equity_change,
        "holding_days":   holding_days,
        "buy_date":       buy_date,
        "entry_price":    entry_price,
        "current_value_metric":        current_value_metric,
        "score_at_buy":                score_at_buy,
        "score_delta":                 score_delta,
        "value_score":                 value_score,
        "quality_score":               quality_score,
        "income_score":                income_score,
        "momentum_score":              momentum_score,
        "conditional_momentum_score":  conditional_momentum_score,
        "rank_percentile":             rank_percentile,
        "rank_at_buy":                 rank_at_buy,
        "rank_delta":                  rank_delta,
        "thesis_intact_score":         thesis_intact_score,
        "exit_confidence":             exit_confidence,
        "premature_exit_probability":  premature_exit_probability,
        "primary_exit_driver":         primary_exit_driver,
        "secondary_exit_driver":       secondary_exit_driver,
        "exit_reason_weights":         json.dumps(exit_reason_weights) if exit_reason_weights else None,
        "rationale_text":              rationale_text,
        "regime":          regime,
        "sector":          sector,
        "industry":        industry,
        "reliability_score": reliability_score,
        "yield_trap_flag":   yield_trap_flag,
        "trim_fraction":     trim_fraction,
        "quantity_sold":     quantity_sold,
        "quantity_remaining": quantity_remaining,
        "cluster":           cluster,
        "archetype":              archetype,
        "archetype_confidence":   archetype_confidence,
        "archetype_drivers":      json.dumps(archetype_drivers) if archetype_drivers else None,
        "archetype_at_entry":     archetype_at_entry,
        "archetype_at_exit":      archetype_at_exit,
        "decision_source":        decision_source,
    })

    _append_row(row)
    _append_journal_row(row)


# ---------------------------------------------------------------------------
# Record a candidate decision
# ---------------------------------------------------------------------------

def record_decision_candidate(
    symbol: str,
    decision_state: str,
    selected_bool: bool,
    skipped_bool: bool,
    *,
    timestamp: str | None     = None,
    skip_reason: str | None   = None,
    current_value_metric: float | None = None,
    value_score: float | None = None,
    quality_score: float | None = None,
    income_score: float | None  = None,
    momentum_score: float | None = None,
    rank_percentile: float | None = None,
    candidate_rank: int | None   = None,
    sentiment_result: str | None = None,
    sentiment_confidence: float | None = None,
    risk_check_passed: bool | None      = None,
    risk_check_fail_reason: str | None  = None,
    proposed_allocation: float | None   = None,
    final_allocation: float | None      = None,
    regime: str | None          = None,
    reliability_score: float | None = None,
) -> None:
    """Append one candidate-evaluation record to decision_outcomes.parquet."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ts  = timestamp or now.isoformat()
    date_str = now.strftime("%Y-%m-%d")

    row: dict = {col: None for col in _SCHEMA}
    row.update({
        "record_type":    "candidate",
        "timestamp":      ts,
        "decision_date":  date_str,
        "symbol":         symbol,
        "decision_state": decision_state,
        "raw_signal":     decision_state,
        "final_action":   decision_state,
        "executed_bool":  selected_bool,
        "current_value_metric": current_value_metric,
        "value_score":    value_score,
        "quality_score":  quality_score,
        "income_score":   income_score,
        "momentum_score": momentum_score,
        "rank_percentile": rank_percentile,
        "regime":          regime,
        "reliability_score": reliability_score,
        "selected_bool":   selected_bool,
        "skipped_bool":    skipped_bool,
        "skip_reason":     skip_reason,
        "candidate_rank":  candidate_rank,
        "sentiment_result":         sentiment_result,
        "sentiment_confidence":     sentiment_confidence,
        "risk_check_passed":        risk_check_passed,
        "risk_check_fail_reason":   risk_check_fail_reason,
        "proposed_allocation":      proposed_allocation,
        "final_allocation":         final_allocation,
    })

    _append_row(row)
    _append_journal_row(row)


# ---------------------------------------------------------------------------
# Append helpers
# ---------------------------------------------------------------------------

def _append_row(row: dict) -> None:
    existing = load_outcomes()
    new_df   = pd.DataFrame([row], columns=_SCHEMA)
    updated  = pd.concat([existing, new_df], ignore_index=True)

    # Dedup: drop identical (symbol, timestamp, decision_state) rows
    dedup_keys = ["symbol", "timestamp", "decision_state"]
    if all(k in updated.columns for k in dedup_keys):
        updated = updated.drop_duplicates(subset=dedup_keys, keep="last")

    _save_outcomes(updated)


def _append_journal_row(row: dict) -> None:
    """Append a human-readable line to position_journal.csv."""
    try:
        path = _journal_path()
        cols = [c for c in _JOURNAL_COLS if c in row]
        line = pd.DataFrame([{c: row.get(c) for c in cols}])
        header = not path.exists()
        line.to_csv(path, mode="a", header=header, index=False)
    except Exception as exc:
        logger.debug("Journal append failed: %s", exc)


# ---------------------------------------------------------------------------
# Backward-compat alias (old signature)
# ---------------------------------------------------------------------------

def record_decision(
    ticker: str,
    decision_output,
    metrics,
    holding_days: int | None,
    portfolio_pnl: float | None,
    regime_cluster: str = "unknown",
    price_at_decision: float | None = None,
) -> None:
    """
    Legacy wrapper — kept for any existing callers.
    Prefers record_decision_holding() for new code.
    """
    record_decision_holding(
        symbol=ticker,
        decision_state=getattr(decision_output, "action", "UNKNOWN"),
        raw_signal=getattr(decision_output, "raw_sell_trigger", ""),
        final_action=getattr(decision_output, "action", "UNKNOWN"),
        executed_bool=False,
        price=price_at_decision,
        percent_change=portfolio_pnl,
        holding_days=holding_days,
        current_value_metric=_sf(metrics.get("value_metric") if metrics is not None else None),
        value_score=_sf(metrics.get("value_score")    if metrics is not None else None),
        quality_score=_sf(metrics.get("quality_score")  if metrics is not None else None),
        momentum_score=_sf(metrics.get("momentum_score") if metrics is not None else None),
        income_score=_sf(metrics.get("income_score")   if metrics is not None else None),
        thesis_intact_score=getattr(decision_output, "thesis_intact_score",        None),
        premature_exit_probability=getattr(decision_output, "premature_exit_probability", None),
        regime=regime_cluster,
    )


# ---------------------------------------------------------------------------
# Fill future returns (backfill job)
# ---------------------------------------------------------------------------

def fill_future_returns(
    current_prices: dict[str, float],
    spy_current_price: float | None = None,
    spy_price_history: dict[str, float] | None = None,
) -> int:
    """
    Fill in realized outcome columns for past decisions.

    current_prices       : {ticker: current_price}  — read from yfinance by caller
    spy_current_price    : today's SPY price
    spy_price_history    : {date_str: spy_price}  — for SPY vs-comparison

    Returns number of rows updated.

    SAFETY: this function writes ONLY to the parquet file.
    It does NOT touch config, factor weights, or calibration state.
    """
    df = load_outcomes()
    if df.empty:
        return 0

    today    = datetime.date.today()
    n_updated = 0

    for idx, row in df.iterrows():
        decision_date_str = str(row.get("decision_date", ""))
        try:
            decision_date = datetime.date.fromisoformat(decision_date_str)
        except ValueError:
            continue

        days_ago   = (today - decision_date).days
        ticker     = str(row.get("symbol") or row.get("ticker", ""))
        entry_px   = _sf(row.get("price") or row.get("price_at_decision"))
        decision_state = str(row.get("decision_state", ""))

        if not ticker or entry_px is None or entry_px <= 0:
            continue

        current_px = current_prices.get(ticker)
        if current_px is None:
            continue

        # ── Per-horizon return + SPY comparison ──────────────────────────────
        for horizon, (ret_col, spy_col) in _SPY_HORIZON_COLS.items():
            if days_ago >= horizon and pd.isna(row.get(ret_col)):
                ret = (current_px / entry_px) - 1.0
                df.at[idx, ret_col] = round(ret, 5)
                n_updated += 1

                # VS-SPY comparison
                if spy_current_price and spy_price_history:
                    spy_entry = spy_price_history.get(decision_date_str)
                    if spy_entry and float(spy_entry) > 0:
                        spy_ret = (spy_current_price / float(spy_entry)) - 1.0
                        df.at[idx, spy_col] = round(ret - spy_ret, 5)

        # ── Summary outcome booleans (need 30d) ───────────────────────────────
        if days_ago >= 30:
            ret_30d = _sf(df.at[idx, "future_30d_return"])
            if ret_30d is None:
                continue
            spy_30d_vs: float | None = _sf(df.at[idx, "future_30d_vs_spy"])

            # outperformed_hold: did this holding beat SPY?
            if pd.isna(row.get("outperformed_hold")) and spy_30d_vs is not None:
                df.at[idx, "outperformed_hold"] = spy_30d_vs > 0

            # premature_exit: EXIT decision, but stock gained ≥2% after
            if decision_state == "EXIT" and pd.isna(row.get("premature_exit")):
                df.at[idx, "premature_exit"] = ret_30d > 0.02
                n_updated += 1

            # bad_hold: HOLD decision, stock fell ≥5% while SPY was flat/up
            if decision_state == "HOLD" and pd.isna(row.get("bad_hold")):
                spy_r = _sf(row.get("future_30d_vs_spy"))
                df.at[idx, "bad_hold"] = ret_30d < -0.05 and (spy_r is None or spy_r <= 0.02)
                n_updated += 1

            # good_trim: TRIM decision, stock subsequently declined or underperformed
            if decision_state == "TRIM" and pd.isna(row.get("good_trim")):
                df.at[idx, "good_trim"] = ret_30d < 0.0 or (spy_30d_vs is not None and spy_30d_vs < 0)
                n_updated += 1

            # good_exit: EXIT decision, stock declined AND underperformed SPY
            if decision_state == "EXIT" and pd.isna(row.get("good_exit")):
                df.at[idx, "good_exit"] = ret_30d < 0.0 and (spy_30d_vs is None or spy_30d_vs < 0)
                n_updated += 1

    if n_updated > 0:
        _save_outcomes(df)

    return n_updated


# ---------------------------------------------------------------------------
# Calibration summary — research / diagnostics only
# NEVER fed back into live scoring or factor weights.
# ---------------------------------------------------------------------------

def get_calibration_summary(df: pd.DataFrame | None = None) -> dict:
    """
    Compute calibration metrics from recorded decision outcomes.

    Returns a dict with the following keys (all floats 0–1, or None if no data):
      premature_exit_rate  — fraction of EXIT decisions where stock gained ≥2% after
      trim_success_rate    — fraction of TRIM decisions where stock subsequently declined
                             or underperformed SPY (i.e. trim was well-timed)
      harvest_regret_rate  — fraction of HARVEST decisions where stock continued rising ≥10%
                             after the exit (regret = sold too early)
      bad_hold_rate        — fraction of HOLD decisions where stock fell ≥5% while SPY flat/up
      n_exit               — number of EXIT decisions with resolved outcomes
      n_trim               — number of TRIM decisions with resolved outcomes
      n_harvest            — number of HARVEST decisions with resolved outcomes
      n_hold               — number of HOLD decisions with resolved outcomes
    """
    if df is None:
        df = load_outcomes()

    if df.empty:
        return {
            "premature_exit_rate": None, "trim_success_rate": None,
            "harvest_regret_rate": None, "bad_hold_rate": None,
            "n_exit": 0, "n_trim": 0, "n_harvest": 0, "n_hold": 0,
        }

    def _rate(mask_state: str, bool_col: str, invert: bool = False) -> tuple[float | None, int]:
        subset = df[df["decision_state"] == mask_state]
        resolved = subset[bool_col].dropna()
        if resolved.empty:
            return None, 0
        rate = float(resolved.astype(bool).mean())
        return (1.0 - rate if invert else rate), len(resolved)

    premature_exit_rate, n_exit    = _rate("EXIT",    "premature_exit")
    trim_success_rate,   n_trim    = _rate("TRIM",    "good_trim")
    bad_hold_rate,       n_hold    = _rate("HOLD",    "bad_hold")

    # Harvest regret: HARVEST exit + stock continued ≥10% (premature_exit proxy at higher bar)
    harvest_subset = df[df["decision_state"] == "HARVEST"]
    h_resolved = harvest_subset["future_30d_return"].dropna()
    if h_resolved.empty:
        harvest_regret_rate, n_harvest = None, 0
    else:
        harvest_regret_rate = float((h_resolved > 0.10).mean())
        n_harvest = len(h_resolved)

    return {
        "premature_exit_rate": premature_exit_rate,
        "trim_success_rate":   trim_success_rate,
        "harvest_regret_rate": harvest_regret_rate,
        "bad_hold_rate":       bad_hold_rate,
        "n_exit":              n_exit,
        "n_trim":              n_trim,
        "n_harvest":           n_harvest,
        "n_hold":              n_hold,
    }


def get_archetype_calibration_summary(df: pd.DataFrame | None = None) -> dict:
    """
    Break down calibration metrics by archetype label.

    Returns a dict keyed by archetype label, each value a dict with:
      trim_success_rate, premature_exit_rate, bad_hold_rate, harvest_regret_rate,
      avg_30d_return, avg_holding_days, n_trim, n_exit, n_hold, n_harvest, n_total

    Only archetypes with at least one resolved record are included.
    NEVER fed back into live scoring or factor weights.
    """
    if df is None:
        df = load_outcomes()

    if df.empty or "archetype" not in df.columns:
        return {}

    result: dict = {}
    archetypes = df["archetype"].dropna().unique()

    for arch in archetypes:
        sub = df[df["archetype"] == arch]
        if sub.empty:
            continue

        def _rate(state: str, bool_col: str, _sub=sub) -> tuple[float | None, int]:
            s = _sub[_sub["decision_state"] == state]
            resolved = s[bool_col].dropna() if bool_col in s.columns else pd.Series(dtype=float)
            if resolved.empty:
                return None, 0
            return float(resolved.astype(bool).mean()), len(resolved)

        trim_rate,  n_trim  = _rate("TRIM",    "good_trim")
        exit_rate,  n_exit  = _rate("EXIT",    "premature_exit")
        hold_rate,  n_hold  = _rate("HOLD",    "bad_hold")

        h_sub = sub[sub["decision_state"] == "HARVEST"]
        if "future_30d_return" in h_sub.columns:
            h_resolved = h_sub["future_30d_return"].dropna()
            harvest_regret = float((h_resolved > 0.10).mean()) if not h_resolved.empty else None
            n_harvest = len(h_resolved)
        else:
            harvest_regret, n_harvest = None, 0

        avg_return: float | None = None
        if "future_30d_return" in sub.columns:
            ret_vals = sub["future_30d_return"].dropna()
            if not ret_vals.empty:
                avg_return = round(float(ret_vals.mean()), 5)

        avg_hold_days: float | None = None
        if "holding_days" in sub.columns:
            hd_vals = pd.to_numeric(sub["holding_days"], errors="coerce").dropna()
            if not hd_vals.empty:
                avg_hold_days = round(float(hd_vals.mean()), 1)

        result[arch] = {
            "trim_success_rate":   trim_rate,
            "premature_exit_rate": exit_rate,
            "bad_hold_rate":       hold_rate,
            "harvest_regret_rate": harvest_regret,
            "avg_30d_return":      avg_return,
            "avg_holding_days":    avg_hold_days,
            "n_trim":              n_trim,
            "n_exit":              n_exit,
            "n_hold":              n_hold,
            "n_harvest":           n_harvest,
            "n_total":             len(sub),
        }

    return result
