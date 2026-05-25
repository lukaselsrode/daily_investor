"""
portfolio/outcome_tracker.py — Decision Outcome Tracker.

Records every portfolio decision with its factor snapshot to
data/decision_outcomes.parquet.

On subsequent runs, fills in future returns (7d / 30d / 90d) for
past decisions whose horizon has elapsed.

This data feeds the calibration engine — it is NEVER read back
into factor scoring or composite formula computation.

Schema
──────
ticker, decision_date, decision_state, decision_confidence,
thesis_intact_score, premature_exit_probability,
composite_score, value_score, quality_score, momentum_score, income_score,
rank_percentile, holding_days, portfolio_pnl, regime_cluster,
future_7d_return, future_30d_return, future_90d_return,
outperformed_spy_30d, is_premature_exit  (bool)
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = [
    "ticker", "decision_date", "decision_state", "decision_confidence",
    "thesis_intact_score", "premature_exit_probability",
    "composite_score", "value_score", "quality_score",
    "momentum_score", "income_score", "rank_percentile",
    "holding_days", "portfolio_pnl", "regime_cluster",
    "price_at_decision",
    "future_7d_return", "future_30d_return", "future_90d_return",
    "outperformed_spy_30d", "is_premature_exit",
]

_FLOAT_COLS = [
    "thesis_intact_score", "premature_exit_probability",
    "composite_score", "value_score", "quality_score",
    "momentum_score", "income_score", "rank_percentile",
    "holding_days", "portfolio_pnl", "price_at_decision",
    "future_7d_return", "future_30d_return", "future_90d_return",
]


def _outcomes_path() -> Path:
    try:
        from ui.utils import DATA_DIR
        return DATA_DIR / "decision_outcomes.parquet"
    except Exception:
        return Path(__file__).parent.parent.parent / "data" / "decision_outcomes.parquet"


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_outcomes() -> pd.DataFrame:
    """Load all recorded decision outcomes. Returns empty DataFrame if none exist."""
    path = _outcomes_path()
    if not path.exists():
        return pd.DataFrame(columns=_SCHEMA)
    try:
        df = pd.read_parquet(path)
        # Ensure all schema columns present
        for col in _SCHEMA:
            if col not in df.columns:
                df[col] = None
        return df
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


# ---------------------------------------------------------------------------
# Record a new decision
# ---------------------------------------------------------------------------

def record_decision(
    ticker: str,
    decision_output,           # DecisionOutput from DecisionAdjustmentEngine
    metrics,                   # pd.Series from agg_data (factor snapshot, read-only)
    holding_days: Optional[int],
    portfolio_pnl: Optional[float],
    regime_cluster: str = "unknown",
    price_at_decision: Optional[float] = None,
) -> None:
    """
    Append one decision record to decision_outcomes.parquet.

    Future return columns are left null — filled by fill_future_returns().
    Called from the main trading loop after each decision, never from the UI.
    """
    def _sf(v):
        try:
            import math
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    row = {
        "ticker":                    ticker,
        "decision_date":             datetime.date.today().isoformat(),
        "decision_state":            getattr(decision_output, "action",      "UNKNOWN"),
        "decision_confidence":       getattr(decision_output, "confidence",  "UNKNOWN"),
        "thesis_intact_score":       getattr(decision_output, "thesis_intact_score", None),
        "premature_exit_probability":getattr(decision_output, "premature_exit_probability", None),
        "composite_score":           _sf(metrics.get("value_metric"))    if metrics is not None else None,
        "value_score":               _sf(metrics.get("value_score"))     if metrics is not None else None,
        "quality_score":             _sf(metrics.get("quality_score"))   if metrics is not None else None,
        "momentum_score":            _sf(metrics.get("momentum_score"))  if metrics is not None else None,
        "income_score":              _sf(metrics.get("income_score"))    if metrics is not None else None,
        "rank_percentile":           None,   # populated by caller if available
        "holding_days":              holding_days,
        "portfolio_pnl":             portfolio_pnl,
        "regime_cluster":            regime_cluster,
        "price_at_decision":         price_at_decision,
        "future_7d_return":          None,
        "future_30d_return":         None,
        "future_90d_return":         None,
        "outperformed_spy_30d":      None,
        "is_premature_exit":         getattr(decision_output, "premature_exit_probability", 0.0) >= 0.45,
    }

    existing = load_outcomes()
    new_row  = pd.DataFrame([row])
    updated  = pd.concat([existing, new_row], ignore_index=True)
    _save_outcomes(updated)


# ---------------------------------------------------------------------------
# Fill future returns for past decisions
# ---------------------------------------------------------------------------

def fill_future_returns(
    current_prices: dict[str, float],
    spy_current_price: Optional[float] = None,
    spy_price_history: Optional[dict[str, float]] = None,
) -> int:
    """
    Fill in future_7d_return, future_30d_return, future_90d_return for past
    decisions whose horizon has elapsed.

    current_prices: {ticker: current_price}
    spy_price_history: {date_str: spy_price} — used for SPY comparison

    Returns number of rows updated.
    """
    df = load_outcomes()
    if df.empty:
        return 0

    today = datetime.date.today()
    n_updated = 0

    for idx, row in df.iterrows():
        decision_date_str = str(row.get("decision_date", ""))
        try:
            decision_date = datetime.date.fromisoformat(decision_date_str)
        except ValueError:
            continue

        days_ago = (today - decision_date).days
        ticker   = str(row.get("ticker", ""))
        entry_px = row.get("price_at_decision")

        if not ticker or entry_px is None or float(entry_px) <= 0:
            continue

        current_px = current_prices.get(ticker)
        if current_px is None:
            continue

        # Fill whichever horizon has elapsed but is still null
        for horizon, col in [(7, "future_7d_return"), (30, "future_30d_return"), (90, "future_90d_return")]:
            if days_ago >= horizon and pd.isna(row.get(col)):
                ret = (current_px / float(entry_px)) - 1.0
                df.at[idx, col] = round(ret, 5)
                n_updated += 1

        # SPY outperformance (30d horizon)
        if days_ago >= 30 and pd.isna(row.get("outperformed_spy_30d")):
            ticker_30d = row.get("future_30d_return")
            if ticker_30d is not None and spy_current_price is not None and spy_price_history:
                spy_entry = spy_price_history.get(decision_date_str)
                if spy_entry and float(spy_entry) > 0:
                    spy_30d = (spy_current_price / float(spy_entry)) - 1.0
                    df.at[idx, "outperformed_spy_30d"] = float(ticker_30d) > spy_30d

    if n_updated > 0:
        _save_outcomes(df)

    return n_updated
