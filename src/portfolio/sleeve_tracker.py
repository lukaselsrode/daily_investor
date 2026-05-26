"""
portfolio/sleeve_tracker.py — Capital source event log for sleeve accounting.

Records every discrete capital movement (contribution, exit proceeds, trim
proceeds, harvest proceeds, cash sweep) to data/sleeve_events.parquet.

ARCHITECTURE CONTRACT
─────────────────────
This file is DIAGNOSTICS / RESEARCH data only.
It is NEVER read back into buy/sell decisions, factor scoring, or alpha signals.
It powers the UI allocation diagnostics page and sleeve drift monitoring.

Event types
───────────
  weekly_contribution  — scheduled cash deposit into the portfolio
  exit_proceeds        — full position exit (stop-loss, thesis, harvest)
  trim_proceeds        — partial position exit (trim_exit)
  harvest_proceeds     — take-profit full exit routed to ETFs
  cash_sweep           — end-of-run cash routed to ETF sleeve to hit index_pct target
  etf_buy              — capital deployed into ETF sleeve (any source)
  stock_buy            — capital deployed into active sleeve

Schema columns
──────────────
  timestamp, event_date, event_type, source_symbol, destination,
  amount, etf_pct_routed, active_pct_routed, notes
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from execution.base import BrokerAdapter

logger = logging.getLogger(__name__)

_SCHEMA: list[str] = [
    "timestamp",
    "event_date",
    "event_type",      # weekly_contribution | exit_proceeds | trim_proceeds |
                       # harvest_proceeds | cash_sweep | etf_buy | stock_buy
    "source_symbol",   # symbol being exited/trimmed (None for contributions)
    "destination",     # "etf_sleeve" | "active_sleeve" | "cash" | "mixed"
    "amount",          # dollar amount of the event
    "etf_pct_routed",  # fraction routed to ETF sleeve (0–1)
    "active_pct_routed",  # fraction routed to active sleeve (0–1)
    "notes",           # free-text annotation
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    try:
        from ui.utils import DATA_DIR
        return DATA_DIR
    except Exception:
        return Path(__file__).parent.parent.parent / "data"


def _events_path() -> Path:
    return _data_dir() / "sleeve_events.parquet"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_sleeve_events() -> pd.DataFrame:
    """Load all recorded sleeve events. Returns empty DataFrame if none exist."""
    path = _events_path()
    if not path.exists():
        return pd.DataFrame(columns=_SCHEMA)
    try:
        df = pd.read_parquet(path)
        for col in _SCHEMA:
            if col not in df.columns:
                df[col] = None
        return df[_SCHEMA]
    except Exception as exc:
        logger.warning("Could not load sleeve_events.parquet: %s", exc)
        return pd.DataFrame(columns=_SCHEMA)


def _save_events(df: pd.DataFrame) -> None:
    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except Exception as exc:
        logger.warning("Could not write sleeve_events.parquet: %s", exc)


# ---------------------------------------------------------------------------
# Record a single event
# ---------------------------------------------------------------------------

def record_event(
    event_type: str,
    amount: float,
    *,
    source_symbol: Optional[str] = None,
    destination: Optional[str]   = None,
    etf_pct_routed: Optional[float]    = None,
    active_pct_routed: Optional[float] = None,
    notes: Optional[str]               = None,
) -> None:
    """Append one sleeve capital event to sleeve_events.parquet."""
    now = datetime.datetime.now(datetime.timezone.utc)
    row = {
        "timestamp":          now.isoformat(),
        "event_date":         now.strftime("%Y-%m-%d"),
        "event_type":         event_type,
        "source_symbol":      source_symbol,
        "destination":        destination,
        "amount":             float(amount),
        "etf_pct_routed":     float(etf_pct_routed)    if etf_pct_routed    is not None else None,
        "active_pct_routed":  float(active_pct_routed) if active_pct_routed is not None else None,
        "notes":              notes,
    }
    existing = load_sleeve_events()
    updated  = pd.concat([existing, pd.DataFrame([row], columns=_SCHEMA)], ignore_index=True)
    _save_events(updated)


# ---------------------------------------------------------------------------
# Allocation state snapshot
# ---------------------------------------------------------------------------

def get_allocation_state(broker: "BrokerAdapter") -> dict:
    """
    Return a point-in-time view of sleeve allocation.

    Returns a dict with:
      total_equity      — portfolio total value
      etf_equity        — current value of ETF sleeve
      active_equity     — current value of active (stock) sleeve
      cash              — uninvested cash
      etf_pct           — etf_equity / total_equity
      active_pct        — active_equity / total_equity
      cash_pct          — cash / total_equity
      target_etf_pct    — INDEX_PCT from config (the target)
      etf_drift_pct     — actual etf_pct - target_etf_pct (positive = overweight ETFs)
    """
    from util import ETFS, INDEX_PCT

    try:
        holdings   = broker.get_holdings()
        total      = broker.get_portfolio_value()
        cash       = broker.get_cash()
    except Exception as exc:
        logger.warning("sleeve_tracker: could not fetch broker state: %s", exc)
        return {}

    etf_equity    = 0.0
    active_equity = 0.0

    for symbol, data in holdings.items():
        try:
            equity = float(data.get("equity") or 0.0)
        except (TypeError, ValueError):
            equity = 0.0

        if symbol in ETFS:
            etf_equity += equity
        else:
            active_equity += equity

    denom = max(total, 1e-9)

    return {
        "total_equity":    total,
        "etf_equity":      etf_equity,
        "active_equity":   active_equity,
        "cash":            cash,
        "etf_pct":         etf_equity    / denom,
        "active_pct":      active_equity / denom,
        "cash_pct":        cash          / denom,
        "target_etf_pct":  INDEX_PCT,
        "etf_drift_pct":   (etf_equity / denom) - INDEX_PCT,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers — call these from manager.py / harvest.py
# ---------------------------------------------------------------------------

def log_contribution(amount: float, notes: Optional[str] = None) -> None:
    record_event(
        "weekly_contribution", amount,
        destination="mixed",
        etf_pct_routed=None,
        active_pct_routed=None,
        notes=notes,
    )


def log_exit_proceeds(
    symbol: str,
    amount: float,
    etf_pct: float = 0.0,
    notes: Optional[str] = None,
) -> None:
    record_event(
        "exit_proceeds", amount,
        source_symbol=symbol,
        destination="etf_sleeve" if etf_pct >= 0.99 else ("mixed" if etf_pct > 0 else "cash"),
        etf_pct_routed=etf_pct,
        active_pct_routed=1.0 - etf_pct,
        notes=notes,
    )


def log_trim_proceeds(
    symbol: str,
    amount: float,
    etf_pct: float,
    notes: Optional[str] = None,
) -> None:
    record_event(
        "trim_proceeds", amount,
        source_symbol=symbol,
        destination="mixed",
        etf_pct_routed=etf_pct,
        active_pct_routed=1.0 - etf_pct,
        notes=notes,
    )


def log_harvest_proceeds(
    symbol: str,
    amount: float,
    etf_pct: float,
    notes: Optional[str] = None,
) -> None:
    record_event(
        "harvest_proceeds", amount,
        source_symbol=symbol,
        destination="etf_sleeve" if etf_pct >= 0.99 else "mixed",
        etf_pct_routed=etf_pct,
        active_pct_routed=1.0 - etf_pct,
        notes=notes,
    )


def log_cash_sweep(amount: float, notes: Optional[str] = None) -> None:
    record_event(
        "cash_sweep", amount,
        destination="etf_sleeve",
        etf_pct_routed=1.0,
        active_pct_routed=0.0,
        notes=notes,
    )
