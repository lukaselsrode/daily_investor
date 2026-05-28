"""
portfolio/buy_context.py — Purchase context: scores at buy, rank at buy.

Persists to data/buy_context.csv.

On first access, backfills from:
  - buy_history.csv      (symbol, bought_date)
  - holdings_*.csv       (average_buy_price)
  - data/snapshots/      (scores closest to buy_date)

Missing context is stored as NaN; display layer falls back to "—" gracefully.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CONTEXT_COLS = [
    "symbol",
    "buy_date",
    "buy_price",
    "composite_score_at_buy",
    "quality_score_at_buy",
    "momentum_score_at_buy",
    "income_score_at_buy",
    "value_score_at_buy",
    "universe_rank_pct_at_buy",
    "regime_at_buy",
]

_SCORE_COLS = {
    "composite_score_at_buy": "value_metric",
    "quality_score_at_buy":   "quality_score",
    "momentum_score_at_buy":  "momentum_score",
    "income_score_at_buy":    "income_score",
    "value_score_at_buy":     "value_score",
}


def _context_path() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR / "buy_context.csv"


def _buy_history_path() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR / "buy_history.csv"


def load_buy_context() -> pd.DataFrame:
    """Load existing buy context, or return empty frame with correct columns."""
    path = _context_path()
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning("Could not load buy_context.csv: %s", exc)
    return pd.DataFrame(columns=_CONTEXT_COLS)


def save_buy_context(df: pd.DataFrame) -> None:
    path = _context_path()
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Could not save buy_context.csv: %s", exc)


def _load_buy_history() -> pd.DataFrame:
    path = _buy_history_path()
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "bought_date"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["symbol", "bought_date"])


def _load_buy_prices() -> dict[str, float]:
    """Attempt to get average_buy_price from latest holdings CSV."""
    try:
        from data.cache import read_data_as_pd
        df = read_data_as_pd("holdings")
        if df is not None and "symbol" in df.columns and "average_buy_price" in df.columns:
            df["average_buy_price"] = pd.to_numeric(df["average_buy_price"], errors="coerce")
            return df.dropna(subset=["average_buy_price"]).set_index("symbol")["average_buy_price"].to_dict()
    except Exception:
        pass
    return {}


def _load_all_snapshots() -> dict[datetime.date, pd.DataFrame]:
    """Load all snapshot parquets into a date → DataFrame map."""
    try:
        from strategy.snapshots import list_snapshots
        result: dict[datetime.date, pd.DataFrame] = {}
        for date, path in list_snapshots():
            try:
                result[date] = pd.read_parquet(path)
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _nearest_snapshot(
    target_date: datetime.date,
    snaps: dict[datetime.date, pd.DataFrame],
    max_days: int = 30,
) -> pd.DataFrame | None:
    """Return snapshot closest to target_date within max_days."""
    if not snaps:
        return None
    best: datetime.date | None = None
    best_diff = 9999
    for d in snaps:
        diff = abs((d - target_date).days)
        if diff < best_diff and diff <= max_days:
            best = d
            best_diff = diff
    return snaps[best] if best is not None else None


def backfill_buy_context() -> pd.DataFrame:
    """
    Build/update buy_context.csv from available history and snapshots.
    Returns the updated DataFrame.
    """
    existing = load_buy_context()
    already  = set(existing["symbol"].dropna().tolist()) if "symbol" in existing.columns else set()

    buy_hist  = _load_buy_history()
    prices    = _load_buy_prices()
    snaps     = _load_all_snapshots()

    new_rows: list[dict] = []

    for _, row in buy_hist.iterrows():
        sym = str(row.get("symbol", "")).strip()
        if not sym or sym in already:
            continue

        buy_date_str = str(row.get("bought_date", "")).strip()
        buy_date: datetime.date | None = None
        try:
            buy_date = datetime.date.fromisoformat(buy_date_str)
        except Exception:
            pass

        snap_df = _nearest_snapshot(buy_date, snaps) if buy_date else None
        scores: dict = {}

        if snap_df is not None and "symbol" in snap_df.columns:
            sym_rows = snap_df[snap_df["symbol"] == sym]
            if not sym_rows.empty:
                sym_row = sym_rows.iloc[0]
                for ctx_col, src_col in _SCORE_COLS.items():
                    if src_col in sym_row.index:
                        scores[ctx_col] = float(pd.to_numeric(sym_row[src_col], errors="coerce"))

                # Universe rank percentile at buy
                if "value_metric" in snap_df.columns:
                    vm_series = pd.to_numeric(snap_df["value_metric"], errors="coerce").dropna()
                    sym_vm    = scores.get("composite_score_at_buy")
                    if sym_vm is not None and len(vm_series) > 1:
                        rank_pct = float((vm_series < sym_vm).mean())
                        scores["universe_rank_pct_at_buy"] = round(rank_pct, 3)

        new_rows.append({
            "symbol":            sym,
            "buy_date":          buy_date_str,
            "buy_price":         prices.get(sym),
            **{c: scores.get(c) for c in _SCORE_COLS},
            "universe_rank_pct_at_buy": scores.get("universe_rank_pct_at_buy"),
            "regime_at_buy":     None,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=_CONTEXT_COLS)
        combined = pd.concat([existing, new_df], ignore_index=True)
        save_buy_context(combined)
        return combined

    return existing
