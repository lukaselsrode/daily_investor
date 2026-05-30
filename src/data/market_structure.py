"""
data/market_structure.py — Market structure signal cache.

Fetches and persists per-symbol market structure fields that are NOT in agg_data:
  maintenance_ratio   — margin maintenance requirement (1.0 = can't hold on margin)
  day_trade_ratio     — day-trade margin requirement
  instrument_type     — "stock" | "adr" | "etp" | ...
  country             — ISO country code (CA, FI, US, ...)
  market_cap          — float (USD)
  description         — company description text
  num_employees       — integer headcount

Source: get_instruments_by_symbols() + get_fundamentals() from robin_stocks.
Cache: data/market_structure.csv  (refreshed when entry is > STALE_DAYS old)

Usage:
  from data.market_structure import load_market_structure
  enriched = load_market_structure(["BB", "GOOG"])
  # enriched["GOOG"]["maintenance_ratio"] → 0.25

SAFE: this module is data-fetch only.
Never modifies factor scores, composite formula, or config.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STALE_DAYS = 7

_INSTRUMENT_FIELDS = [
    "maintenance_ratio",
    "day_trade_ratio",
    "country",
    "type",           # stored as instrument_type
    "margin_initial_ratio",
]

_FUNDAMENTAL_FIELDS = [
    "market_cap",
    "description",
    "num_employees",
]

_SCHEMA_COLS = [
    "symbol",
    "fetched_date",
    "maintenance_ratio",
    "day_trade_ratio",
    "instrument_type",
    "country",
    "margin_initial_ratio",
    "market_cap",
    "description",
    "num_employees",
    "analyst_buy_pct",
    "analyst_num_ratings",
]


def _data_dir() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR


def _cache_path() -> Path:
    return _data_dir() / "market_structure.csv"


def _load_cache() -> pd.DataFrame:
    path = _cache_path()
    if not path.exists():
        return pd.DataFrame(columns=_SCHEMA_COLS)
    try:
        df = pd.read_csv(path, dtype=str)
        for col in _SCHEMA_COLS:
            if col not in df.columns:
                df[col] = None
        return df[_SCHEMA_COLS]
    except Exception as exc:
        logger.warning("Could not load market_structure.csv: %s", exc)
        return pd.DataFrame(columns=_SCHEMA_COLS)


def _save_cache(df: pd.DataFrame) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        logger.warning("Could not save market_structure.csv: %s", exc)


def _is_fresh(row: pd.Series) -> bool:
    try:
        fetched = datetime.date.fromisoformat(str(row["fetched_date"]))
        return (datetime.date.today() - fetched).days <= STALE_DAYS
    except Exception:
        return False


def _fetch_instruments(symbols: list[str]) -> dict[str, dict]:
    """Fetch instrument-level fields for a list of symbols."""
    result: dict[str, dict] = {}
    try:
        import robin_stocks.robinhood as rb
        raw = rb.get_instruments_by_symbols(symbols) or []
        for item in raw:
            if not item or "symbol" not in item:
                continue
            sym = item["symbol"]
            result[sym] = {
                "maintenance_ratio":   item.get("maintenance_ratio"),
                "day_trade_ratio":     item.get("day_trade_ratio"),
                "instrument_type":     item.get("type"),
                "country":             item.get("country"),
                "margin_initial_ratio": item.get("margin_initial_ratio"),
            }
    except Exception as exc:
        logger.warning("get_instruments_by_symbols failed: %s", exc)
    return result


def _fetch_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch fundamental-level fields for a list of symbols."""
    result: dict[str, dict] = {}
    batch_size = 50
    try:
        import robin_stocks.robinhood as rb
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                raw = rb.get_fundamentals(batch) or []
                for item in raw:
                    if not item or "symbol" not in item:
                        continue
                    sym = item["symbol"]
                    result[sym] = {
                        "market_cap":    item.get("market_cap"),
                        "description":   item.get("description"),
                        "num_employees": item.get("num_employees"),
                    }
            except Exception as exc:
                logger.warning("get_fundamentals batch %d failed: %s", i, exc)
    except Exception as exc:
        logger.warning("get_fundamentals import failed: %s", exc)
    return result


def _fetch_ratings(symbols: list[str]) -> dict[str, dict]:
    """Fetch real analyst buy/hold/sell counts per symbol via Robinhood get_ratings.

    Returns {symbol: {"analyst_buy_pct": float|None, "analyst_num_ratings": int}}.
    analyst_buy_pct = num_buy / (num_buy + num_hold + num_sell) — the exact
    consensus fraction, replacing the buy_to_sell_ratio bucket heuristic in the
    archetype classifier (which was off by up to ~28pp on real names).

    get_ratings is a standard-tier endpoint (no Gold required). Per-symbol call;
    failures degrade gracefully to no rating (classifier falls back to heuristic).
    """
    result: dict[str, dict] = {}
    try:
        import robin_stocks.robinhood as rb
    except Exception as exc:
        logger.warning("get_ratings import failed: %s", exc)
        return result
    for sym in symbols:
        try:
            raw = rb.get_ratings(sym) or {}
            summary = raw.get("summary") if isinstance(raw, dict) else None
            if not summary:
                continue
            b = int(summary.get("num_buy_ratings", 0) or 0)
            h = int(summary.get("num_hold_ratings", 0) or 0)
            s = int(summary.get("num_sell_ratings", 0) or 0)
            tot = b + h + s
            if tot <= 0:
                continue
            result[sym] = {
                "analyst_buy_pct":     round(b / tot, 4),
                "analyst_num_ratings": tot,
            }
        except Exception as exc:
            logger.debug("get_ratings failed for %s: %s", sym, exc)
    return result


def refresh_market_structure(symbols: list[str]) -> dict[str, dict]:
    """
    Force-fetch market structure data for the given symbols and update the cache.
    Returns a dict {symbol: {field: value}}.
    Only callable when logged in to Robinhood.
    """
    if not symbols:
        return {}

    instruments = _fetch_instruments(symbols)
    fundamentals = _fetch_fundamentals(symbols)
    ratings = _fetch_ratings(symbols)

    today = datetime.date.today().isoformat()
    rows = []
    merged: dict[str, dict] = {}
    for sym in symbols:
        inst = instruments.get(sym, {})
        fund = fundamentals.get(sym, {})
        rate = ratings.get(sym, {})
        combined = {**inst, **fund, **rate}
        merged[sym] = combined
        rows.append({
            "symbol":             sym,
            "fetched_date":       today,
            "maintenance_ratio":  combined.get("maintenance_ratio"),
            "day_trade_ratio":    combined.get("day_trade_ratio"),
            "instrument_type":    combined.get("instrument_type"),
            "country":            combined.get("country"),
            "margin_initial_ratio": combined.get("margin_initial_ratio"),
            "market_cap":         combined.get("market_cap"),
            "description":        combined.get("description"),
            "num_employees":      combined.get("num_employees"),
            "analyst_buy_pct":    combined.get("analyst_buy_pct"),
            "analyst_num_ratings": combined.get("analyst_num_ratings"),
        })

    existing = _load_cache()
    if not existing.empty:
        existing = existing[~existing["symbol"].isin(symbols)]

    updated = pd.concat([existing, pd.DataFrame(rows, columns=_SCHEMA_COLS)], ignore_index=True)
    _save_cache(updated)
    logger.info("market_structure: refreshed %d symbols", len(rows))
    return merged


def load_market_structure(
    symbols: list[str],
    auto_refresh: bool = True,
) -> dict[str, dict]:
    """
    Return market structure fields for the given symbols from cache.
    If auto_refresh=True (default), stale or missing entries are re-fetched.

    Returns {symbol: {"maintenance_ratio": ..., "day_trade_ratio": ..., ...}}
    Missing symbols return an empty dict — callers must handle gracefully.
    """
    cache = _load_cache()

    if cache.empty:
        stale = list(symbols)
    else:
        cached_syms = set(cache["symbol"].dropna())
        stale = [s for s in symbols if s not in cached_syms]
        if not stale:
            # Check freshness for cached entries
            sub = cache[cache["symbol"].isin(symbols)]
            stale = [
                str(r["symbol"]) for _, r in sub.iterrows()
                if not _is_fresh(r)
            ]

    if stale and auto_refresh:
        try:
            refresh_market_structure(stale)
            cache = _load_cache()
        except Exception as exc:
            logger.debug("auto-refresh of market_structure failed: %s", exc)

    result: dict[str, dict] = {}
    if cache.empty:
        return result

    sub = cache[cache["symbol"].isin(symbols)]
    for _, row in sub.iterrows():
        sym = str(row["symbol"])

        def _f(col, row=row):
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) in ("", "None", "nan"):
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        def _s(col, row=row):
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) in ("", "None", "nan"):
                return None
            return str(v)

        def _i(col):
            v = _f(col)
            return int(v) if v is not None else None

        result[sym] = {
            "maintenance_ratio":    _f("maintenance_ratio"),
            "day_trade_ratio":      _f("day_trade_ratio"),
            "instrument_type":      _s("instrument_type"),
            "country":              _s("country"),
            "margin_initial_ratio": _f("margin_initial_ratio"),
            "market_cap":           _f("market_cap"),
            "description":          _s("description"),
            "num_employees":        _i("num_employees"),
            "analyst_buy_pct":      _f("analyst_buy_pct"),
            "analyst_num_ratings":  _i("analyst_num_ratings"),
        }

    return result
