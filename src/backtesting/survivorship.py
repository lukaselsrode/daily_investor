"""
backtesting/survivorship.py — Survivorship-free price/universe inputs from the FMP cache.

The default loader (yfinance + current agg_data) only ever sees today's survivors, so backtests
are inflated by the ~35% of names that delisted. This module assembles a survivorship-FREE input
set from the split-adjusted FMP cache (data/fmp_cache_adj/): the current universe PLUS the liquid
delisted names (prices to their delisting date). It returns the two things load_and_precompute
needs to splice in — a `closes` frame and an extended `agg_df` — leaving all the downstream
daily-feature/array machinery untouched.

Design choices (v1, documented):
- Point-in-time membership = price availability. A name is selectable on date D iff it has an
  adjusted price on D; dead names' series naturally end at delisting.
- Delisting handling = trailing forward-fill to the final price. A held position marks at its last
  traded value (bankruptcy → ~0 captured; acquisition → deal price), and its crashed/flat momentum
  keeps it from being re-bought. (A later version can liquidate-and-redeploy in the sim instead.)
- Dead-name fundamentals = NEUTRAL (zero) — we don't have point-in-time fundamentals for delisted
  names, so they are scored on price/momentum only, never on quality/value (an honest abstention).
- A causal trailing dollar-volume array is returned for cap-proxy position sizing (no look-ahead,
  unlike the static market_caps snapshot).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

_CACHE_DIR = os.environ.get("FMP_CACHE_DIR", "data/fmp_cache_adj")
_PRICE_DIR = os.path.join(_CACHE_DIR, "prices")
_DEAD_PARQUET = os.path.join(_CACHE_DIR, "dead_universe.parquet")

_NEUTRAL_FUND_COLS = {
    "pe_comp": 0.0, "pb_comp": 0.0, "quality_score": 0.0, "income_score": 0.0,
    "value_metric": 0.0, "position_52w": np.nan, "return_1m": np.nan,
    "baseline_score": 0.0, "market_cap": np.nan,
}


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _load_series(symbol: str) -> pd.DataFrame | None:
    path = os.path.join(_PRICE_DIR, f"{_safe(symbol)}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    return df if "close" in df.columns else None


def dead_universe() -> pd.DataFrame:
    """The pre-scanned liquid delisted names (symbol, first_date, delist_date, max_adv)."""
    if os.path.exists(_DEAD_PARQUET):
        return pd.read_parquet(_DEAD_PARQUET)
    logger.warning("dead_universe.parquet missing — run the dead-universe scan; no dead names added")
    return pd.DataFrame(columns=["symbol", "first_date", "delist_date", "max_adv"])


def assemble(agg_df: pd.DataFrame, symbols: list[str], etf_list: list[str],
             benchmark_symbol: str, n_days: int,
             add_dead: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build survivorship-free (closes, extended_agg_df, dollar_volume) from the FMP cache.

    closes:          date×symbol adjusted Close, dead names trailing-ffilled to final price.
    extended_agg_df: agg_df + neutral rows for the added dead names (price/momentum-only scoring).
    dollar_volume:   (n_days, n_stocks) causal Close×Volume, aligned to the final stock column order,
                     for cap-proxy position sizing without the static-market_caps look-ahead.
    """
    # Calendar from the benchmark series (last n_days of trading days FMP has).
    bench = _load_series(benchmark_symbol)
    if bench is None:
        raise RuntimeError(
            f"Benchmark {benchmark_symbol} not in FMP cache ({_PRICE_DIR}). "
            "Backfill it first (data.fmp_client.eod_prices)."
        )
    cal = bench.index.tolist()[-n_days:]

    want = list(dict.fromkeys(symbols + etf_list + [benchmark_symbol]))
    close_cols: dict[str, pd.Series] = {}
    vol_cols: dict[str, pd.Series] = {}
    for sym in want:
        s = _load_series(sym)
        if s is not None:
            close_cols[sym] = s["close"].astype(float)
            vol_cols[sym] = s["volume"].astype(float) if "volume" in s.columns else s["close"] * 0.0

    dead_added: list[str] = []
    if add_dead:
        have = set(close_cols)
        start = cal[0]
        for _, row in dead_universe().iterrows():
            sym = str(row["symbol"])
            if sym in have:
                continue
            if str(row["delist_date"]) <= start:   # delisted before the window starts → never tradeable here
                continue
            s = _load_series(sym)
            if s is None or len(s) < 30:
                continue
            close_cols[sym] = s["close"].astype(float)
            vol_cols[sym] = s["volume"].astype(float) if "volume" in s.columns else s["close"] * 0.0
            dead_added.append(sym)

    closes = pd.DataFrame(close_cols).reindex(cal)
    # Trailing forward-fill ONLY for dead names (mark held positions at final price post-delisting).
    if dead_added:
        closes[dead_added] = closes[dead_added].ffill()
    logger.info("Survivorship-free inputs: %d survivors+etfs, +%d liquid dead names",
                len(want), len(dead_added))

    # Extend agg_df with neutral rows for the dead names so compute_metric scores them on momentum.
    if dead_added:
        rows = []
        for sym in dead_added:
            rec = {"symbol": sym, "volume": float(np.nanmax(vol_cols[sym].reindex(cal).values) or 0.0),
                   "sector": "Unknown", "industry": "Unknown", **_NEUTRAL_FUND_COLS}
            rows.append(rec)
        dead_df = pd.DataFrame(rows)
        for c in agg_df.columns:
            if c not in dead_df.columns:
                dead_df[c] = np.nan
        agg_df = pd.concat([agg_df, dead_df[agg_df.columns]], ignore_index=True)

    return closes, agg_df, _dollar_volume_matrix(vol_cols, close_cols, cal)


def _dollar_volume_matrix(vol_cols, close_cols, cal) -> np.ndarray:
    """Causal trailing dollar-volume per (day, symbol) for cap-proxy sizing — placeholder aligned
    later to the final stock_cols order by load_and_precompute (returned here symbol-keyed)."""
    syms = list(close_cols)
    dv = np.full((len(cal), len(syms)), np.nan)
    for j, s in enumerate(syms):
        c = close_cols[s].reindex(cal).values
        v = vol_cols.get(s, pd.Series(dtype=float)).reindex(cal).values
        dv[:, j] = c * v
    # return as a DataFrame so the caller can align columns to its stock order
    return pd.DataFrame(dv, index=cal, columns=syms)
