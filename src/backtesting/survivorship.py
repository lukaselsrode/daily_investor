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
  traded value (bankruptcy → ~0 captured; acquisition → deal price). A per-day `tradeable` mask
  (True until the last NATIVE price print) is returned alongside, so the simulator can refuse to
  BUY a name after its delist date even though its ffilled price stays finite.
- Dead-name fundamentals = cross-sectional MEDIAN of the alive universe (computed at assemble
  time) for quality/income/pe/pb. We don't have point-in-time fundamentals for delisted names;
  zero looked "neutral" but sat below the live min_quality_score buy gate (0.38), so every dead
  name was auto-rejected and survivorship bias silently survived. The alive-universe median is an
  honest neutral: gates treat dead names as AVERAGE rather than uninvestable, and selection still
  differentiates them on price/momentum.
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
    "value_metric": 0.0, "position_52w": np.nan, "return_1m": np.nan,
    "baseline_score": 0.0, "market_cap": np.nan,
}
# Fundamentals filled with the cross-sectional MEDIAN of the alive universe (see module
# docstring): zero put dead names below the live min_quality_score buy gate, making them
# structurally unbuyable in a run labelled survivorship-free.
_MEDIAN_FUND_COLS = ("pe_comp", "pb_comp", "quality_score", "income_score")


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
             add_dead: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build survivorship-free (closes, extended_agg_df, dollar_volume, tradeable) from the FMP cache.

    closes:          date×symbol adjusted Close, dead names trailing-ffilled to final price.
    extended_agg_df: agg_df + rows for the added dead names — quality/income/pe/pb at the alive
                     universe's cross-sectional median (honest neutral; see module docstring).
    dollar_volume:   date×symbol causal Close×Volume (symbol-keyed; the caller aligns columns),
                     for cap-proxy position sizing without the static-market_caps look-ahead.
    tradeable:       date×symbol bool, True through each symbol's LAST NATIVE price print (i.e.
                     before the trailing ffill); the simulator must not buy past it.
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
    # Tradeability: True through each symbol's last NATIVE print (captured BEFORE the trailing
    # ffill below). Post-delist ffilled prices are finite, so the simulator needs this mask to
    # refuse buys past the delist date. Reverse-cummax marks every day <= the last native print.
    tradeable = closes.notna().iloc[::-1].cummax().iloc[::-1]
    # Trailing forward-fill ONLY for dead names (mark held positions at final price post-delisting).
    if dead_added:
        closes[dead_added] = closes[dead_added].ffill()
    logger.info("Survivorship-free inputs: %d survivors+etfs, +%d liquid dead names",
                len(want), len(dead_added))

    # Extend agg_df with rows for the dead names: median-neutral fundamentals (so the live
    # quality gate treats them as average, not auto-rejected) + momentum from prices.
    if dead_added:
        medians = {
            c: float(pd.to_numeric(agg_df.get(c), errors="coerce").median())
            if c in agg_df.columns and np.isfinite(pd.to_numeric(agg_df.get(c), errors="coerce").median())
            else 0.0
            for c in _MEDIAN_FUND_COLS
        }
        rows = []
        for sym in dead_added:
            rec = {"symbol": sym, "volume": float(np.nanmax(vol_cols[sym].reindex(cal).values) or 0.0),
                   "sector": "Unknown", "industry": "Unknown", **_NEUTRAL_FUND_COLS, **medians}
            rows.append(rec)
        dead_df = pd.DataFrame(rows)
        for c in agg_df.columns:
            if c not in dead_df.columns:
                dead_df[c] = np.nan
        agg_df = pd.concat([agg_df, dead_df[agg_df.columns]], ignore_index=True)

    return closes, agg_df, _dollar_volume_matrix(vol_cols, close_cols, cal), tradeable


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
