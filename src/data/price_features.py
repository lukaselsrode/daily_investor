"""data/price_features.py — point-in-time price/momentum features from the FMP price cache.

Reconstructs the yfinance-derived momentum block (returns, SPY-relative strength,
realized vol, DMA flags, 52-week geometry) strictly as-of a given date, mirroring the
live definitions in data.fundamentals._enrich_with_momentum exactly:

  return_Xd/Xm         close[-1]/close[-lookback] - 1   (5/21/63/126 trading days)
  realized_vol_3m      std(last 63 daily pct changes) × √252
  risk_adj_momentum_3m return_3m / realized_vol_3m
  above_50dma/200dma   last close > mean of last 50/200 closes
  rs_1m/3m/6m          return − SPY return over the same window (arithmetic difference)
  low/high_52w         min/max close over the trailing 252 rows; position_52w in [0,1]

Built for rescoring SPARSE snapshot vintages that never stored these columns; with
fill_missing_only=True (default) present values are never overwritten. Read-only over
data/fmp_cache_adj/prices/*.parquet (never networks).
"""

from __future__ import annotations

import datetime as _dt
import logging

import numpy as np
import pandas as pd

from .fmp_client import _price_path

logger = logging.getLogger(__name__)

PRICE_FEATURE_COLS = (
    "current_price",
    "return_5d", "return_1m", "return_3m", "return_6m",
    "rs_1m", "rs_3m", "rs_6m",
    "realized_vol_3m", "risk_adj_momentum_3m",
    "above_50dma", "above_200dma",
    "low_52w", "high_52w", "position_52w",
)

_RETURN_LOOKBACKS = {"return_5d": 5, "return_1m": 21, "return_3m": 63, "return_6m": 126}
_RS_SOURCE = {"rs_1m": "return_1m", "rs_3m": "return_3m", "rs_6m": "return_6m"}
_MIN_ROWS_52W = 60
_MIN_VOL_RETURNS = 20

COVERAGE_WARN_THRESHOLD = 0.60


class PriceSeriesCache:
    """Lazy per-symbol loader of (dates, close) arrays from the FMP price cache.

    One instance should be shared across a whole migration run.
    """

    def __init__(self) -> None:
        self._series: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def series(self, symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        if not symbol or not isinstance(symbol, str):
            return None
        if symbol not in self._series:
            self._series[symbol] = self._load(symbol)
        return self._series[symbol]

    @staticmethod
    def _load(symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            df = pd.read_parquet(_price_path(symbol), columns=["close"])
        except (FileNotFoundError, OSError):
            return None
        except Exception:
            logger.warning("price_features: unreadable price cache for %s", symbol)
            return None
        if df.empty:
            return None
        return df.index.to_numpy(dtype=str), df["close"].to_numpy(dtype=float)


def _asof_iso(asof: _dt.date | str) -> str:
    if isinstance(asof, str):
        return asof[:10]
    return asof.isoformat()


def _window_asof(
    cache: PriceSeriesCache, symbol: str, iso: str
) -> np.ndarray | None:
    loaded = cache.series(symbol)
    if loaded is None:
        return None
    dates, close = loaded
    end = int(np.searchsorted(dates, iso, side="right"))
    window = close[:end]
    window = window[np.isfinite(window)]
    return window if len(window) else None


def _features_for_window(window: np.ndarray) -> dict[str, float]:
    """Mirror of _enrich_with_momentum arithmetic on one trailing close window."""
    vals: dict[str, float] = {col: np.nan for col in PRICE_FEATURE_COLS}
    n = len(window)
    if n < 5:
        return vals
    last = float(window[-1])
    vals["current_price"] = round(last, 4)

    for col, lb in _RETURN_LOOKBACKS.items():
        if n >= lb and window[-lb] > 0:
            vals[col] = round(last / float(window[-lb]) - 1.0, 4)

    if n >= 63:
        w = window[-64:]
        rets = np.diff(w) / w[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) >= _MIN_VOL_RETURNS:
            vol = round(float(rets[-63:].std(ddof=1) * (252 ** 0.5)), 4)
            vals["realized_vol_3m"] = vol
            r3m = vals["return_3m"]
            if np.isfinite(r3m) and vol > 0:
                vals["risk_adj_momentum_3m"] = round(r3m / vol, 4)

    if n >= 50:
        vals["above_50dma"] = float(last > window[-50:].mean())
    if n >= 200:
        vals["above_200dma"] = float(last > window[-200:].mean())

    if n >= _MIN_ROWS_52W:
        w52 = window[-252:]
        lo, hi = float(w52.min()), float(w52.max())
        vals["low_52w"] = round(lo, 4)
        vals["high_52w"] = round(hi, 4)
        if hi > lo:
            vals["position_52w"] = round((last - lo) / (hi - lo), 4)
    return vals


def compute_price_features_asof(
    symbols: list[str],
    asof: _dt.date | str,
    cache: PriceSeriesCache,
) -> pd.DataFrame:
    """Per-symbol PRICE_FEATURE_COLS using only cache rows dated <= asof.

    Index = symbol. rs_* stay NaN when the SPY reference window is unavailable.
    """
    iso = _asof_iso(asof)
    spy_window = _window_asof(cache, "SPY", iso)
    spy_feats = _features_for_window(spy_window) if spy_window is not None else {}

    rows: dict[str, list[float]] = {}
    for sym in symbols:
        window = _window_asof(cache, sym, iso)
        vals = _features_for_window(window) if window is not None else {
            col: np.nan for col in PRICE_FEATURE_COLS
        }
        for rs_col, ret_col in _RS_SOURCE.items():
            r = vals.get(ret_col, np.nan)
            spy_r = spy_feats.get(ret_col, np.nan) if spy_feats else np.nan
            if np.isfinite(r) and np.isfinite(spy_r):
                vals[rs_col] = round(r - spy_r, 4)
        rows[sym] = [vals[c] for c in PRICE_FEATURE_COLS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(PRICE_FEATURE_COLS))


def add_price_momentum_features(
    df: pd.DataFrame,
    asof: _dt.date | str,
    cache: PriceSeriesCache | None = None,
    fill_missing_only: bool = True,
) -> float:
    """Merge PRICE_FEATURE_COLS into df (in place, keyed on df['symbol']); returns coverage.

    With fill_missing_only=True only NaN cells are written, so vintages that carry
    live yfinance/Robinhood values keep them. Coverage = post-fill fraction of rows
    with a non-null return_3m.
    """
    if "symbol" not in df.columns or df.empty:
        for col in PRICE_FEATURE_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return 0.0

    cache = cache if cache is not None else PriceSeriesCache()
    symbols = ["" if pd.isna(s) else str(s) for s in df["symbol"].tolist()]
    feats = compute_price_features_asof(symbols, asof, cache)
    aligned = feats.reindex(symbols)

    for col in PRICE_FEATURE_COLS:
        computed = pd.Series(aligned[col].to_numpy(), index=df.index)
        if col not in df.columns:
            df[col] = computed
        elif fill_missing_only:
            existing = pd.to_numeric(df[col], errors="coerce") if df[col].dtype == object else df[col]
            df[col] = existing.where(existing.notna(), computed)
        else:
            df[col] = computed

    coverage = float(pd.to_numeric(df["return_3m"], errors="coerce").notna().mean())
    log = logger.warning if coverage < COVERAGE_WARN_THRESHOLD else logger.info
    log(
        "price_features: asof=%s coverage=%.1f%% (n=%d, fill_missing_only=%s)",
        _asof_iso(asof), coverage * 100, len(df), fill_missing_only,
    )
    return coverage
