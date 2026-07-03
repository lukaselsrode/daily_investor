"""data/volume_features.py — point-in-time dollar-volume features from the FMP price cache.

Computes trailing average dollar volume (close × volume) over 5/21/63 trading-day
windows plus a 63-day consistency measure (coefficient of variation), all strictly
as-of a given date: a window is the last N cache rows with row-date <= asof, so the
same call is valid for today's ETL and for rescoring a 2025 snapshot.

Read-only over data/fmp_cache_adj/prices/*.parquet (never networks). Symbols absent
from the cache degrade to an ADV×price approximation for the 21d horizon when the
caller allows it; the other horizons stay NaN and the peer machinery falls back.
"""

from __future__ import annotations

import datetime as _dt
import logging

import numpy as np
import pandas as pd

from .fmp_client import _price_path

logger = logging.getLogger(__name__)

FEATURE_COLS = ("dollar_vol_5d", "dollar_vol_21d", "dollar_vol_63d", "dollar_vol_cv_63d")

# Minimum cache rows inside the window before a horizon is trusted (else NaN).
_MIN_ROWS = {"dollar_vol_5d": 3, "dollar_vol_21d": 12, "dollar_vol_63d": 40}
_HORIZON_ROWS = {"dollar_vol_5d": 5, "dollar_vol_21d": 21, "dollar_vol_63d": 63}

COVERAGE_WARN_THRESHOLD = 0.70


class DollarVolumeCache:
    """Lazy per-symbol loader of (dates, close×volume) arrays from the FMP price cache.

    One instance should be shared across a whole migration run — repeated
    construction rereads thousands of parquets.
    """

    def __init__(self) -> None:
        self._series: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def series(self, symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(dates as ISO-string ndarray, dollar volume ndarray) or None when uncached."""
        if symbol not in self._series:
            self._series[symbol] = self._load(symbol)
        return self._series[symbol]

    @staticmethod
    def _load(symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        if not symbol or not isinstance(symbol, str):
            return None
        path = _price_path(symbol)
        try:
            df = pd.read_parquet(path, columns=["close", "volume"])
        except (FileNotFoundError, OSError):
            return None
        except Exception:  # corrupt file — treat as absent
            logger.warning("volume_features: unreadable price cache for %s", symbol)
            return None
        if df.empty or "close" not in df.columns or "volume" not in df.columns:
            return None
        dates = df.index.to_numpy(dtype=str)
        dollar = (df["close"].to_numpy(dtype=float) * df["volume"].to_numpy(dtype=float))
        return dates, dollar


def _asof_iso(asof: _dt.date | str) -> str:
    if isinstance(asof, str):
        return asof[:10]
    return asof.isoformat()


def compute_features_asof(
    symbols: list[str],
    asof: _dt.date | str,
    cache: DollarVolumeCache,
) -> pd.DataFrame:
    """Trailing dollar-volume features per symbol, using only cache rows dated <= asof.

    Index = symbol, columns = FEATURE_COLS. A horizon is NaN when the window holds
    fewer than its minimum row count (symbol too young / delisted early / uncached).
    """
    iso = _asof_iso(asof)
    rows: dict[str, list[float]] = {}
    for sym in symbols:
        loaded = cache.series(sym)
        if loaded is None:
            rows[sym] = [np.nan] * len(FEATURE_COLS)
            continue
        dates, dollar = loaded
        end = int(np.searchsorted(dates, iso, side="right"))
        vals: list[float] = []
        for col in ("dollar_vol_5d", "dollar_vol_21d", "dollar_vol_63d"):
            n = _HORIZON_ROWS[col]
            window = dollar[max(0, end - n):end]
            window = window[np.isfinite(window)]
            vals.append(float(window.mean()) if len(window) >= _MIN_ROWS[col] else np.nan)
        w63 = dollar[max(0, end - 63):end]
        w63 = w63[np.isfinite(w63)]
        if len(w63) >= _MIN_ROWS["dollar_vol_63d"] and w63.mean() > 0:
            vals.append(float(w63.std() / w63.mean()))
        else:
            vals.append(np.nan)
        rows[sym] = vals
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(FEATURE_COLS))


def add_dollar_volume_features(
    df: pd.DataFrame,
    asof: _dt.date | str,
    cache: DollarVolumeCache | None = None,
    fallback_from_adv: bool = True,
) -> float:
    """Merge FEATURE_COLS into df (in place, keyed on df['symbol']); returns coverage.

    Coverage = fraction of rows with a non-null dollar_vol_63d from the cache proper
    (the ADV×price fallback fills dollar_vol_21d only and does not count as covered).
    """
    if "symbol" not in df.columns or df.empty:
        for col in FEATURE_COLS:
            df[col] = np.nan
        return 0.0

    cache = cache if cache is not None else DollarVolumeCache()
    # pandas string dtype keeps NaN through .astype(str) — map missing symbols to ""
    # (never in the cache → NaN features) instead of letting float NaN reach the
    # cache path builder.
    symbols = ["" if pd.isna(s) else str(s) for s in df["symbol"].tolist()]
    feats = compute_features_asof(symbols, asof, cache)
    aligned = feats.reindex(symbols)
    for col in FEATURE_COLS:
        df[col] = aligned[col].to_numpy()

    coverage = float(df["dollar_vol_63d"].notna().mean())

    if fallback_from_adv and "volume" in df.columns and "current_price" in df.columns:
        missing = df["dollar_vol_21d"].isna()
        if missing.any():
            adv = pd.to_numeric(df["volume"], errors="coerce")
            px = pd.to_numeric(df["current_price"], errors="coerce")
            df.loc[missing, "dollar_vol_21d"] = (adv * px)[missing]

    log = logger.warning if coverage < COVERAGE_WARN_THRESHOLD else logger.info
    log(
        "volume_features: asof=%s coverage=%.1f%% (n=%d, fallback=%s)",
        _asof_iso(asof), coverage * 100, len(df), fallback_from_adv,
    )
    return coverage
