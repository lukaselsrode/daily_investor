"""
strategy/snapshots.py — Historical scored-universe snapshot store.

Saves each scored universe as a dated Parquet file in data/snapshots/.
Enables rolling IC analysis and regime-conditioned factor diagnostics.

Public API
----------
save_snapshot(df, date, overwrite)         → Path
list_snapshots()                           → list[tuple[date, Path]]
load_snapshots(start, end, columns)        → pd.DataFrame  (snapshot_date col added)
prune_snapshots(keep_days)                 → int (files removed)
compute_forward_ic(horizon_days, factors)  → pd.DataFrame  (date, factor, ic, n, p_value)
backfill_from_csvs()                       → int (files written)
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FACTOR_COLS_DEFAULT = ["value_score", "momentum_score", "quality_score", "income_score"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _snapshot_dir() -> Path:
    from util import DATA_DIRECTORY, SNAPSHOT_PARAMS  # noqa: deferred import
    d = Path(DATA_DIRECTORY) / SNAPSHOT_PARAMS.get("subdir", "snapshots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _date_to_stem(d: datetime.date) -> str:
    return d.strftime("%Y_%m_%d")


def _stem_to_date(stem: str) -> datetime.date:
    return datetime.datetime.strptime(stem, "%Y_%m_%d").date()


def _snapshot_path(d: datetime.date) -> Path:
    return _snapshot_dir() / f"{_date_to_stem(d)}.parquet"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_snapshot(
    df: pd.DataFrame,
    date: datetime.date | None = None,
    overwrite: bool = False,
) -> Path:
    """Save the scored universe as a dated Parquet snapshot.

    Returns the path written. If today's snapshot already exists and
    overwrite=False the existing path is returned without re-writing.
    Returns an empty Path if snapshots are disabled in config.
    """
    from util import SNAPSHOT_PARAMS  # noqa: deferred import

    if not SNAPSHOT_PARAMS.get("enabled", True):
        return Path()

    if date is None:
        date = datetime.date.today()

    path = _snapshot_path(date)

    if path.exists() and not overwrite:
        logger.debug("Snapshot for %s already exists — skipping (%s)", date, path.name)
        return path

    compression = SNAPSHOT_PARAMS.get("compression", "snappy")

    out = df.copy()
    out["snapshot_date"] = date.isoformat()

    try:
        out.to_parquet(path, index=False, compression=compression)
        logger.info("Saved snapshot: %s  (%d rows)", path.name, len(df))
    except Exception as exc:
        logger.error("Failed to save snapshot %s: %s", path.name, exc)
        raise

    # Auto-prune after successful write
    retention = SNAPSHOT_PARAMS.get("retention_days", 365)
    prune_snapshots(retention)

    return path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_snapshots() -> list[tuple[datetime.date, Path]]:
    """Return sorted (ascending) list of (date, path) tuples for all snapshots."""
    result: list[tuple[datetime.date, Path]] = []
    for f in sorted(_snapshot_dir().glob("*.parquet")):
        try:
            result.append((_stem_to_date(f.stem), f))
        except ValueError:
            continue
    return result


def load_snapshots(
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate all snapshots within [start, end].

    The returned DataFrame always contains a `snapshot_date` column (datetime.date).
    Returns an empty DataFrame when no snapshots match.
    """
    snaps = list_snapshots()
    if start:
        snaps = [(d, p) for d, p in snaps if d >= start]
    if end:
        snaps = [(d, p) for d, p in snaps if d <= end]

    if not snaps:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for date, path in snaps:
        try:
            # parquet may already have snapshot_date col from save_snapshot
            read_cols = None
            if columns is not None:
                # ensure snapshot_date is always present
                read_cols = list(set(columns + ["snapshot_date"]))
                # trim to columns that actually exist in the file
                import pyarrow.parquet as pq
                file_cols = set(pq.read_schema(path).names)
                read_cols = [c for c in read_cols if c in file_cols]

            frame = pd.read_parquet(path, columns=read_cols)
            frame["snapshot_date"] = date  # overwrite string col with date object
            frames.append(frame)
        except Exception as exc:
            logger.warning("Failed to load snapshot %s: %s", path.name, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def prune_snapshots(keep_days: int) -> int:
    """Delete snapshots older than keep_days. Returns count removed."""
    cutoff = datetime.date.today() - datetime.timedelta(days=keep_days)
    removed = 0
    for date, path in list_snapshots():
        if date < cutoff:
            try:
                path.unlink()
                removed += 1
                logger.debug("Pruned old snapshot: %s", path.name)
            except Exception as exc:
                logger.warning("Failed to prune %s: %s", path.name, exc)
    if removed:
        logger.info("Pruned %d snapshots older than %d days", removed, keep_days)
    return removed


def backfill_from_csvs() -> int:
    """Convert existing agg_data_*.csv files to Parquet snapshots.

    Skips dates that already have a Parquet snapshot.
    Returns the number of new snapshots written.
    """
    from util import DATA_DIRECTORY, SNAPSHOT_PARAMS  # noqa: deferred import

    if not SNAPSHOT_PARAMS.get("enabled", True):
        return 0

    data_dir = Path(DATA_DIRECTORY)
    written = 0

    for csv_path in sorted(data_dir.glob("agg_data_*.csv")):
        date_part = csv_path.stem.replace("agg_data_", "")
        try:
            date = _stem_to_date(date_part)
        except ValueError:
            continue

        if _snapshot_path(date).exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            save_snapshot(df, date=date, overwrite=False)
            written += 1
        except Exception as exc:
            logger.warning("backfill failed for %s: %s", csv_path.name, exc)

    if written:
        logger.info("Backfilled %d CSV → Parquet snapshots", written)
    return written


# ---------------------------------------------------------------------------
# Rolling IC
# ---------------------------------------------------------------------------

def compute_forward_ic(
    horizon_days: int = 21,
    factor_cols: list[str] | None = None,
    min_overlap: int = 20,
    max_horizon_slop_pct: float = 0.5,
) -> pd.DataFrame:
    """Compute rolling Spearman IC for each factor vs realised forward returns.

    Algorithm
    ---------
    For each snapshot at date T:
      1. Find the snapshot closest to T + horizon_days (within ±50% of horizon).
      2. Forward return = current_price[T_fwd] / current_price[T] - 1.
         Falls back to return_1m from the forward snapshot when prices are sparse.
      3. Compute Spearman IC between factor_score[T] and forward_return.

    Returns
    -------
    DataFrame with columns: date, factor, ic, n_stocks, p_value
    Empty DataFrame if fewer than 2 snapshots exist.
    """
    from scipy.stats import spearmanr  # noqa: deferred import

    if factor_cols is None:
        factor_cols = _FACTOR_COLS_DEFAULT

    # Load everything we need in one pass
    needed_cols = list({"symbol", "current_price", "return_1m"} | set(factor_cols))
    all_df = load_snapshots(columns=needed_cols)

    if all_df.empty or "snapshot_date" not in all_df.columns:
        return pd.DataFrame(columns=["date", "factor", "ic", "n_stocks", "p_value"])

    # Coerce numeric
    for col in ["current_price", "return_1m"] + factor_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    dates = sorted(all_df["snapshot_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["date", "factor", "ic", "n_stocks", "p_value"])

    results: list[dict] = []
    slop_limit = int(horizon_days * max_horizon_slop_pct)

    for t_date in dates:
        t_df = all_df[all_df["snapshot_date"] == t_date]

        # Find nearest forward snapshot
        target_fwd = t_date + datetime.timedelta(days=horizon_days)
        fwd_dates = [d for d in dates if d > t_date]
        if not fwd_dates:
            continue
        f_date = min(fwd_dates, key=lambda d: abs((d - target_fwd).days))
        if abs((f_date - target_fwd).days) > slop_limit:
            continue

        f_df = (
            all_df[all_df["snapshot_date"] == f_date][["symbol", "current_price", "return_1m"]]
            .rename(columns={"current_price": "price_fwd", "return_1m": "return_1m_fwd"})
        )

        merged = t_df.merge(f_df, on="symbol", how="inner")
        if merged.empty:
            continue

        # Forward return: prefer price-based, fall back to return_1m from fwd snapshot
        t_price = merged.get("current_price", pd.Series(dtype=float))
        f_price = merged["price_fwd"]
        price_ok = t_price.notna() & f_price.notna() & (t_price > 0)

        fwd_ret = pd.Series(float("nan"), index=merged.index)
        if price_ok.sum() >= min_overlap:
            fwd_ret[price_ok] = f_price[price_ok].values / t_price[price_ok].values - 1.0
        else:
            fwd_ret = merged["return_1m_fwd"]

        merged = merged.copy()
        merged["_fwd_ret"] = fwd_ret

        valid_fwd = merged["_fwd_ret"].notna()
        if valid_fwd.sum() < min_overlap:
            continue

        for factor in factor_cols:
            if factor not in merged.columns:
                continue
            mask = valid_fwd & merged[factor].notna()
            n = mask.sum()
            if n < min_overlap:
                continue
            ic, pval = spearmanr(merged.loc[mask, factor], merged.loc[mask, "_fwd_ret"])
            results.append({
                "date":     t_date,
                "factor":   factor,
                "ic":       round(float(ic),   4),
                "n_stocks": int(n),
                "p_value":  round(float(pval), 4),
            })

    return pd.DataFrame(results)
