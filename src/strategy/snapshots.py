"""
strategy/snapshots.py — Historical scored-universe snapshot store.

Saves each scored universe as a datetime-stamped Parquet file in data/snapshots/
so that multiple fetch-data runs on the same calendar day are all preserved.

Filename format: YYYY_MM_DD_HH_MM.parquet  (e.g. 2026_05_27_14_30.parquet)
Old date-only files (YYYY_MM_DD.parquet) are still read correctly.

Public API
----------
save_snapshot(df, date, overwrite)         → Path
list_snapshots()                           → list[tuple[date, Path]]
load_snapshots(start, end, columns)        → pd.DataFrame  (snapshot_date + snapshot_datetime cols)
prune_snapshots(keep_days)                 → int (files removed)
compute_forward_ic(horizon_days, factors)  → pd.DataFrame  (date, factor, ic, n, p_value)
backfill_from_csvs()                       → int (files written)
rescore_snapshots(...)                     → MigrationReport (under unified peer engine)

IC note
-------
IC and other 30-day metrics deduplicate to ONE snapshot per calendar day (the
latest intraday run) before computing forward returns.  Multiple intraday runs
on the same date therefore count as one observation, not N observations.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_FACTOR_COLS_DEFAULT = ["value_score", "momentum_score", "quality_score", "income_score"]

# Stem formats, newest first — used for parsing existing files
_STEM_FORMATS = ["%Y_%m_%d_%H_%M", "%Y_%m_%d"]

# A real scored-universe frame has thousands of rows; degraded pipeline runs
# (universe collapse, scoring skipped) produce tiny frames that must never
# enter the store — they poison list_snapshots(), IC, and rescore loops.
MIN_SNAPSHOT_ROWS = 50
_REQUIRED_SNAPSHOT_COLUMNS = ("symbol", "value_metric")


def _is_valid_snapshot(path: Path) -> bool:
    """True when the parquet schema carries the scored-universe contract columns."""
    try:
        import pyarrow.parquet as pq
        names = set(pq.read_schema(path).names)
    except Exception:
        return False
    return all(col in names for col in _REQUIRED_SNAPSHOT_COLUMNS)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _snapshot_dir() -> Path:
    from util import DATA_DIRECTORY, SNAPSHOT_PARAMS
    d = Path(DATA_DIRECTORY) / SNAPSHOT_PARAMS.get("subdir", "snapshots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dt_to_stem(dt: datetime.datetime) -> str:
    return dt.strftime("%Y_%m_%d_%H_%M")


def _stem_to_datetime(stem: str) -> datetime.datetime:
    """Parse both new YYYY_MM_DD_HH_MM and legacy YYYY_MM_DD stems."""
    for fmt in _STEM_FORMATS:
        try:
            return datetime.datetime.strptime(stem, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse snapshot stem: {stem!r}")


def _stem_to_date(stem: str) -> datetime.date:
    return _stem_to_datetime(stem).date()


def _snapshot_path(dt: datetime.datetime) -> Path:
    return _snapshot_dir() / f"{_dt_to_stem(dt)}.parquet"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_snapshot(
    df: pd.DataFrame,
    date: datetime.date | datetime.datetime | None = None,
    overwrite: bool = False,
) -> Path:
    """Save the scored universe as a datetime-stamped Parquet snapshot.

    Multiple calls on the same calendar day each produce a distinct file
    (e.g. 2026_05_27_09_00.parquet, 2026_05_27_14_30.parquet).

    The `snapshot_date` column stored inside the file is always the calendar
    date so IC computation and date-range queries remain unchanged.

    Returns an empty Path if snapshots are disabled in config.
    """
    from util import SNAPSHOT_PARAMS

    if not SNAPSHOT_PARAMS.get("enabled", True):
        return Path()

    missing = [c for c in _REQUIRED_SNAPSHOT_COLUMNS if c not in df.columns]
    if len(df) < MIN_SNAPSHOT_ROWS or missing:
        logger.warning(
            "Refusing snapshot save: %d rows (min %d), missing columns %s — "
            "degraded pipeline frame (universe collapse or scoring skipped)",
            len(df), MIN_SNAPSHOT_ROWS, missing or "none",
        )
        return Path()

    if date is None:
        dt = datetime.datetime.now()
    elif isinstance(date, datetime.datetime):
        dt = date
    else:
        # Legacy callers (backfill) pass a datetime.date → treat as midnight
        dt = datetime.datetime.combine(date, datetime.time(0, 0))

    path = _snapshot_path(dt)

    if path.exists() and not overwrite:
        logger.debug("Snapshot %s already exists — skipping", path.name)
        return path

    compression = SNAPSHOT_PARAMS.get("compression", "snappy")

    out = df.copy()
    out["snapshot_date"] = dt.date().isoformat()      # calendar date for IC / queries
    out["snapshot_datetime"] = dt.isoformat(timespec="minutes")  # full timestamp

    try:
        out.to_parquet(path, index=False, compression=compression)
        logger.info("Saved snapshot: %s  (%d rows)", path.name, len(df))
    except Exception as exc:
        logger.error("Failed to save snapshot %s: %s", path.name, exc)
        raise

    retention = SNAPSHOT_PARAMS.get("retention_days", 365)
    prune_snapshots(retention)

    return path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_snapshots() -> list[tuple[datetime.date, Path]]:
    """Return sorted (ascending) list of (calendar_date, path) tuples.

    Multiple intraday files for the same calendar date appear as separate
    entries — callers that build a date → DataFrame dict will naturally
    keep the last (latest) run of each day.
    """
    result: list[tuple[datetime.date, Path]] = []
    for f in sorted(_snapshot_dir().glob("*.parquet")):
        if not _is_valid_snapshot(f):
            logger.debug("list_snapshots: skipping malformed file %s", f.name)
            continue
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

    Columns always present in the result:
        snapshot_date     datetime.date  — calendar date of the run
        snapshot_datetime str            — full ISO timestamp (HH:MM precision)
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
            read_cols = None
            if columns is not None:
                read_cols = list(set(columns + ["snapshot_date", "snapshot_datetime"]))
                import pyarrow.parquet as pq
                file_cols = set(pq.read_schema(path).names)
                read_cols = [c for c in read_cols if c in file_cols]

            frame = pd.read_parquet(path, columns=read_cols)
            frame["snapshot_date"] = date  # always a date object
            # Preserve snapshot_datetime from file when present; fall back to date
            if "snapshot_datetime" not in frame.columns:
                frame["snapshot_datetime"] = date.isoformat()
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

    Handles both legacy YYYY_MM_DD and new YYYY_MM_DD_HH_MM filename formats.
    Skips CSV files whose corresponding snapshot already exists.
    Returns the number of new snapshots written.
    """
    from util import DATA_DIRECTORY, SNAPSHOT_PARAMS

    if not SNAPSHOT_PARAMS.get("enabled", True):
        return 0

    data_dir = Path(DATA_DIRECTORY)
    written = 0

    for csv_path in sorted(data_dir.glob("agg_data_*.csv")):
        stem_part = csv_path.stem.replace("agg_data_", "")
        try:
            dt = _stem_to_datetime(stem_part)
        except ValueError:
            continue

        if _snapshot_path(dt).exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            save_snapshot(df, date=dt, overwrite=False)
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
    from scipy.stats import spearmanr

    if factor_cols is None:
        factor_cols = _FACTOR_COLS_DEFAULT

    # Load everything we need in one pass
    needed_cols = list({"symbol", "current_price", "return_1m", "snapshot_datetime"} | set(factor_cols))
    all_df = load_snapshots(columns=needed_cols)

    if all_df.empty or "snapshot_date" not in all_df.columns:
        return pd.DataFrame(columns=["date", "factor", "ic", "n_stocks", "p_value"])

    # Coerce numeric
    for col in ["current_price", "return_1m"] + factor_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    # Deduplicate: multiple intraday runs on the same calendar day count as ONE
    # observation for IC purposes.  Keep the latest run per (symbol, date).
    # Files are loaded in chronological order so the last row per group is newest.
    if "snapshot_datetime" in all_df.columns:
        all_df = (
            all_df.sort_values("snapshot_datetime")
            .groupby(["symbol", "snapshot_date"], sort=False)
            .last()
            .reset_index()
        )

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


# ---------------------------------------------------------------------------
# Snapshot rescore (hard cutover — writes canonical column names only)
# ---------------------------------------------------------------------------

# Legacy `*_v3`-suffixed columns that may still exist on disk from prior runs.
# rescore_snapshots() strips these on write to converge to a single canonical schema.
_LEGACY_SUFFIXED_COLUMNS = (
    "value_score_v3", "quality_score_v3", "momentum_score_v3", "income_score_v3",
    "yield_trap_flag_v3", "value_metric_v3",
    "scoring_v3_config_hash", "scoring_v3_migration_timestamp", "scoring_v3_source_snapshot",
)

_SCORING_META_COLUMNS = (
    "scoring_model_version",
    "scoring_config_hash",
    "rescore_timestamp",
    "rescore_source_snapshot",
)


@dataclass
class MigrationReport:
    """Summary of a `snapshots rescore` run (under the unified peer engine)."""
    files_processed: int = 0
    files_rescored: int = 0
    files_skipped_already_migrated: int = 0
    files_skipped_error: int = 0
    rows_rescored: int = 0
    rows_missing_sector: int = 0
    rows_missing_industry: int = 0
    fallback_usage: dict[str, int] = field(default_factory=dict)
    nan_value_metric_rows: int = 0
    backups_created: int = 0
    dry_run: bool = False
    before_value_metric_mean: float = 0.0
    after_value_metric_mean: float = 0.0
    top_score_shifts: list[tuple[str, float, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # peer-2 volume backfill: per-file fraction of rows with cache-backed dollar_vol_63d.
    volume_feature_coverage: dict[str, float] = field(default_factory=dict)
    # peer-3 fundamental backfill: per-file fraction of rows with a cache-backed roe_ttm.
    fundamental_feature_coverage: dict[str, float] = field(default_factory=dict)
    # SPARSE-vintage rescue: per-file pe_ratio coverage after FMP valuation
    # reconstruction, and return_3m coverage after price-feature reconstruction.
    valuation_backfill_coverage: dict[str, float] = field(default_factory=dict)
    price_backfill_coverage: dict[str, float] = field(default_factory=dict)
    # income de-degeneracy: per-file max single-value mass among NONZERO income scores.
    income_top_mass: dict[str, float] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [
            f"Snapshot rescore ({'DRY RUN' if self.dry_run else 'WRITE'})",
            f"  files processed:           {self.files_processed}",
            f"  files rescored:            {self.files_rescored}",
            f"  skipped (already current): {self.files_skipped_already_migrated}",
            f"  skipped (errors):          {self.files_skipped_error}",
            f"  backups created:           {self.backups_created}",
            f"  rows rescored:             {self.rows_rescored}",
            f"  rows missing sector:       {self.rows_missing_sector}",
            f"  rows missing industry:     {self.rows_missing_industry}",
            f"  nan value_metric rows:     {self.nan_value_metric_rows}",
            f"  value_metric mean (before/after): {self.before_value_metric_mean:.3f} / {self.after_value_metric_mean:.3f}",
        ]
        if self.volume_feature_coverage:
            covs = list(self.volume_feature_coverage.values())
            low = {f: c for f, c in self.volume_feature_coverage.items() if c < 0.50}
            lines.append(
                f"  dollar-volume coverage:    mean {sum(covs) / len(covs):.1%}, "
                f"min {min(covs):.1%} ({len(low)} file(s) < 50%)"
            )
            for f, c in sorted(low.items()):
                lines.append(f"    LOW {f}: {c:.1%}")
        for label, cov_map, warn_at in (
            ("fundamental coverage:     ", self.fundamental_feature_coverage, 0.50),
            ("SPARSE valuation rescue:  ", self.valuation_backfill_coverage, 0.60),
            ("SPARSE price rescue:      ", self.price_backfill_coverage, 0.80),
        ):
            if cov_map:
                covs = list(cov_map.values())
                low = {f: c for f, c in cov_map.items() if c < warn_at}
                lines.append(
                    f"  {label}mean {sum(covs) / len(covs):.1%}, "
                    f"min {min(covs):.1%} ({len(low)} file(s) < {warn_at:.0%})"
                )
                for f, c in sorted(low.items()):
                    lines.append(f"    LOW {f}: {c:.1%}")
        if self.income_top_mass:
            worst = max(self.income_top_mass.items(), key=lambda kv: kv[1])
            lines.append(
                f"  income top-mass (nonzero): worst {worst[1]:.1%} ({worst[0]})"
            )
        if self.fallback_usage:
            lines.append("  fallback usage:")
            for k, v in sorted(self.fallback_usage.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<20s} {v}")
        if self.top_score_shifts:
            lines.append("  largest score shifts (top 10):")
            for sym, before, after in self.top_score_shifts[:10]:
                lines.append(f"    {sym:<10s} {before:+.3f} → {after:+.3f}  (Δ {after - before:+.3f})")
        if self.errors:
            lines.append("  errors:")
            for err in self.errors[:20]:
                lines.append(f"    {err}")
        return "\n".join(lines)


def _is_already_current_engine(df: pd.DataFrame) -> bool:
    """True if the snapshot is already stamped with the current engine revision."""
    from strategy.scoring.composite import SCORING_MODEL_VERSION
    return (
        "scoring_model_version" in df.columns
        and (df["scoring_model_version"] == SCORING_MODEL_VERSION).any()
    )


def rescore_snapshots(
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    *,
    dry_run: bool = False,
    overwrite_existing: bool = False,
    in_place_with_backup: bool = False,
    scoring_cfg: dict | None = None,
    backfill_volume: bool = True,
    backfill_fundamentals: bool = True,
    backfill_sparse: bool = True,
) -> MigrationReport:
    """Rescore snapshots under the unified peer engine. Writes canonical column names.

    REGIME CONVENTION: rescoring is regime-NEUTRAL (compute_metric without a regime
    argument), while live snapshots are scored with the day's regime tilt. A rescored
    file is therefore not bit-comparable to the live snapshot from the same day; this
    is the accepted convention (the tilt is a portfolio-construction overlay, not a
    property of the factor scores).

    Behavior
    --------
    - Iterates every *.parquet under input_dir (defaults to data/snapshots/).
    - Reads each file, recomputes scores via strategy.scoring.compute_metric using
      the in-memory universe as the peer set (no lookahead).
    - WRITES canonical column names (value_score, value_metric, etc.) and drops
      any legacy `*_v3` columns.
    - When output_dir is None and in_place_with_backup=True: creates a
      <file>.bak.parquet copy first, then writes the rescored file in place.
    - When output_dir is set: writes rescored copies into output_dir.
    - dry_run=True: writes nothing, only logs and tallies.
    - overwrite_existing=False (default): files already at the current engine
      version are skipped (idempotent).
    - backfill_volume=True (default): before rescoring, point-in-time dollar-volume
      features (data.volume_features) are merged from the FMP price cache using
      trailing windows ending at the snapshot's stem date; files that pre-date the
      volume column also get a synthesized share-ADV `volume` (dollar_vol_21d /
      current_price).
    - backfill_fundamentals=True (default, peer-3): PIT fundamental features
      (data.fundamental_features — quality + income sustainability inputs) are
      merged as-of the stem date for EVERY vintage (no historical file carries them).
    - backfill_sparse=True (default, peer-3): vintages missing pe_ratio or the
      momentum feature block entirely (2025 SPARSE files) get FMP-reconstructed
      valuation ratios (fill-missing-only — Robinhood values are never overwritten)
      and price/momentum features as-of the stem date.
    """
    from strategy.scoring.composite import (
        SCORING_MODEL_VERSION,
        compute_metric,
        scoring_config_hash,
    )

    if input_dir is None:
        input_dir = _snapshot_dir()
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"snapshots input dir not found: {input_path}")

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = input_path  # in-place mode

    cfg_hash = scoring_config_hash(scoring_cfg)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    report = MigrationReport(dry_run=dry_run)

    before_means: list[float] = []
    after_means: list[float] = []
    shifts: list[tuple[str, float, float]] = []
    # One shared FMP-cache reader of each kind across all files — per-file
    # construction would reread thousands of price/statement files.
    _volume_cache = None
    _fund_cache = None
    _price_cache = None

    def _coverage(frame: pd.DataFrame, col: str) -> float:
        if col not in frame.columns or not len(frame):
            return 0.0
        return float(pd.to_numeric(frame[col], errors="coerce").notna().mean())

    for snap_path in sorted(input_path.glob("*.parquet")):
        if snap_path.name.endswith(".bak.parquet"):
            continue
        if not _is_valid_snapshot(snap_path):
            report.files_skipped_error += 1
            report.errors.append(f"{snap_path.name}: malformed snapshot (missing contract columns) — skipped")
            continue
        report.files_processed += 1

        try:
            df = pd.read_parquet(snap_path)
        except Exception as exc:
            report.files_skipped_error += 1
            report.errors.append(f"{snap_path.name}: read failed — {exc}")
            continue

        if _is_already_current_engine(df) and not overwrite_existing:
            report.files_skipped_already_migrated += 1
            continue

        if backfill_volume and (
            "dollar_vol_63d" not in df.columns or overwrite_existing
        ):
            if _volume_cache is None:
                from data.volume_features import DollarVolumeCache
                _volume_cache = DollarVolumeCache()
            try:
                from data.volume_features import add_dollar_volume_features
                asof = _stem_to_date(snap_path.stem)
                coverage = add_dollar_volume_features(
                    df, asof, _volume_cache, fallback_from_adv=True
                )
                report.volume_feature_coverage[snap_path.name] = coverage
                if "volume" not in df.columns and "current_price" in df.columns:
                    px = pd.to_numeric(df["current_price"], errors="coerce")
                    df["volume"] = (df["dollar_vol_21d"] / px.where(px > 0)).round(0)
            except ValueError:
                report.errors.append(
                    f"{snap_path.name}: unparseable stem date — volume backfill skipped"
                )

        try:
            _stem_asof = _stem_to_date(snap_path.stem)
        except ValueError:
            _stem_asof = None

        # SPARSE-vintage rescue (peer-3): 2025 files that never stored broker ratios
        # or the yfinance momentum block get FMP reconstructions as-of the stem date.
        if backfill_sparse and _stem_asof is not None:
            if _coverage(df, "pe_ratio") < 0.30:
                if _fund_cache is None:
                    from data.fundamental_features import FundamentalsCache
                    _fund_cache = FundamentalsCache()
                from data.fundamental_features import add_valuation_ratios_asof
                cov = add_valuation_ratios_asof(
                    df, _stem_asof, _fund_cache, fill_missing_only=True
                )
                report.valuation_backfill_coverage[snap_path.name] = cov
            # Top-up trigger is generous (< 0.95): PARTIAL vintages carry live
            # yfinance momentum for most rows — fill_missing_only reconstructs
            # only the gaps, never touching a live value.
            if _coverage(df, "return_3m") < 0.95:
                if _price_cache is None:
                    from data.price_features import PriceSeriesCache
                    _price_cache = PriceSeriesCache()
                from data.price_features import add_price_momentum_features
                cov = add_price_momentum_features(
                    df, _stem_asof, _price_cache, fill_missing_only=True
                )
                report.price_backfill_coverage[snap_path.name] = cov

        # peer-3 fundamental features for EVERY vintage (no historical file has them).
        if backfill_fundamentals and _stem_asof is not None and (
            "roe_ttm" not in df.columns or overwrite_existing
        ):
            if _fund_cache is None:
                from data.fundamental_features import FundamentalsCache
                _fund_cache = FundamentalsCache()
            from data.fundamental_features import add_fundamental_features
            cov = add_fundamental_features(df, _stem_asof, _fund_cache)
            report.fundamental_feature_coverage[snap_path.name] = cov

        if "sector" not in df.columns:
            df["sector"] = pd.NA
        if "industry" not in df.columns:
            df["industry"] = pd.NA
        report.rows_missing_sector += int(df["sector"].isna().sum())
        report.rows_missing_industry += int(df["industry"].isna().sum())

        # df.get(col, scalar) returns the bare scalar when the column is absent —
        # .fillna would crash; default to a zero Series instead.
        _vm_before = df["value_metric"] if "value_metric" in df.columns else pd.Series(0.0, index=df.index)
        before_metric = pd.to_numeric(_vm_before, errors="coerce").fillna(0.0)
        before_means.append(float(before_metric.mean()))

        try:
            from util import SCORE_WEIGHTS, SCORING_PARAMS
            cfg = scoring_cfg if scoring_cfg is not None else SCORING_PARAMS
            compute_metric(df, SCORE_WEIGHTS, cfg)
        except Exception as exc:
            report.files_skipped_error += 1
            report.errors.append(f"{snap_path.name}: rescore failed — {exc}")
            continue

        _vm_after = df["value_metric"] if "value_metric" in df.columns else pd.Series(0.0, index=df.index)
        after_metric = pd.to_numeric(_vm_after, errors="coerce").fillna(0.0)
        after_means.append(float(after_metric.mean()))
        report.nan_value_metric_rows += int(after_metric.isna().sum())
        report.rows_rescored += len(df)
        report.files_rescored += 1

        if "income_score" in df.columns:
            _inc = pd.to_numeric(df["income_score"], errors="coerce")
            _nz = _inc[_inc.notna() & (_inc != 0.0)]
            if len(_nz):
                report.income_top_mass[snap_path.name] = float(
                    _nz.round(4).value_counts().iloc[0] / len(_nz)
                )

        for col in ("value_fallback_reason", "quality_fallback_reason",
                    "momentum_fallback_reason", "income_fallback_reason"):
            if col in df.columns:
                vc = df[col].value_counts(dropna=False).to_dict()
                for reason, n in vc.items():
                    key = f"{col}:{reason}"
                    report.fallback_usage[key] = report.fallback_usage.get(key, 0) + int(n)

        # Strip legacy `*_v3`-suffixed columns to converge on canonical names.
        for col in _LEGACY_SUFFIXED_COLUMNS:
            if col in df.columns:
                del df[col]

        df["scoring_model_version"] = SCORING_MODEL_VERSION
        df["scoring_config_hash"] = cfg_hash
        df["rescore_timestamp"] = timestamp
        df["rescore_source_snapshot"] = snap_path.name

        if "symbol" in df.columns:
            diff = (after_metric - before_metric).abs()
            top_idx = diff.nlargest(min(20, len(diff))).index
            for i in top_idx:
                sym = str(df.at[i, "symbol"])
                shifts.append((sym, float(before_metric.at[i]), float(after_metric.at[i])))

        if dry_run:
            continue

        if output_dir is None:
            if in_place_with_backup:
                bak_path = snap_path.with_suffix(".bak.parquet")
                if not bak_path.exists():
                    shutil.copy2(snap_path, bak_path)
                    report.backups_created += 1
            target = snap_path
        else:
            target = out_path / snap_path.name

        try:
            from util import SNAPSHOT_PARAMS
            compression = SNAPSHOT_PARAMS.get("compression", "snappy")
            df.to_parquet(target, index=False, compression=compression)
        except Exception as exc:
            report.errors.append(f"{target.name}: write failed — {exc}")

    if before_means:
        report.before_value_metric_mean = float(sum(before_means) / len(before_means))
    if after_means:
        report.after_value_metric_mean = float(sum(after_means) / len(after_means))
    shifts.sort(key=lambda x: -abs(x[2] - x[1]))
    report.top_score_shifts = shifts[:20]
    return report
