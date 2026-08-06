"""
cli/migrate_outcomes.py — migrate frozen decision-ledger scores to the current engine.

data/decision_outcomes.parquet and data/buy_context.csv store factor scores frozen at
decision time. After a scoring-model migration (peer-3) those frozen values are on the
OLD scale and can't be compared with rescored snapshots or new decisions. This tool
rewrites the score columns from the RESCORED snapshot nearest each decision date
(<= MAX_JOIN_DAYS away), with timestamped backups, and stamps `scores_model_version`
for idempotency.

Run AFTER `daily-investor snapshots rescore` — the join reads the snapshot store and
inherits whatever engine version the snapshots carry.

Columns rewritten (when a joined snapshot row exists):
  decision_outcomes: value/quality/income/momentum_score, current_value_metric,
                     rank_percentile (+ score_at_buy/rank_at_buy/deltas via buy_date)
  buy_context:       composite/quality/momentum/income/value_score_at_buy,
                     universe_rank_pct_at_buy
`conditional_momentum_score` is a decision-time interaction feature with no snapshot
column — left frozen.
"""
from __future__ import annotations

import datetime
import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MAX_JOIN_DAYS = 7

_CURRENT_COLS = {
    "value_score": "value_score",
    "quality_score": "quality_score",
    "income_score": "income_score",
    "momentum_score": "momentum_score",
    "current_value_metric": "value_metric",
}

_BUY_CTX_COLS = {
    "composite_score_at_buy": "value_metric",
    "quality_score_at_buy": "quality_score",
    "momentum_score_at_buy": "momentum_score",
    "income_score_at_buy": "income_score",
    "value_score_at_buy": "value_score",
}


class _SnapshotJoiner:
    """Lazy date → scored-frame lookup over the snapshot store (latest run per day)."""

    def __init__(self) -> None:
        from strategy.snapshots import list_snapshots
        by_date: dict[datetime.date, Path] = {}
        for d, p in list_snapshots():
            by_date[d] = p  # ascending → last intraday run wins
        self._paths = by_date
        self._dates = sorted(by_date)
        self._frames: dict[datetime.date, pd.DataFrame | None] = {}

    def nearest_date(self, target: datetime.date) -> datetime.date | None:
        best, best_diff = None, MAX_JOIN_DAYS + 1
        for d in self._dates:
            diff = abs((d - target).days)
            if diff < best_diff:
                best, best_diff = d, diff
        return best

    def frame(self, d: datetime.date) -> pd.DataFrame | None:
        if d not in self._frames:
            try:
                cols = ["symbol", "value_metric", "value_score", "quality_score",
                        "income_score", "momentum_score", "scoring_model_version"]
                df = pd.read_parquet(self._paths[d])
                keep = [c for c in cols if c in df.columns]
                df = df[keep].copy()
                df["_rank_pct"] = pd.to_numeric(df["value_metric"], errors="coerce").rank(pct=True)
                self._frames[d] = df.drop_duplicates(subset=["symbol"], keep="last").set_index("symbol")
            except Exception as exc:
                logger.warning("migrate-outcomes: unreadable snapshot for %s: %s", d, exc)
                self._frames[d] = None
        return self._frames[d]

    def lookup(self, symbol: str, target: datetime.date):
        """(snapshot row, engine version) for symbol near target, or (None, None)."""
        d = self.nearest_date(target)
        if d is None:
            return None, None
        frame = self.frame(d)
        if frame is None or symbol not in frame.index:
            return None, None
        row = frame.loc[symbol]
        version = row.get("scoring_model_version")
        return row, (str(version) if pd.notna(version) else None)


def _parse_date(value) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _backup(path: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_name(f"{path.name}.pre_peer3_{stamp}.bak")
    shutil.copy2(path, bak)
    return bak


def migrate_decision_outcomes(dry_run: bool = False) -> dict:
    """Rewrite decision_outcomes.parquet score columns from rescored snapshots."""
    from portfolio.outcome_tracker import _outcomes_path
    from strategy.scoring.composite import SCORING_MODEL_VERSION

    path = Path(_outcomes_path())
    stats = {"rows": 0, "rewritten": 0, "skipped_current": 0, "no_snapshot": 0, "backup": ""}
    if not path.exists():
        logger.info("migrate-outcomes: %s not found — nothing to migrate", path)
        return stats

    df = pd.read_parquet(path)
    stats["rows"] = len(df)
    if "scores_model_version" not in df.columns:
        df["scores_model_version"] = pd.NA

    joiner = _SnapshotJoiner()
    for i in df.index:
        if str(df.at[i, "scores_model_version"]) == SCORING_MODEL_VERSION:
            stats["skipped_current"] += 1
            continue
        sym = str(df.at[i, "symbol"]) if pd.notna(df.at[i, "symbol"]) else ""
        ddate = _parse_date(df.at[i, "decision_date"]) if "decision_date" in df.columns else None
        if not sym or ddate is None:
            stats["no_snapshot"] += 1
            continue
        row, version = joiner.lookup(sym, ddate)
        if row is None:
            stats["no_snapshot"] += 1
            continue
        for out_col, snap_col in _CURRENT_COLS.items():
            if out_col in df.columns and snap_col in row.index and pd.notna(row[snap_col]):
                df.at[i, out_col] = float(row[snap_col])
        if "rank_percentile" in df.columns and pd.notna(row["_rank_pct"]):
            df.at[i, "rank_percentile"] = float(row["_rank_pct"])

        bdate = _parse_date(df.at[i, "buy_date"]) if "buy_date" in df.columns else None
        if bdate is not None:
            brow, _ = joiner.lookup(sym, bdate)
            if brow is not None and pd.notna(brow.get("value_metric")):
                at_buy = float(brow["value_metric"])
                if "score_at_buy" in df.columns:
                    df.at[i, "score_at_buy"] = at_buy
                if "score_delta" in df.columns and pd.notna(row.get("value_metric")):
                    df.at[i, "score_delta"] = float(row["value_metric"]) - at_buy
                if "rank_at_buy" in df.columns and pd.notna(brow["_rank_pct"]):
                    df.at[i, "rank_at_buy"] = float(brow["_rank_pct"])
                    if "rank_delta" in df.columns and pd.notna(row["_rank_pct"]):
                        df.at[i, "rank_delta"] = float(row["_rank_pct"]) - float(brow["_rank_pct"])
        df.at[i, "scores_model_version"] = version or SCORING_MODEL_VERSION
        stats["rewritten"] += 1

    if dry_run:
        return stats
    stats["backup"] = str(_backup(path))
    df.to_parquet(path, index=False)
    logger.info("migrate-outcomes: %s — %d/%d rows rewritten (backup %s)",
                path.name, stats["rewritten"], stats["rows"], stats["backup"])
    return stats


def migrate_buy_context(dry_run: bool = False) -> dict:
    """Rewrite buy_context.csv score columns from rescored snapshots."""
    from portfolio.buy_context import _context_path
    from strategy.scoring.composite import SCORING_MODEL_VERSION

    path = Path(_context_path())
    stats = {"rows": 0, "rewritten": 0, "skipped_current": 0, "no_snapshot": 0, "backup": ""}
    if not path.exists():
        logger.info("migrate-outcomes: %s not found — nothing to migrate", path)
        return stats

    df = pd.read_csv(path)
    stats["rows"] = len(df)
    if "scores_model_version" not in df.columns:
        df["scores_model_version"] = pd.NA

    joiner = _SnapshotJoiner()
    for i in df.index:
        if str(df.at[i, "scores_model_version"]) == SCORING_MODEL_VERSION:
            stats["skipped_current"] += 1
            continue
        sym = str(df.at[i, "symbol"]) if pd.notna(df.at[i, "symbol"]) else ""
        bdate = _parse_date(df.at[i, "buy_date"]) if "buy_date" in df.columns else None
        if not sym or bdate is None:
            stats["no_snapshot"] += 1
            continue
        row, version = joiner.lookup(sym, bdate)
        if row is None:
            stats["no_snapshot"] += 1
            continue
        for ctx_col, snap_col in _BUY_CTX_COLS.items():
            if ctx_col in df.columns and snap_col in row.index and pd.notna(row[snap_col]):
                df.at[i, ctx_col] = float(row[snap_col])
        if "universe_rank_pct_at_buy" in df.columns and pd.notna(row["_rank_pct"]):
            df.at[i, "universe_rank_pct_at_buy"] = round(float(row["_rank_pct"]), 3)
        df.at[i, "scores_model_version"] = version or SCORING_MODEL_VERSION
        stats["rewritten"] += 1

    if dry_run:
        return stats
    stats["backup"] = str(_backup(path))
    df.to_csv(path, index=False)
    logger.info("migrate-outcomes: %s — %d/%d rows rewritten (backup %s)",
                path.name, stats["rewritten"], stats["rows"], stats["backup"])
    return stats


def cmd_migrate_outcomes(dry_run: bool = False) -> None:
    """CLI entry: migrate both ledgers, print a summary."""
    for name, fn in (("decision_outcomes", migrate_decision_outcomes),
                     ("buy_context", migrate_buy_context)):
        stats = fn(dry_run=dry_run)
        print(f"{name}: {stats['rewritten']}/{stats['rows']} rewritten, "
              f"{stats['skipped_current']} already current, "
              f"{stats['no_snapshot']} without a joinable snapshot"
              + (f", backup: {stats['backup']}" if stats["backup"] else "")
              + (" [DRY RUN]" if dry_run else ""))
