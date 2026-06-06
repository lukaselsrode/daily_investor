"""
data/fmp_ops.py — Operational helpers for the Financial Modeling Prep cache.

These functions are the CLI-facing layer around data.fmp_client. They keep live
network calls explicit (allow_fetch=True only), quota-aware, resumable, and safe
for research workflows. Backtests continue to read the cache only.
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = "data/fmp_cache_adj"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2030-01-01"
STATEMENT_KINDS = ("income-statement", "balance-sheet-statement", "cash-flow-statement")


@dataclass(frozen=True)
class FMPCacheStatus:
    cache_dir: str
    key_present: bool
    calls_remaining: int | None
    price_files: int
    statement_files: int
    negative_meta_files: int
    delisted_roster_rows: int
    dead_universe_rows: int

    def pretty(self) -> str:
        rem = "unknown" if self.calls_remaining is None else str(self.calls_remaining)
        return "\n".join([
            "FMP cache status",
            f"  cache_dir:             {self.cache_dir}",
            f"  FMP_KEY present:       {self.key_present}",
            f"  calls remaining:       {rem}",
            f"  price files:           {self.price_files}",
            f"  statement files:       {self.statement_files}",
            f"  negative meta files:   {self.negative_meta_files}",
            f"  delisted roster rows:  {self.delisted_roster_rows}",
            f"  dead universe rows:    {self.dead_universe_rows}",
        ])


@dataclass(frozen=True)
class FMPBackfillReport:
    action: str
    requested: int
    fetched_or_cached: int
    skipped: int
    failed: int
    quota_remaining: int | None = None
    output_path: str | None = None

    def pretty(self) -> str:
        lines = [
            f"FMP {self.action} complete",
            f"  requested:         {self.requested}",
            f"  fetched_or_cached: {self.fetched_or_cached}",
            f"  skipped:           {self.skipped}",
            f"  failed:            {self.failed}",
        ]
        if self.quota_remaining is not None:
            lines.append(f"  calls remaining:   {self.quota_remaining}")
        if self.output_path:
            lines.append(f"  output:            {self.output_path}")
        return "\n".join(lines)


def _cache_dir() -> Path:
    return Path(os.environ.get("FMP_CACHE_DIR", DEFAULT_CACHE_DIR))


def _safe(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _price_path(symbol: str) -> Path:
    return _cache_dir() / "prices" / f"{_safe(symbol)}.parquet"


def _statement_path(symbol: str, kind: str) -> Path:
    return _cache_dir() / "statements" / kind / f"{_safe(symbol)}.json"


def _count_files(path: Path, pattern: str = "*") -> int:
    return sum(1 for p in path.rglob(pattern) if p.is_file()) if path.exists() else 0


def _read_parquet_len(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def _calls_remaining_from_cache(cache: Path) -> int | None:
    daily_cap = int(os.environ.get("FMP_DAILY_CAP", "0"))
    if daily_cap <= 0:
        return None
    qpath = cache / "meta" / "_quota.json"
    try:
        data = json.loads(qpath.read_text())
        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        count = int(data.get("count", 0)) if data.get("date") == today else 0
    except (OSError, ValueError, TypeError):
        count = 0
    return max(0, daily_cap - count)


def fmp_cache_status() -> FMPCacheStatus:
    """Summarize the local FMP cache without hitting the network."""
    cache = _cache_dir()
    calls_remaining = _calls_remaining_from_cache(cache)

    negative = 0
    meta_dir = cache / "meta"
    if meta_dir.exists():
        for path in meta_dir.glob("*.json"):
            if path.name == "_quota.json":
                continue
            try:
                if any(s in path.read_text() for s in ("premium", "empty", "error")):
                    negative += 1
            except OSError:
                pass

    return FMPCacheStatus(
        cache_dir=str(cache),
        key_present=bool(os.getenv("FMP_KEY")),
        calls_remaining=calls_remaining,
        price_files=_count_files(cache / "prices", "*.parquet"),
        statement_files=_count_files(cache / "statements", "*.json"),
        negative_meta_files=negative,
        delisted_roster_rows=_read_parquet_len(cache / "delisted_roster.parquet"),
        dead_universe_rows=_read_parquet_len(cache / "dead_universe.parquet"),
    )


def load_symbol_list(source: str = "current", *, max_symbols: int | None = None) -> list[str]:
    """Load symbols from the latest agg_data CSV, a CSV/text file, or a comma list."""
    symbols: list[str] = []
    if source == "current":
        from data.cache import read_data_as_pd
        df = read_data_as_pd("agg_data")
        if df is None or "symbol" not in df.columns:
            raise RuntimeError("No agg_data CSV with a symbol column found; run fetch-data first.")
        symbols = df["symbol"].dropna().astype(str).tolist()
    elif os.path.exists(source):
        path = Path(source)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = "symbol" if "symbol" in df.columns else df.columns[0]
            symbols = df[col].dropna().astype(str).tolist()
        else:
            symbols = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    else:
        symbols = [s.strip() for s in source.split(",") if s.strip()]

    deduped = list(dict.fromkeys(s.upper() for s in symbols if s and s != "nan"))
    return deduped[:max_symbols] if max_symbols else deduped


def backfill_prices(
    symbols: list[str],
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
) -> FMPBackfillReport:
    """Backfill adjusted EOD prices for symbols, resuming through fmp_client's cache."""
    from data import fmp_client as fmp

    ok = skipped = failed = 0
    for sym in symbols:
        if not force and _price_path(sym).exists():
            ok += 1
            continue
        try:
            df = fmp.eod_prices(sym, start, end, allow_fetch=True)
            if df is None or df.empty:
                skipped += 1
            else:
                ok += 1
        except fmp.FMPQuotaExceeded:
            logger.warning("FMP quota reached while backfilling prices at symbol %s", sym)
            break
        except Exception as exc:
            logger.warning("FMP price backfill failed for %s: %s", sym, exc)
            failed += 1
    return FMPBackfillReport(
        action="price backfill",
        requested=len(symbols),
        fetched_or_cached=ok,
        skipped=skipped,
        failed=failed,
        quota_remaining=fmp.calls_remaining(),
    )


def backfill_statements(
    symbols: list[str],
    *,
    kinds: list[str] | None = None,
    limit: int = 44,
    force: bool = False,
) -> FMPBackfillReport:
    """Backfill raw FMP statement JSON for symbols/kinds."""
    from data import fmp_client as fmp

    kinds = kinds or list(STATEMENT_KINDS)
    requested = len(symbols) * len(kinds)
    ok = skipped = failed = 0
    for sym in symbols:
        for kind in kinds:
            if not force and _statement_path(sym, kind).exists():
                ok += 1
                continue
            try:
                df = fmp.statement(sym, kind, limit=limit, allow_fetch=True)
                if df is None or df.empty:
                    skipped += 1
                else:
                    ok += 1
            except fmp.FMPQuotaExceeded:
                logger.warning("FMP quota reached while backfilling %s at symbol %s", kind, sym)
                return FMPBackfillReport(
                    action="statement backfill",
                    requested=requested,
                    fetched_or_cached=ok,
                    skipped=skipped,
                    failed=failed,
                    quota_remaining=fmp.calls_remaining(),
                )
            except Exception as exc:
                logger.warning("FMP %s backfill failed for %s: %s", kind, sym, exc)
                failed += 1
    return FMPBackfillReport(
        action="statement backfill",
        requested=requested,
        fetched_or_cached=ok,
        skipped=skipped,
        failed=failed,
        quota_remaining=fmp.calls_remaining(),
    )


def backfill_delisted_roster(*, max_pages: int = 50) -> FMPBackfillReport:
    """Fetch/cache the FMP delisted roster."""
    from data import fmp_client as fmp

    try:
        df = fmp.delisted_companies(max_pages=max_pages, allow_fetch=True)
        rows = 0 if df is None else len(df)
        failed = 0 if rows else 1
    except fmp.FMPQuotaExceeded:
        rows = 0
        failed = 1
    return FMPBackfillReport(
        action="delisted roster backfill",
        requested=max_pages,
        fetched_or_cached=rows,
        skipped=0,
        failed=failed,
        quota_remaining=fmp.calls_remaining(),
        output_path=str(_cache_dir() / "delisted_roster.parquet"),
    )


def build_dead_universe(
    *,
    min_adv: float = 500_000.0,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_symbols: int | None = None,
    allow_fetch_prices: bool = False,
) -> FMPBackfillReport:
    """Build dead_universe.parquet from delisted roster + cached/FMP prices.

    A delisted name is included when its max observed adjusted dollar volume over
    [start, end] clears min_adv. With allow_fetch_prices=False this never spends
    calls and only scans the existing cache.
    """
    from data import fmp_client as fmp

    roster_path = _cache_dir() / "delisted_roster.parquet"
    if roster_path.exists():
        roster = pd.read_parquet(roster_path)
    else:
        roster = fmp.delisted_companies(allow_fetch=True)
    if roster is None or roster.empty or "symbol" not in roster.columns:
        raise RuntimeError("No delisted roster available; run `daily-investor fmp backfill-delisted` first.")

    rows: list[dict] = []
    scanned = failed = skipped = 0
    for _, rec in roster.head(max_symbols or len(roster)).iterrows():
        sym = str(rec.get("symbol", "")).upper().strip()
        if not sym:
            continue
        scanned += 1
        try:
            px = fmp.eod_prices(sym, start, end, allow_fetch=allow_fetch_prices)
        except fmp.FMPQuotaExceeded:
            logger.warning("FMP quota reached while building dead universe at %s", sym)
            break
        except Exception as exc:
            logger.warning("Dead-universe scan failed for %s: %s", sym, exc)
            failed += 1
            continue
        if px is None or px.empty or "close" not in px.columns:
            skipped += 1
            continue
        vol = px["volume"] if "volume" in px.columns else pd.Series(0.0, index=px.index)
        close_arr = np.asarray(px["close"], dtype=float)
        vol_arr = np.asarray(vol, dtype=float)
        max_adv = float(np.nanmax(close_arr * vol_arr))
        if not np.isfinite(max_adv) or max_adv < min_adv:
            skipped += 1
            continue
        rows.append({
            "symbol": sym,
            "first_date": str(px.index.min()),
            "delist_date": str(rec.get("delistedDate") or rec.get("delisted_date") or px.index.max()),
            "max_adv": max_adv,
        })

    out = _cache_dir() / "dead_universe.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=["symbol", "first_date", "delist_date", "max_adv"])
    if out.exists():
        try:
            existing = pd.read_parquet(out)
            if "symbol" in existing.columns and not existing.empty:
                # Delisted roster endpoints can be plan/range-limited. Never shrink a previously
                # richer dead-universe cache just because a later roster fetch was partial.
                new_df = pd.concat([new_df, existing], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["symbol"], keep="first")
        except Exception as exc:
            logger.warning("Could not merge existing dead universe %s: %s", out, exc)
    new_df.to_parquet(out)
    return FMPBackfillReport(
        action="dead universe build",
        requested=scanned,
        fetched_or_cached=len(new_df),
        skipped=skipped,
        failed=failed,
        quota_remaining=fmp.calls_remaining(),
        output_path=str(out),
    )


def validate_cache(sample_symbols: list[str] | None = None) -> FMPBackfillReport:
    """Read-only sanity check for the FMP cache."""
    sample_symbols = sample_symbols or ["SPY", "AAPL"]
    ok = failed = skipped = 0
    from data import fmp_client as fmp

    for sym in sample_symbols:
        px = fmp.eod_prices(sym, "2023-01-01", "2024-01-01", allow_fetch=False)
        if px is not None and not px.empty and "close" in px.columns:
            ok += 1
        else:
            failed += 1
    for kind in STATEMENT_KINDS:
        stmt = fmp.statement("AAPL", kind, allow_fetch=False)
        if stmt is not None and not stmt.empty:
            ok += 1
        else:
            skipped += 1
    status = fmp_cache_status()
    if status.dead_universe_rows <= 0:
        skipped += 1
    return FMPBackfillReport(
        action="cache validation",
        requested=len(sample_symbols) + len(STATEMENT_KINDS) + 1,
        fetched_or_cached=ok,
        skipped=skipped,
        failed=failed,
        quota_remaining=status.calls_remaining,
    )
