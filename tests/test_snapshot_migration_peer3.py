"""
tests/test_snapshot_migration_peer3.py — peer-3 fundamental/SPARSE backfill inside
strategy.snapshots.rescore_snapshots.

Covers:
  1. Fundamental features are merged as-of the snapshot's STEM date (PIT — a filing
     after the stem date is invisible even though it exists in the cache).
  2. SPARSE vintages (no pe_ratio / no momentum block) get FMP-reconstructed
     valuation ratios and price features, fill-missing-only.
  3. Coverage tallies land in the MigrationReport.
  4. Idempotency: a second run skips files stamped with the current version.
  5. Rescored files carry the peer-3 stamp and finite quality/income scores.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from data import fmp_client
from strategy.scoring.composite import SCORING_MODEL_VERSION
from strategy.snapshots import rescore_snapshots


@pytest.fixture
def caches(tmp_path, monkeypatch):
    stmt = tmp_path / "statements"
    for kind in ("income-statement", "balance-sheet-statement", "cash-flow-statement", "dividends"):
        (stmt / kind).mkdir(parents=True)
    prices = tmp_path / "prices"
    prices.mkdir()
    monkeypatch.setattr(fmp_client, "_STMT_DIR", str(stmt))
    monkeypatch.setattr(fmp_client, "_PRICE_DIR", str(prices))
    return stmt, prices


def _seed_statements(stmt_dir, symbol: str, last_filing: str = "2025-07-15", n: int = 12) -> None:
    e = pd.Timestamp(last_filing)
    dates = [(e - pd.DateOffset(months=3 * i)).strftime("%Y-%m-%d") for i in range(n)][::-1]
    inc, cf, bal = [], [], []
    for d in dates:
        fy, q = int(d[:4]), f"Q{pd.Timestamp(d).quarter}"
        inc.append({"fiscalYear": fy, "period": q, "filingDate": d, "netIncome": 100.0,
                    "revenue": 1000.0, "grossProfit": 400.0, "epsDiluted": 1.0,
                    "weightedAverageShsOutDil": 100.0})
        cf.append({"fiscalYear": fy, "period": q, "filingDate": d, "operatingCashFlow": 120.0,
                   "freeCashFlow": 90.0, "commonDividendsPaid": -30.0})
        bal.append({"fiscalYear": fy, "period": q, "filingDate": d, "totalAssets": 4000.0,
                    "totalStockholdersEquity": 2000.0, "totalDebt": 800.0})
    for kind, rows in (("income-statement", inc), ("cash-flow-statement", cf),
                       ("balance-sheet-statement", bal)):
        with open(os.path.join(stmt_dir, kind, f"{symbol}.json"), "w") as fh:
            json.dump(rows, fh)


def _seed_prices(price_dir, symbol: str, last_date: str = "2025-08-01", n: int = 300,
                 close: float = 40.0) -> None:
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=last_date, periods=n)]
    pd.DataFrame(
        {"close": np.full(n, close), "volume": np.full(n, 2e6)},
        index=pd.Index(dates, name="date"),
    ).to_parquet(price_dir / f"{symbol}.parquet")


def _full_universe(n: int = 30) -> pd.DataFrame:
    """A modern-vintage frame: broker ratios + momentum present, fundamentals absent."""
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "symbol":   [f"M{i:03d}" for i in range(n)],
        "industry": ["banks"] * (n // 2) + ["software"] * (n - n // 2),
        "sector":   ["financials"] * (n // 2) + ["technology"] * (n - n // 2),
        "pe_ratio": rng.uniform(5, 40, n),
        "pb_ratio": rng.uniform(0.8, 8.0, n),
        "dividend_yield": rng.uniform(0.0, 0.06, n),
        "current_price":  rng.uniform(20, 200, n),
        "position_52w":   rng.uniform(0.1, 0.95, n),
        "return_1m": rng.uniform(-0.1, 0.1, n),
        "return_3m": rng.uniform(-0.2, 0.2, n),
        "volume":    rng.uniform(5e5, 5e7, n),
        "value_metric": rng.uniform(-0.2, 0.6, n),
    })


def _sparse_universe(n: int = 30) -> pd.DataFrame:
    """2025 SPARSE vintage: symbols + scores only, no ratios, no momentum block."""
    rng = np.random.default_rng(13)
    return pd.DataFrame({
        "symbol":   [f"M{i:03d}" for i in range(n)],
        "industry": ["banks"] * (n // 2) + ["software"] * (n - n // 2),
        "sector":   ["financials"] * (n // 2) + ["technology"] * (n - n // 2),
        "current_price": rng.uniform(20, 200, n),
        "return_1m": rng.uniform(-0.1, 0.1, n),
        "value_metric": rng.uniform(-0.2, 0.6, n),
    })


def test_fundamental_backfill_is_pit_at_stem_date(tmp_path, caches):
    stmt, prices = caches
    for i in range(30):
        _seed_statements(stmt, f"M{i:03d}", last_filing="2025-07-15")
        _seed_prices(prices, f"M{i:03d}")
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    _full_universe().to_parquet(snap_dir / "2025_08_01.parquet", index=False)
    # A snapshot dated BEFORE any filing exists: fundamentals must stay NaN.
    _full_universe().to_parquet(snap_dir / "2022_01_03.parquet", index=False)

    report = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    assert report.files_rescored == 2
    fresh = pd.read_parquet(snap_dir / "2025_08_01.parquet")
    assert fresh["roe_ttm"].notna().all()
    assert fresh.loc[0, "roe_ttm"] == pytest.approx(400.0 / 2000.0)
    assert (fresh["scoring_model_version"] == SCORING_MODEL_VERSION).all()
    assert fresh["quality_score"].notna().all()
    # PIT: at 2022-01-03 the earliest filing (~2022-10) hasn't happened yet.
    stale = pd.read_parquet(snap_dir / "2022_01_03.parquet")
    assert stale["roe_ttm"].isna().all()
    assert (stale["quality_fallback_reason"] == "no_fundamentals").all()
    assert report.fundamental_feature_coverage["2025_08_01.parquet"] == pytest.approx(1.0)


def test_sparse_vintage_gets_valuation_and_price_rescue(tmp_path, caches):
    stmt, prices = caches
    for i in range(30):
        _seed_statements(stmt, f"M{i:03d}", last_filing="2025-07-15")
        _seed_prices(prices, f"M{i:03d}", close=40.0)
    _seed_prices(prices, "SPY", close=500.0)
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    _sparse_universe().to_parquet(snap_dir / "2025_08_01.parquet", index=False)

    report = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    out = pd.read_parquet(snap_dir / "2025_08_01.parquet")
    # Valuation reconstructed from statements + frame price (fill-missing-only).
    assert out["pe_ratio"].notna().all()
    assert out["pb_ratio"].notna().all()
    # Price/momentum features reconstructed from the FMP price cache.
    assert out["return_3m"].notna().all()
    assert out["rs_3m"].notna().all()
    assert report.valuation_backfill_coverage["2025_08_01.parquet"] == pytest.approx(1.0)
    assert report.price_backfill_coverage["2025_08_01.parquet"] == pytest.approx(1.0)
    # Existing current_price / return_1m were NOT overwritten.
    src = _sparse_universe()
    pd.testing.assert_series_equal(
        out["return_1m"], src["return_1m"], check_names=False, check_exact=False, atol=1e-9
    )


def test_rescore_idempotent(tmp_path, caches):
    stmt, prices = caches
    for i in range(30):
        _seed_statements(stmt, f"M{i:03d}")
        _seed_prices(prices, f"M{i:03d}")
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    _full_universe().to_parquet(snap_dir / "2025_08_01.parquet", index=False)

    r1 = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    assert r1.files_rescored == 1
    r2 = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    assert r2.files_skipped_already_migrated == 1
    assert r2.fundamental_feature_coverage == {}
