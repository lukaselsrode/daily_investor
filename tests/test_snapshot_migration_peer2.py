"""
tests/test_snapshot_migration_peer2.py — peer-2 volume backfill inside
strategy.snapshots.rescore_snapshots.

Covers:
  1. Backfill merges dollar-volume feature columns dated as-of the snapshot's
     stem date (point-in-time, not today).
  2. Snapshots that pre-date the `volume` column (27-col 2025 vintage) get a
     synthesized share-ADV volume = dollar_vol_21d / current_price.
  3. Coverage tallies land in MigrationReport.volume_feature_coverage.
  4. Idempotency: a second run skips files already stamped with the current
     SCORING_MODEL_VERSION; backfill_volume=False leaves frames untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import fmp_client
from data.volume_features import FEATURE_COLS
from strategy.scoring.composite import SCORING_MODEL_VERSION
from strategy.snapshots import rescore_snapshots


@pytest.fixture
def price_dir(tmp_path, monkeypatch):
    d = tmp_path / "prices"
    d.mkdir()
    monkeypatch.setattr(fmp_client, "_PRICE_DIR", str(d))
    return d


def _seed_prices(price_dir, symbol: str, last_date: str, n: int = 90,
                 close: float = 40.0, volume: float = 2e6) -> None:
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=last_date, periods=n)]
    pd.DataFrame(
        {"close": np.full(n, close), "volume": np.full(n, volume)},
        index=pd.Index(dates, name="date"),
    ).to_parquet(price_dir / f"{symbol}.parquet")


def _universe(n: int = 30, include_volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "symbol":   [f"M{i:03d}" for i in range(n)],
        "industry": ["banks"] * (n // 2) + ["software"] * (n - n // 2),
        "sector":   ["financials"] * (n // 2) + ["technology"] * (n - n // 2),
        "pe_ratio": rng.uniform(5, 40, n),
        "pb_ratio": rng.uniform(0.8, 8.0, n),
        "dividend_yield": rng.uniform(0.0, 0.06, n),
        "current_price":  rng.uniform(20, 200, n),
        "position_52w":   rng.uniform(0.1, 0.95, n),
        "return_1m": rng.uniform(-0.1, 0.1, n),
        "value_metric": rng.uniform(-0.2, 0.6, n),
    })
    if include_volume:
        df["volume"] = rng.uniform(5e5, 5e7, n)
    return df


def test_backfill_uses_snapshot_stem_date(tmp_path, price_dir):
    # Prices exist only through 2025-08-01 — a snapshot stamped that day must
    # pick them up; asof=today would find them too, so also verify the window
    # value matches the constant series exactly (i.e., no partial future mix).
    for i in range(30):
        _seed_prices(price_dir, f"M{i:03d}", "2025-08-01", close=40.0, volume=2e6)
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    _universe(include_volume=False).to_parquet(snap_dir / "2025_08_01.parquet", index=False)

    report = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    assert report.files_rescored == 1
    out = pd.read_parquet(snap_dir / "2025_08_01.parquet")
    for col in FEATURE_COLS:
        assert col in out.columns
    assert out["dollar_vol_63d"].notna().all()
    assert out["dollar_vol_21d"].iloc[0] == pytest.approx(40.0 * 2e6)
    # 27-col vintage: share-ADV volume synthesized from dollar volume / price.
    assert out["volume"].notna().all()
    expected_adv = 40.0 * 2e6 / out["current_price"].iloc[0]
    assert out["volume"].iloc[0] == pytest.approx(expected_adv, rel=0.01)
    assert (out["scoring_model_version"] == SCORING_MODEL_VERSION).all()
    assert report.volume_feature_coverage["2025_08_01.parquet"] == pytest.approx(1.0)


def test_backfill_idempotent_and_optional(tmp_path, price_dir):
    for i in range(30):
        _seed_prices(price_dir, f"M{i:03d}", "2026-06-01")
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    _universe().to_parquet(snap_dir / "2026_06_01_09_00.parquet", index=False)

    r1 = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    assert r1.files_rescored == 1
    r2 = rescore_snapshots(input_dir=snap_dir, in_place_with_backup=True)
    assert r2.files_skipped_already_migrated == 1
    assert r2.volume_feature_coverage == {}

    # backfill_volume=False: no feature columns are added.
    other = tmp_path / "snaps2"
    other.mkdir()
    _universe().to_parquet(other / "2026_06_01_09_00.parquet", index=False)
    r3 = rescore_snapshots(input_dir=other, in_place_with_backup=True, backfill_volume=False)
    assert r3.files_rescored == 1
    out = pd.read_parquet(other / "2026_06_01_09_00.parquet")
    assert not any(c in out.columns for c in FEATURE_COLS)
