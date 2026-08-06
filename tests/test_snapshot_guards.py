"""
tests/test_snapshot_guards.py — store-integrity guards in strategy.snapshots.

A degraded pipeline run (universe collapse, scoring skipped) once wrote 2-row
frames without score columns into data/snapshots/, poisoning list_snapshots(),
forward-IC, and the rescore loop. Guards:

  1. save_snapshot refuses frames below MIN_SNAPSHOT_ROWS or missing the
     contract columns (symbol, value_metric) — returns an empty Path, no file.
  2. list_snapshots skips parquets whose schema lacks the contract columns.
  3. rescore_snapshots tallies malformed files as errors instead of rescoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategy.snapshots as snap
from strategy.snapshots import (
    MIN_SNAPSHOT_ROWS,
    list_snapshots,
    rescore_snapshots,
    save_snapshot,
)


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    d = tmp_path / "snapshots"
    d.mkdir()
    monkeypatch.setattr(snap, "_snapshot_dir", lambda: d)
    return d


def _scored_frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "symbol": [f"S{i:04d}" for i in range(n)],
        "value_metric": rng.uniform(-0.5, 1.0, n),
        "current_price": rng.uniform(5, 500, n),
    })


def test_save_refuses_tiny_frame(snap_dir):
    path = save_snapshot(_scored_frame(2))
    assert path == snap.Path()
    assert not list(snap_dir.glob("*.parquet"))


def test_save_refuses_missing_contract_columns(snap_dir):
    df = _scored_frame(MIN_SNAPSHOT_ROWS + 10).drop(columns=["value_metric"])
    path = save_snapshot(df)
    assert path == snap.Path()
    assert not list(snap_dir.glob("*.parquet"))


def test_save_accepts_valid_frame(snap_dir):
    path = save_snapshot(_scored_frame(MIN_SNAPSHOT_ROWS))
    assert path.exists()


def test_list_snapshots_skips_malformed(snap_dir):
    _scored_frame(60).to_parquet(snap_dir / "2026_08_01_09_00.parquet", index=False)
    # Malformed: profile-only frame with no scores (the observed junk shape).
    pd.DataFrame({"symbol": ["AAPL", "MSFT"], "pe_ratio": [12.0, 30.0]}).to_parquet(
        snap_dir / "2026_08_02_09_00.parquet", index=False
    )
    entries = list_snapshots()
    assert [p.name for _, p in entries] == ["2026_08_01_09_00.parquet"]


def test_rescore_skips_malformed(snap_dir, tmp_path):
    d = tmp_path / "rescore_in"
    d.mkdir()
    pd.DataFrame({"symbol": ["AAPL", "MSFT"], "pe_ratio": [12.0, 30.0]}).to_parquet(
        d / "2026_08_02_09_00.parquet", index=False
    )
    report = rescore_snapshots(input_dir=d, dry_run=True, backfill_volume=False)
    assert report.files_rescored == 0
    assert report.files_skipped_error == 1
    assert "malformed" in report.errors[0]
