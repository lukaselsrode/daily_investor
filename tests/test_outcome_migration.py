"""
tests/test_outcome_migration.py — decision-ledger score migration (cli.migrate_outcomes).

Covers:
  1. Score columns are rewritten from the nearest (<= 7d) rescored snapshot;
     rows without a joinable snapshot are left untouched.
  2. scores_model_version stamping makes the migration idempotent.
  3. Backups are created; dry_run writes nothing.
  4. buy_context.csv gets the same treatment keyed on buy_date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategy.snapshots as snap
from strategy.scoring.composite import SCORING_MODEL_VERSION


@pytest.fixture
def snap_dir(tmp_path, monkeypatch):
    d = tmp_path / "snapshots"
    d.mkdir()
    monkeypatch.setattr(snap, "_snapshot_dir", lambda: d)
    return d


def _snapshot_frame(n: int = 60, vm: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(n)],
        "value_metric": rng.uniform(-0.5, 1.0, n),
        "value_score": rng.uniform(-0.5, 1.0, n),
        "quality_score": rng.uniform(-0.5, 1.0, n),
        "income_score": rng.uniform(0.0, 1.0, n),
        "momentum_score": rng.uniform(-0.5, 1.0, n),
        "scoring_model_version": SCORING_MODEL_VERSION,
    })
    df.loc[0, "value_metric"] = vm  # S000 pinned for assertions
    return df


def test_outcomes_migrated_from_nearest_snapshot(snap_dir, tmp_path, monkeypatch):
    _snapshot_frame(vm=0.42).to_parquet(snap_dir / "2026_08_01_09_00.parquet", index=False)

    outcomes = pd.DataFrame({
        "symbol": ["S000", "ZZZZ", "S001"],
        "decision_date": ["2026-08-03", "2026-08-03", "2025-01-01"],  # S001: no snapshot in 7d
        "buy_date": ["2026-08-01", None, None],
        "value_score": [9.9, 9.9, 9.9],
        "quality_score": [9.9, 9.9, 9.9],
        "income_score": [9.9, 9.9, 9.9],
        "momentum_score": [9.9, 9.9, 9.9],
        "current_value_metric": [9.9, 9.9, 9.9],
        "score_at_buy": [9.9, 9.9, 9.9],
        "score_delta": [9.9, 9.9, 9.9],
        "rank_percentile": [9.9, 9.9, 9.9],
        "rank_at_buy": [9.9, 9.9, 9.9],
        "rank_delta": [9.9, 9.9, 9.9],
    })
    opath = tmp_path / "decision_outcomes.parquet"
    outcomes.to_parquet(opath, index=False)
    import portfolio.outcome_tracker as ot
    monkeypatch.setattr(ot, "_outcomes_path", lambda: opath)

    from cli.migrate_outcomes import migrate_decision_outcomes
    stats = migrate_decision_outcomes()
    assert stats["rewritten"] == 1          # S000 joined
    assert stats["no_snapshot"] == 2        # ZZZZ unknown symbol, S001 too far

    out = pd.read_parquet(opath)
    row = out[out["symbol"] == "S000"].iloc[0]
    assert row["current_value_metric"] == pytest.approx(0.42)
    assert row["score_at_buy"] == pytest.approx(0.42)   # same snapshot serves buy_date
    assert row["score_delta"] == pytest.approx(0.0)
    assert row["scores_model_version"] == SCORING_MODEL_VERSION
    untouched = out[out["symbol"] == "S001"].iloc[0]
    assert untouched["value_score"] == pytest.approx(9.9)
    # Backup created
    assert list(tmp_path.glob("decision_outcomes.parquet.pre_peer3_*.bak"))

    # Idempotent: second run skips the migrated row.
    stats2 = migrate_decision_outcomes()
    assert stats2["skipped_current"] == 1
    assert stats2["rewritten"] == 0


def test_dry_run_writes_nothing(snap_dir, tmp_path, monkeypatch):
    _snapshot_frame().to_parquet(snap_dir / "2026_08_01_09_00.parquet", index=False)
    outcomes = pd.DataFrame({
        "symbol": ["S000"], "decision_date": ["2026-08-03"], "buy_date": [None],
        "value_score": [9.9], "quality_score": [9.9], "income_score": [9.9],
        "momentum_score": [9.9], "current_value_metric": [9.9],
        "rank_percentile": [9.9],
    })
    opath = tmp_path / "decision_outcomes.parquet"
    outcomes.to_parquet(opath, index=False)
    import portfolio.outcome_tracker as ot
    monkeypatch.setattr(ot, "_outcomes_path", lambda: opath)

    from cli.migrate_outcomes import migrate_decision_outcomes
    stats = migrate_decision_outcomes(dry_run=True)
    assert stats["rewritten"] == 1
    out = pd.read_parquet(opath)
    assert out.loc[0, "value_score"] == pytest.approx(9.9)  # untouched on disk
    assert not list(tmp_path.glob("*.bak"))


def test_buy_context_migrated(snap_dir, tmp_path, monkeypatch):
    _snapshot_frame(vm=0.33).to_parquet(snap_dir / "2026_08_01_09_00.parquet", index=False)
    ctx = pd.DataFrame({
        "symbol": ["S000"],
        "buy_date": ["2026-08-02"],
        "composite_score_at_buy": [9.9],
        "quality_score_at_buy": [9.9],
        "momentum_score_at_buy": [9.9],
        "income_score_at_buy": [9.9],
        "value_score_at_buy": [9.9],
        "universe_rank_pct_at_buy": [9.9],
    })
    cpath = tmp_path / "buy_context.csv"
    ctx.to_csv(cpath, index=False)
    import portfolio.buy_context as bc
    monkeypatch.setattr(bc, "_context_path", lambda: cpath)

    from cli.migrate_outcomes import migrate_buy_context
    stats = migrate_buy_context()
    assert stats["rewritten"] == 1
    out = pd.read_csv(cpath)
    assert out.loc[0, "composite_score_at_buy"] == pytest.approx(0.33)
    assert out.loc[0, "universe_rank_pct_at_buy"] <= 1.0
    assert out.loc[0, "scores_model_version"] == SCORING_MODEL_VERSION
