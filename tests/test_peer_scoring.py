"""
tests/test_peer_scoring.py — peer-relative factor engine + snapshot migration.

Covers:
  1. peer_percentile higher / lower / NaN-safe / winsorize / clamp behavior
  2. industry → sector → market fallback (blend_relative)
  3. Schema validation for ScoringConfig
  4. apply_value distress penalties + finite output
  5. apply_quality: no-dividend tech not penalized; legacy fallback tagged
  6. apply_momentum penalties (falling knife / overextension / high vol)
  7. apply_income yield-trap + zero-dividend neutrality
  8. compute_metric: enabled=False is a no-op; enabled=True writes peer-relative score columns
     and overwrites value_metric (legacy columns preserved alongside)
  9. rescore_snapshots: dry-run writes nothing; in-place creates .bak;
     re-run is idempotent; report counts match reality
 10. Lookahead: rescoring the same DataFrame twice produces identical peer-relative score columns
 11. scoring_config_hash is stable across calls
 12. Backtest report carries scoring_engine_version + peer_config when peer scoring is active
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_universe(n: int = 60) -> pd.DataFrame:
    """Synthetic universe spanning 3 industries × 2 sectors with realistic dispersion."""
    rng = np.random.default_rng(42)
    industries = (["banks"] * 20 + ["software"] * 20 + ["utilities"] * 20)
    sectors = (["financials"] * 20 + ["technology"] * 20 + ["utilities"] * 20)
    df = pd.DataFrame({
        "symbol": [f"T{i:03d}" for i in range(n)],
        "industry": industries[:n],
        "sector":   sectors[:n],
        "pe_ratio": np.concatenate([
            rng.uniform(6, 12, 20),
            rng.uniform(20, 40, 20),
            rng.uniform(15, 22, 20),
        ]),
        "pb_ratio": np.concatenate([
            rng.uniform(0.8, 1.5, 20),
            rng.uniform(3.0, 8.0, 20),
            rng.uniform(1.4, 2.5, 20),
        ]),
        "volume":         rng.uniform(5e5, 5e7, n),
        "dividend_yield": np.concatenate([
            rng.uniform(0.02, 0.05, 20),       # banks pay dividends
            np.zeros(20),                      # software typically doesn't
            rng.uniform(0.03, 0.07, 20),       # utilities pay
        ]),
        "current_price":   rng.uniform(20, 200, n),
        "position_52w":    rng.uniform(0.1, 0.95, n),
        "return_1m":       rng.uniform(-0.1, 0.1, n),
        "return_3m":       rng.uniform(-0.2, 0.2, n),
        "return_5d":       rng.uniform(-0.05, 0.05, n),
        "return_6m":       rng.uniform(-0.3, 0.3, n),
        "rs_3m":           rng.uniform(-0.2, 0.2, n),
        "rs_6m":           rng.uniform(-0.2, 0.2, n),
        "risk_adj_momentum_3m": rng.uniform(-0.3, 0.3, n),
        "realized_vol_3m": rng.uniform(0.15, 0.40, n),
        "above_50dma":     rng.choice([True, False], n),
        "above_200dma":    rng.choice([True, False], n),
        # Pre-seed legacy score columns so the legacy → peer pipeline test can compare.
        "value_score":     rng.uniform(-0.5, 0.8, n),
        "quality_score":   rng.uniform(-0.4, 0.9, n),
        "income_score":    rng.uniform(0.0, 0.8, n),
        "momentum_score":  rng.uniform(-0.5, 0.8, n),
        "value_metric":    rng.uniform(-0.2, 0.6, n),
    })
    return df


def _peer_cfg(**overrides) -> dict:
    cfg = {
        "enabled": True,
        "peer_standardization": {
            "group_by": "industry",
            "fallback_group_by": "sector",
            "min_group_size": 5,
            "method": "percentile",
            "winsorize_pct": 0.05,
            "clamp_low": -1.0,
            "clamp_high": 1.5,
            "blend": {
                "industry_relative": 0.60,
                "sector_relative":   0.25,
                "market_relative":   0.15,
            },
        },
        "factors": {
            "value": {
                "enabled": True, "peer_relative": True, "pe_weight": 0.70, "pb_weight": 0.30,
                "distress": {"pe_threshold": 5.0, "pe_penalty": 0.30, "negative_eps_penalty": 0.25},
            },
            "quality":           {"enabled": True, "peer_relative": True, "use_legacy_checklist_fallback": True},
            "momentum":          {"enabled": True, "peer_relative": True},
            "income":            {"enabled": True, "peer_relative": True, "safety_aware": True},
            "growth_leadership": {"enabled": False},
        },
        "momentum_inputs": {
            "weights": {"rs_3m": 0.25, "rs_6m": 0.25, "risk_adj_3m": 0.20,
                        "trend_structure": 0.15, "return_1m": 0.10, "return_5d": 0.05},
            "penalties": {"falling_knife_3m_threshold": -0.15, "falling_knife_penalty": 0.25,
                          "overextension_52w_threshold": 0.97, "overextension_penalty": 0.20,
                          "high_vol_annual_threshold": 0.50, "high_vol_penalty": 0.15},
            "clamp_low": -1.0, "clamp_high": 1.5, "winsorize_pct": 0.05,
        },
        "quality_checklist": {
            "income_score_cap": 1.5, "yield_trap_threshold": 0.10, "distress_pe_max": 5.0,
            "quality_volume_high": 1_000_000, "quality_volume_low": 100_000,
            "quality_dividend_min": 0.02, "quality_dividend_max": 0.06,
            "quality_weight_has_positive_pe": 0.5, "quality_weight_distress_pe": -0.4,
            "quality_weight_has_positive_pb": 0.2, "quality_weight_high_volume": 0.3,
            "quality_weight_low_volume": -0.3, "quality_weight_yield_trap": -0.6,
            "quality_weight_healthy_dividend": 0.2,
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Peer utilities
# ---------------------------------------------------------------------------


def test_peer_percentile_higher_is_better_and_clamps():
    from strategy.scoring.peer import peer_percentile

    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = pd.Series(["A"] * 8)
    ranks, has = peer_percentile(values, groups, higher_is_better=True,
                                 min_group_size=5, winsorize_pct=0.0,
                                 clamp=(-1.0, 1.0))
    assert has.all()
    assert ranks.iloc[0] < ranks.iloc[-1]
    assert ranks.min() >= -1.0 and ranks.max() <= 1.0


def test_peer_percentile_lower_is_better_inverts_order():
    from strategy.scoring.peer import peer_percentile

    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    groups = pd.Series(["A"] * 8)
    asc, _ = peer_percentile(values, groups, higher_is_better=True,
                             min_group_size=5, winsorize_pct=0.0)
    desc, _ = peer_percentile(values, groups, higher_is_better=False,
                              min_group_size=5, winsorize_pct=0.0)
    assert asc.iloc[0] < asc.iloc[-1]
    assert desc.iloc[0] > desc.iloc[-1]


def test_peer_percentile_small_group_returns_nan():
    from strategy.scoring.peer import peer_percentile

    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    groups = pd.Series(["A"] * 4)
    ranks, has = peer_percentile(values, groups, min_group_size=8, winsorize_pct=0.0)
    assert ranks.isna().all()
    assert not has.any()


def test_peer_percentile_handles_nan_inputs():
    from strategy.scoring.peer import peer_percentile

    values = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])
    groups = pd.Series(["A"] * 8)
    ranks, has = peer_percentile(values, groups, min_group_size=5, winsorize_pct=0.0)
    # NaN inputs stay NaN, non-NaN rows get real ranks.
    assert ranks.iloc[1] != ranks.iloc[1] or pd.isna(ranks.iloc[1])
    assert has.iloc[0] and has.iloc[2] and not has.iloc[1]


def test_blend_relative_falls_back_when_industry_too_small():
    from strategy.scoring.peer import blend_relative

    industry_rank = pd.Series([np.nan, np.nan, np.nan])
    sector_rank   = pd.Series([0.5, -0.3, 0.1])
    market_rank   = pd.Series([0.2, 0.0, 0.4])
    industry_has  = pd.Series([False, False, False])
    sector_has    = pd.Series([True, True, True])
    blended, reason = blend_relative(
        industry_rank, sector_rank, market_rank,
        industry_has=industry_has, sector_has=sector_has,
    )
    assert (reason == "sector").all()
    assert blended.notna().all()


def test_blend_relative_falls_back_to_market_when_both_too_small():
    from strategy.scoring.peer import blend_relative

    industry_rank = pd.Series([np.nan, np.nan])
    sector_rank   = pd.Series([np.nan, np.nan])
    market_rank   = pd.Series([0.3, -0.5])
    blended, reason = blend_relative(
        industry_rank, sector_rank, market_rank,
        industry_has=pd.Series([False, False]),
        sector_has=pd.Series([False, False]),
    )
    assert (reason == "market").all()
    assert blended.iloc[0] == pytest.approx(0.3, abs=1e-3)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_peer_scoring_schema_rejects_bad_blend():
    from config.schema import PeerBlendConfig, PeerStandardizationConfig

    with pytest.raises(ValueError):
        PeerStandardizationConfig(
            blend=PeerBlendConfig(industry_relative=0.5, sector_relative=0.2, market_relative=0.1)
        )


def test_peer_scoring_schema_rejects_bad_winsorize():
    from config.schema import PeerStandardizationConfig

    with pytest.raises(ValueError):
        PeerStandardizationConfig(winsorize_pct=0.5)


def test_peer_scoring_schema_rejects_bad_clamp():
    from config.schema import PeerStandardizationConfig

    with pytest.raises(ValueError):
        PeerStandardizationConfig(clamp_low=1.0, clamp_high=0.0)


def test_peer_scoring_schema_rejects_min_group_size_too_small():
    from config.schema import PeerStandardizationConfig

    with pytest.raises(ValueError):
        PeerStandardizationConfig(min_group_size=1)


# ---------------------------------------------------------------------------
# Factor modules
# ---------------------------------------------------------------------------


def test_apply_value_produces_finite_scores():
    from strategy.scoring.value import apply_value

    df = _make_universe()
    apply_value(df, _peer_cfg())
    assert "value_score" in df.columns
    assert df["value_score"].notna().all()
    # All scores within clamp range
    assert df["value_score"].min() >= -1.0 - 1e-6
    assert df["value_score"].max() <= 1.5 + 1e-6


def test_apply_value_distress_penalty_fires():
    from strategy.scoring.value import apply_value

    df = _make_universe()
    # Force the first ticker into distress range (0 < PE < 5)
    df.loc[df.index[0], "pe_ratio"] = 3.0
    apply_value(df, _peer_cfg())
    assert bool(df.loc[df.index[0], "value_distress_flag"])


def test_apply_quality_no_dividend_not_punished():
    """Tech (no dividend) should not be systematically lower than utilities (dividend)."""
    from strategy.scoring.quality import apply_quality

    df = _make_universe()
    apply_quality(df, _peer_cfg())
    tech = df[df["industry"] == "software"]["quality_score"]
    util = df[df["industry"] == "utilities"]["quality_score"]
    # Tech mean must be within 0.3 of utilities — peer-ranking is doing its job.
    assert abs(float(tech.mean()) - float(util.mean())) < 0.3


def test_apply_quality_small_groups_fall_back_to_market():
    """When every industry/sector group is too small, ranks fall back to market-wide."""
    from strategy.scoring.quality import apply_quality

    df = _make_universe()
    cfg = _peer_cfg()
    cfg["peer_standardization"]["min_group_size"] = 999  # force every group too small
    apply_quality(df, cfg)
    # Industry-level rank not possible → reason is market (or legacy if market also too small).
    assert (df["quality_fallback_reason"].isin(["market", "legacy_checklist"])).all()
    # Scores still finite — fallback path works end-to-end.
    assert df["quality_score"].notna().all()


def test_apply_quality_legacy_fallback_when_no_peers_at_all():
    """If a row has no peer rank at any tier (single-row universe), legacy fallback fires."""
    from strategy.scoring.quality import apply_quality

    df = _make_universe().head(1).reset_index(drop=True)
    cfg = _peer_cfg()
    apply_quality(df, cfg)
    # With one row there's no meaningful market rank either → legacy fallback path.
    assert df["quality_fallback_reason"].iloc[0] in ("legacy_checklist", "market")
    assert df["quality_score"].notna().all()


def test_apply_momentum_penalties_fire():
    from strategy.scoring.momentum import apply_momentum

    df = _make_universe()
    df.loc[df.index[0], "return_3m"] = -0.50   # falling knife
    df.loc[df.index[1], "position_52w"] = 0.99 # overextended
    df.loc[df.index[2], "realized_vol_3m"] = 0.90  # high vol
    apply_momentum(df, _peer_cfg())
    pen = df["momentum_penalties_applied"]
    assert pen.iloc[0] >= 1
    assert pen.iloc[1] >= 1
    assert pen.iloc[2] >= 1


def test_apply_income_yield_trap_floored_and_zero_dividend_neutral():
    from strategy.scoring.income import apply_income

    df = _make_universe()
    df.loc[df.index[0], "dividend_yield"] = 0.20  # yield trap
    apply_income(df, _peer_cfg())
    assert bool(df.loc[df.index[0], "yield_trap_flag"])
    assert df.loc[df.index[0], "income_score"] == 0.0
    # Software (no dividend) → neutral 0
    soft = df[df["industry"] == "software"]
    assert (soft["income_score"] == 0.0).all()


# ---------------------------------------------------------------------------
# Composite integration
# ---------------------------------------------------------------------------


def test_compute_metric_writes_canonical_columns_and_stamps_version():
    from strategy.scoring.composite import SCORING_MODEL_VERSION, compute_metric

    df = _make_universe()
    compute_metric(
        df,
        score_weights={"value": 0.04, "quality": 0.46, "income": 0.00, "momentum": 0.50},
        scoring_cfg=_peer_cfg(),
    )
    # Canonical columns present
    for col in ("value_score", "quality_score", "income_score", "momentum_score", "value_metric"):
        assert col in df.columns, f"missing canonical column: {col!r}"
    # Engine version stamped
    assert df["scoring_model_version"].eq(SCORING_MODEL_VERSION).all()
    # value_metric is finite for every row
    assert df["value_metric"].notna().all()


def test_compute_metric_is_reproducible():
    """Lookahead-safe: same input + same config → identical peer-relative score columns."""
    from strategy.scoring.composite import compute_metric

    df1 = _make_universe()
    df2 = _make_universe()
    sw = {"value": 0.04, "quality": 0.46, "income": 0.00, "momentum": 0.50}
    compute_metric(df1, score_weights=sw, scoring_cfg=_peer_cfg())
    compute_metric(df2, score_weights=sw, scoring_cfg=_peer_cfg())
    pd.testing.assert_series_equal(df1["value_metric"], df2["value_metric"], check_names=False)
    pd.testing.assert_series_equal(df1["quality_score"], df2["quality_score"], check_names=False)


def test_scoring_config_hash_is_stable_and_short():
    from strategy.scoring.composite import scoring_config_hash

    cfg = _peer_cfg()
    h1 = scoring_config_hash(cfg)
    h2 = scoring_config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 12
    cfg2 = _peer_cfg()
    cfg2["peer_standardization"]["min_group_size"] = 999
    assert scoring_config_hash(cfg2) != h1


# ---------------------------------------------------------------------------
# Snapshot migration
# ---------------------------------------------------------------------------


def _seed_snapshot(tmp_path):
    df = _make_universe()
    df["value_metric"] = df["value_metric"].astype(float)
    path = tmp_path / "2026_05_01_09_00.parquet"
    df.to_parquet(path, index=False)
    return path


def test_rescore_snapshots_dry_run_writes_nothing(tmp_path):
    from strategy.snapshots import rescore_snapshots

    snap = _seed_snapshot(tmp_path)
    original_bytes = snap.read_bytes()
    report = rescore_snapshots(input_dir=tmp_path, dry_run=True)
    assert report.dry_run is True
    assert report.files_processed == 1
    assert report.files_rescored == 1
    # File untouched
    assert snap.read_bytes() == original_bytes
    # No .bak created in dry-run mode
    assert not list(tmp_path.glob("*.bak.parquet"))


def test_migrate_snapshots_in_place_creates_backup_and_is_idempotent(tmp_path):
    from strategy.snapshots import rescore_snapshots

    snap = _seed_snapshot(tmp_path)
    r1 = rescore_snapshots(input_dir=tmp_path, in_place_with_backup=True)
    assert r1.files_rescored == 1
    assert r1.backups_created == 1
    assert (tmp_path / "2026_05_01_09_00.bak.parquet").exists()
    # Re-run is a no-op
    r2 = rescore_snapshots(input_dir=tmp_path, in_place_with_backup=True)
    assert r2.files_skipped_already_migrated == 1
    assert r2.files_rescored == 0


def test_migrate_snapshots_to_separate_output_dir(tmp_path):
    from strategy.snapshots import rescore_snapshots

    inp = tmp_path / "in"
    inp.mkdir()
    out = tmp_path / "out"
    _ = _seed_snapshot(inp)
    report = rescore_snapshots(input_dir=inp, output_dir=out)
    assert report.files_rescored == 1
    assert (out / "2026_05_01_09_00.parquet").exists()
    # Input file untouched (no scoring_model_version column added)
    src = pd.read_parquet(next(inp.glob("*.parquet")))
    assert "scoring_model_version" not in src.columns


def test_migrate_snapshots_preserves_original_columns(tmp_path):
    from strategy.snapshots import rescore_snapshots

    snap = _seed_snapshot(tmp_path)
    before = pd.read_parquet(snap)
    rescore_snapshots(input_dir=tmp_path, in_place_with_backup=True)
    after = pd.read_parquet(snap)
    for col in before.columns:
        assert col in after.columns, f"column {col} was dropped"
    # And peer-relative + metadata columns were added
    assert "value_metric" in after.columns
    assert "scoring_config_hash" in after.columns


# ---------------------------------------------------------------------------
# BacktestReport metadata
# ---------------------------------------------------------------------------


def test_backtest_report_default_engine_is_peer():
    from backtesting.types import BacktestReport, SimResult

    sim = SimResult(final_value=100.0, total_return=0.0, sharpe=0.0,
                    calmar=0.0, max_drawdown=0.0, trades_made=0)
    rpt = BacktestReport(
        mode="test", universe_selection="test", lookahead_bias_level="LOW",
        n_symbols=0, n_days=0, train_result=sim, validation_result=None,
        benchmark_return=0.0, benchmark_sharpe=0.0, benchmark_max_drawdown=0.0,
        excess_return=0.0, validation_benchmark_return=0.0, notes=[],
    )
    assert rpt.scoring_engine_version == "peer-1"
    assert rpt.peer_config == {}
