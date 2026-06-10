"""Tests for the regime sizing random-window research harness."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_build_sizing_params_updates_index_pct_and_max_buys_without_mutating_base():
    from research.regime_sizing import SizingVariant, build_sizing_params

    base = np.arange(60, dtype=float)
    variant = SizingVariant(name="neutral_60pct_6buys", index_pct=0.60, max_buys=6)

    out = build_sizing_params(base, variant)

    assert out is not base
    assert base[4] == 4.0
    assert base[44] == 44.0
    assert out[4] == 0.60
    assert out[44] == 6.0


def test_default_neutral_sizing_grid_includes_current_config_first():
    from research.regime_sizing import default_neutral_sizing_grid

    variants = default_neutral_sizing_grid(current_index_pct=0.77, current_max_buys=4)

    assert variants[0].name == "current"
    assert variants[0].index_pct == 0.77
    assert variants[0].max_buys == 4
    assert any(v.index_pct < 0.77 for v in variants[1:])
    assert any(v.max_buys > 4 for v in variants[1:])


def _fake_window_result(params, start, window_days):
    """WindowResult whose benchmark return is a deterministic function of the start."""
    from backtesting.random_walk import WindowResult

    return WindowResult(
        window_id=0,
        start_day=int(start),
        end_day=int(start) + int(window_days),
        strategy_return=float(params[4]),
        benchmark_return=float(start) / 100.0,
        excess_return=float(params[4]) - float(start) / 100.0,
        sharpe=float(params[4]),
        max_drawdown=-0.05,
        calmar=1.0,
        turnover=0.1,
        trades=1,
        avg_positions=1,
        wins_benchmark=True,
    )


def test_run_regime_sizing_grid_is_paired_across_variants(monkeypatch):
    """run_regime_sizing_grid samples ONE shared start set from the seed and evaluates
    every variant on it — deltas are paired, not confounded with window luck."""
    from research import regime_sizing
    from research.regime_sizing import SizingVariant, run_regime_sizing_grid

    seen_scopes = []

    def fake_eligible(precomp, window_days, regime_scope):
        seen_scopes.append(regime_scope)
        return np.arange(0, 120, 10), {}

    monkeypatch.setattr(regime_sizing, "eligible_window_starts", fake_eligible)

    calls = []

    def fake_run_window(precomp, params, start, window_days, **kwargs):
        calls.append((float(params[4]), int(params[44]), int(start), kwargs.get("scope")))
        return _fake_window_result(params, start, window_days)

    monkeypatch.setattr(regime_sizing, "run_single_window", fake_run_window)

    base = np.zeros(60, dtype=float)
    variants = [SizingVariant("a", 0.77, 4), SizingVariant("b", 0.60, 8)]
    rows = run_regime_sizing_grid(
        precomp=object(),
        base_params=base,
        variants=variants,
        regime_scope="neutral",
        n_windows=3,
        window_days=45,
        seed=11,
    )

    assert [r.variant.name for r in rows] == ["a", "b"]
    assert seen_scopes == ["neutral"]
    assert all(c[3] == "overall_strategy" for c in calls)
    # Variant params reach the simulation (index_pct slot 4, max_buys slot 44).
    starts_a = [c[2] for c in calls if (c[0], c[1]) == (0.77, 4)]
    starts_b = [c[2] for c in calls if (c[0], c[1]) == (0.60, 8)]
    assert len(starts_a) == 3 and len(starts_b) == 3
    # PAIRED: both variants evaluated on the identical window starts …
    assert starts_a == starts_b
    # … hence identical benchmark medians (the natural pairing invariant).
    assert rows[0].summary.median_benchmark_return == rows[1].summary.median_benchmark_return


def test_sample_regime_window_starts_splits_temporal_segments(monkeypatch):
    from research import regime_sizing

    monkeypatch.setattr(
        regime_sizing,
        "eligible_window_starts",
        lambda precomp, window_days, regime_scope: (np.array([0, 10, 20, 30, 40, 50]), {}),
    )

    assert regime_sizing.sample_regime_window_starts(
        object(), window_days=10, regime_scope="neutral", n_windows=10, seed=1,
        segment="train", split_day=35,
    ).tolist() == [0, 10, 20]
    assert regime_sizing.sample_regime_window_starts(
        object(), window_days=10, regime_scope="neutral", n_windows=10, seed=1,
        segment="holdout", split_day=35,
    ).tolist() == [40, 50]


def test_run_regime_sizing_grid_on_starts_uses_identical_starts_for_variants(monkeypatch):
    from backtesting.random_walk import WindowResult
    from research import regime_sizing
    from research.regime_sizing import SizingVariant, run_regime_sizing_grid_on_starts

    calls = []

    def fake_run_window(precomp, params, start, window_days, **kwargs):
        calls.append((float(params[4]), int(params[44]), start, window_days, kwargs["scope"]))
        return WindowResult(
            window_id=0,
            start_day=start,
            end_day=start + window_days,
            strategy_return=float(params[4]) + start / 1000,
            benchmark_return=0.1,
            excess_return=float(params[4]) - 0.1,
            sharpe=float(params[4]),
            max_drawdown=-0.05,
            calmar=1.0,
            turnover=0.1,
            trades=1,
            avg_positions=1,
            wins_benchmark=True,
        )

    monkeypatch.setattr(regime_sizing, "run_single_window", fake_run_window)

    base = np.zeros(60, dtype=float)
    variants = [SizingVariant("current", 0.77, 4), SizingVariant("candidate", 0.60, 6)]
    results = run_regime_sizing_grid_on_starts(
        precomp=object(),
        base_params=base,
        variants=variants,
        starts=np.array([5, 15]),
        window_days=45,
    )

    assert [r.variant.name for r in results] == ["current", "candidate"]
    assert [(c[0], c[1], c[2]) for c in calls] == [
        (0.77, 4, 5), (0.77, 4, 15), (0.60, 6, 5), (0.60, 6, 15),
    ]
    assert results[0].summary.n_windows == 2
    assert results[1].summary.n_windows == 2
