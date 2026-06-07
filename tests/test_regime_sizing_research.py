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


def test_run_regime_sizing_grid_passes_regime_scope_and_variant_params(monkeypatch):
    from backtesting.random_walk import RandomWindowSummary
    from research import regime_sizing
    from research.regime_sizing import SizingVariant, run_regime_sizing_grid

    calls = []

    def fake_random_window_backtest(precomp, params, **kwargs):
        calls.append((params.copy(), kwargs))
        return RandomWindowSummary(
            n_windows=kwargs["n_windows"],
            window_days=kwargs["window_days"],
            params_used=params.copy(),
            median_excess_return=float(params[4]),
            pct_beating_benchmark=0.5,
            robust_score=float(params[4]),
        )

    monkeypatch.setattr(regime_sizing, "random_window_backtest", fake_random_window_backtest)

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
    assert [c[0][4] for c in calls] == [0.77, 0.60]
    assert [c[0][44] for c in calls] == [4.0, 8.0]
    assert all(c[1]["regime_scope"] == "neutral" for c in calls)
    assert all(c[1]["scope"] == "overall_strategy" for c in calls)
