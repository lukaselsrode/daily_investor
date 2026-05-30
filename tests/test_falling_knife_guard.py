"""
tests/test_falling_knife_guard.py — below-200DMA value-trap guard in live composite.

The guard (regime.defensive.falling_knife_guard) multiplicatively penalizes the
composite of below-200DMA names whose value_metric is in the top fraction of their
below-200DMA peers. OFF by default (0.0) => behavior-preserving.

Evidence basis: 4h research session found high-composite below-200DMA names
systematically underperform (monotonic pooled quintiles, t up to -7). This guard is
downside protection, validated as OFF-by-default so existing behavior is unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest


def _df():
    """5 below-200DMA names with varied composites + 2 above-200DMA names."""
    return pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E", "U1", "U2"],
        "value_metric": [1.0, 0.8, 0.6, 0.4, 0.2, 1.2, 0.9],
        "above_200dma": [False, False, False, False, False, True, True],
    })


def test_guard_off_is_behavior_preserving(monkeypatch):
    import strategy.scoring.composite as comp
    monkeypatch.setattr(comp, "REGIME_PARAMS", None, raising=False)
    # Even with the import inside the function, REGIME_PARAMS comes from util; patch there.
    import util
    monkeypatch.setattr(util, "REGIME_PARAMS",
                        {"defensive": {"falling_knife_guard": 0.0, "falling_knife_top_frac": 0.5}},
                        raising=False)
    df = _df()
    before = df["value_metric"].copy()
    comp._apply_falling_knife_guard(df)
    pd.testing.assert_series_equal(df["value_metric"], before)


def test_guard_penalizes_top_below200_composites(monkeypatch):
    import util
    monkeypatch.setattr(util, "REGIME_PARAMS",
                        {"defensive": {"falling_knife_guard": 0.5, "falling_knife_top_frac": 0.5}},
                        raising=False)
    import strategy.scoring.composite as comp
    df = _df()
    comp._apply_falling_knife_guard(df)
    vm = df.set_index("symbol")["value_metric"]
    # Below-200DMA composites: [1.0,0.8,0.6,0.4,0.2]; top 50% threshold = quantile(0.5)=0.6
    # so A(1.0), B(0.8), C(0.6) are penalized x0.5; D, E untouched.
    assert vm["A"] == pytest.approx(0.5)
    assert vm["B"] == pytest.approx(0.4)
    assert vm["C"] == pytest.approx(0.3)
    assert vm["D"] == pytest.approx(0.4)   # below threshold, untouched
    assert vm["E"] == pytest.approx(0.2)   # untouched
    # Above-200DMA names NEVER touched, even though U1 has the highest composite.
    assert vm["U1"] == pytest.approx(1.2)
    assert vm["U2"] == pytest.approx(0.9)


def test_guard_noop_when_column_missing(monkeypatch):
    import util
    monkeypatch.setattr(util, "REGIME_PARAMS",
                        {"defensive": {"falling_knife_guard": 0.5}}, raising=False)
    import strategy.scoring.composite as comp
    df = pd.DataFrame({"symbol": ["A"], "value_metric": [1.0]})  # no above_200dma
    before = df["value_metric"].copy()
    comp._apply_falling_knife_guard(df)
    pd.testing.assert_series_equal(df["value_metric"], before)


def test_guard_noop_too_few_below200(monkeypatch):
    import util
    monkeypatch.setattr(util, "REGIME_PARAMS",
                        {"defensive": {"falling_knife_guard": 0.5, "falling_knife_top_frac": 0.5}},
                        raising=False)
    import strategy.scoring.composite as comp
    # only 2 below-200DMA names (< 5 minimum) -> no-op
    df = pd.DataFrame({
        "symbol": ["A", "B", "U1"],
        "value_metric": [1.0, 0.5, 0.9],
        "above_200dma": [False, False, True],
    })
    before = df["value_metric"].copy()
    comp._apply_falling_knife_guard(df)
    pd.testing.assert_series_equal(df["value_metric"], before)
