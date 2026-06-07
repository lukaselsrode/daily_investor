"""Regime-scoped backtest/tuning data selection tests."""
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _precomp_with_benchmark(bench: np.ndarray, n_stocks: int = 3):
    from backtesting.types import PrecomputedData

    n_days = len(bench)
    rng = np.random.default_rng(0)
    prices = np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, n_stocks)), axis=0) * 100
    zeros_s = np.zeros(n_stocks)
    return PrecomputedData(
        symbols=[f"S{i}" for i in range(n_stocks)],
        prices=prices,
        pe_comp=zeros_s,
        pb_comp=zeros_s,
        quality_scores=np.ones(n_stocks),
        income_scores=zeros_s,
        yield_trap_mask=np.zeros(n_stocks, bool),
        bin_indices=np.zeros(n_stocks, np.int32),
        has_position_52w=np.ones(n_stocks, bool),
        position_52w_arr=zeros_s,
        return_1m_arr=zeros_s,
        etf_symbols=[],
        etf_prices=np.zeros((n_days, 0)),
        baseline_scores=zeros_s,
        sector_labels=["X"] * n_stocks,
        volume_arr=np.ones(n_stocks) * 1e6,
        mode="liquid_universe_full",
        universe_selection="liquid_all",
        lookahead_bias_level="MEDIUM",
        benchmark_prices=bench.astype(float),
        benchmark_symbol="SPY",
        position_52w_daily=np.zeros((n_days, n_stocks)),
        return_1m_daily=np.zeros((n_days, n_stocks)),
        bin_indices_daily=np.zeros((n_days, n_stocks), np.int32),
        has_position_52w_daily=np.ones((n_days, n_stocks), bool),
    )


def test_bearish_alias_maps_to_defensive_and_slices_longest_contiguous_block():
    from backtesting.regime_scope import apply_regime_scope, regime_labels

    bench = np.concatenate([
        np.linspace(100, 150, 220),
        np.linspace(150, 130, 12),   # neutral-ish warm transition
        np.linspace(130, 100, 40),   # defensive below 95% of MA200
        np.linspace(100, 120, 20),
    ])
    pc = _precomp_with_benchmark(bench)

    scoped, meta = apply_regime_scope(pc, "bearish")
    labels = regime_labels(scoped)

    assert meta["requested"] == "bearish"
    assert meta["effective"] == "defensive"
    assert scoped.prices.shape[0] > 0
    assert set(labels) == {"defensive"}


def test_all_regime_scope_is_noop_identity():
    from backtesting.regime_scope import apply_regime_scope

    pc = _precomp_with_benchmark(np.linspace(100, 160, 230))
    scoped, meta = apply_regime_scope(pc, "all")

    assert scoped is pc
    assert meta["effective"] == "all"


def test_random_window_backtest_samples_only_selected_regime_starts():
    from backtesting.random_walk import random_window_backtest

    bench = np.concatenate([
        np.linspace(100, 160, 230),
        np.linspace(160, 110, 35),
    ])
    pc = _precomp_with_benchmark(bench)

    with patch("backtesting.random_walk.run_simulation") as mock_sim:
        from backtesting.types import SimResult
        mock_sim.return_value = SimResult(
            final_value=10_500,
            total_return=0.05,
            sharpe=1.0,
            calmar=1.0,
            max_drawdown=-0.05,
            trades_made=20,
            average_positions=5.0,
            equity_curve=np.array([10_000, 10_500]),
        )
        summary = random_window_backtest(
            pc,
            params=None,
            n_windows=4,
            window_days=5,
            regime_scope="bearish",
        )

    assert summary.n_windows > 0
    assert all(w.start_day >= 230 for w in summary.window_results)


def test_parameter_tuner_threads_regime_scope_to_run_tuner():
    from backtesting.types import SimResult
    from tuning.constants import PARAM_NAMES
    from tuning.tuner import ParameterTuner

    p = np.zeros(len(PARAM_NAMES))
    s = SimResult(final_value=10_000, total_return=0.0, sharpe=0.0, calmar=0.0, max_drawdown=0.0, trades_made=0)
    with patch("tuning.tuner.run_tuner", return_value=(p, s)) as mock_rt:
        ParameterTuner().tune(n_days=90, regime_scope="neutral")

    assert mock_rt.call_args.kwargs["regime_scope"] == "neutral"


def test_small_regime_block_warns_overfit_risk(caplog):
    """A regime block below MIN_REGIME_DAYS_FOR_TUNING must loudly warn — rare regimes
    (esp. defensive) span only a few dozen days in a multi-year window, so tuning on them
    is severe overfitting. This is the guardrail against fake regime 'alpha'."""
    import logging

    from backtesting.regime_scope import (
        MIN_REGIME_DAYS_FOR_TUNING,
        apply_regime_scope,
    )

    # ~40-day defensive segment — far below the tuning threshold.
    bench = np.concatenate([
        np.linspace(100, 150, 220),
        np.linspace(150, 100, 40),
    ])
    pc = _precomp_with_benchmark(bench)
    with caplog.at_level(logging.WARNING, logger="backtesting.regime_scope"):
        _scoped, meta = apply_regime_scope(pc, "defensive")

    assert meta["selected_days"] < MIN_REGIME_DAYS_FOR_TUNING
    assert "too few to tune" in caplog.text


def test_large_regime_block_does_not_warn(caplog):
    """A regime block at/above the threshold must NOT emit the overfit warning."""
    import logging

    from backtesting.regime_scope import apply_regime_scope

    pc = _precomp_with_benchmark(np.linspace(100, 220, 600))  # long bullish run
    with caplog.at_level(logging.WARNING, logger="backtesting.regime_scope"):
        _scoped, meta = apply_regime_scope(pc, "bullish")

    assert meta["selected_days"] >= 90
    assert "too few to tune" not in caplog.text
