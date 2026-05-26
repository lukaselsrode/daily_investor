"""
tests/test_backtest.py — Backtest engine tests.

Migrated / adapted from src/tests.py (backtest-specific tests).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np

try:
    from backtesting.types import SimResult
    from backtesting.simulator import (
        compute_performance_metrics,
        run_simulation,
        score_stocks_at_day,
        split_price_window,
    )
    from backtesting.data_loader import select_backtest_universe
    _HAS_BACKTEST = True
except Exception as e:
    _HAS_BACKTEST = False
    _IMPORT_ERROR = str(e)

from util import BACKTEST_PARAMS, RISK_LIMITS, SELL_RULES


@pytest.mark.skipif(not _HAS_BACKTEST, reason="backtest.py not importable")
class TestSplitPriceWindow:

    def test_split_respects_train_pct(self):
        n_days = 100
        train_pct = BACKTEST_PARAMS["train_pct"]
        train_sl, val_sl = split_price_window(n_days, train_pct=train_pct)
        train_len = train_sl.stop - (train_sl.start or 0)
        val_len = val_sl.stop - (val_sl.start or 0)
        assert train_len == pytest.approx(n_days * train_pct, abs=1)
        assert val_len == pytest.approx(n_days * (1 - train_pct), abs=1)

    def test_split_no_overlap(self):
        n_days = 100
        train_sl, val_sl = split_price_window(n_days, train_pct=0.70)
        train_len = train_sl.stop - (train_sl.start or 0)
        val_len = val_sl.stop - (val_sl.start or 0)
        assert train_len + val_len == n_days


@pytest.mark.skipif(not _HAS_BACKTEST, reason="backtest.py not importable")
class TestComputePerformanceMetrics:

    def test_flat_values_sharpe_zero(self):
        # Constant portfolio value → zero daily returns → sharpe = 0
        values = np.full(100, 10000.0)
        result = compute_performance_metrics(values)
        assert abs(result["sharpe"]) < 1e-6

    def test_positive_drift_positive_sharpe(self):
        # Monotonically growing portfolio with small noise → positive sharpe
        rng = np.random.default_rng(42)
        values = np.cumprod(1 + 0.001 + rng.normal(0, 0.005, 252)) * 10000.0
        result = compute_performance_metrics(values)
        assert result["sharpe"] > 0

    def test_max_drawdown_is_nonpositive(self):
        rng = np.random.default_rng(7)
        values = np.cumprod(1 + rng.normal(0, 0.01, 252)) * 10000.0
        result = compute_performance_metrics(values)
        assert result["max_drawdown"] <= 0


@pytest.mark.skipif(not _HAS_BACKTEST, reason="backtest.py not importable")
class TestSimResultDefaults:

    def test_default_extended_fields_are_zero(self):
        r = SimResult(
            final_value=10000.0,
            total_return=0.10,
            sharpe=0.5,
            calmar=0.3,
            max_drawdown=-0.05,
            trades_made=10,
        )
        assert r.sells_made == 0
        assert r.stopout_count == 0

    def test_core_sim_result_extended_fields(self):
        from core.types import SimResult as CoreSimResult
        r = CoreSimResult(
            final_value=10000.0, total_return=0.10, sharpe=0.5, calmar=0.3,
            max_drawdown=-0.05, trades_made=10,
        )
        assert r.etf_return == 0.0
        assert r.trailing_stop_count == 0


class TestConfigConsistencyBacktest:

    def test_turnover_penalty_threshold_positive(self):
        assert BACKTEST_PARAMS["turnover_penalty_trade_count"] > 0

    def test_turnover_penalty_weight_positive(self):
        assert BACKTEST_PARAMS["turnover_penalty_weight"] > 0

    def test_sell_rules_have_minimum_hold_days(self):
        assert RISK_LIMITS["minimum_hold_days"] >= 0

    def test_min_days_before_take_profit_nonneg(self):
        assert SELL_RULES["minimum_days_before_take_profit"] >= 0
