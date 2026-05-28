"""
tests/test_backtesting.py — BacktestResult, ValidationResult, WalkForwardValidator tests (Phase 6).

All tests that touch backtest.py simulation internals are already in src/tests.py.
This file tests the new layer: results containers, validator gate logic,
and BacktestEngine orchestration — without loading real market data.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backtesting.results import BacktestResult, ValidationResult
from backtesting.types import BacktestReport, SimResult
from backtesting.validator import WalkForwardValidator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sim(
    total_return: float = 0.12,
    sharpe: float = 0.80,
    calmar: float = 1.2,
    max_drawdown: float = -0.08,
    trades_made: int = 40,
) -> SimResult:
    return SimResult(
        final_value=11_200.0,
        total_return=total_return,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=max_drawdown,
        trades_made=trades_made,
    )


def _report(
    train: SimResult = None,
    val: SimResult = None,
    val_bench_return: float = 0.05,
    benchmark_return: float = 0.08,
    excess: float = 0.04,
) -> BacktestReport:
    return BacktestReport(
        mode="liquid_universe_sanity_test",
        universe_selection="liquid_all",
        lookahead_bias_level="LOW",
        n_symbols=50,
        n_days=63,
        train_result=train or _sim(),
        validation_result=val,
        benchmark_return=benchmark_return,
        benchmark_sharpe=0.60,
        benchmark_max_drawdown=-0.06,
        excess_return=excess,
        validation_benchmark_return=val_bench_return,
        notes=[],
    )


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def test_total_return_delegates_to_report(self):
        r = BacktestResult(report=_report(train=_sim(total_return=0.15)), n_days=63, mode="test")
        assert r.total_return == pytest.approx(0.15)

    def test_excess_return_from_report(self):
        r = BacktestResult(report=_report(excess=0.07), n_days=63, mode="test")
        assert r.excess_return == pytest.approx(0.07)

    def test_sharpe_delegates(self):
        r = BacktestResult(report=_report(train=_sim(sharpe=1.2)), n_days=63, mode="test")
        assert r.sharpe == pytest.approx(1.2)

    def test_max_drawdown_delegates(self):
        r = BacktestResult(report=_report(train=_sim(max_drawdown=-0.12)), n_days=63, mode="test")
        assert r.max_drawdown == pytest.approx(-0.12)

    def test_passes_basic_validation_true(self):
        r = BacktestResult(report=_report(train=_sim(sharpe=0.5, max_drawdown=-0.10)), n_days=63, mode="t")
        assert r.passes_basic_validation

    def test_passes_basic_validation_false_on_low_sharpe(self):
        r = BacktestResult(report=_report(train=_sim(sharpe=0.10, max_drawdown=-0.10)), n_days=63, mode="t")
        assert not r.passes_basic_validation

    def test_passes_basic_validation_false_on_deep_drawdown(self):
        r = BacktestResult(report=_report(train=_sim(sharpe=1.0, max_drawdown=-0.25)), n_days=63, mode="t")
        assert not r.passes_basic_validation

    def test_summary_dict_keys(self):
        r = BacktestResult(report=_report(), n_days=63, mode="test")
        d = r.summary_dict()
        for key in ("n_days", "mode", "total_return", "benchmark_return", "excess_return",
                    "sharpe", "calmar", "max_drawdown", "trades", "universe_size"):
            assert key in d

    def test_str_contains_mode(self):
        r = BacktestResult(report=_report(), n_days=63, mode="walk_forward")
        assert "walk_forward" in str(r)

    def test_universe_size(self):
        r = BacktestResult(report=_report(), n_days=63, mode="t")
        assert r.universe_size == 50

    def test_benchmark_return(self):
        r = BacktestResult(report=_report(benchmark_return=0.09), n_days=63, mode="t")
        assert r.benchmark_return == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:

    def _vr(self, passed=True, reasons=None, val=None, val_bench=0.04) -> ValidationResult:
        return ValidationResult(
            passed=passed,
            reasons=reasons or [],
            report=_report(val=val or _sim(total_return=0.10), val_bench_return=val_bench),
            train_slice=slice(0, 44),
            val_slice=slice(44, 63),
        )

    def test_val_result_property(self):
        val = _sim(total_return=0.08)
        vr = self._vr(val=val)
        assert vr.val_result is val

    def test_val_excess_return(self):
        val = _sim(total_return=0.10)
        vr = self._vr(val=val, val_bench=0.04)
        assert vr.val_excess_return == pytest.approx(0.06)

    def test_val_excess_return_no_val_result(self):
        vr = ValidationResult(
            passed=False,
            reasons=["No validation window"],
            report=_report(val=None),
            train_slice=slice(0, 63),
            val_slice=None,
        )
        assert vr.val_excess_return == 0.0

    def test_summary_passed(self):
        vr = self._vr(passed=True)
        assert "PASSED" in vr.summary()

    def test_summary_failed_includes_reasons(self):
        vr = self._vr(passed=False, reasons=["Sharpe too low", "Drawdown exceeded"])
        s = vr.summary()
        assert "FAILED" in s
        assert "Sharpe too low" in s


# ---------------------------------------------------------------------------
# WalkForwardValidator — gate logic (no simulation I/O)
# ---------------------------------------------------------------------------

class TestWalkForwardValidatorSplit:

    def _v(self) -> WalkForwardValidator:
        return WalkForwardValidator()

    def test_split_returns_non_overlapping_slices(self):
        train_sl, val_sl = self._v().split(100)
        assert train_sl.stop == val_sl.start

    def test_split_train_pct_respected(self):
        train_sl, val_sl = self._v().split(100, train_pct=0.70)
        assert train_sl.stop == 70
        assert val_sl.start == 70
        assert val_sl.stop == 100

    def test_split_covers_all_days(self):
        train_sl, val_sl = self._v().split(90, train_pct=0.70)
        assert val_sl.stop == 90

    def test_custom_train_pct(self):
        train_sl, _ = self._v().split(100, train_pct=0.80)
        assert train_sl.stop == 80


class TestWalkForwardValidatorGates:

    _GATES = {
        "min_validation_excess_return": 0.0,
        "max_validation_drawdown": -0.20,
        "min_validation_sharpe": 0.25,
    }

    def _v(self) -> WalkForwardValidator:
        return WalkForwardValidator()

    def _rep_with_val(self, total_return=0.10, sharpe=0.80, max_drawdown=-0.08, val_bench=0.05):
        val = _sim(total_return=total_return, sharpe=sharpe, max_drawdown=max_drawdown)
        return _report(val=val, val_bench_return=val_bench)

    def test_all_gates_pass(self):
        rep = self._rep_with_val(total_return=0.10, sharpe=0.80, max_drawdown=-0.08, val_bench=0.05)
        passed, reasons = self._v().validate_report(rep, self._GATES)
        assert passed
        assert reasons == []

    def test_no_val_result_fails(self):
        rep = _report(val=None)
        passed, reasons = self._v().validate_report(rep, self._GATES)
        assert not passed
        assert any("No validation" in r for r in reasons)

    def test_excess_return_gate_fails(self):
        rep = self._rep_with_val(total_return=0.02, val_bench=0.05)
        passed, reasons = self._v().validate_report(rep, self._GATES)
        assert not passed
        assert any("excess return" in r.lower() for r in reasons)

    def test_drawdown_gate_fails(self):
        rep = self._rep_with_val(max_drawdown=-0.25)
        passed, reasons = self._v().validate_report(rep, {**self._GATES, "max_validation_drawdown": -0.20})
        assert not passed
        assert any("drawdown" in r.lower() for r in reasons)

    def test_sharpe_gate_fails(self):
        rep = self._rep_with_val(sharpe=0.10)
        passed, reasons = self._v().validate_report(rep, self._GATES)
        assert not passed
        assert any("sharpe" in r.lower() for r in reasons)

    def test_multiple_gate_failures_all_reported(self):
        rep = self._rep_with_val(total_return=0.01, sharpe=0.10, val_bench=0.05)
        passed, reasons = self._v().validate_report(rep, self._GATES)
        assert not passed
        assert len(reasons) >= 2

    def test_should_apply_requires_validation_pass(self):
        vr = ValidationResult(passed=False, reasons=["x"], report=_report(), train_slice=slice(0, 63), val_slice=None)
        assert not self._v().should_apply(vr, apply_flag=True, backtest_cfg=self._GATES)

    def test_should_apply_passes_when_valid_and_flag_set(self):
        vr = ValidationResult(passed=True, reasons=[], report=_report(), train_slice=slice(0, 63), val_slice=None)
        assert self._v().should_apply(vr, apply_flag=True, backtest_cfg=self._GATES)

    def test_should_apply_force_bypasses_gates(self):
        vr = ValidationResult(passed=False, reasons=["x"], report=_report(), train_slice=slice(0, 63), val_slice=None)
        assert self._v().should_apply(vr, apply_flag=False, force_apply=True)

    def test_should_apply_auto_apply_from_cfg(self):
        vr = ValidationResult(passed=True, reasons=[], report=_report(), train_slice=slice(0, 63), val_slice=None)
        cfg = {**self._GATES, "auto_apply_if_valid": True}
        assert self._v().should_apply(vr, apply_flag=False, backtest_cfg=cfg)

    def test_should_apply_auto_false_and_no_flag(self):
        vr = ValidationResult(passed=True, reasons=[], report=_report(), train_slice=slice(0, 63), val_slice=None)
        cfg = {**self._GATES, "auto_apply_if_valid": False}
        assert not self._v().should_apply(vr, apply_flag=False, backtest_cfg=cfg)


# ---------------------------------------------------------------------------
# WalkForwardValidator.split_and_validate — mocked I/O
# Patches target backtesting.validator.run_backtest_report directly.
# ---------------------------------------------------------------------------

class TestWalkForwardValidatorSplitAndValidate:

    _GATES = {
        "min_validation_excess_return": 0.0,
        "max_validation_drawdown": -0.20,
        "min_validation_sharpe": 0.25,
        "train_pct": 0.70,
    }

    def _precomp_mock(self, n_days: int = 90):
        pc = MagicMock()
        pc.prices = np.zeros((n_days, 5))
        return pc

    def _good_report(self) -> BacktestReport:
        val = _sim(total_return=0.10, sharpe=0.80, max_drawdown=-0.08)
        return _report(val=val, val_bench_return=0.05)

    def test_returns_validation_result(self):
        precomp = self._precomp_mock()
        with patch("backtesting.validator.run_backtest_report", return_value=self._good_report()):
            result = WalkForwardValidator().split_and_validate(
                precomp, None, n_days=90, backtest_cfg=self._GATES
            )
        assert isinstance(result, ValidationResult)

    def test_passed_when_all_gates_pass(self):
        precomp = self._precomp_mock()
        with patch("backtesting.validator.run_backtest_report", return_value=self._good_report()):
            result = WalkForwardValidator().split_and_validate(
                precomp, None, n_days=90, backtest_cfg=self._GATES
            )
        assert result.passed

    def test_failed_when_sharpe_low(self):
        precomp = self._precomp_mock()
        val = _sim(total_return=0.10, sharpe=0.05, max_drawdown=-0.08)
        bad_report = _report(val=val, val_bench_return=0.05)
        with patch("backtesting.validator.run_backtest_report", return_value=bad_report):
            result = WalkForwardValidator().split_and_validate(
                precomp, None, n_days=90, backtest_cfg=self._GATES
            )
        assert not result.passed
        assert any("Sharpe" in r for r in result.reasons)

    def test_slices_set_on_result(self):
        precomp = self._precomp_mock(100)
        with patch("backtesting.validator.run_backtest_report", return_value=self._good_report()):
            result = WalkForwardValidator().split_and_validate(
                precomp, None, n_days=100, train_pct=0.70, backtest_cfg=self._GATES
            )
        assert result.train_slice == slice(0, 70)
        assert result.val_slice is not None

    def test_no_val_slice_when_use_validation_false(self):
        precomp = self._precomp_mock()
        with patch("backtesting.validator.run_backtest_report", return_value=_report(val=None)) as mock_rbr:
            WalkForwardValidator().split_and_validate(
                precomp, None, n_days=90, backtest_cfg=self._GATES, use_validation=False
            )
        _, _, _, val_arg = mock_rbr.call_args[0]
        assert val_arg is None


# ---------------------------------------------------------------------------
# BacktestEngine orchestration — mocked I/O
# ---------------------------------------------------------------------------

class TestBacktestEngine:

    _PRECOMP_DAYS = 90

    def _precomp(self):
        pc = MagicMock()
        pc.prices = np.zeros((self._PRECOMP_DAYS, 5))
        pc.mode = "liquid_universe_sanity_test"
        return pc

    def _good_report(self):
        val = _sim(total_return=0.10, sharpe=0.80, max_drawdown=-0.08)
        return _report(val=val)

    def test_run_returns_backtest_result(self):
        from backtesting.engine import BacktestEngine
        with patch("backtesting.engine.load_and_precompute", return_value=self._precomp()), \
             patch("backtesting.engine.run_backtest_report", return_value=self._good_report()):
            result = BacktestEngine().run(n_days=90)
        assert isinstance(result, BacktestResult)

    def test_run_passes_mode(self):
        from backtesting.engine import BacktestEngine
        with patch("backtesting.engine.load_and_precompute", return_value=self._precomp()) as mock_lp, \
             patch("backtesting.engine.run_backtest_report", return_value=self._good_report()):
            BacktestEngine().run(n_days=90, mode="walk_forward_price_only_test")
        mock_lp.assert_called_once_with(90, mode="walk_forward_price_only_test")

    def test_run_no_validation_passes_none_val_slice(self):
        from backtesting.engine import BacktestEngine
        with patch("backtesting.engine.load_and_precompute", return_value=self._precomp()), \
             patch("backtesting.engine.run_backtest_report", return_value=_report(val=None)) as mock_rbr:
            BacktestEngine().run(n_days=90, with_validation=False)
        _, _, _, val_arg = mock_rbr.call_args[0]
        assert val_arg is None

    def test_run_walk_forward_returns_validation_result(self):
        from backtesting.engine import BacktestEngine
        with patch("backtesting.engine.load_and_precompute", return_value=self._precomp()), \
             patch("backtesting.validator.run_backtest_report", return_value=self._good_report()):
            result = BacktestEngine().run_walk_forward(n_days=90, params=None)
        assert isinstance(result, ValidationResult)

    def test_run_report_delegates_correctly(self):
        from backtesting.engine import BacktestEngine
        precomp = self._precomp()
        with patch("backtesting.engine.run_backtest_report", return_value=self._good_report()) as mock_rbr:
            result = BacktestEngine().run_report(precomp, None, slice(0, 63), slice(63, 90))
        mock_rbr.assert_called_once_with(precomp, None, slice(0, 63), slice(63, 90))
        assert isinstance(result, BacktestReport)
