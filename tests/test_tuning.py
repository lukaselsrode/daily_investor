"""
tests/test_tuning.py — ParameterTuner, StabilityAnalyzer, and result-type tests (Phase 7).

Heavy optimizer internals (run_tuner, run_auto_tune, run_stability_scan) are already
covered in src/tests.py. This file tests the new typed wrappers only, using
patch.object to avoid running scipy.optimize.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backtesting.types import SimResult
from tuning.results import AutoTuneResult, StabilityReport, TuneResult
from tuning.stability import StabilityAnalyzer
from tuning.tuner import ParameterTuner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_PARAMS = 15   # len(PARAM_NAMES)


def _params(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.5, _N_PARAMS)


def _sim(total_return=0.12, sharpe=0.80, calmar=1.2, max_drawdown=-0.08, trades=40) -> SimResult:
    return SimResult(
        final_value=11_200.0,
        total_return=total_return,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=max_drawdown,
        trades_made=trades,
    )


# ---------------------------------------------------------------------------
# TuneResult
# ---------------------------------------------------------------------------

class TestTuneResult:

    def _tr(self, objective="sharpe") -> TuneResult:
        return TuneResult(
            params=_params(),
            sim=_sim(sharpe=0.90, calmar=1.1),
            objective=objective,
            n_days=90,
            active_params=["sw_quality", "sw_momentum", "index_pct"],
        )

    def test_score_is_sharpe_for_sharpe_objective(self):
        tr = self._tr("sharpe")
        assert tr.score == pytest.approx(0.90)

    def test_score_is_calmar_for_calmar_objective(self):
        tr = self._tr("calmar")
        assert tr.score == pytest.approx(1.1)

    def test_summary_contains_objective(self):
        s = self._tr("sharpe").summary()
        assert "sharpe" in s

    def test_summary_contains_return(self):
        s = self._tr().summary()
        assert "%" in s

    def test_summary_contains_active_params(self):
        s = self._tr().summary()
        assert "sw_quality" in s

    def test_params_stored_as_array(self):
        tr = self._tr()
        assert isinstance(tr.params, np.ndarray)
        assert len(tr.params) == _N_PARAMS


# ---------------------------------------------------------------------------
# AutoTuneResult
# ---------------------------------------------------------------------------

class TestAutoTuneResult:

    def _atr(self, val_passed=True, config_written=False) -> AutoTuneResult:
        p = _params()
        return AutoTuneResult(
            avg_params=p,
            sharpe_params=p,
            calmar_params=p + 0.01,
            sharpe_result=_sim(sharpe=0.85),
            calmar_result=_sim(calmar=1.3),
            avg_result=_sim(total_return=0.10, sharpe=0.80),
            n_days=90,
            validation_passed=val_passed,
            validation_reasons=[] if val_passed else ["Sharpe too low"],
            config_written=config_written,
            active_params=["sw_quality", "index_pct"],
        )

    def test_summary_written_when_config_written(self):
        s = self._atr(val_passed=True, config_written=True).summary()
        assert "WRITTEN" in s

    def test_summary_passed_when_val_passed_not_written(self):
        s = self._atr(val_passed=True, config_written=False).summary()
        assert "PASSED" in s

    def test_summary_failed_when_val_failed(self):
        s = self._atr(val_passed=False).summary()
        assert "FAILED" in s

    def test_param_spread_is_dict(self):
        spread = self._atr().param_spread
        assert isinstance(spread, dict)
        assert len(spread) == _N_PARAMS

    def test_param_spread_non_negative(self):
        spread = self._atr().param_spread
        assert all(v >= 0 for v in spread.values())

    def test_validation_reasons_stored(self):
        atr = self._atr(val_passed=False)
        assert "Sharpe too low" in atr.validation_reasons

    def test_active_params_stored(self):
        atr = self._atr()
        assert "sw_quality" in atr.active_params


# ---------------------------------------------------------------------------
# StabilityReport
# ---------------------------------------------------------------------------

class TestStabilityReport:

    def _report(self, n_windows=3) -> StabilityReport:
        return StabilityReport(
            window_results=[{"n_days": d, "params": _params()} for d in [30, 60, 90][:n_windows]],
            output_paths={"csv": "/tmp/stability.csv"},
        )

    def test_n_windows(self):
        assert self._report(3).n_windows == 3

    def test_unstable_params_empty_when_no_df(self):
        assert self._report().unstable_params == []

    def test_summary_contains_window_count(self):
        s = self._report(2).summary()
        assert "2 windows" in s

    def test_output_paths_stored(self):
        r = self._report()
        assert r.output_paths["csv"] == "/tmp/stability.csv"

    def test_stability_df_none_by_default(self):
        r = self._report()
        assert r.stability_df is None


# ---------------------------------------------------------------------------
# ParameterTuner introspection (no optimizer call)
# ---------------------------------------------------------------------------

class TestParameterTunerIntrospection:

    def _tuner(self) -> ParameterTuner:
        return ParameterTuner()

    def test_param_names_is_list_of_strings(self):
        names = self._tuner().param_names
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) == _N_PARAMS

    def test_active_params_subset_of_param_names(self):
        t = self._tuner()
        active = set(t.active_params)
        all_names = set(t.param_names)
        assert active <= all_names

    def test_frozen_and_active_are_complementary(self):
        t = self._tuner()
        assert set(t.active_params) | set(t.frozen_params) == set(t.param_names)
        assert set(t.active_params) & set(t.frozen_params) == set()

    def test_effective_bounds_length_matches_params(self):
        bounds = self._tuner().effective_bounds
        assert len(bounds) == _N_PARAMS

    def test_effective_bounds_are_valid_intervals(self):
        for lo, hi in self._tuner().effective_bounds:
            assert lo <= hi

    def test_current_params_returns_array(self):
        p = self._tuner().current_params()
        assert isinstance(p, np.ndarray)
        assert len(p) == _N_PARAMS


# ---------------------------------------------------------------------------
# ParameterTuner.tune() — mocked _t.run_tuner
# ---------------------------------------------------------------------------

class TestParameterTunerTune:

    def test_returns_tune_result(self):
        p, s = _params(), _sim()
        with patch("tuning.tuner.run_tuner", return_value=(p, s)):
            result = ParameterTuner().tune(n_days=90)
        assert isinstance(result, TuneResult)

    def test_passes_n_days_and_objective(self):
        p, s = _params(), _sim()
        with patch("tuning.tuner.run_tuner", return_value=(p, s)) as mock_rt:
            ParameterTuner().tune(n_days=60, objective="calmar")
        mock_rt.assert_called_once_with(
            n_days=60, objective="calmar", starting_capital=10_000.0, mode=None
        )

    def test_result_params_matches_tuner_output(self):
        p, s = _params(seed=7), _sim()
        with patch("tuning.tuner.run_tuner", return_value=(p, s)):
            result = ParameterTuner().tune(n_days=90)
        assert np.array_equal(result.params, p)

    def test_result_sim_matches_tuner_output(self):
        p, s = _params(), _sim(sharpe=1.5)
        with patch("tuning.tuner.run_tuner", return_value=(p, s)):
            result = ParameterTuner().tune(n_days=90)
        assert result.sim.sharpe == pytest.approx(1.5)

    def test_result_n_days_stored(self):
        p, s = _params(), _sim()
        with patch("tuning.tuner.run_tuner", return_value=(p, s)):
            result = ParameterTuner().tune(n_days=120)
        assert result.n_days == 120

    def test_passes_mode(self):
        p, s = _params(), _sim()
        with patch("tuning.tuner.run_tuner", return_value=(p, s)) as mock_rt:
            ParameterTuner().tune(n_days=90, mode="liquid_universe_sanity_test")
        mock_rt.assert_called_once_with(
            n_days=90, objective="sharpe", starting_capital=10_000.0,
            mode="liquid_universe_sanity_test"
        )

    def test_calmar_objective_score(self):
        p, s = _params(), _sim(calmar=2.5)
        with patch("tuning.tuner.run_tuner", return_value=(p, s)):
            result = ParameterTuner().tune(n_days=90, objective="calmar")
        assert result.score == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# ParameterTuner.apply_params() — mocked _t.apply_config_params
# ---------------------------------------------------------------------------

class TestParameterTunerApplyParams:

    def test_delegates_to_apply_config_params(self):
        p = _params()
        with patch("tuning.tuner.apply_config_params") as mock_acp:
            ParameterTuner().apply_params(p)
        mock_acp.assert_called_once_with(p)


# ---------------------------------------------------------------------------
# StabilityAnalyzer.scan() — mocked _t.run_stability_scan
# ---------------------------------------------------------------------------

class TestStabilityAnalyzerScan:

    def _raw_result(self) -> dict:
        return {
            "window_results": [
                {"n_days": 30, "params": _params(0)},
                {"n_days": 60, "params": _params(1)},
            ],
            "stability_df": None,
            "output_paths": {"csv": "/tmp/out.csv", "heatmap": "/tmp/heatmap.png"},
        }

    def test_returns_stability_report(self):
        with patch("tuning.stability.run_stability_scan", return_value=self._raw_result()):
            result = StabilityAnalyzer().scan()
        assert isinstance(result, StabilityReport)

    def test_n_windows_from_raw(self):
        with patch("tuning.stability.run_stability_scan", return_value=self._raw_result()):
            result = StabilityAnalyzer().scan()
        assert result.n_windows == 2

    def test_output_paths_forwarded(self):
        with patch("tuning.stability.run_stability_scan", return_value=self._raw_result()):
            result = StabilityAnalyzer().scan()
        assert "csv" in result.output_paths

    def test_passes_windows_to_scan(self):
        with patch("tuning.stability.run_stability_scan", return_value=self._raw_result()) as mock_rss:
            StabilityAnalyzer().scan(windows=[30, 45, 60])
        mock_rss.assert_called_once_with(windows=[30, 45, 60], mode=None, output_dir=None)

    def test_passes_mode_and_output_dir(self):
        with patch("tuning.stability.run_stability_scan", return_value=self._raw_result()) as mock_rss:
            StabilityAnalyzer().scan(mode="walk_forward_price_only_test", output_dir="/out")
        mock_rss.assert_called_once_with(
            windows=None, mode="walk_forward_price_only_test", output_dir="/out"
        )

    def test_stability_df_forwarded(self):
        raw = self._raw_result()
        raw["stability_df"] = MagicMock()  # stand-in for a DataFrame
        with patch("tuning.stability.run_stability_scan", return_value=raw):
            result = StabilityAnalyzer().scan()
        assert result.stability_df is raw["stability_df"]

    def test_param_names_returns_list(self):
        names = StabilityAnalyzer().param_names()
        assert isinstance(names, list)
        assert len(names) == _N_PARAMS
