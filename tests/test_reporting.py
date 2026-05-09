"""
tests/test_reporting.py — Reporting and attribution tests (Phase 8).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from core.types import SimResult, BacktestReport, TradeRecord
from backtest import BacktestReport as BtReport, SimResult as BtSim
from backtesting.results import BacktestResult as BtResult

import _reporting_legacy as _rl
from reporting.attribution import AttributionReporter
from reporting.diagnostics import DiagnosticsReporter
from reporting.plots import PlotManager


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PARAM_NAMES = ["sw_quality", "sw_momentum", "index_pct"]
_N = len(_PARAM_NAMES)


def _window_results(n: int = 2) -> list[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "window": 30 * (i + 1),
            "params_avg":    rng.uniform(0.1, 0.5, _N),
            "params_sharpe": rng.uniform(0.1, 0.5, _N),
            "params_calmar": rng.uniform(0.1, 0.5, _N),
            "val_excess_return": 0.05,
            "val_sharpe":        0.80,
            "val_drawdown":     -0.07,
            "turnover":          0.30,
            "trades":            20,
            "unstable_params":   1,
            "validation_passed": True,
        }
        for i in range(n)
    ]


def _stability_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "param": "sw_quality", "mean": 0.25, "stddev": 0.02, "cv": 0.08,
        "sharpe_calmar_spread": 0.03, "convergence_frequency": 0.95,
        "instability_score": 0.15, "stability": "STABLE",
    }])


def _make_bt_report(
    total_return=0.20, sharpe=1.2, calmar=0.8, max_drawdown=-0.08,
    trades_made=20, benchmark_return=0.15, n_symbols=300, excess=0.05,
) -> BtReport:
    sim = BtSim(
        final_value=12000.0, total_return=total_return,
        sharpe=sharpe, calmar=calmar, max_drawdown=max_drawdown,
        trades_made=trades_made,
    )
    return BtReport(
        mode="liquid_universe_sanity_test",
        universe_selection="liquid_all",
        lookahead_bias_level="LOW",
        n_symbols=n_symbols,
        n_days=63,
        train_result=sim,
        validation_result=None,
        benchmark_return=benchmark_return,
        benchmark_sharpe=0.60,
        benchmark_max_drawdown=-0.06,
        excess_return=excess,
        validation_benchmark_return=0.0,
        notes=[],
    )


class TestSimResultExtended:

    def test_sim_result_extended_fields(self):
        r = SimResult(
            final_value=12000.0,
            total_return=0.20,
            sharpe=1.2,
            calmar=0.9,
            max_drawdown=-0.08,
            trades_made=25,
            stopout_count=2,
            trailing_stop_count=3,
            take_profit_count=5,
        )
        assert r.stopout_count == 2
        assert r.trailing_stop_count == 3
        assert r.take_profit_count == 5
        assert r.etf_return == 0.0  # default


class TestBacktestResult:

    def _make_result(self) -> BtResult:
        return BtResult(
            report=_make_bt_report(total_return=0.20, sharpe=1.2, max_drawdown=-0.08,
                                   benchmark_return=0.15, excess=0.05),
            n_days=365,
            mode="liquid_universe_sanity_test",
        )

    def test_excess_return(self):
        result = self._make_result()
        assert result.excess_return == pytest.approx(0.05)

    def test_passes_basic_validation(self):
        result = self._make_result()
        assert result.passes_basic_validation

    def test_fails_validation_low_sharpe(self):
        r = BtResult(
            report=_make_bt_report(sharpe=0.10, max_drawdown=-0.10),
            n_days=90,
            mode="test",
        )
        assert not r.passes_basic_validation

    def test_summary_dict_keys(self):
        result = self._make_result()
        d = result.summary_dict()
        assert "total_return" in d
        assert "sharpe" in d
        assert "max_drawdown" in d
        assert "trades" in d

    def test_str_representation(self):
        result = self._make_result()
        s = str(result)
        assert "365d" in s
        assert "sharpe" in s


# ---------------------------------------------------------------------------
# Phase 8: AttributionReporter
# ---------------------------------------------------------------------------

class TestAttributionReporter:

    def test_compute_stability_delegates(self):
        wr = _window_results()
        mock_df = _stability_df()
        with patch.object(_rl, "compute_parameter_stability", return_value=mock_df) as mock_fn:
            result = AttributionReporter().compute_stability(wr, _PARAM_NAMES)
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, cv_threshold=0.30, spread_threshold=0.15)
        assert result is mock_df

    def test_compute_stability_returns_dataframe(self):
        wr = _window_results(3)
        result = AttributionReporter().compute_stability(wr, _PARAM_NAMES)
        assert isinstance(result, pd.DataFrame)
        assert "param" in result.columns

    def test_compute_stability_custom_thresholds(self):
        wr = _window_results()
        with patch.object(_rl, "compute_parameter_stability", return_value=_stability_df()) as mock_fn:
            AttributionReporter().compute_stability(wr, _PARAM_NAMES, cv_threshold=0.20, spread_threshold=0.10)
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, cv_threshold=0.20, spread_threshold=0.10)

    def test_classify_delegates(self):
        with patch.object(_rl, "classify_stability", return_value="STABLE") as mock_fn:
            result = AttributionReporter().classify(0.1, 0.05)
        mock_fn.assert_called_once_with(0.1, 0.05, 0.30, 0.15)
        assert result == "STABLE"

    def test_classify_stable(self):
        assert AttributionReporter().classify(0.1, 0.05) == "STABLE"

    def test_classify_unstable(self):
        assert AttributionReporter().classify(0.5, 0.3) == "UNSTABLE"

    def test_classify_moderately_stable(self):
        # cv between 0.18 (0.6 * 0.30) and 0.30 → MODERATELY_STABLE
        assert AttributionReporter().classify(0.20, 0.05) == "MODERATELY_STABLE"

    def test_factor_attribution_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AttributionReporter().factor_attribution([])

    def test_sleeve_attribution_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AttributionReporter().sleeve_attribution(MagicMock())

    def test_exit_type_breakdown_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AttributionReporter().exit_type_breakdown([])


# ---------------------------------------------------------------------------
# Phase 8: DiagnosticsReporter
# ---------------------------------------------------------------------------

class TestDiagnosticsReporter:

    def test_write_stability_csv_delegates(self):
        df, wr = _stability_df(), _window_results()
        with patch.object(_rl, "write_stability_summary_csv", return_value="/tmp/out.csv") as mock_fn:
            result = DiagnosticsReporter().write_stability_csv(df, wr, "/tmp")
        mock_fn.assert_called_once_with(df, wr, "/tmp", None)
        assert result == "/tmp/out.csv"

    def test_write_stability_csv_passes_date(self):
        df, wr = _stability_df(), _window_results()
        with patch.object(_rl, "write_stability_summary_csv", return_value="/tmp/out.csv") as mock_fn:
            DiagnosticsReporter().write_stability_csv(df, wr, "/tmp", date="2026-01-01")
        mock_fn.assert_called_once_with(df, wr, "/tmp", "2026-01-01")

    def test_write_robustness_txt_delegates(self):
        df, wr = _stability_df(), _window_results()
        with patch.object(_rl, "write_robustness_report_txt", return_value="/tmp/r.txt") as mock_fn:
            result = DiagnosticsReporter().write_robustness_txt(df, wr, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(df, wr, _PARAM_NAMES, "/tmp", None)
        assert result == "/tmp/r.txt"

    def test_generate_all_delegates(self):
        df, wr = _stability_df(), _window_results()
        expected = {"stability_csv": "/tmp/s.csv", "robustness_txt": "/tmp/r.txt"}
        with patch.object(_rl, "generate_all_reports", return_value=expected) as mock_fn:
            result = DiagnosticsReporter().generate_all(wr, df, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(wr, df, _PARAM_NAMES, "/tmp")
        assert result is expected

    def test_generate_all_returns_dict(self):
        df, wr = _stability_df(), _window_results()
        with patch.object(_rl, "generate_all_reports", return_value={"stability_csv": "/x"}):
            result = DiagnosticsReporter().generate_all(wr, df, _PARAM_NAMES, "/tmp")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Phase 8: PlotManager
# ---------------------------------------------------------------------------

class TestPlotManager:

    def test_param_heatmap_delegates(self):
        wr = _window_results()
        with patch.object(_rl, "generate_param_heatmap", return_value="/tmp/h.png") as mock_fn:
            result = PlotManager().param_heatmap(wr, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, "/tmp", None)
        assert result == "/tmp/h.png"

    def test_param_heatmap_passes_date(self):
        wr = _window_results()
        with patch.object(_rl, "generate_param_heatmap", return_value="/tmp/h.png") as mock_fn:
            PlotManager().param_heatmap(wr, _PARAM_NAMES, "/tmp", date="2026-01-01")
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, "/tmp", "2026-01-01")

    def test_objective_heatmap_delegates(self):
        wr = _window_results()
        with patch.object(_rl, "generate_objective_heatmap", return_value="/tmp/o.png") as mock_fn:
            result = PlotManager().objective_heatmap(wr, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, "/tmp", None)
        assert result == "/tmp/o.png"

    def test_validation_heatmap_delegates(self):
        wr = _window_results()
        with patch.object(_rl, "generate_validation_heatmap", return_value="/tmp/v.png") as mock_fn:
            result = PlotManager().validation_heatmap(wr, "/tmp")
        mock_fn.assert_called_once_with(wr, "/tmp", None)
        assert result == "/tmp/v.png"

    def test_validation_heatmap_passes_date(self):
        wr = _window_results()
        with patch.object(_rl, "generate_validation_heatmap", return_value="/tmp/v.png") as mock_fn:
            PlotManager().validation_heatmap(wr, "/tmp", date="2026-01-01")
        mock_fn.assert_called_once_with(wr, "/tmp", "2026-01-01")
