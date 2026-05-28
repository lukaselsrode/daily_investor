"""
tests/test_reporting.py — Reporting and attribution tests (Phase 8).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtesting.results import BacktestResult as BtResult
from backtesting.types import BacktestReport as BtReport
from backtesting.types import SimResult
from backtesting.types import SimResult as BtSim
from core.types import TradeRecord
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
            trim_count=3,
            harvest_count=5,
        )
        assert r.stopout_count == 2
        assert r.trim_count == 3
        assert r.harvest_count == 5
        assert r.sells_made == 0  # default


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
        with patch("reporting.attribution.compute_parameter_stability", return_value=mock_df) as mock_fn:
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
        with patch("reporting.attribution.compute_parameter_stability", return_value=_stability_df()) as mock_fn:
            AttributionReporter().compute_stability(wr, _PARAM_NAMES, cv_threshold=0.20, spread_threshold=0.10)
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, cv_threshold=0.20, spread_threshold=0.10)

    def test_classify_delegates(self):
        with patch("reporting.attribution.classify_stability", return_value="STABLE") as mock_fn:
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

    def _sell_trades(self) -> list:
        return [
            TradeRecord(date="1", symbol="AAPL", side="sell", quantity=10, price=100,
                        amount=1000, exit_type="stop_loss", pnl=-50.0, hold_days=5),
            TradeRecord(date="2", symbol="MSFT", side="sell", quantity=5, price=200,
                        amount=1000, exit_type="take_profit", pnl=100.0, hold_days=20),
            TradeRecord(date="3", symbol="GOOG", side="buy", quantity=2, price=150,
                        amount=300),  # buys should be ignored
        ]

    def test_factor_attribution_returns_dict(self):
        result = AttributionReporter().factor_attribution(self._sell_trades())
        assert isinstance(result, dict)
        assert "stop_loss" in result
        assert result["stop_loss"]["count"] == 1
        assert result["stop_loss"]["total_pnl"] == pytest.approx(-50.0)

    def test_factor_attribution_ignores_buys(self):
        result = AttributionReporter().factor_attribution(self._sell_trades())
        keys = set(result.keys())
        assert len(keys) == 2  # stop_loss + take_profit only (buy ignored)

    def test_sleeve_attribution_returns_dict(self):
        mock_report = MagicMock()
        mock_report.train.etf_return = 0.05
        mock_report.train.stock_return = 0.10
        mock_report.train.etf_allocation_avg = 0.20
        result = AttributionReporter().sleeve_attribution(mock_report)
        assert isinstance(result, dict)
        assert "etf" in result
        assert "stock" in result
        assert result["etf"]["return"] == pytest.approx(0.05)

    def test_exit_type_breakdown_returns_dict(self):
        result = AttributionReporter().exit_type_breakdown(self._sell_trades())
        assert isinstance(result, dict)
        assert "stop_loss" in result
        assert result["stop_loss"]["win_rate"] == pytest.approx(0.0)
        assert result["take_profit"]["win_rate"] == pytest.approx(1.0)

    def test_exit_type_breakdown_win_loss_counts(self):
        result = AttributionReporter().exit_type_breakdown(self._sell_trades())
        assert result["stop_loss"]["losses"] == 1
        assert result["take_profit"]["wins"] == 1


# ---------------------------------------------------------------------------
# Phase 8: DiagnosticsReporter
# ---------------------------------------------------------------------------

class TestDiagnosticsReporter:

    def test_write_stability_csv_delegates(self):
        df, wr = _stability_df(), _window_results()
        with patch("reporting.diagnostics.write_stability_summary_csv", return_value="/tmp/out.csv") as mock_fn:
            result = DiagnosticsReporter().write_stability_csv(df, wr, "/tmp")
        mock_fn.assert_called_once_with(df, wr, "/tmp", None)
        assert result == "/tmp/out.csv"

    def test_write_stability_csv_passes_date(self):
        df, wr = _stability_df(), _window_results()
        with patch("reporting.diagnostics.write_stability_summary_csv", return_value="/tmp/out.csv") as mock_fn:
            DiagnosticsReporter().write_stability_csv(df, wr, "/tmp", date="2026-01-01")
        mock_fn.assert_called_once_with(df, wr, "/tmp", "2026-01-01")

    def test_write_robustness_txt_delegates(self):
        df, wr = _stability_df(), _window_results()
        with patch("reporting.diagnostics.write_robustness_report_txt", return_value="/tmp/r.txt") as mock_fn:
            result = DiagnosticsReporter().write_robustness_txt(df, wr, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(df, wr, _PARAM_NAMES, "/tmp", None)
        assert result == "/tmp/r.txt"

    def test_generate_all_delegates(self):
        df, wr = _stability_df(), _window_results()
        expected = {"stability_csv": "/tmp/s.csv", "robustness_txt": "/tmp/r.txt"}
        with patch("reporting.diagnostics.generate_all_reports", return_value=expected) as mock_fn:
            result = DiagnosticsReporter().generate_all(wr, df, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(wr, df, _PARAM_NAMES, "/tmp")
        assert result is expected

    def test_generate_all_returns_dict(self):
        df, wr = _stability_df(), _window_results()
        with patch("reporting.diagnostics.generate_all_reports", return_value={"stability_csv": "/x"}):
            result = DiagnosticsReporter().generate_all(wr, df, _PARAM_NAMES, "/tmp")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Phase 8: PlotManager
# ---------------------------------------------------------------------------

class TestPlotManager:

    def test_param_heatmap_delegates(self):
        wr = _window_results()
        with patch("reporting.plots.generate_param_heatmap", return_value="/tmp/h.png") as mock_fn:
            result = PlotManager().param_heatmap(wr, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, "/tmp", None)
        assert result == "/tmp/h.png"

    def test_param_heatmap_passes_date(self):
        wr = _window_results()
        with patch("reporting.plots.generate_param_heatmap", return_value="/tmp/h.png") as mock_fn:
            PlotManager().param_heatmap(wr, _PARAM_NAMES, "/tmp", date="2026-01-01")
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, "/tmp", "2026-01-01")

    def test_objective_heatmap_delegates(self):
        wr = _window_results()
        with patch("reporting.plots.generate_objective_heatmap", return_value="/tmp/o.png") as mock_fn:
            result = PlotManager().objective_heatmap(wr, _PARAM_NAMES, "/tmp")
        mock_fn.assert_called_once_with(wr, _PARAM_NAMES, "/tmp", None)
        assert result == "/tmp/o.png"

    def test_validation_heatmap_delegates(self):
        wr = _window_results()
        with patch("reporting.plots.generate_validation_heatmap", return_value="/tmp/v.png") as mock_fn:
            result = PlotManager().validation_heatmap(wr, "/tmp")
        mock_fn.assert_called_once_with(wr, "/tmp", None)
        assert result == "/tmp/v.png"

    def test_validation_heatmap_passes_date(self):
        wr = _window_results()
        with patch("reporting.plots.generate_validation_heatmap", return_value="/tmp/v.png") as mock_fn:
            PlotManager().validation_heatmap(wr, "/tmp", date="2026-01-01")
        mock_fn.assert_called_once_with(wr, "/tmp", "2026-01-01")
