"""
tests/test_cli.py — CLI command handlers and dispatcher tests (Phase 9).

All tests mock the underlying engines/analyzers at the class-method level
so no real backtest, optimizer, or file I/O runs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch

import numpy as np
import pytest

import tuning.reports as _t
from cli.commands import (
    cmd_auto_tune,
    cmd_backtest,
    cmd_report,
    cmd_stability_scan,
    cmd_tune,
)
from cli.main import main as cli_main

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sim(total_return=0.12, sharpe=0.80, calmar=1.2, max_drawdown=-0.08, trades=40):
    from backtesting.types import SimResult
    return SimResult(
        final_value=11_200.0, total_return=total_return,
        sharpe=sharpe, calmar=calmar, max_drawdown=max_drawdown,
        trades_made=trades,
    )


def _bt_result():
    from backtesting.results import BacktestResult
    from backtesting.types import BacktestReport
    from backtesting.types import SimResult as BtSim
    sim = BtSim(
        final_value=12_000.0, total_return=0.20, sharpe=1.2,
        calmar=0.8, max_drawdown=-0.08, trades_made=20,
    )
    report = BacktestReport(
        mode="liquid_universe_full",
        universe_selection="liquid_all",
        lookahead_bias_level="LOW",
        n_symbols=300, n_days=90,
        train_result=sim, validation_result=None,
        benchmark_return=0.10, benchmark_sharpe=0.50,
        benchmark_max_drawdown=-0.05,
        excess_return=0.10, validation_benchmark_return=0.0, notes=[],
    )
    return BacktestResult(report=report, n_days=90, mode="liquid_universe_full")


def _tune_result(objective="sharpe"):
    from tuning.results import TuneResult
    p = np.random.default_rng(0).uniform(0.1, 0.5, 15)
    return TuneResult(
        params=p, sim=_sim(), objective=objective, n_days=90,
        active_params=["sw_quality", "sw_momentum", "index_pct"],
    )


def _auto_tune_result(val_passed=True):
    from tuning.results import AutoTuneResult
    p = np.random.default_rng(0).uniform(0.1, 0.5, 15)
    return AutoTuneResult(
        avg_params=p, sharpe_params=p, calmar_params=p + 0.01,
        sharpe_result=_sim(sharpe=0.85), calmar_result=_sim(calmar=1.3),
        avg_result=_sim(total_return=0.10),
        n_days=90,
        validation_passed=val_passed,
        validation_reasons=[] if val_passed else ["Sharpe too low"],
        config_written=False,
        active_params=["sw_quality", "sw_momentum", "index_pct"],
    )


def _stability_report():
    from tuning.results import StabilityReport
    return StabilityReport(
        window_results=[{"n_days": 30}, {"n_days": 60}],
        output_paths={"csv": "/tmp/s.csv"},
    )


# ---------------------------------------------------------------------------
# cmd_backtest
# ---------------------------------------------------------------------------

class TestCmdBacktest:

    def test_calls_engine_run(self):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()) as mock_run:
            cmd_backtest(n_days=90)
        mock_run.assert_called_once()

    def test_passes_n_days(self):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()) as mock_run:
            cmd_backtest(n_days=180)
        mock_run.assert_called_once_with(n_days=180, params=None, mode=None, scope="overall_strategy")

    def test_passes_mode(self):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()) as mock_run:
            cmd_backtest(n_days=90, mode="liquid_universe_full")
        mock_run.assert_called_once_with(n_days=90, params=None, mode="liquid_universe_full", scope="overall_strategy")

    def test_default_scope_is_overall_strategy(self):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()) as mock_run:
            cmd_backtest(n_days=90)
        assert mock_run.call_args.kwargs["scope"] == "overall_strategy"

    def test_passes_scope_active_sleeve(self):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()) as mock_run:
            cmd_backtest(n_days=90, scope="active_sleeve_compounding")
        mock_run.assert_called_once_with(n_days=90, params=None, mode=None,
                                         scope="active_sleeve_compounding")

    def test_prints_result(self, capsys):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()):
            cmd_backtest(n_days=90)
        assert capsys.readouterr().out.strip()


# ---------------------------------------------------------------------------
# cmd_tune
# ---------------------------------------------------------------------------

class TestCmdTune:

    def test_calls_tuner_tune(self):
        with patch("tuning.tuner.ParameterTuner.tune", return_value=_tune_result()) as mock_tune, \
             patch.object(_t, "print_config_diff"):
            cmd_tune(n_days=90)
        mock_tune.assert_called_once_with(n_days=90, objective="sharpe", mode=None,
                                          scope="overall_strategy", preset=None)

    def test_passes_objective(self):
        with patch("tuning.tuner.ParameterTuner.tune", return_value=_tune_result("calmar")) as mock_tune, \
             patch.object(_t, "print_config_diff"):
            cmd_tune(n_days=60, objective="calmar")
        mock_tune.assert_called_once_with(n_days=60, objective="calmar", mode=None,
                                          scope="overall_strategy", preset=None)

    def test_passes_mode(self):
        with patch("tuning.tuner.ParameterTuner.tune", return_value=_tune_result()) as mock_tune, \
             patch.object(_t, "print_config_diff"):
            cmd_tune(n_days=90, mode="walk_forward_price_only_test")
        mock_tune.assert_called_once_with(n_days=90, objective="sharpe",
                                          mode="walk_forward_price_only_test",
                                          scope="overall_strategy", preset=None)

    def test_passes_scope_active_sleeve(self):
        with patch("tuning.tuner.ParameterTuner.tune", return_value=_tune_result()) as mock_tune, \
             patch.object(_t, "print_config_diff"):
            cmd_tune(n_days=90, scope="active_sleeve_compounding")
        mock_tune.assert_called_once_with(n_days=90, objective="sharpe", mode=None,
                                          scope="active_sleeve_compounding", preset=None)

    def test_passes_preset(self):
        with patch("tuning.tuner.ParameterTuner.tune", return_value=_tune_result()) as mock_tune, \
             patch.object(_t, "print_config_diff"):
            cmd_tune(n_days=90, preset="active_factor_internals")
        mock_tune.assert_called_once_with(n_days=90, objective="sharpe", mode=None,
                                          scope="overall_strategy", preset="active_factor_internals")

    def test_default_preset_is_none(self):
        with patch("tuning.tuner.ParameterTuner.tune", return_value=_tune_result()) as mock_tune, \
             patch.object(_t, "print_config_diff"):
            cmd_tune(n_days=90)
        assert mock_tune.call_args.kwargs["preset"] is None

    def test_calls_print_config_diff_with_typed_fields(self):
        result = _tune_result()
        with patch("tuning.tuner.ParameterTuner.tune", return_value=result), \
             patch.object(_t, "print_config_diff") as mock_pcd:
            cmd_tune(n_days=90)
        mock_pcd.assert_called_once_with(result.params, result.sim)


# ---------------------------------------------------------------------------
# cmd_auto_tune
# ---------------------------------------------------------------------------

class TestCmdAutoTune:

    def test_calls_auto_tune(self):
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=_auto_tune_result()) as mock_at, \
             patch.object(_t, "_diff_table"):
            cmd_auto_tune(n_days=90)
        mock_at.assert_called_once_with(
            n_days=90, apply=False, force_apply=False, mode=None, llm_review=False,
            scope="overall_strategy", preset=None,
        )

    def test_passes_scope_active_sleeve(self):
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=_auto_tune_result()) as mock_at, \
             patch.object(_t, "_diff_table"):
            cmd_auto_tune(n_days=90, scope="active_sleeve_compounding")
        mock_at.assert_called_once_with(
            n_days=90, apply=False, force_apply=False, mode=None, llm_review=False,
            scope="active_sleeve_compounding", preset=None,
        )

    def test_passes_preset(self):
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=_auto_tune_result()) as mock_at, \
             patch.object(_t, "_diff_table"):
            cmd_auto_tune(n_days=90, preset="active_core_weights")
        mock_at.assert_called_once_with(
            n_days=90, apply=False, force_apply=False, mode=None, llm_review=False,
            scope="overall_strategy", preset="active_core_weights",
        )

    def test_default_preset_is_none(self):
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=_auto_tune_result()) as mock_at, \
             patch.object(_t, "_diff_table"):
            cmd_auto_tune(n_days=90)
        assert mock_at.call_args.kwargs["preset"] is None

    def test_passes_apply_flag(self):
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=_auto_tune_result()) as mock_at, \
             patch.object(_t, "_diff_table"):
            cmd_auto_tune(n_days=90, apply=True)
        assert mock_at.call_args.kwargs["apply"] is True

    def test_calls_diff_table_with_typed_fields(self):
        result = _auto_tune_result()
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=result), \
             patch.object(_t, "_diff_table") as mock_dt:
            cmd_auto_tune(n_days=90)
        call_args, call_kwargs = mock_dt.call_args
        assert "sharpe_ref" in call_kwargs
        assert "calmar_ref" in call_kwargs
        assert "sharpe_params" in call_kwargs
        assert "calmar_params" in call_kwargs

    def test_prints_summary(self, capsys):
        with patch("tuning.tuner.ParameterTuner.auto_tune", return_value=_auto_tune_result()), \
             patch.object(_t, "_diff_table"):
            cmd_auto_tune(n_days=90)
        out = capsys.readouterr().out
        assert any(kw in out for kw in ("PASSED", "FAILED", "WRITTEN", "ret="))


# ---------------------------------------------------------------------------
# cmd_stability_scan
# ---------------------------------------------------------------------------

class TestCmdStabilityScan:

    def test_calls_scan(self):
        with patch("tuning.stability.StabilityAnalyzer.scan", return_value=_stability_report()) as mock_scan:
            cmd_stability_scan()
        mock_scan.assert_called_once_with(windows=None, mode=None, output_dir=None)

    def test_passes_windows_mode_output_dir(self):
        with patch("tuning.stability.StabilityAnalyzer.scan", return_value=_stability_report()) as mock_scan:
            cmd_stability_scan(windows=[30, 60], mode="walk_forward_price_only_test", output_dir="/out")
        mock_scan.assert_called_once_with(
            windows=[30, 60], mode="walk_forward_price_only_test", output_dir="/out"
        )

    def test_prints_summary(self, capsys):
        with patch("tuning.stability.StabilityAnalyzer.scan", return_value=_stability_report()):
            cmd_stability_scan()
        assert "windows" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_report
# ---------------------------------------------------------------------------

class TestCmdReport:

    def test_calls_backtest_engine(self):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()) as mock_run:
            cmd_report()
        mock_run.assert_called_once_with(n_days=90)

    def test_prints_output(self, capsys):
        with patch("backtesting.engine.BacktestEngine.run", return_value=_bt_result()):
            cmd_report()
        assert capsys.readouterr().out.strip()


# ---------------------------------------------------------------------------
# cli.main dispatcher
# ---------------------------------------------------------------------------

class TestCliDispatch:

    def test_dispatch_backtest(self):
        with patch("cli.commands.cmd_backtest") as mock_cmd:
            cli_main(["backtest", "180"])
        mock_cmd.assert_called_once_with(n_days=180, mode=None, compare=False,
                                         archetype_compare=False, scope="overall_strategy")

    def test_dispatch_backtest_with_scope(self):
        with patch("cli.commands.cmd_backtest") as mock_cmd:
            cli_main(["backtest", "90", "--scope", "active_sleeve_compounding"])
        mock_cmd.assert_called_once_with(n_days=90, mode=None, compare=False,
                                         archetype_compare=False,
                                         scope="active_sleeve_compounding")

    def test_dispatch_backtest_default_scope(self):
        with patch("cli.commands.cmd_backtest") as mock_cmd:
            cli_main(["backtest", "90"])
        assert mock_cmd.call_args.kwargs["scope"] == "overall_strategy"

    def test_dispatch_tune(self):
        with patch("cli.commands.cmd_tune") as mock_cmd:
            cli_main(["tune", "90", "--objective", "calmar"])
        mock_cmd.assert_called_once_with(n_days=90, objective="calmar", mode=None,
                                         scope="overall_strategy", preset=None)

    def test_dispatch_tune_with_scope(self):
        with patch("cli.commands.cmd_tune") as mock_cmd:
            cli_main(["tune", "90", "--scope", "active_sleeve_compounding"])
        mock_cmd.assert_called_once_with(n_days=90, objective="sharpe", mode=None,
                                         scope="active_sleeve_compounding", preset=None)

    def test_dispatch_tune_with_preset(self):
        with patch("cli.commands.cmd_tune") as mock_cmd:
            cli_main(["tune", "90", "--preset", "active_factor_internals"])
        mock_cmd.assert_called_once_with(n_days=90, objective="sharpe", mode=None,
                                         scope="overall_strategy",
                                         preset="active_factor_internals")

    def test_dispatch_tune_omitted_preset_is_none(self):
        with patch("cli.commands.cmd_tune") as mock_cmd:
            cli_main(["tune", "90"])
        assert mock_cmd.call_args.kwargs["preset"] is None

    def test_dispatch_auto_tune_defaults(self):
        with patch("cli.commands.cmd_auto_tune") as mock_cmd:
            cli_main(["auto-tune"])
        mock_cmd.assert_called_once_with(
            n_days=90, mode=None, apply=False, force_apply=False, llm_review=False,
            scope="overall_strategy", preset=None,
        )

    def test_dispatch_auto_tune_with_apply(self):
        with patch("cli.commands.cmd_auto_tune") as mock_cmd:
            cli_main(["auto-tune", "120", "--apply"])
        mock_cmd.assert_called_once_with(
            n_days=120, mode=None, apply=True, force_apply=False, llm_review=False,
            scope="overall_strategy", preset=None,
        )

    def test_dispatch_auto_tune_with_scope(self):
        with patch("cli.commands.cmd_auto_tune") as mock_cmd:
            cli_main(["auto-tune", "90", "--scope", "active_sleeve_compounding"])
        mock_cmd.assert_called_once_with(
            n_days=90, mode=None, apply=False, force_apply=False, llm_review=False,
            scope="active_sleeve_compounding", preset=None,
        )

    def test_dispatch_auto_tune_with_preset(self):
        with patch("cli.commands.cmd_auto_tune") as mock_cmd:
            cli_main(["auto-tune", "90", "--preset", "active_core_weights"])
        mock_cmd.assert_called_once_with(
            n_days=90, mode=None, apply=False, force_apply=False, llm_review=False,
            scope="overall_strategy", preset="active_core_weights",
        )

    def test_dispatch_auto_tune_scope_and_preset_together(self):
        with patch("cli.commands.cmd_auto_tune") as mock_cmd:
            cli_main(["auto-tune", "90",
                      "--scope", "active_sleeve_compounding",
                      "--preset", "active_core_weights"])
        mock_cmd.assert_called_once_with(
            n_days=90, mode=None, apply=False, force_apply=False, llm_review=False,
            scope="active_sleeve_compounding", preset="active_core_weights",
        )

    def test_dispatch_auto_tune_omitted_preset_is_none(self):
        with patch("cli.commands.cmd_auto_tune") as mock_cmd:
            cli_main(["auto-tune", "90"])
        assert mock_cmd.call_args.kwargs["preset"] is None

    def test_dispatch_stability_scan(self):
        with patch("cli.commands.cmd_stability_scan") as mock_cmd:
            cli_main(["stability-scan", "--mode", "walk_forward_price_only_test"])
        mock_cmd.assert_called_once_with(mode="walk_forward_price_only_test", output_dir=None)

    def test_dispatch_fmp_status_default(self):
        with patch("cli.commands.cmd_fmp") as mock_cmd:
            cli_main(["fmp"])
        mock_cmd.assert_called_once_with(action="status")

    def test_dispatch_fmp_backfill_prices(self):
        with patch("cli.commands.cmd_fmp") as mock_cmd:
            cli_main([
                "fmp", "backfill-prices",
                "--symbols", "AAPL,MSFT",
                "--start", "2020-01-01",
                "--end", "2024-01-01",
                "--max-symbols", "2",
                "--force",
            ])
        mock_cmd.assert_called_once_with(
            action="backfill-prices",
            symbols_source="AAPL,MSFT",
            start="2020-01-01",
            end="2024-01-01",
            max_symbols=2,
            force=True,
        )

    def test_dispatch_fmp_build_dead_universe(self):
        with patch("cli.commands.cmd_fmp") as mock_cmd:
            cli_main([
                "fmp", "build-dead-universe",
                "--min-adv", "750000",
                "--max-symbols", "10",
                "--fetch-prices",
            ])
        mock_cmd.assert_called_once_with(
            action="build-dead-universe",
            start="2015-01-01",
            end="2030-01-01",
            min_adv=750000.0,
            max_symbols=10,
            allow_fetch_prices=True,
        )

    def test_dispatch_unknown_exits(self):
        with pytest.raises(SystemExit):
            cli_main(["no-such-command"])
