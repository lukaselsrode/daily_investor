"""
tests/test_exit_cadence.py — simulator exit cadence vs live weekly execution.

Live buy/sell cycles run once weekly (mid-day Wednesday); the simulator's
legacy behavior evaluated exits EVERY day, firing stop-losses days before the
live system could — optimistic drawdown numbers exactly where they matter
(crash tapes). exit_check_frequency_days=5 defers mid-week breaches to the
next check day, mirroring live.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # sibling test-module fixtures

import numpy as np
from test_active_sleeve_accounting import _OPEN_CS, _make_precomp, _no_exit_params

from backtesting.simulator import run_simulation
from util import BACKTEST_PARAMS, SELL_RULES

STOP = float(SELL_RULES["stop_loss_pct"])


def _crashing_precomp(n_days: int = 20):
    """Stock 0 trades flat at 100, breaches the stop on day 7 (not an exit-check
    day at cadence 5), and keeps sliding — so a deferred exit sells lower."""
    prices = np.full((n_days, 4), 100.0)
    path = {7: 100 * (1 + STOP - 0.20), 8: 45.0, 9: 42.0}  # deep breach day 7
    for d in range(7, n_days):
        prices[d, 0] = path.get(d, 40.0)
    return _make_precomp(n_days=n_days, stock_prices=prices)


def _run(precomp, cadence: int):
    return run_simulation(
        precomp, _no_exit_params(index_pct=0.30), 10_000.0,
        rebalance_frequency_days=5,
        exit_check_frequency_days=cadence,
        cs_params=_OPEN_CS,
    )


def _stop_sell_day(result) -> int:
    days = [int(t.date) for t in result.trade_log
            if t.side == "sell" and t.exit_type == "stop_loss"]
    assert days, "expected a stop-loss sell in the trade log"
    return min(days)


class TestExitCadence:

    def test_daily_cadence_fires_on_breach_day(self):
        res = _run(_crashing_precomp(), cadence=1)
        assert _stop_sell_day(res) == 7

    def test_weekly_cadence_defers_to_next_check_day(self):
        res = _run(_crashing_precomp(), cadence=5)
        assert _stop_sell_day(res) == 10  # 7, 8, 9 skipped; next multiple of 5

    def test_weekly_cadence_is_costlier_in_a_slide(self):
        """The whole point: deferring the exit through a falling tape ends
        worse — daily-exit backtests overstate crash protection."""
        daily = _run(_crashing_precomp(), cadence=1)
        weekly = _run(_crashing_precomp(), cadence=5)
        assert weekly.final_value < daily.final_value

    def test_final_day_always_evaluates(self):
        """A breach just before the window ends must not be silently held past
        the terminal mark."""
        n = 13  # last day index 12 — not a multiple of 5
        prices = np.full((n, 4), 100.0)
        prices[12, 0] = 40.0
        res = _run(_make_precomp(n_days=n, stock_prices=prices), cadence=5)
        assert _stop_sell_day(res) == 12

    def test_config_documents_the_knob(self):
        assert "exit_check_frequency_days" in BACKTEST_PARAMS
        assert int(BACKTEST_PARAMS["exit_check_frequency_days"]) >= 1
