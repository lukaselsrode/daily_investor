"""
tests/test_stress_gauntlet.py — the named stress-episode falsification gate.

The gauntlet's contract: episodes are date-anchored regime samples; the
candidate must SURVIVE each one relative to the incumbent (catastrophe-scale
floors), episodes outside the data axis or below coverage are SKIPPED visibly
(never silently passed as evidence), and a disabled gauntlet defers to the
prior gates.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtesting.types import SimResult
from util import BACKTEST_PARAMS

SEL = np.full(4, 1.0)
INC = np.full(4, 2.0)


def _sim(excess, dd=-0.10, turnover=1.0):
    return SimResult(final_value=1.0, total_return=excess, sharpe=1.0, calmar=1.0,
                     max_drawdown=dd, trades_made=10, turnover_estimate=turnover,
                     benchmark_twr=0.0)


def _precomp(start="2007-01-02", n=520, n_stocks=5, dates=None):
    cal = tuple(d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n))
    return SimpleNamespace(
        dates=cal if dates is None else dates,
        prices=np.ones((n, n_stocks)),
        tradeable_mask_daily=None,
    )


def _bp(episodes, **over):
    cfg = {
        "stress_gauntlet": {
            "enabled": True,
            "catastrophe_excess": 0.10,
            "catastrophe_drawdown": 0.05,
            "min_symbols": 3,
            "episodes": episodes,
            **over,
        },
        "max_turnover_multiple": 2.0,
        "starting_capital": 10_000.0,
        "slippage_bps": 0.0,
        "commission_per_trade": 0.0,
        "weekly_contribution": 0.0,
        "rebalance_frequency_days": 5,
    }
    return cfg


def _wire(monkeypatch, sel_sim, inc_sim):
    """Patch the gauntlet's collaborators; sims keyed by params identity."""
    import backtesting.regime_scope as rs
    import tuning.gauntlet as g

    monkeypatch.setattr(rs, "slice_precomp",
                        lambda pc, sl: SimpleNamespace(prices=pc.prices[sl]))
    monkeypatch.setattr(g, "gate_simulation",
                        lambda pc, params, *a, **k: sel_sim if np.allclose(params, SEL) else inc_sim)
    monkeypatch.setattr(g, "_dead_names_in_window", lambda s, e: 7)
    return g


EP = {"ep_2007": {"start": "2007-06-01", "end": "2007-12-31"}}


class TestStressGauntlet:

    def test_disabled_defers_to_prior_gates(self):
        from tuning.gauntlet import stress_gauntlet
        bp = _bp(EP)
        bp["stress_gauntlet"]["enabled"] = False
        passed, reasons, rows = stress_gauntlet(SEL, INC, bp, precomp=_precomp())
        assert passed and not reasons and not rows

    def test_missing_dates_fails_loudly(self, monkeypatch):
        g = _wire(monkeypatch, _sim(0.0), _sim(0.0))
        passed, reasons, _ = g.stress_gauntlet(
            SEL, INC, _bp(EP), precomp=SimpleNamespace(dates=None))
        assert not passed
        assert any("no dates" in r for r in reasons)

    def test_episode_predating_axis_skipped_visibly(self, monkeypatch):
        g = _wire(monkeypatch, _sim(0.01), _sim(0.0))
        eps = {"pre_axis": {"start": "2001-01-01", "end": "2001-12-31"}, **EP}
        passed, reasons, rows = g.stress_gauntlet(SEL, INC, _bp(eps), precomp=_precomp())
        assert passed, reasons
        by_name = {r["episode"]: r for r in rows}
        assert "predates data axis" in by_name["pre_axis"]["status"]
        assert by_name["ep_2007"]["status"] == "ok"

    def test_low_coverage_skipped_visibly(self, monkeypatch):
        g = _wire(monkeypatch, _sim(0.01), _sim(0.0))
        bp = _bp(EP)
        bp["stress_gauntlet"]["min_symbols"] = 50   # precomp has 5 stocks
        passed, _, rows = g.stress_gauntlet(SEL, INC, bp, precomp=_precomp())
        assert passed
        assert "only 5 symbols" in rows[0]["status"]

    def test_catastrophic_excess_regression_blocks(self, monkeypatch):
        g = _wire(monkeypatch, _sim(-0.20), _sim(0.0))
        passed, reasons, rows = g.stress_gauntlet(SEL, INC, _bp(EP), precomp=_precomp())
        assert not passed
        assert any("ep_2007" in r and "regression" in r for r in reasons)
        assert rows[0]["delta"] == pytest.approx(-0.20)

    def test_drawdown_breach_blocks(self, monkeypatch):
        g = _wire(monkeypatch, _sim(0.0, dd=-0.30), _sim(0.0, dd=-0.10))
        passed, reasons, _ = g.stress_gauntlet(SEL, INC, _bp(EP), precomp=_precomp())
        assert not passed
        assert any("drawdown" in r for r in reasons)

    def test_turnover_blowup_blocks(self, monkeypatch):
        g = _wire(monkeypatch, _sim(0.0, turnover=5.0), _sim(0.0, turnover=1.0))
        passed, reasons, _ = g.stress_gauntlet(SEL, INC, _bp(EP), precomp=_precomp())
        assert not passed
        assert any("turnover" in r for r in reasons)

    def test_survival_passes_and_reports_bias_signal(self, monkeypatch):
        """A surviving candidate passes; every run row carries the dead-name
        count so pre-2021 survivor bias stays visible in reports."""
        g = _wire(monkeypatch, _sim(0.01, dd=-0.12), _sim(0.0, dd=-0.10))
        passed, reasons, rows = g.stress_gauntlet(SEL, INC, _bp(EP), precomp=_precomp())
        assert passed, reasons
        assert rows[0]["n_dead"] == 7
        assert rows[0]["n_symbols"] == 5

    def test_tradeable_mask_drives_coverage(self, monkeypatch):
        """Coverage must count NATIVE-tradeable names, not ffilled corpses."""
        g = _wire(monkeypatch, _sim(0.01), _sim(0.0))
        pc = _precomp()
        mask = np.ones(pc.prices.shape, dtype=bool)
        mask[:, 3:] = False                      # only 3 of 5 names tradeable
        pc.tradeable_mask_daily = mask
        bp = _bp(EP)
        bp["stress_gauntlet"]["min_symbols"] = 4
        passed, _, rows = g.stress_gauntlet(SEL, INC, bp, precomp=pc)
        assert "only 3 symbols" in rows[0]["status"]

    def test_live_config_documents_keys(self):
        sg = BACKTEST_PARAMS.get("stress_gauntlet", {})
        for key in ("enabled", "history_days", "catastrophe_excess",
                    "catastrophe_drawdown", "min_symbols", "episodes"):
            assert key in sg, f"stress_gauntlet.{key} missing from live config"
        assert "gfc_2008" in sg["episodes"]
        for ep in sg["episodes"].values():
            assert str(ep["start"]) < str(ep["end"])

    def test_live_confirm_config_uses_symmetric_tolerance(self):
        mh = BACKTEST_PARAMS.get("multi_horizon_confirm", {})
        assert "regress_tolerance" in mh
        assert "mid_regress_tolerance" not in mh
        assert "short_regress_tolerance" not in mh
