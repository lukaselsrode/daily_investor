"""
tests/test_tuner_gates.py — incumbent-relative + random-window tuning gates.

Motivating regression (2026-06-10): an auto-tuned config passed the absolute
gates with +0.34% validation excess-vs-SPY while the incumbent config scored
+8.87% on the same split with 5x less turnover. The gates here ensure a tuned
candidate must BEAT the running config out-of-sample, reproducibly, before
--apply may write it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from backtesting.random_walk import RandomWindowSummary, WindowResult
from backtesting.types import BacktestReport, SimResult
from tuning.tuner import paired_random_window_gate, validate_tuned_params

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sim(
    total_return: float = 0.12,
    sharpe: float = 1.0,
    max_drawdown: float = -0.08,
    turnover: float = 1.0,
) -> SimResult:
    return SimResult(
        final_value=11_200.0,
        total_return=total_return,
        sharpe=sharpe,
        calmar=1.2,
        max_drawdown=max_drawdown,
        trades_made=40,
        turnover_estimate=turnover,
    )


def _report(val: SimResult, val_bench_return: float = 0.05) -> BacktestReport:
    return BacktestReport(
        mode="liquid_universe_full",
        universe_selection="liquid_all",
        lookahead_bias_level="LOW",
        n_symbols=50,
        n_days=63,
        train_result=_sim(),
        validation_result=val,
        benchmark_return=0.08,
        benchmark_sharpe=0.60,
        benchmark_max_drawdown=-0.06,
        excess_return=0.04,
        validation_benchmark_return=val_bench_return,
        notes=[],
    )


# Gate config: absolute floors permissive so the incumbent-relative gates are
# what each test exercises. Keys mirror cfg/config.yaml backtest.*.
_CFG = {
    "min_validation_excess_return": 0.0,
    "max_validation_drawdown": -0.20,
    "min_validation_sharpe": 0.25,
    "min_excess_vs_incumbent": 0.0,
    "max_turnover_multiple": 2.0,
}


# ---------------------------------------------------------------------------
# Incumbent-relative excess gate
# ---------------------------------------------------------------------------

class TestIncumbentExcessGate:

    def test_motivating_overfit_case_rejected(self):
        # Tuned: +0.34% val excess. Incumbent: +8.87%. Absolute gates pass; the
        # incumbent-relative gate must reject.
        bench = 0.0752
        tuned = _report(_sim(total_return=bench + 0.0034, turnover=5.03), val_bench_return=bench)
        incumbent = _report(_sim(total_return=bench + 0.0887, turnover=0.93), val_bench_return=bench)

        ok_absolute, _ = validate_tuned_params(tuned, _CFG)
        assert ok_absolute  # the old gates let this through — that was the bug

        ok, reasons = validate_tuned_params(tuned, _CFG, incumbent_report=incumbent)
        assert not ok
        assert any("incumbent" in r for r in reasons)

    def test_beating_incumbent_passes(self):
        bench = 0.05
        tuned = _report(_sim(total_return=bench + 0.10), val_bench_return=bench)
        incumbent = _report(_sim(total_return=bench + 0.08), val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, _CFG, incumbent_report=incumbent)
        assert ok, reasons

    def test_margin_is_honored(self):
        bench = 0.05
        cfg = dict(_CFG, min_excess_vs_incumbent=0.05)
        # Beats incumbent by 2% but margin demands 5%.
        tuned = _report(_sim(total_return=bench + 0.10), val_bench_return=bench)
        incumbent = _report(_sim(total_return=bench + 0.08), val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, cfg, incumbent_report=incumbent)
        assert not ok
        assert any("incumbent" in r for r in reasons)

    def test_no_incumbent_keeps_legacy_behavior(self):
        bench = 0.05
        tuned = _report(_sim(total_return=bench + 0.001), val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, _CFG)
        assert ok, reasons

    def test_incumbent_without_validation_window_is_ignored(self):
        bench = 0.05
        tuned = _report(_sim(total_return=bench + 0.001), val_bench_return=bench)
        incumbent = _report(None, val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, _CFG, incumbent_report=incumbent)
        assert ok, reasons


# ---------------------------------------------------------------------------
# Turnover gate
# ---------------------------------------------------------------------------

class TestTurnoverGate:

    def test_turnover_blowup_rejected_even_when_excess_beats(self):
        bench = 0.05
        tuned = _report(_sim(total_return=bench + 0.10, turnover=5.0), val_bench_return=bench)
        incumbent = _report(_sim(total_return=bench + 0.08, turnover=0.9), val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, _CFG, incumbent_report=incumbent)
        assert not ok
        assert any("turnover" in r.lower() for r in reasons)

    def test_turnover_within_multiple_passes(self):
        bench = 0.05
        tuned = _report(_sim(total_return=bench + 0.10, turnover=1.7), val_bench_return=bench)
        incumbent = _report(_sim(total_return=bench + 0.08, turnover=0.9), val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, _CFG, incumbent_report=incumbent)
        assert ok, reasons

    def test_turnover_multiple_from_cfg(self):
        bench = 0.05
        cfg = dict(_CFG, max_turnover_multiple=10.0)
        tuned = _report(_sim(total_return=bench + 0.10, turnover=5.0), val_bench_return=bench)
        incumbent = _report(_sim(total_return=bench + 0.08, turnover=0.9), val_bench_return=bench)
        ok, reasons = validate_tuned_params(tuned, cfg, incumbent_report=incumbent)
        assert ok, reasons


# ---------------------------------------------------------------------------
# Paired random-window gate
# ---------------------------------------------------------------------------

def _summary(excesses: list[float], robust: float, params=None) -> RandomWindowSummary:
    windows = [
        WindowResult(
            window_id=i, start_day=i * 10, end_day=i * 10 + 120,
            strategy_return=e + 0.05, benchmark_return=0.05, excess_return=e,
            sharpe=1.0, max_drawdown=-0.05, calmar=1.0, turnover=1.0,
            trades=10, avg_positions=20.0, wins_benchmark=e > 0,
        )
        for i, e in enumerate(excesses)
    ]
    return RandomWindowSummary(
        n_windows=len(windows),
        window_days=120,
        params_used=params if params is not None else np.zeros(16),
        window_results=windows,
        median_excess_return=float(np.median(excesses)),
        robust_score=robust,
    )


def _gate_with_fake_summaries(monkeypatch, tuned_sum, inc_sum, cfg=None):
    import backtesting.random_walk as rw

    calls = []

    def fake_rwb(precomp, params, **kw):
        calls.append(kw)
        # First call = tuned, second = incumbent (gate calls in this order).
        return tuned_sum if len(calls) == 1 else inc_sum

    monkeypatch.setattr(rw, "random_window_backtest", fake_rwb)
    bp = {"random_window_gate": cfg if cfg is not None else {}}
    passed, reasons, stats = paired_random_window_gate(
        np.zeros(16), np.ones(16), bp, precomp=object(),
    )
    # Shared windows demand a shared seed: both calls must use identical kwargs.
    assert len(calls) == 2 and calls[0] == calls[1]
    return passed, reasons, stats


class TestPairedRandomWindowGate:

    def test_disabled_gate_skips(self):
        bp = {"random_window_gate": {"enabled": False}}
        passed, reasons, stats = paired_random_window_gate(np.zeros(16), np.ones(16), bp)
        assert passed and not reasons and stats.get("skipped")

    def test_tuned_dominating_passes(self, monkeypatch):
        tuned = _summary([0.04, 0.05, 0.06, 0.03], robust=0.05)
        inc   = _summary([0.01, 0.02, 0.01, 0.02], robust=0.01)
        passed, reasons, stats = _gate_with_fake_summaries(monkeypatch, tuned, inc)
        assert passed, reasons
        assert stats["win_rate"] == 1.0

    def test_low_win_rate_fails(self, monkeypatch):
        # Tuned wins only 1 of 4 paired windows.
        tuned = _summary([0.04, 0.00, 0.00, 0.00], robust=0.05)
        inc   = _summary([0.01, 0.02, 0.01, 0.02], robust=0.01)
        passed, reasons, _ = _gate_with_fake_summaries(monkeypatch, tuned, inc)
        assert not passed
        assert any("win rate" in r.lower() for r in reasons)

    def test_lower_median_excess_fails(self, monkeypatch):
        # Wins half the windows narrowly, but loses on median excess: small wins,
        # big losses — exactly the asymmetry the median gate exists to catch.
        tuned = _summary([0.02, 0.02, 0.00, 0.00], robust=0.05)  # median 0.01
        inc   = _summary([0.01, 0.01, 0.03, 0.03], robust=0.01)  # median 0.02
        passed, reasons, _ = _gate_with_fake_summaries(monkeypatch, tuned, inc)
        assert not passed
        assert any("median excess" in r.lower() for r in reasons)

    def test_equal_median_excess_fails(self, monkeypatch):
        tuned = _summary([0.02, 0.02, 0.02, 0.01], robust=0.05)  # median 0.02
        inc   = _summary([0.02, 0.02, 0.02, 0.02], robust=0.01)  # median 0.02
        passed, reasons, _ = _gate_with_fake_summaries(monkeypatch, tuned, inc)
        assert not passed
        assert any("median excess" in r.lower() for r in reasons)

    def test_lower_robust_score_fails(self, monkeypatch):
        tuned = _summary([0.04, 0.05, 0.06, 0.03], robust=0.01)
        inc   = _summary([0.01, 0.02, 0.01, 0.02], robust=0.02)
        passed, reasons, _ = _gate_with_fake_summaries(monkeypatch, tuned, inc)
        assert not passed
        assert any("robust score" in r.lower() for r in reasons)

    def test_min_win_rate_from_cfg(self, monkeypatch):
        # 50% wins passes the default, fails a 0.75 requirement.
        tuned = _summary([0.04, 0.04, 0.00, 0.00], robust=0.05)
        inc   = _summary([0.01, 0.02, 0.01, 0.02], robust=0.01)
        passed, _, _ = _gate_with_fake_summaries(monkeypatch, tuned, inc)
        assert passed
        tuned2 = _summary([0.04, 0.04, 0.00, 0.00], robust=0.05)
        inc2   = _summary([0.01, 0.02, 0.01, 0.02], robust=0.01)
        passed2, reasons2, _ = _gate_with_fake_summaries(
            monkeypatch, tuned2, inc2, cfg={"min_win_rate": 0.75},
        )
        assert not passed2
        assert any("win rate" in r.lower() for r in reasons2)


# ---------------------------------------------------------------------------
# Live config sanity — the documented keys exist with gate-compatible values
# ---------------------------------------------------------------------------

def test_live_config_documents_gate_keys():
    from util import BACKTEST_PARAMS

    assert "min_excess_vs_incumbent" in BACKTEST_PARAMS
    assert "max_turnover_multiple" in BACKTEST_PARAMS
    gw = BACKTEST_PARAMS.get("random_window_gate")
    assert isinstance(gw, dict)
    for key in ("enabled", "history_days", "n_windows", "window_days", "seed", "min_win_rate"):
        assert key in gw
    # Windows must fit inside the gate history with room for variety.
    assert gw["history_days"] >= 2 * gw["window_days"]
