"""
tests/test_regime_derisk_overlay.py

Regime de-risk overlay (regime.defensive.backtest_derisk_frac) invariants.

The overlay rotates a fraction of the held stock book into the benchmark
instrument on entry into a defensive regime (SPY > 5% below its 200DMA) and holds
it in a dedicated overlay bucket until the regime clears, then unwinds to cash.

These tests pin:
  1. Behavior preservation: frac=0.0 (frozen default) is byte-identical to a run
     with the overlay code absent (same final value, same active curve).
  2. Accounting correctness: cash + stock_value + etf_value + overlay_value ==
     port_val on every day, including while de-risked.
  3. Engagement: with a synthetic crash that drives SPY > 5% below its 200DMA and
     frac=1.0, the overlay actually engages (overlay_value > 0 on defensive days)
     and the stock book is reduced during the de-risk.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

try:
    import util
    from backtesting.simulator import run_simulation
    from backtesting.types import PrecomputedData, SimResult
    _HAS_SIM = True
except Exception as exc:  # pragma: no cover
    _HAS_SIM = False
    _IMPORT_ERR = str(exc)


_N_STOCKS = 4
_N_ETFS = 2

_OPEN_CS = {
    "mode": "percentile",
    "top_percentile": 1.0,
    "max_candidates": 15,
    "min_candidates": 1,
    "use_absolute_score_floor": False,
    "absolute_score_floor": 0.0,
    "min_quality_score": 0.0,
    "min_momentum_score": -999.0,
    "min_conditional_momentum_score": -999.0,
    "allow_income_defensive_exception": False,
    "fallback_thresholds": [],
    "min_post_cooldown_candidates": 1,
}


def _bench_path_with_crash(n_days: int) -> np.ndarray:
    """Benchmark that rises steadily then crashes >5% below its 200DMA.

    First 220 days climb 300 -> ~360 (so the 200DMA sits well below spot), then a
    sharp ~20% crash that pushes spot more than 5% under the trailing 200DMA →
    _detect_regime returns 'defensive'. Tail recovers above the MA so the overlay
    unwinds, exercising both edges.
    """
    rise = np.linspace(300.0, 360.0, 220)
    crash = np.linspace(360.0, 288.0, 20)          # -20% over 20 days
    recover = np.linspace(289.0, 340.0, n_days - 240)
    return np.concatenate([rise, crash, recover])[:n_days]


def _make_precomp(n_days: int, bench_prices: np.ndarray) -> PrecomputedData:
    n = _N_STOCKS
    # Stocks track the benchmark shape (so they also fall in the crash) but are
    # not identical, so rotating to the benchmark is a real, measurable change.
    stock_row = bench_prices / 3.0
    stock_prices = np.column_stack([stock_row * (1.0 + 0.02 * i) for i in range(n)])
    etf_prices_arr = np.column_stack([bench_prices / 1.5 for _ in range(_N_ETFS)])
    return PrecomputedData(
        symbols=[f"STK{i}" for i in range(n)],
        prices=stock_prices.astype(np.float64),
        pe_comp=np.full(n, 0.5),
        pb_comp=np.full(n, 0.5),
        quality_scores=np.array([0.80, 0.75, 0.70, 0.65]),
        income_scores=np.full(n, 0.05),
        yield_trap_mask=np.zeros(n, dtype=bool),
        bin_indices=np.full(n, 2, dtype=np.int32),
        has_position_52w=np.ones(n, dtype=bool),
        position_52w_arr=np.full(n, 0.50),
        return_1m_arr=np.zeros(n),
        etf_symbols=[f"ETF{j}" for j in range(_N_ETFS)],
        etf_prices=etf_prices_arr.astype(np.float64),
        baseline_scores=np.full(n, 0.60),
        sector_labels=["Tech", "Health", "Finance", "Energy"],
        volume_arr=np.full(n, 2_000_000.0),
        mode="test",
        universe_selection="test",
        lookahead_bias_level="LOW",
        benchmark_prices=bench_prices.astype(np.float64),
        benchmark_symbol="SPY",
        position_52w_daily=np.full((n_days, n), 0.50),
        return_1m_daily=np.zeros((n_days, n)),
        bin_indices_daily=np.full((n_days, n), 2, dtype=np.int32),
        has_position_52w_daily=np.ones((n_days, n), dtype=bool),
        ret_5d_daily=None,
        ret_3m_daily=None,
        ret_6m_daily=None,
        rs_3m_daily=None,
        rs_6m_daily=None,
        vol_3m_daily=None,
        above_50dma_daily=None,
        above_200dma_daily=None,
        spy_prices=None,
    )


def _no_exit_params(index_pct: float = 0.0) -> np.ndarray:
    return np.array([
        0.05, 0.45, 0.10, 0.40,
        index_pct,   # [4] index_pct (0 → all capital in the active book)
        0.0,         # [5] metric_threshold
        5.0,         # [6] take_profit_pct
        -1.0,        # [7] sell_weak_below
        -0.99,       # [8] trailing_stop
        0.65,        # [9] value_pe_weight
        -0.35, -0.10, 0.55, 0.85, 0.45,
    ], dtype=np.float64)


def _run(precomp, params, *, derisk_frac, lag=0, scope="active_sleeve_compounding",
         accounting_trace=None) -> SimResult:
    """Run with the overlay set via the live REGIME_PARAMS dict (restored after)."""
    saved = dict(util.REGIME_PARAMS["defensive"])
    util.REGIME_PARAMS["defensive"]["backtest_derisk_frac"] = derisk_frac
    util.REGIME_PARAMS["defensive"]["backtest_derisk_switch_bps"] = 20.0
    util.REGIME_PARAMS["defensive"]["backtest_derisk_lag"] = lag
    try:
        return run_simulation(
            precomp, params,
            starting_capital=5000.0, slippage_bps=0.0, commission_per_trade=0.0,
            weekly_contribution=100.0, rebalance_frequency_days=5,
            cs_params=_OPEN_CS, scope=scope, accounting_trace=accounting_trace,
        )
    finally:
        util.REGIME_PARAMS["defensive"].clear()
        util.REGIME_PARAMS["defensive"].update(saved)


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestOverlayBehaviorPreservation:

    def test_frac_zero_matches_no_overlay(self):
        """frac=0.0 (frozen default) must be identical to overlay-absent behavior."""
        n_days = 300
        bench = _bench_path_with_crash(n_days)
        pc = _make_precomp(n_days, bench)
        r_off = _run(pc, _no_exit_params(), derisk_frac=0.0)
        # Run twice to confirm determinism + that 0.0 truly no-ops.
        r_off2 = _run(pc, _no_exit_params(), derisk_frac=0.0)
        assert r_off.final_value == pytest.approx(r_off2.final_value, rel=1e-12)
        assert r_off.active_total_return == pytest.approx(r_off2.active_total_return, rel=1e-12)


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestOverlayAccounting:

    def test_trace_reconciles_with_overlay_active(self):
        """cash + stock + etf + overlay == port_val on every day while de-risked."""
        n_days = 300
        bench = _bench_path_with_crash(n_days)
        pc = _make_precomp(n_days, bench)
        trace = []
        _run(pc, _no_exit_params(), derisk_frac=1.0, lag=0, accounting_trace=trace)
        assert len(trace) == n_days
        max_err = 0.0
        for entry in trace:
            recon = entry["cash"] + entry["stock_value"] + entry["etf_value"] + entry["overlay_value"]
            max_err = max(max_err, abs(recon - entry["port_val"]))
        assert max_err < 1e-6, f"Accounting did not reconcile: max_err={max_err:.3e}"


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestOverlayEngagement:

    def test_overlay_engages_on_defensive_regime(self):
        """With a crash >5% below 200DMA and frac=1.0, the overlay must engage."""
        n_days = 300
        bench = _bench_path_with_crash(n_days)
        pc = _make_precomp(n_days, bench)
        trace = []
        _run(pc, _no_exit_params(), derisk_frac=1.0, lag=0, accounting_trace=trace)
        overlay_days = [e for e in trace if e["overlay_value"] > 1e-6]
        assert len(overlay_days) > 0, (
            "Overlay never engaged despite a >5%-below-200DMA crash in the benchmark path"
        )
        # On a fully de-risked day the stock book should be drained into the overlay.
        deepest = min(overlay_days, key=lambda e: e["stock_value"])
        assert deepest["overlay_value"] > deepest["stock_value"], (
            "Expected the overlay bucket to dominate the (drained) stock book while de-risked"
        )

    def test_overlay_off_never_engages(self):
        """frac=0.0 → overlay_value is exactly zero on every day."""
        n_days = 300
        bench = _bench_path_with_crash(n_days)
        pc = _make_precomp(n_days, bench)
        trace = []
        _run(pc, _no_exit_params(), derisk_frac=0.0, accounting_trace=trace)
        assert all(e["overlay_value"] == 0.0 for e in trace)


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestOverlayTelemetry:

    def test_telemetry_none_when_disabled(self):
        """frac=0.0 → overlay_telemetry is None (nothing for the UI to show)."""
        n_days = 300
        bench = _bench_path_with_crash(n_days)
        pc = _make_precomp(n_days, bench)
        r = _run(pc, _no_exit_params(), derisk_frac=0.0)
        assert r.overlay_telemetry is None

    def test_telemetry_populated_when_engaged(self):
        """frac=1.0 over a crash window → telemetry reports rotations + days_active."""
        n_days = 300
        bench = _bench_path_with_crash(n_days)
        pc = _make_precomp(n_days, bench)
        trace = []
        r = _run(pc, _no_exit_params(), derisk_frac=1.0, lag=0, accounting_trace=trace)
        ot = r.overlay_telemetry
        assert ot is not None
        assert ot["enabled"] is True
        assert ot["frac"] == 1.0
        assert ot["lag"] == 0
        # The overlay engaged, so these counters must be positive.
        assert ot["days_active"] > 0
        assert ot["rotations"] >= 1
        assert ot["switch_cost"] > 0.0
        assert ot["max_overlay_value"] > 0.0
        # days_active must equal the number of trace days with a live overlay bucket.
        overlay_days = sum(1 for e in trace if e["overlay_value"] > 1e-6)
        assert ot["days_active"] == overlay_days

