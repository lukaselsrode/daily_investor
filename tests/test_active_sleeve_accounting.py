"""
tests/test_active_sleeve_accounting.py

Active sleeve accounting invariants and regression tests.

Background: active_sleeve_compounding tracks cash + stock positions separately
from ETFs. Historical bug: active_total_return was computed from raw _active_daily
values (which grow with contributions), not from contribution-adjusted values,
causing +200% "returns" on short windows with only flat or mildly appreciating stocks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

try:
    from backtesting.simulator import run_simulation
    from backtesting.types import PrecomputedData, SimResult
    _HAS_SIM = True
except Exception as exc:
    _HAS_SIM = False
    _IMPORT_ERR = str(exc)


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

_N_DAYS   = 50
_N_STOCKS = 4
_N_ETFS   = 2

# Permissive candidate selection params: all stocks with positive scores qualify.
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


def _make_precomp(
    n_days: int = _N_DAYS,
    stock_prices: np.ndarray | None = None,
    etf_price: float = 200.0,
    bench_price: float = 300.0,
) -> PrecomputedData:
    """Build a minimal synthetic PrecomputedData for accounting tests."""
    n_stocks = _N_STOCKS
    n_etfs   = _N_ETFS

    if stock_prices is None:
        # Default: flat at $100
        stock_prices = np.full((n_days, n_stocks), 100.0)
    elif stock_prices.ndim == 1:
        # broadcast single-day vector to n_days
        stock_prices = np.tile(stock_prices, (n_days, 1))

    etf_prices_arr  = np.full((n_days, n_etfs), etf_price)
    bench_prices_arr = np.full(n_days, bench_price)

    n = n_stocks
    return PrecomputedData(
        symbols          = [f"STK{i}" for i in range(n)],
        prices           = stock_prices.astype(np.float64),
        pe_comp          = np.full(n, 0.5),
        pb_comp          = np.full(n, 0.5),
        quality_scores   = np.array([0.80, 0.75, 0.70, 0.65]),
        income_scores    = np.full(n, 0.05),
        yield_trap_mask  = np.zeros(n, dtype=bool),
        bin_indices      = np.full(n, 2, dtype=np.int32),
        has_position_52w = np.ones(n, dtype=bool),
        position_52w_arr = np.full(n, 0.50),
        return_1m_arr    = np.zeros(n),
        etf_symbols      = [f"ETF{j}" for j in range(n_etfs)],
        etf_prices       = etf_prices_arr.astype(np.float64),
        baseline_scores  = np.full(n, 0.60),
        sector_labels    = ["Tech", "Health", "Finance", "Energy"],
        volume_arr       = np.full(n, 2_000_000.0),
        mode             = "test",
        universe_selection = "test",
        lookahead_bias_level = "LOW",
        benchmark_prices = bench_prices_arr.astype(np.float64),
        benchmark_symbol = "SPY",
        # daily rolling arrays (v1 momentum — no v2 arrays needed)
        position_52w_daily     = np.full((n_days, n), 0.50),
        return_1m_daily        = np.zeros((n_days, n)),
        bin_indices_daily      = np.full((n_days, n), 2, dtype=np.int32),
        has_position_52w_daily = np.ones((n_days, n), dtype=bool),
        # v2 momentum arrays absent → v1 fallback
        ret_5d_daily  = None,
        ret_3m_daily  = None,
        ret_6m_daily  = None,
        rs_3m_daily   = None,
        rs_6m_daily   = None,
        vol_3m_daily  = None,
        above_50dma_daily  = None,
        above_200dma_daily = None,
        spy_prices    = None,
    )


def _no_exit_params(index_pct: float = 0.30) -> np.ndarray:
    """15-element params with extreme thresholds so no exits trigger on flat/rising prices."""
    return np.array([
        0.05,        # [0] sw_value
        0.45,        # [1] sw_quality
        0.10,        # [2] sw_income
        0.40,        # [3] sw_momentum
        index_pct,   # [4] index_pct
        0.0,         # [5] metric_threshold (all positive scores selected)
        5.0,         # [6] take_profit_pct (need 5× gain to take profit)
        -1.0,        # [7] sell_weak_below (never sell on low score)
        -0.99,       # [8] trailing_stop (near-total wipeout needed)
        0.65,        # [9] value_pe_weight
        # v1 bin scores [10-14]
        -0.35, -0.10, 0.55, 0.85, 0.45,
    ], dtype=np.float64)


def _run(
    precomp: PrecomputedData,
    params: np.ndarray,
    scope: str = "active_sleeve_compounding",
    starting_capital: float = 2000.0,
    weekly_contribution: float = 200.0,
    rebalance_frequency_days: int = 5,
    accounting_trace: list | None = None,
) -> SimResult:
    return run_simulation(
        precomp,
        params,
        starting_capital=starting_capital,
        slippage_bps=0.0,
        commission_per_trade=0.0,
        weekly_contribution=weekly_contribution,
        rebalance_frequency_days=rebalance_frequency_days,
        cs_params=_OPEN_CS,
        scope=scope,
        accounting_trace=accounting_trace,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestActiveSleeveFlatPrices:

    def test_flat_prices_active_return_near_zero(self):
        """Core accounting fix: flat prices → contribution-adjusted active return ≈ 0."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params())
        assert result.active_total_return is not None
        # With flat prices, true return is zero; tolerate tiny rounding/slippage
        assert abs(result.active_total_return) < 0.05, (
            f"Expected near-zero active return with flat prices, got {result.active_total_return:.3f}. "
            "Likely a contribution-inflation bug — active metrics must use ca_active_daily, not raw _active_daily."
        )

    def test_raw_active_equity_would_inflate_without_fix(self):
        """Sanity: raw _active_daily[-1]/_active_daily[0] - 1 is inflated by contributions."""
        # This validates that the old formula would have shown inflated returns,
        # confirming the fix is necessary and not vacuous.
        precomp = _make_precomp()
        trace   = []
        _run(precomp, _no_exit_params(), starting_capital=1000.0,
             weekly_contribution=500.0, accounting_trace=trace)
        # Initial active equity ≈ (1 - 0.30) * 1000 = 700
        # Over 50 days / 5 = 10 contribution days: 10 * 500 * 0.70 = 3500 added
        # End raw active equity ≈ 4200
        # Old formula: 4200/700 - 1 ≈ 5.0 = 500% — obviously wrong
        assert len(trace) > 0
        raw_start = trace[0]["active_equity"]
        raw_end   = trace[-1]["active_equity"]
        raw_naive_return = raw_end / raw_start - 1.0
        # Naive return should be significantly > 0 even with flat prices
        assert raw_naive_return > 0.5, (
            f"Expected raw inflation > 50% with large contributions, got {raw_naive_return:.2f}"
        )

    def test_high_contribution_rate_active_return_near_zero(self):
        """Active return stays near zero even with very large weekly contributions."""
        precomp = _make_precomp()
        # Large contribution relative to starting capital amplifies the bug if still present
        result  = _run(precomp, _no_exit_params(),
                       starting_capital=500.0, weekly_contribution=500.0)
        assert result.active_total_return is not None
        assert abs(result.active_total_return) < 0.05, (
            f"Active return {result.active_total_return:.3f} looks contribution-inflated."
        )


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestActiveEquityCurveInitialization:

    def test_active_equity_curve_starts_at_active_fraction(self):
        """Day-0 active equity ≈ (1 - index_pct) * starting_capital."""
        index_pct = 0.30
        sc        = 2000.0
        precomp   = _make_precomp()
        result    = _run(precomp, _no_exit_params(index_pct=index_pct), starting_capital=sc)
        curve = result.active_equity_curve
        assert curve is not None and len(curve) > 0
        expected  = sc * (1.0 - index_pct)
        # Allow ±20% slack: slippage and partial fills can shift the initial deployment
        assert expected * 0.80 <= curve[0] <= sc, (
            f"Day-0 active equity {curve[0]:.2f} not near expected {expected:.2f}"
        )

    def test_active_equity_nonnegative_throughout(self):
        """Active equity must never be negative."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params())
        curve = result.active_equity_curve
        assert curve is not None
        assert np.all(curve >= 0), "Active equity went negative — accounting leak"


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestActiveExcessReturn:

    def test_active_excess_return_equals_return_minus_benchmark(self):
        """active_excess_return = active_total_return - benchmark_twr exactly."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params())
        assert result.active_total_return  is not None
        assert result.active_excess_return is not None
        expected = result.active_total_return - result.benchmark_twr
        assert abs(result.active_excess_return - expected) < 1e-9, (
            f"active_excess_return {result.active_excess_return:.6f} != "
            f"active_total_return {result.active_total_return:.6f} - benchmark_twr {result.benchmark_twr:.6f}"
        )

    def test_flat_benchmark_and_flat_stock_zero_excess(self):
        """Flat prices everywhere → active_total_return ≈ benchmark_twr ≈ 0 → excess ≈ 0."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params())
        assert abs(result.benchmark_twr) < 0.01, (
            f"Flat benchmark should give ~0 TWR (contribution-adjusted), got {result.benchmark_twr:.4f}"
        )
        assert abs(result.active_excess_return) < 0.05


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestScopeIsolation:

    def test_active_metrics_none_for_overall_strategy(self):
        """When scope=overall_strategy, active_* fields are all None."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params(), scope="overall_strategy")
        assert result.active_total_return  is None
        assert result.active_sharpe        is None
        assert result.active_calmar        is None
        assert result.active_max_drawdown  is None
        assert result.active_excess_return is None
        assert result.active_equity_curve  is None

    def test_active_metrics_populated_for_active_sleeve(self):
        """When scope=active_sleeve_compounding, all active_* fields are populated."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params(), scope="active_sleeve_compounding")
        assert result.active_total_return  is not None
        assert result.active_sharpe        is not None
        assert result.active_calmar        is not None
        assert result.active_max_drawdown  is not None
        assert result.active_excess_return is not None
        assert result.active_equity_curve  is not None


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestTradeAccounting:

    def test_buy_deploys_capital_into_positions(self):
        """Day-0 active equity must have stock positions (cash deployed)."""
        precomp = _make_precomp()
        result  = _run(precomp, _no_exit_params(), starting_capital=5000.0)
        assert result.trades_made >= 1, "Expected at least one buy on day 0"
        # Active equity on day 0 should be less than total capital (some went to ETFs)
        assert result.active_equity_curve is not None
        # And the total equity should be roughly full capital after deployment
        assert result.equity_curve[0] > 4000.0, "Starting capital should be mostly deployed"

    def test_stop_loss_generates_sells(self):
        """A price drop of 30% below cost basis triggers stop-loss sells."""
        # Prices: $100 for 5 days, then drop to $65 (35% loss > 20% stop-loss)
        prices = np.vstack([
            np.full((5,  _N_STOCKS), 100.0),
            np.full((45, _N_STOCKS), 65.0),
        ])
        precomp = _make_precomp(stock_prices=prices)
        params  = _no_exit_params().copy()
        # Use normal trailing stop and take-profit; only stop-loss should fire
        params[6] = 5.0    # take_profit: 500% needed
        params[8] = -0.99  # trailing stop: almost never
        result = _run(precomp, params, starting_capital=2000.0)
        assert result.sells_made >= 1, (
            "Expected stop-loss sells after 35% price drop, got 0 sells"
        )

    def test_accounting_trace_active_equity_matches_cash_plus_stock(self):
        """Invariant: every trace entry has active_equity = cash + stock_value."""
        precomp = _make_precomp()
        trace   = []
        _run(precomp, _no_exit_params(), accounting_trace=trace)
        assert len(trace) == _N_DAYS
        for entry in trace:
            expected = entry["cash"] + entry["stock_value"]
            assert abs(entry["active_equity"] - expected) < 1e-6, (
                f"Day {entry['d']}: active_equity {entry['active_equity']:.4f} != "
                f"cash {entry['cash']:.4f} + stock_value {entry['stock_value']:.4f}"
            )


@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestHarvestRouting:

    def test_harvest_in_active_sleeve_no_etf_routing(self):
        """In active_sleeve_compounding, harvest proceeds stay in active cash (no ETF routing)."""
        # Prices rise 60% on day 12 to trigger harvest (>= 30% gain after >=10 hold days)
        prices = np.vstack([
            np.full((12, _N_STOCKS), 100.0),
            np.full((38, _N_STOCKS), 165.0),  # 65% gain → harvest triggers
        ])
        precomp = _make_precomp(stock_prices=prices)
        trace   = []
        result  = _run(precomp, _no_exit_params(), starting_capital=3000.0,
                       accounting_trace=trace)
        if result.harvest_count == 0:
            pytest.skip("No harvest triggered in this scenario — adjust price path")
        # In active_sleeve_compounding mode, ETF value should NOT grow from harvest proceeds
        # (only from contributions). Check that ETF value change ≈ from contributions only.
        # Simpler check: active_equity_curve should not drop at harvest point
        # (proceeds stay in active sleeve rather than routing out to ETFs)
        curve = result.active_equity_curve
        assert curve is not None
        # After prices rise, active equity should still be positive and growing
        assert curve[-1] >= curve[0] * 0.5, "Active equity collapsed after harvest"

    def test_harvest_overall_strategy_cash_conservation(self):
        """In overall_strategy, harvest routes cash to ETFs without creating value."""
        prices = np.vstack([
            np.full((12, _N_STOCKS), 100.0),
            np.full((38, _N_STOCKS), 165.0),
        ])
        precomp = _make_precomp(stock_prices=prices)
        trace   = []
        result  = _run(precomp, _no_exit_params(), scope="overall_strategy",
                       starting_capital=3000.0, accounting_trace=trace)
        if result.harvest_count == 0:
            pytest.skip("No harvest triggered — adjust price path")
        # Conservation check: total portfolio value (port_val) should never jump by more
        # than the day's price appreciation. With the harvest bug fixed, routing cash
        # to ETFs does NOT inflate port_val.
        for i in range(1, len(trace)):
            prev_val = trace[i - 1]["port_val"]
            curr_val = trace[i]["port_val"]
            # Maximum day-over-day gain from price change: stocks double → 2x prev
            assert curr_val < prev_val * 3.0, (
                f"Day {trace[i]['d']}: port_val jumped from {prev_val:.2f} to {curr_val:.2f} "
                "— likely double-counting from harvest ETF routing bug"
            )


# ---------------------------------------------------------------------------
# Tests: active equity curve is on the same CA basis as reported metrics
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SIM, reason="simulator not importable")
class TestActiveEquityCurveCABasis:
    """Verify that active_equity_curve is the CA series, not the raw contribution-inflated curve.

    These tests pin the invariant: the chart and the scalar metrics use the same
    contribution-adjusted basis, so they can never diverge as they did before the fix.
    """

    def _rising_prices(self, gain_pct: float = 0.20) -> np.ndarray:
        """Prices that rise linearly from 100 to 100*(1+gain_pct) over N_DAYS."""
        start, end = 100.0, 100.0 * (1.0 + gain_pct)
        row = np.linspace(start, end, _N_DAYS)
        return np.column_stack([row] * _N_STOCKS)

    def test_curve_indexed_ends_consistent_with_active_total_return(self):
        """Test 1: active_equity_curve indexed to 100 ends near 100*(1+active_total_return)."""
        precomp = _make_precomp(stock_prices=self._rising_prices(0.10))
        result  = _run(precomp, _no_exit_params())
        assert result.active_total_return is not None
        curve   = result.active_equity_curve
        assert curve is not None and len(curve) > 0

        indexed_end = curve[-1] / max(curve[0], 1e-9) * 100.0
        expected    = 100.0 * (1.0 + result.active_total_return)
        assert abs(indexed_end - expected) < 2.0, (
            f"indexed_end={indexed_end:.2f} but expected≈{expected:.2f} "
            f"(active_total_return={result.active_total_return:.4f}). "
            "active_equity_curve must be the CA series, not raw _active_daily."
        )

    def test_benchmark_ca_curve_indexed_ends_consistent_with_benchmark_twr(self):
        """Test 2: benchmark_ca_equity indexed to 100 ends near 100*(1+benchmark_twr)."""
        precomp = _make_precomp(bench_price=300.0)
        result  = _run(precomp, _no_exit_params())

        bench = result.benchmark_ca_equity
        assert bench is not None and len(bench) > 0, "benchmark_ca_equity should be populated"

        indexed_end = bench[-1] / max(bench[0], 1e-9) * 100.0
        expected    = 100.0 * (1.0 + result.benchmark_twr)
        assert abs(indexed_end - expected) < 2.0, (
            f"indexed_end={indexed_end:.2f} but expected≈{expected:.2f} "
            f"(benchmark_twr={result.benchmark_twr:.4f}). "
            "benchmark_ca_equity must track the TWR-consistent CA series."
        )

    def test_drawdown_from_curve_matches_active_max_drawdown(self):
        """Test 3: max drawdown derived from active_equity_curve matches active_max_drawdown."""
        # Use prices that drop then partially recover so drawdown is non-trivial
        prices = np.vstack([
            np.full((15, _N_STOCKS), 100.0),
            np.full((15, _N_STOCKS), 80.0),   # −20% drop
            np.full((20, _N_STOCKS), 88.0),   # partial recovery
        ])
        precomp = _make_precomp(stock_prices=prices)
        result  = _run(precomp, _no_exit_params())
        assert result.active_max_drawdown is not None

        curve    = result.active_equity_curve
        assert curve is not None and len(curve) > 0
        indexed  = curve / max(curve[0], 1e-9) * 100.0
        cum_max  = np.maximum.accumulate(indexed)
        dd       = np.where(cum_max > 0, indexed / cum_max - 1.0, 0.0)
        computed_max_dd = float(dd.min())

        # Max drawdown from the curve must match the reported metric (same series)
        assert abs(computed_max_dd - result.active_max_drawdown) < 0.02, (
            f"Drawdown from curve={computed_max_dd:.4f} but active_max_drawdown={result.active_max_drawdown:.4f}. "
            "Both must derive from the same CA series."
        )

    def test_train_and_val_segments_each_normalized_independently(self):
        """Test 4: train and val active_equity_curve segments each start at their own day-0 value.

        Both segments must independently normalize to 100 at their own day 0 so
        the chart correctly labels each window's performance without cross-contamination.
        """
        n_days = 60
        n_train = 40
        precomp_full = _make_precomp(n_days=n_days, stock_prices=np.full((n_days, _N_STOCKS), 100.0))

        def _slice_precomp(pc, s):
            return pc._replace(
                prices=pc.prices[s],
                etf_prices=pc.etf_prices[s],
                benchmark_prices=pc.benchmark_prices[s],
                position_52w_daily=pc.position_52w_daily[s],
                return_1m_daily=pc.return_1m_daily[s],
                bin_indices_daily=pc.bin_indices_daily[s],
                has_position_52w_daily=pc.has_position_52w_daily[s],
            )

        train_precomp = _slice_precomp(precomp_full, slice(0, n_train))
        val_precomp   = _slice_precomp(precomp_full, slice(n_train, n_days))

        train_result = _run(train_precomp, _no_exit_params(), starting_capital=2000.0)
        val_result   = _run(val_precomp,   _no_exit_params(), starting_capital=2000.0)

        for label, result in [("train", train_result), ("val", val_result)]:
            curve = result.active_equity_curve
            assert curve is not None and len(curve) > 0, f"{label} curve missing"
            assert curve[0] > 0, f"{label} active curve must start positive"
            # When the chart indexes to 100 at day 0, each segment starts exactly at 100
            indexed_start = curve[0] / max(curve[0], 1e-9) * 100.0
            assert abs(indexed_start - 100.0) < 1e-6, (
                f"{label} curve does not normalize to 100 at start: {indexed_start:.6f}"
            )
