"""
tests.py — Unit tests for pure trading logic functions.

Does not require Robinhood API access. Run with:
    cd src && python tests.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# ---------------------------------------------------------------------------
# Import pure functions
# ---------------------------------------------------------------------------

import numpy as np

from source_data import _position_52w, get_momentum_score

try:
    from main import can_buy_symbol, evaluate_sell_candidate, get_position_value
    from util import BACKTEST_PARAMS, MOMENTUM_PARAMS, RISK_LIMITS, SELL_RULES
    _HAS_MAIN = True
except Exception as e:
    print(f"Warning: could not import main.py ({e}) — main-dependent tests will be skipped")
    _HAS_MAIN = False

try:
    from backtest import (
        BacktestReport,
        PrecomputedData,
        SimResult,
        compute_performance_metrics,
        score_stocks_at_day,
        select_backtest_universe,
        split_price_window,
    )
    _HAS_BACKTEST = True
except Exception as e:
    print(f"Warning: could not import backtest.py ({e}) — backtest tests will be skipped")
    _HAS_BACKTEST = False

try:
    from tuner import (
        should_apply_tuned_config,
        validate_tuned_params,
        validate_llm_review_response,
        merge_llm_recommendation_with_config,
    )
    _HAS_TUNER = True
except Exception as e:
    print(f"Warning: could not import tuner.py ({e}) — tuner tests will be skipped")
    _HAS_TUNER = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert(condition: bool, msg: str = "") -> None:
    if not condition:
        raise AssertionError(msg or "assertion failed")


def _make_metrics(
    value_metric: float = 1.0,
    quality_score: float = 0.5,
    yield_trap_flag: bool = False,
) -> pd.Series:
    return pd.Series({
        "value_metric":    value_metric,
        "quality_score":   quality_score,
        "yield_trap_flag": yield_trap_flag,
    })


def _make_holding(
    percent_change: float = 0.0,
    avg_buy: float = 100.0,
    price: float = 100.0,
) -> dict:
    # Robinhood returns percent_change as a percentage string e.g. "-15.3" for -15.3%
    return {
        "percent_change":    str(percent_change),
        "average_buy_price": str(avg_buy),
        "price":             str(price),
        "quantity":          "10",
    }


def _make_holdings(symbol: str = "AAPL", equity: float = 0.0) -> dict:
    return {symbol: {"equity": str(equity), "quantity": "1", "average_buy_price": "100"}}


# ---------------------------------------------------------------------------
# position_52w tests
# ---------------------------------------------------------------------------

def test_position_52w_normal():
    result = _position_52w(55.0, 40.0, 80.0)
    expected = (55.0 - 40.0) / (80.0 - 40.0)  # 0.375
    _assert(abs(result - expected) < 1e-9, f"expected {expected}, got {result}")


def test_position_52w_missing_price():
    _assert(_position_52w(None, 40.0, 80.0) is None)


def test_position_52w_missing_low():
    _assert(_position_52w(55.0, None, 80.0) is None)


def test_position_52w_missing_high():
    _assert(_position_52w(55.0, 40.0, None) is None)


def test_position_52w_high_equals_low():
    _assert(_position_52w(55.0, 80.0, 80.0) is None)


def test_position_52w_high_less_than_low():
    _assert(_position_52w(55.0, 80.0, 40.0) is None)


def test_position_52w_clamp_above_one():
    result = _position_52w(100.0, 40.0, 80.0)   # current > high
    _assert(result == 1.0, f"expected 1.0, got {result}")


def test_position_52w_clamp_below_zero():
    result = _position_52w(20.0, 40.0, 80.0)    # current < low
    _assert(result == 0.0, f"expected 0.0, got {result}")


def test_position_52w_at_low():
    result = _position_52w(40.0, 40.0, 80.0)
    _assert(result == 0.0, f"expected 0.0, got {result}")


def test_position_52w_at_high():
    result = _position_52w(80.0, 40.0, 80.0)
    _assert(result == 1.0, f"expected 1.0, got {result}")


# ---------------------------------------------------------------------------
# momentum_score tests — use live config values so auto-tune doesn't break them
# ---------------------------------------------------------------------------

def test_momentum_score_none():
    _assert(get_momentum_score(None) == 0.0)


def _mbin(idx: int) -> float:
    """Return the bin score rounded to match get_momentum_score's output precision."""
    raw = MOMENTUM_PARAMS["position_bin_scores"][idx] if _HAS_MAIN else [-0.4, 0.1, 0.3, 0.5, 0.2][idx]
    return round(raw, 3)


def test_momentum_score_below_015():
    s = _mbin(0)
    _assert(get_momentum_score(0.0)  == s)
    _assert(get_momentum_score(0.10) == s)
    _assert(get_momentum_score(0.14) == s)


def test_momentum_score_015_to_035():
    s = _mbin(1)
    _assert(get_momentum_score(0.15) == s)
    _assert(get_momentum_score(0.25) == s)
    _assert(get_momentum_score(0.34) == s)


def test_momentum_score_035_to_075():
    s = _mbin(2)
    _assert(get_momentum_score(0.35) == s)
    _assert(get_momentum_score(0.50) == s)
    _assert(get_momentum_score(0.74) == s)


def test_momentum_score_075_to_095():
    s = _mbin(3)
    _assert(get_momentum_score(0.75) == s)
    _assert(get_momentum_score(0.90) == s)
    _assert(get_momentum_score(0.94) == s)  # 0.95 boundary is exclusive


def test_momentum_score_above_095():
    s = _mbin(4)
    _assert(get_momentum_score(0.95) == s)  # at boundary → bin 4
    _assert(get_momentum_score(0.96) == s)
    _assert(get_momentum_score(1.0)  == s)


# ---------------------------------------------------------------------------
# sell decision engine tests
# ---------------------------------------------------------------------------

def test_sell_hard_stop_loss():
    if not _HAS_MAIN:
        return
    # Use a loss 1pp below the configured stop_loss_pct so the test doesn't need updating
    stop_pct_as_pct = (SELL_RULES["stop_loss_pct"] - 0.01) * 100  # e.g. -0.21 → -21.0
    holding  = _make_holding(percent_change=stop_pct_as_pct)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics())
    _assert(decision["should_sell"],         "should_sell must be True")
    _assert(decision["severity"] == "hard",  f"expected hard, got {decision['severity']}")
    _assert("stop loss" in decision["reason"], decision["reason"])


def test_sell_hard_yield_trap():
    if not _HAS_MAIN:
        return
    holding  = _make_holding(percent_change=0.0)
    metrics  = _make_metrics(value_metric=0.10, yield_trap_flag=True)
    decision = evaluate_sell_candidate("TEST", holding, metrics)
    _assert(decision["should_sell"],        "should_sell must be True")
    _assert(decision["severity"] == "hard", f"expected hard, got {decision['severity']}")
    _assert("yield trap" in decision["reason"], decision["reason"])


def test_sell_hard_quality_floor():
    if not _HAS_MAIN:
        return
    floor    = SELL_RULES["sell_low_quality_below"]   # -0.25
    holding  = _make_holding(percent_change=0.0)
    metrics  = _make_metrics(quality_score=floor - 0.1)
    decision = evaluate_sell_candidate("TEST", holding, metrics)
    _assert(decision["should_sell"],        "should_sell must be True")
    _assert(decision["severity"] == "hard", f"expected hard, got {decision['severity']}")


def test_sell_soft_take_profit():
    if not _HAS_MAIN:
        return
    # Use a gain 1pp above the configured take_profit_pct so the test tracks config changes
    tp_pct_as_pct = (SELL_RULES["take_profit_pct"] + 0.01) * 100  # e.g. 0.4171 → 41.71
    holding  = _make_holding(percent_change=tp_pct_as_pct)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics())
    _assert(decision["should_sell"],        "should_sell must be True")
    _assert(decision["severity"] == "soft", f"expected soft, got {decision['severity']}")
    _assert("take profit" in decision["reason"], decision["reason"])


def test_sell_soft_weak_value():
    if not _HAS_MAIN:
        return
    below    = SELL_RULES["sell_weak_value_below"] - 0.05   # below threshold
    holding  = _make_holding(percent_change=0.0)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics(value_metric=below))
    _assert(decision["should_sell"],        "should_sell must be True")
    _assert(decision["severity"] == "soft", f"expected soft, got {decision['severity']}")


def test_no_sell_healthy_holding():
    if not _HAS_MAIN:
        return
    holding  = _make_holding(percent_change=5.0)   # +5%, no issues
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics(value_metric=1.0))
    _assert(not decision["should_sell"], f"should not sell; reason={decision['reason']}")


def test_sell_no_metrics_no_crash():
    if not _HAS_MAIN:
        return
    holding  = _make_holding(percent_change=0.0)
    decision = evaluate_sell_candidate("TEST", holding, None)
    # With no metrics, none of the metric-based rules fire; should not sell on 0% change
    _assert(not decision["should_sell"])


# ---------------------------------------------------------------------------
# position cap tests
# ---------------------------------------------------------------------------

def test_position_value_from_holdings():
    if not _HAS_MAIN:
        return
    holdings = _make_holdings("AAPL", equity=430.0)
    _assert(get_position_value("AAPL", holdings) == 430.0)
    _assert(get_position_value("MSFT", holdings) == 0.0)


def test_can_buy_within_position_cap():
    if not _HAS_MAIN:
        return
    # portfolio=$10k, max_single=5%=$500, current_pos=$200, propose $100 → ok
    holdings = _make_holdings("AAPL", equity=200.0)
    ok, reason, adj = can_buy_symbol("AAPL", 100.0, holdings, None, 10_000.0, 5_000.0)
    _assert(ok, f"expected ok, got: {reason}")
    _assert(abs(adj - 100.0) < 0.01, f"expected adj=100, got {adj}")


def test_can_buy_reduced_by_position_cap():
    if not _HAS_MAIN:
        return
    # portfolio=$10k, max_single=5%=$500, current_pos=$450, propose $200 → capped to $50
    holdings = _make_holdings("AAPL", equity=450.0)
    ok, reason, adj = can_buy_symbol("AAPL", 200.0, holdings, None, 10_000.0, 5_000.0)
    _assert(ok, f"expected ok after reduction, got: {reason}")
    _assert(abs(adj - 50.0) < 0.01, f"expected adj=50, got {adj}")


def test_can_buy_blocked_by_position_cap():
    if not _HAS_MAIN:
        return
    # portfolio=$10k, max_single=5%=$500, current_pos=$500 → no room
    holdings = _make_holdings("AAPL", equity=500.0)
    ok, reason, adj = can_buy_symbol("AAPL", 100.0, holdings, None, 10_000.0, 5_000.0)
    _assert(not ok, f"expected blocked, got ok with adj={adj}")


def test_can_buy_skipped_below_min_order():
    if not _HAS_MAIN:
        return
    min_order = RISK_LIMITS["min_order_amount"]   # 5.00
    # room=$2 (equity $498 / cap $500), proposal $100 capped to $2 < min_order=$5 → skip
    holdings = _make_holdings("AAPL", equity=498.0)
    ok, reason, adj = can_buy_symbol("AAPL", 100.0, holdings, None, 10_000.0, 5_000.0)
    _assert(not ok, f"expected skip (allocation below min_order), got ok adj={adj}")
    _assert(adj == 0.0, f"expected adj=0.0, got {adj}")


def test_can_buy_blocked_when_cash_below_min_order():
    if not _HAS_MAIN:
        return
    min_order = RISK_LIMITS["min_order_amount"]   # 5.00
    # room=$2, cash=$3 (< min_order) → blocked
    holdings = _make_holdings("AAPL", equity=498.0)
    ok, reason, adj = can_buy_symbol("AAPL", 100.0, holdings, None, 10_000.0, 3.0)
    _assert(not ok, f"expected blocked (cash < min_order), got ok with adj={adj}")


def test_can_buy_order_size_capped():
    if not _HAS_MAIN:
        return
    # max_order_pct=10% of cash=$1000. propose $200 when cash=$1000 → capped to $100
    holdings = _make_holdings("AAPL", equity=0.0)
    ok, reason, adj = can_buy_symbol("AAPL", 200.0, holdings, None, 10_000.0, 1_000.0)
    max_order = 1_000.0 * RISK_LIMITS["max_order_pct_of_cash"]
    _assert(ok, f"expected ok after order cap, got: {reason}")
    _assert(abs(adj - max_order) < 0.01, f"expected adj={max_order}, got {adj}")


# ---------------------------------------------------------------------------
# Sector cap tests (use agg_df with sector info)
# ---------------------------------------------------------------------------

def _make_agg_df_multi(rows: list[dict]) -> pd.DataFrame:
    base = {"volume": 1_000_000}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_sector_cap_reduced():
    if not _HAS_MAIN:
        return
    # portfolio=$100k so per-stock cap (5%=$5k) won't fire
    # Technology cap = 25% of $100k = $25k
    # MSFT holds $23900 (Technology), AAPL holds $100 (Technology) → total=$24000, room=$1000
    # propose $1500 for AAPL → reduced to $1000
    agg_df = _make_agg_df_multi([
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
    ])
    holdings = {
        "AAPL": {"equity": "100",   "quantity": "1"},
        "MSFT": {"equity": "23900", "quantity": "100"},
    }
    ok, reason, adj = can_buy_symbol("AAPL", 1_500.0, holdings, agg_df, 100_000.0, 50_000.0)
    _assert(ok, f"expected ok after sector reduction, got: {reason}")
    _assert(abs(adj - 1000.0) < 0.01, f"expected adj=1000, got {adj}")


def test_sector_cap_blocked():
    if not _HAS_MAIN:
        return
    # Technology already at cap ($25k / $25k)
    agg_df = _make_agg_df_multi([
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
    ])
    holdings = {
        "AAPL": {"equity": "100",   "quantity": "1"},
        "MSFT": {"equity": "24900", "quantity": "100"},
    }
    ok, reason, adj = can_buy_symbol("AAPL", 200.0, holdings, agg_df, 100_000.0, 50_000.0)
    _assert(not ok, f"expected blocked by sector cap, got ok with adj={adj}")


def test_liquidity_gate():
    if not _HAS_MAIN:
        return
    min_vol = RISK_LIMITS["min_liquidity_volume"]
    agg_df  = _make_agg_df_multi([{"symbol": "ILLQ", "sector": "Technology", "volume": min_vol - 1}])
    ok, reason, _ = can_buy_symbol("ILLQ", 100.0, {}, agg_df, 10_000.0, 5_000.0)
    _assert(not ok, f"expected blocked by liquidity gate, got ok")
    _assert("volume" in reason.lower(), reason)


# ---------------------------------------------------------------------------
# exit_type classification tests
# ---------------------------------------------------------------------------

def test_exit_type_hard_sell_is_failure():
    if not _HAS_MAIN:
        return
    stop_pct_as_pct = (SELL_RULES["stop_loss_pct"] - 0.01) * 100
    holding  = _make_holding(percent_change=stop_pct_as_pct)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics())
    _assert(decision["exit_type"] == "failure_exit", f"expected failure_exit, got {decision['exit_type']}")


def test_exit_type_take_profit_is_harvest():
    if not _HAS_MAIN:
        return
    tp_pct_as_pct = (SELL_RULES["take_profit_pct"] + 0.01) * 100
    holding  = _make_holding(percent_change=tp_pct_as_pct)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics(value_metric=0.5))
    _assert(decision["exit_type"] == "harvest_exit", f"expected harvest_exit, got {decision['exit_type']}")


def test_exit_type_weak_value_is_thesis():
    if not _HAS_MAIN:
        return
    below    = SELL_RULES["sell_weak_value_below"] - 0.05
    holding  = _make_holding(percent_change=0.0)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics(value_metric=below))
    _assert(decision["exit_type"] == "thesis_exit", f"expected thesis_exit, got {decision['exit_type']}")


def test_exit_type_no_sell_is_none():
    if not _HAS_MAIN:
        return
    holding  = _make_holding(percent_change=5.0)
    decision = evaluate_sell_candidate("TEST", holding, _make_metrics(value_metric=1.0))
    _assert(decision["exit_type"] is None, f"expected None, got {decision['exit_type']}")


# ---------------------------------------------------------------------------
# split_price_window / compute_performance_metrics tests
# ---------------------------------------------------------------------------

def test_split_price_window_70_30():
    if not _HAS_BACKTEST:
        return
    train_sl, val_sl = split_price_window(100, 0.70)
    _assert(train_sl == slice(0, 70), f"expected train 0:70, got {train_sl}")
    _assert(val_sl == slice(70, 100), f"expected val 70:100, got {val_sl}")


def test_split_price_window_full():
    if not _HAS_BACKTEST:
        return
    train_sl, val_sl = split_price_window(50, 1.0)
    _assert(train_sl.stop == 50)
    _assert(val_sl.start == 50)
    _assert(val_sl.stop == 50)


def test_compute_performance_metrics_flat():
    if not _HAS_BACKTEST:
        return
    vals = np.ones(100) * 10_000.0
    m = compute_performance_metrics(vals)
    _assert(abs(m["total_return"]) < 1e-9, f"flat series should have 0 return, got {m['total_return']}")
    _assert(m["max_drawdown"] == 0.0, f"flat series should have 0 drawdown, got {m['max_drawdown']}")


def test_compute_performance_metrics_growing():
    if not _HAS_BACKTEST:
        return
    vals = np.linspace(10_000.0, 12_000.0, 252)
    m = compute_performance_metrics(vals)
    _assert(abs(m["total_return"] - 0.2) < 1e-6, f"expected +20% return, got {m['total_return']:.4f}")
    _assert(m["max_drawdown"] >= -0.01, f"monotone growth should have ~0 drawdown, got {m['max_drawdown']}")
    _assert(m["sharpe"] > 0, f"positive-return series should have positive Sharpe, got {m['sharpe']}")


def test_compute_performance_metrics_drawdown():
    if not _HAS_BACKTEST:
        return
    # Goes up to 11000 then drops to 8800 = -20% drawdown
    vals = np.array([10_000.0] * 50 + [11_000.0] * 50 + [8_800.0] * 50, dtype=float)
    m = compute_performance_metrics(vals)
    _assert(m["max_drawdown"] <= -0.19, f"expected ~-20% drawdown, got {m['max_drawdown']:.3f}")


# ---------------------------------------------------------------------------
# select_backtest_universe tests
# ---------------------------------------------------------------------------

def _make_agg_df_for_backtest(n: int = 50) -> pd.DataFrame:
    """Build a minimal agg_df with volume, value_metric, sector."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(n)],
        "volume": rng.uniform(100_000, 2_000_000, n),
        "value_metric": rng.uniform(0, 2, n),
        "sector": rng.choice(["Technology", "Healthcare", "Financials", "Energy"], n),
    })


def test_universe_liquid_sample_respects_max():
    if not _HAS_BACKTEST:
        return
    df = _make_agg_df_for_backtest(100)
    selected, bias = select_backtest_universe(df, "liquid_universe_sanity_test", "liquid_sample", 30, 100_000, 42)
    _assert(len(selected) <= 30, f"expected <= 30 symbols, got {len(selected)}")
    _assert(bias == "MEDIUM", f"expected MEDIUM bias, got {bias}")


def test_universe_top_scores_bias_high():
    if not _HAS_BACKTEST:
        return
    df = _make_agg_df_for_backtest(50)
    selected, bias = select_backtest_universe(df, "current_universe_stress_test", "top_current_scores", 20, 100_000, 42)
    _assert(bias == "HIGH", f"expected HIGH bias for top_current_scores, got {bias}")
    # Should be sorted descending by value_metric
    metrics = selected["value_metric"].values
    _assert(all(metrics[i] >= metrics[i+1] for i in range(len(metrics)-1)),
            "top_current_scores should be sorted descending")


def test_universe_walk_forward_bias_low():
    if not _HAS_BACKTEST:
        return
    df = _make_agg_df_for_backtest(50)
    _, bias = select_backtest_universe(df, "walk_forward_price_only_test", "liquid_all", 50, 100_000, 42)
    _assert(bias == "LOW", f"expected LOW bias for walk_forward, got {bias}")


def _make_minimal_precomp(n_days: int = 30, n_stocks: int = 5) -> "PrecomputedData":
    """Build a minimal PrecomputedData for unit tests that need score_stocks_at_day."""
    if not _HAS_BACKTEST:
        return None
    rng = np.random.default_rng(0)
    prices = rng.uniform(10, 200, (n_days, n_stocks)).astype(np.float64)
    zeros = np.zeros(n_stocks)
    # Compute rolling daily features
    from backtest import MOMENTUM_PARAMS as _mp
    boundaries = np.array(_mp["position_bin_boundaries"])
    pos_daily = np.full((n_days, n_stocks), np.nan)
    ret_daily = np.full((n_days, n_stocks), np.nan)
    bin_daily = np.zeros((n_days, n_stocks), dtype=np.int32)
    for d in range(n_days):
        ws = max(0, d - 251)
        lo = prices[ws:d+1].min(axis=0)
        hi = prices[ws:d+1].max(axis=0)
        r = hi - lo
        pos_daily[d] = np.where(r > 0, np.clip((prices[d] - lo) / r, 0, 1), np.nan)
        if d >= 21:
            ret_daily[d] = prices[d] / prices[d - 21] - 1.0
        vp = np.where(np.isfinite(pos_daily[d]), pos_daily[d], 0.5)
        bin_daily[d] = np.searchsorted(boundaries, vp, side="right").astype(np.int32)
    return PrecomputedData(
        symbols=[f"S{i}" for i in range(n_stocks)],
        prices=prices,
        pe_comp=zeros.copy(), pb_comp=zeros.copy(),
        quality_scores=zeros.copy(), income_scores=zeros.copy(),
        yield_trap_mask=np.zeros(n_stocks, dtype=bool),
        bin_indices=bin_daily[0], has_position_52w=np.isfinite(pos_daily[0]),
        position_52w_arr=pos_daily[0], return_1m_arr=ret_daily[0],
        etf_symbols=[], etf_prices=np.zeros((n_days, 0)),
        baseline_scores=zeros.copy(),
        sector_labels=["Unknown"] * n_stocks, volume_arr=np.ones(n_stocks) * 1e6,
        mode="liquid_universe_sanity_test", universe_selection="liquid_sample",
        lookahead_bias_level="MEDIUM",
        benchmark_prices=np.ones(n_days) * 100.0, benchmark_symbol="SPY",
        position_52w_daily=pos_daily, return_1m_daily=ret_daily,
        bin_indices_daily=bin_daily, has_position_52w_daily=np.isfinite(pos_daily),
    )


def test_score_stocks_at_day_uses_rolling_features():
    """Verify that scores differ between day 0 and day 29 (prices differ → momentum differs)."""
    if not _HAS_BACKTEST:
        return
    precomp = _make_minimal_precomp(n_days=30, n_stocks=5)
    params = np.array([0.25, 0.35, 0.10, 0.30,  # sw
                       0.70, 0.80, 0.40, 0.25, -0.15, 0.60,  # other
                       -0.3, 0.1, 0.3, 0.6, 0.2])
    scores_d0  = score_stocks_at_day(precomp, params, 0)
    scores_d29 = score_stocks_at_day(precomp, params, 29)
    # Prices are random — 52w position will differ between day 0 and day 29
    _assert(not np.allclose(scores_d0, scores_d29),
            "scores at day 0 and day 29 should differ (rolling momentum changes with price path)")


def test_universe_min_volume_filter():
    if not _HAS_BACKTEST:
        return
    df = _make_agg_df_for_backtest(50)
    df["volume"] = 100.0  # all below min_volume
    df.loc[0, "volume"] = 2_000_000  # only one passes
    selected, _ = select_backtest_universe(df, "liquid_universe_sanity_test", "liquid_all", 50, 500_000, 42)
    _assert(len(selected) >= 1, "at least one symbol should pass the volume filter")


# ---------------------------------------------------------------------------
# validate_tuned_params tests
# ---------------------------------------------------------------------------

def _make_sim_result(**kwargs) -> "SimResult":
    defaults = dict(
        final_value=10000.0, total_return=0.1, sharpe=0.5,
        calmar=0.8, max_drawdown=-0.1, trades_made=30,
    )
    defaults.update(kwargs)
    return SimResult(**defaults)


def _make_backtest_report(val_result=None, excess_return=0.05, val_bench_return=0.05) -> "BacktestReport":
    tr = _make_sim_result()
    return BacktestReport(
        mode="liquid_universe_sanity_test",
        universe_selection="liquid_sample",
        lookahead_bias_level="MEDIUM",
        n_symbols=100,
        n_days=126,
        train_result=tr,
        validation_result=val_result,
        benchmark_return=tr.total_return - excess_return,
        benchmark_sharpe=0.3,
        benchmark_max_drawdown=-0.08,
        excess_return=excess_return,
        validation_benchmark_return=val_bench_return,
        notes=[],
    )


def test_validation_passes_all_gates():
    if not _HAS_TUNER:
        return
    cfg = {"min_validation_excess_return": 0.0, "max_validation_drawdown": -0.20, "min_validation_sharpe": 0.25}
    # val return=15%, val bench=5% → excess=+10% (passes >=0%)
    vr = _make_sim_result(total_return=0.15, sharpe=0.6, max_drawdown=-0.10)
    report = _make_backtest_report(val_result=vr, excess_return=0.05, val_bench_return=0.05)
    passed, reasons = validate_tuned_params(report, cfg)
    _assert(passed, f"expected pass, got failures: {reasons}")


def test_validation_fails_sharpe_gate():
    if not _HAS_TUNER:
        return
    cfg = {"min_validation_excess_return": 0.0, "max_validation_drawdown": -0.20, "min_validation_sharpe": 0.5}
    vr = _make_sim_result(total_return=0.05, sharpe=0.1, max_drawdown=-0.08)
    report = _make_backtest_report(val_result=vr, val_bench_return=0.03)
    passed, reasons = validate_tuned_params(report, cfg)
    _assert(not passed, "expected fail (sharpe < 0.5)")
    _assert(any("Sharpe" in r for r in reasons), f"expected Sharpe failure, got {reasons}")


def test_validation_fails_drawdown_gate():
    if not _HAS_TUNER:
        return
    cfg = {"min_validation_excess_return": 0.0, "max_validation_drawdown": -0.20, "min_validation_sharpe": 0.0}
    vr = _make_sim_result(total_return=0.05, sharpe=0.4, max_drawdown=-0.35)
    report = _make_backtest_report(val_result=vr, val_bench_return=0.03)
    passed, reasons = validate_tuned_params(report, cfg)
    _assert(not passed, "expected fail (drawdown -35% < -20%)")
    _assert(any("drawdown" in r.lower() for r in reasons), f"expected drawdown failure, got {reasons}")


def test_validation_no_val_window():
    if not _HAS_TUNER:
        return
    cfg = {"min_validation_excess_return": 0.0, "max_validation_drawdown": -0.20, "min_validation_sharpe": 0.25}
    report = _make_backtest_report(val_result=None)
    passed, reasons = validate_tuned_params(report, cfg)
    _assert(not passed, "expected fail when no validation window")


def test_should_apply_blocks_when_validation_fails():
    if not _HAS_TUNER:
        return
    cfg = {"auto_apply_if_valid": False}
    # --apply with failed validation must NOT write config
    _assert(not should_apply_tuned_config(apply_flag=True, validation_passed=False, backtest_cfg=cfg),
            "--apply should not write when validation_passed=False")


def test_should_apply_allows_when_validation_passes():
    if not _HAS_TUNER:
        return
    cfg = {"auto_apply_if_valid": False}
    _assert(should_apply_tuned_config(apply_flag=True, validation_passed=True, backtest_cfg=cfg),
            "--apply should write when validation passes")


def test_should_force_apply_bypasses_validation():
    if not _HAS_TUNER:
        return
    cfg = {"auto_apply_if_valid": False}
    _assert(should_apply_tuned_config(apply_flag=False, validation_passed=False, backtest_cfg=cfg, force_apply=True),
            "--force-apply should write regardless of validation")


# ---------------------------------------------------------------------------
# LLM review tests
# ---------------------------------------------------------------------------

def test_llm_review_rejects_forbidden_param():
    if not _HAS_TUNER:
        return
    candidates = [{"candidate_id": "c1"}]
    response = {
        "recommended_candidate_id": "c1",
        "apply_candidate_as_is": False,
        "proposed_adjustments": {"stop_loss_pct": -0.10},  # FORBIDDEN
        "rationale": "test",
        "risk_warnings": [],
        "confidence": 0.5,
    }
    valid, errors = validate_llm_review_response(response, candidates)
    _assert(not valid, "expected invalid response (forbidden param)")
    _assert(any("forbidden" in e.lower() or "stop_loss_pct" in e for e in errors), f"errors: {errors}")


def test_llm_review_rejects_unknown_candidate():
    if not _HAS_TUNER:
        return
    candidates = [{"candidate_id": "c1"}]
    response = {
        "recommended_candidate_id": "c99",  # doesn't exist
        "apply_candidate_as_is": True,
        "proposed_adjustments": {},
        "rationale": "test",
        "risk_warnings": [],
        "confidence": 0.7,
    }
    valid, errors = validate_llm_review_response(response, candidates)
    _assert(not valid, "expected invalid response (unknown candidate id)")


def test_llm_review_accepts_valid_response():
    if not _HAS_TUNER:
        return
    candidates = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]
    response = {
        "recommended_candidate_id": "c1",
        "apply_candidate_as_is": True,
        "proposed_adjustments": {"metric_threshold": 0.9, "index_pct": 0.7},
        "rationale": "Better risk-adjusted return",
        "risk_warnings": [],
        "confidence": 0.8,
    }
    valid, errors = validate_llm_review_response(response, candidates)
    _assert(valid, f"expected valid response, got errors: {errors}")


def test_llm_merge_applies_alpha_params():
    if not _HAS_TUNER:
        return
    base = {"metric_threshold": 1.0, "index_pct": 0.85, "sell_rules": {}, "scoring": {}, "momentum": {}}
    response = {
        "recommended_candidate_id": "c1",
        "apply_candidate_as_is": False,
        "proposed_adjustments": {
            "metric_threshold": 0.9,
            "take_profit_pct": 0.45,
            "value_pe_weight": 0.65,
        },
        "rationale": "",
        "risk_warnings": [],
        "confidence": 0.7,
    }
    result = merge_llm_recommendation_with_config(base, response)
    _assert(abs(result["metric_threshold"] - 0.9) < 1e-6, f"expected 0.9, got {result['metric_threshold']}")
    _assert(abs(result["sell_rules"]["take_profit_pct"] - 0.45) < 1e-6)
    _assert(abs(result["scoring"]["value_pe_weight"] - 0.65) < 1e-6)


def test_llm_merge_skips_forbidden_params():
    if not _HAS_TUNER:
        return
    base = {"metric_threshold": 1.0}
    response = {
        "recommended_candidate_id": "c1",
        "apply_candidate_as_is": False,
        "proposed_adjustments": {
            "stop_loss_pct": -0.05,          # forbidden
            "max_single_position_pct": 0.10, # forbidden
            "metric_threshold": 0.8,         # allowed
        },
        "rationale": "",
        "risk_warnings": [],
        "confidence": 0.5,
    }
    result = merge_llm_recommendation_with_config(base, response)
    _assert("stop_loss_pct" not in result, "forbidden param should not appear in merged config")
    _assert("max_single_position_pct" not in result, "forbidden param should not appear in merged config")
    _assert(abs(result["metric_threshold"] - 0.8) < 1e-6, "allowed param should be applied")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_ALL_TESTS = [
    test_position_52w_normal,
    test_position_52w_missing_price,
    test_position_52w_missing_low,
    test_position_52w_missing_high,
    test_position_52w_high_equals_low,
    test_position_52w_high_less_than_low,
    test_position_52w_clamp_above_one,
    test_position_52w_clamp_below_zero,
    test_position_52w_at_low,
    test_position_52w_at_high,
    test_momentum_score_none,
    test_momentum_score_below_015,
    test_momentum_score_015_to_035,
    test_momentum_score_035_to_075,
    test_momentum_score_075_to_095,
    test_momentum_score_above_095,
    test_sell_hard_stop_loss,
    test_sell_hard_yield_trap,
    test_sell_hard_quality_floor,
    test_sell_soft_take_profit,
    test_sell_soft_weak_value,
    test_no_sell_healthy_holding,
    test_sell_no_metrics_no_crash,
    test_position_value_from_holdings,
    test_can_buy_within_position_cap,
    test_can_buy_reduced_by_position_cap,
    test_can_buy_blocked_by_position_cap,
    test_can_buy_skipped_below_min_order,
    test_can_buy_blocked_when_cash_below_min_order,
    test_can_buy_order_size_capped,
    test_sector_cap_reduced,
    test_sector_cap_blocked,
    test_liquidity_gate,
    test_exit_type_hard_sell_is_failure,
    test_exit_type_take_profit_is_harvest,
    test_exit_type_weak_value_is_thesis,
    test_exit_type_no_sell_is_none,
    # backtest helpers
    test_split_price_window_70_30,
    test_split_price_window_full,
    test_compute_performance_metrics_flat,
    test_compute_performance_metrics_growing,
    test_compute_performance_metrics_drawdown,
    test_score_stocks_at_day_uses_rolling_features,
    test_universe_liquid_sample_respects_max,
    test_universe_top_scores_bias_high,
    test_universe_walk_forward_bias_low,
    test_universe_min_volume_filter,
    # validation gates
    test_validation_passes_all_gates,
    test_validation_fails_sharpe_gate,
    test_validation_fails_drawdown_gate,
    test_validation_no_val_window,
    test_should_apply_blocks_when_validation_fails,
    test_should_apply_allows_when_validation_passes,
    test_should_force_apply_bypasses_validation,
    # LLM review
    test_llm_review_rejects_forbidden_param,
    test_llm_review_rejects_unknown_candidate,
    test_llm_review_accepts_valid_response,
    test_llm_merge_applies_alpha_params,
    test_llm_merge_skips_forbidden_params,
]


if __name__ == "__main__":
    passed = 0
    failed = 0
    skipped = 0
    for t in _ALL_TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed", end="")
    if skipped:
        print(f", {skipped} skipped", end="")
    print()
    sys.exit(0 if failed == 0 else 1)
