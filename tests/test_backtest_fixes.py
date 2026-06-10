"""
tests/test_backtest_fixes.py — Regression coverage for the backtesting-layer bug fixes.

Fix 1: window slicing must keep EVERY per-day array calendar-aligned (vix/spy/dollar-volume/…).
Fix 2: regime labels computed on the FULL load are attached/sliced, so offset windows are not
       hard-coded "bullish" for their first 200 days.
Fix 3: NaN-priced positions mark AND fill forced exits at the last valid traded price, not cost.
Fix 4: survivorship-free dead names get median-neutral fundamentals (buyable pre-delist) and a
       per-day tradeability mask (never buyable post-delist).
Fix 5: warmup bin scores from config (a), NaN-neutral momentum ranks (b), no bfill price
       fabrication (c), commission actually deducted from cash (d), take-profit floor
       multiplier read from config (e).
Fix 6: robust score is excess-vs-SPY dominant (weights exposed as module constants).

All thresholds asserted here come from the LIVE config (util / config manager) — never
hardcoded copies.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

N_DAYS_DEFAULT = 25


def _make_precomp(
    n_days: int,
    n_stocks: int,
    prices: np.ndarray | None = None,
    benchmark: np.ndarray | None = None,
    **overrides,
):
    """Minimal warm-up-path PrecomputedData (ret_3m_daily=None) with sane defaults."""
    from backtesting.types import PrecomputedData

    if prices is None:
        prices = np.full((n_days, n_stocks), 100.0)
    if benchmark is None:
        benchmark = np.linspace(100.0, 105.0, n_days)
    zeros_s = np.zeros(n_stocks)
    pc = PrecomputedData(
        symbols=[f"S{i}" for i in range(n_stocks)],
        prices=prices.astype(float),
        pe_comp=np.ones(n_stocks),
        pb_comp=np.ones(n_stocks),
        quality_scores=np.ones(n_stocks),
        income_scores=zeros_s.copy(),
        yield_trap_mask=np.zeros(n_stocks, bool),
        bin_indices=np.zeros(n_stocks, np.int32),
        has_position_52w=np.zeros(n_stocks, bool),
        position_52w_arr=np.full(n_stocks, np.nan),
        return_1m_arr=np.full(n_stocks, np.nan),
        etf_symbols=[],
        etf_prices=np.zeros((n_days, 0)),
        baseline_scores=zeros_s.copy(),
        sector_labels=["X"] * n_stocks,
        volume_arr=np.ones(n_stocks) * 1e6,
        mode="liquid_universe_full",
        universe_selection="liquid_all",
        lookahead_bias_level="MEDIUM",
        benchmark_prices=benchmark.astype(float),
        benchmark_symbol="SPY",
        position_52w_daily=np.full((n_days, n_stocks), np.nan),
        return_1m_daily=np.full((n_days, n_stocks), np.nan),
        bin_indices_daily=np.zeros((n_days, n_stocks), np.int32),
        has_position_52w_daily=np.zeros((n_days, n_stocks), bool),
    )
    return pc._replace(**overrides) if overrides else pc


def _permissive_cs() -> dict:
    """Candidate-selection params with every gate off (alive/tradeability still applies)."""
    return {
        "mode": "percentile",
        "top_percentile": 1.0,
        "max_candidates": 10,
        "min_candidates": 0,
        "use_absolute_score_floor": False,
        "absolute_score_floor": -999.0,
        "min_quality_score": -999.0,
        "min_momentum_score": -999.0,
        "min_conditional_momentum_score": -999.0,
        "allow_income_defensive_exception": False,
    }


def _sim_kwargs(**extra) -> dict:
    kw = dict(
        starting_capital=10_000.0,
        slippage_bps=0.0,
        commission_per_trade=0.0,
        weekly_contribution=0.0,
        rebalance_frequency_days=50,  # > n_days: no mid-window rebuys, deterministic book
        cs_params=_permissive_cs(),
    )
    kw.update(extra)
    return kw


# ---------------------------------------------------------------------------
# Fix 1 — canonical slicer aligns EVERY per-day array
# ---------------------------------------------------------------------------

def _make_full_precomp(n_days: int, n_stocks: int, n_etfs: int):
    """PrecomputedData with EVERY per-day field populated with day-distinguishable values."""
    from backtesting.types import PrecomputedData

    days_f = np.arange(n_days, dtype=float)
    per_day_stock = np.tile(days_f[:, None], (1, n_stocks))
    per_day_bool = np.tile((np.arange(n_days) % 2 == 0)[:, None], (1, n_stocks))
    return PrecomputedData(
        symbols=[f"S{i}" for i in range(n_stocks)],
        prices=per_day_stock + 100.0,
        pe_comp=np.ones(n_stocks),
        pb_comp=np.ones(n_stocks),
        quality_scores=np.ones(n_stocks),
        income_scores=np.zeros(n_stocks),
        yield_trap_mask=np.zeros(n_stocks, bool),
        bin_indices=np.zeros(n_stocks, np.int32),
        has_position_52w=np.ones(n_stocks, bool),
        position_52w_arr=np.full(n_stocks, 0.5),
        return_1m_arr=np.zeros(n_stocks),
        etf_symbols=[f"E{j}" for j in range(n_etfs)],
        etf_prices=np.tile(days_f[:, None], (1, n_etfs)) + 200.0,
        baseline_scores=np.zeros(n_stocks),
        sector_labels=["X"] * n_stocks,
        volume_arr=np.ones(n_stocks) * 1e6,
        mode="liquid_universe_full",
        universe_selection="liquid_all",
        lookahead_bias_level="MEDIUM",
        benchmark_prices=days_f + 300.0,
        benchmark_symbol="SPY",
        position_52w_daily=per_day_stock / max(n_days, 1),
        return_1m_daily=per_day_stock * 0.001,
        bin_indices_daily=np.tile((np.arange(n_days) % 5)[:, None], (1, n_stocks)).astype(np.int32),
        has_position_52w_daily=per_day_bool,
        ret_5d_daily=per_day_stock + 1.0,
        ret_3m_daily=per_day_stock + 2.0,
        ret_6m_daily=per_day_stock + 3.0,
        rs_3m_daily=per_day_stock + 4.0,
        rs_6m_daily=per_day_stock + 5.0,
        vol_3m_daily=per_day_stock * 0.01 + 0.1,
        above_50dma_daily=per_day_bool,
        above_200dma_daily=~per_day_bool,
        spy_prices=days_f + 400.0,
        industry_labels=tuple(["Ind"] * n_stocks),
        market_caps=np.ones(n_stocks) * 1e9,
        momentum_scores=np.zeros(n_stocks),
        dollar_volume_daily=per_day_stock + 500.0,
        excluded_mask=np.zeros(n_stocks, bool),
        regime_labels_daily=np.array([f"L{d}" for d in range(n_days)], dtype=object),
        vix_prices=days_f + 600.0,
        tradeable_mask_daily=per_day_bool,
    )


def test_slice_precomp_aligns_every_per_day_array():
    """Slicing at offset k must take rows [k, k+w) of EVERY (n_days, …) array —
    omitting any one (the old vix/spy/dollar-volume bug) leaves it in full-load
    coordinates while the window is day-relative."""
    from backtesting.regime_scope import slice_precomp
    from backtesting.types import PrecomputedData

    n_days, k, w = 12, 4, 5
    pc = _make_full_precomp(n_days=n_days, n_stocks=3, n_etfs=2)
    win = slice_precomp(pc, slice(k, k + w))

    n_checked = 0
    for name in PrecomputedData._fields:
        full = getattr(pc, name)
        if isinstance(full, np.ndarray) and full.ndim >= 1 and full.shape[0] == n_days:
            sliced = getattr(win, name)
            assert isinstance(sliced, np.ndarray), f"{name} dropped by slice_precomp"
            assert sliced.shape[0] == w, f"{name} not sliced (shape {sliced.shape})"
            assert np.array_equal(sliced, full[k:k + w]), f"{name} misaligned after slicing"
            n_checked += 1
    # vix/spy/dollar-volume/regime-labels/tradeable + the core daily arrays must all be present
    assert n_checked >= 20, f"only {n_checked} per-day arrays found — builder out of date?"
    # The headline cases of the original bug, asserted explicitly:
    assert win.vix_prices[0] == pc.vix_prices[k]
    assert win.spy_prices[0] == pc.spy_prices[k]
    assert win.dollar_volume_daily[0, 0] == pc.dollar_volume_daily[k, 0]


def test_all_slicer_call_sites_share_one_helper():
    """random_walk must delegate to the canonical regime_scope.slice_precomp."""
    from backtesting import random_walk, regime_scope

    assert random_walk._slice_precomp is regime_scope.slice_precomp


# ---------------------------------------------------------------------------
# Fix 2 — regime labels from the FULL load survive window slicing
# ---------------------------------------------------------------------------

def _bear_after_300_precomp(n_days: int = 600):
    """Benchmark rises 300d then crashes below its 200DMA; VIX elevated after day 300."""
    from config.manager import ConfigManager

    vix_def = ConfigManager.get().regime.vix_defensive_threshold
    bench = np.concatenate([
        np.linspace(100.0, 200.0, 300),
        np.linspace(200.0, 120.0, 100),
        np.full(n_days - 400, 120.0),
    ])
    vix = np.concatenate([
        np.full(300, 12.0),
        np.full(n_days - 300, vix_def + 5.0),
    ])
    rng = np.random.default_rng(7)
    prices = np.cumprod(1 + rng.normal(0.0003, 0.01, (n_days, 3)), axis=0) * 50
    return _make_precomp(n_days, 3, prices=prices, benchmark=bench, vix_prices=vix)


def test_offset_window_keeps_full_load_regime_not_bullish():
    """A window sliced at day 400 of a bear tape must NOT classify as bullish just
    because the window itself has < 200 days of history."""
    from backtesting.regime_scope import regime_labels, slice_precomp
    from backtesting.simulator import _detect_regime

    pc = _bear_after_300_precomp()

    # The defect path: slicing WITHOUT precomputed labels resets the 200DMA context.
    win_broken = slice_precomp(pc, slice(400, 450))
    assert _detect_regime(win_broken, 0) == "bullish"  # documents why labels must be attached

    labeled = pc._replace(regime_labels_daily=regime_labels(pc))
    win = slice_precomp(labeled, slice(400, 450))
    assert _detect_regime(win, 0) != "bullish"
    assert all(lbl != "bullish" for lbl in win.regime_labels_daily)


def test_random_window_backtest_attaches_full_load_labels_for_all_scope():
    """Every random window (even regime_scope='all') must carry full-load regime labels —
    previously they were attached only for scoped runs, so every short window was bullish."""
    from backtesting.random_walk import random_window_backtest
    from backtesting.regime_scope import regime_labels
    from backtesting.types import SimResult

    pc = _bear_after_300_precomp()
    full_labels = regime_labels(pc)

    captured = []

    def _fake_sim(win_precomp, params, **kwargs):
        captured.append(win_precomp)
        return SimResult(
            final_value=10_100, total_return=0.01, sharpe=0.5, calmar=0.5,
            max_drawdown=-0.03, trades_made=3, average_positions=2.0,
            equity_curve=np.array([10_000.0, 10_100.0]),
        )

    window_days = 20
    with patch("backtesting.random_walk.run_simulation", side_effect=_fake_sim):
        summary = random_window_backtest(
            pc, params=None, n_windows=3, window_days=window_days, seed=1, regime_scope="all",
        )

    assert summary.n_windows > 0 and len(captured) == len(summary.window_results)
    for win_pc, wr in zip(captured, summary.window_results):
        assert win_pc.regime_labels_daily is not None
        assert len(win_pc.regime_labels_daily) == window_days
        assert np.array_equal(
            np.asarray(win_pc.regime_labels_daily, dtype=object),
            full_labels[wr.start_day:wr.end_day],
        )


# ---------------------------------------------------------------------------
# Fix 3 — NaN-priced positions mark and fill at the LAST TRADED price, not cost
# ---------------------------------------------------------------------------

def test_nan_priced_position_marks_and_exits_at_last_traded_price():
    from backtesting.simulator import get_default_params, run_simulation
    from util import EXIT_DECISION_PARAMS, RISK_LIMITS, SELL_RULES

    min_hold = int(RISK_LIMITS["minimum_hold_days"])
    stall_max = min_hold + 2
    n_days = stall_max + 8
    nan_from = 10
    assert nan_from < stall_max, "price must already be NaN when the forced exit fires"

    last_traded = 96.0
    prices = np.full((n_days, 1), np.nan)
    prices[0, 0] = 100.0
    prices[1:nan_from, 0] = np.linspace(100.0, last_traded, nan_from - 1)
    # Sanity (live config): the pre-NaN decline must not trip the stop-loss/trailing stop,
    # so the only exit available during the NaN stretch is the forced (opportunity-cost) one.
    decline = last_traded / 100.0 - 1.0
    assert decline > float(SELL_RULES["stop_loss_pct"])
    assert decline > float(SELL_RULES["trailing_stop_pct"])

    pc = _make_precomp(n_days, 1, prices=prices)

    # Enable the opportunity-cost exit (config-gated, OFF live) to force a sell while the
    # price is NaN. Restored afterwards; thresholds asserted elsewhere stay live-config.
    saved = EXIT_DECISION_PARAMS.get("opportunity_cost")
    EXIT_DECISION_PARAMS["opportunity_cost"] = {
        "enabled": True, "stall_max_days": stall_max,
        "reclaim_band": 0.0, "progress_momentum_floor": 0.10,
    }
    try:
        sim = run_simulation(pc, get_default_params(), **_sim_kwargs())
    finally:
        if saved is None:
            EXIT_DECISION_PARAMS.pop("opportunity_cost", None)
        else:
            EXIT_DECISION_PARAMS["opportunity_cost"] = saved

    buys = [t for t in sim.trade_log if t.side == "buy"]
    sells = [t for t in sim.trade_log if t.side == "sell"]
    assert len(buys) == 1 and len(sells) == 1
    qty, alloc = buys[0].quantity, buys[0].amount
    assert buys[0].price == pytest.approx(100.0)

    # Marking: during the NaN stretch (before the exit) the book must be valued at the
    # last traded price — valuing at cost (100) erased the loss.
    cash_after_buy = 10_000.0 - alloc
    mark_day = nan_from + 1
    assert int(sells[0].date) > mark_day
    assert sim.equity_curve[mark_day] == pytest.approx(cash_after_buy + qty * last_traded)
    assert sim.equity_curve[mark_day] < 10_000.0  # the loss is visible, not erased

    # Fill: the forced exit must execute at the last traded price, not cost basis.
    assert sells[0].price == pytest.approx(last_traded)
    assert sells[0].amount == pytest.approx(qty * last_traded)
    assert sells[0].pnl < 0.0


# ---------------------------------------------------------------------------
# Fix 4 — survivorship-free: dead names buyable pre-delist, never post-delist
# ---------------------------------------------------------------------------

def test_assemble_gives_dead_names_median_fundamentals_and_tradeable_mask(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from backtesting import survivorship
    from util import CANDIDATE_SELECTION_PARAMS

    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    cal = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=100)]
    delist_idx = 59

    def _write(sym: str, days: list[str], base: float):
        df = pd.DataFrame(
            {"close": np.linspace(base, base * 1.5, len(days)),
             "volume": np.full(len(days), 1e6)},
            index=days,
        )
        df.to_parquet(price_dir / f"{sym}.parquet")

    _write("SPY", cal, 400.0)
    _write("ALIVE", cal, 50.0)
    _write("DEADCO", cal[: delist_idx + 1], 20.0)  # strong momentum, delists mid-window
    dead = pd.DataFrame([{
        "symbol": "DEADCO", "first_date": cal[0],
        "delist_date": cal[delist_idx], "max_adv": 1e6,
    }])
    dead_path = tmp_path / "dead_universe.parquet"
    dead.to_parquet(dead_path)
    monkeypatch.setattr(survivorship, "_PRICE_DIR", str(price_dir))
    monkeypatch.setattr(survivorship, "_DEAD_PARQUET", str(dead_path))

    alive_quality = 0.62
    agg = pd.DataFrame([{
        "symbol": "ALIVE", "volume": 1e7, "sector": "Tech", "industry": "Software",
        "quality_score": alive_quality, "income_score": 0.2, "pe_comp": 0.3, "pb_comp": 0.4,
        "value_metric": 0.5, "position_52w": 0.6, "return_1m": 0.02, "market_cap": 1e9,
    }])
    n_days = 80
    closes, ext_agg, dv, tradeable = survivorship.assemble(
        agg, ["ALIVE"], [], "SPY", n_days, add_dead=True,
    )

    dead_row = ext_agg[ext_agg["symbol"] == "DEADCO"].iloc[0]
    # Median-neutral fundamentals: the alive-universe MEDIAN, not 0.0 — zero sat below the
    # live min_quality_score buy gate, auto-rejecting every dead name (survivorship bias
    # survived in a run labelled survivorship-free).
    assert float(dead_row["quality_score"]) == pytest.approx(alive_quality)
    assert float(dead_row["pe_comp"]) == pytest.approx(0.3)
    assert float(dead_row["pb_comp"]) == pytest.approx(0.4)
    assert float(dead_row["income_score"]) == pytest.approx(0.2)
    assert float(dead_row["quality_score"]) >= float(CANDIDATE_SELECTION_PARAMS["min_quality_score"])

    # Prices ffilled to window end so held positions can mark…
    assert closes["DEADCO"].notna().iloc[-1]
    # …but tradeability ends at the last NATIVE print.
    cal_win = closes.index.tolist()
    last_native = cal[delist_idx]
    for day, ok in zip(cal_win, tradeable["DEADCO"].tolist()):
        assert ok == (day <= last_native), f"tradeable wrong on {day}"
    assert tradeable["ALIVE"].all()


def test_select_candidates_blocks_dead_name_after_delist():
    from backtesting.simulator import get_default_params, select_candidates

    n_days, delist_day = 20, 10
    tradeable = np.ones((n_days, 2), dtype=bool)
    tradeable[delist_day:, 1] = False  # stock 1 delists at day 10 (prices stay ffilled-finite)
    pc = _make_precomp(n_days, 2, tradeable_mask_daily=tradeable)
    scores = np.array([0.5, 0.9])  # the dead name scores HIGHER — only tradeability blocks it
    params = get_default_params()

    before, _ = select_candidates(delist_day - 1, scores, pc, params, _permissive_cs())
    after, _ = select_candidates(delist_day + 1, scores, pc, params, _permissive_cs())
    assert before[1], "dead name must be buyable BEFORE its delist date"
    assert not after[1], "dead name must NOT be buyable after its delist date"
    assert before[0] and after[0]


def test_simulation_never_buys_past_last_native_print():
    from backtesting.simulator import get_default_params, run_simulation

    n_days = 12
    prices = np.column_stack([
        np.full(n_days, 100.0),
        np.full(n_days, 50.0),  # finite all window (post-delist ffill) but never tradeable
    ])
    tradeable = np.ones((n_days, 2), dtype=bool)
    tradeable[:, 1] = False
    pc = _make_precomp(n_days, 2, prices=prices, tradeable_mask_daily=tradeable)

    sim = run_simulation(pc, get_default_params(), **_sim_kwargs())
    buys = [t for t in sim.trade_log if t.side == "buy"]
    assert buys, "the alive name should still be bought"
    assert all(t.symbol != "S1" for t in buys), "untradeable (delisted) name was bought"


# ---------------------------------------------------------------------------
# Fix 5a — warm-up momentum uses configured position_bin_scores, not params[10:16]
# ---------------------------------------------------------------------------

def test_warmup_momentum_uses_config_bin_scores_not_momentum_subweights():
    from backtesting.simulator import (
        _momentum_score_at_day,
        _momentum_score_warmup_vec,
        get_default_params,
        score_stocks_at_day,
    )
    from util import SCORING_PARAMS

    warmup_cfg = SCORING_PARAMS["momentum_warmup"]
    boundaries = np.array(warmup_cfg["position_bin_boundaries"])
    bin_scores = np.array(warmup_cfg["position_bin_scores"], dtype=float)

    n_days, n_stocks = 3, 3
    pos = np.array([0.05, 0.50, 0.99])
    pc = _make_precomp(
        n_days, n_stocks,
        position_52w_daily=np.tile(pos, (n_days, 1)),
        has_position_52w_daily=np.ones((n_days, n_stocks), bool),
        bin_indices_daily=np.tile(
            np.searchsorted(boundaries, pos, side="right").astype(np.int32), (n_days, 1)
        ),
        return_1m_daily=np.zeros((n_days, n_stocks)),
    )
    assert pc.ret_3m_daily is None  # warm-up path

    params = get_default_params()
    mom = _momentum_score_at_day(pc, params, 0)
    expected = _momentum_score_warmup_vec(
        pc.bin_indices_daily[0], pc.has_position_52w_daily[0],
        pc.position_52w_daily[0], pc.return_1m_daily[0], bin_scores,
    )
    assert np.array_equal(mom, expected)

    # The momentum sub-weights (params[10:16]) must NOT leak into warm-up bin scoring.
    params_perturbed = params.copy()
    params_perturbed[10:16] = 9.99
    assert np.array_equal(
        score_stocks_at_day(pc, params, 0),
        score_stocks_at_day(pc, params_perturbed, 0),
    )


# ---------------------------------------------------------------------------
# Fix 5b — NaN momentum inputs rank NEUTRAL (0.0), never mid-pack imputed
# ---------------------------------------------------------------------------

def test_multifactor_momentum_scores_nan_inputs_neutral():
    from backtesting.simulator import _momentum_score_multifactor_vec

    n_days, n_stocks = 2, 5
    rs3m = np.array([[np.nan, -0.5, -0.3, -0.1, 0.2]] * n_days)
    zeros = np.zeros((n_days, n_stocks))
    pc = _make_precomp(
        n_days, n_stocks,
        rs_3m_daily=rs3m,
        rs_6m_daily=zeros.copy(),
        ret_3m_daily=zeros.copy(),
        ret_5d_daily=zeros.copy(),
        vol_3m_daily=np.full((n_days, n_stocks), 0.15),
        return_1m_daily=zeros.copy(),
        position_52w_daily=np.full((n_days, n_stocks), 0.5),
    )
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # rs_3m only
    score = _momentum_score_multifactor_vec(0, pc, weights)

    # Live behavior (_pct_rank_series): NaN is excluded from the rank and scored 0.0.
    # The old pre-imputation (NaN→0.0 BEFORE ranking) put the missing-data name at
    # rank 4/5 here (above three real negative-RS names) — top-decile in a bear tape.
    assert score[0] == pytest.approx(0.0)
    assert score[1] < 0.0 and score[1] < score[0]  # worst real name ranks BELOW neutral
    finite_scores = score[1:]
    assert np.all(np.diff(finite_scores) > 0)  # real values still strictly ranked


def test_multifactor_risk_adj_nan_propagates_to_neutral_rank():
    from backtesting.simulator import MOMENTUM_INPUT_PARAMS, _momentum_score_multifactor_vec

    n_days, n_stocks = 1, 4
    zeros = np.zeros((n_days, n_stocks))
    # Keep all real ret3m values above the live falling-knife threshold so the
    # penalty never fires — this test isolates rank propagation, not penalties.
    knife = MOMENTUM_INPUT_PARAMS["penalties"]["falling_knife_3m_threshold"]
    mild_neg, milder_neg = knife / 2, knife / 4
    pc = _make_precomp(
        n_days, n_stocks,
        rs_3m_daily=zeros.copy(),
        rs_6m_daily=zeros.copy(),
        # risk_adj = ret3m/vol: NaN in EITHER input must make the rank input NaN.
        ret_3m_daily=np.array([[np.nan, mild_neg, milder_neg, 0.3]]),
        ret_5d_daily=zeros.copy(),
        vol_3m_daily=np.array([[0.15, np.nan, 0.15, 0.15]]),
        return_1m_daily=zeros.copy(),
        position_52w_daily=np.full((n_days, n_stocks), 0.5),
    )
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # risk_adj only
    score = _momentum_score_multifactor_vec(0, pc, weights)
    assert score[0] == pytest.approx(0.0)  # NaN ret3m → neutral
    assert score[1] == pytest.approx(0.0)  # NaN vol → neutral
    assert score[2] < 0.0 < score[3]       # the two real values still rank


# ---------------------------------------------------------------------------
# Fix 5c — no bfill: pre-IPO prices must stay NaN (no look-ahead fabrication)
# ---------------------------------------------------------------------------

def test_extract_closes_does_not_backfill_pre_ipo_prices():
    from backtesting.data_loader import _extract_closes

    raw = pd.DataFrame({"Close": [np.nan, np.nan, 10.0, np.nan, 11.0]})
    closes = _extract_closes(raw, ["AAA"])
    col = closes["AAA"]
    assert col.iloc[0] != col.iloc[0] and col.iloc[1] != col.iloc[1]  # NaN preserved (no bfill)
    assert col.iloc[2] == 10.0
    assert col.iloc[3] == 10.0  # interior gap still forward-filled
    assert col.iloc[4] == 11.0


# ---------------------------------------------------------------------------
# Fix 5d — commission_per_trade is actually deducted from cash
# ---------------------------------------------------------------------------

def test_commission_is_deducted_from_cash():
    from backtesting.simulator import get_default_params, run_simulation

    pc = _make_precomp(8, 1)  # flat price: any value gap can only come from commission
    params = get_default_params()
    sim_free = run_simulation(pc, params, **_sim_kwargs(commission_per_trade=0.0))
    commission = 1.0
    sim_paid = run_simulation(pc, params, **_sim_kwargs(commission_per_trade=commission))

    n_buys = len([t for t in sim_paid.trade_log if t.side == "buy"])
    assert n_buys == 1
    # With zero slippage all friction is commission — and it must now cost real cash.
    assert sim_paid.friction_cost == pytest.approx(n_buys * commission)
    assert sim_free.final_value - sim_paid.final_value == pytest.approx(sim_paid.friction_cost)


# ---------------------------------------------------------------------------
# Fix 5e — take-profit floor multiplier reads the live config key
# ---------------------------------------------------------------------------

def test_take_profit_floor_multiplier_matches_live_config():
    from backtesting import simulator
    from util import SELL_RULES

    assert simulator._TAKE_PROFIT_FLOOR_MULTIPLIER == pytest.approx(
        float(SELL_RULES["take_profit_value_floor_multiplier"])
    )


# ---------------------------------------------------------------------------
# Fix 6 — robust score is excess-vs-SPY dominant
# ---------------------------------------------------------------------------

def test_robust_score_matches_published_weights():
    from backtesting import random_walk as rw

    me, ms, pb, dd, to, std = 0.02, 1.2, 0.6, -0.10, 1.0, 0.03
    expected = (
        me
        + rw.ROBUST_W_SHARPE * ms
        + rw.ROBUST_W_PCT_BEATING * pb
        - rw.ROBUST_W_DRAWDOWN * abs(dd)
        - rw.ROBUST_W_TURNOVER * to
        - rw.ROBUST_W_STD_EXCESS * std
    )
    assert rw.compute_robust_score(me, ms, pb, dd, to, std) == pytest.approx(expected)


def test_robust_score_excess_dominates_sharpe():
    """Realistic spread: +3% median excess must beat a zero-excess config whose only
    edge is short-window Sharpe — the old 0.50 sharpe weight ranked ~95% on Sharpe."""
    from backtesting.random_walk import compute_robust_score

    common = dict(pct_beating=0.55, worst_decile_dd=-0.10, median_turnover=1.0, std_excess=0.03)
    high_excess = compute_robust_score(median_excess=0.03, median_sharpe=1.0, **common)
    high_sharpe = compute_robust_score(median_excess=0.00, median_sharpe=1.5, **common)
    assert high_excess > high_sharpe
