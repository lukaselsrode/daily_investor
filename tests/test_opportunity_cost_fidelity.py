"""
tests/test_opportunity_cost_fidelity.py — the opportunity-cost ("max hold without
progress") exit must behave identically live and in the backtest.

The shared surface is the progress classifier: the simulator's vectorized
`_progress_vec` must match the live scalar `exit_analysis.is_progress` elementwise.
We also pin the live SellDecisionEngine opportunity-cost gate against the exact
boolean the simulator's `oc_mask` uses, and assert the two invariants that protect
"let winners run": a progressing position is never culled, and the exit is inert
when disabled.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from backtesting.simulator import _progress_vec
from portfolio.exit_analysis import is_progress
from portfolio.sell_engine import SellDecisionEngine
from util import EXIT_DECISION_PARAMS, SELL_RULES

# grids
_RATIO = [0.80, 0.90, 0.96, 0.97, 0.985, 1.0, 1.05]   # price / peak
_MOM   = [-0.30, -0.05, 0.0, 0.05, 0.10, 0.25]
_BAND  = [0.00, 0.03, 0.05]
_FLOOR = [0.0, 0.10, 0.20]


def test_progress_vec_matches_scalar():
    """Vectorized _progress_vec == live scalar is_progress, elementwise."""
    peak = 100.0
    for band, floor in product(_BAND, _FLOOR):
        ratios = np.array(_RATIO, float)
        moms   = np.array(_MOM, float)
        for m in moms:
            prices = ratios * peak
            peaks  = np.full_like(prices, peak)
            momv   = np.full_like(prices, m)
            vec = _progress_vec(prices, peaks, momv, band, floor)
            for j, r in enumerate(_RATIO):
                scalar = is_progress(r * peak, peak, float(m), band, floor)
                assert bool(vec[j]) == scalar, (r, m, band, floor)


def test_progress_vec_handles_nan_like_scalar():
    """NaN price/peak/momentum contribute no progress on that term (both forms)."""
    prices = np.array([np.nan, 100.0, 100.0])
    peaks  = np.array([100.0, np.nan, 100.0])
    moms   = np.array([0.5, 0.5, np.nan])
    vec = _progress_vec(prices, peaks, moms, 0.03, 0.10)
    # row0: price NaN but momentum 0.5>=0.10 -> progress; row1: peak NaN, mom 0.5 -> progress;
    # row2: price==peak -> progress regardless of NaN momentum.
    assert vec.tolist() == [True, True, True]
    nan_none = is_progress(None, 100.0, 0.5, 0.03, 0.10)   # mom term carries
    assert nan_none is True
    assert is_progress(90.0, 100.0, None, 0.03, 0.10) is False  # no peak-reclaim, no mom


def _holding(pct: float, price: float, peak: float):
    """A holding that trips NO earlier sell rule, so evaluate() reaches the oc gate."""
    return {
        "percent_change": str(pct * 100.0),   # Robinhood-style percent string
        "average_buy_price": price / (1.0 + pct),
        "price": price,
        "created_at": "2000-01-01T00:00:00Z",  # very old -> days_held huge
    }, peak


def _metrics(momentum: float):
    # value_metric well above sell_weak; quality above floor; no yield trap.
    return pd.Series({
        "value_metric": SELL_RULES["sell_weak_value_below"] + 0.5,
        "quality_score": SELL_RULES["sell_low_quality_below"] + 0.5,
        "momentum_score": momentum,
        "yield_trap_flag": False,
    })


def test_live_oc_gate_matches_sim_formula(monkeypatch):
    """SellDecisionEngine fires opportunity_cost iff the sim's oc_mask boolean holds."""
    oc = dict(EXIT_DECISION_PARAMS["opportunity_cost"])
    oc.update(enabled=True, stall_max_days=120, reclaim_band=0.03, progress_momentum_floor=0.10)
    monkeypatch.setitem(EXIT_DECISION_PARAMS, "opportunity_cost", oc)

    eng = SellDecisionEngine()
    peak = 100.0
    # price below peak by > band AND momentum below floor => not progressing
    for ratio, mom, stall in product([0.80, 0.96, 0.985, 1.0], _MOM, [0, 60, 119, 120, 200]):
        price = ratio * peak
        holding, pk = _holding(pct=0.03, price=price, peak=peak)
        decision = eng.evaluate(
            "T", holding, _metrics(mom), peak_price=pk, stall_days=stall,
        )
        progressing = is_progress(price, peak, mom, 0.03, 0.10)
        # sim oc_mask boolean (days_held huge, minhold satisfied): ~progress & stall>=max
        sim_oc = (not progressing) and (stall >= 120)
        got_oc = decision.exit_type == "opportunity_cost"
        assert got_oc == sim_oc, (ratio, mom, stall, progressing, decision.exit_type)


def test_progressing_position_never_culled(monkeypatch):
    """A position making progress is never opportunity-cost exited, even if very stale."""
    oc = dict(EXIT_DECISION_PARAMS["opportunity_cost"])
    oc.update(enabled=True, stall_max_days=120, reclaim_band=0.03, progress_momentum_floor=0.10)
    monkeypatch.setitem(EXIT_DECISION_PARAMS, "opportunity_cost", oc)
    eng = SellDecisionEngine()
    # at peak (ratio 1.0) -> progressing via fresh high; huge stall should not matter
    holding, pk = _holding(pct=0.03, price=100.0, peak=100.0)
    decision = eng.evaluate("T", holding, _metrics(-0.5), peak_price=pk, stall_days=9999)
    assert decision.exit_type != "opportunity_cost"


def test_disabled_is_inert():
    """With opportunity_cost disabled (default config), the gate never fires."""
    assert EXIT_DECISION_PARAMS["opportunity_cost"]["enabled"] is False
    eng = SellDecisionEngine()
    holding, pk = _holding(pct=0.03, price=80.0, peak=100.0)
    decision = eng.evaluate("T", holding, _metrics(-0.5), peak_price=pk, stall_days=9999)
    assert decision.exit_type != "opportunity_cost"
