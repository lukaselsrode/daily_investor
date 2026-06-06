"""
tests/test_backtest_exclusions.py — discretionary never-buy exclusions in the backtest.

The live gate (data/fundamentals.py) drops excluded industries/sectors from scoring; this
wires the SAME exclusions into the simulator's candidate selection via precomp.excluded_mask
so backtests evaluate the same investable universe (config: backtest.apply_discretionary_exclusions).

Covers:
  1. select_candidates never selects a name flagged in precomp.excluded_mask, even when it
     would otherwise pass every gate; excluded_mask=None leaves the full universe selectable.
  2. The util config flag resolves to a bool.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtesting.simulator import select_candidates
from backtesting.types import PrecomputedData
from tuning.constants import _current_params

# Wide-open candidate selection so the ONLY thing that can drop a name is excluded_mask.
_OPEN_CS = {
    "mode": "percentile", "top_percentile": 1.0, "max_candidates": 15, "min_candidates": 1,
    "use_absolute_score_floor": False, "absolute_score_floor": 0.0, "min_quality_score": 0.0,
    "min_momentum_score": -999.0, "min_conditional_momentum_score": -999.0,
    "allow_income_defensive_exception": False, "fallback_thresholds": [],
    "min_post_cooldown_candidates": 1,
}


def _mk_precomp(n_days: int = 6, excluded_mask=None) -> PrecomputedData:
    n, n_etfs = 4, 2
    return PrecomputedData(
        symbols=[f"STK{i}" for i in range(n)],
        prices=np.full((n_days, n), 100.0),
        pe_comp=np.full(n, 0.5), pb_comp=np.full(n, 0.5),
        quality_scores=np.array([0.80, 0.75, 0.70, 0.65]),
        income_scores=np.full(n, 0.05), yield_trap_mask=np.zeros(n, dtype=bool),
        bin_indices=np.full(n, 2, dtype=np.int32), has_position_52w=np.ones(n, dtype=bool),
        position_52w_arr=np.full(n, 0.50), return_1m_arr=np.zeros(n),
        etf_symbols=[f"ETF{j}" for j in range(n_etfs)], etf_prices=np.full((n_days, n_etfs), 200.0),
        baseline_scores=np.full(n, 0.60),
        sector_labels=["Tech", "Health", "Finance", "Energy"],
        volume_arr=np.full(n, 2_000_000.0), mode="test", universe_selection="test",
        lookahead_bias_level="LOW", benchmark_prices=np.full(n_days, 300.0), benchmark_symbol="SPY",
        position_52w_daily=np.full((n_days, n), 0.50), return_1m_daily=np.zeros((n_days, n)),
        bin_indices_daily=np.full((n_days, n), 2, dtype=np.int32),
        has_position_52w_daily=np.ones((n_days, n), dtype=bool),
        ret_5d_daily=None, ret_3m_daily=None, ret_6m_daily=None,
        rs_3m_daily=None, rs_6m_daily=None, vol_3m_daily=None,
        above_50dma_daily=None, above_200dma_daily=None, spy_prices=None,
        excluded_mask=excluded_mask,
    )


def test_excluded_mask_drops_candidate():
    params = np.asarray(_current_params(), float)
    composite = np.array([0.9, 0.8, 0.7, 0.6])   # STK0 is the top-scored name
    mask = np.array([True, False, False, False])  # ...and is flagged never-buy

    sel_on, _ = select_candidates(0, composite, _mk_precomp(excluded_mask=mask), params, _OPEN_CS)
    sel_off, _ = select_candidates(0, composite, _mk_precomp(excluded_mask=None), params, _OPEN_CS)
    # The mask must drop the excluded name and change NOTHING else.
    assert sel_off[0] and not sel_on[0], "mask must drop the excluded top name"
    assert np.array_equal(sel_on[1:], sel_off[1:]), "mask must not affect other names"


def test_no_mask_leaves_full_universe():
    params = np.asarray(_current_params(), float)
    composite = np.array([0.9, 0.8, 0.7, 0.6])
    sel, _ = select_candidates(0, composite, _mk_precomp(excluded_mask=None), params, _OPEN_CS)
    assert sel[0], "with no exclusion mask the top name must be selectable"


def test_config_flag_is_bool():
    from util import BACKTEST_PARAMS
    assert isinstance(BACKTEST_PARAMS.get("apply_discretionary_exclusions"), bool)
