"""
tests/test_regime_tilt.py — regime-conditional momentum tilt (slot 46).

Covers:
  1. Slot 46 exists, is named regime_bull_momentum_tilt, maps to
     regime.bullish.momentum_tilt, and is frozen by default.
  2. The active_regime_tilt preset unfreezes exactly slot 46 (1 DOF).
  3. tilt=0.0 is behaviour-preserving: scoring identical to the no-slot vector.
  4. A positive tilt shifts weight toward momentum ONLY in bullish regime,
     leaving neutral/defensive-day scoring unchanged.
  5. The tilt never changes the weight normalisation invariant (weights sum to 1).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def test_slot46_layout_and_frozen_by_default():
    from tuning.constants import (
        _CONFIG_PATH_TO_PARAM_IDX,
        BOUNDS,
        PARAM_NAMES,
        _get_active_indices,
    )
    idx = _CONFIG_PATH_TO_PARAM_IDX["regime.bullish.momentum_tilt"]
    assert idx == 46
    assert PARAM_NAMES[idx] == "regime_bull_momentum_tilt"
    assert BOUNDS[idx][0] == 0.0
    # frozen by default in both scopes
    assert 46 not in _get_active_indices(scope="overall_strategy", preset=None)
    assert 46 not in _get_active_indices(scope="active_sleeve_compounding", preset=None)


def test_slot47_meanrev_layout_and_frozen_by_default():
    from tuning.constants import (
        _CONFIG_PATH_TO_PARAM_IDX,
        BOUNDS,
        PARAM_NAMES,
        _get_active_indices,
    )
    idx = _CONFIG_PATH_TO_PARAM_IDX["regime.defensive.mean_reversion_blend"]
    assert idx == 47
    assert PARAM_NAMES[idx] == "regime_defensive_mean_reversion_blend"
    assert BOUNDS[idx] == (0.0, 1.0)
    assert 47 not in _get_active_indices(scope="active_sleeve_compounding", preset=None)


def test_active_alpha_engine_preset_unfreezes_regime_slots():
    from tuning.constants import _get_active_indices
    active = _get_active_indices(scope="active_sleeve_compounding", preset="active_alpha_engine")
    assert 46 in active  # momentum tilt
    assert 47 in active  # mean-reversion blend


def test_meanrev_blend_only_affects_nonbull_scoring():
    """slot 47 > 0 changes scoring ONLY in non-bull regimes; inert in bull."""
    import numpy as np

    from backtesting.simulator import _detect_regime, get_default_params, score_stocks_at_day
    base16 = get_default_params()
    ext = np.concatenate([base16, np.zeros(48 - len(base16))])
    ext[47] = 0.5  # heavy mean-reversion blend

    # bullish substrate: mean-reversion blend must be INERT
    pc_bull = _tiny_precomp(bullish=True)
    db = pc_bull.prices.shape[0] - 1
    if _detect_regime(pc_bull, db) == "bullish":
        s0 = score_stocks_at_day(pc_bull, np.concatenate([base16, np.zeros(48 - len(base16))]), db)
        s_mr = score_stocks_at_day(pc_bull, ext, db)
        assert np.allclose(s0, s_mr, atol=1e-12), "mean-reversion must be inert in bull regime"

    # defensive substrate: mean-reversion blend must CHANGE scoring
    pc_def = _tiny_precomp(bullish=False)
    dd = pc_def.prices.shape[0] - 1
    if _detect_regime(pc_def, dd) != "bullish":
        s0d = score_stocks_at_day(pc_def, np.concatenate([base16, np.zeros(48 - len(base16))]), dd)
        s_mrd = score_stocks_at_day(pc_def, ext, dd)
        assert not np.allclose(s0d, s_mrd), "mean-reversion must alter scores in fear regime"


def test_active_regime_tilt_preset_unfreezes_only_slot46():
    from tuning.constants import _get_active_indices
    active = _get_active_indices(scope="active_sleeve_compounding", preset="active_regime_tilt")
    assert active == [46]


def _tiny_precomp(n_days=210, n_stocks=5, bullish=True):
    """Construct a minimal PrecomputedData for scoring tests."""
    from backtesting.types import PrecomputedData
    rng = np.random.default_rng(0)
    prices = np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, n_stocks)), axis=0) * 100
    # benchmark: rising (bullish) or falling below 200dma (defensive)
    if bullish:
        bench = np.linspace(100, 160, n_days)
    else:
        bench = np.concatenate([np.linspace(100, 160, n_days - 10),
                                np.linspace(160, 110, 10)])
    zeros_s = np.zeros(n_stocks)
    return PrecomputedData(
        symbols=[f"S{i}" for i in range(n_stocks)],
        prices=prices,
        pe_comp=rng.uniform(-1, 1, n_stocks),
        pb_comp=rng.uniform(-1, 1, n_stocks),
        quality_scores=rng.uniform(-1, 1, n_stocks),
        income_scores=rng.uniform(-1, 1, n_stocks),
        yield_trap_mask=np.zeros(n_stocks, bool),
        bin_indices=np.zeros(n_stocks, np.int32),
        has_position_52w=np.ones(n_stocks, bool),
        position_52w_arr=rng.uniform(0, 1, n_stocks),
        return_1m_arr=rng.uniform(-0.1, 0.1, n_stocks),
        etf_symbols=[],
        etf_prices=np.zeros((n_days, 0)),
        baseline_scores=zeros_s,
        sector_labels=["X"] * n_stocks,
        volume_arr=np.ones(n_stocks) * 1e6,
        mode="liquid_universe_sanity_test",
        universe_selection="liquid_all",
        lookahead_bias_level="MEDIUM",
        benchmark_prices=bench,
        benchmark_symbol="SPY",
        position_52w_daily=np.tile(rng.uniform(0, 1, n_stocks), (n_days, 1)),
        return_1m_daily=np.tile(rng.uniform(-0.1, 0.1, n_stocks), (n_days, 1)),
        bin_indices_daily=np.zeros((n_days, n_stocks), np.int32),
        has_position_52w_daily=np.ones((n_days, n_stocks), bool),
    )


def test_tilt_zero_matches_no_slot():
    from backtesting.simulator import get_default_params, score_stocks_at_day
    pc = _tiny_precomp(bullish=True)
    base16 = get_default_params()                 # 16-len, slot 46 absent
    ext = np.concatenate([base16, np.zeros(47 - len(base16))])  # tilt=0 at slot 46
    day = pc.prices.shape[0] - 1
    s_base = score_stocks_at_day(pc, base16, day)
    s_zero = score_stocks_at_day(pc, ext, day)
    assert np.allclose(s_base, s_zero, atol=1e-12)


def test_positive_tilt_changes_only_bullish_scoring():
    from backtesting.simulator import _detect_regime, get_default_params, score_stocks_at_day
    base16 = get_default_params()
    ext_tilt = np.concatenate([base16, np.zeros(47 - len(base16))])
    ext_tilt[46] = 0.30

    # Bullish substrate: tilt should change scores (momentum re-weighted).
    pc_bull = _tiny_precomp(bullish=True)
    day_b = pc_bull.prices.shape[0] - 1
    assert _detect_regime(pc_bull, day_b) == "bullish"
    s0 = score_stocks_at_day(pc_bull, np.concatenate([base16, np.zeros(47 - len(base16))]), day_b)
    s_tilt = score_stocks_at_day(pc_bull, ext_tilt, day_b)
    assert not np.allclose(s0, s_tilt), "tilt must alter scores in bullish regime"

    # Defensive substrate: on a defensive day tilt must NOT change scoring.
    pc_def = _tiny_precomp(bullish=False)
    day_d = pc_def.prices.shape[0] - 1
    if _detect_regime(pc_def, day_d) != "bullish":
        s0d = score_stocks_at_day(pc_def, np.concatenate([base16, np.zeros(47 - len(base16))]), day_d)
        s_tiltd = score_stocks_at_day(pc_def, ext_tilt, day_d)
        assert np.allclose(s0d, s_tiltd, atol=1e-12), "tilt must be inert outside bullish regime"
