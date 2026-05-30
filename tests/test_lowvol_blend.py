"""
Tests for the low-volatility quality blend (slot 48).

Low-vol is a price-derived quality proxy validated at the signal level
(full-sample forward-IC +0.04@21d / +0.067@63d on the 1550d substrate, orthogonal
to momentum). It is shipped as a tunable, FROZEN-BY-DEFAULT (0.0), because the
rolling-window backtest showed it is net-neutral on the momentum-alpha config
(win-rate flat at 50%, Sharpe-win slightly worse) — same disposition as the
mean-reversion blend (slot 47): available, validated, off by default.
"""
from __future__ import annotations

import numpy as np


def test_lowvol_slot_registered_and_frozen_by_default():
    from tuning.constants import (
        _CONFIG_PATH_TO_PARAM_IDX,
        PARAM_NAMES,
        _get_active_indices,
    )
    idx = _CONFIG_PATH_TO_PARAM_IDX["scoring.quality_low_vol_blend"]
    assert PARAM_NAMES[idx] == "quality_low_vol_blend"
    # Frozen by default in both scopes (no preset) -> not in active indices.
    assert idx not in _get_active_indices(scope="active_sleeve_compounding")
    assert idx not in _get_active_indices(scope="overall_strategy")


def test_lowvol_blend_zero_is_behavior_preserving():
    """slot48=0.0 must leave the composite score identical to no slot at all."""
    import pickle

    from backtesting.simulator import get_default_params, score_stocks_at_day

    try:
        PC = pickle.load(open(".session_tmp/substrate_1550.pkl", "rb"))
    except FileNotFoundError:
        import pytest
        pytest.skip("pinned substrate not present")

    base = get_default_params()
    day = 300
    s_base = score_stocks_at_day(PC, base, day)
    # extend to slot 48 with blend = 0.0
    ext = np.concatenate([base, np.zeros(49 - len(base))])
    s_ext = score_stocks_at_day(PC, ext, day)
    np.testing.assert_array_almost_equal(s_base, s_ext)


def test_lowvol_score_helper_ranks_low_vol_highest():
    import pickle

    from backtesting.simulator import _low_vol_score_at_day

    try:
        PC = pickle.load(open(".session_tmp/substrate_1550.pkl", "rb"))
    except FileNotFoundError:
        import pytest
        pytest.skip("pinned substrate not present")

    day = 400
    score = _low_vol_score_at_day(PC, day)
    vol = np.asarray(PC.vol_3m_daily[day], dtype=float)
    finite = np.isfinite(vol) & np.isfinite(score)
    if finite.sum() < 10:
        import pytest
        pytest.skip("insufficient vol coverage at test day")
    # lowest-vol names should have the highest score: rank correlation negative
    from scipy.stats import spearmanr
    rho = spearmanr(vol[finite], score[finite]).correlation
    assert rho < -0.5
