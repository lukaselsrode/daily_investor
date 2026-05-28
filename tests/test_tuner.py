"""
tests/test_tuner.py — Tuner tests.

Migrated / adapted from src/tests.py (tuner-specific tests).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

try:
    import numpy as np

    from tuning.constants import (
        PARAM_NAMES,
        _current_params,
        _effective_bounds,
        _expand_params,
        _get_active_indices,
    )
    _HAS_TUNER = True
except Exception:
    _HAS_TUNER = False

from util import TUNING_PARAMS


@pytest.mark.skipif(not _HAS_TUNER, reason="tuner.py not importable")
class TestActiveIndices:

    def test_frozen_params_excluded(self):
        active = _get_active_indices()
        frozen_paths = TUNING_PARAMS["frozen_parameters"]
        # Active set should NOT contain any frozen parameter indices
        from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX
        frozen_indices = {_CONFIG_PATH_TO_PARAM_IDX[p] for p in frozen_paths if p in _CONFIG_PATH_TO_PARAM_IDX}
        for idx in active:
            assert idx not in frozen_indices

    def test_active_count_matches_3_free_params(self):
        active = _get_active_indices()
        # With current config: sw_quality, sw_momentum, index_pct = 3 active
        assert len(active) == 3

    def test_active_param_names(self):
        active = _get_active_indices()
        active_names = [PARAM_NAMES[i] for i in active]
        assert "sw_quality" in active_names
        assert "sw_momentum" in active_names
        assert "index_pct" in active_names


@pytest.mark.skipif(not _HAS_TUNER, reason="tuner.py not importable")
class TestExpandParams:

    def test_expand_roundtrip(self):
        active = _get_active_indices()
        frozen_vals = _current_params()
        reduced = np.array([frozen_vals[i] for i in active])
        full = _expand_params(reduced, active, frozen_vals)
        np.testing.assert_array_almost_equal(full, frozen_vals)

    def test_frozen_values_preserved(self):
        active = _get_active_indices()
        frozen_vals = _current_params()
        frozen_indices = [i for i in range(len(PARAM_NAMES)) if i not in active]

        modified_reduced = np.zeros(len(active))
        full = _expand_params(modified_reduced, active, frozen_vals)

        for idx in frozen_indices:
            assert full[idx] == pytest.approx(frozen_vals[idx])


@pytest.mark.skipif(not _HAS_TUNER, reason="tuner.py not importable")
class TestEffectiveBounds:

    def test_bounds_respect_config_overrides(self):
        eff = _effective_bounds()
        from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX
        # Check that quality bounds match config
        if "score_weights.quality" in TUNING_PARAMS["parameter_bounds"]:
            q_idx = _CONFIG_PATH_TO_PARAM_IDX["score_weights.quality"]
            config_lo = TUNING_PARAMS["parameter_bounds"]["score_weights.quality"]["min"]
            config_hi = TUNING_PARAMS["parameter_bounds"]["score_weights.quality"]["max"]
            assert eff[q_idx][0] == pytest.approx(config_lo)
            assert eff[q_idx][1] == pytest.approx(config_hi)

    def test_all_bounds_lo_less_than_hi(self):
        for lo, hi in _effective_bounds():
            assert lo < hi, f"Bound {lo} >= {hi}"


@pytest.mark.skipif(not _HAS_TUNER, reason="tuner.py not importable")
class TestCurrentParams:

    def test_current_params_length(self):
        params = _current_params()
        assert len(params) == len(PARAM_NAMES)

    def test_index_pct_matches_config(self):
        from util import INDEX_PCT
        params = _current_params()
        idx_pct_idx = PARAM_NAMES.index("index_pct")
        assert params[idx_pct_idx] == pytest.approx(INDEX_PCT)
