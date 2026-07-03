"""
tests/test_peer2_tuning.py — peer-2 tuner slots (mom_rel_volume / value_benchmark_blend).

Covers:
  1. Slot layout: the two add-on slots are appended LAST and their indices match
     the literal the simulator uses (simulator hardcodes it to avoid a module-level
     tuning import cycle — this test is the pin that keeps them in sync).
  2. Pre-existing named slots keep their historical indices (no shift).
  3. _current_params seeds the add-on slots from live config and stays inside BOUNDS.
  4. Default-frozen: the slots are inactive under every scope without a preset;
     the presets that list them unfreeze exactly them.
  5. Simulator weight helper: legacy/short vectors degrade to rel_volume weight 0.
"""

from __future__ import annotations

import numpy as np

from backtesting import simulator
from tuning.constants import (
    _CONFIG_PATH_TO_PARAM_IDX,
    _REL_VOLUME_SLOT,
    _VALUE_BENCH_SLOT,
    BOUNDS,
    PARAM_NAMES,
    _current_params,
    _get_active_indices,
)
from tuning.presets import apply_preset_to_frozen


def test_addon_slots_are_last_and_pinned_to_simulator_literal():
    assert PARAM_NAMES.index("mom_rel_volume") == _REL_VOLUME_SLOT
    assert PARAM_NAMES.index("value_benchmark_blend") == _VALUE_BENCH_SLOT
    assert {_REL_VOLUME_SLOT, _VALUE_BENCH_SLOT} == {len(PARAM_NAMES) - 2, len(PARAM_NAMES) - 1}
    # The simulator's hardcoded literal must match the constructed index.
    assert simulator._REL_VOLUME_SLOT == _REL_VOLUME_SLOT


def test_preexisting_slots_did_not_shift():
    # Slots the simulator/config reader hardcode; a shift silently corrupts tunes.
    assert PARAM_NAMES[0:4] == ["sw_value", "sw_quality", "sw_income", "sw_momentum"]
    assert _CONFIG_PATH_TO_PARAM_IDX["regime.bullish.momentum_tilt"] == 46
    assert _CONFIG_PATH_TO_PARAM_IDX["regime.defensive.mean_reversion_blend"] == 47
    assert _CONFIG_PATH_TO_PARAM_IDX["scoring.quality_low_vol_blend"] == 48
    assert _CONFIG_PATH_TO_PARAM_IDX["scoring.momentum_residual_blend"] == 49


def test_current_params_seeds_addon_slots_within_bounds():
    from util import SCORING_PARAMS
    cur = _current_params()
    assert len(cur) == len(PARAM_NAMES) == len(BOUNDS)
    assert cur[_REL_VOLUME_SLOT] == float(
        SCORING_PARAMS["momentum_inputs"]["weights"].get("rel_volume", 0.0)
    )
    assert cur[_VALUE_BENCH_SLOT] == float(
        SCORING_PARAMS["factors"]["value"].get("benchmark_blend", 0.0)
    )
    for slot in (_REL_VOLUME_SLOT, _VALUE_BENCH_SLOT):
        lo, hi = BOUNDS[slot]
        assert lo <= cur[slot] <= hi


def test_addon_slots_frozen_by_default_and_preset_unfrozen():
    for scope in ("overall_strategy", "active_sleeve_compounding"):
        active = set(_get_active_indices(scope))
        assert _REL_VOLUME_SLOT not in active
        assert _VALUE_BENCH_SLOT not in active

    all_frozen = set(range(len(PARAM_NAMES)))
    after_vb = apply_preset_to_frozen(set(all_frozen), "active_value_benchmark")
    assert _VALUE_BENCH_SLOT not in after_vb
    assert _REL_VOLUME_SLOT in after_vb

    after_fi = apply_preset_to_frozen(set(all_frozen), "active_factor_internals")
    assert _REL_VOLUME_SLOT not in after_fi


def test_mom_weight_helper_degrades_on_short_vectors():
    cur = _current_params()
    short = cur[:50]
    w = simulator._mom_weights_with_rel_volume(short)
    assert len(w) == 7 and w[6] == 0.0
    np.testing.assert_allclose(w[:6], np.asarray(short[10:16], dtype=float))

    full = cur.copy()
    full[_REL_VOLUME_SLOT] = 0.25
    w2 = simulator._mom_weights_with_rel_volume(full)
    assert w2[6] == 0.25
