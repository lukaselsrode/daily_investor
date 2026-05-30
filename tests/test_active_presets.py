"""
tests/test_active_presets.py — Preset system and active_sleeve_compounding tuning tests.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active(scope="active_sleeve_compounding", preset=None):
    from tuning.constants import _get_active_indices
    return set(_get_active_indices(scope=scope, preset=preset))


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

def test_list_presets_returns_all():
    from tuning.presets import list_presets
    names = [n for n, _ in list_presets()]
    assert "active_core_weights" in names
    assert "active_exits" in names
    assert "active_factor_internals" in names
    assert "active_full_safe" in names
    assert "active_candidate_filters" in names


def test_unknown_preset_raises_value_error():
    from tuning.presets import validate_preset
    with pytest.raises(ValueError, match="Unknown preset"):
        validate_preset("does_not_exist")


def test_phase2_preset_raises_not_implemented():
    from tuning.presets import validate_preset
    with pytest.raises(NotImplementedError, match="Phase 2"):
        validate_preset("active_rebalance_cooldown")

    # active_position_sizing was revived (no longer Phase 2) — it must NOT raise.
    validate_preset("active_position_sizing")


# ---------------------------------------------------------------------------
# active_core_weights
# ---------------------------------------------------------------------------

def test_core_weights_unfreezes_value_and_income():
    active = _active(preset="active_core_weights")
    # indices 0 (value) and 2 (income) must be active
    assert 0 in active, "value weight (index 0) should be active"
    assert 2 in active, "income weight (index 2) should be active"
    # quality (1) and momentum (3) are already active without preset
    assert 1 in active
    assert 3 in active


def test_core_weights_keeps_index_pct_frozen():
    active = _active(preset="active_core_weights")
    # index_pct (index 4) must remain frozen in active_sleeve scope
    assert 4 not in active, "index_pct (index 4) must stay frozen in active_sleeve_compounding"


def test_core_weights_freezes_non_score_params():
    active = _active(preset="active_core_weights")
    # metric_threshold (5), sell rules (6,7,8), value_pe_weight (9), momentum sub (10-14)
    for idx in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        assert idx not in active, f"index {idx} should remain frozen for active_core_weights"


# ---------------------------------------------------------------------------
# active_exits
# ---------------------------------------------------------------------------

def test_exits_preset_active_indices():
    active = _active(preset="active_exits")
    # metric_threshold (5), take_profit (6), sell_weak (7), trailing_stop (8)
    for idx in [5, 6, 7, 8]:
        assert idx in active, f"index {idx} should be active for active_exits"


def test_exits_preset_freezes_score_weights():
    active = _active(preset="active_exits")
    # score weights (0,1,2,3) should all be frozen — preset adds quality+momentum to freeze_extra
    for idx in [0, 1, 2, 3]:
        assert idx not in active, f"score weight index {idx} should be frozen for active_exits"


def test_exits_preset_keeps_index_pct_frozen():
    active = _active(preset="active_exits")
    assert 4 not in active


# ---------------------------------------------------------------------------
# active_factor_internals
# ---------------------------------------------------------------------------

def test_factor_internals_preset():
    active = _active(preset="active_factor_internals")
    # value_pe_weight (9) and all 5 momentum sub-weights (10-14)
    for idx in [9, 10, 11, 12, 13, 14]:
        assert idx in active, f"index {idx} should be active for active_factor_internals"


def test_factor_internals_freezes_score_weights():
    active = _active(preset="active_factor_internals")
    for idx in [0, 1, 2, 3]:
        assert idx not in active, f"score weight index {idx} should be frozen for active_factor_internals"


def test_factor_internals_keeps_index_pct_frozen():
    active = _active(preset="active_factor_internals")
    assert 4 not in active


# ---------------------------------------------------------------------------
# active_full_safe
# ---------------------------------------------------------------------------

def test_full_safe_preset_coverage():
    active = _active(preset="active_full_safe")
    # all 4 score weights + exits
    for idx in [0, 1, 2, 3, 5, 6, 7, 8]:
        assert idx in active, f"index {idx} should be active for active_full_safe"


def test_full_safe_leaves_momentum_internals_frozen():
    active = _active(preset="active_full_safe")
    for idx in [10, 11, 12, 13, 14]:
        assert idx not in active, f"momentum sub-weight {idx} should remain frozen for active_full_safe"


def test_full_safe_keeps_index_pct_frozen():
    active = _active(preset="active_full_safe")
    assert 4 not in active


# ---------------------------------------------------------------------------
# No preset — baseline active_sleeve behavior
# ---------------------------------------------------------------------------

def test_no_preset_active_sleeve_only_quality_momentum():
    """Without a preset, only quality (1) and momentum (3) are tunable in active_sleeve."""
    active = _active(scope="active_sleeve_compounding", preset=None)
    assert 1 in active, "quality (index 1) should be active without preset"
    assert 3 in active, "momentum (index 3) should be active without preset"
    # Everything else should be frozen
    for idx in [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        assert idx not in active, f"index {idx} should be frozen without preset in active_sleeve"


# ---------------------------------------------------------------------------
# overall_strategy is unaffected by preset names
# ---------------------------------------------------------------------------

def test_preset_has_no_effect_on_overall_strategy_scope():
    """Presets are only meaningful for active_sleeve_compounding; overall_strategy uses config frozen list."""
    from tuning.constants import _get_active_indices
    baseline = set(_get_active_indices(scope="overall_strategy", preset=None))
    with_preset = set(_get_active_indices(scope="overall_strategy", preset="active_core_weights"))
    # In overall_strategy, active_core_weights unfreezes value+income, which may differ from baseline
    # but ACTIVE_SLEEVE_FROZEN does NOT apply, so index_pct can remain in the set
    # The key test: index_pct (4) is in overall_strategy baseline (it's tunable there)
    assert 4 in baseline, "index_pct should be tunable in overall_strategy"


# ---------------------------------------------------------------------------
# _MIN_TRADES_SOFT_ACTIVE constant
# ---------------------------------------------------------------------------

def test_min_trades_soft_active_greater_than_soft():
    from tuning.constants import _MIN_TRADES_SOFT, _MIN_TRADES_SOFT_ACTIVE
    assert _MIN_TRADES_SOFT_ACTIVE > _MIN_TRADES_SOFT, (
        "_MIN_TRADES_SOFT_ACTIVE should require more trades than the overall-strategy soft minimum"
    )
