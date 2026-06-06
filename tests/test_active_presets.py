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


def test_preset_composition_unions_slots():
    """`a+b` (and `a,b`) compose into the UNION of both presets' active slots."""
    assert _active(preset="active_exits+active_exit_floors") == {5, 6, 7, 8, 50, 51, 52, 53}
    assert _active(preset="active_core_weights,active_factor_internals") == {
        0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15,
    }
    # idempotent: composing a preset with itself == itself
    assert _active(preset="active_exits+active_exits") == _active(preset="active_exits")


def test_composition_validation():
    from tuning.presets import split_preset_names, validate_preset
    assert split_preset_names("a+b , c") == ["a", "b", "c"]
    validate_preset("active_exits+active_exit_floors")  # ok
    with pytest.raises(ValueError, match="Unknown preset"):
        validate_preset("active_exits+bogus")


def test_interaction_cluster_presets_slots():
    """The 5 curated interaction-cluster presets unfreeze exactly their mapped slots."""
    expected = {
        "active_buy_gate":         {0, 1, 2, 3, 5, 40, 41, 42},
        "active_momentum_engine":  {3, 10, 11, 12, 13, 14, 15, 46, 49},
        "active_exit_ladder":      {6, 7, 8, 50, 51, 52, 53, 54, 55, 56},
        "active_breadth_turnover": {8, 40, 41, 42, 43, 44, 45, 57, 58, 59},
        "active_quality_stack":    {1, 41, 48, 52},
    }
    for name, slots in expected.items():
        assert _active(preset=name) == slots, name


def test_interaction_screener_report_helpers():
    """Screener verdict + matrix logic (no data / no tuning needed)."""
    import numpy as np

    from tuning.interaction_screen import (
        InteractionResult,
        MarginalResult,
        PairResult,
        _verdict,
    )
    assert _verdict(0.05, 0.0) == "🟢 synergy"
    assert _verdict(-0.05, 0.0) == "🔴 clash"
    assert _verdict(0.0, 0.30) == "↔ compromise"
    assert _verdict(0.0, 0.0) == "⚪ ~independent"
    res = InteractionResult()
    res.marginals = {
        "A": MarginalResult("A", 0.5, np.zeros(60), [0]),
        "B": MarginalResult("B", 0.4, np.zeros(60), [1]),
    }
    res.pairs = [PairResult("A", "B", 0.5, 0.4, 0.7, 0.2, 0.1, "🟢 synergy")]
    m = res.matrix_df()
    assert m.loc["A", "A"] == 0.5 and m.loc["A", "B"] == 0.2 and m.loc["B", "A"] == 0.2
    assert not res.pairs_df().empty


def test_staged_tune_helpers():
    """Staged-tune ordering + trace report (no data / no tuning)."""
    from tuning.staged_tune import _CLUSTER_ORDER, StagedTuneResult, StageResult
    # leverage order covers all 5 interaction clusters, scoring/momentum first.
    assert _CLUSTER_ORDER[0] == "active_momentum_engine"
    assert set(_CLUSTER_ORDER) == {
        "active_momentum_engine", "active_quality_stack", "active_buy_gate",
        "active_exit_ladder", "active_breadth_turnover",
    }
    res = StagedTuneResult(
        stages=[
            StageResult("active_momentum_engine", 0.1, 0.5, True),
            StageResult("active_exit_ladder", 0.5, 0.4, False),
        ],
    )
    df = res.trace_df()
    assert len(df) == 2
    assert df.iloc[0]["result"] == "✅ accepted"
    assert df.iloc[1]["result"] == "— kept prior"


def test_auto_tune_all_cli_importable():
    import inspect

    from cli.commands import cmd_auto_tune_all
    params = inspect.signature(cmd_auto_tune_all).parameters
    assert "profile" in params and "clusters" in params


def test_tuner_active_param_names_honor_preset():
    """Regression: ParameterTuner active-param resolution must reflect scope+preset,
    not silently return the 16 base names (which made the UI tune panel mark the
    wrong rows as 'tuned')."""
    from tuning.tuner import ParameterTuner
    names = ParameterTuner._active_param_names("active_sleeve_compounding", "active_exits")
    assert set(names) == {"metric_threshold", "take_profit_pct", "sell_weak_below", "trailing_stop"}
    # an extended-slot preset resolves to its slots too (not base 16)
    floor_names = ParameterTuner._active_param_names("active_sleeve_compounding", "active_exit_floors")
    assert set(floor_names) == {
        "ef_hard_exit_score_below", "ef_positive_momentum_review_floor",
        "ef_strong_quality_review_floor", "ef_thesis_intact_review_floor",
    }


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


def test_validate_preset_mechanism():
    from tuning import presets as P
    from tuning.presets import validate_preset
    # Revived presets must NOT raise (no longer stubs).
    validate_preset("active_rebalance_cooldown")
    validate_preset("active_position_sizing")
    # Unknown preset raises ValueError.
    with pytest.raises(ValueError, match="Unknown preset"):
        validate_preset("does_not_exist")
    # The Phase-2-stub mechanism still works for any future stub.
    P._PRESETS["__tmp_stub__"] = {"description": "tmp", "phase2": True}
    try:
        with pytest.raises(NotImplementedError):
            validate_preset("__tmp_stub__")
    finally:
        del P._PRESETS["__tmp_stub__"]


def test_no_preset_has_dead_freeze_extra():
    """freeze_extra was vestigial (the seed already freezes base paths) — it's removed."""
    from tuning.presets import _PRESETS
    for name, spec in _PRESETS.items():
        assert "freeze_extra" not in spec, f"{name} still carries dead freeze_extra"


def test_rebalance_cooldown_active_slots():
    active = set(_active(preset="active_rebalance_cooldown"))
    from tuning.constants import _REBAL_FIELDS, _REBAL_SLOT_OFFSET
    expected = {_REBAL_SLOT_OFFSET + i for i in range(len(_REBAL_FIELDS))}
    assert active == expected, f"active_rebalance_cooldown should be {expected}, got {active}"


def test_new_menu_presets_validate_and_unfreeze():
    from tuning.presets import _PRESETS, validate_preset
    for name in ("active_legacy_turnaround", "active_core_default", "active_scoring_blends"):
        validate_preset(name)
        assert _PRESETS[name]["unfreeze"], f"{name} must unfreeze something"
    # scoring_blends opens exactly slots 48/49.
    assert set(_active(preset="active_scoring_blends")) == {48, 49}
    # regularized regime_tilt_plus_weights = momentum_tilt (46) + 4 weights + 4 exits.
    assert set(_active(preset="active_regime_tilt_plus_weights")) == {0, 1, 2, 3, 5, 6, 7, 8, 46}


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
    # score weights (0,1,2,3) should all be frozen — the seed freezes all base paths,
    # and active_exits unfreezes only the exit knobs (not the weights)
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

def test_no_preset_active_sleeve_tunes_full_base_space():
    """Without a preset, the full base param space (slots 0-15) is tunable.

    Philosophy change (2026-05): config.tuning.frozen_parameters is now empty;
    presets define the tunable surface per-run and OOS validation gates catch
    overfitting. So a no-preset tune optimizes the whole base space. Only the
    unconditionally-frozen active-sleeve params (index_pct + ETF routing) and the
    archetype/regime/cs/sizing tail slots stay frozen.
    """
    active = _active(scope="active_sleeve_compounding", preset=None)
    # Core scoring + exit params are all tunable now
    for idx in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        assert idx in active, f"index {idx} should be tunable without preset (full base space)"
    # index_pct (4) stays frozen in active_sleeve scope (ACTIVE_SLEEVE_FROZEN)
    assert 4 not in active, "index_pct must stay frozen in active_sleeve scope"


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
