"""
tests/test_experiment_runner.py — CLI experiment runner smoke tests.

The full CLI invocation requires live yfinance data, so we test:
  - The 7 variants are well-formed (have name + overrides).
  - _apply_overrides correctly patches in-memory config dicts.
  - The override-then-restore pattern leaves globals untouched between variants.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import copy

from cli.commands import EXPERIMENT_VARIANTS, _apply_overrides


def test_all_seven_variants_defined():
    expected = {
        "A_baseline",
        "B_no_defensive_income_buys",
        "C_defensive_income_strict_gate",
        "D_quality_compounder_only",
        "E_cluster_cap_60",
        "F_cluster_cap_50",
        "G_cluster_cap_plus_strict_di",
    }
    assert set(EXPERIMENT_VARIANTS) == expected


def test_each_variant_has_description_and_overrides():
    for vid, spec in EXPERIMENT_VARIANTS.items():
        assert "description" in spec, f"{vid} missing description"
        assert "overrides" in spec, f"{vid} missing overrides"
        assert isinstance(spec["overrides"], dict), f"{vid} overrides not a dict"


def test_baseline_has_empty_overrides():
    assert EXPERIMENT_VARIANTS["A_baseline"]["overrides"] == {}


def test_apply_overrides_patches_concentration_warn_only():
    """E_cluster_cap_60 flips warn_only=false; verify _apply_overrides applies it."""
    from util import CONCENTRATION_LIMIT_PARAMS
    orig = copy.deepcopy(CONCENTRATION_LIMIT_PARAMS)
    try:
        _apply_overrides(EXPERIMENT_VARIANTS["E_cluster_cap_60"]["overrides"])
        assert CONCENTRATION_LIMIT_PARAMS["warn_only"] is False
        assert CONCENTRATION_LIMIT_PARAMS["max_cluster_weight"] == 0.60
    finally:
        CONCENTRATION_LIMIT_PARAMS.clear()
        CONCENTRATION_LIMIT_PARAMS.update(orig)


def test_apply_overrides_patches_archetype_classifier_gate():
    """C_defensive_income_strict_gate enables the gate; verify it applies."""
    from util import ARCHETYPE_CLASSIFIER_PARAMS
    orig = copy.deepcopy(ARCHETYPE_CLASSIFIER_PARAMS)
    try:
        _apply_overrides(EXPERIMENT_VARIANTS["C_defensive_income_strict_gate"]["overrides"])
        assert ARCHETYPE_CLASSIFIER_PARAMS["enabled"] is True
        assert ARCHETYPE_CLASSIFIER_PARAMS["defensive_income"]["require_yield"] is True
    finally:
        ARCHETYPE_CLASSIFIER_PARAMS.clear()
        ARCHETYPE_CLASSIFIER_PARAMS.update(orig)


def test_apply_overrides_patches_archetype_management():
    """B_no_defensive_income_buys disables DI archetype management."""
    from util import ARCHETYPE_PARAMS
    orig = copy.deepcopy(ARCHETYPE_PARAMS)
    try:
        _apply_overrides(EXPERIMENT_VARIANTS["B_no_defensive_income_buys"]["overrides"])
        # The path is archetype_management.defensive_income.enabled = False
        assert ARCHETYPE_PARAMS.get("defensive_income", {}).get("enabled") is False
    finally:
        ARCHETYPE_PARAMS.clear()
        ARCHETYPE_PARAMS.update(orig)


def test_apply_overrides_handles_unknown_root_safely():
    """Unknown override roots (typos) shouldn't raise; just no-op."""
    _apply_overrides({"unknown_top_level.foo.bar": 42})   # should not throw
