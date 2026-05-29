"""
tuning/presets.py — Named tuning presets for active_sleeve_compounding scope.

Each preset controls which of the 15-element params vector is active (tunable)
for a given run. Presets override the config frozen_parameters list — unfreezing
some params, optionally re-freezing others — before ACTIVE_SLEEVE_FROZEN is applied.

Phase 1 presets (working, no vector extension needed):
  active_core_weights     — all 4 score weights
  active_exits            — metric_threshold + 3 sell rules
  active_factor_internals — value PE weight + 5 momentum sub-weights
  active_full_safe        — score weights + exits combined

Phase 2 stubs (raise NotImplementedError; require vector extension):
  active_candidate_filters
  active_rebalance_cooldown
  active_position_sizing

Public API
----------
apply_preset_to_frozen(base_frozen, preset_name) -> set[int]
list_presets()                                   -> list[tuple[str, str]]
validate_preset(preset_name)                     -> None  (raises on unknown/phase2)
"""
from __future__ import annotations

_PRESETS: dict[str, dict] = {
    # ── Phase 1 — working ────────────────────────────────────────────────────
    "active_core_weights": {
        "description": "All 4 score weights (value, quality, income, momentum). "
                       "Unfreezes value + income from the global frozen list.",
        "unfreeze": [
            "score_weights.value",
            "score_weights.income",
        ],
        "freeze_extra": [],
        "phase2": False,
    },
    "active_exits": {
        "description": "Exit / stop / threshold rules (metric_threshold, take_profit, "
                       "sell_weak_value_below, trailing_stop). Score weights stay at defaults.",
        "unfreeze": [
            "metric_threshold",
            "sell_rules.take_profit_pct",
            "sell_rules.sell_weak_value_below",
            "sell_rules.trailing_stop_pct",
        ],
        "freeze_extra": [
            "score_weights.quality",
            "score_weights.momentum",
        ],
        "phase2": False,
    },
    "active_factor_internals": {
        "description": "Value PE weight + all 5 momentum_v2 sub-weights. "
                       "Score weights stay at defaults.",
        "unfreeze": [
            "scoring.value_pe_weight",
            "momentum_v2.weights.rs_3m",
            "momentum_v2.weights.rs_6m",
            "momentum_v2.weights.risk_adj_3m",
            "momentum_v2.weights.trend_structure",
            "momentum_v2.weights.return_1m",
        ],
        "freeze_extra": [
            "score_weights.quality",
            "score_weights.momentum",
        ],
        "phase2": False,
    },
    "active_full_safe": {
        "description": "Score weights + exit rules combined. "
                       "Leaves momentum_v2 internals frozen. Conservative but broad.",
        "unfreeze": [
            "score_weights.value",
            "score_weights.income",
            "metric_threshold",
            "sell_rules.take_profit_pct",
            "sell_rules.sell_weak_value_below",
            "sell_rules.trailing_stop_pct",
        ],
        "freeze_extra": [],
        "phase2": False,
    },

    # ── Phase 2 stubs ─────────────────────────────────────────────────────────
    "active_candidate_filters": {
        "description": "Candidate selection thresholds (top_percentile, min_quality_score, "
                       "min_momentum_score). Requires Phase 2 vector extension.",
        "phase2": True,
    },
    "active_rebalance_cooldown": {
        "description": "Rebalance frequency + cooldown days. "
                       "Requires Phase 2 vector extension.",
        "phase2": True,
    },
    "active_position_sizing": {
        "description": "max_single_position_pct + max_buys_per_rebalance. "
                       "Requires Phase 2 vector extension.",
        "phase2": True,
    },

    # ── Archetype-targeted presets (use lifecycle slots 15-38) ────────────────
    "active_quality_compounders": {
        "description": "Test whether quality_compounder archetype can beat SPY with "
                       "longer holds, wider stops, and higher harvest thresholds.",
        "unfreeze": [
            "archetype_management.quality_compounder.harvest_profit_threshold",
            "archetype_management.quality_compounder.trim_profit_threshold",
            "archetype_management.quality_compounder.trailing_stop_pct",
            "archetype_management.quality_compounder.minimum_hold_days",
        ],
        "freeze_extra": [],
        "phase2": False,
    },
    "active_speculative_momentum": {
        "description": "Test whether momentum/speculative names can create alpha when "
                       "capped and exited quickly. Tunes spec_momentum lifecycle only.",
        "unfreeze": [
            "archetype_management.speculative_momentum.harvest_profit_threshold",
            "archetype_management.speculative_momentum.trim_profit_threshold",
            "archetype_management.speculative_momentum.trailing_stop_pct",
            "archetype_management.speculative_momentum.minimum_hold_days",
        ],
        "freeze_extra": [],
        "phase2": False,
    },
    "active_value_recovery": {
        "description": "Test whether value/recovery names work when filtered for "
                       "quality and improving momentum. Tunes value_recovery lifecycle.",
        "unfreeze": [
            "archetype_management.value_recovery.harvest_profit_threshold",
            "archetype_management.value_recovery.trim_profit_threshold",
            "archetype_management.value_recovery.trailing_stop_pct",
            "archetype_management.value_recovery.minimum_hold_days",
        ],
        "freeze_extra": [],
        "phase2": False,
    },
    "active_defensive_income": {
        "description": "Test whether income names actually add active alpha or should "
                       "just be ETF exposure. Tunes defensive_income lifecycle only.",
        "unfreeze": [
            "archetype_management.defensive_income.harvest_profit_threshold",
            "archetype_management.defensive_income.trim_profit_threshold",
            "archetype_management.defensive_income.trailing_stop_pct",
            "archetype_management.defensive_income.minimum_hold_days",
        ],
        "freeze_extra": [],
        "phase2": False,
    },
    "active_archetype_lifecycle": {
        "description": "Tune ALL archetype lifecycle thresholds (trim/harvest/trail/hold "
                       "across all 6 archetypes — 24 params). Score weights stay frozen.",
        "unfreeze": [
            f"archetype_management.{a}.{f}"
            for a in (
                "quality_compounder", "legacy_turnaround", "speculative_momentum",
                "value_recovery", "defensive_income", "core_default",
            )
            for f in (
                "harvest_profit_threshold", "trim_profit_threshold",
                "trailing_stop_pct", "minimum_hold_days",
            )
        ],
        "freeze_extra": [
            "score_weights.value", "score_weights.quality",
            "score_weights.income", "score_weights.momentum",
        ],
        "phase2": False,
    },
    "active_archetype_rotation": {
        "description": "Test whether some archetypes should be capped, favored, or "
                       "suppressed — tunes the four most divergent archetype harvests.",
        "unfreeze": [
            "archetype_management.quality_compounder.harvest_profit_threshold",
            "archetype_management.speculative_momentum.harvest_profit_threshold",
            "archetype_management.value_recovery.harvest_profit_threshold",
            "archetype_management.defensive_income.harvest_profit_threshold",
        ],
        "freeze_extra": [],
        "phase2": False,
    },
    "active_archetype_alpha": {
        "description": "Safer combined archetype preset — tunes harvest + trailing stop "
                       "for the two highest-volume archetypes (quality_compounder + "
                       "speculative_momentum). Score weights frozen.",
        "unfreeze": [
            "archetype_management.quality_compounder.harvest_profit_threshold",
            "archetype_management.quality_compounder.trailing_stop_pct",
            "archetype_management.speculative_momentum.harvest_profit_threshold",
            "archetype_management.speculative_momentum.trailing_stop_pct",
        ],
        "freeze_extra": [
            "score_weights.value", "score_weights.quality",
            "score_weights.income", "score_weights.momentum",
        ],
        "phase2": False,
    },
}


def validate_preset(preset_name: str) -> None:
    """Raise ValueError for unknown names, NotImplementedError for Phase 2 stubs."""
    if preset_name not in _PRESETS:
        known = ", ".join(sorted(_PRESETS))
        raise ValueError(
            f"Unknown preset {preset_name!r}. Available: {known}"
        )
    if _PRESETS[preset_name].get("phase2"):
        raise NotImplementedError(
            f"Preset {preset_name!r} requires Phase 2 vector extension "
            "(candidate_selection / risk params not yet in the 15-element params vector)."
        )


def apply_preset_to_frozen(base_frozen: set[int], preset_name: str) -> set[int]:
    """
    Return a modified frozen-index set after applying the named preset.

    Unfreezes params listed in preset["unfreeze"] and re-freezes those in
    preset["freeze_extra"]. Called before ACTIVE_SLEEVE_FROZEN is applied.
    """
    from .constants import _CONFIG_PATH_TO_PARAM_IDX

    validate_preset(preset_name)
    spec = _PRESETS[preset_name]

    result = set(base_frozen)

    for path in spec.get("unfreeze", []):
        idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
        if idx is not None:
            result.discard(idx)

    for path in spec.get("freeze_extra", []):
        idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
        if idx is not None:
            result.add(idx)

    return result


def list_presets() -> list[tuple[str, str]]:
    """Return (name, description) pairs for all presets."""
    return [(name, spec["description"]) for name, spec in _PRESETS.items()]
