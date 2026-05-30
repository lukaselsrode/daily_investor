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

Phase 2 (now wired in — slots 40-42):
  active_candidate_filters — top_percentile, min_quality_score, min_momentum_score

Phase 2 stubs (still NotImplementedError; require further vector extension):
  active_rebalance_cooldown
  active_position_sizing

Public API
----------
apply_preset_to_frozen(base_frozen, preset_name) -> set[int]
list_presets()                                   -> list[tuple[str, str]]
validate_preset(preset_name)                     -> None  (raises on unknown/phase2)
"""
from __future__ import annotations

# All non-archetype tunable params. Archetype presets re-freeze this entire list
# via `freeze_extra` so the optimizer only moves archetype lifecycle slots —
# otherwise the four score weights / sell rules / momentum sub-weights drift
# (they are not in the base frozen_parameters list) and dominate the result.
_NON_ARCHETYPE_PATHS: tuple[str, ...] = (
    "score_weights.value",
    "score_weights.quality",
    "score_weights.income",
    "score_weights.momentum",
    "index_pct",
    "metric_threshold",
    "sell_rules.take_profit_pct",
    "sell_rules.sell_weak_value_below",
    "sell_rules.trailing_stop_pct",
    "scoring.factors.value.pe_weight",
    "scoring.momentum_inputs.weights.rs_3m",
    "scoring.momentum_inputs.weights.rs_6m",
    "scoring.momentum_inputs.weights.risk_adj_3m",
    "scoring.momentum_inputs.weights.trend_structure",
    "scoring.momentum_inputs.weights.return_1m",
)


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
        "description": "Value PE weight + all 6 momentum-input weights. "
                       "Score weights stay at defaults.",
        "unfreeze": [
            "scoring.factors.value.pe_weight",
            "scoring.momentum_inputs.weights.rs_3m",
            "scoring.momentum_inputs.weights.rs_6m",
            "scoring.momentum_inputs.weights.risk_adj_3m",
            "scoring.momentum_inputs.weights.trend_structure",
            "scoring.momentum_inputs.weights.return_1m",
            "scoring.momentum_inputs.weights.return_5d",
        ],
        "freeze_extra": [
            "score_weights.quality",
            "score_weights.momentum",
        ],
        "phase2": False,
    },
    "active_full_safe": {
        "description": "Score weights + exit rules combined. "
                       "Leaves momentum-input internals frozen. Conservative but broad.",
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

    "active_candidate_filters": {
        "description": "Candidate selection thresholds (top_percentile, min_quality_score, "
                       "min_momentum_score). Score weights + momentum sub-weights stay frozen.",
        "unfreeze": [
            "candidate_selection.top_percentile",
            "candidate_selection.min_quality_score",
            "candidate_selection.min_momentum_score",
        ],
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
        "phase2": False,
    },

    "active_alpha_engine": {
        "description": "Full high-risk alpha engine: momentum-led selection in bull "
                       "(regime momentum tilt) + contrarian mean-reversion in fear regimes "
                       "(regime mean_reversion_blend), with ride-winners exits. Unfreezes "
                       "score weights, all momentum sub-weights, exits, and both regime "
                       "slots (46+47). Highest DOF — rolling/non-overlapping windows are "
                       "the overfit guard. This is the 'chase alpha, convert to beta' sleeve.",
        "unfreeze": [
            "score_weights.value",
            "score_weights.quality",
            "score_weights.income",
            "score_weights.momentum",
            "metric_threshold",
            "sell_rules.take_profit_pct",
            "sell_rules.sell_weak_value_below",
            "sell_rules.trailing_stop_pct",
            "scoring.momentum_inputs.weights.rs_3m",
            "scoring.momentum_inputs.weights.rs_6m",
            "scoring.momentum_inputs.weights.risk_adj_3m",
            "scoring.momentum_inputs.weights.trend_structure",
            "scoring.momentum_inputs.weights.return_1m",
            "scoring.momentum_inputs.weights.return_5d",
            "regime.bullish.momentum_tilt",
            "regime.defensive.mean_reversion_blend",
        ],
        "freeze_extra": [],
        "phase2": False,
    },

    "active_regime_tilt": {
        "description": "Regime-conditional bull aggressiveness. Unfreezes the single "
                       "regime.bullish.momentum_tilt scalar (slot 46): in confirmed-bull "
                       "regime (SPY>200DMA) shift this fraction of score weight from "
                       "value/quality/income into momentum; stay defensive otherwise. "
                       "All other params frozen at config defaults. Low-DOF, scoring-only.",
        "unfreeze": [
            "regime.bullish.momentum_tilt",
        ],
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
        "phase2": False,
    },

    "active_regime_tilt_plus_weights": {
        "description": "Regime tilt + base score weights + core exits. Tests whether the "
                       "bull/defensive split works best alongside a re-tuned base book. "
                       "Higher DOF — rolling-window stability is the overfit guard.",
        "unfreeze": [
            "regime.bullish.momentum_tilt",
            "score_weights.value",
            "score_weights.income",
            "metric_threshold",
            "sell_rules.take_profit_pct",
            "sell_rules.trailing_stop_pct",
        ],
        "freeze_extra": [],
        "phase2": False,
    },

    # ── Phase 2 stubs ─────────────────────────────────────────────────────────
    "active_rebalance_cooldown": {
        "description": "Rebalance frequency + cooldown days. "
                       "Requires Phase 2 vector extension.",
        "phase2": True,
    },
    "active_position_sizing": {
        "description": "Breadth / sizing — does the active sleeve do better concentrated "
                       "or broader? Tunes max_single_position_pct, max_buys_per_rebalance, "
                       "and candidate-pool max_candidates. Everything else frozen.",
        "unfreeze": [
            "risk.max_single_position_pct",
            "risk.max_buys_per_rebalance",
            "candidate_selection.max_candidates",
        ],
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
        "phase2": False,
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
        "freeze_extra": list(_NON_ARCHETYPE_PATHS),
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
