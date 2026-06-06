"""
tuning/presets.py — Named tuning presets for the active_sleeve_compounding scope.

Each preset names a subset of the 60-slot params vector to make TUNABLE (active) for
one auto-tune run; everything else stays frozen at config values. The optimizer only
moves the active slots, and rolling/OOS validation gates catch overfitting.

Preset families
---------------
Core scoring/exits (base slots 0-15):
  active_core_weights      — all 4 score weights
  active_exits             — metric_threshold + 3 sell rules
  active_factor_internals  — value PE weight + 6 momentum-input weights
  active_full_safe         — score weights + core exits (core_weights ∪ exits)
  active_alpha_engine      — weights + momentum internals + exits + both regime slots
  active_scoring_blends    — low-vol quality + residual-momentum blends (slots 48/49)

Selection / sizing / cadence (extended slots):
  active_candidate_filters — top_percentile, min_quality, min_momentum (40-42)
  active_position_sizing   — max_single_position_pct, max_buys, max_candidates (43-45)
  active_rebalance_cooldown— rebalance_frequency_days + post-sell/stopout cooldowns (57-59)

Regime (scoring-only):
  active_regime_tilt            — bull momentum_tilt scalar (46)
  active_regime_tilt_plus_weights — momentum_tilt + full_safe (regularized)

Exit decision:
  active_exit_floors       — DAE soft-exit floors (50-53)
  active_opportunity_cost  — stall exit thresholds (54-56)

Archetype lifecycle (slots 16-39) — one preset per archetype + combined views:
  active_quality_compounders / active_legacy_turnaround / active_speculative_momentum /
  active_value_recovery / active_defensive_income / active_core_default
  active_archetype_lifecycle (all 24) / active_archetype_rotation / active_archetype_alpha

Public API
----------
apply_preset_to_frozen(base_frozen, preset_name) -> set[int]
list_presets()                                   -> list[tuple[str, str]]
validate_preset(preset_name)                     -> None  (raises on unknown/phase2 stub)
"""
from __future__ import annotations

# All non-archetype base params (slots 0-15). apply_preset_to_frozen() seeds these
# frozen before applying a preset's `unfreeze`, so a preset opens ONLY what it lists —
# the four score weights / sell rules / momentum sub-weights can never silently drift.
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
    "scoring.momentum_inputs.weights.return_5d",
)

# Reused building blocks (keeps composite presets in sync with their parts).
_CORE_WEIGHTS = (
    "score_weights.value",
    "score_weights.quality",
    "score_weights.income",
    "score_weights.momentum",
)
_CORE_EXITS = (
    "metric_threshold",
    "sell_rules.take_profit_pct",
    "sell_rules.sell_weak_value_below",
    "sell_rules.trailing_stop_pct",
)
_MOMENTUM_INPUTS = (
    "scoring.momentum_inputs.weights.rs_3m",
    "scoring.momentum_inputs.weights.rs_6m",
    "scoring.momentum_inputs.weights.risk_adj_3m",
    "scoring.momentum_inputs.weights.trend_structure",
    "scoring.momentum_inputs.weights.return_1m",
    "scoring.momentum_inputs.weights.return_5d",
)
_ARCH_FIELDS = (
    "harvest_profit_threshold",
    "trim_profit_threshold",
    "trailing_stop_pct",
    "minimum_hold_days",
)


def _archetype_unfreeze(archetype: str) -> list[str]:
    return [f"archetype_management.{archetype}.{f}" for f in _ARCH_FIELDS]


_PRESETS: dict[str, dict] = {
    # ── Core scoring / exits ──────────────────────────────────────────────────
    "active_core_weights": {
        "description": "All 4 score weights (value, quality, income, momentum).",
        "unfreeze": list(_CORE_WEIGHTS),
        "phase2": False,
    },
    "active_exits": {
        "description": "Exit / stop / threshold rules (metric_threshold, take_profit, "
                       "sell_weak_value_below, trailing_stop). Score weights stay at defaults.",
        "unfreeze": list(_CORE_EXITS),
        "phase2": False,
    },
    "active_factor_internals": {
        "description": "Value PE weight + all 6 momentum-input weights. "
                       "Score weights stay at defaults.",
        "unfreeze": ["scoring.factors.value.pe_weight", *_MOMENTUM_INPUTS],
        "phase2": False,
    },
    "active_full_safe": {
        "description": "Score weights + core exit rules combined (core_weights ∪ exits). "
                       "Leaves momentum-input internals frozen. Conservative but broad.",
        "unfreeze": [*_CORE_WEIGHTS, *_CORE_EXITS],
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
            *_CORE_WEIGHTS,
            *_CORE_EXITS,
            *_MOMENTUM_INPUTS,
            "regime.bullish.momentum_tilt",
            "regime.defensive.mean_reversion_blend",
        ],
        "phase2": False,
    },
    "active_scoring_blends": {
        "description": "Price-derived factor blends: scoring.quality_low_vol_blend (48) + "
                       "scoring.momentum_residual_blend (49). NOTE: both are frozen OFF in "
                       "config by design — they have genuine forward-IC but FAILED prior "
                       "multi-seed backtests on the concentrated sleeve. This preset exists "
                       "to RE-TEST them on the full universe; adopt only if OOS-robust.",
        "unfreeze": [
            "scoring.quality_low_vol_blend",
            "scoring.momentum_residual_blend",
        ],
        "phase2": False,
    },

    # ── Selection / sizing / cadence ──────────────────────────────────────────
    "active_candidate_filters": {
        "description": "Candidate selection thresholds (top_percentile, min_quality_score, "
                       "min_momentum_score). Score weights + momentum sub-weights stay frozen.",
        "unfreeze": [
            "candidate_selection.top_percentile",
            "candidate_selection.min_quality_score",
            "candidate_selection.min_momentum_score",
        ],
        "phase2": False,
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
        "phase2": False,
    },
    "active_rebalance_cooldown": {
        "description": "Rebalance cadence + re-buy cooldowns: rebalance_frequency_days "
                       "(also gates contribution timing) + cooldown_days_after_sell + "
                       "cooldown_days_after_stopout. Tunes the cadence/cooldown params the "
                       "simulator actually consumes. Everything else frozen.",
        "unfreeze": [
            "backtest.rebalance_frequency_days",
            "backtest.cooldown_days_after_sell",
            "backtest.cooldown_days_after_stopout",
        ],
        "phase2": False,
    },

    # ── Regime (scoring-only) ─────────────────────────────────────────────────
    "active_regime_tilt": {
        "description": "Regime-conditional bull aggressiveness. Unfreezes the single "
                       "regime.bullish.momentum_tilt scalar (slot 46): in confirmed-bull "
                       "regime (SPY>200DMA) shift this fraction of score weight from "
                       "value/quality/income into momentum; stay defensive otherwise. "
                       "All other params frozen at config defaults. Low-DOF, scoring-only.",
        "unfreeze": [
            "regime.bullish.momentum_tilt",
        ],
        "phase2": False,
    },
    "active_regime_tilt_plus_weights": {
        "description": "Regime tilt + a full re-tune of the base book (all 4 score weights "
                       "+ all 4 core exits = regime_tilt ∪ full_safe). Tests whether the "
                       "bull/defensive split works best alongside a re-tuned base. Higher "
                       "DOF — rolling-window stability is the overfit guard.",
        "unfreeze": [
            "regime.bullish.momentum_tilt",
            *_CORE_WEIGHTS,
            *_CORE_EXITS,
        ],
        "phase2": False,
    },

    # ── Exit decision ─────────────────────────────────────────────────────────
    "active_exit_floors": {
        "description": "DAE soft-exit floors (hard_exit_score_below, positive_momentum / "
                       "strong_quality / thesis_intact review floors). These decide when a "
                       "score-below-threshold position is held vs. fully exited. Everything "
                       "else (weights, exit knobs, momentum internals) stays frozen.",
        "unfreeze": [
            "exit_decision.hard_exit_score_below",
            "exit_decision.positive_momentum_review_floor",
            "exit_decision.strong_quality_review_floor",
            "exit_decision.thesis_intact_review_floor",
        ],
        "phase2": False,
    },
    "active_opportunity_cost": {
        "description": "Opportunity-cost stall exit (max hold WITHOUT progress): "
                       "stall_max_days, reclaim_band, progress_momentum_floor. Culls "
                       "stalled active names to recycle capital; never cuts a progressing "
                       "winner. Takes effect only when exit_decision.opportunity_cost.enabled "
                       "is true in config (set it for the tuning run). Everything else frozen.",
        "unfreeze": [
            "exit_decision.opportunity_cost.stall_max_days",
            "exit_decision.opportunity_cost.reclaim_band",
            "exit_decision.opportunity_cost.progress_momentum_floor",
        ],
        "phase2": False,
    },

    # ── Archetype lifecycle (slots 16-39) — one preset per archetype ──────────
    "active_quality_compounders": {
        "description": "Test whether quality_compounder archetype can beat SPY with "
                       "longer holds, wider stops, and higher harvest thresholds.",
        "unfreeze": _archetype_unfreeze("quality_compounder"),
        "phase2": False,
    },
    "active_legacy_turnaround": {
        "description": "Test whether legacy_turnaround names earn their place with tighter "
                       "harvests/stops and shorter holds. Tunes legacy_turnaround lifecycle.",
        "unfreeze": _archetype_unfreeze("legacy_turnaround"),
        "phase2": False,
    },
    "active_speculative_momentum": {
        "description": "Test whether momentum/speculative names can create alpha when "
                       "capped and exited quickly. Tunes spec_momentum lifecycle only.",
        "unfreeze": _archetype_unfreeze("speculative_momentum"),
        "phase2": False,
    },
    "active_value_recovery": {
        "description": "Test whether value/recovery names work when filtered for "
                       "quality and improving momentum. Tunes value_recovery lifecycle.",
        "unfreeze": _archetype_unfreeze("value_recovery"),
        "phase2": False,
    },
    "active_defensive_income": {
        "description": "Test whether income names actually add active alpha or should "
                       "just be ETF exposure. Tunes defensive_income lifecycle only.",
        "unfreeze": _archetype_unfreeze("defensive_income"),
        "phase2": False,
    },
    "active_core_default": {
        "description": "Tune the core_default (fallback) archetype lifecycle — the policy "
                       "applied to names that match no specialised archetype.",
        "unfreeze": _archetype_unfreeze("core_default"),
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
            for f in _ARCH_FIELDS
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
        "phase2": False,
    },

    # ── Interaction-cluster meta-presets ──────────────────────────────────────
    # Curated unions of parameters that CO-DETERMINE one decision surface, so they
    # are best tuned jointly (tuning one in isolation pushes a local optimum that
    # conflicts with the others). These pre-package the recommended joins; the
    # '+' compose syntax covers ad-hoc combinations. See the interaction screener.
    "active_buy_gate": {
        "description": "Buy-gate cluster — what enters the book: all 4 score weights + "
                       "metric_threshold + candidate filters (top_percentile / min_quality / "
                       "min_momentum). Weights change the score distribution the bar/filters gate.",
        "unfreeze": [
            *_CORE_WEIGHTS,
            "metric_threshold",
            "candidate_selection.top_percentile",
            "candidate_selection.min_quality_score",
            "candidate_selection.min_momentum_score",
        ],
        "phase2": False,
    },
    "active_momentum_engine": {
        "description": "Momentum-emphasis cluster: momentum score weight + all 6 momentum "
                       "sub-weights + regime bull momentum_tilt + residual-momentum blend. The "
                       "tilt re-weights momentum, so these co-load it (double-counting risk).",
        "unfreeze": [
            "score_weights.momentum",
            *_MOMENTUM_INPUTS,
            "regime.bullish.momentum_tilt",
            "scoring.momentum_residual_blend",
        ],
        "phase2": False,
    },
    "active_exit_ladder": {
        "description": "Exit-ladder cluster — everything that fires on a held position: "
                       "take_profit + sell_weak + trailing_stop + DAE exit floors (4) + "
                       "opportunity-cost stall exit (3). Thresholds and firing order interact.",
        "unfreeze": [
            "sell_rules.take_profit_pct",
            "sell_rules.sell_weak_value_below",
            "sell_rules.trailing_stop_pct",
            "exit_decision.hard_exit_score_below",
            "exit_decision.positive_momentum_review_floor",
            "exit_decision.strong_quality_review_floor",
            "exit_decision.thesis_intact_review_floor",
            "exit_decision.opportunity_cost.stall_max_days",
            "exit_decision.opportunity_cost.reclaim_band",
            "exit_decision.opportunity_cost.progress_momentum_floor",
        ],
        "phase2": False,
    },
    "active_breadth_turnover": {
        "description": "Breadth/turnover cluster: position sizing (max_single_position, "
                       "max_buys, max_candidates) + candidate filters + rebalance cadence / "
                       "cooldowns + trailing_stop. Jointly set concentration and turnover.",
        "unfreeze": [
            "risk.max_single_position_pct",
            "risk.max_buys_per_rebalance",
            "candidate_selection.max_candidates",
            "candidate_selection.top_percentile",
            "candidate_selection.min_quality_score",
            "candidate_selection.min_momentum_score",
            "backtest.rebalance_frequency_days",
            "backtest.cooldown_days_after_sell",
            "backtest.cooldown_days_after_stopout",
            "sell_rules.trailing_stop_pct",
        ],
        "phase2": False,
    },
    "active_quality_stack": {
        "description": "Quality cluster — the quality tilt across score/gate/exit: quality "
                       "score weight + low-vol quality blend + candidate min_quality + the "
                       "strong_quality review floor.",
        "unfreeze": [
            "score_weights.quality",
            "scoring.quality_low_vol_blend",
            "candidate_selection.min_quality_score",
            "exit_decision.strong_quality_review_floor",
        ],
        "phase2": False,
    },
}


def split_preset_names(preset: str) -> list[str]:
    """
    Split a (possibly composed) preset spec into individual names. Presets compose
    with '+' or ',' — e.g. "active_exits+active_exit_floors" tunes the UNION of both
    surfaces in one run. A single name returns a one-element list.
    """
    return [p.strip() for p in preset.replace(",", "+").split("+") if p.strip()]


def validate_preset(preset_name: str) -> None:
    """
    Validate a preset spec (single or composed). Raise ValueError for an unknown
    name / empty spec, NotImplementedError for a Phase 2 stub.
    """
    names = split_preset_names(preset_name)
    if not names:
        raise ValueError(f"Empty preset spec: {preset_name!r}")
    for name in names:
        if name not in _PRESETS:
            known = ", ".join(sorted(_PRESETS))
            raise ValueError(f"Unknown preset {name!r}. Available: {known}")
        if _PRESETS[name].get("phase2"):
            raise NotImplementedError(
                f"Preset {name!r} is a Phase 2 stub (its config params are not yet "
                "wired into the params vector)."
            )


def apply_preset_to_frozen(base_frozen: set[int], preset_name: str) -> set[int]:
    """
    Return a modified frozen-index set after applying the named preset spec.

    Accepts a single preset or a '+'/','-composed spec (e.g.
    "active_buy_gate+active_exit_ladder"), in which case the UNION of every listed
    preset's `unfreeze` surface is opened — this is how interacting parameter groups
    are co-tuned in one run.

    Self-contained semantics: each preset's `unfreeze` list is the AUTHORITATIVE
    definition of its tunable surface. We seed the frozen set with the full base
    tunable space (all _NON_ARCHETYPE_PATHS) plus whatever was already frozen, then
    unfreeze exactly what the listed presets union to. OOS validation gates catch
    overfitting; the DOF advisory warns on large composed surfaces.
    """
    from .constants import _CONFIG_PATH_TO_PARAM_IDX

    validate_preset(preset_name)
    names = split_preset_names(preset_name)

    # Seed with the full base tunable set frozen, so a preset only opens what it
    # explicitly unfreezes (no reliance on the global config frozen list).
    result = set(base_frozen)
    for path in _NON_ARCHETYPE_PATHS:
        idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
        if idx is not None:
            result.add(idx)

    for name in names:
        for path in _PRESETS[name].get("unfreeze", []):
            idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
            if idx is not None:
                result.discard(idx)

    return result


def list_presets() -> list[tuple[str, str]]:
    """Return (name, description) pairs for all presets."""
    return [(name, spec["description"]) for name, spec in _PRESETS.items()]
