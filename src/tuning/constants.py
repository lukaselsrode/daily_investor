"""
tuning/constants.py — Parameter space definition and config-state helpers.

Constants:
    PARAM_NAMES, BOUNDS, _CONFIG_PATH_TO_PARAM_IDX
    _MIN_TRADES_HARD, _MIN_TRADES_SOFT

Helpers (pure functions of config state, no optimizer logic):
    _effective_bounds(), _get_active_indices(), _expand_params(), _current_params()
"""

from __future__ import annotations

import numpy as np

from util import (
    ARCHETYPE_PARAMS,
    INDEX_PCT,
    METRIC_THRESHOLD,
    MOMENTUM_V2_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
    TUNING_PARAMS,
)

# ── Archetype lifecycle slot layout ────────────────────────────────────────
# 24 slots appended after slot 14 — slots 15-38 — driven by the tuner's
# `active_archetype_lifecycle` preset. The simulator reads these when
# `len(params) > 15` (otherwise it falls back to ARCHETYPE_PARAMS from config).
_ARCH_KEYS: tuple[str, ...] = (
    "quality_compounder",
    "legacy_turnaround",
    "speculative_momentum",
    "value_recovery",
    "defensive_income",
    "core_default",
)
_ARCH_SHORT: dict[str, str] = {
    "quality_compounder":   "qc",
    "legacy_turnaround":    "lt",
    "speculative_momentum": "sm",
    "value_recovery":       "vr",
    "defensive_income":     "di",
    "core_default":         "cd",
}
_ARCH_FIELDS: tuple[tuple[str, str, tuple[float, float]], ...] = (
    # (config field name, vector-name suffix, bounds)
    ("harvest_profit_threshold", "harvest", (0.10, 0.80)),
    ("trim_profit_threshold",    "trim",    (0.05, 0.50)),
    ("trailing_stop_pct",        "trail",   (-0.25, -0.04)),
    ("minimum_hold_days",        "hold",    (1.0, 60.0)),
)

_MIN_TRADES_HARD = 20
_MIN_TRADES_SOFT = 40
_MIN_TRADES_SOFT_ACTIVE = 60  # tighter bar for active_sleeve_compounding (more trades needed)

# Params unconditionally frozen when scope == "active_sleeve_compounding".
# These control passive/index allocation and must not be tuned by the active optimizer.
ACTIVE_SLEEVE_FROZEN: frozenset[str] = frozenset({
    "index_pct",
    "risk.min_index_pct",
    "regime.defensive.index_pct_override",
    "regime.neutral.index_pct_override",
    "harvest.harvest_to_etfs_pct",
    "exit_decision.trim_to_etfs_pct",
})

PARAM_NAMES: list[str] = [
    "sw_value",
    "sw_quality",
    "sw_income",
    "sw_momentum",
    "index_pct",
    "metric_threshold",
    "take_profit_pct",
    "sell_weak_below",
    "trailing_stop",
    "value_pe_weight",
    "mom_rs3m",
    "mom_rs6m",
    "mom_radj",
    "mom_trend",
    "mom_r1m",
]

BOUNDS: list[tuple[float, float]] = [
    (0.05, 0.80),
    (0.05, 0.60),
    (0.00, 0.40),
    (0.00, 0.40),
    (RISK_LIMITS["min_index_pct"], 0.95),
    (0.30, 3.00),
    (0.15, 1.00),
    (0.10, 0.90),
    (-0.30, -0.05),
    (0.30, 0.90),
    (0.00, 0.60),
    (0.00, 0.60),
    (0.00, 0.60),
    (0.00, 0.60),
    (0.00, 0.60),
]

_CONFIG_PATH_TO_PARAM_IDX: dict[str, int] = {
    "score_weights.value":                 0,
    "score_weights.quality":               1,
    "score_weights.income":                2,
    "score_weights.momentum":              3,
    "index_pct":                           4,
    "metric_threshold":                    5,
    "sell_rules.take_profit_pct":          6,
    "sell_rules.sell_weak_value_below":    7,
    "sell_rules.trailing_stop_pct":        8,
    "scoring.value_pe_weight":             9,
    "momentum_v2.weights.rs_3m":           10,
    "momentum_v2.weights.rs_6m":           11,
    "momentum_v2.weights.risk_adj_3m":     12,
    "momentum_v2.weights.trend_structure": 13,
    "momentum_v2.weights.return_1m":       14,
}

# ── Append archetype lifecycle slots 15-38 (24 entries) ────────────────────
_ARCH_SLOT_OFFSET = len(PARAM_NAMES)  # 15
for _ai, _alabel in enumerate(_ARCH_KEYS):
    _short = _ARCH_SHORT[_alabel]
    for _fi, (_field, _suffix, _bnd) in enumerate(_ARCH_FIELDS):
        _name = f"arch_{_short}_{_suffix}"
        _idx  = _ARCH_SLOT_OFFSET + _ai * len(_ARCH_FIELDS) + _fi
        PARAM_NAMES.append(_name)
        BOUNDS.append(_bnd)
        _CONFIG_PATH_TO_PARAM_IDX[f"archetype_management.{_alabel}.{_field}"] = _idx


def archetype_cfg_from_params(params) -> dict:
    """
    Build an archetype-config override dict from the lifecycle slots in *params*.
    Returns {} when params has only the original 15 slots.
    """
    if params is None or len(params) <= _ARCH_SLOT_OFFSET:
        return {}
    cfg: dict = {"enabled": True}
    for _ai, _alabel in enumerate(_ARCH_KEYS):
        entry: dict = {}
        for _fi, (_field, _, _) in enumerate(_ARCH_FIELDS):
            _idx = _ARCH_SLOT_OFFSET + _ai * len(_ARCH_FIELDS) + _fi
            if _idx < len(params):
                _v = float(params[_idx])
                if _field == "minimum_hold_days":
                    entry[_field] = int(round(_v))
                else:
                    entry[_field] = _v
        cfg[_alabel] = entry
    return cfg


def _effective_bounds(scope: str = "overall_strategy", preset: str | None = None) -> list[tuple[float, float]]:
    bounds = list(BOUNDS)
    for path, rng in TUNING_PARAMS.get("parameter_bounds", {}).items():
        idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
        if idx is None:
            continue
        lo = float(rng.get("min", bounds[idx][0]))
        hi = float(rng.get("max", bounds[idx][1]))
        bounds[idx] = (lo, min(hi, bounds[idx][1]))
    return bounds


def _archetype_default_frozen_indices() -> set[int]:
    """All archetype lifecycle slots default to frozen — they unfreeze only via an archetype preset."""
    return {
        _idx for _path, _idx in _CONFIG_PATH_TO_PARAM_IDX.items()
        if _path.startswith("archetype_management.")
    }


def _get_active_indices(scope: str = "overall_strategy", preset: str | None = None) -> list[int]:
    frozen = {
        _CONFIG_PATH_TO_PARAM_IDX[p]
        for p in TUNING_PARAMS.get("frozen_parameters", [])
        if p in _CONFIG_PATH_TO_PARAM_IDX
    }
    # Archetype lifecycle slots are frozen-by-default; an archetype preset unfreezes them.
    frozen |= _archetype_default_frozen_indices()
    if preset is not None:
        from .presets import apply_preset_to_frozen
        frozen = apply_preset_to_frozen(frozen, preset)
    if scope == "active_sleeve_compounding":
        frozen |= {
            _CONFIG_PATH_TO_PARAM_IDX[p]
            for p in ACTIVE_SLEEVE_FROZEN
            if p in _CONFIG_PATH_TO_PARAM_IDX
        }
    return [i for i in range(len(PARAM_NAMES)) if i not in frozen]


def _expand_params(reduced: np.ndarray, active: list[int], frozen_vals: np.ndarray) -> np.ndarray:
    full = frozen_vals.copy()
    for j, i in enumerate(active):
        full[i] = reduced[j]
    return full


def _current_params() -> np.ndarray:
    sw = SCORE_WEIGHTS
    v2w = MOMENTUM_V2_PARAMS.get("weights", {})
    mom_sub = [
        v2w.get("rs_3m",           0.25),
        v2w.get("rs_6m",           0.25),
        v2w.get("risk_adj_3m",     0.20),
        v2w.get("trend_structure", 0.15),
        v2w.get("return_1m",       0.10),
    ]
    base = [
        sw["value"], sw["quality"], sw["income"], sw["momentum"],
        INDEX_PCT,
        METRIC_THRESHOLD,
        SELL_RULES["take_profit_pct"],
        SELL_RULES["sell_weak_value_below"],
        SELL_RULES["trailing_stop_pct"],
        SCORING_PARAMS["value_pe_weight"],
        *mom_sub,
    ]
    # Archetype lifecycle slots 15-38 — read from ARCHETYPE_PARAMS, fall back to slot bounds midpoint
    arch_tail: list[float] = []
    for _ai, _alabel in enumerate(_ARCH_KEYS):
        entry = (ARCHETYPE_PARAMS or {}).get(_alabel, {}) or {}
        for _fi, (_field, _suffix, _bnd) in enumerate(_ARCH_FIELDS):
            _idx = _ARCH_SLOT_OFFSET + _ai * len(_ARCH_FIELDS) + _fi
            _default = entry.get(_field)
            if _default is None:
                _default = (_bnd[0] + _bnd[1]) / 2.0
            arch_tail.append(float(_default))
    return np.array(base + arch_tail)
