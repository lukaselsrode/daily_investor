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
    CANDIDATE_SELECTION_PARAMS,
    INDEX_PCT,
    METRIC_THRESHOLD,
    REGIME_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
    TUNING_PARAMS,
)

# Local aliases for nested scoring sub-blocks (used by the param-vector seeding logic).
MOMENTUM_INPUT_PARAMS = SCORING_PARAMS["momentum_inputs"]
VALUE_FACTOR_PARAMS = SCORING_PARAMS["factors"]["value"]

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
    "sw_value",         # 0
    "sw_quality",       # 1
    "sw_income",        # 2
    "sw_momentum",      # 3
    "index_pct",        # 4
    "metric_threshold", # 5
    "take_profit_pct",  # 6
    "sell_weak_below",  # 7
    "trailing_stop",    # 8
    "value_pe_weight",  # 9
    "mom_rs3m",         # 10
    "mom_rs6m",         # 11
    "mom_radj",         # 12
    "mom_trend",        # 13
    "mom_r1m",          # 14
    "mom_r5d",          # 15  (new in this consolidation — was implicit in v2)
]

BOUNDS: list[tuple[float, float]] = [
    (0.05, 0.80),  # 0 sw_value
    (0.05, 0.60),  # 1 sw_quality
    (0.00, 0.40),  # 2 sw_income
    (0.00, 0.90),  # 3 sw_momentum (widened 0.40->0.90 for momentum-alpha engine; see active_alpha_engine preset)
    (RISK_LIMITS["min_index_pct"], 0.95),  # 4 index_pct
    (0.30, 3.00),  # 5 metric_threshold
    (0.15, 1.00),  # 6 take_profit_pct
    (0.10, 0.90),  # 7 sell_weak_below
    (-0.45, -0.05),# 8 trailing_stop (widened to -0.45 to allow ride-winners regime)
    (0.30, 0.90),  # 9 value_pe_weight
    (0.00, 0.60),  # 10 mom_rs3m
    (0.00, 0.60),  # 11 mom_rs6m
    (0.00, 0.60),  # 12 mom_radj
    (0.00, 0.60),  # 13 mom_trend
    (0.00, 0.60),  # 14 mom_r1m
    (0.00, 0.30),  # 15 mom_r5d (new)
]

_CONFIG_PATH_TO_PARAM_IDX: dict[str, int] = {
    "score_weights.value":                       0,
    "score_weights.quality":                     1,
    "score_weights.income":                      2,
    "score_weights.momentum":                    3,
    "index_pct":                                 4,
    "metric_threshold":                          5,
    "sell_rules.take_profit_pct":                6,
    "sell_rules.sell_weak_value_below":          7,
    "sell_rules.trailing_stop_pct":              8,
    "scoring.factors.value.pe_weight":           9,
    "scoring.momentum_inputs.weights.rs_3m":           10,
    "scoring.momentum_inputs.weights.rs_6m":           11,
    "scoring.momentum_inputs.weights.risk_adj_3m":     12,
    "scoring.momentum_inputs.weights.trend_structure": 13,
    "scoring.momentum_inputs.weights.return_1m":       14,
    "scoring.momentum_inputs.weights.return_5d":       15,
}

# ── Append archetype lifecycle slots 16-39 (24 entries) ────────────────────
_ARCH_SLOT_OFFSET = len(PARAM_NAMES)  # 16
for _ai, _alabel in enumerate(_ARCH_KEYS):
    _short = _ARCH_SHORT[_alabel]
    for _fi, (_field, _suffix, _bnd) in enumerate(_ARCH_FIELDS):
        _name = f"arch_{_short}_{_suffix}"
        _idx  = _ARCH_SLOT_OFFSET + _ai * len(_ARCH_FIELDS) + _fi
        PARAM_NAMES.append(_name)
        BOUNDS.append(_bnd)
        _CONFIG_PATH_TO_PARAM_IDX[f"archetype_management.{_alabel}.{_field}"] = _idx

# ── Append candidate-selection filter slots 40-42 ──────────────────────────
# Three entry-side knobs that gate which stocks become buy candidates.
# Frozen by default; the `active_candidate_filters` preset unfreezes them.
_CS_FILTER_SLOT_OFFSET = len(PARAM_NAMES)  # 40
_CS_FILTER_FIELDS: tuple[tuple[str, str, tuple[float, float]], ...] = (
    # (PARAM_NAMES entry, config path under app.candidate_selection.*, bounds)
    ("cs_top_percentile",     "top_percentile",     (0.05, 0.50)),
    ("cs_min_quality_score",  "min_quality_score",  (0.00, 0.70)),
    ("cs_min_momentum_score", "min_momentum_score", (-0.30, 0.40)),
)
for _i, (_name, _cs_field, _bnd) in enumerate(_CS_FILTER_FIELDS):
    PARAM_NAMES.append(_name)
    BOUNDS.append(_bnd)
    # Config paths follow the same `app.candidate_selection.<field>` convention
    # but we expose them as `candidate_selection.<field>` for preset readability.
    _CONFIG_PATH_TO_PARAM_IDX[f"candidate_selection.{_cs_field}"] = _CS_FILTER_SLOT_OFFSET + _i

# ── Append position-sizing slots 43-45 ─────────────────────────────────────
# Breadth / sizing knobs: max position size, buys per rebalance, candidate-pool
# size. Frozen by default; the `active_position_sizing` preset unfreezes them.
_PS_SLOT_OFFSET = len(PARAM_NAMES)  # 43
_PS_FIELDS: tuple[tuple[str, str, tuple[float, float]], ...] = (
    ("ps_max_single_position_pct", "risk.max_single_position_pct",       (0.03, 0.20)),
    ("ps_max_buys_per_rebalance",  "risk.max_buys_per_rebalance",        (3.0, 15.0)),
    ("ps_max_candidates",          "candidate_selection.max_candidates", (5.0, 50.0)),
)
for _i, (_name, _path, _bnd) in enumerate(_PS_FIELDS):
    PARAM_NAMES.append(_name)
    BOUNDS.append(_bnd)
    _CONFIG_PATH_TO_PARAM_IDX[_path] = _PS_SLOT_OFFSET + _i

# ── Append regime-conditional tilt slot 46 ─────────────────────────────────
# Single scalar: in CONFIRMED-BULL regime (SPY > 200DMA), shift this fraction of
# total score weight from value/quality/income into momentum (renormalised),
# making the active sleeve more aggressive in bull tapes while keeping its
# defensive quality/income tilt in neutral/defensive regimes. Frozen by default
# (0.0 = behaviour-preserving). Unfrozen only via the `active_regime_tilt` preset.
# Scoring-only lever: changes *which* stocks rank high, never cash/share accounting.
_REGIME_TILT_SLOT = len(PARAM_NAMES)  # 46
PARAM_NAMES.append("regime_bull_momentum_tilt")
BOUNDS.append((0.0, 0.50))
_CONFIG_PATH_TO_PARAM_IDX["regime.bullish.momentum_tilt"] = _REGIME_TILT_SLOT

# ── Append regime-conditional mean-reversion blend slot 47 ─────────────────
# In NON-bull regimes, blend an 'oversold' (below-25d-MA) contrarian score into the
# composite. Mirror of momentum: contrarian works in fear regimes (+2-3% fwd edge),
# momentum works in bull. Frozen by default (0.0). Unfrozen via active_alpha_engine.
# Scoring-only lever — changes which stocks rank high, never cash/share accounting.
_MEANREV_SLOT = len(PARAM_NAMES)  # 47
PARAM_NAMES.append("regime_defensive_mean_reversion_blend")
BOUNDS.append((0.0, 1.0))
_CONFIG_PATH_TO_PARAM_IDX["regime.defensive.mean_reversion_blend"] = _MEANREV_SLOT

# ── Append low-volatility quality blend slot 48 ────────────────────────────
# Blend a causal cross-sectional low-vol score into the QUALITY factor. Low-vol
# is a documented quality anomaly: on the 1550d substrate it has positive
# full-sample forward-IC (+0.04@21d, +0.067@63d), is strongest in bull regimes,
# and is largely orthogonal to momentum (corr(-vol, rs_6m)≈+0.13) so it adds new
# information rather than re-proxying it. Frozen by default (0.0 = pure configured
# quality). Scoring-only lever — changes which stocks rank high, never accounting.
_LOWVOL_SLOT = len(PARAM_NAMES)  # 48
PARAM_NAMES.append("quality_low_vol_blend")
BOUNDS.append((0.0, 1.0))
_CONFIG_PATH_TO_PARAM_IDX["scoring.quality_low_vol_blend"] = _LOWVOL_SLOT

# ── Append residual-momentum blend slot 49 ─────────────────────────────────
# Blend a causal cross-sectional RESIDUAL-momentum score into the MOMENTUM factor.
# Residual momentum (Blitz-Huij-Martens 2011) strips market beta: rolling 252d beta
# vs SPY, cumulate the market-residual return t-252..t-21, standardize by residual
# std. On the 1550d substrate it has the strongest forward-IC of any factor tested
# (+0.069@63d / t+3.3, +0.095 in bull / t+5.3), beats raw rs_6m, and is only ~0.56
# correlated with it (additive, not a duplicate). A 0.2-0.7 blend beats rs_6m-alone
# in the top-quintile PoC across every weight and horizon; its edge is regime-shaped
# (cushions momentum's weak era — the crash-resistance Blitz documents). Frozen by
# default (0.0 = pure configured momentum). Scoring-only lever — never accounting.
_RESIDMOM_SLOT = len(PARAM_NAMES)  # 49
PARAM_NAMES.append("momentum_residual_blend")
BOUNDS.append((0.0, 1.0))
_CONFIG_PATH_TO_PARAM_IDX["scoring.momentum_residual_blend"] = _RESIDMOM_SLOT


def position_sizing_cfg_from_params(params) -> dict:
    """
    Build a position-sizing override dict from the sizing slots in *params*.
    Returns {} when params lacks the appended sizing slots (len <= _PS_SLOT_OFFSET).
    max_buys_per_rebalance and max_candidates are integers.
    """
    if params is None or len(params) <= _PS_SLOT_OFFSET:
        return {}
    out: dict = {}
    if _PS_SLOT_OFFSET < len(params):
        out["max_single_position_pct"] = float(params[_PS_SLOT_OFFSET])
    if _PS_SLOT_OFFSET + 1 < len(params):
        out["max_buys_per_rebalance"] = round(float(params[_PS_SLOT_OFFSET + 1]))
    if _PS_SLOT_OFFSET + 2 < len(params):
        out["max_candidates"] = round(float(params[_PS_SLOT_OFFSET + 2]))
    return out


def _position_sizing_default_frozen_indices() -> set[int]:
    """Sizing slots default to frozen — unfrozen only via active_position_sizing."""
    _ps_paths = {p for _, p, _ in _PS_FIELDS}
    return {
        _idx for _path, _idx in _CONFIG_PATH_TO_PARAM_IDX.items()
        if _path in _ps_paths
    }


def candidate_cfg_from_params(params) -> dict:
    """
    Build a candidate-selection override dict from filter slots in *params*.
    Returns {} when params has only the original 40 slots (no cs filters appended).
    """
    if params is None or len(params) <= _CS_FILTER_SLOT_OFFSET:
        return {}
    out: dict = {}
    for _i, (_, _cs_field, _) in enumerate(_CS_FILTER_FIELDS):
        _idx = _CS_FILTER_SLOT_OFFSET + _i
        if _idx < len(params):
            out[_cs_field] = float(params[_idx])
    return out


def _candidate_filter_default_frozen_indices() -> set[int]:
    """All candidate-filter slots default to frozen — unfrozen only via active_candidate_filters preset."""
    return {
        _idx for _path, _idx in _CONFIG_PATH_TO_PARAM_IDX.items()
        if _path.startswith("candidate_selection.")
    }


def archetype_cfg_from_params(params) -> dict:
    """
    Build an archetype-config override dict from the lifecycle slots in *params*.
    Returns {} when params has only the original 15 slots.

    Cross-parameter sanity enforced here (the optimizer's bounds are per-slot
    and would otherwise allow harvest < trim, which is incoherent):
      - harvest_profit_threshold >= trim_profit_threshold + 0.01
      - trailing_stop_pct <= -0.01  (must be negative)
      - minimum_hold_days >= 1
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
                    entry[_field] = max(1, round(_v))
                elif _field == "trailing_stop_pct":
                    entry[_field] = min(_v, -0.01)
                else:
                    entry[_field] = _v
        trim = entry.get("trim_profit_threshold")
        harv = entry.get("harvest_profit_threshold")
        if trim is not None and harv is not None and harv < trim + 0.01:
            entry["harvest_profit_threshold"] = trim + 0.01
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


def _regime_tilt_default_frozen_indices() -> set[int]:
    """Regime tilt + mean-reversion + low-vol + residual-momentum blend slots default
    to frozen — unfrozen via presets."""
    out: set[int] = set()
    for path in ("regime.bullish.momentum_tilt", "regime.defensive.mean_reversion_blend",
                 "scoring.quality_low_vol_blend", "scoring.momentum_residual_blend"):
        idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
        if idx is not None:
            out.add(idx)
    return out


def _get_active_indices(scope: str = "overall_strategy", preset: str | None = None) -> list[int]:
    frozen = {
        _CONFIG_PATH_TO_PARAM_IDX[p]
        for p in TUNING_PARAMS.get("frozen_parameters", [])
        if p in _CONFIG_PATH_TO_PARAM_IDX
    }
    # Archetype lifecycle slots are frozen-by-default; an archetype preset unfreezes them.
    frozen |= _archetype_default_frozen_indices()
    # Candidate-filter slots are frozen-by-default; active_candidate_filters unfreezes them.
    frozen |= _candidate_filter_default_frozen_indices()
    frozen |= _position_sizing_default_frozen_indices()
    # Regime-conditional tilt slot is frozen-by-default; active_regime_tilt unfreezes it.
    frozen |= _regime_tilt_default_frozen_indices()
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
    mi_w = SCORING_PARAMS["momentum_inputs"]["weights"]
    mom_sub = [
        mi_w.get("rs_3m",           0.25),
        mi_w.get("rs_6m",           0.25),
        mi_w.get("risk_adj_3m",     0.20),
        mi_w.get("trend_structure", 0.15),
        mi_w.get("return_1m",       0.10),
        mi_w.get("return_5d",       0.05),
    ]
    base = [
        sw["value"], sw["quality"], sw["income"], sw["momentum"],
        INDEX_PCT,
        METRIC_THRESHOLD,
        SELL_RULES["take_profit_pct"],
        SELL_RULES["sell_weak_value_below"],
        SELL_RULES["trailing_stop_pct"],
        VALUE_FACTOR_PARAMS["pe_weight"],
        *mom_sub,
    ]
    # Archetype lifecycle slots 16-39 — read from ARCHETYPE_PARAMS, fall back to slot bounds midpoint
    arch_tail: list[float] = []
    for _ai, _alabel in enumerate(_ARCH_KEYS):
        entry = (ARCHETYPE_PARAMS or {}).get(_alabel, {}) or {}
        for _fi, (_field, _suffix, _bnd) in enumerate(_ARCH_FIELDS):
            _idx = _ARCH_SLOT_OFFSET + _ai * len(_ARCH_FIELDS) + _fi
            _default = entry.get(_field)
            if _default is None:
                _default = (_bnd[0] + _bnd[1]) / 2.0
            arch_tail.append(float(_default))
    # Candidate-filter slots 40-42 — read from live CANDIDATE_SELECTION_PARAMS
    cs_tail: list[float] = []
    for _name, _cs_field, _bnd in _CS_FILTER_FIELDS:
        _default = (CANDIDATE_SELECTION_PARAMS or {}).get(_cs_field)
        if _default is None:
            _default = (_bnd[0] + _bnd[1]) / 2.0
        cs_tail.append(float(_default))
    # Position-sizing slots 43-45 — read from live RISK_LIMITS / CANDIDATE_SELECTION_PARAMS
    ps_tail: list[float] = []
    for _name, _path, _bnd in _PS_FIELDS:
        if _path == "candidate_selection.max_candidates":
            _default = (CANDIDATE_SELECTION_PARAMS or {}).get("max_candidates")
        else:
            _default = (RISK_LIMITS or {}).get(_path.split(".", 1)[1])
        if _default is None:
            _default = (_bnd[0] + _bnd[1]) / 2.0
        ps_tail.append(float(_default))
    # Regime-tilt slot 46 — read from live REGIME_PARAMS; default 0.0 (off).
    _tilt_default = 0.0
    try:
        _tilt_default = float(
            (REGIME_PARAMS or {}).get("bullish", {}).get("momentum_tilt", 0.0)
        )
    except (TypeError, ValueError, AttributeError):
        _tilt_default = 0.0
    regime_tail: list[float] = [_tilt_default]
    # Mean-reversion blend slot 47 — read from REGIME_PARAMS defensive block; default 0.0.
    _mr_default = 0.0
    try:
        _mr_default = float(
            (REGIME_PARAMS or {}).get("defensive", {}).get("mean_reversion_blend", 0.0)
        )
    except (TypeError, ValueError, AttributeError):
        _mr_default = 0.0
    regime_tail.append(_mr_default)
    # Low-vol quality blend slot 48 — read from SCORING_PARAMS; default 0.0 (off).
    _lv_default = 0.0
    try:
        _lv_default = float((SCORING_PARAMS or {}).get("quality_low_vol_blend", 0.0))
    except (TypeError, ValueError, AttributeError):
        _lv_default = 0.0
    regime_tail.append(_lv_default)
    # Residual-momentum blend slot 49 — read from SCORING_PARAMS; default 0.0 (off).
    _rm_default = 0.0
    try:
        _rm_default = float((SCORING_PARAMS or {}).get("momentum_residual_blend", 0.0))
    except (TypeError, ValueError, AttributeError):
        _rm_default = 0.0
    regime_tail.append(_rm_default)
    return np.array(base + arch_tail + cs_tail + ps_tail + regime_tail)
