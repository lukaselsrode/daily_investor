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
    INDEX_PCT,
    METRIC_THRESHOLD,
    MOMENTUM_V2_PARAMS,
    RISK_LIMITS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    SELL_RULES,
    TUNING_PARAMS,
)

_MIN_TRADES_HARD = 20
_MIN_TRADES_SOFT = 40

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


def _effective_bounds() -> list[tuple[float, float]]:
    bounds = list(BOUNDS)
    for path, rng in TUNING_PARAMS.get("parameter_bounds", {}).items():
        idx = _CONFIG_PATH_TO_PARAM_IDX.get(path)
        if idx is None:
            continue
        lo = float(rng.get("min", bounds[idx][0]))
        hi = float(rng.get("max", bounds[idx][1]))
        bounds[idx] = (lo, min(hi, bounds[idx][1]))
    return bounds


def _get_active_indices() -> list[int]:
    frozen = {
        _CONFIG_PATH_TO_PARAM_IDX[p]
        for p in TUNING_PARAMS.get("frozen_parameters", [])
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
    return np.array([
        sw["value"], sw["quality"], sw["income"], sw["momentum"],
        INDEX_PCT,
        METRIC_THRESHOLD,
        SELL_RULES["take_profit_pct"],
        SELL_RULES["sell_weak_value_below"],
        SELL_RULES["trailing_stop_pct"],
        SCORING_PARAMS["value_pe_weight"],
        *mom_sub,
    ])
