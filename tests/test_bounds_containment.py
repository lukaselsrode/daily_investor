"""
tests/test_bounds_containment.py — every live config value must sit inside tuner BOUNDS.

Hard project rule: the DE tuner seeds from the live config; a live value outside its
slot's bounds silently clips at seed time, so the tuner explores a different point
than the one live trades (the peer-2 era shipped exactly this bug). Applies to every
recalibrated peer-3 threshold (metric_threshold, sell floors, quality gates, DAE
floors) via the canonical slot mapping.
"""

from __future__ import annotations

from tuning.constants import BOUNDS, PARAM_NAMES, _current_params


def test_all_live_values_inside_bounds():
    params = _current_params()
    violations = []
    for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, BOUNDS)):
        v = float(params[i])
        if not (lo <= v <= hi):
            violations.append(f"slot {i} {name}: {v} outside [{lo}, {hi}]")
    assert not violations, "\n".join(violations)
