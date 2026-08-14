"""
tests/test_bounds_containment.py — the tuner must be able to express what we've validated.

Two distinct guarantees:

  1. Every LIVE config value sits inside its slot's bounds. The DE tuner seeds from the
     live config; a value outside its bounds silently clips at seed time, so the tuner
     explores a different point than the one live trades.

  2. Every HISTORICALLY-VALIDATED score weighting stays EXPRESSIBLE. This is the failure
     mode (1) cannot catch, and it has now bitten twice:
       - June: sell_weak_below bounds (0.10, 0.90) excluded the validated -0.18, so every
         re-tune silently dragged it back and reverted the "stop selling winners" finding.
       - 2026-08-13: sw_income bounds (0.00, 0.40) excluded the 2026-06-28 DE-tuned config
         (raw income 0.5747), so the gauntlet could not have rediscovered it even had it
         been optimal — the search was structurally biased against income.
     A bound that excludes a config we once validated makes every future tune a quiet
     revert of that finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from tuning.constants import BOUNDS, PARAM_NAMES, _current_params

# Score-weight regimes this project has actually run live or validated. Weights are
# normalized at scoring time (simulator: sw = raw_sw / sum), so only the RATIO matters —
# expressibility means "some positive scaling of this ratio fits inside the bounds".
_HISTORICAL_WEIGHTS = {
    "manual 2026-07-01 (current)":      (0.25, 0.40, 0.25, 0.10),
    "DE tune 2026-06-28 (income-led)":  (0.1354, 0.1263, 0.5747, 0.1636),
    "bear-validated 2026-06-11":        (0.064, 0.323, 0.051, 0.562),
    "equal-weight control":             (0.25, 0.25, 0.25, 0.25),
}


def test_all_live_values_inside_bounds():
    params = _current_params()
    violations = []
    for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, BOUNDS)):
        v = float(params[i])
        if not (lo <= v <= hi):
            violations.append(f"slot {i} {name}: {v} outside [{lo}, {hi}]")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("label", sorted(_HISTORICAL_WEIGHTS))
def test_historical_weightings_remain_expressible(label):
    """A previously-validated weighting must remain reachable by the optimizer.

    Scale-invariance: normalize the ratio so its largest component sits at that slot's
    upper bound, then every component must fall inside its own bounds.
    """
    w = np.asarray(_HISTORICAL_WEIGHTS[label], dtype=float)
    lo = np.array([BOUNDS[i][0] for i in range(4)], dtype=float)
    hi = np.array([BOUNDS[i][1] for i in range(4)], dtype=float)

    # Largest achievable scaling that keeps every component under its ceiling.
    scale = float(np.min(hi / np.maximum(w, 1e-12)))
    scaled = w * scale
    bad = [
        f"{PARAM_NAMES[i]}={scaled[i]:.4f} outside [{lo[i]}, {hi[i]}]"
        for i in range(4)
        if not (lo[i] - 1e-9 <= scaled[i] <= hi[i] + 1e-9)
    ]
    assert not bad, (
        f"{label} is NOT expressible within the weight bounds: {'; '.join(bad)}. "
        "A tune can never rediscover it — widen the bounds or retire the config."
    )


def test_weight_bounds_are_symmetric():
    """No factor may carry a structural advantage in the search box.

    Weights are normalized, so asymmetric ceilings bias a Latin-hypercube draw toward
    whichever factor has the widest range — a prior imposed by engineering, not evidence.
    """
    weight_bounds = {PARAM_NAMES[i]: BOUNDS[i] for i in range(4)}
    assert len(set(weight_bounds.values())) == 1, (
        f"score-weight bounds are asymmetric: {weight_bounds} — this imposes a prior on "
        "which factor the optimizer favors before it sees any data"
    )
    lo, hi = BOUNDS[0]
    assert lo >= 0.0, (
        "score weights must stay non-negative: the composite is only a convex combination "
        "of [-1, 1.5] factors while weights are >= 0, and every absolute threshold "
        "(metric_threshold, entry ladder, sell floors) is calibrated on that scale"
    )
    assert hi > 0.0
