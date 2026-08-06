"""
tests/test_sim_live_value_parity.py — the simulator's value factor equals live apply_value.

Regression: the sim previously mixed raw 0-5-scale pe_comp/pb_comp (ratios.yaml
components) while the live factor used peer ranks + anchor blend + distress
penalties on [-1, 1.5] — weight retunes were fitting a different factor than the
one live trades. pit_precompute._value_components must now reproduce live
apply_value for ANY pe_weight mix (benchmark_blend excluded — live-only knob).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.pit_precompute import _value_components
from strategy.scoring.value import apply_value
from util import SCORING_PARAMS


def _cfg(pe_weight: float) -> dict:
    cfg = {
        **{k: v for k, v in SCORING_PARAMS.items()},
        "peer_standardization": {**SCORING_PARAMS["peer_standardization"], "min_group_size": 5},
    }
    cfg["factors"] = {
        name: dict(f) for name, f in SCORING_PARAMS["factors"].items()
    }
    cfg["factors"]["value"] = {
        **cfg["factors"]["value"],
        "pe_weight": pe_weight,
        "pb_weight": 1.0 - pe_weight,
        "benchmark_blend": 0.0,  # live-only knob, excluded from parity by design
    }
    return cfg


def _universe(n: int = 80, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "symbol":   [f"V{i:03d}" for i in range(n)],
        "industry": ["banks"] * (n // 2) + ["software"] * (n - n // 2),
        "sector":   ["financials"] * (n // 2) + ["technology"] * (n - n // 2),
        "pe_ratio": rng.uniform(6, 45, n),
        "pb_ratio": rng.uniform(0.8, 9.0, n),
    })
    # Exercise every missing-input branch and the penalty masks.
    df.loc[0, "pe_ratio"] = np.nan            # pb_only
    df.loc[1, "pb_ratio"] = np.nan            # pe_only
    df.loc[2, ["pe_ratio", "pb_ratio"]] = np.nan  # neither → floor
    df.loc[3, "pe_ratio"] = 3.0               # distress (0 < pe <= 5)
    df.loc[4, "pe_ratio"] = -8.0              # negative EPS
    return df


@pytest.mark.parametrize("pe_weight", [0.8757, 0.5, 0.2])
def test_component_mix_reproduces_live_apply_value(pe_weight):
    cfg = _cfg(pe_weight)
    ps = cfg["peer_standardization"]
    clamp = (float(ps["clamp_low"]), float(ps["clamp_high"]))

    live = _universe()
    apply_value(live, cfg)

    frame = _universe()
    pe_c, pb_c, pen = _value_components(frame, cfg)
    sim = np.clip(pe_weight * pe_c + (1.0 - pe_weight) * pb_c - pen, clamp[0], clamp[1])

    live_scores = live["value_score"].to_numpy()
    # Exact parity except live's inner clamp before anchor blending — which only
    # binds when the pre-anchor composite escapes [-1, 1.5]. Compare away from it.
    inner = pd.Series(sim).between(clamp[0] + 1e-9, clamp[1] - 1e-9).to_numpy()
    assert inner.sum() > len(live) * 0.8
    np.testing.assert_allclose(sim[inner], live_scores[inner], atol=0.005)


def test_penalties_present_in_components():
    cfg = _cfg(0.8757)
    frame = _universe()
    _pe_c, _pb_c, pen = _value_components(frame, cfg)
    dist = cfg["factors"]["value"]["distress"]
    assert pen[3] == pytest.approx(float(dist["pe_penalty"]))
    assert pen[4] == pytest.approx(float(dist["negative_eps_penalty"]))
    assert pen[5] == 0.0
