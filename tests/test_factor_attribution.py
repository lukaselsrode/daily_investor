"""
tests/test_factor_attribution.py — UI factor attribution uses live tilted weights.

Regression: position_rationale.factor_contributions carried stale hardcoded
fallback weights (0.45/0.45/0.05/0.05) and ignored the bull-regime momentum
tilt, so the UI decomposition disagreed with the actual composite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio.position_rationale import factor_contributions
from strategy.scoring.composite import regime_tilted_weights
from util import SCORE_WEIGHTS


@pytest.fixture
def metrics() -> pd.Series:
    return pd.Series({
        "value_score": 0.40, "quality_score": 0.60,
        "income_score": 0.20, "momentum_score": 0.80,
        "value_metric": 0.55,
    })


def test_neutral_attribution_matches_live_config_weights(metrics):
    contribs = factor_contributions(metrics, regime=None)
    assert contribs["Value"] == pytest.approx(SCORE_WEIGHTS["value"] * 0.40, abs=1e-4)
    assert contribs["Quality"] == pytest.approx(SCORE_WEIGHTS["quality"] * 0.60, abs=1e-4)
    assert contribs["Income"] == pytest.approx(SCORE_WEIGHTS["income"] * 0.20, abs=1e-4)
    assert contribs["Momentum"] == pytest.approx(SCORE_WEIGHTS["momentum"] * 0.80, abs=1e-4)


def test_bullish_attribution_applies_momentum_tilt(metrics):
    tilted = regime_tilted_weights(SCORE_WEIGHTS, "bullish")
    contribs = factor_contributions(metrics, regime="bullish")
    assert contribs["Momentum"] == pytest.approx(tilted["momentum"] * 0.80, abs=1e-4)
    assert contribs["Quality"] == pytest.approx(tilted["quality"] * 0.60, abs=1e-4)
    # Attribution sums reconstruct the composite under the same weights.
    reconstructed = sum(contribs.values())
    expected = sum(
        tilted[f] * metrics[f"{f}_score"]
        for f in ("value", "quality", "income", "momentum")
    )
    assert reconstructed == pytest.approx(expected, abs=1e-3)


def test_none_metrics_returns_empty():
    assert factor_contributions(None) == {}
