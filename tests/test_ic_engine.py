"""Tests for robust information-coefficient correlation handling."""

from __future__ import annotations

import datetime
import warnings

import pandas as pd
import pytest
from scipy import stats

from research.ic_engine import FactorResearchEngine


@pytest.mark.parametrize("ic_type", ["spearman", "pearson"])
@pytest.mark.parametrize(
    ("factor_values", "forward_prices"),
    [
        ([1.0, 1.0, 1.0], [101.0, 102.0, 103.0]),
        ([1.0, 2.0, 3.0], [101.0, 101.0, 101.0]),
    ],
    ids=["constant-factor", "constant-return"],
)
def test_constant_ic_input_is_skipped_without_scipy_warning(
    monkeypatch: pytest.MonkeyPatch,
    ic_type: str,
    factor_values: list[float],
    forward_prices: list[float],
) -> None:
    symbols = ["A", "B", "C"]
    dates_map = {
        datetime.date(2026, 1, 1): pd.DataFrame(
            {
                "symbol": symbols,
                "current_price": [100.0, 100.0, 100.0],
                "constant_factor": factor_values,
            }
        ),
        datetime.date(2026, 1, 21): pd.DataFrame(
            {
                "symbol": symbols,
                "current_price": forward_prices,
            }
        ),
    }
    engine = FactorResearchEngine(
        factors=["constant_factor"],
        horizons=[20],
        min_overlap=3,
        max_horizon_slop_pct=0.0,
    )
    monkeypatch.setattr(engine, "_load_dates_map", lambda: dates_map)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = engine.compute_multi_horizon_ic(ic_type=ic_type)

    assert result.empty
    assert not any(issubclass(item.category, stats.ConstantInputWarning) for item in caught)


def test_nonconstant_ic_input_is_still_computed(monkeypatch: pytest.MonkeyPatch) -> None:
    symbols = ["A", "B", "C"]
    dates_map = {
        datetime.date(2026, 1, 1): pd.DataFrame(
            {
                "symbol": symbols,
                "current_price": [100.0, 100.0, 100.0],
                "factor": [1.0, 2.0, 3.0],
            }
        ),
        datetime.date(2026, 1, 21): pd.DataFrame(
            {
                "symbol": symbols,
                "current_price": [101.0, 102.0, 103.0],
            }
        ),
    }
    engine = FactorResearchEngine(
        factors=["factor"],
        horizons=[20],
        min_overlap=3,
        max_horizon_slop_pct=0.0,
    )
    monkeypatch.setattr(engine, "_load_dates_map", lambda: dates_map)

    result = engine.compute_multi_horizon_ic()

    assert len(result) == 1
    assert result.iloc[0]["ic"] == pytest.approx(1.0)
