"""Regression tests for the per-stock aggregate row schema."""

from data.fundamentals import (
    _BASE_AGG_COLUMNS,
    _FUNDAMENTAL_FEATURE_COLS,
    _evaluate_stock,
)


def test_per_stock_row_matches_base_aggregate_schema() -> None:
    stock = {
        "industry": "Software",
        "sector": "Technology",
        "volume": 1_000_000,
        "pe_ratio": 20.0,
        "pb_ratio": 3.0,
    }

    metrics = _evaluate_stock("TEST", stock)

    assert metrics is not None
    assert len(["TEST", *metrics]) == len(_BASE_AGG_COLUMNS)
    assert _FUNDAMENTAL_FEATURE_COLS.isdisjoint(_BASE_AGG_COLUMNS)
