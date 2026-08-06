"""
tests/test_price_features.py — PIT price/momentum backfill (data.price_features).

Covers:
  1. Arithmetic parity with the live _enrich_with_momentum definitions
     (returns, rs vs SPY, realized vol, DMA flags, 52w geometry).
  2. Strict as-of windowing: rows after the as-of date are invisible.
  3. fill_missing_only never overwrites present values.
  4. Uncached symbols stay NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import fmp_client
from data.price_features import (
    PRICE_FEATURE_COLS,
    PriceSeriesCache,
    add_price_momentum_features,
    compute_price_features_asof,
)


@pytest.fixture
def price_dir(tmp_path, monkeypatch):
    d = tmp_path / "prices"
    d.mkdir()
    monkeypatch.setattr(fmp_client, "_PRICE_DIR", str(d))
    return d


def _seed(price_dir, symbol: str, closes: np.ndarray, end: str = "2026-06-01") -> None:
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=end, periods=len(closes))]
    pd.DataFrame({"close": closes}, index=pd.Index(dates, name="date")).to_parquet(
        price_dir / f"{symbol}.parquet"
    )


def test_arithmetic_parity(price_dir):
    n = 260
    # Deterministic upward drift: close_t = 100 × 1.001^t.
    closes = 100.0 * np.power(1.001, np.arange(n))
    _seed(price_dir, "UP", closes)
    # Flat SPY → rs_* equal the raw returns.
    _seed(price_dir, "SPY", np.full(n, 500.0))

    feats = compute_price_features_asof(["UP"], "2026-06-01", PriceSeriesCache())
    row = feats.loc["UP"]
    last = closes[-1]
    assert row["current_price"] == pytest.approx(round(float(last), 4))
    for col, lb in (("return_5d", 5), ("return_1m", 21), ("return_3m", 63), ("return_6m", 126)):
        assert row[col] == pytest.approx(round(float(last / closes[-lb] - 1.0), 4)), col
    assert row["rs_3m"] == pytest.approx(row["return_3m"])  # SPY flat → rs == return
    # Constant 0.1% daily growth → near-zero vol; risk_adj huge but finite.
    daily = np.diff(closes[-64:]) / closes[-64:-1]
    assert row["realized_vol_3m"] == pytest.approx(
        round(float(daily[-63:].std(ddof=1) * np.sqrt(252)), 4), abs=1e-4
    )
    assert row["above_50dma"] == 1.0 and row["above_200dma"] == 1.0
    w52 = closes[-252:]
    assert row["low_52w"] == pytest.approx(round(float(w52.min()), 4))
    assert row["high_52w"] == pytest.approx(round(float(w52.max()), 4))
    assert row["position_52w"] == pytest.approx(1.0)  # monotonically rising → at the top


def test_strictly_asof_window(price_dir):
    n = 260
    closes = np.concatenate([np.full(130, 50.0), np.full(130, 100.0)])
    _seed(price_dir, "STEP", closes, end="2026-06-01")
    cache = PriceSeriesCache()
    dates = pd.bdate_range(end="2026-06-01", periods=n)
    # As-of the last day of the 50.0 regime: the 100.0 rows must be invisible.
    early_asof = dates[129].date()
    feats = compute_price_features_asof(["STEP"], early_asof, cache)
    assert feats.loc["STEP", "current_price"] == pytest.approx(50.0)
    assert feats.loc["STEP", "high_52w"] == pytest.approx(50.0)


def test_fill_missing_only(price_dir):
    closes = 100.0 * np.power(1.001, np.arange(260))
    _seed(price_dir, "AAA", closes)
    _seed(price_dir, "SPY", np.full(260, 500.0))
    df = pd.DataFrame({
        "symbol": ["AAA"],
        "return_3m": [0.1234],          # present → untouched
        "return_6m": [np.nan],          # missing → filled
    })
    cov = add_price_momentum_features(df, "2026-06-01")
    assert df.loc[0, "return_3m"] == pytest.approx(0.1234)
    assert df.loc[0, "return_6m"] == pytest.approx(
        round(float(closes[-1] / closes[-126] - 1.0), 4)
    )
    assert cov == 1.0
    for col in PRICE_FEATURE_COLS:
        assert col in df.columns


def test_uncached_symbol_stays_nan(price_dir):
    df = pd.DataFrame({"symbol": ["NOPE"]})
    cov = add_price_momentum_features(df, "2026-06-01")
    assert cov == 0.0
    assert pd.isna(df.loc[0, "return_3m"])
