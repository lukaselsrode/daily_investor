"""
tests/test_volume_features.py — PIT dollar-volume features (data.volume_features).

Covers:
  1. Point-in-time no-lookahead: appending future price rows never changes
     features computed as-of an earlier date.
  2. Trailing windows are inclusive of asof and exclude anything after it.
  3. Coverage minimums: symbols with too little history get NaN per horizon.
  4. Uncached symbols: NaN features, then the ADV×price fallback fills the
     21d horizon only.
  5. Consistency (cv) is ~0 for constant dollar volume and grows with variance.

Thresholds come from the live module constants (_MIN_ROWS / FEATURE_COLS),
never hardcoded copies.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from data import fmp_client, volume_features
from data.volume_features import (
    _MIN_ROWS,
    FEATURE_COLS,
    DollarVolumeCache,
    add_dollar_volume_features,
    compute_features_asof,
)


def _write_prices(price_dir, symbol: str, dates: list[str], close, volume) -> None:
    df = pd.DataFrame({"close": close, "volume": volume}, index=pd.Index(dates, name="date"))
    df.to_parquet(price_dir / f"{symbol}.parquet")


@pytest.fixture
def price_dir(tmp_path, monkeypatch):
    d = tmp_path / "prices"
    d.mkdir()
    monkeypatch.setattr(fmp_client, "_PRICE_DIR", str(d))
    return d


def _biz_dates(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def test_pit_no_lookahead(price_dir):
    dates = _biz_dates("2025-01-01", 80)
    close = np.linspace(100, 120, 80)
    volume = np.full(80, 1_000_000.0)
    _write_prices(price_dir, "PIT", dates, close, volume)

    asof = datetime.date.fromisoformat(dates[70])
    before = compute_features_asof(["PIT"], asof, DollarVolumeCache())

    # Append a violent future regime — must not leak into the asof features.
    future_dates = dates + _biz_dates("2025-06-01", 40)
    future_close = np.concatenate([close, np.full(40, 9_999.0)])
    future_volume = np.concatenate([volume, np.full(40, 9e9)])
    _write_prices(price_dir, "PIT", future_dates, future_close, future_volume)

    after = compute_features_asof(["PIT"], asof, DollarVolumeCache())
    pd.testing.assert_frame_equal(before, after)


def test_trailing_window_inclusive_of_asof(price_dir):
    dates = _biz_dates("2025-01-01", 70)
    close = np.full(70, 10.0)
    volume = np.arange(1, 71, dtype=float)
    _write_prices(price_dir, "INC", dates, close, volume)

    asof = datetime.date.fromisoformat(dates[-1])
    feats = compute_features_asof(["INC"], asof, DollarVolumeCache())
    expected_5d = float(np.mean(10.0 * volume[-5:]))  # last 5 rows INCLUDING asof
    assert feats.loc["INC", "dollar_vol_5d"] == pytest.approx(expected_5d)

    # As-of one row earlier: window shifts back by exactly one row.
    asof_prev = datetime.date.fromisoformat(dates[-2])
    feats_prev = compute_features_asof(["INC"], asof_prev, DollarVolumeCache())
    expected_prev = float(np.mean(10.0 * volume[-6:-1]))
    assert feats_prev.loc["INC", "dollar_vol_5d"] == pytest.approx(expected_prev)


def test_coverage_minimums(price_dir):
    n_young = _MIN_ROWS["dollar_vol_63d"] - 1  # enough for 5d/21d, not 63d
    dates = _biz_dates("2025-01-01", n_young)
    _write_prices(price_dir, "YOUNG", dates, np.full(n_young, 50.0), np.full(n_young, 2e6))

    feats = compute_features_asof(
        ["YOUNG"], datetime.date.fromisoformat(dates[-1]), DollarVolumeCache()
    )
    assert np.isfinite(feats.loc["YOUNG", "dollar_vol_5d"])
    assert np.isfinite(feats.loc["YOUNG", "dollar_vol_21d"])
    assert np.isnan(feats.loc["YOUNG", "dollar_vol_63d"])
    assert np.isnan(feats.loc["YOUNG", "dollar_vol_cv_63d"])


def test_uncached_symbol_and_adv_fallback(price_dir):
    dates = _biz_dates("2025-01-01", 70)
    _write_prices(price_dir, "HAVE", dates, np.full(70, 20.0), np.full(70, 3e6))

    df = pd.DataFrame({
        "symbol": ["HAVE", "MISSING"],
        "volume": [3e6, 4e6],
        "current_price": [20.0, 7.5],
    })
    coverage = add_dollar_volume_features(
        df, datetime.date.fromisoformat(dates[-1]), DollarVolumeCache(), fallback_from_adv=True
    )
    assert coverage == pytest.approx(0.5)
    have = df[df.symbol == "HAVE"].iloc[0]
    missing = df[df.symbol == "MISSING"].iloc[0]
    assert have["dollar_vol_63d"] == pytest.approx(20.0 * 3e6)
    # Fallback: 21d approximated from share ADV × price; other horizons stay NaN.
    assert missing["dollar_vol_21d"] == pytest.approx(4e6 * 7.5)
    assert np.isnan(missing["dollar_vol_63d"])
    assert np.isnan(missing["dollar_vol_5d"])


def test_no_fallback_when_disabled(price_dir):
    df = pd.DataFrame({"symbol": ["GONE"], "volume": [1e6], "current_price": [10.0]})
    add_dollar_volume_features(df, datetime.date(2025, 6, 1), DollarVolumeCache(), fallback_from_adv=False)
    assert df[list(FEATURE_COLS)].isna().all().all()


def test_consistency_cv(price_dir):
    dates = _biz_dates("2025-01-01", 70)
    _write_prices(price_dir, "FLAT", dates, np.full(70, 30.0), np.full(70, 1e6))
    rng = np.random.default_rng(7)
    _write_prices(price_dir, "WILD", dates, np.full(70, 30.0), rng.uniform(1e5, 1e7, 70))

    feats = compute_features_asof(
        ["FLAT", "WILD"], datetime.date.fromisoformat(dates[-1]), DollarVolumeCache()
    )
    assert feats.loc["FLAT", "dollar_vol_cv_63d"] == pytest.approx(0.0, abs=1e-12)
    assert feats.loc["WILD", "dollar_vol_cv_63d"] > feats.loc["FLAT", "dollar_vol_cv_63d"]


def test_feature_cols_constant_matches_module():
    # Guards the schema contract: util.METRIC_KEYS carries exactly these columns.
    from util import METRIC_KEYS
    for col in FEATURE_COLS:
        assert col in METRIC_KEYS, f"{col} missing from util.METRIC_KEYS"
    assert volume_features.COVERAGE_WARN_THRESHOLD > 0
