"""
tests/test_fundamental_features.py — PIT correctness of data.fundamental_features.

Covers:
  1. filingDate STRICTLY before as-of (a same-day filing is invisible).
  2. Restatement dedup by (fiscalYear, period) keeps the latest filing.
  3. Feature arithmetic sanity (ROE, margins, leverage, accruals, share shrink,
     dividend coverage/growth/streak) against hand-computed values.
  4. Staleness guard: a filing older than the max age scores as uncovered.
  5. add_fundamental_features merges in place and reports coverage.
  6. add_valuation_ratios_asof fills ONLY missing cells (fill_missing_only).
  7. Cache-only: symbols with no statement JSONs stay all-NaN.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from data import fmp_client
from data.fundamental_features import (
    FUND_FEATURE_COLS,
    FundamentalsCache,
    add_fundamental_features,
    add_valuation_ratios_asof,
    compute_fundamental_features_asof,
)


@pytest.fixture
def stmt_dir(tmp_path, monkeypatch):
    d = tmp_path / "statements"
    for kind in ("income-statement", "balance-sheet-statement", "cash-flow-statement", "dividends"):
        (d / kind).mkdir(parents=True)
    monkeypatch.setattr(fmp_client, "_STMT_DIR", str(d))
    monkeypatch.setattr(fmp_client, "_PRICE_DIR", str(tmp_path / "prices"))
    (tmp_path / "prices").mkdir()
    return d


def _write(stmt_dir, kind: str, symbol: str, rows: list[dict]) -> None:
    with open(os.path.join(stmt_dir, kind, f"{symbol}.json"), "w") as fh:
        json.dump(rows, fh)


def _qdates(n: int, end: str = "2026-04-01") -> list[pd.Timestamp]:
    """n quarterly-spaced dates ending at `end`, ascending."""
    e = pd.Timestamp(end)
    return [e - pd.DateOffset(months=3 * i) for i in range(n)][::-1]


def _quarters(n: int):
    """(fiscalYear, period, filingDate) tuples, newest LAST, quarterly spacing."""
    return [(d.year, f"Q{d.quarter}", d.strftime("%Y-%m-%d")) for d in _qdates(n)]


def _seed_symbol(
    stmt_dir,
    symbol: str = "TST",
    n: int = 12,
    ni: float = 100.0,
    rev: float = 1000.0,
    gp: float = 400.0,
    eps: float = 1.0,
    shares: float = 100.0,
    cfo: float = 120.0,
    fcf: float = 90.0,
    div_paid: float = 30.0,
    assets: float = 4000.0,
    equity: float = 2000.0,
    debt: float = 800.0,
) -> None:
    qs = _quarters(n)
    _write(stmt_dir, "income-statement", symbol, [
        {"fiscalYear": fy, "period": p, "filingDate": fd, "netIncome": ni, "revenue": rev,
         "grossProfit": gp, "epsDiluted": eps, "weightedAverageShsOutDil": shares}
        for fy, p, fd in qs
    ])
    _write(stmt_dir, "cash-flow-statement", symbol, [
        {"fiscalYear": fy, "period": p, "filingDate": fd, "operatingCashFlow": cfo,
         "freeCashFlow": fcf, "commonDividendsPaid": -div_paid}
        for fy, p, fd in qs
    ])
    _write(stmt_dir, "balance-sheet-statement", symbol, [
        {"fiscalYear": fy, "period": p, "filingDate": fd, "totalAssets": assets,
         "totalStockholdersEquity": equity, "totalDebt": debt}
        for fy, p, fd in qs
    ])


def _seed_dividends(stmt_dir, symbol: str, ex_dates: list[str], amount: float = 0.25) -> None:
    _write(stmt_dir, "dividends", symbol, [
        {"date": d, "adjDividend": amount, "dividend": amount} for d in ex_dates
    ])


def test_feature_arithmetic(stmt_dir):
    _seed_symbol(stmt_dir)
    feats = compute_fundamental_features_asof(["TST"], "2026-06-01", FundamentalsCache())
    row = feats.loc["TST"]
    assert row["roe_ttm"] == pytest.approx(400.0 / 2000.0)          # ttm_ni / equity
    assert row["gross_margin_ttm"] == pytest.approx(1600.0 / 4000.0)
    assert row["gm_trend_yoy"] == pytest.approx(0.0)                # constant margins
    assert row["debt_to_assets"] == pytest.approx(800.0 / 4000.0)
    assert row["neg_accruals"] == pytest.approx(-(400.0 - 480.0) / 4000.0)
    assert row["fcf_to_assets"] == pytest.approx(360.0 / 4000.0)
    assert row["share_count_shrink_yoy"] == pytest.approx(0.0)      # constant shares
    assert row["div_fcf_coverage_ttm"] == pytest.approx(360.0 / 120.0)


def test_filing_strictly_before_asof(stmt_dir):
    _seed_symbol(stmt_dir, n=8)
    cache = FundamentalsCache()
    tl = cache.timeline("TST")
    last_filing = pd.Timestamp(tl["_fd"].iloc[-1])
    # As-of exactly the last filing date: that filing must be INVISIBLE.
    on_day = compute_fundamental_features_asof(["TST"], last_filing.date(), cache)
    after = compute_fundamental_features_asof(
        ["TST"], (last_filing + pd.Timedelta(days=1)).date(), cache
    )
    # Both resolve (earlier filings exist), but the on-day call must use the
    # PREVIOUS filing — verify via a field that differs per filing count.
    assert on_day.loc["TST"].notna().sum() >= 6
    assert after.loc["TST"].notna().sum() >= 6
    # Directly assert the boundary on the row selector.
    from data.fundamental_features import _timeline_row_asof
    row_on = _timeline_row_asof(tl, last_filing)
    assert pd.Timestamp(row_on["_fd"]) < last_filing


def test_restatement_dedup_keeps_latest_filing(stmt_dir):
    qs = _quarters(8)
    rows = [
        {"fiscalYear": fy, "period": p, "filingDate": fd, "netIncome": 100.0, "revenue": 1000.0,
         "grossProfit": 400.0, "epsDiluted": 1.0, "weightedAverageShsOutDil": 100.0}
        for fy, p, fd in qs
    ]
    # Restate the final quarter 10 days later with different net income.
    fy, p, fd = qs[-1]
    restated_fd = (pd.Timestamp(fd) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    rows.append({"fiscalYear": fy, "period": p, "filingDate": restated_fd, "netIncome": 200.0,
                 "revenue": 1000.0, "grossProfit": 400.0, "epsDiluted": 2.0,
                 "weightedAverageShsOutDil": 100.0})
    _write(stmt_dir, "income-statement", "RST", rows)

    tl = FundamentalsCache().timeline("RST")
    # One row per fiscal quarter — the restated one, not both.
    assert len(tl) == 8
    assert pd.Timestamp(tl["_fd"].iloc[-1]).strftime("%Y-%m-%d") == restated_fd


def test_staleness_guard(stmt_dir):
    _seed_symbol(stmt_dir, n=8)
    cache = FundamentalsCache()
    last_filing = pd.Timestamp(cache.timeline("TST")["_fd"].iloc[-1])
    stale_asof = (last_filing + pd.Timedelta(days=500)).date()
    feats = compute_fundamental_features_asof(["TST"], stale_asof, cache)
    assert feats.loc["TST", list(FUND_FEATURE_COLS[:8])].isna().all()


def test_uncached_symbol_stays_nan(stmt_dir):
    feats = compute_fundamental_features_asof(["NOPE"], "2026-06-01", FundamentalsCache())
    assert feats.loc["NOPE"].isna().all()


def test_dividend_growth_and_streak(stmt_dir):
    # 8 quarterly payments: first 4 at 0.20, last 4 at 0.25 → growth = 0.25.
    ex = [d.strftime("%Y-%m-%d") for d in _qdates(8)]
    _write(stmt_dir, "dividends", "DIV", (
        [{"date": d, "adjDividend": 0.20, "dividend": 0.20} for d in ex[:4]]
        + [{"date": d, "adjDividend": 0.25, "dividend": 0.25} for d in ex[4:]]
    ))
    feats = compute_fundamental_features_asof(["DIV"], "2026-06-01", FundamentalsCache())
    assert feats.loc["DIV", "div_growth_1y"] == pytest.approx(0.25, rel=1e-6)
    assert feats.loc["DIV", "div_streak_quarters"] == 8.0


def test_dividend_streak_zero_after_stop(stmt_dir):
    # Payer whose last ex-div was ~2 years before as-of → streak 0, growth NaN.
    ex = [d.strftime("%Y-%m-%d") for d in _qdates(6, end="2024-04-01")]
    _seed_dividends(stmt_dir, "STP", ex)
    feats = compute_fundamental_features_asof(["STP"], "2026-06-01", FundamentalsCache())
    assert feats.loc["STP", "div_streak_quarters"] == 0.0
    assert np.isnan(feats.loc["STP", "div_growth_1y"])


def test_add_fundamental_features_merges_and_covers(stmt_dir):
    _seed_symbol(stmt_dir, "AAA")
    df = pd.DataFrame({"symbol": ["AAA", "ZZZ"]})
    coverage = add_fundamental_features(df, "2026-06-01")
    for col in FUND_FEATURE_COLS:
        assert col in df.columns
    assert coverage == pytest.approx(0.5)  # AAA covered, ZZZ not
    assert df.loc[df["symbol"] == "ZZZ", "roe_ttm"].isna().all()


def test_valuation_fill_missing_only(stmt_dir):
    _seed_symbol(stmt_dir, "VAL", eps=1.0, shares=100.0, equity=2000.0)
    _seed_dividends(
        stmt_dir, "VAL",
        [d.strftime("%Y-%m-%d") for d in _qdates(4)],
        amount=0.25,
    )
    df = pd.DataFrame({
        "symbol": ["VAL", "VAL2"],
        "current_price": [40.0, 40.0],
        "pe_ratio": [12.34, np.nan],       # present → must NOT be overwritten
    })
    _seed_symbol(stmt_dir, "VAL2", eps=2.0, shares=50.0, equity=1000.0)
    cov = add_valuation_ratios_asof(df, "2026-06-01")
    assert df.loc[0, "pe_ratio"] == pytest.approx(12.34)            # untouched
    assert df.loc[1, "pe_ratio"] == pytest.approx(40.0 / 8.0)       # filled: px / ttm_eps
    assert df.loc[1, "pb_ratio"] == pytest.approx(40.0 * 50.0 / 1000.0)
    assert df.loc[1, "market_cap"] == pytest.approx(2000.0)
    assert df.loc[0, "dividend_yield"] == pytest.approx(1.0 / 40.0)  # 4 × 0.25 TTM
    assert df.loc[1, "dividend_yield"] == 0.0                        # no dividend cache → non-payer
    assert cov == 1.0
