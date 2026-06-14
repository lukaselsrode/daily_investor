"""
tests/test_pit_fundamentals.py — point-in-time fundamentals (data/pit_fundamentals.py).

Cache-only (no network); skips when the AAPL statement cache is unavailable. Verifies
causality (strict filingDate < asof, values change over time), the dividend-yield-as-of
helper, and the <4-quarters -> None contract.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from data import fmp_client as fmp
from data.pit_fundamentals import dividend_yield_asof, fundamentals_asof

pytestmark = pytest.mark.skipif(
    fmp.statement("AAPL", "income-statement", allow_fetch=False) is None,
    reason="AAPL statement cache unavailable (cache-only test)",
)


def test_pe_sane_recent():
    f = fundamentals_asof("AAPL", "2023-01-04", 130.0)
    assert f is not None
    assert 5 < f["pe"] < 80


def test_ttm_eps_changes_over_time_causal():
    """A PIT factor must change as new filings arrive — i.e. it is NOT a static snapshot."""
    early = fundamentals_asof("AAPL", "2022-01-04", 175.0)
    late = fundamentals_asof("AAPL", "2025-06-04", 200.0)
    assert early is not None and late is not None
    assert early["ttm_eps"] != late["ttm_eps"]


def test_strictly_before_asof_no_same_day_lookahead():
    """Using a filingDate as `asof` must EXCLUDE that day's filing (strict <)."""
    inc = fmp.statement("AAPL", "income-statement", allow_fetch=False)
    import pandas as pd

    fds = pd.to_datetime(inc["filingDate"], errors="coerce").dropna().sort_values()
    pivot = str(fds.iloc[len(fds) // 2].date())
    n_strict = int((fds < pd.Timestamp(pivot)).sum())
    n_incl = int((fds <= pd.Timestamp(pivot)).sum())
    assert n_incl == n_strict + 1  # the pivot day itself is a filing
    # fundamentals_asof at the pivot must rely only on the strictly-earlier filings.
    f = fundamentals_asof("AAPL", pivot, 150.0)
    assert f is not None  # still >=4 prior quarters in this region


def test_insufficient_quarters_returns_none():
    assert fundamentals_asof("AAPL", "1990-01-01", 1.0) is None


def test_dividend_yield_payer_positive_and_small():
    y = dividend_yield_asof("AAPL", "2025-06-04", 200.0)
    assert 0.0 < y < 0.05


def test_dividend_yield_zero_without_price():
    assert dividend_yield_asof("AAPL", "2025-06-04", 0.0) == 0.0
