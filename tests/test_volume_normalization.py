"""
tests/test_volume_normalization.py — the liquidity gate must see AVERAGE volume.

Regression (2026-06-11): Robinhood's fundamentals `volume` field is the
current day's cumulative volume. A 10:19 AM refresh stored partial-day numbers
(HST "288,362" vs ~3M ADV) and `min_liquidity_volume` rejected every BUY
candidate for the entire run — 6,184 of ~6,967 symbols were below 2/3 of their
prior-close volume. `_normalize_volume` swaps in average daily volume at
ingestion, and the production ledgers must never receive test tickers again.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.fundamentals import _normalize_volume
from util import RISK_LIMITS


class TestNormalizeVolume:

    def test_partial_intraday_volume_replaced_by_average(self):
        # The literal failure case: HST at 10:19 AM.
        item = {"symbol": "HST", "volume": "288362", "average_volume": "3100000"}
        _normalize_volume(item)
        assert float(item["volume"]) == 3_100_000.0
        assert float(item["volume"]) >= RISK_LIMITS["min_liquidity_volume"]

    def test_falls_back_to_two_week_average(self):
        item = {"symbol": "X", "volume": "1000", "average_volume": None,
                "average_volume_2_weeks": "750000"}
        _normalize_volume(item)
        assert float(item["volume"]) == 750_000.0

    def test_keeps_raw_volume_when_no_average_available(self):
        item = {"symbol": "X", "volume": "420000"}
        _normalize_volume(item)
        assert item["volume"] == "420000"  # untouched — no average to prefer

    def test_garbage_average_ignored(self):
        item = {"symbol": "X", "volume": "420000", "average_volume": "not-a-number"}
        _normalize_volume(item)
        assert item["volume"] == "420000"


class TestLedgerIsolation:

    def test_outcome_journal_redirected_to_tmp(self):
        """The autouse conftest fixture must point the ledgers away from data/."""
        import portfolio.outcome_tracker as ot
        assert "data" != ot._journal_path().parent.name or "pytest" in str(ot._journal_path())
        assert str(ot._journal_path()).endswith("outcome_journal.csv")
        assert "daily_investor/data" not in str(ot._journal_path())

    def test_production_ledger_clean_of_test_tickers(self):
        """The one-time purge stays clean: no ZZZQ* rows in the real journal."""
        import pandas as pd
        path = os.path.join(os.path.dirname(__file__), "..", "data", "outcome_journal.csv")
        if not os.path.exists(path):
            return
        j = pd.read_csv(path)
        assert not j["symbol"].astype(str).str.startswith("ZZZQ").any()
