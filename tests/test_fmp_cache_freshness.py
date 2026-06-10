"""tests/test_fmp_cache_freshness.py — FMP cache coverage honesty + transient-error handling.

Covers two cache-poisoning bugs:
  1. Future-dated coverage metadata (req_end=2030-01-01 from backfill defaults) permanently
     froze the price cache — `covers()`/the eod_prices cache-hit must cap claimed coverage
     at the meta's fetch date, healing already-poisoned metas without a forced re-backfill.
  2. Transient HTTP errors (429 / 5xx) were negative-cached as "empty" forever — they must
     raise FMPNetworkError instead, so the symbol is retried on a later run.
No network: the `_get` HTTP layer is monkeypatched and cache dirs point at tmp_path.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import fmp_client as fmp


@pytest.fixture
def cache(tmp_path: Path, monkeypatch):
    """Redirect the whole FMP cache into tmp_path and stub the API key."""
    monkeypatch.setattr(fmp, "_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fmp, "_PRICE_DIR", str(tmp_path / "prices"))
    monkeypatch.setattr(fmp, "_STMT_DIR", str(tmp_path / "statements"))
    monkeypatch.setattr(fmp, "_META_DIR", str(tmp_path / "meta"))
    monkeypatch.setattr(fmp, "_QUOTA_PATH", str(tmp_path / "meta" / "_quota.json"))
    monkeypatch.setenv("FMP_KEY", "test-key")
    return tmp_path


def _days_ago(n: int) -> str:
    return (datetime.date.fromisoformat(fmp._today()) - datetime.timedelta(days=n)).isoformat()


def _write_raw_meta(symbol: str, **rec) -> None:
    """Write a meta file verbatim — simulates legacy on-disk metas that predate the
    _write_meta clamp (i.e. poisoned with a future req_end)."""
    os.makedirs(fmp._META_DIR, exist_ok=True)
    with open(fmp._meta_path(symbol), "w") as fh:
        json.dump(rec, fh)


def _write_prices(symbol: str, dates: list[str]) -> None:
    os.makedirs(fmp._PRICE_DIR, exist_ok=True)
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100},
        index=pd.Index(dates, name="date"),
    )
    df.to_parquet(fmp._price_path(symbol))


def _rows(dates: list[str]) -> list[dict]:
    """Payload rows as the dividend-adjusted endpoint returns them."""
    return [
        {"date": d, "adjOpen": 1.0, "adjHigh": 1.0, "adjLow": 1.0, "adjClose": 2.0, "volume": 100}
        for d in dates
    ]


def test_future_dated_meta_is_stale_and_tops_up(cache, monkeypatch):
    """A poisoned meta (req_end=2030, fetched days ago) must NOT count as covering today —
    covers() says no, and eod_prices fetches the missing tail (from the fetch date, so the
    deep history is not re-pulled)."""
    today = fmp._today()
    stale = _days_ago(7)
    _write_prices("AAPL", [_days_ago(9), _days_ago(8), stale])
    _write_raw_meta("AAPL", status="ok", note="legacy", fetched=stale,
                    req_start="2010-01-01", req_end="2030-01-01")

    assert not fmp.covers("AAPL", "2020-01-01", today)

    calls: list[str] = []

    def fake_get(url: str):
        calls.append(url)
        return 200, _rows([_days_ago(1), today])

    monkeypatch.setattr(fmp, "_get", fake_get)
    df = fmp.eod_prices("AAPL", "2020-01-01", today, allow_fetch=True)

    assert len(calls) == 1, "stale cache must trigger a live top-up"
    assert f"from={stale}" in calls[0], "only the missing tail should be fetched"
    assert df is not None and today in df.index
    # New meta records honest coverage and now satisfies covers() for today.
    meta = fmp._read_meta("AAPL")
    assert meta["req_end"] == today
    assert fmp.covers("AAPL", "2020-01-01", today)


def test_future_dated_meta_fetched_today_still_covers(cache, monkeypatch):
    """A meta fetched TODAY with req_end=2030 still covers today — no refetch loop."""
    today = fmp._today()
    _write_prices("AAPL", [_days_ago(2), _days_ago(1)])
    _write_raw_meta("AAPL", status="ok", note="legacy", fetched=today,
                    req_start="2010-01-01", req_end="2030-01-01")

    assert fmp.covers("AAPL", "2020-01-01", today)

    def boom(url: str):
        raise AssertionError("cache-hit path must not reach the network")

    monkeypatch.setattr(fmp, "_get", boom)
    df = fmp.eod_prices("AAPL", "2020-01-01", today, allow_fetch=True)
    assert df is not None and len(df) == 2


def test_new_meta_clamps_requested_future_end(cache, monkeypatch):
    """A fresh backfill to end=2030-01-01 must record coverage only through today."""
    today = fmp._today()
    monkeypatch.setattr(fmp, "_get", lambda url: (200, _rows([_days_ago(1), today])))

    df = fmp.eod_prices("NEW", "2020-01-01", "2030-01-01", allow_fetch=True)

    assert df is not None
    meta = fmp._read_meta("NEW")
    assert meta["status"] == "ok"
    assert meta["req_end"] == today, "requested future end must be clamped to the fetch date"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_error_raises_and_is_not_negative_cached(cache, monkeypatch, status):
    monkeypatch.setattr(fmp, "_get", lambda url: (status, {"error": "try later"}))

    with pytest.raises(fmp.FMPNetworkError):
        fmp.eod_prices("XYZ", "2020-01-01", "2025-01-01", allow_fetch=True)

    assert fmp._read_meta("XYZ") is None, "transient errors must not write a meta"


def test_genuinely_empty_response_is_still_negative_cached(cache, monkeypatch):
    calls: list[str] = []

    def fake_get(url: str):
        calls.append(url)
        return 200, []

    monkeypatch.setattr(fmp, "_get", fake_get)

    assert fmp.eod_prices("DEAD", "2020-01-01", "2025-01-01", allow_fetch=True) is None
    assert fmp._read_meta("DEAD")["status"] == "empty"
    # Second call is served by the negative cache — no extra spend.
    assert fmp.eod_prices("DEAD", "2020-01-01", "2025-01-01", allow_fetch=True) is None
    assert len(calls) == 1


def test_statement_transient_error_raises_and_empty_is_cached(cache, monkeypatch):
    monkeypatch.setattr(fmp, "_get", lambda url: (429, {"error": "rate limited"}))
    with pytest.raises(fmp.FMPNetworkError):
        fmp.statement("XYZ", "income-statement", allow_fetch=True)
    assert fmp._read_meta("income-statement:XYZ") is None

    monkeypatch.setattr(fmp, "_get", lambda url: (200, []))
    assert fmp.statement("XYZ", "income-statement", allow_fetch=True) is None
    assert fmp._read_meta("income-statement:XYZ")["status"] == "empty"
