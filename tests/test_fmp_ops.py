"""tests/test_fmp_ops.py — FMP cache operational helpers."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import fmp_ops


class _FakeResponse:
    status_code = 200

    def json(self):
        return [{"ok": True}]


def test_fmp_client_daily_cap_zero_means_rate_limited_only(tmp_path: Path, monkeypatch):
    from data import fmp_client as fmp

    monkeypatch.setattr(fmp, "_DAILY_CAP", 0)
    monkeypatch.setattr(fmp, "_THROTTLE_S", 0.0)
    monkeypatch.setattr(fmp, "_META_DIR", str(tmp_path / "meta"))
    monkeypatch.setattr(fmp, "_QUOTA_PATH", str(tmp_path / "meta" / "_quota.json"))
    monkeypatch.setattr(fmp.requests, "get", lambda *_args, **_kwargs: _FakeResponse())

    status, payload = fmp._get("https://example.test/fmp")

    assert status == 200
    assert payload == [{"ok": True}]


def test_load_symbol_list_from_comma_list():
    assert fmp_ops.load_symbol_list("aapl, MSFT,AAPL") == ["AAPL", "MSFT"]


def test_load_symbol_list_from_csv(tmp_path: Path):
    path = tmp_path / "symbols.csv"
    pd.DataFrame({"symbol": ["aapl", "msft", "aapl"]}).to_csv(path, index=False)
    assert fmp_ops.load_symbol_list(str(path), max_symbols=1) == ["AAPL"]


def test_build_dead_universe_preserves_existing_rows_when_roster_is_partial(tmp_path: Path, monkeypatch):
    from data import fmp_ops

    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({
        "symbol": ["OLD1", "OLD2"],
        "first_date": ["2020-01-01", "2020-01-01"],
        "delist_date": ["2021-01-01", "2021-01-01"],
        "max_adv": [1_000_000.0, 2_000_000.0],
    }).to_parquet(cache / "dead_universe.parquet")
    pd.DataFrame({"symbol": ["NEW1"], "delistedDate": ["2022-01-01"]}).to_parquet(
        cache / "delisted_roster.parquet"
    )

    px = pd.DataFrame({"close": [10.0], "volume": [200_000]}, index=["2021-12-31"])

    class FakeFMP:
        @staticmethod
        def eod_prices(*_args, **_kwargs):
            return px

        @staticmethod
        def calls_remaining():
            return None

    monkeypatch.setenv("FMP_CACHE_DIR", str(cache))
    monkeypatch.setattr(fmp_ops, "fmp", FakeFMP, raising=False)

    # Patch the import used inside build_dead_universe.
    import data.fmp_client as real_fmp
    monkeypatch.setattr(real_fmp, "eod_prices", FakeFMP.eod_prices)
    monkeypatch.setattr(real_fmp, "calls_remaining", FakeFMP.calls_remaining)

    report = fmp_ops.build_dead_universe(min_adv=500_000.0)
    out = pd.read_parquet(cache / "dead_universe.parquet")

    assert report.fetched_or_cached == 3
    assert set(out["symbol"]) == {"OLD1", "OLD2", "NEW1"}


def test_cache_status_reads_custom_cache_dir(tmp_path: Path, monkeypatch):
    cache = tmp_path / "fmp_cache_adj"
    (cache / "prices").mkdir(parents=True)
    (cache / "statements" / "income-statement").mkdir(parents=True)
    pd.DataFrame({"close": [1.0]}).to_parquet(cache / "prices" / "AAPL.parquet")
    (cache / "statements" / "income-statement" / "AAPL.json").write_text("[]")
    pd.DataFrame({"symbol": ["DEAD"]}).to_parquet(cache / "dead_universe.parquet")
    pd.DataFrame({"symbol": ["DEAD"]}).to_parquet(cache / "delisted_roster.parquet")
    monkeypatch.setenv("FMP_CACHE_DIR", str(cache))

    status = fmp_ops.fmp_cache_status()

    assert status.price_files == 1
    assert status.statement_files == 1
    assert status.dead_universe_rows == 1
    assert status.delisted_roster_rows == 1
    assert "price files" in status.pretty()
