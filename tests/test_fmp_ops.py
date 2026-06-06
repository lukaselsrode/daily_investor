"""tests/test_fmp_ops.py — FMP cache operational helpers."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import fmp_ops


def test_load_symbol_list_from_comma_list():
    assert fmp_ops.load_symbol_list("aapl, MSFT,AAPL") == ["AAPL", "MSFT"]


def test_load_symbol_list_from_csv(tmp_path: Path):
    path = tmp_path / "symbols.csv"
    pd.DataFrame({"symbol": ["aapl", "msft", "aapl"]}).to_csv(path, index=False)
    assert fmp_ops.load_symbol_list(str(path), max_symbols=1) == ["AAPL"]


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
