"""
tests/test_cache_na_ticker.py — the literal ticker "NA" must survive a CSV round-trip.

Regression (2026-06-10): the expanded universe includes Nano Labs ("NA").
pandas' default NaN tokens turned it into float NaN on every cached read, which
crashed `sorted()` over the cached symbol set in gen_symbols_list with
"'<' not supported between instances of 'float' and 'str'" — breaking every
`run --skip-data` invocation. Genuine missing values are written by our own
writers as the empty string, so they must STILL parse as NaN.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd


def _patch_data_dir(monkeypatch, tmp_path):
    import data.cache as dc
    monkeypatch.setattr(dc, "DATA_DIRECTORY", str(tmp_path))
    return dc


def test_na_ticker_survives_csv_round_trip(monkeypatch, tmp_path):
    dc = _patch_data_dir(monkeypatch, tmp_path)

    df = pd.DataFrame({"symbol": ["AAPL", "NA", "MSFT"], "score": [1.0, 2.0, 3.0]})
    dc.store_data_as_csv("stock_tickers", "", df)

    out = dc.read_data_as_pd("stock_tickers")
    assert out is not None
    assert "NA" in out["symbol"].tolist()
    assert sorted(out["symbol"].tolist()) == ["AAPL", "MSFT", "NA"]  # sortable: all str


def test_empty_cells_still_parse_as_nan(monkeypatch, tmp_path):
    dc = _patch_data_dir(monkeypatch, tmp_path)

    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "pe_ratio": [12.5, np.nan]})
    dc.store_data_as_csv("agg_data", "", df)

    out = dc.read_data_as_pd("agg_data")
    assert out is not None
    assert out["pe_ratio"].dtype.kind == "f"  # numeric column stays numeric
    assert np.isnan(out["pe_ratio"].iloc[1])  # empty cell -> NaN, not ""


def test_gen_symbols_list_filters_non_string_symbols(monkeypatch):
    import data.universe as du

    cached = pd.DataFrame({"symbol": ["MSFT", np.nan, "AAPL", ""]})
    monkeypatch.setattr(du, "read_data_as_pd", lambda dataset: cached)

    out = du.gen_symbols_list(force_refresh=False)
    assert out == ["AAPL", "MSFT"]
