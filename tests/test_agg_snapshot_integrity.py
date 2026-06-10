"""
tests/test_agg_snapshot_integrity.py — skip-data runs must not degrade the cache.

Regression (2026-06-10): get_data(refresh=False) re-persisted the frame it had
just read from cache — but market-structure + co-mention-graph enrichment only
runs when refresh=True, so every `run --skip-data` overwrote the latest enriched
agg_data snapshot (75 cols) with a stripped 54-col version. Live archetype
classification degraded to fundamentals-only and backtest scoring silently lost
market_cap / analyst / instrument_type until the next full refresh.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd


def _patch_pipeline(monkeypatch, stored: list):
    import data.market as dm

    frame = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "pe_ratio": [12.0, 30.0]})
    monkeypatch.setattr(dm, "gen_symbols_list", lambda *a, **k: ["AAPL", "MSFT"])
    monkeypatch.setattr(dm, "get_fundamentals_df", lambda *a, **k: frame.copy())
    monkeypatch.setattr(dm, "get_news_df", lambda *a, **k: None)
    monkeypatch.setattr(dm, "store_data_as_csv", lambda *a, **k: stored.append(a))
    monkeypatch.setattr(dm.time, "sleep", lambda *_: None)
    return dm


def test_skip_data_pass_does_not_persist(monkeypatch):
    stored: list = []
    dm = _patch_pipeline(monkeypatch, stored)
    out = dm.get_data(refresh=False)
    assert not out.empty
    assert stored == []  # cached pass must NOT overwrite the enriched snapshot


def test_refresh_pass_persists(monkeypatch):
    stored: list = []
    dm = _patch_pipeline(monkeypatch, stored)
    # refresh=True path calls rb.build_holdings + enrichment; stub them out.
    monkeypatch.setattr(dm.rb, "build_holdings", lambda *a, **k: {})
    out = dm.get_data(refresh=True)
    assert not out.empty
    assert len(stored) == 1 and stored[0][0] == "agg_data"
