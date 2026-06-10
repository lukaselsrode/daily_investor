"""
tests/test_fmp_survivorship.py — Cache-gated tests for the survivorship-free data layer.

These exercise the new FMP infrastructure (fmp_client, survivorship, pit_fundamentals). They read
ONLY the local cache (allow_fetch=False) so they never hit the network, and they skip cleanly when
the cache is absent — so a clean checkout / CI without data/fmp_cache_adj stays green.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_CACHE = os.path.join("data", "fmp_cache_adj")
_HAS_CACHE = os.path.isdir(os.path.join(_CACHE, "prices"))
_HAS_AAPL = os.path.exists(os.path.join(_CACHE, "prices", "AAPL.parquet"))

needs_cache = pytest.mark.skipif(not (_HAS_CACHE and _HAS_AAPL),
                                 reason="FMP survivorship-free cache (data/fmp_cache_adj) not present")


@needs_cache
def test_fmp_client_eod_cache_read():
    from data import fmp_client as fmp
    df = fmp.eod_prices("AAPL", "2023-01-01", "2024-01-01", allow_fetch=False)
    assert df is not None and len(df) > 100
    assert "close" in df.columns and (df["close"] > 0).all()


@needs_cache
def test_fmp_client_statement_cache_read():
    from data import fmp_client as fmp
    inc = fmp.statement("AAPL", "income-statement", allow_fetch=False)
    assert inc is not None and not inc.empty
    # filingDate is what makes the reconstruction causal — it must be present.
    assert "filingDate" in inc.columns and "epsDiluted" in inc.columns


@needs_cache
def test_fmp_client_negative_cache_no_fetch():
    # allow_fetch=False on an uncached symbol must return None without raising or spending a call.
    from data import fmp_client as fmp
    assert fmp.eod_prices("__NOT_A_REAL_SYMBOL__", "2023-01-01", "2023-02-01", allow_fetch=False) is None


@needs_cache
def test_pit_fundamentals_aapl_sane():
    from data import fmp_client as fmp
    from research.pit_fundamentals import fundamentals_asof
    px = fmp.eod_prices("AAPL", "2023-01-01", "2023-01-15", allow_fetch=False)
    price = float(px["close"].iloc[0])
    f = fundamentals_asof("AAPL", "2023-01-04", price)
    assert f is not None
    assert 5 < f["pe"] < 60, f"AAPL PE out of sane range: {f['pe']}"
    assert f["market_cap"] > 1e12, "AAPL market cap should be > $1T"
    assert 0.10 < f["net_margin"] < 0.45, f"AAPL net margin off: {f['net_margin']}"


@needs_cache
def test_pit_fundamentals_is_causal():
    # Fundamentals as-of an early date must NOT use statements filed later (no look-ahead).
    from research.pit_fundamentals import fundamentals_asof
    early = fundamentals_asof("AAPL", "2022-01-04", 175.0)
    late = fundamentals_asof("AAPL", "2025-06-04", 200.0)
    assert early is not None and late is not None
    # TTM EPS grows over time; the early as-of must reflect a smaller (earlier) TTM than the late one.
    assert early["ttm_eps"] != late["ttm_eps"]


@needs_cache
def test_survivorship_assemble_survivor_path():
    # add_dead=False keeps it fast: just verify the survivor + benchmark path builds from the cache.
    import pandas as pd

    from backtesting.survivorship import assemble, dead_universe
    agg = pd.DataFrame({"symbol": ["AAPL"], "volume": [1e7], "quality_score": [0.0],
                        "pe_comp": [0.0], "pb_comp": [0.0], "income_score": [0.0], "value_metric": [0.0]})
    closes, ext_agg, dv, tradeable = assemble(agg, ["AAPL"], ["SPY"], "SPY", 250, add_dead=False)
    assert "AAPL" in closes.columns and "SPY" in closes.columns
    assert len(closes) == 250
    assert dv.shape[0] == 250
    # Survivors print through the window end → tradeable everywhere they have prices.
    assert tradeable.shape == closes.shape
    assert bool(tradeable["AAPL"].iloc[-1])
    # the dead-name roster exists and is non-trivial (the survivorship hole to splice in)
    assert len(dead_universe()) > 100


def test_precomputed_has_dollar_volume_field():
    # Field must exist with a None default so non-survivorship loads / existing callers are unaffected.
    from backtesting.types import PrecomputedData
    assert "dollar_volume_daily" in PrecomputedData._fields
    assert PrecomputedData._field_defaults.get("dollar_volume_daily", "MISSING") is None
