"""tests/test_odte_fmp_context.py — FMP meme/squeeze context (pure; network stubbed, no real calls).

build_context is pure (raw FMP JSON in -> classified context out). fetch_fmp_raw/run_fmp_context take
an injectable fetch_json so NO real network is touched. Covers parsing, squeeze classification,
fail-closed (no key / endpoint errors), the always-false options flag, Markdown render, artifact
writing, secret-safety, and the NVDA employer tag.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_fmp_context as fc

PROFILE = [{"symbol": "WEN", "price": 12.5, "marketCap": 2.5e9, "beta": 0.8,
            "range": "9.50-22.30", "averageVolume": 5_000_000, "volume": 6_000_000}]
QUOTE = [{"symbol": "WEN", "price": 12.5, "marketCap": 2.5e9, "volume": 6_000_000,
          "avgVolume": 5_000_000, "yearHigh": 22.3, "yearLow": 9.5}]
SFLOAT = [{"symbol": "WEN", "floatShares": 150_000_000, "outstandingShares": 200_000_000,
           "freeFloat": 0.75}]
KMET = [{"symbol": "WEN", "netDebtToEBITDATTM": 3.2}]
NEWS = [{"title": "Wendy's pops on buyout chatter"}, {"title": "Squeeze talk on socials"},
        {"title": "Earnings beat"}]


def _stub(payloads, missing_err="http_404"):
    def fj(path):
        for frag, val in payloads.items():
            if frag in path:
                return val, None
        return None, missing_err
    return fj


def _full_stub():
    return _stub({"/profile": PROFILE, "/quote": QUOTE, "/shares-float": SFLOAT,
                  "/key-metrics-ttm": KMET, "/news/stock": NEWS})


def _full_raw():
    return {"profile": PROFILE, "quote": QUOTE, "shares_float": SFLOAT,
            "key_metrics": KMET, "news": NEWS}


# --- parsing ---------------------------------------------------------------------------------

def test_build_context_parses_fields():
    c = fc.build_context("WEN", _full_raw())
    assert c["symbol"] == "WEN"
    assert c["price"] == 12.5 and c["market_cap"] == 2.5e9 and c["beta"] == 0.8
    assert c["year_low"] == 9.5 and c["year_high"] == 22.3       # parsed from profile "range"
    assert c["volume"] == 6_000_000 and c["average_volume"] == 5_000_000
    assert c["relative_volume"] == 1.2
    assert c["float_shares"] == 150_000_000 and c["outstanding_shares"] == 200_000_000
    assert c["free_float_pct"] == 75.0                           # 0.75 fraction -> 75.0 pct
    assert c["net_debt_to_ebitda"] == 3.2
    assert c["news_count"] == 3 and len(c["recent_news"]) == 3


def test_free_float_derived_from_shares_when_absent():
    sf = [{"floatShares": 50_000_000, "outstandingShares": 100_000_000}]   # no freeFloat field
    c = fc.build_context("XYZ", {"shares_float": sf})
    assert c["free_float_pct"] == 50.0


# --- classification --------------------------------------------------------------------------

def test_classify_squeeze_buckets():
    assert fc.classify_squeeze(None) == "no_float_data"
    assert fc.classify_squeeze(10_000_000) == "tiny_float_squeeze_candidate"
    assert fc.classify_squeeze(50_000_000) == "small_float_momentum"
    assert fc.classify_squeeze(150_000_000) == "mid_float_meme_momentum"
    assert fc.classify_squeeze(500_000_000) == "large_float_meme_momentum_not_tiny_float"


def test_squeeze_profile_and_implication_in_context():
    c = fc.build_context("WEN", _full_raw())
    assert c["squeeze_profile"] == "mid_float_meme_momentum"
    assert c["trade_implication"] and "tiny-float" in c["trade_implication"].lower()
    tiny = fc.build_context("AA", {"shares_float": [{"floatShares": 5_000_000}]})
    assert tiny["squeeze_profile"] == "tiny_float_squeeze_candidate"


# --- options always unavailable --------------------------------------------------------------

def test_fmp_options_always_unavailable():
    c = fc.build_context("WEN", _full_raw())
    assert c["fmp_options_available"] is False
    assert any("Robinhood remains the option-chain" in w for w in c["warnings"])


# --- fail-closed -----------------------------------------------------------------------------

def test_no_key_fails_closed(monkeypatch):
    monkeypatch.delenv("FMP_KEY", raising=False)
    raw, warnings = fc.fetch_fmp_raw("WEN")        # default fetcher, no key
    assert raw == {} and any("FMP_KEY not set" in w for w in warnings)
    c = fc.build_context("WEN", raw, warnings=warnings)
    assert c["squeeze_profile"] == "no_float_data" and c["fmp_options_available"] is False


def test_endpoint_errors_are_graceful():
    raw, warnings = fc.fetch_fmp_raw("WEN", fetch_json=_stub({}, missing_err="http_404"))
    assert raw == {}
    assert all("http_404" in w for w in warnings if ":" in w)
    c = fc.build_context("WEN", raw, warnings=warnings)
    assert c["squeeze_profile"] == "no_float_data"               # partial data, no exception


def test_run_offline_no_fetch():
    c = fc.run_fmp_context("WEN", allow_fetch=False)
    assert any("offline" in w for w in c["warnings"])
    assert c["fmp_options_available"] is False


# --- render + write --------------------------------------------------------------------------

def test_render_markdown_contains_key_sections():
    md = fc.render_markdown(fc.build_context("WEN", _full_raw()))
    assert "FMP Context — WEN" in md
    assert "mid_float_meme_momentum" in md
    assert "Robinhood remains the option-chain" in md
    assert "Recent news (3)" in md


def test_run_fmp_context_writes_artifacts(tmp_path):
    c = fc.run_fmp_context("WEN", fetch_json=_full_stub(), out_dir=str(tmp_path), write=True)
    assert c["squeeze_profile"] == "mid_float_meme_momentum"
    assert (tmp_path / "odte_fmp_context_wen.md").exists()
    assert (tmp_path / "odte_fmp_context_wen.json").exists()
    assert "squeeze" in (tmp_path / "odte_fmp_context_wen.md").read_text().lower()


# --- secret safety ---------------------------------------------------------------------------

def test_output_never_contains_api_key(monkeypatch):
    monkeypatch.setenv("FMP_KEY", "SECRET_TEST_KEY_123")
    c = fc.run_fmp_context("WEN", fetch_json=_full_stub())   # stub: key never used/leaked
    blob = json.dumps(c) + fc.render_markdown(c)
    assert "SECRET_TEST_KEY_123" not in blob and "apikey" not in blob.lower()


# --- NVDA employer restriction ---------------------------------------------------------------

def test_nvda_tagged_restricted_context_only():
    c = fc.build_context("NVDA", {"shares_float": [{"floatShares": 1_000_000_000}]})
    assert c["restricted"] is True
    assert any("RESTRICTED_EMPLOYER" in w for w in c["warnings"])
