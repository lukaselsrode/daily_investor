"""
tests/test_classification_arbiter.py — FMP cross-validation layer (mocked).

Covers GICS→benchmark mapping, material-swing detection, manual-override skip,
verdict-cache reuse (no second LLM call), and the disabled kill-switch. FMP and
Claude are mocked — no network.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import classification_arbiter as ca
from data.valuation import _fmp_to_benchmark_key


def test_fmp_sector_mapping_with_industry_disambiguation():
    assert _fmp_to_benchmark_key("Technology", "Software-Infrastructure") == "Technology Services"
    assert _fmp_to_benchmark_key("Consumer Cyclical", "Auto Manufacturers") == "Consumer Durables"
    assert _fmp_to_benchmark_key("Consumer Cyclical", "Specialty Retail") == "Retail Trade"
    assert _fmp_to_benchmark_key("Financial Services", "Banks - Regional") == "Financial"
    assert _fmp_to_benchmark_key("Healthcare", "Biotechnology") == "Health Technology"
    assert _fmp_to_benchmark_key("Definitely Not A Sector", "") is None


def test_detect_material_flags_large_swing():
    prof = {"sector": "Technology", "industry": "Software", "companyName": "Acme", "description": "cloud"}
    disc = ca._detect_material("XYZ", "Retail Trade", "Internet Retail", prof, 0.20)
    assert disc is not None
    assert disc["fmp_key"] == "Technology Services"
    assert disc["rh_key"] == "Retail Trade"
    assert disc["swing"] > 0.20


def test_detect_no_flag_when_sources_agree():
    # FMP "Financial Services/Banks" maps to the same benchmark key as RH "Finance".
    prof = {"sector": "Financial Services", "industry": "Banks - Diversified"}
    assert ca._detect_material("BAC", "Finance", "Major Banks", prof, 0.20) is None


def test_etf_and_fund_profiles_are_never_flagged():
    # Pooled vehicles are excluded from active scoring; their sector is meaningless.
    etf = {"sector": "Financial Services", "industry": "Asset Management", "isEtf": True}
    fund = {"sector": "Financial Services", "industry": "Asset Management", "isFund": True}
    assert ca._detect_material("SMH", "Miscellaneous", "", etf, 0.20) is None
    assert ca._detect_material("BND", "Miscellaneous", "", fund, 0.20) is None


def test_detect_no_flag_below_threshold():
    prof = {"sector": "Technology", "industry": "Software"}
    # An absurd threshold no real swing can clear → never flags.
    assert ca._detect_material("XYZ", "Retail Trade", "", prof, 5.0) is None


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr("util.CROSS_VALIDATION_PARAMS", {"enabled": False})
    out = ca.cross_validate({"X": {"sector": "Retail Trade"}}, allow_fetch=True)
    assert out == {"checked": 0, "flagged": 0, "applied": 0}


def test_apply_and_persist_then_reuse_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("util.CROSS_VALIDATION_PARAMS", {
        "enabled": True, "swing_threshold": 0.20, "profile_fetch_per_run": 10, "model": "test-model",
    })
    monkeypatch.setattr("util.CLASSIFICATION_OVERRIDES", {})
    monkeypatch.setattr(ca, "ADJUDICATIONS_PATH", str(tmp_path / "adj.json"))

    prof = {"sector": "Technology", "industry": "Software", "companyName": "Acme", "description": "cloud"}
    monkeypatch.setattr(ca.fmp_client, "company_profile", lambda s, allow_fetch=False: prof)

    calls = []

    def fake_adjudicate(disc, model):
        calls.append(disc["symbol"])
        return {"choice": "fmp", "applied_sector": disc["fmp_key"], "reasoning": "cloud platform"}

    monkeypatch.setattr(ca, "_adjudicate", fake_adjudicate)

    funds = {"XYZ": {"sector": "Retail Trade", "industry": "Internet Retail"}}
    out = ca.cross_validate(funds, allow_fetch=True)
    assert out["applied"] == 1
    assert funds["XYZ"]["sector"] == "Technology Services"
    assert calls == ["XYZ"]

    # Second run, fresh frame: verdict is on disk → no new adjudication call.
    calls.clear()
    funds2 = {"XYZ": {"sector": "Retail Trade", "industry": "Internet Retail"}}
    out2 = ca.cross_validate(funds2, allow_fetch=True)
    assert out2["applied"] == 1
    assert funds2["XYZ"]["sector"] == "Technology Services"
    assert calls == []  # cache hit — Claude not re-queried


def test_manual_override_symbols_are_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr("util.CROSS_VALIDATION_PARAMS", {
        "enabled": True, "swing_threshold": 0.20, "profile_fetch_per_run": 10, "model": "m",
    })
    monkeypatch.setattr("util.CLASSIFICATION_OVERRIDES", {"XYZ": {"sector": "Technology Services"}})
    monkeypatch.setattr(ca, "ADJUDICATIONS_PATH", str(tmp_path / "adj.json"))

    seen = []
    monkeypatch.setattr(ca.fmp_client, "company_profile",
                        lambda s, allow_fetch=False: seen.append(s) or None)

    out = ca.cross_validate({"XYZ": {"sector": "Retail Trade"}}, allow_fetch=True)
    assert out["checked"] == 0
    assert seen == []  # never even fetched a profile for an override symbol
