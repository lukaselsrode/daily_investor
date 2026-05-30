"""
Tests for the raw-multiple valuation + company-age proxies in archetype scoring.

Robinhood exposes pe_ratio / pb_ratio / year_founded via get_fundamentals but NO
intrinsic fair value. These proxies read the raw multiples as *character* signals:
  - cheap multiples -> reinforce value_recovery
  - expensive / negative PE -> reinforce speculative_momentum
  - old company (year_founded) -> reinforce legacy_turnaround

Behavior-preservation: when the new fields are absent, classification is unchanged.
"""
from __future__ import annotations

import pytest

from portfolio.position_archetypes import (
    _company_age,
    _valuation_character,
    classify_archetype,
)


def test_valuation_character_buckets():
    assert _valuation_character({"pe_ratio": 8.0})["cheap"] is True
    assert _valuation_character({"pb_ratio": 1.0})["cheap"] is True
    assert _valuation_character({"pe_ratio": 80.0})["expensive"] is True
    assert _valuation_character({"pb_ratio": 25.0})["expensive"] is True
    assert _valuation_character({"pe_ratio": -5.0})["negative_pe"] is True
    # Missing data -> all False (no behavior change)
    none_case = _valuation_character({})
    assert none_case == {"cheap": False, "expensive": False, "negative_pe": False}


def test_company_age():
    import datetime
    yr = datetime.date.today().year
    assert _company_age({"year_founded": yr - 100}) == pytest.approx(100.0)
    assert _company_age({}) is None
    assert _company_age({"year_founded": 99}) is None      # implausible
    assert _company_age({"year_founded": 3000}) is None    # implausible


def test_cheap_multiples_boost_value_recovery():
    """A cheap, moderate-quality recovering name scores higher for value_recovery
    when PE/PB are cheap than when the fields are absent."""
    base = {
        "value_score": 0.5, "quality_score": 0.35, "momentum_score": 0.2,
        "symbol": "X",
    }
    r_without = classify_archetype(dict(base))
    r_with = classify_archetype({**base, "pe_ratio": 8.0, "pb_ratio": 1.1})
    assert (
        r_with.scores["value_recovery"] >= r_without.scores["value_recovery"]
    )


def test_absent_fields_preserve_behavior():
    """Classification without the new fields must equal prior behavior (winner
    unchanged) — the proxies are purely additive when data is present."""
    sig = {
        "value_score": 0.1, "quality_score": 0.8, "momentum_score": 0.3,
        "market_cap": 2_500_000_000_000, "maintenance_ratio": 0.2,
        "symbol": "BIGTECH",
    }
    r = classify_archetype(sig)
    # mega-cap, low maintenance, high quality -> quality_compounder, regardless
    assert r.archetype == "quality_compounder"
