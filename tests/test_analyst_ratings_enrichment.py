"""
Tests for real analyst-ratings enrichment of archetype classification.

Before: the classifier back-derived analyst_buy_pct from buy_to_sell_ratio via a
bucket lookup, which was off by up to ~28pp on real names (e.g. T: real 54% buy
consensus vs heuristic 82%). Now market_structure._fetch_ratings pulls exact
buy/hold/sell counts from Robinhood get_ratings, and the classifier consumes the
exact analyst_buy_pct when present.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_fetch_ratings_computes_exact_buy_pct():
    """_fetch_ratings turns buy/hold/sell counts into the exact consensus fraction."""
    from data import market_structure

    fake = {
        "summary": {
            "num_buy_ratings": 15,
            "num_hold_ratings": 13,
            "num_sell_ratings": 0,
        }
    }
    with patch("robin_stocks.robinhood.get_ratings", return_value=fake):
        out = market_structure._fetch_ratings(["T"])

    assert "T" in out
    # 15 / (15+13+0) = 0.5357 — the real value, NOT the ~0.82 the b/s heuristic gave
    assert out["T"]["analyst_buy_pct"] == pytest.approx(15 / 28, abs=1e-3)
    assert out["T"]["analyst_num_ratings"] == 28


def test_fetch_ratings_skips_empty_summary():
    """No ratings → symbol omitted (classifier falls back to heuristic gracefully)."""
    from data import market_structure

    with patch("robin_stocks.robinhood.get_ratings", return_value={"summary": None}):
        assert market_structure._fetch_ratings(["XYZ"]) == {}

    with patch("robin_stocks.robinhood.get_ratings", return_value={}):
        assert market_structure._fetch_ratings(["XYZ"]) == {}


def test_fetch_ratings_survives_api_error():
    """A get_ratings exception for one symbol must not abort the batch."""
    from data import market_structure

    def _boom(sym, info=None):
        if sym == "BAD":
            raise RuntimeError("api down")
        return {"summary": {"num_buy_ratings": 5, "num_hold_ratings": 0, "num_sell_ratings": 0}}

    with patch("robin_stocks.robinhood.get_ratings", side_effect=_boom):
        out = market_structure._fetch_ratings(["BAD", "GOOD"])

    assert "BAD" not in out
    assert out["GOOD"]["analyst_buy_pct"] == pytest.approx(1.0)


def test_classifier_prefers_exact_analyst_pct_over_ratio_heuristic():
    """When analyst_buy_pct is present, it overrides the buy_to_sell_ratio guess."""
    from portfolio.position_archetypes import _analyst_buy_pct

    # buy_to_sell_ratio=15 would bucket to ~0.82 via the heuristic, but the real
    # consensus is 0.54 — the exact value must win.
    signals = {"analyst_buy_pct": 0.5357, "buy_to_sell_ratio": 15.0}
    assert _analyst_buy_pct(signals) == pytest.approx(0.5357, abs=1e-4)

    # With no exact value, it still falls back to the ratio heuristic (unchanged).
    assert _analyst_buy_pct({"buy_to_sell_ratio": 15.0}) is not None
