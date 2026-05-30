"""
Tests for the fund (ETF/CEF/MLP/ETN) short-circuit in archetype classification.

Reuses the shared core fund predicate (core.instruments) — the same etp/cef/mlp
detection the Factor Map "Stocks only" scope uses. Pooled funds must NOT be run
through the stock factor scorecards (which produced nonsense labels live, e.g. a
leveraged muni-bond CEF scored "speculative_momentum" off its day-trade ratio).
"""
from __future__ import annotations

from core.instruments import is_fund_asset_value, is_fund_instrument_type
from portfolio.position_archetypes import classify_archetype


def test_core_predicate():
    for t in ("etp", "ETP", "cef", "mlp", "etn"):
        assert is_fund_instrument_type(t)
    for t in ("stock", "adr", "reit", None, "", 123):
        assert not is_fund_instrument_type(t)
    assert is_fund_asset_value("ETF")
    assert is_fund_asset_value("fund")
    assert not is_fund_asset_value("equity")
    assert not is_fund_asset_value(None)


def test_cef_routes_to_fund_not_speculative():
    """A muni-bond CEF with a high day-trade ratio must classify as 'fund', not
    'speculative_momentum' (the live misclassification this fixes)."""
    sig = {
        "symbol": "MMU",
        "instrument_type": "cef",
        "day_trade_ratio": 0.45,   # would trigger speculative_momentum as a stock
        "maintenance_ratio": 0.50,
        "income_score": 0.6,
    }
    r = classify_archetype(sig)
    assert r.archetype == "fund"
    assert r.confidence == 1.0
    assert r.policy.archetype == "fund"
    # fund policy is conservative: wide stop, long min-hold
    assert r.policy.trailing_stop_pct <= -0.10
    assert r.policy.minimum_hold_days >= 30


def test_etp_and_explicit_is_etf_route_to_fund():
    assert classify_archetype({"symbol": "SPY", "instrument_type": "etp"}).archetype == "fund"
    assert classify_archetype({"symbol": "X", "is_etf": True}).archetype == "fund"
    assert classify_archetype({"symbol": "Y", "asset_type": "ETF"}).archetype == "fund"


def test_stock_still_classifies_normally():
    """A real operating company (no fund signal) is unaffected by the short-circuit."""
    sig = {
        "symbol": "BIGTECH",
        "instrument_type": "stock",
        "quality_score": 0.8,
        "market_cap": 2_500_000_000_000,
        "maintenance_ratio": 0.2,
    }
    r = classify_archetype(sig)
    assert r.archetype == "quality_compounder"


def test_adr_and_reit_are_not_funds():
    """ADR / REIT are individual equities, must NOT route to fund."""
    assert classify_archetype({"symbol": "NVO", "instrument_type": "adr",
                               "quality_score": 0.6, "market_cap": 3e11}).archetype != "fund"
    assert classify_archetype({"symbol": "O", "instrument_type": "reit",
                               "income_score": 0.7}).archetype != "fund"
