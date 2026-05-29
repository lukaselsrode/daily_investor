"""
tests/test_cluster_enforcement.py — Cluster concentration enforcement helpers.

Tests the pure-function decision logic in
src/portfolio/exposure/cluster_enforcement.py used by both live (manager.buy_cycle)
and backtest (simulator._do_buy) to enforce / downsize / block buys that would
push cluster weight above the configured cap.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio.exposure.cluster_enforcement import (
    cluster_buy_decision,
    current_cluster_weight,
    projected_cluster_weight,
)


def _enf(**overrides) -> dict:
    base = {
        "block_new_buys": True,
        "allow_existing_positions": True,
        "allow_trim_only": True,
        "allow_sell": True,
        "allow_if_underweight": True,
        "downsize_to_fit": True,
        "min_remaining_alloc_multiple": 1.0,
    }
    base.update(overrides)
    return base


def test_cluster_cap_blocks_new_buy_when_no_downsize():
    """When downsize disabled and proposed buy would exceed cap → blocked."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.49, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.50, enforcement_cfg=_enf(downsize_to_fit=False),
        min_order_amount=30.0,
    )
    assert decision == "blocked"
    assert alloc == 0.0


def test_cluster_cap_downsizes_to_fit():
    """When downsize_to_fit=true and headroom > min_order, alloc is reduced to fit."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.40, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.50, enforcement_cfg=_enf(),
        min_order_amount=30.0,
    )
    assert decision == "downsized"
    assert abs(alloc - 100.0) < 1e-6   # headroom = (0.50 - 0.40) * 1000 = 100


def test_cluster_cap_blocks_when_downsize_below_min_order():
    """Downsize fits below min_order_amount → block."""
    decision, alloc, reason = cluster_buy_decision(
        "X", "c1", current_weight=0.495, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.50, enforcement_cfg=_enf(min_remaining_alloc_multiple=2.0),
        min_order_amount=30.0,
    )
    assert decision == "blocked"
    assert alloc == 0.0
    assert "min" in reason.lower()


def test_cluster_cap_allows_buy_within_cap():
    """When projected weight stays under cap → allowed at full alloc."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.20, alloc=100.0, portfolio_value=1000,
        cluster_limit=0.60, enforcement_cfg=_enf(),
        min_order_amount=30.0,
    )
    assert decision == "allowed"
    assert alloc == 100.0


def test_cluster_cap_no_assignment_fallback():
    """cluster_id=None → allowed (we don't block on missing data)."""
    decision, alloc, _ = cluster_buy_decision(
        "X", None, current_weight=0.99, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.50, enforcement_cfg=_enf(),
        min_order_amount=30.0,
    )
    assert decision == "allowed"
    assert alloc == 500.0


def test_cluster_cap_underweight_exemption():
    """When sleeve is underweight AND allow_if_underweight=true → allowed at full size."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.80, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.50, enforcement_cfg=_enf(allow_if_underweight=True),
        min_order_amount=30.0,
        is_underweight=True,
    )
    assert decision == "allowed"
    assert alloc == 500.0


def test_current_cluster_weight_sums_held_equity():
    holdings = {
        "A": {"equity": 100.0},
        "B": {"equity": 200.0},
        "C": {"equity": 300.0},
    }
    lookup = {"A": "c1", "B": "c1", "C": "c2"}
    w = current_cluster_weight(holdings, lookup, portfolio_value=1000.0, cluster_id="c1")
    assert abs(w - 0.30) < 1e-6   # (100 + 200) / 1000


def test_projected_cluster_weight():
    p = projected_cluster_weight(current_weight=0.40, alloc=100.0, portfolio_value=1000.0)
    assert abs(p - 0.50) < 1e-6


def test_zero_portfolio_value_returns_allowed():
    """No portfolio value → no enforcement possible → allow."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.0, alloc=500.0, portfolio_value=0,
        cluster_limit=0.50, enforcement_cfg=_enf(),
        min_order_amount=30.0,
    )
    assert decision == "allowed"
    assert alloc == 500.0


def test_zero_cap_returns_allowed():
    """cluster_limit=0 means 'unset' — don't enforce."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.50, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.0, enforcement_cfg=_enf(),
        min_order_amount=30.0,
    )
    assert decision == "allowed"
    assert alloc == 500.0
