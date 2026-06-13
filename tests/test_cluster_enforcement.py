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


# ---------------------------------------------------------------------------
# Active-sleeve denominator: caps must bind the SLEEVE, not total PV.
# These encode the bug→fix: the buy-cycle/simulator now pass active_sleeve_value
# (= (1-index_pct)*PV) as the `portfolio_value` denominator, so a 60%/40% cap
# constrains the stock book even when the sleeve is a tiny fraction of total PV.
# ---------------------------------------------------------------------------

def test_cluster_cap_binds_against_small_active_sleeve():
    """A cluster at 80% of a tiny active sleeve must trip the 0.60 cap, even though
    that same dollar amount is only ~4% of total PV (the old, broken denominator)."""
    total_pv      = 10_000.0
    index_pct     = 0.95
    sleeve_value  = (1.0 - index_pct) * total_pv   # $500 active sleeve
    cluster_equity = 400.0                          # 80% of sleeve, 4% of total PV
    new_buy        = 100.0

    cur_w = cluster_equity / sleeve_value           # 0.80
    # Under the OLD denominator the weight would be 400/10000 = 0.04 — allowed.
    assert cluster_equity / total_pv < 0.60

    decision, alloc, _ = cluster_buy_decision(
        "MARINE", "c1", current_weight=cur_w, alloc=new_buy,
        portfolio_value=sleeve_value,           # active-sleeve denominator (the fix)
        cluster_limit=0.60, enforcement_cfg=_enf(),
        min_order_amount=30.0, is_underweight=False,
    )
    # 0.80 already over the 0.60 cap → no headroom → blocked.
    assert decision == "blocked"
    assert alloc == 0.0


def test_underweight_exemption_not_applied_at_buy_time():
    """The live buy path passes is_underweight=False so the per-cluster cap binds
    during the initial sleeve build-out (the cold-start gap). Confirm that without
    the exemption an over-cap cluster is still blocked."""
    decision, alloc, _ = cluster_buy_decision(
        "X", "c1", current_weight=0.80, alloc=500.0, portfolio_value=1000,
        cluster_limit=0.60, enforcement_cfg=_enf(allow_if_underweight=True),
        min_order_amount=30.0, is_underweight=False,   # buy path always passes False
    )
    assert decision == "blocked"


def test_sector_cap_uses_same_decision_fn_shipping_banks_reits():
    """The active-sleeve sector cap reuses cluster_buy_decision with the sector label.
    Three sectors each under the cap are fine; a 4th buy that would push one sector
    over max_sector_weight of the sleeve is downsized/blocked. Uses the LIVE config
    cap so the test tracks config, not a hardcoded threshold."""
    from util import CONCENTRATION_LIMIT_PARAMS

    max_sec_w = float(CONCENTRATION_LIMIT_PARAMS["max_sector_weight"])
    sleeve    = 1000.0
    # Banks already at the cap (max_sec_w of the sleeve).
    banks_equity = max_sec_w * sleeve
    decision, alloc, _ = cluster_buy_decision(
        "JPM", "Financial", current_weight=banks_equity / sleeve, alloc=200.0,
        portfolio_value=sleeve, cluster_limit=max_sec_w,
        enforcement_cfg=CONCENTRATION_LIMIT_PARAMS["enforcement"],
        min_order_amount=30.0, is_underweight=False,
    )
    assert decision in ("blocked", "downsized")
    # A different sector well under the cap is still allowed at full size.
    decision2, alloc2, _ = cluster_buy_decision(
        "XOM", "Energy", current_weight=0.0, alloc=200.0,
        portfolio_value=sleeve, cluster_limit=max_sec_w,
        enforcement_cfg=CONCENTRATION_LIMIT_PARAMS["enforcement"],
        min_order_amount=30.0, is_underweight=False,
    )
    assert decision2 == "allowed"
    assert alloc2 == 200.0


def test_cold_start_universe_lookup_is_populated(monkeypatch):
    """Regression: at cold start (only ETFs held, no active stock) the concentration
    report must still carry universe_cluster_lookup so the buy-time cluster cap can
    constrain the FIRST active-sleeve buys. Previously the empty-sleeve branch
    returned an empty lookup → the cap was inert during build-out."""
    import pandas as pd

    import portfolio.exposure.cluster_concentration as cc

    # Stub the (heavy) PCA/KMeans factor map with a deterministic universe→cluster map.
    def _fake_factor_map(agg_df, **_kw):
        mapped = pd.DataFrame({
            "symbol":  ["AAA", "BBB", "CCC"],
            "cluster": ["c0", "c0", "c1"],
        })
        return None, mapped, None

    monkeypatch.setattr(
        "portfolio.visualization.factor_map.build_factor_map", _fake_factor_map
    )

    # Holdings = ETF only → active sleeve is empty.
    holdings_df = pd.DataFrame({"symbol": ["SPY"], "equity": [1000.0], "quantity": [10.0]})
    agg_df      = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "sector": ["T", "T", "F"]})

    report = cc.compute_concentration(
        holdings_df, agg_df, etfs=["SPY"], n_clusters=2,
    )
    assert report.n_active_positions == 0
    # The fix: lookup is populated even though the active sleeve is empty.
    assert report.universe_cluster_lookup == {"AAA": "c0", "BBB": "c0", "CCC": "c1"}
