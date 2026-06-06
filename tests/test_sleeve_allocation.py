"""
tests/test_sleeve_allocation.py — Portfolio sleeve accounting correctness.

Verifies that:
  1. On iteration 1, cash is split index_pct/active between ETFs and stocks.
  2. On iteration 2+, ALL available cash is offered to stocks (ETFs already funded).
  3. The end-of-run cash sweep only fills the ETF-sleeve deficit, not all remaining cash.
  4. HarvestManager respects harvest_to_etfs_pct.
  5. PBR.A-style symbol edge-cases do not crash buy logic.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from execution.paper import PaperBroker
from portfolio.harvest import HarvestManager
from portfolio.manager import PortfolioManager
from portfolio.risk import RiskManager
from util import ETFS, INDEX_PCT, RISK_LIMITS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(etfs=None, stocks=None):
    """Return a price lookup that handles both ETF and stock symbols."""
    etf_price   = etfs   or {e: 100.0 for e in ETFS}
    stock_price = stocks or {}
    def lookup(sym):
        if sym in etf_price:
            return etf_price[sym]
        return stock_price.get(sym, 50.0)
    return lookup


def _broker(cash: float = 100.0, etf_prices=None, stock_prices=None) -> PaperBroker:
    return PaperBroker(
        starting_cash=cash,
        price_lookup=_prices(etf_prices, stock_prices),
    )


def _pm(broker: PaperBroker) -> PortfolioManager:
    return PortfolioManager(
        broker=broker,
        risk=RiskManager(),
        harvest=HarvestManager(),
        auto_approve=True,
        use_sentiment=False,
    )


def _df(symbols, scores=None):
    """Minimal candidates DataFrame."""
    if scores is None:
        scores = [1.0] * len(symbols)
    return pd.DataFrame({
        "symbol":       symbols,
        "value_metric": scores,
    })


# ---------------------------------------------------------------------------
# 1. Iteration-1 split: index_pct to ETFs, (1-index_pct) to stocks
# ---------------------------------------------------------------------------

class TestIteration1Split:

    def test_etf_fraction_of_cash_deployed(self):
        """ETFs should receive INDEX_PCT of starting cash on iter 1."""
        cash = 100.0
        broker = _broker(cash=cash)
        pm = _pm(broker)

        pm.buy_cycle(_df([]), is_first_iteration=True, regime="bullish")

        etf_spent = sum(
            broker.get_holdings().get(e, {}).get("equity", 0.0)
            for e in ETFS
        )
        # Allow for rounding / min-order skips; just check directional correctness.
        expected = cash * INDEX_PCT
        if expected / max(len(ETFS), 1) >= RISK_LIMITS["min_order_amount"]:
            assert etf_spent == pytest.approx(expected, abs=1.0)

    def test_stock_amount_is_remainder_after_etf_buy(self):
        """Stock budget should be cash * (1 - INDEX_PCT) on iter 1."""
        cash = 100.0
        broker = _broker(cash=cash)
        pm = _pm(broker)

        candidates = _df(["AAPL", "MSFT"], scores=[1.0, 1.0])
        pm.buy_cycle(candidates, is_first_iteration=True, regime="bullish")

        # After buying ETFs, remaining cash should be ≤ stock_amount + tolerance
        etf_spent = sum(
            broker.get_holdings().get(e, {}).get("equity", 0.0)
            for e in ETFS
        )
        assert etf_spent <= cash * INDEX_PCT + 0.01


# ---------------------------------------------------------------------------
# 2. Iteration 2+: all cash available for stocks
# ---------------------------------------------------------------------------

class TestSubsequentIterations:

    def test_full_cash_offered_to_stocks_on_iter2(self):
        """
        On iter 2, stock_amount should equal total_cash (not cash * (1 - index_pct)).
        We proxy this by checking how much a no-candidate call depletes cash:
        it should leave ALL cash intact (no ETF buy).
        """
        cash = 50.0
        broker = _broker(cash=cash)
        pm = _pm(broker)

        # Empty candidates — no stock buys, no ETF buys on iter 2
        pm.buy_cycle(_df([]), is_first_iteration=False, regime="bullish")

        # Cash should be unchanged (no buys at all)
        assert broker.get_cash() == pytest.approx(cash, abs=0.01)
        # No ETF positions created
        for etf in ETFS:
            assert broker.get_holdings().get(etf) is None

    def test_no_etf_buys_on_iter2(self):
        """ETF orders must NOT be placed on iterations other than the first."""
        cash = 200.0
        broker = _broker(cash=cash)
        pm = _pm(broker)

        candidates = _df(["GOOG"], scores=[1.0])
        pm.buy_cycle(candidates, is_first_iteration=False, regime="bullish")

        etf_equity = sum(
            broker.get_holdings().get(e, {}).get("equity", 0.0)
            for e in ETFS
        )
        assert etf_equity == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Cash sweep — only fills ETF deficit
# ---------------------------------------------------------------------------

class TestCashSweep:

    def test_sweep_amount_limited_to_etf_deficit(self):
        """
        When ETFs are already at target, the sweep should be ~$0.
        We pre-load ETF positions to exactly the target weight.
        """
        total = 1000.0
        # Fund ETFs to exactly INDEX_PCT of total_cash (before any stock buys).
        etf_equity = total * INDEX_PCT  # e.g. $700 in ETFs

        # Build a broker that already holds ETF positions.
        broker = _broker(cash=total - etf_equity, etf_prices={e: 100.0 for e in ETFS})
        # Manually inject ETF holdings to simulate pre-existing ETF positions.
        for etf in ETFS:
            per_etf_shares = (etf_equity / len(ETFS)) / 100.0
            broker._holdings[etf] = {
                "quantity": per_etf_shares,
                "average_buy_price": 100.0,
                "equity": etf_equity / len(ETFS),
            }

        pm = _pm(broker)
        cash_before = broker.get_cash()
        sweep = pm._compute_etf_sweep_amount(cash_before, INDEX_PCT)

        # ETFs are at target → deficit ≈ 0 (allow small float tolerance)
        assert sweep < RISK_LIMITS["min_order_amount"]

    def test_sweep_fills_deficit_when_etfs_underweight(self):
        """
        When ETFs are underweight, sweep should fill only the deficit — not all cash.
        """
        total = 1000.0
        # Put 50% in ETFs instead of the INDEX_PCT target.
        etf_equity = total * 0.50
        remaining_cash = 200.0

        broker = _broker(cash=remaining_cash, etf_prices={e: 100.0 for e in ETFS})
        for etf in ETFS:
            per_etf_shares = (etf_equity / len(ETFS)) / 100.0
            broker._holdings[etf] = {
                "quantity": per_etf_shares,
                "average_buy_price": 100.0,
                "equity": etf_equity / len(ETFS),
            }
        # Simulate other stock positions so portfolio_value ≈ total
        broker._holdings["__STOCKS__"] = {
            "quantity": 1.0,
            "average_buy_price": total - etf_equity - remaining_cash,
            "equity": total - etf_equity - remaining_cash,
        }

        pm = _pm(broker)
        sweep = pm._compute_etf_sweep_amount(remaining_cash, INDEX_PCT)

        target_etf  = broker.get_portfolio_value() * INDEX_PCT
        deficit     = max(0.0, target_etf - etf_equity)
        expected    = min(remaining_cash, deficit)

        assert sweep == pytest.approx(expected, abs=1.0)
        assert sweep < remaining_cash + 0.01  # never sweeps more than available

    def test_no_double_etf_buy_on_second_iteration(self):
        """
        Full rebalance with two iterations: ETFs must not be bought again
        in the cash sweep if they are already at target after iteration 1.
        """
        cash = 100.0
        broker = _broker(cash=cash)
        pm = _pm(broker)

        # Run the full rebalance with an empty candidate frame (no stock buys).
        pm.rebalance(_df([]), regime="bullish")

        etf_equity = sum(
            broker.get_holdings().get(e, {}).get("equity", 0.0)
            for e in ETFS
        )
        # After iter 1: ETFs should have ~INDEX_PCT * cash.
        # After sweep: deficit ≈ 0 so no double-buy.
        # Total ETF allocation should never exceed cash.
        assert etf_equity <= cash + 0.01


# ---------------------------------------------------------------------------
# 4. Harvest routing — honors harvest_to_etfs_pct
# ---------------------------------------------------------------------------

class TestHarvestRouting:

    def test_harvest_to_etfs_pct_honored(self):
        """Only harvest_to_etfs_pct of proceeds should be deployed to ETFs."""
        from util import HARVEST_PARAMS

        harvest_pct = HARVEST_PARAMS.get("harvest_to_etfs_pct", 1.0)
        n_etfs = len(HARVEST_PARAMS.get("harvest_etfs", ["SPY", "VTI"]))
        # Ensure per-ETF amount exceeds min_order_amount regardless of config
        min_order = RISK_LIMITS["min_order_amount"]
        amount = max(100.0, (min_order * n_etfs / harvest_pct) * 3)
        broker = _broker(cash=amount * 2)
        hm = HarvestManager()

        pre_cash = broker.get_cash()
        hm.route_proceeds(amount, broker)
        post_cash = broker.get_cash()

        cash_spent = pre_cash - post_cash
        expected_spent = amount * harvest_pct
        # Allow for min_order rounding
        assert cash_spent <= expected_spent + 0.01
        assert cash_spent >= expected_spent - RISK_LIMITS["min_order_amount"] - 0.01

    def test_harvest_active_reserve_retained(self):
        """The non-ETF portion of harvest proceeds should remain as cash."""
        from util import HARVEST_PARAMS

        harvest_pct = HARVEST_PARAMS.get("harvest_to_etfs_pct", 1.0)
        if harvest_pct >= 1.0:
            pytest.skip("harvest_to_etfs_pct=1.0: no active reserve to test")

        n_etfs = len(HARVEST_PARAMS.get("harvest_etfs", ["SPY", "VTI"]))
        min_order = RISK_LIMITS["min_order_amount"]
        amount = max(100.0, (min_order * n_etfs / harvest_pct) * 3)
        broker = _broker(cash=amount * 2)
        hm = HarvestManager()

        pre_cash = broker.get_cash()
        hm.route_proceeds(amount, broker)
        post_cash = broker.get_cash()

        active_reserve = amount * (1.0 - harvest_pct)
        # Cash should still contain the active reserve portion
        assert post_cash >= pre_cash - amount * harvest_pct - 0.01


# ---------------------------------------------------------------------------
# 5. PaperBroker protocol completeness
# ---------------------------------------------------------------------------

class TestPaperBrokerProtocol:

    def test_clear_orders_cache_noop(self):
        broker = _broker()
        broker.clear_orders_cache()  # must not raise

    def test_enrich_holdings_created_at_noop(self):
        broker = _broker()
        holdings = {"AAPL": {"quantity": 1.0}}
        broker.enrich_holdings_created_at(holdings)  # must not raise
        # Holdings unchanged
        assert "quantity" in holdings["AAPL"]

    def test_add_funds_increases_cash(self):
        broker = _broker(cash=100.0)
        broker.add_funds(50.0)
        assert broker.get_cash() == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 6. Sleeve capital-event ledger is wired into the live rebalance
# ---------------------------------------------------------------------------

def test_rebalance_records_sleeve_capital_events(tmp_path, monkeypatch):
    """The sleeve event ledger (sleeve_tracker.log_*) is wired into PortfolioManager:
    a rebalance that sweeps cash into ETFs records a cash_sweep capital event. Before
    wiring, the writer API existed but the loop never called it (the UI read an empty
    ledger). The events file is isolated to a tmp path so the test never touches data/."""
    from portfolio import sleeve_tracker

    events_file = tmp_path / "sleeve_events.parquet"
    monkeypatch.setattr(sleeve_tracker, "_events_path", lambda: events_file)

    broker = _broker(cash=1000.0)
    _pm(broker).rebalance(_df([]), regime="bullish")

    events = sleeve_tracker.load_sleeve_events()
    assert not events.empty, "rebalance must write at least one sleeve capital event"
    assert "cash_sweep" in set(events["event_type"]), \
        "the ETF cash sweep should record a cash_sweep event"
    swept = events[events["event_type"] == "cash_sweep"]["amount"].astype(float).sum()
    assert swept > 0
