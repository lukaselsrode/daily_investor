"""
tests/test_risk.py — RiskManager and portfolio sizing tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd

from util import RISK_LIMITS

# ---------------------------------------------------------------------------
# RiskManager tests
# ---------------------------------------------------------------------------

from portfolio.risk import BuyDecision, RiskManager


class TestRiskManagerCanBuy:

    def _agg(self, symbol: str, volume: float = 1_000_000, sector: str = "Technology") -> pd.DataFrame:
        return pd.DataFrame([{"symbol": symbol, "volume": volume, "sector": sector}])

    def _rm(self) -> RiskManager:
        return RiskManager()

    def test_approves_valid_buy(self):
        d = self._rm().can_buy("AAPL", 100.0, {}, self._agg("AAPL"), 10_000.0, 500.0)
        assert d.approved
        assert d.adjusted_allocation > 0

    def test_rejects_below_min_order(self):
        min_order = RISK_LIMITS["min_order_amount"]
        d = self._rm().can_buy("AAPL", min_order * 0.5, {}, self._agg("AAPL"), 10_000.0, 500.0)
        assert not d.approved
        assert "min_order" in d.reason

    def test_rejects_low_liquidity(self):
        min_vol = RISK_LIMITS["min_liquidity_volume"]
        d = self._rm().can_buy("TINY", 100.0, {}, self._agg("TINY", volume=min_vol * 0.5), 10_000.0, 500.0)
        assert not d.approved
        assert "volume" in d.reason.lower()

    def test_caps_to_max_order_pct(self):
        max_pct = RISK_LIMITS["max_order_pct_of_cash"]
        cash = 1_000.0
        large_alloc = cash * max_pct * 2
        d = self._rm().can_buy("AAPL", large_alloc, {}, self._agg("AAPL"), 20_000.0, cash)
        assert d.approved
        assert d.adjusted_allocation <= cash * max_pct + 1e-6

    def test_sector_cap_blocks_oversized_sector(self):
        max_sector = RISK_LIMITS["max_sector_pct"]
        portfolio_value = 10_000.0
        sector_exposure = {"Technology": portfolio_value * max_sector}
        d = self._rm().can_buy(
            "AAPL", 100.0, {}, self._agg("AAPL"), portfolio_value, 1_000.0,
            sector_exposure=sector_exposure,
        )
        assert not d.approved
        assert "sector" in d.reason.lower()

    def test_single_position_cap_blocks_oversize(self):
        max_single = RISK_LIMITS["max_single_position_pct"]
        portfolio_value = 10_000.0
        # Already at the cap
        holdings = {"AAPL": {"equity": str(portfolio_value * max_single)}}
        d = self._rm().can_buy("AAPL", 100.0, holdings, self._agg("AAPL"), portfolio_value, 1_000.0)
        assert not d.approved
        assert "position cap" in d.reason

    def test_returns_buy_decision_dataclass(self):
        d = self._rm().can_buy("AAPL", 100.0, {}, self._agg("AAPL"), 10_000.0, 500.0)
        assert isinstance(d, BuyDecision)

    def test_get_sector_exposure_sums_by_sector(self):
        holdings = {
            "AAPL": {"equity": "300.0"},
            "MSFT": {"equity": "200.0"},
        }
        agg = pd.DataFrame([
            {"symbol": "AAPL", "sector": "Technology"},
            {"symbol": "MSFT", "sector": "Technology"},
        ])
        exp = self._rm().get_sector_exposure(holdings, agg)
        assert exp.get("Technology", 0.0) == pytest.approx(500.0)

    def test_get_sector_exposure_no_agg(self):
        holdings = {"AAPL": {"equity": "100.0"}}
        exp = self._rm().get_sector_exposure(holdings, None)
        assert exp.get("Unknown", 0.0) == pytest.approx(100.0)
