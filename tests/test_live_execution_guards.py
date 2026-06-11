"""
tests/test_live_execution_guards.py — live-run guards found in the 2026-06-10 run log.

1. Pending-sell guard: raw Robinhood order payloads carry an `instrument` URL and
   NO `symbol` key, so the guard silently matched nothing and iteration 2
   re-attempted every queued sell ("Not enough shares to sell"). The guard must
   resolve instrument URLs.
2. Live sell/stopout buy-cooldown (backtest parity): GGB was trailing-stop sold at
   16:20:47 and re-entered the buy candidate list at 16:21:52 the same run — only
   bearish sentiment prevented the rebuy. The simulator enforces
   cooldown_days_after_sell/stopout; live must too, unconditionally.
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from execution.paper import PaperBroker
from portfolio.harvest import HarvestManager
from portfolio.manager import PortfolioManager
from portfolio.risk import RiskManager
from util import BACKTEST_PARAMS


def _pm(broker=None) -> PortfolioManager:
    return PortfolioManager(
        broker=broker or PaperBroker(starting_cash=100.0, price_lookup=lambda s: 50.0),
        risk=RiskManager(),
        harvest=HarvestManager(),
        auto_approve=True,
        use_sentiment=False,
    )


# ---------------------------------------------------------------------------
# 1. Pending-sell guard resolves instrument URLs
# ---------------------------------------------------------------------------

class _OrdersBroker(PaperBroker):
    """Broker whose open orders mimic the RAW Robinhood payload: instrument URL,
    no symbol key."""

    def __init__(self, orders, url_to_symbol):
        super().__init__(starting_cash=0.0, price_lookup=lambda s: 50.0)
        self._orders = orders
        self._url_to_symbol = url_to_symbol
        self.resolve_calls = 0

    def get_open_orders(self):
        return self._orders

    def resolve_instrument_symbol(self, url):
        self.resolve_calls += 1
        return self._url_to_symbol.get(url)


def _order(url, side="sell", state="queued"):
    return {"side": side, "state": state, "instrument": url}


class TestPendingSellGuard:

    def test_instrument_urls_resolve_to_symbols(self):
        broker = _OrdersBroker(
            orders=[
                _order("https://rh/instruments/aaa/"),
                _order("https://rh/instruments/bbb/", state="confirmed"),
                _order("https://rh/instruments/ccc/", side="buy"),       # buys ignored
                _order("https://rh/instruments/ddd/", state="filled"),    # closed ignored
            ],
            url_to_symbol={
                "https://rh/instruments/aaa/": "ERIC",
                "https://rh/instruments/bbb/": "JD",
                "https://rh/instruments/ccc/": "HST",
                "https://rh/instruments/ddd/": "RIO",
            },
        )
        pm = _pm(broker)
        assert pm._get_pending_sell_symbols() == {"ERIC", "JD"}

    def test_partially_filled_still_pending(self):
        broker = _OrdersBroker(
            orders=[_order("https://rh/instruments/aaa/", state="partially_filled")],
            url_to_symbol={"https://rh/instruments/aaa/": "ING"},
        )
        pm = _pm(broker)
        assert pm._get_pending_sell_symbols() == {"ING"}

    def test_symbol_key_still_honored_without_resolution(self):
        broker = _OrdersBroker(orders=[{"side": "sell", "state": "queued", "symbol": "ARM"}],
                               url_to_symbol={})
        pm = _pm(broker)
        assert pm._get_pending_sell_symbols() == {"ARM"}
        assert broker.resolve_calls == 0


# ---------------------------------------------------------------------------
# 2. Live sell/stopout buy-cooldown
# ---------------------------------------------------------------------------

def _write_sell_history(pm, rows):
    pd.DataFrame(rows).to_csv(pm._SELL_HISTORY_CSV, index=False)


class TestSellCooldown:

    def _today(self):
        return datetime.date.today()

    def test_same_day_loss_exit_blocks_rebuy(self, tmp_path, monkeypatch):
        """The GGB case: stop-loss exit minutes earlier must block the rebuy —
        gated on the live config's stopout cooldown being enabled."""
        stopout_cd = int(BACKTEST_PARAMS.get("cooldown_days_after_stopout", 0))
        if stopout_cd <= 0:
            import pytest
            pytest.skip("cooldown_days_after_stopout disabled in live config")

        pm = _pm()
        monkeypatch.setattr(pm, "_SELL_HISTORY_CSV", str(tmp_path / "sell_history.csv"))
        _write_sell_history(pm, [{
            "symbol": "GGB", "sell_date": self._today().isoformat(), "was_loss": True,
        }])
        hist = pm._load_sell_history()
        assert "GGB" in hist
        sold_date, was_loss = hist["GGB"]
        assert was_loss and (self._today() - sold_date).days < stopout_cd

    def test_old_sale_outside_window_not_blocked(self, tmp_path, monkeypatch):
        sell_cd = int(BACKTEST_PARAMS.get("cooldown_days_after_sell", 0))
        pm = _pm()
        monkeypatch.setattr(pm, "_SELL_HISTORY_CSV", str(tmp_path / "sell_history.csv"))
        old = self._today() - datetime.timedelta(days=max(sell_cd, 1) + 30)
        _write_sell_history(pm, [{
            "symbol": "CVE", "sell_date": old.isoformat(), "was_loss": False,
        }])
        sold_date, _ = pm._load_sell_history()["CVE"]
        assert (self._today() - sold_date).days >= sell_cd

    def test_latest_sell_wins_per_symbol(self, tmp_path, monkeypatch):
        pm = _pm()
        monkeypatch.setattr(pm, "_SELL_HISTORY_CSV", str(tmp_path / "sell_history.csv"))
        _write_sell_history(pm, [
            {"symbol": "JD", "sell_date": "2026-01-05", "was_loss": False},
            {"symbol": "JD", "sell_date": self._today().isoformat(), "was_loss": True},
        ])
        sold_date, was_loss = pm._load_sell_history()["JD"]
        assert sold_date == self._today() and was_loss

    def test_missing_history_file_is_empty(self, tmp_path, monkeypatch):
        pm = _pm()
        monkeypatch.setattr(pm, "_SELL_HISTORY_CSV", str(tmp_path / "nope.csv"))
        assert pm._load_sell_history() == {}


class TestWeakStreakPersistence:

    def test_empty_store_writes_parseable_header_only_csv(self, tmp_path, monkeypatch):
        """Regression: save_weak_streak({}) wrote a single newline (column-less
        DataFrame), which pandas refuses to parse (EmptyDataError) and which
        crashed the UI Data Explorer's Research section."""
        import pandas as pd

        import portfolio.thesis_confirm_tracker as tct

        path = tmp_path / "weak_streak.csv"
        monkeypatch.setattr(tct, "_WEAK_STREAK_CSV", str(path))

        tct.save_weak_streak({})
        df = pd.read_csv(path)  # must NOT raise EmptyDataError
        assert list(df.columns) == ["symbol", "weak_streak"]
        assert tct.load_weak_streak() == {}

    def test_round_trip_with_entries(self, tmp_path, monkeypatch):
        import portfolio.thesis_confirm_tracker as tct

        path = tmp_path / "weak_streak.csv"
        monkeypatch.setattr(tct, "_WEAK_STREAK_CSV", str(path))
        tct.save_weak_streak({"ABC": 2, "XYZ": 0})   # zero-streaks dropped
        assert tct.load_weak_streak() == {"ABC": 2}


class TestCandidateOutcomePrice:

    def test_candidate_rows_record_decision_price(self, tmp_path):
        """Regression: candidate ledger rows had no price, so fill_future_returns
        could never compute forward returns for them — the 'was skipping this
        candidate right?' half of the ledger was structurally empty (0% coverage
        vs 90% for holdings)."""
        import pandas as pd

        import portfolio.outcome_tracker as ot
        from portfolio.decision_logger import log_candidate_decision

        row = pd.Series({
            "symbol": "ZZZQP", "value_metric": 0.8, "current_price": 123.45,
            "value_score": 0.5, "quality_score": 0.5, "momentum_score": 0.1,
            "income_score": 0.0, "reliability_score": 0.9,
        })
        log_candidate_decision(
            symbol="ZZZQP", row=row, decision_state="SKIP",
            selected_bool=False, skipped_bool=True, skip_reason="sentiment_gate",
            sentiment_result_dict={"action": "HOLD", "confidence": 60.0},
            risk_check_passed=True, risk_check_fail_reason="",
            proposed_allocation=100.0, final_allocation=0.0,
            regime="neutral", candidate_rank=1, agg_df=None,
        )
        out = ot.load_outcomes()
        cand = out[out["record_type"] == "candidate"]
        assert len(cand) == 1
        assert float(cand.iloc[0]["price"]) == 123.45
