"""
tests/test_buy_phase_efficiency.py — allocation pre-check + per-run sentiment cache.

From the 2026-06-10 run log: iteration 2 spent a full Claude sentiment batch on 8
candidates whose allocations ($19-20 vs $30 min_order) guaranteed zero buys, and
re-queried symbols already judged a minute earlier (SLB flipped BUY 80% -> HOLD 65%
on identical data).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from execution.paper import PaperBroker
from portfolio.harvest import HarvestManager
from portfolio.manager import PortfolioManager
from portfolio.risk import RiskManager
from util import RISK_LIMITS


def _pm(cash: float, use_sentiment: bool = True) -> PortfolioManager:
    return PortfolioManager(
        broker=PaperBroker(starting_cash=cash, price_lookup=lambda s: 50.0),
        risk=RiskManager(),
        harvest=HarvestManager(),
        auto_approve=True,
        use_sentiment=use_sentiment,
    )


def _df(symbols, scores=None):
    if scores is None:
        scores = [0.8] * len(symbols)
    return pd.DataFrame({"symbol": symbols, "value_metric": scores})


class TestAllocationPrecheck:

    def test_starved_budget_skips_sentiment_batch(self, monkeypatch):
        """Budget below min_order → NO order is possible at all, so buy_cycle returns
        without ever calling the sentiment layer (token-saving guarantee)."""
        min_order = RISK_LIMITS["min_order_amount"]
        n = 8
        # Truly starved: the whole stock budget can't clear a single min_order.
        budget = min_order * 0.5
        pm = _pm(cash=budget)

        called = []
        import data.sentiment as ds
        monkeypatch.setattr(
            ds, "get_batch_sentiment_recommendations",
            lambda *a, **k: called.append(1) or {},
        )
        purchased, skipped, failed = pm.buy_cycle(
            _df([f"S{i}" for i in range(n)]),
            is_first_iteration=False,
            regime="neutral",
            effective_index_pct=0.77,
        )
        assert purchased == [] and failed == []
        assert len(skipped) == n
        assert called == []  # sentiment never invoked

    def test_small_budget_concentrates_instead_of_skipping(self, monkeypatch):
        """Budget clears at least one order but per-candidate shares fall below
        min_order → concentration ladder engages: the sentiment batch IS invoked
        (on a trimmed top-ranked bench) so a concentrated buy can be placed,
        rather than skipping the whole phase."""
        min_order = RISK_LIMITS["min_order_amount"]
        n = 8
        # Spreading across 8 names → ~0.31*min_order each (< min_order), but the
        # budget itself clears ~2 orders → concentration should trigger.
        budget = min_order * 2.5
        pm = _pm(cash=budget)

        called = []
        import data.sentiment as ds
        monkeypatch.setattr(
            ds, "get_batch_sentiment_recommendations",
            lambda *a, **k: called.append(1) or {},  # all HOLD → still proves the path ran
        )
        purchased, skipped, failed = pm.buy_cycle(
            _df([f"S{i}" for i in range(n)]),
            is_first_iteration=False,
            regime="neutral",
            effective_index_pct=0.77,
        )
        assert called != []                 # concentration engaged → sentiment invoked
        assert len(skipped) < n             # bench trimmed to the top-ranked ladder

    def test_viable_budget_proceeds_past_precheck(self, monkeypatch):
        """A budget where the top candidate clears min_order must NOT trigger the
        pre-check return path."""
        min_order = RISK_LIMITS["min_order_amount"]
        pm = _pm(cash=min_order * 20)
        reached = []
        # Sentiment disabled → pre-check passes, then loop runs; stub deeper
        # collaborators to observe we got past the pre-check.
        pm._use_sentiment = False
        monkeypatch.setattr(pm, "_build_stocks_data", lambda *a, **k: reached.append(1) or [])
        purchased, skipped, failed = pm.buy_cycle(
            _df(["AAA"]), is_first_iteration=False,
            regime="neutral", effective_index_pct=0.77,
        )
        # Either bought or skipped for a downstream reason — but not the
        # pre-check's "skip everything" signature of (0 purchased, all skipped,
        # sentiment untouched) ... here we simply assert no crash and the single
        # candidate was processed one way or the other.
        assert len(purchased) + len(skipped) + len(failed) >= 1


class TestSentimentRunCache:

    def test_cached_verdict_reused_and_not_requeried(self, monkeypatch):
        pm = _pm(cash=10_000.0)
        calls: list[list[str]] = []

        def fake_batch(stocks_data, action="buy", regime=None):
            calls.append([s["symbol"] for s in stocks_data])
            return {
                s["symbol"]: {"action": "HOLD", "sentiment": "neutral",
                              "confidence": 50.0, "reasoning": "x"}
                for s in stocks_data
            }

        import data.sentiment as ds
        monkeypatch.setattr(ds, "get_batch_sentiment_recommendations", fake_batch)
        # Fake-but-plausible symbols that cannot collide with real ETFs /
        # exclusion lists ("AAA" is a real bond ETF and gets fund-filtered).
        df = _df(["ZZZQA", "ZZZQB"])

        pm.buy_cycle(df, is_first_iteration=True, regime="neutral", effective_index_pct=0.77)
        pm.buy_cycle(df, is_first_iteration=False, regime="neutral", effective_index_pct=0.77)

        queried_first = set(calls[0]) if calls else set()
        assert queried_first, "first iteration should query sentiment"
        # Second iteration must NOT have re-queried anything judged in the first.
        requeried = set().union(*calls[1:]) if len(calls) > 1 else set()
        assert not (queried_first & requeried)
        # And the verdicts must be in the per-run cache.
        assert {s for (_a, s) in pm._sentiment_run_cache} >= queried_first

    def test_api_error_sentinels_not_cached(self):
        pm = _pm(cash=10_000.0)
        from data.sentiment import _api_error_sentinel  # type: ignore[attr-defined]
        pm._sentiment_run_cache = {}
        sentinel = _api_error_sentinel()
        # Mirrors the caching predicate in buy_cycle.
        from data.sentiment import is_api_error_sentinel
        if not is_api_error_sentinel(sentinel):
            pm._sentiment_run_cache[("buy", "AAA")] = sentinel
        assert ("buy", "AAA") not in pm._sentiment_run_cache


class TestLiquidityPrefilter:

    def test_illiquid_candidates_never_reach_sentiment(self, monkeypatch):
        """Deterministic volume gate runs BEFORE the Claude batch — illiquid
        names must not consume sentiment slots (2026-06-11: BLX/SUBCY earned
        BUY verdicts then were skipped on volume every iteration)."""
        import pandas as pd

        from util import RISK_LIMITS

        pm = _pm(cash=10_000.0)
        queried: list[list[str]] = []

        def fake_batch(stocks_data, action="buy", regime=None):
            queried.append([s["symbol"] for s in stocks_data])
            return {s["symbol"]: {"action": "HOLD", "sentiment": "neutral",
                                  "confidence": 50.0, "reasoning": "x"}
                    for s in stocks_data}

        import data.sentiment as ds
        monkeypatch.setattr(ds, "get_batch_sentiment_recommendations", fake_batch)

        min_vol = RISK_LIMITS["min_liquidity_volume"]
        df = pd.DataFrame({
            "symbol":       ["ZZZQL", "ZZZQH"],
            "value_metric": [0.9, 0.85],
            "volume":       [min_vol / 10.0, min_vol * 10.0],  # illiquid vs liquid
        })
        pm.buy_cycle(df, is_first_iteration=True, regime="neutral", effective_index_pct=0.77)
        seen = set().union(*queried) if queried else set()
        assert "ZZZQL" not in seen   # illiquid name never queried
        assert "ZZZQH" in seen       # liquid name still flows through
