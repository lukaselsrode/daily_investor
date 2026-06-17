"""
tests/test_single_stock_analyzer.py — Single Stock Analyzer (decision-support, analysis-only).

No network: yfinance is injected as a fake module and the social fetchers + data cache are
monkeypatched with synthetic data. Covers holdings exposure, cached factors, price/trend +
leverage diagnostics, social spam/symbol filtering, fail-closed behavior without yfinance, the
end-to-end structured result, the pure position-structure math, a Streamlit render smoke test,
and the guardrail that the feature imports/calls no broker/execution code.
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

import data.cache as cache
import research.single_stock_analyzer as ssa

# Deterministic, oscillating series so beta/correlation are non-degenerate (300d → all windows).
_BASE = [100.0]
for _i in range(1, 300):
    _BASE.append(_BASE[-1] * (1 + 0.003 * math.sin(_i / 5.0)))
_LEV = [50.0]
for _i in range(1, 300):
    _r = _BASE[_i] / _BASE[_i - 1] - 1.0
    _LEV.append(_LEV[-1] * (1 + 2 * _r))          # daily-reset 2x of the base
_SPY = [400.0 * (1.0004 ** _i) for _i in range(300)]


def _hist(prices):
    idx = pd.date_range("2024-08-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=idx)


class _Chain:
    def __init__(self):
        self.calls = pd.DataFrame([
            {"strike": 120, "lastPrice": 2.1, "bid": 2.0, "ask": 2.2, "volume": 900,
             "openInterest": 5000, "impliedVolatility": 0.4}])
        self.puts = pd.DataFrame([
            {"strike": 110, "lastPrice": 1.8, "bid": 1.7, "ask": 1.9, "volume": 400,
             "openInterest": 3000, "impliedVolatility": 0.45}])


def _install_fake_yf(monkeypatch, hist_map, *, info=None, news=None, expiries=None):
    mod = types.ModuleType("yfinance")

    class _Tk:
        def __init__(self, sym):
            self.sym = sym

        def history(self, **_k):
            return hist_map.get(self.sym, pd.DataFrame())

        def get_info(self):
            return info or {}

        @property
        def news(self):
            return news or []

        @property
        def options(self):
            return list(expiries or [])

        def option_chain(self, _exp):
            return _Chain()

    mod.Ticker = _Tk
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    return mod


def _synthetic_repo(monkeypatch):
    holdings = pd.DataFrame([
        {"symbol": "SPY", "equity": 18000.0, "percentage": 60.0, "quantity": 25.0,
         "current_price": 700.0, "name": "S&P 500"},
        {"symbol": "BABA", "equity": 2000.0, "percentage": 6.7, "quantity": 17.0,
         "current_price": 117.0, "name": "Alibaba"},
    ])
    agg = pd.DataFrame([{"symbol": "BABA", "value_metric": 1.2, "value_score": 0.8,
                         "quality_score": 0.5, "momentum_score": -0.1, "pe_ratio": 17.5,
                         "pb_ratio": 1.6, "dividend_yield": 0.0}])

    def _fake_read(dataset):
        if "holdings" in dataset:
            return holdings
        if dataset == "agg_data":
            return agg
        return None

    monkeypatch.setattr(cache, "read_data_as_pd", _fake_read)


# ---------------------------------------------------------------------------
# Repo data (read-only)
# ---------------------------------------------------------------------------

def test_holdings_exposure_and_cached_factors(monkeypatch):
    _synthetic_repo(monkeypatch)
    exp = ssa.holdings_exposure(["BABA", "SPY", "BABU"])
    assert exp.status == "ok"
    assert exp.total_equity == pytest.approx(20000.0)
    assert "BABA" in exp.positions and exp.positions["BABA"]["equity"] == pytest.approx(2000.0)
    assert "BABU" not in exp.positions          # not held
    fac = ssa.cached_factors("baba")            # case-insensitive
    assert fac["value_metric"] == 1.2 and fac["pe_ratio"] == 17.5


def test_holdings_exposure_no_snapshot(monkeypatch):
    monkeypatch.setattr(cache, "read_data_as_pd", lambda ds: None)
    exp = ssa.holdings_exposure(["BABA"])
    assert exp.status == "no holdings snapshot found" and exp.total_equity == 0.0


# ---------------------------------------------------------------------------
# Pure position-structure math
# ---------------------------------------------------------------------------

def test_position_structure_math():
    t = ssa.position_structure(10000.0, 10.0, 5.0, 5.0)
    assert t["common_dollars"] == pytest.approx(1000.0)
    assert t["levered_dollars"] == pytest.approx(500.0)
    assert t["cash_dollars"] == pytest.approx(500.0)
    assert t["allocated_pct"] == pytest.approx(20.0)
    assert t["unallocated_pct"] == pytest.approx(80.0)
    assert t["warning"] is None
    over = ssa.position_structure(10000.0, 80.0, 30.0, 10.0)
    assert over["warning"] is not None          # sums to 120%


# ---------------------------------------------------------------------------
# Price / trend + leverage diagnostics (fake yfinance)
# ---------------------------------------------------------------------------

def test_price_snapshot_and_leverage_diagnostics(monkeypatch):
    _install_fake_yf(monkeypatch, {"BABA": _hist(_BASE), "BABU": _hist(_LEV), "SPY": _hist(_SPY)})
    trends, hist = ssa.price_snapshot(["BABA", "BABU", "SPY"])
    assert trends["BABA"].error is None and trends["BABA"].price is not None
    assert trends["BABA"].returns.get("1y") is not None       # 300d history → 1y window present
    lev = ssa.leveraged_diagnostics(hist, "BABA", "BABU")
    assert lev.realized_daily_beta is not None and lev.daily_corr is not None
    assert lev.realized_daily_beta > 1.5          # ~2x daily by construction
    assert "1m" in lev.periods and "tracking_gap" in lev.periods["1m"]


def test_price_snapshot_fail_closed(monkeypatch):
    # offline
    trends, _ = ssa.price_snapshot(["BABA"], allow_fetch=False)
    assert trends["BABA"].error and "disabled" in trends["BABA"].error
    # yfinance missing → import raises → fail closed
    monkeypatch.setitem(sys.modules, "yfinance", None)
    trends2, _ = ssa.price_snapshot(["BABA"])
    assert trends2["BABA"].error == "yfinance unavailable"


# ---------------------------------------------------------------------------
# Social scan — spam filtered, off-symbol dropped, provenance preserved
# ---------------------------------------------------------------------------

def test_social_scan_filters_spam_and_requires_symbol(monkeypatch):
    def _fake_reddit(subreddit, listing, limit, ttl_s, allow_fetch):
        if listing != "hot":
            return []
        return [
            {"title": "BABA calls into earnings", "selftext": "long $BABA setup",
             "permalink": "https://reddit.com/u1", "score": 50, "created_utc": 1.7e9},
            {"title": "Join my Telegram VIP 100X $BABA signals", "selftext": "",
             "permalink": "u2", "score": 5, "created_utc": 1.7e9},          # promo spam
            {"title": "AAPL to the moon buy AAPL", "selftext": "",
             "permalink": "u3", "score": 99, "created_utc": 1.7e9},          # off-symbol
        ]

    def _fake_x(query, limit, ttl_s):
        return ([{"text": "$BABA breakout looks strong", "created_at": "2025-06-16T14:00:00.000Z",
                  "id": "9"}], "ok")

    monkeypatch.setattr(ssa, "fetch_reddit_posts", _fake_reddit)
    monkeypatch.setattr(ssa, "fetch_x_mentions", _fake_x)
    scan = ssa.social_scan("BABA", "BABU")
    titles = " ".join(e["title"] for e in scan.evidence)
    assert "BABA calls" in titles
    assert "Telegram" not in titles and "AAPL" not in titles
    assert scan.mentions.get("BABA", 0) >= 2
    assert scan.quality_docs >= 2
    # provenance preserved on evidence rows
    assert all("source" in e and "url" in e for e in scan.evidence)
    assert any(e["source"] == "x" for e in scan.evidence)


def test_social_scan_offline():
    scan = ssa.social_scan("BABA", "BABU", allow_fetch=False)
    assert scan.raw_docs == 0 and scan.quality_docs == 0
    assert scan.statuses.get("status") == "live fetch disabled"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_analyze_end_to_end_structured(monkeypatch):
    _synthetic_repo(monkeypatch)
    _install_fake_yf(
        monkeypatch, {"BABA": _hist(_BASE), "BABU": _hist(_LEV), "SPY": _hist(_SPY)},
        info={"marketCap": 2.8e11, "trailingPE": 17.5, "beta": 0.6, "recommendationKey": "buy"},
        news=[{"title": "BABA earnings beat estimates", "publisher": "Reuters",
               "link": "https://x/y", "providerPublishTime": 1_700_000_000}],
        expiries=["2026-06-19"],
    )
    monkeypatch.setattr(ssa, "fetch_reddit_posts",
                        lambda **k: [{"title": "BABA calls", "selftext": "$BABA",
                                      "permalink": "u", "score": 9, "created_utc": 1.7e9}]
                        if k.get("listing") == "hot" else [])
    monkeypatch.setattr(ssa, "fetch_x_mentions", lambda *a, **k: ([], "skipped: no token"))

    res = ssa.analyze("baba", "babu", allow_fetch=True, include_social=True,
                      include_news=True, include_options=True)
    assert res.symbol == "BABA" and res.leverage_symbol == "BABU"
    assert "no orders" in res.disclaimer.lower()
    assert res.exposure.total_equity == pytest.approx(20000.0)
    assert res.cached_factors.get("value_metric") == 1.2
    assert res.price_trends["BABA"].price is not None
    assert res.fundamentals.get("trailingPE") == 17.5
    assert res.news and res.news[0]["title"].startswith("BABA")
    assert res.social is not None and res.social.quality_docs >= 1
    assert res.leverage is not None and res.leverage.periods
    assert res.options.get("first_expiry") == "2026-06-19" and res.options["calls_top_oi"]


def test_analyze_fail_closed_without_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    monkeypatch.setattr(cache, "read_data_as_pd", lambda ds: None)
    monkeypatch.setattr(ssa, "fetch_reddit_posts", lambda **k: [])
    monkeypatch.setattr(ssa, "fetch_x_mentions", lambda *a, **k: ([], "skipped"))
    res = ssa.analyze("BABA", "BABU")
    assert res.symbol == "BABA"
    assert res.price_trends["BABA"].error == "yfinance unavailable"
    assert res.fundamentals.get("status") == "yfinance unavailable"
    assert res.options.get("status") == "yfinance unavailable"


# ---------------------------------------------------------------------------
# UI render smoke (Streamlit testing harness — not a browser)
# ---------------------------------------------------------------------------

def test_single_stock_component_render_smoke():
    from streamlit.testing.v1 import AppTest

    def _app():
        from ui.components.single_stock_analyzer import render
        render()

    at = AppTest.from_function(_app).run()
    assert not at.exception                      # initial render (no Run click) must not crash


def test_single_stock_component_run_button_path(monkeypatch):
    """Exercise the non-default/run path without network by stubbing the UI service."""
    from dataclasses import dataclass, field

    from streamlit.testing.v1 import AppTest

    @dataclass
    class _Exposure:
        total_equity: float = 10_000.0
        positions: dict = field(default_factory=lambda: {"BABA": {"equity": 300.0}})
        status: str = "ok"

    @dataclass
    class _Trend:
        symbol: str = "BABA"
        price: float = 100.0
        returns: dict = field(default_factory=lambda: {"5d": 0.01, "1m": -0.02})
        sma20_gap: float = 0.01
        sma50_gap: float = -0.02
        sma200_gap: float = -0.10
        from_52w_high: float = -0.25
        vol20_ann: float = 0.35
        error: str | None = None

    @dataclass
    class _Social:
        raw_docs: int = 2
        quality_docs: int = 1
        mentions: dict = field(default_factory=lambda: {"BABA": 1})
        statuses: dict = field(default_factory=lambda: {"x": "skipped"})
        evidence: list = field(default_factory=lambda: [{
            "source": "x", "title": "$BABA test evidence", "url": "", "score": 0,
            "age_hours": 1,
        }])

    @dataclass
    class _Leverage:
        base_symbol: str = "BABA"
        leverage_symbol: str = "BABU"
        realized_daily_beta: float = 2.0
        daily_corr: float = 0.99
        periods: dict = field(default_factory=lambda: {"1m": {
            "base": 0.10, "lev": 0.19, "daily_2x_synth": 0.20, "tracking_gap": -0.01,
        }})
        note: str | None = None

    @dataclass
    class _Result:
        symbol: str = "BABA"
        leverage_symbol: str = "BABU"
        generated_at: str = "2026-06-15T00:00:00+00:00"
        disclaimer: str = "DECISION-SUPPORT / ANALYSIS ONLY — places NO orders."
        exposure: _Exposure = field(default_factory=_Exposure)
        cached_factors: dict = field(default_factory=lambda: {
            "symbol": "BABA", "value_metric": 0.5, "momentum_score": -0.2,
        })
        price_trends: dict = field(default_factory=lambda: {
            "BABA": _Trend("BABA"), "BABU": _Trend("BABU"),
        })
        fundamentals: dict = field(default_factory=lambda: {"marketCap": 1e9})
        news: list = field(default_factory=lambda: [{
            "title": "BABA headline", "publisher": "Reuters", "link": "",
        }])
        social: _Social = field(default_factory=_Social)
        leverage: _Leverage = field(default_factory=_Leverage)
        options: dict = field(default_factory=lambda: {"status": "skipped"})
        statuses: dict = field(default_factory=lambda: {"fetch": "offline"})

    import ui.services.single_stock_service as svc
    monkeypatch.setattr(svc, "analyze_single_stock", lambda *a, **k: _Result())
    monkeypatch.setattr(svc, "position_targets", lambda total, common, lev, cash: {
        "total_equity": total,
        "common_dollars": total * common / 100,
        "levered_dollars": total * lev / 100,
        "cash_dollars": total * cash / 100,
        "allocated_pct": common + lev + cash,
        "unallocated_pct": 100 - common - lev - cash,
        "warning": None,
    })

    def _app():
        from ui.components.single_stock_analyzer import render
        render()

    at = AppTest.from_function(_app).run()
    at.button[0].click().run()
    assert not at.exception
    rendered = "\n".join(
        [m.value for m in at.markdown]
        + [c.value for c in at.caption]
        + [w.value for w in at.warning]
    )
    assert "Current portfolio exposure" in rendered
    assert "Social evidence" in rendered
    assert "Leverage diagnostics" in rendered
    assert "Position structure helper" in rendered


# ---------------------------------------------------------------------------
# Guardrail — no broker / execution / order code anywhere in the feature
# ---------------------------------------------------------------------------

def test_feature_places_no_orders():
    import inspect

    import ui.components.single_stock_analyzer as comp
    import ui.services.single_stock_service as svc
    forbidden = ("from execution", "import execution", "buy_fractional(", "place_order(",
                 "submit_order(", "broker.sell(", "robin_stocks")
    for mod in (ssa, svc, comp):
        src = inspect.getsource(mod)
        for f in forbidden:
            assert f not in src, f"{mod.__name__} must not reference {f!r}"
