"""
tests/test_position_watch.py — the daily intra-week watchtower.

Execution is weekly (Wednesday); the watchtower is what makes daily fetches
useful between runs: stop breaches, near-stops, profit levels, and regime
transitions must be surfaced loudly, because the human is the only intra-week
circuit breaker. Pure event-builder tests + a wiring smoke test.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reporting.position_watch import build_watch_events, run_position_watch

RULES = {"stop_loss_pct": -0.30, "take_profit_pct": 0.80}
HARVEST = 0.50


def _holding(pnl_pct: float) -> dict:
    return {"percent_change": str(pnl_pct * 100), "price": "100", "equity": "1000"}


def _events(holdings, today="bullish", prev="bullish"):
    return build_watch_events(holdings, today, prev, sell_rules=RULES,
                              harvest_threshold=HARVEST)


class TestEventBuilder:

    def test_calm_book_no_events(self):
        assert _events({"AAA": _holding(0.05), "BBB": _holding(-0.10)}) == []

    def test_stop_breach_flagged(self):
        ev = _events({"AAA": _holding(-0.35)})
        assert [e["type"] for e in ev] == ["stop_breach"]
        assert "cannot act until Wednesday" in ev[0]["detail"]

    def test_near_stop_flagged_inside_band(self):
        ev = _events({"AAA": _holding(-0.27)})   # within 5pp of -30%
        assert [e["type"] for e in ev] == ["stop_near"]
        assert _events({"AAA": _holding(-0.20)}) == []   # outside the band

    def test_profit_levels(self):
        ev = _events({"WIN": _holding(0.55), "BIGWIN": _holding(0.85)})
        types = {e["symbol"]: e["type"] for e in ev}
        assert types == {"WIN": "harvest_reached", "BIGWIN": "take_profit_reached"}

    def test_regime_transition_flagged(self):
        ev = _events({}, today="defensive", prev="bullish")
        assert [e["type"] for e in ev] == ["regime_change"]
        # Unknown previous state (first run) must not fabricate a transition.
        assert _events({}, today="defensive", prev=None) == []

    def test_malformed_percent_change_skipped(self):
        assert _events({"BAD": {"percent_change": "n/a"}}) == []


class TestWiring:

    def test_run_writes_ledger_and_state(self, monkeypatch, tmp_path):
        import data.cache as dc
        import reporting.position_watch as pw
        monkeypatch.setattr(dc, "DATA_DIRECTORY", str(tmp_path))
        monkeypatch.setattr(pw, "_NEAR_BAND", 0.05)

        events = run_position_watch({"AAA": _holding(-0.50)})
        assert any(e["type"] == "stop_breach" for e in events)
        ledger = tmp_path / "position_watch.csv"
        state = tmp_path / "position_watch_state.json"
        assert ledger.exists() and state.exists()
        assert "stop_breach" in ledger.read_text()

    def test_run_never_raises_on_garbage(self):
        assert run_position_watch(None) == [] or isinstance(run_position_watch(None), list)
