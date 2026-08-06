"""tests/test_odte_options_trend.py — the pre-open phantom guard on fetch_spy_trend.

2026-08-05: before the bell, yfinance's latest session is YESTERDAY — fetch_spy_trend served
Aug-4's +1.80% as live tape for 29 watchdog fires, minting a candidate whose watch TTL burned
out before the market opened. The latest session must BE the current ET date or the fetch
fails closed. yfinance is faked via sys.modules — no network.
"""
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

_ET = ZoneInfo("America/New_York")


def _bars(day):
    idx = pd.date_range(f"{day} 09:30", periods=6, freq="5min", tz="America/New_York")
    px = [100 + i for i in range(6)]
    return pd.DataFrame({"Open": px, "High": [p + 1 for p in px], "Low": [p - 1 for p in px],
                         "Close": [p + 0.5 for p in px], "Volume": [1000] * 6}, index=idx)


def _fake_yf(frame):
    class _Ticker:
        def __init__(self, _symbol):
            pass

        def history(self, period=None, interval=None):
            return frame
    return SimpleNamespace(Ticker=_Ticker)


def test_prior_session_bars_fail_closed(monkeypatch):
    import data.odte_options as oo
    yesterday = (datetime.now(_ET) - timedelta(days=1)).date().isoformat()
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(_bars(yesterday)))
    out = oo.fetch_spy_trend()
    assert out["ok"] is False
    assert "no cash-session bars yet" in out["status"]
    assert "pct_vs_prev_close" not in out                      # no phantom numbers escape


def test_same_day_bars_unchanged(monkeypatch):
    import data.odte_options as oo
    today = datetime.now(_ET).date().isoformat()
    two_day = pd.concat([_bars((datetime.now(_ET) - timedelta(days=1)).date().isoformat()),
                         _bars(today)])
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(two_day))
    out = oo.fetch_spy_trend()
    assert out["ok"] is True
    assert out["prev_close"] is not None and out["pct_vs_prev_close"] is not None
