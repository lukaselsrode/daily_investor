"""tests/test_odte_build_snapshot_cli.py — the MCP envelope, pinned to shapes seen in production.

The first version of scripts/odte_build_snapshot.py was written against a HAND-GUESSED fixture and
failed on its very first live invocation (2026-08-10 10:48:31, `no last prices — pass --quotes or
--last`, exit 2) because the real Robinhood MCP payloads differ in two ways nobody would guess:

  * the payload is DOUBLE-ENCODED — a JSON *string* under ``result``;
  * in quotes, the symbol lives INSIDE the ``quote`` sub-object, not on the list entry.

These fixtures are trimmed copies of the actual 10:45:58 payloads. A guessed fixture proves the
parser agrees with the guess; this one proves it agrees with the broker.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "odte_build_snapshot",
    Path(__file__).resolve().parents[1] / "scripts" / "odte_build_snapshot.py")
bs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bs)


def _bar(ts, o, c, h, low, v):
    return {"begins_at": ts, "open_price": o, "close_price": c,
            "high_price": h, "low_price": low, "volume": v, "session": "reg"}


REAL_HISTORICALS = {"result": json.dumps({"data": {"results": [
    {"symbol": "SPY", "interval": "5minute", "bounds": "regular", "bars": [
        _bar("2026-08-10T13:30:00Z", "772.640000", "772.620000", "773.630000", "772.580000", 301336),
        _bar("2026-08-10T13:35:00Z", "772.630000", "772.775000", "773.120000", "772.500000", 214619),
        _bar("2026-08-10T13:40:00Z", "772.815700", "772.690000", "773.110000", "772.540000", 260011),
        _bar("2026-08-10T13:45:00Z", "772.650000", "773.100000", "773.250000", "772.620000", 305562),
        _bar("2026-08-10T13:50:00Z", "773.130000", "773.100000", "773.400000", "772.500000", 301975),
        _bar("2026-08-10T13:55:00Z", "773.125000", "773.480100", "773.520000", "773.010000", 199741),
        _bar("2026-08-10T14:00:00Z", "773.420000", "773.700000", "773.850000", "773.400000", 528582),
    ]},
]}})}

REAL_QUOTES = {"result": json.dumps({"data": {"results": [
    {"quote": {"symbol": "SPY", "last_trade_price": "774.905000",
               "previous_close": "773.260000", "adjusted_previous_close": "773.260000",
               "bid_price": "774.900000", "ask_price": "774.920000", "state": "active"},
     "close": {"symbol": "SPY", "date": "2026-08-07", "price": "773.26"}},
]}})}


# ── the two shapes that broke it live ───────────────────────────────────────────────────────

def test_unwraps_the_double_encoded_result_envelope():
    inner = bs._unwrap(REAL_QUOTES)
    assert isinstance(inner, dict), "a JSON string under `result` must be decoded, not passed through"
    assert "data" in inner


def test_quotes_symbol_is_found_inside_the_quote_block():
    last, prev = bs.extract_quotes(bs._unwrap(REAL_QUOTES))
    assert last["SPY"] == pytest.approx(774.905), "symbol is nested under 'quote', not on the entry"
    assert prev["SPY"] == pytest.approx(773.26)


def test_historicals_bars_are_extracted_from_the_real_wrapper():
    bars = bs.extract_bars(bs._unwrap(REAL_HISTORICALS))
    assert set(bars) == {"SPY"}
    assert len(bars["SPY"]) == 7
    assert bars["SPY"][0]["high_price"] == "773.630000"


# ── end to end, against what the live lane actually decided ─────────────────────────────────

def test_end_to_end_reproduces_the_live_tape_read(tmp_path):
    """SPY on 2026-08-10: opening range 772.50–773.63, last 774.905 -> above both."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from data import odte_breadth as breadth
    from data import odte_snapshot_build as sb

    hist = tmp_path / "h.json"
    hist.write_text(json.dumps(REAL_HISTORICALS))
    quotes = tmp_path / "q.json"
    quotes.write_text(json.dumps(REAL_QUOTES))
    out = tmp_path / "market.json"
    rc = bs.main(["--historicals", str(hist), "--quotes", str(quotes),
                  "--gap-pct", "-0.0787", "--out", str(out)])
    assert rc == 0, "must not exit 2 on the real envelope (the 2026-08-10 live failure)"

    snap = json.loads(out.read_text())
    assert snap["SPY"]["orb_high"] == pytest.approx(773.63), "the level that became the live stop"
    assert snap["SPY"]["orb_state"] == "above"
    assert snap["spy_orb_state"] == snap["SPY"]["orb_state"]
    assert snap["SPY"]["above_vwap"] is True
    assert breadth.alignment(snap, "SPY", "bullish") == breadth.FULL_ALIGNMENT
    assert sb.audit_orb_vocabulary(snap) == []


def test_missing_prices_still_fails_loudly(tmp_path):
    """The original live failure mode must stay a clean exit 2, not a silent empty snapshot."""
    hist = tmp_path / "h.json"
    hist.write_text(json.dumps(REAL_HISTORICALS))
    empty = tmp_path / "q.json"
    empty.write_text(json.dumps({"result": json.dumps({"data": {"results": []}})}))
    assert bs.main(["--historicals", str(hist), "--quotes", str(empty)]) == 2


def test_plain_unwrapped_payloads_still_work(tmp_path):
    """Not every caller double-encodes — the unwrap must be tolerant, not mandatory."""
    plain = json.loads(REAL_HISTORICALS["result"])
    assert set(bs.extract_bars(bs._unwrap(plain))) == {"SPY"}
