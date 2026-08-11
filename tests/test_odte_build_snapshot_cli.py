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

sb_select = bs.select_contract
sb_build = bs.build_contract_snapshot


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


# ── contract mode: instruments x quotes, real 2026-08-11 shapes ─────────────────────────────
# Instruments live under data.instruments; quotes under data.results[].quote, and the SAME uuid
# is `id` in one and `instrument_id` in the other. Both inside the double-encoded envelope.

_OPT_ID = "acaa59c9-1fa1-404e-bb18-ea292a0cafa5"

REAL_INSTRUMENTS = {"result": json.dumps({"data": {"instruments": [
    {"id": _OPT_ID, "chain_symbol": "IWM", "strike_price": "301.0000", "type": "call",
     "expiration_date": "2026-08-11", "state": "active", "tradability": "tradable",
     "chain_id": "c0ffee", "underlying_type": "equity"},
]}})}

REAL_OPT_QUOTES = {"result": json.dumps({"data": {"results": [
    {"quote": {"instrument_id": _OPT_ID, "ask_price": "0.860000", "bid_price": "0.850000",
               "mark_price": "0.855000", "adjusted_mark_price": "0.860000",
               "break_even_price": "301.860000", "delta": "0.654895", "gamma": "0.268876",
               "implied_volatility": "0.175908", "open_interest": 1698, "volume": 14589,
               "bid_size": 6, "ask_size": 5},
     "close": {"price": "0.90"}},
]}})}


def test_instruments_and_quotes_join_on_the_same_uuid():
    insts = bs.extract_list(bs._unwrap(REAL_INSTRUMENTS))
    quotes = bs.extract_list(bs._unwrap(REAL_OPT_QUOTES), ("results", "quotes", "data"))
    picked = sb_select(insts, quotes)
    assert picked is not None
    inst, quote = picked
    assert inst["id"] == quote["instrument_id"] == _OPT_ID


def test_contract_always_carries_generated_at():
    """The 2026-08-10 `contract_quote_undated` refusal: convert cannot age an undated quote."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import data.odte_convert as oc
    insts = bs.extract_list(bs._unwrap(REAL_INSTRUMENTS))
    quotes = bs.extract_list(bs._unwrap(REAL_OPT_QUOTES), ("results", "quotes", "data"))
    inst, quote = sb_select(insts, quotes)
    c = sb_build(inst, quote)
    assert oc._payload_ts(c) is not None, "convert must be able to age this quote"
    assert c["option_id"] == _OPT_ID
    assert c["strike_price"] == pytest.approx(301.0)
    assert c["option_type"] == "call" and c["type"] == "call"
    assert c["bid_price"] == pytest.approx(0.85) and c["ask_price"] == pytest.approx(0.86)


@pytest.mark.parametrize("field,bad", [("state", "inactive"), ("tradability", "untradable")])
def test_untradable_instruments_are_refused(field, bad):
    insts = bs.extract_list(bs._unwrap(REAL_INSTRUMENTS))
    insts[0][field] = bad
    quotes = bs.extract_list(bs._unwrap(REAL_OPT_QUOTES), ("results", "quotes", "data"))
    assert sb_select(insts, quotes) is None, "a contract we cannot trade must never be emitted"


def test_instrument_without_a_quote_is_refused():
    insts = bs.extract_list(bs._unwrap(REAL_INSTRUMENTS))
    assert sb_select(insts, []) is None


def test_strike_and_type_filters_select():
    insts = bs.extract_list(bs._unwrap(REAL_INSTRUMENTS))
    quotes = bs.extract_list(bs._unwrap(REAL_OPT_QUOTES), ("results", "quotes", "data"))
    assert sb_select(insts, quotes, strike_price="301.0", option_type="call") is not None
    assert sb_select(insts, quotes, strike_price="999.0") is None
    assert sb_select(insts, quotes, option_type="put") is None


def test_cli_contract_mode_end_to_end(tmp_path):
    i = tmp_path / "i.json"
    i.write_text(json.dumps(REAL_INSTRUMENTS))
    q = tmp_path / "q.json"
    q.write_text(json.dumps(REAL_OPT_QUOTES))
    out = tmp_path / "contract.json"
    assert bs.main(["--instruments", str(i), "--option-quotes", str(q),
                    "--out-contract", str(out)]) == 0
    c = json.loads(out.read_text())
    assert c["option_id"] == _OPT_ID and "generated_at" in c


def test_cli_reports_the_cold_read_instead_of_emitting_nothing(tmp_path):
    """First get_option_instruments call returns [] and a retry ~10s later succeeds — 14/14 over
    2026-08-10..11 with IDENTICAL arguments. Exit 2 so the caller re-fetches rather than
    proceeding with a contract it cannot describe."""
    i = tmp_path / "i.json"
    i.write_text(json.dumps({"result": json.dumps({"data": {"instruments": []}})}))
    q = tmp_path / "q.json"
    q.write_text(json.dumps(REAL_OPT_QUOTES))
    assert bs.main(["--instruments", str(i), "--option-quotes", str(q)]) == 2


def test_contract_mode_requires_both_inputs(tmp_path):
    i = tmp_path / "i.json"
    i.write_text(json.dumps(REAL_INSTRUMENTS))
    assert bs.main(["--instruments", str(i)]) == 3
