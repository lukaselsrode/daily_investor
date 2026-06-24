"""tests/test_odte_gamma_map.py — 0DTE gamma/pin map (pure/offline, no broker/network/LLM).

Builds normalized option-quote rows and exercises build_gamma_map directly: per-strike call/put
aggregation, call/put wall + max-gamma strike, ATM-straddle expected move, pin-risk levels, quote
freshness, OI fallback when gamma is absent, the honest (non-dealer-GEX) labeling, and a source
guardrail that the module makes no network/broker calls.
"""
import inspect
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_gamma_map as gm

NOW = datetime(2026, 6, 24, 18, 0, 0, tzinfo=timezone.utc)
FRESH = NOW.isoformat()


def _row(side, strike, gamma, oi, mark=None, vol=0, updated=FRESH):
    return {"type": side, "strike_price": strike, "gamma": gamma, "open_interest": oi,
            "mark_price": mark, "volume": vol, "underlying": "SPY",
            "expiration_date": "2026-06-24", "updated_at": updated}


def _scenario_a():
    # spot=600. gamma_notional ∝ gamma*OI (spot^2*0.01 constant). Designed so:
    #   max-gamma strike = 600 (total 60), call wall = 605 (call 40), put wall = 595 (put 45).
    return [
        _row("call", 595, 0.01, 100, mark=6.0), _row("put", 595, 0.05, 900, mark=0.4),
        _row("call", 600, 0.06, 500, mark=2.0, vol=300), _row("put", 600, 0.06, 500, mark=1.8, vol=250),
        _row("call", 605, 0.04, 1000, mark=0.5), _row("put", 605, 0.01, 100, mark=5.5),
    ]


def test_aggregates_calls_and_puts_by_strike():
    g = gm.build_gamma_map(_scenario_a(), spot=600, now=NOW)
    assert g["n_strikes"] == 3
    by = {r["strike"]: r for r in g["by_strike"]}
    assert by[600.0]["call_oi"] == 500 and by[600.0]["put_oi"] == 500
    assert by[600.0]["call_volume"] == 300 and by[600.0]["put_volume"] == 250
    # gamma_notional_1pct = gamma*OI*100*spot^2*0.01 ; spot^2*0.01 = 3600
    assert round(by[600.0]["call_gamma_notional_1pct"], 2) == round(0.06 * 500 * 100 * 3600, 2)
    assert round(by[595.0]["put_gamma_notional_1pct"], 2) == round(0.05 * 900 * 100 * 3600, 2)


def test_identifies_walls_and_max_gamma_strike():
    g = gm.build_gamma_map(_scenario_a(), spot=600, now=NOW)
    assert g["call_wall"] == 605.0
    assert g["put_wall"] == 595.0
    assert g["max_gamma_strike"] == 600.0
    assert g["gamma_available"] is True and g["concentration_basis"] == "gamma_notional_1pct"


def test_expected_move_from_atm_straddle():
    g = gm.build_gamma_map(_scenario_a(), spot=600, now=NOW)
    em = g["expected_move"]
    assert em["available"] is True and em["atm_strike"] == 600.0
    assert em["straddle_mark"] == 3.8                 # 2.0 + 1.8
    assert em["lower"] == 596.2 and em["upper"] == 603.8
    assert em["spot_location"] == "at_atm"


def test_expected_move_unavailable_without_both_marks():
    rows = [_row("call", 600, 0.05, 100, mark=2.0)]   # no put mark at any strike
    g = gm.build_gamma_map(rows, spot=600, now=NOW)
    assert g["expected_move"]["available"] is False


def test_pin_risk_high_when_spot_at_concentrated_peak():
    rows = [
        _row("call", 600, 0.10, 1000, mark=2.0), _row("put", 600, 0.10, 1000, mark=2.0),
        _row("call", 595, 0.01, 50), _row("put", 595, 0.01, 50),
        _row("call", 605, 0.01, 50), _row("put", 605, 0.01, 50),
    ]
    g = gm.build_gamma_map(rows, spot=600, now=NOW)
    assert g["max_gamma_strike"] == 600.0
    pr = g["pin_risk"]
    assert pr["level"] == "high" and pr["within_pin_zone"] is True
    assert pr["peak_vs_median_ratio"] >= 1.5


def test_pin_risk_low_when_spot_far_and_flat():
    rows = [
        _row("call", 590, 0.02, 100), _row("put", 590, 0.02, 100),
        _row("call", 600, 0.02, 100), _row("put", 600, 0.02, 100),
        _row("call", 610, 0.02, 100), _row("put", 610, 0.02, 100),
    ]
    g = gm.build_gamma_map(rows, spot=575, now=NOW)   # spot far from any peak; flat concentration
    assert g["pin_risk"]["level"] == "low"


def test_stale_quotes_flag_and_pin_stale():
    stale = (NOW - timedelta(minutes=60)).isoformat()
    rows = [_row("call", 600, 0.06, 500, mark=2.0, updated=stale),
            _row("put", 600, 0.06, 500, mark=1.8, updated=stale)]
    g = gm.build_gamma_map(rows, spot=600, now=NOW)
    assert g["freshness"]["quote_fresh"] is False
    assert g["freshness"]["age_minutes"] == 60.0
    assert g["pin_risk"]["level"] == "stale"   # never assert a pin on stale quotes


def test_missing_timestamps_are_not_fresh():
    rows = [_row("call", 600, 0.06, 500, mark=2.0, updated=None),
            _row("put", 600, 0.06, 500, mark=1.8, updated=None)]
    g = gm.build_gamma_map(rows, spot=600, now=NOW)
    assert g["freshness"]["quote_fresh"] is False
    assert "no quote timestamps" in g["freshness"]["reason"]


def test_oi_fallback_when_gamma_missing():
    rows = [
        _row("call", 600, None, 800), _row("put", 600, None, 200),
        _row("call", 605, None, 100), _row("put", 605, None, 100),
    ]
    g = gm.build_gamma_map(rows, spot=600, now=NOW)
    assert g["gamma_available"] is False
    assert g["concentration_basis"] == "open_interest"
    assert g["max_gamma_strike"] == 600.0   # 1000 total OI vs 200
    assert g["call_wall"] == 600.0          # 800 call OI vs 100


def test_no_fake_dealer_gex_label():
    g = gm.build_gamma_map(_scenario_a(), spot=600, now=NOW)
    assert g["gamma_regime"] == "pin_risk_only_not_dealer_gex"
    assert "not dealer" in g["disclaimer"].lower()
    # No field anywhere claims a real dealer-GEX / gamma-flip / sign number.
    def _keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from _keys(v)
        elif isinstance(o, list):
            for x in o:
                yield from _keys(x)
    bad = {k.lower() for k in _keys(g)} & {"gex", "dealer_gex", "net_gex", "gamma_flip", "flip_point"}
    assert not bad, f"unexpected dealer-GEX-style fields: {bad}"


def test_module_makes_no_network_or_broker_calls():
    src = inspect.getsource(gm)
    for forbidden in ("requests", "urllib", "httpx", "socket", "robin_stocks",
                      "openai", "anthropic", "yfinance"):
        assert forbidden not in src, f"odte_gamma_map must not reference {forbidden!r}"


# --- realistic RH nested / quote-only shapes ------------------------------------------------

# (side, strike, gamma, oi, mark, volume) — same as _scenario_a: call wall 605, put wall 595,
# max-gamma strike 600.
_SCEN = [
    ("call", 595, 0.01, 100, 6.0, 0), ("put", 595, 0.05, 900, 0.4, 0),
    ("call", 600, 0.06, 500, 2.0, 300), ("put", 600, 0.06, 500, 1.8, 250),
    ("call", 605, 0.04, 1000, 0.5, 0), ("put", 605, 0.01, 100, 5.5, 0),
]


def _nested_rows():
    # {"instrument": {...}, "quote": {...}} — strike/type/expiration in instrument; greeks/OI in quote.
    return [{"instrument": {"chain_symbol": "SPY", "expiration_date": "2026-06-24",
                            "strike_price": f"{k:.4f}", "type": side},
             "quote": {"mark_price": f"{mark}", "gamma": f"{g}", "open_interest": oi,
                       "volume": vol, "updated_at": FRESH}}
            for side, k, g, oi, mark, vol in _SCEN]


def _quote_only_rows_and_instruments(as_list=False, url_ids=False):
    rows, idx = [], {}
    for i, (side, k, g, oi, mark, vol) in enumerate(_SCEN):
        iid = f"id{i}"
        idx[iid] = {"id": iid, "chain_symbol": "SPY", "expiration_date": "2026-06-24",
                    "strike_price": f"{k:.4f}", "type": side}
        ref = f"https://api.robinhood.com/options/instruments/{iid}/" if url_ids else iid
        rows.append({"quote": {"instrument_id" if not url_ids else "instrument": ref,
                               "mark_price": f"{mark}", "gamma": f"{g}", "open_interest": oi,
                               "volume": vol, "updated_at": FRESH}})
    instruments = list(idx.values()) if as_list else idx
    return rows, instruments


def _assert_scenario_a(g):
    assert g["call_wall"] == 605.0 and g["put_wall"] == 595.0 and g["max_gamma_strike"] == 600.0
    assert g["gamma_available"] is True
    by = {r["strike"]: r for r in g["by_strike"]}
    assert by[600.0]["call_oi"] == 500 and by[600.0]["put_oi"] == 500
    assert by[600.0]["call_volume"] == 300


def test_nested_instrument_quote_rows_compute_walls():
    g = gm.build_gamma_map(_nested_rows(), spot=600, now=NOW)
    _assert_scenario_a(g)
    assert g["freshness"]["quote_fresh"] is True
    assert g["underlying"] == "SPY" and g["expiration"] == "2026-06-24"


def test_quote_only_rows_joined_via_instruments_map():
    rows, instruments = _quote_only_rows_and_instruments()
    g = gm.build_gamma_map(rows, spot=600, instruments=instruments, now=NOW)
    _assert_scenario_a(g)


def test_quote_only_rows_joined_via_instruments_list():
    rows, instruments = _quote_only_rows_and_instruments(as_list=True)
    g = gm.build_gamma_map(rows, spot=600, instruments=instruments, now=NOW)
    _assert_scenario_a(g)


def test_quote_only_rows_joined_via_instrument_url():
    rows, instruments = _quote_only_rows_and_instruments(url_ids=True)
    g = gm.build_gamma_map(rows, spot=600, instruments=instruments, now=NOW)
    _assert_scenario_a(g)


def test_top_level_scalar_overrides_nested():
    # An explicit top-level strike/type must win over the nested instrument block.
    row = {"instrument": {"chain_symbol": "SPY", "strike_price": "595", "type": "put"},
           "quote": {"gamma": "0.06", "open_interest": 500, "mark_price": "2.0", "updated_at": FRESH},
           "strike_price": "600", "type": "call"}
    n = gm.normalize_row(row)
    assert n["strike"] == 600.0 and n["side"] == "call"


def test_run_gamma_map_wrapper_with_instruments_and_quote_only(tmp_path):
    rows, instruments = _quote_only_rows_and_instruments()
    wrapper = {"underlying": "SPY", "expiration": "2026-06-24", "spot": 600,
               "instruments": instruments, "rows": rows}
    inp = tmp_path / "merged_chain.json"
    inp.write_text(json.dumps(wrapper))
    g = gm.run_gamma_map(input_path=str(inp), now=NOW)
    _assert_scenario_a(g)
    assert g["underlying"] == "SPY"


def test_run_gamma_map_from_wrapper_and_writes_artifacts(tmp_path):
    wrapper = {"underlying": "SPY", "expiration": "2026-06-24", "spot": 600, "rows": _scenario_a()}
    inp = tmp_path / "chain.json"
    inp.write_text(json.dumps(wrapper))
    out = tmp_path / "reports"
    g = gm.run_gamma_map(input_path=str(inp), out_dir=str(out), write=True, now=NOW)
    assert g["underlying"] == "SPY" and g["max_gamma_strike"] == 600.0
    assert (out / "odte_gamma_map_spy.md").exists()
    assert (out / "odte_gamma_map_spy.json").exists()
    assert "pin_risk_only_not_dealer_gex" in (out / "odte_gamma_map_spy.md").read_text()
