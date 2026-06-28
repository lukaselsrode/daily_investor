"""Tests for the offline 0DTE day-regime scorecard (GOOD_DAY / CHOP / AVOID).

Pure unit tests — no network/broker/LLM. All inputs are caller-supplied JSON.
"""
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_day_score as ods
from data.odte_day_score import AVOID, CHOP, GOOD_DAY, run_day_score, score_day


def test_good_day_when_indices_trend_with_tradable_vol_and_range():
    payload = score_day(market={
        "spy_above_vwap": True, "qqq_above_vwap": True, "iwm_above_vwap": True,
        "spy_orb_state": "above", "qqq_orb_state": "above", "iwm_orb_state": "above",
        "vix": 16, "gap_pct": 0.8, "expected_move_pct": 1.0, "minutes_to_close": 320,
    })
    assert payload["verdict"] == GOOD_DAY
    assert payload["components"]["trend"] == 3
    assert payload["components"]["volatility"] == 1
    assert payload["places_orders"] is False


def test_chop_when_indices_stuck_inside_opening_range():
    payload = score_day(market={
        "spy_orb_state": "inside", "qqq_orb_state": "inside", "iwm_orb_state": "inside",
        "vix": 13, "gap_pct": 0.1, "minutes_to_close": 250,
    })
    assert payload["verdict"] == CHOP
    assert any("inside the opening range" in r for r in payload["reasons"])


def test_avoid_when_too_late_in_session_even_if_trend_is_clean():
    payload = score_day(market={
        "spy_above_vwap": True, "qqq_above_vwap": True, "iwm_above_vwap": True,
        "spy_orb_state": "above", "qqq_orb_state": "above", "iwm_orb_state": "above",
        "vix": 16, "expected_move_pct": 1.0, "minutes_to_close": 20,
    })
    assert payload["verdict"] == AVOID
    assert any("too late to open" in r for r in payload["reasons"])


def test_avoid_when_volatility_is_spiking():
    payload = score_day(market={
        "spy_above_vwap": True, "qqq_above_vwap": True, "iwm_above_vwap": True,
        "spy_orb_state": "above", "qqq_orb_state": "above", "iwm_orb_state": "above",
        "vix": 36, "vix_change_pct": 18, "minutes_to_close": 300,
    })
    assert payload["verdict"] == AVOID
    assert any("very elevated" in r or "spiking" in r for r in payload["reasons"])


def test_split_book_with_tight_move_is_negative_chop():
    # A genuine 1-up / 1-down split (third index inside) — the indices disagree on direction.
    payload = score_day(market={
        "spy_above_vwap": True, "qqq_above_vwap": False, "iwm_orb_state": "inside",
        "spy_orb_state": "above", "qqq_orb_state": "below",
        "vix": 14, "expected_move_pct": 0.3, "minutes_to_close": 240,
    })
    assert payload["verdict"] in {CHOP, AVOID}
    assert any("split above/below VWAP" in r for r in payload["reasons"])
    assert any("tight" in r for r in payload["reasons"])


def test_expected_move_derived_from_gamma_band(tmp_path):
    market = {"spy_above_vwap": True, "qqq_above_vwap": True, "iwm_above_vwap": True,
              "spy_orb_state": "above", "qqq_orb_state": "above", "iwm_orb_state": "above",
              "vix": 16, "minutes_to_close": 300}
    # No expected_move_pct in market; derived from band half-width / spot = (722-714)/2/718 ≈ 0.56%.
    gamma = {"spot": 718, "expected_move": {"lower": 714, "upper": 722}}
    mpath, gpath = tmp_path / "m.json", tmp_path / "g.json"
    mpath.write_text(json.dumps(market))
    gpath.write_text(json.dumps(gamma))
    payload = run_day_score(market_path=str(mpath), gamma_path=str(gpath))
    assert payload["verdict"] == GOOD_DAY
    assert any("from gamma band" in r for r in payload["reasons"])
    assert payload["components"]["expected_move"] == 1


def test_empty_snapshot_defaults_to_chop():
    payload = score_day()
    assert payload["verdict"] == CHOP
    assert payload["score"] == 0
    assert payload["places_orders"] is False


def test_run_day_score_writes_artifact(tmp_path):
    payload = run_day_score(market_json=json.dumps({"vix": 16, "minutes_to_close": 300}),
                            out_dir=str(tmp_path), write=True)
    out = tmp_path / "odte_day_score.json"
    assert out.exists()
    assert payload["artifact"] == str(out)
    assert json.loads(out.read_text())["verdict"] == payload["verdict"]


def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(ods)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket", "yfinance"):
        assert forbidden not in src, f"odte_day_score must not reference {forbidden!r}"
