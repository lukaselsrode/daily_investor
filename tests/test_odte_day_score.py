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


# --- good_day_min_score knob wiring + component-presence telemetry (2026-08-05) ---------------

def test_good_day_boundary_uses_the_config_knob(monkeypatch):
    # The knob was defined in odte_config but the scorer used a literal 4 — a dead knob.
    market = {"spy_above_vwap": True, "spy_orb_state": "above",
              "qqq_above_vwap": True, "qqq_orb_state": "above",
              "iwm_above_vwap": True, "iwm_orb_state": "above",
              "gap_pct": 0.5, "minutes_to_close": 300}
    from data.odte_config import GOOD_DAY_MIN_SCORE
    assert ods.GOOD_DAY_MIN_SCORE == GOOD_DAY_MIN_SCORE       # wired, not re-hardcoded
    assert ods.score_day(market=market)["verdict"] == "GOOD_DAY"   # score 4 at default knob 4
    chop = dict(market)
    del chop["gap_pct"]                                       # the Aug-4 flip: 4 -> 3
    assert ods.score_day(market=chop)["verdict"] == "CHOP"
    monkeypatch.setattr(ods, "GOOD_DAY_MIN_SCORE", 3)
    assert ods.score_day(market=chop)["verdict"] == "GOOD_DAY"  # knob now actually turns


def test_component_presence_telemetry_shows_zero_headroom():
    # On the live tape vix + expected_move are structurally absent: max achievable == threshold.
    market = {"spy_above_vwap": True, "spy_orb_state": "above",
              "qqq_above_vwap": True, "qqq_orb_state": "above",
              "iwm_above_vwap": True, "iwm_orb_state": "above",
              "gap_pct": 0.5, "vixy_change_pct": -1.0, "minutes_to_close": 300}
    p = ods.score_day(market=market)
    assert p["components_supplied"] == 4                      # all but expected_move
    assert p["components_missing"] == ["expected_move"]
    assert p["max_possible_score"] == 4                       # 3 trend + 1 gap; vixy adds nothing
    assert p["max_possible_score"] == ods.GOOD_DAY_MIN_SCORE   # the zero-headroom fact, pinned
    with_vix = ods.score_day(market={**market, "vix": 18.0})
    assert with_vix["max_possible_score"] == 5                # a literal vix restores headroom
