"""Tests for the offline 0DTE vehicle/contract scorecard."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.odte_vehicle_score import BAD_BET, GOOD_BET, WATCH, run_vehicle_score, score_vehicle


def test_good_call_when_tape_gamma_and_liquidity_align():
    payload = score_vehicle(
        {"underlying": "QQQ", "option_type": "call", "strike": 718, "bid": 0.70, "ask": 0.72,
         "volume": 12000, "open_interest": 5000},
        direction="bullish",
        buying_power=108,
        market={"spy_above_vwap": True, "qqq_above_vwap": True, "iwm_above_vwap": True,
                "vixy_above_vwap": False},
        gamma={"spot": 717.5, "pin_risk": {"level": "low"}, "call_wall": 719,
               "expected_move": {"lower": 714, "upper": 720},
               "freshness": {"quote_fresh": True}},
    )
    assert payload["verdict"] == GOOD_BET
    assert payload["components"]["market"] > 0
    assert payload["components"]["gamma"] > 0


def test_bad_call_when_strike_is_beyond_expected_move_and_wall_with_conflicting_tape():
    payload = score_vehicle(
        {"underlying": "IWM", "option_type": "call", "strike": 302, "bid": 0.26, "ask": 0.38,
         "volume": 500, "open_interest": 400},
        direction="bullish",
        buying_power=108,
        market={"spy_above_vwap": False, "qqq_above_vwap": False, "iwm_above_vwap": False,
                "vixy_above_vwap": True},
        gamma={"spot": 298.2, "pin_risk": {"level": "high"}, "call_wall": 299,
               "expected_move": {"lower": 296.4, "upper": 300.0},
               "freshness": {"quote_fresh": True}},
    )
    assert payload["verdict"] == BAD_BET
    assert any("above expected-move" in r for r in payload["reasons"])
    assert any("beyond call wall" in r for r in payload["reasons"])
    assert any("VIXY/risk vol conflicts" in r for r in payload["reasons"])


def test_late_day_theta_gate_can_downgrade_otherwise_good_setup():
    payload = score_vehicle(
        {"underlying": "SPY", "option_type": "call", "strike": 734, "bid": 0.50, "ask": 0.51,
         "volume": 20000, "open_interest": 10000},
        direction="bullish",
        buying_power=108,
        market={"spy_above_vwap": True, "qqq_above_vwap": True, "iwm_above_vwap": True,
                "vixy_change_pct": -1.5, "minutes_to_close": 15},
        gamma={"spot": 733.5, "pin_risk": {"level": "low"}, "call_wall": 735,
               "expected_move": {"lower": 731, "upper": 736},
               "freshness": {"quote_fresh": True}},
    )
    assert payload["verdict"] == WATCH
    assert any("minutes to close" in r for r in payload["reasons"])


def test_watch_when_market_and_gamma_are_missing_but_contract_is_liquid(tmp_path):
    contract = {"underlying": "SPY", "option_type": "put", "strike": 730, "bid": 0.50, "ask": 0.51,
                "volume": 20000, "open_interest": 10000}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    payload = run_vehicle_score(contract_path=str(path), direction="bearish", buying_power="108")
    assert payload["verdict"] in {WATCH, GOOD_BET}
    assert payload["places_orders"] is False


# --- bp_fit: the gate's budget_check dollars at selection time (2026-08-05) -------------------

def test_bp_fit_names_the_aug4_mismatch():
    # Aug-4 replay: ask 2.88, BP 357.06 -> scored GOOD_BET against raw BP while the gate
    # refused 12x against the 60% tier cap. bp_fit must expose the gate's arithmetic verbatim
    # WITHOUT changing the score or verdict.
    from data.odte_config import MAX_DEBIT_FRACTION
    contract = {"underlying": "SPY", "option_type": "call", "bid": 2.85, "ask": 2.88,
                "volume": 184746, "open_interest": 6734}
    p = score_vehicle(contract, direction="bullish", buying_power=357.06)
    fit = p["bp_fit"]
    assert fit["fits"] is False and fit["debit"] == 288.0
    assert fit["cap"] == round(MAX_DEBIT_FRACTION * 357.06, 2)
    assert fit["max_affordable_ask"] == round(MAX_DEBIT_FRACTION * 357.06 / 100.0, 2)
    assert any("CANNOT pass the gate's budget_check" in r for r in p["reasons"])
    # Verdict/score are UNCHANGED by bp_fit (additive telemetry only).
    baseline = score_vehicle(contract, direction="bullish", buying_power=357.06)
    assert p["verdict"] == baseline["verdict"] and p["score"] == baseline["score"]


def test_bp_fit_tier_selects_fraction_and_fitting_contract_passes():
    from data.odte_config import B_PLUS_DEBIT_FRACTION
    contract = {"underlying": "IWM", "option_type": "call", "bid": 0.78, "ask": 0.80,
                "volume": 50000, "open_interest": 4000}
    p = score_vehicle(contract, direction="bullish", buying_power=357.06, tier="b_plus")
    fit = p["bp_fit"]
    assert fit["tier"] == "b_plus" and fit["fraction"] == B_PLUS_DEBIT_FRACTION
    assert fit["fits"] is (80.0 <= round(B_PLUS_DEBIT_FRACTION * 357.06, 2))
    assert any("fits the gate's budget_check" in r for r in p["reasons"])


def test_bp_fit_absent_without_buying_power():
    p = score_vehicle({"underlying": "SPY", "ask": 1.0}, direction="bullish")
    assert p["bp_fit"] is None
