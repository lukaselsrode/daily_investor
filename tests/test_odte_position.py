"""tests/test_odte_position.py — 0DTE live-position decision watchdog (decision-only).

No Robinhood, no network, no LLM. evaluate_position() is pure: a trade plan + a caller-supplied
snapshot in, structured triggers out. Covers single-contract scalp take-profit (+35-50%), the +60%
strong exit, thesis-death levels, bid-floor, time-risk, monitoring-degraded, the no-position quiet
path, the NVDA employer-restriction refusal, and the run_position_watchdog file writer.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_position as op


def _scalp_plan(**over):
    plan = {"status": "open", "mode": "scalp", "underlying": "SPY", "option_type": "call",
            "option_id": "SPY260623C00505000", "entry_price": 1.00, "quantity": 1,
            "thesis": {"underlying_stop": 500.0}, "time_rules": {"flat_before": "15:40"}}
    plan.update(over)
    return plan


def _types(result):
    return {t["type"] for t in result["triggers"]}


# --- take-profit -----------------------------------------------------------------------------

def test_scalp_take_profit_band_35_to_50():
    # Single-contract scalp up +40% -> TAKE_PROFIT scale, action exit (a single contract can't scale).
    # time_rules={} isolates the profit axis (no wall-clock TIME_RISK interference).
    for mark in (1.35, 1.42, 1.50):
        r = op.evaluate_position(_scalp_plan(time_rules={}), {"option_mark": mark})
        assert r["decision"] == "TAKE_PROFIT"
        tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
        assert tp["stage"] == "scale" and tp["action"] == "exit"


def test_take_profit_below_band_holds():
    r = op.evaluate_position(_scalp_plan(time_rules={}), {"option_mark": 1.20})   # +20% < 35%
    assert r["decision"] == "HOLD"
    assert "TAKE_PROFIT" not in _types(r)


def test_strong_exit_at_60pct():
    r = op.evaluate_position(_scalp_plan(time_rules={}), {"option_mark": 1.62})   # +62%
    assert r["decision"] == "TAKE_PROFIT"
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["stage"] == "strong" and tp["action"] == "exit_all"


def test_multi_contract_scalp_scales_keeps_runner():
    # Multi-contract scalp at +40% -> sell partial and KEEP a runner (not a full exit).
    plan = _scalp_plan(quantity=3, thesis={}, time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.40})
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["stage"] == "scale" and tp["action"] == "scale_keep_runner"


def test_pnl_pct_supplied_directly_wins():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}), {"pnl_pct": 0.61})
    assert r["decision"] == "TAKE_PROFIT" and r["pnl_pct"] == 0.61


# --- mode-specific profit semantics (single contract, +40%) ----------------------------------

def _mode_plan(mode, **over):
    plan = _scalp_plan(mode=mode, quantity=1, thesis={}, time_rules={})
    plan.update(over)
    return plan


def _tp_action(res):
    return next(t for t in res["triggers"] if t["type"] == "TAKE_PROFIT")["action"]


def test_single_contract_trend_protects_not_exits():
    # trend +40% single-contract -> alert to protect profit, NOT a forced exit.
    r = op.evaluate_position(_mode_plan("trend"), {"option_mark": 1.40})
    assert r["decision"] == "TAKE_PROFIT"
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["stage"] == "scale" and tp["action"] == "protect_profit"
    assert tp["action"] not in ("exit", "exit_all")


def test_single_contract_lotto_not_forced_exit():
    r = op.evaluate_position(_mode_plan("lotto"), {"option_mark": 1.40})
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["action"] == "hold_but_alert"
    assert tp["action"] not in ("exit", "exit_all")


def test_single_contract_runner_trails_not_exits():
    r = op.evaluate_position(_mode_plan("runner"), {"option_mark": 1.40})
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["action"] == "trail_runner"
    assert tp["action"] not in ("exit", "exit_all")


def test_strong_60_mode_semantics_no_accidental_exit_all():
    # +62% must NOT force exit_all for trend/runner; scalp still exits all.
    trend = op.evaluate_position(_mode_plan("trend"), {"option_mark": 1.62})
    runner = op.evaluate_position(_mode_plan("runner"), {"option_mark": 1.62})
    scalp = op.evaluate_position(_mode_plan("scalp"), {"option_mark": 1.62})
    assert _tp_action(trend) == "trail_or_exit_on_stall" and _tp_action(trend) != "exit_all"
    assert _tp_action(runner) == "trail_runner" and _tp_action(runner) != "exit_all"
    assert _tp_action(scalp) == "exit_all"


def test_profit_rules_override_can_force_trend_exit():
    # An explicit override can force a forced exit on a trend trade if the trader wants it.
    plan = _mode_plan("trend", profit_rules={"take_profit_action": "exit_all"})
    r = op.evaluate_position(plan, {"option_mark": 1.40})   # +40% scale stage
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["action"] == "exit_all"
    # ...and a strong-stage override too.
    plan2 = _mode_plan("runner", profit_rules={"strong_exit_action": "exit_all"})
    r2 = op.evaluate_position(plan2, {"option_mark": 1.62})
    assert next(t for t in r2["triggers"] if t["type"] == "TAKE_PROFIT")["action"] == "exit_all"


def test_profit_rules_threshold_override():
    # Override the take-profit threshold; +25% should now trigger for a trend single-contract.
    plan = _mode_plan("trend", profit_rules={"take_profit_pct": 0.20})
    r = op.evaluate_position(plan, {"option_mark": 1.25})
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["stage"] == "scale" and tp["action"] == "protect_profit"


def test_thesis_death_outranks_profit_when_both_fire():
    # trend +40% (protect_profit) AND a dead thesis -> primary decision is THESIS_DEAD.
    plan = _mode_plan("trend", thesis={"underlying_stop": 500.0})
    r = op.evaluate_position(plan, {"option_mark": 1.40, "underlying_last": 498.0})
    assert r["decision"] == "THESIS_DEAD"
    assert {"TAKE_PROFIT", "THESIS_DEAD"} <= _types(r)   # both fired


# --- thesis death ----------------------------------------------------------------------------

def test_thesis_dead_call_support_lost():
    # Call: underlying below the support stop kills the thesis even at a small loss.
    r = op.evaluate_position(_scalp_plan(time_rules={}),
                             {"option_mark": 0.90, "underlying_last": 498.0})
    assert r["decision"] == "THESIS_DEAD"
    td = next(t for t in r["triggers"] if t["type"] == "THESIS_DEAD")
    assert "underlying" in td["detail"] and td["action"] == "exit"


def test_thesis_alive_call_above_support():
    r = op.evaluate_position(_scalp_plan(time_rules={}),
                             {"option_mark": 1.05, "underlying_last": 506.0})
    assert "THESIS_DEAD" not in _types(r)


def test_thesis_dead_put_resistance_reclaimed_and_vix_fade():
    plan = _scalp_plan(option_type="put", thesis={"underlying_stop": 500.0, "vix_stop": 18.0},
                       time_rules={})
    # Put: underlying back ABOVE stop (resistance reclaimed) and VIX faded BELOW its floor.
    r = op.evaluate_position(plan, {"option_mark": 0.80, "underlying_last": 503.0, "vix": 16.0})
    assert r["decision"] == "THESIS_DEAD"
    reasons = next(t for t in r["triggers"] if t["type"] == "THESIS_DEAD")["reasons"]
    assert any("resistance reclaimed" in x for x in reasons)
    assert any("vol faded" in x for x in reasons)


# --- bid floor / time risk -------------------------------------------------------------------

def test_bid_floor_near_worthless():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}),
                             {"option_mark": 0.04, "option_bid": 0.03})
    assert r["decision"] == "BID_FLOOR"


def test_time_risk_flat_before():
    r = op.evaluate_position(_scalp_plan(thesis={}),
                             {"option_mark": 1.00, "now_et": "2026-06-23T15:45:00-04:00"})
    assert r["decision"] == "TIME_RISK"
    tr = next(t for t in r["triggers"] if t["type"] == "TIME_RISK")
    assert tr["stage"] == "flat"


def test_time_risk_tighten_after():
    plan = _scalp_plan(thesis={}, time_rules={"tighten_after": "15:00", "flat_before": "15:40"})
    r = op.evaluate_position(plan, {"option_mark": 1.00, "now_et": "2026-06-23T15:10:00-04:00"})
    tr = next(t for t in r["triggers"] if t["type"] == "TIME_RISK")
    assert tr["stage"] == "tighten"


# --- monitoring degraded / no position -------------------------------------------------------

def test_monitoring_degraded_when_cannot_value():
    # Active position but the snapshot can't value it (no mark/bid/pnl) -> degraded, not a guess.
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}), {})
    assert r["decision"] == "MONITORING_DEGRADED"


def test_monitoring_degraded_explicit_flag():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}),
                             {"option_mark": 1.10, "monitoring_ok": False})
    assert "MONITORING_DEGRADED" in _types(r)


def test_no_position_quiet_paths():
    for plan in ({}, {"status": "closed", "underlying": "SPY"}, {"underlying": "SPY", "active": False}):
        r = op.evaluate_position(plan, {"option_mark": 5.0})
        assert r["decision"] == "NO_POSITION" and r["triggers"] == []


# --- NVDA employer restriction ---------------------------------------------------------------

def test_nvda_position_is_restricted_no_management():
    # Even handed a juicy +80% NVDA position, the watchdog refuses to manage it.
    plan = _scalp_plan(underlying="NVDA", thesis={}, time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.80})
    assert r["decision"] == "RESTRICTED"
    assert _types(r) == {"RESTRICTED"}
    assert next(t for t in r["triggers"] if t["type"] == "RESTRICTED")["reason"] == "employer"


# --- run_position_watchdog file writer -------------------------------------------------------

def test_run_position_watchdog_writes_files_and_alerts(tmp_path):
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text(json.dumps(_scalp_plan(thesis={}, time_rules={})))
    payload = op.run_position_watchdog(plan_path=str(plan_path), snapshot={"option_mark": 1.62},
                                       state_dir=str(tmp_path))
    assert payload["alert"] is True
    assert payload["decision"] == "TAKE_PROFIT"
    assert payload["option_id"] == "SPY260623C00505000"
    assert (tmp_path / "position_state.json").exists()
    assert (tmp_path / "position_decision.json").exists()


def test_run_position_watchdog_no_plan_is_quiet(tmp_path):
    payload = op.run_position_watchdog(plan_path=str(tmp_path / "nope.json"),
                                       snapshot={}, state_dir=str(tmp_path))
    assert payload["alert"] is False
    assert payload["decision"] == "NO_POSITION"
    assert payload["plan_status"] == "missing"
