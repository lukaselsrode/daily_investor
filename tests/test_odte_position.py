"""tests/test_odte_position.py — 0DTE live-position decision watchdog (decision-only).

No Robinhood, no network, no LLM. evaluate_position() is pure: a trade plan + a caller-supplied
snapshot in, structured triggers out. Covers single-contract scalp take-profit (+20-25%), the +25%
strong exit, thesis-death levels (incl. IWM + the zero-stop-is-absent guard), bid-floor (incl. the
risk_rules.initial_bid_floor fallback), percent-vs-fraction threshold normalization, the
BID_MEMORY_PROTECT giveback harvest (incl. the 2026-08-03 tape replay), time-risk,
monitoring-degraded, the no-position quiet path, the NVDA employer-restriction refusal, and the
run_position_watchdog file writer.
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

def test_scalp_take_profit_band_20_to_25():
    # Single-contract scalp up +20-24% -> TAKE_PROFIT scale, action exit (a single contract can't scale).
    # time_rules={} isolates the profit axis (no wall-clock TIME_RISK interference).
    for mark in (1.20, 1.22, 1.24):
        r = op.evaluate_position(_scalp_plan(time_rules={}), {"option_mark": mark})
        assert r["decision"] == "TAKE_PROFIT"
        tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
        assert tp["stage"] == "scale" and tp["action"] == "exit"


def test_take_profit_below_band_holds():
    r = op.evaluate_position(_scalp_plan(time_rules={}), {"option_mark": 1.19})   # +19% < 20%
    assert r["decision"] == "HOLD"
    assert "TAKE_PROFIT" not in _types(r)


def test_strong_exit_at_25pct_for_scalp():
    r = op.evaluate_position(_scalp_plan(time_rules={}), {"option_mark": 1.26})   # +26%
    assert r["decision"] == "TAKE_PROFIT"
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["stage"] == "strong" and tp["action"] == "exit_all"


def test_multi_contract_scalp_scales_keeps_runner():
    # Multi-contract scalp at +20% -> sell partial and KEEP a runner (not a full exit).
    plan = _scalp_plan(quantity=3, thesis={}, time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.20})
    tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
    assert tp["stage"] == "scale" and tp["action"] == "scale_keep_runner"


def test_pnl_pct_supplied_directly_wins():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}), {"pnl_pct": 0.26})
    assert r["decision"] == "TAKE_PROFIT" and r["pnl_pct"] == 0.26


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


# --- MONITORING_DEGRADED names its cause (2026-08-11) ----------------------------------------
# One generic string for three problems meant a CALLER ERROR read like a bad feed: the controller
# ran `make odte-position JSON=1` twice with no SNAPSHOT= while a position was live, got a verdict
# that looked legitimate, and repeated it.

def _degraded(result):
    return next(t for t in result["triggers"] if t["type"] == "MONITORING_DEGRADED")


def test_bare_invocation_says_no_snapshot_was_supplied():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}), {})
    assert r["decision"] == "MONITORING_DEGRADED"
    t = _degraded(r)
    assert t["cause"] == "no_snapshot_supplied"
    assert t["action"] == "supply_snapshot"
    assert "SNAPSHOT=" in t["detail"], "the message must name the flag that fixes it"


def test_priced_snapshot_missing_price_fields_is_a_feed_problem():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}), {"underlying_last": 500.0})
    assert _degraded(r)["cause"] == "snapshot_unpriceable"


def test_explicit_monitoring_ok_false_is_its_own_cause():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}),
                             {"option_mark": 1.10, "monitoring_ok": False})
    assert _degraded(r)["cause"] == "monitoring_ok_false"


def test_a_priced_snapshot_does_not_degrade_at_all():
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}), {"option_mark": 1.10})
    assert "MONITORING_DEGRADED" not in _types(r)


# --- bid-memory persistence (2026-08-10) -----------------------------------------------------
# evaluate_position RETURNS the updated peak but stays pure; until now nothing wrote it back, so
# BID_MEMORY_PROTECT's entire state depended on the LLM controller hand-persisting it between
# polls. On the live 14:19 SPY 774C trade the plan went unwritten across 7+ management checks.

def _poll(tmp_path, snapshot, **plan_over):
    plan_path = tmp_path / "active_trade.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(_scalp_plan(thesis={}, time_rules={}, **plan_over)))
    return op.run_position_watchdog(plan_path=str(plan_path), snapshot=snapshot,
                                    state_dir=str(tmp_path)), plan_path


def test_best_seen_bid_is_persisted_into_the_plan(tmp_path):
    _, plan_path = _poll(tmp_path, {"option_mark": 1.10, "option_bid": 1.20})
    assert json.loads(plan_path.read_text())["management"]["best_seen_bid"] == 1.20


def test_persisted_peak_only_ratchets_up(tmp_path):
    _, plan_path = _poll(tmp_path, {"option_mark": 1.10, "option_bid": 1.20})
    _poll(tmp_path, {"option_mark": 1.05, "option_bid": 1.02})     # bid fell
    assert json.loads(plan_path.read_text())["management"]["best_seen_bid"] == 1.20, \
        "a later, lower bid must never walk the recorded peak down"


def test_persistence_preserves_the_thesis_the_controller_owns(tmp_path):
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text(json.dumps(_scalp_plan(
        time_rules={}, thesis={"underlying_stop": 773.4753, "orb_level": 773.63},
        entry_order_id="abc123")))
    op.run_position_watchdog(plan_path=str(plan_path),
                             snapshot={"option_mark": 1.10, "option_bid": 1.20},
                             state_dir=str(tmp_path))
    plan = json.loads(plan_path.read_text())
    assert plan["thesis"] == {"underlying_stop": 773.4753, "orb_level": 773.63}
    assert plan["entry_order_id"] == "abc123"
    assert plan["management"]["best_seen_bid"] == 1.20


def test_no_position_never_writes_a_plan(tmp_path):
    op.run_position_watchdog(plan_path=str(tmp_path / "nope.json"), snapshot={"option_bid": 9.0},
                             state_dir=str(tmp_path))
    assert not (tmp_path / "nope.json").exists(), "must never create or repair a plan"


def test_invalid_plan_is_left_untouched(tmp_path):
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text("{not json")
    op.run_position_watchdog(plan_path=str(plan_path), snapshot={"option_bid": 9.0},
                             state_dir=str(tmp_path))
    assert plan_path.read_text() == "{not json"


def test_bid_memory_rail_now_fires_across_polls(tmp_path):
    """The point of the whole change: the rail is stateful, so it needs state that survives.

    Poll 1 lifts the bid past the arm threshold; poll 2 fades it through the giveback floor.
    Without persistence poll 2 starts from no peak, never arms, and the winner round-trips —
    which is exactly the scenario BID_MEMORY_PROTECT was written for.
    """
    entry = 1.00
    armed_bid = entry * (1.0 + op.BID_MEMORY_ARM_GAIN_PCT) + 0.05          # comfortably armed
    floor = entry + (armed_bid - entry) * (1.0 - op.BID_MEMORY_GIVEBACK_FRACTION)

    first, plan_path = _poll(tmp_path, {"option_mark": 1.10, "option_bid": armed_bid})
    assert first["best_seen_bid"] == armed_bid
    assert json.loads(plan_path.read_text())["management"]["best_seen_bid"] == armed_bid

    second, _ = _poll(tmp_path, {"option_mark": 1.10, "option_bid": floor - 0.01})
    assert second["decision"] == "BID_MEMORY_PROTECT"
    assert "BID_MEMORY_PROTECT" in {t["type"] for t in second["triggers"]}


def test_without_the_peak_the_rail_cannot_arm(tmp_path):
    """Control for the test above — same second poll, no persisted peak, no trigger."""
    entry = 1.00
    armed_bid = entry * (1.0 + op.BID_MEMORY_ARM_GAIN_PCT) + 0.05
    floor = entry + (armed_bid - entry) * (1.0 - op.BID_MEMORY_GIVEBACK_FRACTION)
    r = op.evaluate_position(_scalp_plan(thesis={}, time_rules={}),
                             {"option_mark": 1.10, "option_bid": floor - 0.01})
    assert "BID_MEMORY_PROTECT" not in _types(r)


# --- standardized journal wiring (fail-safe) -------------------------------------------------

def _read_journal(tmp_path):
    import data.odte_journal as oj
    return oj.read_events(str(tmp_path / "decision_journal.jsonl"))


def test_position_watchdog_appends_standardized_management_check(tmp_path):
    """An actionable decision is folded into the co-located journal as a non-executable
    management_check, with provenance pointing at position_decision.json."""
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text(json.dumps(_scalp_plan(thesis={}, time_rules={})))
    payload = op.run_position_watchdog(plan_path=str(plan_path), snapshot={"option_mark": 1.62},
                                       state_dir=str(tmp_path))
    rows = _read_journal(tmp_path)
    assert len(rows) == 1
    e = rows[0]
    assert e["event_type"] == "management_check" and e["source"] == "position"
    assert e["decision"] == payload["decision"].lower() or e.get("decision_detail")
    assert e["execution_allowed"] is False                       # never an execution authorization
    assert e["raw_artifact_path"].endswith("position_decision.json")
    assert e["raw_artifact_sha"]


def test_position_watchdog_hold_and_no_position_stdout_unchanged(tmp_path):
    """The decision/stdout contract is unchanged by journaling (HOLD/NO_POSITION still quiet)."""
    # NO_POSITION (missing plan)
    p1 = op.run_position_watchdog(plan_path=str(tmp_path / "nope.json"), snapshot={},
                                  state_dir=str(tmp_path))
    assert p1["alert"] is False and p1["decision"] == "NO_POSITION"
    # journaling still records the check, but the payload is untouched
    assert "raw_artifact_path" not in p1 and set(p1) >= {"alert", "decision", "triggers", "pnl_pct"}


def test_position_watchdog_journal_failure_never_crashes(monkeypatch, tmp_path):
    """If journaling raises, the watchdog still returns its decision and writes its files."""
    import data.odte_journal as oj

    def _boom(*a, **k):
        raise RuntimeError("journal exploded")
    monkeypatch.setattr(oj, "append_decision_journal", _boom)
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text(json.dumps(_scalp_plan(thesis={}, time_rules={})))
    payload = op.run_position_watchdog(plan_path=str(plan_path), snapshot={"option_mark": 1.62},
                                       state_dir=str(tmp_path))
    assert payload["decision"] == "TAKE_PROFIT"                  # unchanged
    assert (tmp_path / "position_decision.json").exists()        # files still written
    assert _read_journal(tmp_path) == []                         # nothing journaled, no crash


# --- iwm stop / zero-stop guard (2026-08-04 exit-fidelity fixes) -----------------------------

def test_thesis_dead_call_iwm_support_lost():
    # iwm_stop was silently ignored by _thesis_breaches — a declared IWM support never fired.
    plan = _scalp_plan(thesis={"iwm_stop": 295.20}, time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.00, "iwm_last": 295.00})
    assert r["decision"] == "THESIS_DEAD"
    reasons = next(t for t in r["triggers"] if t["type"] == "THESIS_DEAD")["reasons"]
    assert any("IWM" in x and "support lost" in x for x in reasons)


def test_thesis_dead_put_iwm_resistance_reclaimed():
    plan = _scalp_plan(option_type="put", thesis={"iwm_stop": 295.20}, time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.00, "iwm_last": 295.50})
    assert r["decision"] == "THESIS_DEAD"


def test_thesis_alive_call_iwm_above_support():
    plan = _scalp_plan(thesis={"iwm_stop": 295.20}, time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.00, "iwm_last": 295.50})
    assert "THESIS_DEAD" not in _types(r)


def test_zero_stop_levels_are_absent_not_live():
    # The 2026-08-03 live plan shipped vix_stop 0.0; read literally "vix >= 0" is always true for
    # a call and would kill every position the moment a snapshot carried vix.
    plan = _scalp_plan(thesis={"vix_stop": 0.0, "underlying_stop": 0.0, "vixy_stop": 20.20},
                       time_rules={})
    r = op.evaluate_position(plan, {"option_mark": 1.00, "vix": 17.5, "underlying_last": 505.0,
                                    "vixy": 19.0})
    assert "THESIS_DEAD" not in _types(r)
    # A real (nonzero) vol stop still fires.
    r2 = op.evaluate_position(plan, {"option_mark": 1.00, "vixy": 20.50})
    assert r2["decision"] == "THESIS_DEAD"


# --- bid floor mapping (risk_rules.initial_bid_floor) ----------------------------------------

def test_risk_rules_initial_bid_floor_fires():
    # The 2026-08-03 plan declared its floor ONLY as risk_rules.initial_bid_floor (0.427) — the
    # evaluator read top-level bid_floor and fell back to the 0.05 default, so it never fired.
    plan = _scalp_plan(thesis={}, time_rules={}, risk_rules={"initial_bid_floor": 0.427})
    r = op.evaluate_position(plan, {"option_bid": 0.42})
    assert r["decision"] == "BID_FLOOR"
    r2 = op.evaluate_position(plan, {"option_bid": 0.45})
    assert "BID_FLOOR" not in _types(r2)


def test_top_level_bid_floor_wins_over_risk_rules():
    plan = _scalp_plan(thesis={}, time_rules={}, bid_floor=0.10,
                       risk_rules={"initial_bid_floor": 0.427})
    r = op.evaluate_position(plan, {"option_bid": 0.42})
    assert "BID_FLOOR" not in _types(r)


# --- percent-vs-fraction normalization -------------------------------------------------------

def test_percent_thresholds_equal_fraction_thresholds():
    # The 2026-08-03 plan carried take_profit_pct: 20 (percent). Read as a fraction (+2000%) the
    # declared take-profit could never fire. 20 must mean exactly what 0.20 means.
    for tp in (20, 0.20):
        plan = _scalp_plan(thesis={}, time_rules={}, profit_rules={"take_profit_pct": tp})
        r = op.evaluate_position(plan, {"option_mark": 1.21})
        assert r["decision"] == "TAKE_PROFIT", f"take_profit_pct={tp}"
    # Legacy top-level fields normalize the same way; strong stage too.
    for strong in (25, 0.25):
        plan = _scalp_plan(thesis={}, time_rules={}, strong_exit_pct=strong)
        r = op.evaluate_position(plan, {"option_mark": 1.26})
        tp = next(t for t in r["triggers"] if t["type"] == "TAKE_PROFIT")
        assert tp["stage"] == "strong", f"strong_exit_pct={strong}"


# --- bid-memory giveback protection (BID_MEMORY_PROTECT) -------------------------------------

def _bm_floor(entry, best):
    return entry + (best - entry) * (1.0 - op.BID_MEMORY_GIVEBACK_FRACTION)


def test_bid_memory_not_armed_below_gain_threshold():
    # Best-seen just under the arm threshold: never fires, whatever the fade.
    entry = 1.00
    best = entry * (1.0 + op.BID_MEMORY_ARM_GAIN_PCT) - 0.01
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       management={"best_seen_bid": best})
    r = op.evaluate_position(plan, {"option_bid": entry * 0.9})
    assert "BID_MEMORY_PROTECT" not in _types(r)


def test_bid_memory_armed_no_fire_at_peak():
    # At a fresh peak the bid IS best-seen — above the giveback floor by construction.
    entry = 1.00
    peak = entry * (1.0 + op.BID_MEMORY_ARM_GAIN_PCT) + 0.10
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       management={"best_seen_bid": peak - 0.05})
    r = op.evaluate_position(plan, {"option_bid": peak})
    assert "BID_MEMORY_PROTECT" not in _types(r)
    assert r["best_seen_bid"] == peak                      # evaluator returns the updated peak


def test_bid_memory_fires_on_giveback():
    entry = 1.00
    best = 1.40                                            # comfortably armed
    fade = _bm_floor(entry, best) - 0.01
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       management={"best_seen_bid": best})
    r = op.evaluate_position(plan, {"option_bid": fade})
    assert r["decision"] == "BID_MEMORY_PROTECT"
    bm = next(t for t in r["triggers"] if t["type"] == "BID_MEMORY_PROTECT")
    assert bm["action"] == "harvest_now"
    assert abs(bm["giveback_floor"] - _bm_floor(entry, best)) < 1e-6


def test_bid_memory_aug3_tape_replay():
    # The trade that proved the rule: SPY 756C 2026-08-03 — entry 0.61, peak bid 0.76, fade to
    # 0.67. The manual close harvested at 0.70; the rule must fire on the same tape.
    entry, peak, fade = 0.61, 0.76, 0.67
    assert peak >= entry * (1.0 + op.BID_MEMORY_ARM_GAIN_PCT)      # armed on this tape
    assert fade <= _bm_floor(entry, peak)                          # fade breaches the floor
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       management={"best_seen_bid": peak})
    r = op.evaluate_position(plan, {"option_bid": fade})
    assert r["decision"] == "BID_MEMORY_PROTECT"


def test_bid_memory_outranks_take_profit_but_not_thesis_death():
    entry = 1.00
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       management={"best_seen_bid": 1.40})
    both = {"option_mark": 1.22, "option_bid": _bm_floor(entry, 1.40) - 0.01}
    r = op.evaluate_position(plan, both)
    assert {"BID_MEMORY_PROTECT", "TAKE_PROFIT"} <= _types(r)
    assert r["decision"] == "BID_MEMORY_PROTECT"
    dead = dict(both, underlying_last=498.0)
    r2 = op.evaluate_position(_scalp_plan(thesis={"underlying_stop": 500.0}, time_rules={},
                                          entry_price=entry,
                                          management={"best_seen_bid": 1.40}), dead)
    assert r2["decision"] == "THESIS_DEAD"


def test_bid_memory_overrides_via_plan():
    # Plan-level bid_memory overrides the module constants (accepted top-level or in profit_rules).
    entry = 1.00
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       bid_memory={"arm_gain_pct": 0.05, "giveback_fraction": 0.50},
                       management={"best_seen_bid": 1.10})       # +10%: armed only via override
    r = op.evaluate_position(plan, {"option_bid": 1.04})         # floor = 1 + 0.10*0.5 = 1.05
    assert r["decision"] == "BID_MEMORY_PROTECT"


def test_best_seen_bid_tracks_and_survives_missing_bid():
    entry = 1.00
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=entry,
                       management={"best_seen_bid": 1.05})
    r = op.evaluate_position(plan, {"option_bid": 1.12})
    assert r["best_seen_bid"] == 1.12                            # new peak returned for persisting
    r2 = op.evaluate_position(plan, {"option_mark": 1.10})       # no bid this poll
    assert r2["best_seen_bid"] == 1.05                           # unchanged, not reset


def test_watchdog_payload_carries_best_seen_bid(tmp_path):
    plan = _scalp_plan(thesis={}, time_rules={}, entry_price=1.00,
                       management={"best_seen_bid": 1.05})
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text(json.dumps(plan))
    payload = op.run_position_watchdog(plan_path=str(plan_path),
                                       snapshot={"option_bid": 1.12, "option_mark": 1.12},
                                       state_dir=str(tmp_path))
    assert payload["best_seen_bid"] == 1.12


# --- 2026-08-07: unconditional max-loss backstop ----------------------------------------------
# THESIS_DEAD and TIME_RISK only exist when the CONTROLLER wrote a thesis / time_rules block:
# _thesis_breaches returns [] on an absent thesis, _time_risk returns None on absent rules. So a
# thin plan had exactly one unconditional exit — BID_FLOOR, which on a $1.00 entry waits for -95%.

def test_thin_plan_now_has_an_unconditional_exit():
    from data.odte_position import DEFAULT_MAX_LOSS_PCT, evaluate_position
    # time_rules={} isolates the loss axis — otherwise the 15:45 clock backstop fires whenever
    # the suite happens to run after that hour.
    thin = {"underlying": "IWM", "option_type": "call", "entry_price": 1.00,
            "quantity": 1, "status": "open", "time_rules": {}}
    # Both prices are DERIVED from the live constant — a fixture that hardcodes the stop stops
    # testing the stop the moment it moves (these read 0.80/0.60 while it was -0.40).
    inside = round(1.00 * (1 + DEFAULT_MAX_LOSS_PCT / 2), 4)      # half-way to the stop
    at_stop = round(1.00 * (1 + DEFAULT_MAX_LOSS_PCT), 4)
    # comfortably inside the stop: still held, as before
    held = evaluate_position(thin, {"option_bid": inside, "option_mark": inside,
                                    "underlying_last": 300.0})
    assert held["decision"] == "HOLD"
    # at the stop: fires even though the plan declares no thesis at all
    hit = evaluate_position(thin, {"option_bid": at_stop, "option_mark": at_stop,
                                   "underlying_last": 300.0})
    assert hit["decision"] == "MAX_LOSS"
    assert abs(hit["pnl_pct"] - DEFAULT_MAX_LOSS_PCT) < 1e-9
    assert any(t["type"] == "MAX_LOSS" and t["action"] == "exit" for t in hit["triggers"])


def test_loss_stop_sits_below_every_winner_drawdown_ever_measured():
    """The rail changed KIND on 2026-08-12: -0.40 was a catastrophe backstop sitting outside the
    working range (worst observed excursion -33.8%); -0.25 is a loss stop INSIDE it that will cut
    live trades. So the old invariant ("below every excursion ever seen") no longer applies and
    would be vacuous — what must hold instead is that the stop never cuts a trade we know recovered.

    Deepest drawdown among WINNERS across the 11 trades with tool-measured excursion
    (`odte_journal.summarize()["measured_excursions"]`, joined on option_id): -0.186, the +$22 SPY
    771p/772p pair's winner. The full winner distribution is -0.186, -0.13, -0.09, -0.056, +0.047,
    so -0.25 sits in a genuine gap rather than just below the worst point.

    Margin matters more than the raw comparison: MAE is the MINIMUM over ~35s-spaced polls, so it
    understates true drawdown and a winner recorded at -0.186 may really have touched lower. -0.20
    was rejected for leaving only 1.4pp. Require a real buffer, not just ordering."""
    from data.odte_position import DEFAULT_MAX_LOSS_PCT
    deepest_winner_drawdown = -0.186
    assert DEFAULT_MAX_LOSS_PCT < deepest_winner_drawdown, DEFAULT_MAX_LOSS_PCT
    assert deepest_winner_drawdown - DEFAULT_MAX_LOSS_PCT >= 0.05, (
        "stop is within 5pp of a known winner's drawdown, and MAE understates drawdown")


def test_plan_may_override_the_backstop_either_signed():
    from data.odte_position import DEFAULT_MAX_LOSS_PCT, evaluate_position
    # time_rules={} isolates the loss axis — otherwise the 15:45 clock backstop fires whenever
    # the suite happens to run after that hour.
    thin = {"underlying": "IWM", "option_type": "call", "entry_price": 1.00,
            "quantity": 1, "status": "open", "time_rules": {}}
    # Every level is derived from the live default so the three cases keep their MEANING when the
    # rail moves: sit the position inside the default, then override looser and tighter than it.
    level = round(1.00 * (1 + DEFAULT_MAX_LOSS_PCT * 0.8), 4)     # 80% of the way to the stop
    snap = {"option_bid": level, "option_mark": level, "underlying_last": 300.0}
    assert evaluate_position(thin, snap)["decision"] == "HOLD"                    # inside default
    loose = evaluate_position({**thin, "max_loss_pct": DEFAULT_MAX_LOSS_PCT * 2}, snap)
    assert loose["decision"] == "HOLD"
    # live plans put risk keys under risk_rules, and may write the magnitude unsigned
    tight = evaluate_position({**thin, "risk_rules": {"max_loss_pct": abs(DEFAULT_MAX_LOSS_PCT) / 2}},
                              snap)
    assert tight["decision"] == "MAX_LOSS"


def test_thesis_death_still_outranks_the_backstop():
    """The backstop is a floor, not a replacement: a named tape invalidation stays the reported
    reason so the postmortem attributes the exit to the thesis, not to the catch-all."""
    from data.odte_position import evaluate_position
    plan = {"underlying": "IWM", "option_type": "call", "entry_price": 1.00, "quantity": 1,
            "status": "open", "thesis": {"underlying_stop": 305.0}}
    out = evaluate_position(plan, {"option_bid": 0.50, "option_mark": 0.50,
                                   "underlying_last": 300.0})
    assert out["decision"] == "THESIS_DEAD"
    assert {"MAX_LOSS", "THESIS_DEAD"} <= {t["type"] for t in out["triggers"]}


def test_bid_floor_outranks_the_backstop_as_the_more_specific_reason():
    """A near-worthless option is BOTH past the backstop and at the floor. The floor is the
    specific diagnosis, so it stays the reported decision — the backstop must never mask a named
    reason in the postmortem."""
    from data.odte_position import evaluate_position
    # time_rules={} isolates the loss axis — otherwise the 15:45 clock backstop fires whenever
    # the suite happens to run after that hour.
    thin = {"underlying": "IWM", "option_type": "call", "entry_price": 1.00,
            "quantity": 1, "status": "open", "time_rules": {}}
    out = evaluate_position(thin, {"option_bid": 0.03, "option_mark": 0.04,
                                   "underlying_last": 300.0})
    assert out["decision"] == "BID_FLOOR"
    assert {"BID_FLOOR", "MAX_LOSS"} <= {t["type"] for t in out["triggers"]}


def test_absent_time_rules_get_the_clock_backstop_but_empty_ones_are_honoured():
    """A WINNING 0DTE position at 15:59 ET used to read HOLD and ride into expiry, where an
    out-of-the-money contract is worth zero — the max-loss backstop cannot catch that (it is a
    winner) and neither can BID_FLOOR (the bid is healthy).

    ABSENT vs EXPLICITLY-EMPTY are different declarations. Missing means the author never
    considered the clock and gets the backstop; `time_rules: {}` is a deliberate "no clock exit"
    and is honoured — the contract the rest of this suite pins by using it to isolate other axes.
    """
    from data.odte_position import DEFAULT_FLAT_BEFORE_ET, evaluate_position
    base = {"underlying": "IWM", "option_type": "call", "entry_price": 1.00,
            "quantity": 1, "status": "open"}
    winner = {"option_mark": 1.10, "option_bid": 1.09, "underlying_last": 300.0,
              "now_et": "2026-08-07T15:59:00-04:00"}
    absent = evaluate_position(base, winner)
    assert absent["decision"] == "TIME_RISK"
    tr = next(t for t in absent["triggers"] if t["type"] == "TIME_RISK")
    assert tr["stage"] == "flat" and DEFAULT_FLAT_BEFORE_ET in tr["detail"]

    explicit = evaluate_position({**base, "time_rules": {}}, winner)
    assert explicit["decision"] != "TIME_RISK"

    # and well before the cutoff the backstop is silent
    early = evaluate_position(base, {**winner, "now_et": "2026-08-07T11:00:00-04:00"})
    assert early["decision"] != "TIME_RISK"


def test_clock_backstop_never_bound_a_historical_exit():
    """Measured, not chosen: no exit in the journal has occurred after 15:30 ET (latest 14:20,
    n=12), so a 15:45 flatten pre-empts nothing that actually happened."""
    from data.odte_position import DEFAULT_FLAT_BEFORE_ET
    hh, mm = (int(x) for x in DEFAULT_FLAT_BEFORE_ET.split(":"))
    assert (hh, mm) > (15, 30)


def test_position_payload_echoes_the_plan_trade_id():
    """Dropping it is why only 10 of 455 management_check events ever carried a trade_id:
    `event_from_position_decision` stamps `trade_id or payload["trade_id"]` and this payload had
    neither, so grouping the poll stream by trade_id silently yielded zero polls for nearly every
    trade and analysis had to join through option_id — which merged 2026-08-11's two SPY trades."""
    from data.odte_position import evaluate_position
    plan = {"trade_id": "iwm-20260813-abc", "underlying": "IWM", "option_type": "put",
            "entry_price": 0.46, "quantity": 1, "status": "open", "option_id": "opt-abc",
            "time_rules": {}}
    out = evaluate_position(plan, {"option_bid": 0.44, "option_mark": 0.44,
                                   "underlying_last": 300.0})
    assert out["trade_id"] == "iwm-20260813-abc"


def test_trade_id_reaches_the_journal_event():
    from data.odte_journal import event_from_position_decision
    from data.odte_position import evaluate_position
    plan = {"trade_id": "spy-20260813-xyz", "underlying": "SPY", "option_type": "call",
            "entry_price": 1.00, "quantity": 1, "status": "open", "option_id": "opt-xyz",
            "time_rules": {}}
    out = evaluate_position(plan, {"option_bid": 0.95, "option_mark": 0.95,
                                   "underlying_last": 500.0})
    assert event_from_position_decision(out)["trade_id"] == "spy-20260813-xyz"
