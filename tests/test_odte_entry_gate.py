"""tests/test_odte_entry_gate.py — PURE/OFFLINE thesis->entry gate (no broker/network/LLM/orders).

Covers the conservative invariants the live execution manager relies on:
  * scan_only can NEVER be execution-allowed,
  * a missing gate input fails CLOSED (no execution, observe not enter),
  * execution is allowed ONLY when every required gate is explicitly true and scan_only is false,
  * a restricted (NVDA) underlying can never execute,
  * the journal event built from a gate decision carries reason/veto/thesis/tier fields and the
    journal re-enforces the scan_only/restricted guards on append.
"""
import inspect
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_entry_gate as eg
import data.odte_journal as oj


def _journal(tmp_path):
    return str(tmp_path / "decision_journal.jsonl")


# All gates explicitly positive — the only shape that may authorize execution.
_GOOD_DAY = {"verdict": "GOOD_DAY", "score": 5}
_GOOD_VEHICLE = {"verdict": "GOOD_BET", "score": 6, "direction": "bullish",
                 "contract": {"underlying": "QQQ", "option_type": "call",
                              "option_id": "QQQ260727C00718000", "expiration_date": "2026-07-27",
                              "strike_price": 718.0},
                 "reasons": ["market: VWAP confirms calls on SPY,QQQ", "gamma: low pin risk"]}
_GOOD_BROKER = {"buying_power": 250.0, "day_trades_left": 3,
                "nonzero_option_positions_count": 0, "open_option_orders_count": 0,
                "today_option_orders_count": 0}
_CANDIDATE = {"ticker": "QQQ", "direction": "bullish", "option_type": "call",
              "option_id": "QQQ260727C00718000", "expiration_date": "2026-07-27",
              "strike_price": 718.0, "selection_timestamp": "2026-07-27T14:00:00+00:00"}
_CONFIRMATIONS = {name: True for name in eg.REQUIRED_CONFIRMATIONS}


def _all_gates_kwargs():
    return dict(candidate=dict(_CANDIDATE), day_score=dict(_GOOD_DAY),
                vehicle_score=dict(_GOOD_VEHICLE), broker_snapshot=dict(_GOOD_BROKER),
                confirmations=dict(_CONFIRMATIONS))


# --- happy path: every explicit gate true, not scan_only -> enter + execution allowed -------------

def test_all_explicit_gates_allow_only_when_not_scan_only():
    d = eg.build_entry_gate_decision(**_all_gates_kwargs())
    assert d["symbol"] == "QQQ" and d["direction"] == "bullish"
    assert all(d["gates"][g] is True for g in d["required_gates"])
    assert d["veto_reasons"] == []
    assert d["execution_allowed"] is True
    assert d["decision"] == "enter" and d["intent"] == "enter"
    # required live confirmations are still recorded even when the gate authorizes execution
    assert set(d["required_confirmations"]) == set(eg.REQUIRED_CONFIRMATIONS)


def test_scan_only_can_never_execute_even_with_all_gates_true():
    d = eg.build_entry_gate_decision(scan_only=True, **_all_gates_kwargs())
    assert d["scan_only"] is True
    assert d["execution_allowed"] is False, "scan_only must force execution_allowed False"
    assert d["decision"] == "observe"


# --- tier boundary: scan_only inherited from the trigger, only promoted by explicit opt-in ---------

def test_scan_only_trigger_is_inherited_and_stays_observe_by_default():
    # A watchdog trigger carries scan_only=True; the gate must INHERIT it (no explicit scan_only),
    # so even with all gates good the record stays observe / non-executable.
    trigger = {"scan_only": True, "execution_allowed": False,
               "candidate": dict(_CANDIDATE)}
    d = eg.build_entry_gate_decision(trigger=trigger, day_score=dict(_GOOD_DAY),
                                     vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER))
    assert all(d["gates"][g] is True for g in d["required_gates"])  # gates would otherwise allow
    assert d["scan_only"] is True, "scan_only must be inherited from the trigger"
    assert d["promoted_to_execution"] is False
    assert d["execution_allowed"] is False and d["decision"] == "observe"
    assert "scan_only_inherited" in d["reason_codes"]


def test_promote_to_execution_is_deprecated_and_fails_closed():
    # REGRESSION (2026-07-23 delayed-fill incident): a bare promote_to_execution=True on a
    # scan_only trigger must NOT make the record executable anymore. The attempt is recorded and
    # answered with execution_lease_required — the lease is minted only via odte-convert.
    trigger = {"scan_only": True, "execution_allowed": False,
               "candidate": dict(_CANDIDATE)}
    d = eg.build_entry_gate_decision(trigger=trigger, day_score=dict(_GOOD_DAY),
                                     vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER),
                                     promote_to_execution=True)
    assert d["scan_only"] is True, "bare promotion can no longer demote the scan tier"
    assert d["promoted_to_execution"] is False
    assert d["promotion_requested"] is True
    assert d["execution_allowed"] is False and d["decision"] == "observe"
    assert eg.EXECUTION_LEASE_REQUIRED in d["reason_codes"]
    assert "scan_only_inherited" in d["reason_codes"]
    assert d["next_command"] == "odte-convert"


def test_candidate_scan_only_is_also_inherited():
    d = eg.build_entry_gate_decision(candidate={**_CANDIDATE, "scan_only": True},
                                     day_score=dict(_GOOD_DAY), vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER))
    assert d["scan_only"] is True and d["execution_allowed"] is False


def test_promote_cannot_override_restricted():
    # Promotion is NOT a restriction bypass — NVDA still vetoes regardless of promote_to_execution.
    d = eg.build_entry_gate_decision(trigger={"scan_only": True},
                                     candidate={"ticker": "NVDA", "direction": "bullish"},
                                     day_score=dict(_GOOD_DAY), vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER), promote_to_execution=True)
    assert "restricted_employer" in d["veto_reasons"]
    assert d["execution_allowed"] is False and d["decision"] == "veto"


def test_run_entry_gate_inherits_scan_only_from_trigger_json():
    # run_entry_gate default scan_only=None must inherit from the trigger payload (CLI default path).
    import json
    d = eg.run_entry_gate(
        trigger_json=json.dumps({"scan_only": True, "candidate": _CANDIDATE}),
        day_score_json=json.dumps(_GOOD_DAY),
        vehicle_score_json=json.dumps(_GOOD_VEHICLE),
        broker_json=json.dumps(_GOOD_BROKER))
    assert d["scan_only"] is True and d["execution_allowed"] is False
    # ...and the deprecated promote flag STAYS fail-closed on the CLI path too (2026-07-23).
    d2 = eg.run_entry_gate(
        trigger_json=json.dumps({"scan_only": True, "candidate": _CANDIDATE}),
        day_score_json=json.dumps(_GOOD_DAY),
        vehicle_score_json=json.dumps(_GOOD_VEHICLE),
        broker_json=json.dumps(_GOOD_BROKER), promote_to_execution=True)
    assert d2["scan_only"] is True and d2["execution_allowed"] is False
    assert eg.EXECUTION_LEASE_REQUIRED in d2["reason_codes"]


# --- fail-closed: missing inputs ----------------------------------------------------------------

def test_missing_gates_fail_closed_to_observe_not_execute():
    # Only a candidate/direction — no day_score, vehicle_score, or broker snapshot.
    d = eg.build_entry_gate_decision(candidate=dict(_CANDIDATE))
    assert d["gates"]["day_regime"] is None
    assert d["gates"]["vehicle"] is None
    assert d["gates"]["account"] is None
    assert d["gates"]["directional_thesis"] is True  # direction was supplied
    assert d["execution_allowed"] is False
    assert d["decision"] == "observe"  # unknown gates -> keep watching, do not deny outright
    assert any(c.endswith(":unknown") for c in d["reason_codes"])


def test_no_broker_snapshot_blocks_execution():
    kw = _all_gates_kwargs()
    kw.pop("broker_snapshot")
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["account"] is None
    assert d["execution_allowed"] is False, "no confirmed funds -> never execution-allowed"


# --- hard vetoes --------------------------------------------------------------------------------

def test_avoid_day_vetoes():
    kw = _all_gates_kwargs()
    kw["day_score"] = {"verdict": "AVOID", "score": -4}
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["day_regime"] is False
    assert "day_regime_avoid" in d["veto_reasons"]
    assert d["decision"] == "veto" and d["execution_allowed"] is False


def test_bad_vehicle_vetoes():
    kw = _all_gates_kwargs()
    kw["vehicle_score"] = {"verdict": "BAD_BET", "score": -3, "direction": "bullish"}
    d = eg.build_entry_gate_decision(**kw)
    assert "vehicle_bad_bet" in d["veto_reasons"]
    assert d["execution_allowed"] is False


def test_insufficient_buying_power_vetoes():
    kw = _all_gates_kwargs()
    kw["broker_snapshot"] = {"buying_power": 0.0, "day_trades_left": 3}
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["account"] is False
    assert "insufficient_buying_power" in d["veto_reasons"]
    assert d["execution_allowed"] is False


def test_no_day_trades_left_vetoes():
    kw = _all_gates_kwargs()
    kw["broker_snapshot"] = {"buying_power": 250.0, "day_trades_left": 0}
    d = eg.build_entry_gate_decision(**kw)
    assert "no_day_trades_left" in d["veto_reasons"]
    assert d["execution_allowed"] is False


# --- restricted underlying ----------------------------------------------------------------------

def test_restricted_symbol_can_never_execute():
    kw = _all_gates_kwargs()
    kw["candidate"] = {"ticker": "NVDA", "direction": "bullish"}
    kw["vehicle_score"] = {**_GOOD_VEHICLE,
                           "contract": {"underlying": "NVDA", "option_type": "call", "strike": 130}}
    d = eg.build_entry_gate_decision(**kw)
    assert d["symbol"] == "NVDA"
    assert "restricted_employer" in d["veto_reasons"]
    assert d["decision"] == "veto" and d["execution_allowed"] is False


# --- not-all-positive but no hard veto -> deny --------------------------------------------------

def test_known_but_not_all_positive_denies():
    # day GOOD, vehicle WATCH (neither GOOD_BET nor BAD_BET -> gate None), broker good, direction set.
    kw = _all_gates_kwargs()
    kw["vehicle_score"] = {"verdict": "WATCH", "score": 1, "direction": "bullish"}
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["vehicle"] is None
    assert d["execution_allowed"] is False
    assert d["decision"] == "observe"  # an unknown (WATCH) gate -> observe, not deny


def test_thesis_block_shape_present():
    d = eg.build_entry_gate_decision(**_all_gates_kwargs())
    t = d["thesis"]
    assert t["direction"] == "bullish"
    assert isinstance(t["basis"], list) and t["basis"]  # falls back to vehicle reasons
    assert t["day_regime"] == "GOOD_DAY" and t["vehicle_verdict"] == "GOOD_BET"


# --- journal converter + re-enforced guards -----------------------------------------------------

def test_event_from_entry_gate_shape_has_reason_veto_thesis_fields():
    d = eg.build_entry_gate_decision(**_all_gates_kwargs())
    ev = oj.event_from_entry_gate(d, trade_id="t1")
    assert ev["event_type"] == "entry_decision" and ev["trade_id"] == "t1"
    assert ev["underlying"] == "QQQ"
    assert ev["decision"]["action"] == "enter"
    assert "reason_codes" in ev and "veto_reasons" in ev and "thesis" in ev
    assert ev["required_confirmations"] and "gates" in ev
    assert ev["confirmations"] == _CONFIRMATIONS
    assert ev["candidate_fingerprint"] == d["candidate_fingerprint"]
    assert ev["option_id"] == d["option_id"]


def test_entry_gate_event_round_trips_and_journal_enforces_scan_only(tmp_path):
    jp = _journal(tmp_path)
    # A scan_only gate must remain non-executable through the journal, even if a caller forced True.
    d = eg.build_entry_gate_decision(scan_only=True, **_all_gates_kwargs())
    ev = oj.event_from_entry_gate(d)
    ev["execution_allowed"] = True  # adversarial: try to sneak execution past the journal guard
    res = oj.append_decision_journal(ev, source="entry_gate", event_type="entry_decision",
                                     journal_path=jp)
    assert res["status"] == "appended"
    assert res["event"]["scan_only"] is True
    assert res["event"]["execution_allowed"] is False


def test_entry_gate_event_journal_tags_restricted(tmp_path):
    jp = _journal(tmp_path)
    d = eg.build_entry_gate_decision(candidate={"ticker": "NVDA", "direction": "bullish"},
                                     day_score=dict(_GOOD_DAY), vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER))
    ev = oj.event_from_entry_gate(d)
    ev["execution_allowed"] = True
    res = oj.append_decision_journal(ev, source="entry_gate", event_type="entry_decision",
                                     journal_path=jp)
    e = res["event"]
    assert e["restricted"] is True and e["restricted_reason"] == "employer"
    assert e["execution_allowed"] is False


# --- next_action / next_command: machine-readable "what advances this gate" ---------------------

def test_passed_gate_next_step_is_broker_review_place_then_watch():
    # An execution-allowed gate must SAY the full sequence: the lease was already minted in-process
    # by odte-convert, so consume it at review/place, then the order guard, then the position watch.
    d = eg.build_entry_gate_decision(**_all_gates_kwargs())
    assert d["execution_allowed"] is True
    assert "odte-execution-authorize" not in d["next_command"]
    assert "broker-review-place" in d["next_command"]
    assert "odte-order-guard" in d["next_command"]
    assert "odte-position" in d["next_command"]
    assert "standing auth" in d["next_action"]
    assert d["failing_gates"] == [] and d["unknown_gates"] == []


def test_bp_vehicle_veto_guides_multi_etf_scan_before_no_trade():
    # One bullet a day must not die on the first contract checked: a BP/vehicle-fit veto points at
    # scanning the other index ETF vehicles before declaring no-trade.
    for kw_over in ({"broker_snapshot": {"buying_power": 0.0, "day_trades_left": 3}},
                    {"vehicle_score": {"verdict": "BAD_BET", "score": -3, "direction": "bullish"}}):
        kw = _all_gates_kwargs()
        kw.update(kw_over)
        d = eg.build_entry_gate_decision(**kw)
        assert d["execution_allowed"] is False
        assert "QQQ/SPY/IWM" in d["next_action"]
        assert "odte-vehicle-score" in d["next_command"]


def test_missing_gate_inputs_name_the_producing_commands():
    d = eg.build_entry_gate_decision(candidate=dict(_CANDIDATE))
    assert d["decision"] == "observe"
    assert set(d["unknown_gates"]) == {"day_regime", "vehicle", "account"}
    for expected in ("odte-day-score", "odte-vehicle-score", "broker snapshot"):
        assert expected in d["next_action"], f"missing-input guidance should mention {expected}"


def test_hard_veto_next_step_stands_down_to_scan():
    kw = _all_gates_kwargs()
    kw["day_score"] = {"verdict": "AVOID", "score": -4}
    d = eg.build_entry_gate_decision(**kw)
    assert d["decision"] == "veto"
    assert d["next_command"] == "odte-watchdog"


def test_scan_only_all_green_next_step_is_atomic_conversion():
    # Bare promotion is dead: an all-green scan-tier record points at the atomic conversion
    # (which mints the lease in-process), never at the deprecated --promote-to-execution flag.
    d = eg.build_entry_gate_decision(scan_only=True, **_all_gates_kwargs())
    assert d["decision"] == "observe" and d["execution_allowed"] is False
    assert d["next_command"] == "odte-convert"
    assert "--promote-to-execution" not in d["next_command"]


# --- green-day preservation: no re-entry after a banked green scalp -----------------------------

_LOCK_NOW = datetime(2026, 7, 10, 18, 0, tzinfo=timezone.utc)   # 14:00 ET, mid-session


def _green_scalp_events(now=_LOCK_NOW, hours_ago=2.0, realized_pnl=21.0):
    """Today's exact live sequence (2026-07-10): 3x SPY 753C filled @0.98, all 3 closed @1.05."""
    base = {"trade_id": "SPY-753C-1", "underlying": "SPY", "option_type": "call", "strike": 753}
    return [
        {**base, "event_type": "order_filled", "seq": 1, "quantity": 3, "entry_price": 0.98,
         "ts": (now - timedelta(hours=hours_ago + 0.5)).isoformat()},
        {**base, "event_type": "order_closed", "seq": 2, "quantity": 3, "exit_price": 1.05,
         "ts": (now - timedelta(hours=hours_ago)).isoformat(),
         "outcome": {"realized_pnl": realized_pnl}},
    ]


def _spy_gates_kwargs():
    kw = _all_gates_kwargs()
    kw["candidate"] = {**_CANDIDATE, "ticker": "SPY", "selected_vehicle": "SPY",
                       "option_id": "SPY260727C00753000", "strike_price": 753.0}
    kw["vehicle_score"] = {**_GOOD_VEHICLE,
                           "contract": {**_GOOD_VEHICLE["contract"], "underlying": "SPY",
                                        "option_id": "SPY260727C00753000", "strike_price": 753.0}}
    return kw


def test_green_scalp_locks_out_same_day_reentry():
    # REGRESSION (live 2026-07-10): after the green 3x SPY 753C scalp, a later gate for the same
    # contract re-entered @1.27 for a scratch. With the day's journal supplied, that later gate must
    # VETO with green_day_preservation_lockout even though every ordinary gate reads True.
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW,
                                     **_spy_gates_kwargs())
    assert d["execution_allowed"] is False
    assert d["decision"] == "veto"
    assert eg.GREEN_LOCKOUT_VETO in d["veto_reasons"]
    assert d["green_day_preservation"]["locked"] is True
    assert d["green_day_preservation"]["banked_pnl"] == 21.0
    assert "green-day preservation" in d["next_action"]


def test_reentry_override_with_ample_bp_is_allowed():
    # The explicit escape hatch: allow_reentry_after_green + BP at/above the BP-scaled floor
    # (GREEN_REENTRY_MIN_BP_MULTIPLE × the contract's own estimated cost — 2026-08-02 retune; the
    # old flat $500 floor sat above the whole account's BP and could never arm).
    kw = _spy_gates_kwargs()
    kw["trigger"] = {"allow_reentry_after_green": True}
    kw["vehicle_score"] = {**kw["vehicle_score"],
                           "contract": {**kw["vehicle_score"]["contract"], "ask": 1.2}}
    cost = 1.2 * 100.0
    kw["broker_snapshot"] = {**_GOOD_BROKER,
                             "buying_power": eg.GREEN_REENTRY_MIN_BP_MULTIPLE * cost + 10.0}
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW, **kw)
    assert d["execution_allowed"] is True
    assert d["allow_reentry_after_green"] is True
    assert eg.GREEN_LOCKOUT_VETO not in d["veto_reasons"]
    assert "green_reentry_override_armed" in d["reason_codes"]


def test_reentry_override_below_bp_multiple_stays_locked():
    # BP-aware guard: the override alone is NOT enough — below the multiple×cost floor the green
    # day is preserved no matter what (the live re-entry burned most of the remaining BP on 1x @1.27).
    kw = _spy_gates_kwargs()
    kw["vehicle_score"] = {**kw["vehicle_score"],
                           "contract": {**kw["vehicle_score"]["contract"], "ask": 1.2}}
    cost = 1.2 * 100.0
    kw["broker_snapshot"] = {**_GOOD_BROKER,
                             "buying_power": eg.GREEN_REENTRY_MIN_BP_MULTIPLE * cost - 1.0,
                             "allow_reentry_after_green": True}
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW, **kw)
    assert d["execution_allowed"] is False
    assert eg.GREEN_REENTRY_BP_VETO in d["veto_reasons"]
    assert d["allow_reentry_after_green"] is False


def test_reentry_override_with_unknown_contract_cost_stays_locked():
    # Fail closed: a re-entry whose size cannot be estimated (no ask/mark on the contract) must
    # stay locked regardless of buying power.
    kw = _spy_gates_kwargs()
    kw["broker_snapshot"] = {**_GOOD_BROKER,
                             "buying_power": 10_000.0,
                             "allow_reentry_after_green": True}
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW, **kw)
    assert d["execution_allowed"] is False
    assert eg.GREEN_REENTRY_BP_VETO in d["veto_reasons"]


def test_prior_day_green_scalp_does_not_lock():
    # New-day reset: yesterday's banked green must not lock today's first entry.
    events = _green_scalp_events(now=_LOCK_NOW - timedelta(days=1))
    d = eg.build_entry_gate_decision(journal_events=events, now=_LOCK_NOW, **_spy_gates_kwargs())
    assert d["green_day_preservation"]["locked"] is False
    assert d["execution_allowed"] is True


def test_red_close_does_not_lock():
    # Falsification: a completed LOSING trade is post-loss-cooldown territory, not green-day lockout.
    events = _green_scalp_events(realized_pnl=-30.0)
    d = eg.build_entry_gate_decision(journal_events=events, now=_LOCK_NOW, **_spy_gates_kwargs())
    assert d["green_day_preservation"]["locked"] is False
    assert d["execution_allowed"] is True


def test_run_entry_gate_reads_journal_for_lockout(tmp_path):
    # The CLI path feeds the day's journal file; a banked green scalp in it must veto the fresh gate.
    import json
    now = datetime.now(timezone.utc)
    jp = _journal(tmp_path)
    with open(jp, "w") as f:
        for e in _green_scalp_events(now=now, hours_ago=0.0):
            f.write(json.dumps(e) + "\n")
    spy = _spy_gates_kwargs()
    d = eg.run_entry_gate(candidate_json=json.dumps(spy["candidate"]),
                          day_score_json=json.dumps(_GOOD_DAY),
                          vehicle_score_json=json.dumps(spy["vehicle_score"]),
                          broker_json=json.dumps(_GOOD_BROKER),
                          confirmations_json=json.dumps(_CONFIRMATIONS),
                          journal_path=jp)
    assert d["execution_allowed"] is False
    assert eg.GREEN_LOCKOUT_VETO in d["veto_reasons"]
    # ...and a missing journal file leaves the lockout disengaged.
    d2 = eg.run_entry_gate(candidate_json=json.dumps(spy["candidate"]),
                           day_score_json=json.dumps(_GOOD_DAY),
                           vehicle_score_json=json.dumps(spy["vehicle_score"]),
                           broker_json=json.dumps(_GOOD_BROKER),
                           confirmations_json=json.dumps(_CONFIRMATIONS),
                           journal_path=str(tmp_path / "missing.jsonl"))
    assert d2["execution_allowed"] is True


# --- guardrail: no broker / network / LLM -------------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(eg)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket"):
        assert forbidden not in src, f"odte_entry_gate must not reference {forbidden!r}"


# --- Confirmation tiers (2026-08-02 retune: CHOP tradeable at A+/B+, B+ at half size) -----------

def test_chop_day_with_b_plus_tier_passes_day_regime_at_half_size():
    kw = _all_gates_kwargs()
    kw["day_score"] = {"verdict": "CHOP"}
    kw["candidate"]["tier"] = "b_plus"
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["day_regime"] is True
    assert d["execution_allowed"] is True
    assert d["tier"] == "b_plus"
    assert d["sizing_tier"] == "half"


def test_chop_day_with_a_plus_tier_passes_at_full_size():
    kw = _all_gates_kwargs()
    kw["day_score"] = {"verdict": "CHOP"}
    kw["candidate"]["tier"] = "a_plus"
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["day_regime"] is True
    assert d["execution_allowed"] is True
    assert d["sizing_tier"] == "full"


def test_bare_chop_without_tier_still_fails_closed():
    kw = _all_gates_kwargs()
    kw["day_score"] = {"verdict": "CHOP"}
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["day_regime"] is None
    assert d["execution_allowed"] is False
    assert d["decision"] == "observe"


def test_watch_vehicle_passes_only_for_b_plus_tier_with_score_floor():
    import data.odte_config as oc
    kw = _all_gates_kwargs()
    kw["candidate"]["tier"] = "b_plus"
    kw["vehicle_score"] = {**_GOOD_VEHICLE, "verdict": "WATCH",
                           "score": oc.B_PLUS_MIN_VEHICLE_SCORE}
    d = eg.build_entry_gate_decision(**kw)
    assert d["gates"]["vehicle"] is True
    assert d["execution_allowed"] is True

    kw["vehicle_score"] = {**_GOOD_VEHICLE, "verdict": "WATCH",
                           "score": oc.B_PLUS_MIN_VEHICLE_SCORE - 1}
    d2 = eg.build_entry_gate_decision(**kw)
    assert d2["gates"]["vehicle"] is None
    assert d2["execution_allowed"] is False

    # A WATCH vehicle without the B+ tier stays unknown regardless of score.
    kw2 = _all_gates_kwargs()
    kw2["vehicle_score"] = {**_GOOD_VEHICLE, "verdict": "WATCH", "score": 10}
    d3 = eg.build_entry_gate_decision(**kw2)
    assert d3["gates"]["vehicle"] is None


# --- Daily trade budget + cooldown vetoes ------------------------------------------------------

def _red_trade_events(now, n=1, minutes_since_close=60.0):
    events = []
    for i in range(n):
        events.append({"event_type": "order_filled", "trade_id": f"t{i}",
                       "ts": (now - timedelta(hours=2 + i)).isoformat()})
        events.append({"event_type": "order_closed", "trade_id": f"t{i}", "realized_pnl": -5.0,
                       "ts": (now - timedelta(minutes=minutes_since_close + i)).isoformat()})
    return events


def test_daily_budget_exhausted_vetoes_new_entry():
    import data.odte_config as oc
    events = _red_trade_events(_LOCK_NOW, n=oc.DAILY_TRADE_BUDGET)
    d = eg.build_entry_gate_decision(journal_events=events, now=_LOCK_NOW, **_spy_gates_kwargs())
    assert eg.DAILY_BUDGET_VETO in d["veto_reasons"]
    assert d["execution_allowed"] is False
    assert d["daily_trade_budget"]["exhausted"] is True


def test_post_trade_cooldown_vetoes_immediate_reentry():
    events = _red_trade_events(_LOCK_NOW, n=1, minutes_since_close=1.0)
    d = eg.build_entry_gate_decision(journal_events=events, now=_LOCK_NOW, **_spy_gates_kwargs())
    assert eg.COOLDOWN_VETO in d["veto_reasons"]
    assert d["execution_allowed"] is False


def test_budget_allows_second_entry_after_cooldown():
    import data.odte_config as oc
    assert oc.DAILY_TRADE_BUDGET >= 2, "retune contract: at least 2 trades/day"
    events = _red_trade_events(_LOCK_NOW, n=1,
                               minutes_since_close=oc.REENTRY_COOLDOWN_MINUTES + 5.0)
    d = eg.build_entry_gate_decision(journal_events=events, now=_LOCK_NOW, **_spy_gates_kwargs())
    assert eg.DAILY_BUDGET_VETO not in d["veto_reasons"]
    assert eg.COOLDOWN_VETO not in d["veto_reasons"]
    assert d["execution_allowed"] is True
    assert d["daily_trade_budget"]["remaining"] >= 1


# --- green re-entry AUTO-ARM (2026-08-03: same-or-better tier + budget + cooldown + BP) ---------

def _auto_arm_kwargs(tier="full", ask=1.2):
    kw = _spy_gates_kwargs()
    kw["candidate"]["tier"] = tier
    kw["vehicle_score"] = {**kw["vehicle_score"],
                           "contract": {**kw["vehicle_score"]["contract"], "ask": ask}}
    return kw


def test_green_reentry_auto_arms_on_same_or_better_tier():
    # Winning trade carries no tier (legacy journal) -> ranks "full"; a "full" candidate with a
    # budget slot, cooldown clear, and BP >= 1.5x cost auto-arms WITHOUT any manual flag.
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW,
                                     **_auto_arm_kwargs(tier="full"))
    assert d["execution_allowed"] is True
    assert "green_reentry_auto_armed_tier" in d["reason_codes"]
    assert d["green_reentry"]["auto_armed"] is True
    assert d["green_reentry"]["manual_override"] is False
    assert d["green_reentry"]["winning_tier_today"] == "full"
    assert d["allow_reentry_after_green"] is True
    # ...and the journal event carries the survival marker + tier.
    ev = oj.event_from_entry_gate(d)
    assert ev["allow_reentry_after_green"] is True and ev["tier"] == "full"


def test_green_reentry_below_winner_tier_stays_locked():
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW,
                                     **_auto_arm_kwargs(tier="b_plus"))
    assert d["execution_allowed"] is False
    assert eg.GREEN_LOCKOUT_VETO in d["veto_reasons"]
    assert "green_reentry_tier_below_winner" in d["reason_codes"]


def test_green_reentry_auto_arm_kill_switch(monkeypatch):
    monkeypatch.setattr(eg, "GREEN_REENTRY_AUTO_ARM", False)
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW,
                                     **_auto_arm_kwargs(tier="a_plus"))
    assert d["execution_allowed"] is False
    assert eg.GREEN_LOCKOUT_VETO in d["veto_reasons"]
    assert d["green_reentry"]["auto_arm_enabled"] is False


def test_green_reentry_auto_arm_blocked_by_budget_and_cooldown():
    import data.odte_config as oc
    # Budget exhausted: seed DAILY_TRADE_BUDGET completed green trades.
    events = []
    for i in range(oc.DAILY_TRADE_BUDGET):
        base = {"trade_id": f"t{i}", "underlying": "SPY"}
        events += [{**base, "event_type": "order_filled", "ts": _ts_hours_ago(3.0 - i * 0.1)},
                   {**base, "event_type": "order_closed", "realized_pnl": 5.0,
                    "ts": _ts_hours_ago(2.0 - i * 0.1)}]
    d = eg.build_entry_gate_decision(journal_events=events, now=_LOCK_NOW,
                                     **_auto_arm_kwargs(tier="a_plus"))
    assert d["execution_allowed"] is False
    assert eg.DAILY_BUDGET_VETO in d["veto_reasons"]
    assert d["green_reentry"]["auto_armed"] is False
    # Cooldown active: close 1 minute ago.
    hot = _green_scalp_events(hours_ago=1 / 60)
    d2 = eg.build_entry_gate_decision(journal_events=hot, now=_LOCK_NOW,
                                      **_auto_arm_kwargs(tier="a_plus"))
    assert d2["execution_allowed"] is False
    assert eg.COOLDOWN_VETO in d2["veto_reasons"]


def test_green_reentry_auto_arm_bp_short_names_bp_veto():
    # Everything qualifies except BP (ask 7.0 -> needs 1.5x$700=$1050 vs $250): the honest blocker.
    d = eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW,
                                     **_auto_arm_kwargs(tier="a_plus", ask=7.0))
    assert d["execution_allowed"] is False
    assert eg.GREEN_REENTRY_BP_VETO in d["veto_reasons"]


def _ts_hours_ago(hours):
    return (_LOCK_NOW - timedelta(hours=hours)).isoformat()


def test_no_next_command_points_at_retired_authorize_ladder():
    # 2026-08-03 sweep: no gate outcome may steer the agent at the retired multi-command ladder.
    scenarios = [
        eg.build_entry_gate_decision(**_all_gates_kwargs()),                       # enter
        eg.build_entry_gate_decision(scan_only=True, **_all_gates_kwargs()),       # scan-tier green
        eg.build_entry_gate_decision(candidate={"ticker": "QQQ", "direction": "bullish"}),  # missing
        eg.build_entry_gate_decision(journal_events=_green_scalp_events(), now=_LOCK_NOW,
                                     **_spy_gates_kwargs()),                       # green veto
        eg.build_entry_gate_decision(trigger={"scan_only": True, "candidate": dict(_CANDIDATE)},
                                     day_score=dict(_GOOD_DAY), vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER),
                                     promote_to_execution=True),                   # deprecated flag
    ]
    for d in scenarios:
        assert "odte-execution-authorize" not in (d.get("next_command") or ""), d.get("next_command")


def test_stale_confirm_transition_steers_to_odte_convert():
    # Cycle-3 regression (2026-08-03): a stale CONFIRM transition produced next_command
    # odte-execution-authorize and the agent narrated the confirm as terminal. It must now say
    # odte-convert and NOT-terminal. (Pure-function path — the CLI wrapper refuses outright.)
    scan_candidate = {**_CANDIDATE, "scan_only": True}
    stale_cd = {"decision": "CONFIRM_ENTRY",
                "generated_at": (datetime(2026, 7, 27, 15, 34, 38, tzinfo=timezone.utc)
                                 - timedelta(seconds=eg.CONFIRMED_CANDIDATE_MAX_AGE_SECONDS + 16)
                                 ).isoformat(),
                "candidate": dict(scan_candidate)}
    d = eg.build_entry_gate_decision(
        candidate=dict(scan_candidate), candidate_decision=stale_cd,
        day_score=dict(_GOOD_DAY), vehicle_score=dict(_GOOD_VEHICLE),
        broker_snapshot=dict(_GOOD_BROKER), confirmations=dict(_CONFIRMATIONS),
        now=datetime(2026, 7, 27, 15, 34, 38, tzinfo=timezone.utc))
    assert "candidate_decision_stale_for_transition" in d["reason_codes"]
    assert d["execution_allowed"] is False
    assert d["next_command"] == "odte-convert"
    assert "NOT terminal" in d["next_action"]
