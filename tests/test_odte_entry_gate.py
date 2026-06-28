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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_entry_gate as eg
import data.odte_journal as oj


def _journal(tmp_path):
    return str(tmp_path / "decision_journal.jsonl")


# All gates explicitly positive — the only shape that may authorize execution.
_GOOD_DAY = {"verdict": "GOOD_DAY", "score": 5}
_GOOD_VEHICLE = {"verdict": "GOOD_BET", "score": 6, "direction": "bullish",
                 "contract": {"underlying": "QQQ", "option_type": "call", "strike": 718},
                 "reasons": ["market: VWAP confirms calls on SPY,QQQ", "gamma: low pin risk"]}
_GOOD_BROKER = {"buying_power": 250.0, "day_trades_left": 3}
_CANDIDATE = {"ticker": "QQQ", "direction": "bullish"}


def _all_gates_kwargs():
    return dict(candidate=dict(_CANDIDATE), day_score=dict(_GOOD_DAY),
                vehicle_score=dict(_GOOD_VEHICLE), broker_snapshot=dict(_GOOD_BROKER))


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


def test_promote_to_execution_allows_inherited_scan_only_trigger_to_enter():
    # The manager EXPLICITLY promotes the same scan_only trigger; with all gates good it may enter.
    trigger = {"scan_only": True, "execution_allowed": False,
               "candidate": dict(_CANDIDATE)}
    d = eg.build_entry_gate_decision(trigger=trigger, day_score=dict(_GOOD_DAY),
                                     vehicle_score=dict(_GOOD_VEHICLE),
                                     broker_snapshot=dict(_GOOD_BROKER),
                                     promote_to_execution=True)
    assert d["scan_only"] is False and d["promoted_to_execution"] is True
    assert d["execution_allowed"] is True and d["decision"] == "enter"
    assert "scan_only_promoted_to_execution" in d["reason_codes"]


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
    # ...and the explicit promote opt-in flips it.
    d2 = eg.run_entry_gate(
        trigger_json=json.dumps({"scan_only": True, "candidate": _CANDIDATE}),
        day_score_json=json.dumps(_GOOD_DAY),
        vehicle_score_json=json.dumps(_GOOD_VEHICLE),
        broker_json=json.dumps(_GOOD_BROKER), promote_to_execution=True)
    assert d2["scan_only"] is False and d2["execution_allowed"] is True


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


# --- guardrail: no broker / network / LLM -------------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(eg)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket"):
        assert forbidden not in src, f"odte_entry_gate must not reference {forbidden!r}"
