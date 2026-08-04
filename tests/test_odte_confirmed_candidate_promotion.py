"""Regression coverage for the safe confirmed-candidate promotion seam.

Pure/offline: proves a fresh identity-matched CONFIRM_ENTRY can build the non-scan gate required by
the execution lease, while stale/mismatched/bare promotion paths remain fail-closed.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from cli.main import _cmd_odte_entry_gate
from data import odte_entry_gate as eg
from data import odte_execution_policy as xp
from data import odte_journal as oj
from data import odte_loop_status as ls

NOW = datetime(2026, 7, 27, 15, 34, 38, tzinfo=timezone.utc)


def _candidate() -> dict:
    candidate = {
        "ticker": "IWM",
        "direction": "bearish",
        "selected_vehicle": "IWM",
        "selection_timestamp": "2026-07-27T15:32:38+00:00",
        "expiration": "2026-07-27",
        "strike": 292.0,
        "option_type": "put",
        "option_id": "iwm-20260727-292p",
        "scan_only": True,
        # Chase-band anchor, stamped by candidate-watch at CONFIRM_ENTRY (2026-08-02 retune).
        "anchor_quote": 0.70,
    }
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    return candidate


def _candidate_decision(*, generated_at: datetime = NOW, candidate: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "generated_at": generated_at.isoformat(),
        "decision": "CONFIRM_ENTRY",
        "scan_only": True,
        "execution_allowed": False,
        "candidate": candidate or _candidate(),
    }


def _vehicle(*, ask: float = 0.70) -> dict:
    return {
        "schema_version": 1,
        "generated_at": (NOW - timedelta(seconds=1)).isoformat(),
        "verdict": "GOOD_BET",
        "direction": "bearish",
        "contract": {
            "option_id": "iwm-20260727-292p",
            "underlying": "IWM",
            "option_type": "put",
            "expiration_date": "2026-07-27",
            "strike_price": 292.0,
            "bid_price": ask - 0.01,
            "ask_price": ask,
            "mark_price": ask - 0.005,
        },
    }


def _broker() -> dict:
    return {
        "generated_at": NOW.isoformat(),
        "buying_power": 348.16,
        "day_trades_left": 3,
        "nonzero_option_positions_count": 0,
        "open_option_orders_count": 0,
        "today_option_orders_count": 0,
    }


def _market() -> dict:
    return {
        "generated_at": NOW.isoformat(),
        "IWM": {"last": 292.11, "above_vwap": False, "orb_state": "below"},
        "SPY": {"last": 737.7, "above_vwap": False, "orb_state": "below"},
        "QQQ": {"last": 678.3, "above_vwap": False, "orb_state": "below"},
        "VIXY": {"last": 21.55, "above_vwap": True},
    }


def _confirmations() -> dict:
    return {name: True for name in eg.REQUIRED_CONFIRMATIONS}


def _gate(*, candidate: dict | None = None, candidate_decision: dict | None = None,
          confirmations: dict | None = None) -> dict:
    return eg.build_entry_gate_decision(
        candidate=candidate or _candidate(),
        candidate_decision=candidate_decision or _candidate_decision(),
        day_score={"verdict": "GOOD_DAY"},
        vehicle_score=_vehicle(),
        broker_snapshot=_broker(),
        confirmations=confirmations if confirmations is not None else _confirmations(),
        now=NOW,
    )


def test_fresh_matching_confirmed_candidate_crosses_scan_to_gate_and_leases() -> None:
    gate = _gate()
    assert gate["confirmed_candidate_transition"] is True
    assert gate["scan_only"] is False
    assert gate["execution_allowed"] is True
    assert gate["confirmations"] == _confirmations()
    assert "confirmed_candidate_transition" in gate["reason_codes"]

    result = xp.authorize_entry(
        gate=gate,
        candidate_decision=_candidate_decision(),
        vehicle_score=_vehicle(),
        broker_snapshot=_broker(),
        market_snapshot=_market(),
        now=NOW,
        policy={"quantity": 1, "limit_price": 0.70},
    )
    assert result["authorized"] is True, result["reason_codes"]
    # 2026-08-02 retune: the lease ceiling extends to the chase band above the 0.70 anchor
    # (still capped by the tier's BP fraction; see test_odte_execution_policy for the cap case).
    ceiling = round(0.70 * (1 + xp.CHASE_BAND_FRACTION), 2)
    assert result["lease"]["max_limit_price"] == ceiling
    assert result["lease"]["max_debit"] == round(ceiling * 100.0, 2)
    assert result["lease"]["anchor_quote"] == 0.70


def test_stale_confirmed_candidate_cannot_cross_scan_boundary() -> None:
    stale = _candidate_decision(
        generated_at=NOW - timedelta(seconds=eg.CONFIRMED_CANDIDATE_MAX_AGE_SECONDS + 1)
    )
    gate = _gate(candidate_decision=stale)
    assert gate["confirmed_candidate_transition"] is False
    assert gate["scan_only"] is True
    assert gate["execution_allowed"] is False
    assert "candidate_decision_stale_for_transition" in gate["reason_codes"]


def test_mismatched_candidate_cannot_cross_scan_boundary() -> None:
    other = _candidate()
    other.update({"ticker": "QQQ", "selected_vehicle": "QQQ"})
    other["candidate_fingerprint"] = xp.candidate_fingerprint(other)
    gate = _gate(candidate=other, candidate_decision=_candidate_decision())
    assert gate["confirmed_candidate_transition"] is False
    assert gate["scan_only"] is True
    assert gate["execution_allowed"] is False
    assert "candidate_fingerprint_mismatch" in gate["reason_codes"]
    assert "candidate_symbol_mismatch" in gate["reason_codes"]


def test_tampered_candidate_fields_cannot_hide_behind_copied_fingerprint() -> None:
    original = _candidate()
    decision = _candidate_decision(candidate=original)
    tampered = dict(original)
    tampered["selection_timestamp"] = (NOW - timedelta(seconds=20)).isoformat()
    # Deliberately retain the original fingerprint: the gate must recompute, not trust this claim.
    tampered["candidate_fingerprint"] = original["candidate_fingerprint"]
    decision["candidate"] = tampered
    gate = _gate(candidate_decision=decision)
    assert gate["scan_only"] is True
    assert gate["execution_allowed"] is False
    assert "candidate_decision_fingerprint_invalid" in gate["reason_codes"]
    assert "candidate_fingerprint_mismatch" in gate["reason_codes"]


def test_contract_switch_cannot_hide_behind_copied_fingerprint() -> None:
    original = _candidate()
    switched = dict(original)
    switched["option_id"] = "iwm-20260727-291p"
    switched["strike"] = 291.0
    switched["candidate_fingerprint"] = original["candidate_fingerprint"]
    decision = _candidate_decision(candidate=switched)
    gate = _gate(candidate_decision=decision)
    assert gate["scan_only"] is True
    assert gate["execution_allowed"] is False
    assert "candidate_decision_fingerprint_invalid" in gate["reason_codes"]
    assert "candidate_contract_option_id_mismatch" in gate["reason_codes"]
    assert "candidate_contract_strike_price_mismatch" in gate["reason_codes"]


def test_missing_candidate_cycle_timestamp_cannot_cross_boundary() -> None:
    undated = _candidate()
    undated.pop("selection_timestamp")
    undated["candidate_fingerprint"] = xp.candidate_fingerprint(undated)
    gate = _gate(candidate=undated, candidate_decision=_candidate_decision(candidate=undated))
    assert gate["scan_only"] is True
    assert "candidate_cycle_timestamp_missing" in gate["reason_codes"]


def test_non_scan_gate_without_final_confirmations_is_not_executable() -> None:
    candidate = {**_candidate(), "scan_only": False}
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    gate = eg.build_entry_gate_decision(
        candidate=candidate,
        day_score={"verdict": "GOOD_DAY"},
        vehicle_score=_vehicle(),
        broker_snapshot=_broker(),
        confirmations={},
        now=NOW,
    )
    assert gate["scan_only"] is False
    assert gate["execution_allowed"] is False
    assert gate["confirmation_needed"] is True
    assert "final_confirmation_live_chain_recheck_missing" in gate["veto_reasons"]


def test_confirm_entry_is_rechecked_at_lease_time_with_sixty_second_ttl() -> None:
    later = NOW + timedelta(seconds=xp.INPUT_TTLS_SECONDS["candidate_decision"] + 1)
    gate = _gate()
    gate["generated_at"] = later.isoformat()
    vehicle = _vehicle()
    vehicle["generated_at"] = later.isoformat()
    broker = _broker()
    broker["generated_at"] = later.isoformat()
    market = _market()
    market["generated_at"] = later.isoformat()
    denied = xp.authorize_entry(
        gate=gate,
        candidate_decision=_candidate_decision(generated_at=NOW),
        vehicle_score=vehicle,
        broker_snapshot=broker,
        market_snapshot=market,
        now=later,
        policy={"quantity": 1, "limit_price": 0.70},
    )
    assert denied["authorized"] is False
    assert "candidate_decision_stale" in denied["reason_codes"]


def test_future_dated_confirm_entry_cannot_cross_boundary() -> None:
    future = _candidate_decision(
        generated_at=NOW + timedelta(seconds=eg.CONFIRMED_CANDIDATE_MAX_FUTURE_SKEW_SECONDS + 1)
    )
    gate = _gate(candidate_decision=future)
    assert gate["scan_only"] is True
    assert gate["execution_allowed"] is False
    assert "candidate_decision_from_future" in gate["reason_codes"]


def test_missing_broker_truth_counts_block_gate_and_lease() -> None:
    broker = _broker()
    broker.pop("open_option_orders_count")
    candidate = {**_candidate(), "scan_only": False}
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    gate = eg.build_entry_gate_decision(
        candidate=candidate,
        day_score={"verdict": "GOOD_DAY"},
        vehicle_score=_vehicle(),
        broker_snapshot=broker,
        confirmations=_confirmations(),
        now=NOW,
    )
    assert gate["execution_allowed"] is False
    assert gate["gates"]["account"] is None
    assert "broker_open_orders_count_missing" in gate["reason_codes"]

    denied = xp.authorize_entry(
        gate=_gate(),
        candidate_decision=_candidate_decision(),
        vehicle_score=_vehicle(),
        broker_snapshot=broker,
        market_snapshot=_market(),
        now=NOW,
        policy={"quantity": 1, "limit_price": 0.70},
    )
    assert denied["authorized"] is False
    assert "broker_open_orders_count_missing" in denied["reason_codes"]


def test_explicit_scan_only_remains_a_final_veto_even_with_confirmation() -> None:
    gate = eg.build_entry_gate_decision(
        candidate=_candidate(),
        candidate_decision=_candidate_decision(),
        day_score={"verdict": "GOOD_DAY"},
        vehicle_score=_vehicle(),
        broker_snapshot=_broker(),
        confirmations=_confirmations(),
        scan_only=True,
        now=NOW,
    )
    assert gate["confirmed_candidate_transition"] is False
    assert gate["scan_only"] is True
    assert gate["execution_allowed"] is False


def test_run_entry_gate_json_wrapper_wires_confirmed_transition_inputs() -> None:
    gate = eg.run_entry_gate(
        candidate_json=json.dumps(_candidate()),
        candidate_decision_json=json.dumps(_candidate_decision()),
        day_score_json=json.dumps({"verdict": "GOOD_DAY"}),
        vehicle_score_json=json.dumps(_vehicle()),
        broker_json=json.dumps(_broker()),
        confirmations_json=json.dumps(_confirmations()),
        now=NOW,
    )
    assert gate["confirmed_candidate_transition"] is True
    assert gate["scan_only"] is False
    assert gate["execution_allowed"] is True


def test_cli_journaled_gate_is_visible_to_loop_status(tmp_path, capsys) -> None:
    current = datetime.now(timezone.utc).replace(microsecond=0)
    candidate = _candidate()
    candidate["selection_timestamp"] = current.isoformat()
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    decision = _candidate_decision(generated_at=current, candidate=candidate)
    vehicle = _vehicle()
    vehicle["generated_at"] = current.isoformat()
    broker = _broker()
    broker["generated_at"] = current.isoformat()
    journal_path = tmp_path / "decision_journal.jsonl"

    _cmd_odte_entry_gate([
        "--candidate-json", json.dumps(candidate),
        "--candidate-decision-json", json.dumps(decision),
        "--day-score-json", json.dumps({"verdict": "GOOD_DAY"}),
        "--vehicle-score-json", json.dumps(vehicle),
        "--broker-json", json.dumps(broker),
        "--confirmations-json", json.dumps(_confirmations()),
        "--journal", "--journal-path", str(journal_path), "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_allowed"] is True
    assert payload["journal_append"]["status"] == "appended"

    events = oj.read_events(str(journal_path))
    loop = ls.derive_loop_state(journal_events=events, now=current)
    assert loop["state"] == "PROMOTED"
    assert loop["posture"] == "EXECUTION_READY"


def test_cli_journal_append_failure_exits_nonzero(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        oj,
        "append_decision_journal",
        lambda *args, **kwargs: {"status": "error", "event_id": None},
    )
    with pytest.raises(SystemExit) as exc:
        _cmd_odte_entry_gate([
            "--candidate-json", "{}",
            "--journal", "--journal-path", str(tmp_path / "decision_journal.jsonl"), "--json",
        ])
    assert exc.value.code == 2
    assert "journal append failed" in capsys.readouterr().err


def test_missing_live_confirmations_block_gate_and_lease_after_safe_transition() -> None:
    gate = _gate(confirmations={})
    assert gate["scan_only"] is False
    assert gate["confirmed_candidate_transition"] is True
    assert gate["execution_allowed"] is False
    assert gate["decision"] == "veto"
    assert "final_confirmation_live_chain_recheck_missing" in gate["veto_reasons"]
    denied = xp.authorize_entry(
        gate=gate,
        candidate_decision=_candidate_decision(),
        vehicle_score=_vehicle(),
        broker_snapshot=_broker(),
        market_snapshot=_market(),
        now=NOW,
        policy={"quantity": 1, "limit_price": 0.70},
    )
    assert denied["authorized"] is False
    assert "gate_not_execution_allowed" in denied["reason_codes"]
    assert all(
        f"confirmation_not_boolean:{name}" in denied["reason_codes"]
        for name in eg.REQUIRED_CONFIRMATIONS
    )
