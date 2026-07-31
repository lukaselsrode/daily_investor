"""tests/test_odte_execution_policy.py — execution-authorization lease (2026-07-23 remediation).

Pure/offline: no broker, network, LLM, or orders. Pins the structural invariants that make a repeat
of the 2026-07-23 delayed-fill loss impossible:
  * the replay: an authorization derived from the 11:11 signal is EXPIRED before the 11:15 fill;
  * the incident's 2-contract / 83.9%-of-BP order is rejected under the DEFAULT policy;
  * bare promotion (scan-tier gate) can never lease;
  * exact identity, freshness, explicit-boolean confirmations, broker truth all fail closed;
  * leases are short-lived (TTL clamped), single-use, and bound to one candidate fingerprint;
  * FULL_ACCOUNT_A_PLUS needs the complete management plan — an A+ label alone never sets size.

Thresholds/TTLs are imported from the live module — never re-hardcoded.
"""
import inspect
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_entry_gate as eg
import data.odte_execution_policy as xp

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "odte" / "2026-07-23-delayed-fill.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


FX = _fixture()
PROMOTION_AT = _ts(FX["timeline"]["promotion_at"])
SUBMITTED_AT = _ts(FX["timeline"]["order_submitted_at"])
FILLED_AT = _ts(FX["timeline"]["order_filled_at"])


def _package(now: datetime, *, sym: str | None = None, direction: str | None = None,
             contract: dict | None = None) -> dict:
    """A fresh, all-matching authorization package built from the sanitized incident fixture."""
    fx = _fixture()
    sym = sym or fx["candidate"]["ticker"]
    direction = direction or fx["candidate"]["direction"]
    contract = contract if contract is not None else dict(fx["contract"])
    cand = {
        **fx["candidate"],
        "ticker": sym,
        "direction": direction,
        "selected_vehicle": sym,
        "selection_timestamp": _iso(now - timedelta(seconds=10)),
        "option_id": contract.get("option_id"),
        "expiration_date": contract.get("expiration_date"),
        "strike_price": contract.get("strike_price"),
        "option_type": contract.get("option_type"),
    }
    cand["candidate_fingerprint"] = xp.candidate_fingerprint(cand)
    gate = {"generated_at": _iso(now), "symbol": sym, "direction": direction,
            "candidate_fingerprint": cand["candidate_fingerprint"],
            "option_id": contract.get("option_id"),
            "expiration_date": contract.get("expiration_date"),
            "strike_price": contract.get("strike_price"),
            "option_type": contract.get("option_type"),
            "scan_only": False, "execution_allowed": True,
            "required_confirmations": list(xp.DEFAULT_REQUIRED_CONFIRMATIONS),
            "confirmations": {c: True for c in xp.DEFAULT_REQUIRED_CONFIRMATIONS}}
    cd = {"decision": "CONFIRM_ENTRY", "generated_at": _iso(now - timedelta(seconds=10)),
          "candidate": cand}
    vs = {"verdict": "GOOD_BET", "direction": direction,
          "generated_at": _iso(now - timedelta(seconds=8)), "contract": contract}
    broker = {**fx["broker_snapshot"], "as_of": _iso(now - timedelta(seconds=5))}
    market = {**fx["tape"]["at_trigger"], "as_of": _iso(now - timedelta(seconds=3))}
    return {"gate": gate, "candidate_decision": cd, "vehicle_score": vs,
            "broker_snapshot": broker, "market_snapshot": market}


def _full_account_policy(fx: dict | None = None, **over) -> dict:
    """The explicit FULL_ACCOUNT_A_PLUS policy the incident-size order would have needed."""
    fx = fx or _fixture()
    policy = {
        "risk_mode": "FULL_ACCOUNT_A_PLUS",
        "quantity": fx["order"]["quantity"],
        "limit_price": fx["order"]["limit_price"],
        "management_plan": {
            "trigger": "SPY momentum extension below ORB low with VIXY firming",
            "invalidation": fx["invalidation"]["description"],
            "target": "put +40% or SPY flush to 735.5",
            "scratch_rail": "scratch at -10% if SPY basing above 736.5",
            "management_cadence_seconds": 10,
            "max_premium_loss": fx["order"]["debit"],
        },
    }
    policy.update(over)
    return policy


# --- Task 1: deterministic replay of the incident ------------------------------------------------

def test_replay_lease_from_1111_signal_expires_before_1115_fill():
    # A lease minted at the 11:11:17 ET promotion moment — even for the exact incident order under
    # an explicit FULL_ACCOUNT_A_PLUS policy — expires long before the 11:15:35 ET fill.
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy=_full_account_policy())
    assert res["authorized"] is True, res["reason_codes"]
    lease = res["lease"]
    expires = _ts(lease["expires_at"])
    assert (expires - PROMOTION_AT).total_seconds() == xp.DEFAULT_LEASE_TTL_SECONDS
    assert expires < FILLED_AT, "the 11:11 authorization must be dead before the 11:15 fill"
    assert expires < SUBMITTED_AT, "it is dead even before the 11:13 submission"
    assert xp.lease_expired(lease, FILLED_AT) is True
    assert xp.lease_expired(lease, SUBMITTED_AT) is True
    # Even at the MAXIMUM permitted TTL the lease cannot reach the fill.
    res_max = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                                 policy=_full_account_policy(ttl_seconds=10_000))
    assert _ts(res_max["lease"]["expires_at"]) < FILLED_AT


def test_incident_two_contract_order_rejected_under_default_policy():
    # 2 contracts × $1.68 = $336 = 83.9% of $400.34 BP: rejected on BOTH default caps.
    fx = _fixture()
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy={"quantity": fx["order"]["quantity"],
                                     "limit_price": fx["order"]["limit_price"]})
    assert res["authorized"] is False
    assert "quantity_exceeds_policy" in res["reason_codes"]
    assert "debit_exceeds_policy" in res["reason_codes"]
    assert res["lease"] is None
    # Live-constant sanity: the incident debit really exceeds the default fraction of the real BP.
    assert fx["order"]["debit"] > xp.DEFAULT_MAX_DEBIT_FRACTION * fx["broker_snapshot"]["buying_power"]
    assert fx["order"]["quantity"] > xp.DEFAULT_MAX_CONTRACTS


# --- TTL policy ----------------------------------------------------------------------------------

def test_ttl_configurable_downward_never_above_max():
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy={"ttl_seconds": 5})
    lease = res["lease"]
    assert (_ts(lease["expires_at"]) - _ts(lease["issued_at"])).total_seconds() == 5
    res_up = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                                policy={"ttl_seconds": 10_000})
    lease_up = res_up["lease"]
    assert ((_ts(lease_up["expires_at"]) - _ts(lease_up["issued_at"])).total_seconds()
            == xp.MAX_LEASE_TTL_SECONDS)


def test_lease_expires_exactly_at_boundary():
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    lease = res["lease"]
    boundary = _ts(lease["expires_at"])
    assert xp.lease_expired(lease, boundary - timedelta(seconds=1)) is False
    assert xp.lease_expired(lease, boundary) is True, "boundary-inclusive: exactly expired IS expired"
    assert xp.lease_seconds_remaining(lease, boundary) == 0.0


# --- bare promotion stays non-executable ----------------------------------------------------------

def test_bare_promoted_scan_gate_cannot_lease():
    # The exact incident hole: a scan_only watchdog record run through the (deprecated) promote
    # flag. The gate itself now stays non-executable, and authorize_entry refuses it on top.
    gate = eg.build_entry_gate_decision(
        trigger={"scan_only": True, "candidate": {"ticker": "SPY", "direction": "bearish"}},
        day_score={"verdict": "GOOD_DAY"},
        vehicle_score={"verdict": "GOOD_BET", "direction": "bearish",
                       "contract": dict(FX["contract"])},
        broker_snapshot={"buying_power": 400.34, "day_trades_left": 3},
        promote_to_execution=True, now=PROMOTION_AT)
    assert gate["scan_only"] is True and gate["execution_allowed"] is False
    pkg = _package(PROMOTION_AT)
    pkg["gate"] = gate
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False
    assert "gate_scan_only_not_false" in res["reason_codes"]
    assert "gate_not_execution_allowed" in res["reason_codes"]


# --- freshness + confirmation fail-closed ---------------------------------------------------------

def test_missing_or_stale_candidate_confirmation_rejected():
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"] = {**pkg["candidate_decision"], "decision": "KEEP_WATCHING"}
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False and "candidate_not_confirmed" in res["reason_codes"]

    stale_by = timedelta(seconds=xp.INPUT_TTLS_SECONDS["candidate_decision"] + 1)
    pkg2 = _package(PROMOTION_AT)
    pkg2["candidate_decision"]["generated_at"] = _iso(PROMOTION_AT - stale_by)
    res2 = xp.authorize_entry(**pkg2, now=PROMOTION_AT, policy={})
    assert "candidate_decision_stale" in res2["reason_codes"]

    pkg3 = _package(PROMOTION_AT)
    pkg3["candidate_decision"].pop("generated_at")
    res3 = xp.authorize_entry(**pkg3, now=PROMOTION_AT, policy={})
    assert "candidate_decision_undated" in res3["reason_codes"]


def test_every_input_must_be_fresh_within_its_own_ttl():
    for name, key in (("entry_gate", "gate"), ("vehicle_score", "vehicle_score"),
                      ("market_snapshot", "market_snapshot"), ("broker_snapshot", "broker_snapshot")):
        pkg = _package(PROMOTION_AT)
        stale_by = timedelta(seconds=xp.INPUT_TTLS_SECONDS[name] + 1)
        payload = pkg[key]
        for ts_key in ("generated_at", "as_of", "ts"):
            payload.pop(ts_key, None)
        payload["generated_at"] = _iso(PROMOTION_AT - stale_by)
        res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
        assert res["authorized"] is False
        assert f"{name}_stale" in res["reason_codes"], (name, res["reason_codes"])


def test_future_dated_input_artifacts_fail_closed() -> None:
    for name, key in (("entry_gate", "gate"), ("candidate_decision", "candidate_decision"),
                      ("vehicle_score", "vehicle_score"), ("market_snapshot", "market_snapshot"),
                      ("broker_snapshot", "broker_snapshot")):
        pkg = _package(PROMOTION_AT)
        payload = pkg[key]
        for ts_key in ("generated_at", "as_of", "ts"):
            payload.pop(ts_key, None)
        payload["generated_at"] = _iso(
            PROMOTION_AT + timedelta(seconds=xp.MAX_FUTURE_SKEW_SECONDS + 1)
        )
        res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
        assert res["authorized"] is False
        assert f"{name}_from_future" in res["reason_codes"], (name, res["reason_codes"])


def test_confirmations_must_be_explicit_booleans():
    pkg = _package(PROMOTION_AT)
    pkg["gate"]["confirmations"]["budget_check"] = "true"      # a string is NOT a confirmation
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert "confirmation_not_boolean:budget_check" in res["reason_codes"]

    pkg2 = _package(PROMOTION_AT)
    del pkg2["gate"]["confirmations"]["spread_cap_check"]      # missing fails closed
    res2 = xp.authorize_entry(**pkg2, now=PROMOTION_AT, policy={})
    assert "confirmation_not_boolean:spread_cap_check" in res2["reason_codes"]

    pkg3 = _package(PROMOTION_AT)
    pkg3["gate"]["confirmations"]["live_chain_recheck"] = False
    res3 = xp.authorize_entry(**pkg3, now=PROMOTION_AT, policy={})
    assert "confirmation_failed:live_chain_recheck" in res3["reason_codes"]

    pkg4 = _package(PROMOTION_AT)
    pkg4["gate"]["required_confirmations"] = ["budget_check"]
    res4 = xp.authorize_entry(**pkg4, now=PROMOTION_AT, policy={})
    assert "confirmation_schema_incomplete" in res4["reason_codes"]


# --- exact identity binding -----------------------------------------------------------------------

def test_candidate_and_gate_must_bind_exact_option_contract() -> None:
    for container, field, reason in (
        ("candidate", "option_id", "candidate_option_id_missing"),
        ("candidate", "expiration_date", "candidate_expiration_missing"),
        ("candidate", "strike_price", "candidate_strike_missing"),
        ("candidate", "option_type", "candidate_option_type_missing"),
        ("gate", "option_id", "gate_option_id_missing"),
        ("gate", "expiration_date", "gate_expiration_date_missing"),
        ("gate", "strike_price", "gate_strike_price_missing"),
        ("gate", "option_type", "gate_option_type_missing"),
    ):
        pkg = _package(PROMOTION_AT)
        target = (pkg["candidate_decision"]["candidate"]
                  if container == "candidate" else pkg["gate"])
        target.pop(field)
        res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
        assert res["authorized"] is False
        assert reason in res["reason_codes"], (container, field, res["reason_codes"])


def test_missing_candidate_cycle_timestamp_rejected_at_authorization() -> None:
    pkg = _package(PROMOTION_AT)
    candidate = pkg["candidate_decision"]["candidate"]
    for field in ("selection_timestamp", "created_at", "ts", "generated_at"):
        candidate.pop(field, None)
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    pkg["gate"]["candidate_fingerprint"] = candidate["candidate_fingerprint"]
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False
    assert "candidate_cycle_timestamp_missing" in res["reason_codes"]


def test_symbol_mismatch_rejected_qqq_thesis_cannot_lease_spy_contract():
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"]["candidate"]["ticker"] = "QQQ"   # thesis says QQQ, contract is SPY
    pkg["gate"]["symbol"] = "QQQ"
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False and "symbol_mismatch" in res["reason_codes"]


def test_direction_and_option_type_mismatches_rejected():
    pkg = _package(PROMOTION_AT)
    pkg["gate"]["direction"] = "bullish"                       # gate disagrees with candidate
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert "direction_mismatch" in res["reason_codes"]

    pkg2 = _package(PROMOTION_AT)
    pkg2["vehicle_score"]["contract"]["option_type"] = "call"  # a call cannot carry a bearish thesis
    pkg2["vehicle_score"]["direction"] = "bearish"
    res2 = xp.authorize_entry(**pkg2, now=PROMOTION_AT, policy={})
    assert "option_type_direction_mismatch" in res2["reason_codes"]


def test_option_id_strike_expiration_binding():
    pkg = _package(PROMOTION_AT)
    del pkg["vehicle_score"]["contract"]["option_id"]
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert "option_id_missing" in res["reason_codes"]

    pkg2 = _package(PROMOTION_AT)
    pkg2["candidate_decision"]["candidate"]["expiration_date"] = "2026-07-24"
    res2 = xp.authorize_entry(**pkg2, now=PROMOTION_AT, policy={})
    assert "expiration_mismatch" in res2["reason_codes"]

    pkg3 = _package(PROMOTION_AT)
    pkg3["candidate_decision"]["candidate"]["strike_price"] = 736.0
    res3 = xp.authorize_entry(**pkg3, now=PROMOTION_AT, policy={})
    assert "strike_mismatch" in res3["reason_codes"]


def test_restricted_underlying_rejected():
    contract = {**FX["contract"], "underlying": "NVDA"}
    pkg = _package(PROMOTION_AT, sym="NVDA", contract=contract)
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False and "restricted_underlying" in res["reason_codes"]


# --- broker truth prerequisites --------------------------------------------------------------------

def test_missing_buying_power_rejected():
    pkg = _package(PROMOTION_AT)
    del pkg["broker_snapshot"]["buying_power"]
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False and "buying_power_missing" in res["reason_codes"]


def test_missing_counts_open_orders_or_positions_fail_closed():
    pkg = _package(PROMOTION_AT)
    del pkg["broker_snapshot"]["open_option_orders_count"]
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert "broker_open_orders_count_missing" in res["reason_codes"]

    pkg2 = _package(PROMOTION_AT)
    pkg2["broker_snapshot"]["open_option_orders_count"] = 1
    res2 = xp.authorize_entry(**pkg2, now=PROMOTION_AT, policy={})
    assert "open_order_outstanding" in res2["reason_codes"]

    pkg3 = _package(PROMOTION_AT)
    pkg3["broker_snapshot"]["nonzero_option_positions_count"] = 1
    res3 = xp.authorize_entry(**pkg3, now=PROMOTION_AT, policy={})
    assert "position_already_open" in res3["reason_codes"]

    pkg4 = _package(PROMOTION_AT)
    pkg4["broker_snapshot"]["controller_locked"] = True
    res4 = xp.authorize_entry(**pkg4, now=PROMOTION_AT, policy={})
    assert "controller_locked" in res4["reason_codes"]


# --- a valid fresh all-matching package issues exactly one lease -----------------------------------

def test_valid_fresh_package_issues_lease_with_maximums_and_fingerprints():
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    assert res["authorized"] is True and res["reason_codes"] == []
    lease = res["lease"]
    fx = _fixture()
    assert lease["symbol"] == "SPY" and lease["direction"] == "bearish"
    assert lease["option_type"] == "put" and lease["option_id"] == fx["contract"]["option_id"]
    assert lease["strike_price"] == fx["contract"]["strike_price"]
    assert lease["expiration_date"] == fx["contract"]["expiration_date"]
    assert lease["quantity"] == xp.DEFAULT_MAX_CONTRACTS
    assert lease["max_limit_price"] == fx["contract"]["ask"]
    assert lease["max_debit"] == round(xp.DEFAULT_MAX_CONTRACTS * fx["contract"]["ask"] * 100, 2)
    assert lease["candidate_fingerprint"] and lease["market_fingerprint"]
    assert lease["risk_mode"] == xp.DEFAULT_RISK_MODE
    # Policy values are serialized into the payload (and journaled from there).
    assert res["policy"]["ttl_seconds"] == xp.DEFAULT_LEASE_TTL_SECONDS
    assert res["policy"]["max_debit_fraction"] == xp.DEFAULT_MAX_DEBIT_FRACTION
    assert res["places_orders"] is False


def test_partial_policy_can_only_tighten_debit_cap():
    # A caller-supplied fraction ABOVE the default is clamped back down (never widened).
    fx = _fixture()
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy={"quantity": 1, "limit_price": fx["order"]["limit_price"],
                                     "max_debit_fraction": 0.99})
    # 1 × 1.68 × 100 = $168 ≤ 50% of 400.34 — allowed; the 0.99 never applied as a widening.
    assert res["authorized"] is True
    tight = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                               policy={"quantity": 1, "limit_price": fx["order"]["limit_price"],
                                       "max_debit_fraction": 0.01})
    assert tight["authorized"] is False and "debit_exceeds_policy" in tight["reason_codes"]


# --- single-use consumption ------------------------------------------------------------------------

def test_consumed_lease_reuse_rejected():
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    lease = res["lease"]
    consumed: set[str] = set()
    just_after = PROMOTION_AT + timedelta(seconds=1)
    first = xp.consume_lease(lease, consumed_ids=consumed, now=just_after)
    assert first["status"] == "consumed"
    second = xp.consume_lease(lease, consumed_ids=consumed, now=just_after)
    assert second["status"] == "already_consumed"


def test_expired_lease_cannot_be_consumed():
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    late = PROMOTION_AT + timedelta(seconds=xp.DEFAULT_LEASE_TTL_SECONDS + 1)
    out = xp.consume_lease(res["lease"], consumed_ids=set(), now=late)
    assert out["status"] == "expired"


def test_consumed_ledger_roundtrip(tmp_path):
    ledger = tmp_path / "consumed_leases.json"
    assert xp.load_consumed_ids(ledger) == set()
    xp.record_consumed(ledger, "abc123")
    xp.record_consumed(ledger, "abc123")           # idempotent
    xp.record_consumed(ledger, "def456")
    assert xp.load_consumed_ids(ledger) == {"abc123", "def456"}


# --- Task 3: FULL_ACCOUNT_A_PLUS -------------------------------------------------------------------

def test_full_account_with_complete_plan_allows_incident_size():
    # Equivalent size to the incident IS allowed — but only under a still-fresh explicit
    # FULL_ACCOUNT_A_PLUS lease with the complete management plan.
    fx = _fixture()
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy=_full_account_policy())
    assert res["authorized"] is True, res["reason_codes"]
    lease = res["lease"]
    assert lease["risk_mode"] == "FULL_ACCOUNT_A_PLUS"
    assert lease["quantity"] == fx["order"]["quantity"]
    assert lease["max_debit"] == fx["order"]["debit"]
    assert lease["max_premium_loss"] == fx["order"]["debit"]
    assert res["policy"]["risk_mode"] == "FULL_ACCOUNT_A_PLUS"
    assert res["policy"]["max_premium_loss"] == fx["order"]["debit"]


def test_full_account_missing_any_management_field_rejected():
    for field in xp.FULL_ACCOUNT_PLAN_FIELDS:
        policy = _full_account_policy()
        del policy["management_plan"][field]
        res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy=policy)
        assert res["authorized"] is False, field
        assert f"full_account_missing:{field}" in res["reason_codes"], field


def test_full_account_debit_above_accepted_max_loss_rejected():
    policy = _full_account_policy()
    policy["management_plan"]["max_premium_loss"] = 100.0     # accepts less than the $336 debit
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy=policy)
    assert res["authorized"] is False
    assert "debit_exceeds_accepted_max_loss" in res["reason_codes"]


def test_a_plus_label_alone_never_sets_size():
    # A model-generated "A+" grade on the candidate/gate does NOT flip the risk mode: without an
    # explicit FULL_ACCOUNT_A_PLUS policy the default 1-contract/debit-cap policy still rejects.
    fx = _fixture()
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"]["candidate"]["grade"] = "A+"
    pkg["gate"]["grade"] = "A+"
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT,
                             policy={"quantity": fx["order"]["quantity"],
                                     "limit_price": fx["order"]["limit_price"]})
    assert res["authorized"] is False
    assert res["risk_mode"] == xp.DEFAULT_RISK_MODE
    assert "quantity_exceeds_policy" in res["reason_codes"]


def test_invalid_risk_mode_fails_closed():
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy={"risk_mode": "YOLO_FULL_SEND"})
    assert res["authorized"] is False and "risk_mode_invalid" in res["reason_codes"]


# --- Task 5: vehicle lock / candidate fingerprint binding ------------------------------------------

def test_vehicle_switch_changes_fingerprint_and_invalidates_old_lease():
    qqq = {"ticker": "QQQ", "direction": "bearish", "created_at": _iso(PROMOTION_AT)}
    spy = {"ticker": "SPY", "direction": "bearish", "created_at": _iso(PROMOTION_AT)}
    assert xp.candidate_fingerprint(qqq) != xp.candidate_fingerprint(spy)
    pkg = _package(PROMOTION_AT)
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    lease = res["lease"]
    exact_candidate = pkg["candidate_decision"]["candidate"]
    assert xp.lease_matches_candidate(lease, exact_candidate) is True
    switched = {**_fixture()["candidate"], "ticker": "QQQ", "selected_vehicle": "QQQ"}
    assert xp.lease_matches_candidate(lease, switched) is False


def test_qqq_candidate_with_qqq_contract_can_lease():
    contract = {**FX["contract"], "underlying": "QQQ", "option_id": "QQQ260723P00718000",
                "strike_price": 718.0}
    pkg = _package(PROMOTION_AT, sym="QQQ", contract=contract)
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    assert res["lease"]["symbol"] == "QQQ"


# --- guardrail: no broker / network / LLM ----------------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(xp)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket"):
        assert forbidden not in src, f"odte_execution_policy must not reference {forbidden!r}"
