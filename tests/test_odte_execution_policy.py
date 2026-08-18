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
        # Chase-band anchor, stamped by candidate-watch at CONFIRM_ENTRY (2026-08-02 retune).
        "anchor_quote": contract.get("ask") or contract.get("mark"),
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
    # 2026-08-02 retune: the lease ceiling extends to the chase band above the CONFIRM_ENTRY
    # anchor (the fixture anchors at the contract ask), still capped by the BP fraction.
    ceiling = round(fx["contract"]["ask"] * (1 + xp.CHASE_BAND_FRACTION), 2)
    assert lease["max_limit_price"] == ceiling
    assert lease["max_debit"] == round(
        min(xp.DEFAULT_MAX_CONTRACTS * ceiling * 100,
            xp.DEFAULT_MAX_DEBIT_FRACTION * fx["broker_snapshot"]["buying_power"]), 2)
    assert lease["anchor_quote"] == fx["contract"]["ask"]
    assert lease["candidate_fingerprint"] and lease["market_fingerprint"]
    assert lease["risk_mode"] == xp.DEFAULT_RISK_MODE
    # Policy values are serialized into the payload (and journaled from there).
    assert res["policy"]["ttl_seconds"] == xp.DEFAULT_LEASE_TTL_SECONDS
    assert res["policy"]["max_debit_fraction"] == xp.DEFAULT_MAX_DEBIT_FRACTION
    assert res["policy"]["chase_band_fraction"] == xp.CHASE_BAND_FRACTION
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


# --- 2026-08-02 retune: chase band + tiered BP-proportional sizing --------------------------------

def test_limit_above_chase_band_is_refused():
    # The fill ceiling is anchor*(1+band): one cent above refuses. The band is measured from the
    # CONFIRM_ENTRY anchor, not the precompute quote — re-pricing inside the band is allowed.
    fx = _fixture()
    anchor = fx["contract"]["ask"]
    too_high = round(anchor * (1 + xp.CHASE_BAND_FRACTION) + 0.01, 2)
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                             policy={"quantity": 1, "limit_price": too_high})
    assert res["authorized"] is False
    assert "limit_exceeds_chase_band" in res["reason_codes"]
    # At the ceiling exactly (still inside every BP cap for one contract) it authorizes.
    at_ceiling = round(anchor * (1 + xp.CHASE_BAND_FRACTION), 2)
    ok = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                            policy={"quantity": 1, "limit_price": at_ceiling})
    assert ok["authorized"] is True, ok["reason_codes"]


def test_missing_anchor_quote_fails_closed():
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"]["candidate"].pop("anchor_quote", None)
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False
    assert "anchor_quote_missing" in res["reason_codes"]


def test_b_plus_tier_halves_debit_fraction():
    # The tape-computed B+ tier (CHOP half-size) applies B_PLUS_DEBIT_FRACTION instead of the full
    # fraction: the incident contract's $168 single-contract debit fits the full tier but not B+.
    fx = _fixture()
    bp = fx["broker_snapshot"]["buying_power"]
    debit = round(fx["contract"]["ask"] * 100.0, 2)
    assert xp.B_PLUS_DEBIT_FRACTION * bp < debit <= xp.DEFAULT_MAX_DEBIT_FRACTION * bp, \
        "fixture must straddle the two tier caps for this test to bite"
    full = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    assert full["authorized"] is True, full["reason_codes"]
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"]["candidate"]["tier"] = "b_plus"
    pkg["gate"]["tier"] = "b_plus"
    half = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert half["authorized"] is False
    assert "debit_exceeds_policy" in half["reason_codes"]
    assert half["policy"]["max_debit_fraction"] == xp.B_PLUS_DEBIT_FRACTION


def test_candidate_gate_tier_mismatch_is_refused():
    # The gate's recorded tier must agree with the candidate's tape-computed tier — a divergence
    # means the artifacts describe two different setups.
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"]["candidate"]["tier"] = "b_plus"
    pkg["gate"]["tier"] = "a_plus"
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is False
    assert "tier_mismatch" in res["reason_codes"]


def test_snapshot_ttls_widened_but_in_process_artifacts_stay_tight():
    # 2026-08-02 retune contract: ONLY the two snapshot TTLs widened (fetch→authorize latency);
    # the three in-process artifacts stay at 60s and the lease hard cap is untouched.
    assert xp.INPUT_TTLS_SECONDS["market_snapshot"] == xp.SNAPSHOT_TTL_SECONDS
    assert xp.INPUT_TTLS_SECONDS["broker_snapshot"] == xp.SNAPSHOT_TTL_SECONDS
    assert xp.SNAPSHOT_TTL_SECONDS > 60.0
    assert xp.INPUT_TTLS_SECONDS["entry_gate"] == 60.0
    assert xp.INPUT_TTLS_SECONDS["candidate_decision"] == 60.0
    assert xp.INPUT_TTLS_SECONDS["vehicle_score"] == 60.0
    assert xp.MAX_LEASE_TTL_SECONDS == 60.0
    assert xp.DEFAULT_LEASE_TTL_SECONDS <= xp.MAX_LEASE_TTL_SECONDS


def test_jul31_snapshot_replay_now_converts():
    # REPLAY (2026-07-31 15:35 ET): every gate passed and the lease was refused because the
    # market/broker snapshots were 60-120s old. Under the widened snapshot TTL the same package
    # authorizes; at the new bound it still fails closed.
    inside = _package(PROMOTION_AT)
    for key in ("market_snapshot", "broker_snapshot"):
        inside[key]["as_of"] = _iso(PROMOTION_AT - timedelta(seconds=90))
    res = xp.authorize_entry(**inside, now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    beyond = _package(PROMOTION_AT)
    beyond["market_snapshot"]["as_of"] = _iso(
        PROMOTION_AT - timedelta(seconds=xp.SNAPSHOT_TTL_SECONDS + 1))
    res2 = xp.authorize_entry(**beyond, now=PROMOTION_AT, policy={})
    assert res2["authorized"] is False
    assert "market_snapshot_stale" in res2["reason_codes"]


def test_lease_ceilings_are_mutually_consistent():
    # 2026-08-06 QQQ 722C: the lease published max_limit_price 0.86 (anchor 0.75 x 1.15) beside
    # max_debit 84.61 (B+ 30% of $282.02 BP) — an order at the lease's OWN limit ceiling
    # violated its debit ceiling and the hook blocked it. The limit ceiling now clamps to the
    # affordable cent: qty x max_limit_price x 100 <= max_debit ALWAYS.
    fx = _fixture()
    contract = {**fx["contract"], "bid": 0.74, "ask": 0.75, "mark": 0.745}
    pkg = _package(PROMOTION_AT, contract=contract)
    pkg["broker_snapshot"]["buying_power"] = 282.02
    pkg["candidate_decision"]["candidate"]["tier"] = "b_plus"
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT,
                             policy={"quantity": 1, "limit_price": 0.75})
    assert res["authorized"] is True, res["reason_codes"]
    lease = res["lease"]
    assert lease["max_debit"] == round(xp.B_PLUS_DEBIT_FRACTION * 282.02, 2)   # 84.61
    assert lease["max_limit_price"] == 0.84                                     # floored cent
    assert lease["quantity"] * lease["max_limit_price"] * 100.0 <= lease["max_debit"]
    # The clamp never falls below the reviewed limit itself.
    assert lease["max_limit_price"] >= 0.75


def test_lease_ceiling_unclamped_when_bp_is_not_binding():
    # Plenty of BP: the chase-band ceiling stands untouched (the original 2026-08-02 contract).
    fx = _fixture()
    contract = {**fx["contract"], "bid": 1.15, "ask": 1.19, "mark": 1.17}
    pkg = _package(PROMOTION_AT, contract=contract)
    pkg["broker_snapshot"]["buying_power"] = 348.16
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT,
                             policy={"quantity": 1, "limit_price": 1.19})
    assert res["authorized"] is True, res["reason_codes"]
    lease = res["lease"]
    assert lease["max_limit_price"] == round(1.19 * (1 + xp.CHASE_BAND_FRACTION), 2)
    assert lease["quantity"] * lease["max_limit_price"] * 100.0 <= lease["max_debit"]


# --- 2026-08-07: the single-use ledger is an atomic test-and-set ------------------------------

def test_record_consumed_reports_whether_it_claimed_the_lease(tmp_path):
    """This was a bare read-modify-write — load, membership check, add, atomic_write_text — with no
    lock, so two consumers could both pass their check before either recorded and both place
    against ONE single-use lease. The return value is what makes the claim atomic; callers treat
    False as "someone else consumed this"."""
    from data.odte_execution_policy import load_consumed_ids, record_consumed
    ledger = tmp_path / "consumed_leases.json"
    assert record_consumed(ledger, "L1") is True         # we claimed it
    assert record_consumed(ledger, "L1") is False        # someone (us) already had it
    assert record_consumed(ledger, "L2") is True
    assert load_consumed_ids(ledger) == {"L1", "L2"}


def test_ledger_lock_is_a_sidecar_not_the_ledger_itself(tmp_path):
    """`atomic_write_text` ends in `os.replace`, so a lock held on the ledger inode would be
    released to a stale inode and a second process could lock the replacement independently. The
    sidecar is never replaced."""
    from data.odte_execution_policy import record_consumed
    ledger = tmp_path / "consumed_leases.json"
    record_consumed(ledger, "L1")
    assert (tmp_path / ".consumed_leases.json.lock").exists()
    assert ledger.exists()


# --- tier-scaled contract sizing (2026-08-14, operator decision) --------------------------------
# At 1 contract the debit fractions never bind: a +20% winner moved the account +3.3% while tier
# quality changed nothing about size. full/a_plus may now size up to their configured max WITHIN
# the fraction and BP; b_plus (the tier that carried every recent loss) stays at
# DEFAULT_MAX_CONTRACTS. Code default is 1 (off); cfg/config.yaml arms full/a_plus at 2.

def _cheap_package(at, tier, ask=0.67):
    # the contract rides INSIDE vehicle_score; _package takes it as a parameter
    contract = dict(_fixture()["contract"])
    contract["ask"] = ask                               # 0.67 = today's real SPY 776P premium
    pkg = _package(at, contract=contract)
    pkg["candidate_decision"]["candidate"]["tier"] = tier
    pkg["gate"]["tier"] = tier
    return pkg


def test_full_tier_sizes_to_two_contracts_within_the_fraction(monkeypatch):
    monkeypatch.setattr("data.odte_config.TIER_MAX_CONTRACTS_FULL", 2)
    res = xp.authorize_entry(**_cheap_package(PROMOTION_AT, "full"), now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    lease = res["lease"]
    assert lease["quantity"] == 2
    bp = 400.34
    assert lease["max_debit"] <= xp.DEFAULT_MAX_DEBIT_FRACTION * bp + 0.01
    assert res["policy"]["max_contracts"] == 2


def test_b_plus_stays_one_contract_even_when_affordable(monkeypatch):
    """The loss-carrying tier NEVER sizes up, no matter the knobs or affordability."""
    monkeypatch.setattr("data.odte_config.TIER_MAX_CONTRACTS_FULL", 2)
    monkeypatch.setattr("data.odte_config.TIER_MAX_CONTRACTS_APLUS", 2)
    pkg = _cheap_package(PROMOTION_AT, "b_plus", ask=0.30)   # b_plus fraction affords 4
    res = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    assert res["lease"]["quantity"] == 1
    assert res["policy"]["max_contracts"] == xp.DEFAULT_MAX_CONTRACTS


def test_full_tier_caps_at_one_when_the_fraction_affords_only_one(monkeypatch):
    """The fraction is the binding rail, not the tier ceiling — the incident-priced contract
    (debit between the two caps) still sizes 1 at full tier."""
    monkeypatch.setattr("data.odte_config.TIER_MAX_CONTRACTS_FULL", 2)
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    assert res["lease"]["quantity"] == 1                # 2 x $168 would breach 0.60 x BP


def test_knobs_off_is_exactly_the_old_behavior(monkeypatch):
    monkeypatch.setattr("data.odte_config.TIER_MAX_CONTRACTS_FULL", 1)
    monkeypatch.setattr("data.odte_config.TIER_MAX_CONTRACTS_APLUS", 1)
    res = xp.authorize_entry(**_cheap_package(PROMOTION_AT, "full"), now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    assert res["lease"]["quantity"] == 1
    assert res["policy"]["max_contracts"] == xp.DEFAULT_MAX_CONTRACTS


def test_live_posture_pins_tier_scaled_sizing():
    """2026-08-18 RESUME posture: back to 1-lot everywhere — the 08-14 2-lot decision rode the
    a_plus losers (-$32 was a 2-lot). Size returns only when the freshness spec earns it
    (data/odte/reports/entry_signal_autopsy_2026-08-18.md); updating these pins IS that act."""
    import data.odte_config as _oc
    assert _oc.TIER_MAX_CONTRACTS_FULL == 1
    assert _oc.TIER_MAX_CONTRACTS_APLUS == 1
    assert xp.DEFAULT_MAX_CONTRACTS == 1               # b_plus/default ceiling unchanged
