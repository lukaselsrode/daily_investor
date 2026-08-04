"""tests/test_odte_execution_integration.py — broker-lane integration simulator (Task 8).

End-to-end lease → consume → submit → guard flows against the in-memory FakeOptionBroker (which
records EVERY controller-invoked broker call, so prohibited calls are provably never made). No
network, no real broker, no money, no orders. All 8 plan scenarios:

  1. immediate fill inside the lease  → accepted and managed (FILLED_FRESH)
  2. pending past TTL                 → cancel requested and broker-verified cancelled
  3. invalidation before fill         → cancel requested
  4. cancel/fill race                 → broker truth wins; safety incident; no duplicate order
  5. delayed fill on today's exact timestamps → incident path, never normal A+ management
  6. concurrent controller ticks      → one lease consumed once; no duplicate submission
  7. symbol mismatch                  → no review or placement call
  8. quantity/debit violation         → no review or placement call
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fakes.fake_option_broker import FakeOptionBroker

import data.odte_execution_policy as xp
import data.odte_journal as oj
import data.odte_order_guard as og

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "odte" / "2026-07-23-delayed-fill.json"
FX = json.loads(FIXTURE_PATH.read_text())


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


PROMOTION_AT = _ts(FX["timeline"]["promotion_at"])
SUBMITTED_AT = _ts(FX["timeline"]["order_submitted_at"])
FILLED_AT = _ts(FX["timeline"]["order_filled_at"])


def _package(now: datetime, *, sym: str | None = None, contract: dict | None = None) -> dict:
    sym = sym or FX["candidate"]["ticker"]
    contract = contract if contract is not None else dict(FX["contract"])
    candidate = {
        **FX["candidate"], "ticker": sym, "selected_vehicle": sym,
        "selection_timestamp": _iso(now - timedelta(seconds=10)),
        "option_id": contract.get("option_id"),
        "expiration_date": contract.get("expiration_date"),
        "strike_price": contract.get("strike_price"),
        "option_type": contract.get("option_type"),
        # Chase-band anchor, stamped by candidate-watch at CONFIRM_ENTRY (2026-08-02 retune).
        "anchor_quote": contract.get("ask") or contract.get("mark"),
    }
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    return {
        "gate": {"generated_at": _iso(now), "symbol": sym, "direction": "bearish",
                 "candidate_fingerprint": candidate["candidate_fingerprint"],
                 "option_id": contract.get("option_id"),
                 "expiration_date": contract.get("expiration_date"),
                 "strike_price": contract.get("strike_price"),
                 "option_type": contract.get("option_type"),
                 "scan_only": False, "execution_allowed": True,
                 "required_confirmations": list(xp.DEFAULT_REQUIRED_CONFIRMATIONS),
                 "confirmations": {c: True for c in xp.DEFAULT_REQUIRED_CONFIRMATIONS}},
        "candidate_decision": {"decision": "CONFIRM_ENTRY",
                               "generated_at": _iso(now - timedelta(seconds=10)),
                               "candidate": candidate},
        "vehicle_score": {"verdict": "GOOD_BET", "direction": "bearish",
                          "generated_at": _iso(now - timedelta(seconds=8)), "contract": contract},
        "broker_snapshot": {**FX["broker_snapshot"], "as_of": _iso(now - timedelta(seconds=5))},
        "market_snapshot": {**FX["tape"]["at_trigger"], "as_of": _iso(now - timedelta(seconds=3))},
    }


def _full_account_policy() -> dict:
    return {"risk_mode": "FULL_ACCOUNT_A_PLUS", "quantity": FX["order"]["quantity"],
            "limit_price": FX["order"]["limit_price"],
            "management_plan": {"trigger": "SPY momentum extension below ORB low",
                                "invalidation": FX["invalidation"]["description"],
                                "target": "put +40%", "scratch_rail": "scratch at -10%",
                                "management_cadence_seconds": 10,
                                "max_premium_loss": FX["order"]["debit"]}}


def _controller_submit(broker: FakeOptionBroker, auth: dict, consumed: set[str],
                       now: datetime) -> tuple[dict | None, dict]:
    """The controller's deterministic submit path: refuse-first. NO broker call unless the
    authorization passed AND the single-use lease consumption succeeded."""
    if not auth.get("authorized"):
        return None, {"status": "refused", "reason_codes": auth.get("reason_codes")}
    lease = auth["lease"]
    res = xp.consume_lease(lease, consumed_ids=consumed, now=now)
    if res["status"] != "consumed":
        return None, res
    broker.review_order(symbol=lease["symbol"], option_id=lease["option_id"],
                        quantity=lease["quantity"], limit_price=lease["max_limit_price"])
    order = broker.place_order(symbol=lease["symbol"], option_id=lease["option_id"],
                               option_type=lease["option_type"],
                               strike_price=lease["strike_price"],
                               expiration_date=lease["expiration_date"],
                               quantity=lease["quantity"],
                               limit_price=lease["max_limit_price"], submitted_at=_iso(now))
    return order, res


# --- 1. immediate fill inside the lease -----------------------------------------------------------

def test_immediate_fill_inside_lease_is_accepted_and_managed():
    broker = FakeOptionBroker()
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    consumed: set[str] = set()
    t_submit = PROMOTION_AT + timedelta(seconds=2)
    order, res = _controller_submit(broker, auth, consumed, t_submit)
    assert res["status"] == "consumed" and order is not None
    broker.sim_fill(order["order_ref"], PROMOTION_AT + timedelta(seconds=6))
    truth = broker.order_status(order["order_ref"])
    guard = og.evaluate_order_guard(truth, lease=auth["lease"],
                                    now=PROMOTION_AT + timedelta(seconds=7))
    assert guard["state"] == og.FILLED_FRESH
    assert guard["safety_incident"] is False
    assert oj.event_from_order_guard(guard)["event_type"] == "order_filled"
    assert len(broker.calls_of("place_order")) == 1
    assert len(broker.calls_of("cancel_order")) == 0


# --- 2. pending past TTL: cancel requested and broker-verified cancelled ---------------------------

def test_pending_past_ttl_cancel_requested_and_verified():
    broker = FakeOptionBroker()
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    order, _ = _controller_submit(broker, auth, set(), PROMOTION_AT + timedelta(seconds=2))
    late = PROMOTION_AT + timedelta(seconds=xp.DEFAULT_LEASE_TTL_SECONDS + 5)
    guard = og.evaluate_order_guard(broker.order_status(order["order_ref"]),
                                    lease=auth["lease"], now=late)
    assert guard["state"] == og.CANCEL_STALE_ENTRY and guard["cancel_required"] is True
    # Controller obeys: cancel at the broker, then VERIFY with fresh order truth.
    broker.cancel_order(order["order_ref"])
    verified = og.evaluate_order_guard(broker.order_status(order["order_ref"]),
                                       lease=auth["lease"], now=late + timedelta(seconds=1))
    assert verified["state"] == og.NO_ORDER
    assert len(broker.calls_of("cancel_order")) == 1
    assert len(broker.calls_of("place_order")) == 1            # never re-placed


# --- 3. invalidation before fill: cancel requested --------------------------------------------------

def test_invalidation_before_fill_requests_cancel():
    broker = FakeOptionBroker()
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    order, _ = _controller_submit(broker, auth, set(), PROMOTION_AT + timedelta(seconds=2))
    inside_lease = PROMOTION_AT + timedelta(seconds=10)
    guard = og.evaluate_order_guard(broker.order_status(order["order_ref"]), lease=auth["lease"],
                                    market_snapshot=dict(FX["tape"]["at_fill"]),  # SPY reclaimed VWAP
                                    now=inside_lease)
    assert guard["state"] == og.CANCEL_THESIS_INVALID and guard["cancel_required"] is True
    broker.cancel_order(order["order_ref"])
    assert broker.order_status(order["order_ref"])["status"] == "cancelled"


# --- 4. cancel/fill race: broker truth wins; safety incident; no duplicate order --------------------

def test_cancel_fill_race_broker_truth_wins_and_no_duplicate_order():
    broker = FakeOptionBroker()
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    consumed: set[str] = set()
    order, _ = _controller_submit(broker, auth, consumed, PROMOTION_AT + timedelta(seconds=2))
    # The exchange fills a moment before the controller's cancel lands (after TTL).
    fill_time = PROMOTION_AT + timedelta(seconds=xp.DEFAULT_LEASE_TTL_SECONDS + 4)
    broker.sim_fill(order["order_ref"], fill_time)
    cancel_response = broker.cancel_order(order["order_ref"])
    assert cancel_response["status"] == "filled", "broker truth wins: a filled order stays filled"
    guard = og.evaluate_order_guard(cancel_response, lease=auth["lease"],
                                    now=fill_time + timedelta(seconds=2))
    assert guard["state"] == og.FILLED_WITHOUT_VALID_LEASE
    assert guard["safety_incident"] is True and guard["prohibit_new_entries"] is True
    assert oj.event_from_order_guard(guard)["event_type"] == "execution_safety_incident"
    # No duplicate order: the burned/expired lease refuses a second submission outright.
    order2, res2 = _controller_submit(broker, auth, consumed,
                                      fill_time + timedelta(seconds=3))
    assert order2 is None and res2["status"] in ("already_consumed", "expired")
    assert len(broker.calls_of("place_order")) == 1


# --- 5. delayed fill matching today's timestamps: incident path, never normal A+ management ---------

def test_replay_todays_delayed_fill_is_incident_path_never_normal():
    broker = FakeOptionBroker()
    # The lease the 11:11:17 promotion could (at most) have minted — full-account, exact order.
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                              policy=_full_account_policy())
    assert auth["authorized"] is True
    lease = auth["lease"]
    # Under the new layer the 11:13:11 submission is IMPOSSIBLE: the lease is already dead.
    blocked, res = _controller_submit(broker, auth, set(), SUBMITTED_AT)
    assert blocked is None and res["status"] == "expired"
    assert broker.calls == [], "no review/place call may happen on a dead lease"
    # Replay the legacy (unguarded) submission the incident actually made:
    order = broker.place_order(symbol=FX["order"]["symbol"], option_id=FX["order"]["option_id"],
                               option_type=FX["order"]["option_type"],
                               strike_price=FX["order"]["strike_price"],
                               expiration_date=FX["order"]["expiration_date"],
                               quantity=FX["order"]["quantity"],
                               limit_price=FX["order"]["limit_price"],
                               submitted_at=FX["timeline"]["order_submitted_at"])
    # First guard tick after submission: cancel decision exists BEFORE the recorded fill time.
    tick = SUBMITTED_AT + timedelta(seconds=1)
    guard = og.evaluate_order_guard(broker.order_status(order["order_ref"]), lease=lease, now=tick)
    assert guard["state"] == og.CANCEL_STALE_ENTRY
    assert _ts(guard["generated_at"]) < FILLED_AT
    # If (as on the day) the fill still lands at 15:15:35Z, it is the INCIDENT path — never
    # reclassified into normal position management.
    broker.sim_fill(order["order_ref"], FX["timeline"]["order_filled_at"])
    incident = og.evaluate_order_guard(broker.order_status(order["order_ref"]), lease=lease,
                                       now=FILLED_AT + timedelta(seconds=2))
    assert incident["state"] == og.FILLED_WITHOUT_VALID_LEASE
    assert incident["state"] != og.FILLED_FRESH
    assert incident["safety_incident"] is True and incident["prohibit_new_entries"] is True
    assert oj.event_from_order_guard(incident)["event_type"] == "execution_safety_incident"


# --- 6. concurrent controller ticks: one lease consumed once; no duplicate submission ---------------

def test_concurrent_ticks_consume_lease_once_no_duplicate_submission():
    broker = FakeOptionBroker()
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    consumed: set[str] = set()          # the shared single-use ledger
    t = PROMOTION_AT + timedelta(seconds=2)
    order_a, res_a = _controller_submit(broker, auth, consumed, t)
    order_b, res_b = _controller_submit(broker, auth, consumed, t)   # the concurrent second tick
    assert order_a is not None and res_a["status"] == "consumed"
    assert order_b is None and res_b["status"] == "already_consumed"
    assert len(broker.calls_of("place_order")) == 1
    assert len(broker.calls_of("review_order")) == 1


# --- 7. symbol mismatch: no review or placement call -------------------------------------------------

def test_symbol_mismatch_makes_no_broker_calls():
    broker = FakeOptionBroker()
    pkg = _package(PROMOTION_AT)
    pkg["candidate_decision"]["candidate"]["ticker"] = "QQQ"    # QQQ thesis, SPY contract
    pkg["gate"]["symbol"] = "QQQ"
    auth = xp.authorize_entry(**pkg, now=PROMOTION_AT, policy={})
    assert auth["authorized"] is False and "symbol_mismatch" in auth["reason_codes"]
    order, res = _controller_submit(broker, auth, set(), PROMOTION_AT + timedelta(seconds=1))
    assert order is None and res["status"] == "refused"
    assert broker.calls == [], "a refused authorization must produce ZERO broker calls"


# --- 8. quantity/debit violation: no review or placement call ---------------------------------------

def test_quantity_debit_violation_makes_no_broker_calls():
    broker = FakeOptionBroker()
    auth = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT,
                              policy={"quantity": FX["order"]["quantity"],
                                      "limit_price": FX["order"]["limit_price"]})
    assert auth["authorized"] is False
    assert {"quantity_exceeds_policy", "debit_exceeds_policy"} <= set(auth["reason_codes"])
    order, res = _controller_submit(broker, auth, set(), PROMOTION_AT + timedelta(seconds=1))
    assert order is None and res["status"] == "refused"
    assert broker.calls == []
