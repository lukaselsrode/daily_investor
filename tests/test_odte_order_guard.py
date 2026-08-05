"""tests/test_odte_order_guard.py — pending-order TTL / cancel-first state machine.

Pure/offline: no broker, network, LLM, or orders. Replays the 2026-07-23 delayed-fill incident
deterministically from the sanitized fixture and pins the invariants:
  * an unfilled entry order past its lease outputs CANCEL_STALE_ENTRY immediately — and the replay
    produces that cancel decision BEFORE the recorded fill timestamp;
  * a fill after lease expiry is FILLED_WITHOUT_VALID_LEASE — a safety incident that prohibits new
    entries, never the normal path;
  * thesis invalidation before the fill cancels;
  * broker/lease identity or maximum disagreement blocks;
  * a lease can never retroactively cover an order submitted before it was issued;
  * an order at the broker burns its lease (single use) in the ledger.

TTLs/policy values come from the live modules — never re-hardcoded.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


def _package(now: datetime) -> dict:
    contract = dict(FX["contract"])
    candidate = {
        **FX["candidate"], "selected_vehicle": "SPY",
        "selection_timestamp": _iso(now - timedelta(seconds=10)),
        "option_id": contract.get("option_id"),
        "expiration_date": contract.get("expiration_date"),
        "strike_price": contract.get("strike_price"),
        "option_type": contract.get("option_type"),
        # Chase-band anchor, stamped by candidate-watch at CONFIRM_ENTRY (2026-08-02 retune).
        "anchor_quote": contract.get("ask") or contract.get("mark"),
    }
    candidate["candidate_fingerprint"] = xp.candidate_fingerprint(candidate)
    gate = {"generated_at": _iso(now), "symbol": "SPY", "direction": "bearish",
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "option_id": contract.get("option_id"),
            "expiration_date": contract.get("expiration_date"),
            "strike_price": contract.get("strike_price"),
            "option_type": contract.get("option_type"),
            "scan_only": False, "execution_allowed": True,
            "required_confirmations": list(xp.DEFAULT_REQUIRED_CONFIRMATIONS),
            "confirmations": {c: True for c in xp.DEFAULT_REQUIRED_CONFIRMATIONS}}
    cd = {"decision": "CONFIRM_ENTRY", "generated_at": _iso(now - timedelta(seconds=10)),
          "candidate": candidate}
    vs = {"verdict": "GOOD_BET", "direction": "bearish",
          "generated_at": _iso(now - timedelta(seconds=8)), "contract": contract}
    broker = {**FX["broker_snapshot"], "as_of": _iso(now - timedelta(seconds=5))}
    market = {**FX["tape"]["at_trigger"], "as_of": _iso(now - timedelta(seconds=3))}
    return {"gate": gate, "candidate_decision": cd, "vehicle_score": vs,
            "broker_snapshot": broker, "market_snapshot": market}


def _full_account_lease(now: datetime = PROMOTION_AT) -> dict:
    """The exact incident order, authorized the only way it now could be: FULL_ACCOUNT_A_PLUS."""
    policy = {
        "risk_mode": "FULL_ACCOUNT_A_PLUS",
        "quantity": FX["order"]["quantity"],
        "limit_price": FX["order"]["limit_price"],
        "management_plan": {
            "trigger": "SPY momentum extension below ORB low",
            "invalidation": FX["invalidation"]["description"],
            "target": "put +40%",
            "scratch_rail": "scratch at -10%",
            "management_cadence_seconds": 10,
            "max_premium_loss": FX["order"]["debit"],
        },
    }
    res = xp.authorize_entry(**_package(now), now=now, policy=policy)
    assert res["authorized"] is True, res["reason_codes"]
    return res["lease"]


def _pending_order(lease: dict, submitted_at: datetime) -> dict:
    return {**FX["order"], "submitted_at": _iso(submitted_at), "filled_at": None,
            "status": "pending"}


# --- NO_ORDER --------------------------------------------------------------------------------------

def test_no_order_is_no_order():
    r = og.evaluate_order_guard(None, lease=_full_account_lease(), now=PROMOTION_AT)
    assert r["state"] == og.NO_ORDER
    assert r["cancel_required"] is False and r["safety_incident"] is False
    assert r["places_orders"] is False


def test_cancelled_order_is_no_order():
    lease = _full_account_lease()
    order = {**_pending_order(lease, PROMOTION_AT), "status": "cancelled"}
    r = og.evaluate_order_guard(order, lease=lease, now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] == og.NO_ORDER


# --- PENDING_FRESH ----------------------------------------------------------------------------------

def test_pending_inside_lease_with_valid_thesis_is_pending_fresh():
    lease = _full_account_lease()
    submitted = PROMOTION_AT + timedelta(seconds=3)
    now = PROMOTION_AT + timedelta(seconds=10)
    r = og.evaluate_order_guard(_pending_order(lease, submitted), lease=lease,
                                market_snapshot=dict(FX["tape"]["at_trigger"]), now=now)
    assert r["state"] == og.PENDING_FRESH
    assert r["cancel_required"] is False
    assert r["pending_order_age_seconds"] == 7.0
    assert r["lease_seconds_remaining"] == xp.DEFAULT_LEASE_TTL_SECONDS - 10


# --- CANCEL_STALE_ENTRY: the replay acceptance -------------------------------------------------------

def test_replay_incident_order_cancels_before_recorded_fill():
    # ACCEPTANCE (plan Task 4): today's order must produce CANCEL_STALE_ENTRY before the recorded
    # 15:15:35Z fill. The order was submitted 15:13:11Z; the 11:11:17 lease died at +TTL.
    lease = _full_account_lease(PROMOTION_AT)
    guard_time = SUBMITTED_AT + timedelta(seconds=1)          # first guard tick after submission
    assert guard_time < FILLED_AT
    r = og.evaluate_order_guard(_pending_order(lease, SUBMITTED_AT), lease=lease,
                                market_snapshot=dict(FX["tape"]["at_trigger"]), now=guard_time)
    assert r["state"] == og.CANCEL_STALE_ENTRY
    assert r["cancel_required"] is True
    assert _ts(r["generated_at"]) < FILLED_AT, \
        "the cancel decision must exist strictly before the recorded fill time"
    assert r["lease_seconds_remaining"] < 0


def test_pending_past_ttl_cancels_immediately_at_boundary():
    lease = _full_account_lease()
    expiry = _ts(lease["expires_at"])
    r = og.evaluate_order_guard(_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
                                lease=lease, now=expiry)
    assert r["state"] == og.CANCEL_STALE_ENTRY, "TTL expiry cancels AT the boundary, not after"


def test_pending_with_no_lease_cancels():
    r = og.evaluate_order_guard({**FX["order"]}, lease=None, now=SUBMITTED_AT)
    assert r["state"] == og.CANCEL_STALE_ENTRY
    assert r["cancel_required"] is True


def test_stale_order_cannot_be_extended_by_a_new_lease():
    # A brand-new lease minted AFTER the order was submitted can never retroactively cover it —
    # "extend the old order with a fresh lease" is structurally impossible.
    fresh_lease = _full_account_lease(SUBMITTED_AT + timedelta(seconds=30))
    old_order = _pending_order(fresh_lease, SUBMITTED_AT)     # submitted before lease issuance
    r = og.evaluate_order_guard(old_order, lease=fresh_lease,
                                now=SUBMITTED_AT + timedelta(seconds=35))
    assert r["state"] == og.CANCEL_STALE_ENTRY
    assert any("predates the lease" in reason for reason in r["reasons"])


# --- CANCEL_THESIS_INVALID ---------------------------------------------------------------------------

def test_thesis_invalidation_before_ttl_cancels():
    # SPY reclaimed VWAP (the fixture's at-fill tape) while the bearish order was still pending
    # INSIDE the lease window: cancel on the dead thesis, don't wait for the TTL.
    lease = _full_account_lease()
    now = PROMOTION_AT + timedelta(seconds=10)
    r = og.evaluate_order_guard(_pending_order(lease, PROMOTION_AT + timedelta(seconds=3)),
                                lease=lease, market_snapshot=dict(FX["tape"]["at_fill"]), now=now)
    assert r["state"] == og.CANCEL_THESIS_INVALID
    assert r["cancel_required"] is True
    assert any("reclaimed VWAP" in reason for reason in r["reasons"])


def test_explicit_invalidation_level_cancels():
    lease = _full_account_lease()
    market = {"SPY": {"last": FX["invalidation"]["level"] + 0.1},
              "invalidation_level": FX["invalidation"]["level"]}
    r = og.evaluate_order_guard(_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
                                lease=lease, market_snapshot=market,
                                now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] == og.CANCEL_THESIS_INVALID


# --- fills -------------------------------------------------------------------------------------------

def test_filled_inside_lease_is_filled_fresh():
    lease = _full_account_lease()
    order = {**FX["order"], "status": "filled",
             "submitted_at": _iso(PROMOTION_AT + timedelta(seconds=2)),
             "filled_at": _iso(PROMOTION_AT + timedelta(seconds=8))}
    r = og.evaluate_order_guard(order, lease=lease, now=PROMOTION_AT + timedelta(seconds=9))
    assert r["state"] == og.FILLED_FRESH
    assert r["safety_incident"] is False and r["prohibit_new_entries"] is False


def test_replay_delayed_fill_is_safety_incident_never_normal():
    # The incident's actual fill (15:15:35Z, minutes after lease death) is the incident path:
    # FILLED_WITHOUT_VALID_LEASE — journal + prohibit new entries + flatten/alert, never "A+".
    lease = _full_account_lease(PROMOTION_AT)
    order = {**FX["order"], "status": "filled", "filled_at": FX["timeline"]["order_filled_at"]}
    r = og.evaluate_order_guard(order, lease=lease, now=FILLED_AT + timedelta(seconds=2))
    assert r["state"] == og.FILLED_WITHOUT_VALID_LEASE
    assert r["safety_incident"] is True
    assert r["prohibit_new_entries"] is True
    assert r["state"] != og.FILLED_FRESH


def test_fill_with_no_lease_is_safety_incident():
    order = {**FX["order"], "status": "filled", "filled_at": FX["timeline"]["order_filled_at"]}
    r = og.evaluate_order_guard(order, lease=None, now=FILLED_AT + timedelta(seconds=2))
    assert r["state"] == og.FILLED_WITHOUT_VALID_LEASE


# --- BROKER_MISMATCH_BLOCKED ---------------------------------------------------------------------------

def test_symbol_mismatch_blocks():
    lease = _full_account_lease()
    order = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)), "symbol": "QQQ"}
    r = og.evaluate_order_guard(order, lease=lease, now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] == og.BROKER_MISMATCH_BLOCKED
    assert r["safety_incident"] is True and r["prohibit_new_entries"] is True
    assert r["cancel_required"] is True                        # still pending — cancel it too


def test_quantity_or_limit_above_lease_blocks():
    lease = _full_account_lease()
    over_qty = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
                "quantity": lease["quantity"] + 1}
    r = og.evaluate_order_guard(over_qty, lease=lease, now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] == og.BROKER_MISMATCH_BLOCKED
    over_limit = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
                  "limit_price": lease["max_limit_price"] + 0.05}
    r2 = og.evaluate_order_guard(over_limit, lease=lease, now=PROMOTION_AT + timedelta(seconds=5))
    assert r2["state"] == og.BROKER_MISMATCH_BLOCKED


# --- run_order_guard: IO wrapper, lease burning, journal mapping ------------------------------------

def test_run_order_guard_burns_lease_once_and_writes_artifact(tmp_path):
    lease = _full_account_lease()
    lease_path = tmp_path / "execution_lease.json"
    lease_path.write_text(json.dumps({"authorized": True, "lease": lease}))
    order = _pending_order(lease, PROMOTION_AT + timedelta(seconds=2))
    payload = og.run_order_guard(order_json=json.dumps(order), lease_path=str(lease_path),
                                 market_json=json.dumps(FX["tape"]["at_trigger"]),
                                 state_dir=str(tmp_path), write=True,
                                 now=PROMOTION_AT + timedelta(seconds=10))
    assert payload["state"] == og.PENDING_FRESH
    assert payload["lease_consumed_now"] is True
    assert (tmp_path / og.ORDER_GUARD_FILENAME).exists()
    ledger = tmp_path / xp.CONSUMED_LEASES_FILENAME
    assert lease["lease_id"] in xp.load_consumed_ids(ledger)
    # Second run: the lease is already burned — never "consumed" twice.
    payload2 = og.run_order_guard(order_json=json.dumps(order), lease_path=str(lease_path),
                                  market_json=json.dumps(FX["tape"]["at_trigger"]),
                                  state_dir=str(tmp_path),
                                  now=PROMOTION_AT + timedelta(seconds=12))
    assert payload2["lease_consumed_now"] is False


def test_run_order_guard_defaults_to_state_dir_lease(tmp_path):
    lease = _full_account_lease()
    (tmp_path / xp.LEASE_FILENAME).write_text(json.dumps({"authorized": True, "lease": lease}))
    order = _pending_order(lease, PROMOTION_AT + timedelta(seconds=2))
    payload = og.run_order_guard(order_json=json.dumps(order), state_dir=str(tmp_path),
                                 market_json=json.dumps(FX["tape"]["at_trigger"]),
                                 now=PROMOTION_AT + timedelta(seconds=10))
    assert payload["lease_id"] == lease["lease_id"]
    assert payload["state"] == og.PENDING_FRESH


def test_guard_states_map_to_journal_event_types():
    lease = _full_account_lease()
    pending = og.evaluate_order_guard(_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
                                      lease=lease,
                                      market_snapshot=dict(FX["tape"]["at_trigger"]),
                                      now=PROMOTION_AT + timedelta(seconds=5))
    assert oj.event_from_order_guard(pending)["event_type"] == "entry_order_pending"

    stale = og.evaluate_order_guard(_pending_order(lease, SUBMITTED_AT), lease=lease,
                                    now=SUBMITTED_AT + timedelta(seconds=1))
    assert oj.event_from_order_guard(stale)["event_type"] == "entry_order_cancelled_stale"

    incident_order = {**FX["order"], "status": "filled",
                      "filled_at": FX["timeline"]["order_filled_at"]}
    incident = og.evaluate_order_guard(incident_order, lease=lease,
                                       now=FILLED_AT + timedelta(seconds=1))
    ev = oj.event_from_order_guard(incident)
    assert ev["event_type"] == "execution_safety_incident"
    assert ev["prohibit_new_entries"] is True

    empty = og.evaluate_order_guard(None, lease=lease, now=PROMOTION_AT)
    assert oj.event_from_order_guard(empty) is None


# --- guardrail: no broker / network / LLM -----------------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    import inspect
    src = inspect.getsource(og)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic",
                      "submit_order", "urllib", "httpx", "socket"):
        assert forbidden not in src, f"odte_order_guard must not reference {forbidden!r}"


# --- 2026-08-02 retune: chase-band lease ceiling composes with the (unchanged) guard ---------------

def test_order_limit_above_chase_band_ceiling_is_blocked():
    # A PARTIAL_ACCOUNT lease's max_limit_price is now anchor*(1+CHASE_BAND_FRACTION). The guard
    # itself is unchanged — this proves an order one cent above that ceiling is still a hard block.
    res = xp.authorize_entry(**_package(PROMOTION_AT), now=PROMOTION_AT, policy={})
    assert res["authorized"] is True, res["reason_codes"]
    lease = res["lease"]
    assert lease["max_limit_price"] == round(
        lease["anchor_quote"] * (1 + xp.CHASE_BAND_FRACTION), 2)
    order = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
             "quantity": lease["quantity"],
             "limit_price": round(lease["max_limit_price"] + 0.01, 2)}
    r = og.evaluate_order_guard(order, lease=lease, now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] == og.BROKER_MISMATCH_BLOCKED
    inside = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
              "quantity": lease["quantity"], "limit_price": lease["max_limit_price"],
              "debit": round(lease["quantity"] * lease["max_limit_price"] * 100.0, 2)}
    r2 = og.evaluate_order_guard(inside, lease=lease, now=PROMOTION_AT + timedelta(seconds=5))
    assert r2["state"] == og.PENDING_FRESH


def test_run_order_guard_materializes_consumed_ledger(tmp_path):
    # 2026-08-03 forensics: consumed_leases.json had NEVER existed on disk because the guard was
    # never invoked on the live path. Pin the contract: one run_order_guard call on a live order
    # writes the ledger and burns the lease exactly once.
    import json as _json
    lease = _full_account_lease()
    order = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2))}
    r = og.run_order_guard(order_json=_json.dumps(order), lease_json=_json.dumps(lease),
                           state_dir=str(tmp_path), write=True,
                           now=PROMOTION_AT + timedelta(seconds=5))
    ledger = tmp_path / "consumed_leases.json"
    assert ledger.exists(), "the single-use ledger must materialize on first guarded order"
    assert lease["lease_id"] in _json.loads(ledger.read_text())
    assert r["lease_consumed_now"] is True
    r2 = og.run_order_guard(order_json=_json.dumps(order), lease_json=_json.dumps(lease),
                            state_dir=str(tmp_path),
                            now=PROMOTION_AT + timedelta(seconds=6))
    assert r2["lease_consumed_now"] is False


# --- partial fills / pending cancels stay guarded (2026-08-04 fast-lane prerequisite) --------------

def test_partially_filled_inside_lease_is_pending_fresh():
    # A partial fill is a LIVE working order. Before this fix `partially_filled` fell through to
    # "gone" (NO_ORDER) — and the broker stamps filled_at on partials, which must never classify
    # the working remainder as a completed fill.
    lease = _full_account_lease()
    order = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
             "status": "partially_filled", "filled_quantity": 1,
             "filled_at": _iso(PROMOTION_AT + timedelta(seconds=4))}
    r = og.evaluate_order_guard(order, lease=lease,
                                market_snapshot={"SPY": {"above_vwap": False}},
                                now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] == og.PENDING_FRESH
    assert r["order"]["filled_quantity"] == 1          # passed through for partial-fill handling


def test_partially_filled_past_lease_cancels_remainder():
    lease = _full_account_lease()
    ttl = xp.lease_seconds_remaining(lease, PROMOTION_AT) or 0
    order = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
             "status": "partially_filled", "filled_quantity": 1}
    r = og.evaluate_order_guard(order, lease=lease,
                                now=PROMOTION_AT + timedelta(seconds=ttl + 1))
    assert r["state"] == og.CANCEL_STALE_ENTRY
    assert r["cancel_required"] is True


def test_pending_cancelled_is_still_guarded_not_no_order():
    # A cancel REQUEST is not a confirmed cancel — the order may still fill; keep guarding.
    lease = _full_account_lease()
    order = {**_pending_order(lease, PROMOTION_AT + timedelta(seconds=2)),
             "status": "pending_cancelled"}
    r = og.evaluate_order_guard(order, lease=lease,
                                market_snapshot={"SPY": {"above_vwap": False}},
                                now=PROMOTION_AT + timedelta(seconds=5))
    assert r["state"] != og.NO_ORDER
    assert r["state"] == og.PENDING_FRESH
