"""tests/test_odte_pre_place_guard.py — the fast lane's consume-before-place safety gate.

Pure/offline. Pins: every gate blocks independently (restricted underlying, no lease, identity
mismatch, expired lease, consumed ledger), closing orders bypass the lease gates but never the
NVDA scan, the guard fails CLOSED on internal errors, and — the crash-safety invariant — the
consumed-ledger write happens BEFORE the place call is awaited, so a crash in between burns the
lease rather than leaving it replayable. Ledger semantics come from the live policy module.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import data.odte_execution_policy as xp
import execution.odte_pre_place_guard as ppg

NOW = datetime(2026, 8, 5, 14, 31, 2, tzinfo=timezone.utc)


def _lease(**over):
    lease = {
        "lease_id": "lease-abc123", "symbol": "SPY", "option_id": "opt-756c",
        "option_type": "call", "strike_price": 756.0, "expiration_date": "2026-08-05",
        "quantity": 1, "max_limit_price": 0.73, "max_debit": 130.0,
        "issued_at": (NOW - timedelta(seconds=5)).isoformat(),
        "expires_at": (NOW + timedelta(seconds=55)).isoformat(),
    }
    lease.update(over)
    return lease


def _order(**over):
    order = {
        "symbol": "SPY", "option_id": "opt-756c", "option_type": "call",
        "strike_price": 756.0, "expiration_date": "2026-08-05",
        "quantity": 1, "limit_price": 0.64,
        "legs": [{"option_id": "opt-756c", "side": "buy", "position_effect": "open",
                  "ratio_quantity": 1}],
    }
    order.update(over)
    return order


def _check(order, lease, tmp_path, now=NOW):
    return ppg.pre_place_check(order, lease, ledger_path=str(tmp_path / "consumed.json"),
                               now=now)


# --- individual gates ------------------------------------------------------------------------

def test_clean_open_is_allowed(tmp_path):
    check = _check(_order(), _lease(), tmp_path)
    assert check == {"allowed": True, "closing": False, "lease_id": "lease-abc123",
                     "reasons": []}


def test_no_lease_blocks_opens(tmp_path):
    check = _check(_order(), None, tmp_path)
    assert check["allowed"] is False and "no_execution_lease" in check["reasons"]


def test_identity_mismatch_blocks(tmp_path):
    for bad in ({"symbol": "QQQ"},
                {"legs": [{"option_id": "opt-OTHER", "side": "buy",
                           "position_effect": "open", "ratio_quantity": 1}],
                 "option_id": "opt-OTHER"},
                {"limit_price": 0.80},                          # over the chase ceiling
                {"quantity": 2}):                               # over the leased size
        check = _check(_order(**bad), _lease(), tmp_path)
        assert check["allowed"] is False, bad
        assert any(r.startswith("lease_identity_mismatch") for r in check["reasons"]), bad


def test_expired_lease_blocks(tmp_path):
    check = _check(_order(), _lease(), tmp_path, now=NOW + timedelta(seconds=56))
    assert check["allowed"] is False and "lease_expired" in check["reasons"]


def test_consumed_lease_blocks_single_use(tmp_path):
    ledger = tmp_path / "consumed.json"
    xp.record_consumed(str(ledger), "lease-abc123")
    check = _check(_order(), _lease(), tmp_path)
    assert check["allowed"] is False and "lease_already_consumed" in check["reasons"]


def test_nvda_blocks_everything_even_closes(tmp_path):
    open_check = _check(_order(symbol="NVDA"), _lease(symbol="NVDA"), tmp_path)
    assert open_check["allowed"] is False
    assert any(r.startswith("restricted_underlying") for r in open_check["reasons"])
    close = _order(symbol="NVDA",
                   legs=[{"option_id": "opt-756c", "side": "sell",
                          "position_effect": "close", "ratio_quantity": 1}])
    close_check = _check(close, None, tmp_path)
    assert close_check["allowed"] is False                     # closing never bypasses NVDA


def test_closing_bypasses_lease_gates(tmp_path):
    close = _order(legs=[{"option_id": "opt-756c", "side": "sell",
                          "position_effect": "close", "ratio_quantity": 1}])
    # No lease, expired lease, consumed lease: a close always passes.
    assert _check(close, None, tmp_path)["allowed"] is True
    assert _check(close, _lease(), tmp_path,
                  now=NOW + timedelta(minutes=10))["allowed"] is True
    xp.record_consumed(str(tmp_path / "consumed.json"), "lease-abc123")
    check = _check(close, _lease(), tmp_path)
    assert check["allowed"] is True and check["closing"] is True


def test_guard_fails_closed_on_internal_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ppg, "_identity_mismatches",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    check = _check(_order(), _lease(), tmp_path)
    assert check["allowed"] is False
    assert any(r.startswith("guard_error") for r in check["reasons"])


# --- order_for_guard normalization -----------------------------------------------------------

def test_order_for_guard_maps_mcp_args():
    args = {"account_number": "435050133", "direction": "debit", "type": "limit",
            "quantity": "1", "price": "0.64", "time_in_force": "gfd",
            "legs": [{"option_id": "opt-756c", "side": "buy", "position_effect": "open",
                      "ratio_quantity": 1}]}
    order = ppg.order_for_guard(args, symbol="SPY", option_type="call", strike_price=756.0,
                                expiration_date="2026-08-05")
    assert order["option_id"] == "opt-756c" and order["limit_price"] == "0.64"
    assert order["quantity"] == "1" and order["symbol"] == "SPY"
    assert order["position_effect"] == "open"


# --- consume_then: the crash-safety ordering -------------------------------------------------

def test_consume_happens_before_place(tmp_path):
    ledger = str(tmp_path / "consumed.json")
    observed = {}

    async def place():
        # At the moment the broker call fires, the lease MUST already be burned.
        observed["consumed_at_place_time"] = "lease-abc123" in xp.load_consumed_ids(ledger)
        return {"order_id": "ord-1"}

    check, result = asyncio.run(ppg.consume_then(_order(), _lease(), ledger_path=ledger,
                                                 now=NOW, place=place))
    assert check["allowed"] is True and result == {"order_id": "ord-1"}
    assert observed["consumed_at_place_time"] is True


def test_crash_between_consume_and_place_burns_the_lease(tmp_path):
    ledger = str(tmp_path / "consumed.json")

    async def place():
        raise ConnectionError("crash mid-transmit")

    with pytest.raises(ConnectionError):
        asyncio.run(ppg.consume_then(_order(), _lease(), ledger_path=ledger, now=NOW,
                                     place=place))
    assert "lease-abc123" in xp.load_consumed_ids(ledger)      # burned, never replayable
    # And the burned lease refuses a second attempt outright.
    check = ppg.pre_place_check(_order(), _lease(), ledger_path=ledger, now=NOW)
    assert check["allowed"] is False and "lease_already_consumed" in check["reasons"]


def test_refused_check_never_touches_ledger_or_places(tmp_path):
    ledger = str(tmp_path / "consumed.json")
    called = []

    async def place():
        called.append(True)

    check, result = asyncio.run(ppg.consume_then(_order(quantity=5), _lease(),
                                                 ledger_path=ledger, now=NOW, place=place))
    assert check["allowed"] is False and result is None
    assert not called
    assert not json.loads((tmp_path / "consumed.json").read_text()) \
        if (tmp_path / "consumed.json").exists() else True


def test_closing_consume_then_places_without_burning(tmp_path):
    ledger = str(tmp_path / "consumed.json")
    close = _order(legs=[{"option_id": "opt-756c", "side": "sell",
                          "position_effect": "close", "ratio_quantity": 1}])

    async def place():
        return {"order_id": "close-1"}

    check, result = asyncio.run(ppg.consume_then(close, _lease(), ledger_path=ledger,
                                                 now=NOW, place=place))
    assert check["allowed"] is True and result == {"order_id": "close-1"}
    assert "lease-abc123" not in xp.load_consumed_ids(ledger)  # closes never burn entry leases


def test_two_consumers_racing_one_lease_place_exactly_once(tmp_path):
    """`pre_place_check` READS the ledger to verify the lease is unconsumed; recording it is a
    separate WRITE. Between the two, a second lane — the fast-lane daemon shares this exact ledger
    path (`odte_fast_lane.ledger_path`) — could pass its own check and place against the SAME
    single-use lease. Losing the claim is now a refusal, never a place."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    from execution.odte_pre_place_guard import consume_then
    now = datetime.now(timezone.utc)
    ledger = str(tmp_path / "consumed_leases.json")
    lease = {"lease_id": "L2", "option_id": "opt", "symbol": "IWM", "direction": "bullish",
             "quantity": 1, "max_limit_price": 1.0, "max_debit": 100.0,
             "issued_at": now.isoformat(),
             "expires_at": (now + timedelta(seconds=40)).isoformat()}
    order = {"option_id": "opt", "symbol": "IWM", "quantity": 1, "price": 0.9,
             "position_effect": "open", "type": "limit", "direction": "debit"}
    placed = []

    async def place():
        placed.append(1)
        return {"ok": True}

    async def both():
        return (await consume_then(order, lease, ledger_path=ledger, now=now, place=place),
                await consume_then(order, lease, ledger_path=ledger, now=now, place=place))

    first, second = asyncio.run(both())
    assert first[0]["allowed"] is True and first[1] is not None
    assert second[0]["allowed"] is False
    assert "lease_already_consumed" in second[0]["reasons"]
    assert len(placed) == 1, "one single-use lease produced more than one placement"
