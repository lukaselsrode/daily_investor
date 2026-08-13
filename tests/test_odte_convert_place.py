"""tests/test_odte_convert_place.py — the slow lane's in-process lease consumption.

No broker, no network, no LLM: the MCP client is faked. What matters here is not that an order is
transmitted but WHEN the lease is claimed relative to the place, and that every failure leaves the
lease unconsumed rather than half-used.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from execution.odte_convert_place import place_converted, place_limit_for

NOW = datetime(2026, 8, 13, 15, 54, 15, tzinfo=timezone.utc)
ACCT = "435050133"


def _lease(**over):
    d = {"lease_id": "232ff1290e46aa01", "option_id": "opt-iwm-303p", "symbol": "IWM",
         "option_type": "put", "strike_price": 303.0, "expiration_date": "2026-08-13",
         "quantity": 1, "max_limit_price": 0.50, "max_debit": 50.0,
         "expires_at": (NOW + timedelta(seconds=60)).isoformat()}
    d.update(over)
    return d


def _payload(**over):
    d = {"converted": True, "lease": _lease(), "contract": {"ask": 0.46, "bid": 0.44}}
    d.update(over)
    return d


class FakeClient:
    """Records call ORDER — the whole point is that the ledger claim precedes the place."""

    def __init__(self, ledger_path=None, fail_place=False):
        self.calls = []
        self.ledger_path = ledger_path
        self.fail_place = fail_place
        self.ledger_at_place = None

    def build_order_args(self, *, account_number, option_id, quantity, limit_price):
        self.calls.append("build")
        return {"account_number": account_number, "quantity": str(quantity),
                "price": f"{limit_price:.2f}", "direction": "debit",
                "legs": [{"option_id": option_id, "side": "buy", "position_effect": "open",
                          "ratio_quantity": 1}]}

    async def review_option_order(self, order_args):
        self.calls.append("review")
        return {"alerts": []}

    async def place_option_order(self, order_args, ref_id):
        self.calls.append("place")
        if self.ledger_path:                     # snapshot the ledger AT the moment of placing
            try:
                self.ledger_at_place = json.loads(open(self.ledger_path).read())
            except Exception:
                self.ledger_at_place = None
        if self.fail_place:
            raise RuntimeError("broker exploded")
        return {"data": {"id": "order-abc-123"}}

    async def close(self):
        self.calls.append("close")


def _run(payload, tmp_path, client, now=NOW):
    ledger = str(tmp_path / "consumed_leases.json")
    if not os.path.exists(ledger):
        (tmp_path / "consumed_leases.json").write_text("[]")
    client.ledger_path = ledger
    return asyncio.run(place_converted(payload, account_number=ACCT, ledger_path=ledger,
                                       client=client, now=now)), ledger


# --- the ordering property this module exists to provide ---------------------------------------

def test_lease_is_claimed_BEFORE_the_order_is_placed(tmp_path):
    """The slow lane records the ledger claim AFTER the place today (odte_order_guard.py:448), so
    between placing and the first guard poll a second place would pass the ledger check.
    `consume_then` claims first; a crash in between burns the lease, never the reverse."""
    c = FakeClient()
    report, ledger = _run(_payload(), tmp_path, c)
    assert report["placed"] is True
    # No "close": the module only closes a client it CREATED, never a caller-supplied one.
    assert c.calls == ["build", "review", "place"]
    assert "232ff1290e46aa01" in (c.ledger_at_place or []), "lease was NOT claimed before placing"


def test_exactly_one_review_never_a_re_review(tmp_path):
    """Three re-reviews missed a lease by 2.2s on 2026-08-04."""
    c = FakeClient()
    _run(_payload(), tmp_path, c)
    assert c.calls.count("review") == 1


# --- fail closed --------------------------------------------------------------------------------

def test_expired_lease_is_never_placed(tmp_path):
    c = FakeClient()
    stale = _payload(lease=_lease(expires_at=(NOW - timedelta(seconds=1)).isoformat()))
    report, _ = _run(stale, tmp_path, c)
    assert report["placed"] is False
    assert "lease_expired_before_place" in report["reasons"]
    assert "place" not in c.calls


def test_unconverted_payload_places_nothing(tmp_path):
    c = FakeClient()
    report, _ = _run(_payload(converted=False), tmp_path, c)
    assert report["placed"] is False and report["reasons"] == ["not_converted"]
    assert c.calls == []


def test_no_priceable_quote_stands_down(tmp_path):
    c = FakeClient()
    report, _ = _run(_payload(lease=_lease(max_limit_price=None), contract={}), tmp_path, c)
    assert report["placed"] is False and "no_priceable_quote" in report["reasons"]
    assert "place" not in c.calls


def test_a_broker_exception_never_escapes(tmp_path):
    """A raise here would abort the controller tick mid-conversion."""
    c = FakeClient(fail_place=True)
    report, _ = _run(_payload(), tmp_path, c)
    assert report["placed"] is False
    assert any(r.startswith("place_error:") for r in report["reasons"])


def test_already_consumed_lease_is_refused(tmp_path):
    (tmp_path / "consumed_leases.json").write_text(json.dumps(["232ff1290e46aa01"]))
    c = FakeClient()
    report, _ = _run(_payload(), tmp_path, c)
    assert report["placed"] is False
    assert "lease_already_consumed" in " ".join(report["reasons"])
    assert "place" not in c.calls


# --- the limit rule -----------------------------------------------------------------------------

def test_limit_is_the_ask_capped_by_the_lease_ceiling():
    assert place_limit_for({"max_limit_price": 1.17}, {"ask": 1.10}) == 1.10   # ask wins
    assert place_limit_for({"max_limit_price": 1.17}, {"ask": 1.30}) == 1.17   # ceiling caps
    assert place_limit_for({"max_limit_price": 1.17}, {}) == 1.17              # ceiling fallback
    assert place_limit_for({}, {}) is None                                     # nothing to price


def test_limit_never_exceeds_the_lease_ceiling(tmp_path):
    c = FakeClient()
    report, _ = _run(_payload(contract={"ask": 99.0}), tmp_path, c)
    assert report["limit_price"] == 0.50
