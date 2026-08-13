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

import pytest

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
    # `_contract` is a TEST carrier only: run_convert's real payload has no contract key, which is
    # exactly the bug this file now pins. The helper forwards it as the explicit argument.
    d = {"converted": True, "lease": _lease(), "_contract": {"ask": 0.46, "bid": 0.44}}
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

    async def aclose(self):
        # `aclose`, not `close`: OdteMcpClient has no close(), and naming it wrong here would let
        # the module's AttributeError-swallowing finally hide a leaked MCP session.
        self.calls.append("aclose")


def _run(payload, tmp_path, client, now=NOW):
    ledger = str(tmp_path / "consumed_leases.json")
    if not os.path.exists(ledger):
        (tmp_path / "consumed_leases.json").write_text("[]")
    client.ledger_path = ledger
    return asyncio.run(place_converted(payload, account_number=ACCT, ledger_path=ledger,
                                       contract=payload.get("_contract"), client=client,
                                       now=now)), ledger


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
    report, _ = _run(_payload(lease=_lease(max_limit_price=None), _contract={}), tmp_path, c)
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
    report, _ = _run(_payload(_contract={"ask": 99.0}), tmp_path, c)
    assert report["limit_price"] == 0.50


# --- the CLI seam (2026-08-13) ------------------------------------------------------------------
# The placement lives in `execution` and the decision in `data`, because the import-linter contract
# "data must not import from higher layers" forbids data -> execution. The CLI is the only place
# allowed to touch both, so these pin that wiring rather than the module in isolation.

def _cli_main_module():
    """`cli.main` resolves to the re-exported main() FUNCTION, not the module — import it properly."""
    import importlib
    return importlib.import_module("cli.main")


def test_place_flag_is_opt_in_and_needs_an_account(monkeypatch, capsys):
    """Without --account (and no env var) the CLI refuses rather than guessing an account."""
    m = _cli_main_module()
    monkeypatch.delenv("ODTE_ACCOUNT_NUMBER", raising=False)
    monkeypatch.setattr("data.odte_convert.run_convert",
                        lambda **kw: {"converted": True, "lease": {"lease_id": "x"}})
    monkeypatch.setattr("data.odte_convert.render_markdown", lambda p: "")
    with pytest.raises(SystemExit) as exc:
        m._cmd_odte_convert(["--place"])
    assert exc.value.code == 3
    assert "ODTE_ACCOUNT_NUMBER" in capsys.readouterr().err


def test_without_the_flag_no_placement_is_attempted(monkeypatch, capsys):
    """The whole safety story is that omitting --place leaves path 1 byte-identical."""
    m = _cli_main_module()
    called = {"n": 0}

    async def _boom(*a, **k):                      # must never be awaited
        called["n"] += 1
        return {}

    monkeypatch.setattr("data.odte_convert.run_convert",
                        lambda **kw: {"converted": True, "lease": {"lease_id": "x"}})
    monkeypatch.setattr("data.odte_convert.render_markdown", lambda p: "CONVERTED")
    monkeypatch.setattr("execution.odte_convert_place.place_converted", _boom)
    m._cmd_odte_convert([])                        # no --place
    assert called["n"] == 0
    assert "placement" not in capsys.readouterr().out


def test_missing_contract_does_not_silently_pay_the_ceiling(tmp_path):
    """THE bug this argument exists for. `run_convert`'s payload has no "contract" key, so reading
    payload["contract"] always yielded {} and the limit fell back to the lease ceiling — paying the
    top of the chase band on every fill instead of the ask. The caller must pass it."""
    c = FakeClient()
    payload = {"converted": True, "lease": _lease()}           # no contract anywhere
    ledger = str(tmp_path / "consumed_leases.json")
    (tmp_path / "consumed_leases.json").write_text("[]")
    c.ledger_path = ledger
    report = asyncio.run(place_converted(payload, account_number=ACCT, ledger_path=ledger,
                                         contract={"ask": 0.46}, client=c, now=NOW))
    assert report["limit_price"] == 0.46, "paid the lease ceiling instead of the ask"


def test_the_ask_is_used_when_it_is_below_the_ceiling(tmp_path):
    c = FakeClient()
    report, _ = _run(_payload(_contract={"ask": 0.41}), tmp_path, c)
    assert report["limit_price"] == 0.41 and report["limit_price"] < 0.50


def test_a_client_it_created_is_torn_down_with_aclose(monkeypatch, tmp_path):
    """OdteMcpClient exposes `aclose`, never `close`. Calling the wrong name raised
    AttributeError straight into the best-effort handler, silently leaking the session."""
    from execution.odte_mcp_client import OdteMcpClient
    assert hasattr(OdteMcpClient, "aclose") and not hasattr(OdteMcpClient, "close")

    made = FakeClient()
    ledger = str(tmp_path / "consumed_leases.json")
    (tmp_path / "consumed_leases.json").write_text("[]")
    made.ledger_path = ledger
    monkeypatch.setattr("execution.odte_convert_place.OdteMcpClient", lambda *a, **k: made)
    asyncio.run(place_converted({"converted": True, "lease": _lease()},
                                account_number=ACCT, ledger_path=ledger,
                                contract={"ask": 0.46}, now=NOW))       # client=None -> we own it
    assert "aclose" in made.calls, "a client we created was never torn down"


def test_the_order_placed_event_cannot_be_miscounted(tmp_path):
    """`order_placed` is a NEW vocabulary term (EVENT_TYPES is a partial legacy declaration, not a
    whitelist — `entry_fill` and `no_trade_decision` are absent from it too). This week produced
    four spellings of one close and three readers that each understood a different subset, so a new
    term has to be proven inert before it ships: informational only, never a fill, never a trade."""
    import data.odte_journal as oj
    jp = str(tmp_path / "j.jsonl")
    oj.append_decision_journal({"event_type": "order_placed", "underlying": "IWM",
                                "option_id": "x", "lease_id": "L1", "order_id": "o1",
                                "limit_price": 0.46},
                               source="odte_convert_place", event_type="order_placed",
                               journal_path=jp)
    evs = oj.read_events(jp)
    assert evs[0]["event_type"] == "order_placed"
    assert oj._classify_day_stream(evs[0]) == "controller_events"   # NOT "trades"
    assert oj.summarize(evs)["n_trades"] == 0                       # creates no trade row
    assert oj.summarize(evs)["n_closed"] == 0
    assert oj.daily_trade_budget(evs)["trades_today"] == 0          # never burns a budget slot
