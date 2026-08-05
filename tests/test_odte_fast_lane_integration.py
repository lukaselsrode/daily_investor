"""tests/test_odte_fast_lane_integration.py — full fast-lane lifecycle against the fake broker.

LIVE mode (stage entries_live), one scripted session: arm → trigger fires → convert mints a real
lease → ONE review → consumed-ledger write PROVABLY precedes the place call (asserted by
snapshotting the ledger inside the fake's place handler) → PENDING → fill inside the lease →
active_trade.json + entry_fill → MANAGING → bid-memory giveback harvest → sell-to-close placed
→ exit fill → order_closed with realized P/L → IDLE. Also pins the exit reprice loop and the
partial-fill branch. No network, no real broker, no LLM.
"""
import asyncio
import json
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fakes.fake_mcp_session import FakeMcpSession
from test_odte_fast_lane import (
    NOW,
    _historicals_payload,
    _intent,
    _iso,
    _option_quote_payload,
    _portfolio_payload,
    _quotes_payload,
)

import execution.odte_fast_lane as fl
import execution.odte_mcp_client as mc


class LedgerSnappingSession(FakeMcpSession):
    """Snapshots the consumed-leases ledger at the exact moment place_option_order arrives —
    the only airtight way to prove consume-BEFORE-place ordering."""

    def __init__(self, ledger_path):
        super().__init__()
        self.ledger_path = ledger_path
        self.ledger_at_place: list[list[str]] = []

    async def call_tool(self, name, arguments=None):
        if name == "place_option_order":
            try:
                ids = json.loads(open(self.ledger_path).read())
            except OSError:
                ids = []
            self.ledger_at_place.append(ids)
        return await super().call_tool(name, arguments)


def _order_row(order_id, state, *, price="0.63", processed="0", created, last_tx=None,
               option_id="opt-756c"):
    row = {"id": order_id, "state": state, "chain_symbol": "SPY", "price": price,
           "quantity": "1", "processed_quantity": processed, "created_at": created,
           "legs": [{"option_id": option_id}]}
    if last_tx:
        row["last_transaction_at"] = last_tx
    if state == "filled":
        row["average_price"] = price
    return row


def _orders_payload(*rows):
    return {"data": {"orders": list(rows)}, "guide": ""}


def _setup_live(tmp_path):
    (tmp_path / fl.STAGE_FILENAME).write_text(json.dumps({"stage": "entries_live"}))
    (tmp_path / "armed_intents.json").write_text(
        json.dumps({"schema_version": 1, "intents": [_intent()]}))
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"access_token": "fake", "expires_at": time.time() + 999999}))
    session = LedgerSnappingSession(str(tmp_path / "consumed_leases.json"))

    async def factory():
        return session
    client = mc.OdteMcpClient(token_path=str(token), session_factory=factory)
    daemon = fl.FastLaneDaemon(client, account_number="435050133", state_dir=tmp_path)
    assert daemon.mode == fl.MODE_LIVE
    return daemon, session


def _tick(daemon, now):
    return asyncio.run(daemon.tick(now))


def _events(tmp_path):
    path = tmp_path / "decision_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_full_lifecycle_entry_to_giveback_exit(tmp_path):
    daemon, session = _setup_live(tmp_path)
    t0 = NOW

    # ── tick 1: startup reconcile + tape + trigger fires + convert + review + place ──────────
    session.queue("get_option_orders", _orders_payload())              # startup: nothing open
    session.queue("get_equity_historicals", _historicals_payload())
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_quotes", _option_quote_payload(bid=0.61, ask=0.63))
    session.queue("get_portfolio", _portfolio_payload())
    session.queue("get_option_positions", {"data": {"positions": []}, "guide": ""})
    session.queue("get_option_orders", _orders_payload())              # broker_truth
    session.queue("review_option_order", {"data": {"alerts": []}, "guide": ""})
    session.queue("place_option_order",
                  {"data": {"id": "ord-entry", "state": "confirmed"}, "guide": ""})
    status = _tick(daemon, t0)
    assert status["state"] == fl.PENDING and status["counts"]["placed"] == 1

    # ONE review, then place — and the ledger already held the lease when place arrived.
    assert len(session.calls_of("review_option_order")) == 1
    assert len(session.calls_of("place_option_order")) == 1
    lease_id = daemon.lease["lease_id"]
    assert session.ledger_at_place == [[lease_id]]
    place_args = session.calls_of("place_option_order")[0]["arguments"]
    assert place_args["legs"][0]["option_id"] == "opt-756c"
    assert place_args["quantity"] == "1" and place_args["ref_id"]
    assert float(place_args["price"]) <= daemon.lease["max_limit_price"]
    # Convert journaled the lease + entry decision in the REAL journal (live mode).
    assert any(e["event_type"] == "execution_lease_issued" for e in _events(tmp_path))

    # ── tick 2: fill inside the lease → active_trade + entry_fill → MANAGING ────────────────
    t1 = t0 + timedelta(seconds=5)
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_orders", _orders_payload(
        _order_row("ord-entry", "filled", price="0.61", processed="1",
                   created=_iso(t0), last_tx=_iso(t0 + timedelta(seconds=3)))))
    status = _tick(daemon, t1)
    assert status["state"] == fl.MANAGING
    plan = json.loads((tmp_path / "active_trade.json").read_text())
    assert plan["status"] == "open" and plan["entry_price"] == 0.61
    assert plan["execution_lease_id"] == lease_id
    assert plan["profit_rules"] == {"take_profit_pct": 0.60, "strong_exit_pct": 0.80}
    fills = [e for e in _events(tmp_path) if e["event_type"] == "entry_fill"]
    assert len(fills) == 1 and fills[0]["fill_price"] == 0.61

    # ── tick 3: bid runs to 0.76 — bid-memory arms, nothing fires at the peak ───────────────
    t2 = t1 + timedelta(seconds=4)
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_quotes", _option_quote_payload(bid=0.76, ask=0.78))
    status = _tick(daemon, t2)
    assert status["state"] == fl.MANAGING and daemon.exit_order is None
    plan = json.loads((tmp_path / "active_trade.json").read_text())
    assert plan["management"]["best_seen_bid"] == 0.76           # persisted for the next poll

    # ── tick 4: fade to 0.67 breaches the giveback floor → sell-to-close placed ─────────────
    t3 = t2 + timedelta(seconds=4)
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_quotes", _option_quote_payload(bid=0.67, ask=0.69))
    session.queue("place_option_order",
                  {"data": {"id": "ord-exit", "state": "confirmed"}, "guide": ""})
    status = _tick(daemon, t3)
    assert daemon.exit_order and daemon.exit_order["order_id"] == "ord-exit"
    assert daemon.exit_order["trigger_type"] == "BID_MEMORY_PROTECT"
    exit_args = session.calls_of("place_option_order")[1]["arguments"]
    assert exit_args["direction"] == "credit" and exit_args["price"] == "0.67"
    assert exit_args["legs"][0]["position_effect"] == "close"
    # The close never burned a second ledger entry.
    assert json.loads((tmp_path / "consumed_leases.json").read_text()) == [lease_id]

    # ── tick 5: exit fills → order_closed with realized P/L → IDLE ──────────────────────────
    t4 = t3 + timedelta(seconds=4)
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_orders", _orders_payload(
        _order_row("ord-exit", "filled", price="0.67", processed="1",
                   created=_iso(t3), last_tx=_iso(t4))))
    status = _tick(daemon, t4)
    assert status["state"] == fl.IDLE
    plan = json.loads((tmp_path / "active_trade.json").read_text())
    assert plan["status"] == "closed" and plan["exit_price"] == 0.67
    assert plan["realized_pnl"] == 6.0                          # (0.67-0.61)*100
    closed = [e for e in _events(tmp_path) if e["event_type"] == "order_closed"]
    assert len(closed) == 1 and closed[0]["trigger_type"] == "BID_MEMORY_PROTECT"
    # Never a cancel in the whole lifecycle.
    assert not session.calls_of("cancel_option_order")


def test_exit_reprice_cancels_and_replaces_at_fresh_bid(tmp_path):
    daemon, session = _setup_live(tmp_path)
    daemon._started = True
    daemon.state = fl.MANAGING
    (tmp_path / "active_trade.json").write_text(json.dumps(
        {"status": "open", "underlying": "SPY", "option_id": "opt-756c", "entry_price": 0.61,
         "quantity": 1, "mode": "scalp", "management": {"best_seen_bid": 0.76}}))
    daemon.exit_order = {"order_id": "ord-exit", "limit": 0.70, "trigger_type": "TAKE_PROFIT",
                         "placed_at": _iso(NOW - timedelta(seconds=15)), "replacements": 0}
    session.queue("get_equity_historicals", _historicals_payload())
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_orders", _orders_payload(
        _order_row("ord-exit", "queued", price="0.70", created=_iso(NOW))))
    session.queue("get_option_quotes", _option_quote_payload(bid=0.66, ask=0.68))
    session.queue("cancel_option_order", {"data": {"state": "cancelled"}, "guide": ""})
    session.queue("place_option_order",
                  {"data": {"id": "ord-exit-2", "state": "confirmed"}, "guide": ""})
    _tick(daemon, NOW)
    assert session.calls_of("cancel_option_order")
    assert daemon.exit_order["order_id"] == "ord-exit-2"
    assert daemon.exit_order["replacements"] == 1
    args = session.calls_of("place_option_order")[0]["arguments"]
    assert args["price"] == "0.66"                              # repriced AT the fresh bid


def test_partial_fill_past_lease_cancels_remainder_and_manages_filled_qty(tmp_path):
    daemon, session = _setup_live(tmp_path)
    daemon._started = True
    daemon.state = fl.PENDING
    daemon.lease = {"lease_id": "lease-p", "symbol": "SPY", "option_id": "opt-756c",
                    "option_type": "call", "strike_price": 756.0,
                    "expiration_date": "2026-08-05", "quantity": 2,
                    "max_limit_price": 0.73, "max_debit": 150.0,
                    "issued_at": _iso(NOW - timedelta(seconds=70)),
                    "expires_at": _iso(NOW - timedelta(seconds=10))}
    daemon.pending_order = {"order_id": "ord-p", "ref_id": "r", "limit_price": 0.63,
                            "placed_at": _iso(NOW - timedelta(seconds=65)), "intent_id": "x"}
    session.queue("get_equity_historicals", _historicals_payload())
    session.queue("get_equity_quotes", _quotes_payload())
    session.queue("get_option_orders", _orders_payload(
        _order_row("ord-p", "partially_filled", price="0.63", processed="1",
                   created=_iso(NOW - timedelta(seconds=65)))))
    session.queue("cancel_option_order", {"data": {"state": "pending_cancelled"}, "guide": ""})
    status = _tick(daemon, NOW)
    assert session.calls_of("cancel_option_order")              # the remainder is cancelled
    assert status["state"] == fl.MANAGING                       # the filled contract is managed
    plan = json.loads((tmp_path / "active_trade.json").read_text())
    assert plan["quantity"] == 1                                # the FILLED qty, not the ordered


def test_session_runner_once_and_stale_token_downgrade(tmp_path):
    # run_fast_lane(once=True) drives one tick; a <48h token forces shadow regardless of stage.
    (tmp_path / fl.STAGE_FILENAME).write_text(json.dumps({"stage": "entries_live"}))
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"access_token": "fake",
                                 "expires_at": time.time() + 3600.0}))  # 1h left
    session = FakeMcpSession()

    async def factory():
        return session
    client = mc.OdteMcpClient(token_path=str(token), session_factory=factory)
    session.queue("get_option_orders", _orders_payload())
    session.queue("get_equity_historicals", _historicals_payload())
    session.queue("get_equity_quotes", _quotes_payload())
    status = asyncio.run(fl.run_fast_lane(client, account_number="435050133",
                                          state_dir=tmp_path, mode="live", once=True))
    assert status["mode"] == "shadow"                           # downgraded, loudly
    shadow_path = tmp_path / "shadow" / "decision_journal.jsonl"
    events = [json.loads(line) for line in shadow_path.read_text().splitlines()]
    starts = [e for e in events if e["event_type"] == "shadow_session_start"]
    assert starts and starts[0]["effective_mode"] == "shadow"
