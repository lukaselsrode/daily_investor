"""tests/test_odte_mcp_client.py — direct MCP broker client against the scripted fake session.

No network, no real broker, no real tokens (fixture tokens are fake strings). Pins the transport
contracts the fast lane depends on: the output-schema DEFANG after list_tools, payload extraction
(structuredContent / stringified-JSON text / raw text), the 48h token preflight vs the
unexpired-only connect gate, exception-group unwrapping, the single reconnect-retry, SAME-ref_id
resend on place timeout, quote batching under the server cap, and the broker_truth composition
that feeds authorize_entry.
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from fakes.fake_mcp_session import FakeMcpSession, SchemaViolating, Timeout

import execution.odte_mcp_client as mc


def _token_file(tmp_path, hours_remaining: float) -> str:
    path = tmp_path / "robinhood.json"
    path.write_text(json.dumps({"access_token": "fake-token-for-tests-only",
                                "expires_at": time.time() + hours_remaining * 3600.0}))
    return str(path)


def _client(tmp_path, sessions, hours_remaining: float = 100.0) -> mc.OdteMcpClient:
    """Client wired to pop fake sessions from `sessions` on each (re)connect."""
    remaining = list(sessions)

    async def factory():
        return remaining.pop(0)

    return mc.OdteMcpClient(token_path=_token_file(tmp_path, hours_remaining),
                            session_factory=factory)


def _run(coro):
    return asyncio.run(coro)


# --- token preflight -------------------------------------------------------------------------

def test_preflight_boundaries(tmp_path):
    ok = mc.OdteMcpClient(token_path=_token_file(tmp_path, mc.TOKEN_MIN_HOURS + 1))
    meta = ok.preflight()
    assert meta["hours_remaining"] > mc.TOKEN_MIN_HOURS
    assert "access_token" not in json.dumps(meta)              # metadata only, never the token
    stale = mc.OdteMcpClient(token_path=_token_file(tmp_path, mc.TOKEN_MIN_HOURS - 1))
    with pytest.raises(mc.McpAuthStale):
        stale.preflight()


def test_connect_needs_only_unexpired_token(tmp_path):
    # A 1h-left token must still CONNECT (a live position needs its exit lane) even though the
    # daemon's 48h go-live preflight refuses it.
    session = FakeMcpSession()
    session.queue("get_portfolio", {"buying_power": 350.0})
    client = _client(tmp_path, [session], hours_remaining=1.0)
    assert _run(client.call(mc.TOOL_PORTFOLIO)) == {"buying_power": 350.0}
    with pytest.raises(mc.McpAuthStale):
        client.preflight()
    expired = _client(tmp_path, [FakeMcpSession()], hours_remaining=-0.1)
    with pytest.raises(mc.McpAuthStale):
        _run(expired.connect())


def test_missing_token_file_is_auth_stale(tmp_path):
    client = mc.OdteMcpClient(token_path=str(tmp_path / "nope.json"))
    with pytest.raises(mc.McpAuthStale):
        client.preflight()


# --- defang + extraction ---------------------------------------------------------------------

def test_connect_defangs_output_schema_validation(tmp_path):
    session = FakeMcpSession()
    session.queue("get_portfolio", SchemaViolating({"buying_power": 348.16}))
    client = _client(tmp_path, [session])
    # Without the defang the SchemaViolating marker raises; with it the payload flows through.
    assert _run(client.call(mc.TOOL_PORTFOLIO)) == {"buying_power": 348.16}
    assert session._tool_output_schemas == {}
    assert session.initialized and session.list_tools_count == 1


def test_extraction_structured_then_json_text_then_raw(tmp_path):
    session = FakeMcpSession()
    session.queue("get_portfolio", {"structured": True}, '{"from_text": 1}', "plain prose")
    client = _client(tmp_path, [session])

    async def scenario():
        assert await client.call(mc.TOOL_PORTFOLIO) == {"structured": True}
        assert await client.call(mc.TOOL_PORTFOLIO) == {"from_text": 1}
        assert await client.call(mc.TOOL_PORTFOLIO) == "plain prose"
    _run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "ord-1"},
        {"data": {"id": "ord-1"}},
        {"data": {"order": {"id": "ord-1"}}},
        {"result": '{"data":{"id":"ord-1"}}'},
        '{"data":{"order":{"order_id":"ord-1"}}}',
        "11111111-2222-3333-4444-555555555555",
    ],
)
def test_normalize_order_placement_row_recovers_live_envelopes(payload):
    row = mc.normalize_order_placement_row(payload)
    assert row.get("id") == "ord-1" or row.get("order_id") == "ord-1" or row.get("id") == payload


def test_is_error_result_raises_tool_error(tmp_path):
    from fakes.fake_mcp_session import FakeToolResult
    session = FakeMcpSession()
    session.queue("place_option_order", FakeToolResult(text="insufficient buying power",
                                                       isError=True))
    client = _client(tmp_path, [session])
    with pytest.raises(mc.McpToolError, match="insufficient buying power"):
        _run(client.call(mc.TOOL_PLACE_OPTION_ORDER, {}))


# --- reconnect-once + unwrap -----------------------------------------------------------------

def test_transport_error_reconnects_once_and_succeeds(tmp_path):
    s1, s2 = FakeMcpSession(), FakeMcpSession()
    s1.queue("get_portfolio", ConnectionError("wire dropped"))
    s2.queue("get_portfolio", {"buying_power": 350.0})
    client = _client(tmp_path, [s1, s2])
    assert _run(client.call(mc.TOOL_PORTFOLIO)) == {"buying_power": 350.0}
    assert client.reconnects == 1
    assert s2._tool_output_schemas == {}                       # the NEW session is defanged too


def test_second_failure_raises_transport_error_with_unwrapped_cause(tmp_path):
    s1, s2 = FakeMcpSession(), FakeMcpSession()
    s1.queue("get_portfolio",
             BaseExceptionGroup("taskgroup", [ValueError("401 Unauthorized")]))
    s2.queue("get_portfolio",
             BaseExceptionGroup("taskgroup", [ValueError("401 Unauthorized")]))
    client = _client(tmp_path, [s1, s2])
    with pytest.raises(mc.McpTransportError, match="401 Unauthorized"):
        _run(client.call(mc.TOOL_PORTFOLIO))


def test_unwrap_exception_group_nested():
    inner = ValueError("root cause")
    wrapped = BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [inner])])
    assert mc._unwrap_exception_group(wrapped) is inner


# --- ref_id idempotency ----------------------------------------------------------------------

def test_place_timeout_resends_same_ref_id(tmp_path):
    s1, s2 = FakeMcpSession(), FakeMcpSession()
    s1.queue("place_option_order", Timeout())
    s2.queue("place_option_order", {"order_id": "ord-1", "state": "confirmed"})
    client = _client(tmp_path, [s1, s2])
    args = client.build_order_args(account_number="435050133", option_id="opt-756c",
                                   quantity=1, limit_price=0.64)
    result = _run(client.place_option_order(args, ref_id="11111111-2222-3333-4444-555555555555"))
    assert result["order_id"] == "ord-1"
    sent = s1.calls_of("place_option_order") + s2.calls_of("place_option_order")
    assert len(sent) == 2
    assert (sent[0]["arguments"]["ref_id"] == sent[1]["arguments"]["ref_id"]
            == "11111111-2222-3333-4444-555555555555")


def test_order_args_match_the_v5_templates():
    open_args = mc.OdteMcpClient.build_order_args(account_number="435050133",
                                                  option_id="opt-1", quantity=1,
                                                  limit_price=0.64)
    assert open_args["direction"] == "debit" and open_args["type"] == "limit"
    assert open_args["quantity"] == "1" and open_args["price"] == "0.64"
    assert open_args["legs"] == [{"option_id": "opt-1", "side": "buy",
                                  "position_effect": "open", "ratio_quantity": 1}]
    close_args = mc.OdteMcpClient.build_order_args(account_number="435050133",
                                                   option_id="opt-1", quantity=1,
                                                   limit_price=0.70, position_effect="close")
    assert close_args["direction"] == "credit"
    assert close_args["legs"][0]["side"] == "sell"
    assert close_args["legs"][0]["position_effect"] == "close"


# --- market-data helpers ---------------------------------------------------------------------

def test_equity_quotes_batches_under_server_cap(tmp_path):
    session = FakeMcpSession()
    symbols = [f"S{i:02d}" for i in range(mc.EQUITY_QUOTE_BATCH_MAX + 5)]
    session.queue("get_equity_quotes",
                  {"results": [{"symbol": s} for s in symbols[:mc.EQUITY_QUOTE_BATCH_MAX]]},
                  {"results": [{"symbol": s} for s in symbols[mc.EQUITY_QUOTE_BATCH_MAX:]]})
    client = _client(tmp_path, [session])
    rows = _run(client.equity_quotes(symbols))
    assert len(rows) == len(symbols)
    calls = session.calls_of("get_equity_quotes")
    assert len(calls) == 2
    assert all(len(c["arguments"]["symbols"]) <= mc.EQUITY_QUOTE_BATCH_MAX for c in calls)


def test_option_quote_by_id_unwraps_single_row(tmp_path):
    session = FakeMcpSession()
    session.queue("get_option_quotes", {"results": [{"bid_price": 0.61, "ask_price": 0.63}]})
    client = _client(tmp_path, [session])
    quote = _run(client.option_quote_by_id("opt-756c"))
    assert quote == {"bid_price": 0.61, "ask_price": 0.63}
    # The server calls them instrument_ids (discovered live 2026-08-04); option_id IS that UUID.
    assert session.calls_of("get_option_quotes")[0]["arguments"] == {
        "instrument_ids": ["opt-756c"]}


def test_equity_historicals_uses_start_time_never_span(tmp_path):
    # The live schema REQUIRES start_time and rejects any 'span' property (discovered 2026-08-04:
    # `invalid params: unexpected additional properties ["span"]`).
    session = FakeMcpSession()
    session.queue("get_equity_historicals", {"data": {"results": []}})
    client = _client(tmp_path, [session])
    _run(client.equity_historicals(["SPY", "QQQ"], start_time="2026-08-05T13:30:00Z"))
    args = session.calls_of("get_equity_historicals")[0]["arguments"]
    assert args == {"symbols": ["SPY", "QQQ"], "start_time": "2026-08-05T13:30:00Z",
                    "interval": "5minute"}
    assert "span" not in args


def test_protocol_error_is_tool_error_not_retried(tmp_path):
    session = FakeMcpSession()
    session.queue("get_equity_historicals",
                  mc.McpProtocolError(code=-32602,
                                      message="invalid params: unexpected additional properties"))
    client = _client(tmp_path, [session])
    with pytest.raises(mc.McpToolError, match="invalid params"):
        _run(client.call(mc.TOOL_EQUITY_HISTORICALS, {}))
    assert client.reconnects == 0                              # semantic refusal: never retried


def test_cancel_and_order_state_require_account_number(tmp_path):
    session = FakeMcpSession()
    session.queue("cancel_option_order", {"data": {"state": "pending_cancelled"}})
    session.queue("get_option_orders", {"data": {"orders": [{"id": "o1", "state": "filled"}]}})
    client = _client(tmp_path, [session])

    async def scenario():
        await client.cancel_option_order("o1", account_number="435050133")
        return await client.order_state("o1", account_number="435050133")
    row = _run(scenario())
    assert row == {"id": "o1", "state": "filled"}
    assert session.calls_of("cancel_option_order")[0]["arguments"] == {
        "account_number": "435050133", "order_id": "o1"}
    assert session.calls_of("get_option_orders")[0]["arguments"] == {
        "account_number": "435050133", "order_id": "o1"}


# --- broker truth ----------------------------------------------------------------------------

def test_broker_truth_composes_authorize_entry_contract(tmp_path):
    # Payload shapes mirror the LIVE server responses probed 2026-08-04: portfolio nests
    # data.buying_power.buying_power (a string); positions under data.positions; orders under
    # data.orders with `state`/`id` keys. day_trades_left does not exist in the live payload —
    # None is correct (authorize_entry only vetoes when it is present and <= 0).
    from datetime import datetime, timezone
    now = datetime(2026, 8, 5, 14, 31, 2, tzinfo=timezone.utc)
    session = FakeMcpSession()
    session.queue("get_portfolio",
                  {"data": {"cash": "12.03",
                            "buying_power": {"buying_power": "348.16",
                                             "unleveraged_buying_power": "348.16"}},
                   "guide": "prose"})
    session.queue("get_option_positions",
                  {"data": {"positions": [{"option_id": "a", "quantity": "1.0000"},
                                          {"option_id": "b", "quantity": "0.0000"}]},
                   "guide": "prose"})
    session.queue("get_option_orders",
                  {"data": {"orders": [{"id": "o1", "state": "queued",
                                        "created_at": "2026-08-05T14:00:00Z"},
                                       {"id": "o2", "state": "filled",
                                        "created_at": "2026-08-05T13:00:00Z"},
                                       {"id": "o3", "state": "cancelled",
                                        "created_at": "2026-08-01T13:00:00Z"}]},
                   "guide": "prose"})
    client = _client(tmp_path, [session])
    truth = _run(client.broker_truth("435050133", now=now))
    assert truth["buying_power"] == 348.16
    assert truth["day_trades_left"] is None
    assert truth["nonzero_option_positions_count"] == 1
    assert truth["open_option_orders_count"] == 1              # queued only
    assert truth["today_option_orders_count"] == 2             # the two Aug-5 orders
    assert truth["account_number"] == "435050133"
    assert truth["as_of"] == now.isoformat(timespec="seconds")
