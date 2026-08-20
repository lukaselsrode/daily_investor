"""Direct Robinhood MCP client for the 0DTE fast lane — the no-model-turn broker transport.

The two-lane architecture (2026-08-04) quarantines the LLM to arming intents; this client is how
the deterministic daemon talks to the broker with ZERO model turns in the latency window. It
speaks the SAME sanctioned agentic MCP lane Hermes uses (streamable HTTP + Bearer token), so
nothing here bypasses broker-side controls — it only bypasses the 5-40s LLM turns that killed the
2026-08-04 lease by 2.2 seconds.

Facts this design is built on (verified 2026-08-04):
  * Auth is a plain Bearer token from ``~/.hermes/mcp-tokens/robinhood.json`` with an epoch
    ``expires_at`` (~7-day life, MANUAL ``hermes mcp reauth`` only — the Aug 1-2 "blackout" was
    an expired token). The client RE-READS the file on every (re)connect and NEVER refreshes it.
  * The MCP wire is fast (p50 0.21s over 2051 observed calls); concurrent sessions are safe.
  * Robinhood ships schema-violating responses (the ``get_accounts``/``unsettled_funds``
    precedent), so client-side output-schema validation MUST be defanged after ``list_tools`` —
    an SDK raise on a SUCCESSFUL ``place_option_order`` would be catastrophic.
  * ``place_option_order`` takes a ``ref_id`` idempotency key: on a timeout the SAME ref_id is
    re-sent, never a fresh one — worst case is one order, never two.

Token-policy split: ``preflight()`` (default >= 48h remaining) is the DAEMON's liveness gate for
going live; ``connect()`` itself only requires an unexpired token — a client that refused to
reconnect mid-session would strand a live position that needs an exit.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from data.odte_snapshot_build import (
    PENDING_ORDER_STATES,
    broker_rows,
    build_broker_snapshot,
)

try:                            # the vendored SDK error type for JSON-RPC-level refusals
    from mcp.shared.exceptions import MCPError as McpProtocolError
except ImportError:             # pragma: no cover — tests run without needing the real SDK
    class McpProtocolError(Exception):  # type: ignore[no-redef]
        pass


class _NeverRaised(BaseException):
    """py3.10 stand-in: nothing ever raises this, so the except clauses are inert there."""
    exceptions: tuple[BaseException, ...] = ()


# BaseExceptionGroup is a builtin only from py3.11 (the repo floor is 3.10) — resolve via
# builtins so the module lints/runs on both.
_ExcGroup: type[BaseException] = getattr(builtins, "BaseExceptionGroup", _NeverRaised)

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

DEFAULT_TOKEN_PATH = "~/.hermes/mcp-tokens/robinhood.json"
DEFAULT_MCP_URL = "https://agent.robinhood.com/mcp/trading"
TOKEN_PATH_ENV = "ODTE_MCP_TOKEN_PATH"
MCP_URL_ENV = "ODTE_MCP_URL"

TOKEN_MIN_HOURS = 48.0          # daemon go-live gate; connect() itself only needs unexpired
DEFAULT_CALL_TIMEOUT_SECONDS = 8.0

# Tool names exactly as served by the Robinhood MCP server (mirrors the Hermes include list).
TOOL_PORTFOLIO = "get_portfolio"
TOOL_OPTION_POSITIONS = "get_option_positions"
TOOL_OPTION_ORDERS = "get_option_orders"
TOOL_EQUITY_QUOTES = "get_equity_quotes"
TOOL_EQUITY_HISTORICALS = "get_equity_historicals"
TOOL_OPTION_QUOTES = "get_option_quotes"
TOOL_REVIEW_OPTION_ORDER = "review_option_order"
TOOL_PLACE_OPTION_ORDER = "place_option_order"
TOOL_CANCEL_OPTION_ORDER = "cancel_option_order"

EQUITY_QUOTE_BATCH_MAX = 20     # above 20 the server drops `closes` from quote responses
EQUITY_HISTORICALS_BATCH_MAX = 10   # server cap: up to 10 symbols per historicals call




class McpAuthStale(RuntimeError):
    """Token missing/expired/inside the reauth horizon — the fix is `hermes mcp reauth robinhood`."""


class McpTransportError(RuntimeError):
    """Transport failed twice (original attempt + one reconnect-retry)."""


class McpToolError(RuntimeError):
    """The server answered with isError — the tool itself refused."""


def _unwrap_exception_group(exc: BaseException) -> Exception:
    """anyio task groups wrap the real cause in (Base)ExceptionGroup — unwrap for honest errors
    (re-implemented locally so the daemon imports zero Hermes code)."""
    while isinstance(exc, _ExcGroup) and getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]  # type: ignore[attr-defined]
    return exc if isinstance(exc, Exception) else RuntimeError(str(exc))


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def normalize_order_placement_row(value: Any) -> dict:
    """Recover an order row from Robinhood's inconsistent placement envelopes.

    The live server has returned direct dicts, nested ``data``/``order`` dicts, a
    stringified JSON payload, and ``{"result": "<json>"}``. Placement is already a
    side effect, so callers must recover the identity instead of crashing and
    accidentally submitting the same close again on the next tick.
    """
    current = value
    for _ in range(8):
        if isinstance(current, str):
            text = current.strip()
            try:
                current = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                if len(text) == 36 and text.count("-") == 4:
                    return {"id": text}
                return {}
            continue
        if isinstance(current, list):
            current = current[0] if current else None
            continue
        if not isinstance(current, dict):
            return {}
        if current.get("id") or current.get("order_id"):
            return current
        nested = next(
            (current.get(key) for key in
             ("data", "order", "result", "structuredContent", "orders", "results")
             if current.get(key) is not None),
            None,
        )
        if nested is None or nested is current:
            return {}
        current = nested
    return {}


class OdteMcpClient:
    """One long-lived MCP session with typed broker helpers. All calls are async.

    ``session_factory`` (tests) is an async callable returning a ClientSession-compatible object;
    when provided, no network transport is opened and close is a no-op for the session itself.
    """

    def __init__(self, token_path: str | None = None, url: str | None = None,
                 timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
                 session_factory: Any = None) -> None:
        self.token_path = os.path.expanduser(
            token_path or os.environ.get(TOKEN_PATH_ENV) or DEFAULT_TOKEN_PATH)
        self.url = url or os.environ.get(MCP_URL_ENV) or DEFAULT_MCP_URL
        self.timeout = timeout
        self.session: Any = None
        self.last_preflight: dict | None = None
        self.reconnects = 0
        self._session_factory = session_factory
        self._stack: AsyncExitStack | None = None

    # --- token preflight (sync, no network) ----------------------------------------------------

    def preflight(self, min_hours: float = TOKEN_MIN_HOURS, now: float | None = None) -> dict:
        """Read the token FILE (never the token value into logs) and enforce the reauth horizon.
        Raises McpAuthStale below `min_hours` remaining. Returns metadata only."""
        now = time.time() if now is None else now
        try:
            with open(self.token_path) as fh:
                token = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise McpAuthStale(f"token file unreadable at {self.token_path}: {exc}") from exc
        if not token.get("access_token"):
            raise McpAuthStale(f"no access_token in {self.token_path}")
        expires_at = _num(token.get("expires_at"))
        hours_remaining = ((expires_at - now) / 3600.0) if expires_at is not None else None
        if hours_remaining is None or hours_remaining < min_hours:
            raise McpAuthStale(
                f"robinhood MCP token has {hours_remaining if hours_remaining is None else round(hours_remaining, 1)}h "
                f"remaining (< {min_hours}h) — run `hermes mcp reauth robinhood`")
        self.last_preflight = {
            "token_path": self.token_path,
            "expires_at_iso": datetime.fromtimestamp(expires_at, tz=timezone.utc)
            .isoformat(timespec="seconds"),
            "hours_remaining": round(hours_remaining, 1),
            "checked_at": datetime.fromtimestamp(now, tz=timezone.utc)
            .isoformat(timespec="seconds"),
        }
        return self.last_preflight

    def _bearer(self) -> str:
        with open(self.token_path) as fh:
            return str(json.load(fh)["access_token"])

    # --- connection lifecycle ------------------------------------------------------------------

    async def connect(self) -> None:
        """(Re)connect: re-read the token file, open the transport, initialize, list_tools, and
        DEFANG client-side output-schema validation."""
        await self.aclose()
        # Unexpired is enough to CONNECT (a live position must always be exitable); the >=48h
        # go-live horizon is enforced by the daemon via preflight(TOKEN_MIN_HOURS).
        self.preflight(min_hours=0.0)
        if self._session_factory is not None:
            self.session = await self._session_factory()
        else:                                               # pragma: no cover — network path
            from mcp import ClientSession
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

            self._stack = AsyncExitStack()
            # NO explicit timeout object: mcp 2.x vendors httpx as `httpx2`, so a caller-built
            # httpx.Timeout leaks a foreign type into anyio's fail_after (float + Timeout crash,
            # seen live 2026-08-04). Per-call deadlines come from asyncio.wait_for in call().
            http_client = create_mcp_http_client(
                headers={"Authorization": f"Bearer {self._bearer()}"})
            await self._stack.enter_async_context(http_client)
            read, write = await self._stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client))
            self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        await self.session.list_tools()
        # THE DEFANG: Robinhood ships schema-violating responses; a client-side validation raise
        # on a successful order call would be catastrophic. Structural, not best-effort.
        if hasattr(self.session, "_tool_output_schemas"):
            self.session._tool_output_schemas = {}

    async def aclose(self) -> None:
        if self._stack is not None:                          # pragma: no cover — network path
            try:
                await self._stack.aclose()
            except Exception as exc:
                logger.debug("mcp transport close error swallowed: %s", exc)
            self._stack = None
        self.session = None

    # --- core call with one reconnect-retry ----------------------------------------------------

    async def call(self, tool: str, arguments: dict | None = None,
                   timeout: float | None = None) -> Any:
        """Call a tool and extract its payload (structuredContent first, then stringified JSON,
        then raw text). ONE immediate reconnect-and-retry on transport error/timeout — place
        stays idempotent because its ref_id is inside `arguments` and is re-sent verbatim.
        No backoff loops: the second failure raises (a lease window has no time for retries)."""
        if self.session is None:
            await self.connect()
        timeout = self.timeout if timeout is None else timeout
        try:
            result = await asyncio.wait_for(self.session.call_tool(tool, arguments or {}),
                                            timeout=timeout)
        except McpProtocolError as exc:
            # A JSON-RPC error (invalid params, unknown tool) is a SEMANTIC refusal — retrying
            # the identical call cannot help and must not burn lease time.
            raise McpToolError(f"{tool}: {exc}") from exc
        except (TimeoutError, OSError, ConnectionError, RuntimeError, _ExcGroup) as exc:
            cause = _unwrap_exception_group(exc)
            logger.warning("mcp call %s failed (%s) — one reconnect-retry", tool, cause)
            self.reconnects += 1
            try:
                await self.connect()
                result = await asyncio.wait_for(self.session.call_tool(tool, arguments or {}),
                                                timeout=timeout)
            except McpProtocolError as exc2:
                raise McpToolError(f"{tool}: {exc2}") from exc2
            except (TimeoutError, OSError, ConnectionError, RuntimeError,
                    _ExcGroup) as exc2:
                raise McpTransportError(
                    f"{tool} failed twice: {_unwrap_exception_group(exc2)}") from exc2
        if getattr(result, "isError", False):
            raise McpToolError(f"{tool}: {self._text_of(result) or 'tool returned isError'}")
        return self._extract(result)

    @staticmethod
    def _text_of(result: Any) -> str:
        return "\n".join(str(getattr(c, "text", "") or "")
                         for c in getattr(result, "content", []) or [])

    @classmethod
    def _extract(cls, result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        text = cls._text_of(result)
        if text:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return text
        return None

    # --- typed market-data helpers -------------------------------------------------------------

    async def equity_historicals(self, symbols: list[str], start_time: str,
                                 interval: str = "5minute", end_time: str | None = None,
                                 bounds: str | None = None) -> Any:
        """Bars since `start_time` (RFC3339 UTC, REQUIRED by the server — there is no `span`
        param; discovered live 2026-08-04). <=10 symbols per call."""
        args: dict[str, Any] = {"symbols": list(symbols)[:EQUITY_HISTORICALS_BATCH_MAX],
                                "start_time": start_time, "interval": interval}
        if end_time:
            args["end_time"] = end_time
        if bounds:
            args["bounds"] = bounds
        return await self.call(TOOL_EQUITY_HISTORICALS, args)

    async def equity_quotes(self, symbols: list[str]) -> list[Any]:
        """Quotes for any number of symbols, batched under the server's cap."""
        out: list[Any] = []
        symbols = list(symbols)
        for i in range(0, len(symbols), EQUITY_QUOTE_BATCH_MAX):
            payload = await self.call(TOOL_EQUITY_QUOTES,
                                      {"symbols": symbols[i:i + EQUITY_QUOTE_BATCH_MAX]})
            rows = broker_rows(payload)
            out.extend(rows if rows else [payload])
        return out

    async def option_quote_by_id(self, option_id: str) -> Any:
        # The server calls them instrument_ids; the repo's option_id IS the instrument UUID.
        payload = await self.call(TOOL_OPTION_QUOTES, {"instrument_ids": [str(option_id)]})
        rows = broker_rows(payload)
        return rows[0] if rows else payload

    # --- broker truth (the authorize_entry snapshot contract) ----------------------------------

    async def broker_truth(self, account_number: str,
                           now: datetime | None = None) -> dict:
        """Compose portfolio + option positions + option orders into the broker-snapshot key
        contract authorize_entry sizes against. The three reads run concurrently."""
        now = now or datetime.now(timezone.utc)
        portfolio, positions, orders = await asyncio.gather(
            self.call(TOOL_PORTFOLIO, {"account_number": account_number}),
            self.call(TOOL_OPTION_POSITIONS, {"account_number": account_number}),
            self.call(TOOL_OPTION_ORDERS, {"account_number": account_number}),
        )
        # Fetch here, compose in `data`. This method is the ONLY caller that can reach the broker,
        # so the composition had no business being welded to it — `odte_build_snapshot --out-broker`
        # needs the same logic against saved payloads, and a second implementation is how the two
        # drifted within hours of being duplicated (the copy lost `cash_available_for_trading`,
        # `submitted` and `pending_cancelled`).
        return build_broker_snapshot(portfolio, positions, orders,
                                     account_number=account_number, now=now,
                                     source="odte_mcp_client.broker_truth")

    # --- order lane ----------------------------------------------------------------------------

    @staticmethod
    def build_order_args(*, account_number: str, option_id: str, quantity: int,
                         limit_price: float, position_effect: str = "open") -> dict:
        """The exact single-leg order shape the v5 skill templates pinned (quantity/price as
        strings; direction follows position_effect)."""
        opening = position_effect == "open"
        return {
            "account_number": str(account_number),
            "direction": "debit" if opening else "credit",
            "type": "limit",
            "quantity": str(int(quantity)),
            "price": f"{limit_price:.2f}",
            "time_in_force": "gfd",
            "legs": [{"option_id": str(option_id), "side": "buy" if opening else "sell",
                      "position_effect": position_effect, "ratio_quantity": 1}],
        }

    async def review_option_order(self, order_args: dict) -> Any:
        """The ONE review before place — its alerts are the only place broker warnings surface."""
        return await self.call(TOOL_REVIEW_OPTION_ORDER, dict(order_args))

    async def place_option_order(self, order_args: dict, ref_id: str) -> Any:
        """Place with the caller-minted ref_id (UUID). A timeout inside call() re-sends the SAME
        ref_id — the broker dedupes, so worst case is one order, never two."""
        return await self.call(TOOL_PLACE_OPTION_ORDER, {**order_args, "ref_id": str(ref_id)})

    async def open_option_orders(self, account_number: str) -> list[dict]:
        """All currently-working option orders (pending/partial/pending-cancel states)."""
        payload = await self.call(TOOL_OPTION_ORDERS, {"account_number": str(account_number)})
        return [r for r in broker_rows(payload)
                if str(r.get("state") or r.get("status") or "").strip().lower()
                in PENDING_ORDER_STATES]

    async def cancel_option_order(self, order_id: str, account_number: str) -> Any:
        # account_number is REQUIRED by the server schema (mismatches are rejected).
        return await self.call(TOOL_CANCEL_OPTION_ORDER,
                               {"account_number": str(account_number),
                                "order_id": str(order_id)})

    async def order_state(self, order_id: str, account_number: str) -> Any:
        """Fresh broker truth for one order via get_option_orders' order_id filter."""
        payload = await self.call(TOOL_OPTION_ORDERS,
                                  {"account_number": str(account_number),
                                   "order_id": str(order_id)})
        rows = broker_rows(payload)
        return rows[0] if rows else payload
