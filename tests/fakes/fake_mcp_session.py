"""In-memory MCP session fake for fast-lane broker-client tests. TESTS ONLY.

No network, no real broker, no tokens. Mimics the surface of ``mcp.ClientSession`` that
``execution.odte_mcp_client.OdteMcpClient`` touches: ``list_tools()``, ``call_tool()``, and the
``_tool_output_schemas`` dict the client must DEFANG (Robinhood ships schema-violating responses;
client-side output validation raising on a SUCCESSFUL place would be catastrophic).

Follows the FakeOptionBroker pattern: every call is recorded in ``calls`` (tool + arguments, in
order) so tests can prove prohibited calls were never made and assert call ORDER (e.g. the
consumed-ledger write preceding place_option_order). Responses are scripted per tool with
``queue()``; special markers simulate the failure modes the client must survive:

  * ``Timeout()``              — raises TimeoutError (ref_id-resend tests)
  * ``SchemaViolating(data)``  — raises IF the client failed to defang ``_tool_output_schemas``,
                                 otherwise returns ``data`` normally
  * an Exception instance     — raised as-is (wrap in BaseExceptionGroup for unwrap tests)
  * a dict                    — returned as ``structuredContent``
  * a str                     — returned as text content (stringified-JSON extraction tests)
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any


class Timeout:
    """Marker: this call times out."""


class SchemaViolating:
    """Marker: this payload violates the tool's output schema — the SDK raises unless defanged."""

    def __init__(self, data: Any) -> None:
        self.data = data


class FakeToolResult:
    """Shape-compatible stand-in for mcp.types.CallToolResult."""

    def __init__(self, structuredContent: Any = None, text: str | None = None,
                 isError: bool = False) -> None:
        self.structuredContent = structuredContent
        self.content = [SimpleNamespace(type="text", text=text)] if text is not None else []
        self.isError = isError


class FakeMcpSession:
    """Deterministic fake MCP session. Every call_tool self-records into ``calls``."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.latency_s = latency_s
        self.initialized = False
        self.list_tools_count = 0
        self._queues: dict[str, list[Any]] = {}
        # Populated by list_tools() exactly like the real SDK; the client must CLEAR it.
        self._tool_output_schemas: dict[str, dict | None] = {}

    # --- scripting (the exchange side; not recorded) -------------------------------------------

    def queue(self, tool: str, *responses: Any) -> None:
        self._queues.setdefault(tool, []).extend(responses)

    def calls_of(self, tool: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["tool"] == tool]

    # --- ClientSession surface (recorded) ------------------------------------------------------

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> Any:
        self.list_tools_count += 1
        # The real SDK caches every listed tool's output schema for client-side validation.
        self._tool_output_schemas = {tool: {"type": "object"} for tool in self._queues}
        return SimpleNamespace(tools=[SimpleNamespace(name=t) for t in self._queues])

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        self.calls.append({"tool": name, "arguments": dict(arguments or {})})
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        queue = self._queues.get(name)
        if not queue:
            raise AssertionError(f"FakeMcpSession: no scripted response for tool '{name}'")
        response = queue.pop(0)
        if isinstance(response, Timeout):
            raise TimeoutError(f"fake timeout on {name}")
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, SchemaViolating):
            if self._tool_output_schemas:
                raise RuntimeError(f"output schema validation failed for {name} "
                                   "(client did not defang _tool_output_schemas)")
            response = response.data
        if isinstance(response, FakeToolResult):
            return response
        if isinstance(response, str):
            return FakeToolResult(text=response)
        return FakeToolResult(structuredContent=response)
