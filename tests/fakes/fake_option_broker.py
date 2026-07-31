"""In-memory option-broker simulator for execution-safety integration tests. TESTS ONLY.

No network, no real broker, no money. Simulates the submit/pending/cancel/fill lifecycle of one or
more single-leg option orders so the lease + order-guard layer can be exercised against races
(cancel-vs-fill, delayed fill, duplicate submission) deterministically.

The fake records EVERY controller-invoked broker method in ``calls`` (name + kwargs, in order), so
tests can prove prohibited calls (review/place on a refused authorization, duplicate placements)
were NEVER made. Simulation controls (``sim_fill``, ``sim_reject_cancel``) model the exchange and
are deliberately NOT recorded as controller calls.

BROKER TRUTH WINS: ``cancel_order`` on an already-filled order does not un-fill it — it returns the
filled truth, exactly like the real race.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


class FakeOptionBroker:
    """Deterministic fake broker. Every controller-facing method self-records into ``calls``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self._seq = 0

    # --- internals -----------------------------------------------------------------------------

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append({"method": method, **kwargs})

    def calls_of(self, method: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method]

    # --- controller-facing methods (ALL recorded) ---------------------------------------------

    def review_order(self, **contract: Any) -> dict[str, Any]:
        """Pre-placement review. Recorded; always passes (the fake reviews nothing)."""
        self._record("review_order", **contract)
        return {"ok": True, "reviewed": dict(contract)}

    def place_order(self, *, symbol: str, option_id: str, option_type: str, strike_price: float,
                    expiration_date: str, quantity: int, limit_price: float,
                    submitted_at: str) -> dict[str, Any]:
        """Submit a limit order. Returns the broker order-truth dict (status 'pending')."""
        self._record("place_order", symbol=symbol, option_id=option_id, quantity=quantity,
                     limit_price=limit_price, submitted_at=submitted_at)
        self._seq += 1
        ref = f"fake-order-{self._seq}"
        order = {
            "order_ref": ref, "status": "pending",
            "symbol": symbol, "option_id": option_id, "option_type": option_type,
            "strike_price": strike_price, "expiration_date": expiration_date,
            "quantity": quantity, "limit_price": limit_price,
            "submitted_at": submitted_at, "filled_at": None,
        }
        self.orders[ref] = order
        return dict(order)

    def cancel_order(self, order_ref: str) -> dict[str, Any]:
        """Request a cancel. BROKER TRUTH WINS: an already-filled order stays filled."""
        self._record("cancel_order", order_ref=order_ref)
        order = self.orders.get(order_ref)
        if order is None:
            return {"order_ref": order_ref, "status": "none"}
        if order["status"] == "filled":
            return dict(order)          # the race: cancel arrived after the fill
        order["status"] = "cancelled"
        return dict(order)

    def order_status(self, order_ref: str) -> dict[str, Any]:
        """Fresh broker order truth for one order."""
        self._record("order_status", order_ref=order_ref)
        order = self.orders.get(order_ref)
        return dict(order) if order else {"order_ref": order_ref, "status": "none"}

    def open_orders(self) -> list[dict[str, Any]]:
        self._record("open_orders")
        return [dict(o) for o in self.orders.values() if o["status"] == "pending"]

    # --- simulation controls (the exchange; NOT controller calls, NOT recorded) ----------------

    def sim_fill(self, order_ref: str, filled_at: str | datetime) -> None:
        """Exchange-side fill of a pending order at ``filled_at``."""
        order = self.orders[order_ref]
        if order["status"] == "pending":
            order["status"] = "filled"
            order["filled_at"] = (filled_at.isoformat(timespec="seconds")
                                  if isinstance(filled_at, datetime) else str(filled_at))
