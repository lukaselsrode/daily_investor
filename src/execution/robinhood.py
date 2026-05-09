"""
execution/robinhood.py — RobinhoodBroker: live robin_stocks execution adapter.

All rb.* calls go through _retry() for rate-limit / 429 handling.
Confirmation prompts (AUTO_APPROVE) are NOT here — that is a CLI / portfolio-manager concern.
"""

from __future__ import annotations

import logging

from .base import BrokerAdapter, OrderResult

logger = logging.getLogger(__name__)


class RobinhoodBroker(BrokerAdapter):
    """
    Live broker backed by robin_stocks.

    Migrated from main.py:
      _place_fractional_buy      → buy_fractional
      _place_whole_share_buy     → buy_whole
      _place_sell                → sell  (without AUTO_APPROVE gate — caller's responsibility)
      get_available_cash         → get_cash  (subtracts pending committed orders)
      get_portfolio_value        → get_portfolio_value
      get_current_positions      → get_holdings
      rb.get_all_open_stock_orders → get_open_orders
    """

    def __init__(self) -> None:
        try:
            import robin_stocks.robinhood as rb
            self._rb = rb
        except ImportError as e:
            raise RuntimeError("robin_stocks not installed") from e

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------

    def _retry(self, fn, *args, max_retries: int = 3, **kwargs):
        import random
        import time
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "too many requests" in msg or "throttle" in msg:
                    wait = (2 ** attempt) * (0.5 + random.random())
                    logger.warning(
                        f"Rate-limited on {fn.__name__} (attempt {attempt + 1}/{max_retries}), "
                        f"sleeping {wait:.1f}s"
                    )
                    time.sleep(wait)
                else:
                    raise
        return fn(*args, **kwargs)  # final attempt — let it raise

    # ------------------------------------------------------------------
    # Buy orders
    # ------------------------------------------------------------------

    def buy_fractional(self, symbol: str, amount: float) -> OrderResult:
        try:
            res = self._retry(
                self._rb.orders.order_buy_fractional_by_price, symbol, amount
            )
        except Exception as e:
            logger.error(f"{symbol}: fractional buy exception: {e}")
            return OrderResult(symbol, "buy", amount, 0.0, False, None, "error", str(e))

        if res and res.get("id"):
            logger.info(
                f"Buy {symbol} ${amount:.2f}: state={res.get('state')}, id={res['id'][:8]}..."
            )
            return OrderResult(
                symbol=symbol,
                side="buy",
                amount=amount,
                quantity=0.0,  # fractional qty not returned synchronously
                success=True,
                order_id=res["id"],
                state=res.get("state", "confirmed"),
            )

        detail = (res.get("detail") or repr(res)) if res else "None response"
        logger.warning(f"{symbol}: fractional buy rejected — {detail}")
        return OrderResult(symbol, "buy", amount, 0.0, False, None, "rejected", detail)

    def buy_whole(self, symbol: str, quantity: int) -> OrderResult:
        try:
            res = self._retry(self._rb.orders.order_buy_market, symbol, quantity)
        except Exception as e:
            logger.error(f"{symbol}: whole-share market order failed: {e}")
            return OrderResult(symbol, "buy", 0.0, float(quantity), False, None, "error", str(e))

        if res and res.get("id"):
            logger.info(
                f"Buy {symbol} {quantity} share(s): state={res.get('state')}, id={res['id'][:8]}..."
            )
            return OrderResult(
                symbol=symbol,
                side="buy",
                amount=0.0,  # market order — dollar amount unknown until fill
                quantity=float(quantity),
                success=True,
                order_id=res["id"],
                state=res.get("state", "confirmed"),
            )

        detail = (res.get("detail") if res else "None response") or "rejected"
        logger.warning(f"{symbol}: whole-share order rejected — {detail}")
        return OrderResult(symbol, "buy", 0.0, float(quantity), False, None, "rejected", detail)

    # ------------------------------------------------------------------
    # Sell orders
    # ------------------------------------------------------------------

    def sell(self, symbol: str, quantity: float) -> OrderResult:
        # Re-fetch live position to guard against a stale holdings snapshot.
        try:
            live_positions = self._rb.get_all_positions()
            live_pos = next(
                (p for p in live_positions if p.get("symbol") == symbol), None
            )
            live_qty = float(live_pos.get("quantity", 0)) if live_pos else 0.0
            if live_qty <= 0:
                logger.warning(
                    f"Skipping sell for {symbol}: position already closed (live qty={live_qty})"
                )
                return OrderResult(symbol, "sell", 0.0, 0.0, False, None, "skipped", "position already closed")
            quantity = live_qty  # use live quantity to avoid partial-fill mismatch
        except Exception as e:
            logger.warning(
                f"Could not verify live position for {symbol}: {e} — using cached quantity"
            )

        is_fractional = quantity != int(quantity)
        try:
            if is_fractional:
                res = self._retry(
                    self._rb.orders.order_sell_fractional_by_quantity, symbol, quantity
                )
            else:
                res = self._retry(
                    self._rb.order_sell_market, symbol, int(quantity), timeInForce="gfd"
                )
        except Exception as e:
            logger.error(f"Sell order exception for {symbol}: {e}")
            return OrderResult(symbol, "sell", 0.0, quantity, False, None, "error", str(e))

        if not res:
            logger.warning(f"Sell order returned None for {symbol}")
            return OrderResult(symbol, "sell", 0.0, quantity, False, None, "rejected", "None response")

        order_id = res.get("id")
        if not order_id:
            detail = res.get("detail") or repr(res)
            logger.error(f"Sell rejected for {symbol}: {detail}")
            return OrderResult(symbol, "sell", 0.0, quantity, False, None, "rejected", detail)

        logger.info(
            f"Sell order placed for {symbol}: qty={quantity}, "
            f"state={res.get('state')}, id={order_id[:8]}..."
        )
        return OrderResult(
            symbol=symbol,
            side="sell",
            amount=0.0,
            quantity=quantity,
            success=True,
            order_id=order_id,
            state=res.get("state", "confirmed"),
        )

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    def get_holdings(self) -> dict:
        try:
            return self._rb.build_holdings() or {}
        except Exception as e:
            logger.warning(f"Could not fetch holdings: {e}")
            return {}

    def get_cash(self) -> float:
        """Return cash minus committed-but-not-settled buy orders."""
        try:
            cash = float(self._rb.account.build_user_profile().get("cash", 0))
        except Exception as e:
            logger.warning(f"Could not fetch cash: {e}")
            return 0.0

        try:
            committed = 0.0
            for order in self._rb.orders.get_all_open_stock_orders():
                if order.get("side") != "buy":
                    continue
                if order.get("state") not in ("confirmed", "queued", "unconfirmed"):
                    continue
                order_type = order.get("type")
                if order_type == "market" and order.get("extended_hours", False):
                    continue
                if order_type == "market":
                    for field in ("executed_notional", "total_notional", "dollar_based_amount"):
                        nested = order.get(field) or {}
                        amt = nested.get("amount")
                        if amt:
                            committed += float(amt)
                            break
                elif order_type == "limit":
                    committed += float(order.get("quantity", 0)) * float(order.get("price", 0))

            available = cash - committed
        except Exception as e:
            logger.warning(f"Could not subtract pending orders from cash: {e}")
            available = cash

        logger.info(
            f"Cash: ${available:,.2f} available "
            f"(total=${cash:,.2f}, pending=${cash - available:,.2f})"
        )
        return max(0.0, available)

    def get_portfolio_value(self) -> float:
        try:
            return float(self._rb.account.build_user_profile().get("equity", 0) or 0)
        except Exception as e:
            logger.warning(f"Could not fetch portfolio value: {e}")
            return 0.0

    def get_open_orders(self) -> list[dict]:
        try:
            return list(self._rb.orders.get_all_open_stock_orders() or [])
        except Exception as e:
            logger.warning(f"Could not fetch open orders: {e}")
            return []
