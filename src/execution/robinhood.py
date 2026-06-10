"""
execution/robinhood.py — RobinhoodBroker: live robin_stocks execution adapter.

All rb.* calls go through _retry() for rate-limit / 429 handling.
Confirmation prompts (AUTO_APPROVE) are NOT here — that is a CLI / portfolio-manager concern.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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
      rb.get_all_open_stock_orders → get_open_orders  (cached per-run)
      login()                    → login
      add_funds_to_account()     → add_funds
      _fetch_and_save_dividends() → get_dividends
      _enrich_holdings_with_created_at → enrich_holdings_created_at
    """

    def __init__(self) -> None:
        try:
            import robin_stocks.robinhood as rb
            self._rb = rb
        except ImportError as e:
            raise RuntimeError("robin_stocks not installed") from e
        self._orders_cache: list | None = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, username: str | None, password: str | None, mfa_secret: str | None = None) -> None:
        if not username or not password:
            raise ValueError("Missing required env vars: RB_ACCT and RB_CREDS must be set in .env")

        import pyotp
        mfa_code = None
        if mfa_secret:
            mfa_secret = mfa_secret.strip().replace(" ", "").replace("-", "").upper()
            # Base32 alphabet is A-Z and 2-7 ONLY. Name the offending characters so
            # the .env can actually be fixed — the generic pyotp error ("Non-base32
            # digit found") doesn't say what's wrong. Common causes: pasted a
            # recovery/backup code or hex string instead of the TOTP setup key, or
            # OCR-style confusions (0 vs O, 1 vs I, 8 vs B).
            _bad = sorted({c for c in mfa_secret if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"})
            if _bad:
                logger.error(
                    f"RB_MFA_SECRET contains non-base32 character(s) {_bad} — base32 "
                    f"uses only A-Z and 2-7. Use the TOTP *setup key* from Robinhood's "
                    f"authenticator-app enrollment (not a backup/recovery code). "
                    f"Login will rely on the cached session until this is fixed."
                )
            else:
                try:
                    mfa_code = pyotp.TOTP(mfa_secret).now()
                    logger.info("MFA code generated from RB_MFA_SECRET")
                except Exception as e:
                    logger.error(
                        f"MFA generation failed: {e} — check RB_MFA_SECRET in .env (must be plain base32)"
                    )

        try:
            self._rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
            logger.info("Logged in to Robinhood")
        except Exception as e:
            if "mfa_required" in str(e).lower() and not mfa_code:
                mfa_code = input("Enter MFA code: ").strip()
                self._rb.login(
                    username=username, password=password, mfa_code=mfa_code, store_session=True
                )
                logger.info("Logged in with manual MFA code")
            else:
                raise

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------

    def _retry(self, fn, *args, max_retries: int = 3, retry_on_none: bool = False, **kwargs):
        import random
        for attempt in range(max_retries):
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "too many requests" in msg or "throttle" in msg:
                    wait = (2 ** attempt) * (1.0 + random.random())
                    logger.warning(
                        f"Rate-limited on {fn.__name__} (attempt {attempt + 1}/{max_retries}), "
                        f"sleeping {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                raise
            if result is None and retry_on_none and attempt < max_retries - 1:
                wait = (2 ** attempt) * (1.0 + random.random())
                logger.warning(
                    f"{fn.__name__} returned None (possible 429) — "
                    f"attempt {attempt + 1}/{max_retries}, sleeping {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            return result
        return fn(*args, **kwargs)  # final attempt — let it raise

    # ------------------------------------------------------------------
    # Buy orders
    # ------------------------------------------------------------------

    def buy_fractional(self, symbol: str, amount: float) -> OrderResult:
        try:
            res = self._retry(
                self._rb.orders.order_buy_fractional_by_price, symbol, amount,
                retry_on_none=True,
            )
        except Exception as e:
            logger.error(f"{symbol}: fractional buy exception: {e}")
            return OrderResult(symbol, "buy", amount, 0.0, False, None, "error", str(e))
        finally:
            time.sleep(1.5)

        if res and res.get("id"):
            logger.info(
                f"Buy {symbol} ${amount:.2f}: state={res.get('state')}, id={res['id'][:8]}..."
            )
            return OrderResult(
                symbol=symbol,
                side="buy",
                amount=amount,
                quantity=0.0,
                success=True,
                order_id=res["id"],
                state=res.get("state", "confirmed"),
            )

        detail = (res.get("detail") or repr(res)) if res else "None response"
        logger.warning(f"{symbol}: fractional buy rejected — {detail}")
        return OrderResult(symbol, "buy", amount, 0.0, False, None, "rejected", detail)

    def buy_whole(self, symbol: str, quantity: int) -> OrderResult:
        try:
            res = self._retry(
                self._rb.orders.order_buy_market, symbol, quantity,
                retry_on_none=True,
            )
        except Exception as e:
            logger.error(f"{symbol}: whole-share market order failed: {e}")
            return OrderResult(symbol, "buy", 0.0, float(quantity), False, None, "error", str(e))
        finally:
            time.sleep(1.5)

        if res and res.get("id"):
            logger.info(
                f"Buy {symbol} {quantity} share(s): state={res.get('state')}, id={res['id'][:8]}..."
            )
            return OrderResult(
                symbol=symbol,
                side="buy",
                amount=0.0,
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
        # Raw position dicts carry an instrument URL, not a symbol, so resolve the
        # symbol's instrument URL first and match positions on it.
        try:
            inst_url: str | None = None
            try:
                instruments = self._rb.get_instruments_by_symbols(symbol) or []
                inst_url = next(
                    (
                        i["url"] for i in instruments
                        if i and i.get("url") and i.get("symbol") == symbol
                    ),
                    None,
                )
            except Exception as e:
                logger.warning(f"Instrument URL resolution failed for {symbol}: {e}")

            # Prefer open positions (cheaper; get_all_positions includes every
            # position ever traded, with qty 0 for closed ones).
            get_positions = (
                getattr(self._rb, "get_open_stock_positions", None)
                or self._rb.get_all_positions
            )
            live_positions = get_positions() or []
            live_pos = next(
                (
                    p for p in live_positions
                    if (inst_url is not None and p.get("instrument") == inst_url)
                    or p.get("symbol") == symbol  # fallback for sources that include it
                ),
                None,
            )
            if live_pos is None and inst_url is None:
                # Could not resolve the instrument and nothing matched — cannot
                # verify; fall through to the cached quantity rather than skip.
                raise RuntimeError("instrument URL unresolved and no position matched")
            live_qty = float(live_pos.get("quantity", 0)) if live_pos else 0.0
            if live_qty <= 0:
                logger.warning(
                    f"Skipping sell for {symbol}: position already closed (live qty={live_qty})"
                )
                return OrderResult(
                    symbol, "sell", 0.0, 0.0, False, None, "skipped", "position already closed"
                )
            quantity = min(quantity, live_qty)  # clamp trims; never sell more than held
        except Exception as e:
            logger.warning(
                f"Could not verify live position for {symbol}: {e} — using cached quantity"
            )

        is_fractional = quantity != int(quantity)
        try:
            if is_fractional:
                res = self._retry(
                    self._rb.orders.order_sell_fractional_by_quantity, symbol, quantity,
                    retry_on_none=True,
                )
            else:
                res = self._retry(
                    self._rb.order_sell_market, symbol, int(quantity), timeInForce="gfd",
                    retry_on_none=True,
                )
        except Exception as e:
            logger.error(f"Sell order exception for {symbol}: {e}")
            return OrderResult(symbol, "sell", 0.0, quantity, False, None, "error", str(e))
        finally:
            time.sleep(1.5)

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
            for order in self.get_open_orders():
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
        if getattr(self, "_orders_cache", None) is None:
            try:
                self._orders_cache = list(self._rb.orders.get_all_open_stock_orders() or [])
            except Exception as e:
                logger.warning(f"Could not fetch open orders: {e}")
                return []
        return self._orders_cache  # type: ignore[return-value]

    def clear_orders_cache(self) -> None:
        self._orders_cache = None

    def resolve_instrument_symbol(self, url: str) -> str | None:
        """Resolve an instrument URL to its ticker symbol (cached per adapter).

        Raw order/position payloads carry an `instrument` URL, never a `symbol`
        key — anything matching orders to symbols must go through here."""
        cache = getattr(self, "_instrument_symbol_cache", None)
        if cache is None:
            cache = self._instrument_symbol_cache = {}
        if url in cache:
            return cache[url]
        try:
            inst = self._rb.get_instrument_by_url(url)
            sym = (inst or {}).get("symbol") or None
        except Exception as e:
            logger.warning(f"Could not resolve instrument {url}: {e}")
            sym = None
        cache[url] = sym
        return sym

    # ------------------------------------------------------------------
    # Fund management
    # ------------------------------------------------------------------

    def add_funds(self, target_amount: float) -> None:
        """Initiate an ACH deposit for target_amount. Caller must handle confirmation prompt."""
        try:
            accounts = self._rb.get_linked_bank_accounts()
            ach = accounts[0].get("url") if accounts else None
            if not ach:
                logger.warning("No linked bank account found — cannot deposit")
                return
            resp = self._rb.deposit_funds_to_robinhood_account(ach, round(target_amount, 2))
            logger.info(f"Deposit requested: ${target_amount:,.2f} — state={resp.get('state')}")
        except Exception as e:
            logger.error(f"Deposit failed: {e}")

    # ------------------------------------------------------------------
    # Dividend data
    # ------------------------------------------------------------------

    def get_dividends(self) -> pd.DataFrame:
        """Fetch dividend history and resolve instrument URLs to symbols."""
        import pandas as pd

        dividends = self._rb.get_dividends() or []
        if not dividends:
            return pd.DataFrame()

        instrument_urls = list({d.get("instrument") for d in dividends if d.get("instrument")})
        url_to_symbol: dict[str, str] = {}
        for url in instrument_urls:
            try:
                inst = self._rb.get_instrument_by_url(url)
                if inst and inst.get("symbol"):
                    url_to_symbol[url] = inst["symbol"]
            except Exception:
                pass

        rows = []
        for d in dividends:
            url    = d.get("instrument", "")
            symbol = url_to_symbol.get(url, "")
            amount = float(d.get("amount") or 0)
            rows.append({
                "symbol":      symbol,
                "record_date": d.get("record_date", ""),
                "paid_at":     d.get("paid_at", ""),
                "state":       d.get("state", ""),
                "amount":      amount,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Holdings enrichment
    # ------------------------------------------------------------------

    def enrich_holdings_created_at(self, holdings: dict) -> None:
        """Add 'created_at' (position open date) to each holdings entry in-place.

        Two-phase lookup:
          Phase 1 — get_open_stock_positions(): fast, but Robinhood returns null
                     created_at for old / migrated positions.
          Phase 2 — get_all_stock_orders(): finds the earliest filled buy order
                     per symbol for anything Phase 1 missed.
        """
        if not holdings:
            return

        url_to_symbol: dict[str, str] = {}
        try:
            instruments = self._rb.get_instruments_by_symbols(list(holdings.keys())) or []
            for inst in instruments:
                if inst and inst.get("url") and inst.get("symbol"):
                    url_to_symbol[inst["url"]] = inst["symbol"]
            if not url_to_symbol:
                logger.warning("enrich_holdings_created_at: instrument resolution returned no data")
        except Exception as exc:
            logger.warning(f"Instrument URL resolution failed: {exc}")

        if url_to_symbol:
            try:
                positions = self._rb.get_open_stock_positions() or []
                phase1 = 0
                for pos in positions:
                    inst_url   = pos.get("instrument")
                    created_at = pos.get("created_at")
                    if not inst_url or not created_at:
                        continue
                    symbol = url_to_symbol.get(inst_url)
                    if symbol and symbol in holdings:
                        holdings[symbol]["created_at"] = created_at
                        phase1 += 1
                logger.info(
                    f"Phase 1 (positions): enriched {phase1}/{len(holdings)} holdings with open date"
                )
            except Exception as exc:
                logger.warning(f"Phase 1 (positions) enrichment failed: {exc}")

        missing: set[str] = {s for s in holdings if not holdings[s].get("created_at")}
        if not missing:
            return

        logger.info(
            f"Phase 2 (order history): resolving open date for "
            f"{len(missing)} symbol(s): {sorted(missing)}"
        )
        try:
            all_orders = self._rb.get_all_stock_orders() or []
            first_buy: dict[str, str] = {}
            for order in all_orders:
                if order.get("side") != "buy" or order.get("state") != "filled":
                    continue
                inst_url = order.get("instrument")
                if not inst_url:
                    continue
                sym = url_to_symbol.get(inst_url)
                if not sym or sym not in missing:
                    continue
                created = order.get("created_at")
                if not created:
                    continue
                if sym not in first_buy or created < first_buy[sym]:
                    first_buy[sym] = created

            phase2 = 0
            for sym, date_str in first_buy.items():
                if sym in holdings:
                    holdings[sym]["created_at"] = date_str
                    phase2 += 1

            if phase2:
                logger.info(
                    f"Phase 2: resolved {phase2}/{len(missing)} open dates from order history"
                )
            still_missing = missing - set(first_buy.keys())
            if still_missing:
                logger.warning(f"Phase 2: no buy orders found for {sorted(still_missing)}")
        except Exception as exc:
            logger.warning(f"Phase 2 (order history) enrichment failed: {exc}")
