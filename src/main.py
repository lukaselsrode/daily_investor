"""
main.py — Daily investment strategy entry point.

Responsibilities:
  - Robinhood login
  - Fund top-up
  - Buy cycle: pre-filter → risk-check → batch sentiment → execute orders
  - Sell cycle: hard/soft decision engine → batch sentiment (soft only) → execute
  - Iteration loop until cash exhausted or no more candidates

Changes in this revision:
  - Portfolio risk controls: position cap, sector cap, order cap, liquidity gate
  - Sell decision engine: hard vs soft sells, stop-loss, take-profit, weak-value, yield-trap
  - make_sales() refactored: hard sells execute immediately; soft sells optionally held by sentiment
  - make_buys() passes every buy through can_buy_symbol() before placing order
"""

import datetime
import json
import logging
import os
import sys
import time

import pandas as pd
import pyotp
import robin_stocks.robinhood as rb
import yfinance as yf
from dotenv import load_dotenv

from sentiment_analysis import get_batch_sentiment_recommendations, get_sentiment_recommendation
from source_data import get_data as generate_daily_undervalued_stocks
from util import (
    AUTO_APPROVE,
    BEAR_MARKET_PARAMS,
    CONFIDENCE_THRESHOLD,
    DATA_DIRECTORY,
    ETFS,
    HARVEST_PARAMS,
    INDEX_PCT,
    MAX_ITERATIONS,
    METRIC_KEYS,
    METRIC_THRESHOLD,
    RISK_LIMITS,
    SELL_RULES,
    SELL_SENTIMENT_OVERRIDE_CONFIDENCE,
    USE_SENTIMENT_ANALYSIS,
    WEEKLY_INVESTMENT,
    read_data_as_pd,
    safe_float,
    update_industry_valuations,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("investment_bot.log"),
    ],
)
logger = logging.getLogger("investment_bot")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login() -> None:
    username = os.getenv("RB_ACCT")
    password = os.getenv("RB_CREDS")
    if not username or not password:
        raise ValueError(
            "Missing required env vars: RB_ACCT and RB_CREDS must be set in .env"
        )
    mfa_code = None
    mfa_secret = os.getenv("RB_MFA_SECRET")
    if mfa_secret:
        # Strip whitespace, hyphens, and spaces — common copy-paste artefacts in TOTP secrets
        mfa_secret = mfa_secret.strip().replace(" ", "").replace("-", "").upper()
        try:
            mfa_code = pyotp.TOTP(mfa_secret).now()
            logger.info("MFA code generated from RB_MFA_SECRET")
        except Exception as e:
            logger.error(f"MFA generation failed: {e} — check RB_MFA_SECRET in .env (must be plain base32)")

    try:
        rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
        logger.info("Logged in to Robinhood")
    except Exception as e:
        if "mfa_required" in str(e).lower() and not mfa_code:
            mfa_code = input("Enter MFA code: ").strip()
            rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
            logger.info("Logged in with manual MFA code")
        else:
            raise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    if AUTO_APPROVE:
        logger.info(f"AUTO-APPROVED: {prompt}")
        return True
    return input(f"{prompt} [y/n] ").strip().lower() in ("y", "yes")


def get_available_cash() -> float:
    """Return cash minus committed-but-not-settled buy orders."""
    cash = float(rb.account.build_user_profile().get("cash", 0))

    try:
        committed = 0.0
        for order in rb.orders.get_all_open_stock_orders():
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


def add_funds_to_account() -> None:
    available = get_available_cash()
    if available >= WEEKLY_INVESTMENT:
        logger.info(f"Sufficient cash (${available:,.2f} ≥ ${WEEKLY_INVESTMENT:,.2f}) — no deposit needed")
        return

    needed = WEEKLY_INVESTMENT - available
    # Always require manual confirmation for deposits regardless of AUTO_APPROVE
    resp = input(
        f"Cash ${available:,.2f} < target ${WEEKLY_INVESTMENT:,.2f}. Deposit ${needed:,.2f}? [y/n] "
    ).strip().lower()
    if resp not in ("y", "yes"):
        return

    try:
        accounts = rb.get_linked_bank_accounts()
        ach = (accounts[0].get("url") if accounts else None)
        if not ach:
            logger.warning("No linked bank account found — cannot deposit")
            return
        resp = rb.deposit_funds_to_robinhood_account(ach, round(needed, 2))
        logger.info(f"Deposit requested: ${needed:,.2f} — state={resp.get('state')}")
    except Exception as e:
        logger.error(f"Deposit failed: {e}")


_PEAK_PRICES_CSV = os.path.join(DATA_DIRECTORY, "peak_prices.csv")


def _rb_call_with_retry(fn, *args, max_retries: int = 3, **kwargs):
    """Call a Robinhood API function with exponential back-off on 429 / rate-limit errors."""
    import random as _rand
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "too many requests" in msg or "throttle" in msg:
                wait = (2 ** attempt) * (0.5 + _rand.random())
                logger.warning(
                    f"Rate-limited on {fn.__name__} (attempt {attempt+1}/{max_retries}), "
                    f"sleeping {wait:.1f}s"
                )
                time.sleep(wait)
            else:
                raise
    return fn(*args, **kwargs)  # final attempt — let it raise


def _load_peak_prices() -> dict[str, float]:
    try:
        df = pd.read_csv(_PEAK_PRICES_CSV)
        if "symbol" in df.columns and "peak_price" in df.columns:
            return dict(zip(df["symbol"], df["peak_price"].astype(float)))
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not load peak prices: {e}")
    return {}


def _save_peak_prices(peaks: dict[str, float]) -> None:
    try:
        pd.DataFrame(
            [{"symbol": sym, "peak_price": price} for sym, price in peaks.items()]
        ).to_csv(_PEAK_PRICES_CSV, index=False)
    except Exception as e:
        logger.warning(f"Could not save peak prices: {e}")


def wipe_data() -> None:
    if not confirm("Wipe data directory?"):
        return
    for f in os.listdir(DATA_DIRECTORY):
        path = os.path.join(DATA_DIRECTORY, f)
        try:
            os.remove(path)
            logger.debug(f"Removed {path}")
        except Exception as e:
            logger.error(f"Could not remove {path}: {e}")
    logger.info("Data directory cleared")


def _is_bear_market_regime() -> bool:
    """Return True when SPY is below its 200-day MA and VIX > 25 (risk-off regime)."""
    try:
        ma_period = BEAR_MARKET_PARAMS["spy_ma_period"]
        spy = yf.Ticker("SPY").history(period="1y")["Close"]
        if len(spy) < ma_period:
            return False
        spy_price = float(spy.iloc[-1])
        spy_200ma = float(spy.rolling(ma_period).mean().iloc[-1])
        vix = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
        regime = spy_price < spy_200ma and vix > BEAR_MARKET_PARAMS["vix_threshold"]
        logger.info(
            f"Market regime: {'BEAR — stock buys suspended' if regime else 'normal'} "
            f"(SPY=${spy_price:.2f} vs 200MA=${spy_200ma:.2f}, VIX={vix:.1f})"
        )
        return regime
    except Exception as e:
        logger.warning(f"Regime check failed ({e}) — assuming normal market")
        return False


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------

def _place_fractional_buy(symbol: str, allocation: float) -> tuple[bool, str]:
    """Attempt a fractional buy order. Returns (success, detail_message)."""
    try:
        res = _rb_call_with_retry(rb.orders.order_buy_fractional_by_price, symbol, allocation)
    except Exception as e:
        return False, str(e)
    if res and res.get("id"):
        logger.info(f"Buy {symbol} ${allocation:.2f}: state={res.get('state')}, id={res['id'][:8]}...")
        return True, "ok"
    detail = (res.get("detail") or repr(res)) if res else "None response"
    logger.warning(f"{symbol}: fractional buy rejected — {detail}")
    return False, detail


def _place_whole_share_buy(symbol: str, quantity: int) -> bool:
    """Attempt a whole-share market buy. Returns True on success."""
    try:
        res = _rb_call_with_retry(rb.orders.order_buy_market, symbol, quantity)
    except Exception as e:
        logger.error(f"{symbol}: whole-share market order failed: {e}")
        return False
    if res and res.get("id"):
        logger.info(f"Buy {symbol} {quantity} share(s): state={res.get('state')}, id={res['id'][:8]}...")
        return True
    logger.warning(f"{symbol}: whole-share order rejected — {res.get('detail') if res else 'None'}")
    return False


def _update_local_exposures_after_buy(
    symbol: str,
    allocation: float,
    agg_df: "pd.DataFrame | None",
    sector_exposure: dict[str, float],
) -> None:
    """Mutate sector_exposure in-place after a successful buy to keep intra-run risk checks accurate."""
    if agg_df is None or agg_df.empty or "symbol" not in agg_df.columns:
        return
    row = agg_df[agg_df["symbol"] == symbol]
    if row.empty:
        return
    sector = str(row.iloc[0].get("sector") or "Unknown")
    sector_exposure[sector] = sector_exposure.get(sector, 0.0) + allocation


def _process_whole_share_queue(
    queue: list[tuple[str, float]],
    agg_df: "pd.DataFrame | None",
    sector_exposure: dict[str, float],
    purchased: list[str],
    failed: list[str],
    skipped: list[str],
    holdings: dict | None = None,
    portfolio_value: float = 0.0,
) -> None:
    """Execute whole-share fallback orders for symbols where fractional buy failed."""
    max_ws = RISK_LIMITS["max_whole_share_buys_per_run"]
    max_ws_mult = RISK_LIMITS["max_whole_share_allocation_multiplier"]
    ws_count = 0

    for idx, (symbol, alloc) in enumerate(queue):
        if ws_count >= max_ws:
            logger.info(f"Whole-share limit ({max_ws}) reached — remaining queue skipped")
            skipped.extend(sym for sym, _ in queue[idx:])
            break

        current_price: float | None = None
        if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
            row = agg_df[agg_df["symbol"] == symbol]
            if not row.empty:
                current_price = safe_float(row.iloc[0].get("current_price"))

        if not current_price or current_price <= 0:
            logger.warning(f"{symbol}: no price available for whole-share fallback — failing")
            failed.append(symbol)
            continue

        max_alloc = alloc * max_ws_mult
        if current_price > max_alloc:
            logger.warning(
                f"{symbol}: price ${current_price:.2f} > {max_ws_mult}× original alloc "
                f"${max_alloc:.2f} — skipping whole-share fallback"
            )
            skipped.append(symbol)
            continue

        # Re-run full risk checks — cash and sector exposure may have shifted
        # since the original buy attempt earlier in the same run.
        if holdings is not None:
            available_cash = get_available_cash() * (1 - INDEX_PCT)
            ok, reason, _ = can_buy_symbol(
                symbol, current_price, holdings, agg_df, portfolio_value, available_cash, sector_exposure
            )
            if not ok:
                logger.info(f"Whole-share {symbol} blocked by risk re-check: {reason}")
                skipped.append(symbol)
                continue

        if _place_whole_share_buy(symbol, 1):
            purchased.append(symbol)
            ws_count += 1
            _update_local_exposures_after_buy(symbol, current_price, agg_df, sector_exposure)
        else:
            failed.append(symbol)


def _place_sell(symbol: str, quantity: float) -> bool:
    # Re-fetch live position to guard against a stale holdings snapshot.
    # This matters when AUTO_APPROVE=False and the user takes time to confirm,
    # or when a pending order from a previous run already closed the position.
    try:
        live_positions = rb.get_all_positions()
        live_pos = next(
            (p for p in live_positions if p.get("symbol") == symbol), None
        )
        live_qty = float(live_pos.get("quantity", 0)) if live_pos else 0.0
        if live_qty <= 0:
            logger.warning(f"Skipping sell for {symbol}: position already closed (live qty={live_qty})")
            return False
        quantity = live_qty  # use live quantity to avoid partial-fill mismatch
    except Exception as e:
        logger.warning(f"Could not verify live position for {symbol}: {e} — using cached quantity")

    if not AUTO_APPROVE and not confirm(f"Sell {quantity} shares of {symbol}?"):
        logger.info(f"Sell cancelled for {symbol}")
        return False

    is_fractional = quantity != int(quantity)
    try:
        if is_fractional:
            res = _rb_call_with_retry(rb.orders.order_sell_fractional_by_quantity, symbol, quantity)
        else:
            res = _rb_call_with_retry(rb.order_sell_market, symbol, int(quantity), timeInForce="gfd")
    except Exception as e:
        logger.error(f"Sell order exception for {symbol}: {e}")
        return False

    if not res:
        logger.warning(f"Sell order returned None for {symbol}")
        return False

    # A valid Robinhood order always carries an 'id'; a missing id means the response
    # is an API error dict (e.g. {'detail': 'Not found'}) that was truthy but not an order
    order_id = res.get("id")
    if not order_id:
        logger.error(f"Sell rejected for {symbol}: {res.get('detail') or res}")
        return False

    logger.info(f"Sell order placed for {symbol}: qty={quantity}, state={res.get('state')}, id={order_id[:8]}...")
    return True


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_news_for_symbol(symbol: str, news_df: pd.DataFrame | None) -> dict:
    try:
        if news_df is not None and not news_df.empty:
            rows = news_df[news_df["symbol"] == symbol]["news"]
            if not rows.empty:
                raw = rows.iloc[0] if len(rows) == 1 else rows.tolist()
                if isinstance(raw, str):
                    raw = json.loads(raw)
                return {symbol: raw}
    except Exception as e:
        logger.debug(f"News load failed for {symbol}: {e}")
    return {}


def _build_stocks_data(candidates: pd.DataFrame, action: str) -> list[dict]:
    """
    Build the list consumed by get_batch_sentiment_recommendations.
    Both CSVs are loaded once and reused for all symbols.
    """
    try:
        agg_df = read_data_as_pd("agg_data")
    except Exception:
        agg_df = None
    try:
        news_df = read_data_as_pd("news")
    except Exception:
        news_df = None

    # Index once for O(1) per-symbol lookups instead of a full scan per candidate
    agg_index = (
        agg_df.set_index("symbol")
        if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns
        else None
    )

    stocks_data = []
    for _, row in candidates.iterrows():
        symbol = row["symbol"]

        if agg_index is not None and symbol in agg_index.index:
            agg_row = agg_index.loc[symbol]
            if isinstance(agg_row, pd.DataFrame):
                agg_row = agg_row.iloc[0]
            fundamentals = {k: agg_row.get(k) for k in METRIC_KEYS}
        else:
            fundamentals = {k: row.get(k) for k in METRIC_KEYS}

        stocks_data.append({
            "symbol":              symbol,
            "action":              action,
            "fundamental_metrics": fundamentals,
            "news_sentiment":      _load_news_for_symbol(symbol, news_df),
        })

    return stocks_data


# ---------------------------------------------------------------------------
# Portfolio risk controls
# ---------------------------------------------------------------------------

def get_portfolio_value() -> float:
    """Return total portfolio equity."""
    try:
        return float(rb.account.build_user_profile().get("equity", 0) or 0)
    except Exception as e:
        logger.warning(f"Could not fetch portfolio value: {e}")
        return 0.0


def get_current_positions() -> dict:
    """Return holdings dict {symbol: data}."""
    try:
        return rb.build_holdings() or {}
    except Exception as e:
        logger.warning(f"Could not fetch holdings: {e}")
        return {}


def get_position_value(symbol: str, holdings: dict) -> float:
    """Return current dollar value of a position."""
    try:
        return float(holdings.get(symbol, {}).get("equity", 0) or 0)
    except Exception:
        return 0.0


def get_sector_exposure(holdings: dict, agg_df: pd.DataFrame | None) -> dict[str, float]:
    """Return {sector: total_dollar_value} for all current holdings."""
    totals: dict[str, float] = {}
    for symbol, data in holdings.items():
        equity = safe_float(data.get("equity"), 0.0)
        sector = "Unknown"
        if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
            row = agg_df[agg_df["symbol"] == symbol]
            if not row.empty:
                sector = str(row.iloc[0].get("sector") or "Unknown")
        totals[sector] = totals.get(sector, 0.0) + equity
    return totals


def can_buy_symbol(
    symbol: str,
    allocation: float,
    holdings: dict,
    agg_df: pd.DataFrame | None,
    portfolio_value: float,
    available_cash: float,
    sector_exposure: dict | None = None,
) -> tuple[bool, str, float]:
    """
    Validate and adjust a proposed buy allocation against risk limits.
    Returns (approved, reason, adjusted_allocation).
    Reduces allocation to the maximum allowed rather than blocking outright when possible.
    """
    max_single    = RISK_LIMITS["max_single_position_pct"]
    max_sector    = RISK_LIMITS["max_sector_pct"]
    max_order_pct = RISK_LIMITS["max_order_pct_of_cash"]
    min_order     = RISK_LIMITS["min_order_amount"]
    min_volume    = RISK_LIMITS["min_liquidity_volume"]

    # Liquidity gate
    if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
        row = agg_df[agg_df["symbol"] == symbol]
        if not row.empty:
            vol = safe_float(row.iloc[0].get("volume"), 0.0)
            if vol < min_volume:
                return False, f"volume {vol:,.0f} < min {min_volume:,.0f}", 0.0

    # Order size cap (fraction of available cash)
    max_order = available_cash * max_order_pct
    if allocation > max_order:
        logger.info(
            f"{symbol}: order ${allocation:.2f} capped to {max_order_pct:.0%} of cash "
            f"(${available_cash:.2f}) = ${max_order:.2f}"
        )
        allocation = max_order

    # Single-position cap
    if portfolio_value > 0:
        current_pos = get_position_value(symbol, holdings)
        max_allowed = portfolio_value * max_single
        room = max_allowed - current_pos
        if room <= 0:
            return (
                False,
                f"position cap reached (${current_pos:.2f} / ${max_allowed:.2f})",
                0.0,
            )
        if allocation > room:
            logger.info(
                f"{symbol}: buy reduced ${allocation:.2f} → ${room:.2f} "
                f"(single-position cap {max_single:.0%} of ${portfolio_value:.2f})"
            )
            allocation = room

    # Sector cap
    if portfolio_value > 0 and agg_df is not None and not agg_df.empty:
        row = agg_df[agg_df["symbol"] == symbol]
        sector = str(row.iloc[0].get("sector") or "") if not row.empty else ""
        if sector:
            sector_exp     = sector_exposure if sector_exposure is not None else get_sector_exposure(holdings, agg_df)
            current_sector = sector_exp.get(sector, 0.0)
            max_sector_val = portfolio_value * max_sector
            room = max_sector_val - current_sector
            if room <= 0:
                return (
                    False,
                    f"sector cap reached for {sector!r} "
                    f"(${current_sector:.2f} / ${max_sector_val:.2f})",
                    0.0,
                )
            if allocation > room:
                logger.info(
                    f"{symbol}: buy reduced ${allocation:.2f} → ${room:.2f} "
                    f"(sector {sector!r} cap {max_sector:.0%})"
                )
                allocation = room

    # Final minimum check — skip if allocation is below the minimum order threshold
    if allocation < min_order:
        return (
            False,
            f"allocation ${allocation:.2f} below min_order_amount ${min_order:.2f}",
            0.0,
        )

    return True, "ok", allocation


# ---------------------------------------------------------------------------
# Sell decision engine
# ---------------------------------------------------------------------------

def evaluate_sell_candidate(
    symbol: str,
    holding: dict,
    metrics_row: "pd.Series | None",
    peak_price: float | None = None,
) -> dict:
    """
    Evaluate a single holding for sell conditions.

    Returns:
        should_sell: bool
        reason: str
        severity: "hard" | "soft" | None
        percent_change: float | None   (decimal: -0.12 = -12%)
        value_metric: float | None
        quality_score: float | None
        yield_trap_flag: bool | None
    """
    # Derive percent_change — Robinhood returns it as a percentage string e.g. "-15.3"
    percent_change: float | None = None
    pct_raw = safe_float(holding.get("percent_change"))
    if pct_raw is not None:
        percent_change = pct_raw / 100.0

    # Fallback: compute from average_buy_price and current price
    if percent_change is None:
        avg   = safe_float(holding.get("average_buy_price"))
        price = safe_float(holding.get("price"))
        if avg and avg > 0 and price:
            percent_change = (price / avg) - 1.0

    # Extract metrics
    value_metric:   float | None = None
    quality_score:  float | None = None
    yield_trap_flag: bool | None = None

    if metrics_row is not None:
        value_metric  = safe_float(metrics_row.get("value_metric"))
        quality_score = safe_float(metrics_row.get("quality_score"))
        yt = metrics_row.get("yield_trap_flag")
        if yt is not None:
            try:
                yield_trap_flag = bool(yt) if not pd.isna(yt) else None
            except Exception:
                yield_trap_flag = None

    # Days held (best-effort — holding dict may not carry creation date)
    days_held: int | None = None
    try:
        created = holding.get("created_at") or holding.get("initiation_date")
        if created:
            created_dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
            days_held = (datetime.datetime.now(datetime.timezone.utc) - created_dt).days
    except Exception:
        pass

    stop_loss   = SELL_RULES["stop_loss_pct"]
    take_profit = SELL_RULES["take_profit_pct"]
    sell_weak   = SELL_RULES["sell_weak_value_below"]
    sell_yt     = SELL_RULES["sell_yield_trap"]
    sell_lq     = SELL_RULES["sell_low_quality_below"]
    min_days    = SELL_RULES["min_days_held_before_value_exit"]

    base = {
        "percent_change":  percent_change,
        "value_metric":    value_metric,
        "quality_score":   quality_score,
        "yield_trap_flag": yield_trap_flag,
    }

    # ── Hard sells ────────────────────────────────────────────────────────────

    if percent_change is not None and percent_change <= stop_loss:
        return {
            **base,
            "should_sell": True,
            "reason":      f"stop loss breached ({percent_change:.1%} ≤ {stop_loss:.1%})",
            "severity":    "hard",
            "exit_type":   "failure_exit",
        }

    trailing_stop = SELL_RULES["trailing_stop_pct"]
    if peak_price is not None and peak_price > 0:
        current_p = safe_float(holding.get("price"))
        if current_p is not None:
            drawdown = (current_p / peak_price) - 1.0
            if drawdown <= trailing_stop:
                return {
                    **base,
                    "should_sell": True,
                    "reason":      f"trailing stop: {drawdown:.1%} from peak ${peak_price:.2f}",
                    "severity":    "hard",
                    "exit_type":   "failure_exit",
                }

    if sell_yt and yield_trap_flag and value_metric is not None and value_metric < sell_weak:
        return {
            **base,
            "should_sell": True,
            "reason":      f"yield trap with weak value_metric={value_metric:.3f} < {sell_weak}",
            "severity":    "hard",
            "exit_type":   "failure_exit",
        }

    if quality_score is not None and quality_score < sell_lq:
        return {
            **base,
            "should_sell": True,
            "reason":      f"quality_score {quality_score:.3f} below floor {sell_lq}",
            "severity":    "hard",
            "exit_type":   "failure_exit",
        }

    # ── Soft sells ────────────────────────────────────────────────────────────

    if percent_change is not None and percent_change >= take_profit:
        floor = SELL_RULES["take_profit_value_floor_multiplier"]
        if value_metric is not None and value_metric >= METRIC_THRESHOLD * floor:
            logger.info(
                f"{symbol}: take-profit threshold hit ({percent_change:.1%}) "
                f"but still fundamentally cheap (value_metric={value_metric:.3f}) — holding"
            )
        else:
            return {
                **base,
                "should_sell": True,
                "reason":      f"take profit triggered ({percent_change:.1%} ≥ {take_profit:.1%})",
                "severity":    "soft",
                "exit_type":   "harvest_exit",
            }

    if value_metric is not None and value_metric < sell_weak:
        if days_held is None or days_held >= min_days:
            days_str = f"{days_held}d" if days_held is not None else "unknown days"
            return {
                **base,
                "should_sell": True,
                "reason":      f"value_metric={value_metric:.3f} < {sell_weak} (held {days_str})",
                "severity":    "soft",
                "exit_type":   "thesis_exit",
            }

    return {
        **base,
        "should_sell": False,
        "reason":      "no sell condition met",
        "severity":    None,
        "exit_type":   None,
    }


# ---------------------------------------------------------------------------
# Harvest routing
# ---------------------------------------------------------------------------

def allocate_harvest_proceeds_to_etfs(amount: float) -> None:
    """Reinvest take-profit proceeds into harvest ETFs."""
    harvest_etfs = HARVEST_PARAMS["harvest_etfs"]
    if not harvest_etfs:
        return
    per_etf = amount / len(harvest_etfs)
    min_order = RISK_LIMITS["min_order_amount"]
    if per_etf < min_order:
        logger.info(
            f"Harvest per-ETF ${per_etf:.2f} < min_order ${min_order:.2f} — skipping harvest reinvestment"
        )
        return
    logger.info(f"=== HARVEST: ${amount:.2f} → {harvest_etfs} (${per_etf:.2f} each) ===")
    for etf in harvest_etfs:
        try:
            res = _rb_call_with_retry(rb.orders.order_buy_fractional_by_price, etf, per_etf)
            logger.info(f"Harvest → {etf}: {res.get('state') if res else 'None'}")
        except Exception as e:
            logger.error(f"Harvest reinvestment failed for {etf}: {e}")


# ---------------------------------------------------------------------------
# Sell cycle
# ---------------------------------------------------------------------------

def get_pending_sell_symbols() -> set[str]:
    """Return symbols that already have an open sell order so we can skip re-evaluating them."""
    try:
        pending: set[str] = set()
        for order in rb.orders.get_all_open_stock_orders():
            if order.get("side") != "sell":
                continue
            if order.get("state") not in ("confirmed", "queued", "unconfirmed"):
                continue
            sym = order.get("symbol")
            if sym:
                pending.add(sym)
        return pending
    except Exception as exc:
        logger.warning(f"Could not fetch open sell orders: {exc}")
        return set()


def make_sales() -> list[str]:
    """
    Evaluate all non-ETF holdings for sell conditions.

    Hard sells (stop-loss, yield-trap, quality floor) execute immediately.
    Soft sells (take-profit, weak value) are optionally held by sentiment.
    Sentiment can only override soft sells — never hard sells.
    """
    sold: list[str] = []

    try:
        holdings = rb.build_holdings()
    except Exception as e:
        logger.error(f"Could not fetch holdings: {e}")
        return sold

    try:
        agg_df = read_data_as_pd("agg_data")
    except Exception:
        agg_df = None

    # Load and update trailing-stop peak prices for all active holdings
    peaks = _load_peak_prices()
    for symbol, data in holdings.items():
        if symbol in ETFS or float(data.get("quantity", 0)) <= 0:
            continue
        current_p = safe_float(data.get("price"))
        if current_p and current_p > 0:
            if symbol not in peaks or peaks[symbol] <= 0:
                avg_buy = safe_float(data.get("average_buy_price"), 0.0) or 0.0
                peaks[symbol] = max(avg_buy, current_p)
            else:
                peaks[symbol] = max(peaks[symbol], current_p)
    _save_peak_prices(peaks)

    pending_sells = get_pending_sell_symbols()
    if pending_sells:
        logger.info(f"Skipping {len(pending_sells)} symbol(s) with existing open sell orders: {sorted(pending_sells)}")

    scanned    = 0
    hard_sells: dict[str, dict] = {}
    soft_sells: dict[str, dict] = {}

    for symbol, data in holdings.items():
        if symbol in ETFS:
            continue
        if float(data.get("quantity", 0)) <= 0:
            continue
        if symbol in pending_sells:
            continue

        scanned += 1

        metrics_row = None
        if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
            row = agg_df[agg_df["symbol"] == symbol]
            if not row.empty:
                metrics_row = row.iloc[0]

        decision = evaluate_sell_candidate(symbol, data, metrics_row, peak_price=peaks.get(symbol))

        if not decision["should_sell"]:
            continue

        if decision["severity"] == "hard":
            hard_sells[symbol] = decision
        else:
            soft_sells[symbol] = decision

    logger.info(
        f"Sell scan: {scanned} holdings scanned | "
        f"{len(hard_sells)} hard | {len(soft_sells)} soft | "
        f"{scanned - len(hard_sells) - len(soft_sells)} no-action"
    )

    # ── Execute hard sells ────────────────────────────────────────────────────
    for symbol, decision in hard_sells.items():
        pct = decision.get("percent_change")
        pct_str = f" | P/L={pct:.1%}" if pct is not None else ""
        logger.info(f"HARD SELL {symbol} | {decision['reason']}{pct_str}")
        quantity = float(holdings[symbol].get("quantity", 0))
        if _place_sell(symbol, quantity):
            sold.append(symbol)

    # ── Soft sells with optional sentiment override ───────────────────────────
    held_on_sentiment: set[str] = set()
    sentiment_results: dict[str, dict] = {}

    if soft_sells:
        if USE_SENTIMENT_ANALYSIS:
            soft_df = pd.DataFrame([
                {"symbol": sym, **{k: None for k in METRIC_KEYS}}
                for sym in soft_sells
            ])
            stocks_data = _build_stocks_data(soft_df, action="sell")
            try:
                sentiment_results = get_batch_sentiment_recommendations(stocks_data, action="sell")
            except Exception:
                logger.error("Batch sentiment failed for soft sells — executing all", exc_info=True)

            for sym, result in sentiment_results.items():
                if (
                    result.get("action") == "HOLD"
                    and result.get("sentiment") == "bullish"
                    and result["confidence"] >= SELL_SENTIMENT_OVERRIDE_CONFIDENCE
                ):
                    logger.info(
                        f"HOLD {sym} — sentiment overrides soft sell "
                        f"({result['confidence']}% bullish): {result['reasoning']}"
                    )
                    held_on_sentiment.add(sym)
                else:
                    logger.info(
                        f"SELL confirmed {sym} — action={result.get('action')} "
                        f"sentiment={result.get('sentiment')} ({result['confidence']}%): {result['reasoning']}"
                    )

        harvest_proceeds = 0.0
        for symbol, decision in soft_sells.items():
            if symbol in held_on_sentiment:
                continue
            pct = decision.get("percent_change")
            pct_str = f" | P/L={pct:.1%}" if pct is not None else ""
            logger.info(f"SOFT SELL {symbol} | {decision['reason']}{pct_str}")
            quantity = float(holdings[symbol].get("quantity", 0))
            if _place_sell(symbol, quantity):
                sold.append(symbol)
                if decision.get("exit_type") == "harvest_exit":
                    harvest_proceeds += safe_float(holdings[symbol].get("equity"), 0.0) or 0.0

        if harvest_proceeds >= HARVEST_PARAMS["min_harvest_amount"]:
            allocate_harvest_proceeds_to_etfs(harvest_proceeds)

    # Remove sold positions from peak price tracking
    if sold:
        for sym in sold:
            peaks.pop(sym, None)
        _save_peak_prices(peaks)

    logger.info(
        f"Sell summary: {scanned} scanned | "
        f"{len(hard_sells)} hard | {len(soft_sells)} soft candidates | "
        f"{len(held_on_sentiment)} held on sentiment | "
        f"{len(sold)} executed | "
        f"{len(hard_sells) + len(soft_sells) - len(sold)} skipped/no-action"
    )
    return sold


# ---------------------------------------------------------------------------
# Buy cycle
# ---------------------------------------------------------------------------

def make_buys(df: pd.DataFrame, is_first_iteration: bool = True, bear_market: bool = False) -> tuple[list, list, list]:
    """
    Execute buy orders.
    Returns (purchased, skipped, failed).
    """
    total_cash   = get_available_cash()
    etf_amount   = total_cash * INDEX_PCT
    stock_amount = total_cash - etf_amount
    logger.info(f"Allocating ${etf_amount:.2f} to ETFs, ${stock_amount:.2f} to stocks")

    # ETF buys — first iteration only
    if is_first_iteration and etf_amount > 0 and (AUTO_APPROVE or confirm(f"Buy ETFs (${etf_amount:,.2f})?")):
        per_etf = etf_amount / max(len(ETFS), 1)
        if per_etf < RISK_LIMITS["min_order_amount"]:
            logger.info(
                f"Per-ETF amount ${per_etf:.2f} < min_order ${RISK_LIMITS['min_order_amount']:.2f} — skipping ETF buys"
            )
        else:
            for etf in ETFS:
                try:
                    if AUTO_APPROVE or confirm(f"Buy ${per_etf:,.2f} of {etf}?"):
                        res = rb.orders.order_buy_fractional_by_price(etf, per_etf)
                        logger.info(f"ETF {etf}: {res.get('state') if res else 'None'}")
                except Exception as e:
                    logger.error(f"ETF buy failed for {etf}: {e}")

    if bear_market:
        logger.info("Bear market regime — skipping individual stock buys (remaining cash swept to ETFs at end)")
        return [], [], []

    if df.empty or stock_amount <= 0:
        logger.warning("No stock picks or no funds for stocks")
        return [], [], []

    # Pre-filter by value_metric
    total_before = len(df)
    df["value_metric"] = pd.to_numeric(df["value_metric"], errors="coerce").fillna(0.0)

    logger.info(f"value_metric dtype: {df['value_metric'].dtype}")
    logger.info(f"value_metric max: {df['value_metric'].max()}")
    logger.info(
        "Top value_metric rows:\n%s",
        df[["symbol", "value_metric"]]
        .sort_values("value_metric", ascending=False)
        .head(10)
        .to_string(index=False)
    )
    candidates = df[df["value_metric"] >= METRIC_THRESHOLD].copy()
    logger.info(f"Pre-filter: {total_before} → {len(candidates)} stocks (value_metric ≥ {METRIC_THRESHOLD})")

    if candidates.empty:
        logger.warning(f"No stocks pass value_metric ≥ {METRIC_THRESHOLD}")
        return [], df["symbol"].tolist(), []

    # Exclude contrarian watchlist — tagged for monitoring only, not auto-buy
    if "strategy_bucket" in candidates.columns:
        contrarian_mask = candidates["strategy_bucket"] == "contrarian_watchlist"
        contrarian_syms = candidates.loc[contrarian_mask, "symbol"].tolist()
        candidates = candidates[~contrarian_mask].copy()
        if contrarian_syms:
            logger.info(f"Contrarian watchlist excluded from buys: {contrarian_syms}")

    if candidates.empty:
        logger.warning("No candidates remain after contrarian exclusion")
        return [], df["symbol"].tolist(), []

    # Cap candidates before sentiment to bound API cost and avoid look-ahead selection.
    # BTR is a tie-breaker within equal value_metric — NaN rows are pushed to the end.
    max_sc = RISK_LIMITS["max_sentiment_candidates"]
    if len(candidates) > max_sc:
        sort_cols = ["value_metric"] + (
            ["buy_to_sell_ratio"] if "buy_to_sell_ratio" in candidates.columns else []
        )
        candidates = (
            candidates
            .sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
            .head(max_sc)
            .copy()
        )
        logger.info(f"Candidates capped to {max_sc} (max_sentiment_candidates)")

    # Load portfolio context once before the loop
    holdings        = get_current_positions()
    portfolio_value = get_portfolio_value()
    try:
        agg_df = read_data_as_pd("agg_data")
    except Exception:
        agg_df = None

    # Sentiment results — empty dict acts as "no filter" when sentiment is disabled
    sentiment_results: dict[str, dict] = {}
    if USE_SENTIMENT_ANALYSIS:
        logger.info(f"Running batch sentiment on {len(candidates)} candidates...")
        stocks_data = _build_stocks_data(candidates, action="buy")
        try:
            sentiment_results = get_batch_sentiment_recommendations(stocks_data, action="buy")
        except Exception:
            logger.error("Batch sentiment failed — all candidates skipped", exc_info=True)
            return [], candidates["symbol"].tolist(), []

    # Pre-compute sector exposure once rather than rebuilding it per candidate
    sector_exposure = get_sector_exposure(holdings, agg_df) if portfolio_value > 0 else {}

    purchased, skipped, failed = [], [], []
    whole_share_queue: list[tuple[str, float]] = []
    allow_ws_fallback = RISK_LIMITS["allow_whole_share_fallback"]
    total_value = candidates["value_metric"].sum()

    for _, row in candidates.iterrows():
        symbol = row["symbol"]

        if sentiment_results:
            result = sentiment_results.get(
                symbol,
                {"action": "HOLD", "sentiment": "neutral", "confidence": 0.0, "reasoning": "No result"},
            )
            logger.info(
                f"{'='*60}\nBUY {symbol} | action={result.get('action')} "
                f"sentiment={result.get('sentiment')} {result['confidence']:.1f}% | {result['reasoning']}\n{'='*60}"
            )
            if result.get("action") != "BUY" or result["confidence"] < CONFIDENCE_THRESHOLD:
                logger.info(f"Skipping {symbol}")
                skipped.append(symbol)
                continue

        cash = get_available_cash() * (1 - INDEX_PCT)
        if cash < RISK_LIMITS["min_order_amount"]:
            logger.info(f"Cash ${cash:.2f} below min order ${RISK_LIMITS['min_order_amount']:.2f} — exiting buy loop")
            break

        alloc = (row["value_metric"] / total_value) * cash if total_value else 0
        ok, reason, adj_alloc = can_buy_symbol(
            symbol, alloc, holdings, agg_df, portfolio_value, cash, sector_exposure
        )
        if not ok:
            logger.info(f"Skipping {symbol}: {reason}")
            skipped.append(symbol)
            continue

        try:
            if AUTO_APPROVE or confirm(
                f"Buy ${adj_alloc:,.2f} of {symbol}? ({row['value_metric']/total_value:.1%} of stock budget)"
            ):
                ok_frac, detail = _place_fractional_buy(symbol, adj_alloc)
                if ok_frac:
                    purchased.append(symbol)
                    _update_local_exposures_after_buy(symbol, adj_alloc, agg_df, sector_exposure)
                    time.sleep(0.5)
                elif allow_ws_fallback:
                    logger.warning(f"{symbol}: fractional failed ({detail}) — queued for whole-share fallback")
                    whole_share_queue.append((symbol, adj_alloc))
                else:
                    logger.warning(f"{symbol}: fractional failed ({detail}) — skipping")
                    failed.append(symbol)
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            failed.append(symbol)

    if whole_share_queue:
        _process_whole_share_queue(
            whole_share_queue, agg_df, sector_exposure, purchased, failed, skipped,
            holdings, portfolio_value,
        )

    logger.info(
        f"Buy summary: {len(purchased)} bought, {len(skipped)} skipped, {len(failed)} failed"
    )
    logger.info(f"Cash remaining: ${get_available_cash():,.2f}")
    return purchased, skipped, failed


# ---------------------------------------------------------------------------
# Strategy loop
# ---------------------------------------------------------------------------

def run_daily_strat() -> None:
    logger.info(f"=== Daily Investment Strategy {datetime.datetime.now():%Y-%m-%d %H:%M} ===")
    if USE_SENTIMENT_ANALYSIS:
        logger.info(f"Sentiment ON | METRIC_THRESHOLD={METRIC_THRESHOLD} | CONFIDENCE={CONFIDENCE_THRESHOLD}%")

    if not AUTO_APPROVE and not confirm("Generate new picks and run strategy?"):
        logger.info("Cancelled")
        return

    try:
        skip_data = "--skip-data" in sys.argv
        if not skip_data:
            update_industry_valuations(verbose=True)
            add_funds_to_account()
            refresh = AUTO_APPROVE or confirm("Generate fresh data? (takes several minutes)")
        else:
            logger.info("--skip-data: using existing CSVs")
            refresh = False

        df = generate_daily_undervalued_stocks(refresh=refresh)
    except Exception as e:
        logger.error(f"Strategy setup failed: {e}")
        if not AUTO_APPROVE:
            input("Press Enter to exit...")
        return

    # Allow strategy to run with whatever cash is available as long as it is positive
    cash = get_available_cash()
    if cash <= 0:
        logger.error("No funds available — aborting strategy")
        if not AUTO_APPROVE:
            input("Press Enter to exit...")
        return
    if cash < WEEKLY_INVESTMENT:
        logger.info(f"Proceeding with ${cash:,.2f} available (below ${WEEKLY_INVESTMENT:,.2f} weekly target)")

    bear_market = _is_bear_market_regime()

    permanently_skipped: set[str] = set()

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info(f"\n{'='*60}\nITERATION {iteration}/10 | skipped so far: {len(permanently_skipped)}\n{'='*60}")

        if permanently_skipped:
            df = df[~df["symbol"].isin(permanently_skipped)].copy()

        if df.empty:
            logger.info("No remaining candidates — exiting")
            break

        made_buys = made_sells = False

        # Sell phase always runs first
        try:
            logger.info("=== SELL PHASE ===")
            sold = make_sales()
            if sold:
                made_sells = True
        except Exception as e:
            logger.error(f"Sell phase error (iter {iteration}): {e}")

        cash = get_available_cash()
        if cash < RISK_LIMITS["min_order_amount"]:
            logger.info(f"Cash ${cash:.2f} below min order ${RISK_LIMITS['min_order_amount']:.2f} — skipping buy phase")
            if not made_sells:
                logger.info("No sells either — exiting loop")
                break
            continue

        try:
            logger.info("=== BUY PHASE ===")
            purchased, skipped, failed = make_buys(df, is_first_iteration=(iteration == 1), bear_market=bear_market)
            if purchased:
                made_buys = True
            permanently_skipped.update(skipped)
            permanently_skipped.update(failed)
        except Exception as e:
            logger.error(f"Buy phase error (iter {iteration}): {e}")

        if not made_buys and not made_sells:
            logger.info("No activity this iteration — exiting loop")
            break

    # Sweep any remaining cash into ETFs so nothing sits idle
    remaining = get_available_cash()
    if remaining > 0 and ETFS:
        per_etf = remaining / len(ETFS)
        if per_etf < RISK_LIMITS["min_order_amount"]:
            logger.info(
                f"Sweep per-ETF ${per_etf:.2f} < min_order ${RISK_LIMITS['min_order_amount']:.2f} — skipping sweep"
            )
        else:
            logger.info(f"=== CASH SWEEP: ${remaining:,.2f} → ETFs (${per_etf:.2f} each) ===")
            for etf in ETFS:
                try:
                    res = rb.orders.order_buy_fractional_by_price(etf, per_etf)
                    logger.info(f"Sweep {etf}: {res.get('state') if res else 'None'}")
                except Exception as e:
                    logger.error(f"Sweep failed for {etf}: {e}")

    logger.info(
        f"\n{'='*60}\n"
        f"STRATEGY COMPLETE\n"
        f"Final cash: ${get_available_cash():,.2f}\n"
        f"Total skipped: {len(permanently_skipped)}\n"
        f"{'='*60}"
    )


# ---------------------------------------------------------------------------
# Tuner CLI
# ---------------------------------------------------------------------------

def _run_tuner_cli(n_days: int, objective: str) -> None:
    from tuner import print_config_diff, run_tuner
    try:
        best_params, best_result = run_tuner(
            n_days=n_days,
            objective=objective,
            starting_capital=10_000.0,
        )
        print_config_diff(best_params, best_result)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)


def _run_auto_tune_cli(n_days: int, mode: str | None = None, apply: bool = False, force_apply: bool = False) -> None:
    from tuner import run_auto_tune, _diff_table
    try:
        avg_params, sharpe_result, calmar_result, avg_result = run_auto_tune(
            n_days=n_days,
            starting_capital=10_000.0,
            mode=mode,
            apply=apply,
            force_apply=force_apply,
        )
        _diff_table(
            avg_params,
            label=f"mean of Sharpe + Calmar over {n_days}d",
            sharpe_ref=sharpe_result,
            calmar_ref=calmar_result,
        )
        print(
            f"\nAveraged result:  ret={avg_result.total_return:+.1%}  "
            f"sharpe={avg_result.sharpe:+.3f}  "
            f"calmar={avg_result.calmar:+.3f}  "
            f"trades={avg_result.trades_made}"
        )
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _raise_fd_limit() -> None:
    """Raise the open-file-descriptor soft limit to avoid EMFILE during bulk yfinance downloads."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 4096)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.debug(f"Raised RLIMIT_NOFILE: {soft} → {target} (hard={hard})")
    except Exception as e:
        logger.debug(f"Could not raise fd limit: {e}")


def main() -> None:
    _raise_fd_limit()
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("Usage: python main.py [--skip-data] [--tune DAYS] [--auto-tune [DAYS]] [--objective sharpe|calmar] [--mode MODE] [--apply] [--help]")
        print("  --skip-data          Reuse existing CSV files instead of regenerating")
        print("  --tune DAYS          Back-simulate N days and print suggested config tweaks")
        print("  --auto-tune [DAYS]   Run Sharpe+Calmar, train/val split, validate, optionally write config (default 90d)")
        print("  --objective METRIC   For --tune only: sharpe (default) or calmar")
        print("  --mode MODE          Backtest mode: current_universe_stress_test | liquid_universe_sanity_test | walk_forward_price_only_test")
        print("  --apply              Write config.yaml if validation gates pass")
        print("  --force-apply        Write config.yaml regardless of validation (use with caution)")
        return

    if "--auto-tune" in args:
        idx = args.index("--auto-tune")
        n_days = 90
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            n_days = int(args[idx + 1])
        mode = None
        if "--mode" in args:
            mi = args.index("--mode")
            if mi + 1 < len(args):
                mode = args[mi + 1]
        apply = "--apply" in args
        force_apply = "--force-apply" in args
        _run_auto_tune_cli(n_days, mode=mode, apply=apply, force_apply=force_apply)
        return

    if "--tune" in args:
        idx = args.index("--tune")
        try:
            n_days = int(args[idx + 1])
        except (IndexError, ValueError):
            print("--tune requires an integer argument, e.g. --tune 90")
            sys.exit(1)
        objective = "sharpe"
        if "--objective" in args:
            oi = args.index("--objective")
            try:
                objective = args[oi + 1].lower()
                if objective not in ("sharpe", "calmar"):
                    raise ValueError
            except (IndexError, ValueError):
                print("--objective must be 'sharpe' or 'calmar'")
                sys.exit(1)
        _run_tuner_cli(n_days, objective)
        return

    login()
    run_daily_strat()


if __name__ == "__main__":
    main()
