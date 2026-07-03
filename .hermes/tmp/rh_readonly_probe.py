#!/usr/bin/env python3
"""Read-only Robinhood target-account probe for ODTE cron."""
from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _mask(acct: str | None) -> str | None:
    if not acct:
        return None
    acct = str(acct)
    return "****" + acct[-4:]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    if load_dotenv:
        load_dotenv(root / ".env")

    target = (
        os.environ.get("RB_ACCT_NUM")
        or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER")
        or os.environ.get("RH_ACCOUNT_NUMBER")
        or "435050133"
    )
    # Force all known aliases for any helper code using env defaults.
    os.environ["RB_ACCT_NUM"] = target
    os.environ["ROBINHOOD_ACCOUNT_NUMBER"] = target
    os.environ["RH_ACCOUNT_NUMBER"] = target

    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME") or os.environ.get("RH_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD") or os.environ.get("RH_PASSWORD")
    mfa_secret = os.environ.get("RB_MFA_SECRET") or os.environ.get("ROBINHOOD_MFA_SECRET") or os.environ.get("RH_MFA_SECRET")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_account_masked": _mask(target),
        "account_verified": False,
        "read_only": True,
        "errors": [],
    }
    if not username or not password:
        result["errors"].append("missing_credentials")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:
        result["errors"].append(f"import_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    mfa_code = None
    if mfa_secret:
        try:
            mfa_code = pyotp.TOTP(mfa_secret.strip().replace(" ", "").replace("-", "").upper()).now()
        except Exception as exc:
            result["errors"].append(f"mfa_generation_failed:{type(exc).__name__}")

    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            login_resp = rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
        result["login_ok"] = bool(isinstance(login_resp, dict) and login_resp.get("access_token"))
        if not result["login_ok"]:
            result["errors"].append("login_no_access_token")
    except Exception as exc:
        result["login_ok"] = False
        result["errors"].append(f"login_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    def call(name, fn, *args, **kwargs):
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                return fn(*args, **kwargs)
        except Exception as exc:
            result["errors"].append(f"{name}_failed:{type(exc).__name__}:{exc}")
            return None

    acct = call("account_profile", rb.profiles.load_account_profile, account_number=target)
    portfolio = call("portfolio_profile", rb.profiles.load_portfolio_profile, account_number=target)
    user_profile = call("user_profile", rb.profiles.load_user_profile)
    opt_positions = call("open_option_positions", rb.options.get_open_option_positions, account_number=target) or []
    agg_positions = call("aggregate_option_positions", rb.options.get_aggregate_open_positions, account_number=target) or []
    open_opt_orders = call("open_option_orders", rb.orders.get_all_open_option_orders, account_number=target) or []
    recent_opt_orders = call("recent_option_orders", rb.orders.get_all_option_orders, account_number=target) or []

    acct_url = (acct or {}).get("url") or (acct or {}).get("account") or ""
    result["account_verified"] = target in str(acct_url) or (acct or {}).get("account_number") == target
    result["account"] = {
        "masked": _mask((acct or {}).get("account_number") or target),
        "type": (acct or {}).get("type"),
        "status": (acct or {}).get("status"),
        "option_level": (acct or {}).get("option_level") or (acct or {}).get("options_level"),
        "buying_power": _safe_float((acct or {}).get("buying_power")),
        "cash": _safe_float((acct or {}).get("cash")),
        "portfolio_equity": _safe_float((portfolio or {}).get("equity")),
    }
    if user_profile:
        result["user_profile"] = {
            "instant_eligibility": user_profile.get("instant_eligibility"),
            "option_trading_on_expiration_enabled": user_profile.get("option_trading_on_expiration_enabled"),
        }

    def summarize_order(o):
        return {
            "id": o.get("id"),
            "state": o.get("state"),
            "chain_symbol": o.get("chain_symbol"),
            "direction": o.get("direction"),
            "opening_strategy": o.get("opening_strategy"),
            "closing_strategy": o.get("closing_strategy"),
            "created_at": o.get("created_at"),
            "processed_quantity": o.get("processed_quantity"),
            "quantity": o.get("quantity"),
            "price": o.get("price"),
        }

    def summarize_pos(p):
        return {
            "id": p.get("id"),
            "chain_symbol": p.get("chain_symbol"),
            "type": p.get("type"),
            "quantity": p.get("quantity"),
            "average_price": p.get("average_price"),
            "updated_at": p.get("updated_at"),
            "option": p.get("option"),
        }

    nonzero = []
    for p in opt_positions:
        q = _safe_float(p.get("quantity")) or 0.0
        if abs(q) > 1e-9:
            nonzero.append(summarize_pos(p))
    result["open_option_positions_count"] = len(nonzero)
    result["open_option_positions"] = nonzero
    result["aggregate_open_positions_count"] = len(agg_positions) if isinstance(agg_positions, list) else None
    result["open_option_orders_count"] = len(open_opt_orders)
    result["open_option_orders"] = [summarize_order(o) for o in open_opt_orders[:10] if isinstance(o, dict)]
    if isinstance(recent_opt_orders, list):
        result["recent_option_orders"] = [summarize_order(o) for o in recent_opt_orders[:5] if isinstance(o, dict)]

    symbols = ["SPY", "QQQ", "IWM", "VIXY", "TSLA"]
    quotes = {}
    for s in symbols:
        q = call(f"quote_{s}", rb.stocks.get_quotes, s)
        if isinstance(q, list) and q:
            d = q[0]
            quotes[s] = {
                "last_trade_price": _safe_float(d.get("last_trade_price")),
                "previous_close": _safe_float(d.get("previous_close")),
                "ask_price": _safe_float(d.get("ask_price")),
                "bid_price": _safe_float(d.get("bid_price")),
                "updated_at": d.get("updated_at"),
                "trading_halted": d.get("trading_halted"),
            }
    result["quotes"] = quotes

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["account_verified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
