#!/usr/bin/env python3
"""Read-only Robinhood target-account probe for ODTE cron. Prints JSON only."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def mask(s: Any) -> str | None:
    if s is None:
        return None
    text = str(s)
    if len(text) <= 4:
        return "***" + text
    return "***" + text[-4:]


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except TypeError:
        # Some robin_stocks installs do not support account_number on all helpers.
        kwargs.pop("account_number", None)
        return fn(*args, **kwargs)


def main() -> int:
    load_env()
    target = (
        os.environ.get("RB_ACCT_NUM")
        or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER")
        or os.environ.get("RH_ACCOUNT_NUMBER")
        or "435050133"
    )
    # Force aliases for helpers that read env defaults.
    os.environ["RB_ACCT_NUM"] = target
    os.environ["ROBINHOOD_ACCOUNT_NUMBER"] = target
    os.environ["RH_ACCOUNT_NUMBER"] = target

    out: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "target_account_masked": mask(target),
        "read_only": True,
        "ok": False,
        "errors": [],
    }
    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:
        out["errors"].append(f"import_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(out, sort_keys=True))
        return 0

    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME") or os.environ.get("RH_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD") or os.environ.get("RH_PASSWORD")
    mfa_secret = os.environ.get("RB_MFA_SECRET") or os.environ.get("ROBINHOOD_MFA_SECRET") or os.environ.get("RH_MFA_SECRET")
    mfa_code = None
    if mfa_secret:
        try:
            mfa_code = pyotp.TOTP(mfa_secret.strip().replace(" ", "")).now()
        except Exception as exc:
            out["errors"].append(f"mfa_failed:{type(exc).__name__}")
    if not username or not password:
        out["errors"].append("missing_credentials")
        print(json.dumps(out, sort_keys=True))
        return 0

    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            login_resp = rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
        out["login_ok"] = bool(isinstance(login_resp, dict) and login_resp.get("access_token"))
        if not out["login_ok"]:
            out["errors"].append("login_no_access_token")
            print(json.dumps(out, sort_keys=True))
            return 0
    except Exception as exc:
        out["errors"].append(f"login_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(out, sort_keys=True))
        return 0

    try:
        acct = call(rb.profiles.load_account_profile, account_number=target)
        port = call(rb.profiles.load_portfolio_profile, account_number=target)
        option_positions = call(rb.options.get_open_option_positions, account_number=target) or []
        aggregate_positions = call(rb.options.get_aggregate_open_positions, account_number=target) or []
        open_option_orders = call(rb.orders.get_all_open_option_orders, account_number=target) or []
        quotes = {}
        for sym in ["SPY", "QQQ", "IWM", "VIXY"]:
            try:
                q = rb.stocks.get_quotes(sym) or []
                row = q[0] if q else {}
                quotes[sym] = {
                    "last_trade_price": safe_float(row.get("last_trade_price")),
                    "ask_price": safe_float(row.get("ask_price")),
                    "bid_price": safe_float(row.get("bid_price")),
                    "updated_at": row.get("updated_at"),
                }
            except Exception as exc:
                quotes[sym] = {"error": f"{type(exc).__name__}:{exc}"}
        account_no = acct.get("account_number") if isinstance(acct, dict) else None
        out.update({
            "ok": True,
            "account_verified": str(account_no) == str(target),
            "account_masked": mask(account_no),
            "buying_power": safe_float((acct or {}).get("buying_power") if isinstance(acct, dict) else None),
            "cash": safe_float((acct or {}).get("cash") if isinstance(acct, dict) else None),
            "option_level": (acct or {}).get("option_level") if isinstance(acct, dict) else None,
            "is_pdt": (acct or {}).get("is_pdt") if isinstance(acct, dict) else None,
            "portfolio_market_value": safe_float((port or {}).get("market_value") if isinstance(port, dict) else None),
            "open_option_positions_count": len(option_positions) if isinstance(option_positions, list) else None,
            "aggregate_open_positions_count": len(aggregate_positions) if isinstance(aggregate_positions, list) else None,
            "open_option_orders_count": len(open_option_orders) if isinstance(open_option_orders, list) else None,
            "open_option_order_states": [o.get("state") for o in open_option_orders[:5] if isinstance(o, dict)],
            "quotes": quotes,
        })
    except Exception as exc:
        out["errors"].append(f"probe_failed:{type(exc).__name__}:{exc}")
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
