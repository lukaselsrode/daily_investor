#!/usr/bin/env python3
"""Read-only Robinhood/account + broad tape snapshot for ODTE cron.
Prints JSON only; suppresses third-party login chatter; never places/reviews orders.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def mask_acct(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x)
    if len(s) <= 4:
        return "***" + s
    return "***" + s[-4:]


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def try_call(label: str, fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return {"ok": True, "value": fn(*args, **kwargs)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    load_env(repo / ".env")
    load_env(Path.home() / "0dte" / ".env")

    target = (
        os.environ.get("RB_ACCT_NUM")
        or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER")
        or os.environ.get("RH_ACCOUNT_NUMBER")
        or "435050133"
    )
    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME") or os.environ.get("RH_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD") or os.environ.get("RH_PASSWORD")
    mfa_secret = os.environ.get("RB_MFA_SECRET") or os.environ.get("ROBINHOOD_MFA_SECRET") or os.environ.get("RH_MFA_SECRET")

    out: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_account_masked": mask_acct(target),
        "target_verified": False,
        "login_ok": False,
        "blocked": False,
        "errors": [],
    }
    if not username or not password:
        out.update({"blocked": True, "errors": ["missing Robinhood credentials in environment/.env"]})
        print(json.dumps(out, indent=2, default=str))
        return 0

    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:
        out.update({"blocked": True})
        out["errors"].append(f"import failed: {type(exc).__name__}: {exc}")
        print(json.dumps(out, indent=2, default=str))
        return 0

    mfa_code = None
    if mfa_secret:
        secret = mfa_secret.strip().replace(" ", "").replace("-", "").upper()
        bad = sorted({c for c in secret if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"})
        if bad:
            out["errors"].append(f"RB_MFA_SECRET contains non-base32 characters: {bad}")
        else:
            with contextlib.suppress(Exception):
                mfa_code = pyotp.TOTP(secret).now()
    login_stdout = io.StringIO()
    login_stderr = io.StringIO()
    with contextlib.redirect_stdout(login_stdout), contextlib.redirect_stderr(login_stderr):
        login = try_call("login", rb.login, username=username, password=password, mfa_code=mfa_code, store_session=True)
    out["login_ok"] = bool(login.get("ok") and isinstance(login.get("value"), dict) and login["value"].get("access_token"))
    if not out["login_ok"]:
        out["blocked"] = True
        out["errors"].append(login.get("error") or "login returned no access_token")
        print(json.dumps(out, indent=2, default=str))
        return 0

    prof = try_call("account_profile", rb.profiles.load_account_profile, account_number=target)
    port = try_call("portfolio_profile", rb.profiles.load_portfolio_profile, account_number=target)
    open_opt_orders = try_call("open_option_orders", rb.orders.get_all_open_option_orders, account_number=target)
    opt_positions = try_call("open_option_positions", rb.options.get_open_option_positions, account_number=target)
    agg_positions = try_call("aggregate_option_positions", rb.options.get_aggregate_open_positions, account_number=target)

    profile_val = prof.get("value") if prof.get("ok") else None
    portfolio_val = port.get("value") if port.get("ok") else None
    acct_num_seen = None
    if isinstance(profile_val, dict):
        acct_num_seen = profile_val.get("account_number") or profile_val.get("account")
    out["target_verified"] = bool(str(acct_num_seen or "") == str(target))
    out["account"] = {
        "account_number_masked": mask_acct(acct_num_seen),
        "type": profile_val.get("type") if isinstance(profile_val, dict) else None,
        "option_level": profile_val.get("option_level") or profile_val.get("options_level") if isinstance(profile_val, dict) else None,
        "is_active": profile_val.get("is_active") if isinstance(profile_val, dict) else None,
        "buying_power": safe_float(profile_val.get("buying_power")) if isinstance(profile_val, dict) else None,
        "cash": safe_float(profile_val.get("cash")) if isinstance(profile_val, dict) else None,
        "portfolio_equity": safe_float(portfolio_val.get("equity")) if isinstance(portfolio_val, dict) else None,
        "portfolio_withdrawable_amount": safe_float(portfolio_val.get("withdrawable_amount")) if isinstance(portfolio_val, dict) else None,
    }
    out["open_option_orders"] = {
        "ok": open_opt_orders.get("ok"),
        "count": len(open_opt_orders.get("value") or []) if open_opt_orders.get("ok") else None,
        "states": [o.get("state") for o in (open_opt_orders.get("value") or []) if isinstance(o, dict)][:10],
        "ids_masked": [str(o.get("id", ""))[:8] + "..." for o in (open_opt_orders.get("value") or []) if isinstance(o, dict)][:10],
        "error": open_opt_orders.get("error"),
    }
    positions_val = opt_positions.get("value") or []
    nonzero_positions = []
    for p in positions_val if isinstance(positions_val, list) else []:
        if not isinstance(p, dict):
            continue
        qty = safe_float(p.get("quantity") or p.get("intraday_quantity") or p.get("trade_value_multiplier"))
        if qty and abs(qty) > 1e-9:
            nonzero_positions.append({
                "chain_symbol": p.get("chain_symbol"),
                "option": p.get("option"),
                "quantity": qty,
                "average_price": p.get("average_price"),
                "type": p.get("type"),
                "account_masked": mask_acct(p.get("account_number") or p.get("account")),
            })
    out["open_option_positions"] = {
        "ok": opt_positions.get("ok"),
        "raw_count": len(positions_val) if isinstance(positions_val, list) else None,
        "nonzero_count": len(nonzero_positions),
        "nonzero_sample": nonzero_positions[:10],
        "error": opt_positions.get("error"),
    }
    out["aggregate_option_positions"] = {
        "ok": agg_positions.get("ok"),
        "count": len(agg_positions.get("value") or []) if agg_positions.get("ok") else None,
        "error": agg_positions.get("error"),
    }
    quote_symbols = ["SPY", "QQQ", "IWM", "VIXY", "TLT", "USO"]
    quotes = {}
    quote_res = try_call("quotes", rb.stocks.get_quotes, quote_symbols)
    if quote_res.get("ok") and isinstance(quote_res.get("value"), list):
        for q in quote_res["value"]:
            if not isinstance(q, dict):
                continue
            sym = q.get("symbol")
            quotes[sym] = {
                "last_trade_price": safe_float(q.get("last_trade_price")),
                "last_extended_hours_trade_price": safe_float(q.get("last_extended_hours_trade_price")),
                "previous_close": safe_float(q.get("previous_close")),
                "updated_at": q.get("updated_at") or q.get("last_trade_price_source"),
                "bid_price": safe_float(q.get("bid_price")),
                "ask_price": safe_float(q.get("ask_price")),
            }
    else:
        out["errors"].append("quote fetch failed: " + str(quote_res.get("error")))
    out["quotes"] = quotes
    for label, res in [("account_profile", prof), ("portfolio_profile", port), ("open_option_orders", open_opt_orders), ("open_option_positions", opt_positions), ("aggregate_option_positions", agg_positions)]:
        if not res.get("ok"):
            out["errors"].append(f"{label}: {res.get('error')}")
    if not out["target_verified"]:
        out["blocked"] = True
        out["errors"].append("target account unverified")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
