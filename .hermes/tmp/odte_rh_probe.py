#!/usr/bin/env python3
"""Cron-safe read-only Robinhood probe for target ODTE account. Prints JSON only."""
from __future__ import annotations

import contextlib
import io
import json
import os
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

TARGET = (
    os.environ.get("RH_TARGET_ACCOUNT")
    or os.environ.get("ROBINHOOD_TARGET_ACCOUNT")
    or os.environ.get("RB_ACCT_NUM")
    or "435050133"
)


def _env(*names: str) -> str | None:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return None


def _mask(acct: str | None) -> str | None:
    if not acct:
        return None
    return "****" + acct[-4:]


def _account_from_url(url: str | None) -> str | None:
    if not url:
        return None
    bits = [b for b in url.rstrip("/").split("/") if b]
    return bits[-1] if bits else None


def main() -> None:
    if load_dotenv:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            load_dotenv()
    out: dict = {
        "ok": False,
        "asof_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_masked": _mask(TARGET),
        "target_verified": False,
        "source": "local_robin_stocks_read_only_probe",
        "errors": [],
    }
    try:
        import robin_stocks.robinhood as rh
    except Exception as e:
        out["errors"].append(f"import_failed:{type(e).__name__}:{e}")
        print(json.dumps(out, separators=(",", ":")))
        return

    username = _env("ROBINHOOD_USERNAME", "RH_USERNAME", "ROBINHOOD_USER", "RB_USERNAME", "RB_USER", "RB_ACCT")
    password = _env("ROBINHOOD_PASSWORD", "RH_PASSWORD", "ROBINHOOD_PASS", "RB_PASSWORD", "RB_PASS")
    rb_creds = _env("RB_CREDS")
    if rb_creds and (not username or not password):
        try:
            creds = json.loads(rb_creds)
            if isinstance(creds, dict):
                username = username or creds.get("username") or creds.get("user") or creds.get("email")
                password = password or creds.get("password") or creds.get("pass")
        except Exception:
            if ":" in rb_creds:
                username = username or rb_creds.split(":", 1)[0]
                password = password or rb_creds.split(":", 1)[1]
            elif username:
                password = password or rb_creds
    totp_secret = _env(
        "ROBINHOOD_TOTP_SECRET",
        "RH_TOTP_SECRET",
        "ROBINHOOD_MFA_SECRET",
        "RH_MFA_SECRET",
        "RB_MFA_SECRET",
    )
    if not username or not password:
        out["errors"].append("missing_login_env")
        print(json.dumps(out, separators=(",", ":")))
        return
    mfa_code = None
    if totp_secret:
        try:
            import pyotp
            mfa_code = pyotp.TOTP(totp_secret).now()
        except Exception as e:
            out["errors"].append(f"totp_failed:{type(e).__name__}")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rh.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
    except Exception as e:
        out["errors"].append(f"login_failed:{type(e).__name__}:{e}")
        print(json.dumps(out, separators=(",", ":")))
        return

    try:
        acct = rh.profiles.load_account_profile(account_number=TARGET)
        portfolio = rh.profiles.load_portfolio_profile(account_number=TARGET)
    except TypeError:
        acct = rh.profiles.load_account_profile(TARGET)
        portfolio = rh.profiles.load_portfolio_profile(TARGET)
    except Exception as e:
        out["errors"].append(f"account_fetch_failed:{type(e).__name__}:{e}")
        print(json.dumps(out, separators=(",", ":")))
        return

    acct_num = acct.get("account_number") or _account_from_url(acct.get("url")) or _account_from_url(portfolio.get("account"))
    out.update({
        "ok": True,
        "target_verified": acct_num == TARGET or _account_from_url(portfolio.get("account")) == TARGET,
        "account_masked": _mask(acct_num),
        "agentic_allowed": acct.get("agentic_allowed"),
        "option_level": acct.get("option_level"),
        "buying_power": acct.get("buying_power") or portfolio.get("withdrawable_amount") or portfolio.get("extended_hours_equity"),
    })

    def safe_call(label, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError:
            try:
                kwargs.pop("account_number", None)
                return fn(*args, **kwargs)
            except Exception as e:
                out["errors"].append(f"{label}_failed:{type(e).__name__}:{e}")
                return []
        except Exception as e:
            out["errors"].append(f"{label}_failed:{type(e).__name__}:{e}")
            return []

    open_pos = safe_call("positions", rh.options.get_open_option_positions, account_number=TARGET)
    agg_pos = safe_call("agg_positions", rh.options.get_aggregate_open_positions, account_number=TARGET)
    open_orders = safe_call("open_orders", rh.orders.get_all_open_option_orders, account_number=TARGET)
    nonzero_positions = []
    for p in (open_pos or []):
        try:
            qty = float(p.get("quantity") or p.get("chain_quantity") or 0)
        except Exception:
            qty = 0.0
        if abs(qty) > 0:
            nonzero_positions.append({
                "chain_symbol": p.get("chain_symbol"),
                "option": p.get("option"),
                "quantity": p.get("quantity") or p.get("chain_quantity"),
                "average_price": p.get("average_price"),
                "account_masked": _mask(_account_from_url(p.get("account"))),
            })
    orders_summary = []
    for o in (open_orders or [])[:10]:
        orders_summary.append({
            "id": o.get("id"),
            "state": o.get("state"),
            "direction": o.get("direction"),
            "opening_strategy": o.get("opening_strategy"),
            "closing_strategy": o.get("closing_strategy"),
            "price": o.get("price"),
            "quantity": o.get("quantity"),
            "account_masked": _mask(_account_from_url(o.get("account"))),
        })
    out.update({
        "open_option_positions_count": len(open_pos or []),
        "aggregate_option_positions_count": len(agg_pos or []),
        "nonzero_option_positions_count": len(nonzero_positions),
        "nonzero_positions": nonzero_positions,
        "open_option_orders_count": len(open_orders or []),
        "open_orders": orders_summary,
    })
    print(json.dumps(out, separators=(",", ":")))

if __name__ == "__main__":
    main()
