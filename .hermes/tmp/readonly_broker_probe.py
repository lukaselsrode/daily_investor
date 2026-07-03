#!/usr/bin/env python3
"""Read-only Robinhood probe for the configured Agentic options account.

Prints JSON only. Suppresses third-party stdout/stderr around Robinhood calls.
Never places/cancels orders.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET = os.environ.get("RB_ACCT_NUM") or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER") or os.environ.get("RH_ACCOUNT_NUMBER") or "435050133"
SYMBOLS = [s.strip().upper() for s in os.environ.get("PROBE_SYMBOLS", "SPY,QQQ,IWM,VIXY,TSLA,MU").split(",") if s.strip()]


def load_env() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def mask(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    return "****" + s[-4:] if len(s) >= 4 else "****"


def scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(secret in lk for secret in ("token", "password", "secret", "mfa", "authorization")):
                out[k] = "[redacted]"
            elif lk in {"account_number", "account", "account_id"}:
                out[k] = mask(v)
            elif isinstance(v, str) and TARGET in v:
                out[k] = v.replace(TARGET, mask(TARGET) or "****")
            else:
                out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, str) and TARGET in obj:
        return obj.replace(TARGET, mask(TARGET) or "****")
    return obj


@contextlib.contextmanager
def quiet():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def call(fn, *args, **kwargs):
    with quiet():
        return fn(*args, **kwargs)


def main() -> int:
    load_env()
    # Force target aliases for any helper path that consults env.
    os.environ["RB_ACCT_NUM"] = TARGET
    os.environ["ROBINHOOD_ACCOUNT_NUMBER"] = TARGET
    os.environ["RH_ACCOUNT_NUMBER"] = TARGET

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_account_masked": mask(TARGET),
        "read_only": True,
        "places_orders": False,
        "account_verified": False,
        "errors": [],
    }
    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"import_failed: {type(exc).__name__}: {exc}")
        print(json.dumps(result, sort_keys=True))
        return 2

    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD")
    mfa_secret = os.environ.get("RB_MFA_SECRET") or os.environ.get("ROBINHOOD_MFA_SECRET")
    if not username or not password:
        result["errors"].append("missing_credentials")
        print(json.dumps(result, sort_keys=True))
        return 2
    mfa_code = None
    if mfa_secret:
        try:
            mfa_code = pyotp.TOTP(mfa_secret.strip().replace(" ", "").replace("-", "").upper()).now()
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"mfa_generation_failed: {type(exc).__name__}: {exc}")
    try:
        login_resp = call(rb.login, username=username, password=password, mfa_code=mfa_code, store_session=True)
        result["login_ok"] = bool(isinstance(login_resp, dict) and login_resp.get("access_token"))
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"login_failed: {type(exc).__name__}: {exc}")
        print(json.dumps(result, sort_keys=True))
        return 2

    # Account + portfolio, explicit target when supported.
    try:
        account = call(rb.profiles.load_account_profile, account_number=TARGET)
        result["account"] = scrub(account)
        acct_url = json.dumps(account)
        result["account_verified"] = TARGET in acct_url or (isinstance(account, dict) and str(account.get("account_number")) == TARGET)
    except TypeError:
        account = call(rb.profiles.load_account_profile)
        result["account"] = scrub(account)
        result["account_verified"] = TARGET in json.dumps(account)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"account_profile_failed: {type(exc).__name__}: {exc}")

    try:
        portfolio = call(rb.profiles.load_portfolio_profile, account_number=TARGET)
        result["portfolio"] = scrub(portfolio)
    except TypeError:
        portfolio = call(rb.profiles.load_portfolio_profile)
        result["portfolio"] = scrub(portfolio)
        result["errors"].append("portfolio_profile_did_not_accept_account_number")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"portfolio_failed: {type(exc).__name__}: {exc}")

    # Open option positions and orders.
    try:
        positions = call(rb.options.get_open_option_positions, account_number=TARGET)
        if positions is None:
            positions = []
        result["open_option_positions_count"] = len(positions) if isinstance(positions, list) else None
        result["open_option_positions"] = scrub(positions)
    except TypeError:
        positions = call(rb.options.get_open_option_positions)
        result["open_option_positions_count"] = len(positions) if isinstance(positions, list) else None
        result["open_option_positions"] = scrub(positions)
        result["errors"].append("positions_call_did_not_accept_account_number")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"positions_failed: {type(exc).__name__}: {exc}")

    try:
        agg_positions = call(rb.options.get_aggregate_open_positions, account_number=TARGET)
        result["aggregate_open_positions"] = scrub(agg_positions)
    except TypeError:
        pass
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"aggregate_positions_failed: {type(exc).__name__}: {exc}")

    try:
        open_orders = call(rb.orders.get_all_open_option_orders, account_number=TARGET)
        if open_orders is None:
            open_orders = []
        result["open_option_orders_count"] = len(open_orders) if isinstance(open_orders, list) else None
        result["open_option_orders"] = scrub(open_orders)
    except TypeError:
        open_orders = call(rb.orders.get_all_open_option_orders)
        result["open_option_orders_count"] = len(open_orders) if isinstance(open_orders, list) else None
        result["open_option_orders"] = scrub(open_orders)
        result["errors"].append("open_orders_call_did_not_accept_account_number")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"open_orders_failed: {type(exc).__name__}: {exc}")

    try:
        recent_orders = call(rb.orders.get_all_option_orders, account_number=TARGET)
        if isinstance(recent_orders, list):
            result["recent_option_orders"] = scrub(recent_orders[:5])
            result["recent_option_orders_count_returned"] = len(recent_orders)
    except TypeError:
        pass
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"recent_orders_failed: {type(exc).__name__}: {exc}")

    # Equity quotes for broad tape; read-only.
    quotes = {}
    try:
        q_rows = call(rb.stocks.get_quotes, SYMBOLS) or []
        for row in q_rows:
            if isinstance(row, dict):
                quotes[row.get("symbol") or row.get("instrument") or str(len(quotes))] = scrub(row)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"quotes_failed: {type(exc).__name__}: {exc}")
    result["quotes"] = quotes

    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("account_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
