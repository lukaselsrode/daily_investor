#!/usr/bin/env python3
"""Read-only Robinhood/account probe for ODTE cron. Prints JSON only."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def mask(acct: str | None) -> str | None:
    if not acct:
        return None
    return "****" + str(acct)[-4:]


def main() -> int:
    load_dotenv(Path(".env"))
    target = (
        os.environ.get("RB_ACCT_NUM")
        or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER")
        or os.environ.get("RH_ACCOUNT_NUMBER")
        or "435050133"
    )
    # Force target-account aliases for helpers that inspect env.
    os.environ["RB_ACCT_NUM"] = target
    os.environ["ROBINHOOD_ACCOUNT_NUMBER"] = target
    os.environ["RH_ACCOUNT_NUMBER"] = target

    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME") or os.environ.get("RH_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD") or os.environ.get("RH_PASSWORD")
    mfa_secret = os.environ.get("RB_MFA_SECRET") or os.environ.get("ROBINHOOD_MFA_SECRET") or os.environ.get("RH_MFA_SECRET")

    result = {
        "asof_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_masked": mask(target),
        "source": "local_robin_stocks_read_only_probe_with_explicit_target_override",
        "places_orders": False,
        "errors": [],
        "warnings": [],
    }
    if not username or not password:
        result["errors"].append("missing_robinhood_credentials")
        print(json.dumps(result, separators=(",", ":")))
        return 1

    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:
        result["errors"].append(f"import_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(result, separators=(",", ":")))
        return 1

    mfa_code = None
    if mfa_secret:
        clean = mfa_secret.strip().replace(" ", "").replace("-", "").upper()
        bad = sorted({c for c in clean if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"})
        if bad:
            result["warnings"].append("mfa_secret_non_base32_cached_session_may_be_used")
        else:
            try:
                mfa_code = pyotp.TOTP(clean).now()
            except Exception as exc:
                result["warnings"].append(f"mfa_generation_failed:{type(exc).__name__}")

    try:
        # Suppress robin_stocks login chatter.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
    except Exception as exc:
        result["warnings"].append(f"login_exception_cached_session_may_still_work:{type(exc).__name__}:{exc}")

    try:
        acct = rb.profiles.load_account_profile(account_number=target)
        port = rb.profiles.load_portfolio_profile(account_number=target)
        open_pos = rb.options.get_open_option_positions(account_number=target) or []
        open_orders = rb.orders.get_all_open_option_orders(account_number=target) or []
        quotes = rb.stocks.get_quotes(["SPY", "QQQ", "IWM", "VIXY"], info=None) or []

        # target verification evidence
        acct_num = str(acct.get("account_number") or acct.get("account") or target)
        account_url = str(acct.get("url") or acct.get("account") or "")
        target_verified = (acct_num == str(target)) or (str(target) in account_url)
        nonzero_pos = []
        for p in open_pos:
            qty = fnum(p.get("quantity") or p.get("chain_quantity") or p.get("pending_quantity"))
            if qty and abs(qty) > 1e-9:
                nonzero_pos.append(p)
        result.update(
            {
                "target_verified": bool(target_verified),
                "masked_account": mask(target),
                "agentic_allowed": acct.get("agentic_allowed"),
                "option_level": acct.get("option_level"),
                "buying_power": fnum(acct.get("buying_power") or port.get("withdrawable_amount") or port.get("market_value")),
                "cash_available_for_withdrawal": fnum(acct.get("cash_available_for_withdrawal")),
                "open_option_positions_count": len(open_pos),
                "nonzero_option_positions_count": len(nonzero_pos),
                "open_option_orders_count": len(open_orders),
                "quotes": [
                    {
                        "symbol": q.get("symbol"),
                        "last_trade_price": fnum(q.get("last_trade_price")),
                        "ask_price": fnum(q.get("ask_price")),
                        "bid_price": fnum(q.get("bid_price")),
                        "previous_close": fnum(q.get("previous_close")),
                        "updated_at": q.get("updated_at"),
                    }
                    for q in quotes
                ],
            }
        )
    except Exception as exc:
        result["errors"].append(f"probe_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(result, separators=(",", ":")))
        return 1

    print(json.dumps(result, separators=(",", ":")))
    return 0 if result.get("target_verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
