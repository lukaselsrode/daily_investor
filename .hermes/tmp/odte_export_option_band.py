#!/usr/bin/env python3
"""Read-only Robinhood option band exporter for ODTE gamma/vehicle scoring."""
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
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def safe_float(x: Any) -> float | None:
    try:
        if x in (None, ""):
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    f = safe_float(x)
    return int(f) if f is not None else None


def first(obj: Any) -> dict[str, Any]:
    if isinstance(obj, list) and obj:
        head = obj[0]
        if isinstance(head, list):
            return first(head)
        return head if isinstance(head, dict) else {}
    return obj if isinstance(obj, dict) else {}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    load_env(repo / ".env")
    load_env(Path.home() / "0dte" / ".env")
    target = "435050133"
    os.environ["RB_ACCT_NUM"] = target
    os.environ["ROBINHOOD_ACCOUNT_NUMBER"] = target
    os.environ["RH_ACCOUNT_NUMBER"] = target
    symbols = [s.strip().upper() for s in (sys.argv[1] if len(sys.argv) > 1 else "SPY,QQQ,IWM").split(",") if s.strip()]
    expiration = sys.argv[2] if len(sys.argv) > 2 else dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).date().isoformat()
    out: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "expiration": expiration, "read_only": True, "symbols": {}, "errors": []}
    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:
        out["errors"].append(f"import_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(out, sort_keys=True))
        return 0
    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME") or os.environ.get("RH_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD") or os.environ.get("RH_PASSWORD")
    secret = os.environ.get("RB_MFA_SECRET") or os.environ.get("ROBINHOOD_MFA_SECRET") or os.environ.get("RH_MFA_SECRET")
    code = None
    if secret:
        with contextlib.suppress(Exception):
            code = pyotp.TOTP(secret.strip().replace(" ", "")).now()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        login = rb.login(username=username, password=password, mfa_code=code, store_session=True)
    out["login_ok"] = bool(isinstance(login, dict) and login.get("access_token"))
    for sym in symbols:
        block: dict[str, Any] = {"rows": [], "errors": []}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                q = rb.stocks.get_quotes(sym) or []
            spot = safe_float(first(q).get("last_trade_price"))
            block["spot"] = spot
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                instruments = rb.options.find_options_by_expiration(sym, expiration) or []
            if not isinstance(instruments, list):
                instruments = []
            strike_values = []
            for item in instruments:
                if isinstance(item, dict):
                    value = safe_float(item.get("strike_price"))
                    if value is not None:
                        strike_values.append(value)
            strikes = sorted(set(strike_values))
            if spot is not None:
                strikes = sorted(strikes, key=lambda x: abs(x - spot))[:24]
            else:
                strikes = strikes[:24]
            for strike in sorted(strikes):
                for opt_type in ("call", "put"):
                    inst = next((i for i in instruments if isinstance(i, dict) and safe_float(i.get("strike_price")) == strike and str(i.get("type") or i.get("option_type")).lower() == opt_type), {})
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            md = first(rb.options.get_option_market_data(sym, expiration, str(strike).rstrip("0").rstrip("."), opt_type) or [])
                    except Exception as exc:
                        block["errors"].append(f"market_data_failed:{strike}:{opt_type}:{type(exc).__name__}:{exc}")
                        md = {}
                    row = {
                        "chain_symbol": sym,
                        "expiration_date": expiration,
                        "strike_price": strike,
                        "type": opt_type,
                        "bid_price": safe_float(md.get("bid_price") or md.get("bid") or inst.get("bid_price")),
                        "ask_price": safe_float(md.get("ask_price") or md.get("ask") or inst.get("ask_price")),
                        "mark_price": safe_float(md.get("mark_price") or md.get("adjusted_mark_price") or md.get("mark") or inst.get("mark_price") or inst.get("adjusted_mark_price")),
                        "implied_volatility": safe_float(md.get("implied_volatility") or inst.get("implied_volatility")),
                        "delta": safe_float(md.get("delta") or inst.get("delta")),
                        "gamma": safe_float(md.get("gamma") or inst.get("gamma")),
                        "theta": safe_float(md.get("theta") or inst.get("theta")),
                        "open_interest": safe_int(md.get("open_interest") or inst.get("open_interest")),
                        "volume": safe_int(md.get("volume") or inst.get("volume")),
                        "updated_at": md.get("updated_at") or md.get("previous_close_date"),
                        "state": inst.get("state"),
                        "tradability": inst.get("tradability"),
                        "instrument_id": inst.get("id"),
                        "url": inst.get("url"),
                    }
                    if row["bid_price"] is not None or row["ask_price"] is not None or row["mark_price"] is not None:
                        block["rows"].append(row)
        except Exception as exc:
            block["errors"].append(f"symbol_failed:{type(exc).__name__}:{exc}")
        out["symbols"][sym] = block
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
