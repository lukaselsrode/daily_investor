#!/usr/bin/env python3
"""Read-only intraday market snapshot for ODTE day-score.

Suppresses Robinhood chatter and prints JSON only. No orders/reviews.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
from typing import Any


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def pct(last: float | None, prev: float | None) -> float | None:
    if last is None or prev in (None, 0):
        return None
    return (last - prev) / prev * 100.0


def bar_time(row: dict[str, Any]) -> dt.datetime | None:
    raw = row.get("begins_at") or row.get("updated_at")
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def compute_symbol(row: dict[str, Any], bars: list[dict[str, Any]]) -> dict[str, Any]:
    last = safe_float(row.get("last_trade_price"))
    prev = safe_float(row.get("previous_close"))
    regular = []
    for bar in bars:
        ts = bar_time(bar)
        if ts is None:
            continue
        et = ts.astimezone(dt.timezone(dt.timedelta(hours=-4)))
        if et.time() >= dt.time(9, 30) and et.time() <= dt.time(16, 0):
            regular.append((et, bar))
    pv = 0.0
    vol = 0.0
    for _ts, bar in regular:
        close = safe_float(bar.get("close_price") or bar.get("close"))
        volume = safe_float(bar.get("volume")) or 0.0
        if close is not None and volume > 0:
            pv += close * volume
            vol += volume
    vwap = pv / vol if vol > 0 else None
    first_30 = [(ts, b) for ts, b in regular if ts.time() < dt.time(10, 0)]
    highs = [safe_float(b.get("high_price") or b.get("high")) for _ts, b in first_30]
    lows = [safe_float(b.get("low_price") or b.get("low")) for _ts, b in first_30]
    highs = [x for x in highs if x is not None]
    lows = [x for x in lows if x is not None]
    orb_high = max(highs) if highs else None
    orb_low = min(lows) if lows else None
    orb_state = None
    if last is not None and orb_high is not None and orb_low is not None:
        if last > orb_high:
            orb_state = "above"
        elif last < orb_low:
            orb_state = "below"
        else:
            orb_state = "inside"
    return {
        "last": last,
        "previous_close": prev,
        "day_change_pct": pct(last, prev),
        "vwap": vwap,
        "above_vwap": (last > vwap) if last is not None and vwap is not None else None,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_state": orb_state,
        "quote_updated_at": row.get("updated_at"),
        "bars": len(regular),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    load_env(repo / ".env")
    load_env(Path.home() / "0dte" / ".env")
    target = "435050133"
    os.environ["RB_ACCT_NUM"] = target
    os.environ["ROBINHOOD_ACCOUNT_NUMBER"] = target
    os.environ["RH_ACCOUNT_NUMBER"] = target
    out: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "read_only": True, "errors": []}
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
        with contextlib.suppress(Exception):
            mfa_code = pyotp.TOTP(mfa_secret.strip().replace(" ", "")).now()
    if not username or not password:
        out["errors"].append("missing_credentials")
        print(json.dumps(out, sort_keys=True))
        return 0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        login = rb.login(username=username, password=password, mfa_code=mfa_code, store_session=True)
    out["login_ok"] = bool(isinstance(login, dict) and login.get("access_token"))
    symbols = ["SPY", "QQQ", "IWM", "VIXY"]
    quotes = {}
    histories = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        quote_rows = rb.stocks.get_quotes(symbols) or []
    for q in quote_rows:
        if isinstance(q, dict) and q.get("symbol"):
            quotes[q["symbol"]] = q
    for sym in symbols:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bars = rb.stocks.get_stock_historicals(sym, interval="5minute", span="day", bounds="regular") or []
            histories[sym] = bars if isinstance(bars, list) else []
        except Exception as exc:
            out["errors"].append(f"historical_failed:{sym}:{type(exc).__name__}:{exc}")
            histories[sym] = []
    market: dict[str, Any] = {"symbols": {}}
    for sym in symbols:
        block = compute_symbol(quotes.get(sym, {}), histories.get(sym, []))
        market["symbols"][sym] = block
    spy = market["symbols"].get("SPY", {})
    market.update({
        "spy_above_vwap": market["symbols"].get("SPY", {}).get("above_vwap"),
        "qqq_above_vwap": market["symbols"].get("QQQ", {}).get("above_vwap"),
        "iwm_above_vwap": market["symbols"].get("IWM", {}).get("above_vwap"),
        "vixy_above_vwap": market["symbols"].get("VIXY", {}).get("above_vwap"),
        "spy_orb_state": market["symbols"].get("SPY", {}).get("orb_state"),
        "qqq_orb_state": market["symbols"].get("QQQ", {}).get("orb_state"),
        "iwm_orb_state": market["symbols"].get("IWM", {}).get("orb_state"),
        "vixy_change_pct": market["symbols"].get("VIXY", {}).get("day_change_pct"),
        "gap_pct": spy.get("day_change_pct"),
        "expected_move_pct": 0.8,
    })
    now_et = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=-4)))
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    market["minutes_to_close"] = max(0.0, (close_et - now_et).total_seconds() / 60.0)
    out["market"] = market
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
