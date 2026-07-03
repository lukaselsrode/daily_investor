#!/usr/bin/env python3
"""Read-only live ODTE candidate probe for cron: broker/account, broad tape, SPY 0DTE chain rows.
Prints/writes JSON only; no order review/place/cancel.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import math
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
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def sf(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def mask(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x)
    return "***" + s[-4:]


def try_call(fn, *args, **kwargs) -> tuple[bool, Any]:
    try:
        return True, fn(*args, **kwargs)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def parse_time(s: Any) -> dt.datetime | None:
    if not s:
        return None
    txt = str(s).replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(txt)
    except Exception:
        return None


def quote_price(q: dict[str, Any]) -> float | None:
    return sf(q.get("last_trade_price")) or sf(q.get("last_extended_hours_trade_price")) or sf(q.get("mark_price"))


def hist_stats(rows: list[dict[str, Any]], quote: float | None) -> dict[str, Any]:
    parsed = []
    for r in rows or []:
        ts = parse_time(r.get("begins_at"))
        close = sf(r.get("close_price")) or sf(r.get("close"))
        high = sf(r.get("high_price")) or sf(r.get("high"))
        low = sf(r.get("low_price")) or sf(r.get("low"))
        vol = sf(r.get("volume")) or 0.0
        if ts and close is not None:
            parsed.append((ts, close, high, low, vol))
    if not parsed:
        return {"above_vwap": None, "orb_state": "unknown", "vwap": None, "orb_high": None, "orb_low": None}
    # RH day/5minute regular bars for current session; first 30 minutes define ORB.
    num = sum(c * v for _, c, _, _, v in parsed if v and c is not None)
    den = sum(v for *_, v in parsed if v)
    if den <= 0:
        vwap = sum(c for _, c, *_ in parsed) / len(parsed)
    else:
        vwap = num / den
    first_ts = min(p[0] for p in parsed)
    orb = [p for p in parsed if (p[0] - first_ts).total_seconds() < 30 * 60]
    highs = [h for _, _, h, _, _ in orb if h is not None]
    lows = [l for _, _, _, l, _ in orb if l is not None]
    orb_high = max(highs) if highs else None
    orb_low = min(lows) if lows else None
    px = quote if quote is not None else parsed[-1][1]
    if orb_high is not None and px > orb_high:
        state = "above"
    elif orb_low is not None and px < orb_low:
        state = "below"
    elif orb_high is not None and orb_low is not None:
        state = "inside"
    else:
        state = "unknown"
    return {"above_vwap": bool(px > vwap) if px is not None and vwap is not None else None,
            "orb_state": state, "vwap": round(vwap, 4) if vwap is not None else None,
            "orb_high": orb_high, "orb_low": orb_low, "bars": len(parsed)}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    load_env(repo / ".env")
    load_env(Path.home() / "0dte" / ".env")
    target = os.environ.get("RB_ACCT_NUM") or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER") or os.environ.get("RH_ACCOUNT_NUMBER") or "435050133"
    username = os.environ.get("RB_ACCT") or os.environ.get("ROBINHOOD_USERNAME") or os.environ.get("RH_USERNAME")
    password = os.environ.get("RB_CREDS") or os.environ.get("ROBINHOOD_PASSWORD") or os.environ.get("RH_PASSWORD")
    out: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "target_account_masked": mask(target), "errors": [], "places_orders": False}
    if not username or not password:
        out.update({"blocked": True, "errors": ["missing credentials"]})
        print(json.dumps(out, indent=2, default=str)); return 0
    try:
        import robin_stocks.robinhood as rb
    except Exception as exc:
        out.update({"blocked": True, "errors": [f"import failed: {type(exc).__name__}: {exc}"]})
        print(json.dumps(out, indent=2, default=str)); return 0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ok, login = try_call(rb.login, username=username, password=password, store_session=True)
    out["login_ok"] = bool(ok and isinstance(login, dict) and login.get("access_token"))
    if not out["login_ok"]:
        out.update({"blocked": True}); out["errors"].append(str(login)); print(json.dumps(out, indent=2, default=str)); return 0
    okp, prof = try_call(rb.profiles.load_account_profile, account_number=target)
    oko, orders = try_call(rb.orders.get_all_open_option_orders, account_number=target)
    okpos, positions = try_call(rb.options.get_open_option_positions, account_number=target)
    acct_seen = prof.get("account_number") if okp and isinstance(prof, dict) else None
    out["broker"] = {"target_verified": str(acct_seen) == str(target), "account_masked": mask(acct_seen),
                      "buying_power": sf(prof.get("buying_power")) if isinstance(prof, dict) else None,
                      "cash": sf(prof.get("cash")) if isinstance(prof, dict) else None,
                      "option_level": prof.get("option_level") if isinstance(prof, dict) else None,
                      "open_option_orders_count": len(orders or []) if oko and isinstance(orders, list) else None,
                      "open_option_positions_nonzero_count": 0}
    nonzero = []
    for p in positions or [] if okpos and isinstance(positions, list) else []:
        qty = sf(p.get("quantity") or p.get("intraday_quantity"))
        if qty and abs(qty) > 1e-9:
            nonzero.append({"chain_symbol": p.get("chain_symbol"), "option": p.get("option"), "quantity": qty, "average_price": p.get("average_price")})
    out["broker"]["open_option_positions_nonzero_count"] = len(nonzero)
    out["broker"]["nonzero_positions"] = nonzero[:10]

    symbols = ["SPY", "QQQ", "IWM", "VIXY"]
    okq, qres = try_call(rb.stocks.get_quotes, symbols)
    quotes: dict[str, Any] = {}
    if okq and isinstance(qres, list):
        for q in qres:
            if isinstance(q, dict) and q.get("symbol"):
                quotes[q["symbol"]] = q
    else:
        out["errors"].append("quotes failed: " + str(qres))
    market: dict[str, Any] = {"source": "robinhood_intraday", "approximated": False}
    for sym in symbols:
        q = quotes.get(sym, {})
        px = quote_price(q)
        prev = sf(q.get("previous_close"))
        market[f"{sym.lower()}_last"] = px
        if prev:
            market[f"{sym.lower()}_change_pct"] = (px / prev - 1.0) * 100 if px else None
        okh, hist = try_call(rb.stocks.get_stock_historicals, sym, interval="5minute", span="day", bounds="regular")
        if not okh or not isinstance(hist, list):
            out["errors"].append(f"{sym} historicals failed: {hist}")
            st = {"above_vwap": None, "orb_state": "unknown", "vwap": None, "orb_high": None, "orb_low": None}
        else:
            st = hist_stats(hist, px)
        market[f"{sym.lower()}_above_vwap"] = st["above_vwap"]
        market[f"{sym.lower()}_orb_state"] = st["orb_state"]
        market[f"{sym.lower()}_vwap"] = st["vwap"]
        market[f"{sym.lower()}_orb_high"] = st["orb_high"]
        market[f"{sym.lower()}_orb_low"] = st["orb_low"]
        market[f"{sym.lower()}_bars"] = st.get("bars")
    # Direct VIX often unavailable in RH stocks; VIXY change is still in market, use VIX proxy if not present.
    market["vix"] = None
    market["vixy_change_pct"] = market.get("vixy_change_pct")
    # 16:00 ET close; use UTC now converted roughly via zoneinfo.
    try:
        from zoneinfo import ZoneInfo
        now_et = dt.datetime.now(ZoneInfo("America/New_York"))
        close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        market["minutes_to_close"] = max(0, int((close_et - now_et).total_seconds() // 60))
    except Exception:
        market["minutes_to_close"] = None
    # gap based on SPY previous close / current; expected move approximated from 0DTE straddle later if possible.
    market["gap_pct"] = market.get("spy_change_pct")

    today = dt.datetime.now(dt.timezone.utc).astimezone().date().isoformat()
    underlying = "SPY"
    spot = market.get("spy_last")
    option_rows = []
    if spot:
        center = int(round(float(spot)))
        strikes = list(range(center - 5, center + 7))
        for opt_type in ["call", "put"]:
            for strike in strikes:
                okinst, inst = try_call(rb.options.find_options_by_expiration_and_strike, underlying, today, str(float(strike)), opt_type)
                if not okinst or not inst:
                    okinst, inst = try_call(rb.options.find_options_by_expiration_and_strike, underlying, today, str(strike), opt_type)
                inst_list = inst if isinstance(inst, list) else ([inst] if isinstance(inst, dict) else [])
                for item in inst_list[:1]:
                    if not isinstance(item, dict):
                        continue
                    oid = item.get("id") or str(item.get("url", "")).rstrip("/").split("/")[-1]
                    okmd, md = try_call(rb.options.get_option_market_data_by_id, oid) if oid else (False, None)
                    md_item = md[0] if isinstance(md, list) and md else (md if isinstance(md, dict) else {})
                    row = {**item, **md_item}
                    bid = sf(row.get("bid_price") or row.get("bid"))
                    ask = sf(row.get("ask_price") or row.get("ask"))
                    mark = sf(row.get("mark_price") or row.get("adjusted_mark_price"))
                    option_rows.append({
                        "chain_symbol": underlying,
                        "expiration_date": today,
                        "strike_price": sf(row.get("strike_price")) or float(strike),
                        "type": opt_type,
                        "bid_price": bid,
                        "ask_price": ask,
                        "mark_price": mark,
                        "implied_volatility": sf(row.get("implied_volatility")),
                        "delta": sf(row.get("delta")),
                        "gamma": sf(row.get("gamma")),
                        "open_interest": int(sf(row.get("open_interest")) or 0),
                        "volume": int(sf(row.get("volume")) or 0),
                        "updated_at": row.get("updated_at"),
                        "id": oid,
                    })
                    break
    out["market"] = market
    out["option_chain"] = {"underlying": underlying, "expiration": today, "spot": spot, "rows": option_rows}
    # Pick a primary call nearest $0.40-$0.90 mark with <= ~70 debit; else nearest ATM affordable.
    calls = [r for r in option_rows if r.get("type") == "call" and r.get("ask_price") is not None]
    calls.sort(key=lambda r: (0 if 0.2 <= (r.get("mark_price") or r.get("ask_price") or 999) <= 0.9 else 1, abs((r.get("strike_price") or 0) - (spot or 0))))
    if calls:
        c = calls[0]
        out["candidate_contract"] = {"underlying": underlying, "option_type": "call", "strike": c.get("strike_price"),
                                     "expiration": today, "bid": c.get("bid_price"), "ask": c.get("ask_price"),
                                     "mark": c.get("mark_price"), "delta": c.get("delta"), "gamma": c.get("gamma"),
                                     "volume": c.get("volume"), "open_interest": c.get("open_interest"),
                                     "option_id": c.get("id"), "updated_at": c.get("updated_at")}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
