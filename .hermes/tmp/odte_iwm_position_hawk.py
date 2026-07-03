#!/usr/bin/env python3
"""Read-only HAWK monitor for the current IWM 303C scalp; no orders/reviews."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time
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


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


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
    checks = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    out: dict[str, Any] = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "read_only": True, "checks": [], "errors": []}
    try:
        import pyotp
        import robin_stocks.robinhood as rb
    except Exception as exc:
        out["errors"].append(f"import_failed:{type(exc).__name__}:{exc}")
        print(json.dumps(out, indent=2, sort_keys=True))
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
    stop_decisions = {"TAKE_PROFIT", "THESIS_DEAD", "BID_FLOOR", "TIME_RISK", "MONITORING_DEGRADED"}
    for idx in range(checks):
        check: dict[str, Any] = {"i": idx + 1, "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                positions = rb.options.get_open_option_positions(account_number=target) or []
                orders = rb.orders.get_all_open_option_orders(account_number=target) or []
                md = first(rb.options.get_option_market_data_by_id("0eda8096-cf70-4fc8-b75b-1c8f1df9657b") or [])
                qs = rb.stocks.get_quotes(["IWM", "SPY", "QQQ", "VIXY"]) or []
            quotes = {q.get("symbol"): q for q in qs if isinstance(q, dict)}
            nonzero = []
            for p in positions if isinstance(positions, list) else []:
                qty = safe_float(p.get("quantity") or p.get("intraday_quantity"))
                if qty and abs(qty) > 1e-9:
                    nonzero.append(p)
            snap = {
                "now_et": dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=-4))).isoformat(),
                "option_id": "0eda8096-cf70-4fc8-b75b-1c8f1df9657b",
                "option_bid": safe_float(md.get("bid_price")),
                "option_ask": safe_float(md.get("ask_price")),
                "option_mark": safe_float(md.get("mark_price") or md.get("adjusted_mark_price")),
                "underlying_price": safe_float(quotes.get("IWM", {}).get("last_trade_price")),
                "iwm": safe_float(quotes.get("IWM", {}).get("last_trade_price")),
                "spy": safe_float(quotes.get("SPY", {}).get("last_trade_price")),
                "qqq": safe_float(quotes.get("QQQ", {}).get("last_trade_price")),
                "vixy": safe_float(quotes.get("VIXY", {}).get("last_trade_price")),
                "broker_verified": True,
                "position_quantity": len(nonzero),
                "open_option_orders_count": len(orders) if isinstance(orders, list) else None,
            }
            snap_path = Path(f"/tmp/odte_iwm_hawk_snapshot_{idx+1}.json")
            snap_path.write_text(json.dumps(snap, indent=2, sort_keys=True))
            proc = subprocess.run([
                str(repo / ".venv/bin/daily-investor"), "odte-position", "--snapshot", str(snap_path),
                "--plan", str(repo / "data/odte/active_trade.json"), "--json"
            ], cwd=repo, text=True, capture_output=True, timeout=30)
            decision = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip().startswith("{") else {"decision": "MONITORING_DEGRADED", "error": proc.stderr or proc.stdout}
            check.update({"snapshot": snap, "decision": decision})
            out["checks"].append(check)
            if str(decision.get("decision")) in stop_decisions or not nonzero:
                break
        except Exception as exc:
            check["error"] = f"{type(exc).__name__}:{exc}"
            out["checks"].append(check)
            break
        if idx != checks - 1:
            time.sleep(interval)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
