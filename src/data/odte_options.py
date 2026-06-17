"""
data/odte_options.py — PAPER-ONLY same-day (0DTE) option-chain lookup.

ANALYSIS / DECISION-SUPPORT ONLY. **Places NO orders, imports NO broker/execution code.**
Reads the current same-day expiry option chain for a ticker via yfinance (best-effort,
fails closed when the market is closed / no same-day expiry / no network / yfinance missing)
and returns budget-fitting, liquidity-sorted candidate contracts matching a social direction
(bullish -> calls, bearish -> puts). It does not size, recommend, or place anything.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else default  # NaN -> default
    except (TypeError, ValueError):
        return default


def fetch_same_day_chain(ticker: str, allow_fetch: bool = True, today: str | None = None):
    """Return ({"expiry","calls","puts"}, status) for the SAME-DAY expiry, else (None, status).
    Cache-free, network best-effort, fails closed. `today` (YYYY-MM-DD) overridable for tests."""
    if not allow_fetch:
        return None, "skipped: allow_fetch=False"
    today = today or date.today().isoformat()
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        expiries = list(getattr(tk, "options", []) or [])
        if today not in expiries:
            return None, f"no same-day (0DTE) expiry for {ticker} (today={today}; expiries={expiries[:3]})"
        chain = tk.option_chain(today)
        return {"expiry": today, "calls": chain.calls, "puts": chain.puts}, "ok"
    except Exception as exc:
        logger.warning("0DTE chain fetch failed for %s: %s", ticker, exc)
        return None, f"error: {exc}"


def select_paper_contracts(chain: dict, direction: str, budget_dollars: float,
                           max_contracts: int = 3) -> list[dict]:
    """Pick budget-fitting, liquid contracts matching direction (bullish->calls, bearish->puts).
    cost = premium*100 (premium = ask, or last if ask missing); kept when 0 < cost <= cap, where
    cap = budget_dollars (or $50 if budget is 0). Requires bid>0 & ask>0; prefers nonzero
    volume/open-interest when those columns exist. Sorted by tight spread then liquidity."""
    df = chain.get("calls" if direction == "bullish" else "puts")
    if df is None or getattr(df, "empty", True):
        return []
    cap = float(budget_dollars) if float(budget_dollars) > 0 else 50.0
    has_vol = "volume" in df.columns
    has_oi = "openInterest" in df.columns
    rows: list[dict] = []
    for _, r in df.iterrows():
        bid, ask, last = _f(r.get("bid")), _f(r.get("ask")), _f(r.get("lastPrice"))
        premium = ask if ask > 0 else last
        cost = premium * 100.0
        if not (premium > 0 and 0 < cost <= cap):
            continue
        if not (bid > 0 and ask > 0):
            continue
        vol = _f(r.get("volume")) if has_vol else 0.0
        oi = _f(r.get("openInterest")) if has_oi else 0.0
        if (has_vol or has_oi) and (vol + oi) <= 0:
            continue  # require SOME liquidity when the data is available
        mid = (ask + bid) / 2.0
        rows.append({
            "option_type": "call" if direction == "bullish" else "put",
            "strike": _f(r.get("strike")),
            "bid": round(bid, 4), "ask": round(ask, 4), "last": round(last, 4),
            "premium_cost_estimate": round(cost, 2),
            "spread_pct": round((ask - bid) / mid, 4) if mid > 0 else None,
            "volume": int(vol), "open_interest": int(oi),
        })
    rows.sort(key=lambda c: (c["spread_pct"] if c["spread_pct"] is not None else 9.9,
                             -(c["volume"] + c["open_interest"])))
    return rows[:max_contracts]


def build_paper_options(ticker: str, direction: str, budget_dollars: float,
                        allow_fetch: bool = True, today: str | None = None,
                        max_contracts: int = 3) -> dict:
    """PAPER-ONLY 0DTE option idea for `ticker` matching `direction`. Never places orders."""
    if direction not in ("bullish", "bearish"):
        return {"status": f"no directional signal ({direction}) — no contracts", "expiry": None,
                "option_type": None, "contracts": []}
    chain, status = fetch_same_day_chain(ticker, allow_fetch=allow_fetch, today=today)
    if chain is None:
        return {"status": status, "expiry": None,
                "option_type": "call" if direction == "bullish" else "put", "contracts": []}
    contracts = select_paper_contracts(chain, direction, budget_dollars, max_contracts)
    return {
        "status": "ok" if contracts else "no viable 0DTE same-day contract within budget",
        "expiry": chain["expiry"],
        "option_type": "call" if direction == "bullish" else "put",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contracts": contracts,
        "note": "PAPER/ANALYSIS ONLY — no order is placed; premiums are indicative quotes.",
    }
