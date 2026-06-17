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


def fetch_spy_trend(ticker: str = "SPY", allow_fetch: bool = True) -> dict:
    """PAPER/ANALYSIS-only intraday price context for `ticker` (SPY by default), via yfinance.

    Returns a dict the scorecard consumes: ``last`` vs ``prev_close`` and intraday ``open`` plus a
    volume-weighted ``vwap`` (Σ typical-price·volume / Σ volume over the current session's 5-minute
    bars). **Fails closed** — any error / closed market / no network / yfinance missing yields
    ``{"ok": False, "status": ...}`` so the scorecard degrades to OBSERVE. Places NO orders."""
    if not allow_fetch:
        return {"ok": False, "status": "skipped: allow_fetch=False"}
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2d", interval="5m")
        if hist is None or getattr(hist, "empty", True):
            return {"ok": False, "status": "no intraday history (closed market / no network)"}
        day_of = [d.date() if hasattr(d, "date") else d for d in hist.index]
        dates = sorted(set(day_of))
        today = dates[-1]
        today_bars = hist[[d == today for d in day_of]]
        if getattr(today_bars, "empty", True):
            return {"ok": False, "status": "no same-day intraday bars"}
        prev_bars = hist[[d == dates[-2] for d in day_of]] if len(dates) >= 2 else None
        last = _f(today_bars["Close"].iloc[-1])
        open_ = _f(today_bars["Open"].iloc[0])
        prev_close = (_f(prev_bars["Close"].iloc[-1])
                      if prev_bars is not None and not prev_bars.empty else None)
        tp = (today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3.0
        vol = today_bars["Volume"]
        vsum = _f(vol.sum())
        vwap = _f((tp * vol).sum() / vsum) if vsum > 0 else None
        return {
            "ok": last > 0,
            "status": "ok" if last > 0 else "no usable last price",
            "last": last or None,
            "prev_close": prev_close if prev_close and prev_close > 0 else None,
            "open": open_ if open_ > 0 else None,
            "vwap": vwap if vwap and vwap > 0 else None,
            "pct_vs_prev_close": (last / prev_close - 1.0) if prev_close and prev_close > 0 else None,
            "pct_vs_open": (last / open_ - 1.0) if open_ > 0 else None,
            "above_vwap": (last > vwap) if (vwap and vwap > 0 and last > 0) else None,
        }
    except Exception as exc:
        logger.warning("SPY trend fetch failed for %s: %s", ticker, exc)
        return {"ok": False, "status": f"error: {exc}"}


def select_paper_contracts(chain: dict, direction: str, budget_dollars: float,
                           max_contracts: int = 3) -> list[dict]:
    """Pick liquid contracts matching direction (bullish->calls, bearish->puts).

    Budget-fitting rows (premium*100 <= cap) are returned first, sorted by tight spread then
    liquidity. If no liquid same-day contract fits the cap, return the single cheapest liquid
    contract above budget as PAPER-only context so the report can explain what was just out of
    reach instead of only saying "no viable contract".
    """
    df = chain.get("calls" if direction == "bullish" else "puts")
    if df is None or getattr(df, "empty", True):
        return []
    cap = float(budget_dollars) if float(budget_dollars) > 0 else 50.0
    has_vol = "volume" in df.columns
    has_oi = "openInterest" in df.columns
    budget_rows: list[dict] = []
    above_budget_rows: list[dict] = []
    for _, r in df.iterrows():
        bid, ask, last = _f(r.get("bid")), _f(r.get("ask")), _f(r.get("lastPrice"))
        premium = ask if ask > 0 else last
        cost = premium * 100.0
        if not (premium > 0 and cost > 0):
            continue
        if not (bid > 0 and ask > 0):
            continue
        vol = _f(r.get("volume")) if has_vol else 0.0
        oi = _f(r.get("openInterest")) if has_oi else 0.0
        if (has_vol or has_oi) and (vol + oi) <= 0:
            continue  # require SOME liquidity when the data is available
        mid = (ask + bid) / 2.0
        row = {
            "option_type": "call" if direction == "bullish" else "put",
            "strike": _f(r.get("strike")),
            "bid": round(bid, 4), "ask": round(ask, 4), "last": round(last, 4),
            "premium_cost_estimate": round(cost, 2),
            "above_budget": cost > cap,
            "spread_pct": round((ask - bid) / mid, 4) if mid > 0 else None,
            "volume": int(vol), "open_interest": int(oi),
        }
        if cost <= cap:
            budget_rows.append(row)
        else:
            above_budget_rows.append(row)
    budget_rows.sort(key=lambda c: (c["spread_pct"] if c["spread_pct"] is not None else 9.9,
                                    -(c["volume"] + c["open_interest"])))
    if budget_rows:
        return budget_rows[:max_contracts]

    above_budget_rows.sort(key=lambda c: (c["premium_cost_estimate"],
                                          c["spread_pct"] if c["spread_pct"] is not None else 9.9,
                                          -(c["volume"] + c["open_interest"])))
    return above_budget_rows[:1]


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
    cap = float(budget_dollars) if float(budget_dollars) > 0 else 50.0
    has_budget_fit = any(not c.get("above_budget") for c in contracts)
    return {
        "status": (
            "ok" if has_budget_fit else
            f"no viable 0DTE same-day contract within ${cap:.0f} budget; "
            "showing cheapest above-budget contract" if contracts else
            f"no viable 0DTE same-day contract within ${cap:.0f} budget"
        ),
        "expiry": chain["expiry"],
        "option_type": "call" if direction == "bullish" else "put",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contracts": contracts,
        "note": "PAPER/ANALYSIS ONLY — no order is placed; premiums are indicative quotes.",
    }
