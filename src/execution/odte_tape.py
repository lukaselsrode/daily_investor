"""0DTE fast-lane tape engine — 5-minute bars + live quotes → the market snapshot contract.

Feeds the SAME market-snapshot shape the whole decision chain already consumes (score_day flat
keys AND per-symbol blocks for candidate-watch/_symbol_block readers), built deterministically
from broker data with no LLM anywhere. The 2026-08-04 gap_pct drop — a controller losing an
overnight constant mid-session and silently re-tiering the day — is structurally impossible
here: gap_pct is computed once per ET session and carried in the engine state.

Input shapes are pinned to the LIVE server responses (probed 2026-08-04):
  * get_equity_historicals → data.results[].{symbol, bars[]}, bar =
    {begins_at RFC3339Z, open_price/close_price/high_price/low_price str, volume int,
     session "reg"} — bounds defaults to the regular session
  * get_equity_quotes → data.results[].quote.{symbol, last_trade_price, previous_close,
    bid_price, ask_price}

Math contracts:
  * session VWAP = Σ((H+L+C)/3 · V) / ΣV over today's regular-session bars
  * ORB = high/low of the FIRST 6 five-minute bars (the archived prior art's bars[:15] was a
    15-minute-bar constant — wrong here), FROZEN at 10:00 ET; before the freeze every ORB field
    is OMITTED, never placeholdered — partial tape reads neutral downstream
  * gap_pct = SPY first-bar open vs the quote lane's previous_close, ×100, session-constant
"""
from __future__ import annotations

from datetime import datetime, timezone
from datetime import time as dtime
from typing import Any
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

TAPE_SYMBOLS = ("SPY", "QQQ", "IWM", "VIXY")   # one historicals call covers all four (cap: 10)
ORB_BAR_COUNT = 6                               # first 30 minutes of 5-minute bars
ORB_FREEZE_ET = dtime(10, 0)
SESSION_OPEN_ET = dtime(9, 30)
SESSION_CLOSE_ET = dtime(16, 0)
BAR_INTERVAL = "5minute"
BAR_REFRESH_SECONDS = 300                       # re-pull bars at 5-minute boundaries


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _parse_ts(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def session_open_utc(now: datetime) -> datetime:
    """Today's 09:30 ET as a UTC datetime (the historicals start_time)."""
    et = now.astimezone(_ET)
    return datetime.combine(et.date(), SESSION_OPEN_ET, tzinfo=_ET).astimezone(timezone.utc)


def minutes_to_close(now: datetime) -> float:
    et = now.astimezone(_ET)
    close = datetime.combine(et.date(), SESSION_CLOSE_ET, tzinfo=_ET)
    return max(0.0, round((close - et).total_seconds() / 60.0, 1))


def parse_historicals(payload: Any) -> dict[str, list[dict]]:
    """data.results[].{symbol, bars[]} → {symbol: [normalized bar dicts]}, today-agnostic."""
    results = []
    if isinstance(payload, dict):
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        raw = data.get("results")
        if isinstance(raw, list):
            results = raw
    out: dict[str, list[dict]] = {}
    for res in results:
        if not isinstance(res, dict):
            continue
        symbol = str(res.get("symbol") or "").upper()
        bars = []
        for bar in res.get("bars") or []:
            ts = _parse_ts(bar.get("begins_at"))
            o, h = _num(bar.get("open_price")), _num(bar.get("high_price"))
            low, c = _num(bar.get("low_price")), _num(bar.get("close_price"))
            v = _num(bar.get("volume"))
            if ts is None or None in (o, h, low, c):
                continue
            bars.append({"ts": ts, "open": o, "high": h, "low": low, "close": c,
                         "volume": v or 0.0, "session": str(bar.get("session") or "")})
        if symbol:
            out[symbol] = bars
    return out


def parse_quotes(rows: list[Any]) -> dict[str, dict]:
    """get_equity_quotes rows (each nests .quote) → {symbol: {last, previous_close, bid, ask}}."""
    out: dict[str, dict] = {}
    for row in rows or []:
        quote = row.get("quote") if isinstance(row, dict) else None
        if not isinstance(quote, dict):
            quote = row if isinstance(row, dict) else {}
        symbol = str(quote.get("symbol") or "").upper()
        if not symbol:
            continue
        out[symbol] = {
            "last": _num(quote.get("last_trade_price") or quote.get("last")),
            "previous_close": _num(quote.get("previous_close")
                                   or quote.get("adjusted_previous_close")),
            "bid": _num(quote.get("bid_price")),
            "ask": _num(quote.get("ask_price")),
        }
    return out


def _today_regular_bars(bars: list[dict], now: datetime) -> list[dict]:
    et_today = now.astimezone(_ET).date()
    return [b for b in bars
            if b["ts"].astimezone(_ET).date() == et_today
            and (b["session"] in ("reg", "") or b["session"] is None)
            and b["ts"] <= now]


def _vwap(bars: list[dict]) -> float | None:
    num = sum(((b["high"] + b["low"] + b["close"]) / 3.0) * b["volume"] for b in bars)
    den = sum(b["volume"] for b in bars)
    return (num / den) if den > 0 else None


def _orb(bars: list[dict], now: datetime) -> tuple[float, float] | None:
    """The frozen opening range, or None before the 10:00 ET freeze / with a short tape."""
    if now.astimezone(_ET).time() < ORB_FREEZE_ET or len(bars) < ORB_BAR_COUNT:
        return None
    window = bars[:ORB_BAR_COUNT]
    return (max(b["high"] for b in window), min(b["low"] for b in window))


def compute_tape(bars_by_symbol: dict[str, list[dict]], quotes: dict[str, dict],
                 now: datetime, state: dict | None = None) -> tuple[dict, dict]:
    """PURE: bars + quotes → (market_snapshot, new_state). The snapshot carries BOTH the flat
    score_day keys and the per-symbol blocks, plus generated_at/as_of for the TTL clock."""
    state = dict(state or {})
    et_today = now.astimezone(_ET).date().isoformat()
    if state.get("session_date") != et_today:
        state = {"session_date": et_today}

    snapshot: dict[str, Any] = {
        "generated_at": now.isoformat(timespec="seconds"),
        "as_of": now.isoformat(timespec="seconds"),
        "source": "odte_tape",
        "minutes_to_close": minutes_to_close(now),
    }

    # gap_pct: session constant — computed ONCE from SPY's first bar vs the quote lane's
    # previous_close, then carried in state for the rest of the session.
    if _num(state.get("gap_pct")) is None:
        spy_bars = _today_regular_bars(bars_by_symbol.get("SPY") or [], now)
        prev_close = (quotes.get("SPY") or {}).get("previous_close")
        if spy_bars and prev_close:
            state["gap_pct"] = round((spy_bars[0]["open"] / prev_close - 1.0) * 100.0, 4)
    if _num(state.get("gap_pct")) is not None:
        snapshot["gap_pct"] = state["gap_pct"]

    for symbol in TAPE_SYMBOLS:
        bars = _today_regular_bars(bars_by_symbol.get(symbol) or [], now)
        quote = quotes.get(symbol) or {}
        last = quote.get("last")
        if last is None and bars:
            last = bars[-1]["close"]
        block: dict[str, Any] = {}
        if last is not None:
            block["last"] = last
        prev_close = quote.get("previous_close")
        if last is not None and prev_close:
            block["change_pct"] = round((last / prev_close - 1.0) * 100.0, 4)

        vwap = _vwap(bars)
        if vwap is not None and last is not None:
            block["vwap"] = round(vwap, 4)
            block["above_vwap"] = bool(last > vwap)
        orb = _orb(bars, now)
        if orb is not None and last is not None:
            orb_high, orb_low = orb
            block["orb_high"], block["orb_low"] = round(orb_high, 4), round(orb_low, 4)
            block["orb_state"] = ("above" if last > orb_high
                                  else "below" if last < orb_low else "inside")

        if block:
            snapshot[symbol] = block
        key = symbol.lower()
        # Flat mirrors for score_day — emitted ONLY when computed (omission stays neutral).
        if "above_vwap" in block:
            snapshot[f"{key}_above_vwap"] = block["above_vwap"]
        if "orb_state" in block:
            snapshot[f"{key}_orb_state"] = block["orb_state"]
        if symbol == "VIXY" and "change_pct" in block:
            snapshot["vixy_change_pct"] = block["change_pct"]

    return snapshot, state


async def build_tape(client: Any, state: dict | None, now: datetime) -> tuple[dict, dict]:
    """One tape tick against the broker: quotes every call, bars refreshed only when the 5-minute
    boundary advances (or the state is empty/new-session). Returns (market_snapshot, new_state);
    the caller owns persisting state between ticks."""
    state = dict(state or {})
    et_today = now.astimezone(_ET).date().isoformat()
    boundary = int(now.timestamp() // BAR_REFRESH_SECONDS)
    if (state.get("session_date") != et_today or state.get("bars_boundary") != boundary
            or not state.get("bars")):
        start = session_open_utc(now).isoformat(timespec="seconds").replace("+00:00", "Z")
        payload = await client.equity_historicals(list(TAPE_SYMBOLS), start_time=start,
                                                  interval=BAR_INTERVAL)
        state["bars"] = parse_historicals(payload)
        state["bars_boundary"] = boundary
    quotes = parse_quotes(await client.equity_quotes(list(TAPE_SYMBOLS)))
    bars_state = state.pop("bars", {})
    boundary_state = state.pop("bars_boundary", None)
    snapshot, state = compute_tape(bars_state, quotes, now, state)
    state["bars"], state["bars_boundary"] = bars_state, boundary_state
    return snapshot, state
