"""Build the 0DTE market snapshot from raw session bars — VWAP side and opening-range side.

PURE/OFFLINE: no broker, no network, no LLM, no file I/O. The caller (Hermes) fetches bars and
passes them in.

WHY THIS MODULE EXISTS (2026-08-10). `above_vwap` and `orb_state` are the two inputs that drive
breadth, tier and entry — and until now **no tool produced them**. The controller re-authored an
ad-hoc Python script computing both, from scratch, on every tick:

    def orb(symbol):
        first_30 = bars[symbol][:6]        # assumes exactly 6 bars == 30 minutes
        state = "above" if last > high else "below" if last < low else "inside"

That is `odte_breadth`'s own lesson one level upstream: the shared *reader* was unified on
2026-08-07, but the *writer* stayed improvised. Three consequences, all observed in the journal:

* **Vocabulary drift.** 46 of 1508 recorded `orb_state` values were outside the documented
  `above|below|inside` set — `above_orb`, `above_5m_range`, `above_first_1m_range`,
  `inside_30m_range`, `above_high`. 23 of them carried a *directional* read that
  `odte_breadth.alignment` scores as "no ORB data at all", silently costing a full confirmer each
  (`above` scores 2, `above_orb` scores 1 — identical to absent).
* **A window that moves.** `bars[:6]` is only 30 minutes if the bars are 5-minute bars, and before
  10:00 ET there are fewer than 6 bars — so the "opening range" included the current price and the
  state collapsed to `inside`, suppressing every early-session breakout.
* **~60 seconds inside a 30-second budget**, re-derived every tick, which is what turned the
  2026-08-10 14:07:50 CONFIRM_ENTRY into a 14:10:02 refusal after VIXY firmed.

The opening-range window is defined HERE, once, because the snapshot spec pins the vocabulary
(`docs/hermes/v2/fast_path_snapshots.md`) but never pinned the window — which is what left it to
improvisation in the first place.

Honesty rule, matching the spec's "OMIT fields you cannot compute; never placeholder": while the
opening range is still forming, `orb_state` is omitted rather than reported as `inside`. Both read
identically to `odte_breadth.alignment` (an absent ORB and an `inside` ORB each add 0), so this
changes no score — it stops the snapshot from asserting a range that does not exist yet.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SESSION_OPEN_ET = time(9, 30)
SESSION_CLOSE_ET = time(16, 0)

# THE opening range. Improvised per-tick before 2026-08-10 (1m / 5m / 30m all appear in the
# journal); pinned here so every snapshot means the same thing by construction.
OPENING_RANGE_MINUTES = 30

# The only values odte_breadth scores. Anything else reads as "no ORB data".
ORB_ABOVE, ORB_BELOW, ORB_INSIDE = "above", "below", "inside"
CANONICAL_ORB_STATES = (ORB_ABOVE, ORB_BELOW, ORB_INSIDE)

_TS_KEYS = ("begins_at", "timestamp", "time", "t", "date", "datetime")
_OPEN_KEYS = ("open_price", "open", "o")
_HIGH_KEYS = ("high_price", "high", "h")
_LOW_KEYS = ("low_price", "low", "l")
_CLOSE_KEYS = ("close_price", "close", "c")
_VOL_KEYS = ("volume", "vol", "v")


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN guard


def _first(block: dict, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in block:
            return block[k]
    return None


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    # Hermes/Go writes nanosecond precision; fromisoformat takes at most 6 fractional digits.
    import re
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def normalize_bar(bar: Any) -> dict | None:
    """One bar -> {ts, open, high, low, close, volume}, or None if unusable.

    Accepts the Robinhood historicals shape (`begins_at`/`*_price`, values as strings), generic
    OHLCV dicts, and positional sequences. A bar without high/low/close is dropped rather than
    guessed at.
    """
    if isinstance(bar, dict):
        ts = _parse_ts(_first(bar, _TS_KEYS))
        o = _num(_first(bar, _OPEN_KEYS))
        h = _num(_first(bar, _HIGH_KEYS))
        low = _num(_first(bar, _LOW_KEYS))
        c = _num(_first(bar, _CLOSE_KEYS))
        v = _num(_first(bar, _VOL_KEYS))
    elif isinstance(bar, (list, tuple)) and len(bar) >= 5:
        # positional (ts, o, h, l, c, v) or (o, h, l, c, v)
        if len(bar) >= 6:
            ts, o, h, low, c, v = (_parse_ts(bar[0]), _num(bar[1]), _num(bar[2]),
                                   _num(bar[3]), _num(bar[4]), _num(bar[5]))
        else:
            ts, (o, h, low, c, v) = None, (_num(bar[0]), _num(bar[1]), _num(bar[2]),
                                           _num(bar[3]), _num(bar[4]))
    else:
        return None
    if h is None or low is None or c is None:
        return None
    return {"ts": ts, "open": o, "high": h, "low": low, "close": c,
            "volume": v if v is not None else 0.0}


def session_bars(bars: Iterable[Any], *, session_date=None) -> list[dict]:
    """Normalized bars belonging to the RTH session, oldest first.

    Bars carrying timestamps are filtered to >= 09:30 ET on the session date, so a caller that
    over-fetches (yesterday's tail, or premarket) still gets a true session VWAP rather than a
    rolling window of "whatever I asked for". Undated bars are kept in the given order — the
    caller is then asserting they are the session.
    """
    rows = [b for b in (normalize_bar(x) for x in (bars or [])) if b]
    dated = [b for b in rows if b["ts"] is not None]
    if not dated:
        return rows
    if session_date is None:
        session_date = max(b["ts"] for b in dated).astimezone(ET).date()
    open_dt = datetime.combine(session_date, SESSION_OPEN_ET, tzinfo=ET)
    close_dt = datetime.combine(session_date, SESSION_CLOSE_ET, tzinfo=ET)
    kept = [b for b in dated if open_dt <= b["ts"].astimezone(ET) <= close_dt]
    kept.sort(key=lambda b: b["ts"])
    return kept


def session_vwap(bars: Iterable[Any], *, session_date=None) -> float | None:
    """Volume-weighted average of the typical price ((H+L+C)/3) over the session so far."""
    rows = session_bars(bars, session_date=session_date)
    num = den = 0.0
    for b in rows:
        vol = b["volume"] or 0.0
        if vol <= 0:
            continue
        num += ((b["high"] + b["low"] + b["close"]) / 3.0) * vol
        den += vol
    if den <= 0:
        return None
    return num / den


def _bar_interval(rows: list[dict]) -> timedelta:
    """Median spacing between consecutive dated bars; 5 minutes when undeterminable."""
    gaps = sorted((rows[i]["ts"] - rows[i - 1]["ts"]).total_seconds()
                  for i in range(1, len(rows)) if rows[i]["ts"] and rows[i - 1]["ts"])
    if not gaps:
        return timedelta(minutes=5)
    mid = gaps[len(gaps) // 2]
    return timedelta(seconds=mid) if mid > 0 else timedelta(minutes=5)


def opening_range(bars: Iterable[Any], *, session_date=None,
                  minutes: int = OPENING_RANGE_MINUTES, now: datetime | None = None) -> dict:
    """High/low of the first `minutes` of RTH, plus whether that window has fully elapsed.

    Selected by TIMESTAMP, never by bar count — `bars[:6]` is only 30 minutes if the bars happen
    to be 5-minute bars, and is silently wrong otherwise. `complete` is False until the window has
    closed, which is what stops an unformed range from being reported as a real one.

    `complete` needs BOTH the clock and the bars (2026-08-11). It used to ask only whether the
    newest bar was at/after the window end — but brokers return the last COMPLETED interval, so at
    10:02 the newest 5-minute bar still starts 09:55 and a fully-observed 09:30-10:00 range
    reported itself incomplete for another whole bar interval. Observed live: `orb_high 774.54`
    and `orb_low 772.58` both computed and sitting in the snapshot while `orb_state` stayed None,
    so the tape lane could not manufacture for ~5 minutes after the range it needs had closed.

    Keying off the clock ALONE would be the opposite error: a caller feeding stale bars would get
    `complete` on an under-observed range, and an `orb_high` that is too low manufactures phantom
    breakouts. So the window must also be covered — the last bar in it, plus one interval, must
    reach the end.
    """
    rows = session_bars(bars, session_date=session_date)
    if not rows:
        return {"high": None, "low": None, "complete": False, "bars": 0}
    dated = [b for b in rows if b["ts"] is not None]
    if not dated:
        # undated bars: the caller asserted these are the session, so we cannot time-slice.
        return {"high": None, "low": None, "complete": False, "bars": len(rows)}
    day = (session_date or max(b["ts"] for b in dated).astimezone(ET).date())
    open_dt = datetime.combine(day, SESSION_OPEN_ET, tzinfo=ET)
    end_dt = open_dt + timedelta(minutes=minutes)
    window = [b for b in dated if b["ts"].astimezone(ET) < end_dt]
    if not window:
        return {"high": None, "low": None, "complete": False, "bars": 0}
    interval = _bar_interval(dated)
    covered = (max(b["ts"] for b in window) + interval) >= end_dt
    elapsed = (now or datetime.now(timezone.utc)).astimezone(ET) >= end_dt
    return {"high": max(b["high"] for b in window),
            "low": min(b["low"] for b in window),
            # BOTH: the clock is past the window, and the bars actually reach its end
            "complete": bool(covered and elapsed),
            "bars": len(window)}


def orb_state(last: float | None, orb: dict) -> str | None:
    """Canonical `above|below|inside`, or None while the opening range is still forming.

    Returning None (and omitting the key upstream) is deliberate: an unformed range reported as
    `inside` is a false negative that suppresses early-session breakouts, and any spelling outside
    CANONICAL_ORB_STATES is scored by odte_breadth as no data at all.
    """
    if last is None or not orb.get("complete"):
        return None
    high, low = orb.get("high"), orb.get("low")
    if high is None or low is None:
        return None
    if last > high:
        return ORB_ABOVE
    if last < low:
        return ORB_BELOW
    return ORB_INSIDE


def minutes_to_close(now: datetime | None = None) -> float:
    now = (now or datetime.now(timezone.utc)).astimezone(ET)
    close_dt = datetime.combine(now.date(), SESSION_CLOSE_ET, tzinfo=ET)
    return round(max(0.0, (close_dt - now).total_seconds() / 60.0), 2)


def build_symbol_block(bars: Iterable[Any], last: float | None, *,
                       session_date=None, prev_close: float | None = None,
                       now: datetime | None = None) -> dict:
    """One symbol's tape block. Keys are OMITTED, never placeholdered, when uncomputable."""
    vwap = session_vwap(bars, session_date=session_date)
    orb = opening_range(bars, session_date=session_date, now=now)
    state = orb_state(last, orb)
    block: dict[str, Any] = {}
    if last is not None:
        block["last"] = last
    if vwap is not None:
        block["vwap"] = round(vwap, 6)
        if last is not None:
            block["above_vwap"] = bool(last > vwap)
    if state is not None:
        block["orb_state"] = state
    if orb.get("high") is not None:
        block["orb_high"] = round(orb["high"], 6)
        block["orb_low"] = round(orb["low"], 6)
        block["orb_complete"] = bool(orb.get("complete"))
    if prev_close and last is not None:
        block["change_pct"] = round((last / prev_close - 1.0) * 100.0, 4)
    return block


def build_market_snapshot(bars_by_symbol: dict, last_by_symbol: dict, *,
                          now: datetime | None = None, session_date=None,
                          prev_close_by_symbol: dict | None = None,
                          spot_symbol: str = "SPY", gap_pct: float | None = None,
                          extra: dict | None = None) -> dict:
    """The market.json the fast path consumes — BOTH flat keys and nested blocks.

    Emits the exact shapes in docs/hermes/v2/fast_path_snapshots.md: flat `{sym}_above_vwap` /
    `{sym}_orb_state` for the day scorer, nested `market["SPY"]` blocks for the tier computation.
    `odte_breadth.symbol_block` merges the two, but only agreement makes that safe — writing both
    from ONE computation is what guarantees they agree.
    """
    now = now or datetime.now(timezone.utc)
    prev_close_by_symbol = prev_close_by_symbol or {}
    out: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "minutes_to_close": minutes_to_close(now),
        "opening_range_minutes": OPENING_RANGE_MINUTES,
    }
    if gap_pct is not None:
        out["gap_pct"] = gap_pct

    for sym in (bars_by_symbol or {}):
        up = str(sym).upper()
        block = build_symbol_block(bars_by_symbol.get(sym),
                                   _num((last_by_symbol or {}).get(sym)),
                                   session_date=session_date, now=now,
                                   prev_close=_num(prev_close_by_symbol.get(sym)))
        if not block:
            continue
        out[up] = block
        low = up.lower()
        if "above_vwap" in block:
            out[f"{low}_above_vwap"] = block["above_vwap"]
        if "orb_state" in block:
            out[f"{low}_orb_state"] = block["orb_state"]
        if "change_pct" in block:
            out[f"{low}_change_pct"] = block["change_pct"]

    spot = _num((last_by_symbol or {}).get(spot_symbol))
    if spot is not None:
        out["spot"] = spot
    if extra:
        out.update(extra)
    return out


_INSTRUMENT_ID_KEYS = ("id", "instrument_id", "option_id")
_QUOTE_ID_KEYS = ("instrument_id", "id", "option_id")


def select_contract(instruments: Iterable[dict], quotes: Iterable[dict], *,
                    option_id: str | None = None, strike_price: Any = None,
                    option_type: str | None = None) -> tuple[dict, dict] | None:
    """Join one instrument to its quote, or None when no tradable match exists.

    The two payloads carry disjoint halves of a contract: instruments hold identity
    (strike/expiry/type/chain), quotes hold price and the SAME uuid under a different key
    (`instrument_id`). Joining them is what the controller has been doing by hand every tick.

    Fail-closed: an instrument that is not `active`+`tradable`, or has no matching quote, is never
    returned. A contract we cannot fully describe must not become an order.
    """
    q_by_id: dict[str, dict] = {}
    for q in quotes or []:
        if not isinstance(q, dict):
            continue
        inner = q.get("quote") if isinstance(q.get("quote"), dict) else q
        qid = str(_first(inner, _QUOTE_ID_KEYS) or "").strip()
        if qid:
            q_by_id[qid] = inner

    want_type = str(option_type or "").strip().lower() or None
    want_strike = _num(strike_price)
    for inst in instruments or []:
        if not isinstance(inst, dict):
            continue
        iid = str(_first(inst, _INSTRUMENT_ID_KEYS) or "").strip()
        if not iid:
            continue
        if option_id and iid != str(option_id).strip():
            continue
        if want_type and str(inst.get("type") or "").strip().lower() != want_type:
            continue
        if want_strike is not None and _num(inst.get("strike_price")) != want_strike:
            continue
        if str(inst.get("state") or "active").strip().lower() != "active":
            continue
        if str(inst.get("tradability") or "tradable").strip().lower() != "tradable":
            continue
        quote = q_by_id.get(iid)
        if quote:
            return inst, quote
    return None


def build_contract_snapshot(instrument: dict, quote: dict, *,
                            now: datetime | None = None) -> dict:
    """The contract.json `odte-convert` reads, from the two raw halves.

    ALWAYS stamps `generated_at`. Its absence is what produced `contract_quote_undated` on
    2026-08-10: the controller saved the raw MCP envelope, convert could not age the quote, and
    refused — correctly, since a stale quote misprices the debit.
    """
    now = now or datetime.now(timezone.utc)
    inst = instrument if isinstance(instrument, dict) else {}
    q = quote if isinstance(quote, dict) else {}
    out: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "updated_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "option_id": _first(inst, _INSTRUMENT_ID_KEYS),
        "chain_symbol": inst.get("chain_symbol"),
        "underlying": inst.get("chain_symbol"),
        "expiration_date": inst.get("expiration_date"),
        "state": inst.get("state"),
        "tradability": inst.get("tradability"),
    }
    otype = str(inst.get("type") or "").strip().lower() or None
    if otype:
        out["option_type"] = otype
        out["type"] = otype
    strike = _num(inst.get("strike_price"))
    if strike is not None:
        out["strike_price"] = strike
    for src, dst in (("bid_price", "bid_price"), ("ask_price", "ask_price"),
                     ("mark_price", "mark_price"), ("adjusted_mark_price", "adjusted_mark_price"),
                     ("break_even_price", "break_even_price"), ("delta", "delta"),
                     ("gamma", "gamma"), ("implied_volatility", "implied_volatility"),
                     ("open_interest", "open_interest"), ("volume", "volume"),
                     ("bid_size", "bid_size"), ("ask_size", "ask_size")):
        v = _num(q.get(src))
        if v is not None:
            out[dst] = v
    return {k: v for k, v in out.items() if v is not None}


def audit_orb_vocabulary(market: dict) -> list[dict]:
    """Non-canonical `*orb_state` values in a snapshot someone ELSE built.

    Diagnostic for the hand-built snapshots still in the journal: `odte_breadth` treats an
    unrecognized spelling as absent, so this is the difference between "the index had no opening
    range" and "the index broke out and we threw the reading away".
    """
    found: list[dict] = []

    def walk(node: Any, path: str = "") -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and "orb_state" in key.lower() and isinstance(value, str):
                if value.strip().lower() not in CANONICAL_ORB_STATES:
                    directional = value.strip().lower().startswith(("above", "below"))
                    found.append({"path": here, "value": value,
                                  "directional_signal_lost": directional})
            elif isinstance(value, dict):
                walk(value, here)

    walk(market)
    return found
