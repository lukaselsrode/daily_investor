"""
data/fmp_client.py — Quota-aware, disk-cached Financial Modeling Prep client.

Purpose: survivorship-bias-free price history. FMP's `/stable` API serves daily prices
for DELISTED tickers (acquisitions, some bankruptcies) that yfinance drops entirely —
the missing 34.8% of the universe. The catch is the plan cap (default 250 calls/day),
so this client is built around an aggressive persistent cache:

  - every successful price pull is written to <cache>/prices/<SYMBOL>.parquet (default
    cache = data/fmp_cache_adj, split+dividend-adjusted; override with FMP_CACHE_DIR)
  - 402-premium / empty / error responses are NEGATIVE-cached in meta/<SYMBOL>.json so a
    dead symbol is never retried (a wasted call) on a later run
  - a per-day call counter (meta/_quota.json) hard-stops live fetches before the cap, so
    a large backfill simply resumes the next day where it left off

Endpoints (only the /stable ones validated on the current key):
  /stable/historical-price-eod/full?symbol=&from=&to=   daily EOD bars
  /stable/delisted-companies?page=N                     delisted roster (+ delist dates)

Read paths never hit the network — only `eod_prices(..., allow_fetch=True)` can, and only
within quota. Safe to call from backtests: cache-only by default.
"""
from __future__ import annotations

import datetime
import json
import os
import time

import pandas as pd
import requests

from core.logging import get_logger

logger = get_logger(__name__)

_BASE = "https://financialmodelingprep.com/stable"
# Canonical store is the split+dividend-ADJUSTED cache. (The old unadjusted data/fmp_cache from the
# first backfill is stale — superseded by this and safe to delete.)
_CACHE_DIR = os.environ.get("FMP_CACHE_DIR", "data/fmp_cache_adj")
_PRICE_DIR = os.path.join(_CACHE_DIR, "prices")
_STMT_DIR = os.path.join(_CACHE_DIR, "statements")
_META_DIR = os.path.join(_CACHE_DIR, "meta")
_QUOTA_PATH = os.path.join(_META_DIR, "_quota.json")

# FMP paid plans are rate-limited rather than lifetime/daily-call limited. Set FMP_DAILY_CAP=0
# (the default) for rate-limit-only behavior; set a positive value such as 240 if you want a
# self-imposed daily safety budget for a free/conservative key.
_DAILY_CAP = int(os.environ.get("FMP_DAILY_CAP", "0"))
_THROTTLE_S = float(os.environ.get("FMP_THROTTLE_S", "0.4"))  # polite spacing between live calls


class FMPQuotaExceeded(RuntimeError):
    """Raised when a live fetch is needed but today's call budget is spent."""


def _ensure_dirs() -> None:
    os.makedirs(_PRICE_DIR, exist_ok=True)
    os.makedirs(_META_DIR, exist_ok=True)


def _today() -> str:
    # Date passed explicitly nowhere reachable here; use UTC date for the rolling counter.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")


def _load_quota() -> dict:
    try:
        with open(_QUOTA_PATH) as fh:
            q = json.load(fh)
    except (OSError, ValueError):
        q = {}
    if q.get("date") != _today():
        q = {"date": _today(), "count": 0}
    return q


def _save_quota(q: dict) -> None:
    _ensure_dirs()
    with open(_QUOTA_PATH, "w") as fh:
        json.dump(q, fh)


def calls_remaining() -> int | None:
    """How many self-budgeted live FMP calls remain today; None means rate-limit-only."""
    if _DAILY_CAP <= 0:
        return None
    return max(0, _DAILY_CAP - _load_quota().get("count", 0))


def _safe(symbol: str) -> str:
    """Filename-safe symbol — odd tickers carry '/', '.', ':' (preferreds, foreign, units)."""
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _meta_path(symbol: str) -> str:
    return os.path.join(_META_DIR, f"{_safe(symbol)}.json")


def _price_path(symbol: str) -> str:
    return os.path.join(_PRICE_DIR, f"{_safe(symbol)}.parquet")


def _read_meta(symbol: str) -> dict | None:
    try:
        with open(_meta_path(symbol)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_meta(symbol: str, status: str, note: str = "",
                req_start: str | None = None, req_end: str | None = None) -> None:
    _ensure_dirs()
    rec = {"status": status, "note": note, "fetched": _today()}
    if req_start is not None:
        rec["req_start"] = req_start
    if req_end is not None:
        rec["req_end"] = req_end
    with open(_meta_path(symbol), "w") as fh:
        json.dump(rec, fh)


def _api_key() -> str:
    key = os.getenv("FMP_KEY")
    if not key:
        raise RuntimeError("FMP_KEY not set in environment (.env)")
    return key


class FMPNetworkError(RuntimeError):
    """Transient network failure after retries — caller should skip + retry later (NOT cache)."""


def _get(url: str) -> tuple[int, object]:
    """One live GET, counted against the daily quota. Returns (status_code, parsed|text).

    Retries transient network failures with backoff; only a successful HTTP response counts
    against the quota (timed-out attempts never reached FMP). Raises FMPNetworkError if all
    attempts fail so a long backfill can skip the symbol and retry it on the next run.
    """
    q = _load_quota()
    if _DAILY_CAP > 0 and q.get("count", 0) >= _DAILY_CAP:
        raise FMPQuotaExceeded(f"FMP daily cap {_DAILY_CAP} reached ({q['count']} used)")
    time.sleep(_THROTTLE_S)
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=25)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))  # backoff on timeout/connection errors
            continue
        q["count"] = q.get("count", 0) + 1
        _save_quota(q)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, resp.text
    raise FMPNetworkError(f"{url.split('?')[0]}: {last_exc}")


def eod_prices(symbol: str, start: str, end: str, allow_fetch: bool = False) -> pd.DataFrame | None:
    """Daily EOD prices for `symbol` over [start, end], cache-first.

    Returns a DataFrame indexed by date (columns: open/high/low/close/volume) or None when
    the symbol is unavailable (premium-gated, empty, or not yet cached and allow_fetch=False).
    Never spends a call for a symbol already known-bad (negative cache).
    """
    cached = _price_path(symbol)
    meta = _read_meta(symbol)
    if os.path.exists(cached):
        # Range-aware: the cache is keyed by symbol, so a prior NARROW fetch must NOT satisfy a
        # wider request (that bug truncated pre-probed names like NVDA to a few bars). Trust the
        # cache only when the previously-fetched window covers [start, end]; otherwise re-fetch the
        # union range. Old meta without req_start/req_end is treated as not-covering -> re-fetch.
        fs, fe = (meta or {}).get("req_start"), (meta or {}).get("req_end")
        if fs and fe and fs <= start and fe >= end:
            df = pd.read_parquet(cached)
            return df.loc[(df.index >= start) & (df.index <= end)]
        if not allow_fetch:
            df = pd.read_parquet(cached)  # best effort with what we have
            return df.loc[(df.index >= start) & (df.index <= end)]
        # widen the fetch to the union so we never shrink coverage
        if fs:
            start = min(start, fs)
        if fe:
            end = max(end, fe)

    if meta and meta.get("status") in ("premium", "empty", "error"):
        return None  # known-bad — do not waste a call

    if not allow_fetch:
        return None

    # SPLIT+DIVIDEND-ADJUSTED prices — mandatory for factor backtests (unadjusted prices turn a
    # 10:1 split into a fake -90% crash, which corrupts momentum on exactly the winners).
    url = f"{_BASE}/historical-price-eod/dividend-adjusted?symbol={symbol}&from={start}&to={end}&apikey={_api_key()}"
    status, payload = _get(url)
    if status == 402:
        _write_meta(symbol, "premium", "402 paywalled on current plan")
        return None
    if status != 200 or not isinstance(payload, list) or not payload:
        _write_meta(symbol, "empty", f"status={status}")
        return None

    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"})
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]
    _ensure_dirs()
    df.to_parquet(cached)
    _write_meta(symbol, "ok", f"{len(df)} bars", req_start=start, req_end=end)
    return df.loc[(df.index >= start) & (df.index <= end)]


_PERIOD_ENDPOINTS = {"income-statement", "balance-sheet-statement", "cash-flow-statement"}


def statement(symbol: str, kind: str, period: str = "quarter", limit: int = 44,
              allow_fetch: bool = False) -> pd.DataFrame | None:
    """Cached raw FMP statement for point-in-time fundamentals reconstruction.

    kind ∈ {income-statement, balance-sheet-statement, cash-flow-statement, earnings,
            shares-float, dividends}. Statements carry `filingDate`, so they can be used
            causally (only filings public as-of a backtest date). Cached as JSON per symbol;
            402/empty negative-cached so a bad symbol never re-spends a call.

    Returns a DataFrame (rows = reporting periods, newest first) or None when unavailable.
    """
    sdir = os.path.join(_STMT_DIR, kind)
    path = os.path.join(sdir, f"{_safe(symbol)}.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return pd.DataFrame(json.load(fh))
        except (OSError, ValueError):
            pass

    mkey = f"{kind}:{symbol}"
    meta = _read_meta(mkey)
    if meta and meta.get("status") in ("premium", "empty", "error"):
        return None
    if not allow_fetch:
        return None

    if kind in _PERIOD_ENDPOINTS:
        url = f"{_BASE}/{kind}?symbol={symbol}&period={period}&limit={limit}&apikey={_api_key()}"
    else:
        url = f"{_BASE}/{kind}?symbol={symbol}&limit={limit}&apikey={_api_key()}"
    status, payload = _get(url)
    if status == 402:
        _write_meta(mkey, "premium", "402 paywalled on current plan")
        return None
    if status != 200 or not isinstance(payload, list) or not payload:
        _write_meta(mkey, "empty", f"status={status}")
        return None

    os.makedirs(sdir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh)
    _write_meta(mkey, "ok", f"{len(payload)} periods")
    return pd.DataFrame(payload)


def delisted_companies(max_pages: int = 50, allow_fetch: bool = False) -> pd.DataFrame:
    """The delisted-company roster (symbol, exchange, ipoDate, delistedDate), cache-first.

    Cached as a single parquet so the whole roster costs <= max_pages calls ONCE, then free.
    """
    roster = os.path.join(_CACHE_DIR, "delisted_roster.parquet")
    if os.path.exists(roster):
        return pd.read_parquet(roster)
    if not allow_fetch:
        return pd.DataFrame()
    rows: list[dict] = []
    for page in range(max_pages):
        status, payload = _get(f"{_BASE}/delisted-companies?page={page}&apikey={_api_key()}")
        if status != 200 or not isinstance(payload, list) or not payload:
            break
        rows.extend(payload)
    df = pd.DataFrame(rows)
    if not df.empty:
        _ensure_dirs()
        df.to_parquet(roster)
    return df
