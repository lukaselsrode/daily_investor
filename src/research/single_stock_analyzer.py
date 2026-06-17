"""
research/single_stock_analyzer.py — Single-stock decision-support analyzer (READ-ONLY).

Refactored from the one-off .session_tmp/baba_single_stock_analyzer.py into a reusable,
testable module. Given a symbol (and an optional leveraged-ETF symbol) it returns a structured
``SingleStockAnalysis`` assembled from:

  * latest portfolio exposure from the most recent ``data/holdings_*.csv``
  * the repo's cached ``agg_data`` row (factor scores / value metric)
  * yfinance price/trend, fundamentals, news, and an options-surface summary
  * a social scan (Reddit OAuth/JSON/RSS + X official API) via ``data.social_sentiment``,
    with the existing transparent spam/dedupe filters — social items are treated as ordinary
    sourced evidence (provenance preserved), never as a separate pre-aggregated score
  * leveraged-ETF diagnostics (realized daily beta/correlation, and cumulative return vs a
    synthetic daily-reset 2x series) when a leverage symbol is supplied

Decision-support only — **not financial advice, places NO orders, imports NO broker/execution
code.** Read-only per the ``research/`` contract. Network is optional: every fetch fails closed
(returns a status, never raises) and yfinance is imported lazily so the module is usable (and
testable) without it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from data.social_sentiment import (
    _parse_x_ts,
    _quality_filter,
    extract_ticker_mentions,
    fetch_reddit_posts,
    fetch_x_mentions,
    score_social,
)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "DECISION-SUPPORT / ANALYSIS ONLY — not financial advice, not a recommendation, and places "
    "NO orders. Leveraged ETFs reset daily and can lose money over multi-day periods even if the "
    "underlying rises; single-name/single-country concentration risk is non-diversifiable."
)

# Default Reddit surfaces for a single-name scan (bounded). WSB plus broader investing subs.
_DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")
_FUNDAMENTAL_KEYS = (
    "marketCap", "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
    "profitMargins", "operatingMargins", "grossMargins", "revenueGrowth", "earningsGrowth",
    "freeCashflow", "totalCash", "totalDebt", "dividendYield", "beta", "recommendationMean",
    "recommendationKey", "targetMeanPrice", "targetMedianPrice", "numberOfAnalystOpinions",
)
_RETURN_WINDOWS = (("5d", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252))


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------

@dataclass
class PriceTrend:
    symbol: str
    price: float | None = None
    date: str | None = None
    low_52w: float | None = None
    high_52w: float | None = None
    from_52w_high: float | None = None
    from_52w_low: float | None = None
    sma20_gap: float | None = None
    sma50_gap: float | None = None
    sma200_gap: float | None = None
    vol20_ann: float | None = None
    returns: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class HoldingsExposure:
    total_equity: float = 0.0
    positions: dict = field(default_factory=dict)  # symbol -> {equity, percentage, quantity, current_price}
    status: str = ""


@dataclass
class SocialScan:
    statuses: dict = field(default_factory=dict)
    raw_docs: int = 0
    quality_docs: int = 0
    mentions: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)  # [{source, title, url, score, ts, age_hours}]


@dataclass
class LeverageDiagnostics:
    base_symbol: str
    leverage_symbol: str
    realized_daily_beta: float | None = None
    daily_corr: float | None = None
    periods: dict = field(default_factory=dict)  # label -> {base, lev, daily_2x_synth, tracking_gap}
    note: str | None = None


@dataclass
class SingleStockAnalysis:
    symbol: str
    leverage_symbol: str | None
    generated_at: str
    disclaimer: str
    exposure: HoldingsExposure
    cached_factors: dict
    price_trends: dict          # symbol -> PriceTrend
    fundamentals: dict
    news: list
    social: SocialScan | None
    leverage: LeverageDiagnostics | None
    options: dict
    statuses: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    """Coerce to a finite float or None (never raises)."""
    try:
        if v is None:
            return None
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return None


def _sma_gap(close, price: float, window: int) -> float | None:
    """price / SMA(window) − 1, or None when there isn't enough history."""
    if len(close) < window:
        return None
    ma = float(close.rolling(window).mean().iloc[-1])
    return price / ma - 1.0 if ma else None


def _load_yf():
    """Lazily import yfinance; return the module or None (fail-closed when missing)."""
    try:
        import yfinance as yf
        return yf
    except Exception as exc:  # pragma: no cover - depends on host install
        logger.warning("yfinance unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Repo data (read-only): holdings exposure + cached factor scores
# ---------------------------------------------------------------------------

def holdings_exposure(symbols: list[str]) -> HoldingsExposure:
    """Latest portfolio exposure from the most recent ``data/holdings_*.csv``. Read-only."""
    from data.cache import read_data_as_pd
    try:
        df = read_data_as_pd("holdings")
    except Exception as exc:
        return HoldingsExposure(status=f"holdings unavailable: {exc}")
    if df is None or getattr(df, "empty", True) or "symbol" not in df.columns:
        return HoldingsExposure(status="no holdings snapshot found")
    total = sum(_f(v) or 0.0 for v in df.get("equity", []))
    positions: dict = {}
    wanted = {s.upper() for s in symbols}
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "")).upper()
        if sym in wanted:
            positions[sym] = {
                "equity": _f(r.get("equity")),
                "percentage": _f(r.get("percentage")),
                "quantity": _f(r.get("quantity")),
                "current_price": _f(r.get("current_price")),
                "name": r.get("name"),
            }
    return HoldingsExposure(total_equity=total, positions=positions, status="ok")


def cached_factors(symbol: str) -> dict:
    """The repo's cached ``agg_data`` scoring row for ``symbol`` (factor scores, value metric)."""
    from data.cache import read_data_as_pd
    keep = ("symbol", "value_metric", "value_score", "quality_score", "income_score",
            "momentum_score", "pe_ratio", "pb_ratio", "dividend_yield", "return_1m",
            "position_52w", "buy_to_sell_ratio")
    try:
        df = read_data_as_pd("agg_data")
        if df is None or getattr(df, "empty", True) or "symbol" not in df.columns:
            return {}
        row = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {k: r.get(k) for k in keep if k in df.columns}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# yfinance: price/trend, fundamentals, news, options (all fail-closed)
# ---------------------------------------------------------------------------

def price_snapshot(symbols: list[str], *, allow_fetch: bool = True) -> tuple[dict, dict]:
    """Return ({symbol: PriceTrend}, {symbol: history_df}). Fails closed per symbol."""
    trends: dict = {}
    hist: dict = {}
    if not allow_fetch:
        return {s: PriceTrend(symbol=s, error="live fetch disabled") for s in symbols}, hist
    yf = _load_yf()
    if yf is None:
        return {s: PriceTrend(symbol=s, error="yfinance unavailable") for s in symbols}, hist
    for sym in symbols:
        try:
            h = yf.Ticker(sym).history(period="1y", interval="1d", auto_adjust=True)
        except Exception as exc:
            trends[sym] = PriceTrend(symbol=sym, error=f"history error: {exc}")
            continue
        if h is None or getattr(h, "empty", True) or "Close" not in getattr(h, "columns", []):
            trends[sym] = PriceTrend(symbol=sym, error="no history")
            continue
        hist[sym] = h
        close = h["Close"].dropna()
        if close.empty:
            trends[sym] = PriceTrend(symbol=sym, error="no closes")
            continue
        price = float(close.iloc[-1])
        high_52, low_52 = float(close.max()), float(close.min())
        rets = {lbl: (price / float(close.iloc[-n]) - 1.0) if len(close) > n else None
                for lbl, n in _RETURN_WINDOWS}
        vol20 = None
        if len(close) > 22:
            v = close.pct_change().rolling(20).std().iloc[-1]
            vol20 = float(v) * math.sqrt(252) if v == v else None  # v==v guards NaN
        trends[sym] = PriceTrend(
            symbol=sym, price=price, date=str(close.index[-1].date()),
            low_52w=low_52, high_52w=high_52,
            from_52w_high=(price / high_52 - 1.0) if high_52 else None,
            from_52w_low=(price / low_52 - 1.0) if low_52 else None,
            sma20_gap=_sma_gap(close, price, 20), sma50_gap=_sma_gap(close, price, 50),
            sma200_gap=_sma_gap(close, price, 200), vol20_ann=vol20, returns=rets,
        )
    return trends, hist


def fundamentals_snapshot(symbol: str, *, allow_fetch: bool = True) -> dict:
    """yfinance fundamental/analyst fields for ``symbol`` (subset). {} or {status} on failure."""
    if not allow_fetch:
        return {"status": "live fetch disabled"}
    yf = _load_yf()
    if yf is None:
        return {"status": "yfinance unavailable"}
    try:
        tk = yf.Ticker(symbol)
        info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", {})
        info = info or {}
        return {k: info.get(k) for k in _FUNDAMENTAL_KEYS}
    except Exception as exc:
        return {"status": f"fundamentals error: {exc}"}


def yf_news(symbol: str, limit: int = 8, *, allow_fetch: bool = True) -> list[dict]:
    """yfinance news items relevant to ``symbol`` (best-effort across schema versions)."""
    if not allow_fetch:
        return []
    yf = _load_yf()
    if yf is None:
        return []
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []
    rows: list[dict] = []
    sym_low = symbol.lower()
    for item in items[: max(limit * 3, limit)]:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        title = item.get("title") or content.get("title") or ""
        publisher = (item.get("publisher")
                     or (content.get("provider") or {}).get("displayName") or "yfinance")
        link = (item.get("link")
                or (content.get("canonicalUrl") or {}).get("url") or "")
        ts = item.get("providerPublishTime") or content.get("pubDate")
        if sym_low not in f"{title} {link}".lower():
            # keep relevance loose but require the ticker to appear somewhere
            if title and not link:
                pass  # untagged headline — keep
            else:
                continue
        rows.append({"title": title, "publisher": publisher, "link": link,
                     "api_source": "yfinance", "pub_date": str(ts)})
        if len(rows) >= limit:
            break
    return rows


def options_snapshot(symbol: str, *, allow_fetch: bool = True) -> dict:
    """Nearest-expiry options surface summary (top OI calls/puts). Not a trade recommendation."""
    if not allow_fetch:
        return {"status": "live fetch disabled"}
    yf = _load_yf()
    if yf is None:
        return {"status": "yfinance unavailable"}
    try:
        tk = yf.Ticker(symbol)
        expiries = list(tk.options or [])
    except Exception as exc:
        return {"status": f"options error: {exc}"}
    if not expiries:
        return {"status": "no listed expiries"}
    exp = expiries[0]
    try:
        chain = tk.option_chain(exp)
    except Exception as exc:
        return {"status": f"chain error: {exc}", "first_expiry": exp}

    def _top_oi(frame):
        cols = [c for c in ("strike", "lastPrice", "bid", "ask", "volume",
                            "openInterest", "impliedVolatility") if c in getattr(frame, "columns", [])]
        if not cols:
            return []
        return frame[cols].sort_values(
            [c for c in ("openInterest", "volume") if c in cols], ascending=False
        ).head(5).to_dict("records")

    return {"first_expiry": exp,
            "calls_top_oi": _top_oi(chain.calls),
            "puts_top_oi": _top_oi(chain.puts)}


# ---------------------------------------------------------------------------
# Social scan (reuses data.social_sentiment; social treated as sourced evidence)
# ---------------------------------------------------------------------------

def social_scan(symbol: str, leverage_symbol: str | None = None, *, allow_fetch: bool = True,
                subreddits: tuple[str, ...] = _DEFAULT_SUBREDDITS, reddit_limit: int = 50,
                x_limit: int = 100, now_ts: float | None = None) -> SocialScan:
    """Bounded Reddit + X scan for a single name. Uses the existing fetchers (Reddit OAuth/JSON/RSS,
    X official API only when X_BEARER_TOKEN is set) and the transparent spam/dedupe filter. Items
    are sourced evidence — no pre-aggregated bullish/bearish label is produced for downstream use.
    """
    import time as _t
    now_ts = now_ts if now_ts is not None else _t.time()
    tickers = {symbol.upper()}
    if leverage_symbol:
        tickers.add(leverage_symbol.upper())

    docs: list[dict] = []
    statuses: dict = {}
    if allow_fetch:
        for sub in subreddits:
            for listing in ("hot", "new"):
                try:
                    posts = fetch_reddit_posts(subreddit=sub, listing=listing,
                                               limit=reddit_limit, ttl_s=600, allow_fetch=True)
                except Exception:
                    posts = []
                statuses[f"reddit:{sub}:{listing}"] = len(posts)
                for q in posts:
                    docs.append({
                        "source": f"reddit/{sub}",
                        "text": f"{q.get('title', '')} {q.get('selftext', '')}",
                        "title": (q.get("title", "") or "")[:180],
                        "url": q.get("permalink", ""),
                        "score": _f(q.get("score")) or 0.0,
                        "ts": _f(q.get("created_utc")) or 0.0,
                    })
        x_query = (f"(${symbol} OR {symbol}"
                   + (f" OR ${leverage_symbol}" if leverage_symbol else "")
                   + ") lang:en -is:retweet -crypto -btc -telegram -whatsapp")
        try:
            x_posts, x_status = fetch_x_mentions(x_query, limit=x_limit, ttl_s=600)
        except Exception as exc:
            x_posts, x_status = [], f"error: {exc}"
        statuses["x"] = x_status
        for t in x_posts:
            text = t.get("text", "") or ""
            docs.append({
                "source": "x", "text": text, "title": text[:180],
                "url": f"https://twitter.com/i/web/status/{t.get('id', '')}",
                "score": 0.0, "ts": _parse_x_ts(t.get("created_at", "")),
            })
    else:
        statuses["status"] = "live fetch disabled"

    raw_n = len(docs)
    # Conservative spam/dedupe (no options-context requirement for a stock analyzer)...
    docs = _quality_filter(docs, lambda d: d.get("text", ""),
                           allowed=tickers, require_options_context=False)
    # ...then require the actual ticker(s) to appear, so the scan is about THIS name.
    docs = [d for d in docs
            if any(tk in d["text"].upper() for tk in tickers)]
    mentions = extract_ticker_mentions([d["text"] for d in docs], allowed=tickers)
    scores = score_social(mentions, [{"text": d["text"], "ts": d.get("ts", 0.0),
                                      "weight": d.get("score", 0.0)} for d in docs])
    top = sorted(docs, key=lambda d: (-(d.get("score") or 0.0), -float(d.get("ts") or 0.0)))[:10]
    evidence = [{
        "source": d["source"], "title": " ".join(d["title"].split())[:220], "url": d.get("url", ""),
        "score": d.get("score", 0.0), "ts": d.get("ts", 0.0),
        "age_hours": round((now_ts - float(d["ts"])) / 3600.0, 1) if d.get("ts") else None,
    } for d in top]
    return SocialScan(statuses=statuses, raw_docs=raw_n, quality_docs=len(docs),
                      mentions=dict(mentions), scores=scores, evidence=evidence)


# ---------------------------------------------------------------------------
# Leveraged-ETF diagnostics
# ---------------------------------------------------------------------------

def leveraged_diagnostics(hist: dict, base_symbol: str,
                          leverage_symbol: str) -> LeverageDiagnostics:
    """Realized daily beta/correlation of the leveraged ETF vs its underlying, plus cumulative
    return over 1m/3m/6m/1y compared to a synthetic *daily-reset 2x* series — surfaces the
    path-dependence/decay that makes a daily-reset 2x ETF a poor multi-day proxy for 2x the move.
    """
    import pandas as pd
    diag = LeverageDiagnostics(base_symbol=base_symbol, leverage_symbol=leverage_symbol)
    b = hist.get(base_symbol)
    x = hist.get(leverage_symbol)
    if b is None or x is None or getattr(b, "empty", True) or getattr(x, "empty", True):
        diag.note = "insufficient history for leverage diagnostics"
        return diag
    df = pd.DataFrame({"b": b["Close"], "x": x["Close"]}).dropna()
    common = pd.concat([df["b"].pct_change().rename("b"),
                        df["x"].pct_change().rename("x")], axis=1).dropna()
    if len(common) < 20:
        diag.note = "insufficient overlapping history (<20 days)"
        return diag
    var_b = common["b"].var()
    diag.realized_daily_beta = _f(common["x"].cov(common["b"]) / var_b) if var_b else None
    diag.daily_corr = _f(common["x"].corr(common["b"]))
    for label, n in (("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)):
        if len(common) > n:
            sub = common.iloc[-n:]
            diag.periods[label] = {
                "base": _f((1 + sub["b"]).prod() - 1),
                "lev": _f((1 + sub["x"]).prod() - 1),
                "daily_2x_synth": _f((1 + 2 * sub["b"]).prod() - 1),
                "tracking_gap": _f(((1 + sub["x"]).prod() - 1) - ((1 + 2 * sub["b"]).prod() - 1)),
            }
    return diag


# ---------------------------------------------------------------------------
# Position-structure helper (pure arithmetic; hypothetical sizing only)
# ---------------------------------------------------------------------------

def position_structure(total_equity: float, common_pct: float, levered_pct: float,
                       cash_pct: float) -> dict:
    """Translate target sleeve percentages into dollar targets against the latest portfolio total.
    Pure arithmetic — hypothetical sizing only; this NEVER places or proposes a live order."""
    total = _f(total_equity) or 0.0
    cp, lp, kp = (_f(common_pct) or 0.0, _f(levered_pct) or 0.0, _f(cash_pct) or 0.0)
    allocated = cp + lp + kp
    warning = None
    if allocated > 100.0 + 1e-9:
        warning = f"target allocation sums to {allocated:.1f}% (>100%)"
    return {
        "total_equity": total,
        "common_dollars": total * cp / 100.0,
        "levered_dollars": total * lp / 100.0,
        "cash_dollars": total * kp / 100.0,
        "allocated_pct": allocated,
        "unallocated_pct": max(0.0, 100.0 - allocated),
        "warning": warning,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze(symbol: str, leverage_symbol: str | None = None, *, allow_fetch: bool = True,
            include_social: bool = True, include_news: bool = True,
            include_options: bool = True, now: datetime | None = None) -> SingleStockAnalysis:
    """Assemble a structured single-stock analysis. Every section fails closed independently, so a
    network/yfinance outage degrades gracefully instead of raising. Places NO orders."""
    symbol = (symbol or "").strip().upper()
    leverage_symbol = (leverage_symbol or "").strip().upper() or None
    now = now or datetime.now(timezone.utc)
    if not symbol:
        raise ValueError("symbol is required")

    watch = list(dict.fromkeys([s for s in (symbol, leverage_symbol, "SPY") if s]))
    statuses: dict = {}

    trends, hist = price_snapshot(watch, allow_fetch=allow_fetch)
    exposure = holdings_exposure(watch)
    cached = cached_factors(symbol)
    fundamentals = fundamentals_snapshot(symbol, allow_fetch=allow_fetch)
    news = yf_news(symbol, allow_fetch=allow_fetch) if include_news else []
    social = (social_scan(symbol, leverage_symbol, allow_fetch=allow_fetch)
              if include_social else None)
    leverage = (leveraged_diagnostics(hist, symbol, leverage_symbol)
                if leverage_symbol else None)
    options = (options_snapshot(symbol, allow_fetch=allow_fetch)
               if include_options else {"status": "skipped"})

    statuses["price"] = "ok" if symbol in hist else trends.get(symbol, PriceTrend(symbol)).error
    statuses["holdings"] = exposure.status
    statuses["fetch"] = "live" if allow_fetch else "offline"

    return SingleStockAnalysis(
        symbol=symbol, leverage_symbol=leverage_symbol,
        generated_at=now.isoformat(timespec="seconds"), disclaimer=DISCLAIMER,
        exposure=exposure, cached_factors=cached, price_trends=trends,
        fundamentals=fundamentals, news=news, social=social, leverage=leverage,
        options=options, statuses=statuses,
    )
