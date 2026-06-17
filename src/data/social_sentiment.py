"""
data/social_sentiment.py — social-sentiment / 0DTE watchlist (ANALYSIS / PAPER ONLY).

Decision-support only. **NOT financial advice. NOT auto-trading. This module places NO orders
and imports NO broker/execution code.** It fetches PUBLIC Reddit (r/wallstreetbets) post JSON
and — only when X_BEARER_TOKEN is set — X/Twitter via the OFFICIAL API (never ToS-bypassing
scraping), extracts ticker mentions for a limited universe, scores transparent hype / sentiment
/ momentum heuristics, and builds a 0DTE *idea* report for paper analysis at a tiny fixed budget.

Network is optional: every fetch fails closed (returns empty + status), and results are cached.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests

from core.paths import DATA_DIR

logger = logging.getLogger(__name__)

_UA = "daily-investor-research/1.0 (social-sentiment analysis; non-commercial)"
_CACHE_DIR = DATA_DIR / "social_cache"
_REDDIT_URL = "https://www.reddit.com/r/{sub}/{listing}.json"
_REDDIT_RSS_URL = "https://www.reddit.com/r/{sub}/.rss"
_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_REDDIT_OAUTH_URL = "https://oauth.reddit.com/r/{sub}/{listing}"
_REDDIT_COMMENTS_JSON_URL = "https://www.reddit.com/comments/{id}.json"
_REDDIT_OAUTH_COMMENTS_URL = "https://oauth.reddit.com/comments/{id}"
_X_RECENT_URL = "https://api.twitter.com/2/tweets/search/recent"

# Collection-shape notes (how WSB/0DTE social data is commonly gathered, and what we cover):
#   • Reddit posts via the official app-only OAuth API (preferred) → public listing JSON →
#     Atom/RSS feed. No browser/HTML scraping (brittle + ToS risk; Chrome is unreliable here).
#   • Reddit *comments* for the top posts (people analyze comment threads, not just titles) —
#     bounded + cached + opt-in via reddit_comments_enrich. /comments/{id} (OAuth or public JSON).
#   • X/Twitter via the official recent-search API, only when X_BEARER_TOKEN is set (.env).
#   • Mention counts / bullish ratios / hype-momentum are derived transparently in score_social()
#     for the 0DTE report; the active-sleeve substrate intentionally gets RAW items only.
# TODO (not yet wired, no scraping required): multiple subreddits (r/options, r/stocks, r/Daytrading),
#   StockTwits message-stream API, and post/comment export volume — all fit the same fail-closed,
#   cached, official-API/RSS pattern; add as additional `sources` entries when needed.

# Uppercase tokens that look like tickers but aren't (extend as needed). Sentiment words
# (PUT/CALL/BUY/SELL/...) are excluded as TICKERS here but still counted as lowercase
# sentiment terms in the lexicons below — the two passes are independent.
_STOPWORDS = frozenset({
    "I", "A", "AN", "THE", "DD", "YOLO", "FD", "FDS", "CEO", "CFO", "IPO", "ETF", "ATH", "EOD",
    "EOW", "WSB", "IMO", "IMHO", "TLDR", "US", "USA", "IT", "BE", "TO", "OR", "AND", "FOR", "ON",
    "IN", "OF", "IS", "AT", "BY", "UP", "SO", "NO", "GO", "DO", "AM", "PM", "ER", "OG", "GG",
    "LOL", "LMAO", "PUT", "PUTS", "CALL", "CALLS", "BUY", "SELL", "HOLD", "MOON", "BULL", "BEAR",
    "RH", "API", "CPI", "FED", "GDP", "EPS", "PE", "ROI", "WSJ", "CNBC", "SEC", "IRS", "OK", "TA",
    "HODL", "RIP", "ELI", "EV", "AI", "OP", "IV", "OTM", "ITM", "YOY", "QOQ", "EOY",
})
_BULL = ("call", "calls", "moon", "rocket", "buy", "long", "bull", "bullish", "pump",
         "squeeze", "breakout", "rip", "🚀", "💎")
_BEAR = ("put", "puts", "short", "sell", "bear", "bearish", "crash", "dump", "drill",
         "tank", "collapse", "puke")


# ---------------------------------------------------------------------------
# Tiny TTL JSON cache (so repeated runs / tests don't hammer the network)
# ---------------------------------------------------------------------------

def _cache_path(key: str):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9_]', '_', key)}.json"


def _read_cache(key: str, ttl_s: float):
    p = _cache_path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl_s:
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _write_cache(key: str, data) -> None:
    try:
        _cache_path(key).write_text(json.dumps(data))
    except Exception as exc:
        logger.debug("social cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Fetchers — fail closed (empty + status), never raise to the caller
# ---------------------------------------------------------------------------

def _parse_reddit_rss(text: str, limit: int) -> list[dict]:
    """Parse Reddit's public Atom feed as a fallback when listing JSON is blocked.

    Reddit frequently returns 403 for anonymous ``/hot.json`` requests from servers, while
    the subreddit Atom feed remains publicly accessible. The feed gives title/link/date but
    not score/comment counts, so engagement fields are neutral 0. This keeps the social
    report live without credentialed Reddit API setup and without scraping HTML.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    posts: list[dict] = []
    for entry in root.findall("a:entry", ns)[:limit]:
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:content", default="", namespaces=ns)
                   or entry.findtext("a:summary", default="", namespaces=ns) or "")
        link_el = entry.find("a:link", ns)
        href = link_el.get("href", "") if link_el is not None else ""
        updated = entry.findtext("a:updated", default="", namespaces=ns) or ""
        ts = _parse_x_ts(updated)
        author = entry.findtext("a:author/a:name", default="", namespaces=ns) or None
        posts.append({
            "title": title,
            "selftext": re.sub(r"<[^>]+>", " ", summary).strip(),
            "score": 0,
            "num_comments": 0,
            "permalink": href,
            "created_utc": ts,
            "author": author,
            "api_source": "reddit_rss",
        })
    return posts


def _reddit_user_agent() -> str:
    """Reddit asks for a descriptive, unique UA. Honor REDDIT_USER_AGENT if the operator set one."""
    return os.environ.get("REDDIT_USER_AGENT") or _UA


def _parse_reddit_children(children: list, api_source: str) -> list[dict]:
    """Parse a Reddit listing `data.children` array (same shape for public JSON and OAuth) into
    the post dicts the rest of the module consumes."""
    posts: list[dict] = []
    for c in children:
        d = c.get("data", {}) if isinstance(c, dict) else {}
        posts.append({
            "id": d.get("id", "") or "",   # base36 post id — needed for /comments/{id}.json
            "title": d.get("title", "") or "",
            "selftext": d.get("selftext", "") or "",
            "score": int(d.get("score", 0) or 0),
            "num_comments": int(d.get("num_comments", 0) or 0),
            "permalink": "https://www.reddit.com" + (d.get("permalink", "") or ""),
            "created_utc": float(d.get("created_utc", 0) or 0),
            "author": d.get("author"),
            "api_source": api_source,
        })
    return posts


def _fetch_reddit_oauth(subreddit: str, listing: str, limit: int) -> list[dict] | None:
    """Official Reddit app-only OAuth (``client_credentials``) listing fetch.

    Preferred over anonymous endpoints when REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET are set:
    it is ToS-clean, stable, and not rate-limited/403'd like anonymous server requests. Returns
    parsed posts on success, or ``None`` when creds are absent OR the call fails — the caller
    then falls back to public JSON, then Atom/RSS. No browser/HTML scraping is ever used.
    """
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        logger.debug("reddit OAuth: REDDIT_CLIENT_ID/SECRET not set — using public endpoints")
        return None
    ua = _reddit_user_agent()
    try:
        tok = requests.post(
            _REDDIT_TOKEN_URL,
            auth=(cid, secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": ua},
            timeout=15,
        )
        tok.raise_for_status()
        access = (tok.json() or {}).get("access_token")
        if not access:
            logger.warning("reddit OAuth: token response carried no access_token; falling back")
            return None
        r = requests.get(
            _REDDIT_OAUTH_URL.format(sub=subreddit, listing=listing),
            headers={"Authorization": f"bearer {access}", "User-Agent": ua},
            params={"limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        children = (r.json() or {}).get("data", {}).get("children", [])
        posts = _parse_reddit_children(children, api_source="reddit_oauth")
        logger.info("reddit OAuth fetch ok: %d posts from r/%s/%s", len(posts), subreddit, listing)
        return posts
    except Exception as exc:
        logger.warning("reddit OAuth fetch failed (%s); falling back to public endpoints", exc)
        return None


def fetch_reddit_posts(subreddit: str = "wallstreetbets", listing: str = "hot",
                       limit: int = 50, ttl_s: float = 900, allow_fetch: bool = True) -> list[dict]:
    """Fetch a subreddit listing, cache-backed, in preference order (each fails closed to the
    next): (1) official OAuth app-only API when REDDIT_CLIENT_ID/SECRET are set, (2) public
    listing JSON, (3) public Atom/RSS feed. Returns a list of
    {title, selftext, score, num_comments, permalink, created_utc, author, api_source}; [] on
    total failure. No Selenium / HTML scraping."""
    key = f"reddit_{subreddit}_{listing}_{limit}"
    cached = _read_cache(key, ttl_s)
    if cached is not None:
        return cached
    if not allow_fetch:
        return []
    # 1) Official OAuth (preferred). None => creds absent or call failed → fall through.
    oauth_posts = _fetch_reddit_oauth(subreddit, listing, limit)
    if oauth_posts is not None:
        _write_cache(key, oauth_posts)
        return oauth_posts
    # 2) Public listing JSON.
    url = _REDDIT_URL.format(sub=subreddit, listing=listing)
    try:
        r = requests.get(url, headers={"User-Agent": _reddit_user_agent()},
                         params={"limit": limit}, timeout=15)
        r.raise_for_status()
        children = (r.json() or {}).get("data", {}).get("children", [])
    except Exception as exc:
        logger.warning("reddit JSON fetch failed (%s): %s; trying Atom feed", url, exc)
        # 3) Public Atom/RSS feed.
        try:
            rss = requests.get(
                _REDDIT_RSS_URL.format(sub=subreddit),
                headers={"User-Agent": _reddit_user_agent()},
                params={"limit": limit},
                timeout=15,
            )
            rss.raise_for_status()
            posts = _parse_reddit_rss(rss.text, limit)
            _write_cache(key, posts)
            return posts
        except Exception as rss_exc:
            logger.warning("reddit RSS fetch failed (%s): %s", subreddit, rss_exc)
            return []
    posts = _parse_reddit_children(children, api_source="reddit_json")
    _write_cache(key, posts)
    return posts


def _reddit_oauth_token() -> str | None:
    """Mint an app-only OAuth bearer token (``client_credentials``) when REDDIT_CLIENT_ID +
    REDDIT_CLIENT_SECRET are set; ``None`` otherwise / on failure. Used by the comments fetcher
    (the listing fetcher mints its own inline). No browser/HTML scraping."""
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        return None
    try:
        tok = requests.post(
            _REDDIT_TOKEN_URL, auth=(cid, secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": _reddit_user_agent()}, timeout=15)
        tok.raise_for_status()
        return (tok.json() or {}).get("access_token") or None
    except Exception as exc:
        logger.warning("reddit OAuth token fetch failed (%s)", exc)
        return None


def _parse_reddit_comments(payload, limit: int) -> list[dict]:
    """Parse Reddit's ``/comments/{id}`` JSON ([post_listing, comments_listing]) into top-level
    comment dicts {body, score, author}. Skips non-comment ("more"/load-more) nodes."""
    try:
        children = payload[1]["data"]["children"]
    except Exception:
        return []
    out: list[dict] = []
    for c in children:
        if not isinstance(c, dict) or c.get("kind") != "t1":
            continue
        d = c.get("data", {}) or {}
        body = (d.get("body") or "").strip()
        if not body:
            continue
        out.append({"body": body[:500], "score": int(d.get("score", 0) or 0),
                    "author": d.get("author")})
        if len(out) >= limit:
            break
    return out


def fetch_reddit_comments(post_id: str, limit: int = 5, ttl_s: float = 900,
                          allow_fetch: bool = True) -> list[dict]:
    """Top-level comments for a post (official OAuth when creds present, else public JSON),
    cache-backed and fail-closed. Returns ``[{body, score, author}]`` (<= ``limit``); ``[]`` on
    any failure. WSB analysis often reads *comments*, not just titles — this is small, bounded,
    and cached, and never scrapes HTML. (Atom/RSS carries no comments, so RSS-fallback posts —
    which have no id — simply yield [].)"""
    if not post_id:
        return []
    key = f"reddit_comments_{post_id}_{limit}"
    cached = _read_cache(key, ttl_s)
    if cached is not None:
        return cached
    if not allow_fetch:
        return []
    token = _reddit_oauth_token()
    ua = _reddit_user_agent()
    params = {"limit": limit, "depth": 1, "sort": "top"}
    try:
        if token:
            r = requests.get(_REDDIT_OAUTH_COMMENTS_URL.format(id=post_id),
                             headers={"Authorization": f"bearer {token}", "User-Agent": ua},
                             params=params, timeout=15)
        else:
            r = requests.get(_REDDIT_COMMENTS_JSON_URL.format(id=post_id),
                             headers={"User-Agent": ua}, params=params, timeout=15)
        r.raise_for_status()
        comments = _parse_reddit_comments(r.json() or [], limit)
        _write_cache(key, comments)
        return comments
    except Exception as exc:
        logger.warning("reddit comments fetch failed for %s (%s)", post_id, exc)
        return []


def _fold_comments_into_posts(posts: list[dict], top_posts: int, per_post: int,
                              allow_fetch: bool) -> list[dict]:
    """Bounded comments enrichment: for the top `top_posts` posts by score, fetch up to
    `per_post` top comments and append their text to that post's selftext, so the active-sleeve
    LLM (and ticker extraction) sees the crowd discussion, not just the title. Returns copies;
    fail-closed (a failed fetch just leaves the post unchanged)."""
    enriched = [dict(q) for q in posts]
    ranked = sorted(range(len(enriched)), key=lambda i: -int(enriched[i].get("score", 0) or 0))
    for i in ranked[: max(0, top_posts)]:
        pid = enriched[i].get("id")
        if not pid:
            continue
        comments = fetch_reddit_comments(pid, limit=per_post, allow_fetch=allow_fetch)
        if comments:
            joined = " | ".join(c["body"] for c in comments)
            base = (enriched[i].get("selftext", "") or "").strip()
            enriched[i]["selftext"] = f"{base} Top comments: {joined}".strip()
            enriched[i]["comments_sampled"] = len(comments)
    return enriched


def fetch_x_mentions(query: str, limit: int = 50, ttl_s: float = 900) -> tuple[list[dict], str]:
    """X/Twitter recent search — ONLY via the official API when X_BEARER_TOKEN is set; otherwise
    SKIP cleanly (no ToS-bypassing scraping). Returns (posts, status)."""
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        return [], "skipped: X_BEARER_TOKEN not set"
    key = f"x_{query}_{limit}"
    cached = _read_cache(key, ttl_s)
    if cached is not None:
        return cached, "cache"
    try:
        r = requests.get(
            _X_RECENT_URL,
            headers={"Authorization": f"Bearer {token}", "User-Agent": _UA},
            params={"query": query, "max_results": min(max(limit, 10), 100),
                    "tweet.fields": "public_metrics,created_at"},
            timeout=15,
        )
        r.raise_for_status()
        data = (r.json() or {}).get("data", []) or []
    except Exception as exc:
        logger.warning("X fetch failed: %s", exc)
        return [], f"error: {exc}"
    posts = [{"text": t.get("text", "") or "", "created_at": t.get("created_at", ""),
              "id": t.get("id", "")} for t in data]
    _write_cache(key, posts)
    return posts, "ok"


# ---------------------------------------------------------------------------
# Extraction + transparent scoring
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\$?\b([A-Z]{1,5})\b")


def extract_ticker_mentions(texts, allowed: set[str] | None = None) -> Counter:
    """Count $TICKER / TICKER tokens, dropping stopwords. If `allowed` is given, restrict to it
    (e.g. a vetted universe). Common English/jargon all-caps words are excluded via _STOPWORDS."""
    cnt: Counter = Counter()
    for t in texts:
        for m in _TICKER_RE.findall(t or ""):
            if m in _STOPWORDS:
                continue
            if allowed is not None and m not in allowed:
                continue
            cnt[m] += 1
    return cnt


def score_social(mentions: Counter, documents: list[dict]) -> dict:
    """Per-ticker transparent scores from `documents` (each {text, ts, weight}):
      hype      = share of total mentions (0..1)
      sentiment = (bull-bear)/(bull+bear) over docs mentioning the ticker (-1..1)
      momentum  = share of the ticker's mention-docs newer than the median doc timestamp (0..1)
    """
    total = sum(mentions.values()) or 1
    ts_all = sorted(d.get("ts", 0.0) for d in documents)
    median_ts = ts_all[len(ts_all) // 2] if ts_all else 0.0
    out: dict = {}
    for tk, n in mentions.items():
        tkl = tk.lower()
        pat = re.compile(rf"\$?\b{re.escape(tkl)}\b", re.IGNORECASE)
        bull = bear = recent = hits = 0
        for d in documents:
            txt = (d.get("text", "") or "")
            if not pat.search(txt):
                continue
            hits += 1
            low = txt.lower()
            bull += sum(low.count(w) for w in _BULL)
            bear += sum(low.count(w) for w in _BEAR)
            if d.get("ts", 0.0) >= median_ts:
                recent += 1
        denom = (bull + bear) or 1
        out[tk] = {
            "mentions": int(n),
            "hype": round(n / total, 4),
            "bull": int(bull),
            "bear": int(bear),
            "sentiment": round((bull - bear) / denom, 3),
            "momentum": round(recent / hits, 3) if hits else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Transparent spam / quality filtering (NO ML model, no network). X recent-search in
# particular returns promo/scam noise — Telegram/VIP/100X signal pumps, crypto promo, and
# class-action/legal blasts — that mention SPY/QQQ but carry no 0DTE signal and inflate
# mention counts. These rules are simple, inspectable substring/regex checks.
# ---------------------------------------------------------------------------

# High-precision promo/scam/off-topic substrings (matched case-insensitively).
_SPAM_TERMS = (
    "telegram", "t.me/", "vip", "100x", "1000x", "free signal", "join my", "join our",
    "join the", "whatsapp", "dm me", "dm for", "dm us", "discord", "discord.gg", "giveaway",
    "guaranteed", "link in bio", "copy my trade", "signals group", "premium group",
    "alerts group", "cash app", "cashapp", "venmo", "promo code",
    # class-action / legal blast spam (frequent on $TICKER searches)
    "class action", "law firm", "rosen law", "shareholder rights", "securities fraud",
    "investors who purchased", "investigation on behalf",
)
# Crypto/off-topic terms: only spam when the doc is NOT also about index options (below).
_CRYPTO_TERMS = ("bitcoin", "btc", "ethereum", "crypto", "altcoin", "dogecoin", " doge",
                 "solana", "$sol", "xrp", "forex", "memecoin", "shitcoin", "xauusd")
# Options / day-trading / catalyst context (required for ODTE evidence, NOT for news enrichment).
_OPTIONS_CONTEXT_RE = re.compile(
    r"\b(0dte|odte|calls?|puts?|options?|strikes?|expir\w+|contracts?|theta|gamma|delta|"
    r"premium|scalp\w*|intraday|day[\s-]?trade\w*|swing|hedg\w*|spx|spreads?|fomc|cpi|opex|"
    r"implied\s+vol\w*)\b", re.IGNORECASE)
_CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,5}\b")
_MAX_CASHTAGS = 6   # shotgun ticker-spam: a tweet tagging many tickers is not focused signal
_MAX_MENTIONS = 4   # shotgun @account-spam


def _has_options_context(text: str) -> bool:
    """True if the text carries an options / day-trading / catalyst token (0DTE, call/put,
    strike, scalp, FOMC, ...). Bare tickers do NOT count — that's the separate ticker gate."""
    return bool(_OPTIONS_CONTEXT_RE.search(text or ""))


def _is_spam(text: str) -> bool:
    """Transparent promo/scam/off-topic check (no ML): hard promo terms, crypto-focused-and-not-
    options, or shotgun cashtag/@account spam. Used to keep both the ODTE report and the
    active-sleeve news enrichment clean."""
    low = (text or "").lower()
    if any(t in low for t in _SPAM_TERMS):
        return True
    if any(t in low for t in _CRYPTO_TERMS) and not _has_options_context(low):
        return True
    if len(_CASHTAG_RE.findall(low)) > _MAX_CASHTAGS or low.count("@") > _MAX_MENTIONS:
        return True
    return False


def _dedup_key(text: str) -> str:
    """Normalized key for near-duplicate detection: lowercase, URLs/mentions/punctuation stripped,
    whitespace collapsed, truncated. Catches reposted/boilerplate spam that differs only in
    links, casing, or trailing tags."""
    low = re.sub(r"https?://\S+", " ", (text or "").lower())
    low = re.sub(r"[@#]\w+", " ", low)
    low = re.sub(r"[^a-z0-9 ]+", " ", low)
    return " ".join(low.split())[:160]


def _quality_filter(items: list, text_fn, *, allowed: set[str] | None = None,
                    require_options_context: bool = False) -> list:
    """Drop spam + near-duplicates from `items` (transparent, order-preserving). For ODTE evidence
    pass require_options_context=True (also requires ≥1 allowed ticker); for active-sleeve news
    enrichment pass False (conservative — spam/dedupe only, provenance preserved)."""
    kept: list = []
    seen: set[str] = set()
    for it in items:
        text = text_fn(it) or ""
        if _is_spam(text):
            continue
        if require_options_context:
            if not _has_options_context(text):
                continue
            if allowed is not None and not _tickers_in(text, allowed):
                continue
        key = _dedup_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(it)
    return kept


# ---------------------------------------------------------------------------
# Report (paper / analysis only)
# ---------------------------------------------------------------------------

def _parse_x_ts(s: str) -> float:
    """Best-effort X created_at (ISO) -> epoch seconds; 0.0 if unparseable (X support partial)."""
    if s:
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            continue
    return 0.0


def _eastern() -> object:
    """America/New_York tz (DST-correct via stdlib zoneinfo). Falls back to a fixed EST offset
    if the system has no tz database, so the freshness window degrades gracefully (never raises)."""
    try:
        return ZoneInfo("America/New_York")
    except (ZoneInfoNotFoundError, Exception):  # pragma: no cover - depends on host tzdata
        return timezone(timedelta(hours=-5))


def _parse_hm(value: str | None, default: tuple[int, int]) -> tuple[int, int]:
    """Parse an "HH:MM" clock string to (hour, minute); fall back to `default` on any error."""
    try:
        hh, mm = str(value).split(":")
        return int(hh), int(mm)
    except Exception:
        return default


def market_session_window(
    now: datetime | None = None,
    *,
    open_hm: tuple[int, int] = (9, 30),
    close_hm: tuple[int, int] = (16, 0),
    max_lookback_hours: float = 96.0,
) -> tuple[datetime, datetime]:
    """Return ``(window_start_utc, window_end_utc)`` for social-post freshness, anchored to US
    equity-market sessions in America/New_York (DST-correct via stdlib ``zoneinfo``).

    Policy — so weekend/pre-market sentiment accumulated *since the last regular close* is kept
    for the next session's prep, rather than relying on the UTC calendar day:

      * Weekend (Sat/Sun)         → previous Friday's regular close (``close_hm`` ET).
      * Weekday before the open   → previous trading day's close (``close_hm`` ET).
      * Weekday at/after the open  → the current day's open (``open_hm`` ET).

    Holidays are best-effort: only Sat/Sun are skipped (no market-calendar utility exists here),
    so a holiday is treated as a trading day. The start is floored to ``now − max_lookback_hours``
    so an unusually long gap can't widen the window without bound. ``end`` is ``now``.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    et_tz = _eastern()
    et = now.astimezone(et_tz)

    def _close_of(d) -> datetime:
        return datetime(d.year, d.month, d.day, close_hm[0], close_hm[1], tzinfo=et_tz)

    def _prev_trading_date(d):
        x = d - timedelta(days=1)
        while x.weekday() >= 5:  # skip Sat(5)/Sun(6)
            x -= timedelta(days=1)
        return x

    if et.weekday() >= 5 or et.time() < dtime(open_hm[0], open_hm[1]):
        # Weekend, or a weekday before the open → since the previous trading day's close.
        start_et = _close_of(_prev_trading_date(et.date()))
    else:
        # Weekday at/after the open (incl. post-close same day) → since today's open.
        start_et = et.replace(hour=open_hm[0], minute=open_hm[1], second=0, microsecond=0)

    start_utc = start_et.astimezone(timezone.utc)
    floor = now - timedelta(hours=max_lookback_hours)
    if start_utc < floor:
        start_utc = floor
    return start_utc, now


def _odte_evidence_row(q: dict, now_ts: float) -> dict:
    ts = q.get("ts", 0.0)
    return {"title": q["title"], "score": q["score"], "url": q["url"], "source": q["source"],
            "post_time": (datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")
                          if ts else None),
            "age_hours": round((now_ts - ts) / 3600.0, 1) if ts else None}


def _collect_odte_evidence(ticker: str, ev_pool: list[dict], now_ts: float,
                           max_items: int = 3) -> list[dict]:
    """Up to `max_items` dedup'd evidence rows mentioning `ticker`, ranked by engagement then
    recency. RSS and X often have neutral score=0, so recency keeps the report from always
    showing only Reddit RSS items when fresh X posts are present."""
    ev_pat = re.compile(rf"\$?\b{re.escape(ticker)}\b")
    evidence: list[dict] = []
    seen_evidence: set[str] = set()
    for q in sorted(ev_pool, key=lambda q: (-q["score"], -q.get("ts", 0.0))):
        if not ev_pat.search(q["text"]):
            continue
        dedupe_key = re.sub(r"\s+", " ", q["text"].lower()).strip()[:180]
        if dedupe_key in seen_evidence:
            continue
        seen_evidence.add(dedupe_key)
        evidence.append(_odte_evidence_row(q, now_ts))
        if len(evidence) >= max_items:
            break
    return evidence


def _select_odte_candidate(ranked: list, ev_pool: list[dict], min_mentions: int,
                           now_ts: float) -> dict | None:
    """First ranked ticker clearing the mention floor with nonzero directional sentiment, plus its
    evidence. Returns None when nothing qualifies."""
    for tk, s in ranked:
        if s["mentions"] >= min_mentions and abs(s["sentiment"]) > 0.0:
            return {
                "ticker": tk,
                "direction": "bullish" if s["sentiment"] > 0 else "bearish",
                **s,
                "evidence": _collect_odte_evidence(tk, ev_pool, now_ts),
            }
    return None


def _resolve_paper_options(candidate: dict | None, budget: float, include: bool,
                           allow_fetch: bool) -> dict:
    """PAPER-ONLY 0DTE option idea (yfinance; no orders). Skips cleanly without a candidate, when
    disabled, or under --no-fetch; fails closed on any lookup error."""
    if not (candidate and include and allow_fetch):
        return {"status": "skipped (no candidate / disabled / --no-fetch)", "contracts": []}
    try:
        from data.odte_options import build_paper_options
        return build_paper_options(candidate["ticker"], candidate["direction"], budget,
                                   allow_fetch=allow_fetch)
    except Exception as exc:
        logger.warning("paper options lookup failed: %s", exc)
        return {"status": f"error: {exc}", "contracts": []}


def _gather_odte_sources(p: dict, allowed: set[str] | None, allow_fetch: bool, is_fresh) -> dict:
    """Fetch Reddit + X posts, freshness-filter, run the quality filter ONCE, then score & rank
    tickers. Deriving mentions, scoring, and the evidence pool from the SAME filtered set keeps
    them consistent — so promo/spam can't inflate a ticker's mention count while being excluded
    from evidence (the live "SPY 47" bug). Returns post/spam counts plus `ranked` tickers and the
    quality-gated `ev_pool` evidence set."""
    subreddit = p.get("subreddit", "wallstreetbets")
    sources = p.get("sources", ["reddit"])

    posts_all = []
    if "reddit" in sources:
        posts_all = fetch_reddit_posts(subreddit=subreddit, listing="hot",
                                       limit=int(p.get("reddit_limit", 50)), allow_fetch=allow_fetch)
    posts = [q for q in posts_all if is_fresh(q.get("created_utc", 0.0))]

    x_posts_all: list[dict] = []
    x_status = "disabled (not in sources)"
    if "x" in sources:
        x_posts_all, x_status = fetch_x_mentions(
            query=p.get("x_query", "($SPY OR $QQQ OR 0DTE OR ODTE) lang:en -is:retweet -crypto -gold -xauusd"),
            limit=int(p.get("x_limit", 50)))
    x_posts = [t for t in x_posts_all if is_fresh(_parse_x_ts(t.get("created_at", "")))]

    # Build ONE combined item list (Reddit + X) carrying everything mentions/scoring AND evidence
    # need, then run the transparent spam/quality filter ONCE.
    combined = [{"text": f"{q['title']} {q['selftext']}", "ts": q.get("created_utc", 0.0),
                 "weight": q.get("score", 0), "title": q["title"][:140], "score": q["score"],
                 "url": q["permalink"], "source": "reddit"} for q in posts]
    combined += [{"text": t.get("text", ""), "ts": _parse_x_ts(t.get("created_at", "")),
                  "weight": 0, "title": (t.get("text", "") or "")[:140], "score": 0,
                  "url": f"https://twitter.com/i/web/status/{t.get('id', '')}", "source": "x"}
                 for t in x_posts]
    # ODTE evidence requires an allowed ticker + options/day-trading context (drops generic
    # SPY/QQQ chatter and risk-management platitudes that aren't 0DTE signal).
    kept = _quality_filter(combined, lambda c: c["text"], allowed=allowed,
                           require_options_context=True)
    reddit_spam = sum(1 for c in combined if c["source"] == "reddit") - \
        sum(1 for c in kept if c["source"] == "reddit")
    x_spam = sum(1 for c in combined if c["source"] == "x") - \
        sum(1 for c in kept if c["source"] == "x")

    documents = [{"text": c["text"], "ts": c["ts"], "weight": c["weight"]} for c in kept]
    mentions = extract_ticker_mentions([c["text"] for c in kept], allowed=allowed)
    scores = score_social(mentions, documents)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["mentions"])[: int(p.get("max_tickers", 10))]

    return {
        "subreddit": subreddit,
        "posts": posts, "posts_all": posts_all,
        "reddit_stale": len(posts_all) - len(posts), "reddit_spam": reddit_spam,
        "x_posts": x_posts, "x_posts_all": x_posts_all, "x_status": x_status,
        "x_stale": len(x_posts_all) - len(x_posts), "x_spam": x_spam,
        "ranked": ranked, "ev_pool": kept,  # ev_pool is the same filtered, quality-gated set
    }


def build_odte_social_report(allow_fetch: bool = True, params: dict | None = None,
                             now: datetime | None = None) -> dict:
    """Build the 0DTE social-sentiment IDEA report (paper/analysis only). Never places orders."""
    from util import OPTIONS_SOCIAL_PARAMS
    p = params if params is not None else OPTIONS_SOCIAL_PARAMS

    # Allowed-universe filter (avoids ranking arbitrary all-caps false positives): for the
    # 0DTE report, default to the explicitly configured liquid/core underlyings only
    # (SPY/QQQ by default). A broad agg_data expansion produced real-but-unhelpful names/ETFs
    # that may not have liquid 0DTE chains; users can expand core_universe deliberately later.
    allowed: set[str] | None = {str(s).upper() for s in p.get("core_universe", ["SPY", "QQQ"])}
    if not allowed:
        allowed = None

    # MARKET-SESSION freshness: consider only posts since the relevant session boundary, so
    # weekend/pre-market chatter accumulated since the last regular close is retained for the
    # next session's prep (see market_session_window). `freshness_mode: market_window` (default)
    # uses that session anchor; anything else falls back to a plain rolling max_lookback_hours
    # window. Posts with no/unparseable timestamp are treated as STALE (excluded).
    now = now or datetime.now(timezone.utc)
    now_ts = now.timestamp()
    mode = str(p.get("freshness_mode", "market_window"))
    max_lb = float(p.get("max_lookback_hours", 96))
    open_hm = _parse_hm(p.get("market_open_et", "09:30"), (9, 30))
    close_hm = _parse_hm(p.get("market_close_et", "16:00"), (16, 0))
    if mode == "market_window":
        win_start, win_end = market_session_window(
            now, open_hm=open_hm, close_hm=close_hm, max_lookback_hours=max_lb)
    else:
        win_start, win_end = now - timedelta(hours=max_lb), now
    start_ts, end_ts = win_start.timestamp(), win_end.timestamp()

    def _fresh(ts: float) -> bool:
        return bool(ts and ts > 0 and start_ts <= ts <= end_ts + 1.0)

    src = _gather_odte_sources(p, allowed, allow_fetch, _fresh)
    ranked, ev_pool = src["ranked"], src["ev_pool"]

    min_mentions = int(p.get("min_mentions", 3))
    candidate = _select_odte_candidate(ranked, ev_pool, min_mentions, now_ts)

    # PAPER-ONLY 0DTE option idea (yfinance; no orders). bullish->calls, bearish->puts.
    budget = float(p.get("budget_dollars", 50))
    paper_options = _resolve_paper_options(
        candidate, budget, bool(p.get("include_paper_options", True)), allow_fetch)

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "disclaimer": ("ANALYSIS / PAPER ONLY — not financial advice, not auto-trading. "
                       "No orders are placed by this tool."),
        "budget_dollars": budget,
        "freshness_window": {
            "mode": mode,
            "window_start_et": win_start.astimezone(_eastern()).isoformat(timespec="seconds"),
            "window_start_utc": win_start.isoformat(timespec="seconds"),
            "window_end_utc": win_end.isoformat(timespec="seconds"),
            "max_lookback_hours": max_lb,
        },
        "sources": {
            "reddit": {"subreddit": src["subreddit"], "n_posts": len(src["posts"]),
                       "n_fetched": len(src["posts_all"]), "n_stale_filtered": src["reddit_stale"],
                       "n_filtered": src["reddit_spam"],
                       "n_quality": len(src["posts"]) - src["reddit_spam"]},
            "x": {"status": src["x_status"], "n_posts": len(src["x_posts"]),
                  "n_fetched": len(src["x_posts_all"]), "n_stale_filtered": src["x_stale"],
                  "n_filtered": src["x_spam"], "n_quality": len(src["x_posts"]) - src["x_spam"]},
        },
        "top_tickers": [{"ticker": tk, **s} for tk, s in ranked],
        "candidate": candidate,
        "paper_options": paper_options,
        "risk_notes": [
            "0DTE options decay to zero intraday; treat the entire budget as at-risk to ZERO.",
            "Social hype is a contrarian/lagging signal as often as a leading one — not edge.",
            f"Max paper budget ${budget:.0f}; start at $0 (observe only).",
            "Session-window only: posts before the last market open/close anchor are excluded.",
            "This is decision-support; size, entry, and whether to trade at all are the user's call.",
        ],
    }


def format_report(report: dict) -> str:
    """Human-readable text rendering of build_odte_social_report()'s dict."""
    lines = [
        "=" * 72,
        "0DTE SOCIAL-SENTIMENT WATCHLIST — PAPER / ANALYSIS ONLY",
        "=" * 72,
        report["disclaimer"],
        f"generated_at: {report['generated_at']}  | paper budget: ${report['budget_dollars']:.0f}",
    ]
    fw = report.get("freshness_window", {})
    rd, xs = report["sources"]["reddit"], report["sources"]["x"]
    lines.append(
        f"FRESHNESS [{fw.get('mode', 'n/a')}]: since {fw.get('window_start_et', 'n/a')} ET "
        f"({fw.get('window_start_utc', 'n/a')} UTC) "
        f"| reddit r/{rd['subreddit']}: {rd['n_posts']} fresh / {rd.get('n_fetched', '?')} fetched "
        f"({rd.get('n_stale_filtered', 0)} stale, {rd.get('n_filtered', 0)} spam/off-topic dropped) "
        f"| x: {xs['status']} ({xs['n_posts']} fresh, {xs.get('n_stale_filtered', 0)} stale, "
        f"{xs.get('n_filtered', 0)} spam/off-topic dropped)")
    lines += ["", "TOP MENTIONED (ticker | mentions | hype | sentiment | momentum):"]
    for t in report["top_tickers"]:
        lines.append(f"  {t['ticker']:<6} {t['mentions']:>4}  hype={t['hype']:.3f}  "
                     f"sent={t['sentiment']:+.2f}  mom={t['momentum']:.2f}")
    c = report["candidate"]
    lines.append("")
    if c:
        lines.append(f"CANDIDATE IDEA: {c['ticker']} — {c['direction'].upper()} social pressure "
                     f"(mentions={c['mentions']}, sentiment={c['sentiment']:+.2f}, momentum={c['momentum']:.2f})")
        for e in c["evidence"]:
            _age = f"{e['age_hours']}h ago" if e.get("age_hours") is not None else "time?"
            lines.append(f"   • [{e['source']}/{_age}] [{e['score']}] {e['title']}  {e['url']}")
    else:
        lines.append("CANDIDATE IDEA: none (insufficient mentions / no decisive sentiment) — observe only.")
    # PAPER-ONLY 0DTE option idea
    po = report.get("paper_options", {})
    lines.append("")
    lines.append(f"PAPER 0DTE OPTIONS ({po.get('option_type') or 'n/a'}, expiry "
                 f"{po.get('expiry') or 'n/a'}): {po.get('status', 'n/a')}")
    for k in po.get("contracts", []):
        lines.append(f"   • {k['option_type'].upper()} ${k['strike']:g}  "
                     f"bid/ask {k['bid']}/{k['ask']} last {k['last']}  "
                     f"~${k['premium_cost_estimate']:.0f}/contract  "
                     f"spread={('n/a' if k['spread_pct'] is None else format(k['spread_pct'], '.0%'))}  "
                     f"vol={k['volume']} oi={k['open_interest']}")
    if not po.get("contracts"):
        lines.append(f"   • no viable ${report['budget_dollars']:.0f}/same-day contract surfaced "
                     "(closed market / budget / liquidity)")
    lines.append("")
    lines.append("RISK NOTES:")
    for r in report["risk_notes"]:
        lines.append(f"  - {r}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared enrichment — normalize social items into the data/news.py article shape
# and merge them into the cached "news" substrate so the EXISTING data.sentiment
# context (PortfolioManager active-sleeve sentiment) can consume richer context.
# The 0DTE report above is one consumer; this is the substrate-merge path.
# ---------------------------------------------------------------------------

def _tickers_in(text: str, allowed: set[str] | None = None) -> list[str]:
    """Per-text ticker symbols (stopwords dropped; restricted to `allowed` if given)."""
    out: list[str] = []
    for m in _TICKER_RE.findall(text or ""):
        if m in _STOPWORDS:
            continue
        if allowed is not None and m not in allowed:
            continue
        out.append(m)
    return sorted(set(out))


def _norm_item(*, title, summary, link, author, ts, source, engagement,
               allowed: set[str] | None = None) -> dict:
    """Build ONE normalized item in the data/news.py article-dict shape so social items merge
    into the "news" dataset and render in data.sentiment._format_news exactly like a news
    article. We intentionally do NOT attach a precomputed bullish/bearish/net sentiment to the
    item: the active-sleeve LLM must judge social and news uniformly from the raw text, not from
    a separate social score (the 0DTE report keeps its own transparent heuristics independently).
    Only factual provenance is additive — author + raw engagement counts."""
    text = f"{title} {summary}"
    try:
        iso = (datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
               if ts else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        fdate = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").strftime("%m-%d-%Y")
    except Exception:
        fdate = datetime.now(timezone.utc).strftime("%m-%d-%Y")
    return {
        # --- data/news.py article schema (consumed by data.sentiment._format_news) ---
        "title": (title or "")[:300],
        "publisher": source,
        "link": link or "",
        "summary": (summary or "")[:500],
        "pub_date": iso,
        "formatted_date": fdate,
        "related_symbols": _tickers_in(text, allowed),
        "api_source": source,                # 'reddit_wsb' | 'x'
        # --- additive factual provenance (counts/handles, NOT a sentiment verdict) ---
        "author": author,
        "engagement": engagement,
    }


def normalize_social_items(reddit_posts: list[dict] | None = None,
                           x_posts: list[dict] | None = None,
                           allowed: set[str] | None = None) -> list[dict]:
    """Normalize raw Reddit/X posts into the shared news article-dict shape."""
    items: list[dict] = []
    for q in (reddit_posts or []):
        items.append(_norm_item(
            title=q.get("title", ""), summary=q.get("selftext", ""),
            link=q.get("permalink", ""), author=q.get("author"),
            ts=q.get("created_utc", 0.0), source="reddit_wsb",
            engagement={"score": int(q.get("score", 0) or 0),
                        "num_comments": int(q.get("num_comments", 0) or 0)},
            allowed=allowed,
        ))
    for t in (x_posts or []):
        items.append(_norm_item(
            title=(t.get("text", "") or "")[:120], summary=t.get("text", ""),
            link=f"https://twitter.com/i/web/status/{t.get('id', '')}",
            author=t.get("author_id"),
            ts=_parse_x_ts(t.get("created_at", "")),  # real tweet time, not 'now'
            source="x",
            engagement={"id": t.get("id", "")}, allowed=allowed,
        ))
    return items


def social_items_by_symbol(items: list[dict]) -> dict[str, list[dict]]:
    """Group normalized social items by each related symbol they mention."""
    out: dict[str, list[dict]] = {}
    for it in items:
        for sym in it.get("related_symbols", []):
            out.setdefault(sym, []).append(it)
    return out


def merge_social_into_news(news_df, social_by_symbol: dict[str, list[dict]]):
    """Append normalized social items into the cached "news" frame, PRESERVING its
    ["symbol","news"] schema (news = JSON list of article dicts). Existing news is kept;
    symbols absent from the frame get a new row. Returns a NEW DataFrame."""
    rows: dict[str, list[dict]] = {}
    if (news_df is not None and not getattr(news_df, "empty", True)
            and {"symbol", "news"} <= set(news_df.columns)):
        for _, r in news_df.iterrows():
            try:
                rows[str(r["symbol"])] = json.loads(r["news"]) if r["news"] else []
            except Exception:
                rows[str(r["symbol"])] = []
    for sym, items in social_by_symbol.items():
        rows.setdefault(sym, [])
        rows[sym].extend(items)
    return pd.DataFrame([{"symbol": s, "news": json.dumps(a)} for s, a in rows.items()])


def enrich_news_with_social(news_df=None, allow_fetch: bool = True, params: dict | None = None,
                            persist: bool = False):
    """Fetch social (reddit/x per config) → normalize → merge into the "news" substrate.
    Returns the merged frame. persist=False by default (does NOT mutate the production news
    cache unless explicitly asked). This is the seam PortfolioManager sentiment can use later;
    run-auto is NOT changed."""
    from data.cache import read_data_as_pd, store_data_as_csv
    from util import OPTIONS_SOCIAL_PARAMS
    p = params if params is not None else OPTIONS_SOCIAL_PARAMS
    if news_df is None:
        news_df = read_data_as_pd("news")

    # Allowed-universe so enrichment NEVER injects false-positive all-caps symbols into the
    # active-sleeve sentiment substrate: core_universe + symbols already in the news frame +
    # cached agg_data symbols. None only if all are empty (then stopword-only filtering).
    allowed: set[str] = {str(s).upper() for s in p.get("core_universe", ["SPY", "QQQ"])}
    if (news_df is not None and not getattr(news_df, "empty", True)
            and "symbol" in getattr(news_df, "columns", [])):
        allowed |= {str(s).upper() for s in news_df["symbol"].dropna()}
    try:
        _agg = read_data_as_pd("agg_data")
        if _agg is not None and "symbol" in getattr(_agg, "columns", []):
            allowed |= {str(s).upper() for s in _agg["symbol"].dropna()}
    except Exception:
        pass
    allowed_set: set[str] | None = allowed or None

    sources = p.get("sources", ["reddit"])
    reddit = (fetch_reddit_posts(subreddit=p.get("subreddit", "wallstreetbets"),
                                 limit=int(p.get("reddit_limit", 50)), allow_fetch=allow_fetch)
              if "reddit" in sources else [])
    # Optional, bounded, opt-in comments enrichment: fold top comments of the top posts into
    # their selftext so the LLM (and ticker extraction) sees the discussion, not just the title.
    if reddit and bool(p.get("reddit_comments_enrich", False)):
        reddit = _fold_comments_into_posts(
            reddit, int(p.get("reddit_comments_top_posts", 3)),
            int(p.get("reddit_comments_per_post", 5)), allow_fetch)
    x_posts: list[dict] = []
    if "x" in sources:
        x_posts, _ = fetch_x_mentions(
            query=p.get("x_query", "(SPY OR QQQ OR 0DTE) lang:en -is:retweet"),
            limit=int(p.get("x_limit", 50)))
    items = normalize_social_items(reddit, x_posts, allowed=allowed_set)
    # Conservative quality pass for the active-sleeve substrate: drop promo/scam/off-topic spam
    # and near-duplicates so the sentiment LLM never reads Telegram/VIP/100X/crypto/legal blasts —
    # but DO NOT require options context here (news enrichment is broader than 0DTE) and preserve
    # every surviving item's source/provenance untouched.
    items = _quality_filter(items, lambda it: f"{it.get('title', '')} {it.get('summary', '')}",
                            require_options_context=False)
    merged = merge_social_into_news(news_df, social_items_by_symbol(items))
    if persist:
        store_data_as_csv("news", ["symbol", "news"], merged)
    return merged
