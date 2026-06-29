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

import html
import json
import logging
import math
import os
import re
import time
from collections import Counter
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests

from core.paths import DATA_DIR, ODTE_SCRAPE_DIR

logger = logging.getLogger(__name__)

_UA = "daily-investor-research/1.0 (social-sentiment analysis; non-commercial)"
_CACHE_DIR = DATA_DIR / "social_cache"
_REDDIT_URL = "https://www.reddit.com/r/{sub}/{listing}.json"
_REDDIT_RSS_URL = "https://www.reddit.com/r/{sub}/.rss"
_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_REDDIT_OAUTH_URL = "https://oauth.reddit.com/r/{sub}/{listing}"
_REDDIT_COMMENTS_JSON_URL = "https://www.reddit.com/comments/{id}.json"
_REDDIT_OAUTH_COMMENTS_URL = "https://oauth.reddit.com/comments/{id}"
_REDDIT_MORECHILDREN_URL = "https://www.reddit.com/api/morechildren.json"
_REDDIT_OAUTH_MORECHILDREN_URL = "https://oauth.reddit.com/api/morechildren"
_REDDIT_SEARCH_JSON_URL = "https://www.reddit.com/r/{sub}/search.json"
_REDDIT_OAUTH_SEARCH_URL = "https://oauth.reddit.com/r/{sub}/search"
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
    # 'V' in 0DTE chatter is the V-shaped-recovery / 'V bounce' pattern, NOT Visa — exclude it so
    # "we get a V here" doesn't surface as a Visa mention (the bare-letter false positive the user hit):
    "V",
    # 0DTE/flow jargon that looks like a ticker but isn't (kept out of mention ranking):
    "FOMC", "OPEX", "GEX", "OI", "ALERT", "WHALE", "PR", "DTE", "ODTE", "SPX", "VIX",
    # TA-indicator abbreviations — in 0DTE chatter these mean the indicator, not the (minor) ticker:
    "RSI", "MACD", "EMA", "SMA",
})
_BULL = ("call", "calls", "moon", "mooning", "moonshot", "rocket", "buy", "long", "bull",
         "bullish", "pump", "squeeze", "breakout", "rip", "ripping", "tendies", "printing",
         "diamond hands", "hold the line", "send it", "🚀", "💎")
_BEAR = ("put", "puts", "short", "sell", "bear", "bearish", "crash", "dump", "drill",
         "tank", "tanking", "tanked", "collapse", "puke", "cooked", "toast", "rug", "rugged",
         "rug pull", "bagholder", "bagholders", "bagholding", "exit liquidity", "dead cat",
         "capitulation", "guh")


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


def _comment_dict(d: dict) -> dict | None:
    """One top-level comment's fields, or None for an empty body. Captures the vote signals
    upvote-scoring needs — ``score`` (net karma ups−downs), ``score_hidden`` (Reddit's initial
    vote-fuzzing window), ``controversiality`` (0/1, near-tied votes) — plus ``created_utc`` for
    recency weighting (newer comments weigh more)."""
    body = (d.get("body") or "").strip()
    if not body:
        return None
    return {"body": body[:500], "score": int(d.get("score", 0) or 0),
            "score_hidden": bool(d.get("score_hidden", False)),
            "controversiality": int(d.get("controversiality", 0) or 0),
            "created_utc": float(d.get("created_utc", 0.0) or 0.0),
            "author": d.get("author")}


def _parse_reddit_comment_listing(payload, limit: int) -> tuple[list[dict], list[str]]:
    """Parse Reddit's ``/comments/{id}`` JSON ([post_listing, comments_listing]) into
    (top-level comment dicts, remaining 'more'-child ids). The 'more' ids let the caller page the
    rest of a big thread (_expand_more_comments); deeper-reply nodes are skipped."""
    try:
        children = payload[1]["data"]["children"]
    except Exception:
        return [], []
    out: list[dict] = []
    more_ids: list[str] = []
    for c in children:
        if not isinstance(c, dict):
            continue
        if c.get("kind") == "more":
            more_ids.extend((c.get("data", {}) or {}).get("children", []) or [])
            continue
        if c.get("kind") != "t1":
            continue
        cd = _comment_dict(c.get("data", {}) or {})
        if cd and len(out) < limit:
            out.append(cd)
    return out, more_ids


def _reddit_comment_auth(bearer_token: str | None) -> tuple[bool, dict]:
    """(use_oauth, headers) for the comment endpoints, honoring the auth order app OAuth → explicit
    caller bearer → anonymous public. Headers may carry the token but are NEVER logged."""
    ua = _reddit_user_agent()
    token = _reddit_oauth_token()
    if token:                                  # (A) app-only OAuth
        return True, {"Authorization": f"bearer {token}", "User-Agent": ua}
    if bearer_token:                           # (B) explicit caller-supplied bearer (never logged)
        return True, {"Authorization": f"Bearer {bearer_token}", "User-Agent": ua}
    return False, {"User-Agent": ua}           # (C) anonymous public JSON


def _expand_more_comments(post_id: str, out: list[dict], more_ids: list[str], limit: int,
                          use_oauth: bool, headers: dict, max_pages: int = 120) -> list[dict]:
    """Follow Reddit's ``more``-children links (the morechildren API, 100 ids/request) to pull the
    WHOLE thread's top-level comments — so the caller never manages pagination. Bounded by ``limit``
    and ``max_pages`` (a runaway safety net) and FAIL-CLOSED: any error stops paging and keeps what
    was already collected. Extends and returns ``out`` (<= ``limit``)."""
    url = _REDDIT_OAUTH_MORECHILDREN_URL if use_oauth else _REDDIT_MORECHILDREN_URL
    ids = list(more_ids)
    pages = 0
    while ids and len(out) < limit and pages < max_pages:
        batch, ids = ids[:100], ids[100:]
        pages += 1
        try:
            r = requests.get(url, headers=headers, params={
                "api_type": "json", "link_id": f"t3_{post_id}", "children": ",".join(batch),
                "sort": "new", "raw_json": 1}, timeout=15)
            r.raise_for_status()
            things = (((r.json() or {}).get("json") or {}).get("data") or {}).get("things") or []
        except Exception as exc:
            logger.warning("reddit morechildren fetch failed for %s (%s)", post_id, exc)  # never logs token
            break   # fail-closed: keep the comments already collected
        for c in things:
            if not isinstance(c, dict):
                continue
            if c.get("kind") == "more":
                ids.extend((c.get("data", {}) or {}).get("children", []) or [])
            elif c.get("kind") == "t1" and len(out) < limit:
                cd = _comment_dict(c.get("data", {}) or {})
                if cd:
                    out.append(cd)
    if ids and len(out) >= limit:
        logger.info("reddit comments for %s capped at %d (%d more unfetched)", post_id, limit, len(ids))
    return out[:limit]


def fetch_reddit_comments(post_id: str, limit: int = 5, ttl_s: float = 900,
                          allow_fetch: bool = True, bearer_token: str | None = None) -> list[dict]:
    """Top-level comments for a post, cache-backed and fail-closed. AUTO-PAGINATES Reddit's
    'more comments' links so it returns the WHOLE thread's top-level comments up to ``limit`` (the
    caller never manages paging). Auth order: (A) app-only OAuth when REDDIT_CLIENT_ID/SECRET are
    set; (B) an explicit, caller-supplied ``bearer_token`` (read-only, ephemeral — passed as an
    ARGUMENT, never read from cookies/.env, never logged); (C) anonymous public JSON. Returns
    ``[{body, score, score_hidden, controversiality, created_utc, author}]`` (<= ``limit``). The
    token is used ONLY as an ``Authorization`` header against oauth.reddit.com and is never cached,
    returned, or logged."""
    if not post_id:
        return []
    key = f"reddit_comments_{post_id}_{limit}"
    cached = _read_cache(key, ttl_s)
    if cached is not None:
        return cached
    if not allow_fetch:
        return []
    use_oauth, headers = _reddit_comment_auth(bearer_token)
    base = _REDDIT_OAUTH_COMMENTS_URL if use_oauth else _REDDIT_COMMENTS_JSON_URL
    # Reddit caps a single comments listing near ~500; request that, then page the rest via 'more'.
    params = {"limit": min(limit, 500), "depth": 1, "sort": "new", "raw_json": 1}
    try:
        r = requests.get(base.format(id=post_id), headers=headers, params=params, timeout=15)
        r.raise_for_status()
        out, more_ids = _parse_reddit_comment_listing(r.json() or [], limit)
        out = _expand_more_comments(post_id, out, more_ids, limit, use_oauth, headers)
        _write_cache(key, out)
        return out
    except Exception as exc:
        logger.warning("reddit comments fetch failed for %s (%s)", post_id, exc)  # never logs token
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

# A ticker is either a cashtag ($spy, ANY case → normalized) or a BARE all-caps token (SPY). A
# plain lowercase word ("spy", "put") is intentionally NOT a ticker — it needs the $ or all-caps.
_TICKER_RE = re.compile(r"\$([A-Za-z]{1,5})\b|\b([A-Z]{1,5})\b")


def _iter_symbols(text: str):
    """Yield normalized (uppercase) ticker symbols from `text`: lowercase cashtags are recognized
    and normalized; plain lowercase words are not."""
    for m in _TICKER_RE.finditer(text or ""):
        sym = (m.group(1) or m.group(2) or "").upper()
        if sym:
            yield sym


def extract_ticker_mentions(texts, allowed: set[str] | None = None) -> Counter:
    """Count $ticker / TICKER tokens (cashtags case-insensitive, normalized to uppercase), dropping
    stopwords. If `allowed` is given, restrict to it. Plain lowercase words are not tickers."""
    cnt: Counter = Counter()
    for t in texts:
        for sym in _iter_symbols(t):
            if sym in _STOPWORDS:
                continue
            if allowed is not None and sym not in allowed:
                continue
            cnt[sym] += 1
    return cnt


def engagement_weight(score) -> float:
    """Bounded community-endorsement weight for one doc, from its net Reddit score (ups − downs):
    ``1 + log10(1 + max(score, 0))``. A score of 0 (X posts, brand-new comments) weighs 1.0; +9 → 2.0;
    +99 → 3.0; a viral +5000 → ~4.7. The log damping lets a strongly-upvoted comment lift a ticker's
    rank / directional lean WITHOUT one mega-upvoted post swamping a dozen ordinary ones. Net-negative
    comments are dropped upstream (_is_downvoted), so the max(score,0) floor only ever clamps to 0."""
    return 1.0 + math.log10(1.0 + max(float(score or 0), 0.0))


_RECENCY_HALF_LIFE_S = 7200.0   # 2h: a 0DTE doc loses half its weight every 2 hours of age


def _recency_weight(ts, ref_ts, half_life_s: float = _RECENCY_HALF_LIFE_S, floor: float = 0.1) -> float:
    """Recency multiplier in (floor, 1]: 1.0 for the newest doc, halving every ``half_life_s`` of age
    measured against the most-recent doc (``ref_ts``). In a 0DTE session a comment from an hour ago is
    worth more than one from the open, so newer chatter steers the read. Floored so old-but-relevant
    chatter still counts a little; missing/zero ts (or no ref) → 1.0 (neutral)."""
    if not ts or not ref_ts or ts >= ref_ts:
        return 1.0
    return max(floor, 0.5 ** ((ref_ts - ts) / half_life_s))


def _doc_weight(score, ts, ref_ts) -> float:
    """Combined per-doc weight = community endorsement (upvotes) × recency (newer = heavier)."""
    return engagement_weight(score) * _recency_weight(ts, ref_ts)


def score_social(mentions: Counter, documents: list[dict]) -> dict:
    """RAW-CHATTER ranking heuristic (coarse keyword tally) — per-ticker transparent scores from
    `documents` (each {text, ts, weight}), where ``weight`` is the doc's net Reddit score:
      hype      = share of total mentions (0..1)               [raw mention share — honest count]
      sentiment = (bull-bear)/(bull+bear) over docs mentioning the ticker (-1..1), with each doc's
                  bull/bear keyword hits weighted by _doc_weight (upvotes × recency) so upvoted AND
                  recent comments drive the lean while ignored or stale ones barely register
      momentum  = share of the ticker's mention-docs newer than the median doc timestamp (0..1)
    bull/bear are reported as RAW keyword counts (display honesty); only `sentiment` is weighted.

    NOTE: this is intentionally SEPARATE from the SPY decision scorecard. It drives only the
    'CROWD CHATTER' mention ranking and the candidate's option-chain direction. The scorecard's
    social read uses the context-aware classify_odte_intent / summarize_odte_intent instead.
    """
    total = sum(mentions.values()) or 1
    ts_all = sorted(d.get("ts", 0.0) for d in documents)
    median_ts = ts_all[len(ts_all) // 2] if ts_all else 0.0
    ref_ts = ts_all[-1] if ts_all else 0.0    # newest doc — recency anchor for _doc_weight
    out: dict = {}
    for tk, n in mentions.items():
        tkl = tk.lower()
        pat = re.compile(rf"\$?\b{re.escape(tkl)}\b", re.IGNORECASE)
        bull = bear = recent = hits = 0       # raw keyword / doc counts (display)
        bull_w = bear_w = 0.0                  # upvote × recency-weighted keyword strength (drives sentiment)
        for d in documents:
            txt = (d.get("text", "") or "")
            if not pat.search(txt):
                continue
            hits += 1
            low = txt.lower()
            w = _doc_weight(d.get("weight", 0), d.get("ts", 0.0), ref_ts)
            nb = sum(low.count(x) for x in _BULL)
            nk = sum(low.count(x) for x in _BEAR)
            bull += nb
            bear += nk
            bull_w += w * nb
            bear_w += w * nk
            if d.get("ts", 0.0) >= median_ts:
                recent += 1
        denom_w = (bull_w + bear_w) or 1.0
        out[tk] = {
            "mentions": int(n),
            "hype": round(n / total, 4),
            "bull": int(bull),
            "bear": int(bear),
            "sentiment": round((bull_w - bear_w) / denom_w, 3),
            "momentum": round(recent / hits, 3) if hits else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Contextual ODTE intent classifier (transparent, no ML). Unlike score_social()'s
# whole-document keyword tally, this reads phrase/context windows around the ticker and
# options terms, CANCELS negated cues ("do not chase calls" -> not bullish), and INVERTS
# option-outcome phrases ("puts got smoked" -> bullish; "calls printing" -> bullish; "puts
# printing" -> bearish). Questions / risk-warnings need a wider margin or resolve to neutral;
# ties resolve to neutral (conflict). It exposes transparent bull/bear counts + example spans.
# ---------------------------------------------------------------------------

_OPT_CALL = frozenset({"call", "calls"})
_OPT_PUT = frozenset({"put", "puts"})
# Option-outcome verbs: a GAIN on calls (or LOSS on puts) is bullish for the underlying, and
# vice-versa. "printing"/"smoked" are the canonical WSB gain/loss verbs.
_GAIN_VERBS = frozenset({"print", "prints", "printing", "printed", "ripping", "ripped",
                         "mooning", "mooned", "squeezing", "green", "won", "winning", "banger",
                         "tendies", "hit", "hits", "loaded", "loading"})
_LOSS_VERBS = frozenset({"smoked", "cooked", "crushed", "rekt", "wrecked", "destroyed",
                         "worthless", "expired", "assigned", "tanked", "dead", "red", "losing",
                         "lost", "bust", "busted", "obliterated", "gutted", "toast", "rugged",
                         "rug", "bagholder", "bagholders", "bagholding", "trapped"})
# Bare directional intent words (emojis handled separately; option words handled by the grammar).
_BULL_WORDS = frozenset({"call", "calls", "moon", "rocket", "buy", "buying", "bought", "long",
                         "bull", "bullish", "pump", "squeeze", "breakout", "rip", "bullrun",
                         "upside", "mooning", "moonshot", "tendies"})
_BEAR_WORDS = frozenset({"put", "puts", "short", "shorting", "shorted", "sell", "bear", "bearish",
                         "crash", "dump", "drill", "tank", "tanking", "tanked", "collapse", "puke",
                         "downside", "cooked", "toast", "rug", "rugged", "bagholder", "bagholders",
                         "bagholding", "guh"})
# Negators CANCEL a directional cue (we do not flip to the opposite — safer + transparent).
_NEGATORS = frozenset({"not", "no", "never", "none", "without", "avoid", "stop", "dont", "don't",
                       "isnt", "isn't", "aint", "ain't", "wont", "won't", "cant", "can't", "quit",
                       "fade", "against", "anti"})
_BULL_PHRASES = (("buy", "the", "dip"), ("to", "the", "moon"), ("load", "up"), ("send", "it"),
                 ("bottom", "is", "in"), ("diamond", "hands"), ("hold", "the", "line"),
                 ("loaded", "calls"), ("calls", "loaded"), ("calls", "printing"))
_BEAR_PHRASES = (("go", "to", "zero"), ("going", "to", "zero"), ("bag", "holder"),
                 ("dead", "cat"), ("catching", "knives"), ("falling", "knife"),
                 ("top", "is", "in"), ("exit", "liquidity"), ("rug", "pull"),
                 ("paper", "hands"), ("iv", "crush"), ("puts", "printing"))
_QUESTION_RISK_RE = re.compile(
    r"\?|\bnfa\b|not financial advice|do your own|\bdyor\b|should i\b|is it too late|"
    r"am i (cooked|screwed|fucked)|thoughts\b", re.IGNORECASE)
_CTX_WINDOW = 8       # tokens around a ticker/options anchor to attribute a directional cue
_NEG_WINDOW = 3       # tokens before a cue scanned for a negator
_SHORT_TOKENS = 40    # texts at/under this token length count cues regardless of proximity
_RISK_MARGIN = 3      # questions/risk-warnings need this net margin to stay directional
_ANCHOR_TERMS = frozenset({"0dte", "odte", "strike", "strikes", "contract", "contracts",
                           "option", "options", "spx", "expiry", "expiration", "yolo", "fd", "fds",
                           "lotto", "lottos", "scalp", "scalps", "premium", "theta", "gamma"})
# A bare SINGULAR 'put'/'call' is an option noun (not an English verb) only with this context:
# an options term / a number / a $strike nearby, or a determiner/transaction word right before it.
_OPT_DETERMINERS = frozenset({"a", "an", "the", "my", "this", "that", "some", "sell", "sold",
                              "buy", "buying", "bought", "long", "short", "grab", "grabbed",
                              "load", "loaded", "hold", "holding", "weekly", "weeklies",
                              "atm", "otm", "itm", "leap", "leaps", "scalp"})
# Cover/exit words: a bear cue right after one is EXITING a bearish position (bullish transition),
# so it must not be counted bearish ("closed my short", "out of puts").
_COVER = frozenset({"closed", "closing", "close", "covered", "cover", "covering",
                    "exit", "exited", "exiting", "out"})
# Market-structure / dealer-flow jargon: context, not retail directional intent.
_FLOW_RE = re.compile(r"\b(call\s+wall|put\s+wall|gamma|gex|max\s+pain|dealers?|vanna|charm|"
                      r"gravity|open\s+interest|gamma\s+exposure|0\s*gamma|charm\s+flow)\b",
                      re.IGNORECASE)
_STRUCT_NEXT = frozenset({"wall", "walls", "gravity"})  # 'call wall' / 'put gravity' = structure
# DISPOSITION: closing/selling an option is NOT a fresh directional bet — 'sold calls' isn't
# bullish, 'sold puts' isn't bearish. These cancel the call/put vote that follows them.
_DISPOSED = frozenset({"sold", "selling", "sell", "sells", "dumping", "unloaded", "unload",
                       "closed", "closing", "close", "exited", "exiting", "trimmed", "trimming"})
# PARTY outcomes: 'bears cooked' = bullish, 'bulls trapped' = bearish (who's winning/losing).
_PARTY_BULL = frozenset({"bull", "bulls"})
_PARTY_BEAR = frozenset({"bear", "bears"})
_PARTY_LOSS = _LOSS_VERBS | frozenset({"trapped", "wrong", "done", "finished", "fucked", "screwed"})
_PARTY_GAIN = _GAIN_VERBS | frozenset({"right", "feasting", "eating", "correct", "winning"})
# JINX / sarcasm: a bullish position followed (via a connector) by a bearish DIRECTION word is the
# WSB 'my calls jinx it down' trope → bearish; mirror for puts → bullish.
_BEAR_DIR = frozenset({"drop", "drops", "dropping", "dropped", "tank", "tanks", "tanking", "tanked",
                       "dump", "dumps", "dumping", "fade", "fades", "fading", "faded", "crash",
                       "crashes", "crashing", "red", "lower", "fall", "falls", "falling", "plummet",
                       "tumble", "tumbling"})
_BULL_DIR = frozenset({"pump", "pumps", "pumping", "pumped", "rip", "rips", "ripping", "moon",
                       "mooning", "green", "higher", "rally", "rallies", "rallying", "melt",
                       "melting", "breakout", "ripped"})
_JINX_CONNECTOR = frozenset({"so", "and", "but", "watch", "now", "gonna", "cause", "because",
                             "then", "til", "until", "till", "know", "knew", "everytime",
                             "anytime", "whenever"})
# Validated company-name → ticker aliases (small, hand-checked; NOT broad/hallucinated).
_TICKER_ALIASES = {"microsoft": "MSFT", "micron": "MU", "tesla": "TSLA", "nvidia": "NVDA",
                   "amazon": "AMZN", "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL",
                   "meta": "META", "netflix": "NFLX", "palantir": "PLTR", "broadcom": "AVGO"}
_ALIAS_BY_TICKER: dict[str, list[str]] = {}
for _name, _tk in _TICKER_ALIASES.items():
    _ALIAS_BY_TICKER.setdefault(_tk, []).append(_name)


def _ticker_doc_re(ticker: str) -> re.Pattern:
    """Regex matching a ticker symbol OR a validated company-name alias (Microsoft→MSFT)."""
    parts = [re.escape(ticker)] + [re.escape(n) for n in _ALIAS_BY_TICKER.get(ticker.upper(), [])]
    return re.compile(r"\$?\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


def _intent_tokens(text: str) -> list[str]:
    return re.findall(r"\$?[a-z0-9']+", (text or "").lower())


def _opt_noun_ctx(toks: list[str], i: int) -> bool:
    """True when a bare singular 'put'/'call' at index i is in clear options-NOUN context (option
    term / number / $strike within ±3, or a determiner/transaction word immediately before)."""
    for j in range(max(0, i - 3), min(len(toks), i + 4)):
        if j == i:
            continue
        tj = toks[j]
        if tj in _ANCHOR_TERMS or tj.isdigit() or (tj.startswith("$") and len(tj) > 1):
            return True
    return i > 0 and toks[i - 1] in _OPT_DETERMINERS


_JINX_BULL_POS = frozenset({"calls", "call", "buy", "buying", "bought", "long"})
_JINX_BEAR_POS = frozenset({"puts", "put", "short", "shorting"})
_CTX_WINDOW_MULTI = 5   # tighter window when >1 ticker in a comment (scope cues to the target)
# Curated UNAMBIGUOUS tickers people often write lowercase (not English words) — used only to
# detect a SECOND ticker in a comment so we can scope cues to the target ticker.
_LOWERCASE_TICKERS = frozenset({"spy", "qqq", "iwm", "dia", "spx", "ndx", "vix", "tsla", "nvda",
                                "msft", "amzn", "aapl", "googl", "meta", "nflx", "amd", "soxl",
                                "soxs", "tqqq", "sqqq", "pltr", "avgo", "smci", "mstr"})


def _other_tickers(text: str, toks: list[str], tkl: str) -> set[str]:
    """Distinct OTHER ticker mentions in the text (uppercase/cashtag, plus curated lowercase index
    tickers) — used to decide whether to scope directional cues to the target ticker."""
    out = {s.lower() for s in _iter_symbols(text) if s not in _STOPWORDS and s.lower() != tkl}
    out |= {t for t in toks if t in _LOWERCASE_TICKERS and t != tkl}
    return out


def _jinx_direction(toks: list[str]) -> str | None:
    """WSB jinx/sarcasm: a bullish position token followed (within a window, via a connector word)
    by a bearish DIRECTION word ('bought calls so it keeps dropping') is anti-bullish → 'bearish';
    a bearish position followed by a bullish direction → 'bullish'. Returns None if no jinx."""
    def _hit(positions: list[int], dirset: frozenset) -> bool:
        for p in positions:
            for j in range(p + 1, min(len(toks), p + 13)):
                if toks[j] in dirset and any(
                        toks[k] in _JINX_CONNECTOR for k in range(p + 1, j + 1)):
                    return True
        return False
    bull_pos = [i for i, t in enumerate(toks) if t in _JINX_BULL_POS]
    bear_pos = [i for i, t in enumerate(toks) if t in _JINX_BEAR_POS]
    if _hit(bull_pos, _BEAR_DIR):
        return "bearish"
    if _hit(bear_pos, _BULL_DIR):
        return "bullish"
    return None


def classify_odte_intent(text: str, ticker: str = "SPY") -> dict:
    """Classify the directional ODTE intent of one social text toward `ticker`.

    Returns ``{"intent": "bullish"|"bearish"|"neutral", "bull": int, "bear": int,
    "examples": [...], "flags": [...]}``. Reads context windows around the ticker / options
    terms, cancels negated cues, inverts option-outcome phrases, and resolves questions /
    risk-warnings / ties to neutral. Transparent and inspectable — NOT financial advice."""
    low = (text or "").lower()
    toks = _intent_tokens(low)
    flags: list[str] = []
    examples: list[str] = []
    if not toks:
        return {"intent": "neutral", "bull": 0, "bear": 0, "examples": [],
                "bull_examples": [], "bear_examples": [], "flags": ["empty"]}
    tkl = ticker.lower()
    target_set = {tkl, "$" + tkl} | set(_ALIAS_BY_TICKER.get(ticker.upper(), []))   # MSFT + 'microsoft'
    anchors = [i for i, t in enumerate(toks)
               if t in target_set or t in _OPT_CALL or t in _OPT_PUT or t in _ANCHOR_TERMS]
    short = len(toks) <= _SHORT_TOKENS
    # Multi-ticker scoping: when the comment names ANOTHER ticker, attribute each cue to its NEAREST
    # ticker — so 'qqq calls ... TXRH' counts the calls for QQQ, not TXRH.
    tk_positions = [i for i, t in enumerate(toks) if t in target_set]
    others = _other_tickers(text, toks, tkl)
    other_positions = [i for i, t in enumerate(toks) if t.lstrip("$") in others]
    multi = bool(tk_positions) and bool(other_positions)

    def near(i: int) -> bool:
        if multi:
            dt = min((abs(i - a) for a in tk_positions), default=99)
            do = min((abs(i - a) for a in other_positions), default=99)
            return dt <= do and dt <= _CTX_WINDOW   # target is the nearest ticker (ties to target)
        return short or any(abs(i - a) <= _CTX_WINDOW for a in anchors)

    def negated(i: int) -> bool:
        return any(toks[j] in _NEGATORS for j in range(max(0, i - _NEG_WINDOW), i))

    bull = bear = 0
    consumed: set[int] = set()
    bull_ex: list[str] = []   # spans that VOTED bullish (intent-supporting, never negated)
    bear_ex: list[str] = []   # spans that VOTED bearish

    # 0) JINX / sarcasm override: a bullish position + a later bearish DIRECTION word joined by a
    # connector ('bought calls so it keeps dropping') is anti-bullish → bearish; mirror for puts.
    jinx = _jinx_direction(toks)
    if jinx == "bearish":
        bear += 3
        bear_ex.append("position jinx (calls→down)")
        flags.append("jinx")
    elif jinx == "bullish":
        bull += 3
        bull_ex.append("position jinx (puts→up)")
        flags.append("jinx")

    # 1) Option-outcome grammar: call/put × gain/loss within a tight window → inverse logic.
    # A negator near the verb FLIPS the outcome ('puts won't print' = puts won't gain = bullish).
    for i, t in enumerate(toks):
        if i in consumed or not (t in _OPT_CALL or t in _OPT_PUT) or not near(i):
            continue
        verb = vpos = None
        for j in range(max(0, i - 3), min(len(toks), i + 4)):
            if j == i:
                continue
            if toks[j] in _GAIN_VERBS:
                verb, vpos = "gain", j
                break
            if toks[j] in _LOSS_VERBS:
                verb, vpos = "loss", j
                break
        if not verb:
            continue
        is_call = t in _OPT_CALL
        flip = any(toks[k] in _NEGATORS
                   for k in range(max(0, min(i, vpos) - _NEG_WINDOW), max(i, vpos)))
        outcome_bull = (is_call and verb == "gain") or (not is_call and verb == "loss")
        if flip:
            outcome_bull = not outcome_bull
        span = " ".join(toks[max(0, i - 1):min(len(toks), i + 3)])
        if outcome_bull:
            bull += 2
            bull_ex.append(span)
        else:
            bear += 2
            bear_ex.append(span)
        consumed.update({i, vpos})
        examples.append(span)

    # 1b) PARTY outcome: 'bears cooked' = bullish, 'bulls trapped' = bearish (who's winning/losing).
    # Prefer the nearest outcome word; when tied, prefer the word AFTER the party token so
    # "puts printing, bulls cooked" reads as bulls losing, not bulls printing.
    for i, t in enumerate(toks):
        if i in consumed or not (t in _PARTY_BULL or t in _PARTY_BEAR) or not near(i):
            continue
        candidates: list[tuple[int, int, str]] = []
        for j in range(max(0, i - 2), min(len(toks), i + 4)):
            if j == i:
                continue
            if toks[j] in _PARTY_GAIN:
                candidates.append((abs(j - i), 0 if j > i else 1, "gain"))
            elif toks[j] in _PARTY_LOSS:
                candidates.append((abs(j - i), 0 if j > i else 1, "loss"))
        if not candidates:
            continue
        _, _, outcome = min(candidates)
        is_bull_party = t in _PARTY_BULL
        party_bull = (is_bull_party and outcome == "gain") or (not is_bull_party and outcome == "loss")
        span = " ".join(toks[max(0, i - 1):min(len(toks), i + 3)])
        if party_bull:
            bull += 2
            bull_ex.append(span)
        else:
            bear += 2
            bear_ex.append(span)
        consumed.add(i)
        examples.append(span)

    # 2) Directional multi-word phrases (negation cancels; token-consumed to avoid double count).
    for phrases, pol in ((_BULL_PHRASES, "bull"), (_BEAR_PHRASES, "bear")):
        for ph in phrases:
            L = len(ph)
            for i in range(len(toks) - L + 1):
                if any((i + k) in consumed for k in range(L)):
                    continue
                if tuple(toks[i:i + L]) != ph or not near(i):
                    continue
                consumed.update(range(i, i + L))
                span = " ".join(ph)
                if negated(i):
                    examples.append(span + " (negated)")
                    continue
                if pol == "bull":
                    bull += 2
                    bull_ex.append(span)
                else:
                    bear += 2
                    bear_ex.append(span)
                examples.append(span)

    # 3) Bare single-word cues (negation cancels rather than flips).
    for i, t in enumerate(toks):
        if i in consumed:
            continue
        base = "bull" if t in _BULL_WORDS else ("bear" if t in _BEAR_WORDS else None)
        if base is None or not near(i):
            continue
        # Market-structure jargon: 'call wall' / 'put gravity' is not directional intent.
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in (_OPT_CALL | _OPT_PUT) and nxt in _STRUCT_NEXT:
            continue
        # A bare SINGULAR 'put'/'call' is a directional option noun only with options-noun context
        # ('put on someone' = English verb → ignored).
        if t in ("put", "call") and not _opt_noun_ctx(toks, i):
            continue
        # DISPOSITION: 'sold/selling/closed calls|puts' is not a fresh directional bet → cancel.
        # Window of 4 catches 'sold my odte calls' / 'closed the spy puts'.
        if t in (_OPT_CALL | _OPT_PUT) and any(
                toks[j] in _DISPOSED for j in range(max(0, i - 4), i)):
            examples.append(f"disposed {t}")
            continue
        if negated(i):
            examples.append(f"not {t}")
            continue
        # A bear cue right after a cover/exit word is a bullish transition ('closed my short') —
        # do not count it bearish.
        if base == "bear" and any(toks[j] in _COVER for j in range(max(0, i - _NEG_WINDOW), i)):
            continue
        # Bare single words still VOTE, but are too noisy to surface as examples — only the
        # higher-signal option-outcome / multi-word phrase / emoji spans become examples.
        if base == "bull":
            bull += 1
        else:
            bear += 1

    # 4) Emoji cues (bullish; not negatable). Counted, but NOT surfaced as an example (an emoji
    # is not a natural-language snippet).
    bull += low.count("🚀") + low.count("💎")

    risky = bool(_QUESTION_RISK_RE.search(low))
    if risky:
        flags.append("risk_or_question")
    net = bull - bear

    def _result(intent: str, extra_flags: list[str] | None = None) -> dict:
        return {"intent": intent, "bull": bull, "bear": bear,
                "examples": examples,            # ALL spans (incl. negated) — for transparency
                "bull_examples": bull_ex, "bear_examples": bear_ex,  # intent-supporting spans
                "flags": flags + (extra_flags or [])}

    # Market-structure / dealer-flow jargon with no strong retail signal (option-outcome or phrase)
    # is context, not a directional vote → neutral.
    if _FLOW_RE.search(low) and not (bull_ex or bear_ex):
        return _result("neutral", ["flow"])
    # Explicit HEDGE: 'both calls and puts' (or 'calls and puts' together) is non-directional.
    if jinx is None and (" both " in f" {low} ") and "call" in low and "put" in low:
        return _result("neutral", ["hedge"])
    if bull == 0 and bear == 0:
        return _result("neutral", ["no_signal"])
    if risky and abs(net) < _RISK_MARGIN:
        return _result("neutral")
    if net == 0:
        return _result("neutral", ["conflict"])
    return _result("bullish" if net > 0 else "bearish")


# Junk tokens to strip from example snippets: reddit/handle artifacts, url bits, html escapes.
_EXAMPLE_JUNK = frozenset({"u", "r", "amp", "www", "http", "https", "nbsp", "x200b", "gt", "lt"})
# A clean directional snippet must contain at least one of these signal words.
_EXAMPLE_SIGNAL = _OPT_CALL | _OPT_PUT | _GAIN_VERBS | _LOSS_VERBS | frozenset(
    {"moon", "dip", "zero", "bag", "knife", "knives", "bottom", "top", "load", "send"})


def _clean_example(span: str | None) -> str | None:
    """Sanitize an example snippet for the report: drop handle/url/escape tokens and pure numbers,
    require >=2 real words AND a directional signal word; else return None (so we omit rather than
    show a useless fragment like 'nota bull wait' or a bare username)."""
    if not span:
        return None
    words = [w for w in span.split()
             if w not in _EXAMPLE_JUNK and not w.isdigit() and len(w) > 1 and any(ch.isalpha() for ch in w)]
    if len(words) < 2 or not any(w in _EXAMPLE_SIGNAL for w in words):
        return None
    return " ".join(words)


def summarize_odte_intent(ticker: str, ev_pool: list[dict], max_examples: int = 4) -> dict:
    """Aggregate classify_odte_intent over the quality-gated evidence mentioning `ticker`.

    Sums bull/bear votes across docs and resolves a net intent with a small margin (so a single
    keyword can't drive a verdict). Returns SEPARATED concepts: ``n_docs`` (mention/chatter count),
    ``directional_docs`` (docs that carried an actual bull/bear signal), the bull/bear vote totals,
    the resolved ``intent``, and intent-supporting example spans. ``intent`` is
    'bullish'/'bearish'/'neutral'. Matches the ticker symbol OR a validated company-name alias."""
    pat = _ticker_doc_re(ticker)
    bull = bear = 0.0                       # upvote × recency-weighted bull/bear vote totals (drive intent)
    n_docs = n_bull = n_bear = 0           # raw doc counts (honest display / chatter rank)
    ref_ts = max((c.get("ts", 0.0) or 0.0 for c in ev_pool), default=0.0)   # newest doc — recency anchor
    bull_ex: list[str] = []
    bear_ex: list[str] = []
    for c in ev_pool:
        txt = c.get("text", "") or ""
        if not pat.search(txt):
            continue
        n_docs += 1
        w = _doc_weight(c.get("score", 0), c.get("ts", 0.0), ref_ts)   # upvoted AND recent weigh more
        r = classify_odte_intent(txt, ticker)
        bull += w * r["bull"]
        bear += w * r["bear"]
        if r["intent"] == "bullish":
            n_bull += 1
        elif r["intent"] == "bearish":
            n_bear += 1
        for ex in r.get("bull_examples", []):
            ce = _clean_example(ex)
            if ce and ce not in bull_ex:
                bull_ex.append(ce)
        for ex in r.get("bear_examples", []):
            ce = _clean_example(ex)
            if ce and ce not in bear_ex:
                bear_ex.append(ce)
    net = bull - bear
    intent = "bullish" if net >= 2 else ("bearish" if net <= -2 else "neutral")
    # examples must SUPPORT the resolved intent (not just any matched span). Neutral shows none.
    examples = (bull_ex if intent == "bullish"
                else bear_ex if intent == "bearish" else [])[:max_examples]
    return {"ticker": ticker, "intent": intent, "bull": round(bull, 2), "bear": round(bear, 2),
            "bull_examples": bull_ex[:max_examples], "bear_examples": bear_ex[:max_examples],
            # SEPARATED concepts: mention/chatter vs directional evidence vs resolved intent.
            "n_docs": n_docs,                      # mention_count / chatter rank
            "directional_docs": n_bull + n_bear,   # docs with an actual bull/bear signal
            "n_bullish_docs": n_bull, "n_bearish_docs": n_bear,
            "examples": examples}


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
    # paid-signal / alert-bot promo (passes other gates because it names $SPY + options context)
    "real-time options flow", "real time options flow", "options flow alert", "try free",
    "free trial", "sign up", "subscribe", "members only", "join free",
    # class-action / legal blast spam (frequent on $TICKER searches)
    "class action", "law firm", "rosen law", "shareholder rights", "securities fraud",
    "investors who purchased", "investigation on behalf",
)
# Alert-bot "Conviction 3/5"-style scoring badges (promo signal), matched case-insensitively.
_SPAM_RE = re.compile(r"\bconviction\s*\d\s*/\s*5\b", re.IGNORECASE)
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
    if any(t in low for t in _SPAM_TERMS) or _SPAM_RE.search(low):
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


# ---------------------------------------------------------------------------
# Employer / compliance restricted underlyings — HARD, code-level block.
# These symbols can NEVER be emitted as a tradable 0DTE candidate or carry option contracts in
# the report. Enforced HERE (not only in the controller prompt / controller_policy.json) so a
# misconfigured run, a tampered policy file, or a forgetful agent can never surface a tradable
# chain for them. NVDA is the user's employer — strictly forbidden as a trade vehicle; it may
# appear ONLY as read-only market context tagged RESTRICTED_EMPLOYER with contracts stripped.
# Config (options_social.restricted_underlyings) may ADD names but can never remove this floor.
# ---------------------------------------------------------------------------
RESTRICTED_EMPLOYER_TICKERS = frozenset({"NVDA"})


def _restricted_underlyings(p: dict | None = None) -> set[str]:
    """Underlyings hard-blocked from tradable emission: always the RESTRICTED_EMPLOYER_TICKERS
    floor (NVDA), plus any config-supplied extras (options_social.restricted_underlyings). Config
    can ADD, never remove."""
    extra: set[str] = set()
    try:
        src = p
        if src is None:
            from util import OPTIONS_SOCIAL_PARAMS
            src = OPTIONS_SOCIAL_PARAMS
        extra = {str(s).upper() for s in (src or {}).get("restricted_underlyings", [])}
    except Exception:
        extra = set()
    return set(RESTRICTED_EMPLOYER_TICKERS) | extra


def is_restricted_underlying(ticker: str | None, p: dict | None = None) -> bool:
    """True when `ticker` is employer/compliance-restricted (never tradable in the 0DTE report)."""
    return bool(ticker) and str(ticker).upper() in _restricted_underlyings(p)


def _restricted_card(ticker: str, social_direction: str | None, mentions: int = 0) -> dict:
    """Read-only RESTRICTED_EMPLOYER context card — same shape as build_ticker_card but with every
    contract field stripped and a hard restricted marker. Surfaced as market context, never as a
    tradable idea."""
    return {
        "ticker": str(ticker).upper(), "mentions": int(mentions),
        "verdict": "RESTRICTED_EMPLOYER", "confidence": "n/a", "price_dir": None,
        "social": social_direction, "contracts": [],
        "note": ("RESTRICTED_EMPLOYER — employer restriction; read-only market context only, "
                 "NOT tradable. No orders for this symbol under any circumstance."),
        "restricted": True, "restricted_reason": "employer",
        "available_contracts": [], "options_expiry": None, "options_status": "restricted_employer",
    }


def _select_odte_candidate(ranked: list, ev_pool: list[dict], min_mentions: int,
                           now_ts: float) -> dict | None:
    """First ranked ticker clearing the mention floor with nonzero directional sentiment, plus its
    evidence. Returns None when nothing qualifies. Employer/compliance-restricted symbols (e.g.
    NVDA) are never eligible — they can't be a tradable candidate."""
    for tk, s in ranked:
        if is_restricted_underlying(tk):
            continue   # hard employer/compliance block — never a tradable candidate
        if s["mentions"] >= min_mentions and abs(s["sentiment"]) > 0.0:
            return {
                "ticker": tk,
                "direction": "bullish" if s["sentiment"] > 0 else "bearish",
                **s,
                "evidence": _collect_odte_evidence(tk, ev_pool, now_ts),
            }
    return None


# ---------------------------------------------------------------------------
# SPY 0DTE decision scorecard — transparent, conservative, fail-closed to OBSERVE.
# NOT financial advice, NO orders. A directional PRICE read is mandatory for any
# CALL/PUT lean: social pressure alone can NEVER produce a directional verdict.
# ---------------------------------------------------------------------------

_TREND_PCT = 0.0015   # need |last/prev_close - 1| >= 0.15% to call an intraday direction
_WIDE_SPREAD = 0.25   # top-contract spread above this = poor liquidity (caps confidence)


def _price_direction(price: dict | None) -> str | None:
    """Intraday SPY direction from price context: 'bullish'/'bearish' only when the move clears
    _TREND_PCT vs prior close AND VWAP confirms the same side; 'neutral' when near VWAP / sub-
    threshold / VWAP unavailable; None when price data is missing or unusable (fail closed)."""
    if not price or not price.get("ok"):
        return None
    last, prev, above_vwap = price.get("last"), price.get("prev_close"), price.get("above_vwap")
    if not last or not prev:
        return None
    chg = last / prev - 1.0
    if chg >= _TREND_PCT and above_vwap is True:
        return "bullish"
    if chg <= -_TREND_PCT and above_vwap is False:
        return "bearish"
    return "neutral"


def _liquidity_ok(paper_options: dict | None) -> bool:
    """True when at least one surfaced contract is within budget at a not-too-wide spread."""
    for c in (paper_options or {}).get("contracts", []):
        sp = c.get("spread_pct")
        if not c.get("above_budget") and sp is not None and sp <= _WIDE_SPREAD:
            return True
    return False


def _decide_verdict(price: dict | None, social_direction: str | None, liq_ok: bool) -> dict:
    """Shared, ticker-agnostic gate → {verdict, confidence, price_dir, social_dir, why}. A
    directional PRICE read is mandatory; social is confirmation-only; price↔social conflict and
    missing/neutral price all resolve to OBSERVE. Confidence is 'low'/'medium' only."""
    price_dir = _price_direction(price)
    social_dir = social_direction if social_direction in ("bullish", "bearish") else None
    if price_dir is None:
        return {"verdict": "OBSERVE", "confidence": "low", "price_dir": None,
                "social_dir": social_dir, "why": "no_price"}
    if price_dir == "neutral":
        return {"verdict": "OBSERVE", "confidence": "low", "price_dir": price_dir,
                "social_dir": social_dir, "why": "neutral_price"}
    if social_dir and social_dir != price_dir:
        return {"verdict": "OBSERVE", "confidence": "low", "price_dir": price_dir,
                "social_dir": social_dir, "why": "conflict"}
    verdict = "CALL-leaning" if price_dir == "bullish" else "PUT-leaning"
    confidence = "medium" if (social_dir == price_dir and liq_ok) else "low"
    return {"verdict": verdict, "confidence": confidence, "price_dir": price_dir,
            "social_dir": social_dir, "why": "lean"}


def build_scorecard(price: dict | None, paper_options: dict | None,
                    social_direction: str | None) -> dict:
    """Transparent SPY 0DTE decision scorecard → verdict CALL-leaning / PUT-leaning / OBSERVE.

    Conservative and fail-closed (see _decide_verdict). ``social_direction`` is the SPY contextual
    intent (summarize_odte_intent) — confirmation only; only 'bullish'/'bearish' confirm. Confidence
    is 'low'/'medium' only (this tool claims no edge)."""
    liq_ok = _liquidity_ok(paper_options)
    d = _decide_verdict(price, social_direction, liq_ok)
    price_dir, social_dir = d["price_dir"], d["social_dir"]
    base = {"price_direction": price_dir, "social_direction": social_dir,
            "verdict": d["verdict"], "confidence": d["confidence"]}

    if d["why"] == "no_price":
        base["reasons"] = ["SPY price/trend data unavailable or stale — observe only."]
        return base
    if d["why"] == "neutral_price":
        base["reasons"] = ["SPY near VWAP / no decisive intraday trend — observe only "
                           "(social pressure alone is not a directional signal)."]
        return base
    if d["why"] == "conflict":
        base["reasons"] = [f"Conflict: intraday price reads {price_dir} but social pressure reads "
                           f"{social_dir} — observe only."]
        return base

    chg = price["last"] / price["prev_close"] - 1.0
    reasons = [f"SPY intraday {chg:+.2%} vs prior close and "
               f"{'above' if price_dir == 'bullish' else 'below'} VWAP."]
    if social_dir == price_dir:
        reasons.append(f"Social pressure agrees ({social_dir}) — confirmation only, not the driver.")
    else:
        reasons.append("No confirming SPY social pressure — confidence capped at low.")
    if not liq_ok:
        reasons.append("0DTE liquidity weak/absent or above budget — confidence capped at low.")
    reasons.append("PAPER lean only — max loss is the entire premium; observe-only is always valid.")
    base["reasons"] = reasons
    return base


def _resolve_intraday_trend(ticker: str, allow_fetch: bool) -> dict:
    """Intraday price context for any `ticker` (yfinance; no orders). Fail-closed."""
    if not allow_fetch:
        return {"ok": False, "status": "skipped: --no-fetch"}
    try:
        from data.odte_options import fetch_spy_trend  # generic: takes a ticker
        return fetch_spy_trend(ticker, allow_fetch=allow_fetch)
    except Exception as exc:
        logger.warning("%s trend lookup failed: %s", ticker, exc)
        return {"ok": False, "status": f"error: {exc}"}


def _resolve_spy_trend(allow_fetch: bool) -> dict:
    """SPY intraday price context for the scorecard (yfinance; no orders). Fail-closed."""
    return _resolve_intraday_trend("SPY", allow_fetch)


def _resolve_paper_options(candidate: dict | None, budget: float, include: bool,
                           allow_fetch: bool) -> dict:
    """PAPER-ONLY 0DTE option idea (yfinance; no orders). Skips cleanly without a candidate, when
    disabled, or under --no-fetch; fails closed on any lookup error."""
    if not (candidate and include and allow_fetch):
        return {"status": "skipped (no candidate / disabled / --no-fetch)", "contracts": []}
    if is_restricted_underlying(candidate.get("ticker")):
        return {"status": "restricted_employer", "contracts": []}   # never resolve a chain
    try:
        from data.odte_options import build_paper_options
        return build_paper_options(candidate["ticker"], candidate["direction"], budget,
                                   allow_fetch=allow_fetch)
    except Exception as exc:
        logger.warning("paper options lookup failed: %s", exc)
        return {"status": f"error: {exc}", "contracts": []}


# ---------------------------------------------------------------------------
# TOP CHATTER — compact, paper-only, beginner cards for the most-talked-about OTHER tickers.
# Same gates as the SPY scorecard (price required; social confirm-only; no chain → OBSERVE).
# Derived from the SAME quality-gated evidence pool, so it inherits the spam filtering and the
# (honest) source limitation: today's evidence is SPY/QQQ-biased by the configured query.
# ---------------------------------------------------------------------------

_CASHTAG_ONLY_RE = re.compile(r"\$([A-Za-z]{1,5})\b")


def _cashtag_mentions(texts) -> Counter:
    """Count ONLY $cashtag tickers (normalized, stopwords dropped). Bare all-caps tokens are
    excluded here — they're mostly 0DTE jargon (FOMC/GEX/OI/OPEX), not real tickers — which keeps
    the top-chatter list from surfacing jargon as fake tickers."""
    cnt: Counter = Counter()
    for t in texts:
        for m in _CASHTAG_ONLY_RE.findall(t or ""):
            sym = m.upper()
            if sym not in _STOPWORDS:
                cnt[sym] += 1
    return cnt


def rank_top_chatter(ev_pool: list[dict], exclude: set[str] = frozenset({"SPY"}),
                     max_n: int = 5, min_mentions: int = 2,
                     allowed: set[str] | None = None) -> list[tuple[str, int]]:
    """Top-N (ticker, mentions) from the quality-gated evidence, excluding `exclude` (the SPY
    backdrop) and below-floor names; spam is already gone (ev_pool is post-filter).

    When `allowed` (a VALIDATED optionable universe, e.g. agg_data ∪ core) is given, bare uppercase
    tickers count too — WSB comments often omit the `$` — but only when they're in that universe, so
    jargon ('FOMC'/'GEX'/'OI') is excluded. Without `allowed`, falls back to cashtag-only (the safe
    default that never surfaces jargon).

    ORDER is weighted by _doc_weight (upvotes × recency — a ticker carried by upvoted, recent
    comments outranks one carried by ignored or stale ones), but the floor and the returned count
    stay RAW mention counts so the report's '(N posts)' label remains an honest head-count."""
    raw: Counter = Counter()       # honest mention count — what the report shows / the floor checks
    wt: Counter = Counter()        # upvote × recency-weighted mention strength — what we rank by
    ref_ts = max((c.get("ts", 0.0) or 0.0 for c in ev_pool), default=0.0)   # newest doc — recency anchor
    for c in ev_pool:
        text = c.get("text", "") or ""
        per_doc = (extract_ticker_mentions([text], allowed=allowed) if allowed
                   else _cashtag_mentions([text]))
        w = _doc_weight(c.get("score", 0), c.get("ts", 0.0), ref_ts)
        for sym, k in per_doc.items():
            raw[sym] += k
            wt[sym] += k * w
    items = [(t, n) for t, n in raw.items() if t not in exclude and n >= min_mentions]
    items.sort(key=lambda kv: (-wt[kv[0]], kv[0]))
    return items[:max_n]


def _resolve_ticker_options(ticker: str, price: dict, intent: str, budget: float,
                            allow_fetch: bool) -> dict:
    """PAPER-only same-day chain for `ticker` on the side implied by price (else social) direction.
    Fail-closed; no direction → skip. Never places orders."""
    if not allow_fetch:
        return {"status": "skipped: --no-fetch", "contracts": [], "expiry": None}
    pdir = _price_direction(price)
    side = pdir if pdir in ("bullish", "bearish") else (
        intent if intent in ("bullish", "bearish") else None)
    if side is None:
        return {"status": "no direction", "contracts": [], "expiry": None}
    try:
        from data.odte_options import build_paper_options
        return build_paper_options(ticker, side, budget, allow_fetch=allow_fetch)
    except Exception as exc:
        logger.warning("ticker options lookup failed for %s: %s", ticker, exc)
        return {"status": f"error: {exc}", "contracts": [], "expiry": None}


def build_ticker_card(ticker: str, price: dict | None, paper_options: dict | None,
                      social_direction: str | None, mentions: int = 0) -> dict:
    """One compact, PAPER-only beginner card for a single name. Verdict uses the shared gate
    (price required; social confirm-only). A same-day chain that doesn't exist → OBSERVE (you
    can't 0DTE-trade it). OBSERVE carries NO contracts. Never an instruction to trade."""
    # Employer/compliance-restricted symbols (e.g. NVDA) short-circuit to a read-only
    # RESTRICTED_EMPLOYER card with all contracts stripped — never tradable.
    if is_restricted_underlying(ticker):
        return _restricted_card(ticker, social_direction, mentions)
    po = paper_options or {}
    liq_ok = _liquidity_ok(po)
    d = _decide_verdict(price, social_direction, liq_ok)
    verdict, confidence, note = d["verdict"], d["confidence"], None
    # Only when the gate is DIRECTIONAL but there's no tradable same-day chain do we downgrade to
    # OBSERVE for "no same-day options" (an OBSERVE-by-price card keeps its price-based reason).
    if verdict in ("CALL-leaning", "PUT-leaning") and not po.get("expiry"):
        verdict, confidence, note = "OBSERVE", "low", "no same-day options"
    want = {"CALL-leaning": "call", "PUT-leaning": "put"}.get(verdict)
    contracts = [k for k in po.get("contracts", []) if k.get("option_type") == want] if want else []
    return {"ticker": ticker, "mentions": int(mentions), "verdict": verdict,
            "confidence": confidence, "price_dir": d["price_dir"],
            "social": social_direction, "contracts": contracts, "note": note,
            # Full resolved same-day chain (price-implied side), kept regardless of verdict so a
            # consumer/agent sees the tradable contracts even on OBSERVE. `contracts` stays the
            # human-report-filtered list (only shown on a lean).
            "available_contracts": list(po.get("contracts", [])),
            "options_expiry": po.get("expiry"), "options_status": po.get("status")}


# Per-ticker enrichment is bounded + per-ticker-timeout + fail-closed so one slow/hung name can't
# stall the whole scan. These are SCAN-tier knobs (decision-support only) — see odte_concurrency.
_TOP_CHATTER_MAX_WORKERS = 4
_TOP_CHATTER_TIMEOUT_S = 20.0


def build_top_chatter(ev_pool: list[dict], allow_fetch: bool, budget: float,
                      exclude: set[str] = frozenset({"SPY"}), max_n: int = 5,
                      min_mentions: int = 2, allowed: set[str] | None = None,
                      max_workers: int = _TOP_CHATTER_MAX_WORKERS,
                      timeout_s: float = _TOP_CHATTER_TIMEOUT_S) -> list[dict]:
    """Build up to `max_n` paper-only ticker cards for the most-chattered OTHER names (ranked over a
    VALIDATED `allowed` universe so bare uppercase tickers count but jargon doesn't). Each card
    fetches its own price/chain with BOUNDED concurrency + a per-ticker timeout + fail-closed
    degradation: a slow/hung or failing ticker becomes an OBSERVE card (no contracts, an
    `options_status` of timeout/error) instead of stalling or aborting the scan. On --no-fetch every
    card degrades to OBSERVE. Cards are SCAN-tier: `tier="scan"`, `execution_allowed=False` — this
    list never authorizes a trade. Reuses summarize_odte_intent for the contextual social read."""
    from data.odte_concurrency import bounded_gather

    ranked = list(rank_top_chatter(ev_pool, exclude, max_n, min_mentions, allowed))
    intents = {tk: summarize_odte_intent(tk, ev_pool)["intent"] for tk, _ in ranked}
    fetch_list = [(tk, n) for tk, n in ranked if not is_restricted_underlying(tk)]

    def _enrich(item):
        tk, _n = item
        price = _resolve_intraday_trend(tk, allow_fetch)
        po = _resolve_ticker_options(tk, price, intents[tk], budget, allow_fetch)
        return price, po

    gathered = bounded_gather(_enrich, fetch_list, max_workers=max_workers, timeout_s=timeout_s)
    by_ticker: dict[str, dict] = {fetch_list[i][0]: g for i, g in enumerate(gathered)}

    cards: list[dict] = []
    for tk, n in ranked:
        intent = intents[tk]
        if is_restricted_underlying(tk):
            # Employer/compliance block: surface as read-only context only — no price/chain fetch.
            cards.append(_restricted_card(tk, intent, n))
            continue
        g = by_ticker.get(tk) or {"ok": False, "status": "skipped"}
        if g["ok"]:
            price, po = g["result"]
            card = build_ticker_card(tk, price, po, intent, n)
        else:
            # fail-closed: a timeout/error/skip yields a safe OBSERVE card (no contracts).
            card = build_ticker_card(tk, None, {"status": g["status"]}, intent, n)
            card["fetch_status"] = g["status"]
        # Scan-tier marker: these cards are decision-support, NEVER an execution authorization.
        card["tier"] = "scan"
        card["execution_allowed"] = False
        cards.append(card)
    return cards


def _optionable_universe(p: dict, core: list[str]) -> set[str]:
    """The VALIDATED ticker universe TOP CHATTER ranks against (so bare-uppercase WSB tickers like
    MSFT/MU/TSLA count, but jargon like FOMC/GEX does not). Priority: an explicit
    ``top_chatter_universe`` param (mainly for tests/power-users) ∪ core; else the cached agg_data
    scored universe ∪ core; else just core (graceful fallback). Stopword jargon is excluded downstream
    by extract_ticker_mentions regardless."""
    allowed = {str(s).upper() for s in core}
    explicit = p.get("top_chatter_universe")
    if explicit:
        return allowed | {str(s).upper() for s in explicit}
    try:
        from data.cache import read_data_as_pd
        agg = read_data_as_pd("agg_data")
        if agg is not None and not getattr(agg, "empty", True) and "symbol" in getattr(agg, "columns", []):
            allowed |= {str(s).upper() for s in agg["symbol"].dropna()}
    except Exception as exc:
        logger.debug("optionable universe: agg_data unavailable (%s) — using core only", exc)
    return allowed


# ---------------------------------------------------------------------------
# WSB daily-discussion-thread comments — official/fail-closed STATUS seam (real ingestion is a
# follow-up). NEVER uses cookies or chat-pasted secrets; prefers Reddit OAuth env creds.
# ---------------------------------------------------------------------------

_DAILY_THREAD_RE = re.compile(r"daily\s+discussion\s+thread", re.IGNORECASE)


def _parse_thread_id(url: str | None) -> str:
    """Extract a base36 post id from a reddit /comments/{id}/ URL ('' if none)."""
    m = re.search(r"/comments/([a-z0-9]+)", url or "", re.IGNORECASE)
    return m.group(1) if m else ""


def wsb_daily_thread_title(d: _date) -> dict:
    """The WSB daily-discussion-thread human title + URL slug for an ET calendar date `d`, e.g.
    'Daily Discussion Thread for June 18, 2026' / 'daily_discussion_thread_for_june_18_2026'.

    IMPORTANT: the title/slug ARE derivable from the date, but the post id ('1u9240r') is NOT —
    Reddit assigns it. So ``id`` is always None here; the id must come from listing/search discovery
    or a daily_thread_id/url override. This helper exists to build the search query, not to fake a
    URL the tool can fetch."""
    month = d.strftime("%B")
    title = f"Daily Discussion Thread for {month} {d.day}, {d.year}"
    slug = f"daily_discussion_thread_for_{month.lower()}_{d.day}_{d.year}"
    return {"title": title, "slug": slug, "id": None}


def _fetch_reddit_search(subreddit: str, query: str, allow_fetch: bool = True,
                         limit: int = 10) -> list[dict]:
    """Official Reddit search (OAuth → public JSON), newest-first, restricted to the subreddit and
    the last day. Fail-closed → [] on creds-less 403 / error. No cookies. Used to discover today's
    daily thread by title/date when it isn't in the hot listing (it's usually stickied)."""
    if not allow_fetch:
        return []
    token = _reddit_oauth_token()
    ua = _reddit_user_agent()
    params = {"q": query, "restrict_sr": 1, "sort": "new", "t": "day", "limit": limit}
    try:
        if token:
            r = requests.get(_REDDIT_OAUTH_SEARCH_URL.format(sub=subreddit),
                             headers={"Authorization": f"bearer {token}", "User-Agent": ua},
                             params=params, timeout=15)
        else:
            r = requests.get(_REDDIT_SEARCH_JSON_URL.format(sub=subreddit),
                             headers={"User-Agent": ua}, params=params, timeout=15)
        r.raise_for_status()
        children = (r.json() or {}).get("data", {}).get("children", [])
        return _parse_reddit_children(children, api_source="reddit_search")
    except Exception as exc:
        logger.warning("reddit daily-thread search failed: %s", exc)
        return []


def _find_daily_thread_id(posts: list[dict], params: dict, subreddit: str = "wallstreetbets",
                          allow_fetch: bool = False) -> str:
    """Today's daily-discussion-thread post id, in priority order: (1) configured daily_thread_id /
    daily_thread_url override; (2) the hot listing by title (cheap, already fetched); (3) official
    search by title/date (robust — the thread is stickied and reliably findable by day). '' if none."""
    override = params.get("daily_thread_id") or _parse_thread_id(params.get("daily_thread_url"))
    if override:
        return str(override)
    for q in posts:
        if _DAILY_THREAD_RE.search(q.get("title", "") or ""):
            return q.get("id", "") or ""
    if allow_fetch:
        for q in _fetch_reddit_search(subreddit, "Daily Discussion Thread", allow_fetch):
            if _DAILY_THREAD_RE.search(q.get("title", "") or ""):
                return q.get("id", "") or ""
    return ""


def _reddit_auth_label(bearer_token: str | None) -> str:
    """Which auth path fetch_reddit_comments will use, for a REDACTED status line (no token)."""
    if os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET"):
        return "app oauth"
    return "bearer token" if bearer_token else "public"


_DAILY_LIMIT_DEFAULT = 10000   # default: read the WHOLE thread (fetch_reddit_comments auto-paginates)
_DAILY_LIMIT_CAP = 10000       # hard safety ceiling so a runaway thread can't page forever


def _clamp_daily_limit(n) -> int:
    """Clamp the daily-thread comment read size to [1, _DAILY_LIMIT_CAP]. fetch_reddit_comments
    auto-paginates 'more comments', so by default we read the WHOLE thread (up to the cap); the cap
    is just a runaway safety net. Lower it (daily_thread_limit) for a faster, partial read."""
    try:
        return max(1, min(int(n), _DAILY_LIMIT_CAP))
    except (TypeError, ValueError):
        return _DAILY_LIMIT_DEFAULT


# ---------------------------------------------------------------------------
# Daily-thread comment NOISE filter (runs BEFORE ticker ranking / sentiment). WSB daily threads are
# mostly chatter — bot/banbet commands, one-word/emoji replies, off-topic conversation, and bare
# price questions. We keep only ACTIONABLE lines (a ticker/option/strike/day-trade cue) plus longer
# substantive comments; everything else is dropped so it can't drive mentions or sentiment.
# ---------------------------------------------------------------------------

_BANBET_RE = re.compile(r"!?\bbanbet\b|no active banbet", re.IGNORECASE)
_OFFTOPIC_RE = re.compile(
    r"\b(breakfast|lunch|dinner|sister|brother|frens?|weather|coffee|girlfriend|boyfriend|wife|"
    r"cscareerquestions|autism|autistic|slaves?|handjob|chafed|clifford|africa|casino|balls)\b"
    r"|richard brandon|late for work", re.IGNORECASE)
# Explicit market/trading context — a NON-ticker comment must carry at least one of these to count
# as on-topic (generic 'money/back/profit' alone does NOT — we can't see the parent comment).
_MARKET_CTX_RE = re.compile(
    r"\b(market|stocks?|shares?|equit\w+|calls?|puts?|options?|strikes?|0dte|odte|theta|gamma|"
    r"spy|qqq|spx|ndx|nasdaq|dow|sp500|fed|fomc|cpi|pce|earnings|rally|rip|dump|pump|bull\w*|"
    r"bear\w*|short|position\w*|portfolio|ticker|premarket|tendies|bag\w*|hedge|vix|yields?|"
    r"rate cut|rate hike|green|red candle|bid|ask|volume|iv)\b", re.IGNORECASE)
# Options/strike cue: calls/puts/0dte/strike/theta… or a strike like '400c' / '1150p'.
_OPT_CUE_RE = re.compile(
    r"\b(0dte|odte|calls?|puts?|strikes?|theta|gamma|delta|premium|contracts?|leaps?)\b"
    r"|\b\d{2,5}\s?[cp]\b", re.IGNORECASE)
# Day-trade / direction cue (only counts as actionable WHEN a ticker is also present).
_DAYTRADE_RE = re.compile(
    r"\b(send it|eod|lotto|yolo|moon|rocket|long|short|buy|buying|sell|selling|bull|bear|scalp|"
    r"breakout|squeeze|hold|hodl)\b", re.IGNORECASE)


def _comment_is_actionable(text: str) -> bool:
    """True when a comment carries a tradable signal: an options/strike cue (strong on its own), OR
    a day-trade/direction cue together with a ticker. A bare price QUESTION (has '?' but no options
    cue) is NOT actionable — it's chatter, not directional evidence."""
    has_opt = bool(_OPT_CUE_RE.search(text or ""))
    if "?" in (text or "") and not has_opt:
        return False
    if has_opt:
        return True
    has_ticker = any(s not in _STOPWORDS for s in _iter_symbols(text or ""))
    return bool(_DAYTRADE_RE.search(text or "")) and has_ticker


def _is_noise_comment(text: str) -> bool:
    """True if a daily-thread comment is low-information noise that must NOT drive ranking/sentiment.
    Strong actionable lines are always kept; bot/banbet tables, off-topic conversation, one-word /
    single-number / emoji-only, and very short (<=30 char) non-actionable comments are dropped."""
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    if _BANBET_RE.search(low) or low.startswith("|") or "|--" in low or "|:-" in low:
        return True
    if _comment_is_actionable(t):
        return False
    if _OFFTOPIC_RE.search(low):
        return True
    words = re.findall(r"[a-z0-9]+", low)
    if len(words) <= 1:        # one word / single number / emoji-only
        return True
    # On-topic gate: a non-actionable comment must mention a ticker OR explicit market/trading
    # context — otherwise it's off-topic chatter regardless of length ('Going to Africa...'),
    # including generic-money replies with no ticker ('I hope so. I need my money back.').
    has_ticker = any(s not in _STOPWORDS for s in _iter_symbols(t))
    if not (has_ticker or _MARKET_CTX_RE.search(low)):
        return True
    return len(t) <= 30        # very short, non-actionable low-info


def _is_downvoted(c: dict) -> bool:
    """True if the community actively REJECTED a comment (net score < 0) — a take the crowd voted
    down, which must not drive ranking/sentiment. Comments still inside Reddit's initial
    vote-fuzzing window (``score_hidden``) get the benefit of the doubt and are KEPT: you can't
    penalize a take the crowd hasn't finished voting on yet."""
    if c.get("score_hidden"):
        return False
    return int(c.get("score", 0) or 0) < 0


def fetch_daily_thread_comments(posts: list[dict], params: dict, allow_fetch: bool = True,
                                limit: int = _DAILY_LIMIT_DEFAULT) -> tuple[list[dict], str, str]:
    """Official/fail-closed fetch of today's daily-thread top comments. Auth order: app OAuth →
    explicit ``params['reddit_bearer_token']`` (a caller-supplied, ephemeral, read-only token —
    NEVER cookies, NEVER persisted/logged) → anonymous public JSON. Returns (comments, status,
    thread_id); the status names the auth path but NEVER the token. When all paths are blocked,
    returns 'unavailable: auth needed' rather than a silent empty (no misleadingly low count)."""
    if not allow_fetch:
        return [], "skipped (--no-fetch)", ""
    sub = params.get("subreddit", "wallstreetbets")
    tid = _find_daily_thread_id(posts, params, sub, allow_fetch)
    if not tid:
        return [], "no daily discussion thread found (listing or search)", ""
    bearer = params.get("reddit_bearer_token") or None   # runtime arg only; never persisted/logged
    raw = fetch_reddit_comments(tid, limit=limit, allow_fetch=allow_fetch, bearer_token=bearer)
    # Drop low-information chatter (bot/banbet, off-topic, one-word, bare price questions) AND
    # community-rejected takes (net-negative score, _is_downvoted) BEFORE the comments enter the
    # evidence pool / ticker ranking / sentiment. Upvote scoring then weights the survivors.
    comments = [c for c in raw
                if not _is_noise_comment(c.get("body", "") or "") and not _is_downvoted(c)]
    label = _reddit_auth_label(bearer)
    if comments:
        # Plain English: how many we READ (auto-paginated, up to the safety cap) vs how many survived
        # the noise/downvote filter and were used.
        return comments, f"{len(raw)} read, {len(comments)} included ({label})", tid
    if raw:   # fetched, but every comment read was low-signal noise
        return [], f"{len(raw)} read, 0 included ({label}) — all low-signal/noise", tid
    if not (os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")) and not bearer:
        return [], ("unavailable: auth needed (set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET, "
                    "or pass --reddit-bearer-token)"), tid
    return [], "unavailable (no comments returned)", tid


_ODTE_DIR = os.path.expanduser("~/0dte")


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _unexpired_token(obj: dict) -> str | None:
    """Return the bearer token from a {token|reddit_token, expires|reddit_token_expires} dict, or
    None if absent or already expired. Accepts epoch in seconds or milliseconds."""
    tok = obj.get("token") or obj.get("reddit_token")
    if not tok or not isinstance(tok, str):
        return None
    exp = obj.get("expires") or obj.get("reddit_token_expires")
    if exp:
        try:
            exp = float(exp)
            if exp > 1e12:          # milliseconds → seconds
                exp /= 1000.0
            if exp <= time.time():
                return None         # expired — skip silently
        except (TypeError, ValueError):
            pass
    return tok


def load_0dte_runtime_config() -> dict:
    """Best-effort runtime config so ``make odte-report`` can run with NO env/flags (for the Hermes
    agent). Reads ``~/0dte/`` in priority order; fail-soft (returns {} on any problem); the token
    value is NEVER logged. Returns any subset of {'reddit_bearer_token', 'daily_thread_id'}.

    App OAuth:     ~/0dte/config.json {"reddit_client_id","reddit_client_secret"} — the ROBUST,
                   non-expiring auth (preferred over the bearer token, which expires daily).
    Bearer token:  ~/0dte/config.json → ~/0dte/reddit_token.json → legacy ~/.reddit_token.json
                   (expired tokens are skipped). The reddit_token.json format is the same
                   {"token","expires"} your daily refresh already writes — just move it here.
    Daily thread:  ~/0dte/config.json {"daily_thread_id"} → ~/0dte/daily_thread_id.txt (first line).
    """
    cfg = _read_json(os.path.join(_ODTE_DIR, "config.json"))
    out: dict = {}

    cid, secret = cfg.get("reddit_client_id"), cfg.get("reddit_client_secret")
    if cid and secret:
        out["reddit_client_id"], out["reddit_client_secret"] = str(cid), str(secret)

    token = (_unexpired_token(cfg)
             or _unexpired_token(_read_json(os.path.join(_ODTE_DIR, "reddit_token.json")))
             or _unexpired_token(_read_json(os.path.join(_ODTE_DIR, ".reddit_token.json")))   # dotted variant
             or _unexpired_token(_read_json(os.path.expanduser("~/.reddit_token.json"))))      # legacy home
    if token:
        out["reddit_bearer_token"] = token

    tid = cfg.get("daily_thread_id")
    if not tid:
        try:
            with open(os.path.join(_ODTE_DIR, "daily_thread_id.txt"), encoding="utf-8") as fh:
                tid = fh.readline().strip()
        except Exception:
            tid = ""
    if tid:
        out["daily_thread_id"] = str(tid).strip()
    return out


_RSS_BOILERPLATE = re.compile(r"\s*submitted by\s*/u/\S+\s*\[link\]\s*\[comments\]\s*$", re.I)


def _clean_dump_text(text: str) -> str:
    """Make a raw social doc readable on one line: unescape HTML entities (``&#39;``→``'``,
    ``&quot;``→``"``, ``&#32;``→space, ``&amp;``→``&``), strip the Reddit RSS
    ``submitted by /u/x [link] [comments]`` tail, drop stray ``[link]``/``[comments]`` markers,
    and collapse all whitespace to single spaces."""
    t = html.unescape(text or "")
    t = _RSS_BOILERPLATE.sub("", t)
    return " ".join(t.replace("[link]", " ").replace("[comments]", " ").split()).strip()


# Per-kind cap on retained scrape-text snapshots (keep the most recent N of each); the UI reads the
# whole history from data/odte/scrape/, so this just bounds disk growth over many runs.
_SCRAPE_RETENTION = 500


def _prune_scrape_snapshots(out_dir: str, kind: str, keep: int = _SCRAPE_RETENTION) -> None:
    """Delete all but the most recent ``keep`` ``<kind>_text_*.txt`` snapshots. Best-effort."""
    try:
        files = sorted(Path(out_dir).glob(f"{kind}_text_*.txt"))
        for stale in files[:-keep]:
            stale.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - prune is best-effort
        logger.debug("scrape-snapshot prune (%s) failed: %s", kind, exc)


def _dump_analyzed_texts(combined: list[dict]) -> None:
    """Write the CLEANED texts this run analyzed (one document per line) — Reddit (hot posts + WSB
    daily-thread comments) vs X — as TIMESTAMPED snapshots under ``data/odte/scrape/`` so the history
    accumulates over time (``reddit_text_YYYY_MM_DD_HH_MM.txt`` / ``x_text_YYYY_MM_DD_HH_MM.txt``),
    plus a stable ``reddit_text.txt`` / ``x_text.txt`` "latest" pointer for back-compat. Each file is
    prefixed with a ``# <date> | …`` header showing exactly what was captured (so a small file from a
    failed daily-comments fetch is obvious at a glance). Fail-soft (never breaks the report). A kind is
    only written when it produced ≥1 in-window doc, so a failed X (or Reddit) fetch does NOT write an
    empty snapshot. Old snapshots are pruned to the most recent ``_SCRAPE_RETENTION`` per kind."""
    try:
        out_dir = ODTE_SCRAPE_DIR
        n = {"reddit": 0, "reddit_daily_comment": 0, "x": 0}
        reddit_lines, x_lines = [], []
        for c in combined:
            src = c.get("source", "reddit")
            text = _clean_dump_text(c.get("text", ""))
            if not text:
                continue
            (x_lines if src == "x" else reddit_lines).append(text)
            n[src] = n.get(src, 0) + 1
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M")
        ts = now.strftime("%Y_%m_%d_%H_%M")
        files = (
            ("reddit", reddit_lines,
             f"# {stamp} | reddit: {n['reddit']} hot posts + {n['reddit_daily_comment']} daily-thread comments"),
            ("x", x_lines, f"# {stamp} | x: {n['x']} tweets"),
        )
        wrote = []
        for kind, lines, header in files:
            if not lines:
                continue
            os.makedirs(out_dir, exist_ok=True)
            body = header + "\n" + "\n".join(lines) + "\n"
            # Timestamped snapshot (history) + stable latest pointer (back-compat).
            for name in (f"{kind}_text_{ts}.txt", f"{kind}_text.txt"):
                with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
                    fh.write(body)
            _prune_scrape_snapshots(out_dir, kind)
            wrote.append(f"{len(lines)} {kind}")
        if wrote:
            logger.info("dumped analyzed social texts -> %s (%s)", out_dir, ", ".join(wrote))
    except Exception as exc:  # pragma: no cover - dump is best-effort, never fatal
        logger.debug("analyzed-text dump to %s failed: %s", ODTE_SCRAPE_DIR, exc)


def _base_x_query_tickers(x_query: str) -> set[str]:
    """Tickers the fixed X query already asks about (so cross-source enrichment doesn't re-search
    them). Cashtags in the query plus the SPY/QQQ core."""
    return {m.upper() for m in re.findall(r"\$([A-Za-z]{1,5})", x_query or "")} | {"SPY", "QQQ"}


def _wsb_top_tickers(posts: list[dict], daily_comments: list[dict], universe: set[str] | None,
                     exclude: set[str], top_n: int, min_mentions: int) -> list[str]:
    """Most-mentioned tickers across the WSB chatter (hot posts + daily-thread comments), restricted
    to `universe` (validated/optionable) and minus `exclude`. Empty when no validated universe."""
    if not universe:
        return []
    texts = [f"{q.get('title', '')} {q.get('selftext', '')}" for q in posts]
    texts += [(c.get("body", "") or "") for c in daily_comments]
    counts = extract_ticker_mentions(texts, allowed=universe)
    return [t for t, n in counts.most_common() if t not in exclude and n >= min_mentions][:top_n]


def _cross_source_x_enrich(p: dict, posts: list[dict], daily_comments: list[dict],
                           allowed: set[str] | None, broad_allowed: set[str] | None,
                           allow_fetch: bool, is_fresh) -> tuple[list[dict], str]:
    """Reddit → X cross-pollination: give the tickers trending in WSB their own X search (the fixed
    x_query only ever asks about SPY/QQQ), so the scorer sees BOTH platforms for the same name.
    Bounded (top-N validated tickers, one cached X call) and config-gated (default on). Returns
    (fresh enrichment tweets tagged like base X posts, human-readable status)."""
    if "x" not in p.get("sources", ["reddit"]) or not bool(p.get("cross_source_x_enrich", True)):
        return [], "disabled"
    picks = _wsb_top_tickers(
        posts, daily_comments, universe=(broad_allowed or allowed),
        exclude=_base_x_query_tickers(p.get("x_query", "")),
        top_n=int(p.get("cross_source_x_top_n", 5)),
        min_mentions=int(p.get("cross_source_x_min_mentions", 2)))
    if not picks:
        return [], "no eligible WSB tickers"
    if not allow_fetch:
        return [], f"skipped (--no-fetch): would query {', '.join(picks)}"
    query = "(" + " OR ".join(f"${t}" for t in picks) + ") lang:en -is:retweet"
    enr, st = fetch_x_mentions(query, limit=int(p.get("x_limit", 50)))
    enr = [t for t in enr if is_fresh(_parse_x_ts(t.get("created_at", "")))]
    return enr, f"{st}: +{len(enr)} tweets for {', '.join(picks)}"


def _gather_odte_sources(p: dict, allowed: set[str] | None, allow_fetch: bool, is_fresh,
                         now_ts: float = 0.0, broad_allowed: set[str] | None = None) -> dict:
    """Fetch Reddit posts + X + the WSB daily-thread comments, freshness-filter, run the quality
    filter ONCE, then score & rank tickers. Deriving mentions, scoring, and the evidence pool from
    the SAME filtered set keeps them consistent — so promo/spam can't inflate a ticker's mention
    count while being excluded from evidence (the live "SPY 47" bug). Returns post/spam counts plus
    `ranked` tickers and the quality-gated `ev_pool` evidence set (now incl. daily-thread comments)."""
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

    # WSB daily-discussion-thread comments — INGESTED here (official/fail-closed; no cookies). They
    # carry the real intraday chatter, so we fold them into the SAME combined list (before the one
    # quality filter) and stamp ts=now so they count as fresh. If unavailable, the status says so.
    daily_comments, daily_status, tid = ([], "disabled (reddit not in sources)", "")
    if "reddit" in sources and bool(p.get("daily_thread_comments", True)):
        dlimit = _clamp_daily_limit(p.get("daily_thread_limit", _DAILY_LIMIT_DEFAULT))
        daily_comments, daily_status, tid = fetch_daily_thread_comments(
            posts_all, p, allow_fetch, limit=dlimit)
    thread_url = f"https://www.reddit.com/comments/{tid}" if tid else ""

    # Cross-source enrichment (Reddit → X): the tickers WSB is buzzing about get their OWN X search
    # (the fixed x_query only asks SPY/QQQ), folded in as ordinary X docs so the scorer/cards see
    # both platforms for the same name. On by default; disable via cross_source_x_enrich: false.
    x_enrich, x_enrich_status = _cross_source_x_enrich(
        p, posts, daily_comments, allowed, broad_allowed, allow_fetch, is_fresh)
    x_base_n = len(x_posts)          # base-fetch fresh count (before enrichment) for an accurate stale stat
    x_posts = x_posts + x_enrich

    # Build ONE combined item list (Reddit posts + daily-thread comments + X), then run the
    # transparent spam/quality filter ONCE.
    combined = [{"text": f"{q['title']} {q['selftext']}", "ts": q.get("created_utc", 0.0),
                 "weight": q.get("score", 0), "title": q["title"][:140], "score": q["score"],
                 "url": q["permalink"], "source": "reddit"} for q in posts]
    # Daily-thread comments carry their REAL created_utc (so recency weighting favors fresher intraday
    # chatter); fall back to now_ts only when the timestamp is missing, so they still count as fresh.
    combined += [{"text": c.get("body", "") or "", "ts": (c.get("created_utc") or now_ts),
                  "weight": int(c.get("score", 0) or 0),
                  "title": (c.get("body", "") or "")[:140], "score": int(c.get("score", 0) or 0),
                  "url": thread_url, "source": "reddit_daily_comment"} for c in daily_comments]
    combined += [{"text": t.get("text", ""), "ts": _parse_x_ts(t.get("created_at", "")),
                  "weight": 0, "title": (t.get("text", "") or "")[:140], "score": 0,
                  "url": f"https://twitter.com/i/web/status/{t.get('id', '')}", "source": "x"}
                 for t in x_posts]
    # Dump the in-window texts this run analyzed to ~/0dte/{reddit,x}_text.txt (overwritten each run).
    _dump_analyzed_texts(combined)
    # ODTE evidence requires an allowed ticker + options/day-trading context (drops generic
    # SPY/QQQ chatter and risk-management platitudes that aren't 0DTE signal).
    kept = _quality_filter(combined, lambda c: c["text"], allowed=allowed,
                           require_options_context=True)
    reddit_spam = sum(1 for c in combined if c["source"] == "reddit") - \
        sum(1 for c in kept if c["source"] == "reddit")
    x_spam = sum(1 for c in combined if c["source"] == "x") - \
        sum(1 for c in kept if c["source"] == "x")

    # SECOND quality pass against the BROAD optionable universe (for TOP CHATTER only): keeps items
    # mentioning any validated optionable ticker (e.g. a daily-thread comment about bare MSFT/MU),
    # which the core (SPY/QQQ) ev_pool drops. The SPY scorecard / candidate keep using `kept`.
    broad_ev = (_quality_filter(combined, lambda c: c["text"], allowed=broad_allowed,
                                require_options_context=True)
                if broad_allowed and broad_allowed != allowed else kept)

    documents = [{"text": c["text"], "ts": c["ts"], "weight": c["weight"]} for c in kept]
    mentions = extract_ticker_mentions([c["text"] for c in kept], allowed=allowed)
    scores = score_social(mentions, documents)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["mentions"])[: int(p.get("max_tickers", 10))]

    return {
        "subreddit": subreddit,
        "posts": posts, "posts_all": posts_all,
        "reddit_stale": len(posts_all) - len(posts), "reddit_spam": reddit_spam,
        "x_posts": x_posts, "x_posts_all": x_posts_all, "x_status": x_status,
        "x_stale": len(x_posts_all) - x_base_n, "x_spam": x_spam,
        "x_enrich_status": x_enrich_status, "x_enrich_n": len(x_enrich),
        "ranked": ranked, "ev_pool": kept,  # ev_pool is the same filtered, quality-gated set
        "broad_ev": broad_ev,               # broader pool for TOP CHATTER ranking
        "daily_status": daily_status, "daily_n": len(daily_comments),
    }


def build_odte_social_report(allow_fetch: bool = True, params: dict | None = None,
                             now: datetime | None = None,
                             reddit_bearer_token: str | None = None) -> dict:
    """Build the 0DTE social-sentiment IDEA report (paper/analysis only). Never places orders.

    ``reddit_bearer_token`` is an OPTIONAL, caller-supplied, ephemeral read-only token for the WSB
    daily-thread comments fetch (auth order: app OAuth → this token → public). It is overlaid into
    the params at RUNTIME ONLY — never persisted to config, never logged, never returned in the
    report dict. Omit it to keep the current fail-closed OAuth/public behavior."""
    from util import OPTIONS_SOCIAL_PARAMS
    p = params if params is not None else OPTIONS_SOCIAL_PARAMS
    if reddit_bearer_token:
        p = {**p, "reddit_bearer_token": reddit_bearer_token}   # runtime overlay; not persisted

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

    # Which session anchor the window used — drives context-aware HOW FRESH wording. Mirrors
    # market_session_window: weekend / weekday pre-open → previous close; weekday at/after open →
    # today's open. (Rolling mode has no session anchor.)
    _et_now = now.astimezone(_eastern())
    if mode != "market_window":
        fresh_anchor = "rolling"
    elif _et_now.weekday() >= 5 or _et_now.time() < dtime(open_hm[0], open_hm[1]):
        fresh_anchor = "prev_close"
    else:
        fresh_anchor = "today_open"

    def _fresh(ts: float) -> bool:
        return bool(ts and ts > 0 and start_ts <= ts <= end_ts + 1.0)

    # TOP CHATTER ranks against a VALIDATED broad optionable universe (agg_data ∪ core) so bare
    # uppercase WSB tickers (MSFT/MU/TSLA) count while jargon (FOMC/GEX) is excluded. The SPY
    # scorecard/candidate keep the core (SPY/QQQ) allowed set.
    broad_allowed = _optionable_universe(p, list(p.get("core_universe", ["SPY", "QQQ"])))
    src = _gather_odte_sources(p, allowed, allow_fetch, _fresh, now_ts, broad_allowed)
    ranked, ev_pool = src["ranked"], src["ev_pool"]

    min_mentions = int(p.get("min_mentions", 3))
    candidate = _select_odte_candidate(ranked, ev_pool, min_mentions, now_ts)

    # PAPER-ONLY 0DTE option idea (yfinance; no orders). bullish->calls, bearish->puts.
    budget = float(p.get("budget_dollars", 50))
    paper_options = _resolve_paper_options(
        candidate, budget, bool(p.get("include_paper_options", True)), allow_fetch)

    # SPY-focused decision scorecard (price-led, social confirm-only; fails closed to OBSERVE).
    # Social confirmation uses the CONTEXTUAL classifier over SPY evidence (negation/inverse/
    # conflict aware), NOT the raw keyword sentiment that drives candidate selection.
    spy_trend = _resolve_spy_trend(allow_fetch)
    social_intent = summarize_odte_intent("SPY", ev_pool)
    scorecard = build_scorecard(spy_trend, paper_options, social_intent["intent"])

    # TOP CHATTER — compact paper cards for the most-chattered OTHER names (SPY is the backdrop, not
    # a card). Ranked over the BROAD evidence pool against the validated optionable universe, so
    # daily-thread comments about bare-uppercase names contribute. Coverage is still only the
    # configured sources — surfaced as a caveat, never hidden.
    top_chatter = build_top_chatter(src.get("broad_ev", ev_pool), allow_fetch, budget,
                                    exclude={"SPY"}, max_n=int(p.get("top_chatter_n", 5)),
                                    min_mentions=int(p.get("top_chatter_min_mentions", 2)),
                                    allowed=broad_allowed)
    top_chatter_caveat = ("Top chatter covers only the configured sources (this subreddit + the "
                          "daily thread + the X query) — not the whole market.")

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "disclaimer": ("ANALYSIS / PAPER ONLY — not financial advice, not auto-trading. "
                       "No orders are placed by this tool."),
        "budget_dollars": budget,
        "freshness_window": {
            "mode": mode,
            "anchor": fresh_anchor,   # today_open | prev_close | rolling — drives HOW FRESH wording
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
                  "n_filtered": src["x_spam"], "n_quality": len(src["x_posts"]) - src["x_spam"],
                  "enrich_status": src.get("x_enrich_status", "n/a"), "n_enrich": src.get("x_enrich_n", 0)},
        },
        "top_tickers": [{"ticker": tk, **s} for tk, s in ranked],
        "candidate": candidate,
        "spy_trend": spy_trend,
        "social_intent": social_intent,
        "scorecard": scorecard,
        "paper_options": paper_options,
        "top_chatter": top_chatter,
        "top_chatter_caveat": top_chatter_caveat,
        "daily_thread": {"status": src.get("daily_status", "n/a"), "n_comments": src.get("daily_n", 0)},
        "risk_notes": [
            "These contracts can lose ALL their value the same day — treat the whole budget as money you could lose.",
            "Lots of online hype does NOT mean it will go up — the crowd is wrong just as often.",
            f"Never risk more than ${budget:.0f} of practice money; starting at $0 (just watching) is best.",
            "We only read posts from today's market session — older posts are ignored on purpose.",
            "This is just information to learn from. Whether to trade at all is your choice.",
        ],
    }


def _short_et(iso: str | None) -> str:
    """Trim an ET ISO timestamp ('2026-06-17T09:30:00-04:00') to a friendly 'Jun-17 09:30'."""
    if not iso or len(iso) < 16:
        return "n/a"
    try:
        return f"{iso[5:7]}-{iso[8:10]} {iso[11:16]}"  # MM-DD HH:MM
    except Exception:
        return "n/a"


def _freshness_phrase(fw: dict) -> str:
    """Context-aware 'how far back' wording. Pre-open/weekend windows anchor to the PREVIOUS market
    close (so we must NOT say 'today'); regular-session windows anchor to today's open."""
    start = _short_et(fw.get("window_start_et"))   # 'MM-DD HH:MM'
    anchor = fw.get("anchor")
    if anchor == "today_open":
        return f"since today's open ({start[-5:]} ET)"           # HH:MM
    if anchor == "prev_close":
        return f"since the previous market close ({start} ET)"   # MM-DD HH:MM — not 'today'
    if anchor == "rolling":
        return f"in roughly the last {int(fw.get('max_lookback_hours', 96))}h"
    return f"since {start} ET"


# Plain-language verdict labels (no jargon). Maps the scorecard verdict to a beginner sentence.
_WHATDO = {
    "OBSERVE": ("🛑 DO NOTHING — just watch today",
                "There isn't a clear-enough reason to even practice a trade right now."),
    "CALL-leaning": ("✅ Maybe a tiny PRACTICE 'CALL' — a bet that SPY goes UP",
                     "The price trend gives a mild UP hint. This is practice money only."),
    "PUT-leaning": ("⚠️ Maybe a tiny PRACTICE 'PUT' — a bet that SPY goes DOWN",
                    "The price trend gives a mild DOWN hint. This is practice money only."),
}
_STRENGTH = {"low": "weak", "medium": "medium"}


def _kid_price_line(pt: dict) -> str:
    """Beginner price bullet — no VWAP/jargon, rounded numbers only."""
    if not pt or not pt.get("ok"):
        return "Price: we couldn't check SPY's price today, so we can't tell which way it's going."
    pc, vw = pt.get("pct_vs_prev_close"), pt.get("above_vwap")
    if pc is None:
        return "Price: SPY's change since yesterday is unknown."
    way = "UP" if pc > 0 else ("DOWN" if pc < 0 else "FLAT")
    avg = ""
    if vw is not None:
        avg = (" and it's higher than its average price so far today (an up sign)" if vw
               else " and it's lower than its average price so far today (a down sign)")
    return f"Price: SPY is {way} about {abs(pc):.1%} since yesterday{avg}."


def _kid_social_line(si: dict | None) -> str:
    """Beginner crowd bullet — supporting examples only, never a contradictory span."""
    if not si or not si.get("n_docs"):
        return "Crowd talk: almost nobody is posting about SPY 0DTE right now — no clear mood."
    mood = {"bullish": "hopeful it goes UP", "bearish": "worried it goes DOWN",
            "neutral": "split and unsure"}.get(si.get("intent"), "unsure")
    s = (f"Crowd talk: people online sound {mood} about SPY today ({si.get('n_docs', 0)} posts) — "
         "this is just chatter, not proof")
    ex = si.get("examples", [])  # already filtered to support the resolved mood (no negated spans)
    if ex and si.get("intent") in ("bullish", "bearish"):
        s += " (for example: " + ", ".join(f'"{e}"' for e in ex[:2]) + ")"
    return s + "."


def _kid_trade_line(po: dict, budget: float) -> str:
    """Beginner 'can you even trade it' bullet — no spread/liquidity jargon."""
    cs = po.get("contracts", [])
    if not cs:
        return f"To trade: there is no cheap same-day SPY contract under ${budget:.0f} right now."
    if any(c.get("above_budget") for c in cs):
        return f"To trade: nothing fits ${budget:.0f} today — the cheapest one costs more than that."
    return (f"To trade: practice contracts exist (cheapest about ${cs[0]['premium_cost_estimate']:.0f}), "
            "but that does NOT make the setup good.")


def _has_budget_contract(po: dict) -> bool:
    return any(not c.get("above_budget") for c in (po.get("contracts") or []))


def _kid_change_lines(verdict: str, po: dict, budget: float) -> list[str]:
    """'What would change the answer' — state-aware beginner bullets (don't ask for a cheap
    contract to appear when one already exists)."""
    has_fit = _has_budget_contract(po)
    contract_line = (f"And a cheap contract under ${budget:.0f} would need to stay available and easy to trade."
                     if has_fit else
                     f"And a cheap contract under ${budget:.0f} would need to show up (none does right now).")
    if verdict == "OBSERVE":
        return ["SPY would need to clearly move UP or DOWN and stay there.",
                "The online crowd would need to clearly support that same direction.",
                contract_line]
    flip = "DOWN" if verdict == "CALL-leaning" else "UP"
    return [f"If SPY turns {flip} instead and stays there, the hint flips.",
            "If the online crowd swings the other way, the hint weakens.",
            contract_line]


_CARD_ACTION = {
    "OBSERVE": "🛑 do nothing",
    "CALL-leaning": "✅ practice CALL lean (bets it goes up)",
    "PUT-leaning": "⚠️ practice PUT lean (bets it goes down)",
}


def _card_reason(card: dict) -> str:
    """Plain-language one-clause reason for a ticker card (no jargon)."""
    if card.get("note") == "no same-day options":
        return "no same-day options to practice with"
    pd = card.get("price_dir")
    if pd is None:
        return "couldn't check its price today"
    if card["verdict"] == "OBSERVE":
        return "price isn't clearly moving" if pd == "neutral" else "price and the crowd don't agree"
    return "price is moving " + ("up" if pd == "bullish" else "down")


def _card_line(card: dict, budget: float) -> str:
    """One compact beginner card line. Contract example only on a CALL/PUT lean (never OBSERVE)."""
    action = _CARD_ACTION.get(card["verdict"], "🛑 do nothing")
    line = f"{card['ticker']} ({card.get('mentions', 0)} posts): {action} — {_card_reason(card)}"
    contracts = card.get("contracts", [])
    if card["verdict"] != "OBSERVE" and contracts:
        k = contracts[0]
        tag = " ABOVE budget" if k.get("above_budget") else ""
        line += (f"  [example only: a {k['option_type'].upper()} ~${k['premium_cost_estimate']:.0f}{tag}]")
    return line


def _json_klist(klist: list | None) -> list[dict]:
    return [{"type": k.get("option_type"), "strike": k.get("strike"),
             "cost_est": k.get("premium_cost_estimate"), "above_budget": bool(k.get("above_budget"))}
            for k in (klist or [])]


def _json_contracts(o: dict) -> list[dict]:
    return _json_klist((o or {}).get("contracts"))


def format_report_json(report: dict) -> str:
    """Compact, machine-ingestible JSON of the 0DTE read for an agent — signal only, no human-safety
    prose, banners, or disclaimers. `direction` is up/down/none; everything is plain JSON types."""
    sc = report.get("scorecard") or {}
    pt = report.get("spy_trend", {}) or {}
    si = report.get("social_intent") or {}
    src = report.get("sources", {}) or {}
    xs, rd = src.get("x", {}) or {}, src.get("reddit", {}) or {}
    _dir = {"CALL-leaning": "up", "PUT-leaning": "down", "OBSERVE": "none"}
    _cdir = {"bullish": "up", "bearish": "down", "neutral": "none"}
    payload = {
        "generated_at": report.get("generated_at"),
        "spy": {
            "verdict": sc.get("verdict", "OBSERVE"),
            "direction": _dir.get(sc.get("verdict", "OBSERVE"), "none"),
            "confidence": sc.get("confidence", "low"),
            "reasons": sc.get("reasons", []),
            "price": {"ok": pt.get("ok"), "pct_vs_prev_close": pt.get("pct_vs_prev_close"),
                      "above_vwap": pt.get("above_vwap")},
            "social": {"intent": si.get("intent"), "n_docs": si.get("n_docs", 0)},
            "contracts": _json_contracts(report.get("paper_options", {}) or {}),
            "options_status": (report.get("paper_options", {}) or {}).get("status"),
        },
        "freshness": report.get("freshness_window", {}),
        "sources": {
            "reddit": {"n_posts": rd.get("n_posts"), "n_quality": rd.get("n_quality"),
                       "daily_thread": (report.get("daily_thread", {}) or {}).get("status")},
            "x": {"n_posts": xs.get("n_posts"), "status": xs.get("status"),
                  "cross_source_enrich": xs.get("enrich_status"), "n_enrich": xs.get("n_enrich", 0)},
        },
        # Every chattered ticker carries its full resolved same-day chain (price-implied side), so an
        # executing agent sees the tradable contracts for each name — not just the directional ones.
        "top_chatter": [
            {"ticker": c.get("ticker"), "mentions": c.get("mentions", 0), "verdict": c.get("verdict"),
             "direction": _cdir.get(c.get("price_dir"), "none"),
             # Employer/compliance restriction (e.g. NVDA): contracts are forced empty regardless.
             "restricted": bool(c.get("restricted")),
             "restricted_reason": c.get("restricted_reason"),
             "options_expiry": c.get("options_expiry"), "options_status": c.get("options_status"),
             "contracts": [] if c.get("restricted") else _json_klist(c.get("available_contracts"))}
            for c in (report.get("top_chatter") or [])
        ],
        # Top-level candidate (restored for downstream consumers). _select_odte_candidate never
        # returns a restricted symbol, so the candidate is guaranteed non-restricted / tradable-eligible.
        "candidate": report.get("candidate"),
    }
    return json.dumps(payload, indent=2, default=str)


def format_report(report: dict) -> str:
    """Beginner-friendly text rendering of build_odte_social_report(): answers (1) what to do now,
    (2) why in 3 tiny bullets, (3) what would change the answer — in plain language a newcomer can
    follow. PRACTICE / PAPER ONLY — this tool never buys or sells anything and gives no instructions."""
    sc = report.get("scorecard") or {"verdict": "OBSERVE", "confidence": "low", "reasons": []}
    pt = report.get("spy_trend", {}) or {}
    si = report.get("social_intent")
    po = report.get("paper_options", {}) or {}
    budget = float(report.get("budget_dollars", 50))
    verdict = sc.get("verdict", "OBSERVE")
    action, meaning = _WHATDO.get(verdict, _WHATDO["OBSERVE"])
    strength = _STRENGTH.get(sc.get("confidence", "low"), "weak")

    lines = [
        "=" * 72,
        "SPY 0DTE — PRACTICE / PAPER ONLY  (this tool never buys or sells anything)",
        "=" * 72,
        f"WHAT TO DO NOW:  {action}",
        f"   What that means: {meaning}",
        f"   Confidence: {strength}.  Doing nothing is always fine — it's the default.",
        "",
        "WHY (3 quick reasons):",
        f"  • {_kid_price_line(pt)}",
        f"  • {_kid_social_line(si)}",
        f"  • {_kid_trade_line(po, budget)}",
        "",
        "WHAT WOULD CHANGE THIS:",
    ]
    for ch in _kid_change_lines(verdict, po, budget):
        lines.append(f"  • {ch}")

    # How fresh is the info (plain words).
    fw = report.get("freshness_window", {})
    rd, xs = report["sources"]["reddit"], report["sources"]["x"]
    lines.append("")
    lines.append(f"HOW FRESH: {rd['n_posts']} fresh Reddit posts (and {xs['n_posts']} from X) "
                 f"{_freshness_phrase(fw)}; older or spammy posts were thrown out.")
    if xs.get("n_enrich", 0):
        lines.append(f"   Cross-source: also searched X for WSB's hot tickers ({xs.get('enrich_status', '')}).")

    # Practice-only contract examples — rounded dollars, plainly labeled. Shown ONLY when there is
    # an actual CALL/PUT lean AND the surfaced contracts match that direction; on DO NOTHING
    # (OBSERVE) we SUPPRESS them so the examples never contradict the verdict.
    want_type = {"CALL-leaning": "call", "PUT-leaning": "put"}.get(verdict)
    contracts = po.get("contracts", [])
    aligned = [k for k in contracts if k.get("option_type") == want_type] if want_type else []
    lines += ["", f"IF YOU PRACTICE ANYWAY  (these are examples only, NOT instructions; "
              f"you could lose the whole ${budget:.0f}):"]
    if want_type is None:
        lines.append("  • No example shown — today's read is DO NOTHING. "
                     "Contract examples appear only when there's a clear CALL or PUT lean.")
    elif aligned:
        for k in aligned:
            tag = "  [ABOVE BUDGET — costs more than the limit]" if k.get("above_budget") else ""
            bet = "SPY goes UP" if k["option_type"] == "call" else "SPY goes DOWN"
            lines.append(
                f"  • A {k['option_type'].upper()} at the ${k['strike']:g} price (bets {bet}) — "
                f"costs about ${k['premium_cost_estimate']:.0f} for one.{tag}")
    else:
        lines.append(f"  • Nothing cheap enough today — no same-day SPY {want_type} under "
                     f"${budget:.0f} (market may be closed, or none were cheap/easy to trade).")

    # TOP CHATTER — compact paper cards for the other names people talked about (after SPY).
    lines += ["", "TOP CHATTER  (other tickers people talked about today — context, not advice):"]
    lines.append("  " + report.get("top_chatter_caveat", "limited source coverage."))
    dt = report.get("daily_thread", {}) or {}
    lines.append(f"  Daily thread comments: {dt.get('status', 'n/a')}.")
    cards = report.get("top_chatter", [])
    if cards:
        for c in cards:
            lines.append("  • " + _card_line(c, budget))
    else:
        lines.append("  • none surfaced from today's filtered chatter "
                     "(the configured sources are SPY/QQQ-biased).")

    lines += ["", "REMEMBER:"]
    for r in report.get("risk_notes", []):
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
    """Per-text ticker symbols (cashtags normalized; stopwords dropped; restricted to `allowed`)."""
    out: list[str] = []
    for sym in _iter_symbols(text):
        if sym in _STOPWORDS:
            continue
        if allowed is not None and sym not in allowed:
            continue
        out.append(sym)
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
