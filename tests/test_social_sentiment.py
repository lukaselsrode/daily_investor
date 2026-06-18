"""
tests/test_social_sentiment.py — 0DTE social-sentiment watchlist (analysis-only).

No network: requests is monkeypatched and the cache dir is redirected to tmp. Covers ticker
extraction (stopwords excluded), transparent scoring, Reddit fetch parse + graceful failure,
X skip without token, the report shape (candidate/evidence/disclaimer/risk), and the
hard guardrail that the module places no orders / imports no execution code.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import data.social_sentiment as ss


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.text = payload if isinstance(payload, str) else ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _reddit_payload(created_utc: float | None = None):
    import time as _t
    cu = _t.time() if created_utc is None else created_utc  # fresh by default (day-of filter)

    def _post(title, body, score, pl):
        return {"data": {"title": title, "selftext": body, "score": score,
                         "num_comments": 5, "permalink": pl, "created_utc": cu}}
    return {"data": {"children": [
        _post("GME to the moon 🚀 buy calls", "loading up", 500, "/r/wsb/1"),
        _post("SPY calls printing, bullish breakout", "long SPY", 300, "/r/wsb/2"),
        _post("GME squeeze incoming, more calls", "buy buy", 200, "/r/wsb/3"),
        _post("The DD on YOLO PUTS — I sold THE top", "puts crash dump", 50, "/r/wsb/4"),
    ]}}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # Redirect cache to tmp so tests never touch the real data dir or stale cache.
    monkeypatch.setattr(ss, "_CACHE_DIR", tmp_path / "social_cache")
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    # Clear Reddit OAuth creds by default so anonymous-path tests never hit oauth.reddit.com.
    for _v in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(_v, raising=False)


# America/New_York anchors for the market-session freshness tests. June 2025 is EDT (UTC-4);
# 2025-06-13 is a Friday, 06-15 a Sunday, 06-16 a Monday — no DST/weekend ambiguity.
from datetime import datetime as _datetime  # noqa: E402
from zoneinfo import ZoneInfo as _ZoneInfo  # noqa: E402

_ET = _ZoneInfo("America/New_York")


def _et(y, mo, d, h, mi):
    """A tz-aware America/New_York datetime."""
    return _datetime(y, mo, d, h, mi, tzinfo=_ET)


def _ets(y, mo, d, h, mi):
    """Epoch seconds for an America/New_York wall-clock time (what Reddit's created_utc holds)."""
    return _et(y, mo, d, h, mi).timestamp()


# ---------------------------------------------------------------------------

def test_extract_excludes_stopwords_and_counts_tickers():
    texts = ["GME to the moon buy calls", "SPY calls, the DD on YOLO PUTS"]
    c = ss.extract_ticker_mentions(texts)
    assert c["GME"] == 1 and c["SPY"] == 1
    for sw in ("THE", "DD", "YOLO", "PUTS", "BUY", "CALLS", "TO"):
        assert sw not in c


def test_score_social_sentiment_sign_and_momentum():
    docs = [
        {"text": "GME calls moon buy bullish", "ts": 10.0},
        {"text": "GME puts short bearish crash", "ts": 0.0},
        {"text": "SPY puts dump short", "ts": 10.0},
    ]
    mentions = ss.extract_ticker_mentions([d["text"] for d in docs])
    sc = ss.score_social(mentions, docs)
    assert sc["GME"]["sentiment"] == pytest.approx(0.0, abs=1e-9) or sc["GME"]["bull"] > 0
    assert sc["SPY"]["sentiment"] < 0          # SPY only bearish words
    assert 0.0 <= sc["GME"]["momentum"] <= 1.0


def test_fetch_reddit_parses_and_caches(monkeypatch):
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(_reddit_payload()))
    posts = ss.fetch_reddit_posts(limit=10)
    assert len(posts) == 4
    assert posts[0]["title"].startswith("GME")
    assert posts[0]["permalink"].startswith("https://www.reddit.com/r/wsb/")
    # second call served from cache even if network now fails
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    assert len(ss.fetch_reddit_posts(limit=10)) == 4


def test_fetch_reddit_fails_closed(monkeypatch):
    def _boom(*a, **k):
        raise ss.requests.RequestException("no network")
    monkeypatch.setattr(ss.requests, "get", _boom)
    assert ss.fetch_reddit_posts(limit=10) == []


def test_fetch_reddit_falls_back_to_atom_feed(monkeypatch):
    """Reddit often blocks anonymous JSON with 403; the public Atom feed keeps live runs usable."""
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>SPY calls into close</title>
        <content type="html">QQQ moon</content>
        <updated>2023-01-15T12:00:00+00:00</updated>
        <author><name>u/tester</name></author>
        <link href="https://www.reddit.com/r/wallstreetbets/comments/abc/test/" />
      </entry>
    </feed>"""

    calls = []

    def _fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith(".json"):
            raise ss.requests.HTTPError("blocked")
        return _FakeResp(atom)

    monkeypatch.setattr(ss.requests, "get", _fake_get)
    posts = ss.fetch_reddit_posts(limit=10)
    assert len(posts) == 1
    assert posts[0]["title"] == "SPY calls into close"
    assert posts[0]["permalink"].startswith("https://www.reddit.com/")
    assert any(url.endswith(".rss") for url in calls)


def test_fetch_x_skips_without_token():
    posts, status = ss.fetch_x_mentions("SPY")
    assert posts == [] and "X_BEARER_TOKEN not set" in status


def test_fetch_x_uses_official_api_with_token(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "abc")
    captured = {}

    def _fake_get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        captured["auth"] = (headers or {}).get("Authorization", "")
        return _FakeResp({"data": [{"text": "SPY 0DTE calls", "created_at": "t", "id": "1"}]})

    monkeypatch.setattr(ss.requests, "get", _fake_get)
    posts, status = ss.fetch_x_mentions("SPY")
    assert status == "ok" and len(posts) == 1
    assert captured["url"].startswith("https://api.twitter.com/2/")
    assert captured["auth"] == "Bearer abc"


def test_build_report_shape_and_disclaimer(monkeypatch):
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(_reddit_payload()))
    params = {"sources": ["reddit"], "subreddit": "wallstreetbets", "reddit_limit": 25,
              "max_tickers": 10, "min_mentions": 1, "budget_dollars": 50,
              "core_universe": ["SPY", "QQQ", "GME"], "include_paper_options": False}
    rep = ss.build_odte_social_report(allow_fetch=True, params=params)
    assert "not financial advice" in rep["disclaimer"].lower()
    assert rep["sources"]["x"]["status"].startswith("disabled")
    assert rep["candidate"] is not None
    assert rep["candidate"]["ticker"] in ("GME", "SPY")
    assert rep["candidate"]["evidence"] and rep["candidate"]["evidence"][0]["url"].startswith("http")
    assert any("lose" in r.lower() for r in rep["risk_notes"])
    # text rendering works and carries the paper-only banner
    txt = ss.format_report(rep)
    assert "PAPER ONLY" in txt


def _raw_reddit():
    return [
        {"title": "GME to the moon buy calls", "selftext": "loading SPY too",
         "permalink": "https://www.reddit.com/r/wsb/1", "score": 500, "num_comments": 5,
         "created_utc": 1_700_000_000.0},
        {"title": "the DD on YOLO PUTS", "selftext": "sold THE top",
         "permalink": "https://www.reddit.com/r/wsb/2", "score": 50, "num_comments": 2,
         "created_utc": 1_700_000_500.0},
    ]


def test_normalize_social_items_news_schema():
    items = ss.normalize_social_items(reddit_posts=_raw_reddit())
    assert items, "expected normalized items"
    a = items[0]
    # data/news.py article-dict schema keys present
    for k in ("title", "publisher", "link", "summary", "pub_date", "formatted_date",
              "related_symbols", "api_source"):
        assert k in a, f"missing news-schema key {k}"
    assert a["api_source"] == "reddit_wsb"
    # ticker extraction: GME + SPY captured, stopwords (THE/DD/YOLO/PUTS) excluded
    syms = set(a["related_symbols"]) | set(items[1]["related_symbols"])
    assert "GME" in syms and "SPY" in syms
    assert not ({"THE", "DD", "YOLO", "PUTS"} & syms)


def test_merge_preserves_news_schema_and_includes_social():
    import json as _json

    import pandas as pd
    existing = pd.DataFrame([{"symbol": "AAPL",
                              "news": _json.dumps([{"title": "Apple earnings", "summary": "beat",
                                                    "api_source": "robinhood"}])}])
    items = ss.normalize_social_items(reddit_posts=_raw_reddit())
    merged = ss.merge_social_into_news(existing, ss.social_items_by_symbol(items))
    # schema preserved exactly
    assert list(merged.columns) == ["symbol", "news"]
    by = {r["symbol"]: _json.loads(r["news"]) for _, r in merged.iterrows()}
    # existing AAPL news preserved untouched
    assert any(x.get("api_source") == "robinhood" for x in by["AAPL"])
    # social merged in for a mentioned ticker (e.g. GME or SPY), as reddit_wsb items
    social_syms = [s for s in by if any(x.get("api_source") == "reddit_wsb" for x in by[s])]
    assert "GME" in social_syms or "SPY" in social_syms


def test_social_flows_into_existing_sentiment_context():
    """Normalized social items render in data.sentiment._format_news — the SAME context path
    PortfolioManager active-sleeve sentiment consumes (proves enrichment reaches the substrate)."""
    sentiment = pytest.importorskip("data.sentiment")
    items = ss.normalize_social_items(reddit_posts=_raw_reddit())
    by_sym = ss.social_items_by_symbol(items)
    sym = "GME" if "GME" in by_sym else next(iter(by_sym))
    rendered = sentiment._format_news({sym: by_sym[sym]}, sym)
    assert "GME to the moon" in rendered or by_sym[sym][0]["title"][:20] in rendered


def test_get_news_df_always_enriches_social(monkeypatch):
    """get_news_df ALWAYS merges normalized social into the news dataset (independent of
    options_social.enabled) so the active-sleeve sentiment substrate carries social provenance.
    The only opt-out is disable_social_news_enrichment. Wiring + schema + allow-list test."""
    import json as _json

    import data.news as news
    import util
    store: dict = {}
    monkeypatch.setattr(news, "read_data_as_pd", lambda ds: store.get(ds))
    monkeypatch.setattr(news, "store_data_as_csv",
                        lambda name, cols, df: store.__setitem__(name, df.copy()))
    monkeypatch.setattr(news, "get_news_for_tickers_by_symbol",
                        lambda tickers, max_articles=3: {
                            "AAPL": [{"title": "Apple earnings", "summary": "beat",
                                      "api_source": "yfinance"}]})
    # Post mentions SPY (core, allowed) + FOMO/GETOUT (all-caps noise, NOT allowed).
    monkeypatch.setattr(ss, "fetch_reddit_posts",
                        lambda **k: [{"title": "SPY calls moon FOMO GETOUT", "selftext": "buy",
                                      "permalink": "u", "score": 9, "num_comments": 1,
                                      "created_utc": 1.7e9}])
    # Force agg_data absent so the allow-list is core_universe (SPY/QQQ) + news symbols (AAPL).
    import data.cache as _cache
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    monkeypatch.setitem(util.OPTIONS_SOCIAL_PARAMS, "sources", ["reddit"])
    # enabled stays FALSE — enrichment must happen anyway (decoupled from the 0DTE report gate).
    monkeypatch.setitem(util.OPTIONS_SOCIAL_PARAMS, "enabled", False)
    monkeypatch.setitem(util.OPTIONS_SOCIAL_PARAMS, "disable_social_news_enrichment", False)

    news.get_news_df(["AAPL"], force_refresh=True)
    rows = {r["symbol"]: _json.loads(r["news"]) for _, r in store["news"].iterrows()}
    assert list(store["news"].columns) == ["symbol", "news"]               # schema preserved
    assert any(a.get("api_source") == "reddit_wsb"
               for a in rows.get("SPY", [])), "SPY social not merged though enrichment is always-on"
    assert any(a.get("api_source") == "yfinance" for a in rows.get("AAPL", [])), "base news lost"
    # false-positive all-caps must NOT create rows (allowed-universe filter)
    assert "FOMO" not in rows and "GETOUT" not in rows, f"false positives leaked: {set(rows)}"

    # explicit opt-out -> unchanged (no social merged)
    monkeypatch.setitem(util.OPTIONS_SOCIAL_PARAMS, "disable_social_news_enrichment", True)
    news.get_news_df(["AAPL"], force_refresh=True)
    rows2 = {r["symbol"]: _json.loads(r["news"]) for _, r in store["news"].iterrows()}
    assert not any(a.get("api_source") == "reddit_wsb"
                   for items in rows2.values() for a in items), \
        "social merged despite disable_social_news_enrichment"


def test_fetch_reddit_uses_official_oauth_when_creds_present(monkeypatch):
    """When REDDIT_CLIENT_ID/SECRET are set, fetch goes through the official OAuth app-only API
    (oauth.reddit.com with a bearer token) and never touches the anonymous JSON endpoint."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "daily-investor-test/1.0")
    calls: dict = {"token_auth": None, "get_urls": [], "bearer": None}

    def _fake_post(url, auth=None, data=None, headers=None, timeout=None):
        calls["token_auth"] = auth
        assert url == ss._REDDIT_TOKEN_URL
        assert data == {"grant_type": "client_credentials"}
        return _FakeResp({"access_token": "tok123", "token_type": "bearer"})

    def _fake_get(url, headers=None, params=None, timeout=None):
        calls["get_urls"].append(url)
        calls["bearer"] = (headers or {}).get("Authorization")
        return _FakeResp({"data": {"children": [
            {"data": {"title": "SPY calls", "selftext": "buy", "score": 12, "num_comments": 3,
                      "permalink": "/r/wsb/o", "created_utc": 1.7e9, "author": "u/x"}}]}})

    monkeypatch.setattr(ss.requests, "post", _fake_post)
    monkeypatch.setattr(ss.requests, "get", _fake_get)

    posts = ss.fetch_reddit_posts(limit=10)
    assert calls["token_auth"] == ("cid", "secret")
    assert calls["bearer"] == "bearer tok123"
    assert any("oauth.reddit.com" in u for u in calls["get_urls"]), calls["get_urls"]
    assert not any(u.endswith(".json") for u in calls["get_urls"]), "anonymous JSON must be skipped"
    assert len(posts) == 1 and posts[0]["title"] == "SPY calls"
    assert posts[0]["api_source"] == "reddit_oauth" and posts[0]["score"] == 12


def test_fetch_reddit_oauth_falls_back_to_public_json_on_token_error(monkeypatch):
    """If OAuth creds are present but the token call fails, fetch falls back to public JSON
    (then Atom/RSS). Proves the fallback ORDER without requiring a browser/HTML path."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")

    def _boom_post(*a, **k):
        raise ss.requests.HTTPError("401 unauthorized")

    def _fake_get(url, headers=None, params=None, timeout=None):
        assert url.endswith(".json"), f"should hit public JSON, got {url}"
        return _FakeResp({"data": {"children": [
            {"data": {"title": "QQQ puts", "selftext": "", "score": 5, "num_comments": 1,
                      "permalink": "/r/wsb/j", "created_utc": 1.7e9}}]}})

    monkeypatch.setattr(ss.requests, "post", _boom_post)
    monkeypatch.setattr(ss.requests, "get", _fake_get)

    posts = ss.fetch_reddit_posts(limit=10)
    assert len(posts) == 1 and posts[0]["title"] == "QQQ puts"
    assert posts[0]["api_source"] == "reddit_json"


def _comments_payload():
    """Reddit /comments/{id} JSON shape: [post_listing, comments_listing]."""
    return [
        {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": {"id": "abc"}}]}},
        {"kind": "Listing", "data": {"children": [
            {"kind": "t1", "data": {"body": "SPY calls printing today", "score": 42, "author": "u/a"}},
            {"kind": "t1", "data": {"body": "loading QQQ puts", "score": 7, "author": "u/b"}},
            {"kind": "more", "data": {"children": ["x1", "x2"]}},   # load-more node — must be skipped
        ]}},
    ]


def test_fetch_reddit_comments_parses(monkeypatch):
    """Comments fetch parses top-level (t1) comment bodies/scores and skips load-more nodes."""
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(_comments_payload()))
    comments = ss.fetch_reddit_comments("abc", limit=5)
    assert len(comments) == 2, comments
    assert comments[0]["body"].startswith("SPY calls") and comments[0]["score"] == 42
    assert all("body" in c and c["body"] for c in comments)


def test_fetch_reddit_comments_fail_closed(monkeypatch):
    """Comments fetch fails closed: network error → []; empty id → [] with no request at all."""
    def _boom(*a, **k):
        raise ss.requests.RequestException("no net")
    monkeypatch.setattr(ss.requests, "get", _boom)
    assert ss.fetch_reddit_comments("abc") == []
    assert ss.fetch_reddit_comments("") == []
    assert ss.fetch_reddit_comments("abc", allow_fetch=False) == []


def test_fold_comments_into_posts(monkeypatch):
    """Top comments are folded into the top posts' selftext (bounded), posts without an id are
    left untouched, and the originals are not mutated."""
    monkeypatch.setattr(ss, "fetch_reddit_comments",
                        lambda pid, limit=5, allow_fetch=True: [
                            {"body": "buy SPY calls", "score": 10, "author": "u/a"}])
    posts = [{"id": "p1", "title": "SPY thread", "selftext": "base", "score": 100},
             {"id": "", "title": "no id", "selftext": "x", "score": 5}]
    out = ss._fold_comments_into_posts(posts, top_posts=3, per_post=5, allow_fetch=True)
    assert "Top comments: buy SPY calls" in out[0]["selftext"]
    assert out[0]["comments_sampled"] == 1
    assert out[1]["selftext"] == "x" and "comments_sampled" not in out[1]
    assert posts[0]["selftext"] == "base", "original post must not be mutated"


def test_enrich_folds_comments_when_enabled(monkeypatch):
    """When reddit_comments_enrich is on, comment text is folded into the post → it reaches the
    news substrate and its tickers (e.g. QQQ from a comment) surface as related symbols."""
    import json as _json

    import data.cache as _cache
    monkeypatch.setattr(ss, "fetch_reddit_posts",
                        lambda **k: [{"id": "p1", "title": "SPY calls", "selftext": "",
                                      "permalink": "u", "score": 50, "num_comments": 9,
                                      "created_utc": 1.7e9}])
    monkeypatch.setattr(ss, "fetch_reddit_comments",
                        lambda pid, limit=5, allow_fetch=True: [
                            {"body": "QQQ puts too", "score": 3, "author": "u/z"}])
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    merged = ss.enrich_news_with_social(news_df=None, allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"],
        "reddit_comments_enrich": True, "reddit_comments_top_posts": 3,
        "reddit_comments_per_post": 5})
    rows = {r["symbol"]: _json.loads(r["news"]) for _, r in merged.iterrows()}
    spy_summary = " ".join(a.get("summary", "") for a in rows.get("SPY", []))
    assert "Top comments: QQQ puts too" in spy_summary, rows
    assert "QQQ" in rows, "ticker from a folded comment should surface as a related symbol"


def test_enrich_skips_comments_when_disabled(monkeypatch):
    """Default off: comments are NOT fetched/folded unless reddit_comments_enrich is true."""
    monkeypatch.setattr(ss, "fetch_reddit_posts",
                        lambda **k: [{"id": "p1", "title": "SPY calls", "selftext": "",
                                      "permalink": "u", "score": 50, "num_comments": 9,
                                      "created_utc": 1.7e9}])

    def _must_not_call(*a, **k):
        raise AssertionError("comments fetched while reddit_comments_enrich is off")
    monkeypatch.setattr(ss, "fetch_reddit_comments", _must_not_call)
    import data.cache as _cache
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    merged = ss.enrich_news_with_social(news_df=None, allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"]})  # reddit_comments_enrich absent → off
    assert "SPY" in {r["symbol"] for _, r in merged.iterrows()}


def test_report_evidence_includes_both_reddit_and_x(monkeypatch):
    """ODTE evidence surfaces BOTH Reddit and X for a candidate, using the same row fields
    (source/title/score/url/age). Heuristics stay in the report — they are NOT fed to active
    sentiment (that path is covered by the _format_news uniformity tests)."""
    import data.cache as _cache
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    now = _et(2025, 6, 16, 10, 30)            # Monday, market hours
    reddit_ts = _ets(2025, 6, 16, 10, 0)      # Monday 10:00 ET — fresh

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "twitter.com" in url:
            return _FakeResp({"data": [{"text": "SPY 0DTE calls ripping",
                                        "created_at": "2025-06-16T14:00:00.000Z", "id": "9"}]})
        return _FakeResp({"data": {"children": [
            {"data": {"title": "SPY calls bullish", "selftext": "", "score": 80,
                      "num_comments": 2, "permalink": "/r/wsb/s", "created_utc": reddit_ts}}]}})

    monkeypatch.setattr(ss.requests, "get", _fake_get)
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    rep = ss.build_odte_social_report(allow_fetch=True, now=now, params={
        "sources": ["reddit", "x"], "core_universe": ["SPY"], "min_mentions": 1,
        "x_query": "SPY", "freshness_mode": "market_window", "include_paper_options": False})
    assert rep["candidate"]["ticker"] == "SPY"
    ev_sources = {e["source"] for e in rep["candidate"]["evidence"]}
    assert {"reddit", "x"} <= ev_sources, ev_sources
    assert rep["sources"]["x"]["n_posts"] >= 1
    for e in rep["candidate"]["evidence"]:
        assert {"title", "score", "url", "source"} <= set(e), e


# ---------------------------------------------------------------------------
# Spam / quality filtering (transparent, no ML, no network)
# ---------------------------------------------------------------------------

def test_spam_predicates_are_transparent():
    """_is_spam / _has_options_context are simple inspectable rules: promo/scam, off-topic crypto
    (unless options-focused), and shotgun cashtag/@ spam are flagged; legit options tweets aren't."""
    assert ss._is_spam("Join my Telegram for VIP 0DTE signals 100X gains")
    assert ss._is_spam("BTC to the moon — buy bitcoin now, crypto season")        # crypto, no opts
    assert ss._is_spam("Class action: investors who purchased $QQQ — law firm")   # legal blast
    assert ss._is_spam("$AAA $BBB $CCC $DDD $EEE $FFF $GGG all squeezing")         # shotgun cashtags
    assert not ss._is_spam("SPY 0DTE calls into FOMC, watching the 540 strike")
    assert not ss._is_spam("hedging my SPY puts against BTC exposure")            # crypto + options
    assert ss._has_options_context("buying SPY puts at the 540 strike")
    assert not ss._has_options_context("SPY is a great long-term hold")


def test_quality_filter_drops_spam_and_dedupes_keeps_legit():
    """_quality_filter drops promo/legal spam and near-duplicates and (for ODTE) requires an
    allowed ticker + options context — keeping only the genuine 0DTE signal."""
    docs = [
        {"text": "Join my Telegram VIP for free signals 100X gains $SPY", "id": 1},
        {"text": "Class action lawsuit: investors who purchased $QQQ may recover", "id": 2},
        {"text": "Risk management is essential whenever you trade SPY and QQQ", "id": 3},  # no opts
        {"text": "SPY 0DTE calls ripping into the close, 540 strike", "id": 4},
        {"text": "SPY 0DTE calls ripping into the close, 540 strike!!!", "id": 5},  # near-dup of 4
    ]
    kept = ss._quality_filter(docs, lambda d: d["text"], allowed={"SPY", "QQQ"},
                              require_options_context=True)
    assert [d["id"] for d in kept] == [4], [d["id"] for d in kept]


def test_odte_report_filters_x_spam_and_does_not_inflate_mentions(monkeypatch):
    """Reproduces the live bug: X returns mostly promo/legal/off-topic noise mentioning SPY. After
    filtering, SPY's mention count reflects only the genuine 0DTE tweet (not the spam), and no spam
    appears in the candidate's evidence."""
    import data.cache as _cache
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    now = _et(2025, 6, 16, 10, 30)
    ts = "2025-06-16T14:00:00.000Z"   # Monday 10:00 ET — fresh
    noise = [
        "Join my Telegram VIP for free SPY signals 100X",
        "BTC and SPY pump! join WhatsApp for crypto signals",
        "Class action: investors who purchased SPY may recover — law firm",
        "Risk management is key whenever you trade SPY and QQQ",   # generic, no options context
    ]

    def _fake_get(url, headers=None, params=None, timeout=None):
        if "twitter.com" in url:
            data = [{"text": t, "created_at": ts, "id": str(i)} for i, t in enumerate(noise)]
            data.append({"text": "SPY 0DTE calls bullish into FOMC, 540 strike",
                         "created_at": ts, "id": "99"})
            return _FakeResp({"data": data})
        return _FakeResp({"data": {"children": []}})   # no reddit

    monkeypatch.setattr(ss.requests, "get", _fake_get)
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    rep = ss.build_odte_social_report(allow_fetch=True, now=now, params={
        "sources": ["x"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
        "x_query": "SPY", "freshness_mode": "market_window", "include_paper_options": False})
    spy = next((t for t in rep["top_tickers"] if t["ticker"] == "SPY"), None)
    assert spy is not None and spy["mentions"] == 1, rep["top_tickers"]   # NOT inflated by 4 noise
    assert rep["sources"]["x"]["n_filtered"] == 4
    assert rep["sources"]["x"]["n_quality"] == 1
    assert rep["candidate"]["ticker"] == "SPY"
    ev_text = " ".join(e["title"].lower() for e in rep["candidate"]["evidence"])
    for bad in ("telegram", "whatsapp", "class action", "law firm", "risk management"):
        assert bad not in ev_text, ev_text


def test_enrichment_does_not_inject_spam(monkeypatch):
    """Active-sleeve news enrichment drops promo spam (Telegram/100X) before merging, so the
    sentiment substrate never carries it — but keeps a legitimate SPY options post."""
    import json as _json

    import data.cache as _cache
    monkeypatch.setattr(ss, "fetch_reddit_posts", lambda **k: [
        {"id": "a", "title": "Join my Telegram VIP free SPY signals 100X", "selftext": "",
         "permalink": "u1", "score": 9, "num_comments": 1, "created_utc": 1.7e9},
        {"id": "b", "title": "SPY breakout, buying calls", "selftext": "long SPY",
         "permalink": "u2", "score": 20, "num_comments": 2, "created_utc": 1.7e9},
    ])
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    merged = ss.enrich_news_with_social(news_df=None, allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"]})
    rows = {r["symbol"]: _json.loads(r["news"]) for _, r in merged.iterrows()}
    titles = " ".join(a.get("title", "") for a in rows.get("SPY", [])).lower()
    assert "telegram" not in titles and "100x" not in titles, titles
    assert "breakout" in titles, titles   # the legitimate options post survives


def test_formatted_sentiment_includes_source_provenance_uniformly():
    """data.sentiment._format_news must render social items UNIFORMLY as articles: it surfaces
    source/provenance (api_source, link), raw engagement counts, and the title/summary text — but
    must NOT inject a precomputed social bullish/bearish label or net score, so the LLM judges
    social and news together from raw text rather than a separate aggregated social signal."""
    sentiment = pytest.importorskip("data.sentiment")
    items = ss.normalize_social_items(reddit_posts=_raw_reddit())
    by_sym = ss.social_items_by_symbol(items)
    sym = "GME" if "GME" in by_sym else next(iter(by_sym))
    rendered = sentiment._format_news({sym: by_sym[sym]}, sym)
    # provenance is visible (source + link + factual engagement counts)
    assert "Source: reddit_wsb" in rendered, rendered
    assert "Engagement:" in rendered and "upvotes" in rendered, rendered
    assert "Link:" in rendered and "reddit.com" in rendered, rendered
    # but NO precomputed social sentiment label / net is fed to the active-sleeve prompt
    # (raw title/summary text may say anything — we only assert our injected verdict is gone)
    assert "Social:" not in rendered, rendered
    assert "net=" not in rendered, rendered


def test_normalized_social_item_has_no_precomputed_sentiment():
    """The normalized article dict carries factual provenance only — no social_sentiment/social_net
    fields — so there is a single uniform article shape and no separate social aggregation path."""
    items = ss.normalize_social_items(reddit_posts=_raw_reddit())
    assert items
    for a in items:
        assert "social_sentiment" not in a and "social_net" not in a, a
        assert a["api_source"] == "reddit_wsb"
        assert isinstance(a.get("engagement"), dict)


def test_x_normalized_pub_date_reflects_created_at():
    """X items must carry the tweet's created_at as pub_date (not 'now') so enriched news is
    time-correct for the active-sleeve sentiment substrate."""
    items = ss.normalize_social_items(
        x_posts=[{"text": "SPY 0DTE calls", "created_at": "2023-01-15T12:00:00.000Z", "id": "1"}])
    assert items and items[0]["api_source"] == "x"
    assert items[0]["pub_date"].startswith("2023-01-15"), items[0]["pub_date"]


def test_config_manager_options_social_property():
    """OptionsSocialConfig is wired into the typed config manager (no fake schema parity)."""
    from config.manager import ConfigManager
    o = ConfigManager.from_dict({"options_social": {
        "enabled": True, "budget_dollars": 25, "sources": ["reddit", "x"],
        "core_universe": ["SPY", "QQQ", "IWM"],
        "disable_social_news_enrichment": True}}).options_social
    assert o.enabled is True and o.budget_dollars == 25.0
    assert o.sources == ("reddit", "x") and "IWM" in o.core_universe
    assert o.disable_social_news_enrichment is True
    # default is False (always-on social enrichment)
    assert ConfigManager.from_dict({}).options_social.disable_social_news_enrichment is False


def test_allowed_universe_filters_false_positives(monkeypatch):
    """build_odte_social_report restricts tickers to core_universe (+ agg if present), so random
    all-caps noise is not ranked. Here agg_data is absent -> core_universe is the allow-list."""
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp({"data": {"children": [
        {"data": {"title": "SPY calls vs NVDA puts and GETOUT FOMO", "selftext": "",
                  "score": 9, "num_comments": 1, "permalink": "/r/wsb/x",
                  "created_utc": 1_700_000_000.0}}]}}))
    # build_odte reads agg_data via a lazy `from data.cache import read_data_as_pd`; force it
    # absent so the allow-list is core_universe only (NVDA/FOMO must then be filtered out).
    import data.cache as _cache
    monkeypatch.setattr(_cache, "read_data_as_pd", lambda ds: None)
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "max_tickers": 10,
        "min_mentions": 1})
    tickers = {t["ticker"] for t in rep["top_tickers"]}
    assert tickers <= {"SPY", "QQQ"}, f"non-core false positives leaked: {tickers}"


def test_market_session_window_boundaries():
    """market_session_window anchors to US sessions in America/New_York (not the UTC day):
    weekend & weekday-premarket → previous Friday/last close; weekday at/after open → today 9:30."""
    fri_close = _et(2025, 6, 13, 16, 0)        # Friday regular close

    # Sunday noon → since Friday's close.
    start, end = ss.market_session_window(_et(2025, 6, 15, 12, 0))
    assert start == fri_close.astimezone(start.tzinfo)
    assert end == _et(2025, 6, 15, 12, 0)

    # Monday pre-open (07:00) → still since Friday's close (weekend sentiment retained).
    start, _ = ss.market_session_window(_et(2025, 6, 16, 7, 0))
    assert start == fri_close.astimezone(start.tzinfo)

    # Monday during market hours (10:00) → since Monday's 09:30 open.
    start, _ = ss.market_session_window(_et(2025, 6, 16, 10, 0))
    assert start == _et(2025, 6, 16, 9, 30).astimezone(start.tzinfo)

    # Wednesday afternoon → since that day's open.
    start, _ = ss.market_session_window(_et(2025, 6, 11, 14, 0))
    assert start == _et(2025, 6, 11, 9, 30).astimezone(start.tzinfo)


def test_market_window_cap_limits_lookback():
    """max_lookback_hours floors how far the window can reach (e.g. across a long holiday gap)."""
    start, _ = ss.market_session_window(_et(2025, 6, 15, 12, 0), max_lookback_hours=4.0)
    assert start == _et(2025, 6, 15, 8, 0).astimezone(start.tzinfo)  # now − 4h, not Friday close


def test_report_sunday_keeps_friday_after_close_drops_before(monkeypatch):
    """Sunday report: a Friday *after-close* SPY post is fresh (kept); a Friday *before-close* QQQ
    post is stale (dropped). Proves weekend windows anchor to the prior regular close, not the UTC
    calendar day."""
    payload = {"data": {"children": [
        {"data": {"title": "SPY calls bullish into the weekend", "selftext": "", "score": 100,
                  "num_comments": 1, "permalink": "/r/wsb/fresh",
                  "created_utc": _ets(2025, 6, 13, 17, 0)}},   # Fri 17:00 ET — after close
        {"data": {"title": "QQQ calls midday Friday", "selftext": "", "score": 90,
                  "num_comments": 1, "permalink": "/r/wsb/stale",
                  "created_utc": _ets(2025, 6, 13, 15, 0)}},    # Fri 15:00 ET — before close
    ]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(
        allow_fetch=True, now=_et(2025, 6, 15, 12, 0), params={
            "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
            "freshness_mode": "market_window", "include_paper_options": False})
    tickers = {t["ticker"] for t in rep["top_tickers"]}
    assert "SPY" in tickers and "QQQ" not in tickers, f"window not anchored to last close: {tickers}"
    assert rep["sources"]["reddit"]["n_stale_filtered"] >= 1
    assert rep["candidate"]["ticker"] == "SPY"
    assert rep["freshness_window"]["mode"] == "market_window"
    assert rep["freshness_window"]["window_start_et"].startswith("2025-06-13T16:00")
    assert "HOW FRESH" in ss.format_report(rep)


def test_report_monday_premarket_keeps_friday_after_close(monkeypatch):
    """Monday pre-market report includes Friday-after-close sentiment (since the prior close)."""
    payload = {"data": {"children": [
        {"data": {"title": "SPY calls bullish Friday close ramp", "selftext": "", "score": 100,
                  "num_comments": 1, "permalink": "/r/wsb/fri",
                  "created_utc": _ets(2025, 6, 13, 17, 30)}},   # Fri 17:30 ET — after close
    ]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(
        allow_fetch=True, now=_et(2025, 6, 16, 7, 0), params={   # Monday 07:00 ET pre-open
            "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
            "freshness_mode": "market_window", "include_paper_options": False})
    assert {t["ticker"] for t in rep["top_tickers"]} == {"SPY"}
    assert rep["candidate"]["ticker"] == "SPY"
    assert rep["freshness_window"]["window_start_et"].startswith("2025-06-13T16:00")
    assert rep["freshness_window"]["anchor"] == "prev_close"
    # Wording must NOT claim 'today' when the window anchors to the previous close.
    txt = ss.format_report(rep)
    how_fresh = next(line for line in txt.splitlines() if line.startswith("HOW FRESH"))
    assert "previous market close" in how_fresh and "06-13 16:00" in how_fresh
    assert "today" not in how_fresh.lower()


def test_report_monday_market_hours_uses_monday_open(monkeypatch):
    """During Monday market hours the window starts at Monday 09:30 — Friday posts are now stale."""
    payload = {"data": {"children": [
        {"data": {"title": "SPY calls bullish now", "selftext": "", "score": 100,
                  "num_comments": 1, "permalink": "/r/wsb/mon",
                  "created_utc": _ets(2025, 6, 16, 10, 0)}},     # Mon 10:00 ET — in session
        {"data": {"title": "QQQ calls Friday after close", "selftext": "", "score": 90,
                  "num_comments": 1, "permalink": "/r/wsb/fri",
                  "created_utc": _ets(2025, 6, 13, 17, 0)}},      # Fri 17:00 ET — pre-open, stale
    ]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(
        allow_fetch=True, now=_et(2025, 6, 16, 10, 30), params={
            "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
            "freshness_mode": "market_window", "include_paper_options": False})
    tickers = {t["ticker"] for t in rep["top_tickers"]}
    assert "SPY" in tickers and "QQQ" not in tickers, f"Monday window not anchored to 09:30: {tickers}"
    assert rep["freshness_window"]["window_start_et"].startswith("2025-06-16T09:30")
    assert rep["freshness_window"]["anchor"] == "today_open"
    assert "since today's open (09:30 ET)" in ss.format_report(rep)


def test_report_8am_et_preopen_freshness_wording(monkeypatch):
    """8AM ET day-of (pre-open weekday): the window starts at the PREVIOUS close and the HOW FRESH
    wording says 'previous market close', never 'today' (the misleading wording this fixes)."""
    payload = {"data": {"children": [
        {"data": {"title": "SPY 0dte calls after the close", "selftext": "", "score": 50,
                  "num_comments": 1, "permalink": "/r/wsb/x",
                  "created_utc": _ets(2025, 6, 16, 17, 0)}},   # Mon 17:00 ET — after Monday close
    ]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(
        allow_fetch=True, now=_et(2025, 6, 17, 8, 0), params={   # Tue 08:00 ET — pre-open
            "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
            "freshness_mode": "market_window", "include_paper_options": False})
    fw = rep["freshness_window"]
    assert fw["anchor"] == "prev_close"
    assert fw["window_start_et"].startswith("2025-06-16T16:00")   # previous (Monday) close
    txt = ss.format_report(rep)
    how_fresh = next(line for line in txt.splitlines() if line.startswith("HOW FRESH"))
    assert "previous market close (06-16 16:00 ET)" in how_fresh
    assert "today" not in how_fresh.lower()


def test_reddit_rss_updated_parses_timestamp():
    """Atom <updated> must parse to a real created_utc so RSS-fallback posts are day-of filterable."""
    atom = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            '<entry><title>SPY calls</title><updated>2025-06-14T12:00:00+00:00</updated>'
            '<link href="https://www.reddit.com/r/wsb/z"/></entry></feed>')
    posts = ss._parse_reddit_rss(atom, 10)
    assert posts and posts[0]["created_utc"] > 0
    import datetime as _dt
    got = _dt.datetime.fromtimestamp(posts[0]["created_utc"], _dt.timezone.utc).date().isoformat()
    assert got == "2025-06-14", got


def _fake_yf(expiries, calls_df, puts_df):
    import types
    class _Chain:
        def __init__(self):
            self.calls = calls_df
            self.puts = puts_df
    class _Tk:
        def __init__(self, sym):
            self.options = list(expiries)
        def option_chain(self, exp):
            return _Chain()
    m = types.ModuleType("yfinance")
    m.Ticker = _Tk
    return m


def test_odte_options_budget_and_direction(monkeypatch):
    """PAPER options: budget cap (0 -> $50), liquidity sort, direction mapping. No network."""
    import datetime as _dt
    import sys

    import pandas as pd

    from data import odte_options as oo
    today = _dt.date.today().isoformat()
    calls = pd.DataFrame([
        {"strike": 500, "bid": 0.25, "ask": 0.30, "lastPrice": 0.28, "volume": 1000, "openInterest": 5000},
        {"strike": 495, "bid": 0.95, "ask": 1.00, "lastPrice": 0.98, "volume": 10, "openInterest": 20},
    ])
    puts = pd.DataFrame([
        {"strike": 500, "bid": 0.20, "ask": 0.25, "lastPrice": 0.24, "volume": 500, "openInterest": 300}])
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf([today], calls, puts))

    # bullish, budget 0 -> $50 cap: only the ~$30 call qualifies ($100 call excluded)
    res = oo.build_paper_options("SPY", "bullish", 0.0, today=today)
    assert res["status"] == "ok" and res["option_type"] == "call"
    assert len(res["contracts"]) == 1 and res["contracts"][0]["strike"] == 500
    assert res["contracts"][0]["premium_cost_estimate"] <= 50
    # bearish -> puts
    res2 = oo.build_paper_options("SPY", "bearish", 100.0, today=today)
    assert res2["option_type"] == "put" and res2["contracts"][0]["option_type"] == "put"


def test_odte_options_fail_closed_when_no_same_day(monkeypatch):
    import datetime as _dt
    import sys

    import pandas as pd

    from data import odte_options as oo
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf(["2099-01-01"], pd.DataFrame(), pd.DataFrame()))
    res = oo.build_paper_options("SPY", "bullish", 50.0, today=_dt.date.today().isoformat())
    assert res["contracts"] == [] and "no same-day" in res["status"]


def test_paper_options_attached_in_report(monkeypatch):
    """End-to-end: a fresh bullish SPY candidate attaches a PAPER call contract (mocked chain)."""
    import datetime as _dt
    import sys
    import time as _t

    import pandas as pd
    today = _dt.date.today().isoformat()
    calls = pd.DataFrame([
        {"strike": 500, "bid": 0.25, "ask": 0.30, "lastPrice": 0.28, "volume": 1000, "openInterest": 5000}])
    puts = pd.DataFrame([
        {"strike": 500, "bid": 0.20, "ask": 0.25, "lastPrice": 0.24, "volume": 100, "openInterest": 100}])
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf([today], calls, puts))
    now = _t.time()
    payload = {"data": {"children": [
        {"data": {"title": "SPY calls bullish moon today", "selftext": "buy", "score": 100,
                  "num_comments": 1, "permalink": "/r/wsb/x", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY"], "min_mentions": 1,
        "budget_dollars": 50, "include_paper_options": True})
    assert rep["candidate"]["ticker"] == "SPY" and rep["candidate"]["direction"] == "bullish"
    po = rep["paper_options"]
    assert po["status"] == "ok" and po["option_type"] == "call" and po["contracts"]
    assert po["contracts"][0]["premium_cost_estimate"] <= 50
    assert "IF YOU PRACTICE ANYWAY" in ss.format_report(rep)


def test_odte_cheapest_above_budget_when_none_fit(monkeypatch):
    """No <= $50 call, but liquid calls at $60 and $90: return ONLY the cheapest ($60) above-budget
    contract, flagged above_budget=True, with a status that explains it was just out of reach."""
    import datetime as _dt
    import sys

    import pandas as pd

    from data import odte_options as oo
    today = _dt.date.today().isoformat()
    calls = pd.DataFrame([
        {"strike": 600, "bid": 0.55, "ask": 0.60, "lastPrice": 0.58, "volume": 1000, "openInterest": 5000},
        {"strike": 595, "bid": 0.85, "ask": 0.90, "lastPrice": 0.88, "volume": 500, "openInterest": 300}])
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf([today], calls, pd.DataFrame()))
    res = oo.build_paper_options("SPY", "bullish", 50.0, today=today)
    assert len(res["contracts"]) == 1
    c = res["contracts"][0]
    assert c["strike"] == 600 and c["above_budget"] is True
    assert c["premium_cost_estimate"] == 60.0
    assert "cheapest above-budget" in res["status"]


def test_odte_budget_fit_excludes_above_budget(monkeypatch):
    """If a <= $50 contract exists, above-budget candidates are NOT returned (and status is ok)."""
    import datetime as _dt
    import sys

    import pandas as pd

    from data import odte_options as oo
    today = _dt.date.today().isoformat()
    calls = pd.DataFrame([
        {"strike": 500, "bid": 0.25, "ask": 0.30, "lastPrice": 0.28, "volume": 1000, "openInterest": 5000},
        {"strike": 595, "bid": 0.85, "ask": 0.90, "lastPrice": 0.88, "volume": 500, "openInterest": 300}])
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf([today], calls, pd.DataFrame()))
    res = oo.build_paper_options("SPY", "bullish", 50.0, today=today)
    assert res["status"] == "ok"
    assert [c["strike"] for c in res["contracts"]] == [500]
    assert all(c["above_budget"] is False for c in res["contracts"])


def test_format_report_labels_above_budget(monkeypatch):
    """End-to-end: an above-budget-only chain renders the contract row with an ABOVE BUDGET label
    and a status that mentions the cheapest above-budget fallback."""
    import datetime as _dt
    import sys
    import time as _t

    import pandas as pd
    today = _dt.date.today().isoformat()
    calls = pd.DataFrame([
        {"strike": 600, "bid": 0.55, "ask": 0.60, "lastPrice": 0.58, "volume": 1000, "openInterest": 5000}])
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf([today], calls, pd.DataFrame()))
    now = _t.time()
    payload = {"data": {"children": [
        {"data": {"title": "SPY calls bullish moon today", "selftext": "buy", "score": 100,
                  "num_comments": 1, "permalink": "/r/wsb/x", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    # Force a bullish price read so the verdict is CALL-leaning (practice examples are suppressed
    # on OBSERVE); the above-budget call must then render with its label.
    monkeypatch.setattr(ss, "_resolve_spy_trend", lambda allow_fetch: {
        "ok": True, "last": 505.0, "prev_close": 500.0, "vwap": 503.0,
        "pct_vs_prev_close": 0.01, "above_vwap": True})
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY"], "min_mentions": 1,
        "budget_dollars": 50, "include_paper_options": True})
    po = rep["paper_options"]
    assert po["contracts"] and po["contracts"][0]["above_budget"] is True
    assert "cheapest above-budget" in po["status"]
    assert rep["scorecard"]["verdict"] == "CALL-leaning"
    txt = ss.format_report(rep)
    assert "ABOVE BUDGET" in txt


def test_practice_examples_suppressed_on_observe():
    """On a DO NOTHING (OBSERVE) verdict, directional contract examples must be SUPPRESSED so they
    never contradict the verdict — even when paper_options carries call contracts."""
    rep = {
        "budget_dollars": 50,
        "freshness_window": {"window_start_et": "2026-06-17T09:30:00-04:00"},
        "sources": {"reddit": {"subreddit": "wsb", "n_posts": 2, "n_fetched": 50,
                               "n_stale_filtered": 40, "n_filtered": 0},
                    "x": {"status": "disabled", "n_posts": 0, "n_stale_filtered": 0, "n_filtered": 0}},
        "top_tickers": [{"ticker": "SPY", "mentions": 5}],
        "spy_trend": {"ok": False, "status": "no intraday history"},
        "social_intent": {"intent": "neutral", "bull": 0, "bear": 0, "n_docs": 2, "examples": []},
        "scorecard": {"verdict": "OBSERVE", "confidence": "low",
                      "reasons": ["SPY price/trend data unavailable or stale — observe only."]},
        "paper_options": {"status": "ok", "option_type": "call", "contracts": [
            {"option_type": "call", "strike": 505, "ask": 0.30, "premium_cost_estimate": 30,
             "spread_pct": 0.05, "volume": 1000, "above_budget": False}]},
        "risk_notes": ["practice money only."],
    }
    txt = ss.format_report(rep)
    assert "No example shown — today's read is DO NOTHING" in txt
    assert "A CALL at the $505" not in txt  # the directional example is suppressed


def _spy_price(*, last, prev_close, open_, vwap, ok=True):
    """Build a SPY price-trend dict in the shape build_scorecard consumes."""
    return {"ok": ok, "status": "ok" if ok else "no intraday history",
            "last": last, "prev_close": prev_close, "open": open_, "vwap": vwap,
            "pct_vs_prev_close": (last / prev_close - 1.0) if prev_close else None,
            "pct_vs_open": (last / open_ - 1.0) if open_ else None,
            "above_vwap": (last > vwap) if vwap is not None else None}


def test_scorecard_bullish_price_and_social_call_leaning():
    """Bullish intraday SPY (up vs prior close, above VWAP) + bullish SPY social -> CALL-leaning."""
    price = _spy_price(last=505.0, prev_close=500.0, open_=501.0, vwap=503.0)
    po = {"contracts": [{"spread_pct": 0.05, "above_budget": False}]}
    sc = ss.build_scorecard(price, po, "bullish")
    assert sc["verdict"] == "CALL-leaning"
    assert sc["confidence"] == "medium"
    assert sc["price_direction"] == "bullish"


def test_scorecard_bearish_price_and_social_put_leaning():
    """Bearish intraday SPY (down vs prior close, below VWAP) + bearish SPY social -> PUT-leaning."""
    price = _spy_price(last=495.0, prev_close=500.0, open_=499.0, vwap=497.0)
    po = {"contracts": [{"spread_pct": 0.05, "above_budget": False}]}
    sc = ss.build_scorecard(price, po, "bearish")
    assert sc["verdict"] == "PUT-leaning"
    assert sc["price_direction"] == "bearish"


def test_scorecard_missing_price_observes():
    """No usable price data -> OBSERVE even with strong bullish social (fail closed)."""
    sc = ss.build_scorecard({"ok": False, "status": "no intraday history"}, {}, "bullish")
    assert sc["verdict"] == "OBSERVE"
    assert ss.build_scorecard(None, {}, "bullish")["verdict"] == "OBSERVE"


def test_scorecard_social_only_never_directional():
    """Price near VWAP / sub-threshold + bullish social must NOT yield CALL — social alone can't."""
    price = _spy_price(last=500.2, prev_close=500.0, open_=500.1, vwap=500.15)  # +0.04%, tiny
    sc = ss.build_scorecard(price, {}, "bullish")
    assert sc["verdict"] == "OBSERVE"


def test_scorecard_price_social_conflict_observes():
    """Bullish price but bearish social -> conflict -> OBSERVE."""
    price = _spy_price(last=505.0, prev_close=500.0, open_=501.0, vwap=503.0)
    sc = ss.build_scorecard(price, {}, "bearish")
    assert sc["verdict"] == "OBSERVE"


def test_scorecard_no_social_confirmation_caps_confidence_low():
    """Directional price with NO confirming SPY social (neutral/None) -> still CALL-leaning, but
    only low confidence. (build_odte_social_report only ever feeds SPY intent here, so non-SPY
    chatter can't confirm.)"""
    price = _spy_price(last=505.0, prev_close=500.0, open_=501.0, vwap=503.0)
    sc = ss.build_scorecard(price, {"contracts": []}, None)
    assert sc["verdict"] == "CALL-leaning"
    assert sc["confidence"] == "low"
    assert sc["social_direction"] is None


def test_format_report_includes_scorecard(monkeypatch):
    """End-to-end: report leads with a plain-language 'WHAT TO DO NOW' action."""
    import sys
    import time as _t

    import pandas as pd
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yf([], pd.DataFrame(), pd.DataFrame()))
    now = _t.time()
    payload = {"data": {"children": [
        {"data": {"title": "SPY calls bullish moon today", "selftext": "buy", "score": 100,
                  "num_comments": 1, "permalink": "/r/wsb/x", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY"], "min_mentions": 1,
        "budget_dollars": 50, "include_paper_options": True})
    assert "scorecard" in rep and rep["scorecard"]["verdict"] in ("CALL-leaning", "PUT-leaning", "OBSERVE")
    txt = ss.format_report(rep)
    assert "WHAT TO DO NOW" in txt


def test_report_plain_language_no_jargon_in_primary(monkeypatch):
    """Beginner UX: primary section answers what-to-do / why / do-nothing in plain words, with no
    raw jargon and no raw floats. Uses --no-fetch so it fails closed to OBSERVE/DO NOTHING."""
    rep = ss.build_odte_social_report(allow_fetch=False, params={
        "sources": [], "core_universe": ["SPY"], "budget_dollars": 50})
    txt = ss.format_report(rep)
    head = txt.split("IF YOU PRACTICE ANYWAY")[0]  # the primary (above the practice block)
    # Human labels present, near the top.
    assert "WHAT TO DO NOW" in txt and txt.index("WHAT TO DO NOW") < 260
    assert "DO NOTHING" in head
    assert "Doing nothing is always fine" in head
    assert "WHY (3 quick reasons)" in head
    assert "WHAT WOULD CHANGE THIS" in head
    # Plain 'Confidence:' label (not 'How strong is this hint'); 0DTE is never 'safe'.
    assert "Confidence:" in head and "How strong is this hint" not in txt
    assert "safe-enough" not in txt and "clear-enough" in head
    # No raw jargon in the primary section ('confidence' is now an intended plain label).
    for jargon in ("VWAP", "hype=", "spread", "liquidity", "momentum", "sentiment",
                   "contextual", "sent=", "mom="):
        assert jargon not in head, f"jargon {jargon!r} leaked into primary section"
    # No raw floats with long decimal tails (e.g. 750.5200805664062).
    import re as _re
    assert not _re.search(r"\d+\.\d{5,}", txt), "raw long float leaked into report"


def test_change_lines_state_aware_and_trade_text():
    """'What would change this' must not ask for a cheap contract to appear when one exists; the
    WHY trade line must not over-sell tradability; example cleaner drops junk/handle fragments."""
    has = {"contracts": [{"above_budget": False, "premium_cost_estimate": 30}]}
    none = {"contracts": []}
    fit_lines = ss._kid_change_lines("OBSERVE", has, 50)
    assert any("stay available" in line for line in fit_lines)
    assert not any("show up" in line for line in fit_lines)
    assert any("show up" in line for line in ss._kid_change_lines("OBSERVE", none, 50))
    # WHY trade line: contracts existing does NOT imply a good setup.
    assert "does NOT make the setup good" in ss._kid_trade_line(has, 50)
    # Example cleaner: strip handle/junk + tiny fragments; keep clean directional snippets.
    assert ss._clean_example("u spygod nota") is None      # no directional signal word
    assert ss._clean_example("bull") is None                # single word
    assert ss._clean_example("spy calls printing") == "spy calls printing"


def test_report_examples_support_intent_no_contradiction():
    """A negated phrase among bullish docs must NOT surface as a shown example (no 'not calls').
    The rendered crowd line shows only intent-supporting spans."""
    rep = {
        "budget_dollars": 50,
        "freshness_window": {"window_start_et": "2026-06-17T09:30:00-04:00"},
        "sources": {"reddit": {"subreddit": "wsb", "n_posts": 3, "n_fetched": 50,
                               "n_stale_filtered": 40, "n_filtered": 2},
                    "x": {"status": "disabled", "n_posts": 0, "n_stale_filtered": 0, "n_filtered": 0}},
        "top_tickers": [{"ticker": "SPY", "mentions": 11}],
        "spy_trend": {"ok": True, "last": 505.0, "prev_close": 500.0, "vwap": 503.0,
                      "pct_vs_prev_close": 0.01, "above_vwap": True},
        "social_intent": ss.summarize_odte_intent("SPY", [
            {"text": "SPY calls printing 🚀"}, {"text": "do not chase SPY puts"},
            {"text": "buy the SPY dip"}]),
        "scorecard": {"verdict": "CALL-leaning", "confidence": "medium",
                      "reasons": ["SPY intraday +1.00% vs prior close and above VWAP."]},
        "paper_options": {"status": "ok", "contracts": [
            {"option_type": "call", "strike": 505, "ask": 0.30, "premium_cost_estimate": 30,
             "spread_pct": 0.05, "volume": 1000, "above_budget": False}]},
        "risk_notes": ["practice money only — you could lose it all."],
    }
    txt = ss.format_report(rep)
    crowd = next(line for line in txt.splitlines() if "Crowd talk" in line)
    # No negated span surfaces as a quoted example (e.g. "not puts"), and no debug marker.
    assert '"not ' not in crowd and "(negated)" not in crowd and "not calls" not in txt
    assert rep["social_intent"]["intent"] == "bullish"
    assert "examples only, NOT instructions" in txt


def test_intent_jinx_sarcasm_anti_signal():
    """Real WSB idioms: a position + opposite-direction expectation is anti-signal, not the position.
    'bought calls so it should keep dropping' = bearish, etc."""
    assert ss.classify_odte_intent("bought calls so it should keep dropping", "SPY")["intent"] == "bearish"
    assert ss.classify_odte_intent("buying msft calls so this shit stock tanks", "MSFT")["intent"] == "bearish"
    assert ss.classify_odte_intent("If you buy mu now you just know it starts fading to close", "MU")["intent"] == "bearish"


def test_intent_party_and_outcome_inverse():
    """'bears are cooked' = bullish; 'bulls are trapped' = bearish; 'puts won't print' = bullish."""
    assert ss.classify_odte_intent("bears are cooked", "SPY")["intent"] == "bullish"
    assert ss.classify_odte_intent("bulls are trapped", "SPY")["intent"] == "bearish"
    assert ss.classify_odte_intent("puts are cooked", "SPY")["intent"] == "bullish"
    assert ss.classify_odte_intent("calls are cooked", "SPY")["intent"] == "bearish"
    assert ss.classify_odte_intent("puts won't print", "SPY")["intent"] == "bullish"


def test_intent_disposition_sold_is_not_directional():
    """'sold calls' is not bullish; 'sold puts' is not bearish; 'sold calls, switched to puts' = bearish."""
    assert ss.classify_odte_intent("sold calls", "SPY")["intent"] == "neutral"
    assert ss.classify_odte_intent("sold puts", "SPY")["intent"] == "neutral"
    assert ss.classify_odte_intent(
        "Sold my ODTE calls for profit and switched to puts. Lets go!", "SPY")["intent"] == "bearish"


def test_intent_weak_chatter_and_no_short_not_bearish():
    """Questions / desires are weak chatter (not directional); 'do not short' is not bearish."""
    assert ss.classify_odte_intent("is SPCX going to 135?", "SPCX")["intent"] == "neutral"
    assert ss.classify_odte_intent("spy needs to go up", "SPY")["intent"] == "neutral"
    assert ss.classify_odte_intent("do not short", "SPY")["intent"] != "bearish"
    assert ss.classify_odte_intent("never short again", "SPY")["intent"] != "bearish"
    assert ss.classify_odte_intent("I buy both QQQ calls and puts today", "QQQ")["intent"] == "neutral"


def test_intent_multi_ticker_attribution():
    """Cues attach to the NEAREST ticker — 'qqq calls ... TXRH' is bullish for QQQ, neutral for TXRH."""
    txt = "Mostly qqq calls. Some shares sold on TXRH profit taking"
    assert ss.classify_odte_intent(txt, "QQQ")["intent"] == "bullish"
    assert ss.classify_odte_intent(txt, "TXRH")["intent"] == "neutral"
    txt2 = "MSFT calls but NVDA puts"
    assert ss.classify_odte_intent(txt2, "MSFT")["intent"] == "bullish"
    assert ss.classify_odte_intent(txt2, "NVDA")["intent"] == "bearish"


def test_intent_company_name_alias():
    """Validated company-name aliases map to the ticker for the directional read (and no halluc.)."""
    s = ss.summarize_odte_intent("MSFT", [{"text": "Microsoft calls printing today"}])
    assert s["n_docs"] == 1 and s["intent"] == "bullish"
    assert ss.summarize_odte_intent("MSFT", [{"text": "soft serve ice cream is great"}])["n_docs"] == 0


def test_summarize_separates_mentions_from_directional_evidence():
    """Mere mentions != directional evidence: a chatter-only ticker has n_docs>0 but
    directional_docs==0 and neutral intent (TOP CHATTER won't show a confident lean from mentions)."""
    ev = [{"text": "MU may actually hit 1500 next week"}, {"text": "MU MELTING UP"},
          {"text": "is MU too high?"}]
    s = ss.summarize_odte_intent("MU", ev)
    assert s["n_docs"] == 3 and s["directional_docs"] == 0 and s["intent"] == "neutral"


def test_classify_intent_contextual_phrases():
    """Phrase/context: bullish directional phrases classify bullish, bearish ones bearish."""
    assert ss.classify_odte_intent("SPY calls to the moon, buy the dip 🚀")["intent"] == "bullish"
    assert ss.classify_odte_intent("SPY going to zero, grab puts")["intent"] == "bearish"


def test_classify_intent_negation_not_bullish():
    """Negation cancels: 'do not chase SPY calls' must NOT read bullish."""
    r = ss.classify_odte_intent("do not chase SPY calls here")
    assert r["intent"] != "bullish"
    assert any("not" in e for e in r["examples"])


def test_classify_intent_inverse_option_outcome():
    """Inverse phrases: 'puts got smoked' is bullish for the underlying; 'calls got cooked' bearish;
    'puts printing' is bearish (puts gaining => price down)."""
    assert ss.classify_odte_intent("SPY puts got smoked today")["intent"] == "bullish"
    assert ss.classify_odte_intent("SPY calls got cooked")["intent"] == "bearish"
    assert ss.classify_odte_intent("SPY puts printing all day")["intent"] == "bearish"
    assert ss.classify_odte_intent("SPY calls printing 🚀")["intent"] == "bullish"


def test_classify_intent_question_and_risk_warning_neutral():
    """Questions / risk-warnings without a strong margin resolve to neutral, not directional."""
    assert ss.classify_odte_intent("is it too late to buy SPY calls?")["intent"] == "neutral"
    assert ss.classify_odte_intent("SPY calls? thoughts? NFA")["intent"] == "neutral"


def test_classify_intent_conflict_neutral():
    """Balanced bullish + bearish cues conflict -> neutral."""
    r = ss.classify_odte_intent("SPY calls or puts, bullish but also bearish")
    assert r["intent"] == "neutral"


def test_summarize_odte_intent_aggregates_examples():
    """Aggregate over evidence: net-bullish docs -> bullish with transparent counts/examples."""
    ev = [{"text": "SPY calls to the moon 🚀"}, {"text": "SPY puts got smoked"},
          {"text": "QQQ puts printing"}]  # QQQ doc must not count toward SPY
    s = ss.summarize_odte_intent("SPY", ev)
    assert s["intent"] == "bullish" and s["n_docs"] == 2 and s["bull"] >= 2
    assert s["examples"]


def test_scorecard_uses_contextual_classifier_not_keyword(monkeypatch):
    """End-to-end: the scorecard's social confirmation comes from the CONTEXTUAL classifier, not
    raw keyword sentiment. 'SPY puts got smoked' reads keyword-bearish (candidate.direction) yet
    contextual-bullish (inverse phrase), so a bullish price yields CALL-leaning."""
    import time as _t
    monkeypatch.setattr(ss, "_resolve_spy_trend", lambda allow_fetch: {
        "ok": True, "last": 505.0, "prev_close": 500.0, "vwap": 503.0,
        "pct_vs_prev_close": 0.01, "above_vwap": True})
    payload = {"data": {"children": [
        {"data": {"title": "SPY puts got smoked today", "selftext": "", "score": 50,
                  "num_comments": 1, "permalink": "/r/wsb/x", "created_utc": _t.time()}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(payload))
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY"], "min_mentions": 1, "budget_dollars": 50})
    assert rep["candidate"]["direction"] == "bearish"      # raw keyword view
    assert rep["social_intent"]["intent"] == "bullish"     # contextual view
    assert rep["scorecard"]["verdict"] == "CALL-leaning"   # verdict follows the contextual view


def test_hardening_promo_alert_bot_is_spam():
    """(1) Promo/alert-bot text is spam and never survives as ODTE evidence."""
    t = "Bullish signal on $SPY — Conviction 3/5 Real-time options flow for members. Try free"
    assert ss._is_spam(t) is True
    kept = ss._quality_filter([{"text": t}], lambda c: c["text"],
                              allowed={"SPY"}, require_options_context=True)
    assert kept == []


def test_hardening_phrasal_put_not_bearish():
    """(2) Bare singular 'put' as an English verb ('put on someone') is NOT counted bearish."""
    r = ss.classify_odte_intent("I would put on someone for $SPY $QQQ honestly", "SPY")
    assert r["intent"] != "bearish"
    assert r["bear"] == 0
    # but a real option noun still counts (so we didn't over-suppress)
    assert ss.classify_odte_intent("bought a $SPY put at the 500 strike", "SPY")["bear"] >= 1


def test_hardening_lowercase_cashtags_normalized():
    """(3) Lowercase cashtags '$spy $qqq' normalize to SPY/QQQ; plain lowercase words do not."""
    c = ss.extract_ticker_mentions(["$spy and $qqq are ripping", "AAPL too"])
    assert c["SPY"] == 1 and c["QQQ"] == 1 and c["AAPL"] == 1
    assert "SPY" not in ss.extract_ticker_mentions(["i spy with my little eye"])
    assert ss._tickers_in("$spy calls", allowed={"SPY"}) == ["SPY"]


def test_hardening_flow_jargon_is_neutral():
    """(4) Gamma/flow market-structure jargon is neutral context, not a directional vote."""
    t = "dealers long gamma, call wall 510, put wall 500, GEX positive, max pain 505 on $SPY"
    r = ss.classify_odte_intent(t, "SPY")
    assert r["intent"] == "neutral"
    assert "flow" in r["flags"]


def test_hardening_short_cover_transition_is_bullish():
    """(5) 'closed my short going back to $SPY calls' is a bullish transition, not a conflict."""
    r = ss.classify_odte_intent("closed my short, going back to $SPY calls", "SPY")
    assert r["intent"] == "bullish"


def test_top_chatter_ranks_by_mentions_excludes_spy_and_spam():
    """(8a) Top chatter ranks tickers by mentions from quality-gated evidence, excludes the SPY
    backdrop, and never includes spam tickers."""
    raw = [
        {"text": "$NVDA 0dte calls ripping"}, {"text": "$NVDA puts 0dte strike 500"},
        {"text": "$NVDA 0dte calls scalp"}, {"text": "$AMD 0dte calls"},
        {"text": "$SPY 0dte calls today"},
        {"text": "join my telegram VIP $TSLA 0dte calls free signal"},  # spam → dropped
    ]
    ev = ss._quality_filter(raw, lambda c: c["text"], allowed=None, require_options_context=True)
    ranked = ss.rank_top_chatter(ev, exclude={"SPY"}, max_n=5, min_mentions=1)
    tickers = [t for t, _ in ranked]
    assert tickers[0] == "NVDA"          # most mentions first
    assert "SPY" not in tickers          # backdrop excluded
    assert "TSLA" not in tickers         # spam dropped before ranking
    assert "AMD" in tickers


def test_rank_top_chatter_bare_universe_tickers_excludes_jargon():
    """With a validated optionable universe, BARE uppercase tickers (no $) count — WSB comments
    often omit the $ — while jargon (FOMC/GEX/OI) is excluded."""
    ev = [{"text": "MSFT 0dte calls printing"}, {"text": "MSFT calls 0dte"},
          {"text": "MU 0dte puts"}, {"text": "TSLA 0dte calls"},
          {"text": "FOMC GEX gamma OI 0dte wall"}]   # jargon, no real ticker
    allowed = {"MSFT", "MU", "TSLA", "SPY", "QQQ"}
    ranked = ss.rank_top_chatter(ev, exclude={"SPY"}, max_n=5, min_mentions=1, allowed=allowed)
    tk = [t for t, _ in ranked]
    assert tk[0] == "MSFT" and "MU" in tk and "TSLA" in tk
    assert "FOMC" not in tk and "GEX" not in tk and "OI" not in tk


def test_report_top_chatter_includes_bare_daily_comment_tickers(monkeypatch):
    """End-to-end: daily-thread comments with BARE MSFT/TSLA surface in TOP CHATTER (validated
    against an explicit optionable universe), while FOMC/GEX junk does not."""
    import time as _t
    now = _t.time()
    listing = {"data": {"children": [
        {"data": {"id": "1u9240r", "title": "Daily Discussion Thread for June 18, 2026",
                  "selftext": "", "score": 1, "num_comments": 9000,
                  "permalink": "/r/wsb/dt", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(listing))
    monkeypatch.setattr(ss, "fetch_reddit_comments", lambda *a, **k: [
        {"body": "MSFT 0dte calls printing", "score": 4, "author": "u"},
        {"body": "more MSFT 0dte calls", "score": 3, "author": "u2"},
        {"body": "TSLA 0dte puts", "score": 2, "author": "u3"},
        {"body": "FOMC GEX gamma wall 0dte", "score": 1, "author": "u4"}])   # junk
    monkeypatch.setattr(ss, "_resolve_intraday_trend", lambda tk, af: {"ok": False, "status": "no data"})
    monkeypatch.setattr(ss, "_resolve_ticker_options", lambda *a, **k: {"contracts": [], "expiry": None})
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
        "top_chatter_min_mentions": 1, "top_chatter_universe": ["MSFT", "TSLA", "MU"]})
    tickers = [c["ticker"] for c in rep["top_chatter"]]
    assert "MSFT" in tickers and "TSLA" in tickers
    assert "FOMC" not in tickers and "GEX" not in tickers
    assert "included" in rep["daily_thread"]["status"]


def test_ticker_card_directional_with_chain_is_lean():
    """(8b) Directional price + confirming social + a tradable same-day chain -> paper CALL-lean."""
    price = _spy_price(last=505.0, prev_close=500.0, open_=501.0, vwap=503.0)
    po = {"expiry": "2026-06-17", "contracts": [
        {"option_type": "call", "strike": 505, "ask": 0.30, "premium_cost_estimate": 30,
         "spread_pct": 0.05, "volume": 1000, "above_budget": False}]}
    card = ss.build_ticker_card("NVDA", price, po, "bullish", mentions=7)
    assert card["verdict"] == "CALL-leaning" and card["confidence"] == "medium"
    assert card["contracts"] and card["contracts"][0]["option_type"] == "call"


def test_ticker_card_social_only_observes():
    """(8c) No usable price (social-only) -> OBSERVE, no contracts."""
    card = ss.build_ticker_card("NVDA", {"ok": False, "status": "no data"},
                                {"expiry": "2026-06-17", "contracts": []}, "bullish", mentions=9)
    assert card["verdict"] == "OBSERVE" and card["contracts"] == []


def test_ticker_card_no_chain_observes():
    """(8d) No same-day options chain -> OBSERVE even with a directional price (can't 0DTE-trade)."""
    price = _spy_price(last=505.0, prev_close=500.0, open_=501.0, vwap=503.0)
    card = ss.build_ticker_card("NVDA", price, {"expiry": None, "contracts": []}, "bullish", 4)
    assert card["verdict"] == "OBSERVE" and card["note"] == "no same-day options"
    assert card["contracts"] == []


def test_daily_thread_comments_fail_closed(monkeypatch):
    """(8e) Daily-thread seam is official/fail-closed: --no-fetch -> skipped; thread found but no
    comments and no OAuth creds -> 'auth needed'; no thread -> 'not found'. No cookies, ever."""
    assert ss.fetch_daily_thread_comments([], {}, allow_fetch=False)[1].startswith("skipped")
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(ss, "fetch_reddit_comments", lambda *a, **k: [])
    monkeypatch.setattr(ss, "_fetch_reddit_search", lambda *a, **k: [])  # no network in unit test
    posts = [{"id": "1u85wuy", "title": "Daily Discussion Thread for June 17, 2026"}]
    cmts, status, tid = ss.fetch_daily_thread_comments(posts, {}, allow_fetch=True)
    assert cmts == [] and "auth needed" in status and tid == "1u85wuy"
    _, st2, _ = ss.fetch_daily_thread_comments([{"id": "x", "title": "random post"}], {}, allow_fetch=True)
    assert "no daily discussion thread" in st2


def test_daily_thread_discovered_by_search_when_not_in_listing(monkeypatch):
    """The daily thread is reliably found BY DATE via official search even when it's not in the hot
    listing (it's usually stickied). Discovery returns its id; the fetch then proceeds normally."""
    monkeypatch.setattr(ss, "_fetch_reddit_search", lambda sub, q, af, limit=10: [
        {"id": "1u85wuy", "title": "Daily Discussion Thread for June 17, 2026"}])
    # Not in the (empty) hot listing → must come from search.
    tid = ss._find_daily_thread_id([], {}, "wallstreetbets", allow_fetch=True)
    assert tid == "1u85wuy"
    # And without allow_fetch, search is NOT attempted (stays cheap/offline).
    assert ss._find_daily_thread_id([], {}, "wallstreetbets", allow_fetch=False) == ""


def test_report_includes_spy_backdrop_top_chatter_and_daily_status():
    """(8f/8g) Report keeps SPY backdrop first, then a TOP CHATTER section, a daily-thread status
    line, and an honest source caveat — and the chatter section stays jargon-free."""
    rep = ss.build_odte_social_report(allow_fetch=False, params={
        "sources": [], "core_universe": ["SPY"], "budget_dollars": 50})
    assert isinstance(rep.get("top_chatter"), list)
    assert "status" in rep.get("daily_thread", {})
    assert rep.get("top_chatter_caveat")
    txt = ss.format_report(rep)
    assert txt.index("WHAT TO DO NOW") < txt.index("TOP CHATTER")   # SPY backdrop first
    assert "daily thread" in txt.lower()
    assert "not the whole market" in txt.lower()                    # honest source caveat
    chatter = txt.split("TOP CHATTER")[1]
    for jargon in ("VWAP", "spread", "liquidity", "hype=", "sentiment="):
        assert jargon not in chatter


def test_daily_thread_url_and_id_parsed():
    """A configured daily_thread_url/id override resolves the thread id (no listing needed)."""
    assert ss._parse_thread_id(
        "https://www.reddit.com/r/wallstreetbets/comments/1u85wuy/daily_discussion/") == "1u85wuy"
    assert ss._find_daily_thread_id([], {"daily_thread_url": "https://reddit.com/comments/abc123/x"}) == "abc123"
    assert ss._find_daily_thread_id([], {"daily_thread_id": "zzz999"}) == "zzz999"


def test_daily_thread_comments_folded_into_ev_pool_and_top_chatter(monkeypatch):
    """Daily-thread comments are INGESTED: a thread found in the listing + bounded comments
    mentioning $QQQ flow into the evidence pool, surface in TOP CHATTER, and report 'N included'."""
    import time as _t
    now = _t.time()
    listing = {"data": {"children": [
        {"data": {"id": "1u85wuy", "title": "Daily Discussion Thread for June 17, 2026",
                  "selftext": "", "score": 10, "num_comments": 100,
                  "permalink": "/r/wsb/dt", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(listing))
    monkeypatch.setattr(ss, "fetch_reddit_comments", lambda *a, **k: [
        {"body": "$QQQ 0dte calls printing", "score": 5, "author": "u1"},
        {"body": "grabbing $QQQ 0dte calls", "score": 3, "author": "u2"},
        {"body": "$QQQ calls 0dte scalp", "score": 2, "author": "u3"}])
    # Keep per-ticker price/options offline + deterministic (cards fail closed to OBSERVE).
    monkeypatch.setattr(ss, "_resolve_intraday_trend", lambda tk, af: {"ok": False, "status": "no data"})
    monkeypatch.setattr(ss, "_resolve_ticker_options", lambda *a, **k: {"contracts": [], "expiry": None})
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
        "top_chatter_min_mentions": 2})
    assert rep["daily_thread"]["n_comments"] == 3
    assert "included" in rep["daily_thread"]["status"]
    assert "QQQ" in [c["ticker"] for c in rep["top_chatter"]]   # comments fed top chatter
    assert "3 sampled, 3 included" in ss.format_report(rep)


def test_reddit_comments_uses_oauth_bearer_when_creds_present(monkeypatch):
    """Verify the existing requests-OAuth path: with creds, comments are fetched from
    oauth.reddit.com/comments/{id} with a bearer header (same app-only flow PRAW uses) — so PRAW
    is unnecessary for this. No username/password; read-only application-only."""
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"], captured["headers"] = url, headers or {}
        return _FakeResp([{}, {"data": {"children": [
            {"kind": "t1", "data": {"body": "SPY 0dte calls", "score": 3, "author": "u"}}]}}])

    monkeypatch.setattr(ss, "_reddit_oauth_token", lambda: "TKN")
    monkeypatch.setattr(ss.requests, "get", fake_get)
    out = ss.fetch_reddit_comments("1u85wuy", limit=5, allow_fetch=True)
    assert out and out[0]["body"].startswith("SPY")
    assert captured["url"] == "https://oauth.reddit.com/comments/1u85wuy"
    assert captured["headers"].get("Authorization") == "bearer TKN"


def test_reddit_comments_public_json_without_creds(monkeypatch):
    """Without creds, it falls back to public JSON (no bearer) — the path that 403s server-side,
    which is exactly why REDDIT_CLIENT_ID/SECRET are needed for reliable daily-thread comments."""
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        return _FakeResp([{}, {"data": {"children": []}}])

    monkeypatch.setattr(ss, "_reddit_oauth_token", lambda: None)
    monkeypatch.setattr(ss.requests, "get", fake_get)
    ss.fetch_reddit_comments("xyz789", limit=3, allow_fetch=True)
    assert captured["url"] == "https://www.reddit.com/comments/xyz789.json"


def test_no_praw_dependency_added():
    """Option A stays dependency-light: no PRAW import in the module (requests OAuth only)."""
    import inspect
    src = inspect.getsource(ss)
    assert "import praw" not in src and "from praw" not in src


def test_reddit_comments_bearer_token_path(monkeypatch):
    """Explicit bearer arg hits oauth.reddit.com/comments/{id} with 'Authorization: Bearer <token>'
    and never leaks the token into the returned comments."""
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["url"], captured["headers"] = url, headers or {}
        return _FakeResp([{}, {"data": {"children": [
            {"kind": "t1", "data": {"body": "$QQQ 0dte calls", "score": 2, "author": "u"}}]}}])

    monkeypatch.setattr(ss, "_reddit_oauth_token", lambda: None)   # no app creds
    monkeypatch.setattr(ss.requests, "get", fake_get)
    token = "tok_v2_SECRET_abc123"
    out = ss.fetch_reddit_comments("1u9240r", limit=100, allow_fetch=True, bearer_token=token)
    assert out and out[0]["body"].startswith("$QQQ")
    assert captured["url"] == "https://oauth.reddit.com/comments/1u9240r"
    assert captured["headers"].get("Authorization") == f"Bearer {token}"
    assert token not in str(out)   # token never in returned data


def test_reddit_comments_auth_order_app_over_bearer(monkeypatch):
    """Fetch order: app OAuth preferred over bearer when app creds exist; bearer used when absent."""
    captured = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        captured["auth"] = (headers or {}).get("Authorization")
        return _FakeResp([{}, {"data": {"children": []}}])

    monkeypatch.setattr(ss.requests, "get", fake_get)
    monkeypatch.setattr(ss, "_reddit_oauth_token", lambda: "APPTOKEN")   # app creds present
    ss.fetch_reddit_comments("a1", allow_fetch=True, bearer_token="BEARERTOK")
    assert captured["auth"] == "bearer APPTOKEN"     # app OAuth wins
    monkeypatch.setattr(ss, "_reddit_oauth_token", lambda: None)         # no app creds
    ss.fetch_reddit_comments("a2", allow_fetch=True, bearer_token="BEARERTOK")
    assert captured["auth"] == "Bearer BEARERTOK"    # explicit bearer used


def test_wsb_daily_thread_title_helper():
    """Title/slug ARE derivable from the ET date; the post id is NOT (helper returns id=None)."""
    import datetime as _dt
    h = ss.wsb_daily_thread_title(_dt.date(2026, 6, 18))
    assert h["title"] == "Daily Discussion Thread for June 18, 2026"
    assert h["slug"] == "daily_discussion_thread_for_june_18_2026"
    assert h["id"] is None


def test_daily_thread_status_auth_label_and_no_token(monkeypatch):
    """Status names the auth path but never the token; auth-needed only when NO app creds AND no bearer."""
    monkeypatch.setattr(ss, "fetch_reddit_comments", lambda *a, **k: [])
    posts = [{"id": "1u9240r", "title": "Daily Discussion Thread for June 18, 2026"}]
    _, status, _ = ss.fetch_daily_thread_comments(posts, {}, allow_fetch=True)
    assert "auth needed" in status
    # bearer provided but empty result -> generic unavailable (not 'auth needed')
    _, status2, _ = ss.fetch_daily_thread_comments(posts, {"reddit_bearer_token": "TKN"}, allow_fetch=True)
    assert "unavailable" in status2 and "auth needed" not in status2 and "TKN" not in status2


def test_bearer_token_plumbed_into_report_not_persisted(monkeypatch):
    """build_odte_social_report(reddit_bearer_token=...) plumbs the token into the daily-thread
    fetch, includes the comments, labels status '(bearer token)', and NEVER persists/renders it."""
    import time as _t
    now = _t.time()
    listing = {"data": {"children": [
        {"data": {"id": "1u9240r", "title": "Daily Discussion Thread for June 18, 2026",
                  "selftext": "", "score": 1, "num_comments": 50,
                  "permalink": "/r/wsb/dt", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(listing))
    seen = {}

    def fake_comments(post_id, limit=5, ttl_s=900, allow_fetch=True, bearer_token=None):
        seen["bearer"] = bearer_token
        return [{"body": "$QQQ 0dte calls printing", "score": 3, "author": "u"},
                {"body": "$QQQ 0dte calls", "score": 2, "author": "u2"}]

    monkeypatch.setattr(ss, "fetch_reddit_comments", fake_comments)
    monkeypatch.setattr(ss, "_resolve_intraday_trend", lambda tk, af: {"ok": False, "status": "no data"})
    monkeypatch.setattr(ss, "_resolve_ticker_options", lambda *a, **k: {"contracts": [], "expiry": None})
    token = "ephemeral_TOKEN_xyz"
    myparams = {"sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
                "top_chatter_min_mentions": 2}
    rep = ss.build_odte_social_report(allow_fetch=True, reddit_bearer_token=token, params=myparams)
    assert seen["bearer"] == token                          # plumbed into the fetch
    assert "reddit_bearer_token" not in myparams            # caller's params NOT mutated
    assert rep["daily_thread"]["n_comments"] == 2
    assert "bearer token" in rep["daily_thread"]["status"] and "included" in rep["daily_thread"]["status"]
    assert "QQQ" in [c["ticker"] for c in rep["top_chatter"]]   # comments fed top chatter
    txt = ss.format_report(rep)
    assert token not in txt and token not in str(rep)       # never rendered / never in dict


def test_cli_passes_bearer_token_and_does_not_print_it(monkeypatch, capsys):
    """Command-level: the CLI parses --reddit-bearer-token, passes it through, and never echoes it."""
    import data.social_sentiment as _ss
    from cli.main import main as cli_main
    cap = {}

    def fake_build(allow_fetch=True, params=None, now=None, reddit_bearer_token=None):
        cap["token"], cap["allow_fetch"] = reddit_bearer_token, allow_fetch
        return {"_": 1}

    monkeypatch.setattr(_ss, "build_odte_social_report", fake_build)
    monkeypatch.setattr(_ss, "format_report", lambda rep: "RENDERED REPORT")
    cli_main(["odte-social-report", "--reddit-bearer-token", "TOK123", "--no-fetch"])
    out = capsys.readouterr().out
    assert cap["token"] == "TOK123" and cap["allow_fetch"] is False
    assert "TOK123" not in out


def test_cli_daily_thread_override_and_bearer_plumbing(monkeypatch, capsys):
    """CLI plumbs --daily-thread-id/--daily-thread-url + --reddit-bearer-token into the report at
    runtime (never persisted to the global params), preserves default when absent, and never echoes
    the token."""
    import data.social_sentiment as _ss
    from cli.main import main as cli_main
    from util import OPTIONS_SOCIAL_PARAMS as _OSP
    cap = {}

    def fake_build(allow_fetch=True, params=None, now=None, reddit_bearer_token=None):
        cap["params"], cap["token"], cap["allow_fetch"] = params, reddit_bearer_token, allow_fetch
        return {"_": 1}

    monkeypatch.setattr(_ss, "build_odte_social_report", fake_build)
    monkeypatch.setattr(_ss, "format_report", lambda rep: "RENDERED REPORT")

    # id override + bearer
    cli_main(["odte-social-report", "--daily-thread-id", "1u9240r",
              "--reddit-bearer-token", "TOK123", "--no-fetch"])
    out = capsys.readouterr().out
    assert cap["params"]["daily_thread_id"] == "1u9240r"
    assert cap["token"] == "TOK123" and cap["allow_fetch"] is False
    assert "TOK123" not in out
    assert _OSP.get("daily_thread_id") != "1u9240r"   # global config NOT mutated

    # url override
    cli_main(["odte-social-report", "--daily-thread-url",
              "https://www.reddit.com/r/wallstreetbets/comments/abc123/x/", "--no-fetch"])
    assert cap["params"]["daily_thread_url"].endswith("abc123/x/")

    # default: no override flags -> params stays None (existing behavior preserved)
    cli_main(["odte-social-report", "--no-fetch"])
    assert cap["params"] is None


def test_daily_thread_noise_filter_drops_chatter_keeps_actionable():
    """Noise filter: drop bot/banbet, off-topic, one-word/number/emoji, and bare price questions;
    keep short STRONG actionable lines (ticker + calls/puts/strike/0dte/direction)."""
    noise = [
        "is SPCX going to 135?", "90", "lower", "Need a Hail Mary today 😂",
        "Had a great breakfast then took profit, my sister called",
        "U buying? Asking for a fren", "!banbet SPCX 150 60d", "Banbet broken",
        "No Active BanBet",
    ]
    keep = [
        "TSLA 400c send it", "MSFT puts", "MU 1150 EOD", "SPY 0dte calls",
        "time for spy puts",
        "Mostly qqq calls. Some shares sold on TXRH profit taking",
    ]
    for t in noise:
        assert ss._is_noise_comment(t) is True, f"should be noise: {t!r}"
    for t in keep:
        assert ss._is_noise_comment(t) is False, f"should be kept: {t!r}"


def test_daily_thread_noise_filter_drops_offtopic_longform_and_generic_money():
    """v3 tightening: long off-topic comments with no ticker AND no market context are dropped,
    and no-ticker generic-money replies are dropped (we can't see parent context)."""
    noise = [
        "Going to Africa and freeing slaves",
        "Anyone else browsing r/cscareerquestions during the dump lol my autism is acting up",
        "my balls are chafed from sitting all day",
        "Richard Brandon gave me a handjob clifford",
        "running late for work gonna hit the casino after",
        "I hope so. I need my money back.",
    ]
    keep = [
        "TSLA 400c send it", "MSFT puts", "SPY 0dte calls",
        "market looks weak, fed is hawkish and yields ripping",   # no ticker but market context
    ]
    for t in noise:
        assert ss._is_noise_comment(t) is True, f"should be noise: {t!r}"
    for t in keep:
        assert ss._is_noise_comment(t) is False, f"should be kept: {t!r}"


def test_daily_thread_noise_filter_applied_in_fetch(monkeypatch):
    """The noise filter runs inside fetch_daily_thread_comments: noise is excluded from the included
    comments and the count reflects only the kept (actionable) ones."""
    monkeypatch.setattr(ss, "fetch_reddit_comments", lambda *a, **k: [
        {"body": "SPY 0dte calls", "score": 3, "author": "u1"},     # keep
        {"body": "is SPCX going to 135?", "score": 1, "author": "u2"},  # drop (question)
        {"body": "90", "score": 1, "author": "u3"},                  # drop
        {"body": "!banbet SPCX 150 60d", "score": 1, "author": "u4"}])  # drop
    posts = [{"id": "1u9240r", "title": "Daily Discussion Thread for June 18, 2026"}]
    comments, status, _ = ss.fetch_daily_thread_comments(posts, {"reddit_bearer_token": "T"},
                                                         allow_fetch=True)
    bodies = [c["body"] for c in comments]
    assert bodies == ["SPY 0dte calls"]
    assert "1 included" in status and "sampled" in status


def test_daily_thread_limit_clamped():
    """Sample size clamps to [1, 500] (a big thread is sampled, not read in full)."""
    assert ss._clamp_daily_limit(150) == 150
    assert ss._clamp_daily_limit(99999) == 500     # cap
    assert ss._clamp_daily_limit(0) == 1           # floor
    assert ss._clamp_daily_limit("nope") == ss._DAILY_LIMIT_DEFAULT


def test_daily_thread_limit_passed_to_fetch_and_status_says_sampled(monkeypatch):
    """The configured daily_thread_limit is passed to the comment fetch, and the status says the
    comments are SAMPLED (not exhaustive)."""
    import time as _t
    now = _t.time()
    listing = {"data": {"children": [
        {"data": {"id": "1u9240r", "title": "Daily Discussion Thread for June 18, 2026",
                  "selftext": "", "score": 1, "num_comments": 11000,
                  "permalink": "/r/wsb/dt", "created_utc": now}}]}}
    monkeypatch.setattr(ss.requests, "get", lambda *a, **k: _FakeResp(listing))
    seen = {}

    def fake_comments(post_id, limit=5, ttl_s=900, allow_fetch=True, bearer_token=None):
        seen["limit"] = limit
        return [{"body": "$QQQ 0dte calls", "score": 1, "author": "u"} for _ in range(3)]

    monkeypatch.setattr(ss, "fetch_reddit_comments", fake_comments)
    monkeypatch.setattr(ss, "_resolve_intraday_trend", lambda tk, af: {"ok": False, "status": "no data"})
    monkeypatch.setattr(ss, "_resolve_ticker_options", lambda *a, **k: {"contracts": [], "expiry": None})
    rep = ss.build_odte_social_report(allow_fetch=True, params={
        "sources": ["reddit"], "core_universe": ["SPY", "QQQ"], "min_mentions": 1,
        "daily_thread_limit": 150})
    assert seen["limit"] == 150                                  # configured sample size passed through
    status = rep["daily_thread"]["status"]
    assert "sampled" in status                                  # not implied exhaustive
    assert "all" not in status.split("(")[0].lower()            # no 'all 11k' implication
    assert "11000" not in ss.format_report(rep)                 # never claims the full count


def test_cli_daily_thread_limit_plumbing(monkeypatch):
    """CLI parses --daily-thread-limit into params['daily_thread_limit'] (int), runtime-only."""
    import data.social_sentiment as _ss
    from cli.main import main as cli_main
    cap = {}

    def fake_build(allow_fetch=True, params=None, now=None, reddit_bearer_token=None):
        cap["params"] = params
        return {"_": 1}

    monkeypatch.setattr(_ss, "build_odte_social_report", fake_build)
    monkeypatch.setattr(_ss, "format_report", lambda rep: "RENDERED")
    cli_main(["odte-social-report", "--daily-thread-limit", "200", "--no-fetch"])
    assert cap["params"]["daily_thread_limit"] == 200


def test_no_cookie_secrets_used():
    """Hard rule: the module never uses cookies, a cookie env secret, or the cookie/csrf token-mint
    endpoint — official OAuth + explicit bearer arg only."""
    import inspect
    src = inspect.getsource(ss)
    assert "cookies=" not in src
    assert "REDDIT_COOKIE" not in src
    assert '"Cookie"' not in src and "'Cookie'" not in src
    assert "shreddit/token" not in src   # no automatic cookie/csrf token minting


def test_module_places_no_orders():
    """Guardrail: the social AND options modules import no execution/broker code and call no
    order methods (checks imports + call-sites, not prose in the docstring)."""
    import inspect

    from data import odte_options as oo
    forbidden = ("from execution", "import execution", "buy_fractional(",
                 "place_order(", "submit_order(", "broker.sell(", "robin_stocks")
    for mod in (ss, oo):
        src = inspect.getsource(mod)
        for f in forbidden:
            assert f not in src, f"{mod.__name__} must not reference {f!r}"
