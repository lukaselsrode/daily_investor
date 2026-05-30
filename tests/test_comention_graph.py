"""
Tests for the news co-mention graph ETL artifact + supporting data-layer pieces.

The graph is built from the structured ``related_symbols`` field (resolved from
Robinhood ``related_instruments``) and persisted as a data-layer artifact, consumed
read-only by portfolio/ui. These tests exercise the pure builder offline (no network).
"""
from __future__ import annotations

import json

import pandas as pd


def _news_df(rows):
    return pd.DataFrame(
        [{"symbol": s, "news": json.dumps(arts)} for s, arts in rows],
        columns=["symbol", "news"],
    )


def test_title_sentiment_directional():
    from data.comention_graph import title_sentiment
    assert title_sentiment("Stock surges to record high on strong growth") > 0
    assert title_sentiment("Shares plunge on weak guidance and downgrade") < 0
    assert title_sentiment("Company holds annual meeting") == 0.0


def test_build_graph_edges_and_sentiment():
    from data.comention_graph import build_comention_graph

    news = _news_df([
        ("CVS", [
            {"title": "CVS Health covers Eli Lilly drug", "related_symbols": ["LLY", "NVO"]},
            {"title": "CVS gains on strong earnings beat", "related_symbols": []},
        ]),
        ("NVO", [
            {"title": "Novo Nordisk falls on weak guidance", "related_symbols": ["LLY"]},
        ]),
    ])
    edges, nodes = build_comention_graph(news_df=news, held_symbols={"CVS", "NVO"}, persist=False)

    # Edge CVS--LLY appears in both CVS art1 and NVO is separate; LLY--NVO from CVS art1.
    emap = {(r.source, r.target): r.weight for r in edges.itertuples()}
    assert emap.get(("CVS", "LLY")) == 1
    assert emap.get(("LLY", "NVO")) == 1
    assert emap.get(("CVS", "NVO")) == 1   # co-mentioned in CVS article 1
    assert emap.get(("LLY", "NVO")) == 1

    nmap = {r.symbol: r for r in nodes.itertuples()}
    # held flag propagated
    assert nmap["CVS"].held is True
    assert nmap["LLY"].held is False
    # degree: LLY connects to CVS and NVO -> 2
    assert nmap["LLY"].degree == 2
    # CVS sentiment: art1 neutral(0) + art2 positive(>0) averaged -> > 0
    assert nmap["CVS"].sentiment > 0
    # NVO title is negative -> negative sentiment
    assert nmap["NVO"].sentiment < 0


def test_shared_article_identity_creates_edges():
    """The primary mechanism: same article (by link, else title) under multiple
    tickers' news lists creates a co-mention edge — even when related_symbols is
    empty (the yfinance case, which is the common one in production)."""
    from data.comention_graph import build_comention_graph

    # AAPL and MSFT both carry the SAME article (same link), no related_symbols.
    shared = {"title": "Tech giants rally on AI optimism", "link": "http://x.com/a1",
              "related_symbols": []}
    # CSCO and JNPR share a different article by TITLE only (no link on either) — the
    # title-fallback path.
    shared_titled = {"title": "Networking sector upgraded by analysts", "link": "",
                     "related_symbols": []}
    news = _news_df([
        ("AAPL", [shared, {"title": "Apple unique story", "link": "http://x.com/aapl",
                            "related_symbols": []}]),
        ("MSFT", [shared]),
        ("CSCO", [shared_titled]),
        ("JNPR", [{"title": "Networking sector upgraded by analysts", "link": "",
                   "related_symbols": []}]),
    ])
    edges, nodes = build_comention_graph(news_df=news, persist=False)
    emap = {(r.source, r.target): r.weight for r in edges.itertuples()}
    # AAPL--MSFT share the link identity
    assert emap.get(("AAPL", "MSFT")) == 1
    # CSCO--JNPR share the title identity (link-less fallback)
    assert emap.get(("CSCO", "JNPR")) == 1
    nmap = {r.symbol: r for r in nodes.itertuples()}
    assert nmap["MSFT"].degree >= 1


def test_link_normalization_matches_variants():
    """Links differing only by scheme/www/query/trailing slash are the same article."""
    from data.comention_graph import _article_key
    k1 = _article_key({"link": "https://www.x.com/story?utm=1"})
    k2 = _article_key({"link": "http://x.com/story/"})
    assert k1 == k2 == "L:x.com/story"
    # title fallback when no link
    assert _article_key({"title": "Big  News"}) == "T:big news"
    assert _article_key({}) == ""


def test_build_graph_empty_input():
    from data.comention_graph import build_comention_graph
    edges, nodes = build_comention_graph(news_df=pd.DataFrame(columns=["symbol", "news"]), persist=False)
    assert edges.empty
    assert nodes.empty
    assert list(edges.columns) == ["source", "target", "weight"]


def test_build_graph_tolerates_malformed_news_json():
    from data.comention_graph import build_comention_graph
    news = pd.DataFrame(
        [{"symbol": "X", "news": "not json"}, {"symbol": "Y", "news": json.dumps([{"title": "Y rallies", "related_symbols": ["Z"]}])}],
        columns=["symbol", "news"],
    )
    edges, nodes = build_comention_graph(news_df=news, persist=False)
    # Y--Z survives, malformed X row is skipped without raising
    assert {(r.source, r.target) for r in edges.itertuples()} == {("Y", "Z")}


def test_instrument_resolver_normalizes_id_and_url():
    from data.instrument_resolver import _normalize_id
    assert _normalize_id("abc-123") == "abc-123"
    assert _normalize_id("https://api.robinhood.com/instruments/abc-123/") == "abc-123"
    assert _normalize_id("https://api.robinhood.com/instruments/abc-123") == "abc-123"


def test_market_structure_df_shape():
    """load_market_structure_df returns the documented columns even when empty."""
    from data.market_structure import MARKET_STRUCTURE_DF_COLS, load_market_structure_df
    df = load_market_structure_df([], auto_refresh=False)
    assert list(df.columns) == ["symbol", *MARKET_STRUCTURE_DF_COLS]
    assert df.empty
