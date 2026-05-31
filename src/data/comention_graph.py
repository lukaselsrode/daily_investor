"""
data/comention_graph.py — News co-mention graph ETL artifact.

Builds a connected graph of securities that are mentioned together in news, from
the structured ``related_symbols`` field captured on Robinhood news articles
(data.news). Nodes = tickers; edges = co-mention counts; each node carries a crude
lexical sentiment averaged over the article titles it appears in.

This is a DATA-LAYER artifact: it is built + persisted here, then read by the
portfolio/visualization + ui layers as a pure consumer. No live API calls and no
graph derivation happen in business logic.

Persists two files:
  data/comention_edges.csv  (source, target, weight)
  data/comention_nodes.csv  (symbol, degree, sentiment, n_articles, held)
"""
from __future__ import annotations

import collections
import itertools
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Transparent finance-aware title-sentiment lexicon. Intentionally rule-based (not an
# NLP model) so the graph artifact stays self-contained and deterministic. Expanded
# with negation handling. NOTE: validated offline (.session_tmp/h7) that a heavier
# model (VADER) and this expansion do NOT improve cross-sectional forward-return IC —
# this is a correctness/display upgrade, not an alpha lever. No trading claim attached.
_POS = frozenset({
    "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps", "jumped",
    "rally", "rallies", "rallied", "beat", "beats", "gain", "gains", "gained", "rise",
    "rises", "rose", "record", "strong", "growth", "upgrade", "upgraded", "outperform",
    "buy", "bullish", "boom", "wins", "win", "profit", "profits", "tops", "topped",
    "rebound", "rebounds", "raises", "raise", "raised", "high", "highs", "best", "boost",
    "boosts", "boosted", "soaring", "surging", "climb", "climbs", "climbed", "advance",
    "advances", "leap", "leaps", "expands", "expansion", "accelerate", "accelerates",
    "momentum", "breakout", "upside", "buyback", "approval", "approved", "secures",
    "optimism", "outperforms",
})
_NEG = frozenset({
    "fall", "falls", "fell", "drop", "drops", "dropped", "plunge", "plunges", "plunged",
    "slump", "slumps", "sink", "sinks", "sank", "miss", "misses", "missed", "loss",
    "losses", "weak", "cut", "cuts", "downgrade", "downgraded", "sell", "bearish",
    "crash", "crashes", "fear", "fears", "warn", "warns", "warning", "slash", "slashes",
    "plummet", "plummets", "tumble", "tumbles", "probe", "sue", "sued", "lawsuit", "low",
    "lows", "decline", "declines", "declined", "layoff", "layoffs", "bankruptcy",
    "default", "recall", "halt", "halts", "slowdown", "weakness", "disappoints",
    "disappointing", "investigation", "fraud", "scandal", "selloff", "downside", "woes",
})
# Negators flip the polarity of a sentiment word in the following 1-2 tokens.
_NEGATORS = frozenset({"not", "no", "never", "without", "fails", "failed", "lacks", "lack",
                       "unable", "halts", "halt"})


def title_sentiment(text: str) -> float:
    """Finance-aware lexical sentiment in [-1, 1] from a headline. 0.0 when no hits.

    Counts positive/negative lexicon hits, flipping a hit's sign when a negator
    appears within the preceding two tokens ("not strong" -> negative). Score is the
    mean signed hit, clamped to [-1, 1]. Deterministic and dependency-free.
    """
    toks = re.findall(r"[a-zA-Z]+", (text or "").lower())
    if not toks:
        return 0.0
    total = 0.0
    hits = 0
    for i, t in enumerate(toks):
        base = 1.0 if t in _POS else (-1.0 if t in _NEG else 0.0)
        if base == 0.0:
            continue
        if any(w in _NEGATORS for w in toks[max(0, i - 2):i]):
            base = -base
        total += base
        hits += 1
    if hits == 0:
        return 0.0
    v = total / hits
    return float(max(-1.0, min(1.0, v)))


def _data_dir() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR


def _edges_path() -> Path:
    return _data_dir() / "comention_edges.csv"


def _nodes_path() -> Path:
    return _data_dir() / "comention_nodes.csv"


def _article_key(art: dict) -> str:
    """Stable identity for an article so the same story under multiple tickers maps
    to one node-set. Prefer the normalized link (strip scheme/query/trailing slash);
    fall back to a normalized title. Empty when neither is usable."""
    link = str(art.get("link") or "").strip()
    if link:
        link = re.sub(r"^https?://(www\.)?", "", link.lower())
        link = link.split("?")[0].split("#")[0].rstrip("/")
        if link:
            return f"L:{link}"
    title = str(art.get("title") or "").strip().lower()
    title = re.sub(r"\s+", " ", title)
    return f"T:{title}" if title else ""



def _parse_articles(art_iter):
    """Yield (sentiment, related_symbols, article_key) for each dict article."""
    for art in art_iter:
        if not isinstance(art, dict):
            continue
        yield (
            title_sentiment(art.get("title", "")),
            art.get("related_symbols") or [],
            _article_key(art),
        )


def _accumulate_news(news_df: pd.DataFrame | None):
    """Single pass over the news table -> the accumulators the graph is built from.

    Returns (edges, node_sent, node_articles, article_syms, article_sent):
      edges          Counter[(a,b)] of related_symbols co-mention edges
      node_sent      symbol -> list[float] of attributed title sentiments
      node_articles  Counter[symbol] of article counts
      article_syms   article-identity -> set of symbols carrying it (shared-article edges)
      article_sent   article-identity -> its title sentiment (computed once)
    """
    edges: collections.Counter = collections.Counter()
    node_sent: dict[str, list[float]] = collections.defaultdict(list)
    node_articles: collections.Counter = collections.Counter()
    article_syms: dict[str, set[str]] = collections.defaultdict(set)
    article_sent: dict[str, float] = {}

    if news_df is None or news_df.empty or "news" not in news_df.columns:
        return edges, node_sent, node_articles, article_syms, article_sent

    for _, row in news_df.iterrows():
        sym = str(row.get("symbol", "")).strip().upper()
        if not sym:
            continue
        raw = row.get("news")
        try:
            articles = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            articles = []
        for s, related, akey in _parse_articles(articles):
            node_sent[sym].append(s)
            node_articles[sym] += 1

            # (1) Structured related_symbols edges (kept when the feed provides them).
            touched = {sym}
            for other in related:
                o = str(other).strip().upper()
                if not o or o == sym:
                    continue
                edges[tuple(sorted((sym, o)))] += 1
                touched.add(o)
            for o in touched - {sym}:
                node_sent[o].append(s)

            # (2) Shared-article-identity co-mention: register under the article key.
            if akey:
                article_syms[akey].add(sym)
                if akey not in article_sent:
                    article_sent[akey] = s

    return edges, node_sent, node_articles, article_syms, article_sent


def build_comention_graph(
    news_df: pd.DataFrame | None = None,
    held_symbols: set[str] | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (edges_df, nodes_df) from the news table.

    Edges come from two complementary mechanisms (see _accumulate_news): structured
    ``related_symbols`` when the feed provides them, plus shared article identity
    (the same story under multiple tickers) which is what makes the graph dense and
    source-agnostic. Each article's title sentiment is attributed to every node it
    touches. When news_df is None it is read from the data cache.

    Returns the two DataFrames and (when persist) writes them to the data dir.
    """
    if news_df is None:
        from data.cache import read_data_as_pd
        news_df = read_data_as_pd("news")

    edges, node_sent, node_articles, article_syms, article_sent = _accumulate_news(news_df)

    # Build co-mention edges from shared article identity. Every pair of distinct
    # symbols carrying the same article gets an edge (weight = number of shared
    # articles). Also attribute each shared article's sentiment to every node it
    # touches (so a co-mentioned-but-not-primary symbol still accrues sentiment).
    for akey, syms in article_syms.items():
        if len(syms) < 2:
            continue
        s = article_sent.get(akey, 0.0)
        for a, b in itertools.combinations(sorted(syms), 2):
            edges[(a, b)] += 1
        for sym in syms:
            node_sent[sym].append(s)

    edge_rows = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in sorted(edges.items(), key=lambda kv: -kv[1])
    ]
    edges_df = pd.DataFrame(edge_rows, columns=["source", "target", "weight"])

    degree: collections.Counter = collections.Counter()
    for (a, b), _w in edges.items():
        degree[a] += 1
        degree[b] += 1

    held = {str(s).strip().upper() for s in (held_symbols or set())}
    all_nodes = set(degree) | set(node_sent)
    node_rows = []
    for sym in sorted(all_nodes):
        ss = node_sent.get(sym, [])
        node_rows.append({
            "symbol": sym,
            "degree": int(degree.get(sym, 0)),
            "sentiment": round(sum(ss) / len(ss), 4) if ss else 0.0,
            "n_articles": int(node_articles.get(sym, 0)),
            "held": sym in held,
        })
    nodes_df = pd.DataFrame(
        node_rows, columns=["symbol", "degree", "sentiment", "n_articles", "held"]
    )

    if persist:
        try:
            d = _data_dir()
            d.mkdir(parents=True, exist_ok=True)
            edges_df.to_csv(_edges_path(), index=False)
            nodes_df.to_csv(_nodes_path(), index=False)
            logger.info(
                "comention graph: %d edges, %d nodes persisted",
                len(edges_df), len(nodes_df),
            )
        except Exception as exc:
            logger.warning("Could not persist comention graph: %s", exc)

    return edges_df, nodes_df


def load_comention_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the persisted graph artifact. Empty frames when not yet built."""
    e_cols = ["source", "target", "weight"]
    n_cols = ["symbol", "degree", "sentiment", "n_articles", "held"]
    ep, np_ = _edges_path(), _nodes_path()
    edges = pd.read_csv(ep) if ep.exists() else pd.DataFrame(columns=e_cols)
    nodes = pd.read_csv(np_) if np_.exists() else pd.DataFrame(columns=n_cols)
    return edges, nodes


# Per-node structural features exposed to the enriched data block (agg_data). Prefixed
# ``news_`` so consumers can tell graph-derived columns apart. Computed in ETL; the
# strategy/portfolio layer reads them as columns (pure consumer), never builds a graph.
GRAPH_FEATURE_COLS: list[str] = [
    "news_degree", "news_wdegree", "news_centrality", "news_clustering",
    "news_core", "news_community", "news_community_size",
    "news_sentiment", "news_peer_sentiment", "news_sent_dispersion",
]


def compute_graph_features(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-node structural + sentiment features from the co-mention graph.

    Returns a DataFrame keyed by ``symbol`` with the GRAPH_FEATURE_COLS columns. Uses
    networkx for centrality/community; degrades gracefully (zeros) if networkx is
    unavailable or the graph is empty. This is the research-validated feature set
    (degree/centrality/community + neighbor-sentiment aggregates).
    """
    cols = ["symbol", *GRAPH_FEATURE_COLS]
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(columns=cols)
    try:
        import networkx as nx
    except Exception:
        logger.warning("networkx unavailable — graph features skipped")
        return pd.DataFrame(columns=cols)

    G = nx.Graph()
    for r in edges_df.itertuples(index=False):
        G.add_edge(r.source, r.target, weight=getattr(r, "weight", 1))
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=cols)

    sent_map: dict[str, float] = {}
    if nodes_df is not None and not nodes_df.empty and "sentiment" in nodes_df.columns:
        sent_map = dict(zip(nodes_df["symbol"], nodes_df["sentiment"]))

    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    clustering = nx.clustering(G)
    core = nx.core_number(G)
    try:
        cent = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        cent = {n: 0.0 for n in G.nodes()}
    comm_id: dict[str, int] = {}
    comm_sz: dict[str, int] = {}
    try:
        for ci, cset in enumerate(nx.community.greedy_modularity_communities(G, weight="weight")):
            for n in cset:
                comm_id[n] = ci
                comm_sz[n] = len(cset)
    except Exception:
        pass

    rows = []
    for n in G.nodes():
        nbrs = list(G.neighbors(n))
        psent = [sent_map.get(x, 0.0) for x in nbrs]
        rows.append({
            "symbol": n,
            "news_degree": int(deg.get(n, 0)),
            "news_wdegree": int(wdeg.get(n, 0)),
            "news_centrality": round(float(cent.get(n, 0.0)), 6),
            "news_clustering": round(float(clustering.get(n, 0.0)), 6),
            "news_core": int(core.get(n, 0)),
            "news_community": int(comm_id.get(n, -1)),
            "news_community_size": int(comm_sz.get(n, 0)),
            "news_sentiment": round(float(sent_map.get(n, 0.0)), 4),
            "news_peer_sentiment": round(float(sum(psent) / len(psent)), 4) if psent else 0.0,
            "news_sent_dispersion": round(float(pd.Series(psent).std()), 4) if len(psent) > 1 else 0.0,
        })
    return pd.DataFrame(rows, columns=cols)


def _features_path() -> Path:
    return _data_dir() / "comention_features.csv"


def build_and_persist_features() -> pd.DataFrame:
    """Read the persisted graph, compute features, persist comention_features.csv."""
    edges, nodes = load_comention_graph()
    feats = compute_graph_features(edges, nodes)
    if not feats.empty:
        try:
            feats.to_csv(_features_path(), index=False)
            logger.info("comention features: %d nodes persisted", len(feats))
        except Exception as exc:
            logger.warning("Could not persist comention features: %s", exc)
    return feats


def load_comention_features() -> pd.DataFrame:
    """Read the persisted per-node feature artifact. Empty frame when not built."""
    fp = _features_path()
    if fp.exists():
        return pd.read_csv(fp)
    return pd.DataFrame(columns=["symbol", *GRAPH_FEATURE_COLS])
