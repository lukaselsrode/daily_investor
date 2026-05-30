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
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Crude, transparent title-sentiment lexicon (intentionally small — this is a
# coarse directional tag, not an NLP model). Lives here so the graph artifact is
# self-contained ETL output.
_POS = frozenset({
    "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "beat",
    "beats", "gain", "gains", "rise", "rises", "record", "strong", "growth", "upgrade",
    "outperform", "buy", "bullish", "boom", "wins", "win", "profit", "tops", "rebound",
    "raises", "raise", "high", "highs",
})
_NEG = frozenset({
    "fall", "falls", "drop", "drops", "plunge", "plunges", "slump", "sink", "sinks",
    "miss", "misses", "loss", "losses", "weak", "cut", "cuts", "downgrade", "sell",
    "bearish", "crash", "fear", "fears", "warn", "warns", "slash", "plummet", "tumble",
    "probe", "sue", "lawsuit", "low", "lows",
})


def _data_dir() -> Path:
    from core.paths import DATA_DIR
    return DATA_DIR


def _edges_path() -> Path:
    return _data_dir() / "comention_edges.csv"


def _nodes_path() -> Path:
    return _data_dir() / "comention_nodes.csv"


def title_sentiment(text: str) -> float:
    """Crude lexical sentiment in [-1, 1] from a headline. 0.0 when no hits."""
    toks = re.findall(r"[a-zA-Z]+", (text or "").lower())
    p = sum(1 for t in toks if t in _POS)
    n = sum(1 for t in toks if t in _NEG)
    return 0.0 if (p + n) == 0 else (p - n) / (p + n)


def build_comention_graph(
    news_df: pd.DataFrame | None = None,
    held_symbols: set[str] | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (edges_df, nodes_df) from the news cache's ``related_symbols`` field.

    news_df: the persisted news table (symbol, news=json[list[article]]). When None,
    it is read from the data cache. Each article contributes:
      - an edge (article-symbol -- each related_symbol), weight summed over articles
      - title sentiment attributed to every node it touches

    Returns the two DataFrames and (when persist) writes them to the data dir.
    """
    if news_df is None:
        from data.cache import read_data_as_pd
        news_df = read_data_as_pd("news")

    edges: collections.Counter = collections.Counter()
    node_sent: dict[str, list[float]] = collections.defaultdict(list)
    node_articles: collections.Counter = collections.Counter()

    if news_df is not None and not news_df.empty and "news" in news_df.columns:
        for _, row in news_df.iterrows():
            sym = str(row.get("symbol", "")).strip().upper()
            if not sym:
                continue
            raw = row.get("news")
            try:
                articles = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                articles = []
            for art in articles:
                if not isinstance(art, dict):
                    continue
                s = title_sentiment(art.get("title", ""))
                node_sent[sym].append(s)
                node_articles[sym] += 1
                related = art.get("related_symbols") or []
                touched = {sym}
                for other in related:
                    o = str(other).strip().upper()
                    if not o or o == sym:
                        continue
                    edges[tuple(sorted((sym, o)))] += 1
                    touched.add(o)
                # attribute this title's sentiment to co-mentioned nodes too
                for o in touched - {sym}:
                    node_sent[o].append(s)

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
