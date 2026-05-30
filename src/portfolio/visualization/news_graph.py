"""
portfolio/visualization/news_graph.py — read-only helpers for the news co-mention
graph evolution UI.

Builds per-date graph snapshots from the dated ``news_YYYY_MM_DD*.csv`` files in the
data dir, reusing the ETL builder (data.comention_graph). Returns tidy structures the
UI renders with plotly/networkx. SAFE: read-only, never writes config/orders/data.
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd


def _data_dir():
    from core.paths import DATA_DIR
    return DATA_DIR


def list_news_dates() -> list[str]:
    """Return sorted distinct calendar dates (YYYY_MM_DD) with a news snapshot.

    When multiple intraday files exist for a date, the latest is used downstream.
    """
    out: set[str] = set()
    for f in glob.glob(os.path.join(_data_dir(), "news_2*.csv")):
        m = re.search(r"news_(\d{4}_\d{2}_\d{2})", os.path.basename(f))
        if m:
            out.add(m.group(1))
    return sorted(out)


def _news_file_for_date(date: str) -> str | None:
    """Latest news file for a calendar date (handles intraday suffixes)."""
    matches = sorted(glob.glob(os.path.join(_data_dir(), f"news_{date}*.csv")))
    return matches[-1] if matches else None


def build_graph_for_date(date: str, held_symbols: set[str] | None = None):
    """Build (edges_df, nodes_df, features_df) for one snapshot date. Empty frames
    when the file is missing or unparseable."""
    from data.comention_graph import (
        build_comention_graph,
        compute_graph_features,
    )
    path = _news_file_for_date(date)
    if path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        news_df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    edges, nodes = build_comention_graph(
        news_df=news_df, held_symbols=held_symbols or set(), persist=False
    )
    feats = compute_graph_features(edges, nodes)
    return edges, nodes, feats


def graph_evolution_summary(held_symbols: set[str] | None = None) -> pd.DataFrame:
    """One row per snapshot date with graph-level metrics over time.

    Columns: date, n_nodes, n_edges, n_connected, density, n_communities,
             largest_community, mean_sentiment, neg_fraction, mean_dispersion,
             n_held_in_graph.
    """
    rows = []
    held = {str(s).strip().upper() for s in (held_symbols or set())}
    for date in list_news_dates():
        edges, nodes, feats = build_graph_for_date(date, held)
        if nodes.empty:
            continue
        connected = nodes[nodes["degree"] > 0] if "degree" in nodes.columns else nodes
        n_nodes = len(nodes)
        n_conn = len(connected)
        n_edges = len(edges)
        density = (2.0 * n_edges / (n_conn * (n_conn - 1))) if n_conn > 1 else 0.0
        n_comm = int(feats["news_community"].nunique()) if not feats.empty and "news_community" in feats else 0
        largest = int(feats["news_community_size"].max()) if not feats.empty and "news_community_size" in feats else 0
        sent = nodes["sentiment"] if "sentiment" in nodes.columns else pd.Series(dtype=float)
        disp = feats["news_sent_dispersion"] if not feats.empty and "news_sent_dispersion" in feats else pd.Series(dtype=float)
        n_held = int(connected["symbol"].isin(held).sum()) if held and "symbol" in connected.columns else 0
        rows.append({
            "date": date,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_connected": n_conn,
            "density": round(density, 5),
            "n_communities": n_comm,
            "largest_community": largest,
            "mean_sentiment": round(float(sent.mean()), 4) if len(sent) else 0.0,
            "neg_fraction": round(float((sent < 0).mean()), 4) if len(sent) else 0.0,
            "mean_dispersion": round(float(disp.mean()), 4) if len(disp) else 0.0,
            "n_held_in_graph": n_held,
        })
    return pd.DataFrame(rows)


def node_neighborhood(date: str, symbol: str, hops: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sub_edges, sub_nodes) for the k-hop neighborhood of `symbol` on `date`.

    Used to render the ego-network of a held name (e.g. who NVO is co-mentioned with).
    """
    symbol = str(symbol).strip().upper()
    edges, nodes, feats = build_graph_for_date(date)
    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "weight"]), pd.DataFrame()
    keep = {symbol}
    frontier = {symbol}
    for _ in range(max(1, hops)):
        nxt: set[str] = set()
        for r in edges.itertuples(index=False):
            if r.source in frontier:
                nxt.add(r.target)
            elif r.target in frontier:
                nxt.add(r.source)
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if not frontier:
            break
    sub_e = edges[edges["source"].isin(keep) & edges["target"].isin(keep)].copy()
    sub_n = feats[feats["symbol"].isin(keep)].copy() if not feats.empty else pd.DataFrame()
    return sub_e, sub_n
