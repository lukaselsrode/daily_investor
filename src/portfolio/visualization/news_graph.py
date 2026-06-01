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


def graph_evolution_by_sector(
    group_by: str = "sector",
) -> pd.DataFrame:
    """Per-date × per-sector breakdown of graph metrics.

    Returns a tidy DataFrame with columns:
        date, group (sector or industry), n_nodes, mean_sentiment,
        neg_fraction, intra_edges, cross_edges, attention_share

    ``attention_share`` = fraction of all connected nodes on that date that
    belong to this sector — a normalised measure of how much news attention
    the sector is capturing relative to the rest of the universe.
    ``intra_edges`` = edges where both endpoints share the same group.
    ``cross_edges`` = edges where endpoints are in different groups.
    """
    import numpy as np
    try:
        from util import read_data_as_pd
        agg = read_data_as_pd("agg_data")
        if group_by not in agg.columns or "symbol" not in agg.columns:
            return pd.DataFrame()
        sym_to_group = (
            agg.dropna(subset=[group_by, "symbol"])
            .set_index("symbol")[group_by]
            .str.strip()
            .to_dict()
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for date in list_news_dates():
        edges, nodes, _feats = build_graph_for_date(date)
        if nodes.empty:
            continue
        connected = (
            nodes[nodes["degree"] > 0].copy()
            if "degree" in nodes.columns
            else nodes.copy()
        )
        connected["group"] = connected["symbol"].map(sym_to_group)
        connected = connected.dropna(subset=["group"])
        if connected.empty:
            continue
        total_conn = max(len(connected), 1)

        # Precompute group mappings once per date (outside the inner loop)
        if not edges.empty:
            src_grps = edges["source"].map(sym_to_group)
            tgt_grps = edges["target"].map(sym_to_group)
        else:
            src_grps = tgt_grps = pd.Series(dtype=object)

        for grp, grp_nodes in connected.groupby("group"):
            sent = grp_nodes["sentiment"] if "sentiment" in grp_nodes.columns else pd.Series(dtype=float)
            n_nodes = len(grp_nodes)
            intra = cross = 0
            cross_ratio = 0.0
            if not edges.empty:
                # intra: both endpoints in this group
                intra = int(((src_grps == grp) & (tgt_grps == grp)).sum())
                # cross: exactly one endpoint in this group (each edge counted once)
                cross = int(
                    ((src_grps == grp) & (tgt_grps != grp)).sum()
                    + ((src_grps != grp) & (tgt_grps == grp)).sum()
                )
                total_touching = intra + cross
                cross_ratio = round(cross / total_touching, 4) if total_touching > 0 else 0.0
            _sv = np.asarray(sent, dtype=float) if len(sent) else np.array([], dtype=float)
            mean_sent = float(_sv.mean()) if len(_sv) else 0.0
            neg_frac = float((_sv < 0).mean()) if len(_sv) else 0.0
            rows.append({
                "date": date,
                "group": grp,
                "n_nodes": n_nodes,
                "mean_sentiment": round(mean_sent, 4),
                "neg_fraction": round(neg_frac, 4),
                "intra_edges": intra,
                "cross_edges": cross,
                "cross_ratio": cross_ratio,  # cross / (intra+cross), bounded [0,1]
                "attention_share": round(n_nodes / total_conn, 4),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


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
