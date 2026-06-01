"""
ui/components/news_graph.py — News co-mention graph + evolution view.

Three tabs:
  📈 Evolution   — graph-level metrics over snapshot dates (size, communities,
                   sentiment, dispersion). Tracks how the news network changes.
  🕸️ Network     — force-directed co-mention graph for a chosen date, nodes colored
                   by sentiment or community, sized by degree.
  🎯 Ego network — k-hop neighborhood of a chosen symbol (e.g. a held name), to see
                   which peers it is co-mentioned with.

SAFE: read-only. Renders the persisted/derived graph artifact; never writes anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def _evolution(held: tuple[str, ...]) -> pd.DataFrame:
    from portfolio.visualization.news_graph import graph_evolution_summary
    return graph_evolution_summary(held_symbols=set(held))


@st.cache_data(ttl=300, show_spinner=False)
def _dates() -> list[str]:
    from portfolio.visualization.news_graph import list_news_dates
    return list_news_dates()


@st.cache_data(ttl=300, show_spinner=False)
def _graph(date: str, held: tuple[str, ...]):
    from portfolio.visualization.news_graph import build_graph_for_date
    return build_graph_for_date(date, set(held))


@st.cache_data(ttl=300, show_spinner=False)
def _ego(date: str, symbol: str, hops: int):
    from portfolio.visualization.news_graph import node_neighborhood
    return node_neighborhood(date, symbol, hops)


@st.cache_data(ttl=300, show_spinner=False)
def _sector_evolution(group_by: str = "sector") -> pd.DataFrame:
    from portfolio.visualization.news_graph import graph_evolution_by_sector
    return graph_evolution_by_sector(group_by=group_by)


def _held_symbols() -> list[str]:
    """Best-effort load of currently-held symbols for highlighting."""
    try:
        from portfolio.visualization.factor_map import load_universe_with_holdings
        df = load_universe_with_holdings()
        if "owned" in df.columns and "symbol" in df.columns:
            return sorted(df.loc[df["owned"].astype(bool), "symbol"].astype(str).str.upper().tolist())
    except Exception:
        pass
    return []


def _spring_layout(edges: pd.DataFrame, nodes_index: list[str]) -> dict:
    """networkx spring layout -> {symbol: (x, y)}. Falls back to circle if needed."""
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(nodes_index)
        for r in edges.itertuples(index=False):
            G.add_edge(r.source, r.target, weight=getattr(r, "weight", 1))
        pos = nx.spring_layout(G, seed=7, k=0.5, iterations=50, weight="weight")
        return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}
    except Exception:
        n = len(nodes_index)
        return {s: (np.cos(2 * np.pi * i / max(n, 1)), np.sin(2 * np.pi * i / max(n, 1)))
                for i, s in enumerate(nodes_index)}


def _draw_network(edges: pd.DataFrame, feats: pd.DataFrame, color_by: str,
                  held: set[str], title: str, max_nodes: int = 400):
    import plotly.graph_objects as go

    if feats.empty or edges.empty:
        st.info("No co-mention edges for this date.")
        return

    # cap to the highest-degree nodes for legibility
    f = feats.copy()
    if len(f) > max_nodes:
        f = f.nlargest(max_nodes, "news_degree")
    keep = set(f["symbol"])
    e = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]

    pos = _spring_layout(e, list(keep))
    fmap = f.set_index("symbol")

    # edge trace
    ex, ey = [], []
    for r in e.itertuples(index=False):
        if r.source in pos and r.target in pos:
            x0, y0 = pos[r.source]
            x1, y1 = pos[r.target]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
                            hoverinfo="none", showlegend=False)

    # node trace
    nx_, ny_, color, size, text = [], [], [], [], []
    for sym in keep:
        if sym not in pos:
            continue
        x, y = pos[sym]
        nx_.append(x); ny_.append(y)
        row = fmap.loc[sym]
        deg = float(row.get("news_degree", 1))
        size.append(8 + 3 * np.sqrt(max(deg, 1)))
        if color_by == "sentiment":
            color.append(float(row.get("news_sentiment", 0.0)))
        else:  # community
            color.append(float(row.get("news_community", -1)))
        held_tag = " ★HELD" if sym in held else ""
        text.append(f"{sym}{held_tag}<br>deg={int(deg)} "
                    f"sent={row.get('news_sentiment', 0):.2f} "
                    f"comm={int(row.get('news_community', -1))} "
                    f"size={int(row.get('news_community_size', 0))}")

    cscale = "RdYlGn" if color_by == "sentiment" else "Turbo"
    node_trace = go.Scatter(
        x=nx_, y=ny_, mode="markers", text=text, hoverinfo="text",
        marker=dict(size=size, color=color, colorscale=cscale, showscale=True,
                    colorbar=dict(title=color_by), line=dict(width=0.5, color="#222")),
        showlegend=False,
    )
    # held-name ring overlay
    hx, hy = [], []
    for sym in keep:
        if sym in held and sym in pos:
            hx.append(pos[sym][0]); hy.append(pos[sym][1])
    overlays = []
    if hx:
        overlays.append(go.Scatter(x=hx, y=hy, mode="markers",
                        marker=dict(size=18, color="rgba(0,0,0,0)",
                                    line=dict(width=2.5, color="gold")),
                        hoverinfo="none", name="Held"))

    fig = go.Figure([edge_trace, node_trace, *overlays])
    fig.update_layout(title=title, height=620, showlegend=bool(hx),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.subheader("News Co-Mention Graph")
    st.caption(
        "Stocks linked when they appear in the same news article (edge weight = shared "
        "articles). Built from the dated news snapshots — a window into how the market's "
        "attention network and sentiment evolve. Read-only."
    )

    dates = _dates()
    if not dates:
        st.info("No dated `news_*.csv` snapshots found in the data dir yet.")
        return

    held = _held_symbols()
    held_t = tuple(held)
    tabs = st.tabs(["📈 Evolution", "📊 Sector breakdown", "🕸️ Network", "🎯 Ego network"])

    # ── Evolution ─────────────────────────────────────────────────────────────
    with tabs[0]:
        ev = _evolution(held_t)
        if ev.empty:
            st.info("Could not build evolution summary.")
        else:
            import plotly.graph_objects as go
            st.caption(f"{len(ev)} snapshot dates · {ev['n_connected'].iloc[-1]} connected "
                       f"nodes / {ev['n_edges'].iloc[-1]} edges on the latest.")
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_scatter(x=ev["date"], y=ev["n_connected"], name="connected nodes")
                fig.add_scatter(x=ev["date"], y=ev["n_edges"], name="edges", yaxis="y2")
                fig.update_layout(title="Graph size over time", height=320,
                                  yaxis=dict(title="nodes"),
                                  yaxis2=dict(title="edges", overlaying="y", side="right"),
                                  margin=dict(l=10, r=10, t=40, b=10),
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                fig.add_scatter(x=ev["date"], y=ev["mean_sentiment"], name="mean sentiment")
                fig.add_scatter(x=ev["date"], y=ev["neg_fraction"], name="neg fraction")
                fig.add_scatter(x=ev["date"], y=ev["mean_dispersion"], name="dispersion")
                fig.update_layout(title="Sentiment & disagreement over time", height=320,
                                  margin=dict(l=10, r=10, t=40, b=10),
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "⚠️ Research note: aggregate graph sentiment showed a *suggestive* "
                "contrarian tie to forward market returns (high mood → softer next-week "
                "returns) but n≈13 dates — NOT confirmed. Watch it accumulate; do not trade it yet."
            )
            with st.expander("Evolution table"):
                st.dataframe(ev, use_container_width=True, hide_index=True)

    # ── Sector breakdown ──────────────────────────────────────────────────────
    with tabs[1]:
        gb_choice = st.radio("Group by", ["sector", "industry"], horizontal=True,
                             key="ng_sector_gb")
        with st.spinner("Building sector breakdown…"):
            sec_df = _sector_evolution(gb_choice)

        if sec_df.empty:
            st.info("No sector data available — agg_data may not have sector/industry columns yet.")
        else:
            import plotly.express as px
            import plotly.graph_objects as go

            # Focus on top-N groups by peak attention share (avoids clutter)
            top_n = st.slider("Show top N sectors by peak attention", 5, 20, 10,
                              key="ng_sector_topn")
            peak = sec_df.groupby("group")["attention_share"].max().nlargest(top_n)
            sec_filtered = sec_df[sec_df["group"].isin(peak.index)].copy()

            # ── Chart 1: Attention share heatmap (which sectors dominate news) ──
            st.subheader("News attention share by sector over time")
            st.caption("Fraction of all connected graph nodes in each sector on each date. "
                       "High attention = sector sweeping up many co-mention stories.")
            pivot_att = (sec_filtered.pivot_table(
                index="group", columns="date",
                values="attention_share", aggfunc="sum", fill_value=0.0
            ))
            # Ensure column labels are strings so Plotly renders dates not integers
            pivot_att.columns = [str(c) for c in pivot_att.columns]
            fig_att = px.imshow(
                pivot_att,
                color_continuous_scale="Blues",
                aspect="auto",
                labels=dict(x="Date", y=gb_choice.capitalize(), color="Attention share"),
                x=list(pivot_att.columns),
                y=list(pivot_att.index),
            )
            fig_att.update_layout(height=40 * min(top_n, len(pivot_att)) + 80,
                                  margin=dict(l=10, r=10, t=10, b=60))
            fig_att.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_att, use_container_width=True)

            # ── Chart 2: Sentiment heatmap (which sectors are positive/negative) ──
            st.subheader("Mean sentiment by sector over time")
            st.caption("Positive = bullish news tone in that sector. "
                       "Diverging sectors signal rotation or concentrated risk events.")
            pivot_sent = (sec_filtered.pivot_table(
                index="group", columns="date",
                values="mean_sentiment", aggfunc="mean", fill_value=0.0
            ))
            pivot_sent.columns = [str(c) for c in pivot_sent.columns]
            fig_sent = px.imshow(
                pivot_sent,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                zmin=-0.3, zmax=0.3,
                labels=dict(x="Date", y=gb_choice.capitalize(), color="Sentiment"),
                x=list(pivot_sent.columns),
                y=list(pivot_sent.index),
            )
            fig_sent.update_layout(height=40 * min(top_n, len(pivot_sent)) + 80,
                                   margin=dict(l=10, r=10, t=10, b=60))
            fig_sent.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_sent, use_container_width=True)

            # ── Chart 3: Cross-sector ratio over time ───────────────────────────
            st.subheader("Cross-sector narrative ratio over time")
            st.caption(
                "cross_ratio = cross-sector edges ÷ (intra + cross) for each sector. "
                "Bounded 0–1. Near 1 = sector is swept up in macro narratives linking "
                "it to other sectors. Near 0 = sector stories are self-contained. "
                "A sudden ratio spike signals a broad market event touching that sector."
            )
            fig_cross = go.Figure()
            all_dates = sorted(sec_filtered["date"].astype(str).unique())
            for grp in peak.index[:8]:  # limit to 8 for legibility
                g = sec_filtered[sec_filtered["group"] == grp].sort_values("date")
                if "cross_ratio" not in g.columns:
                    continue
                # Reindex to all dates so missing dates become explicit gaps (not
                # interpolated lines that can shoot outside [0,1] on sparse data)
                g_full = (pd.DataFrame({"date": all_dates})
                          .merge(g[["date", "cross_ratio"]].assign(date=g["date"].astype(str)),
                                 on="date", how="left"))
                fig_cross.add_scatter(x=g_full["date"], y=g_full["cross_ratio"],
                                      name=grp, mode="lines+markers",
                                      marker=dict(size=4),
                                      connectgaps=False)
            fig_cross.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=60),
                                    legend=dict(orientation="h"),
                                    yaxis=dict(title="Cross-sector ratio", range=[0, 1]),
                                    xaxis=dict(type="category", categoryorder="array",
                                               categoryarray=all_dates, tickangle=-45))
            st.plotly_chart(fig_cross, use_container_width=True)

            # ── Latest snapshot snapshot table ────────────────────────────────
            latest_date = sec_df["date"].max()
            latest = (sec_df[sec_df["date"] == latest_date]
                      .sort_values("attention_share", ascending=False)
                      .reset_index(drop=True))
            with st.expander(f"Latest snapshot detail — {latest_date}"):
                st.dataframe(
                    latest[["group", "n_nodes", "mean_sentiment",
                             "neg_fraction", "intra_edges", "cross_edges",
                             "cross_ratio", "attention_share"]].rename(columns={
                        "group": gb_choice, "n_nodes": "nodes",
                        "mean_sentiment": "sentiment", "neg_fraction": "neg %",
                        "intra_edges": "intra-edges", "cross_edges": "cross-edges",
                        "cross_ratio": "cross ratio", "attention_share": "attention %",
                    }),
                    use_container_width=True, hide_index=True,
                )

    # ── Network ───────────────────────────────────────────────────────────────
    with tabs[2]:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            date = st.selectbox("Snapshot date", dates, index=len(dates) - 1, key="ng_date")
        with c2:
            color_by = st.radio("Color nodes by", ["sentiment", "community"],
                                horizontal=True, key="ng_color")
        with c3:
            max_nodes = st.select_slider("Max nodes", [100, 200, 400, 600], value=400, key="ng_max")
        edges, nodes, feats = _graph(date, held_t)
        _draw_network(edges, feats, color_by, set(held),
                      f"Co-mention network · {date}", max_nodes=max_nodes)

    # ── Ego network ───────────────────────────────────────────────────────────
    with tabs[3]:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            date2 = st.selectbox("Snapshot date", dates, index=len(dates) - 1, key="ng_ego_date")
        # prefer held names in the picker
        _, _, feats2 = _graph(date2, held_t)
        graph_syms = sorted(feats2["symbol"].tolist()) if not feats2.empty else []
        default_sym = next((s for s in held if s in graph_syms), graph_syms[0] if graph_syms else "")
        with c2:
            # Always instantiate the selectbox (disabled when the graph is empty) so the
            # widget key is stable across reruns and dates.
            options = graph_syms if graph_syms else ["—"]
            sel = st.selectbox(
                "Symbol", options,
                index=options.index(default_sym) if default_sym in options else 0,
                key="ng_ego_sym", disabled=not graph_syms,
            )
            sym = sel if graph_syms else ""
        with c3:
            hops = st.select_slider("Hops", [1, 2, 3], value=1, key="ng_ego_hops")
        if sym:
            e, n = _ego(date2, sym, hops)
            if e.empty:
                st.info(f"{sym} has no co-mention edges on {date2}.")
            else:
                _draw_network(e, n, "sentiment", {sym} | set(held),
                              f"{sym} · {hops}-hop co-mention neighborhood · {date2}",
                              max_nodes=200)
                st.dataframe(e.sort_values("weight", ascending=False),
                             use_container_width=True, hide_index=True)
