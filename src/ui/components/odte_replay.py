"""ui/components/odte_replay.py — 0DTE day replay.

Replays one session from its day packet: the tape (price vs VWAP vs the opening range) with entry,
exit, lease and refusal markers laid on it, the day-score headroom, and the day's postmortem.

Honest about density: recent packets hold only a handful of market snapshots, so when there are too
few points to draw a line this renders markers on a scatter and says so. It never interpolates a
tape it does not have.

Read-only; no orders, no broker.
"""
from __future__ import annotations

import streamlit as st

from ui.services import odte_service as svc

_SYMBOLS = ("SPY", "QQQ", "IWM", "VIXY")


def _render_tape(packet: dict, symbol: str, markers: dict) -> None:
    frames = svc.tape_frames(packet, symbol)
    if not frames:
        st.info(f"No {symbol} price points in this packet.")
        return

    density = packet.get("tape_density")
    import plotly.graph_objects as go
    fig = go.Figure()
    mode = "lines+markers" if density == "dense" else "markers"
    fig.add_trace(go.Scatter(x=[f["ts"] for f in frames], y=[f["last"] for f in frames],
                             mode=mode, name=symbol, line=dict(color="#3498db"),
                             marker=dict(size=7)))
    vwap = [(f["ts"], f["vwap"]) for f in frames if f.get("vwap") is not None]
    if vwap:
        fig.add_trace(go.Scatter(x=[v[0] for v in vwap], y=[v[1] for v in vwap],
                                 mode="lines", name="VWAP",
                                 line=dict(color="#f39c12", dash="dot")))
    highs = [f["orb_high"] for f in frames if f.get("orb_high") is not None]
    lows = [f["orb_low"] for f in frames if f.get("orb_low") is not None]
    if highs and lows:
        fig.add_hrect(y0=min(lows), y1=max(highs), fillcolor="#7f8c8d", opacity=0.14,
                      line_width=0, annotation_text="opening range")

    for m in markers.get("entries", []):
        if m.get("underlying") == symbol:
            fig.add_vline(x=m["ts"], line_color="#2ecc71", line_dash="dash",
                          annotation_text="entry")
    for m in markers.get("exits", []):
        if m.get("underlying") == symbol:
            fig.add_vline(x=m["ts"], line_color="#6a1b9a", line_dash="dash",
                          annotation_text=m.get("rail_fired") or "exit")

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="price", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch", key=f"odte_replay_tape_{symbol}")

    n = len(frames)
    if density == "markers":
        st.warning(f"**{n} snapshot(s)** for {symbol} — too few to draw a tape, so these are the "
                   "recorded points only. No interpolation.")
    elif density == "sparse":
        st.caption(f"{n} snapshots — a coarse tape. Treat gaps as unobserved, not as flat.")
    else:
        st.caption(f"{n} snapshots.")

    archived = packet.get("archive_snapshots") or 0
    if archived > n:
        st.caption(f"ℹ️ {archived} higher-resolution controller snapshots for this date sit in "
                   "`data/odte/archive/` but are not folded into the packet — the cleanup runs "
                   "before ingest sees them.")


def _render_markers(markers: dict) -> None:
    st.markdown("#### Decisions on the day")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries", len(markers.get("entries") or []))
    c2.metric("Exits", len(markers.get("exits") or []))
    c3.metric("Leases", len(markers.get("leases") or []))
    c4.metric("Refusals", len(markers.get("refusals") or []))

    refusals = markers.get("refusals") or []
    if refusals:
        import pandas as pd
        with st.expander(f"Refusals ({len(refusals)})"):
            st.dataframe(pd.DataFrame([{
                "ts": r.get("ts"), "sym": r.get("underlying"), "stage": r.get("stage") or "unknown",
                "reasons": ", ".join(r.get("reason_codes") or []),
            } for r in refusals]), width="stretch", hide_index=True)

    evaluations = markers.get("evaluations") or []
    if evaluations:
        import pandas as pd
        with st.expander(f"Candidate evaluations ({len(evaluations)})"):
            st.dataframe(pd.DataFrame(evaluations), width="stretch", hide_index=True)


def _render_day_score(trade_date: str) -> None:
    st.markdown("#### Day score & headroom")
    series = [r for r in svc.day_score_series(days=400) if r["date"] == str(trade_date)]
    if not series:
        st.caption("No day-score events for this date. First-party journaling began 2026-08-06; "
                   "earlier dates only have whatever the end-of-day artifact sweep caught.")
        return

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["ts"] for r in series], y=[r["score"] for r in series],
                             mode="lines+markers", name="score", line=dict(color="#3498db")))
    fig.add_trace(go.Scatter(x=[r["ts"] for r in series],
                             y=[r["max_possible_score"] for r in series],
                             mode="lines", name="max possible",
                             line=dict(color="#7f8c8d", dash="dot")))
    fig.add_hline(y=svc.good_day_min_score(), line_dash="dash", line_color="#2ecc71",
                  annotation_text="GOOD_DAY threshold")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.2))
    st.plotly_chart(fig, width="stretch", key="odte_replay_dayscore")

    last = series[-1]
    missing = last.get("components_missing") or []
    st.caption(f"Final verdict **{last.get('verdict')}** · score {last.get('score')} · "
               f"{last.get('components_supplied')} components supplied"
               + (f" · missing: {', '.join(missing)}" if missing else ""))
    if (last.get("max_possible_score") is not None
            and last["max_possible_score"] < svc.good_day_min_score()):
        st.warning("Headroom check: with the components actually supplied, a GOOD_DAY was "
                   "**structurally unreachable** — the ceiling sat below the threshold.")


def _render_postmortem(packet: dict) -> None:
    st.markdown("#### Postmortem")
    pm = packet.get("postmortem")
    if not pm:
        st.caption("No postmortem for this date.")
        return
    if packet.get("postmortem_generated_differs"):
        st.caption("✍️ Human-edited — shown below. A regenerated draft exists alongside it "
                   "(`postmortem.generated.md`) and is not overwriting your notes.")
    st.markdown(pm)


def render() -> None:
    st.subheader("Day replay")
    st.caption("One session, replayed from its day packet: tape, decisions, day-score headroom, "
               "and the written postmortem.")

    dates = svc.day_index()
    if not dates:
        st.info("No day packets yet — build one with `odte-day-packet <date>`.")
        return

    c1, c2 = st.columns([1, 1])
    trade_date = c1.selectbox("Date", dates, key="odte_replay_date")
    symbol = c2.selectbox("Symbol", _SYMBOLS, key="odte_replay_symbol")

    packet = svc.day_packet(trade_date)
    if not packet.get("exists"):
        st.error(f"No packet directory for {trade_date}.")
        return

    counts = {k: len(packet.get(k) or []) for k in
              ("market_snapshots", "candidates", "vehicle_scores", "trades", "controller_events")}
    st.caption(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in counts.items()))
    if packet.get("trades_semantics") != "lifecycle":
        st.caption("⚠️ On this date `trades.jsonl` still included monitoring polls and gate vetoes "
                   "— its row count is not a trade count. (Changed 2026-08-05.)")

    markers = svc.day_markers(trade_date)
    _render_tape(packet, symbol, markers)
    st.divider()
    _render_markers(markers)
    st.divider()
    _render_day_score(trade_date)
    st.divider()
    _render_postmortem(packet)
