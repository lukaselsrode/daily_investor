"""
ui/components/single_stock_analyzer.py — Single Stock Analyzer (Research › Universe tab).

Decision-support only — NOT financial advice and places NO orders. Renders the structured
output of ui.services.single_stock_service (which wraps research.single_stock_analyzer):
holdings exposure, cached factor row, yfinance price/trend + fundamentals, news + social
evidence (with source/provenance), leveraged-ETF diagnostics + risk notes, and a hypothetical
position-structure helper. This component renders only — it contains no business logic and never
imports or calls broker/execution code.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st


def _pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "n/a"
        return f"{float(x) * 100:+.1f}%"
    except Exception:
        return "n/a"


def _money(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v):
        return "n/a"
    for cut, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= cut:
            return f"${v / cut:.2f}{suf}"
    return f"${v:,.0f}"


def _render_exposure(res) -> None:
    exp = res.exposure
    st.markdown("#### Current portfolio exposure")
    if exp.status != "ok":
        st.info(f"Holdings snapshot: {exp.status}")
        return
    cols = st.columns(max(1, len(exp.positions) + 1))
    cols[0].metric("Portfolio total", _money(exp.total_equity))
    for i, (sym, p) in enumerate(exp.positions.items(), start=1):
        if i >= len(cols):
            break
        share = (p.get("equity") or 0.0) / exp.total_equity * 100 if exp.total_equity else 0.0
        cols[i].metric(sym, _money(p.get("equity")), f"{share:.1f}% of book")
    if not exp.positions:
        st.caption("None of the requested symbols are currently held.")


def _render_price_table(res) -> None:
    st.markdown("#### Price / trend")
    rows = []
    for sym, t in res.price_trends.items():
        if t.error:
            rows.append({"symbol": sym, "status": t.error})
            continue
        rows.append({
            "symbol": sym, "price": None if t.price is None else f"${t.price:,.2f}",
            "5d": _pct(t.returns.get("5d")), "1m": _pct(t.returns.get("1m")),
            "3m": _pct(t.returns.get("3m")), "6m": _pct(t.returns.get("6m")),
            "1y": _pct(t.returns.get("1y")),
            "vs 20SMA": _pct(t.sma20_gap), "vs 50SMA": _pct(t.sma50_gap),
            "vs 200SMA": _pct(t.sma200_gap), "from 52w high": _pct(t.from_52w_high),
            "20d ann vol": _pct(t.vol20_ann),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_factors(res) -> None:
    st.markdown("#### Repo cached factor scores")
    if not res.cached_factors:
        st.caption("No cached agg_data row for this symbol (run fetch-data to populate).")
        return
    st.dataframe(pd.DataFrame([res.cached_factors]), hide_index=True, use_container_width=True)


def _render_fundamentals(res) -> None:
    f = res.fundamentals or {}
    if f.get("status"):
        st.caption(f"Fundamentals: {f['status']}")
        return
    money_keys = {"marketCap", "freeCashflow", "totalCash", "totalDebt"}
    pct_keys = {"profitMargins", "operatingMargins", "grossMargins", "revenueGrowth",
                "earningsGrowth"}
    disp = {}
    for k, v in f.items():
        if k in money_keys:
            disp[k] = _money(v)
        elif k in pct_keys:
            disp[k] = _pct(v)
        else:
            disp[k] = "n/a" if v is None else str(v)
    with st.expander("yfinance fundamentals / analyst snapshot", expanded=False):
        st.dataframe(pd.DataFrame([disp]).T.rename(columns={0: "value"}), use_container_width=True)


def _render_news(res) -> None:
    if not res.news:
        return
    st.markdown("#### News (yfinance)")
    for n in res.news[:8]:
        link = n.get("link") or ""
        title = n.get("title") or "(untitled)"
        src = n.get("publisher") or n.get("api_source") or "yfinance"
        st.markdown(f"- {f'[{title}]({link})' if link else title} — *{src}*")


def _render_social(res) -> None:
    soc = res.social
    if soc is None:
        return
    st.markdown("#### Social evidence (Reddit / X)")
    st.caption(
        f"{soc.quality_docs} kept from {soc.raw_docs} raw after spam/dedupe • "
        f"mentions: {soc.mentions or '—'} • sources are shown for provenance "
        "(no pre-aggregated social score is used)."
    )
    if soc.statuses:
        st.caption(f"fetch statuses: {soc.statuses}")
    if not soc.evidence:
        st.caption("No quality-filtered social evidence mentioning this symbol.")
        return
    for d in soc.evidence:
        age = f"{d['age_hours']}h ago" if d.get("age_hours") is not None else "time?"
        url = d.get("url") or ""
        title = d.get("title") or ""
        head = f"[{d.get('source')} / {age} / score {int(d.get('score') or 0)}]"
        st.markdown(f"- {head} {f'[{title}]({url})' if url else title}")


def _render_leverage(res) -> None:
    lev = res.leverage
    if lev is None:
        return
    st.markdown(f"#### Leverage diagnostics — {lev.leverage_symbol} vs {lev.base_symbol}")
    if lev.note and not lev.periods:
        st.info(lev.note)
        return
    cols = st.columns(2)
    cols[0].metric("Realized daily beta", "n/a" if lev.realized_daily_beta is None
                   else f"{lev.realized_daily_beta:.2f}")
    cols[1].metric("Daily correlation", "n/a" if lev.daily_corr is None
                   else f"{lev.daily_corr:.2f}")
    if lev.periods:
        rows = [{
            "period": label, lev.base_symbol: _pct(d.get("base")),
            f"{lev.leverage_symbol} actual": _pct(d.get("lev")),
            "daily-reset 2x synthetic": _pct(d.get("daily_2x_synth")),
            "tracking gap": _pct(d.get("tracking_gap")),
        } for label, d in lev.periods.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.warning(
        "A daily-reset leveraged ETF compounds daily and is path-dependent: over multi-day "
        "horizons it can diverge sharply from 2× the underlying's move (the 'tracking gap' "
        "above) and lose value even if the underlying rises. Treat it as tactical/short-horizon "
        "exposure with a predefined holding-period and loss rule — not a long-term 2× wrapper."
    )


def _render_options(res) -> None:
    opt = res.options or {}
    if not opt or opt.get("status") == "skipped":
        return
    with st.expander("Options surface snapshot (NOT a trade recommendation)", expanded=False):
        st.caption(f"Nearest expiry / status: {opt.get('first_expiry', opt.get('status'))}")
        for side in ("calls_top_oi", "puts_top_oi"):
            if opt.get(side):
                st.markdown(f"**{side.replace('_', ' ')}**")
                st.dataframe(pd.DataFrame(opt[side]), hide_index=True, use_container_width=True)


def _render_position_helper(res) -> None:
    st.markdown("#### Position structure helper")
    st.caption(
        "Hypothetical sizing against your latest portfolio total — **no orders are placed or "
        "proposed.** Enter target sleeve percentages to see dollar targets."
    )
    total = res.exposure.total_equity or 0.0
    c1, c2, c3 = st.columns(3)
    common_pct = c1.number_input("Target common %", 0.0, 100.0, 10.0, step=1.0,
                                 key="ssa_common_pct")
    levered_pct = c2.number_input("Target levered %", 0.0, 100.0, 0.0, step=1.0,
                                  key="ssa_levered_pct")
    cash_pct = c3.number_input("Cash reserve %", 0.0, 100.0, 5.0, step=1.0, key="ssa_cash_pct")
    from ui.services.single_stock_service import position_targets
    tgt = position_targets(total, common_pct, levered_pct, cash_pct)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio total", _money(tgt["total_equity"]))
    m2.metric(f"{res.symbol} common", _money(tgt["common_dollars"]))
    m3.metric(f"{res.leverage_symbol or 'Levered'} ", _money(tgt["levered_dollars"]))
    m4.metric("Cash reserve", _money(tgt["cash_dollars"]))
    if tgt["warning"]:
        st.warning(tgt["warning"])
    else:
        st.caption(f"Allocated {tgt['allocated_pct']:.0f}% • unallocated "
                   f"{tgt['unallocated_pct']:.0f}%")


def _render_result(res) -> None:
    st.warning(res.disclaimer)
    st.caption(f"Generated {res.generated_at} • fetch: {res.statuses.get('fetch', '?')}")
    _render_exposure(res)
    _render_price_table(res)
    _render_factors(res)
    _render_fundamentals(res)
    _render_news(res)
    _render_social(res)
    _render_leverage(res)
    _render_options(res)
    st.divider()
    _render_position_helper(res)


def render() -> None:
    st.subheader("🔎 Single Stock Analyzer")
    st.caption(
        "Decision-support only — not financial advice, places NO orders. Reuses the repo's "
        "holdings/agg cache, yfinance, and the Reddit/X social substrate."
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    symbol = c1.text_input("Symbol", value="BABA", key="ssa_symbol").strip().upper()
    leverage = (c2.text_input("Leverage ETF (optional)", value="BABU", key="ssa_lev")
                .strip().upper() or None)
    run = c3.button("Run analysis", key="ssa_run", type="primary")

    o1, o2, o3 = st.columns(3)
    allow_fetch = o1.checkbox("Allow live fetch", value=True, key="ssa_allow_fetch")
    include_social = o2.checkbox("Social scan (Reddit/X)", value=True, key="ssa_social")
    include_news = o3.checkbox("News", value=True, key="ssa_news")

    if run:
        if not symbol:
            st.error("Enter a symbol.")
            return
        from ui.services.single_stock_service import analyze_single_stock
        with st.spinner(f"Analyzing {symbol}…"):
            try:
                st.session_state["ssa_result"] = analyze_single_stock(
                    symbol, leverage, allow_fetch=allow_fetch,
                    include_social=include_social, include_news=include_news)
            except Exception as exc:
                st.session_state.pop("ssa_result", None)
                st.error(f"Analysis failed: {exc}")
                return

    res = st.session_state.get("ssa_result")
    if res is None:
        st.info("Enter a symbol and click **Run analysis**.")
        return
    _render_result(res)
