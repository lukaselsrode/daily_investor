"""FMP single-name context for 0DTE meme/squeeze SANITY — NOT an entry signal, NO orders.

Enriches a single underlying with cheap FMP *stable* fundamentals (profile, quote, shares-float,
key-metrics-ttm, a few news headlines) so the controller can sanity-check a meme/squeeze candidate
on a trigger — float size, relative volume, leverage, fresh headlines. It is read-only context:

    *** NOT an entry signal. NO orders. NO options / gamma. ***

FMP OPTIONS ENDPOINTS DO NOT WORK on this plan (stable options 404; legacy options 403), so this
module NEVER fetches options/gamma from FMP and always emits ``fmp_options_available: false`` —
**Robinhood remains the option-chain / gamma source** (see ``odte_gamma_map``). This is intentionally
NOT wired into ``odte-watchdog`` (which must stay cheap / no-network); the controller calls it only
on a candidate trigger.

Design: ``build_context`` is PURE (raw FMP JSON in -> classified context out) and fully unit-tested
without network. ``fetch_fmp_raw``/``run_fmp_context`` accept an injectable ``fetch_json`` so tests
stub the network. Fail-closed: a missing FMP_KEY or any endpoint error yields partial data +
warnings, never an exception. Secrets are never returned or logged (the apikey lives only in the URL
inside the default fetcher and is never surfaced).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from core.paths import ODTE_REPORT_DIR

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = ODTE_REPORT_DIR
_BASE = "https://financialmodelingprep.com/stable"
DEFAULT_NEWS_LIMIT = 3

# Stable, non-options endpoints only (path templates; apikey appended by the default fetcher).
_ENDPOINTS = {
    "profile": "/profile?symbol={sym}",
    "quote": "/quote?symbol={sym}",
    "shares_float": "/shares-float?symbol={sym}",
    "key_metrics": "/key-metrics-ttm?symbol={sym}",
    "news": "/news/stock?symbols={sym}&limit={news_limit}",
}

# Float-size buckets (absolute float share count) -> squeeze profile.
_SQUEEZE_TINY = 20_000_000
_SQUEEZE_SMALL = 75_000_000
_SQUEEZE_MID = 300_000_000

OPTIONS_UNAVAILABLE_WARNING = (
    "FMP options endpoints unavailable (stable 404 / legacy 403) — NOT used for gamma/options. "
    "Robinhood remains the option-chain / gamma source.")
DISCLAIMER = ("Meme/squeeze SANITY context only — NOT an entry signal, no orders, no options/gamma "
              "(Robinhood is the options/gamma source).")

_TRADE_IMPLICATION = {
    "tiny_float_squeeze_candidate":
        "Very low float — violent two-way moves possible; size tiny, expect slippage, "
        "squeeze/sympathy risk high. Confirm on Robinhood before any trade.",
    "small_float_momentum":
        "Low float — momentum can run; respect VWAP/levels and manage size.",
    "mid_float_meme_momentum":
        "Moderate float — meme momentum possible but NOT a tiny-float squeeze; needs real "
        "volume/catalyst.",
    "large_float_meme_momentum_not_tiny_float":
        "Large float — NOT a tiny-float squeeze; treat as liquid momentum, do not expect "
        "low-float dynamics.",
    "no_float_data":
        "No float data — cannot assess squeeze profile; treat with caution and verify on Robinhood.",
}


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(x) -> dict:
    """FMP list endpoints return [ {...} ]; return the first dict (or x if already a dict)."""
    if isinstance(x, list):
        return x[0] if x and isinstance(x[0], dict) else {}
    return x if isinstance(x, dict) else {}


def _pick(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


# --- network (injectable; never returns the apikey) ------------------------------------------

def _default_fetch_json(path: str):
    """One live GET against FMP stable. Returns (parsed_json | None, error_str | None). The apikey
    is added here and NEVER returned — error strings carry only status codes / exception types."""
    import requests
    key = os.getenv("FMP_KEY")
    if not key:
        return None, "no_key"
    url = f"{_BASE}{path}{'&' if '?' in path else '?'}apikey={key}"
    try:
        r = requests.get(url, timeout=20)
    except Exception as exc:   # network/DNS/timeout — fail closed
        return None, f"network_error:{type(exc).__name__}"
    if r.status_code != 200:
        return None, f"http_{r.status_code}"
    try:
        return r.json(), None
    except ValueError:
        return None, "bad_json"


def fetch_fmp_raw(symbol: str, news_limit: int = DEFAULT_NEWS_LIMIT, fetch_json=None) -> tuple[dict, list]:
    """Fetch the stable non-options endpoints for `symbol`. Returns (raw_by_endpoint, warnings).
    Fail-closed: missing key or per-endpoint errors become warnings, never exceptions."""
    fetch_json = fetch_json or _default_fetch_json
    if fetch_json is _default_fetch_json and not os.getenv("FMP_KEY"):
        return {}, ["FMP_KEY not set — FMP enrichment skipped (fail-closed)"]
    sym = symbol.upper()
    raw, warnings = {}, []
    for name, tmpl in _ENDPOINTS.items():
        data, err = fetch_json(tmpl.format(sym=sym, news_limit=news_limit))
        if err:
            warnings.append(f"{name}: {err}")
        if data is not None:
            raw[name] = data
    return raw, warnings


# --- pure classification ---------------------------------------------------------------------

def classify_squeeze(float_shares: float | None) -> str:
    if float_shares is None:
        return "no_float_data"
    if float_shares < _SQUEEZE_TINY:
        return "tiny_float_squeeze_candidate"
    if float_shares < _SQUEEZE_SMALL:
        return "small_float_momentum"
    if float_shares < _SQUEEZE_MID:
        return "mid_float_meme_momentum"
    return "large_float_meme_momentum_not_tiny_float"


def build_context(symbol: str, raw: dict | None, warnings: list | None = None,
                  now: datetime | None = None) -> dict:
    """PURE: classify pre-fetched FMP JSON into a context dict. No network, no IO."""
    now = now or datetime.now(timezone.utc)
    raw = raw or {}
    warnings = list(warnings or [])
    sym = symbol.upper()

    prof = _first(raw.get("profile"))
    quote = _first(raw.get("quote"))
    sfloat = _first(raw.get("shares_float"))
    kmet = _first(raw.get("key_metrics"))
    news = raw.get("news") if isinstance(raw.get("news"), list) else []

    price = _num(_pick(prof, "price") or _pick(quote, "price"))
    market_cap = _num(_pick(prof, "marketCap", "mktCap") or _pick(quote, "marketCap"))
    beta = _num(_pick(prof, "beta"))
    volume = _num(_pick(quote, "volume") or _pick(prof, "volume"))
    avg_volume = _num(_pick(prof, "averageVolume", "avgVolume")
                      or _pick(quote, "avgVolume", "averageVolume"))
    rel_volume = round(volume / avg_volume, 2) if (volume and avg_volume) else None

    year_low = year_high = None
    rng = _pick(prof, "range")
    if isinstance(rng, str) and "-" in rng:
        lo, hi = rng.split("-", 1)
        year_low, year_high = _num(lo), _num(hi)
    if year_low is None:
        year_low = _num(_pick(quote, "yearLow"))
    if year_high is None:
        year_high = _num(_pick(quote, "yearHigh"))

    float_shares = _num(_pick(sfloat, "floatShares", "float"))
    outstanding = _num(_pick(sfloat, "outstandingShares", "sharesOutstanding"))
    free_float_pct = None
    ff = _num(_pick(sfloat, "freeFloat", "freeFloatPercent"))
    if ff is not None:
        free_float_pct = round(ff * 100, 2) if ff <= 1 else round(ff, 2)
    elif float_shares and outstanding:
        free_float_pct = round(float_shares / outstanding * 100, 2)

    net_debt_to_ebitda = _num(_pick(kmet, "netDebtToEBITDATTM", "netDebtToEBITDA",
                                    "netDebtToEbitdaTTM"))

    recent_news = [str(n.get("title")) for n in news if isinstance(n, dict) and n.get("title")][:5]
    squeeze = classify_squeeze(float_shares)

    # Preserve the project-wide employer block: NVDA may be read-only context, never a trade vehicle.
    from data.social_sentiment import is_restricted_underlying
    restricted = is_restricted_underlying(sym)
    if restricted:
        warnings.insert(0, f"RESTRICTED_EMPLOYER: {sym} is employer-restricted — context only, never trade.")

    warnings.append(OPTIONS_UNAVAILABLE_WARNING)

    return {
        "symbol": sym,
        "as_of": now.isoformat(timespec="seconds"),
        "price": price, "market_cap": market_cap, "beta": beta,
        "year_low": year_low, "year_high": year_high,
        "volume": volume, "average_volume": avg_volume, "relative_volume": rel_volume,
        "float_shares": float_shares, "outstanding_shares": outstanding,
        "free_float_pct": free_float_pct, "net_debt_to_ebitda": net_debt_to_ebitda,
        "news_count": len(news), "recent_news": recent_news,
        "squeeze_profile": squeeze, "trade_implication": _TRADE_IMPLICATION[squeeze],
        "fmp_options_available": False,
        "restricted": restricted,
        "disclaimer": DISCLAIMER,
        "warnings": warnings,
    }


# --- render ----------------------------------------------------------------------------------

def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float) and x.is_integer():
        return f"{int(x):,}"
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x)


def render_markdown(ctx: dict) -> str:
    c = ctx or {}
    lines = [
        f"# FMP Context — {c.get('symbol')}  (meme/squeeze sanity)",
        f"_as-of {c.get('as_of')} · NOT an entry signal · no orders · no options/gamma_",
        f"> ⚠️ {OPTIONS_UNAVAILABLE_WARNING}",
    ]
    if c.get("restricted"):
        lines.append(f"> ⛔ {c['symbol']} is employer-restricted — context only, never trade.")
    lines += [
        "",
        "## Snapshot",
        f"- Price: **{_fmt(c.get('price'))}** · Market cap: **{_fmt(c.get('market_cap'))}** "
        f"· Beta: **{_fmt(c.get('beta'))}**",
        f"- 52w range: {_fmt(c.get('year_low'))} – {_fmt(c.get('year_high'))} "
        f"· Rel vol: **{_fmt(c.get('relative_volume'))}** "
        f"(vol {_fmt(c.get('volume'))} / avg {_fmt(c.get('average_volume'))})",
        f"- Float: **{_fmt(c.get('float_shares'))}** sh · Outstanding: {_fmt(c.get('outstanding_shares'))} "
        f"· Free float: {_fmt(c.get('free_float_pct'))}%",
        f"- Net debt/EBITDA: {_fmt(c.get('net_debt_to_ebitda'))}",
        "",
        "## Squeeze read",
        f"- Profile: **{c.get('squeeze_profile')}**",
        f"- Implication: {c.get('trade_implication')}",
    ]
    news = c.get("recent_news") or []
    lines += ["", f"## Recent news ({c.get('news_count', 0)})"]
    lines += [f"- {t}" for t in news] if news else ["- _none_"]
    warns = c.get("warnings") or []
    if warns:
        lines += ["", "## Warnings"] + [f"- {w}" for w in warns]
    return "\n".join(lines) + "\n"


def run_fmp_context(symbol: str, allow_fetch: bool = True, news_limit: int = DEFAULT_NEWS_LIMIT,
                    out_dir: str | None = None, write: bool = False, now=None,
                    fetch_json=None) -> dict:
    """Fetch (unless allow_fetch=False) + classify + optionally write artifacts. Fail-closed; the
    returned dict never contains secrets. ``fetch_json`` is injectable for tests."""
    if allow_fetch:
        raw, warnings = fetch_fmp_raw(symbol, news_limit=news_limit, fetch_json=fetch_json)
    else:
        raw, warnings = {}, ["allow_fetch=False — no FMP fetch (offline)"]
    ctx = build_context(symbol, raw, warnings=warnings, now=now)

    artifacts: dict[str, str] = {}
    if write or out_dir:
        odir = Path(os.path.expanduser(out_dir or DEFAULT_REPORT_DIR))
        odir.mkdir(parents=True, exist_ok=True)
        slug = ctx["symbol"].lower()
        md_path, js_path = odir / f"odte_fmp_context_{slug}.md", odir / f"odte_fmp_context_{slug}.json"
        md_path.write_text(render_markdown(ctx))
        js_path.write_text(json.dumps(ctx, indent=2, default=str))
        artifacts = {"markdown": str(md_path), "json": str(js_path)}
    ctx["artifacts"] = artifacts
    return ctx
