"""0DTE option-chain gamma / pin map — pure/offline, NO broker, NO LLM, NO network.

Computes ABSOLUTE option-chain gamma + open-interest concentration from option-quote rows that the
caller (Hermes/Robinhood MCP) exports to a JSON file or string. It is an HONEST concentration map,
not a dealer-positioning model:

    *** This is NOT dealer net GEX, gamma flip, or sign. ***

Robinhood option quotes expose per-contract gamma/delta/IV/OI/volume/marks but NOT dealer
positioning, so this module CANNOT and does NOT infer dealer net gamma or a gamma-flip level. Every
output is labeled ``gamma_regime: pin_risk_only_not_dealer_gex`` and carries the disclaimer below.
Walls / max-gamma-strike / pin-risk are concentration heuristics over supplied quotes only.

----------------------------------------------------------------------------------------------------
input (JSON file or string): either a LIST of row objects, or a wrapper dict
  { "underlying": "SPY", "expiration": "2026-06-24", "spot": 734.8, "rows": [ ...row... ] }
  (the row list may also appear under "results"/"option_quotes"/"quotes"/"data")

each row may be FLAT, NESTED, or QUOTE-ONLY (real Robinhood get_option_quotes shapes):
  flat        { "type":"call", "strike_price":"735", "gamma":"0.047", "open_interest":680, ... }
  nested      { "instrument": {"chain_symbol":"SPY","expiration_date":"2026-06-24",
                               "strike_price":"735.0000","type":"call"},
                "quote": {"mark_price":"2.74","gamma":"0.047","open_interest":680,
                          "volume":69193,"updated_at":"..."} }
              -> instrument/contract/quote sub-objects are merged (quote wins on overlap; explicit
                 top-level scalars win over all).
  quote-only  { "quote": {"instrument_id":"<id>","mark_price":"2.74","gamma":"0.047", ...} }
              joined against a wrapper-level instruments map for strike/type/expiration:
  wrapper with instruments:
    { "spot":734.8, "rows":[{"quote":{...}}, ...],
      "instruments": { "<instrument_id>": {"strike_price":"735","type":"call", ...} } }
    ("instruments" may also be a LIST of instrument dicts, each keyed by its id/url)

row fields (aliases accepted; missing fields tolerated):
  underlying        underlying | chain_symbol | symbol
  expiration_date   expiration_date | expiration
  type              type | option_type            -> normalized to "call"/"put"
  strike            strike_price | strike
  bid / ask         bid_price | bid  /  ask_price | ask
  mark              mark_price | adjusted_mark_price | mark   (falls back to (bid+ask)/2)
  implied_volatility, delta, gamma
  open_interest     open_interest | oi
  volume
  updated_at        updated_at | timestamp | updated_at_utc    (ISO; drives freshness)
  spot              spot | underlying_last | underlying_price   (row-level, optional)
----------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = os.path.expanduser("~/0dte/reports")

# The honest regime label — surfaced on every output so no consumer mistakes this for dealer GEX.
GAMMA_REGIME_LABEL = "pin_risk_only_not_dealer_gex"
DISCLAIMER = ("ABSOLUTE option-chain gamma/OI concentration from SUPPLIED quote rows — NOT dealer "
              "net GEX, gamma flip, or sign. Robinhood/this tool do NOT expose dealer positioning. "
              "Walls, max-gamma strike, and pin-risk are concentration heuristics only.")

# Pin-zone + concentration defaults (all overridable).
DEFAULT_NEAR_PCT = 0.0025          # spot within 0.25% of the max-gamma strike => "in the pin zone"
DEFAULT_NEAR_DOLLARS = 1.0         # ...or within $1.00, whichever is larger
DEFAULT_CONCENTRATION_RATIO = 1.5  # peak strike >= 1.5x the median strike => "materially concentrated"
DEFAULT_MAX_AGE_MINUTES = 15.0     # newest quote older than this => quote_fresh False

_ALIASES = {
    "underlying": ("underlying", "chain_symbol", "symbol"),
    "expiration": ("expiration_date", "expiration"),
    "type": ("type", "option_type"),
    "strike": ("strike_price", "strike"),
    "bid": ("bid_price", "bid"),
    "ask": ("ask_price", "ask"),
    "mark": ("mark_price", "adjusted_mark_price", "mark"),
    "iv": ("implied_volatility", "iv"),
    "delta": ("delta",),
    "gamma": ("gamma",),
    "open_interest": ("open_interest", "oi"),
    "volume": ("volume",),
    "updated_at": ("updated_at", "timestamp", "updated_at_utc"),
    "spot": ("spot", "underlying_last", "underlying_price"),
}


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _alias(raw: dict, key: str):
    for k in _ALIASES[key]:
        if k in raw and raw[k] not in (None, ""):
            return raw[k]
    return None


def _norm_side(v) -> str:
    s = str(v or "").strip().lower()
    if s in ("call", "c", "calls"):
        return "call"
    if s in ("put", "p", "puts"):
        return "put"
    return ""


def _parse_ts(s) -> datetime | None:
    if not s:
        return None
    txt = str(s).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _median(vals) -> float | None:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    n, m = len(v), len(v) // 2
    return v[m] if n % 2 else (v[m - 1] + v[m]) / 2.0


# Sub-objects that real RH payloads nest (get_option_quotes rows carry a "quote"; merged feeds add
# "instrument"/"contract"). We flatten these into one row; the "quote" block wins on key overlap
# (e.g. updated_at) since it's the live quote, and explicit top-level scalars win over everything.
_NESTED_KEYS = ("instrument", "contract", "option", "quote", "greeks", "close", "fundamentals")


def _instrument_id(d: dict) -> str | None:
    """Extract an instrument id from id/url-style fields (last URL path segment, or the id itself)."""
    for k in ("instrument_id", "instrument", "id", "instrument_url", "url"):
        v = d.get(k)
        if v:
            return str(v).rstrip("/").rsplit("/", 1)[-1]
    return None


def _index_instruments(instruments) -> dict:
    """Normalize an instruments map (dict keyed by id) or list of instrument dicts -> {id: dict}."""
    if isinstance(instruments, dict):
        return instruments
    out: dict = {}
    if isinstance(instruments, list):
        for inst in instruments:
            if isinstance(inst, dict):
                iid = _instrument_id(inst)
                if iid:
                    out[iid] = inst
    return out


def _flatten_row(raw: dict, instruments: dict | None = None) -> dict:
    """Merge nested {instrument, quote, contract, ...} sub-objects into one flat dict (quote wins on
    overlap; explicit top-level scalars win over all). If strike/type are still missing and the row
    references an instrument_id present in `instruments`, join those instrument fields in."""
    merged: dict = {}
    for k in _NESTED_KEYS:
        sub = raw.get(k)
        if isinstance(sub, dict):
            merged.update(sub)
    for k, v in raw.items():           # explicit top-level scalars override nested
        if not (k in _NESTED_KEYS and isinstance(v, dict)):
            merged[k] = v
    if instruments and (_num(_alias(merged, "strike")) is None or not _norm_side(_alias(merged, "type"))):
        iid = _instrument_id(merged)
        inst = instruments.get(iid) if iid else None
        if isinstance(inst, dict):
            for k, v in inst.items():
                merged.setdefault(k, v)   # fill gaps only — quote/top-level keep priority
    return merged


def normalize_row(raw: dict, instruments: dict | None = None) -> dict | None:
    """Map one raw quote row (flat, nested {instrument,quote}, or quote-only joined via `instruments`)
    to a normalized row, or None if it has no usable strike/side."""
    if not isinstance(raw, dict):
        return None
    raw = _flatten_row(raw, instruments)
    side = _norm_side(_alias(raw, "type"))
    strike = _num(_alias(raw, "strike"))
    if not side or strike is None:
        return None
    bid, ask = _num(_alias(raw, "bid")), _num(_alias(raw, "ask"))
    mark = _num(_alias(raw, "mark"))
    if mark is None and bid is not None and ask is not None:
        mark = (bid + ask) / 2.0
    oi = _num(_alias(raw, "open_interest"))
    vol = _num(_alias(raw, "volume"))
    und = _alias(raw, "underlying")
    return {
        "underlying": str(und).upper() if und else None,
        "expiration": _alias(raw, "expiration"),
        "side": side, "strike": strike,
        "bid": bid, "ask": ask, "mark": mark,
        "iv": _num(_alias(raw, "iv")), "delta": _num(_alias(raw, "delta")),
        "gamma": _num(_alias(raw, "gamma")),
        "open_interest": int(oi) if oi is not None else 0,
        "volume": int(vol) if vol is not None else 0,
        "updated_at": _parse_ts(_alias(raw, "updated_at")),
        "spot": _num(_alias(raw, "spot")),
    }


def load_rows(obj) -> tuple[list, dict]:
    """Accept a list of rows OR a wrapper dict; return (raw_rows, meta{spot,underlying,expiration})."""
    meta: dict = {}
    if isinstance(obj, list):
        return obj, meta
    if isinstance(obj, dict):
        for k in ("spot", "underlying_price", "underlying_last"):
            if obj.get(k) is not None:
                meta["spot"] = obj[k]
                break
        for k in ("underlying", "chain_symbol", "symbol"):
            if obj.get(k):
                meta["underlying"] = str(obj[k]).upper()
                break
        for k in ("expiration", "expiration_date"):
            if obj.get(k):
                meta["expiration"] = obj[k]
                break
        # Optional instrument-id -> contract-fields map (dict keyed by id, or list of instrument
        # dicts) used to join strike/type/expiration onto quote-only rows.
        if isinstance(obj.get("instruments"), (dict, list)):
            meta["instruments"] = obj["instruments"]
        for k in ("rows", "results", "option_quotes", "quotes", "data"):
            if isinstance(obj.get(k), list):
                return obj[k], meta
    return [], meta


def _empty_strike() -> dict:
    return {"call_oi": 0, "put_oi": 0, "call_volume": 0, "put_volume": 0,
            "call_gamma_notional_1pct": 0.0, "put_gamma_notional_1pct": 0.0}


def _top_strike(strikes: dict, gkey: str, oikey: str):
    """Strike maximizing gamma-notional concentration; falls back to OI when gamma is all-zero."""
    if not strikes:
        return None
    if any(v[gkey] for v in strikes.values()):
        return max(strikes.items(), key=lambda kv: kv[1][gkey])[0]
    if any(v[oikey] for v in strikes.values()):
        return max(strikes.items(), key=lambda kv: kv[1][oikey])[0]
    return None


def build_gamma_map(rows, spot=None, underlying=None, expiration=None, now=None, instruments=None,
                    near_pct: float = DEFAULT_NEAR_PCT, near_dollars: float = DEFAULT_NEAR_DOLLARS,
                    concentration_ratio: float = DEFAULT_CONCENTRATION_RATIO,
                    max_age_minutes: float = DEFAULT_MAX_AGE_MINUTES) -> dict:
    """PURE: build the gamma/pin concentration map from quote rows. No IO/network.

    `instruments` is an optional id->contract-fields map (or list of instrument dicts) used to join
    strike/type/expiration onto quote-only rows that carry just an instrument_id."""
    now = now or datetime.now(timezone.utc)
    inst_map = _index_instruments(instruments)
    norm = [r for r in (normalize_row(x, inst_map) for x in (rows or [])) if r]
    if underlying:
        u = underlying.upper()
        norm = [r for r in norm if r.get("underlying") in (None, u)]
    if expiration:
        norm = [r for r in norm if r.get("expiration") in (None, expiration)]

    underlying = underlying or next((r["underlying"] for r in norm if r.get("underlying")), None)
    expiration = expiration or next((r["expiration"] for r in norm if r.get("expiration")), None)

    spot = _num(spot)
    if spot is None:
        spot = _median([r["spot"] for r in norm if r.get("spot") is not None])

    # Per-row dollar gamma per 1% move: gamma * OI * 100 * spot^2 * 0.01 (absolute concentration).
    for r in norm:
        g = r.get("gamma")
        if g is not None and spot is not None:
            r["gamma_notional_1pct"] = g * (r["open_interest"] or 0) * 100 * spot * spot * 0.01
        else:
            r["gamma_notional_1pct"] = None
    gamma_available = any(r.get("gamma") is not None for r in norm) and spot is not None

    strikes: dict[float, dict] = {}
    for r in norm:
        d = strikes.setdefault(r["strike"], _empty_strike())
        gn = r["gamma_notional_1pct"] or 0.0
        if r["side"] == "call":
            d["call_oi"] += r["open_interest"]
            d["call_volume"] += r["volume"]
            d["call_gamma_notional_1pct"] += gn
        else:
            d["put_oi"] += r["open_interest"]
            d["put_volume"] += r["volume"]
            d["put_gamma_notional_1pct"] += gn
    for d in strikes.values():
        d["total_gamma_notional_1pct"] = d["call_gamma_notional_1pct"] + d["put_gamma_notional_1pct"]
        d["total_oi"] = d["call_oi"] + d["put_oi"]

    call_wall = _top_strike(strikes, "call_gamma_notional_1pct", "call_oi")
    put_wall = _top_strike(strikes, "put_gamma_notional_1pct", "put_oi")
    max_gamma_strike = _top_strike(strikes, "total_gamma_notional_1pct", "total_oi")

    expected_move = _expected_move(norm, spot)
    freshness = _freshness(norm, now, max_age_minutes)
    pin_risk = _pin_risk(strikes, spot, max_gamma_strike, freshness["quote_fresh"],
                         near_pct, near_dollars, concentration_ratio)

    return {
        "underlying": underlying, "expiration": expiration, "spot": spot,
        "generated_at": now.isoformat(timespec="seconds"),
        "gamma_regime": GAMMA_REGIME_LABEL,
        "disclaimer": DISCLAIMER,
        "gamma_available": gamma_available,
        "concentration_basis": "gamma_notional_1pct" if gamma_available else "open_interest",
        "n_rows": len(norm), "n_strikes": len(strikes),
        "call_wall": call_wall, "put_wall": put_wall, "max_gamma_strike": max_gamma_strike,
        "expected_move": expected_move, "pin_risk": pin_risk, "freshness": freshness,
        "by_strike": [{"strike": k, **v} for k, v in sorted(strikes.items())],
    }


def _expected_move(norm: list, spot: float | None) -> dict:
    if spot is None:
        return {"available": False, "reason": "no spot supplied/derivable"}
    call_marks, put_marks = {}, {}
    for r in norm:
        if r.get("mark") is None:
            continue
        (call_marks if r["side"] == "call" else put_marks)[r["strike"]] = r["mark"]
    both = sorted(set(call_marks) & set(put_marks), key=lambda k: (abs(k - spot), k))
    if not both:
        return {"available": False, "reason": "no strike with both call & put marks"}
    atm = both[0]
    straddle = call_marks[atm] + put_marks[atm]
    return {
        "available": True, "atm_strike": atm, "straddle_mark": round(straddle, 4),
        "lower": round(spot - straddle, 4), "upper": round(spot + straddle, 4),
        # spot vs the ATM strike (the natural mid of the chain) — which side of the pin we sit on.
        "spot_location": "above_atm" if spot > atm else "below_atm" if spot < atm else "at_atm",
    }


def _freshness(norm: list, now: datetime, max_age_minutes: float) -> dict:
    ts = [r["updated_at"] for r in norm if r.get("updated_at") is not None]
    if not ts:
        return {"quote_fresh": False, "newest_quote_utc": None, "age_minutes": None,
                "max_age_minutes": max_age_minutes, "reason": "no quote timestamps in input"}
    newest = max(ts)
    age = (now - newest).total_seconds() / 60.0
    fresh = age <= max_age_minutes
    return {"quote_fresh": fresh, "newest_quote_utc": newest.isoformat(timespec="seconds"),
            "age_minutes": round(age, 1), "max_age_minutes": max_age_minutes,
            "reason": None if fresh else f"newest quote {age:.0f}m old > {max_age_minutes:.0f}m"}


def _pin_risk(strikes: dict, spot, max_gamma_strike, quote_fresh: bool,
              near_pct: float, near_dollars: float, concentration_ratio: float) -> dict:
    basis = ("total_gamma_notional_1pct"
             if any(v["total_gamma_notional_1pct"] for v in strikes.values()) else "total_oi")
    median = _median([v[basis] for v in strikes.values()])
    peak = strikes[max_gamma_strike][basis] if max_gamma_strike in strikes else None
    ratio = round(peak / median, 2) if (peak and median) else None
    material = bool(ratio and ratio >= concentration_ratio)
    within = None
    distance = None
    if spot is not None and max_gamma_strike is not None:
        distance = round(abs(spot - max_gamma_strike), 4)
        within = distance <= max(near_dollars, spot * near_pct)

    if not quote_fresh:
        level, reason = "stale", "quotes stale — pin read unreliable, refresh before acting"
    elif spot is None or max_gamma_strike is None:
        level, reason = "unknown", "need spot and a concentration strike"
    elif within and material:
        level, reason = "high", "spot in the pin zone AND concentration materially above median"
    elif within or material:
        level, reason = "medium", "either in the pin zone OR materially concentrated, not both"
    else:
        level, reason = "low", "spot away from the concentration peak"
    return {"level": level, "reason": reason, "max_gamma_strike": max_gamma_strike,
            "distance": distance, "within_pin_zone": within,
            "concentration_basis": basis, "peak_vs_median_ratio": ratio}


# --- render ----------------------------------------------------------------------------------

def render_markdown(gmap: dict, top_n: int = 8) -> str:
    g = gmap or {}
    def f(x, nd=2):
        return "n/a" if x is None else (f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x))
    lines = [
        f"# 0DTE Gamma / Pin Map — {g.get('underlying') or '?'} {g.get('expiration') or ''}".rstrip(),
        f"_Generated {g.get('generated_at')} · regime: **{g.get('gamma_regime')}**_",
        f"> ⚠️ {g.get('disclaimer')}",
        "",
        "## Key levels",
        f"- Spot: **{f(g.get('spot'))}** · basis: **{g.get('concentration_basis')}** "
        f"(gamma_available={g.get('gamma_available')})",
        f"- Call wall: **{f(g.get('call_wall'))}** · Put wall: **{f(g.get('put_wall'))}** "
        f"· Max-gamma strike: **{f(g.get('max_gamma_strike'))}**",
    ]
    em = g.get("expected_move") or {}
    if em.get("available"):
        lines.append(f"- Expected move (ATM {f(em.get('atm_strike'))} straddle {f(em.get('straddle_mark'))}): "
                     f"**{f(em.get('lower'))} – {f(em.get('upper'))}** · spot {em.get('spot_location')}")
    else:
        lines.append(f"- Expected move: n/a ({em.get('reason')})")
    pr = g.get("pin_risk") or {}
    lines.append(f"- Pin risk: **{str(pr.get('level')).upper()}** — {pr.get('reason')} "
                 f"(peak/median {f(pr.get('peak_vs_median_ratio'))}, dist {f(pr.get('distance'))})")
    fr = g.get("freshness") or {}
    flag = "✅ fresh" if fr.get("quote_fresh") else f"⚠️ STALE ({fr.get('reason')})"
    lines.append(f"- Quotes: {flag} · newest {fr.get('newest_quote_utc') or 'n/a'} "
                 f"· age {f(fr.get('age_minutes'), 0)}m")

    rows = sorted(g.get("by_strike", []), key=lambda r: r.get("total_gamma_notional_1pct", 0)
                  or r.get("total_oi", 0), reverse=True)[:top_n]
    lines += ["", f"## Top {len(rows)} strikes by concentration", "",
              "| Strike | Call OI | Put OI | Call γ$1% | Put γ$1% | Total γ$1% |",
              "|-------:|--------:|-------:|----------:|---------:|-----------:|"]
    for r in rows:
        lines.append(f"| {f(r['strike'])} | {r['call_oi']} | {r['put_oi']} "
                     f"| {f(r['call_gamma_notional_1pct'], 0)} | {f(r['put_gamma_notional_1pct'], 0)} "
                     f"| {f(r['total_gamma_notional_1pct'], 0)} |")
    return "\n".join(lines) + "\n"


def run_gamma_map(input_path: str | None = None, input_json: str | None = None,
                  spot=None, underlying=None, expiration=None,
                  out_dir: str | None = None, write: bool = False, now=None) -> dict:
    """Read rows from a JSON file/string, build the map, optionally write Markdown+JSON artifacts.

    PURE/offline — no broker, no network. Raises ValueError on unusable input."""
    if input_json is not None:
        obj = json.loads(input_json)
    elif input_path is not None:
        obj = json.loads(Path(os.path.expanduser(input_path)).read_text())
    else:
        raise ValueError("provide input_path or input_json")
    raw_rows, meta = load_rows(obj)
    gmap = build_gamma_map(
        raw_rows, spot=spot if spot is not None else meta.get("spot"),
        underlying=underlying or meta.get("underlying"),
        expiration=expiration or meta.get("expiration"),
        instruments=meta.get("instruments"), now=now)

    artifacts: dict[str, str] = {}
    if write or out_dir:
        odir = Path(os.path.expanduser(out_dir or DEFAULT_REPORT_DIR))
        odir.mkdir(parents=True, exist_ok=True)
        slug = (gmap.get("underlying") or "chain").lower()
        md_path, js_path = odir / f"odte_gamma_map_{slug}.md", odir / f"odte_gamma_map_{slug}.json"
        md_path.write_text(render_markdown(gmap))
        js_path.write_text(json.dumps(gmap, indent=2, default=str))
        artifacts = {"markdown": str(md_path), "json": str(js_path)}
    gmap["artifacts"] = artifacts
    return gmap
