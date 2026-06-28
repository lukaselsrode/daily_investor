"""0DTE day-regime scorecard — local/offline, no broker/network/LLM.

Companion to `odte_vehicle_score`: that one scores a *specific contract*; this one scores the
*whole trading day* before you pick a vehicle. It answers the controller's first question — "is
today a day to press directional 0DTE bets, scalp a range, or stay flat?" — as a deterministic
GOOD_DAY / CHOP / AVOID classification with reasons.

It places NO orders and fetches NO data. The caller supplies a market/regime snapshot (the same
fields Hermes/Robinhood already collect): VIX/VIXY, opening gap, per-index ORB state + VWAP side,
expected-move %, and minutes-to-close. An optional gamma-map payload lets the expected move be
derived from the ATM-straddle band when `expected_move_pct` isn't supplied directly.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_REPORT_DIR

GOOD_DAY = "GOOD_DAY"
CHOP = "CHOP"
AVOID = "AVOID"

_INDICES = ("spy", "qqq", "iwm")


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
        return out if out == out else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "yes", "1", "above", "above_vwap"}:
            return True
        if key in {"false", "no", "0", "below", "below_vwap"}:
            return False
    return None


def _load_json(path: str | None, raw_json: str | None, default: dict | None = None) -> dict:
    if raw_json:
        obj = json.loads(raw_json)
    elif path:
        obj = json.loads(Path(os.path.expanduser(path)).read_text())
    else:
        return dict(default or {})
    if not isinstance(obj, dict):
        raise ValueError("payload must be a JSON object")
    return obj


def _trend_score(market: dict) -> tuple[int, list[str]]:
    """Score directional conviction from per-index ORB state + VWAP side.

    Each index votes up/down/inside from `{sym}_above_vwap` (bool/string) and `{sym}_orb_state`
    (above|below|inside). Aligned indices = trend day (good for directional 0DTE); a split book or
    most indices stuck inside the opening range = chop.
    """
    ups = dns = inside = 0
    for sym in _INDICES:
        vwap = _bool(market.get(f"{sym}_above_vwap"))
        orb = str(market.get(f"{sym}_orb_state") or "").strip().lower()
        signals: list[int] = []
        if vwap is True:
            signals.append(1)
        elif vwap is False:
            signals.append(-1)
        if orb == "above":
            signals.append(1)
        elif orb == "below":
            signals.append(-1)
        elif orb == "inside":
            signals.append(0)
        if not signals:
            continue
        total = sum(signals)
        if total > 0:
            ups += 1
        elif total < 0:
            dns += 1
        else:
            inside += 1
    aligned, conflict = max(ups, dns), min(ups, dns)
    score = 0
    reasons: list[str] = []
    if aligned == 0 and inside == 0:
        return 0, ["no index ORB/VWAP fields in snapshot"]
    if aligned >= 2 and conflict == 0:
        score += 3
        reasons.append(f"{aligned} indices trend-aligned on ORB/VWAP — clean directional tape")
    elif aligned >= 2 and conflict >= 1:
        score += 1
        reasons.append("majority of indices trend together but one conflicts")
    elif aligned >= 1 and conflict >= 1:
        score -= 2
        reasons.append("indices split above/below VWAP — chop risk")
    if inside >= 2:
        score -= 2
        reasons.append(f"{inside} indices still inside the opening range — no breakout yet")
    return score, reasons


def _vol_score(market: dict) -> tuple[int, list[str], bool]:
    """Score the volatility regime. Returns (score, reasons, hard_avoid)."""
    score = 0
    reasons: list[str] = []
    vix = _num(market.get("vix"))
    vix_chg = _num(market.get("vix_change_pct"))
    vixy_chg = _num(market.get("vixy_change_pct") if "vixy_change_pct" in market
                    else (market.get("vixy") or {}).get("day_change_pct")
                    if isinstance(market.get("vixy"), dict) else None)
    hard_avoid = False
    if vix is not None:
        if vix > 32:
            score -= 2
            hard_avoid = True
            reasons.append(f"VIX {vix:g} very elevated — headline/whipsaw risk, stay flat")
        elif vix >= 28:
            score -= 1
            reasons.append(f"VIX {vix:g} elevated — wide, unstable ranges")
        elif vix < 12:
            score -= 1
            reasons.append(f"VIX {vix:g} compressed — small ranges favor chop over trend")
        else:
            score += 1
            reasons.append(f"VIX {vix:g} in a tradable band")
    spike = max([c for c in (vix_chg, vixy_chg) if c is not None], default=None)
    if spike is not None and spike >= 12:
        score -= 2
        hard_avoid = True
        reasons.append(f"volatility spiking intraday (+{spike:g}%) — unstable, stand aside")
    elif spike is not None and spike >= 6:
        score -= 1
        reasons.append(f"volatility rising intraday (+{spike:g}%)")
    return score, reasons, hard_avoid


def _gap_score(market: dict) -> tuple[int, list[str]]:
    gap = _num(market.get("gap_pct"))
    if gap is None:
        return 0, []
    ag = abs(gap)
    if 0.3 <= ag <= 1.5:
        return 1, [f"clean {gap:+g}% gap supports a directional open"]
    if ag > 2.5:
        return -1, [f"large {gap:+g}% gap — reversion/whipsaw risk into the open"]
    return 0, [f"muted {gap:+g}% gap — little directional push"]


def _expected_move_score(market: dict, gamma: dict) -> tuple[int, list[str]]:
    """Score the day's expected move. Prefer an explicit `expected_move_pct`; else derive it from a
    gamma-map ATM-straddle band (half-width / spot)."""
    em_pct = _num(market.get("expected_move_pct"))
    derived = False
    if em_pct is None and gamma:
        raw_em = gamma.get("expected_move")
        em = raw_em if isinstance(raw_em, dict) else {}
        lower, upper = _num(em.get("lower")), _num(em.get("upper"))
        spot = _num(gamma.get("spot")) or _num(market.get("spot"))
        if lower is not None and upper is not None and spot and spot > 0:
            em_pct = (upper - lower) / 2.0 / spot * 100.0
            derived = True
    if em_pct is None:
        return 0, []
    src = " (from gamma band)" if derived else ""
    if em_pct < 0.4:
        return -2, [f"expected move {em_pct:.2f}%{src} is tight — theta chop, little room to run"]
    if em_pct <= 1.5:
        return 1, [f"expected move {em_pct:.2f}%{src} gives a tradable range"]
    return 0, [f"expected move {em_pct:.2f}%{src} is wide — big range but more risk"]


def _time_score(market: dict) -> tuple[int, list[str], bool]:
    """Late-day theta gate. Returns (score, reasons, hard_avoid)."""
    mtc = _num(market.get("minutes_to_close"))
    if mtc is None:
        return 0, [], False
    if mtc <= 30:
        return -4, [f"only {mtc:g} minutes to close — too late to open a new day-trade"], True
    if mtc <= 60:
        return -2, [f"late session: {mtc:g} minutes to close — reduce/scalp only"], False
    return 0, [], False


def score_day(market: dict | None = None, gamma: dict | None = None) -> dict:
    """Return a deterministic GOOD_DAY/CHOP/AVOID score for the trading day from a market snapshot.

    Inputs (all optional): vix, vix_change_pct, vixy_change_pct/vixy{day_change_pct}, gap_pct,
    {spy,qqq,iwm}_above_vwap, {spy,qqq,iwm}_orb_state, expected_move_pct (or a gamma payload with an
    expected_move band + spot), minutes_to_close. No network/broker/LLM; never places orders.
    """
    market = market or {}
    gamma = gamma or {}
    if not market and not gamma:
        return {
            "schema_version": 1,
            "verdict": CHOP,
            "score": 0,
            "components": {},
            "reasons": ["no market/regime snapshot supplied — defaulting to CHOP (no edge)"],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "basis": "non_sentiment_day_regime_scorecard: trend + volatility + gap + expected-move + time",
            "places_orders": False,
        }
    trend, trend_r = _trend_score(market)
    vol, vol_r, vol_avoid = _vol_score(market)
    gap, gap_r = _gap_score(market)
    em, em_r = _expected_move_score(market, gamma)
    tim, tim_r, time_avoid = _time_score(market)

    components = {"trend": trend, "volatility": vol, "gap": gap,
                  "expected_move": em, "time": tim}
    total = trend + vol + gap + em + tim
    reasons = ([f"trend: {r}" for r in trend_r] + [f"volatility: {r}" for r in vol_r] +
               [f"gap: {r}" for r in gap_r] + [f"expected_move: {r}" for r in em_r] +
               [f"time: {r}" for r in tim_r])

    hard_avoid = vol_avoid or time_avoid
    if hard_avoid:
        verdict = AVOID
    elif total >= 4:
        verdict = GOOD_DAY
    elif total <= -3:
        verdict = AVOID
    else:
        verdict = CHOP
    return {
        "schema_version": 1,
        "verdict": verdict,
        "score": total,
        "components": components,
        "reasons": reasons,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "basis": "non_sentiment_day_regime_scorecard: trend + volatility + gap + expected-move + time",
        "places_orders": False,
    }


def run_day_score(market_json: str | None = None, market_path: str | None = None,
                  gamma_json: str | None = None, gamma_path: str | None = None,
                  out_dir: str | None = None, write: bool = False) -> dict:
    market = _load_json(market_path, market_json, default={})
    gamma = _load_json(gamma_path, gamma_json, default={})
    payload = score_day(market=market, gamma=gamma)
    if write:
        out = Path(os.path.expanduser(out_dir or ODTE_REPORT_DIR))
        out.mkdir(parents=True, exist_ok=True)
        path = out / "odte_day_score.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        payload["artifact"] = str(path)
    return payload


def render_markdown(payload: dict) -> str:
    comps = payload.get("components") or {}
    comp_line = " · ".join(f"{k} {v:+d}" for k, v in comps.items()) if comps else "—"
    lines = ["# 0DTE day score", "",
             f"Verdict: **{payload.get('verdict')}**  ",
             f"Score: `{payload.get('score')}`  ",
             f"Components: {comp_line}  ",
             f"Basis: {payload.get('basis')}", "", "## Reasons"]
    for r in payload.get("reasons") or []:
        lines.append(f"- {r}")
    return "\n".join(lines)
