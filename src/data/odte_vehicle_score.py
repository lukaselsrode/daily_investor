"""0DTE vehicle/contract bet-quality scorecard — local/offline, no broker/network/LLM.

This module gives the live controller a simple non-sentiment answer to the user's question:
"is this contract/vehicle a good or bad bet for the day?"

It intentionally does **not** place orders and does not fetch data. Callers supply the live market
snapshot, candidate contract, and optional gamma-map payload that Hermes/Robinhood already collected.
The output is a deterministic GOOD_BET / WATCH / BAD_BET classification with reasons.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ODTE_REPORT_DIR

GOOD_BET = "GOOD_BET"
WATCH = "WATCH"
BAD_BET = "BAD_BET"


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


def _expected_move_penalty(contract: dict, gamma: dict) -> tuple[int, list[str]]:
    strike = _num(contract.get("strike") or contract.get("strike_price"))
    opt_type = str(contract.get("option_type") or contract.get("type") or "").lower()
    raw_em = gamma.get("expected_move")
    em: dict = raw_em if isinstance(raw_em, dict) else {}
    lower, upper = _num(em.get("lower")), _num(em.get("upper"))
    if strike is None or lower is None or upper is None or opt_type not in {"call", "put"}:
        return 0, []
    premium = (_num(contract.get("ask") or contract.get("ask_price")) or
               _num(contract.get("mark") or contract.get("mark_price")) or 0.0)
    if opt_type == "call":
        breakeven = strike + premium
        if breakeven > upper:
            return -5, [f"call breakeven {breakeven:g} is above expected-move upper {upper:g}"]
    else:
        breakeven = strike - premium
        if breakeven < lower:
            return -5, [f"put breakeven {breakeven:g} is below expected-move lower {lower:g}"]
    return 2, ["breakeven is inside expected-move band"]


def _gamma_score(contract: dict, gamma: dict) -> tuple[int, list[str]]:
    if not gamma:
        return 0, ["gamma/pin unavailable for this score"]
    reasons: list[str] = []
    score = 0
    pin = gamma.get("pin_risk")
    if isinstance(pin, dict):
        pin_level = str(pin.get("level") or "").lower()
    else:
        pin_level = str(pin or "").lower()
    if pin_level in {"high", "medium"}:
        score -= 2 if pin_level == "high" else 1
        reasons.append(f"{pin_level} pin risk")
    elif pin_level == "low":
        score += 1
        reasons.append("low pin risk")
    fresh = gamma.get("freshness") if isinstance(gamma.get("freshness"), dict) else {}
    if fresh and fresh.get("quote_fresh") is False:
        score -= 2
        reasons.append("gamma quotes are stale")
    strike = _num(contract.get("strike") or contract.get("strike_price"))
    opt_type = str(contract.get("option_type") or contract.get("type") or "").lower()
    wall = _num(gamma.get("call_wall") if opt_type == "call" else gamma.get("put_wall"))
    spot = _num(gamma.get("spot"))
    if strike is not None and wall is not None and spot is not None:
        # Calls under/at call wall and puts above/at put wall are more plausible; buying beyond the
        # wall without breakout confirmation is usually lower-quality long premium.
        if opt_type == "call" and strike > wall and spot <= wall:
            score -= 2
            reasons.append(f"call strike beyond call wall {wall:g} before spot clears it")
        elif opt_type == "put" and strike < wall and spot >= wall:
            score -= 2
            reasons.append(f"put strike beyond put wall {wall:g} before spot clears it")
        else:
            score += 1
            reasons.append("strike/wall relationship is plausible")
    em_score, em_reasons = _expected_move_penalty(contract, gamma)
    return score + em_score, reasons + em_reasons


def _market_score(direction: str, market: dict) -> tuple[int, list[str]]:
    """Score non-sentiment market/tape context.

    Expected optional fields:
      spy_above_vwap, qqq_above_vwap, iwm_above_vwap, vixy_above_vwap
      spy_orb_state/qqq_orb_state/iwm_orb_state: above|below|inside
      vix_change_pct or vixy_change_pct
    """
    if not market:
        return 0, ["market/tape snapshot unavailable"]
    bullish = direction in {"call", "bullish", "long_call"}
    score = 0
    reasons: list[str] = []
    above = []
    below = []
    for sym in ("spy", "qqq", "iwm"):
        v = _bool(market.get(f"{sym}_above_vwap") if f"{sym}_above_vwap" in market else market.get(sym, {}).get("above_vwap") if isinstance(market.get(sym), dict) else None)
        if v is True:
            above.append(sym.upper())
        elif v is False:
            below.append(sym.upper())
    if bullish:
        score += len(above) - len(below)
        if len(above) >= 2:
            reasons.append(f"VWAP confirms calls on {','.join(above)}")
        if len(below) >= 2:
            reasons.append(f"VWAP conflicts with calls on {','.join(below)}")
    else:
        score += len(below) - len(above)
        if len(below) >= 2:
            reasons.append(f"VWAP confirms puts on {','.join(below)}")
        if len(above) >= 2:
            reasons.append(f"VWAP conflicts with puts on {','.join(above)}")
    # VIXY confirmation: lower/below VWAP helps calls; higher/above helps puts.
    raw_vixy = market.get("vixy")
    vixy_obj: dict = raw_vixy if isinstance(raw_vixy, dict) else {}
    vixy_above = _bool(market.get("vixy_above_vwap") if "vixy_above_vwap" in market
                       else vixy_obj.get("above_vwap"))
    vixy_chg = _num(market.get("vixy_change_pct") if "vixy_change_pct" in market
                    else vixy_obj.get("day_change_pct"))
    if bullish:
        if vixy_above is False or (vixy_chg is not None and vixy_chg < 0):
            score += 1
            reasons.append("VIXY/risk vol confirms calls")
        elif vixy_above is True or (vixy_chg is not None and vixy_chg > 0):
            score -= 1
            reasons.append("VIXY/risk vol conflicts with calls")
    else:
        if vixy_above is True or (vixy_chg is not None and vixy_chg > 0):
            score += 1
            reasons.append("VIXY/risk vol confirms puts")
        elif vixy_above is False or (vixy_chg is not None and vixy_chg < 0):
            score -= 1
            reasons.append("VIXY/risk vol conflicts with puts")
    minutes_to_close = _num(market.get("minutes_to_close"))
    if minutes_to_close is not None:
        if minutes_to_close <= 20:
            score -= 4
            reasons.append(f"only {minutes_to_close:g} minutes to close/sellout")
        elif minutes_to_close <= 45:
            score -= 2
            reasons.append(f"late-day theta gate: {minutes_to_close:g} minutes to close")
    return score, reasons or ["market snapshot present but no strong tape fields"]


def _liquidity_score(contract: dict, buying_power: float | None) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    bid, ask, mark = _num(contract.get("bid") or contract.get("bid_price")), _num(contract.get("ask") or contract.get("ask_price")), _num(contract.get("mark") or contract.get("mark_price"))
    if bid and ask and ask > 0:
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid if mid > 0 else None
        if spread is not None and spread <= 0.12:
            score += 1
            reasons.append(f"tight spread {spread:.1%}")
        elif spread is not None and spread > 0.25:
            score -= 2
            reasons.append(f"wide spread {spread:.1%}")
    premium = ask or mark
    if buying_power and premium:
        debit = premium * 100.0
        if debit > buying_power:
            score -= 4
            reasons.append(f"estimated debit ${debit:.0f} exceeds buying power ${buying_power:.0f}")
        elif debit <= buying_power * 0.75:
            score += 1
            reasons.append("debit fits buying power with buffer")
        else:
            reasons.append("debit fits but uses most buying power")
    vol = _num(contract.get("volume"))
    oi = _num(contract.get("open_interest") or contract.get("openInterest"))
    if (vol or 0) + (oi or 0) > 1000:
        score += 1
        reasons.append("liquidity/OI is healthy")
    elif vol is not None or oi is not None:
        score -= 1
        reasons.append("liquidity/OI is thin")
    return score, reasons


def score_vehicle(contract: dict, direction: str | None = None, market: dict | None = None,
                  gamma: dict | None = None, buying_power: float | None = None) -> dict:
    """Return deterministic GOOD_BET/WATCH/BAD_BET score for a candidate vehicle/contract."""
    direction = (direction or contract.get("direction") or contract.get("option_type") or contract.get("type") or "").lower()
    if direction == "call":
        direction = "bullish"
    elif direction == "put":
        direction = "bearish"
    total = 0
    components: dict[str, int] = {}
    reasons: list[str] = []
    for name, (score, rs) in {
        "market": _market_score("call" if direction == "bullish" else "put", market or {}),
        "gamma": _gamma_score(contract, gamma or {}),
        "liquidity": _liquidity_score(contract, buying_power),
    }.items():
        components[name] = score
        total += score
        reasons.extend(f"{name}: {r}" for r in rs)
    hard_bad = any("breakeven" in r and "expected-move" in r and ("above" in r or "below" in r)
                   for r in reasons)
    late_cap = (_num((market or {}).get("minutes_to_close")) or 9999) <= 20
    if hard_bad:
        verdict = BAD_BET
    elif total >= 5 and not late_cap:
        verdict = GOOD_BET
    elif total <= -2:
        verdict = BAD_BET
    else:
        verdict = WATCH
    return {
        "schema_version": 1,
        "verdict": verdict,
        "score": total,
        "components": components,
        "direction": direction,
        "contract": contract,
        "reasons": reasons,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "basis": "non_sentiment_vehicle_scorecard: tape + gamma/pin + liquidity/account-fit",
        "places_orders": False,
    }


def run_vehicle_score(contract_json: str | None = None, contract_path: str | None = None,
                      market_json: str | None = None, market_path: str | None = None,
                      gamma_json: str | None = None, gamma_path: str | None = None,
                      direction: str | None = None, buying_power: str | float | None = None,
                      out_dir: str | None = None, write: bool = False) -> dict:
    contract = _load_json(contract_path, contract_json)
    market = _load_json(market_path, market_json, default={})
    gamma = _load_json(gamma_path, gamma_json, default={})
    bp = _num(buying_power)
    payload = score_vehicle(contract, direction=direction, market=market, gamma=gamma, buying_power=bp)
    if write:
        out = Path(os.path.expanduser(out_dir or ODTE_REPORT_DIR))
        out.mkdir(parents=True, exist_ok=True)
        sym = str(contract.get("underlying") or contract.get("symbol") or "vehicle").lower()
        path = out / f"odte_vehicle_score_{sym}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        payload["artifact"] = str(path)
    return payload


def render_markdown(payload: dict) -> str:
    c = payload.get("contract") or {}
    title = f"{c.get('underlying') or c.get('symbol') or 'candidate'} {c.get('strike') or c.get('strike_price') or ''}{str(c.get('option_type') or c.get('type') or '').upper()[:1]}"
    lines = [f"# 0DTE vehicle score: {title}".strip(), "",
             f"Verdict: **{payload.get('verdict')}**  ",
             f"Score: `{payload.get('score')}`  ",
             f"Basis: {payload.get('basis')}", "", "## Reasons"]
    for r in payload.get("reasons") or []:
        lines.append(f"- {r}")
    return "\n".join(lines)
