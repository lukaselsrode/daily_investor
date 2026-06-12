"""
reporting/position_watch.py — daily intra-week watchtower for held positions.

Live buy/sell execution runs ONCE WEEKLY (mid-day Wednesday), but data is
fetched daily. Between executions the human is the only circuit breaker —
this module makes that workable: every fetch, it surfaces LOUD warnings for
state the bot cannot act on until Wednesday (stop breaches, near-stops,
take-profit/harvest levels reached, regime transitions) and appends them to a
small ledger so the pattern of intra-week events is trackable over time.

Read-only with respect to the portfolio: never places orders, never mutates
holdings. Writes only its own ledger/state files under the data directory.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os

from util import HARVEST_PARAMS, SELL_RULES

logger = logging.getLogger(__name__)

_WATCH_CSV = "position_watch.csv"
_STATE_JSON = "position_watch_state.json"
_NEAR_BAND = 0.05   # warn within 5pp of the stop — "one bad day from a breach"


def _data_dir() -> str:
    import data.cache as _dc  # call-time attr so test data-dir redirects apply
    return str(_dc.DATA_DIRECTORY)


def _pnl(holding: dict) -> float | None:
    try:
        return float(holding.get("percent_change")) / 100.0
    except (TypeError, ValueError):
        return None


def build_watch_events(
    holdings: dict,
    regime_today: str | None,
    regime_prev: str | None,
    sell_rules: dict | None = None,
    harvest_threshold: float | None = None,
    near_band: float = _NEAR_BAND,
) -> list[dict]:
    """Pure event builder — returns a list of {type, symbol, detail, value}.

    Thresholds default to the LIVE config (sell_rules/harvest), so warnings
    always describe what the bot itself would do on the next Wednesday run.
    """
    sr = sell_rules if sell_rules is not None else SELL_RULES
    stop = float(sr.get("stop_loss_pct", -0.30))
    tp = float(sr.get("take_profit_pct", 0.80))
    harvest = (
        float(harvest_threshold)
        if harvest_threshold is not None
        else float(HARVEST_PARAMS.get("harvest_profit_threshold", 0.50))
    )

    events: list[dict] = []
    if regime_today and regime_prev and regime_today != regime_prev:
        events.append({
            "type": "regime_change", "symbol": "",
            "detail": f"regime {regime_prev} -> {regime_today}; sleeve sizing and "
                      f"stops change at the next weekly run",
            "value": regime_today,
        })

    for sym, h in (holdings or {}).items():
        pnl = _pnl(h)
        if pnl is None:
            continue
        if pnl <= stop:
            events.append({
                "type": "stop_breach", "symbol": sym,
                "detail": f"{sym} {pnl:+.1%} is through the {stop:.0%} stop — the bot "
                          f"cannot act until Wednesday; manual review warranted",
                "value": f"{pnl:.4f}",
            })
        elif pnl <= stop + near_band:
            events.append({
                "type": "stop_near", "symbol": sym,
                "detail": f"{sym} {pnl:+.1%} is within {near_band:.0%} of the "
                          f"{stop:.0%} stop",
                "value": f"{pnl:.4f}",
            })
        if pnl >= tp:
            events.append({
                "type": "take_profit_reached", "symbol": sym,
                "detail": f"{sym} {pnl:+.1%} is past the +{tp:.0%} full-exit level",
                "value": f"{pnl:.4f}",
            })
        elif pnl >= harvest:
            events.append({
                "type": "harvest_reached", "symbol": sym,
                "detail": f"{sym} {pnl:+.1%} is past the +{harvest:.0%} harvest level",
                "value": f"{pnl:.4f}",
            })
    return events


def _load_state() -> dict:
    path = os.path.join(_data_dir(), _STATE_JSON)
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(os.path.join(_data_dir(), _STATE_JSON), "w") as fh:
            json.dump(state, fh)
    except Exception as exc:
        logger.debug("position-watch state save failed: %s", exc)


def _append_ledger(events: list[dict], today: str) -> None:
    if not events:
        return
    path = os.path.join(_data_dir(), _WATCH_CSV)
    is_new = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if is_new:
                w.writerow(["date", "type", "symbol", "detail", "value"])
            for e in events:
                w.writerow([today, e["type"], e["symbol"], e["detail"], e["value"]])
    except Exception as exc:
        logger.debug("position-watch ledger append failed: %s", exc)


def run_position_watch(holdings: dict) -> list[dict]:
    """Daily watch entry point (called from fetch-data). Never raises."""
    try:
        today = str(datetime.date.today())
        state = _load_state()

        regime_today: str | None = None
        try:
            from strategy.regimes.detector import get_current_regime
            regime_today = get_current_regime()
        except Exception as exc:
            logger.debug("position-watch regime unavailable: %s", exc)

        events = build_watch_events(holdings, regime_today, state.get("regime"))
        for e in events:
            logger.warning("⚠ WATCH %s: %s", e["type"], e["detail"])
        if not events:
            logger.info("Position watch: no intra-week warnings (%d holdings, regime=%s)",
                        len(holdings or {}), regime_today or "n/a")
        _append_ledger(events, today)
        _save_state({"date": today, "regime": regime_today or state.get("regime")})
        return events
    except Exception as exc:
        logger.warning("position watch failed (continuing): %s", exc)
        return []
