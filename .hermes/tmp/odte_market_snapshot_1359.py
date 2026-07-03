#!/usr/bin/env python3
"""Build a compact flattened ODTE market snapshot from yfinance bars."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLS = ["SPY", "QQQ", "IWM", "VIXY"]


def _series(frame: pd.DataFrame, name: str, symbol: str) -> pd.Series:
    data = frame[name]
    if isinstance(data, pd.DataFrame):
        if symbol in data.columns:
            return data[symbol].dropna()
        return data.iloc[:, 0].dropna()
    return data.dropna()


def _regular_bars(symbol: str) -> pd.DataFrame:
    hist = yf.download(symbol, period="2d", interval="1m", auto_adjust=False, progress=False, prepost=False, threads=False)
    if hist.empty:
        raise RuntimeError(f"no bars for {symbol}")
    if isinstance(hist.columns, pd.MultiIndex):
        # yfinance may return Price/Ticker or Ticker/Price depending version.
        if symbol in hist.columns.get_level_values(-1):
            hist = hist.xs(symbol, axis=1, level=-1, drop_level=True)
        elif symbol in hist.columns.get_level_values(0):
            hist = hist.xs(symbol, axis=1, level=0, drop_level=True)
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")
    eastern = hist.tz_convert("America/New_York")
    today = eastern.index[-1].date()
    bars = eastern[eastern.index.date == today].between_time("09:30", "16:00")
    if bars.empty:
        bars = eastern[eastern.index.date == today]
    if bars.empty:
        raise RuntimeError(f"no same-day bars for {symbol}")
    return bars


def _quote(symbol: str) -> dict[str, float | str | bool | None]:
    bars = _regular_bars(symbol)
    close = _series(bars, "Close", symbol)
    high = _series(bars, "High", symbol)
    low = _series(bars, "Low", symbol)
    volume = _series(bars, "Volume", symbol)
    typical = (high + low + close) / 3.0
    vwap = float((typical * volume).sum() / volume.sum()) if float(volume.sum()) > 0 else float(close.iloc[-1])
    first_30 = bars.between_time("09:30", "10:00")
    if first_30.empty:
        first_30 = bars.head(min(30, len(bars)))
    orb_high = float(_series(first_30, "High", symbol).max())
    orb_low = float(_series(first_30, "Low", symbol).min())
    price = float(close.iloc[-1])
    prev = yf.Ticker(symbol).fast_info.get("previous_close")
    prev_close = float(prev) if prev is not None else None
    gap_pct = ((price / prev_close) - 1.0) if prev_close else None
    return {
        "price": price,
        "vwap": vwap,
        "above_vwap": price > vwap,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "above_orb": price > orb_high,
        "below_orb": price < orb_low,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "last_bar_et": bars.index[-1].isoformat(),
    }


def main() -> None:
    out: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "minutes_to_close": None,
        "expected_move_pct": 0.006,
    }
    for sym in SYMBOLS:
        try:
            q = _quote(sym)
        except Exception as exc:  # noqa: BLE001 - artifact should capture degraded fields.
            q = {"error": str(exc)}
        out[sym] = q
        out[sym.lower()] = q
    spy = out.get("SPY", {}) if isinstance(out.get("SPY"), dict) else {}
    qqq = out.get("QQQ", {}) if isinstance(out.get("QQQ"), dict) else {}
    iwm = out.get("IWM", {}) if isinstance(out.get("IWM"), dict) else {}
    vixy = out.get("VIXY", {}) if isinstance(out.get("VIXY"), dict) else {}
    out.update({
        "spy_price": spy.get("price"),
        "spy_vwap": spy.get("vwap"),
        "spy_above_vwap": spy.get("above_vwap"),
        "spy_above_orb": spy.get("above_orb"),
        "spy_gap_pct": spy.get("gap_pct"),
        "qqq_price": qqq.get("price"),
        "qqq_vwap": qqq.get("vwap"),
        "qqq_above_vwap": qqq.get("above_vwap"),
        "qqq_above_orb": qqq.get("above_orb"),
        "qqq_gap_pct": qqq.get("gap_pct"),
        "iwm_price": iwm.get("price"),
        "iwm_vwap": iwm.get("vwap"),
        "iwm_above_vwap": iwm.get("above_vwap"),
        "iwm_above_orb": iwm.get("above_orb"),
        "iwm_gap_pct": iwm.get("gap_pct"),
        "vixy_price": vixy.get("price"),
        "vixy_vwap": vixy.get("vwap"),
        "vixy_below_vwap": (vixy.get("price") is not None and vixy.get("vwap") is not None and vixy.get("price") < vixy.get("vwap")),
        "vixy_gap_pct": vixy.get("gap_pct"),
    })
    Path("/tmp/odte_market_1359.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
