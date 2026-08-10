#!/usr/bin/env python
"""Build market.json from raw Robinhood MCP payloads — the fast path's tape step.

Replaces the ad-hoc Python the controller re-authored every tick. That script did two costly
things: it recomputed VWAP/opening-range from scratch (see `data.odte_snapshot_build` for what
that cost), and it TRANSCRIBED the bar data into Python literals inline, by hand, inside a
30-second budget.

So this takes the MCP responses as-is. Pipe them in; get the snapshot out.

    .venv/bin/python scripts/odte_build_snapshot.py \
        --historicals /tmp/hist.json --quotes /tmp/quotes.json \
        --gap-pct -0.0787 --out /tmp/market.json

`--historicals` accepts whatever `get_equity_historicals` returned (a dict of symbol -> bars, a
list of per-symbol result blocks, or a wrapper carrying either under results/data/historicals).
`--quotes` likewise for `get_equity_quotes`. Fields that cannot be computed are OMITTED, matching
docs/hermes/v2/fast_path_snapshots.md — never placeholdered.

Exit codes: 0 built, 2 nothing usable in the inputs, 3 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.odte_snapshot_build import (  # noqa: E402
    audit_orb_vocabulary, build_market_snapshot,
)

_BAR_LIST_KEYS = ("historicals", "bars", "results", "data", "candles")
_SYMBOL_KEYS = ("symbol", "chain_symbol", "underlying", "instrument_symbol")
_PRICE_KEYS = ("last_trade_price", "last_price", "price", "mark_price", "last")
_PREV_KEYS = ("previous_close", "prev_close", "adjusted_previous_close",
              "last_non_reg_trade_price", "close")


def _unwrap(value, _depth: int = 0):
    """Peel the MCP envelope: {"result": "<json string>"} -> {"data": {"results": [...]}}.

    Verified against production 2026-08-10: both get_equity_quotes and get_equity_historicals
    return the payload DOUBLE-ENCODED — a JSON string under `result`. Guessing this shape from a
    hand-written fixture is what made the first version of this script fail live with
    "no last prices" on its very first real invocation.
    """
    if _depth > 6:
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("{", "[")):
            try:
                return _unwrap(json.loads(s), _depth + 1)
            except ValueError:
                return value
        return value
    if isinstance(value, dict):
        for key in ("result", "content", "payload"):
            if key in value and len(value) <= 2:
                return _unwrap(value[key], _depth + 1)
    return value


def _load(path: str | None) -> object:
    raw = json.load(sys.stdin) if (not path or path == "-") \
        else json.loads(Path(path).expanduser().read_text())
    return _unwrap(raw)


def _as_bar_list(value):
    """Pull a list of bar dicts out of whatever wrapper it arrived in."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k in _BAR_LIST_KEYS:
            inner = value.get(k)
            if isinstance(inner, list):
                return inner
    return None


def extract_bars(payload) -> dict:
    """symbol -> [bar, ...] from any of the shapes the historicals tool returns."""
    out: dict[str, list] = {}
    if isinstance(payload, dict):
        # unwrap one level of results/data if present
        for k in _BAR_LIST_KEYS:
            inner = payload.get(k)
            if isinstance(inner, (list, dict)) and k in ("results", "data", "historicals"):
                sub = extract_bars(inner)
                if sub:
                    return sub
        for key, value in payload.items():
            rows = _as_bar_list(value)
            if rows and isinstance(key, str) and len(key) <= 8:
                out[key.upper()] = rows
    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            sym = next((str(entry[k]) for k in _SYMBOL_KEYS if entry.get(k)), None)
            rows = _as_bar_list(entry)
            if sym and rows:
                out[sym.upper()] = rows
    return out


def _first_num(block: dict, keys) -> float | None:
    for k in keys:
        v = block.get(k)
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            return f
    return None


def extract_quotes(payload) -> tuple[dict, dict]:
    """(last_by_symbol, prev_close_by_symbol) from the quotes tool's shapes."""
    last: dict[str, float] = {}
    prev: dict[str, float] = {}

    def take(sym, block):
        if not isinstance(block, dict):
            return
        q = block.get("quote") if isinstance(block.get("quote"), dict) else block
        px = _first_num(q, _PRICE_KEYS)
        pc = _first_num(q, _PREV_KEYS)
        if px is not None:
            last[str(sym).upper()] = px
        if pc is not None:
            prev[str(sym).upper()] = pc

    if isinstance(payload, dict):
        for k in ("quotes", "results", "data"):
            if isinstance(payload.get(k), (dict, list)):
                return extract_quotes(payload[k])
        for sym, block in payload.items():
            if isinstance(block, dict):
                take(sym, block)
    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            # Production shape is {"quote": {"symbol": ..., ...}, "close": {...}} — the symbol
            # lives INSIDE the quote block, not on the entry. Looking only at the entry level
            # found nothing and silently produced zero prices (2026-08-10 live failure).
            inner = entry.get("quote") if isinstance(entry.get("quote"), dict) else entry
            sym = (next((entry[k] for k in _SYMBOL_KEYS if entry.get(k)), None)
                   or next((inner[k] for k in _SYMBOL_KEYS if inner.get(k)), None))
            if sym:
                take(sym, entry)
                if not prev.get(str(sym).upper()):
                    close = entry.get("close")
                    if isinstance(close, dict):
                        pc = _first_num(close, ("price", *_PREV_KEYS))
                        if pc is not None:
                            prev[str(sym).upper()] = pc
    return last, prev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--historicals", required=True, help="get_equity_historicals payload ('-' = stdin)")
    ap.add_argument("--quotes", help="get_equity_quotes payload; omit to pass --last")
    ap.add_argument("--last", help='JSON {"SPY": 773.86, ...} when quotes are unavailable')
    ap.add_argument("--prev-close", help='JSON {"VIXY": 19.56, ...}')
    ap.add_argument("--gap-pct", type=float, help="session constant; include it EVERY tick")
    ap.add_argument("--spot-symbol", default="SPY")
    ap.add_argument("--out", help="write here as well as stdout")
    args = ap.parse_args(argv)

    try:
        bars = extract_bars(_load(args.historicals))
    except (OSError, ValueError) as exc:
        print(f"could not read --historicals: {exc}", file=sys.stderr)
        return 3
    if not bars:
        print("no usable bars found in --historicals", file=sys.stderr)
        return 2

    last: dict = {}
    prev: dict = {}
    if args.quotes:
        try:
            last, prev = extract_quotes(_load(args.quotes))
        except (OSError, ValueError) as exc:
            print(f"could not read --quotes: {exc}", file=sys.stderr)
            return 3
    if args.last:
        last.update({str(k).upper(): v for k, v in json.loads(args.last).items()})
    if args.prev_close:
        prev.update({str(k).upper(): v for k, v in json.loads(args.prev_close).items()})
    if not last:
        print("no last prices — pass --quotes or --last", file=sys.stderr)
        return 2

    snap = build_market_snapshot(bars, last, now=datetime.now(timezone.utc),
                                 prev_close_by_symbol=prev, spot_symbol=args.spot_symbol,
                                 gap_pct=args.gap_pct)

    drift = audit_orb_vocabulary(snap)
    if drift:                                   # our own writer must never produce this
        print(f"INTERNAL: non-canonical orb_state emitted: {drift}", file=sys.stderr)
        return 2

    text = json.dumps(snap, indent=2) + "\n"
    if args.out:
        Path(args.out).expanduser().write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
