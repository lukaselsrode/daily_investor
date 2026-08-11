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
    audit_orb_vocabulary, build_contract_snapshot, build_market_snapshot, select_contract,
)

_INSTRUMENT_LIST_KEYS = ("instruments", "results", "data")

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


def extract_list(payload, keys=_INSTRUMENT_LIST_KEYS) -> list:
    """Pull a list of dicts out of the MCP envelope, wherever it hid it.

    Real shapes seen in production: instruments arrive under `data.instruments`, option quotes
    under `data.results` (each entry wrapping the real record in `quote`). Both sit inside the
    double-encoded `{"result": "<json>"}` envelope that `_load` has already peeled.
    """
    payload = _unwrap(payload)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in keys:
            inner = payload.get(k)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
            if isinstance(inner, dict):
                got = extract_list(inner, keys)
                if got:
                    return got
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--historicals", help="get_equity_historicals payload ('-' = stdin); required for tape mode")
    ap.add_argument("--quotes", help="get_equity_quotes payload; omit to pass --last")
    ap.add_argument("--last", help='JSON {"SPY": 773.86, ...} when quotes are unavailable')
    ap.add_argument("--prev-close", help='JSON {"VIXY": 19.56, ...}')
    ap.add_argument("--gap-pct", type=float, help="session constant; include it EVERY tick")
    ap.add_argument("--spot-symbol", default="SPY")
    ap.add_argument("--out", help="write here as well as stdout")
    ap.add_argument("--instruments", help="get_option_instruments payload (with --option-quotes)")
    ap.add_argument("--option-quotes", help="get_option_quotes payload (with --instruments)")
    ap.add_argument("--out-contract", help="write the joined contract.json here")
    ap.add_argument("--contract-option-id", help="pick this exact option uuid")
    ap.add_argument("--contract-strike", help="pick this strike (e.g. 301.0)")
    ap.add_argument("--contract-type", help="'call' or 'put'")
    args = ap.parse_args(argv)

    # --- contract mode: join instruments to quotes, no tape needed -----------------------------
    if args.instruments or args.option_quotes:
        if not (args.instruments and args.option_quotes):
            print("contract mode needs BOTH --instruments and --option-quotes", file=sys.stderr)
            return 3
        try:
            insts = extract_list(_load(args.instruments))
            quotes = extract_list(_load(args.option_quotes), ("results", "quotes", "data"))
        except (OSError, ValueError) as exc:
            print(f"could not read contract inputs: {exc}", file=sys.stderr)
            return 3
        if not insts:
            # The FIRST get_option_instruments call reliably returns [] and a retry ~10s later
            # succeeds (14/14 across 2026-08-10..11, identical arguments). Say so plainly rather
            # than emitting a contract we cannot describe.
            print("no instruments in payload — cold read? re-fetch and retry", file=sys.stderr)
            return 2
        picked = select_contract(insts, quotes, option_id=args.contract_option_id,
                                 strike_price=args.contract_strike,
                                 option_type=args.contract_type)
        if not picked:
            print("no tradable instrument matched a quote", file=sys.stderr)
            return 2
        contract = build_contract_snapshot(picked[0], picked[1], now=datetime.now(timezone.utc))
        text = json.dumps(contract, indent=2) + "\n"
        if args.out_contract:
            Path(args.out_contract).expanduser().write_text(text)
        sys.stdout.write(text)
        return 0

    if not args.historicals:
        print("tape mode needs --historicals (or use --instruments/--option-quotes)",
              file=sys.stderr)
        return 3
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
