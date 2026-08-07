#!/usr/bin/env python
"""Read-only replay: what would a tool-emitted DEFAULT max-loss stop have done to past trades?

`odte_position` only produces THESIS_DEAD / TIME_RISK when the controller wrote a `thesis` /
`time_rules` block into the plan. With a thin plan the sole unconditional exit is BID_FLOOR, which
on a $1.00 entry fires at -95%. The 2026-08-07 IWM trade exited at -16% only because the agent
happened to write a thesis stop — badly, at the exact opening-range boundary.

Before giving the module a default stop, measure it: for every closed trade, reconstruct the
intra-trade P/L path and ask which trades the default would have exited EARLIER, and what that does
to realized P/L. A default that cuts winners is the wrong default.

Sources, in preference order per trade:
  1. `management_check.pnl_pct` inside the entry->exit window (joined by timestamp — modern events
     carry trade_id: None, and only one position is ever open, so the window is unambiguous);
  2. the postmortem `excursion` block (mae_pct / lowest_observed_bid_pct) when no series exists.

Writes nothing.

    .venv/bin/python scripts/odte_stop_replay.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_journal as oj  # noqa: E402

JOURNAL = "data/odte/decision_journal.jsonl"
CANDIDATE_STOPS = (-0.20, -0.30, -0.40, -0.50, -0.60)


def _ts(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value):
    """`pnl_pct` is fraction-encoded on 148 of 149 events, and PERCENT-encoded on one legacy row
    (2026-06-24, `pnl_pct: -26.56`). Un-normalised, that single row reads as -2656% and dominates
    every distribution taken over the series. A long option cannot lose more than 100%, so anything
    outside [-1, +5] is a unit error, not a datum."""
    v = _num(value)
    if v is None:
        return None
    return v / 100.0 if (v < -1.0 or v > 5.0) else v


def main() -> int:
    events = oj.read_events(JOURNAL)
    ledger = oj.summarize(events)
    checks = [e for e in events
              if e.get("event_type") == "management_check" and _pct(e.get("pnl_pct")) is not None]
    postmortems = {}
    for e in events:
        if e.get("event_type") == "postmortem" and isinstance(e.get("excursion"), dict):
            postmortems.setdefault(str(e.get("trade_id") or e.get("underlying")), e["excursion"])

    # REUSE the ledger join rather than re-deriving it. odte_service.trade_ledger already encodes
    # the entry_fill|order_filled -> lease -> exit_fill|order_closed -> postmortem joins, including
    # the two-lane merge that would otherwise double-count every fill.
    from ui.services import odte_service as svc
    svc._MEM_CACHE.clear()
    closed = []
    for row in svc.trade_ledger():
        entry, pnl = _num(row.get("entry_price")), _num(row.get("realized_pnl"))
        ets, xts = _ts(row.get("entry_ts")), _ts(row.get("exit_ts"))
        if entry and pnl is not None and ets and xts:
            closed.append({"entry_ts": ets, "exit_ts": xts, "entry": entry,
                           "exit": _num(row.get("exit_price")), "pnl": pnl,
                           "underlying": row.get("underlying"), "trade_id": row.get("trade_id"),
                           "rail": row.get("rail_fired"),
                           "mfe_pct": _num(row.get("mfe_pct")),
                           "mae_pct": _num(row.get("mae_pct"))})
    print(f"closed trades with a usable entry/exit/P&L: {len(closed)} "
          f"(journal reports n_closed={ledger['n_closed']})\n")

    rows = []
    for r in closed:
        series = [(_ts(c.get("ts")), _pct(c.get("pnl_pct")))
                  for c in checks
                  if r["entry_ts"] <= (_ts(c.get("ts")) or r["entry_ts"]) <= r["exit_ts"]]
        series = [(t, p) for t, p in series if t is not None]
        mae = min((p for _, p in series), default=None)
        source = "management_check series"
        if mae is None and r.get("mae_pct") is not None:
            mae, source = r["mae_pct"] / 100.0, "ledger mae_pct"
        if mae is None:
            ex = postmortems.get(str(r["trade_id"])) or postmortems.get(str(r["underlying"])) or {}
            raw = ex.get("mae_pct")
            if raw is None:
                raw = ex.get("lowest_observed_bid_pct")
            if raw is not None:
                mae = _num(raw) / 100.0
                source = "postmortem excursion"
        rows.append({**r, "mae": mae, "n_checks": len(series), "source": source})

    have = [r for r in rows if r["mae"] is not None]
    print(f"trades with a reconstructable adverse excursion: {len(have)}/{len(rows)}")
    for r in sorted(have, key=lambda x: str(x["entry_ts"])):
        print(f"   {str(r['entry_ts'])[:16]} {str(r['underlying'] or '?'):4s} "
              f"entry={r['entry']:.2f} pnl=${r['pnl']:+.2f} MAE={r['mae']:+.1%} "
              f"rail={str(r['rail'] or '-'):22s} ({r['n_checks']} checks, {r['source']})")

    print("\ncandidate default stops — trades the stop would have exited EARLIER:")
    print(f"   {'stop':>6}  {'hit':>4}  {'winners cut':>11}  {'losers capped':>13}   P/L delta")
    for stop in CANDIDATE_STOPS:
        hit = [r for r in have if r["mae"] is not None and r["mae"] <= stop]
        winners_cut = [r for r in hit if r["pnl"] > 0]
        losers = [r for r in hit if r["pnl"] <= 0]
        # capping a loser at `stop` of its entry debit; entry is per-share, x100 per contract
        delta = 0.0
        for r in hit:
            capped = stop * r["entry"] * 100.0
            delta += capped - r["pnl"]
        print(f"   {stop:>6.0%}  {len(hit):>4}  {len(winners_cut):>11}  {len(losers):>13}   "
              f"${delta:+.2f}")

    print("\n  'winners cut' is the number that matters: a default that stops a trade which went on")
    print("  to close green is a default that is too tight, however good the P/L delta looks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
