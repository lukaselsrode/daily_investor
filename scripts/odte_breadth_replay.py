#!/usr/bin/env python
"""Read-only replay: how often would the 2026-08-07 breadth reconciliation flip a decision?

The 0DTE lane has no simulator, so the only honest way to size a confirmation-gate change before
it reaches the live loop is to replay it over every market snapshot the system has actually seen
and count how many decisions move, in which direction, on which dates.

Two changes are measured SEPARATELY so their effects are not confounded:

  breadth  — `confirmations >= B_PLUS_MIN_CONFIRMATIONS` (count of fully aligned indices)
             becomes `breadth_score >= BREADTH_MIN_SCORE` (VWAP side + opening-range side summed,
             so an index above VWAP but inside its opening range finally counts for something).
  vol      — `_vixy_weak` / `_vixy_firming` short-circuit independently and can both be true;
             `vol_bias()` is a single signed read.

Both rules share the same precondition — the candidate's OWN underlying must be above VWAP and
through its opening range — so only snapshots meeting it can possibly flip.

Writes nothing. Reads the journal, the day packets, the archived controller snapshots, and the
convert-time snapshots.

    .venv/bin/python scripts/odte_breadth_replay.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_breadth as breadth  # noqa: E402
import data.odte_candidate_watch as cw  # noqa: E402
import data.odte_journal as oj  # noqa: E402
from data.odte_config import (  # noqa: E402
    B_PLUS_MIN_CONFIRMATIONS,
    DAILY_TRADE_BUDGET,
    EXECUTABLE_UNIVERSE,
)

DAILY_BUDGET = DAILY_TRADE_BUDGET

# The proposed threshold, expressed in the same units the plan defines it in: two fully aligned
# indices' worth of supportive breadth. Kept local so this script can run BEFORE odte_config gains
# the constant, and so re-running it after the fact proves the shipped value is the measured one.
BREADTH_MIN_SCORE = B_PLUS_MIN_CONFIRMATIONS * breadth.FULL_ALIGNMENT

JOURNAL = "data/odte/decision_journal.jsonl"


def _snapshots() -> list[tuple[str, str, dict]]:
    """(trade_date, as_of, snapshot) for every replayable tape, deduped by as_of."""
    seen: dict[str, tuple[str, str, dict]] = {}

    for event in oj.read_events(JOURNAL):
        if event.get("event_type") != "market_snapshot":
            continue
        as_of = str(event.get("as_of") or event.get("ts") or "")
        if as_of:
            seen.setdefault(as_of, (str(event.get("ts") or as_of)[:10], as_of, event))

    for pattern in ("data/odte/archive/**/controller_market_*.json",
                    "data/odte/convert_market_*.json",
                    "data/odte/days/*/market_*.json"):
        for path in glob.glob(pattern, recursive=True):
            try:
                payload = json.load(open(path))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            as_of = str(payload.get("as_of") or payload.get("generated_at") or "")
            if as_of:
                seen.setdefault(as_of, (as_of[:10], as_of, payload))

    return sorted(seen.values(), key=lambda row: row[1])


def _eligible(market: dict, symbol: str, direction: str) -> bool:
    """The precondition both rules share: the candidate's own underlying must be aligned."""
    above = breadth.above_vwap(market, symbol)
    orb = breadth.orb_state(market, symbol)
    if direction == "bullish":
        return above is True and orb == "above"
    return above is False and orb == "below"


def _legacy_counts(market: dict, direction: str) -> tuple[int, list[str]]:
    """The OLD confirmation rule, re-expressed over the SHARED reader.

    `cw._confirmation_counts` resolves symbols through a nested-block-only lookup, so on a
    flat-shaped snapshot it sees an empty tape and returns (0, [], []). Comparing that against the
    new code would measure the shape fix, not the rule change — the two must be held apart.
    """
    hits: list[str] = []
    dissents: list[str] = []
    for symbol in breadth.SCAN_UNIVERSE:
        above = breadth.above_vwap(market, symbol)
        if above is None:
            continue                      # partial tape is neutral, never a silent veto
        orb = breadth.orb_state(market, symbol)
        if direction == "bullish":
            if above is True and orb == "above":
                hits.append(symbol)
            elif above is False or orb == "below":
                dissents.append(symbol)
        else:
            if above is False and orb == "below":
                hits.append(symbol)
            elif above is True or orb == "above":
                dissents.append(symbol)
    return len(hits), dissents


def _legacy_vol(market: dict, direction: str) -> bool:
    """The OLD two-helper volatility read over the shared reader — both can be true at once."""
    block = breadth.symbol_block(market, "VIXY") or breadth.symbol_block(market, "VIX")
    above = breadth._bool_field(block, "above_vwap")
    change = breadth._num(block.get("change_pct") if block.get("change_pct") is not None
                       else block.get("pct_change"))
    if direction == "bullish":                                    # _vixy_weak
        if above is False:
            return True
        return bool(change is not None and change < 0)
    if above is True:                                             # _vixy_firming
        return True
    return bool(change is not None and change > 0)


def main() -> int:
    snapshots = _snapshots()
    if not snapshots:
        print("no replayable snapshots found")
        return 1

    flips: dict[str, Counter] = {"breadth": Counter(), "vol": Counter(), "combined": Counter()}
    by_date: dict[str, Counter] = defaultdict(Counter)
    examples: list[str] = []
    unreadable_by_old_reader: set[tuple[str, str]] = set()
    confirmed_by_date: dict[str, dict[str, bool]] = defaultdict(
        lambda: {"old": False, "new": False})
    tier_of_opens: Counter = Counter()
    considered = 0

    for trade_date, as_of, market in snapshots:
        for symbol in EXECUTABLE_UNIVERSE:
            for direction in ("bullish", "bearish"):
                if not _eligible(market, symbol, direction):
                    continue
                considered += 1

                count, _dissenters = _legacy_counts(market, direction)
                score = breadth.breadth(market, direction)["score"]

                old_breadth = count >= B_PLUS_MIN_CONFIRMATIONS
                new_breadth = score >= BREADTH_MIN_SCORE

                old_vol = _legacy_vol(market, direction)
                bias = breadth.vol_bias(market)
                new_vol = bias < 0 if direction == "bullish" else bias > 0

                nested_blind = not cw._symbol_block(market, symbol)
                if nested_blind:
                    unreadable_by_old_reader.add((trade_date, as_of))

                pairs = {
                    "breadth": (old_breadth and old_vol, new_breadth and old_vol),
                    "vol": (old_breadth and old_vol, old_breadth and new_vol),
                    "combined": (old_breadth and old_vol, new_breadth and new_vol),
                }
                for axis, (was, now) in pairs.items():
                    if was == now:
                        flips[axis]["same"] += 1
                    elif now:
                        flips[axis]["opens"] += 1
                    else:
                        flips[axis]["closes"] += 1

                was, now = pairs["combined"]
                confirmed_by_date[trade_date]["old"] |= was
                confirmed_by_date[trade_date]["new"] |= now
                if now and not was:
                    # Every newly opened entry also carries a TIER, and the tier decides the debit
                    # fraction. Half-derived breadth (fewer than B_PLUS_MIN_CONFIRMATIONS fully
                    # aligned indices) sizes at the B+ rail, so counting opens without counting
                    # tiers overstates the added exposure by 2x on exactly the tapes this change
                    # admits. Report the split rather than leaving it to be assumed.
                    full_n = len(breadth.breadth(market, direction)["full"])
                    tier_of_opens["b_plus" if full_n < B_PLUS_MIN_CONFIRMATIONS else "full"] += 1
                if was != now:
                    by_date[trade_date]["opens" if now else "closes"] += 1
                    if len(examples) < 12:
                        examples.append(
                            f"    {as_of}  {symbol:4s} {direction:8s} "
                            f"{'OPENS ' if now else 'CLOSES'}  "
                            f"count {count}->score {score}  vol {int(old_vol)}->{int(new_vol)}")

    print(f"replayable snapshots ........ {len(snapshots)}")
    print(f"candidate situations ........ {considered}   "
          f"(underlying itself aligned — the only ones that can flip)")
    print(f"threshold ................... breadth_score >= {BREADTH_MIN_SCORE} "
          f"(was confirmations >= {B_PLUS_MIN_CONFIRMATIONS})\n")

    for axis in ("breadth", "vol", "combined"):
        row = flips[axis]
        total = sum(row.values()) or 1
        print(f"  {axis:9s} same {row['same']:5d}   opens {row['opens']:4d}   "
              f"closes {row['closes']:4d}   ({(row['opens'] + row['closes']) / total:.1%} moved)")

    if by_date:
        print("\n  by trade date (combined):")
        for date in sorted(by_date):
            row = by_date[date]
            print(f"    {date}  opens {row['opens']:3d}  closes {row['closes']:3d}")

    # Ticks are not trades. The same setup re-confirms on every tick it survives, and the daily
    # budget caps what any of them can become. The decision-relevant number is how many SESSIONS
    # go from "never confirmed" to "confirmed at least once" — that is the trade-count exposure.
    gained = sorted(d for d, seen in confirmed_by_date.items() if seen["new"] and not seen["old"])
    lost = sorted(d for d, seen in confirmed_by_date.items() if seen["old"] and not seen["new"])
    print(f"\n  sessions replayed ........... {len(confirmed_by_date)}")
    print(f"  sessions gaining a confirm .. {len(gained)}  {', '.join(gained) if gained else ''}")
    print(f"  sessions losing every confirm {len(lost)}  {', '.join(lost) if lost else ''}")
    print(f"  at the {DAILY_BUDGET}/day budget cap that is at most "
          f"{len(gained) * DAILY_BUDGET} added trades across the replayed history.")
    if tier_of_opens:
        half, whole = tier_of_opens.get("b_plus", 0), tier_of_opens.get("full", 0)
        print(f"  newly opened entries by sizing tier: b_plus(half) {half}  ·  full {whole}"
              + ("  — every added entry sizes at the HALF rail" if not whole else ""))

    if examples:
        print("\n  examples:")
        print("\n".join(examples))

    if unreadable_by_old_reader:
        dates = sorted({date for date, _ in unreadable_by_old_reader})
        print(f"\n  NOTE — {len(unreadable_by_old_reader)} of these snapshots are flat-shaped and the")
        print("  nested-only reader in candidate_watch resolves EVERY symbol to {} on them, so the")
        print("  old confirmation count is a structural 0 rather than a tape read. Measured above")
        print("  over the shared reader so this is excluded from the flip counts; reported here")
        print(f"  because it is its own finding. Dates: {', '.join(dates)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
