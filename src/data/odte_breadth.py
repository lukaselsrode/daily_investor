"""Shared 0DTE tape primitives — ONE definition of VWAP side, opening-range side, and per-index
alignment for every module that reads a market snapshot.

PURE/OFFLINE: no broker, no network, no LLM, no file I/O. Callers pass a snapshot dict.

WHY THIS MODULE EXISTS (2026-08-07). ``odte_day_score._trend_score`` and
``odte_candidate_watch._confirmation_counts`` were two independent implementations of "is the tape
aligned?", and they drifted along three axes at once:

* **strictness** — day-score counted an index as aligned on the VWAP side alone; candidate-watch
  demanded VWAP side AND an opening-range breakout;
* **snapshot shape** — day-score read the flat ``{sym}_above_vwap`` / ``{sym}_orb_state`` keys,
  candidate-watch read the nested ``market["QQQ"]["orb_state"]`` blocks;
* **universe** — ``("spy","qqq","iwm")`` hardcoded vs ``SCAN_UNIVERSE`` (which includes XSP).

On 2026-08-07 15:30:03 that produced two contradictory verdicts on the SAME snapshot in the SAME
journal second: day-score logged "3 indices trend-aligned on ORB/VWAP — clean directional tape"
(GOOD_DAY) while candidate-watch logged ``confirmers: ['SPY']`` — 1 of 3, a count that could never
converge. The day was graded a clean trend day, which unlocks the full tier and drops the CHOP B+
requirement, while the entry gate simultaneously reported breadth that made entry impossible.

The two modules SHOULD differ in strictness — day regime and entry confirmation are different
questions. They should not differ in how they read a snapshot, nor disagree by accident. So the
shape-reading and the per-index vote live here once, and the differences that remain are explicit:
``require_vwap`` for strictness, an explicit ``universe`` argument, and thresholds owned by
``odte_config``.
"""
from __future__ import annotations

from typing import Any

from data.odte_config import SCAN_UNIVERSE

# A fully aligned index scores VWAP side + opening-range side. "Half" alignment (VWAP side with the
# index still inside its opening range) is the case the old binary confirmer count discarded
# entirely — it counted neither for nor against, which is how a SPY-led breakout with rangebound
# laggards produced an un-convergeable 1-of-3.
VWAP_WEIGHT = 1
ORB_WEIGHT = 1
FULL_ALIGNMENT = VWAP_WEIGHT + ORB_WEIGHT

_TRUE_WORDS = {"true", "yes", "1", "above", "above_vwap"}
_FALSE_WORDS = {"false", "no", "0", "below", "below_vwap"}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN guard


def _bool_field(block: dict, name: str) -> bool | None:
    value = block.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TRUE_WORDS:
            return True
        if key in _FALSE_WORDS:
            return False
    return None


def symbol_block(market: dict, symbol: str | None) -> dict:
    """Resolve one symbol's tape fields from EITHER snapshot shape.

    Nested ``market["SPY"]`` wins field-by-field over the flat ``market["spy_above_vwap"]`` form,
    but both are merged so a snapshot carrying only one shape reads identically to one carrying
    both. Today's controller happens to write both and they agree — that agreement is a coincidence
    of the writer, not a guarantee, and it is exactly the kind of silent divergence that would make
    two consumers disagree again.
    """
    if not isinstance(market, dict) or not symbol:
        return {}
    sym = str(symbol).strip().upper()
    if not sym:
        return {}

    merged: dict = {}
    prefix = f"{sym.lower()}_"
    for key, value in market.items():
        if isinstance(key, str) and key.lower().startswith(prefix):
            merged[key[len(prefix):].lower()] = value

    for key in (sym, sym.lower(), "underlying"):
        block = market.get(key)
        if isinstance(block, dict):
            merged.update(block)
            break
    return merged


def above_vwap(market: dict, symbol: str | None) -> bool | None:
    return _bool_field(symbol_block(market, symbol), "above_vwap")


def orb_state(market: dict, symbol: str | None) -> str:
    block = symbol_block(market, symbol)
    return str(block.get("orb_state") or block.get("opening_range_state") or "").strip().lower()


def alignment(market: dict, symbol: str | None, direction: str,
              *, require_vwap: bool = True) -> int | None:
    """Signed alignment of one index with `direction`, or None when the tape gives no read.

    ``+2`` fully aligned (VWAP side AND opening-range side) · ``+1`` half aligned (VWAP side, still
    inside the opening range) · ``0`` mixed · ``-1``/``-2`` opposed.

    ``require_vwap=True`` (the confirmation lane) returns None whenever the VWAP side is absent, so
    partial tape stays neutral and can never become a silent veto — XSP snapshots routinely ship
    without ``above_vwap`` and must not read as dissent. ``require_vwap=False`` (the day-regime
    lane) scores whatever fields are present, which is what ``_trend_score`` has always done.
    """
    vwap = above_vwap(market, symbol)
    if vwap is None and require_vwap:
        return None

    orb = orb_state(market, symbol)
    score = 0
    seen = False
    if vwap is not None:
        score += VWAP_WEIGHT if vwap else -VWAP_WEIGHT
        seen = True
    if orb == "above":
        score += ORB_WEIGHT
        seen = True
    elif orb == "below":
        score -= ORB_WEIGHT
        seen = True
    elif orb == "inside":
        seen = True  # a definitive read that contributes nothing either way

    if not seen:
        return None
    return score if str(direction or "").strip().lower() != "bearish" else -score


def breadth(market: dict, direction: str, universe: tuple[str, ...] | None = None) -> dict:
    """Bucket a universe by alignment with `direction`.

    ``score`` sums only the SUPPORTIVE alignments. Opposed indices are tracked separately in
    ``opposed`` rather than subtracted, because the dissent rule is a count with its own tolerance
    (``B_PLUS_MAX_DISSENTERS``) — netting them into the score would silently re-tighten the B+ tier
    that the 2026-08-02 retune deliberately opened.

    ``opposed`` is ``alignment <= 0``, which reproduces the legacy dissent predicate ("a DEFINITIVE
    opposite read") exactly. With a VWAP side required, a zero means one supportive and one opposing
    component — e.g. above VWAP but back below the opening range — which the old rule counted as
    dissent. Treating zero as neutral here would quietly widen the B+ tier a second time.
    """
    out: dict[str, Any] = {"score": 0, "full": [], "half": [], "opposed": [], "neutral": []}
    for symbol in (universe if universe is not None else SCAN_UNIVERSE):
        value = alignment(market, symbol, direction)
        if value is None:
            out["neutral"].append(symbol)
            continue
        if value >= FULL_ALIGNMENT:
            out["full"].append(symbol)
        elif value > 0:
            out["half"].append(symbol)
        else:
            out["opposed"].append(symbol)
        if value > 0:
            out["score"] += value
    return out


def vol_bias(market: dict) -> int:
    """Single signed volatility read: ``-1`` weak (supports calls) · ``+1`` firming (supports puts).

    The VWAP side wins whenever it is present; ``change_pct`` is consulted only as a tiebreak when
    it is absent. This is what makes "weak" and "firming" mutually exclusive by construction.

    Before this, ``_vixy_weak`` short-circuited on ``above_vwap is False`` and ``_vixy_firming`` on
    ``above_vwap is True``, with BOTH falling through to ``change_pct`` otherwise — so a VIXY that
    disagreed with itself (below VWAP but up on the day, or above VWAP but down on the day)
    satisfied both and handed a free volatility confirmation to EITHER direction. Flagged as
    telemetry on 2026-08-05, first observed live 2026-08-07 15:54:42.
    """
    block = symbol_block(market, "VIXY") or symbol_block(market, "VIX")
    if not block:
        return 0
    vwap = _bool_field(block, "above_vwap")
    if vwap is not None:
        return 1 if vwap else -1
    change = _num(block.get("change_pct") if block.get("change_pct") is not None
                  else block.get("pct_change"))
    if change is None or change == 0:
        return 0
    return 1 if change > 0 else -1


def vol_divergence(market: dict) -> bool:
    """True when the VWAP side and the day change disagree about volatility.

    The information the old ``vixy_conflict`` flag carried, kept as telemetry now that the two
    helpers can no longer both be true.
    """
    block = symbol_block(market, "VIXY") or symbol_block(market, "VIX")
    if not block:
        return False
    vwap = _bool_field(block, "above_vwap")
    change = _num(block.get("change_pct") if block.get("change_pct") is not None
                  else block.get("pct_change"))
    if vwap is None or change is None or change == 0:
        return False
    return vwap is not (change > 0)
