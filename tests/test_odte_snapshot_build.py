"""tests/test_odte_snapshot_build.py — the snapshot WRITER, pinned against the reader.

odte_breadth unified how the tape is READ (2026-08-07). This suite covers the other half: the
tape is now BUILT by a tool instead of by ad-hoc Python the controller re-authored each tick.
The properties that matter are the ones the improvised version got wrong — a bar-count opening
range, a rolling VWAP, and a vocabulary that drifted outside `above|below|inside`.

Everything reads its thresholds from the module constants.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data import odte_breadth as breadth
from data import odte_snapshot_build as sb

ET = sb.ET


def bars(day, start_min=0, count=8, *, interval=5, base=100.0, high_add=1.0,
         low_sub=1.0, vol=1000.0, drift=0.0):
    """Session bars starting at 09:30 ET + start_min, `interval` minutes apart."""
    open_dt = datetime.combine(day, sb.SESSION_OPEN_ET, tzinfo=ET)
    out = []
    for i in range(count):
        px = base + drift * i
        out.append({
            "begins_at": (open_dt + timedelta(minutes=start_min + i * interval))
                         .astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "open_price": str(px), "close_price": str(px),
            "high_price": str(px + high_add), "low_price": str(px - low_sub),
            "volume": str(vol),
        })
    return out


DAY = datetime(2026, 8, 10, tzinfo=ET).date()


# ── the bug the improvised version had: a bar-COUNT opening range ────────────────────────────

def test_opening_range_is_selected_by_time_not_bar_count():
    """`bars[:6]` is 30 minutes only for 5-minute bars. One-minute bars must give the SAME range."""
    five = bars(DAY, count=12, interval=5, base=100.0)     # 60 minutes of tape
    one = bars(DAY, count=60, interval=1, base=100.0)      # the same 60 minutes
    r5 = sb.opening_range(five, session_date=DAY)
    r1 = sb.opening_range(one, session_date=DAY)
    assert r5["high"] == r1["high"] and r5["low"] == r1["low"]
    assert r5["bars"] == 6 and r1["bars"] == 30, "same window, different bar counts"


def test_opening_range_incomplete_while_still_forming():
    """Before the window elapses there is no opening range — and saying so is the point."""
    partial = bars(DAY, count=3, interval=5)               # only 15 minutes of tape
    r = sb.opening_range(partial, session_date=DAY)
    assert r["complete"] is False


def test_orb_state_is_none_while_forming_not_inside():
    """The improvised version compared `last` against a range that INCLUDED `last`, so every
    early-session tick read `inside` and no breakout could ever be detected."""
    partial = bars(DAY, count=3, interval=5, base=100.0)
    r = sb.opening_range(partial, session_date=DAY)
    assert sb.orb_state(500.0, r) is None, "a range that has not formed cannot be broken"


def test_complete_needs_the_clock_not_a_bar_printed_at_the_window_end():
    """2026-08-11 live defect. Brokers return the last COMPLETED interval, so at 10:02 the newest
    5-minute bar still starts 09:55. The 09:30-10:00 range is fully observed either way — both
    cases below contain all 6 window bars — but the old rule waited for a bar AT the end and left
    a known range reporting `complete: False` for another whole bar interval."""
    through_0955 = bars(DAY, count=6, interval=5)          # 09:30..09:55, the realistic case
    at_1002 = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET) + timedelta(minutes=32)
    r = sb.opening_range(through_0955, session_date=DAY, now=at_1002)
    assert r["bars"] == 6, "the whole window is present"
    assert r["complete"] is True, "clock is past 10:00 and the bars reach it"


def test_complete_is_false_when_the_clock_has_not_passed_the_window():
    through_0955 = bars(DAY, count=6, interval=5)
    at_0950 = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET) + timedelta(minutes=20)
    assert sb.opening_range(through_0955, session_date=DAY, now=at_0950)["complete"] is False


def test_stale_bars_cannot_fake_completion():
    """The opposite error: clock alone would mark a 2-bar range complete and hand the tape lane an
    orb_high that is far too low, which manufactures phantom breakouts."""
    stale = bars(DAY, count=2, interval=5)                 # only 09:30..09:35
    long_after = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET) + timedelta(minutes=90)
    r = sb.opening_range(stale, session_date=DAY, now=long_after)
    assert r["complete"] is False, "clock elapsed, but the bars do not cover the window"


def test_bar_interval_is_inferred_not_assumed():
    one_min = bars(DAY, count=30, interval=1)              # 09:30..09:59
    at_1002 = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET) + timedelta(minutes=32)
    r = sb.opening_range(one_min, session_date=DAY, now=at_1002)
    assert r["complete"] is True, "1-minute bars reach 09:59; +1m interval covers 10:00"


def test_orb_state_resolves_once_the_window_closes():
    tape = bars(DAY, count=10, interval=5, base=100.0)     # 50 minutes: window closed
    r = sb.opening_range(tape, session_date=DAY)
    assert r["complete"] is True
    assert sb.orb_state(r["high"] + 0.5, r) == sb.ORB_ABOVE
    assert sb.orb_state(r["low"] - 0.5, r) == sb.ORB_BELOW
    assert sb.orb_state((r["high"] + r["low"]) / 2, r) == sb.ORB_INSIDE


# ── vocabulary: the 46 non-canonical values in the journal ───────────────────────────────────

@pytest.mark.parametrize("last_offset", [-5.0, 0.0, 5.0])
def test_emitted_orb_state_is_always_canonical(last_offset):
    tape = bars(DAY, count=10, interval=5, base=100.0)
    r = sb.opening_range(tape, session_date=DAY)
    state = sb.orb_state(r["high"] + last_offset, r)
    assert state in sb.CANONICAL_ORB_STATES


def test_audit_flags_the_real_journal_vocabulary():
    """The exact spellings observed in decision_journal.jsonl on 2026-08-10."""
    market = {
        "spy_orb_state": "above",              # fine
        "qqq_orb_state": "above_orb",          # directional, silently dropped by breadth
        "iwm_orb_state": "above_5m_range",     # directional, silently dropped
        "XSP": {"orb_state": "inside_30m_range"},   # non-canonical but not directional
        "VIXY": {"orb_state": "pre_open_no_orb"},   # genuinely "no ORB"
    }
    found = sb.audit_orb_vocabulary(market)
    values = {f["value"] for f in found}
    assert "above" not in values
    assert {"above_orb", "above_5m_range", "inside_30m_range", "pre_open_no_orb"} == values
    lost = {f["value"] for f in found if f["directional_signal_lost"]}
    assert lost == {"above_orb", "above_5m_range"}


def test_the_cost_of_a_dropped_reading_is_a_full_confirmer():
    """Why the vocabulary matters at all: breadth scores `above_orb` as if there were no ORB."""
    canonical = {"spy_above_vwap": True, "spy_orb_state": "above"}
    drifted = {"spy_above_vwap": True, "spy_orb_state": "above_orb"}
    assert breadth.alignment(canonical, "SPY", "bullish") == breadth.FULL_ALIGNMENT
    assert breadth.alignment(drifted, "SPY", "bullish") == breadth.VWAP_WEIGHT
    assert breadth.alignment(drifted, "SPY", "bullish") == breadth.alignment(
        {"spy_above_vwap": True}, "SPY", "bullish"), "identical to having no ORB data at all"


# ── VWAP ────────────────────────────────────────────────────────────────────────────────────

def test_session_vwap_is_volume_weighted_typical_price():
    tape = [
        {"begins_at": datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET).isoformat(),
         "high_price": 12.0, "low_price": 8.0, "close_price": 10.0, "volume": 100},
        {"begins_at": (datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET)
                       + timedelta(minutes=5)).isoformat(),
         "high_price": 22.0, "low_price": 18.0, "close_price": 20.0, "volume": 300},
    ]
    # typical prices 10 and 20, volumes 100 and 300 -> (10*100 + 20*300) / 400 = 17.5
    assert sb.session_vwap(tape, session_date=DAY) == pytest.approx(17.5)


def test_session_vwap_ignores_premarket_and_prior_day():
    """A caller that over-fetches must still get SESSION vwap, not a rolling window."""
    open_dt = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET)
    tape = [
        {"begins_at": (open_dt - timedelta(minutes=30)).isoformat(),   # premarket
         "high_price": 1000.0, "low_price": 1000.0, "close_price": 1000.0, "volume": 10_000},
        {"begins_at": open_dt.isoformat(),
         "high_price": 10.0, "low_price": 10.0, "close_price": 10.0, "volume": 100},
    ]
    assert sb.session_vwap(tape, session_date=DAY) == pytest.approx(10.0)


def test_zero_volume_bars_do_not_divide_by_zero():
    tape = bars(DAY, count=4, vol=0.0)
    assert sb.session_vwap(tape, session_date=DAY) is None


# ── snapshot assembly ───────────────────────────────────────────────────────────────────────

def test_flat_and_nested_shapes_are_written_from_ONE_computation():
    """odte_breadth merges both shapes; only agreement makes that safe."""
    tape = {s: bars(DAY, count=10, interval=5, base=100.0) for s in ("SPY", "QQQ", "IWM")}
    last = {"SPY": 106.0, "QQQ": 106.0, "IWM": 94.0}
    now = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET) + timedelta(minutes=60)
    snap = sb.build_market_snapshot(tape, last, now=now, session_date=DAY)
    for sym in ("SPY", "QQQ", "IWM"):
        assert snap[f"{sym.lower()}_above_vwap"] == snap[sym]["above_vwap"]
        assert snap[f"{sym.lower()}_orb_state"] == snap[sym]["orb_state"]


def test_snapshot_round_trips_through_the_shared_reader():
    """The writer's output must score the way the reader expects — the whole point of the module."""
    tape = {s: bars(DAY, count=10, interval=5, base=100.0) for s in ("SPY", "QQQ", "IWM")}
    last = {"SPY": 106.0, "QQQ": 106.0, "IWM": 94.0}   # SPY/QQQ break up, IWM breaks down
    now = datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET) + timedelta(minutes=60)
    snap = sb.build_market_snapshot(tape, last, now=now, session_date=DAY)
    assert breadth.alignment(snap, "SPY", "bullish") == breadth.FULL_ALIGNMENT
    assert breadth.alignment(snap, "QQQ", "bullish") == breadth.FULL_ALIGNMENT
    assert breadth.alignment(snap, "IWM", "bullish") <= 0, "IWM below VWAP and below its range"
    assert sb.audit_orb_vocabulary(snap) == [], "our own writer must never emit drift"


def test_uncomputable_fields_are_omitted_never_placeholdered():
    """The spec's rule: OMIT what you cannot compute; never placeholder 0/false."""
    snap = sb.build_market_snapshot({"SPY": bars(DAY, count=2, interval=5)}, {"SPY": 100.0},
                                    now=datetime.combine(DAY, sb.SESSION_OPEN_ET, tzinfo=ET)
                                    + timedelta(minutes=10), session_date=DAY)
    assert "spy_orb_state" not in snap, "range still forming — assert nothing"
    assert "orb_state" not in snap["SPY"]
    assert snap["SPY"]["above_vwap"] in (True, False), "VWAP IS computable, so it is present"


def test_missing_last_price_yields_no_vwap_side():
    snap = sb.build_market_snapshot({"SPY": bars(DAY, count=10)}, {"SPY": None},
                                    session_date=DAY)
    assert "spy_above_vwap" not in snap


def test_opening_range_minutes_is_recorded_in_the_snapshot():
    """The window was improvised per tick; recording it makes a future drift visible."""
    snap = sb.build_market_snapshot({"SPY": bars(DAY, count=10)}, {"SPY": 105.0},
                                    session_date=DAY)
    assert snap["opening_range_minutes"] == sb.OPENING_RANGE_MINUTES


# ── input tolerance ─────────────────────────────────────────────────────────────────────────

def test_robinhood_string_valued_bars_parse():
    b = sb.normalize_bar({"begins_at": "2026-08-10T13:30:00Z", "open_price": "772.64",
                          "close_price": "772.62", "high_price": "773.63",
                          "low_price": "772.58", "volume": "301336"})
    assert b["high"] == pytest.approx(773.63) and b["volume"] == pytest.approx(301336)


def test_nanosecond_timestamps_parse():
    """Hermes writes 9 fractional digits; fromisoformat takes 6."""
    b = sb.normalize_bar({"begins_at": "2026-08-10T13:30:00.062201113Z",
                          "high_price": 1, "low_price": 1, "close_price": 1, "volume": 1})
    assert b["ts"] is not None


def test_unusable_bars_are_dropped_not_guessed():
    assert sb.normalize_bar({"begins_at": "2026-08-10T13:30:00Z"}) is None
    assert sb.normalize_bar(None) is None
    assert sb.normalize_bar("nonsense") is None


def test_undated_bars_are_kept_in_order():
    rows = sb.session_bars([(10.0, 11.0, 9.0, 10.5, 100), (10.5, 12.0, 10.0, 11.0, 200)])
    assert len(rows) == 2 and rows[0]["close"] == pytest.approx(10.5)


def test_intraday_positions_count_as_open():
    # 2026-08-20: same-day positions carry quantity "0.0000" + intraday_quantity "1.0000" —
    # the one-open-idea count must see them.
    from data.odte_snapshot_build import build_broker_snapshot
    snap = build_broker_snapshot(
        {"data": {"buying_power": {"buying_power": "200.00"}}},
        {"data": {"results": [
            {"option_id": "a", "quantity": "0.0000", "intraday_quantity": "1.0000"},
            {"option_id": "b", "quantity": "0.0000", "intraday_quantity": "0.0000"}]}},
        {"data": {"orders": []}}, account_number="435050133")
    assert snap["nonzero_option_positions_count"] == 1
