"""tests/test_odte_tape.py — fast-lane tape engine (bars + quotes → market snapshot).

Pure/offline except the FakeMcpSession transport tests. Pins the math (session VWAP, the
6-bar/30-minute ORB frozen at 10:00 ET, gap_pct as a session constant), the OMIT-never-
placeholder rule for partial tape, and — critically — that the emitted snapshot is accepted
verbatim by the LIVE consumers: score_day's flat keys and candidate-watch's per-symbol blocks.
Payload fixtures mirror the real server shapes probed 2026-08-04.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import execution.odte_tape as tape

UTC = timezone.utc
SESSION_OPEN = datetime(2026, 8, 5, 13, 30, tzinfo=UTC)        # 09:30 ET
MIDDAY = datetime(2026, 8, 5, 15, 1, tzinfo=UTC)               # 11:01 ET (post-ORB-freeze)
EARLY = datetime(2026, 8, 5, 13, 45, tzinfo=UTC)               # 09:45 ET (pre-freeze)


def _bars(n, base=760.0, step=0.5, volume=1000.0, start=SESSION_OPEN):
    out = []
    for i in range(n):
        px = base + i * step
        out.append({"ts": start + timedelta(minutes=5 * i), "open": px, "high": px + 1.0,
                    "low": px - 1.0, "close": px + 0.5, "volume": volume, "session": "reg"})
    return out


def _quotes(**last_by_symbol):
    out = {}
    for sym, last in last_by_symbol.items():
        out[sym.upper()] = {"last": last, "previous_close": 755.0, "bid": last - 0.02,
                            "ask": last + 0.02}
    return out


# --- math ------------------------------------------------------------------------------------

def test_session_vwap_matches_hand_computation():
    bars = [{"ts": SESSION_OPEN, "open": 10, "high": 12, "low": 8, "close": 10,
             "volume": 100, "session": "reg"},
            {"ts": SESSION_OPEN + timedelta(minutes=5), "open": 10, "high": 14, "low": 10,
             "close": 12, "volume": 300, "session": "reg"}]
    # typical prices: (12+8+10)/3 = 10, (14+10+12)/3 = 12 -> vwap = (10*100+12*300)/400 = 11.5
    snap, _ = tape.compute_tape({"SPY": bars}, _quotes(SPY=11.6), MIDDAY)
    assert snap["SPY"]["vwap"] == 11.5
    assert snap["SPY"]["above_vwap"] is True and snap["spy_above_vwap"] is True
    snap2, _ = tape.compute_tape({"SPY": bars}, _quotes(SPY=11.4), MIDDAY)
    assert snap2["SPY"]["above_vwap"] is False


def test_orb_is_first_six_bars_frozen_after_ten_et():
    bars = _bars(12)                                           # a full hour of bars
    orb_high = max(b["high"] for b in bars[:tape.ORB_BAR_COUNT])
    orb_low = min(b["low"] for b in bars[:tape.ORB_BAR_COUNT])
    snap, _ = tape.compute_tape({"SPY": bars}, _quotes(SPY=orb_high + 1), MIDDAY)
    assert snap["SPY"]["orb_high"] == orb_high and snap["SPY"]["orb_low"] == orb_low
    assert snap["SPY"]["orb_state"] == "above" and snap["spy_orb_state"] == "above"
    snap2, _ = tape.compute_tape({"SPY": bars}, _quotes(SPY=orb_low - 1), MIDDAY)
    assert snap2["SPY"]["orb_state"] == "below"
    snap3, _ = tape.compute_tape({"SPY": bars}, _quotes(SPY=(orb_high + orb_low) / 2), MIDDAY)
    assert snap3["SPY"]["orb_state"] == "inside"


def test_pre_freeze_omits_orb_fields_never_placeholders():
    # 09:45 ET with only 3 bars: VWAP computes, ORB is OMITTED (partial tape reads neutral).
    bars = _bars(3)
    snap, _ = tape.compute_tape({"SPY": bars}, _quotes(SPY=762.0), EARLY)
    assert "vwap" in snap["SPY"]
    for key in ("orb_high", "orb_low", "orb_state"):
        assert key not in snap["SPY"]
    assert "spy_orb_state" not in snap


def test_pre_open_quotes_only_snapshot_is_neutral():
    snap, _ = tape.compute_tape({}, _quotes(SPY=760.0, VIXY=20.0),
                                datetime(2026, 8, 5, 13, 20, tzinfo=UTC))
    assert snap["SPY"]["last"] == 760.0 and "above_vwap" not in snap["SPY"]
    assert "spy_above_vwap" not in snap and "gap_pct" not in snap
    assert snap["minutes_to_close"] > 390


# --- gap_pct session constant ----------------------------------------------------------------

def test_gap_pct_computed_once_and_carried_all_session():
    bars = _bars(6, base=760.0)
    quotes = _quotes(SPY=765.0)                                # previous_close 755.0
    snap, state = tape.compute_tape({"SPY": bars}, quotes, MIDDAY)
    expected = round((760.0 / 755.0 - 1.0) * 100.0, 4)
    assert snap["gap_pct"] == expected
    # Later tick loses the quote lane entirely: the session constant still rides the state.
    snap2, state = tape.compute_tape({"SPY": bars}, {}, MIDDAY + timedelta(minutes=30), state)
    assert snap2["gap_pct"] == expected
    # A NEW ET day never inherits yesterday's gap.
    next_day = MIDDAY + timedelta(days=1)
    snap3, _ = tape.compute_tape({}, {}, next_day, state)
    assert "gap_pct" not in snap3


# --- live-consumer contract ------------------------------------------------------------------

def _full_snapshot(now=MIDDAY, spy=770.0, qqq=700.0, iwm=300.0, vixy=19.0):
    bars = {"SPY": _bars(12, base=760.0), "QQQ": _bars(12, base=690.0),
            "IWM": _bars(12, base=290.0), "VIXY": _bars(12, base=20.5, step=-0.01, volume=500)}
    quotes = _quotes(SPY=spy, QQQ=qqq, IWM=iwm)
    quotes["VIXY"] = {"last": vixy, "previous_close": 20.4, "bid": vixy, "ask": vixy}
    return tape.compute_tape(bars, quotes, now)


def test_score_day_accepts_the_flat_keys():
    from data.odte_day_score import score_day
    snap, _ = _full_snapshot()
    day = score_day(market=snap)
    assert day.get("verdict"), day
    assert isinstance(day.get("components"), dict)


def test_candidate_watch_primitives_accept_the_blocks():
    from data.odte_candidate_watch import _above_vwap, _confirmation_counts, _spot, _vixy_weak
    snap, _ = _full_snapshot()
    assert _spot(snap, "SPY") == 770.0
    assert _above_vwap(snap, "SPY") is True
    assert _vixy_weak(snap) is True                            # VIXY below vwap / negative change
    n, confirmers, dissenters = _confirmation_counts(snap, "bullish")
    assert n >= 3 and not dissenters, (confirmers, dissenters)


def test_convert_preflight_accepts_tape_freshness_stamp():
    from data.odte_convert import _payload_ts
    snap, _ = _full_snapshot()
    assert _payload_ts(snap) is not None


# --- parsing of live payload shapes ----------------------------------------------------------

def test_parse_historicals_live_shape_and_malformed_bars():
    payload = {"data": {"results": [
        {"symbol": "SPY", "interval": "5minute", "bounds": "regular", "bars": [
            {"begins_at": "2026-08-05T13:30:00Z", "open_price": "760.620000",
             "close_price": "761.830000", "high_price": "761.880000",
             "low_price": "760.520000", "volume": 606724, "session": "reg"},
            {"begins_at": None, "open_price": "1"},            # malformed: dropped
        ]}]}, "guide": "prose"}
    bars = tape.parse_historicals(payload)
    assert len(bars["SPY"]) == 1
    assert bars["SPY"][0]["open"] == 760.62 and bars["SPY"][0]["volume"] == 606724


def test_parse_quotes_live_nested_shape():
    rows = [{"quote": {"symbol": "SPY", "last_trade_price": "770.100000",
                       "previous_close": "755.000000", "bid_price": "770.05",
                       "ask_price": "770.15"},
             "close": {"symbol": "SPY", "price": "769.00"}}]
    quotes = tape.parse_quotes(rows)
    assert quotes["SPY"]["last"] == 770.1 and quotes["SPY"]["previous_close"] == 755.0


# --- perf ------------------------------------------------------------------------------------

def test_compute_tape_full_day_under_50ms():
    import time
    bars = {s: _bars(78) for s in tape.TAPE_SYMBOLS}
    quotes = _quotes(SPY=770.0, QQQ=700.0, IWM=300.0, VIXY=19.0)
    tape.compute_tape(bars, quotes, MIDDAY)                    # warm
    t0 = time.perf_counter()
    tape.compute_tape(bars, quotes, MIDDAY)
    assert (time.perf_counter() - t0) * 1000.0 < 50


# --- transport orchestration -----------------------------------------------------------------

def _historicals_payload():
    def bars_json(base):
        out = []
        for i in range(12):
            px = base + i * 0.5
            ts = (SESSION_OPEN + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z")
            out.append({"begins_at": ts, "open_price": str(px), "high_price": str(px + 1),
                        "low_price": str(px - 1), "close_price": str(px + 0.5),
                        "volume": 1000, "session": "reg"})
        return out
    return {"data": {"results": [{"symbol": s, "bars": bars_json(b)} for s, b in
                                 (("SPY", 760.0), ("QQQ", 690.0), ("IWM", 290.0),
                                  ("VIXY", 20.5))]}, "guide": "prose"}


def _quotes_payload():
    return {"data": {"results": [
        {"quote": {"symbol": s, "last_trade_price": str(px), "previous_close": str(pc)}}
        for s, px, pc in (("SPY", 770.0, 755.0), ("QQQ", 700.0, 695.0),
                          ("IWM", 300.0, 298.0), ("VIXY", 19.0, 20.4))]}, "guide": "prose"}


def test_build_tape_fetches_bars_once_per_boundary(tmp_path):
    import json
    import time as _time

    from fakes.fake_mcp_session import FakeMcpSession

    import execution.odte_mcp_client as mc
    session = FakeMcpSession()
    session.queue("get_equity_historicals", _historicals_payload())
    session.queue("get_equity_quotes", _quotes_payload(), _quotes_payload())

    token = tmp_path / "token.json"
    token.write_text(json.dumps({"access_token": "fake", "expires_at": _time.time() + 999999}))

    async def factory():
        return session

    client = mc.OdteMcpClient(token_path=str(token), session_factory=factory)

    async def scenario():
        snap1, state = await tape.build_tape(client, None, MIDDAY)
        snap2, state = await tape.build_tape(client, state, MIDDAY + timedelta(seconds=4))
        return snap1, snap2
    snap1, snap2 = asyncio.run(scenario())
    assert len(session.calls_of("get_equity_historicals")) == 1    # same boundary: bars cached
    assert len(session.calls_of("get_equity_quotes")) == 2         # quotes every tick
    args = session.calls_of("get_equity_historicals")[0]["arguments"]
    assert args["start_time"] == "2026-08-05T13:30:00Z"            # today's 09:30 ET, in UTC
    assert "span" not in args
    assert snap1["SPY"]["above_vwap"] is True and snap2["spy_orb_state"] == "above"
    assert snap1["gap_pct"] == snap2["gap_pct"]
