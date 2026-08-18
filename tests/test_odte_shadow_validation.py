"""Shadow validation of the freshness entry spec (2026-08-18 halt) — detection, tracking,
price-path recording, and the fail-open contract. Evidence consumer:
data/odte/reports/entry_signal_autopsy_2026-08-18.md."""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import execution.odte_shadow_validation as sv

NOW = datetime(2026, 8, 19, 15, 0, 0, tzinfo=timezone.utc)


def _confirm_pair(t, *, vetoed=True, option_id="opt-1", direction="bearish", tier="a_plus"):
    ts = t.isoformat(timespec="seconds")
    events = [
        {"event_type": "entry_decision", "ts": ts, "symbol": "SPY", "tier": tier,
         "reasons": ["day_regime:ok", "confirmed_candidate_transition"]},
        {"event_type": "vehicle_score", "ts": ts, "underlying": "SPY", "option_id": option_id,
         "strike": 640.0, "option_type": "put", "vehicle_score": {"direction": direction}},
    ]
    if vetoed:
        events.append({"event_type": "no_trade_decision", "ts": ts,
                       "reason_codes": ["gate:veto", "daily_trade_budget_exhausted"]})
    return events


def _signal(t, direction="bearish"):
    return {"event_type": "candidate_evaluation", "ts": t.isoformat(timespec="seconds"),
            "symbol": "SPY", "direction": direction}


def test_detects_vetoed_confirm_with_signal_age():
    events = [_signal(NOW - timedelta(minutes=12))] + _confirm_pair(NOW)
    hits = sv.find_qualifying_confirms(events, NOW)
    assert len(hits) == 1
    h = hits[0]
    assert h["option_id"] == "opt-1" and h["tier"] == "a_plus"
    assert h["signal_age_minutes"] == 12.0


def test_real_entry_is_not_a_shadow_candidate():
    # No budget veto (an actual fill followed) -> not a would-be entry.
    events = _confirm_pair(NOW, vetoed=False)
    assert sv.find_qualifying_confirms(events, NOW) == []


def _tracker(tmp_path, **kw):
    kw.setdefault("hold_minutes", 45.0)
    kw.setdefault("poll_seconds", 0.0)      # poll every step in tests
    return sv.ShadowValidationTracker(tmp_path / "shadow_validation", **kw)


def test_tracks_polls_and_finalizes_the_price_path(tmp_path):
    tr = _tracker(tmp_path)
    events = [_signal(NOW - timedelta(minutes=5))] + _confirm_pair(NOW)
    quotes = iter([{"ask_price": 0.80, "bid_price": 0.78},
                   {"ask_price": 0.95, "bid_price": 0.92},     # MFE sample
                   {"ask_price": 0.60, "bid_price": 0.55}])    # MAE + final sample

    async def quote_fn(_oid):
        return next(quotes)

    for i in range(3):
        asyncio.run(tr.step(events, quote_fn, NOW + timedelta(minutes=i)))
    assert len(tr.tracking) == 1
    rec = next(iter(tr.tracking.values()))
    assert rec["ref_ask"] == 0.80 and len(rec["samples"]) == 3

    async def dead(_oid):                                      # hold expiry needs no quote
        raise AssertionError("no poll after expiry")
    asyncio.run(tr.step(events, dead, NOW + timedelta(minutes=46)))
    assert tr.tracking == {}
    lines = [json.loads(line) for line in
             (tmp_path / "shadow_validation" / "outcomes.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    o = lines[0]
    assert o["final_reason"] == "hold_expired" and o["n_samples"] == 3
    assert o["mfe_pct"] == round(0.92 / 0.80 - 1, 4)
    assert o["mae_pct"] == round(0.55 / 0.80 - 1, 4)
    assert o["final_pnl_pct"] == round(0.55 / 0.80 - 1, 4)
    assert o["signal_age_minutes"] == 5.0


def test_same_confirm_is_never_tracked_twice(tmp_path):
    tr = _tracker(tmp_path)
    events = _confirm_pair(NOW)

    async def quote_fn(_oid):
        return {"ask_price": 0.80, "bid_price": 0.78}
    asyncio.run(tr.step(events, quote_fn, NOW))
    asyncio.run(tr.step(events, quote_fn, NOW + timedelta(seconds=30)))
    assert len(tr.tracking) == 1


def test_tracking_survives_restart(tmp_path):
    tr = _tracker(tmp_path)

    async def quote_fn(_oid):
        return {"ask_price": 0.80, "bid_price": 0.78}
    asyncio.run(tr.step(_confirm_pair(NOW), quote_fn, NOW))
    reborn = _tracker(tmp_path)                                 # fresh instance, same dir
    assert len(reborn.tracking) == 1


def test_step_swallows_quote_failures(tmp_path):
    tr = _tracker(tmp_path)

    async def broken(_oid):
        raise RuntimeError("MCP down")
    asyncio.run(tr.step(_confirm_pair(NOW), broken, NOW))
    assert tr.errors == 1                                       # counted, never raised


def test_daemon_wires_the_tracker_behind_the_flag(tmp_path, monkeypatch):
    import data.odte_config as oc
    import execution.odte_fast_lane as fl

    class _Client:                                              # bare double; never called
        pass
    (tmp_path / "fast_lane_stage.json").write_text(json.dumps({"stage": "shadow"}))
    monkeypatch.setattr(oc, "SHADOW_VALIDATION_ENABLED", True)
    d = fl.FastLaneDaemon(_Client(), account_number="435050133", state_dir=tmp_path)
    assert d.shadow_validation is not None
    monkeypatch.setattr(oc, "SHADOW_VALIDATION_ENABLED", False)
    d2 = fl.FastLaneDaemon(_Client(), account_number="435050133", state_dir=tmp_path)
    assert d2.shadow_validation is None


@pytest.mark.halt_posture
def test_live_posture_arms_shadow_validation():
    """The halt's evidence engine must be ON while entries are halted — turning it off silently
    starves the un-halt decision of data. Update only with the resume decision."""
    import data.odte_config as oc
    assert oc.SHADOW_VALIDATION_ENABLED is True
