"""tests/test_odte_fast_lane.py — fast-lane daemon state transitions (unit level).

Pure/offline: FakeMcpSession + fake clock + tmp state dirs. Pins mode resolution (the stage
file caps any requested mode; a flag can only be MORE conservative), the off-hours/pause/
entries-locked inertness (proven structurally — the fake RAISES on any unscripted tool call),
shadow mode's zero-order-tool guarantee, convert-refusal cooldown, and the incident path that
locks entries for the session.
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from fakes.fake_mcp_session import FakeMcpSession

import execution.odte_fast_lane as fl
import execution.odte_mcp_client as mc

NOW = datetime(2026, 8, 5, 15, 1, 2, tzinfo=timezone.utc)      # 11:01 ET, Wednesday
SESSION_OPEN = datetime(2026, 8, 5, 13, 30, tzinfo=timezone.utc)


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def _historicals_payload():
    def bars(base):
        out = []
        for i in range(12):
            px = base + i * 0.5
            ts = (SESSION_OPEN + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z")
            out.append({"begins_at": ts, "open_price": str(px), "high_price": str(px + 1),
                        "low_price": str(px - 1), "close_price": str(px + 0.5),
                        "volume": 1000, "session": "reg"})
        return out
    return {"data": {"results": [{"symbol": s, "bars": bars(b)} for s, b in
                                 (("SPY", 760.0), ("QQQ", 690.0), ("IWM", 290.0),
                                  ("VIXY", 20.5))]}, "guide": ""}


def _quotes_payload(spy=770.0):
    rows = [("SPY", spy, 755.0), ("QQQ", 700.0, 695.0), ("IWM", 300.0, 298.0),
            ("VIXY", 19.0, 20.4)]
    return {"data": {"results": [
        {"quote": {"symbol": s, "last_trade_price": str(px), "previous_close": str(pc)}}
        for s, px, pc in rows]}, "guide": ""}


def _portfolio_payload(bp=357.06):
    return {"data": {"buying_power": {"buying_power": str(bp)}}, "guide": ""}


def _option_quote_payload(bid=0.61, ask=0.63):
    return {"data": {"results": [{"bid_price": str(bid), "ask_price": str(ask),
                                  "volume": 5000, "open_interest": 8000}]}, "guide": ""}


def _intent(now=NOW, **over):
    intent = {
        "intent_id": "spy-c-756-test", "schema_version": 1,
        "armed_at": _iso(now - timedelta(minutes=10)),
        "armed_by": "test", "expires_at": _iso(now + timedelta(hours=2)),
        "status": "armed", "symbol": "SPY", "direction": "bullish",
        "contract": {"option_id": "opt-756c", "chain_symbol": "SPY",
                     "expiration_date": "2026-08-05", "strike_price": 756.0,
                     "option_type": "call"},
        "trigger": {"level_acceptance": {"side": "above", "level": 755.60,
                                         "n_consecutive_checks": 1,
                                         "min_seconds_between_checks": 0},
                    "confirmations": {"min_confirmers": 3, "max_dissenters": 0},
                    "vixy_condition": "weak"},
        "exit_plan": {"mode": "scalp",
                      "profit_rules": {"take_profit_pct": 0.60, "strong_exit_pct": 0.80},
                      "risk_rules": {"initial_bid_floor": 0.30},
                      "time_rules": {"flat_before": "15:30"}},
    }
    intent.update(over)
    return intent


def _setup(tmp_path, *, stage="shadow", mode=None, intents=None):
    (tmp_path / fl.STAGE_FILENAME).write_text(json.dumps({"stage": stage}))
    if intents is not None:
        (tmp_path / "armed_intents.json").write_text(
            json.dumps({"schema_version": 1, "intents": intents}))
    token = tmp_path / "token.json"
    token.write_text(json.dumps({"access_token": "fake", "expires_at": time.time() + 999999}))
    session = FakeMcpSession()

    async def factory():
        return session
    client = mc.OdteMcpClient(token_path=str(token), session_factory=factory)
    daemon = fl.FastLaneDaemon(client, account_number="435050133", state_dir=tmp_path,
                               mode=mode)
    return daemon, session


def _tick(daemon, now=NOW):
    return asyncio.run(daemon.tick(now))


def _queue_tape(session, spy=770.0):
    session.queue("get_equity_historicals", _historicals_payload())
    session.queue("get_equity_quotes", _quotes_payload(spy=spy))


def _queue_startup(session):
    session.queue("get_option_orders", {"data": {"orders": []}, "guide": ""})


def _queue_fire_reads(session, bid=0.61, ask=0.63, bp=357.06):
    session.queue("get_option_quotes", _option_quote_payload(bid=bid, ask=ask))
    session.queue("get_portfolio", _portfolio_payload(bp=bp))
    session.queue("get_option_positions", {"data": {"positions": []}, "guide": ""})
    session.queue("get_option_orders", {"data": {"orders": []}, "guide": ""})


def _shadow_events(tmp_path):
    path = tmp_path / "shadow" / "decision_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- mode resolution -------------------------------------------------------------------------

def test_stage_file_caps_requested_mode(tmp_path):
    stage = tmp_path / fl.STAGE_FILENAME
    stage.write_text(json.dumps({"stage": "shadow"}))
    assert fl.resolve_mode(stage, "live") == "shadow"          # flag can't exceed the stage
    assert fl.resolve_mode(stage, None) == "shadow"
    stage.write_text(json.dumps({"stage": "entries_live"}))
    assert fl.resolve_mode(stage, None) == "live"
    assert fl.resolve_mode(stage, "shadow") == "shadow"        # more conservative always OK
    assert fl.resolve_mode(stage, "exits") == "exits"
    stage.write_text(json.dumps({"stage": "exits_live"}))
    assert fl.resolve_mode(stage, "live") == "exits"
    assert fl.resolve_mode(Path(tmp_path / "missing.json"), None) == "shadow"  # default: safest


def test_shadow_client_blocks_order_tools_structurally():
    class Stub:
        async def review_option_order(self, *a, **k):
            return {}

        async def equity_quotes(self, *a, **k):
            return []
    shadow = fl.ShadowClient(Stub())
    with pytest.raises(fl.ShadowOrderBlocked):
        _ = shadow.review_option_order
    with pytest.raises(fl.ShadowOrderBlocked):
        _ = shadow.place_option_order
    assert callable(shadow.equity_quotes)                      # reads pass through


# --- inert ticks -----------------------------------------------------------------------------

def test_off_hours_tick_makes_zero_broker_calls(tmp_path):
    daemon, session = _setup(tmp_path, stage="entries_live")
    night = datetime(2026, 8, 5, 6, 0, tzinfo=timezone.utc)    # 02:00 ET
    status = _tick(daemon, night)
    assert status["session_open"] is False
    assert session.calls == []                                 # unscripted call would raise
    assert (tmp_path / fl.STATUS_FILENAME).exists()            # heartbeat still written


def test_idle_with_no_intents(tmp_path):
    daemon, session = _setup(tmp_path, stage="shadow")
    _queue_startup(session)
    _queue_tape(session)
    status = _tick(daemon)
    assert status["state"] == fl.IDLE and status["intents_armed"] == 0
    heartbeat = json.loads((tmp_path / fl.STATUS_FILENAME).read_text())
    assert heartbeat["mode"] == "shadow" and heartbeat["account_masked"] == "****0133"


def test_pause_file_blocks_firing_but_keeps_watching(tmp_path):
    daemon, session = _setup(tmp_path, stage="entries_live", intents=[_intent()])
    (tmp_path / fl.PAUSE_FILENAME).touch()
    _queue_startup(session)
    _queue_tape(session)                                       # NO fire reads scripted
    status = _tick(daemon)
    assert status["paused"] is True and status["counts"]["fires"] == 0
    assert status["state"] == fl.WATCHING
    assert not session.calls_of("get_option_quotes")           # firing never started


def test_entries_locked_blocks_firing(tmp_path):
    daemon, session = _setup(tmp_path, stage="entries_live", intents=[_intent()])
    daemon.entries_locked = True
    daemon._started = True                                     # skip startup reconcile
    _queue_tape(session)
    status = _tick(daemon)
    assert status["counts"]["fires"] == 0 and status["entries_locked"] is True


# --- shadow firing ---------------------------------------------------------------------------

def test_shadow_fire_journals_everything_and_never_places(tmp_path):
    daemon, session = _setup(tmp_path, stage="shadow", intents=[_intent()])
    _queue_startup(session)
    _queue_tape(session)
    _queue_fire_reads(session)
    status = _tick(daemon)
    assert status["counts"]["fires"] == 1
    events = {e["event_type"] for e in _shadow_events(tmp_path)}
    assert {"shadow_trigger_fired", "shadow_convert_result",
            "shadow_order_intent"} <= events
    assert not session.calls_of("review_option_order")
    assert not session.calls_of("place_option_order")
    intent_evt = [e for e in _shadow_events(tmp_path)
                  if e["event_type"] == "shadow_order_intent"][0]
    assert intent_evt["order_args"]["legs"][0]["option_id"] == "opt-756c"
    assert intent_evt["lease"]["lease_id"]
    # The real journal is untouched — shadow can never pollute budget/green state.
    assert not (tmp_path / "decision_journal.jsonl").exists()
    # And the intent is single-shot: the next tick has nothing armed.
    _queue_tape(session)
    status2 = _tick(daemon, NOW + timedelta(seconds=4))
    assert status2["counts"]["fires"] == 1 and status2["intents_armed"] == 0


def test_convert_refusal_sets_refire_cooldown(tmp_path):
    # An unaffordable contract (debit above the tier cap) refuses at the gate; the intent parks
    # for REFIRE_COOLDOWN_SECONDS instead of hot-looping the still-true trigger.
    daemon, session = _setup(tmp_path, stage="shadow", intents=[_intent()])
    _queue_startup(session)
    _queue_tape(session)
    _queue_fire_reads(session, bid=2.48, ask=2.50, bp=300.0)   # debit 250 > 0.60*300
    status = _tick(daemon)
    assert status["counts"]["fires"] == 1 and status["counts"]["refusals"] == 1
    results = [e for e in _shadow_events(tmp_path) if e["event_type"] == "shadow_convert_result"]
    assert results and results[0]["converted"] is False
    assert not any(e["event_type"] == "shadow_order_intent" for e in _shadow_events(tmp_path))
    # Inside the cooldown: armed count is zero, nothing re-fires.
    _queue_tape(session)
    status2 = _tick(daemon, NOW + timedelta(seconds=10))
    assert status2["counts"]["fires"] == 1 and status2["intents_armed"] == 0
    # After the cooldown the intent is armed again (the tape may have changed).
    _queue_tape(session)
    _queue_fire_reads(session, bid=2.48, ask=2.50, bp=300.0)
    status3 = _tick(daemon, NOW + timedelta(seconds=fl.REFIRE_COOLDOWN_SECONDS + 11))
    assert status3["counts"]["fires"] == 2


# --- incident path ---------------------------------------------------------------------------

def test_fill_after_lease_expiry_is_incident_and_locks_entries(tmp_path):
    daemon, session = _setup(tmp_path, stage="entries_live", intents=[_intent()])
    daemon._started = True
    daemon.state = fl.PENDING
    daemon.lease = {"lease_id": "lease-x", "symbol": "SPY", "option_id": "opt-756c",
                    "option_type": "call", "strike_price": 756.0,
                    "expiration_date": "2026-08-05", "quantity": 1,
                    "max_limit_price": 0.73, "max_debit": 130.0,
                    "issued_at": _iso(NOW - timedelta(seconds=90)),
                    "expires_at": _iso(NOW - timedelta(seconds=30))}
    daemon.pending_order = {"order_id": "ord-1", "ref_id": "r", "limit_price": 0.63,
                            "placed_at": _iso(NOW - timedelta(seconds=85))}
    _queue_tape(session)
    session.queue("get_option_orders",
                  {"data": {"orders": [{"id": "ord-1", "state": "filled",
                                        "chain_symbol": "SPY", "price": "0.63",
                                        "quantity": "1", "processed_quantity": "1",
                                        "created_at": _iso(NOW - timedelta(seconds=85)),
                                        "last_transaction_at": _iso(NOW - timedelta(seconds=5)),
                                        "legs": [{"option_id": "opt-756c"}]}]}, "guide": ""})
    status = _tick(daemon)
    assert daemon.entries_locked is True
    assert status["counts"]["incidents"] == 1
    events = [json.loads(line) for line in
              (tmp_path / "decision_journal.jsonl").read_text().splitlines()]
    assert any(e["event_type"] == "execution_safety_incident" for e in events)
    assert daemon.state == fl.MANAGING                         # the fill is real: manage it
    assert json.loads((tmp_path / "active_trade.json").read_text())["incident_entry"] is True


def test_startup_adopts_open_plan_into_managing(tmp_path):
    daemon, session = _setup(tmp_path, stage="shadow")
    (tmp_path / "active_trade.json").write_text(json.dumps(
        {"status": "open", "underlying": "SPY", "option_id": "opt-756c",
         "entry_price": 0.61, "quantity": 1, "mode": "scalp"}))
    _queue_tape(session)
    session.queue("get_option_quotes", _option_quote_payload(bid=0.62, ask=0.64))
    status = _tick(daemon)
    assert status["state"] == fl.MANAGING


def test_startup_unexplained_order_locks_entries(tmp_path):
    daemon, session = _setup(tmp_path, stage="shadow")
    session.queue("get_option_orders",
                  {"data": {"orders": [{"id": "mystery", "state": "queued",
                                        "chain_symbol": "QQQ", "price": "1.00",
                                        "quantity": "1",
                                        "legs": [{"option_id": "opt-unknown"}]}]},
                   "guide": ""})
    _queue_tape(session)
    status = _tick(daemon)
    assert daemon.entries_locked is True and status["counts"]["incidents"] == 1
    assert any(e["event_type"] == "shadow_incident" for e in _shadow_events(tmp_path))


# --- 2026-08-07: shadow must not mutate the controller's shared position plan ------------------

def test_shadow_writes_the_plan_to_the_shadow_dir_not_the_shared_one(tmp_path):
    """`MODE_SHADOW` wraps the broker client so no order can escape, but the plan WRITES were
    ungated by mode. A shadow daemon started while the CONTROLLER holds a position would read the
    controller's active_trade.json, recompute, and write its own version back — clobbering `thesis`
    (the pre-entry invalidation) and `management.best_seen_bid` (the whole state of the bid-memory
    rail). Wrapping the broker is not enough when two lanes share a state file.
    """
    daemon, session = _setup(tmp_path, stage="shadow")
    canonical = tmp_path / "active_trade.json"
    controller_plan = {"status": "open", "underlying": "SPY", "option_id": "opt-756c",
                       "entry_price": 0.61, "quantity": 1, "mode": "scalp",
                       "thesis": {"underlying_stop": 755.0},
                       "management": {"best_seen_bid": 0.80}}
    canonical.write_text(json.dumps(controller_plan))

    # reads still come from the CANONICAL plan — observing the real position is the point
    assert daemon._plan_path() == canonical
    assert daemon._plan_write_path() != canonical
    assert daemon._plan_write_path().parent.name == fl.SHADOW_DIRNAME

    _queue_tape(session)
    session.queue("get_option_quotes", _option_quote_payload(bid=0.62, ask=0.64))
    _tick(daemon)

    # the controller's plan is byte-identical: thesis and bid-memory survived
    assert json.loads(canonical.read_text()) == controller_plan


def test_live_mode_still_owns_the_canonical_plan(tmp_path):
    """In `entries_live` the fast lane IS the owner, so it writes the real plan — otherwise the
    controller and the guard would manage a position the daemon alone knows about."""
    daemon, _ = _setup(tmp_path, stage="entries_live", mode="live")
    assert daemon.mode == fl.MODE_LIVE
    assert daemon._plan_write_path() == daemon._plan_path() == tmp_path / "active_trade.json"


def test_exits_mode_writes_the_canonical_plan(tmp_path):
    """REVERSED 2026-08-19 on live evidence. At `exits` this lane OWNS exit management, but its
    `management` updates were redirected to the shadow copy — so the canonical plan looked
    unmanaged: the watchdog guard alarmed on an actively managed position and the controller
    (told to defer to `active_trade.json.management.decision`) could never see a decision.
    Only SHADOW redirects; the 2026-08-07 shadow-purity invariant stands."""
    daemon, _ = _setup(tmp_path, stage="exits_live", mode="exits")
    assert daemon.mode == fl.MODE_EXITS
    assert daemon._plan_write_path() == daemon._plan_path() == tmp_path / "active_trade.json"


# --- adopting a position this lane did not open (2026-08-13) ------------------------------------
# `_startup_reconcile` runs once, so a position the CONTROLLER opens mid-session was invisible
# forever: from IDLE the tick only ran `_watch`, which reads armed intents and nothing else.
# Measured live — the controller held IWM 303P from 11:54:29 to 12:07:07 while the daemon sat IDLE
# for 711 ticks and emitted ZERO exit intents. Gate 1 wants `shadow_exit_intent` on "any live
# Hermes position" and so could never have been satisfied; worse, at `exits_live` Hermes DEFERS
# exits to this lane, against a position it cannot see.

def _controller_plan(**over):
    plan = {"status": "open", "underlying": "IWM", "option_id": "opt-iwm-303p",
            "option_type": "put", "strike_price": 303.0, "entry_price": 0.46, "quantity": 1,
            "trade_id": "iwm-20260813-303p-scalp", "mode": "scalp",
            "thesis": {"underlying_stop": 304.0}, "time_rules": {}}
    plan.update(over)
    return plan


def test_daemon_adopts_a_position_the_controller_opened_mid_session(tmp_path):
    daemon, session = _setup(tmp_path)
    _queue_startup(session)
    _queue_tape(session)
    _tick(daemon)                                   # startup: flat -> IDLE
    assert daemon.state == fl.IDLE

    # the controller opens a position AFTER startup
    (tmp_path / fl.DEFAULT_PLAN_FILENAME).write_text(json.dumps(_controller_plan()))
    _queue_tape(session)
    _tick(daemon)
    assert daemon.state == fl.MANAGING, "daemon stayed blind to a controller-opened position"


def test_adoption_is_journaled_so_the_handover_is_auditable(tmp_path):
    daemon, session = _setup(tmp_path)
    _queue_startup(session)
    _queue_tape(session)
    _tick(daemon)
    (tmp_path / fl.DEFAULT_PLAN_FILENAME).write_text(json.dumps(_controller_plan()))
    _queue_tape(session)
    _tick(daemon)
    rows = [json.loads(x) for x in
            (tmp_path / "shadow" / "decision_journal.jsonl").read_text().splitlines() if x.strip()]
    adopted = [e for e in rows if e.get("event_type") == "shadow_position_adopted"]
    assert len(adopted) == 1
    assert adopted[0]["trade_id"] == "iwm-20260813-303p-scalp"
    assert adopted[0]["opened_by"] == "controller"


def test_a_closed_plan_is_never_adopted(tmp_path):
    daemon, session = _setup(tmp_path)
    _queue_startup(session)
    _queue_tape(session)
    _tick(daemon)
    (tmp_path / fl.DEFAULT_PLAN_FILENAME).write_text(
        json.dumps(_controller_plan(status="closed")))
    _queue_tape(session)
    _tick(daemon)
    assert daemon.state == fl.IDLE


def test_entries_locked_blocks_adoption(tmp_path):
    """An unexplained working order means we do not understand the account; managing into that is
    how a misunderstanding becomes an order."""
    daemon, session = _setup(tmp_path)
    _queue_startup(session)
    _queue_tape(session)
    _tick(daemon)
    daemon.entries_locked = True
    (tmp_path / fl.DEFAULT_PLAN_FILENAME).write_text(json.dumps(_controller_plan()))
    _queue_tape(session)
    _tick(daemon)
    assert daemon.state == fl.IDLE


# --- a rejected/dead EXIT order stands down and is journaled (2026-08-13) -----------------------
# _poll_exit compared `status == "gone"` but _order_row_to_guard keeps the RAW broker state
# ("rejected"/"cancelled"), so the branch NEVER matched — a dead exit order fell through into the
# reprice loop. A rejected exit on a live position is exactly what an operator must hear about.

def _rejected_exit_row(state="rejected"):
    return {"data": {"orders": [{
        "id": "exit-ord-1", "chain_symbol": "IWM", "state": state, "quantity": "1.00000",
        "processed_quantity": "0.00000", "price": "0.30000000",
        "created_at": "2026-08-13T15:00:00Z",
        "legs": [{"option_id": "opt-iwm-303p", "side": "sell", "position_effect": "close"}]}]}}


def _managed_daemon(tmp_path, state="rejected"):
    daemon, session = _setup(tmp_path)
    _queue_startup(session)
    _queue_tape(session)
    _tick(daemon)
    daemon.exit_order = {"order_id": "exit-ord-1", "limit": 0.30, "trigger_type": "MAX_LOSS",
                         "placed_at": _iso(NOW), "replacements": 0}
    session.queue("get_option_orders", _rejected_exit_row(state))
    plan = {"trade_id": "iwm-t1", "underlying": "IWM", "option_id": "opt-iwm-303p",
            "entry_price": 0.46, "quantity": 1, "status": "open"}
    asyncio.run(daemon._poll_exit(plan, NOW, paused=False))
    return daemon, tmp_path


def test_rejected_exit_order_stands_down_and_journals(tmp_path):
    daemon, base = _managed_daemon(tmp_path, "rejected")
    assert daemon.exit_order is None, "dead exit order must stand down, not enter the reprice loop"
    rows = [json.loads(x) for x in
            (base / "shadow" / "decision_journal.jsonl").read_text().splitlines() if x.strip()]
    rejected = [e for e in rows if e.get("event_type") == "order_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["side"] == "exit" and rejected[0]["trade_id"] == "iwm-t1"


def test_cancelled_exit_order_stands_down_quietly(tmp_path):
    """Cancelled is churn (the reprice loop itself cancels) — stand down, no rejection event."""
    daemon, base = _managed_daemon(tmp_path, "cancelled")
    assert daemon.exit_order is None
    sj = base / "shadow" / "decision_journal.jsonl"
    rows = ([json.loads(x) for x in sj.read_text().splitlines() if x.strip()]
            if sj.exists() else [])          # nothing journaled at all is the passing case
    assert [e for e in rows if e.get("event_type") == "order_rejected"] == []


# --- phantom-position recheck (2026-08-19: daemon adopted 5s before the controller's exit
# filled and then managed the dead position on quote polls alone for 15 minutes) -----------------

def test_phantom_position_releases_on_broker_recheck(tmp_path):
    daemon, session = _setup(tmp_path, stage="exits_live", mode="exits")
    (tmp_path / "active_trade.json").write_text(json.dumps(
        {"status": "open", "underlying": "QQQ", "option_id": "opt-dead", "quantity": 1,
         "entry_price": 1.34, "mode": "scalp", "trade_id": "t-phantom"}))
    daemon.state = fl.MANAGING
    daemon._started = True
    daemon._manage_polls = fl.POSITION_RECHECK_TICKS - 1    # this poll triggers the recheck
    _queue_tape(session)
    session.queue("get_option_positions", {"data": {"positions": []}, "guide": ""})
    _tick(daemon)
    assert daemon.state == fl.IDLE
    assert "phantom_position_released" in {e["event_type"] for e in _shadow_events(tmp_path)}


def test_position_recheck_failure_keeps_managing(tmp_path):
    # Uncertainty is NOT evidence of absence: a failed positions read must keep managing.
    daemon, session = _setup(tmp_path, stage="exits_live", mode="exits")
    (tmp_path / "active_trade.json").write_text(json.dumps(
        {"status": "open", "underlying": "QQQ", "option_id": "opt-live", "quantity": 1,
         "entry_price": 1.34, "mode": "scalp", "trade_id": "t-live"}))
    daemon.state = fl.MANAGING
    daemon._started = True
    daemon._manage_polls = fl.POSITION_RECHECK_TICKS - 1
    _queue_tape(session)
    # No get_option_positions queued -> the fake session raises -> recheck fails open.
    session.queue("get_option_quotes", _option_quote_payload(bid=1.30, ask=1.36))
    _tick(daemon)
    assert daemon.state == fl.MANAGING


def test_intraday_position_is_not_a_phantom(tmp_path):
    # 2026-08-20 LIVE incident: Robinhood reports same-day holdings with settled quantity
    # "0.0000" and the real size in intraday_quantity — the recheck released an actively held
    # position mid-session. Settled+intraday is the true quantity.
    daemon, session = _setup(tmp_path, stage="exits_live", mode="exits")
    (tmp_path / "active_trade.json").write_text(json.dumps(
        {"status": "open", "underlying": "IWM", "option_id": "opt-intraday", "quantity": 1,
         "entry_price": 1.21, "mode": "scalp", "trade_id": "t-intraday"}))
    daemon.state = fl.MANAGING
    daemon._started = True
    daemon._manage_polls = fl.POSITION_RECHECK_TICKS - 1
    _queue_tape(session)
    session.queue("get_option_positions", {"data": {"results": [
        {"option_id": "opt-intraday", "quantity": "0.0000",
         "intraday_quantity": "1.0000", "type": "long"}]}, "guide": ""})
    session.queue("get_option_quotes", _option_quote_payload(bid=1.18, ask=1.24))
    _tick(daemon)
    assert daemon.state == fl.MANAGING                     # present via intraday quantity


def test_management_heartbeat_stamps_on_losing_trades(tmp_path):
    # 2026-08-20: management.decision only wrote when best_seen_bid IMPROVED — silent on losing
    # trades, so the controller saw `management: None` and exited itself (3 sessions, 0 daemon
    # exits). The verdict now stamps on a cadence regardless of direction.
    daemon, session = _setup(tmp_path, stage="exits_live", mode="exits")
    plan = {"status": "open", "underlying": "IWM", "option_id": "opt-hb", "quantity": 1,
            "entry_price": 1.21, "mode": "scalp", "trade_id": "t-hb",
            "profit_rules": {"take_profit_pct": 0.25},
            "risk_rules": {"initial_bid_floor": 0.30}}
    (tmp_path / "active_trade.json").write_text(json.dumps(plan))
    daemon.state = fl.MANAGING
    daemon._started = True
    _queue_tape(session)
    session.queue("get_option_quotes", _option_quote_payload(bid=1.10, ask=1.16))  # losing, no exit
    _tick(daemon)
    written = json.loads((tmp_path / "active_trade.json").read_text())
    mgmt = written.get("management") or {}
    assert mgmt.get("managed_by") == "fast_lane"
    assert mgmt.get("decision") is not None
    assert mgmt.get("stamped_at")


def test_released_plan_is_not_readopted_until_file_changes(tmp_path):
    daemon, session = _setup(tmp_path, stage="exits_live", mode="exits")
    plan_path = tmp_path / "active_trade.json"
    plan_path.write_text(json.dumps(
        {"status": "open", "underlying": "IWM", "option_id": "opt-dead2", "quantity": 1,
         "entry_price": 1.21, "mode": "scalp", "trade_id": "t-dead2"}))
    daemon.state = fl.MANAGING
    daemon._started = True
    daemon._manage_polls = fl.POSITION_RECHECK_TICKS - 1
    _queue_tape(session)
    session.queue("get_option_positions", {"data": {"results": []}, "guide": ""})
    _tick(daemon)
    assert daemon.state == fl.IDLE                        # released (broker flat)
    _queue_tape(session)
    _tick(daemon)                                          # same stale file: must NOT re-adopt
    assert daemon.state == fl.IDLE
    plan_path.write_text(plan_path.read_text())            # rewrite -> new mtime
    import os as _os
    _os.utime(plan_path, None)
    _queue_tape(session)
    session.queue("get_option_quotes", _option_quote_payload(bid=1.18, ask=1.24))
    _tick(daemon)
    assert daemon.state == fl.MANAGING                     # changed file may re-adopt
