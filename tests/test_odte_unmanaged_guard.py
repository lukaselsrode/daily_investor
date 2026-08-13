"""tests/test_odte_unmanaged_guard.py — the position-unmanaged alarm (local/offline).

Guards the one condition nothing else in the system notices: size is on and the only job that can
exit it has stopped ticking. Pure unit tests over `evaluate` — no IO, no clock, no broker.
"""
import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_SPEC = importlib.util.spec_from_file_location(
    "odte_unmanaged_guard",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "odte_unmanaged_guard.py"))
guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guard)

NOW = datetime(2026, 8, 12, 16, 20, tzinfo=timezone.utc)


def _trade(**over):
    t = {"status": "open", "underlying": "IWM", "strike_price": 301.0, "option_type": "put",
         "updated_at": (NOW - timedelta(seconds=30)).isoformat()}
    t.update(over)
    return t


def _position(age_seconds=30.0):
    return {"updated_at": (NOW - timedelta(seconds=age_seconds)).isoformat(),
            "decision": "HOLD", "active": True}


def test_silent_while_the_position_is_being_managed():
    assert guard.evaluate(_trade(), _position(30), now=NOW) is None


def test_silent_when_flat():
    assert guard.evaluate(_trade(status="closed"), _position(9999), now=NOW) is None
    assert guard.evaluate(None, None, now=NOW) is None


def test_alarms_once_management_goes_stale():
    line = guard.evaluate(_trade(), _position(guard.STALE_AFTER_SECONDS + 60), now=NOW)
    assert line and "POSITION UNMANAGED" in line
    assert "IWM 301.0P" in line
    assert "0dte-live-controller" in line          # names the only job that can exit


def test_boundary_is_not_an_alarm():
    assert guard.evaluate(_trade(), _position(guard.STALE_AFTER_SECONDS), now=NOW) is None
    assert guard.evaluate(_trade(), _position(guard.STALE_AFTER_SECONDS + 0.5), now=NOW)


def test_open_position_with_no_readable_timestamp_alarms():
    """The dangerous ambiguity: we cannot show it is being managed, so we must not stay quiet."""
    line = guard.evaluate(_trade(updated_at=None), {"updated_at": "not-a-date"}, now=NOW)
    assert line and "no readable management timestamp" in line


def test_falls_back_to_the_plan_timestamp():
    trade = _trade(updated_at=(NOW - timedelta(seconds=30)).isoformat())
    assert guard.evaluate(trade, {}, now=NOW) is None
    stale = _trade(updated_at=(NOW - timedelta(seconds=900)).isoformat())
    assert guard.evaluate(stale, {}, now=NOW)


def test_the_16_minute_starvation_window_would_have_alarmed():
    """2026-08-12: the controller missed three free slots while another lane held the gateway."""
    assert guard.evaluate(_trade(), _position(16 * 60), now=NOW)


def test_main_is_silent_and_exit_zero_when_state_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(guard, "ODTE", tmp_path)
    assert guard.main([]) == 0
    assert capsys.readouterr().out == ""


# --- missed-lease alarm (2026-08-13) -----------------------------------------------------------
# The only setup of the day reached a lease and lost it: 4235f2af issued 11:03:36 for SPY 778C,
# expired 11:04:36, never consumed, no order. Nothing alerted. The controller's own Telegram said
# "converted successfully: stage=authorize, no refusal codes" — true and completely misleading.

def _lease(expires, lease_id="4235f2af929d0a25", **over):
    d = {"lease": {"lease_id": lease_id, "underlying": "SPY", "strike_price": 778.0,
                   "option_type": "call", "expires_at": expires.isoformat()}}
    d["lease"].update(over)
    return d


def test_alarms_on_a_lease_that_expired_without_an_order():
    line = guard.evaluate_lease(_lease(NOW - timedelta(seconds=30)), [], now=NOW)
    assert line and "LEASE MISSED" in line
    assert "SPY" in line and "778" in line
    assert "latency" in line.lower()          # names the cause, not just the symptom


def test_silent_when_the_lease_was_consumed():
    """It became an order — that is the happy path, not an alarm."""
    lid = "4235f2af929d0a25"
    assert guard.evaluate_lease(_lease(NOW - timedelta(seconds=30), lease_id=lid), [lid],
                                now=NOW) is None


def test_silent_before_the_lease_has_lapsed():
    assert guard.evaluate_lease(_lease(NOW + timedelta(seconds=20)), [], now=NOW) is None


def test_stops_repeating_once_the_window_passes():
    """The pulse runs every minute; a stateless window keeps it to one or two lines, not forever."""
    fresh = guard.evaluate_lease(_lease(NOW - timedelta(seconds=60)), [], now=NOW)
    stale = guard.evaluate_lease(
        _lease(NOW - timedelta(seconds=guard.LEASE_ALERT_WINDOW_SECONDS + 30)), [], now=NOW)
    assert fresh and stale is None


def test_undated_lease_is_not_an_alarm():
    """Absent expires_at means we cannot show it lapsed; the position guard is the loud one."""
    d = {"lease": {"lease_id": "x", "underlying": "SPY"}}
    assert guard.evaluate_lease(d, [], now=NOW) is None
