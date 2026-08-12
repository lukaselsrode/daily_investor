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
