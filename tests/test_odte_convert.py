"""tests/test_odte_convert.py — atomic CONFIRM_ENTRY→gate→lease conversion (2026-08-02 retune).

Pure/offline: no broker, network, LLM, or orders. Pins the conversion contract that fixes the
2026-07-27..31 zero-trade week:
  * one call, one clock: fresh artifacts convert to a minted lease in a single process;
  * the 2026-07-31 replay (snapshots ~90s old at authorize) now converts under the widened
    snapshot TTL — and still fails closed beyond it;
  * every non-converting stage journals an IDENTITY-BOUND terminal no_trade_decision (no silent
    scan-only dead ends);
  * a journal-append failure withholds the lease (fail closed);
  * the final confirmations are COMPUTED from the artifacts, never asserted.

Thresholds/TTLs come from the live modules — never re-hardcoded.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_config as oc
import data.odte_convert as cv
import data.odte_journal as oj

NOW = datetime(2026, 7, 31, 15, 35, 3, tzinfo=timezone.utc)   # the real failed-conversion moment


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _market(now: datetime = NOW, age_seconds: float = 3.0, **over) -> dict:
    m = {
        "as_of": _iso(now - timedelta(seconds=age_seconds)),
        "minutes_to_close": 240, "vix": 18.0, "gap_pct": 0.5, "expected_move_pct": 0.8,
        # day-score flat keys
        "spy_above_vwap": True, "spy_orb_state": "above",
        "qqq_above_vwap": True, "qqq_orb_state": "above",
        "iwm_above_vwap": True, "iwm_orb_state": "above",
        "vixy_above_vwap": False, "vixy_change_pct": -2.0,
        # candidate-watch block keys
        "SPY": {"last": 741.0, "above_vwap": True, "orb_state": "above"},
        "QQQ": {"last": 724.2, "above_vwap": True, "orb_state": "above"},
        "IWM": {"last": 299.0, "above_vwap": True, "orb_state": "above"},
        "XSP": {"last": 741.0, "above_vwap": True, "orb_state": "above"},
        "VIXY": {"above_vwap": False, "change_pct": -2.0},
    }
    m.update(over)
    return m


def _contract(now: datetime = NOW, age_seconds: float = 2.0, **over) -> dict:
    c = {
        "as_of": _iso(now - timedelta(seconds=age_seconds)),
        "underlying": "QQQ", "option_type": "call", "option_id": "QQQ260731C00725000",
        "expiration_date": "2026-07-31", "strike_price": 725.0,
        "bid": 1.15, "ask": 1.19, "mark": 1.17, "volume": 5000, "open_interest": 8000,
        "direction": "bullish",
    }
    c.update(over)
    return c


def _broker(now: datetime = NOW, age_seconds: float = 2.0, **over) -> dict:
    b = {
        "as_of": _iso(now - timedelta(seconds=age_seconds)),
        "buying_power": 348.16, "day_trades_left": 3,
        "nonzero_option_positions_count": 0, "open_option_orders_count": 0,
        "today_option_orders_count": 0,
    }
    b.update(over)
    return b


def _candidate(now: datetime = NOW) -> dict:
    return {"ticker": "QQQ", "direction": "bullish",
            "created_at": _iso(now - timedelta(minutes=5))}


def _convert(tmp_path, *, now: datetime = NOW, market=None, contract=None, broker=None,
             candidate=None, journal=True, journal_path=None, **kw):
    return cv.run_convert(
        candidate_json=json.dumps(candidate or _candidate(now)),
        market_json=json.dumps(market or _market(now)),
        broker_json=json.dumps(broker or _broker(now)),
        contract_json=json.dumps(contract or _contract(now)),
        journal_path=journal_path or str(tmp_path / "decision_journal.jsonl"),
        state_dir=str(tmp_path), journal=journal, now=now, **kw)


def test_one_call_converts_fresh_package_to_lease(tmp_path):
    payload = _convert(tmp_path)
    assert payload["converted"] is True, payload["reason_codes"]
    assert payload["stage"] == "authorize"
    lease = payload["lease"]
    assert lease["symbol"] == "QQQ" and lease["option_id"] == "QQQ260731C00725000"
    assert lease["anchor_quote"] == 1.19
    assert lease["max_limit_price"] == round(1.19 * (1 + oc.CHASE_BAND_FRACTION), 2)
    # computed confirmations, not asserted booleans
    assert payload["confirmations"] == {"live_chain_recheck": True, "spread_cap_check": True,
                                        "budget_check": True}
    # artifacts persisted through the canonical writers
    assert (tmp_path / "candidate_decision.json").exists()
    assert (tmp_path / "active_candidate.json").exists()
    assert (tmp_path / "execution_lease.json").exists()
    # journal carries the gate decision AND the lease issuance
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    types = [e.get("event_type") for e in events]
    assert "entry_decision" in types
    lease_events = [e for e in events if e.get("event_type") == "execution_lease_issued"]
    assert lease_events and lease_events[-1].get("authorized") is True


def test_jul31_replay_snapshots_90s_old_now_convert(tmp_path):
    # REPLAY: on 2026-07-31 the gate passed and authorize refused on snapshot staleness. Snapshots
    # older than the old 60s bound but inside the widened TTL must now convert.
    assert oc.SNAPSHOT_TTL_SECONDS > 90 > 60
    payload = _convert(tmp_path, market=_market(age_seconds=90.0), broker=_broker(age_seconds=90.0))
    assert payload["converted"] is True, payload["reason_codes"]


def test_snapshot_beyond_ttl_refuses_at_preflight_and_journals_terminally(tmp_path):
    stale = _market(age_seconds=oc.SNAPSHOT_TTL_SECONDS + 1)
    payload = _convert(tmp_path, market=stale)
    assert payload["converted"] is False
    assert payload["stage"] == "preflight"
    assert "market_snapshot_stale" in payload["reason_codes"]
    assert (tmp_path / "execution_lease.json").exists() is False
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    terminal = [e for e in events if e.get("event_type") == "no_trade_decision"]
    assert terminal and terminal[-1].get("stage") == "preflight"


def test_degraded_candidate_watch_journals_identity_bound_no_trade(tmp_path):
    # VIX 40 → day score hard AVOID → candidate watch degrades; the terminal event must carry the
    # candidate identity so loop-status conversion accountability resolves.
    payload = _convert(tmp_path, market=_market(vix=40.0))
    assert payload["converted"] is False
    assert payload["stage"] == "candidate_watch"
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    terminal = [e for e in events if e.get("event_type") == "no_trade_decision"]
    assert terminal
    last = terminal[-1]
    assert last.get("symbol") == "QQQ"
    assert last.get("candidate_fingerprint")


def test_journal_append_failure_withholds_lease(tmp_path):
    # journal_path pointing at a DIRECTORY makes the append fail → the gate cannot be recorded →
    # no lease is minted (fail closed), even though every input would otherwise authorize.
    bad_journal = tmp_path / "journal_dir"
    bad_journal.mkdir()
    payload = _convert(tmp_path, journal_path=str(bad_journal))
    assert payload["converted"] is False
    assert payload["stage"] == "journal"
    assert "journal_append_failed_execution_withheld" in payload["reason_codes"]
    assert (tmp_path / "execution_lease.json").exists() is False


def test_no_journal_mode_still_converts_but_records_nothing(tmp_path):
    payload = _convert(tmp_path, journal=False)
    assert payload["converted"] is True
    assert not (tmp_path / "decision_journal.jsonl").exists()


def test_missing_snapshot_is_named_not_silently_defaulted(tmp_path):
    payload = cv.run_convert(candidate_json=json.dumps(_candidate()),
                             market_json=json.dumps(_market()),
                             broker_json=json.dumps(_broker()),
                             journal_path=str(tmp_path / "decision_journal.jsonl"),
                             state_dir=str(tmp_path), now=NOW)
    assert payload["converted"] is False
    assert payload["stage"] == "preflight"
    assert "contract_quote_missing" in payload["reason_codes"]


def test_convert_places_no_orders_and_module_is_offline():
    import inspect
    src = inspect.getsource(cv)
    for forbidden in ("robin_stocks", "requests", "place_order", "submit_order", "urllib",
                      "httpx", "socket"):
        assert forbidden not in src, f"odte_convert must not reference {forbidden!r}"
