"""tests/test_odte_service.py — the 0DTE UI data seam (headless, offline).

The service is where events get joined into trades, funnels and lease spans, so the joins are
pinned here rather than discovered in a chart. Everything runs WITHOUT a Streamlit runtime: the
components must never be the only way to exercise this code. No broker, network, or LLM.

The dedupe cases below are all real defects this system has produced: one fill journaled by two
lanes read as two trades, and one exit journaled by two lanes doubled every rail count.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ui.services import odte_service as svc


@pytest.fixture(autouse=True)
def _clear_cache():
    svc._MEM_CACHE.clear()
    yield
    svc._MEM_CACHE.clear()


def _journal(tmp_path, events) -> str:
    p = tmp_path / "decision_journal.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in events))
    return str(p)


def _modern_trade(lease_id="lease-1", option_id="QQQ260806C00720000", trade_id="qqq-1"):
    """A fully instrumented 2026-08-03+ trade, journaled by BOTH lanes like the real system."""
    return [
        {"event_type": "execution_lease_issued", "ts": "2026-08-06T15:24:48+00:00",
         "trade_date": "2026-08-06", "lease_id": lease_id, "authorized": True, "decision": "allow",
         "underlying": "QQQ", "tier": "b_plus", "option_id": option_id,
         "issued_at": "2026-08-06T15:24:48+00:00", "expires_at": "2026-08-06T15:25:48+00:00",
         "max_limit_price": 0.84, "max_debit": 84.61, "quantity": 1, "risk_mode": "risk_on"},
        # controller lane: carries trade_id + lease binding
        {"event_type": "entry_fill", "ts": "2026-08-06T15:25:07+00:00", "trade_date": "2026-08-06",
         "trade_id": trade_id, "underlying": "QQQ", "option_id": option_id, "fill_price": 0.62,
         "quantity": 1, "execution_lease_id": lease_id, "lease_valid_at_fill": True},
        # order-guard lane: same fill, no trade_id, carries the submit/fill timestamps
        {"event_type": "order_filled", "ts": "2026-08-06T15:25:12+00:00",
         "trade_date": "2026-08-06", "underlying": "QQQ", "option_id": option_id,
         "execution_lease_id": lease_id, "submitted_at": "2026-08-06T15:25:07+00:00",
         "filled_at": "2026-08-06T15:25:07.26+00:00"},
        {"event_type": "execution_lease_consumed", "ts": "2026-08-06T15:25:33+00:00",
         "trade_date": "2026-08-06", "lease_id": lease_id},
        {"event_type": "management_check", "ts": "2026-08-06T15:27:00+00:00",
         "trade_date": "2026-08-06", "trade_id": None, "pnl_pct": 13.28, "decision": "hold"},
        {"event_type": "management_check", "ts": "2026-08-06T15:30:00+00:00",
         "trade_date": "2026-08-06", "trade_id": None, "pnl_pct": 22.66,
         "decision": "take_profit"},
        {"event_type": "exit_fill", "ts": "2026-08-06T15:31:00+00:00", "trade_date": "2026-08-06",
         "trade_id": trade_id, "underlying": "QQQ", "option_id": option_id, "fill_price": 0.76,
         "rail_fired": "TAKE_PROFIT", "best_seen_bid": 0.76, "gross_pnl": 14.0,
         "estimated_net_pnl": 13.9},
        # exit journaled by the second lane too
        {"event_type": "order_closed", "ts": "2026-08-06T15:31:05+00:00",
         "trade_date": "2026-08-06", "trade_id": trade_id, "underlying": "QQQ",
         "option_id": option_id, "realized_pnl": 14.0, "rail_fired": "TAKE_PROFIT"},
        {"event_type": "postmortem", "ts": "2026-08-06T15:33:00+00:00", "trade_date": "2026-08-06",
         "trade_id": trade_id, "underlying": "QQQ", "process_quality": "good_process",
         "outcome_quality": "good_outcome", "failure_layer": "none",
         "entry": {"tier": "b_plus", "fill_price": 0.62},
         "exit": {"rail_fired": "TAKE_PROFIT"},
         "pnl": {"gross": 14.0, "estimated_net": 13.9},
         "excursion": {"mfe_dollars": 14.0, "mfe_pct": 23.0, "mae_dollars": -2.0,
                       "mae_pct": -3.0, "mfe_capture_pct": 100.0},
         "process_review": {"rule_violations": []}},
    ]


# --- ledger joins ------------------------------------------------------------------------------

def test_ledger_joins_one_trade_from_both_journaling_lanes(tmp_path):
    jp = _journal(tmp_path, _modern_trade())
    rows = svc.trade_ledger(jp)
    assert len(rows) == 1, "one fill journaled by two lanes is ONE trade"
    r = rows[0]
    assert r["underlying"] == "QQQ" and r["tier"] == "b_plus"
    assert r["lease_id"] == "lease-1" and r["lease_valid_at_fill"] is True
    assert r["rail_fired"] == "TAKE_PROFIT"
    assert r["realized_pnl"] == 14.0 and r["net_pnl"] == 13.9
    assert r["mfe_capture_pct"] == 100.0
    assert r["process_quality"] == "good_process" and r["failure_layer"] == "none"
    # exit_fill wins over order_closed for the exit: only it carries the price and the best bid
    assert r["exit_price"] == 0.76 and r["best_seen_bid"] == 0.76
    assert r["held_minutes"] == 5.9 and r["modern"] is True


def test_ledger_survives_a_missing_postmortem(tmp_path):
    events = [e for e in _modern_trade() if e["event_type"] != "postmortem"]
    r = svc.trade_ledger(_journal(tmp_path, events))[0]
    assert r["realized_pnl"] == 14.0            # still readable from the exit
    assert r["mfe_capture_pct"] is None         # never fabricated
    assert r["has_postmortem"] is False


def test_ledger_empty_journal_is_not_an_error(tmp_path):
    assert svc.trade_ledger(_journal(tmp_path, [])) == []


# --- latency -----------------------------------------------------------------------------------

def test_latency_merges_lanes_into_one_row_per_lease(tmp_path):
    rows = svc.latency_rows(_journal(tmp_path, _modern_trade()))
    assert len(rows) == 1, "two lanes journaling one entry must not plot as two trades"
    r = rows[0]
    assert r["trade_id"] == "qqq-1"             # from the controller lane
    assert r["lease_to_submit"] == 19.0         # from the guard lane
    assert r["submit_to_fill"] == 0.26
    assert r["lease_to_fill"] == 19.0
    assert r["lease_to_consume"] == 45.0


def test_latency_skips_fills_with_no_lease(tmp_path):
    # Pre-2026-08-03 fills carry no execution_lease_id; there is no leg to measure.
    jp = _journal(tmp_path, [{"event_type": "entry_fill", "ts": "2026-07-01T14:00:00+00:00",
                              "trade_id": "old", "option_id": "SPY-C"}])
    assert svc.latency_rows(jp) == []


# --- intra-trade series ------------------------------------------------------------------------

def test_intratrade_series_joins_by_timestamp_window(tmp_path):
    jp = _journal(tmp_path, _modern_trade())
    row = svc.trade_ledger(jp)[0]
    series = svc.intratrade_series(row, jp)
    # modern management_check events carry trade_id=None — the bracket is the only join available
    assert [p["pnl_pct"] for p in series] == [13.28, 22.66]
    assert series[-1]["decision"] == "take_profit"
    assert series[0]["minutes"] > 0


# --- funnel ------------------------------------------------------------------------------------

def test_funnel_rows_only_cover_days_with_events(tmp_path):
    from datetime import datetime, timezone
    events = _modern_trade() + [
        {"event_type": "watchdog_trigger", "ts": "2026-08-06T14:00:00+00:00", "alert": True},
        {"event_type": "no_trade_decision", "ts": "2026-08-06T16:00:00+00:00",
         "stage": "entry_gate", "reason_codes": ["budget_check:fail"]},
    ]
    jp = _journal(tmp_path, events)
    svc._MEM_CACHE.clear()
    out = svc._funnel(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    assert len(out["rows"]) == 1, "no invented flat rows for days with no events"
    row = out["rows"][0]
    assert row["fills"] == 1 and row["leases_issued"] == 1
    assert row["watchdog_alerts"] == 1
    assert out["by_stage"] == {"entry_gate": 1}
    assert ("entry_gate:budget_check:fail", 1) in out["pareto"]


# --- rails -------------------------------------------------------------------------------------

def test_rail_counts_are_per_trade_not_per_event(tmp_path):
    from datetime import datetime, timezone
    jp = _journal(tmp_path, _modern_trade())
    svc._MEM_CACHE.clear()
    # exit_fill AND order_closed both carry rail_fired for the same exit
    out = svc._rails(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    assert out["rail_counts"] == [("TAKE_PROFIT", 1)]


def test_lease_span_outcome_and_inferred_expiry(tmp_path):
    from datetime import datetime, timezone
    events = _modern_trade() + [
        {"event_type": "execution_lease_issued", "ts": "2026-08-06T15:12:02+00:00",
         "trade_date": "2026-08-06", "lease_id": "lease-dead", "authorized": True,
         "decision": "allow", "underlying": "QQQ", "tier": "b_plus"},
    ]
    jp = _journal(tmp_path, events)
    svc._MEM_CACHE.clear()
    out = svc._rails(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    by_id = {s["lease_id"]: s for s in out["spans"]}
    assert by_id["lease-1"]["outcome"] == "filled"
    assert by_id["lease-1"]["ttl_seconds"] == 60.0        # read, because expires_at was recorded
    # no fill, no consume, no expires_at -> presumed expired, and labelled as an inference
    assert by_id["lease-dead"]["outcome"] == "expired_inferred"
    assert by_id["lease-dead"]["ttl_seconds"] is None


def test_incident_carries_its_adjudication(tmp_path):
    from datetime import datetime, timezone
    events = [
        {"event_type": "execution_safety_incident", "ts": "2026-08-06T14:14:54+00:00",
         "trade_date": "2026-08-06", "event_id": "inc-1", "seq": 6001, "underlying": "IWM",
         "guard_state": "BROKER_MISMATCH_BLOCKED", "reason_codes": ["option_type_mismatch"]},
        {"event_type": "execution_safety_incident_adjudicated", "ts": "2026-08-06T18:00:00+00:00",
         "trade_date": "2026-08-06", "incident_event_id": "inc-1", "adjudicated_by": "human",
         "reason": "guard read a raw broker row; fixed in bbbca46"},
    ]
    jp = _journal(tmp_path, events)
    svc._MEM_CACHE.clear()
    out = svc._rails(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    inc = out["incidents"][0]
    assert inc["guard_state"] == "BROKER_MISMATCH_BLOCKED"
    assert inc["adjudication"]["adjudicated_by"] == "human"
    assert "bbbca46" in inc["adjudication"]["reason"]


# --- ORB near-miss evidence --------------------------------------------------------------------

def _evaluation(ts, orb_state, *, confirmations=2, decision="keep_watching", vixy_weak=True,
                above_vwap=True):
    return {"event_type": "candidate_evaluation", "ts": ts, "trade_date": ts[:10],
            "underlying": "SPY", "direction": "bullish", "decision": decision,
            "checks": {"underlying_above_vwap": above_vwap, "underlying_orb_state": orb_state,
                       "confirmations": confirmations, "confirmers": ["QQQ", "IWM"],
                       "dissenters": [], "vixy_weak": vixy_weak, "tier": "b_plus"}}


def test_orb_near_miss_counts_only_the_orb_clause_failures(tmp_path):
    from datetime import datetime, timezone
    events = [
        _evaluation("2026-08-06T14:00:00+00:00", "inside"),                     # NEAR MISS
        _evaluation("2026-08-06T14:05:00+00:00", "inside", confirmations=1),    # also short confirmers
        _evaluation("2026-08-06T14:10:00+00:00", "inside", vixy_weak=False),    # vol disagrees
        _evaluation("2026-08-06T14:15:00+00:00", "inside", above_vwap=False),   # wrong VWAP side
        _evaluation("2026-08-06T14:20:00+00:00", "above", decision="CONFIRM_ENTRY"),
    ]
    jp = _journal(tmp_path, events)
    svc._MEM_CACHE.clear()
    out = svc._orb(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    assert out["evaluated"] == 5 and out["confirmed"] == 1
    assert out["near_miss"] == 1, "only the tick failing ONLY the ORB clause counts"
    assert out["collecting"] is False
    assert out["samples"][0]["orb_state"] == "inside"


def test_orb_near_miss_reports_collecting_when_no_evaluations_exist(tmp_path):
    # Before 2026-08-06 the checks were discarded — an empty result means "not collected yet",
    # never "no near misses happened".
    from datetime import datetime, timezone
    jp = _journal(tmp_path, _modern_trade())
    svc._MEM_CACHE.clear()
    out = svc._orb(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    assert out["collecting"] is True and out["near_miss"] == 0


def test_orb_near_miss_respects_direction(tmp_path):
    from datetime import datetime, timezone
    bearish = {"event_type": "candidate_evaluation", "ts": "2026-08-06T14:00:00+00:00",
               "trade_date": "2026-08-06", "underlying": "SPY", "direction": "bearish",
               "decision": "keep_watching",
               "checks": {"underlying_above_vwap": False, "underlying_orb_state": "inside",
                          "confirmations": 3, "vixy_firming": True, "vixy_weak": False}}
    jp = _journal(tmp_path, [bearish])
    svc._MEM_CACHE.clear()
    out = svc._orb(jp, 10, datetime(2026, 8, 6, 23, 0, tzinfo=timezone.utc))
    assert out["near_miss"] == 1, "a bearish near-miss needs below-VWAP + VIXY firming"


# --- day packets -------------------------------------------------------------------------------

def test_tape_frames_read_nested_and_flat_snapshot_shapes():
    packet = {"market_snapshots": [
        # 2026-07-27+ nested per-symbol block
        {"as_of": "2026-08-06T14:00:00+00:00",
         "SPY": {"last": 741.0, "vwap": 740.0, "orb_high": 742.0, "orb_low": 738.0,
                 "orb_state": "inside", "above_vwap": True}},
        # older flat keys
        {"as_of": "2026-06-26T14:00:00+00:00", "spy_price": 700.0, "spy_vwap": 699.0,
         "spy_orb_state": "above", "spy_above_vwap": True},
        {"as_of": "2026-08-06T14:05:00+00:00"},        # no price -> dropped, never zero-filled
    ]}
    rows = svc.tape_frames(packet, "SPY")
    assert [r["last"] for r in rows] == [700.0, 741.0]      # chronological
    assert rows[1]["orb_high"] == 742.0 and rows[0]["vwap"] == 699.0


def test_day_markers_dedupe_the_double_journaled_fill(tmp_path):
    jp = _journal(tmp_path, _modern_trade())
    svc._MEM_CACHE.clear()
    m = svc._markers(jp, "2026-08-06")
    assert len(m["entries"]) == 1 and len(m["exits"]) == 1
    assert m["exits"][0]["rail_fired"] == "TAKE_PROFIT"


# --- constants come from the live modules -------------------------------------------------------

def test_reference_lines_come_from_live_config():
    import data.odte_config as oc
    import data.odte_day_score as ods
    from data.odte_execution_policy import DEFAULT_LEASE_TTL_SECONDS, MAX_LEASE_TTL_SECONDS
    assert svc.conversion_sla_seconds() == float(oc.CONFIRM_CONVERSION_SLA_SECONDS)
    assert svc.good_day_min_score() == float(ods.GOOD_DAY_MIN_SCORE)
    assert svc.lease_ttl_seconds() == (float(DEFAULT_LEASE_TTL_SECONDS),
                                       float(MAX_LEASE_TTL_SECONDS))


def test_fill_vocabularies_track_the_journal_module():
    from data.odte_journal import ENTRY_FILL_EVENTS, EXIT_FILL_EVENTS
    assert svc._ENTRY_TYPES == tuple(ENTRY_FILL_EVENTS)
    assert svc._EXIT_TYPES == tuple(EXIT_FILL_EVENTS)


def test_service_runs_headless_without_streamlit_session_state(tmp_path):
    # Every function above already ran without a ScriptRunContext; assert the cache fell back to
    # the module dict rather than silently no-op'ing.
    svc._MEM_CACHE.clear()
    svc.trade_ledger(_journal(tmp_path, _modern_trade()))
    assert svc._MEM_CACHE, "headless callers must still get memoization"


def test_module_places_no_orders_and_calls_no_broker():
    import inspect
    src = inspect.getsource(svc)
    for forbidden in ("place_order", "submit_order", "cancel_order", "requests.", "httpx.",
                      "anthropic", "openai"):
        assert forbidden not in src, f"0DTE UI service must never reference {forbidden}"
