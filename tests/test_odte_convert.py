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


# --- telemetry recorded at computation time (2026-08-06) ---------------------------------------
# The checks the candidate watch computes and the day score it derives were both discarded every
# tick, surviving only in overwritten artifacts. They are journaled now so the near-miss population
# and the day-score headroom become countable. Both are scan_only telemetry — never gate inputs.

def test_candidate_evaluation_journaled_on_every_tick_including_non_converting(tmp_path):
    # A tick that does NOT convert is exactly the one carrying near-miss evidence.
    payload = _convert(tmp_path, market=_market(**{"QQQ": {"last": 724.2, "above_vwap": True,
                                                          "orb_state": "inside"}}))
    assert payload["converted"] is False
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    evals = [e for e in events if e.get("event_type") == "candidate_evaluation"]
    assert evals, "the evaluation must be recorded even when the candidate does not convert"
    checks = evals[-1]["checks"]
    # the clause-level results that make an ORB near-miss countable
    assert checks["underlying_orb_state"] == "inside"
    assert checks["underlying_above_vwap"] is True
    assert "confirmations" in checks and "vixy_weak" in checks
    assert evals[-1]["scan_only"] is True and evals[-1]["execution_allowed"] is False


def test_candidate_evaluation_journaled_on_converting_tick(tmp_path):
    payload = _convert(tmp_path)
    assert payload["converted"] is True, payload["reason_codes"]
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    evals = [e for e in events if e.get("event_type") == "candidate_evaluation"]
    assert evals and evals[-1]["checks"].get("tier")


def test_telemetry_events_do_not_disturb_trade_or_refusal_counters(tmp_path):
    # The 2026-08-06 lesson: a new event type reaching the aggregators is how a money/volume
    # counter silently drifts. Telemetry must be inert to every one of them.
    _convert(tmp_path)
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    assert any(e.get("event_type") == "candidate_evaluation" for e in events)
    assert any(e.get("event_type") == "day_score" for e in events)
    assert any(e.get("event_type") == "vehicle_score" for e in events)
    s = oj.summarize(events)
    assert s["n_trades"] == 0 and s["n_closed"] == 0 and s["total_realized_pnl"] == 0
    wt = oj.weekly_telemetry(events, now=NOW)
    assert wt["trades_this_week"] == 0
    assert wt["no_trade_decisions"] == 0 and wt["lease_refusals"] == 0
    assert oj.daily_trade_budget(events, now=NOW)["trades_today"] == 0


def test_vehicle_score_journaled_with_bp_fit_at_computation_time(tmp_path):
    # bp_fit shipped 2026-08-05 into a stream that had already stopped flowing: the ingest lane
    # (loose top-level artifacts) went dry at the v5 port, and odte_convert — the only live caller
    # of score_vehicle — kept {verdict, score} and dropped the rest. Result: zero bp_fit events
    # ever recorded, while 2026-08-06 scored 14 gate ticks.
    _convert(tmp_path)
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    scores = [e for e in events if e.get("event_type") == "vehicle_score"]
    assert scores
    last = scores[-1]
    assert last["verdict"] and last["score"] is not None
    assert isinstance(last["bp_fit"], dict) and last["bp_fit"].get("tier")
    assert last["bp_fit"].get("max_affordable_ask") is not None
    assert last["scan_only"] is True and last["execution_allowed"] is False
    # No trade_id: summarize() would open a trade row for it and inflate n_trades.
    assert last.get("trade_id") is None
    assert last["source"] == "odte_convert"


def test_day_score_journaled_with_headroom_telemetry(tmp_path):
    _convert(tmp_path)
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    scores = [e for e in events if e.get("event_type") == "day_score"]
    assert scores
    last = scores[-1]
    assert last["verdict"] and isinstance(last["components"], dict)
    # headroom: was a GOOD_DAY even reachable with the components actually supplied?
    assert last["components_supplied"] is not None
    assert last["max_possible_score"] is not None
    assert isinstance(last["components_missing"], list)


def test_lease_event_carries_ttl_and_ceilings(tmp_path):
    # The atomic path used to journal a 7-key subset, so a lease's expiry and its price/debit
    # ceilings were unreadable after the fact and "expired" had to be inferred.
    payload = _convert(tmp_path)
    assert payload["converted"] is True, payload["reason_codes"]
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    lease_ev = [e for e in events if e.get("event_type") == "execution_lease_issued"][-1]
    for key in ("issued_at", "expires_at", "max_limit_price", "max_debit", "quantity",
                "risk_mode", "lease_id"):
        assert lease_ev.get(key) is not None, f"lease event missing {key}"
    assert lease_ev["authorized"] is True
    assert lease_ev["tier"] and lease_ev["option_id"]
    # mutual consistency: the recorded ceilings can never contradict each other
    assert lease_ev["max_limit_price"] * lease_ev["quantity"] * 100.0 <= lease_ev["max_debit"] + 1e-9


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


# --- post-green auto-arm end-to-end (2026-08-03) ------------------------------------------------

def _seed_green_journal(tmp_path, now):
    events = [
        {"event_type": "execution_lease_issued", "trade_id": "t1", "underlying": "SPY",
         "option_id": "spy-756c", "tier": "full", "authorized": True,
         "ts": (now - timedelta(hours=2)).isoformat()},
        {"event_type": "order_filled", "trade_id": "t1", "underlying": "SPY",
         "option_id": "spy-756c", "ts": (now - timedelta(hours=2)).isoformat()},
        {"event_type": "order_closed", "trade_id": "t1", "underlying": "SPY",
         "option_id": "spy-756c", "realized_pnl": 9.0,
         "ts": (now - timedelta(hours=1)).isoformat()},
    ]
    jp = tmp_path / "decision_journal.jsonl"
    with open(jp, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return str(jp)


def test_post_green_second_trade_auto_arms_and_converts(tmp_path):
    # The 2026-08-03 failure: after the +$9 green close, trade #2 was structurally impossible.
    # Now an a_plus tape (>= the winning "full" tier) with a budget slot, cooldown clear, and BP
    # covering the multiple converts straight through — no manual flag anywhere.
    jp = _seed_green_journal(tmp_path, NOW)
    payload = _convert(tmp_path, journal_path=jp)
    assert payload["converted"] is True, payload["reason_codes"]
    gate = payload["entry_gate"]
    assert "green_reentry_auto_armed_tier" in gate["reason_codes"]
    assert gate["green_reentry"]["auto_armed"] is True
    assert gate["green_reentry"]["winning_tier_today"] == "full"


def test_post_green_kill_switch_off_refuses_at_gate(tmp_path, monkeypatch):
    import data.odte_entry_gate as eg
    monkeypatch.setattr(eg, "GREEN_REENTRY_AUTO_ARM", False)
    jp = _seed_green_journal(tmp_path, NOW)
    payload = _convert(tmp_path, journal_path=jp)
    assert payload["converted"] is False
    assert payload["stage"] == "entry_gate"
    assert any("green_day_preservation_lockout" in str(r) for r in payload["reason_codes"])


# --- 2026-08-04 day-1 fixes: gap backfill, components, deadline, guidance pass-through ----------

def test_gap_pct_persists_and_backfills_same_et_day(tmp_path):
    # THE 24-CENT KILL: gap_pct (an overnight constant) vanished mid-session -> GOOD_DAY->CHOP ->
    # tier halved -> a $214.00 debit refused against $214.24. First sighting persists; a later
    # snapshot that lost the key is backfilled, loudly flagged.
    p1 = _convert(tmp_path)                                   # market has gap_pct 0.5
    assert p1["converted"] is True
    sc = json.loads((tmp_path / cv.SESSION_CONSTANTS_FILENAME).read_text())
    assert sc["gap_pct"] == 0.5
    assert p1["gap_pct_backfilled"] is False
    assert p1["day_score"]["components"]["gap"] == 1

    # A later tick (distinct clock so journal event_ids differ) whose snapshot lost the key.
    from datetime import timedelta
    later = NOW + timedelta(minutes=5)
    m2 = _market(now=later)
    del m2["gap_pct"]                                         # the Aug-4 dropped key
    p2 = _convert(tmp_path, now=later, market=m2, contract=_contract(now=later),
                  broker=_broker(now=later), candidate=_candidate(now=later))
    assert p2["gap_pct_backfilled"] is True
    assert p2["day_score"]["components"]["gap"] == 1, "backfilled gap must restore the component"
    assert p2["converted"] is True
    events = oj.read_events(str(tmp_path / "decision_journal.jsonl"))
    flagged = [e for e in events if e.get("gap_pct_backfilled")]
    assert flagged, "the backfill must be visible in the journal"


def test_gap_backfill_never_crosses_et_dates(tmp_path):
    from datetime import timedelta
    p1 = _convert(tmp_path)
    assert (tmp_path / cv.SESSION_CONSTANTS_FILENAME).exists()
    tomorrow = NOW + timedelta(days=1)
    m2 = _market(now=tomorrow)
    del m2["gap_pct"]
    p2 = _convert(tmp_path, now=tomorrow, market=m2,
                  contract=_contract(now=tomorrow), broker=_broker(now=tomorrow),
                  candidate=_candidate(now=tomorrow))
    assert p2["gap_pct_backfilled"] is False, "yesterday's gap must never leak into today"


def test_converted_payload_carries_machine_deadline(tmp_path):
    p = _convert(tmp_path)
    assert p["converted"] is True
    assert p["place_deadline"] == p["lease"]["expires_at"]
    assert p["lease_seconds_remaining"] is not None
    assert 0 < p["lease_seconds_remaining"] <= 60.0


def test_entry_gate_refusal_passes_through_rotation_guidance(tmp_path):
    # 2026-08-04: convert's generic refusal prose OVERWROTE the gate's rotate-vehicle guidance.
    # A budget-check refusal must surface the gate's own next_action/next_command.
    # Inside BP (vehicle stays GOOD_BET) but over the 60% tier cap -> budget_check fails at the
    # gate, which is exactly the Aug-4 shape.
    expensive = _contract(ask=2.50, bid=2.48, mark=2.49)       # debit $250 vs cap ~$208.9
    p = _convert(tmp_path, contract=expensive)
    assert p["converted"] is False
    assert p["stage"] == "entry_gate"
    assert "QQQ/SPY/IWM" in p["next_action"]
    assert "odte-candidate-watch" in p["next_command"]
    assert "refresh the named inputs" not in p["next_action"]


def test_zero_buying_power_fails_closed_not_fallback(tmp_path):
    broke = _broker(buying_power=0.0, options_buying_power=5000.0)
    p = _convert(tmp_path, broker=broke)
    assert p["converted"] is False
    assert p["confirmation_detail"]["budget_check"]["buying_power"] == 0.0


# --- fast-lane prerequisites (2026-08-04): dict inputs + journal_events read override ------------

def test_dict_inputs_identical_to_json_string_inputs(tmp_path):
    # The daemon holds the artifacts in memory — passing dicts must produce the exact decision
    # a json.dumps round trip produces.
    kw_a = dict(state_dir=str(tmp_path / "a"), journal=False, write=False, now=NOW,
                journal_path=str(tmp_path / "a" / "decision_journal.jsonl"))
    kw_b = dict(state_dir=str(tmp_path / "b"), journal=False, write=False, now=NOW,
                journal_path=str(tmp_path / "b" / "decision_journal.jsonl"))
    p_str = cv.run_convert(candidate_json=json.dumps(_candidate()),
                           market_json=json.dumps(_market()),
                           broker_json=json.dumps(_broker()),
                           contract_json=json.dumps(_contract()), **kw_a)
    p_dict = cv.run_convert(candidate_json=_candidate(), market_json=_market(),
                            broker_json=_broker(), contract_json=_contract(), **kw_b)
    assert p_dict["converted"] is True
    assert (json.dumps(p_dict, sort_keys=True, default=str)
            == json.dumps(p_str, sort_keys=True, default=str))


def test_journal_events_override_enforces_budget_with_zero_appends(tmp_path):
    # Shadow mode: read the REAL journal's budget/green state via journal_events while journal=False
    # guarantees no append anywhere. A 2-entry day must refuse daily_trade_budget_exhausted.
    # (Day kept net-RED so the 2026-08-06 a_plus-uncapped green-day exception cannot apply —
    # this test pins the zero-append mechanics, not the exception.)
    real_events = []
    for i, hours_ago in enumerate((4, 2)):
        ts = (NOW - timedelta(hours=hours_ago)).isoformat()
        real_events += [
            {"event_type": "execution_lease_issued", "trade_id": f"t{i}", "underlying": "SPY",
             "option_id": f"spy-{i}", "tier": "full", "authorized": True, "ts": ts},
            {"event_type": "order_filled", "trade_id": f"t{i}", "underlying": "SPY",
             "option_id": f"spy-{i}", "ts": ts},
            {"event_type": "order_closed", "trade_id": f"t{i}", "underlying": "SPY",
             "option_id": f"spy-{i}", "realized_pnl": -5.0,
             "ts": (NOW - timedelta(hours=hours_ago - 1)).isoformat()},
        ]
    assert len([e for e in real_events if e["event_type"] == "order_filled"]) \
        == oc.DAILY_TRADE_BUDGET
    payload = cv.run_convert(candidate_json=_candidate(), market_json=_market(),
                             broker_json=_broker(), contract_json=_contract(),
                             state_dir=str(tmp_path), write=False, journal=False,
                             journal_events=real_events,
                             journal_path=str(tmp_path / "decision_journal.jsonl"), now=NOW)
    assert payload["converted"] is False
    assert payload["stage"] == "entry_gate"
    import data.odte_entry_gate as eg
    assert any(eg.DAILY_BUDGET_VETO in str(r) for r in payload["reason_codes"])
    assert not (tmp_path / "decision_journal.jsonl").exists()   # zero appends anywhere


def test_warm_convert_on_dicts_is_fast(tmp_path):
    # The daemon calls run_convert inside the FIRING window — the warm in-process conversion must
    # stay far under the lease clock (measured ~30ms; loose CI bound).
    import time
    kw = dict(state_dir=str(tmp_path), write=False, journal=False, journal_events=[],
              journal_path=str(tmp_path / "decision_journal.jsonl"))
    cv.run_convert(candidate_json=_candidate(), market_json=_market(), broker_json=_broker(),
                   contract_json=_contract(), now=NOW, **kw)          # warm imports/caches
    t0 = time.perf_counter()
    p = cv.run_convert(candidate_json=_candidate(), market_json=_market(), broker_json=_broker(),
                       contract_json=_contract(), now=NOW, **kw)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert p["converted"] is True
    assert elapsed_ms < 150, f"warm convert took {elapsed_ms:.1f}ms"


def test_mint_seeds_the_consumed_ledger(tmp_path):
    # 2026-08-05 audit: the ledger only materialized on the first guard/consume call, and the
    # Hermes pre-order hook SKIPS its replay check when the file is absent — Aug-3's lease was
    # replayable for its whole TTL. Minting now seeds an empty ledger first.
    ledger = tmp_path / "consumed_leases.json"
    assert not ledger.exists()
    payload = _convert(tmp_path)
    assert payload["converted"] is True
    assert json.loads(ledger.read_text()) == []
    # Seeding is idempotent and NEVER truncates an existing ledger.
    import data.odte_execution_policy as xp
    xp.record_consumed(str(ledger), "burned-1")
    p2 = _convert(tmp_path, now=NOW + timedelta(minutes=5))
    assert p2["converted"] is True
    assert "burned-1" in json.loads(ledger.read_text())


# --- A+ uncapped daily budget (2026-08-06 user policy) ----------------------------------------

def _budget_exhausted_events(now, day_pnl=5.0):
    """Two completed trades today (base cap consumed) with a chosen net day P/L."""
    events = []
    for i, hours_ago in enumerate((4, 2)):
        ts = (now - timedelta(hours=hours_ago)).isoformat()
        events += [
            {"event_type": "entry_fill", "trade_id": f"t{i}", "underlying": "SPY",
             "option_id": f"spy-{i}", "ts": ts},
            {"event_type": "order_closed", "trade_id": f"t{i}", "underlying": "SPY",
             "option_id": f"spy-{i}", "realized_pnl": day_pnl / 2.0,
             "ts": (now - timedelta(hours=hours_ago - 1)).isoformat()},
        ]
    return events


def test_aplus_converts_past_exhausted_budget_on_green_day():
    # a_plus tape (the default _market: 3 confirmers, 0 dissenters) + budget 2/2 used + day
    # net-green -> the gate grants the a_plus exception and the conversion mints a lease.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        payload = cv.run_convert(candidate_json=_candidate(), market_json=_market(),
                                 broker_json=_broker(), contract_json=_contract(),
                                 state_dir=td, write=False, journal=False,
                                 journal_events=_budget_exhausted_events(NOW, day_pnl=9.0),
                                 journal_path=str(Path(td) / "j.jsonl"), now=NOW)
    assert payload["converted"] is True, payload["reason_codes"]
    import data.odte_entry_gate as eg
    assert eg.APLUS_BUDGET_EXCEPTION in payload["entry_gate"]["reason_codes"]


def test_non_aplus_and_red_days_stay_capped(monkeypatch):
    import tempfile
    from pathlib import Path

    import data.odte_entry_gate as eg
    import data.odte_journal as oj

    def _run(market, events, td):
        return cv.run_convert(candidate_json=_candidate(), market_json=market,
                              broker_json=_broker(), contract_json=_contract(),
                              state_dir=td, write=False, journal=False,
                              journal_events=events,
                              journal_path=str(Path(td) / "j.jsonl"), now=NOW)

    # B+ tier (CHOP + one dissenter) on a green day: still vetoed at the cap.
    bplus_market = _market()
    bplus_market["IWM"] = {"last": 298.0, "above_vwap": False, "orb_state": "inside"}
    bplus_market["gap_pct"] = 0.1                      # muted gap -> CHOP day
    with tempfile.TemporaryDirectory() as td:
        refused = _run(bplus_market, _budget_exhausted_events(NOW, day_pnl=9.0), td)
    assert refused["converted"] is False
    assert any(eg.DAILY_BUDGET_VETO in str(r) for r in refused["reason_codes"])

    # a_plus tape but the day is net-RED: the anti-tilt half holds — vetoed.
    with tempfile.TemporaryDirectory() as td:
        red = _run(_market(), _budget_exhausted_events(NOW, day_pnl=-6.0), td)
    assert red["converted"] is False
    assert any(eg.DAILY_BUDGET_VETO in str(r) for r in red["reason_codes"])

    # Kill switch off: vetoed regardless.
    monkeypatch.setattr(oj, "green_day_preservation", oj.green_day_preservation)  # no-op guard
    import data.odte_config as oc2
    monkeypatch.setattr(oc2, "DAILY_BUDGET_APLUS_UNCAPPED", False)
    with tempfile.TemporaryDirectory() as td:
        off = _run(_market(), _budget_exhausted_events(NOW, day_pnl=9.0), td)
    assert off["converted"] is False
