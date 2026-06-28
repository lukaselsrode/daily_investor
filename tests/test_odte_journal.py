"""tests/test_odte_journal.py — 0DTE decision journal (local/offline, no broker/LLM/network).

Pure unit tests over tmp_path: append/read JSONL roundtrip, summary metrics (hit rate, avg P/L,
by-mode, MFE capture, rule violations, timing), Markdown/CSV artifacts, no-data behavior, experiment
extraction, the NVDA employer-restriction tag/exclusion, the position->event helper, and a source
guardrail that the module makes no broker/network/LLM calls.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_journal as oj


def _journal(tmp_path):
    return str(tmp_path / "decision_journal.jsonl")


def _seed_one_trade(jp, trade_id="t1", mode="scalp", realized=12.0, mfe=20.0):
    oj.append_event({"event_type": "pre_trade_thesis", "trade_id": trade_id, "mode": mode,
                     "underlying": "SPY", "thesis": {"direction": "call", "catalyst": "CPI"}},
                    journal_path=jp)
    oj.append_event({"event_type": "entry_decision", "trade_id": trade_id, "mode": mode,
                     "underlying": "SPY", "ts": "2026-06-23T13:30:00-04:00",
                     "decision": {"action": "open", "confidence": "medium"}}, journal_path=jp)
    oj.append_event({"event_type": "management_check", "trade_id": trade_id,
                     "decision": {"action": "HOLD"}}, journal_path=jp)
    oj.append_event({"event_type": "order_closed", "trade_id": trade_id, "mode": mode,
                     "underlying": "SPY", "ts": "2026-06-23T13:55:00-04:00",
                     "outcome": {"realized_pnl": realized, "mfe": mfe,
                                 "rule_violations": [], "lessons": ["sized fine"]}},
                    journal_path=jp)


# --- append / read ---------------------------------------------------------------------------

def test_append_and_read_roundtrip(tmp_path):
    jp = _journal(tmp_path)
    a = oj.append_event({"event_type": "note", "trade_id": "x"}, journal_path=jp)
    b = oj.append_event({"type": "entry_decision", "trade_id": "x"}, journal_path=jp)  # `type` alias
    assert a["seq"] == 0 and b["seq"] == 1
    assert a["ts"] and b["event_type"] == "entry_decision"
    events = oj.read_events(jp)
    assert len(events) == 2 and [e["event_type"] for e in events] == ["note", "entry_decision"]


def test_read_skips_malformed_lines(tmp_path):
    jp = tmp_path / "decision_journal.jsonl"
    jp.write_text('{"event_type":"note"}\nNOT JSON\n\n{"event_type":"postmortem"}\n')
    events = oj.read_events(str(jp))
    assert [e["event_type"] for e in events] == ["note", "postmortem"]


# --- summary metrics -------------------------------------------------------------------------

def test_summary_metrics_hit_rate_pnl_mode_capture(tmp_path):
    jp = _journal(tmp_path)
    _seed_one_trade(jp, "t1", "scalp", realized=12.0, mfe=20.0)   # win, capture 0.6
    _seed_one_trade(jp, "t2", "scalp", realized=-8.0, mfe=5.0)    # loss
    oj.append_event({"event_type": "order_closed", "trade_id": "t3", "mode": "runner",
                     "outcome": {"realized_pnl": 30.0, "mfe": 60.0,
                                 "rule_violations": ["held_past_flat"]}}, journal_path=jp)
    s = oj.summarize(oj.read_events(jp))
    assert s["n_trades"] == 3 and s["n_closed"] == 3
    assert s["hit_rate"] == round(2 / 3, 4)
    assert s["total_realized_pnl"] == 34.0
    assert s["avg_realized_pnl"] == round(34.0 / 3, 4)
    # capture = mean(12/20, -8/5, 30/60) = mean(0.6, -1.6, 0.5)
    assert s["avg_mfe_capture"] == round((0.6 - 1.6 + 0.5) / 3, 4)
    assert s["by_mode"]["scalp"]["trades"] == 2 and s["by_mode"]["scalp"]["wins"] == 1
    assert s["by_mode"]["runner"]["realized_pnl"] == 30.0
    assert s["rule_violations"].get("held_past_flat") == 1
    assert s["n_management_checks"] == 2   # one per seeded trade (t1, t2)


def test_summary_held_minutes(tmp_path):
    jp = _journal(tmp_path)
    _seed_one_trade(jp, "t1", "scalp")
    s = oj.summarize(oj.read_events(jp))
    assert s["avg_held_minutes"] == 25.0   # 13:30 -> 13:55


def test_open_trade_not_counted_closed(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "entry_decision", "trade_id": "open1", "mode": "lotto"},
                    journal_path=jp)
    s = oj.summarize(oj.read_events(jp))
    assert s["n_trades"] == 1 and s["n_closed"] == 0 and s["hit_rate"] is None


# --- experiments / lessons -------------------------------------------------------------------

def test_experiment_extraction(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "experiment", "hypothesis": "VWAP reclaim entries beat opening-range",
                     "metric": "hit_rate", "promote_if": ">55% over 10 trades",
                     "kill_if": "<40% over 10 trades", "status": "open"}, journal_path=jp)
    s = oj.summarize(oj.read_events(jp))
    assert len(s["experiments"]) == 1
    assert s["experiments"][0]["metric"] == "hit_rate"
    assert s["experiments"][0]["status"] == "open"


# --- no data ---------------------------------------------------------------------------------

def test_no_data_behavior(tmp_path):
    res = oj.build_report(journal_path=_journal(tmp_path))
    s = res["summary"]
    assert s["n_events"] == 0 and s["hit_rate"] is None and s["total_realized_pnl"] == 0.0
    assert "No journal events yet" in res["markdown"]
    assert res["csv"].startswith("mode,trades")
    assert res["artifacts"] == {}   # nothing written without --write/out_dir


# --- artifacts ------------------------------------------------------------------------------

def test_report_writes_md_and_csv(tmp_path):
    jp = _journal(tmp_path)
    _seed_one_trade(jp, "t1", "scalp", realized=15.0, mfe=20.0)
    out = tmp_path / "reports"
    res = oj.build_report(journal_path=jp, out_dir=str(out), write_artifacts=True)
    md, csv = out / "odte_journal_report.md", out / "odte_journal_summary.csv"
    assert md.exists() and csv.exists()
    assert "Decision Journal" in md.read_text() and "Trades by mode" in md.read_text()
    assert "scalp" in csv.read_text()
    assert res["artifacts"]["markdown"] == str(md)


# --- NVDA restriction ------------------------------------------------------------------------

def test_nvda_event_tagged_and_excluded(tmp_path):
    jp = _journal(tmp_path)
    stored = oj.append_event({"event_type": "entry_decision", "trade_id": "bad", "mode": "scalp",
                              "underlying": "nvda",
                              "outcome": {"realized_pnl": 99.0, "mfe": 100.0}}, journal_path=jp)
    assert stored["restricted"] is True and stored["restricted_reason"] == "employer"
    # An experiment that names NVDA must never surface as a forward recommendation.
    oj.append_event({"event_type": "experiment", "underlying": "NVDA",
                     "hypothesis": "trade NVDA 0DTE"}, journal_path=jp)
    s = oj.summarize(oj.read_events(jp))
    assert "NVDA" in s["restricted_flags"]
    assert s["experiments"] == []                      # restricted experiment excluded
    assert s["n_closed"] == 0                          # restricted trade excluded from metrics
    assert any("RESTRICTED_EMPLOYER" in v for v in s["rule_violations"])


# --- sentiment / gamma status rollups --------------------------------------------------------

def test_sentiment_status_rollup(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "pre_trade_thesis", "trade_id": "t1", "underlying": "SPY",
                     "sentiment": {"verdict": "OBSERVE", "direction": "bullish",
                                   "confidence": "low", "sentiment": 0.4, "mentions": 12}},
                    journal_path=jp)
    oj.append_event({"event_type": "management_check", "trade_id": "t1", "underlying": "SPY",
                     "sentiment": {"verdict": "BUY", "intent": "bullish",   # `intent` alias
                                   "confidence": "medium", "sentiment": 0.8}}, journal_path=jp)
    oj.append_event({"event_type": "note", "trade_id": "t2", "underlying": "QQQ",
                     "sentiment": {"verdict": "OBSERVE", "direction": "bearish",
                                   "confidence": "low", "sentiment": -0.6}}, journal_path=jp)
    ss = oj.summarize(oj.read_events(jp))["sentiment_status"]
    assert ss["n_readings"] == 3
    assert ss["latest"]["verdict"] == "OBSERVE" and ss["latest"]["direction"] == "bearish"
    assert ss["by_verdict"] == {"OBSERVE": 2, "BUY": 1}
    assert ss["by_direction"] == {"bullish": 2, "bearish": 1}
    assert ss["avg_score"] == round((0.4 + 0.8 - 0.6) / 3, 4)
    assert ss["restricted_readings"] == []


def test_sentiment_status_restricted_excluded(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "note", "trade_id": "t1", "underlying": "SPY",
                     "sentiment": {"verdict": "BUY", "direction": "bullish", "sentiment": 0.5}},
                    journal_path=jp)
    oj.append_event({"event_type": "note", "trade_id": "bad", "underlying": "NVDA",
                     "sentiment": {"verdict": "BUY", "direction": "bullish", "sentiment": 0.9}},
                    journal_path=jp)
    ss = oj.summarize(oj.read_events(jp))["sentiment_status"]
    assert ss["n_readings"] == 2                       # both counted as records...
    assert ss["latest"]["underlying"] == "SPY"         # ...but NVDA never the latest/forward read
    assert ss["avg_score"] == 0.5                      # restricted score excluded from the mean
    assert ss["restricted_readings"] == ["NVDA"]


def test_sentiment_status_empty(tmp_path):
    jp = _journal(tmp_path)
    _seed_one_trade(jp, "t1", "scalp")                 # trade events carry no `sentiment` block
    ss = oj.summarize(oj.read_events(jp))["sentiment_status"]
    assert ss == {"n_readings": 0, "latest": None, "by_verdict": {}, "by_direction": {},
                  "by_status": {}, "avg_score": None, "restricted_readings": []}


def test_gamma_status_rollup_and_no_dealer_gex(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "pre_trade_thesis", "trade_id": "t1", "underlying": "SPY",
                     "gamma": {"gamma_regime": "pin_risk_only_not_dealer_gex", "gamma_available": True,
                               "max_gamma_strike": 600.0, "call_wall": 605.0, "put_wall": 595.0,
                               "pin_risk": {"level": "medium"},
                               "freshness": {"quote_fresh": True}}}, journal_path=jp)
    oj.append_event({"event_type": "management_check", "trade_id": "t1", "underlying": "SPY",
                     "gamma": {"max_gamma_strike": 600.0, "pin_risk_level": "high",  # flat alias
                               "quote_fresh": True}}, journal_path=jp)
    gs = oj.summarize(oj.read_events(jp))["gamma_status"]
    assert gs["n_readings"] == 2
    assert gs["latest"]["pin_risk"] == "high" and gs["latest"]["max_gamma_strike"] == 600.0
    assert gs["by_pin_risk"] == {"medium": 1, "high": 1}
    assert gs["regime"] == "pin_risk_only_not_dealer_gex"
    assert gs["includes_dealer_positioning"] is False
    # No field anywhere claims a real dealer-GEX / gamma-flip / sign number.
    def _keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from _keys(v)
        elif isinstance(o, list):
            for x in o:
                yield from _keys(x)
    bad = {k.lower() for k in _keys(gs)} & {"gex", "dealer_gex", "net_gex", "gamma_flip", "flip_point"}
    assert not bad, f"gamma_status must not expose dealer-GEX fields: {bad}"


def test_gamma_status_restricted_excluded(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "note", "trade_id": "bad", "underlying": "NVDA",
                     "gamma": {"max_gamma_strike": 130.0, "pin_risk": {"level": "high"}}},
                    journal_path=jp)
    gs = oj.summarize(oj.read_events(jp))["gamma_status"]
    assert gs["n_readings"] == 1 and gs["latest"] is None   # restricted read never surfaces as latest
    assert gs["by_pin_risk"] == {} and gs["restricted_readings"] == ["NVDA"]


def test_sentiment_status_flat_aliases(tmp_path):
    jp = _journal(tmp_path)
    # Flat top-level status field (no nested `sentiment` dict) must still produce a reading row.
    oj.append_event({"event_type": "pre_trade_thesis", "trade_id": "t1", "underlying": "SPY",
                     "sentiment_status": "useful_context"}, journal_path=jp)
    oj.append_event({"event_type": "management_check", "trade_id": "t1", "underlying": "SPY",
                     "sentiment_state": "diverged_warning"}, journal_path=jp)  # `sentiment_state` alias
    # Free-text freshness/pulse context-only, no explicit status code.
    oj.append_event({"event_type": "note", "trade_id": "t2", "underlying": "QQQ",
                     "social_freshness": "stale"}, journal_path=jp)  # stale -> stale_unavailable
    ss = oj.summarize(oj.read_events(jp))["sentiment_status"]
    assert ss["n_readings"] == 3
    assert ss["by_status"] == {"useful_context": 1, "diverged_warning": 1, "stale_unavailable": 1}
    assert ss["latest"]["status"] == "stale_unavailable" and ss["latest"]["context"] == "stale"


def test_sentiment_status_thesis_social_pulse(tmp_path):
    jp = _journal(tmp_path)
    # thesis.social_pulse is captured as context even with no nested sentiment / explicit status.
    oj.append_event({"event_type": "pre_trade_thesis", "trade_id": "t1", "underlying": "SPY",
                     "thesis": {"direction": "call", "social_pulse": "quiet, no clear lean"}},
                    journal_path=jp)
    ss = oj.summarize(oj.read_events(jp))["sentiment_status"]
    assert ss["n_readings"] == 1
    assert ss["latest"]["context"] == "quiet, no clear lean" and ss["latest"]["status"] is None
    assert ss["by_status"] == {}


def test_gamma_status_flat_pin_state_no_export(tmp_path):
    jp = _journal(tmp_path)
    # Live "no Robinhood export" shape: flat gamma_pin_state, no nested `gamma` dict. Must still
    # produce a row normalized to the honest unavailable_no_export status.
    oj.append_event({"event_type": "pre_trade_thesis", "trade_id": "t1", "underlying": "SPY",
                     "gamma_pin_state": "unknown_no_export_available"}, journal_path=jp)
    oj.append_event({"event_type": "management_check", "trade_id": "t1", "underlying": "SPY",
                     "gamma_pin_state": "unknown_no_robinhood_export_for_odte_gamma_map"},
                    journal_path=jp)
    oj.append_event({"event_type": "note", "trade_id": "t2", "underlying": "QQQ",
                     "gamma_status": "source_limited"}, journal_path=jp)
    gs = oj.summarize(oj.read_events(jp))["gamma_status"]
    assert gs["n_readings"] == 3
    assert gs["by_status"] == {"unavailable_no_export": 2, "source_limited": 1}
    assert gs["latest"]["status"] == "source_limited"
    assert gs["includes_dealer_positioning"] is False
    assert gs["regime"] == "pin_risk_only_not_dealer_gex"


def test_old_live_journal_shape_not_invisible(tmp_path):
    jp = _journal(tmp_path)
    # The exact shape today's live journal writes: flat social_freshness + gamma_pin_state on one
    # event, with NO nested sentiment/gamma dicts. Both rollups must surface a status row.
    oj.append_event({"event_type": "management_check", "trade_id": "t1", "underlying": "SPY",
                     "social_freshness": "fresh",
                     "gamma_pin_state": "unknown_no_robinhood_export_for_odte_gamma_map"},
                    journal_path=jp)
    s = oj.summarize(oj.read_events(jp))
    ss, gs = s["sentiment_status"], s["gamma_status"]
    assert ss["n_readings"] == 1 and ss["latest"]["context"] == "fresh"
    assert ss["by_status"] == {"useful_context": 1}              # fresh -> useful_context
    assert gs["n_readings"] == 1 and gs["by_status"] == {"unavailable_no_export": 1}
    md = oj.render_markdown(s)
    assert "Sentiment & gamma context" in md and "unavailable_no_export" in md


def test_status_sections_in_markdown(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "note", "trade_id": "t1", "underlying": "SPY",
                     "sentiment": {"verdict": "BUY", "direction": "bullish", "sentiment": 0.5},
                     "gamma": {"max_gamma_strike": 600.0, "pin_risk": {"level": "medium"},
                               "freshness": {"quote_fresh": True}}}, journal_path=jp)
    md = oj.render_markdown(oj.summarize(oj.read_events(jp)))
    assert "Sentiment & gamma context" in md
    assert "NOT dealer GEX" in md and "pin_risk_only_not_dealer_gex" in md


# --- position -> event helper ----------------------------------------------------------------

def test_event_from_position_decision(tmp_path):
    payload = {"decision": "TAKE_PROFIT", "underlying": "SPY", "mode": "scalp", "pnl_pct": 0.62,
               "option_id": "SPY_C", "triggers": [{"type": "TAKE_PROFIT", "detail": "+62% >= 60%"}]}
    ev = oj.event_from_position_decision(payload, trade_id="t9")
    assert ev["event_type"] == "management_check" and ev["trade_id"] == "t9"
    assert ev["triggers"] == ["TAKE_PROFIT"] and ev["decision"]["action"] == "TAKE_PROFIT"
    stored = oj.append_event(ev, journal_path=_journal(tmp_path))   # round-trips through append
    assert stored["seq"] == 0


# --- vehicle-score -> event helper -----------------------------------------------------------

def test_event_from_vehicle_score(tmp_path):
    payload = {"verdict": "GOOD_BET", "score": 6, "direction": "bullish",
               "components": {"market": 3, "gamma": 2, "liquidity": 1},
               "contract": {"underlying": "QQQ", "option_type": "call", "strike": 718},
               "reasons": ["market: VWAP confirms calls on SPY,QQQ", "gamma: low pin risk"]}
    ev = oj.event_from_vehicle_score(payload, trade_id="t7")
    assert ev["event_type"] == "pre_trade_thesis" and ev["trade_id"] == "t7"
    assert ev["underlying"] == "QQQ" and ev["option_type"] == "call" and ev["strike"] == 718
    assert ev["decision"]["action"] == "GOOD_BET"
    assert ev["decision"]["reasons"] == payload["reasons"]
    assert ev["vehicle_score"]["score"] == 6 and ev["vehicle_score"]["direction"] == "bullish"
    stored = oj.append_event(ev, journal_path=_journal(tmp_path))   # round-trips through append
    assert stored["seq"] == 0 and stored["underlying"] == "QQQ"


def test_event_from_vehicle_score_nvda_tagged_on_append(tmp_path):
    # A restricted underlying flowing through the vehicle-score helper is tagged on store.
    payload = {"verdict": "BAD_BET", "score": -3,
               "contract": {"underlying": "NVDA", "option_type": "put", "strike": 130},
               "reasons": ["gamma: high pin risk"]}
    ev = oj.event_from_vehicle_score(payload, trade_id="bad", extra={"mode": "scalp"})
    assert ev["mode"] == "scalp"
    stored = oj.append_event(ev, journal_path=_journal(tmp_path))
    assert stored["restricted"] is True and stored["restricted_reason"] == "employer"


# --- guardrail: no broker / network / LLM ----------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(oj)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket"):
        assert forbidden not in src, f"odte_journal must not reference {forbidden!r}"


# --- standardized decision-journal layer (append_decision_journal) ---------------------------

def test_append_decision_journal_stamps_envelope_and_returns_appended(tmp_path):
    jp = _journal(tmp_path)
    res = oj.append_decision_journal(
        {"underlying": "spy", "decision": "veto", "reason_codes": ["wide_spread"],
         "thesis": "chop day", "confidence": "low"},
        source="controller", event_type="entry_decision", journal_path=jp)
    assert res["status"] == "appended" and res["event_id"]
    e = res["event"]
    assert e["schema"] == oj.DECISION_SCHEMA and e["source"] == "controller"
    assert e["symbol"] == "SPY" and e["decision"] == "veto"
    assert e["event_type"] == "entry_decision" and e["trade_date"] == oj._derive_trade_date(e["ts"])
    # conservative defaults: not execution-allowed, not scan-only unless asked
    assert e["execution_allowed"] is False and e["scan_only"] is False
    # round-trips through the normal reader
    rows = oj.read_events(jp)
    assert len(rows) == 1 and rows[0]["event_id"] == res["event_id"]


def test_append_decision_journal_is_idempotent_on_artifact_path(tmp_path):
    jp = _journal(tmp_path)
    art = "data/odte/controller_event_20260626_0904.json"
    first = oj.append_decision_journal({"raw_artifact_path": art, "decision": "skip"},
                                       source="ingest", event_type="controller_event", journal_path=jp)
    dup = oj.append_decision_journal({"raw_artifact_path": art, "decision": "skip"},
                                     source="ingest", event_type="controller_event", journal_path=jp)
    assert first["status"] == "appended"
    assert dup["status"] == "duplicate" and dup["event_id"] == first["event_id"]
    assert len(oj.read_events(jp)) == 1, "duplicate must not be re-appended"


def test_scan_only_can_never_be_execution_allowed(tmp_path):
    jp = _journal(tmp_path)
    # Even if a caller wrongly passes execution_allowed=True on a scan_only event, the guard wins.
    res = oj.append_decision_journal(
        {"symbol": "XSP", "scan_only": True, "execution_allowed": True, "decision": "observe"},
        source="social_scan", event_type="scan", journal_path=jp)
    assert res["event"]["scan_only"] is True
    assert res["event"]["execution_allowed"] is False, "scan-only must never be execution-allowed"


def test_restricted_underlying_is_never_execution_allowed(tmp_path):
    jp = _journal(tmp_path)
    res = oj.append_decision_journal(
        {"underlying": "NVDA", "execution_allowed": True, "decision": "enter"},
        source="controller", event_type="entry_decision", journal_path=jp)
    e = res["event"]
    assert e["restricted"] is True and e["restricted_reason"] == "employer"
    assert e["execution_allowed"] is False, "restricted (NVDA) must never be execution-allowed"


def test_append_decision_journal_fails_safe(monkeypatch, tmp_path):
    """A journaling failure must return status=error, not raise (trading loop must not crash)."""
    jp = _journal(tmp_path)

    def _boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(oj, "_append_jsonl_locked", _boom)
    res = oj.append_decision_journal({"decision": "hold"}, source="position",
                                     event_type="management_check", journal_path=jp)
    assert res["status"] == "error" and res["event"] is None


def test_decision_dict_normalized_to_scalar_verb_keeps_detail(tmp_path):
    jp = _journal(tmp_path)
    res = oj.append_decision_journal(
        {"underlying": "SPY", "decision": {"action": "open", "reasons": ["breakout"]}},
        source="controller", event_type="entry_decision", journal_path=jp)
    e = res["event"]
    assert e["decision"] == "enter"                          # 'open' -> canonical 'enter'
    assert e["decision_detail"] == {"action": "open", "reasons": ["breakout"]}


def test_symbol_pulled_from_nested_contract(tmp_path):
    jp = _journal(tmp_path)
    res = oj.append_decision_journal(
        {"contract": {"underlying": "iwm", "strike": 300, "option_type": "call"}},
        source="ingest", event_type="vehicle_score", journal_path=jp)
    assert res["event"]["symbol"] == "IWM"


# --- loose-artifact ingestion (ingest_loose_artifacts) ---------------------------------------

def _write(p, obj):
    import json as _j
    p.write_text(_j.dumps(obj))
    return p


def test_ingest_folds_known_artifacts_and_is_idempotent(tmp_path):
    ddir = tmp_path / "odte"
    ddir.mkdir()
    jp = str(ddir / "decision_journal.jsonl")
    _write(ddir / "controller_event_20260626_0904.json",
           {"ts": "2026-06-26T09:04:00-04:00", "decision": "no_trade", "underlying": "SPY"})
    _write(ddir / "candidate_iwm_300c_20260626_1014.json",
           {"ts": "2026-06-26T10:14:00-04:00", "candidate": {"ticker": "IWM"}})
    _write(ddir / "spy_gamma_map_20260626_1122.json",
           {"ts": "2026-06-26T11:22:00-04:00", "underlying": "SPY"})
    _write(ddir / "unrelated_notes.json", {"hello": "world"})        # not a known pattern -> ignored

    s1 = oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp)
    assert s1["dry_run"] is False
    assert s1["files_scanned"] == 3 and s1["events_appended"] == 3 and s1["duplicates_skipped"] == 0
    rows = oj.read_events(jp)
    assert len(rows) == 3
    # decision verb normalized; symbol pulled from nested candidate; raw path preserved
    ctrl = next(r for r in rows if r["source"] == "ingest:controller")
    assert ctrl["decision"] == "skip" and ctrl["execution_allowed"] is False
    cand = next(r for r in rows if r["source"] == "ingest:candidate")
    assert cand["symbol"] == "IWM" and cand["raw_artifact_path"].endswith("candidate_iwm_300c_20260626_1014.json")

    # re-run: everything is a duplicate, nothing re-appended (idempotent)
    s2 = oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp)
    assert s2["events_appended"] == 0 and s2["duplicates_skipped"] == 3
    assert len(oj.read_events(jp)) == 3


def test_ingest_changed_artifact_reingests_via_content_hash(tmp_path):
    ddir = tmp_path / "odte"
    ddir.mkdir()
    jp = str(ddir / "decision_journal.jsonl")
    f = _write(ddir / "controller_event_20260626_0904.json",
               {"ts": "2026-06-26T09:04:00-04:00", "decision": "wait"})
    oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp)
    # content changes (decision flips) -> new content hash -> re-ingested as a new record
    _write(f, {"ts": "2026-06-26T09:04:00-04:00", "decision": "veto"})
    s = oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp)
    assert s["events_appended"] == 1
    assert len(oj.read_events(jp)) == 2


def test_ingest_dry_run_does_not_mutate_journal(tmp_path):
    ddir = tmp_path / "odte"
    ddir.mkdir()
    jp = str(ddir / "decision_journal.jsonl")
    _write(ddir / "event_no_trade_20260626_1115.json", {"ts": "2026-06-26T11:15:00-04:00"})
    s = oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp, dry_run=True)
    # dry_run: events_appended stays 0; events_would_append reports what WOULD be folded in.
    assert s["dry_run"] is True and s["files_scanned"] == 1
    assert s["events_appended"] == 0 and s["events_would_append"] == 1
    assert oj.read_events(jp) == [], "dry-run must not write to the journal"


def test_ingest_date_filter_from_filename_without_ts(tmp_path):
    """Artifacts with NO `ts` must still be day-filtered by their FILENAME date, not bucketed to
    today (the bug the reviewer flagged)."""
    ddir = tmp_path / "odte"
    ddir.mkdir()
    jp = str(ddir / "decision_journal.jsonl")
    _write(ddir / "market_snapshot_20260626_0931.json", {"underlying": "SPY"})   # no ts
    _write(ddir / "market_snapshot_2026_06_25_0931.json", {"underlying": "QQQ"})  # no ts, other day
    (ddir / "controller_event_bad.json").write_text("{not json")
    s = oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp, trade_date="2026-06-26")
    assert s["events_appended"] == 1            # only the 06-26 filename-dated snapshot
    assert s["errors"] == 1                     # malformed file counted, not fatal
    assert oj.read_events(jp)[0]["trade_date"] == "2026-06-26"


def test_ingest_nested_nvda_is_restricted_and_not_executable(tmp_path):
    """A loose artifact whose ticker is only nested (contract.underlying=NVDA) must still be tagged
    restricted and forced non-executable, even if the payload claims execution_allowed=True."""
    ddir = tmp_path / "odte"
    ddir.mkdir()
    jp = str(ddir / "decision_journal.jsonl")
    _write(ddir / "candidate_nvda_120c_20260626_1014.json",
           {"ts": "2026-06-26T10:14:00-04:00", "execution_allowed": True,
            "contract": {"underlying": "NVDA", "strike": 120, "option_type": "call"}})
    oj.ingest_loose_artifacts(data_dir=str(ddir), journal_path=jp)
    e = oj.read_events(jp)[0]
    assert e["symbol"] == "NVDA"
    assert e["restricted"] is True and e["restricted_reason"] == "employer"
    assert e["execution_allowed"] is False
    assert e["raw_execution_allowed"] is True   # original value preserved for audit


# --- additive day packet (build_day_packet) --------------------------------------------------

def test_build_day_packet_routes_streams_and_is_idempotent(tmp_path):
    ddir = tmp_path / "odte"
    ddir.mkdir()
    jp = str(ddir / "decision_journal.jsonl")
    td = "2026-06-26"
    # seed a mix of standardized events for the day
    oj.append_decision_journal({"underlying": "SPY", "ts": f"{td}T09:31:00-04:00"},
                               source="ingest:market_snapshot", event_type="market_snapshot", journal_path=jp)
    oj.append_decision_journal({"candidate": {"ticker": "IWM"}, "ts": f"{td}T10:14:00-04:00"},
                               source="ingest:candidate", event_type="candidate", journal_path=jp)
    oj.append_decision_journal({"contract": {"underlying": "SPY"}, "ts": f"{td}T10:20:00-04:00"},
                               source="ingest:vehicle_score", event_type="vehicle_score", journal_path=jp)
    oj.append_decision_journal({"underlying": "IWM", "ts": f"{td}T11:00:00-04:00",
                                "decision": {"action": "open"}},
                               source="controller", event_type="entry_decision", journal_path=jp)
    oj.append_decision_journal({"underlying": "SPY", "ts": f"{td}T09:00:00-04:00", "decision": "no_trade"},
                               source="controller", event_type="controller_event", journal_path=jp)
    # event from a DIFFERENT day must not leak into this packet
    oj.append_decision_journal({"underlying": "QQQ", "ts": "2026-06-25T10:00:00-04:00"},
                               source="ingest:candidate", event_type="candidate", journal_path=jp)

    s = oj.build_day_packet(trade_date=td, journal_path=jp, out_root=str(ddir))
    assert s["events_written"] == 5            # the 06-25 event excluded
    assert s["files"]["market_snapshots.jsonl"] == 1
    assert s["files"]["candidates.jsonl"] == 1
    assert s["files"]["vehicle_scores.jsonl"] == 1
    assert s["files"]["trades.jsonl"] == 1
    assert s["files"]["controller_events.jsonl"] == 1
    root = ddir / "days" / td
    assert (root / "postmortem.md").exists()
    cand_lines = (root / "candidates.jsonl").read_text().strip().splitlines()
    assert len(cand_lines) == 1 and "IWM" in cand_lines[0]

    # postmortem edits are preserved on rebuild; streams are regenerated (idempotent)
    (root / "postmortem.md").write_text("# my notes")
    s2 = oj.build_day_packet(trade_date=td, journal_path=jp, out_root=str(ddir))
    assert s2["events_written"] == 5
    assert (root / "postmortem.md").read_text() == "# my notes"


def test_build_day_packet_fail_safe_on_bad_root(monkeypatch, tmp_path):
    jp = str(tmp_path / "decision_journal.jsonl")
    oj.append_decision_journal({"underlying": "SPY"}, source="x", event_type="note", journal_path=jp)

    def _boom(*a, **k):
        raise OSError("nope")
    monkeypatch.setattr(oj.Path, "mkdir", _boom)
    s = oj.build_day_packet(trade_date="2026-06-26", journal_path=jp, out_root=str(tmp_path / "d"))
    assert "error" in s and s["events_written"] == 0       # never raises


# --- self-eval / process quality (summarize.process_quality) ---------------------------------

def _closed(jp, tid, realized, mfe=None, violations=None, loss_category=None, diagnosis=None):
    ev = {"event_type": "order_closed", "trade_id": tid, "mode": "scalp",
          "underlying": "SPY", "realized_pnl": realized}
    if mfe is not None:
        ev["mfe"] = mfe
    if violations:
        ev["rule_violations"] = violations
    if loss_category:
        ev["loss_category"] = loss_category
    if diagnosis:
        ev["diagnosis"] = diagnosis
    oj.append_event(ev, journal_path=jp)


def test_process_quality_separates_process_from_outcome(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "w_clean", 20.0, mfe=22.0)                       # good process, good outcome (clean win)
    _closed(jp, "l_clean", -15.0, mfe=0.0)                       # good process, bad outcome; thesis wrong
    _closed(jp, "lucky", 10.0, mfe=12.0, violations=["no_stop"])  # bad process, lucky win
    pq = oj.summarize(oj.read_events(jp))["process_quality"]
    assert pq["n_diagnosed"] == 3
    assert pq["process_outcome"]["good_process_good_outcome"] == 1
    assert pq["process_outcome"]["good_process_bad_outcome"] == 1
    assert pq["process_outcome"]["bad_process_lucky_outcome"] == 1


def test_execution_diagnosis_from_mfe(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "clean", 20.0, mfe=22.0)            # kept most -> clean_win
    _closed(jp, "gaveback_win", 4.0, mfe=30.0)      # won but captured <half -> good_entry_bad_exit
    _closed(jp, "roundtrip", -5.0, mfe=18.0)        # was green, ended red -> good_thesis_bad_exit
    _closed(jp, "wrong", -12.0, mfe=0.0)            # never favorable -> thesis_wrong
    diag = oj.summarize(oj.read_events(jp))["process_quality"]["execution_diagnosis"]
    assert diag["clean_win"] == 1 and diag["good_entry_bad_exit"] == 1
    assert diag["good_thesis_bad_exit"] == 1 and diag["thesis_wrong"] == 1


def test_loss_categories_use_explicit_tags_only(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "l1", -10.0, mfe=0.0, loss_category="vehicle")
    _closed(jp, "l2", -8.0, mfe=0.0)                # loser with no tag -> uncategorized
    _closed(jp, "win", 5.0, mfe=6.0)               # winners never counted as a loss cause
    pq = oj.summarize(oj.read_events(jp))["process_quality"]
    assert pq["loss_categories"] == {"vehicle": 1, "uncategorized": 1}


def test_explicit_diagnosis_field_is_respected(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "t1", -3.0, mfe=9.0, diagnosis="good_signal_bad_vehicle")
    diag = oj.summarize(oj.read_events(jp))["process_quality"]["execution_diagnosis"]
    assert diag.get("good_signal_bad_vehicle") == 1   # explicit overrides the mfe heuristic


def test_process_quality_renders_in_markdown_report(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "lucky", 10.0, mfe=12.0, violations=["no_stop"])
    md = oj.build_report(journal_path=jp)["markdown"]
    assert "Process quality & loss diagnosis" in md
    assert "bad process lucky outcome" in md
