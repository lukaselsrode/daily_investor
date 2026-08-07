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
    oj.append_decision_journal({"underlying": "IWM", "ts": f"{td}T11:01:00-04:00",
                                "trade_id": "t-iwm", "fill_price": 0.80},
                               source="controller", event_type="entry_fill", journal_path=jp)
    oj.append_decision_journal({"underlying": "SPY", "ts": f"{td}T09:00:00-04:00", "decision": "no_trade"},
                               source="controller", event_type="controller_event", journal_path=jp)
    # event from a DIFFERENT day must not leak into this packet
    oj.append_decision_journal({"underlying": "QQQ", "ts": "2026-06-25T10:00:00-04:00"},
                               source="ingest:candidate", event_type="candidate", journal_path=jp)

    s = oj.build_day_packet(trade_date=td, journal_path=jp, out_root=str(ddir))
    assert s["events_written"] == 6            # the 06-25 event excluded
    assert s["files"]["market_snapshots.jsonl"] == 1
    assert s["files"]["candidates.jsonl"] == 1
    assert s["files"]["vehicle_scores.jsonl"] == 1
    # 2026-08-05 fix: trades.jsonl is trade LIFECYCLE only — the entry_fill lands there; the
    # entry_decision (a gate verdict) now routes to controller_events, never to trades.
    assert s["files"]["trades.jsonl"] == 1
    assert s["files"]["controller_events.jsonl"] == 2
    root = ddir / "days" / td
    assert (root / "postmortem.md").exists()
    cand_lines = (root / "candidates.jsonl").read_text().strip().splitlines()
    assert len(cand_lines) == 1 and "IWM" in cand_lines[0]

    # postmortem edits are preserved on rebuild; streams are regenerated (idempotent)
    (root / "postmortem.md").write_text("# my notes")
    s2 = oj.build_day_packet(trade_date=td, journal_path=jp, out_root=str(ddir))
    assert s2["events_written"] == 6
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


# --- 2026-08-03 postmortem schema (excursion / process_review / explicit grades) ---------------
# The modern postmortem moved the self-eval fields into typed sub-objects. Reading only the legacy
# top-level keys made avg_mfe_capture report 7.7% on a day whose trade captured 100% of its MFE,
# and silently graded a rule-violating trade as good process.

def _modern_postmortem(jp, trade_id, *, realized, capture_pct, mfe_dollars,
                       process_quality="good_process", outcome_quality="good_outcome",
                       failure_layer="none", violations=None, rail="TAKE_PROFIT"):
    """Verbatim shape of the 2026-08-06 QQQ postmortem (structured entry/exit/pnl/excursion/rails)."""
    oj.append_event({"event_type": "entry_fill", "trade_id": trade_id, "mode": "scalp",
                     "underlying": "QQQ", "ts": "2026-08-06T14:10:00-04:00"}, journal_path=jp)
    oj.append_event({"event_type": "exit_fill", "trade_id": trade_id, "mode": "scalp",
                     "underlying": "QQQ", "ts": "2026-08-06T14:16:00-04:00",
                     "rail_fired": rail}, journal_path=jp)
    oj.append_event({"event_type": "postmortem", "trade_id": trade_id, "mode": "scalp",
                     "underlying": "QQQ", "ts": "2026-08-06T14:18:00-04:00",
                     "realized_pnl": realized,
                     "process_quality": process_quality,
                     "outcome_quality": outcome_quality,
                     "failure_layer": failure_layer,
                     "entry": {"tier": "b_plus", "day_regime": "CHOP"},
                     "exit": {"rail_fired": rail},
                     "pnl": {"gross": realized, "estimated_net": realized},
                     "excursion": {"mfe_dollars": mfe_dollars, "mfe_pct": 23.0,
                                   "mae_dollars": -2.0, "mae_pct": -3.0,
                                   "mfe_capture_pct": capture_pct},
                     "rails": {"take_profit_fired": rail == "TAKE_PROFIT",
                               "framework_honored": True},
                     "process_review": {"worked": "rails held",
                                        "rule_violations": list(violations or []),
                                        "lessons": []}},
                    journal_path=jp)


def test_modern_postmortem_excursion_drives_mfe_capture(tmp_path):
    jp = _journal(tmp_path)
    _modern_postmortem(jp, "qqq_720c", realized=13.9, capture_pct=100.0, mfe_dollars=14.0)
    s = oj.summarize(oj.read_events(jp))
    # The postmortem's own capture figure wins: it is measured against the best bid actually seen,
    # not re-derived from a realized/MFE ratio.
    assert s["avg_mfe_capture"] == 1.0
    assert s["n_closed"] == 1 and s["hit_rate"] == 1.0


def test_modern_postmortem_grades_are_used_verbatim(tmp_path):
    jp = _journal(tmp_path)
    _modern_postmortem(jp, "iwm_301c", realized=-0.1, capture_pct=0.0, mfe_dollars=3.0,
                       process_quality="bad_process", outcome_quality="bad_outcome",
                       failure_layer="execution_order_guard_artifact_mapping",
                       violations=["order-guard input omitted top-level option_type"],
                       rail="execution_safety_incident")
    pq = oj.summarize(oj.read_events(jp))["process_quality"]
    assert pq["process_outcome"]["bad_process_bad_outcome"] == 1
    assert pq["failure_layers"] == {"execution_order_guard_artifact_mapping": 1}


def test_nested_rule_violations_are_counted(tmp_path):
    jp = _journal(tmp_path)
    _modern_postmortem(jp, "t1", realized=-1.0, capture_pct=0.0, mfe_dollars=2.0,
                       process_quality="bad_process", outcome_quality="bad_outcome",
                       violations=["no_stop"])
    s = oj.summarize(oj.read_events(jp))
    assert s["n_rule_violations"] == 1 and "no_stop" in s["rule_violations"]


def test_clean_failure_layer_is_not_counted_as_a_failure(tmp_path):
    jp = _journal(tmp_path)
    # "none_normal_execution_slippage" is the author saying nothing failed — a qualified "none".
    _modern_postmortem(jp, "spy_ok", realized=9.0, capture_pct=60.0, mfe_dollars=15.0,
                       failure_layer="none_normal_execution_slippage")
    pq = oj.summarize(oj.read_events(jp))["process_quality"]
    assert pq["failure_layers"] == {}
    assert pq["process_outcome"]["good_process_good_outcome"] == 1


def test_legacy_postmortem_capture_still_derived(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "old", 6.0, mfe=12.0)          # pre-08-03 shape: flat mfe, no excursion block
    s = oj.summarize(oj.read_events(jp))
    assert s["avg_mfe_capture"] == 0.5         # realized/MFE fallback intact


def test_both_eras_average_together(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "old", 6.0, mfe=12.0)                                            # 0.50 derived
    _modern_postmortem(jp, "new", realized=13.9, capture_pct=100.0, mfe_dollars=14.0)  # 1.00 stated
    s = oj.summarize(oj.read_events(jp))
    assert s["avg_mfe_capture"] == 0.75
    assert s["n_closed"] == 2


def test_process_quality_renders_in_markdown_report(tmp_path):
    jp = _journal(tmp_path)
    _closed(jp, "lucky", 10.0, mfe=12.0, violations=["no_stop"])
    md = oj.build_report(journal_path=jp)["markdown"]
    assert "Process quality & loss diagnosis" in md
    assert "bad process lucky outcome" in md


def test_experiments_and_lessons_are_deduped(tmp_path):
    jp = _journal(tmp_path)
    exp = {"event_type": "experiment", "trade_id": "t1", "hypothesis": "Wait for reclaim",
           "promote_if": "better", "kill_if": "worse"}
    oj.append_event(exp, journal_path=jp)
    oj.append_event({**exp, "source": "ingest:event"}, journal_path=jp)
    lesson = {"event_type": "postmortem", "trade_id": "t1", "lessons": ["Harvest target zone"]}
    oj.append_event(lesson, journal_path=jp)
    oj.append_event({**lesson, "ts": "2026-06-30T15:00:00Z"}, journal_path=jp)
    s = oj.summarize(oj.read_events(jp))
    assert len(s["experiments"]) == 1
    assert len(s["lessons"]) == 1


def test_time_bucket_high_action_window_and_markdown(tmp_path):
    jp = _journal(tmp_path)
    oj.append_event({"event_type": "order_filled", "trade_id": "morning", "mode": "scalp",
                     "ts": "2026-06-30T13:45:00+00:00"}, journal_path=jp)  # 09:45 ET
    oj.append_event({"event_type": "order_closed", "trade_id": "morning", "mode": "scalp",
                     "ts": "2026-06-30T13:50:00+00:00", "realized_pnl": 5.0}, journal_path=jp)
    oj.append_event({"event_type": "order_filled", "trade_id": "late", "mode": "scalp",
                     "ts": "2026-06-30T19:45:00+00:00"}, journal_path=jp)  # 15:45 ET
    oj.append_event({"event_type": "order_closed", "trade_id": "late", "mode": "scalp",
                     "ts": "2026-06-30T19:55:00+00:00", "realized_pnl": -2.0}, journal_path=jp)
    res = oj.build_report(journal_path=jp)
    buckets = res["summary"]["by_time_bucket"]
    assert buckets["high_action_0930_1300"]["closed"] == 1
    assert buckets["late_1530_close"]["closed"] == 1
    assert "Time-of-day buckets" in res["markdown"]
    assert "high_action_0930_1300" in res["markdown"]
    assert "Canonical strategy guardrails" in res["markdown"]
    assert "Stale artifacts are never authority" in res["markdown"]


def test_day_packet_generates_postmortem_sections_without_overwriting_human_notes(tmp_path):
    jp = _journal(tmp_path)
    td = "2026-06-30"
    oj.append_event({"event_type": "postmortem", "trade_date": td, "trade_id": "QQQ-T1",
                     "contract": "QQQ 736C", "entry_price": 0.85, "exit_price": 1.14,
                     "realized_pnl_dollars_gross": 29.0,
                     "what_worked": ["Harvested target zone"],
                     "what_failed_or_risked": ["Broker lane stale"],
                     "durable_rule": "Harvest target zone wins"}, journal_path=jp)
    oj.append_event({"event_type": "experiment", "trade_date": td, "trade_id": "QQQ-T1",
                     "hypothesis": "Reclaim/retest after support break",
                     "promote_if": "reduces whipsaws", "kill_if": "worsens losses"}, journal_path=jp)
    out_root = str(tmp_path / "odte")
    s = oj.build_day_packet(trade_date=td, journal_path=jp, out_root=out_root)
    pm = tmp_path / "odte" / "days" / td / "postmortem.md"
    text = pm.read_text()
    assert s["files"]["postmortem.md"] == 1
    assert "## What went well" in text
    assert "Harvested target zone" in text
    assert "## What did not go well / risks" in text
    assert "Broker lane stale" in text
    assert "## Experiments for tomorrow" in text
    assert "Reclaim/retest" in text
    assert "## Canonical strategy guardrails" in text
    assert "stale_artifact_veto" in text

    pm.write_text("# human notes")
    s2 = oj.build_day_packet(trade_date=td, journal_path=jp, out_root=out_root)
    assert pm.read_text() == "# human notes"
    generated = tmp_path / "odte" / "days" / td / "postmortem.generated.md"
    assert s2["files"]["postmortem.generated.md"] == 1
    assert "Harvested target zone" in generated.read_text()


# --- execution-safety events (2026-07-23 delayed-fill remediation) ------------------------------

def test_execution_safety_event_types_are_canonical():
    for et in oj.EXECUTION_SAFETY_EVENT_TYPES:
        assert et in oj.EVENT_TYPES, et
    assert set(oj.EXECUTION_SAFETY_EVENT_TYPES) == {
        "execution_lease_issued", "execution_lease_consumed", "entry_order_pending",
        "entry_order_cancelled_stale", "execution_safety_incident"}


def test_event_from_execution_lease_serializes_policy_and_maximums(tmp_path):
    auth = {"authorized": True, "risk_mode": "PARTIAL_ACCOUNT",
            "policy": {"risk_mode": "PARTIAL_ACCOUNT", "ttl_seconds": 30.0,
                       "max_contracts": 1, "max_debit_fraction": 0.5, "max_premium_loss": None},
            "reason_codes": [],
            "lease": {"lease_id": "abc123", "symbol": "SPY", "direction": "bearish",
                      "option_id": "SPY260723P00737000", "option_type": "put",
                      "strike_price": 737.0, "expiration_date": "2026-07-23", "quantity": 1,
                      "issued_at": "2026-07-23T15:11:17+00:00",
                      "expires_at": "2026-07-23T15:11:47+00:00",
                      "max_limit_price": 1.68, "max_debit": 168.0,
                      "risk_mode": "PARTIAL_ACCOUNT", "max_premium_loss": None,
                      "candidate_fingerprint": "fp1", "market_fingerprint": "fp2"}}
    ev = oj.event_from_execution_lease(auth, trade_id="t1")
    assert ev["event_type"] == "execution_lease_issued" and ev["trade_id"] == "t1"
    assert ev["lease_id"] == "abc123" and ev["underlying"] == "SPY"
    assert ev["max_debit"] == 168.0 and ev["max_limit_price"] == 1.68
    assert ev["policy"]["ttl_seconds"] == 30.0
    assert ev["decision"]["action"] == "allow"
    # The journal RECORD of a lease is never itself execution authority — even adversarially:
    # build_decision_event forces execution_allowed=False for the whole safety-event family.
    ev["execution_allowed"] = True
    jp = str(tmp_path / "decision_journal.jsonl")
    res = oj.append_decision_journal(ev, source="execution_authorize",
                                     event_type="execution_lease_issued", journal_path=jp)
    assert res["status"] == "appended"
    assert res["event"]["execution_allowed"] is False
    ev2 = oj.event_from_execution_lease(auth)
    assert ev2["execution_allowed"] is False


def test_event_from_execution_lease_denied_records_reasons():
    auth = {"authorized": False, "risk_mode": "PARTIAL_ACCOUNT", "lease": None,
            "reason_codes": ["quantity_exceeds_policy", "debit_exceeds_policy"]}
    ev = oj.event_from_execution_lease(auth)
    assert ev["decision"]["action"] == "deny"
    assert "quantity_exceeds_policy" in ev["reason_codes"]
    assert ev["authorized"] is False


def test_execution_safety_lockout_locks_today_only(tmp_path):
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 23, 18, 0, tzinfo=timezone.utc)
    today_incident = {"event_type": "execution_safety_incident", "seq": 1,
                      "ts": (now - timedelta(hours=1)).isoformat(),
                      "underlying": "SPY", "guard_state": "FILLED_WITHOUT_VALID_LEASE"}
    out = oj.execution_safety_lockout([today_incident], now=now)
    assert out["locked"] is True
    assert out["incidents"][0]["underlying"] == "SPY"
    # A prior-day incident does not lock today (ET calendar-day reset).
    old = {**today_incident, "ts": (now - timedelta(days=2)).isoformat()}
    assert oj.execution_safety_lockout([old], now=now)["locked"] is False
    # Non-incident events never lock.
    assert oj.execution_safety_lockout(
        [{"event_type": "entry_order_pending", "ts": now.isoformat()}], now=now)["locked"] is False
    assert oj.execution_safety_lockout([], now=now)["locked"] is False


def test_new_event_types_roundtrip_through_journal(tmp_path):
    jp = str(tmp_path / "decision_journal.jsonl")
    for et in oj.EXECUTION_SAFETY_EVENT_TYPES:
        res = oj.append_decision_journal({"event_type": et, "underlying": "SPY",
                                          "lease_id": f"lease-{et}"},
                                         source="test", event_type=et, journal_path=jp)
        assert res["status"] == "appended", et
    stored = oj.read_events(jp)
    assert [e["event_type"] for e in stored] == list(oj.EXECUTION_SAFETY_EVENT_TYPES)
    # Every stored record stays non-authoritative.
    assert all(e["execution_allowed"] is False for e in stored)


# --- daily trade budget (2026-08-02 retune: 2/day cap + post-close cooldown) --------------------

def test_daily_trade_budget_counts_fills_and_exhausts_at_budget():
    from datetime import datetime, timedelta, timezone

    import data.odte_config as oc
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    fills = [{"event_type": "order_filled", "trade_id": f"t{i}",
              "ts": (now - timedelta(hours=2 + i)).isoformat()}
             for i in range(oc.DAILY_TRADE_BUDGET)]
    b = oj.daily_trade_budget(fills, now=now)
    assert b["trades_today"] == oc.DAILY_TRADE_BUDGET
    assert b["budget"] == oc.DAILY_TRADE_BUDGET
    assert b["remaining"] == 0
    assert b["exhausted"] is True
    under = oj.daily_trade_budget(fills[:-1], now=now)
    assert under["exhausted"] is False and under["remaining"] == 1


def test_daily_trade_budget_dedupes_same_trade_and_ignores_other_days():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    dup = [{"event_type": "order_filled", "trade_id": "t1", "ts": now.isoformat()},
           {"event_type": "order_filled", "trade_id": "t1", "ts": now.isoformat()}]
    assert oj.daily_trade_budget(dup, now=now)["trades_today"] == 1
    yesterday = [{"event_type": "order_filled", "trade_id": "t9",
                  "ts": (now - timedelta(days=1)).isoformat()}]
    assert oj.daily_trade_budget(yesterday, now=now)["trades_today"] == 0


def test_daily_trade_budget_cooldown_window():
    from datetime import datetime, timedelta, timezone

    import data.odte_config as oc
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    closed = [{"event_type": "order_closed", "trade_id": "t1", "realized_pnl": -5.0,
               "ts": (now - timedelta(minutes=1)).isoformat()}]
    hot = oj.daily_trade_budget(closed, now=now)
    assert hot["cooldown_active"] is True
    assert hot["cooldown_until"] is not None
    cold = oj.daily_trade_budget(closed, now=now + timedelta(minutes=oc.REENTRY_COOLDOWN_MINUTES))
    assert cold["cooldown_active"] is False


# --- weekly telemetry + zero-trade tripwire (2026-08-02 retune) ---------------------------------

def test_weekly_telemetry_counts_funnel_and_fires_tripwire():
    from datetime import datetime, timedelta, timezone

    import data.odte_config as oc
    # 2026-07-31 was a Friday: the tripwire weekday (default Wed) has long passed.
    now = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
    events = [
        {"event_type": "entry_decision", "decision": "enter", "execution_allowed": True,
         "ts": (now - timedelta(hours=5)).isoformat()},
        {"event_type": "execution_lease_issued", "authorized": False, "decision": "deny",
         "reason_codes": ["market_snapshot_stale", "broker_snapshot_stale"],
         "ts": (now - timedelta(hours=5)).isoformat()},
        {"event_type": "no_trade_decision", "stage": "entry_gate",
         "reason_codes": ["final_confirmation_budget_check_failed"],
         "ts": (now - timedelta(days=1)).isoformat()},
        # last ISO week: must not count
        {"event_type": "order_filled", "trade_id": "old", "ts": (now - timedelta(days=8)).isoformat()},
    ]
    wk = oj.weekly_telemetry(events, now=now)
    assert wk["trades_this_week"] == 0
    assert wk["gates_passed"] == 1
    # 2026-08-05: lease_refusals totals refusals across ALL stages (1 lease-stage denial +
    # 1 entry_gate no_trade_decision) — the lease-only scalar read 0 on an 18-refusal week.
    assert wk["lease_refusals"] == 2
    assert wk["leases_issued"] == 0
    assert wk["no_trade_decisions"] == 1
    assert ("authorize:market_snapshot_stale", 1) in wk["top_refusal_reasons"]
    # 2026-08-04: v5 refusals (no_trade_decision with stage) must be tallied too.
    assert wk["refusals_by_stage"].get("entry_gate", 0) >= 0    # key present
    assert wk["refusals_by_stage"] == {"entry_gate": 1}
    assert ("entry_gate:final_confirmation_budget_check_failed", 1) in wk["top_refusal_reasons"]
    assert wk["weekly_target"] == [oc.WEEKLY_TRADE_TARGET_MIN, oc.WEEKLY_TRADE_TARGET_MAX]
    assert wk["tripwire"]["armed"] is True
    assert wk["tripwire"]["fired"] is True, "a Friday with zero trades must fire the tripwire"


def test_weekly_telemetry_tripwire_quiet_early_week_or_with_trades():
    from datetime import datetime, timedelta, timezone
    monday = datetime(2026, 7, 27, 15, 0, tzinfo=timezone.utc)
    wk = oj.weekly_telemetry([], now=monday)
    assert wk["tripwire"]["armed"] is False and wk["tripwire"]["fired"] is False
    friday = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
    traded = [{"event_type": "order_filled", "trade_id": "t1",
               "ts": (friday - timedelta(days=2)).isoformat()}]
    wk2 = oj.weekly_telemetry(traded, now=friday)
    assert wk2["trades_this_week"] == 1
    assert wk2["tripwire"]["fired"] is False


# --- winning-tier helper (2026-08-03 green re-entry auto-arm) -----------------------------------

def _green_day(now, tier="a_plus", trade_id="t1", with_tier_event=True):
    from datetime import timedelta
    events = []
    if with_tier_event:
        events.append({"event_type": "execution_lease_issued", "trade_id": trade_id,
                       "underlying": "SPY", "option_id": "spy-756c", "tier": tier,
                       "authorized": True, "ts": (now - timedelta(hours=2)).isoformat()})
    events.append({"event_type": "order_filled", "trade_id": trade_id, "underlying": "SPY",
                   "option_id": "spy-756c", "ts": (now - timedelta(hours=2)).isoformat()})
    events.append({"event_type": "order_closed", "trade_id": trade_id, "underlying": "SPY",
                   "option_id": "spy-756c", "realized_pnl": 9.0,
                   "ts": (now - timedelta(hours=1)).isoformat()})
    return events


def test_green_day_winning_tier_joins_by_trade_id():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc)
    wt = oj.green_day_winning_tier(_green_day(now, tier="a_plus"), now=now)
    assert wt["winning_tier"] == "a_plus"
    assert wt["winning_rank"] == oj.TIER_RANK["a_plus"]
    assert wt["trades"][0]["tier_source"] == "trade_id"


def test_green_day_winning_tier_option_id_and_underlying_fallbacks():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc)
    # tier event has no trade_id -> joins on option_id
    ev = _green_day(now, with_tier_event=False)
    ev.insert(0, {"event_type": "entry_decision", "option_id": "spy-756c", "tier": "b_plus",
                  "ts": (now - timedelta(hours=2)).isoformat()})
    wt = oj.green_day_winning_tier(ev, now=now)
    assert wt["winning_tier"] == "b_plus" and wt["trades"][0]["tier_source"] == "option_id"
    # no id match at all -> same-underlying fallback
    ev2 = _green_day(now, with_tier_event=False)
    ev2.insert(0, {"event_type": "entry_decision", "underlying": "SPY", "tier": "full",
                   "ts": (now - timedelta(hours=2)).isoformat()})
    wt2 = oj.green_day_winning_tier(ev2, now=now)
    assert wt2["winning_tier"] == "full" and wt2["trades"][0]["tier_source"] == "underlying"


def test_green_day_winning_tier_defaults_full_for_legacy_journals():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc)
    wt = oj.green_day_winning_tier(_green_day(now, with_tier_event=False), now=now)
    assert wt["winning_tier"] == "full"
    assert wt["winning_rank"] == oj.TIER_RANK["full"]
    assert wt["trades"][0]["tier_source"] == "default_full"


def test_green_day_winning_tier_none_without_green_trades():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 3, 18, 0, tzinfo=timezone.utc)
    assert oj.green_day_winning_tier([], now=now)["winning_tier"] is None
    assert oj.tier_rank(None) == 0 and oj.tier_rank("bogus") == 0


def test_entry_gate_and_lease_events_carry_tier():
    gate = {"symbol": "SPY", "decision": "enter", "tier": "b_plus", "sizing_tier": "half"}
    e = oj.event_from_entry_gate(gate)
    assert e["tier"] == "b_plus" and e["sizing_tier"] == "half"
    auth = {"authorized": True, "lease": {"symbol": "SPY", "lease_id": "x", "tier": "a_plus",
                                          "anchor_quote": 0.63}}
    le = oj.event_from_execution_lease(auth)
    assert le["tier"] == "a_plus" and le["anchor_quote"] == 0.63


# --- LIVE fill vocabulary (2026-08-04 regression) -----------------------------------------------
# The live controller journals `entry_fill`/`exit_fill`; the repo schema originally said
# `order_filled`. An order_filled-only counter read ZERO trades on 2026-08-03 (a real traded day),
# which would have left the daily budget cap and the zero-trade tripwire non-functional.

def _live_vocab_day(now, trade_id="spy-20260803-756c-scalp-114245"):
    from datetime import timedelta
    return [
        {"event_type": "execution_lease_issued", "trade_id": trade_id, "underlying": "SPY",
         "option_id": "opt-756c", "tier": "a_plus", "authorized": True,
         "ts": (now - timedelta(hours=3)).isoformat()},
        {"event_type": "entry_fill", "trade_id": trade_id, "underlying": "SPY",
         "option_id": "opt-756c", "ts": (now - timedelta(hours=3)).isoformat()},
        {"event_type": "exit_fill", "trade_id": trade_id, "underlying": "SPY",
         "option_id": "opt-756c", "ts": (now - timedelta(hours=2)).isoformat()},
        {"event_type": "order_closed", "trade_id": trade_id, "underlying": "SPY",
         "option_id": "opt-756c", "realized_pnl": 9.0,
         "ts": (now - timedelta(hours=2)).isoformat()},
    ]


def test_daily_budget_counts_live_entry_fill_vocabulary():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
    b = oj.daily_trade_budget(_live_vocab_day(now), now=now)
    assert b["trades_today"] == 1, "an entry_fill IS a trade — the budget must decrement"
    assert b["remaining"] == b["budget"] - 1
    assert b["cooldown_active"] is False          # closed 2h ago


def test_weekly_telemetry_counts_live_entry_fill_vocabulary():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
    wk = oj.weekly_telemetry(_live_vocab_day(now), now=now)
    assert wk["trades_this_week"] == 1
    assert wk["tripwire"]["fired"] is False, "a traded week must not fire the zero-trade tripwire"


def test_winning_tier_joins_live_entry_fill_vocabulary():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
    wt = oj.green_day_winning_tier(_live_vocab_day(now), now=now)
    assert wt["winning_tier"] == "a_plus"
    assert wt["trades"][0]["tier_source"] != "default_full"


def test_both_fill_spellings_dedupe_to_one_trade():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
    events = _live_vocab_day(now) + [
        {"event_type": "order_filled", "trade_id": "spy-20260803-756c-scalp-114245",
         "option_id": "opt-756c", "ts": (now - timedelta(hours=3)).isoformat()}]
    assert oj.daily_trade_budget(events, now=now)["trades_today"] == 1


# --- 2026-08-06: sanctioned incident adjudication (human-only, named, same-day) ---------------

def test_adjudication_unlocks_only_the_named_same_day_incident():
    from datetime import datetime, timezone
    now = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    incident = {"event_type": "execution_safety_incident", "event_id": "ev-abc", "seq": 5937,
                "ts": "2026-08-06T14:14:54+00:00", "underlying": "IWM",
                "reason_codes": ["broker order disagrees with the lease: option_type_mismatch"],
                "guard_state": "BROKER_MISMATCH_BLOCKED"}
    adjudication = {"event_type": "execution_safety_incident_adjudicated",
                    "incident_event_id": "ev-abc", "ts": "2026-08-06T15:30:00+00:00",
                    "adjudicated_by": "lukas",
                    "reason": "guard key-collision false positive; contract identity verified"}
    # Named + attributed + same ET day: unlocked, and the record shows WHO and WHY.
    out = oj.execution_safety_lockout([incident, adjudication], now=now)
    assert out["locked"] is False
    assert out["adjudicated"][0]["adjudicated_by"] == "lukas"
    assert out["incidents"] == []
    # Fallback matching by incident_seq also works.
    by_seq = {**adjudication, "incident_event_id": None, "incident_seq": 5937}
    assert oj.execution_safety_lockout([incident, by_seq], now=now)["locked"] is False
    # An adjudication that names the WRONG incident clears nothing.
    wrong = {**adjudication, "incident_event_id": "ev-other"}
    assert oj.execution_safety_lockout([incident, wrong], now=now)["locked"] is True
    # Unattributed or reasonless adjudications clear nothing.
    assert oj.execution_safety_lockout(
        [incident, {**adjudication, "adjudicated_by": ""}], now=now)["locked"] is True
    assert oj.execution_safety_lockout(
        [incident, {**adjudication, "reason": ""}], now=now)["locked"] is True
    # A prior-day adjudication clears nothing (same-ET-day only).
    stale = {**adjudication, "ts": "2026-08-05T15:30:00+00:00"}
    assert oj.execution_safety_lockout([incident, stale], now=now)["locked"] is True
    # A second unadjudicated incident keeps the day locked even with one cleared.
    incident2 = {**incident, "event_id": "ev-def", "seq": 6000,
                 "ts": "2026-08-06T15:40:00+00:00"}
    both = oj.execution_safety_lockout([incident, incident2, adjudication], now=now)
    assert both["locked"] is True and len(both["adjudicated"]) == 1


def test_guard_and_controller_journaling_the_same_fill_counts_once():
    # 2026-08-06 QQQ 720C: the guard journaled order_filled (no trade_id) and the controller
    # journaled entry_fill (with trade_id) for the SAME fill 7 seconds apart — the old
    # trade_id-or-option_id key counted trades_today=3 on a 2-trade day. Same option within the
    # join window = one entry; a later same-contract re-entry still counts separately.
    from datetime import datetime, timezone

    import data.odte_config as oc
    now = datetime(2026, 8, 6, 16, 0, tzinfo=timezone.utc)
    events = [
        {"event_type": "entry_fill", "trade_id": "iwm-20260806-301c", "option_id": "opt-iwm",
         "ts": "2026-08-06T14:14:32+00:00"},
        {"event_type": "order_filled", "trade_id": None, "option_id": "opt-qqq",
         "ts": "2026-08-06T15:25:33+00:00"},                       # guard lane
        {"event_type": "entry_fill", "trade_id": "qqq-20260806-720c", "option_id": "opt-qqq",
         "ts": "2026-08-06T15:25:40+00:00"},                       # controller lane, same fill
    ]
    b = oj.daily_trade_budget(events, now=now)
    assert b["trades_today"] == 2
    assert b["exhausted"] is (2 >= oc.DAILY_TRADE_BUDGET)
    wk = oj.weekly_telemetry(events, now=now)
    assert wk["trades_this_week"] == 2
    # A genuine re-entry on the SAME contract hours later (post-cooldown) counts as a new trade.
    reentry = events + [{"event_type": "entry_fill", "trade_id": "qqq-2-20260806",
                         "option_id": "opt-qqq", "ts": "2026-08-06T18:40:00+00:00"}]
    assert oj.daily_trade_budget(reentry, now=now.replace(hour=19))["trades_today"] == 3


# --- 2026-08-06 EOD: ingest can never mint countable lifecycle events -------------------------

def test_ingest_refuses_lifecycle_event_payloads(tmp_path):
    # A loose artifact whose payload carries a fill event_type is a COPY of something the
    # trading lane already journaled; re-ingesting it doubled the day packet to +$28 on a
    # +$14 day. Ingest now refuses lifecycle types outright.
    import json
    jp = str(tmp_path / "decision_journal.jsonl")
    state = tmp_path
    (state / "event_entry_fill_copy.json").write_text(json.dumps(
        {"event_type": "entry_fill", "trade_id": "t-dup", "option_id": "opt-x",
         "fill_price": 0.64, "ts": "2026-08-06T15:25:40+00:00"}))
    (state / "event_note.json").write_text(json.dumps(
        {"event_type": "controller_event", "note": "benign", "ts": "2026-08-06T15:00:00+00:00"}))
    s = oj.ingest_loose_artifacts(data_dir=str(state), journal_path=jp)
    assert s.get("lifecycle_skipped", 0) == 1
    events = oj.read_events(jp)
    assert not any(e.get("event_type") == "entry_fill" for e in events)
    assert any(e.get("event_type") == "controller_event" for e in events)


def test_ingest_refuses_first_party_telemetry_types(tmp_path):
    # candidate_evaluation and day_score are written first-party at COMPUTATION time. An EOD
    # artifact sweep re-appending the same decision would mint a fresh event_id that dedupe
    # cannot catch, so the series would double-count.
    import json
    jp = str(tmp_path / "decision_journal.jsonl")
    (tmp_path / "candidate_decision_copy.json").write_text(json.dumps(
        {"event_type": "candidate_evaluation", "state": "WATCHING_CONFIRMATION",
         "ts": "2026-08-06T15:00:00+00:00"}))
    (tmp_path / "odte_day_score.json").write_text(json.dumps(
        {"verdict": "CHOP", "score": 1, "ts": "2026-08-06T15:00:00+00:00"}))
    s = oj.ingest_loose_artifacts(data_dir=str(tmp_path), journal_path=jp)
    assert s.get("lifecycle_skipped", 0) == 2
    assert oj.read_events(jp) == []


def test_candidate_evaluation_routes_to_the_candidates_stream(tmp_path):
    # Left unrouted it would land in controller_events and be buried among ~470 rows/day.
    td = "2026-08-06"
    oj.append_event({"event_type": "candidate_evaluation", "trade_date": td, "underlying": "SPY",
                     "ts": f"{td}T14:00:00+00:00", "state": "WATCHING_CONFIRMATION",
                     "checks": {"underlying_orb_state": "inside"}},
                    journal_path=str(tmp_path / "decision_journal.jsonl"))
    s = oj.build_day_packet(td, journal_path=str(tmp_path / "decision_journal.jsonl"),
                            out_root=str(tmp_path / "days"))
    assert s["files"]["candidates.jsonl"] == 1
    assert s["files"].get("controller_events.jsonl", 0) == 0


def test_event_from_candidate_evaluation_carries_checks_verbatim():
    cd = {"state": "WATCHING_CONFIRMATION", "decision": "keep_watching",
          "generated_at": "2026-08-06T14:00:00+00:00",
          "reasons": ["CHOP requires at least B+ ETF confirmation"],
          "candidate": {"ticker": "SPY", "direction": "bullish", "option_id": "SPY-C",
                        "candidate_fingerprint": "fp-1", "tier": "full"},
          "checks": {"underlying_above_vwap": True, "underlying_orb_state": "inside",
                     "confirmations": 2, "confirmers": ["QQQ", "IWM"], "dissenters": [],
                     "vixy_weak": True, "vixy_firming": False, "tier": "b_plus",
                     "minutes_to_close": 180}}
    ev = oj.event_from_candidate_evaluation(cd)
    assert ev["event_type"] == "candidate_evaluation"
    assert ev["checks"] == cd["checks"]           # verbatim — the clause results are the point
    assert ev["underlying"] == "SPY" and ev["candidate_fingerprint"] == "fp-1"
    assert ev["tier"] == "b_plus"                 # the tier the watch would have minted
    assert ev["scan_only"] is True and ev["execution_allowed"] is False


def test_event_from_day_score_carries_headroom():
    payload = {"verdict": "AVOID", "score": -1,
               "components": {"trend": 3, "volatility": 0, "gap": 0, "expected_move": 0, "time": -4},
               "components_supplied": 4, "components_missing": ["expected_move"],
               "max_possible_score": 4, "reasons": ["late session"], "basis": "scorecard",
               "generated_at": "2026-08-06T19:00:00+00:00"}
    ev = oj.event_from_day_score(payload)
    assert ev["event_type"] == "day_score" and ev["verdict"] == "AVOID"
    assert ev["components_supplied"] == 4 and ev["max_possible_score"] == 4
    assert ev["components_missing"] == ["expected_move"]
    assert ev["execution_allowed"] is False


def test_telemetry_events_are_inert_to_summarize():
    # summarize() keys trades on trade_id; telemetry carries none, so it can only ever show up
    # in by_type. Pinned because a countable telemetry event is how P/L drifts.
    events = [{"event_type": "candidate_evaluation", "ts": "2026-08-06T14:00:00+00:00",
               "checks": {"confirmations": 3}},
              {"event_type": "day_score", "ts": "2026-08-06T14:00:00+00:00", "score": 2}]
    s = oj.summarize(events)
    assert s["n_trades"] == 0 and s["n_closed"] == 0 and s["total_realized_pnl"] == 0
    assert s["by_type"]["candidate_evaluation"] == 1 and s["by_type"]["day_score"] == 1


def test_money_counters_ignore_ingest_sourced_fills():
    # Belt and suspenders: even if a lifecycle event slips in under an ingest: source, the
    # budget and green-day counters never count it.
    from datetime import datetime, timezone
    now = datetime(2026, 8, 6, 20, 0, tzinfo=timezone.utc)
    real = [{"event_type": "entry_fill", "trade_id": "t1", "option_id": "opt-a",
             "ts": "2026-08-06T15:25:40+00:00"},
            {"event_type": "order_closed", "trade_id": "t1", "option_id": "opt-a",
             "realized_pnl": 14.0, "ts": "2026-08-06T15:30:00+00:00"}]
    dupes = [{**real[0], "source": "ingest:event", "seq": 900},
             {**real[1], "trade_id": None, "source": "ingest:event", "seq": 901}]
    b = oj.daily_trade_budget(real + dupes, now=now)
    assert b["trades_today"] == 1
    g = oj.green_day_preservation(real + dupes, now=now)
    assert g["net_day_pnl"] == 14.0                             # not 28
