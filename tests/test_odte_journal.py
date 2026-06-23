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


# --- position -> event helper ----------------------------------------------------------------

def test_event_from_position_decision(tmp_path):
    payload = {"decision": "TAKE_PROFIT", "underlying": "SPY", "mode": "scalp", "pnl_pct": 0.62,
               "option_id": "SPY_C", "triggers": [{"type": "TAKE_PROFIT", "detail": "+62% >= 60%"}]}
    ev = oj.event_from_position_decision(payload, trade_id="t9")
    assert ev["event_type"] == "management_check" and ev["trade_id"] == "t9"
    assert ev["triggers"] == ["TAKE_PROFIT"] and ev["decision"]["action"] == "TAKE_PROFIT"
    stored = oj.append_event(ev, journal_path=_journal(tmp_path))   # round-trips through append
    assert stored["seq"] == 0


# --- guardrail: no broker / network / LLM ----------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(oj)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket"):
        assert forbidden not in src, f"odte_journal must not reference {forbidden!r}"
