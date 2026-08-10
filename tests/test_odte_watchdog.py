"""tests/test_odte_watchdog.py — the scan/trigger lane's safety invariants and alert state machine.

`odte_watchdog` had ZERO test coverage while running every minute (`0dte-watchdog-pulse`,
`*/1 9-15 * * 1-5`) and producing `triggers.json` — the artifact the entire candidate lane is
built from. This suite pins the properties that must never regress:

  * the lane can never authorize execution (`scan_only` / `execution_allowed`),
  * a restricted (employer-blocked) underlying can never become the candidate,
  * a non-executable single name is demoted to context, never parked as the candidate key,
  * the persistence re-alert actually re-fires, which is the 2026-08-02 retune's whole point.

Thresholds are read from the live module constants, never re-hardcoded — a config change must
move these tests, not silently invalidate them.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import data.odte_watchdog as w
from data.social_sentiment import is_restricted_underlying

UTC = timezone.utc
NOW = datetime(2026, 8, 10, 14, 30, tzinfo=UTC)


def _report(*, verdict="OBSERVE", candidate=None, top_chatter=None, intent="neutral"):
    """A minimal build_odte_social_report() shape."""
    return {
        "scorecard": {"verdict": verdict, "confidence": 0.61, "reasons": ["r1", "r2"]},
        "candidate": candidate,
        "spy_trend": {"pct_vs_prev_close": 0.12, "above_vwap": True},
        "social_intent": {"intent": intent, "n_docs": 7},
        "top_chatter": top_chatter or [],
    }


@pytest.fixture()
def wd(tmp_path, monkeypatch):
    """run_watchdog bound to a tmp state dir with a stubbed report (no network, no real journal)."""
    policy = tmp_path / "controller_policy.json"
    policy.write_text(json.dumps({"account": "redacted"}))

    def run(report, *, now=NOW, policy_path=None, state_dir=None):
        import data.social_sentiment as ss
        monkeypatch.setattr(ss, "build_odte_social_report",
                            lambda allow_fetch=True: report, raising=False)
        return w.run_watchdog(state_dir=str(state_dir or tmp_path),
                              policy_path=str(policy if policy_path is None else policy_path),
                              allow_fetch=False, now=now)

    run.tmp = tmp_path
    run.policy = policy
    return run


# ── the invariant that matters most: this lane cannot authorize a trade ──────────────────────

@pytest.mark.parametrize("verdict", ["OBSERVE", "CALL-leaning", "PUT-leaning"])
@pytest.mark.parametrize("policy_ok", [True, False])
@pytest.mark.parametrize("report_error", [None, "boom"])
def test_decision_context_never_authorizes_execution(verdict, policy_ok, report_error):
    ctx = w._decision_context(_report(verdict=verdict), {"ticker": "SPY", "direction": "bullish"},
                              {"verdict": verdict}, policy_ok,
                              "ok" if policy_ok else "missing", report_error)
    assert ctx["scan_only"] is True
    assert ctx["execution_allowed"] is False
    assert ctx["confirmation_needed"] is True
    # the required-confirmation list is non-negotiable, not advisory
    assert set(w._REQUIRED_CONFIRMATIONS).issubset(set(ctx["required_confirmations"]))


def test_payload_carries_the_scan_only_invariants(wd):
    p = wd(_report(verdict="CALL-leaning"))
    assert p["scan_only"] is True
    assert p["execution_allowed"] is False
    assert p["decision_context"]["execution_allowed"] is False


# ── restricted underlyings: the employer block ──────────────────────────────────────────────

def test_restricted_symbol_is_actually_restricted():
    """Guard the guard — if this flips, the block below stops proving anything."""
    assert is_restricted_underlying("NVDA") is True


def test_restricted_candidate_never_becomes_the_candidate(wd):
    p = wd(_report(verdict="OBSERVE",
                   candidate={"ticker": "NVDA", "direction": "bullish", "mentions": 99}))
    assert p["candidate"] is None
    state = json.loads((wd.tmp / w.STATE_FILENAME).read_text())
    assert state["candidate_key"] is None


def test_candidate_key_rejects_restricted_and_shapes_the_rest():
    assert w._candidate_key({"ticker": "NVDA", "direction": "bullish"}) is None
    assert w._candidate_key({"ticker": "spy", "direction": "bearish"}) == "SPY:bearish"
    assert w._candidate_key({"ticker": "SPY"}) == "SPY:?"
    assert w._candidate_key(None) is None
    assert w._candidate_key({}) is None


def test_restricted_chatter_is_reported_as_context(wd):
    p = wd(_report(top_chatter=[{"ticker": "NVDA", "restricted": True},
                                {"ticker": "AMD", "restricted": False}]))
    assert p["restricted_chatter"] == ["NVDA"]


# ── executable universe: a single name must never park as the candidate ─────────────────────

def test_non_executable_single_name_is_demoted_to_context(wd):
    single = {"ticker": "AAPL", "direction": "bearish", "mentions": 40}
    p = wd(_report(verdict="OBSERVE", candidate=single))
    assert p["candidate"] is None, "AAPL is unconvertible — it must never be the candidate"
    assert p["single_name_context"] == single


def test_non_executable_single_name_falls_back_to_scorecard_candidate(wd):
    single = {"ticker": "AAPL", "direction": "bearish", "mentions": 40}
    p = wd(_report(verdict="CALL-leaning", candidate=single))
    assert p["candidate"]["ticker"] == "SPY"
    assert p["candidate"]["direction"] == "bullish"
    assert p["single_name_context"] == single


@pytest.mark.parametrize("ticker", list(w.EXECUTABLE_UNIVERSE))
def test_executable_universe_members_are_kept(wd, ticker):
    p = wd(_report(verdict="OBSERVE", candidate={"ticker": ticker, "direction": "bullish"}))
    assert p["candidate"]["ticker"] == ticker
    assert p["single_name_context"] is None


# ── scorecard → synthetic market candidate ──────────────────────────────────────────────────

@pytest.mark.parametrize("verdict,direction", [("CALL-leaning", "bullish"),
                                               ("PUT-leaning", "bearish")])
def test_scorecard_market_candidate_direction_mapping(verdict, direction):
    c = w._scorecard_market_candidate(_report(verdict=verdict))
    assert c["ticker"] == "SPY" and c["direction"] == direction
    assert c["source"] == "market_scorecard"


@pytest.mark.parametrize("verdict", ["OBSERVE", "", "nonsense", None])
def test_scorecard_market_candidate_requires_a_direction(verdict):
    assert w._scorecard_market_candidate(_report(verdict=verdict)) is None


def test_scorecard_market_candidate_survives_junk_subfields():
    """Snapshot shapes drift; a non-dict spy_trend must degrade, not raise."""
    r = _report(verdict="CALL-leaning")
    r["spy_trend"] = "not-a-dict"
    r["social_intent"] = None
    c = w._scorecard_market_candidate(r)
    assert c["observed_market_context"] == {"pct_vs_prev_close": None, "above_vwap": None}


# ── the alert state machine ─────────────────────────────────────────────────────────────────

def test_new_candidate_fires_once_then_goes_quiet_until_the_realert_window(wd):
    p1 = wd(_report(verdict="CALL-leaning"), now=NOW)
    assert p1["alert"] is True
    assert [t["type"] for t in p1["triggers"]] == ["new_candidate"]

    # same candidate, one minute later: no new alert yet
    p2 = wd(_report(verdict="CALL-leaning"), now=NOW + timedelta(minutes=1))
    assert p2["alert"] is False, "unchanged candidate must not re-alert every minute"


def test_persistence_realert_fires_at_the_configured_window(wd):
    wd(_report(verdict="CALL-leaning"), now=NOW)
    later = NOW + timedelta(minutes=w.WATCHDOG_REALERT_MINUTES)
    p = wd(_report(verdict="CALL-leaning"), now=later)
    assert p["alert"] is True
    assert [t["type"] for t in p["triggers"]] == ["candidate_persisting"], (
        "the 2026-08-02 retune exists so an unchanged candidate cannot go silent forever")


def test_candidate_first_seen_is_carried_across_ticks(wd):
    wd(_report(verdict="CALL-leaning"), now=NOW)
    first = json.loads((wd.tmp / w.STATE_FILENAME).read_text())["candidate_first_seen_utc"]
    wd(_report(verdict="CALL-leaning"), now=NOW + timedelta(minutes=3))
    again = json.loads((wd.tmp / w.STATE_FILENAME).read_text())["candidate_first_seen_utc"]
    assert first == again, "candidate age must not reset while the candidate is unchanged"


def test_direction_flip_is_a_new_candidate(wd):
    wd(_report(verdict="CALL-leaning"), now=NOW)
    p = wd(_report(verdict="PUT-leaning"), now=NOW + timedelta(minutes=1))
    assert [t["type"] for t in p["triggers"]] == ["new_candidate"]


def test_missing_policy_raises_a_trigger_and_vetoes(wd, tmp_path):
    p = wd(_report(), policy_path=tmp_path / "does_not_exist.json")
    assert p["policy_ok"] is False
    assert "policy_missing" in [t["type"] for t in p["triggers"]]
    assert "policy_missing" in p["decision_context"]["veto_reasons"]


def test_an_unrelated_persistent_trigger_defers_the_candidate_persistence_realert(wd, tmp_path):
    """DOCUMENTS A REAL COUPLING (2026-08-10 audit), deliberately not changed live.

    `last_alert_utc` is refreshed whenever ANY trigger fires, but the candidate persistence
    re-alert gates on that same field. So a permanently-failing unrelated trigger — a missing
    policy file, a stuck report_error — refreshes the stamp every minute and the
    `candidate_persisting` trigger can never come due.

    Impact is limited because `alert` is still True every tick (the policy trigger fires), so the
    controller is not silenced — but `render_pulse` will name the policy fault and never the live
    candidate, and a consumer filtering on trigger *type* will not see the persistence signal.
    Pinned rather than fixed: changing alert semantics on the live trigger lane during market
    hours is exactly the unforced change that should wait for the close.
    """
    missing = tmp_path / "gone.json"
    wd(_report(verdict="CALL-leaning"), now=NOW, policy_path=missing)
    well_past = NOW + timedelta(minutes=w.WATCHDOG_REALERT_MINUTES * 3)
    # ticks in between keep refreshing last_alert_utc via the policy trigger
    for m in range(1, w.WATCHDOG_REALERT_MINUTES * 3):
        wd(_report(verdict="CALL-leaning"), now=NOW + timedelta(minutes=m), policy_path=missing)
    p = wd(_report(verdict="CALL-leaning"), now=well_past, policy_path=missing)
    types = [t["type"] for t in p["triggers"]]
    assert "policy_missing" in types
    assert "candidate_persisting" not in types, "current (pinned) behaviour — see docstring"
    assert p["alert"] is True, "the controller is still alerted, which bounds the impact"


# ── io contracts ────────────────────────────────────────────────────────────────────────────

def test_read_json_status_contract(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": 1}))
    assert w._read_json(good) == ({"a": 1}, "ok")
    assert w._read_json(tmp_path / "nope.json") == (None, "missing")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert w._read_json(bad) == (None, "invalid")
    notdict = tmp_path / "list.json"
    notdict.write_text("[1,2]")
    assert w._read_json(notdict) == (None, "invalid")


def test_parse_ts_assumes_utc_when_naive():
    assert w._parse_ts(None) is None
    assert w._parse_ts("not-a-date") is None
    assert w._parse_ts("2026-08-10T14:30:00").tzinfo is not None
    assert w._parse_ts("2026-08-10T14:30:00Z") == NOW.replace(minute=30)


def test_journal_is_written_beside_the_state_dir_not_the_repo(wd):
    """A tmp state_dir must never append to the live decision journal."""
    wd(_report(verdict="CALL-leaning"))
    assert (wd.tmp / "decision_journal.jsonl").exists()
    events = [json.loads(x) for x in
              (wd.tmp / "decision_journal.jsonl").read_text().splitlines() if x.strip()]
    assert events and events[-1]["event_type"] == "watchdog_trigger"
    assert events[-1]["scan_only"] is True
    assert events[-1]["execution_allowed"] is False


def test_state_and_triggers_are_valid_json_after_a_run(wd):
    wd(_report(verdict="PUT-leaning"))
    state = json.loads((wd.tmp / w.STATE_FILENAME).read_text())
    trig = json.loads((wd.tmp / w.TRIGGERS_FILENAME).read_text())
    assert state["version"] == w.STATE_VERSION
    assert trig["candidate"]["direction"] == "bearish"
