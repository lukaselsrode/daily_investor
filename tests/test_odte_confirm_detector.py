"""Confirm detector (2026-08-18 latency fix) — detection, poke rail, transition journaling,
cooldown, and the fail-open contract. Design: data/odte/reports/entry_signal_autopsy_2026-08-18.md
+ the approved latency plan (median 297s tick-spacing gap → seconds)."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import execution.odte_confirm_detector as cd

# 15:00Z = 11:00 ET — ORB formed, pre-midday-fence.
NOW = datetime(2026, 8, 19, 15, 0, 0, tzinfo=timezone.utc)


def _tape(aligned=True, orb=True):
    """Daemon-tape shape: nested + flat mirrors, generated_at stamped (like odte_tape)."""
    state = "above" if aligned else "inside"
    t = {"generated_at": NOW.isoformat(), "minutes_to_close": 240.0, "gap_pct": 0.1}
    for sym in ("SPY", "QQQ", "IWM"):
        blk = {"last": 100.0, "above_vwap": aligned}
        if orb:
            blk["orb_state"] = state
        t[sym] = blk
        t[f"{sym.lower()}_above_vwap"] = aligned
        if orb:
            t[f"{sym.lower()}_orb_state"] = state
    t["VIXY"] = {"above_vwap": False, "change_pct": -2.0}
    t["vixy_change_pct"] = -2.0
    return t


def _detector(tmp_path, cooldown=90.0, **kw):
    calls = []
    det = cd.ConfirmDetector(tmp_path, journal_path=str(tmp_path / "journal.jsonl"),
                             poke_cooldown_seconds=cooldown,
                             run_cmd=calls.append, **kw)
    det._controller_pokeable = lambda: True          # tests never read the real jobs.json
    return det, calls


def _seed_candidate(tmp_path, synthetic=False):
    cand = {"ticker": "QQQ", "direction": "bullish",
            "created_at": (NOW - timedelta(minutes=3)).isoformat()}
    if synthetic:
        cand = {"ticker": "SPY", "direction": "bullish", "source": "market_scorecard",
                "created_at": (NOW - timedelta(minutes=3)).isoformat()}
    (tmp_path / "active_candidate.json").write_text(json.dumps(cand))
    return cand


def _events(tmp_path):
    p = tmp_path / "journal.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_confirm_pokes_once_then_cooldown_repokes(tmp_path):
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    det.step(_tape(), NOW)
    assert len(calls) == 1
    chain = calls[0]
    # Ordered chain: notepad set BEFORE cron run, absolute hermes path, right job id.
    assert chain[0][:5] == [cd.HERMES_BIN, "cron", "notepad", cd.CONTROLLER_JOB_ID, "set"]
    assert chain[0][5] == cd.NOTEPAD_KEY
    assert chain[1] == [cd.HERMES_BIN, "cron", "run", cd.CONTROLLER_JOB_ID]
    # Inside cooldown: held confirm does NOT re-poke.
    det.step(_tape(), NOW + timedelta(seconds=30))
    assert len(calls) == 1
    # Past cooldown with the confirm still true: re-poke (covers the held-claim race).
    det.step(_tape(), NOW + timedelta(seconds=91))
    assert len(calls) == 2


def test_synthetic_ready_pokes_via_empty_candidate_lane(tmp_path):
    # A scorecard synthetic can never confirm itself (forced KEEP_WATCHING) — the detector's
    # SECOND evaluation with an empty candidate proves the tape is confirm-ready.
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path, synthetic=True)
    det.step(_tape(), NOW)
    assert len(calls) == 1
    note = json.loads(calls[0][0][6])
    assert note["kind"] == "synthetic_ready"
    assert note["ts"] == NOW.isoformat(timespec="seconds")


def test_unaligned_tape_never_pokes(tmp_path):
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    det.step(_tape(aligned=False), NOW)
    assert calls == []


def test_no_orb_keys_pre_1000_never_pokes(tmp_path):
    # Before 10:00 ET the tape omits orb_state entirely — confirmation is structurally
    # impossible and the detector must stay silent.
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    det.step(_tape(orb=False), NOW - timedelta(hours=1, minutes=30))
    assert calls == []


def test_pause_file_suppresses_poke(tmp_path):
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    (tmp_path / cd.PAUSE_FILENAME).write_text("")
    det.step(_tape(), NOW)
    assert calls == []


def test_broker_blocked_suppresses_poke(tmp_path):
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    (tmp_path / "broker_health.json").write_text(json.dumps({"blocked": True}))
    det.step(_tape(), NOW)
    assert calls == []                                   # BROKER_BLOCKED decision, no confirm


def test_transitions_journal_once_not_per_tick(tmp_path):
    # N held ticks journal TRANSITIONS, never per-tick appends. When first-sighting and ready
    # land in the same second the journal's identity dedupe (source/type/symbol/decision/ts)
    # collapses them to one event — honest; the invariant is NO GROWTH while the state holds.
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    det.step(_tape(), NOW)
    after_first_tick = len(_events(tmp_path))
    assert 2 <= after_first_tick <= 3                    # 1-2 evaluations + 1 poke event
    for i in range(1, 10):
        det.step(_tape(), NOW + timedelta(seconds=3 * i))
    evs = _events(tmp_path)
    assert len(evs) == after_first_tick                  # held stream added NOTHING
    evals = [e for e in evs if e.get("event_type") == "candidate_evaluation"]
    pokes = [e for e in evs if e.get("event_type") == "confirm_detector_poke"]
    assert evals[0].get("detector_transition") == "first_sighting"
    assert len(pokes) == 1
    assert all(e.get("scan_only") is True for e in evals)


def test_first_sighting_moves_the_freshness_clock(tmp_path):
    det, _ = _detector(tmp_path)
    _seed_candidate(tmp_path)
    det.step(_tape(), NOW)
    from data.odte_journal import first_signal_age_minutes
    evs = _events(tmp_path)
    age = first_signal_age_minutes(evs, "QQQ", "bullish", now=NOW + timedelta(minutes=7))
    assert age == 7.0                                    # clock starts at daemon detection


def test_state_survives_restart_no_double_poke(tmp_path):
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    det.step(_tape(), NOW)
    assert len(calls) == 1
    # Daemon crash + launchd restart: fresh instance, same state dir, confirm still true.
    det2, calls2 = _detector(tmp_path)
    det2.step(_tape(), NOW + timedelta(seconds=20))       # inside the original cooldown
    assert calls2 == []


def test_step_swallows_errors(tmp_path, monkeypatch):
    det, calls = _detector(tmp_path)
    _seed_candidate(tmp_path)
    monkeypatch.setattr(cd, "evaluate_candidate_watch",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    det.step(_tape(), NOW)
    assert det.errors == 1 and calls == []


def test_shadow_mode_never_fires_the_real_poke(tmp_path):
    # SHADOW PURITY: live=False journals the would-poke but never runs the chain — a shadow
    # daemon rehearses, it does not wake the real controller.
    det, calls = _detector(tmp_path)
    det.live = False
    _seed_candidate(tmp_path)
    det.step(_tape(), NOW)
    assert calls == []
    pokes = [e for e in _events(tmp_path) if e.get("event_type") == "confirm_detector_poke"]
    assert len(pokes) == 1 and pokes[0].get("would_poke") is True


def test_shadow_daemon_routes_detector_to_shadow_journal(tmp_path, monkeypatch):
    import data.odte_config as oc
    import execution.odte_fast_lane as fl

    class _Client:
        pass
    (tmp_path / "fast_lane_stage.json").write_text(json.dumps({"stage": "shadow"}))
    monkeypatch.setattr(oc, "CONFIRM_DETECTOR_ENABLED", True)
    d = fl.FastLaneDaemon(_Client(), account_number="435050133", state_dir=tmp_path)
    assert d.confirm_detector.live is False
    assert d.confirm_detector.journal_path == d.shadow_journal_path


def test_daemon_wires_detector_behind_flag(tmp_path, monkeypatch):
    import data.odte_config as oc
    import execution.odte_fast_lane as fl

    class _Client:
        pass
    (tmp_path / "fast_lane_stage.json").write_text(json.dumps({"stage": "shadow"}))
    monkeypatch.setattr(oc, "CONFIRM_DETECTOR_ENABLED", True)
    d = fl.FastLaneDaemon(_Client(), account_number="435050133", state_dir=tmp_path)
    assert d.confirm_detector is not None
    monkeypatch.setattr(oc, "CONFIRM_DETECTOR_ENABLED", False)
    d2 = fl.FastLaneDaemon(_Client(), account_number="435050133", state_dir=tmp_path)
    assert d2.confirm_detector is None


@pytest.mark.halt_posture
def test_live_posture_arms_the_confirm_detector():
    """The resume's entry-latency rail: OFF silently reverts confirm detection to the */5 LLM
    cadence (median 297s measured). Update only with an operator decision."""
    import data.odte_config as oc
    assert oc.CONFIRM_DETECTOR_ENABLED is True
    assert oc.CONFIRM_POKE_COOLDOWN_SECONDS == 90.0


def test_stale_move_is_parked_never_poked(tmp_path, monkeypatch):
    # 2026-08-19 live poke-storm: a confirm-ready move past the freshness limit re-poked every
    # cooldown — and candidate TTL re-seeds minted FRESH detector clocks that kept the storm
    # going under new keys. The detector now uses the GATE'S journal clock
    # (first_signal_age_minutes), which spans re-seeds: one ready_but_stale transition, zero
    # pokes, parked for the session.
    import data.odte_config as oc
    monkeypatch.setattr(oc, "MAX_SIGNAL_AGE_MINUTES", 20.0)
    det, calls = _detector(tmp_path, cooldown=0.0)
    # The tape-lane confirm resolves to SPY; the journal carries SPY-bullish history from 25
    # minutes ago (the controller's scan telemetry in production).
    (tmp_path / "active_candidate.json").write_text(json.dumps(
        {"ticker": "SPY", "direction": "bullish",
         "created_at": (NOW - timedelta(minutes=3)).isoformat()}))
    history = [{"event_type": "candidate_evaluation", "symbol": "SPY", "direction": "bullish",
                "checks": {"underlying_orb_state": "above"},
                "ts": (NOW - timedelta(minutes=25)).isoformat()}]
    for i in range(5):
        det.step(_tape(), NOW + timedelta(seconds=3 * i), events=history)
    assert calls == []                                       # stale from the first look
    evs = _events(tmp_path)
    stale = [e for e in evs if e.get("detector_transition") == "ready_but_stale"]
    # Same-second identity dedupe may fold the stale event into the first-sighting append; the
    # park is STATE-based either way and the phase is visible in the heartbeat summary.
    assert len(stale) <= 1
    move = next(iter((det.state.get("moves") or {}).values()))
    assert move["phase"] == "stale"


def test_fresh_journal_clock_still_pokes(tmp_path, monkeypatch):
    # The journal clock must not over-block: a move whose first signal is recent pokes normally.
    import data.odte_config as oc
    monkeypatch.setattr(oc, "MAX_SIGNAL_AGE_MINUTES", 20.0)
    det, calls = _detector(tmp_path)
    (tmp_path / "active_candidate.json").write_text(json.dumps(
        {"ticker": "SPY", "direction": "bullish",
         "created_at": (NOW - timedelta(minutes=3)).isoformat()}))
    history = [{"event_type": "candidate_evaluation", "symbol": "SPY", "direction": "bullish",
                "checks": {"underlying_orb_state": "above"},
                "ts": (NOW - timedelta(minutes=8)).isoformat()}]
    det.step(_tape(), NOW, events=history)
    assert len(calls) == 1


def test_detached_chain_refuses_to_spawn_under_pytest(monkeypatch):
    # 2026-08-19 incident: suite runs fired REAL pokes at the REAL controller job. The default
    # spawner must be inert whenever pytest is in the environment — this test runs under pytest
    # by construction, so a spawn here would be the bug itself.
    spawned = []
    monkeypatch.setattr(cd.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    cd._detached_chain([["echo", "poke"]])
    assert spawned == []
