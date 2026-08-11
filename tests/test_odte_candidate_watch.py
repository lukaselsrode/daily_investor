"""tests/test_odte_candidate_watch.py — pre-entry candidate HAWK lane.

Pure/offline: no broker, network, LLM, or orders. Candidate watch may confirm that a fresh entry gate
should be built, but it must never set execution_allowed=True by itself.
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_breadth as breadth
import data.odte_candidate_watch as cw
import data.odte_config as oc
import data.odte_execution_policy as xp
import data.odte_loop_status as ls

NOW = datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)


def _market(**over):
    m = {
        "day_verdict": "GOOD_DAY",
        "minutes_to_close": 240,
        "SPY": {"last": 741.0, "above_vwap": True, "orb_state": "above"},
        "QQQ": {"last": 724.2, "above_vwap": True, "orb_state": "above"},
        "IWM": {"last": 299.0, "above_vwap": True, "orb_state": "above"},
        "XSP": {"last": 741.0, "above_vwap": True, "orb_state": "above"},
        "VIXY": {"above_vwap": False, "change_pct": -2.0},
    }
    m.update(over)
    return m


def test_bullish_etf_candidate_confirms_from_tape_only():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(), now=NOW
    )
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["execution_allowed"] is False
    assert payload["scan_only"] is True
    assert payload["places_orders"] is False


def test_confirmed_candidate_locks_exact_option_contract_identity() -> None:
    vehicle_score = {
        "verdict": "GOOD_BET",
        "direction": "bullish",
        "contract": {
            "underlying": "QQQ",
            "option_type": "call",
            "option_id": "QQQ260629C00725000",
            "expiration_date": "2026-06-29",
            "strike_price": 725.0,
        },
    }
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"},
        market=_market(),
        vehicle_score=vehicle_score,
        now=NOW,
    )
    candidate = payload["candidate"]
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert candidate["option_id"] == "QQQ260629C00725000"
    assert candidate["expiration_date"] == "2026-06-29"
    assert candidate["strike_price"] == 725.0
    assert candidate["option_type"] == "call"
    assert candidate["selection_timestamp"] == NOW.isoformat()
    assert candidate["candidate_fingerprint"] == xp.candidate_fingerprint(candidate)


def test_contract_switch_starts_new_candidate_cycle() -> None:
    old = {
        "ticker": "QQQ", "direction": "bullish", "selected_vehicle": "QQQ",
        "option_id": "QQQ260629C00724000", "expiration_date": "2026-06-29",
        "strike_price": 724.0, "option_type": "call",
        "selection_timestamp": "2026-06-29T13:50:00+00:00",
    }
    vehicle_score = {
        "verdict": "GOOD_BET", "direction": "bullish",
        "contract": {
            "underlying": "QQQ", "option_type": "call",
            "option_id": "QQQ260629C00725000", "expiration_date": "2026-06-29",
            "strike_price": 725.0,
        },
    }
    payload = cw.evaluate_candidate_watch(old, market=_market(), vehicle_score=vehicle_score, now=NOW)
    candidate = payload["candidate"]
    assert candidate["selection_timestamp"] == NOW.isoformat()
    assert candidate["selection_reason"] == "exact_contract_switch_new_candidate_cycle"
    assert candidate["option_id"] == "QQQ260629C00725000"
    assert candidate["candidate_fingerprint"] == xp.candidate_fingerprint(candidate)


def test_market_only_etf_lane_creates_candidate_without_social():
    payload = cw.evaluate_candidate_watch({}, market=_market(), now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["candidate"]["ticker"] in cw.ETF_UNIVERSE
    assert payload["candidate"]["source"] == "etf_momentum_tape"
    assert payload["execution_allowed"] is False


def test_chop_confirms_b_plus_tier_with_one_dissenter():
    # 2026-08-02 risk-on retune: CHOP no longer demands A+. Enough confirmers with at most
    # B_PLUS_MAX_DISSENTERS definitive dissenters converts at the half-size B+ tier.
    m = _market(day_verdict="CHOP")
    m["IWM"] = {"last": 298.0, "above_vwap": False, "orb_state": "inside"}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=m, now=NOW
    )
    assert payload["checks"]["confirmations"] >= oc.B_PLUS_MIN_CONFIRMATIONS
    assert len(payload["checks"]["dissenters"]) <= oc.B_PLUS_MAX_DISSENTERS
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["candidate"]["tier"] == "b_plus"
    assert payload["execution_allowed"] is False


def test_chop_keeps_watching_below_b_plus_tier():
    # More definitive dissenters than B_PLUS_MAX_DISSENTERS -> no tier qualifies on CHOP.
    m = _market(day_verdict="CHOP")
    m["IWM"] = {"last": 298.0, "above_vwap": False, "orb_state": "inside"}
    m["XSP"] = {"last": 741.0, "above_vwap": False, "orb_state": "below"}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=m, now=NOW
    )
    assert len(payload["checks"]["dissenters"]) > oc.B_PLUS_MAX_DISSENTERS
    assert payload["decision"] == cw.KEEP_WATCHING
    assert any("CHOP" in r for r in payload["reasons"])


def test_partial_tape_is_neutral_never_a_dissenter():
    # XSP snapshots routinely lack above_vwap: without a definitive VWAP side the ETF must be
    # neutral, not a silent dissenter that blocks the A+/B+ tiers.
    m = _market(day_verdict="CHOP")
    m["XSP"] = {"last": 741.0, "orb_state": "below"}     # no above_vwap -> neutral
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=m, now=NOW
    )
    assert "XSP" not in payload["checks"]["dissenters"]
    assert payload["decision"] == cw.CONFIRM_ENTRY


def test_full_alignment_confirms_a_plus_tier():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(), now=NOW
    )
    assert payload["checks"]["confirmations"] >= oc.A_PLUS_MIN_CONFIRMATIONS
    assert payload["checks"]["dissenters"] == []
    assert payload["candidate"]["tier"] == "a_plus"


def test_confirm_entry_stamps_chase_band_anchor_from_contract_ask():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(),
        vehicle_score={"verdict": "GOOD_BET", "direction": "bullish",
                       "contract": {"underlying": "QQQ", "option_type": "call",
                                    "option_id": "QQQ260629C00725000",
                                    "expiration_date": "2026-06-29", "strike_price": 725.0,
                                    "ask": 1.19, "mark": 1.17}},
        now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["candidate"]["anchor_quote"] == 1.19
    assert payload["candidate"]["anchor_ts"] == NOW.isoformat(timespec="seconds")


def test_scan_only_symbol_never_becomes_executable_candidate():
    # XSP tape counts toward confirmation, but an XSP candidate itself can never convert.
    assert "XSP" in cw.ETF_UNIVERSE and "XSP" not in cw.EXECUTABLE_UNIVERSE
    payload = cw.evaluate_candidate_watch(
        {"ticker": "XSP", "direction": "bullish"}, market=_market(), now=NOW
    )
    assert payload["decision"] == cw.DEGRADED_NO_TRADE
    assert payload["execution_allowed"] is False


def test_synthetic_market_scorecard_never_confirms_entry():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish", "source": "market_scorecard",
         "synthetic_market_scorecard": True, "scan_only": True,
         "execution_allowed": False},
        market=_market(), now=NOW,
    )
    assert payload["decision"] == cw.KEEP_WATCHING
    assert payload["execution_allowed"] is False
    assert any("synthetic market_scorecard" in reason for reason in payload["reasons"])


def test_pin_wall_blocks_confirmation_until_acceptance_above_wall():
    m = _market(QQQ={"last": 724.97, "above_vwap": True, "orb_state": "above"})
    gamma = {"pin_risk": {"level": "high"}, "call_wall": 725.0, "max_gamma_strike": 725.0}
    cand = {"ticker": "QQQ", "direction": "bullish", "strike": 725.0}
    payload = cw.evaluate_candidate_watch(cand, market=m, gamma_map=gamma, now=NOW)
    assert payload["decision"] == cw.KEEP_WATCHING
    assert any("wall acceptance" in r for r in payload["reasons"])

    m2 = _market(QQQ={"last": 725.25, "above_vwap": True, "orb_state": "above"})
    payload2 = cw.evaluate_candidate_watch(cand, market=m2, gamma_map=gamma, now=NOW)
    assert payload2["decision"] == cw.CONFIRM_ENTRY


def test_degraded_or_expired_candidate_never_authorizes_execution():
    avoid = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=_market(day_verdict="AVOID"), now=NOW
    )
    assert avoid["decision"] == cw.DEGRADED_NO_TRADE
    assert avoid["execution_allowed"] is False

    old = {"ticker": "SPY", "direction": "bullish", "created_at": "2026-06-29T13:00:00+00:00"}
    # Strong tape: expiry closes the OLD cycle and the tape lane re-seeds a FRESH candidate in
    # the same run (2026-08-05 zombie-trap fix) — still scan-only, never execution authority.
    reseeded = cw.evaluate_candidate_watch(old, market=_market(), now=NOW, max_watch_minutes=20)
    assert reseeded["prior_candidate_expired"] is True
    assert reseeded["execution_allowed"] is False
    assert any("expired" in r for r in reseeded["reasons"])
    # No tape to re-seed from: expiry is final.
    expired = cw.evaluate_candidate_watch(old, market={}, now=NOW, max_watch_minutes=20)
    assert expired["decision"] == cw.EXPIRED_NO_CONFIRMATION
    assert expired["execution_allowed"] is False


def test_broker_blocked_is_loud_and_not_normal_watch():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(),
        broker_health={"execution_lane": "blocked"}, now=NOW,
    )
    assert payload["decision"] == cw.BROKER_BLOCKED
    assert payload["state"] == "BROKER_BLOCKED"
    assert payload["execution_allowed"] is False


def test_loop_status_surfaces_active_candidate_before_stale_gated_gate():
    active_candidate = {"ticker": "QQQ", "direction": "bullish", "state": "WATCHING_CONFIRMATION"}
    candidate_decision = {
        "decision": cw.KEEP_WATCHING,
        "candidate": active_candidate,
        "scan_only": True,
        "execution_allowed": False,
    }
    gate = {
        "event_type": "entry_decision",
        "seq": 5,
        "ts": "2026-06-29T13:30:00+00:00",
        "underlying": "SPY",
        "decision": "observe",
        "scan_only": True,
        "execution_allowed": False,
    }
    r = ls.derive_loop_state(active_candidate=active_candidate, candidate_decision=candidate_decision,
                             journal_events=[gate], now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["next_command"] == "odte-candidate-watch"
    assert r["context"]["candidate_watch"] is True
    assert r["executable"] is False


def test_loop_status_ignores_degraded_candidate_and_falls_through_to_scan():
    r = ls.derive_loop_state(
        active_candidate={"ticker": "QQQ", "direction": "bullish"},
        candidate_decision={"decision": cw.DEGRADED_NO_TRADE, "candidate": {"ticker": "QQQ"}},
        now=NOW,
    )
    assert r["state"] == "SCAN"


def test_loop_status_surfaces_broker_blocked_candidate_as_execution_lane_blocker():
    active_candidate = {"ticker": "QQQ", "direction": "bullish", "state": "BROKER_BLOCKED"}
    candidate_decision = {
        "decision": cw.BROKER_BLOCKED,
        "candidate": active_candidate,
        "scan_only": True,
        "execution_allowed": False,
    }
    r = ls.derive_loop_state(active_candidate=active_candidate, candidate_decision=candidate_decision,
                             now=NOW)
    assert r["state"] == "CANDIDATE"
    assert r["next_command"] == "verify-broker-review-lane"
    assert "blocked" in r["next_action"]
    assert r["executable"] is False


# --- VEHICLE LOCK (2026-07-23 remediation): one thesis, one underlying, one fingerprint ---------

def _bearish_market(**over):
    m = {
        "day_verdict": "GOOD_DAY",
        "minutes_to_close": 240,
        "SPY": {"last": 736.4, "above_vwap": False, "orb_state": "below"},
        "QQQ": {"last": 718.6, "above_vwap": False, "orb_state": "below"},
        "IWM": {"last": 296.0, "above_vwap": False, "orb_state": "below"},
        "XSP": {"last": 736.4, "above_vwap": False, "orb_state": "below"},
        "VIXY": {"above_vwap": True, "change_pct": 2.0},
    }
    m.update(over)
    return m


def test_qqq_candidate_with_spy_contract_is_hard_mismatch():
    # The exact failure the lock closes: the controller discusses QQQ but the vehicle score carries
    # a SPY contract. That is a HARD mismatch — degrade, never substitute.
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bearish"}, market=_bearish_market(),
        vehicle_score={"verdict": "GOOD_BET", "direction": "bearish",
                       "contract": {"underlying": "SPY", "option_type": "put", "strike": 737}},
        now=NOW)
    assert payload["decision"] == cw.DEGRADED_NO_TRADE
    assert payload["checks"]["vehicle_underlying"] == "SPY"
    assert any("mismatch" in r for r in payload["reasons"])
    assert payload["execution_allowed"] is False


def test_qqq_candidate_with_qqq_contract_matching_direction_proceeds():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bearish"}, market=_bearish_market(),
        vehicle_score={"verdict": "GOOD_BET", "direction": "bearish",
                       "contract": {"underlying": "QQQ", "option_type": "put", "strike": 718}},
        now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert payload["candidate"]["selected_vehicle"] == "QQQ"
    assert payload["execution_allowed"] is False               # a watch never executes by itself


def test_vehicle_lock_fields_persisted_on_candidate_and_active_file(tmp_path):
    import json
    # No created_at: run_candidate_watch uses the wall clock, so a dated candidate would expire.
    payload = cw.run_candidate_watch(
        candidate_json=json.dumps({"ticker": "QQQ", "direction": "bearish"}),
        market_json=json.dumps(_bearish_market()),
        state_dir=str(tmp_path), write=True)
    cand = payload["candidate"]
    for field in ("selected_vehicle", "selection_reason", "relative_strength_rank",
                  "selection_timestamp", "candidate_fingerprint"):
        assert cand.get(field) is not None, field
    assert cand["selected_vehicle"] == "QQQ"
    active = json.loads((tmp_path / cw.ACTIVE_CANDIDATE_FILENAME).read_text())
    assert active["selected_vehicle"] == "QQQ"
    assert active["candidate_fingerprint"] == cand["candidate_fingerprint"]
    assert active["selection_timestamp"] == cand["selection_timestamp"]


def test_vehicle_switch_mints_new_fingerprint_and_invalidates_old_lease():
    from data.odte_execution_policy import lease_matches_candidate
    qqq = cw.evaluate_candidate_watch({"ticker": "QQQ", "direction": "bearish",
                                       "created_at": NOW.isoformat()},
                                      market=_bearish_market(), now=NOW)
    spy = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bearish",
                                       "created_at": NOW.isoformat()},
                                      market=_bearish_market(), now=NOW)
    fp_qqq = qqq["candidate"]["candidate_fingerprint"]
    fp_spy = spy["candidate"]["candidate_fingerprint"]
    assert fp_qqq and fp_spy and fp_qqq != fp_spy
    # A lease bound to the QQQ cycle can never bind the SPY cycle.
    fake_lease = {"lease_id": "x", "candidate_fingerprint": fp_qqq}
    assert lease_matches_candidate(fake_lease, qqq["candidate"]) is True
    assert lease_matches_candidate(fake_lease, spy["candidate"]) is False


def test_relative_strength_rank_is_deterministic_selection_context():
    # SPY is the only ETF fully aligned bearish -> rank 1; QQQ half-aligned -> rank 2.
    m = _bearish_market(QQQ={"last": 718.6, "above_vwap": False, "orb_state": "inside"},
                        IWM={"last": 296.0, "above_vwap": True, "orb_state": "above"},
                        XSP={})
    spy = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bearish"}, market=m, now=NOW)
    assert spy["candidate"]["relative_strength_rank"] == 1
    qqq = cw.evaluate_candidate_watch({"ticker": "QQQ", "direction": "bearish"}, market=m, now=NOW)
    assert qqq["candidate"]["relative_strength_rank"] == 2


def test_vixy_divergence_is_recorded_but_the_helpers_are_mutually_exclusive():
    # SUPERSEDES test_vixy_conflict_flag_when_both_directions_confirmed (2026-08-07). That test
    # pinned the BUG: _vixy_weak short-circuited on above_vwap=False and _vixy_firming on
    # above_vwap=True, both falling through to change_pct otherwise, so a VIXY that disagreed with
    # itself satisfied BOTH and handed a free vol confirmation to either direction. The flag was
    # predicted 2026-08-05 and fired live 2026-08-07 15:54:42. vol_bias() now resolves it: the VWAP
    # side wins, change_pct is only a tiebreak. The disagreement is still recorded, as divergence.
    for block, weak, firming in (
        ({"above_vwap": True, "change_pct": -0.7371, "last": 20.2}, False, True),   # 2026-08-04
        ({"above_vwap": False, "change_pct": 0.0512, "last": 19.54}, True, False),  # 2026-08-07
    ):
        market = {
            "SPY": {"last": 770.0, "above_vwap": True, "orb_state": "above"},
            "QQQ": {"last": 700.0, "above_vwap": True, "orb_state": "above"},
            "IWM": {"last": 300.0, "above_vwap": True, "orb_state": "above"},
            "VIXY": block,
        }
        assert cw._vixy_weak(market) is weak
        assert cw._vixy_firming(market) is firming
        assert not (cw._vixy_weak(market) and cw._vixy_firming(market))
        result = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bullish"},
                                             market=market, now=NOW)
        assert result["checks"].get("vixy_divergence") is True
        assert "vixy_conflict" not in result["checks"]      # unreachable by construction now


def test_agreeing_vixy_is_not_flagged_as_divergent():
    market = {
        "SPY": {"last": 770.0, "above_vwap": True, "orb_state": "above"},
        "QQQ": {"last": 700.0, "above_vwap": True, "orb_state": "above"},
        "IWM": {"last": 300.0, "above_vwap": True, "orb_state": "above"},
        "VIXY": {"above_vwap": False, "change_pct": -0.7371, "last": 20.2},
    }
    result = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bullish"},
                                         market=market, now=NOW)
    assert "vixy_divergence" not in result["checks"]
    assert cw._vixy_weak(market) is True and cw._vixy_firming(market) is False


# --- 2026-08-05 zombie-trap fixes: session anchor, tombstone, expiry fall-through -------------

def test_pre_open_candidate_clock_starts_at_the_bell():
    # The Aug-5 phantom was minted 09:01 ET and expired 09:22 — before the market opened. A
    # same-ET-day pre-open candidate now starts its watch clock at 09:30.
    from datetime import datetime, timezone
    pre_open_cand = {"ticker": "SPY", "direction": "bullish",
                     "created_at": "2026-08-05T13:01:07+00:00"}          # 09:01 ET
    at_0922 = datetime(2026, 8, 5, 13, 22, tzinfo=timezone.utc)          # 09:22 ET
    assert cw._candidate_age_minutes(pre_open_cand, at_0922) == 0.0      # clock not started
    at_0945 = datetime(2026, 8, 5, 13, 45, tzinfo=timezone.utc)          # 09:45 ET
    assert cw._candidate_age_minutes(pre_open_cand, at_0945) == 15.0     # 15m since the bell
    # A PRIOR-day candidate keeps its raw age and expires immediately, as it must.
    stale = {"ticker": "SPY", "direction": "bullish",
             "created_at": "2026-08-04T14:00:00+00:00"}
    assert cw._candidate_age_minutes(stale, at_0945) > 1000


def test_expiry_writes_inert_tombstone_and_reopens_tape_lane(tmp_path):
    import json as _json
    old = {"ticker": "SPY", "direction": "bullish", "created_at": "2026-06-29T13:00:00+00:00"}
    payload = cw.run_candidate_watch(candidate_json=_json.dumps(old), market_json="{}",
                                     state_dir=str(tmp_path), write=True)
    assert payload["decision"] == cw.EXPIRED_NO_CONFIRMATION
    tomb = _json.loads((tmp_path / "active_candidate.json").read_text())
    assert "ticker" not in tomb and "direction" not in tomb              # inert to the tape lane
    assert tomb["decision"] == cw.EXPIRED_NO_CONFIRMATION
    assert tomb["prior"]["ticker"] == "SPY"                              # audit trail preserved
    # Next run consumes the tombstone as the candidate: the tape lane now manufactures.
    payload2 = cw.run_candidate_watch(candidate_json=_json.dumps(tomb),
                                      market_json=_json.dumps(_market()),
                                      state_dir=str(tmp_path), write=True)
    assert payload2["decision"] in {cw.KEEP_WATCHING, cw.CONFIRM_ENTRY}
    assert payload2["candidate"].get("ticker")                           # a REAL fresh candidate


def test_aug5_zombie_replay_expires_with_session_anchored_age():
    # The keystone replay: the real zombie (created 09:01 ET) against the best tape of the day
    # (10:16 ET: all three ETFs above VWAP but INSIDE their opening ranges — the archived
    # snapshot shape). The session-anchored age is 46m (09:30→10:16), not the raw 75m; the tape
    # lane cannot re-seed (no ORB breakout), so the cycle expires cleanly — the CORRECT verdict
    # for Aug-5, reached while still reading the tape instead of blind.
    from datetime import datetime, timezone
    zombie = {"ticker": "SPY", "direction": "bullish",
              "created_at": "2026-08-05T13:01:07+00:00"}
    tape_1016 = {
        "SPY": {"last": 776.24, "above_vwap": True, "orb_state": "inside"},
        "QQQ": {"last": 727.11, "above_vwap": True, "orb_state": "inside"},
        "IWM": {"last": 302.83, "above_vwap": True, "orb_state": "inside"},
        "VIXY": {"above_vwap": False, "change_pct": -0.34},
    }
    at_1016 = datetime(2026, 8, 5, 14, 16, tzinfo=timezone.utc)
    r = cw.evaluate_candidate_watch(zombie, market=tape_1016, now=at_1016,
                                    max_watch_minutes=20)
    assert r["decision"] == cw.EXPIRED_NO_CONFIRMATION
    assert r["checks"]["age_minutes"] == 46.0                            # anchored, not 75
    # Same moment, ORB breakout tape: the SAME call re-seeds a fresh cycle instead.
    breakout = {k: dict(v) for k, v in tape_1016.items()}
    for sym in ("SPY", "QQQ", "IWM"):
        breakout[sym]["orb_state"] = "above"
    r2 = cw.evaluate_candidate_watch(zombie, market=breakout, now=at_1016,
                                     max_watch_minutes=20)
    assert r2.get("prior_candidate_expired") is True
    assert r2["candidate"]["ticker"] in ("SPY", "QQQ", "IWM")


# --- 2026-08-07: graded breadth replaces the binary confirmer count ----------------------------
# The session that motivated this: a SPY 773C candidate sat in WATCHING_CONFIRMATION for 45 minutes
# at confirmations=1 while day_score graded the SAME snapshot GOOD_DAY off "3 indices trend-aligned
# on ORB/VWAP". QQQ and IWM were above VWAP but pinned inside their opening ranges (QQQ 722.66 vs an
# ORB high of 722.71; IWM 301.17 vs 301.19), which the binary count discarded entirely.

def _leader_led_market(**over):
    """Verbatim shape of the tape odte-convert refused at 2026-08-07T15:29:34Z."""
    m = {
        "day_verdict": "GOOD_DAY",
        "minutes_to_close": 270,
        "SPY": {"last": 773.655, "above_vwap": True, "orb_state": "above"},
        "QQQ": {"last": 722.66, "above_vwap": True, "orb_state": "inside"},
        "IWM": {"last": 301.17, "above_vwap": True, "orb_state": "inside"},
        "VIXY": {"last": 19.48, "above_vwap": False, "orb_state": "below"},
    }
    m.update(over)
    return m


def test_leader_led_breakout_with_rangebound_laggards_now_confirms():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=_leader_led_market(), now=NOW)
    checks = payload["checks"]
    assert checks["confirmations"] == 1                      # still honestly 1 fully aligned
    assert checks["half_confirmers"] == ["QQQ", "IWM"]       # the population the old count dropped
    assert checks["breadth_score"] == oc.BREADTH_MIN_SCORE   # exactly at the bar, not past it
    assert checks["breadth_required"] == oc.BREADTH_MIN_SCORE
    assert payload["decision"] == cw.CONFIRM_ENTRY


def test_one_index_alone_still_refuses():
    # The change must not let a single index convert by itself — that is the failure mode a graded
    # score could introduce if the threshold were set at FULL_ALIGNMENT.
    solo = _leader_led_market()
    for sym in ("QQQ", "IWM"):
        solo.pop(sym)
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=solo, now=NOW)
    assert payload["checks"]["breadth_score"] < oc.BREADTH_MIN_SCORE
    assert payload["decision"] == cw.KEEP_WATCHING


def test_breadth_thresholds_are_expressed_in_full_alignment_units():
    # The defaults must keep the old confirmer counts' meaning: N fully aligned indices is worth
    # N * FULL_ALIGNMENT points. Pinned so a config edit cannot silently change what a tier means.
    assert oc.BREADTH_MIN_SCORE == oc.B_PLUS_MIN_CONFIRMATIONS * breadth.FULL_ALIGNMENT
    assert oc.A_PLUS_MIN_BREADTH == oc.A_PLUS_MIN_CONFIRMATIONS * breadth.FULL_ALIGNMENT


def test_full_alignment_still_reaches_a_plus_under_the_score():
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=_market(), now=NOW)
    assert payload["checks"]["breadth_score"] >= oc.A_PLUS_MIN_BREADTH
    assert payload["candidate"]["tier"] == "a_plus"


def test_half_alignment_alone_cannot_reach_a_plus():
    # Four half-aligned indices score the A+ bar arithmetically but none is fully aligned; the
    # candidate's own underlying must still clear its opening range, so this refuses upstream.
    halves = {"day_verdict": "GOOD_DAY", "minutes_to_close": 270,
              "VIXY": {"above_vwap": False, "change_pct": -2.0}}
    for sym in ("SPY", "QQQ", "IWM", "XSP"):
        halves[sym] = {"last": 100.0, "above_vwap": True, "orb_state": "inside"}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=halves, now=NOW)
    assert payload["decision"] != cw.CONFIRM_ENTRY
    assert payload["candidate"].get("tier") != "a_plus"


def test_bearish_side_is_symmetric():
    # Mirror of the 2026-08-07 leader-led tape: SPY through its range, the laggards below VWAP but
    # still inside theirs. XSP is dropped so the score is the exact 2+1+1 mirror of the bullish case.
    market = _bearish_market(QQQ={"last": 718.6, "above_vwap": False, "orb_state": "inside"},
                             IWM={"last": 296.0, "above_vwap": False, "orb_state": "inside"})
    market.pop("XSP")
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bearish"}, market=market, now=NOW)
    assert payload["checks"]["half_confirmers"] == ["QQQ", "IWM"]
    assert payload["checks"]["breadth_score"] == oc.BREADTH_MIN_SCORE
    assert payload["decision"] == cw.CONFIRM_ENTRY


# --- 2026-08-07: the scanning lane journals its evaluations ------------------------------------
# candidate_evaluation previously fired only from odte_convert, which the controller runs only on
# the fast path — 3 events against ~78 scanning ticks on 2026-08-07. The near-miss population lives
# in the ticks that never convert, so the scanning lane has to record them too. That means BOTH
# lanes journal on a converting tick, which is exactly how a counter defect gets born (the fill
# double-count and the ingest P/L doubling both started this way). Every counter gets an assertion.

def _journaled(tmp_path, **kw):
    jp = str(tmp_path / "decision_journal.jsonl")
    cw.run_candidate_watch(candidate_json=json.dumps({"ticker": "SPY", "direction": "bullish"}),
                           market_json=json.dumps(_market()),
                           journal=True, journal_path=jp, **kw)
    return jp


def test_scanning_lane_journals_a_candidate_evaluation(tmp_path):
    import data.odte_journal as oj
    jp = _journaled(tmp_path)
    events = [e for e in oj.read_events(jp) if e.get("event_type") == "candidate_evaluation"]
    assert len(events) == 1
    ev = events[0]
    assert ev["source"] == "odte_candidate_watch"
    assert ev["scan_only"] is True and ev["execution_allowed"] is False
    assert ev.get("trade_id") is None                     # never joins a trade row
    assert ev["checks"]["breadth_score"] >= 0             # the near-miss payload rides along
    assert "confirmers" in ev["checks"] and "half_confirmers" in ev["checks"]


def test_scanning_lane_does_not_journal_by_default(tmp_path):
    import data.odte_journal as oj
    jp = str(tmp_path / "decision_journal.jsonl")
    cw.run_candidate_watch(candidate_json=json.dumps({"ticker": "SPY", "direction": "bullish"}),
                           market_json=json.dumps(_market()), journal_path=jp)
    assert oj.read_events(jp) == []


def test_both_lanes_journaling_moves_no_counter(tmp_path):
    # The converting-tick case: odte_convert and odte_candidate_watch each append their own
    # evaluation for the same tick. Neither creates a trade row, a refusal, or a budget slot.
    import data.odte_journal as oj
    jp = _journaled(tmp_path)
    before_events = oj.read_events(jp)
    before = (oj.summarize(before_events), oj.weekly_telemetry(before_events),
              oj.daily_trade_budget(before_events))
    # a second lane journaling the same evaluation, verbatim
    payload = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bullish"},
                                          market=_market(), now=NOW)
    oj.append_decision_journal(oj.event_from_candidate_evaluation(payload),
                              source="odte_convert", event_type="candidate_evaluation",
                              journal_path=jp)
    after_events = oj.read_events(jp)
    after = (oj.summarize(after_events), oj.weekly_telemetry(after_events),
             oj.daily_trade_budget(after_events))
    assert len(after_events) == len(before_events) + 1     # the event IS recorded
    for name, b, a in zip(("summarize", "weekly_telemetry", "daily_trade_budget"), before, after):
        for key in ("n_trades", "n_closed", "total_realized_pnl", "no_trade_decisions",
                    "lease_refusals", "entry_decisions", "trades_today", "budget", "remaining"):
            if key in b:
                assert a[key] == b[key], f"{name}.{key} moved: {b[key]} -> {a[key]}"


def test_ingest_never_synthesizes_a_second_candidate_evaluation():
    # First-party at computation time in BOTH lanes now, so an EOD artifact sweep must never
    # re-add the same decision under its own dedupe key.
    import data.odte_journal as oj
    assert "candidate_evaluation" in oj._INGEST_PROTECTED_LIFECYCLE


def test_synthetic_scorecard_candidate_can_expire(tmp_path):
    # 2026-08-07 zombie: the scan-only refusal sat ABOVE the expiry check, so a synthetic
    # market_scorecard candidate returned KEEP_WATCHING forever. Because _extract_candidate
    # early-returns on anything carrying ticker+direction, that immortal identity also kept the
    # tape lane from minting an executable candidate. Observed live at 24m against a 20m TTL and
    # still KEEP_WATCHING when evaluated three hours forward.
    from datetime import timedelta
    synthetic = {"ticker": "SPY", "direction": "bullish", "source": "market_scorecard",
                 "synthetic_market_scorecard": True,
                 "created_at": (NOW - timedelta(minutes=25)).isoformat()}
    flat = {"day_verdict": "CHOP", "minutes_to_close": 240,
            "SPY": {"last": 741.0, "above_vwap": True, "orb_state": "inside"},
            "QQQ": {"last": 724.2, "above_vwap": True, "orb_state": "inside"},
            "IWM": {"last": 299.0, "above_vwap": True, "orb_state": "inside"},
            "VIXY": {"above_vwap": False, "change_pct": -2.0}}
    payload = cw.evaluate_candidate_watch(synthetic, market=flat, now=NOW)
    assert payload["decision"] == cw.EXPIRED_NO_CONFIRMATION
    assert payload["checks"]["age_minutes"] == 25.0

    # ...and expiry re-seeds from the live tape in the SAME run when the tape qualifies, so the
    # slot is never left occupied by a dead synthetic.
    reseeded = cw.evaluate_candidate_watch(synthetic, market=_market(), now=NOW)
    assert reseeded.get("prior_candidate_expired") is True
    assert reseeded["candidate"]["source"] == "etf_momentum_tape"
    assert reseeded["candidate"]["ticker"] in cw.EXECUTABLE_UNIVERSE


def test_candidate_with_only_updated_at_can_still_age_out():
    """2026-08-11 zombie: the watchdog's synthetic persists into active_candidate.json with
    ticker+direction but ONLY an `updated_at` — no created_at/ts/generated_at. Age was therefore
    None, the TTL could never retire it, and `_extract_candidate` early-returns on anything with a
    ticker. Replayed against a perfect next-session tape the leftover answered KEEP_WATCHING where
    the same tape with no candidate answered CONFIRM_ENTRY. It must be able to EXPIRE."""
    from datetime import timedelta
    stale = {"ticker": "SPY", "direction": "bearish", "source": "market_scorecard",
             "updated_at": (NOW - timedelta(hours=18)).isoformat()}
    age = cw._candidate_age_minutes(stale, NOW)
    assert age is not None and age > 20, "a candidate with only updated_at must still have a clock"
    payload = cw.evaluate_candidate_watch(stale, market=_market(), now=NOW)
    assert payload["decision"] != cw.KEEP_WATCHING, "an expired synthetic must not hold the slot"
    assert payload.get("prior_candidate_expired") is True


def test_real_created_at_still_wins_over_updated_at():
    """The new fallbacks rank LAST — a real mint time must never be overridden by a later write."""
    from datetime import timedelta
    cand = {"ticker": "SPY", "direction": "bullish",
            "created_at": NOW.isoformat(),
            "updated_at": (NOW - timedelta(hours=18)).isoformat()}
    assert (cw._candidate_age_minutes(cand, NOW) or 0) < 1.0


def test_fresh_synthetic_scorecard_is_still_scan_only():
    # The guarantee that matters is unchanged: within its TTL a synthetic candidate is refused,
    # and it can never reach CONFIRM_ENTRY however well ETF breadth aligns.
    synthetic = {"ticker": "SPY", "direction": "bullish", "source": "market_scorecard",
                 "created_at": NOW.isoformat()}
    payload = cw.evaluate_candidate_watch(synthetic, market=_market(), now=NOW)
    assert payload["decision"] == cw.KEEP_WATCHING
    assert any("scan-only" in r for r in payload["reasons"])
    assert payload["execution_allowed"] is False


def test_expiry_precedes_every_other_disposition():
    # Pin the ordering itself: whatever else is wrong with a candidate, an aged-out one expires
    # (and re-seeds) rather than returning some other terminal state that strands the slot.
    from datetime import timedelta
    old = (NOW - timedelta(minutes=25)).isoformat()
    for extra in ({"source": "market_scorecard"},
                  {"synthetic_market_scorecard": True},
                  {}):
        cand = {"ticker": "SPY", "direction": "bullish", "created_at": old, **extra}
        payload = cw.evaluate_candidate_watch(cand, market=_market(), now=NOW)
        assert payload.get("prior_candidate_expired") is True, extra


# --- 2026-08-07: tier evidence is counted in FULLY aligned indices, not breadth points ---------
# Grading half-alignment was meant to change WHETHER a leader-led tape converts. Scoring the tiers
# off the breadth number alone also changed how large it trades and what escapes the daily cap.

def _leader_led_good_day():
    m = _leader_led_market()
    m["day_verdict"] = "GOOD_DAY"
    return m


def test_half_derived_breadth_sizes_at_the_b_plus_rail_even_on_a_good_day():
    # 1 full confirmer + 2 halves scores the same 4 as 2 full confirmers, but it is weaker evidence
    # than the two fully aligned indices that used to be the price of admission. It converts (that
    # is the intended change) at HALF size (that is the part that must not have moved).
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=_leader_led_good_day(), now=NOW)
    checks = payload["checks"]
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert checks["breadth_score"] == oc.BREADTH_MIN_SCORE
    assert checks["tier_basis"]["full_confirmers"] < oc.B_PLUS_MIN_CONFIRMATIONS
    assert checks["tier_basis"]["half_derived"] is True
    assert payload["candidate"]["tier"] == "b_plus"


def test_two_fully_aligned_indices_still_size_full_on_a_good_day():
    # The pre-change configuration keeps the pre-change tier — this is the control.
    m = _leader_led_good_day()
    m["QQQ"] = {"last": 723.0, "above_vwap": True, "orb_state": "above"}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=m, now=NOW)
    assert payload["checks"]["tier_basis"]["full_confirmers"] >= oc.B_PLUS_MIN_CONFIRMATIONS
    assert payload["checks"]["tier_basis"]["half_derived"] is False
    assert payload["candidate"]["tier"] == "full"


def test_a_plus_still_requires_three_fully_aligned_indices():
    # A+ carries the daily_budget_aplus_uncapped exemption, so it is the tier that must NOT be
    # reachable on arithmetic alone: 2 full + 2 half scores A_PLUS_MIN_BREADTH but has never
    # satisfied the 3-confirmer bar.
    m = _leader_led_good_day()
    m["QQQ"] = {"last": 723.0, "above_vwap": True, "orb_state": "above"}   # 2nd full confirmer
    m["XSP"] = {"last": 741.0, "above_vwap": True, "orb_state": "inside"}  # a half
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=m, now=NOW)
    checks = payload["checks"]
    assert checks["breadth_score"] >= oc.A_PLUS_MIN_BREADTH          # arithmetic says A+
    assert checks["tier_basis"]["full_confirmers"] < oc.A_PLUS_MIN_CONFIRMATIONS
    assert payload["candidate"]["tier"] != "a_plus"                   # the count says no

    # ...and a genuinely A+ tape still reaches it.
    full = _market()
    payload2 = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=full, now=NOW)
    assert payload2["checks"]["tier_basis"]["full_confirmers"] >= oc.A_PLUS_MIN_CONFIRMATIONS
    assert payload2["candidate"]["tier"] == "a_plus"


def test_late_day_window_still_demands_a_plus_under_the_count_rule():
    # mtc < 45 requires a_plus; half-derived breadth must not sneak through it.
    m = _leader_led_good_day()
    m["minutes_to_close"] = 20
    payload = cw.evaluate_candidate_watch(
        {"ticker": "SPY", "direction": "bullish"}, market=m, now=NOW)
    assert payload["decision"] == cw.KEEP_WATCHING
    assert any("late-day" in r for r in payload["reasons"])


def test_tape_minted_candidate_carries_a_birth_timestamp_so_it_can_expire():
    # Only the expiry fall-through stamped `created_at`. A candidate the tape lane minted from an
    # EMPTY slot carried none, so _candidate_age_minutes returned None and the watch TTL never
    # applied to it — it would hold the slot until something else displaced it. Every locked
    # candidate now gets a birth timestamp.
    from datetime import timedelta
    payload = cw.evaluate_candidate_watch({}, market=_market(), now=NOW)
    cand = payload["candidate"]
    assert cand["source"] == "etf_momentum_tape"
    assert cand.get("created_at"), "tape-minted candidate has no birth timestamp"
    assert cw._candidate_age_minutes(cand, NOW) == 0.0

    # ...and it ages out on a later tick rather than living forever.
    later = NOW + timedelta(minutes=25)
    assert cw._candidate_age_minutes(cand, later) == 25.0
    aged = cw.evaluate_candidate_watch(cand, market=_market(), now=later)
    assert aged.get("prior_candidate_expired") is True or \
        aged["decision"] == cw.EXPIRED_NO_CONFIRMATION


def test_birth_timestamp_does_not_change_the_candidate_fingerprint():
    # candidate_fingerprint hashes selection_timestamp (falling back to created_at), and
    # selection_timestamp is defaulted from created_at immediately after. Stamping created_at must
    # therefore be identity-neutral — a changed fingerprint would invalidate any bound lease.
    import data.odte_execution_policy as xp
    payload = cw.evaluate_candidate_watch({}, market=_market(), now=NOW)
    cand = payload["candidate"]
    assert cand["candidate_fingerprint"] == xp.candidate_fingerprint(cand)
    without = {k: v for k, v in cand.items() if k != "created_at"}
    assert xp.candidate_fingerprint(without) == cand["candidate_fingerprint"]


# --- 2026-08-07: the confirmed candidate carries its own invalidation level -------------------
# From the loop's postmortem of iwm-20260807-301c-scalp-6a761e01, logged as a rule violation:
# "Post-fill invalidation was hand-derived at the exact ORB boundary instead of being carried from
# a pre-entry machine-readable plan." Entry trigger and exit stop were the same price; the trade
# lived and died inside 2.5 cents and realized -$8.

def test_confirmed_candidate_carries_a_machine_read_invalidation_level():
    m = _market()
    m["QQQ"] = {"last": 724.2, "above_vwap": True, "orb_state": "above", "orb_high": 723.0}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=m, now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    inv = payload["candidate"]["invalidation"]
    assert inv["orb_level"] == 723.0
    # the stop sits BELOW the breakout level by the acceptance buffer — never ON it
    assert inv["underlying_stop"] < inv["orb_level"]
    assert inv["acceptance_buffer"] == max(0.03, 723.0 * 0.0002)
    assert inv["qqq_stop"] == inv["underlying_stop"]      # odte_position reads {sym}_stop too
    assert payload["checks"]["invalidation"] == inv


def test_invalidation_would_have_survived_the_iwm_whipsaw():
    # The real numbers: ORB high 301.19, entry at 301.205, the hand-derived stop fired at 301.18.
    # With the acceptance buffer the thesis is still alive at 301.18.
    m = _market()
    m["IWM"] = {"last": 301.205, "above_vwap": True, "orb_state": "above", "orb_high": 301.19}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "IWM", "direction": "bullish"}, market=m, now=NOW)
    inv = payload["candidate"]["invalidation"]
    assert inv["underlying_stop"] < 301.18, inv       # 301.18 would NOT have killed the thesis
    assert inv["underlying_stop"] < inv["orb_level"]


def test_invalidation_is_direction_symmetric():
    m = _bearish_market()
    m["QQQ"] = {"last": 718.6, "above_vwap": False, "orb_state": "below", "orb_low": 719.0}
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bearish"}, market=m, now=NOW)
    inv = payload["candidate"]["invalidation"]
    assert inv["orb_level"] == 719.0
    assert inv["underlying_stop"] > inv["orb_level"]   # a bearish thesis dies ABOVE the level


def test_missing_opening_range_emits_no_invalidation_rather_than_a_zero_stop():
    # The convert-time snapshot ships orb_state without orb_high. A missing level must never
    # masquerade as a stop at 0, which would read as "thesis dead" on every tick.
    m = _market()
    m["QQQ"] = {"last": 724.2, "above_vwap": True, "orb_state": "above"}   # no orb_high
    payload = cw.evaluate_candidate_watch(
        {"ticker": "QQQ", "direction": "bullish"}, market=m, now=NOW)
    assert payload["decision"] == cw.CONFIRM_ENTRY
    assert "invalidation" not in payload["candidate"]


def test_invalidation_survives_the_convert_round_trip():
    """The lane that CAN compute the stop is not the lane that converts.

    The scanning snapshot carries orb_high; the convert-time snapshot ships orb_state without it
    (verified on the 2026-08-07 convert_market_* artifacts). odte_convert re-runs candidate-watch
    with its own snapshot, so if the second pass overwrote `invalidation` with None the machine-read
    plan would be lost at exactly the moment it is needed.
    """
    scan = _market()
    scan["SPY"] = {"last": 773.0, "above_vwap": True, "orb_state": "above", "orb_high": 772.35}
    first = cw.evaluate_candidate_watch({"ticker": "SPY", "direction": "bullish"},
                                        market=scan, now=NOW)
    inv = first["candidate"]["invalidation"]
    assert inv["underlying_stop"] < inv["orb_level"]

    convert_tape = _market()
    convert_tape["SPY"] = {"last": 773.0, "above_vwap": True, "orb_state": "above"}  # no orb_high
    second = cw.evaluate_candidate_watch(first["candidate"], market=convert_tape, now=NOW)
    assert second["decision"] == cw.CONFIRM_ENTRY
    assert second["candidate"]["invalidation"] == inv      # carried, never recomputed to None
