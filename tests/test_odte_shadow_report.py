"""tests/test_odte_shadow_report.py — shadow-vs-live divergence joining.

Pure/offline. Pins the join semantics (ET day + underlying, exact option_id preferred, ±5-min
window fallback), the latency/limit deltas on matched pairs, the shadow_only / live_only /
exit-divergence classifications the rollout gates adjudicate, and the `clean` flag that Gate 1
requires. Event vocabulary comes from the live journal module — never re-hardcoded.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_shadow_report as sr

NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def _live_entry(ts, option_id="opt-756c", underlying="SPY", event_type="execution_lease_issued",
                limit=0.64):
    return {"event_type": event_type, "ts": _iso(ts), "underlying": underlying,
            "option_id": option_id, "max_limit_price": limit}


def _shadow_intent(ts, option_id="opt-756c", underlying="SPY", limit=0.63):
    return {"event_type": sr.SHADOW_ORDER_INTENT, "ts": _iso(ts), "underlying": underlying,
            "option_id": option_id, "limit_price": limit, "intent_id": "spy-c-756"}


def test_matched_pair_reports_latency_and_limit_deltas():
    report = sr.build_shadow_report(
        [_live_entry(NOW)], [_shadow_intent(NOW - timedelta(seconds=42))], now=NOW)
    assert report["counts"]["both_fired"] == 1 and report["clean"] is True
    pair = report["both_fired"][0]
    assert pair["latency_delta_s"] == -42.0                    # negative: shadow was FASTER
    assert pair["limit_delta"] == round(0.63 - 0.64, 4)


def test_option_id_match_beats_window_match():
    other = _shadow_intent(NOW + timedelta(seconds=10), option_id="opt-OTHER")
    exact = _shadow_intent(NOW + timedelta(minutes=4), option_id="opt-756c")
    report = sr.build_shadow_report([_live_entry(NOW)], [other, exact], now=NOW)
    assert report["counts"]["both_fired"] == 1
    assert report["both_fired"][0]["latency_delta_s"] == 240.0  # the exact-id one, not closest
    assert report["counts"]["shadow_only"] == 1                 # the other intent stands alone


def test_shadow_only_and_live_only_classification():
    live = [_live_entry(NOW, option_id="opt-live", underlying="QQQ")]
    shadow = [_shadow_intent(NOW + timedelta(hours=2), option_id="opt-shadow")]
    report = sr.build_shadow_report(live, shadow, now=NOW)
    assert report["counts"]["both_fired"] == 0
    assert [m["option_id"] for m in report["live_only"]] == ["opt-live"]
    assert [m["option_id"] for m in report["shadow_only"]] == ["opt-shadow"]
    assert report["clean"] is False


def test_fill_vocabulary_anchors_live_entries():
    # entry_fill (the 2026-08-03 live vocabulary) anchors a live entry; the earliest event per
    # identity wins so lease + fill for the same trade count ONCE.
    from data.odte_journal import ENTRY_FILL_EVENTS
    assert "entry_fill" in ENTRY_FILL_EVENTS
    events = [_live_entry(NOW, event_type="execution_lease_issued"),
              _live_entry(NOW + timedelta(seconds=8), event_type="entry_fill")]
    report = sr.build_shadow_report(events, [_shadow_intent(NOW)], now=NOW)
    assert report["counts"]["live_entries"] == 1
    assert report["counts"]["both_fired"] == 1


def test_exit_divergences_and_incidents_break_clean():
    live_close = {"event_type": "order_closed", "ts": _iso(NOW), "underlying": "SPY",
                  "option_id": "opt-756c", "realized_pnl": 9.0}
    matched_exit = {"event_type": sr.SHADOW_EXIT_INTENT, "ts": _iso(NOW - timedelta(seconds=30)),
                    "underlying": "SPY", "option_id": "opt-756c",
                    "trigger_type": "BID_MEMORY_PROTECT"}
    report = sr.build_shadow_report([live_close], [matched_exit], now=NOW)
    div = report["exit_divergences"][0]
    assert div["divergence"] == "matched" and div["timing_delta_s"] == -30.0
    assert div["shadow_trigger"] == "BID_MEMORY_PROTECT"
    assert report["clean"] is True

    # An unmatched live close, an orphan shadow exit, or an incident all break `clean`.
    lonely = sr.build_shadow_report([live_close], [], now=NOW)
    assert lonely["exit_divergences"][0]["divergence"] == "live_exit_without_shadow_intent"
    assert lonely["clean"] is False
    orphan = sr.build_shadow_report([], [matched_exit], now=NOW)
    assert orphan["exit_divergences"][0]["divergence"] == "shadow_exit_without_live_close"
    assert orphan["clean"] is False
    incident = sr.build_shadow_report([], [{"event_type": sr.SHADOW_INCIDENT, "ts": _iso(NOW),
                                            "underlying": "SPY"}], now=NOW)
    assert incident["counts"]["shadow_incidents"] == 1 and incident["clean"] is False


def test_different_et_days_never_join():
    report = sr.build_shadow_report([_live_entry(NOW)],
                                    [_shadow_intent(NOW + timedelta(days=1))], now=NOW)
    assert report["counts"]["both_fired"] == 0
    assert report["counts"]["shadow_only"] == 1 and report["counts"]["live_only"] == 1


def test_render_markdown_names_every_divergence():
    report = sr.build_shadow_report(
        [_live_entry(NOW, option_id="opt-a")],
        [_shadow_intent(NOW + timedelta(hours=3), option_id="opt-b")], now=NOW)
    text = sr.render_markdown(report)
    assert "LIVE-ONLY" in text and "SHADOW-ONLY" in text and "clean: False" in text


# --- the window (2026-08-12) -------------------------------------------------------------------
# Comparing every live event ever journaled against the shadow stream made `clean` unreachable by
# construction: the daemon's first run reported 20 live_only entries and 17 exit divergences going
# back to 2026-06-24, against 90 seconds of shadow data. Gate 1 requires `clean: true`, so it could
# never have been satisfied.

def _session_start(ts, underlying="SPY"):
    return {"event_type": sr.SHADOW_SESSION_START, "ts": _iso(ts), "underlying": underlying}


def test_live_events_before_the_session_start_are_out_of_window():
    report = sr.build_shadow_report(
        [_live_entry(NOW - timedelta(hours=3))],          # the lane was not running yet
        [_session_start(NOW - timedelta(minutes=5))], now=NOW)
    assert report["counts"]["live_entries"] == 0
    assert report["window"]["live_events_out_of_window"] == 1
    assert report["clean"] is True


def test_a_live_entry_the_lane_SHOULD_have_seen_is_still_a_divergence():
    """The window must not swallow the signal Gate 1 exists to adjudicate."""
    report = sr.build_shadow_report(
        [_live_entry(NOW)],                                # after the lane started, no intent
        [_session_start(NOW - timedelta(minutes=5))], now=NOW)
    assert [d["option_id"] for d in report["live_only"]] == ["opt-756c"]
    assert report["clean"] is False


def test_window_is_reported_not_implied():
    """A narrowing that is not reported is indistinguishable from clean."""
    start = NOW - timedelta(minutes=5)
    report = sr.build_shadow_report([_live_entry(NOW - timedelta(hours=3))],
                                    [_session_start(start)], now=NOW)
    assert report["window"]["observed_from"] == {"2026-08-05": _iso(start)}


def test_shadow_journal_without_a_session_marker_is_unwindowed():
    """Legacy/partial shadow data keeps the original semantics rather than silently comparing
    against nothing."""
    report = sr.build_shadow_report([_live_entry(NOW)], [_shadow_intent(NOW)], now=NOW)
    assert report["counts"]["both_fired"] == 1
    assert report["window"]["live_events_out_of_window"] == 0
