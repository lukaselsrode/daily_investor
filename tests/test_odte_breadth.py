"""Tests for the shared 0DTE snapshot primitives (VWAP side / opening-range side / alignment / vol).

Pure unit tests — no network/broker/LLM. Fixtures are verbatim shapes from the 2026-08-07 session
that exposed the day_score vs candidate_watch breadth contradiction.

Not to be confused with `test_odte_tape.py`, which covers `execution.odte_tape` — the fast-lane
engine that PRODUCES snapshots. This module READS them.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data.odte_breadth as breadth
from data.odte_config import SCAN_UNIVERSE

# The exact snapshot odte-convert fetched at 2026-08-07T15:29:34Z and refused on. SPY had cleared
# its opening range; QQQ and IWM were above VWAP but pinned inside theirs (QQQ 722.66 vs an ORB
# high of 722.71, IWM 301.17 vs 301.19). The old binary count read this as 1-of-3 forever.
NESTED_20260807 = {
    "as_of": "2026-08-07T15:29:34Z",
    "SPY": {"last": 773.655, "above_vwap": True, "orb_state": "above"},
    "QQQ": {"last": 722.66, "above_vwap": True, "orb_state": "inside"},
    "IWM": {"last": 301.17, "above_vwap": True, "orb_state": "inside"},
    "VIXY": {"last": 19.48, "above_vwap": False, "orb_state": "below"},
}

FLAT_20260807 = {
    "as_of": "2026-08-07T15:29:34Z",
    "spy_above_vwap": True, "spy_orb_state": "above",
    "qqq_above_vwap": True, "qqq_orb_state": "inside",
    "iwm_above_vwap": True, "iwm_orb_state": "inside",
    "vixy_above_vwap": False, "vixy_orb_state": "below",
}


def _both() -> dict:
    return {**FLAT_20260807, **NESTED_20260807}


# --- snapshot-shape equivalence ----------------------------------------------------------------

def test_both_snapshot_shapes_resolve_identically():
    # day_score read the flat keys, candidate_watch read the nested blocks. They agreed on
    # 2026-08-07 only because the fast-lane tape engine emits both; a snapshot carrying one shape
    # must not read differently from one carrying the other.
    for symbol in ("SPY", "QQQ", "IWM", "VIXY"):
        reads = {shape: (breadth.above_vwap(m, symbol), breadth.orb_state(m, symbol))
                 for shape, m in (("nested", NESTED_20260807),
                                  ("flat", FLAT_20260807),
                                  ("both", _both()))}
        assert len(set(reads.values())) == 1, f"{symbol} reads differ by shape: {reads}"


def test_nested_block_wins_field_by_field_over_flat_keys():
    mixed = {**FLAT_20260807, "QQQ": {"above_vwap": True, "orb_state": "above"}}
    assert breadth.orb_state(mixed, "QQQ") == "above"        # nested wins
    assert breadth.orb_state(mixed, "IWM") == "inside"       # flat still fills the gap


def test_unknown_symbol_and_non_dict_market_are_empty_not_raising():
    assert breadth.symbol_block(NESTED_20260807, "NVDA") == {}
    assert breadth.symbol_block(NESTED_20260807, None) == {}
    assert breadth.symbol_block([], "SPY") == {}             # type: ignore[arg-type]
    assert breadth.above_vwap({}, "SPY") is None
    assert breadth.orb_state({}, "SPY") == ""


# --- per-index alignment -----------------------------------------------------------------------

def test_alignment_grades_the_2026_08_07_tape():
    m = NESTED_20260807
    assert breadth.alignment(m, "SPY", "bullish") == breadth.FULL_ALIGNMENT     # +2 VWAP and ORB
    assert breadth.alignment(m, "QQQ", "bullish") == 1                          # +1 VWAP only
    assert breadth.alignment(m, "IWM", "bullish") == 1
    assert breadth.alignment(m, "VIXY", "bullish") == -breadth.FULL_ALIGNMENT   # -2 opposed


def test_alignment_is_sign_symmetric_in_direction():
    for symbol in ("SPY", "QQQ", "IWM", "VIXY"):
        bull = breadth.alignment(NESTED_20260807, symbol, "bullish")
        bear = breadth.alignment(NESTED_20260807, symbol, "bearish")
        assert bull is not None and bear == -bull


def test_mixed_index_scores_zero_not_negative():
    # Above VWAP but below its opening range: a genuine internal conflict.
    m = {"SPY": {"above_vwap": True, "orb_state": "below"}}
    assert breadth.alignment(m, "SPY", "bullish") == 0


def test_partial_tape_is_neutral_and_never_opposed():
    # XSP routinely ships without above_vwap, and the fast-lane engine OMITS ORB fields before the
    # 10:00 ET freeze. Under the confirmation lane that must read as "no opinion", never a veto.
    partial = {"XSP": {"last": 741.0, "orb_state": "below"}}
    assert breadth.alignment(partial, "XSP", "bullish") is None
    buckets = breadth.breadth(partial, "bullish", ("XSP",))
    assert buckets["opposed"] == [] and buckets["neutral"] == ["XSP"]


def test_day_regime_lane_scores_partial_tape_instead_of_dropping_it():
    # require_vwap=False is the ONLY difference day_score keeps — its vote has always scored
    # whatever fields were present. Made explicit here rather than re-implemented there.
    partial = {"XSP": {"last": 741.0, "orb_state": "below"}}
    assert breadth.alignment(partial, "XSP", "bullish", require_vwap=False) == -1


def test_no_read_at_all_is_none_under_either_strictness():
    assert breadth.alignment({"XSP": {"last": 741.0}}, "XSP", "bullish") is None
    assert breadth.alignment({"XSP": {"last": 741.0}}, "XSP", "bullish", require_vwap=False) is None


# --- breadth -----------------------------------------------------------------------------------

def test_breadth_on_the_refused_tape_reaches_two_full_confirmers_worth():
    b = breadth.breadth(NESTED_20260807, "bullish", ("SPY", "QQQ", "IWM"))
    assert b["full"] == ["SPY"]
    assert b["half"] == ["QQQ", "IWM"]
    assert b["opposed"] == []
    # One full confirmer plus two halves is worth the same as two full confirmers. This is the
    # whole behavioural claim of the 2026-08-07 reconciliation, pinned to the tape that motivated it.
    assert b["score"] == 2 * breadth.FULL_ALIGNMENT


def test_breadth_score_counts_only_supportive_alignment():
    # An opposed index is tracked in `opposed` for the dissent rule, never netted out of the score
    # — subtracting it would silently re-tighten the B+ tier the 2026-08-02 retune opened.
    m = {"SPY": {"above_vwap": True, "orb_state": "above"},
         "QQQ": {"above_vwap": True, "orb_state": "above"},
         "IWM": {"above_vwap": False, "orb_state": "below"}}
    b = breadth.breadth(m, "bullish", ("SPY", "QQQ", "IWM"))
    assert b["score"] == 2 * breadth.FULL_ALIGNMENT
    assert b["opposed"] == ["IWM"]


def test_opposed_reproduces_the_legacy_dissent_predicate_exactly():
    # The old rule called an index a dissenter when its VWAP side was present AND it carried a
    # definitive opposite read (`above is False or orb == "below"`). A mixed index — above VWAP but
    # back below its opening range — scores 0, and the old rule DID count it as dissent. Treating
    # zero as neutral would widen the B+ tier a second time by accident, so pin the whole truth
    # table rather than trusting the sign convention.
    for above in (True, False, None):
        for orb in ("above", "below", "inside", ""):
            block = {"last": 100.0}
            if above is not None:
                block["above_vwap"] = above
            if orb:
                block["orb_state"] = orb
            m = {"SPY": block}
            legacy_dissent = above is not None and (above is False or orb == "below")
            is_opposed = "SPY" in breadth.breadth(m, "bullish", ("SPY",))["opposed"]
            assert is_opposed == legacy_dissent, f"above={above} orb={orb!r}"


def test_breadth_defaults_to_the_configured_scan_universe():
    b = breadth.breadth(NESTED_20260807, "bullish")
    assert set(b["full"] + b["half"] + b["opposed"] + b["neutral"]) == set(SCAN_UNIVERSE)
    assert "XSP" in b["neutral"]          # absent from the snapshot -> no opinion


# --- volatility read ---------------------------------------------------------------------------

def test_vol_bias_is_single_signed_on_the_block_that_produced_the_conflict():
    # 2026-08-07 15:54: VIXY below VWAP (supports calls) but +0.05% on the day (supports puts).
    # The old helpers both returned True here. The VWAP side wins; the disagreement is telemetry.
    conflicted = {"VIXY": {"above_vwap": False, "change_pct": 0.0512, "last": 19.54}}
    assert breadth.vol_bias(conflicted) == -1
    assert breadth.vol_divergence(conflicted) is True


def test_vol_bias_resolves_the_mirror_case_too():
    # The 2026-08-04 shape named in the original comment: above VWAP but down on the day.
    mirrored = {"VIXY": {"above_vwap": True, "change_pct": -0.7371, "last": 20.2}}
    assert breadth.vol_bias(mirrored) == 1
    assert breadth.vol_divergence(mirrored) is True


def test_vol_bias_falls_back_to_day_change_only_without_a_vwap_side():
    assert breadth.vol_bias({"VIXY": {"change_pct": -0.9}}) == -1
    assert breadth.vol_bias({"VIXY": {"change_pct": 0.9}}) == 1
    assert breadth.vol_divergence({"VIXY": {"change_pct": 0.9}}) is False


def test_vol_bias_is_zero_and_never_divergent_without_a_block():
    assert breadth.vol_bias({}) == 0
    assert breadth.vol_divergence({}) is False
    assert breadth.vol_bias({"VIXY": {"last": 19.5}}) == 0


def test_agreeing_volatility_read_is_not_divergent():
    assert breadth.vol_divergence({"VIXY": {"above_vwap": False, "change_pct": -0.5}}) is False
    assert breadth.vol_divergence({"VIXY": {"above_vwap": True, "change_pct": 0.5}}) is False


# --- purity ------------------------------------------------------------------------------------

def test_module_makes_no_broker_or_network_calls():
    src = inspect.getsource(breadth)
    for forbidden in ("robin_stocks", "requests", "openai", "anthropic", "place_order",
                      "submit_order", "urllib", "httpx", "socket", "yfinance", "open("):
        assert forbidden not in src, f"odte_breadth must not reference {forbidden!r}"
