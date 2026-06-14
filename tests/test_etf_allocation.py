"""
tests/test_etf_allocation.py — ETF/core sleeve allocation (Milestone A, configured_only).

Covers the hard rules: enabled:false / equal_weight preserve behavior exactly; weights
normalize; regime selection; invalid allocations rejected; every constraint enforced;
bucket→ETF mapping; turnover-bounded rebalance; ETF allocation touches ONLY the ETF
sleeve (active book unchanged); configured_only cannot add ETFs; the tuning slots are
isolated from active-stock params; and the writer touches only the etf_allocation section.

Per the project rule, thresholds come from LIVE config constants (util), never hardcoded.
"""
import copy
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from portfolio import etf_allocation as ea
from util import ETF_ALLOCATION_PARAMS

_UNIVERSE = list(ETF_ALLOCATION_PARAMS["configured_universe"])
_BUCKETS = ETF_ALLOCATION_PARAMS["buckets"]
_CONS = ETF_ALLOCATION_PARAMS["constraints"]


# ---------------------------------------------------------------------------
# 1. enabled:false / equal_weight preserve the historical equal weight exactly
# ---------------------------------------------------------------------------

def test_disabled_is_equal_weight():
    p = {**ETF_ALLOCATION_PARAMS, "enabled": False, "mode": "regime_weights"}
    w = ea.etf_target_weights("bullish", _UNIVERSE, params=p)
    assert w == pytest.approx({e: 1.0 / len(_UNIVERSE) for e in _UNIVERSE})


def test_equal_weight_mode_is_equal_weight():
    p = {**ETF_ALLOCATION_PARAMS, "enabled": True, "mode": "equal_weight"}
    w = ea.etf_target_weights("defensive", _UNIVERSE, params=p)
    assert w == pytest.approx({e: 1.0 / len(_UNIVERSE) for e in _UNIVERSE})


def test_equal_weight_sums_to_one_and_stays_in_universe():
    w = ea.equal_weights(_UNIVERSE)
    assert sum(w.values()) == pytest.approx(1.0)
    assert set(w) == set(_UNIVERSE)


# ---------------------------------------------------------------------------
# 2. static_weights normalize; all-null falls back to equal
# ---------------------------------------------------------------------------

def test_static_weights_normalize():
    p = copy.deepcopy(ETF_ALLOCATION_PARAMS)
    p["enabled"] = True
    p["mode"] = "static_weights"
    p["default_weights"] = {e: None for e in _UNIVERSE}
    p["default_weights"]["SPY"] = 2.0
    p["default_weights"]["VTI"] = 2.0
    w = ea.etf_target_weights("bullish", _UNIVERSE, params=p)
    assert w == pytest.approx({"SPY": 0.5, "VTI": 0.5})
    assert sum(w.values()) == pytest.approx(1.0)


def test_static_weights_all_null_falls_back_to_equal():
    p = copy.deepcopy(ETF_ALLOCATION_PARAMS)
    p["enabled"] = True
    p["mode"] = "static_weights"
    p["default_weights"] = {e: None for e in _UNIVERSE}
    w = ea.etf_target_weights("bullish", _UNIVERSE, params=p)
    assert w == pytest.approx({e: 1.0 / len(_UNIVERSE) for e in _UNIVERSE})


# ---------------------------------------------------------------------------
# 3. regime_weights select the correct regime allocation
# ---------------------------------------------------------------------------

def test_regime_weights_select_per_regime():
    p = copy.deepcopy(ETF_ALLOCATION_PARAMS)
    p["enabled"] = True
    p["mode"] = "regime_weights"
    p["regime_weights"] = {
        "bullish":   {"core_market": 0.6, "growth": 0.2, "small_cap": 0.2},
        "neutral":   {"core_market": 0.7, "dividend_defensive": 0.3},
        "defensive": {"core_market": 1.0},
    }
    wb = ea.etf_target_weights("bullish", _UNIVERSE, params=p)
    wd = ea.etf_target_weights("defensive", _UNIVERSE, params=p)
    # defensive is 100% core_market → only SPY/VOO/VTI hold weight.
    assert set(k for k, v in wd.items() if v > 0) == {"SPY", "VOO", "VTI"}
    # bullish holds growth (QQQ) and small_cap (IWM); defensive does not.
    assert wb.get("QQQ", 0) > 0 and wb.get("IWM", 0) > 0
    assert wd.get("QQQ", 0) == 0


# ---------------------------------------------------------------------------
# 4. bucket → ETF expansion (equal within bucket, restricted to universe)
# ---------------------------------------------------------------------------

def test_expand_bucket_weights_equal_within_bucket():
    w = ea.expand_bucket_weights({"core_market": 0.6, "growth": 0.4}, _BUCKETS, _UNIVERSE)
    # core_market (SPY/VOO/VTI) shares 0.6 equally; growth (QQQ) gets 0.4.
    assert w["SPY"] == pytest.approx(0.2)
    assert w["QQQ"] == pytest.approx(0.4)
    assert sum(w.values()) == pytest.approx(1.0)


def test_expand_drops_buckets_with_no_universe_member():
    # cashlike_bonds is empty in configured_only → its weight is renormalized away.
    w = ea.expand_bucket_weights({"core_market": 0.5, "cashlike_bonds": 0.5}, _BUCKETS, _UNIVERSE)
    assert sum(w.values()) == pytest.approx(1.0)
    assert set(k for k, v in w.items() if v > 0) == {"SPY", "VOO", "VTI"}


# ---------------------------------------------------------------------------
# 5. invalid allocations rejected (fall back to equal); never leave universe
# ---------------------------------------------------------------------------

def test_invalid_allocation_falls_back_to_equal():
    p = copy.deepcopy(ETF_ALLOCATION_PARAMS)
    p["enabled"] = True
    p["mode"] = "regime_weights"
    # 100% semis violates min_core_market and max_semis → must fall back to equal.
    p["regime_weights"] = {"bullish": {"semis": 1.0}, "neutral": {}, "defensive": {}}
    w = ea.etf_target_weights("bullish", _UNIVERSE, params=p)
    assert w == pytest.approx({e: 1.0 / len(_UNIVERSE) for e in _UNIVERSE})


def test_never_emits_weight_outside_universe():
    p = copy.deepcopy(ETF_ALLOCATION_PARAMS)
    p["enabled"] = True
    p["mode"] = "regime_weights"
    p["regime_weights"] = {"bullish": {"core_market": 1.0}, "neutral": {}, "defensive": {}}
    small_uni = ["SPY", "VOO", "VTI"]  # configured_only cannot add ETFs beyond this
    w = ea.etf_target_weights("bullish", small_uni, params=p)
    assert set(w).issubset(set(small_uni))


# ---------------------------------------------------------------------------
# 6. each constraint enforced by validate_allocation (uses LIVE caps)
# ---------------------------------------------------------------------------

def _mk(bucket_weights):
    return ea.expand_bucket_weights(bucket_weights, _BUCKETS, _UNIVERSE)


def test_min_core_market_enforced():
    v = ea.validate_allocation(_mk({"core_market": 0.2, "growth": 0.3, "dividend_defensive": 0.5}),
                               _BUCKETS, _CONS, _UNIVERSE)
    assert any("core_market" in s for s in v)


def test_max_single_etf_enforced():
    # QQQ alone would be 100% of one ETF (> max_single_etf_weight).
    w = {"QQQ": 1.0}
    v = ea.validate_allocation(w, _BUCKETS, _CONS, _UNIVERSE)
    assert any("max_single_etf_weight" in s for s in v)


def test_max_semis_and_thematic_enforced():
    # Heavy semis: violates max_semis_weight and/or max_thematic_combined.
    v = ea.validate_allocation(_mk({"core_market": 0.4, "semis": 0.6}), _BUCKETS, _CONS, _UNIVERSE)
    assert any("semis" in s or "thematic" in s for s in v)


def test_valid_core_heavy_allocation_passes():
    w = _mk({"core_market": 0.4, "growth": 0.1, "semis": 0.1, "dividend_defensive": 0.1,
             "international": 0.1, "real_estate": 0.1, "small_cap": 0.1})
    assert ea.validate_allocation(w, _BUCKETS, _CONS, _UNIVERSE) == []


# ---------------------------------------------------------------------------
# 7. rebalance_plan: band gate + turnover cap (drives ETF turnover)
# ---------------------------------------------------------------------------

def test_rebalance_noop_within_band():
    band = _CONS["rebalance_band"]
    deltas = ea.rebalance_plan({"SPY": 500, "QQQ": 500}, {"SPY": 0.5 + band / 2, "QQQ": 0.5 - band / 2},
                               band, _CONS["max_turnover_per_rebalance"])
    assert deltas == {}


def test_rebalance_capped_at_max_turnover():
    cap = _CONS["max_turnover_per_rebalance"]
    deltas = ea.rebalance_plan({"SPY": 1000.0}, {"SPY": 0.0, "VTI": 1.0},
                               _CONS["rebalance_band"], cap)
    one_way = sum(v for v in deltas.values() if v > 0)
    assert one_way == pytest.approx(cap * 1000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. Tuning slots: ETF scope isolates active-stock params (hard rule)
# ---------------------------------------------------------------------------

def test_etf_scope_only_unfreezes_etf_weight_slots():
    from tuning.constants import _ETF_ENABLED_SLOT, _etf_weight_slot_indices, _get_active_indices
    act = set(_get_active_indices("etf_allocation", "etf_allocation"))
    assert act == _etf_weight_slot_indices()
    assert _ETF_ENABLED_SLOT not in act  # enabled flag carried per-candidate, not optimized


def test_active_stock_scope_never_touches_etf_slots():
    from tuning.constants import _ETF_ENABLED_SLOT, _etf_weight_slot_indices, _get_active_indices
    etf_idx = _etf_weight_slot_indices() | {_ETF_ENABLED_SLOT}
    for preset in ("active_core_weights", "active_full_safe", "active_exit_ladder"):
        act = set(_get_active_indices("active_sleeve_compounding", preset))
        assert not (act & etf_idx), f"{preset} leaked into ETF slots"


def test_incumbent_vector_is_equal_weight():
    from tuning.constants import _ETF_ENABLED_SLOT, _current_params
    assert _current_params()[_ETF_ENABLED_SLOT] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. ETF allocation affects ONLY the ETF sleeve (active book unchanged)
# ---------------------------------------------------------------------------

def test_etf_allocation_changes_only_etf_sleeve():
    from backtesting.simulator import run_simulation
    from tuning.constants import (
        _ETF_ENABLED_SLOT,
        _ETF_WEIGHT_SLOT_OFFSET,
        _current_params,
    )
    from ui.services.backtest_service import load_precomp
    try:
        pc = load_precomp(365)
    except Exception as exc:  # no full-universe data in this sandbox — integration-only
        pytest.skip(f"requires data/agg_data.csv (full-universe precomp): {exc}")
    base = _current_params()
    r_eq = run_simulation(pc, base, scope="etf_allocation")  # enabled=0 → equal weight
    cand = base.copy()
    cand[_ETF_ENABLED_SLOT] = 1.0
    for k in range(21):  # valid core-heavy: core 0.4, satellites 0.1
        cand[_ETF_WEIGHT_SLOT_OFFSET + k] = 0.4 if (k % 7 == 0) else 0.1
    r_alt = run_simulation(pc, cand, scope="etf_allocation")
    # Active sleeve (stock book) is driven by the frozen active-stock params → identical
    # trade count; only the ETF sleeve weighting differs.
    assert r_eq.trades_made == r_alt.trades_made
    assert r_eq.etf_turnover == pytest.approx(0.0)  # equal weight never rebalances
    assert abs(r_alt.total_return - r_eq.total_return) > 1e-9  # ETF weighting did change totals


# ---------------------------------------------------------------------------
# 10. Writer touches ONLY the etf_allocation section (preserves the rest)
# ---------------------------------------------------------------------------

def test_apply_etf_allocation_writes_only_etf_section(tmp_path, monkeypatch):
    import yaml

    import core.paths
    from tuning import reports
    from tuning.constants import (
        _ETF_ENABLED_SLOT,
        _ETF_WEIGHT_SLOT_OFFSET,
        _current_params,
    )

    src_cfg = str(core.paths.CONFIG_FILE)
    tmp_cfg = tmp_path / "config.yaml"
    shutil.copy(src_cfg, tmp_cfg)

    with open(tmp_cfg) as f:
        before = yaml.safe_load(f)

    vec = _current_params().copy()
    vec[_ETF_ENABLED_SLOT] = 1.0
    for k in range(21):
        vec[_ETF_WEIGHT_SLOT_OFFSET + k] = 0.4 if (k % 7 == 0) else 0.1

    monkeypatch.setattr(reports, "CONFIG_FILE", str(tmp_cfg))
    reports.apply_etf_allocation_params(vec, provenance={"candidate_id": "test", "n_days": 365})

    with open(tmp_cfg) as f:
        after = yaml.safe_load(f)

    # etf_allocation updated...
    assert after["etf_allocation"]["enabled"] is True
    assert after["etf_allocation"]["mode"] == "regime_weights"
    assert after["etf_allocation"]["regime_weights"]["bullish"]["core_market"] == pytest.approx(0.4)
    # ...and NOTHING else changed (active-stock params, scoring, sell rules, etc.).
    for key in before:
        if key == "etf_allocation":
            continue
        assert after[key] == before[key], f"writer mutated unrelated section: {key}"
    # A provenance comment was prepended.
    assert "tune-etf-allocation" in tmp_cfg.read_text()
