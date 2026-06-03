"""
tests/test_position_sizing_preset.py — active_position_sizing preset + audit fix.

Covers:
  1. The revived active_position_sizing preset (no longer a Phase-2 stub) unfreezes
     exactly the 3 sizing slots and nothing else.
  2. Audit fix: active_factor_internals now also tunes return_5d (slot 15), matching
     its "all 6 momentum weights" description.
  3. position_sizing_cfg_from_params extraction (ints rounded; empty for short vectors).
  4. _current_params() grew by 3 and seeds from live config.
  5. apply_config_params persists the sizing slots into the risk / candidate_selection
     config blocks (written to a temp file, never live config).
  6. run_simulation accepts both the extended (46) and legacy (15) vectors.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # sibling test modules

import pytest
import yaml

from tuning import constants, presets

# ── 1. preset revived + isolates the sizing slots ───────────────────────────

def test_active_position_sizing_no_longer_phase2():
    presets.validate_preset("active_position_sizing")  # must not raise
    assert presets._PRESETS["active_position_sizing"]["phase2"] is False


def test_active_position_sizing_active_indices():
    active = set(constants._get_active_indices(preset="active_position_sizing"))
    assert active == {
        constants._PS_SLOT_OFFSET,
        constants._PS_SLOT_OFFSET + 1,
        constants._PS_SLOT_OFFSET + 2,
    }
    # weights / exits / factor internals must stay frozen
    assert not (active & set(range(0, 16)))


def test_active_position_sizing_paths_registered():
    for p in ("risk.max_single_position_pct", "risk.max_buys_per_rebalance",
              "candidate_selection.max_candidates"):
        assert p in constants._CONFIG_PATH_TO_PARAM_IDX


# ── 2. audit fix: factor_internals tunes all 6 momentum weights ─────────────

def test_factor_internals_includes_return_5d():
    active = set(constants._get_active_indices(preset="active_factor_internals"))
    r5d = constants._CONFIG_PATH_TO_PARAM_IDX["scoring.momentum_inputs.weights.return_5d"]
    assert r5d in active  # slot 15 — previously omitted despite the description


# ── 3. cfg extraction ────────────────────────────────────────────────────────

def test_position_sizing_cfg_from_params():
    cp = constants._current_params()
    cfg = constants.position_sizing_cfg_from_params(cp)
    assert set(cfg) == {"max_single_position_pct", "max_buys_per_rebalance", "max_candidates"}
    assert isinstance(cfg["max_buys_per_rebalance"], int)
    assert isinstance(cfg["max_candidates"], int)
    assert constants.position_sizing_cfg_from_params(cp[:15]) == {}
    assert constants.position_sizing_cfg_from_params(None) == {}


# ── 4. _current_params length + seeding ──────────────────────────────────────

def test_current_params_extended_length():
    cp = constants._current_params()
    # _PS_SLOT_OFFSET + 3 position-sizing + 1 regime-tilt (46) + 1 mean-reversion (47)
    # + 1 low-vol quality blend (48) + 1 residual-momentum blend (49)
    # + 4 DAE exit-floor slots (50-53) + 3 opportunity-cost slots (54-56, last group)
    assert len(cp) == constants._OC_SLOT_OFFSET + len(constants._OC_FIELDS)
    # exit-floor slots seed from config (EXIT_DECISION_PARAMS) and sit within bounds
    for off in range(len(constants._EXIT_FLOOR_FIELDS)):
        idx = constants._EXIT_FLOOR_SLOT_OFFSET + off
        lo, hi = constants.BOUNDS[idx]
        assert lo <= cp[idx] <= hi
    # within bounds for each position-sizing slot
    for off in range(3):
        lo, hi = constants.BOUNDS[constants._PS_SLOT_OFFSET + off]
        assert lo <= cp[constants._PS_SLOT_OFFSET + off] <= hi
    # regime slots present, default to 0.0 (behaviour-preserving) and in bounds
    for path in ("regime.bullish.momentum_tilt", "regime.defensive.mean_reversion_blend"):
        idx = constants._CONFIG_PATH_TO_PARAM_IDX[path]
        lo, hi = constants.BOUNDS[idx]
        assert lo <= cp[idx] <= hi


# ── 5. write-back persists sizing slots (temp file, not live config) ─────────

def test_apply_config_params_persists_sizing(tmp_path, monkeypatch):
    from tuning import reports
    src_cfg = os.path.join(os.path.dirname(__file__), "..", "cfg", "config.yaml")
    tmp_cfg = tmp_path / "config.yaml"
    shutil.copy(src_cfg, tmp_cfg)
    monkeypatch.setattr(reports, "CONFIG_FILE", str(tmp_cfg))

    params = constants._current_params().copy()
    params[constants._PS_SLOT_OFFSET]     = 0.11    # max_single_position_pct
    params[constants._PS_SLOT_OFFSET + 1] = 12.0    # max_buys_per_rebalance
    params[constants._PS_SLOT_OFFSET + 2] = 25.0    # max_candidates
    reports.apply_config_params(params)

    written = yaml.safe_load(open(tmp_cfg))
    assert written["risk"]["max_single_position_pct"] == 0.11
    assert written["risk"]["max_buys_per_rebalance"] == 12
    assert written["candidate_selection"]["max_candidates"] == 25


# ── 6. simulator accepts extended + legacy vectors ──────────────────────────

def test_run_simulation_accepts_extended_and_legacy_vectors():
    try:
        from test_active_sleeve_accounting import _make_precomp, _no_exit_params, _run
    except Exception:
        pytest.skip("sibling precomp helper unavailable")
    precomp = _make_precomp()
    legacy = _no_exit_params()                      # 15-element
    res_legacy = _run(precomp, legacy)
    assert res_legacy is not None

    extended = constants._current_params().copy()   # 46-element
    extended[:15] = legacy
    extended[constants._PS_SLOT_OFFSET + 2] = 50.0   # loose max_candidates
    res_ext = _run(precomp, extended)
    assert res_ext is not None
