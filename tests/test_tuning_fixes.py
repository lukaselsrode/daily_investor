"""
tests/test_tuning_fixes.py — regression coverage for the tuning-layer bug fixes.

Covers:
  1. apply_config_params round-trips EVERY tuned slot (the writer previously dropped
     candidate-filter / exit-floor / opportunity-cost / rebalance slots, so a tuned
     preset vector validated one config and persisted another).
  1b. random_tune._IDX_TO_CONFIG_PATH covers every slot, so best_weights_yaml emits
     real nested config paths instead of invalid flat keys.
  2. Score-weight normalization is gate-equivalent without threshold rescaling,
     because the simulator itself normalizes raw weights before composing the score.
  3. run_auto_tune / ParameterTuner.auto_tune gate the OOS report at the scope the
     parameters were tuned at (previously always "overall_strategy").
  4. Legacy random-tune path evaluates all candidates AND the current-config baseline
     on one shared seed (identical windows — paired comparison).
  5. Staged tune trains on the leading segment only, accepts a stage only when it
     improves on BOTH the tuning windows and a disjoint-seed re-evaluation, and
     validate_full_windowed scans tuning-disjoint windows (terminal holdout + offset
     seeds), with an honest residual-overlap note when no horizon fits the holdout.
  6. a) crashing configs are penalized (not scored 0.0, which outranked valid
        negative robust scores); b) _effective_bounds clamps config bounds on BOTH
        sides; c) AutoTuneResult.config_written mirrors the actual write predicate.
"""

import importlib
import os
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import yaml

from backtesting.types import SimResult


def _sim(total_return=0.12, sharpe=0.80, calmar=1.2, max_drawdown=-0.08, trades=40) -> SimResult:
    return SimResult(
        final_value=11_200.0,
        total_return=total_return,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=max_drawdown,
        trades_made=trades,
    )


# ---------------------------------------------------------------------------
# Fix 1 — apply_config_params round-trips every tuned slot
# ---------------------------------------------------------------------------

def _perturbed_full_vector(tc):
    """A full params vector with EVERY slot perturbed from the live config, built to
    be a fixed point of the writer's coercions (normalized weight groups, 4-decimal
    floats, ints for count/day slots, 0/1 booleans, harvest > trim + 0.01)."""
    cur = tc._current_params().copy()
    target = cur.copy()
    by_path = dict(tc._CONFIG_PATH_TO_PARAM_IDX)

    explicit = {
        # normalized score-weight group (sums to 1.0 — fixed point of normalization)
        "score_weights.value":    0.30,
        "score_weights.quality":  0.30,
        "score_weights.income":   0.20,
        "score_weights.momentum": 0.20,
        "index_pct":        0.70,   # above risk.min_index_pct so the clamp is a no-op
        "metric_threshold": 1.00,
        "sell_rules.take_profit_pct":       0.90,
        "sell_rules.sell_weak_value_below": 0.20,
        "sell_rules.trailing_stop_pct":    -0.20,
        "scoring.factors.value.pe_weight":  0.60,
        # normalized momentum sub-weight group (sums to 1.0)
        "scoring.momentum_inputs.weights.rs_3m":           0.30,
        "scoring.momentum_inputs.weights.rs_6m":           0.25,
        "scoring.momentum_inputs.weights.risk_adj_3m":     0.20,
        "scoring.momentum_inputs.weights.trend_structure": 0.15,
        "scoring.momentum_inputs.weights.return_1m":       0.05,
        "scoring.momentum_inputs.weights.return_5d":       0.05,
        # candidate-filter slots 40-42
        "candidate_selection.top_percentile":     0.25,
        "candidate_selection.min_quality_score":  0.50,
        "candidate_selection.min_momentum_score": 0.10,
        # position-sizing slots 43-45 (counts are ints)
        "risk.max_single_position_pct":       0.13,
        "risk.max_buys_per_rebalance":        6.0,
        "candidate_selection.max_candidates": 15.0,
        # regime / scoring blend slots 46-49
        "regime.bullish.momentum_tilt":          0.20,
        "regime.defensive.mean_reversion_blend": 0.10,
        "scoring.quality_low_vol_blend":         0.15,
        "scoring.momentum_residual_blend":       0.25,
        # DAE soft-exit floor slots 50-53
        "exit_decision.hard_exit_score_below":          0.30,
        "exit_decision.positive_momentum_review_floor": 0.05,
        "exit_decision.strong_quality_review_floor":    0.60,
        "exit_decision.thesis_intact_review_floor":     0.50,
        # opportunity-cost slots 54-56 (stall days is an int)
        "exit_decision.opportunity_cost.stall_max_days":          90.0,
        "exit_decision.opportunity_cost.reclaim_band":            0.05,
        "exit_decision.opportunity_cost.progress_momentum_floor": 0.20,
        # rebalance / cooldown slots 57-59 (all ints)
        "backtest.rebalance_frequency_days":    7.0,
        "backtest.cooldown_days_after_sell":    4.0,
        "backtest.cooldown_days_after_stopout": 10.0,
        # contribution-timing slots 72-79 (last group)
        "contribution_timing.multiplier.dip_sensitivity":      1.50,
        "contribution_timing.multiplier.neutral_dip_score":    0.40,
        "contribution_timing.multiplier.min_multiplier":       0.60,
        "contribution_timing.multiplier.max_multiplier":       2.50,
        "contribution_timing.dip_signal.weights.return_1w":    0.30,
        "contribution_timing.dip_signal.weights.return_1m":    0.30,
        "contribution_timing.dip_signal.weights.drawdown_20d": 0.25,
        "contribution_timing.dip_signal.weights.drawdown_60d": 0.10,
    }
    for path, val in explicit.items():
        target[by_path[path]] = val

    # archetype lifecycle + policy-switch boolean slots
    for arch in tc._ARCH_KEYS:
        trim_i = by_path[f"archetype_management.{arch}.trim_profit_threshold"]
        harv_i = by_path[f"archetype_management.{arch}.harvest_profit_threshold"]
        trail_i = by_path[f"archetype_management.{arch}.trailing_stop_pct"]
        hold_i = by_path[f"archetype_management.{arch}.minimum_hold_days"]
        target[trim_i] = round(float(cur[trim_i]) + 0.02, 4)
        target[harv_i] = round(float(target[trim_i]) + 0.30, 4)  # > trim + 0.01 sanity
        target[trail_i] = -0.15 if abs(float(cur[trail_i]) + 0.15) > 1e-9 else -0.16
        target[hold_i] = float(round(float(cur[hold_i])) + 3)
        for bfield, _suffix in tc._ARCH_BOOL_FIELDS:
            bi = by_path[f"archetype_management.{arch}.{bfield}"]
            target[bi] = 1.0 - round(float(cur[bi]))
    return cur, target


def test_apply_config_params_round_trips_every_slot(tmp_path):
    import core.paths
    import tuning.constants as tc
    import util
    from tuning import reports

    src_cfg = str(core.paths.CONFIG_FILE)
    tmp_cfg = tmp_path / "config.yaml"
    shutil.copy(src_cfg, tmp_cfg)

    cur, target = _perturbed_full_vector(tc)
    names = list(tc.PARAM_NAMES)
    unperturbed = [names[i] for i in range(len(target)) if abs(target[i] - cur[i]) < 1e-9]
    assert not unperturbed, (
        f"test setup must perturb every slot to detect drops; equal to current config: "
        f"{unperturbed} — adjust _perturbed_full_vector"
    )

    orig_reports_cfg = reports.CONFIG_FILE
    reports.CONFIG_FILE = str(tmp_cfg)
    try:
        reports.apply_config_params(target)
    finally:
        reports.CONFIG_FILE = orig_reports_cfg

    # Reload the written config through the SAME loader chain the tuner uses to BUILD
    # vectors: util module constants → tuning.constants._current_params().
    core.paths.CONFIG_FILE = str(tmp_cfg)
    try:
        importlib.reload(util)
        importlib.reload(tc)
        reloaded = tc._current_params()
        reloaded_names = list(tc.PARAM_NAMES)
    finally:
        core.paths.CONFIG_FILE = src_cfg
        importlib.reload(util)
        importlib.reload(tc)

    assert reloaded_names == names
    assert len(reloaded) == len(target)
    mismatches = {
        names[i]: {"written": float(target[i]), "reloaded": float(reloaded[i])}
        for i in range(len(target))
        if abs(float(reloaded[i]) - float(target[i])) > 1e-6
    }
    assert not mismatches, f"slots dropped or mangled by apply_config_params: {mismatches}"


# ---------------------------------------------------------------------------
# Fix 1b — best_weights_yaml emits real nested config paths for every slot
# ---------------------------------------------------------------------------

def test_best_weights_yaml_nests_extended_slots():
    from backtesting.random_walk import RandomWindowSummary
    from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX, PARAM_NAMES
    from tuning.random_tune import _IDX_TO_CONFIG_PATH, RandomTuneResult, WeightCandidate

    # The export mapping covers EVERY slot with its canonical config path.
    assert set(_IDX_TO_CONFIG_PATH) == set(range(len(PARAM_NAMES)))
    assert set(_IDX_TO_CONFIG_PATH.values()) == set(_CONFIG_PATH_TO_PARAM_IDX)

    paths = [
        "candidate_selection.top_percentile",
        "exit_decision.hard_exit_score_below",
        "backtest.rebalance_frequency_days",
    ]
    idxs = [_CONFIG_PATH_TO_PARAM_IDX[p] for p in paths]
    names = [PARAM_NAMES[i] for i in idxs]
    values = np.array([0.25, 0.31, 7.0])
    full = np.zeros(len(PARAM_NAMES))
    for i, v in zip(idxs, values):
        full[i] = v
    cand = WeightCandidate(
        sample_id=0,
        active_values=values,
        active_names=names,
        full_params=full,
        summary=RandomWindowSummary(n_windows=3, window_days=20, params_used=full.copy()),
        robust_score=0.1,
    )
    res = RandomTuneResult(
        n_samples=1, n_windows=3, window_days=20, seed=1,
        active_param_names=names, candidates=[cand], best_candidate=cand,
    )
    parsed = yaml.safe_load(res.best_weights_yaml())
    assert parsed["candidate_selection"]["top_percentile"] == pytest.approx(0.25)
    assert parsed["exit_decision"]["hard_exit_score_below"] == pytest.approx(0.31)
    assert parsed["backtest"]["rebalance_frequency_days"] == pytest.approx(7.0)
    # No invalid flat top-level keys (the old behavior emitted e.g. `cs_top_percentile:`).
    assert all(not k.startswith(("cs_", "ef_", "rb_", "oc_", "ps_")) for k in parsed)


# ---------------------------------------------------------------------------
# Fix 2 — weight normalization is gate-equivalent without threshold rescaling
# ---------------------------------------------------------------------------

def test_score_weight_normalization_is_gate_equivalent(tmp_path, monkeypatch):
    """The simulator normalizes raw score weights before composing the score, so the
    gate the optimizer validates is normalized_weights·scores >= raw_threshold. The
    writer persists normalized weights with the UNSCALED threshold — and that written
    config reproduces the exact buy-gate decisions of the validated raw vector."""
    import tuning.constants as tc
    from backtesting.simulator import _regime_tilted_weights
    from tuning import reports

    raw = np.array([0.20, 0.10, 0.12, 0.08])  # raw weights summing to 0.5 — NOT normalized
    sum_raw = float(raw.sum())
    assert sum_raw == pytest.approx(0.5)

    base = tc._current_params().copy()
    base[:4] = raw
    threshold = float(base[5])

    # 1. The simulator's weight pipeline normalizes (precomp untouched for len<=46 vectors).
    sim_w = _regime_tilted_weights(raw, base[:16], None, 0)
    np.testing.assert_allclose(sim_w, raw / sum_raw)

    # 2. The writer persists normalized weights and the threshold UNCHANGED.
    src_cfg = os.path.join(os.path.dirname(__file__), "..", "cfg", "config.yaml")
    tmp_cfg = tmp_path / "config.yaml"
    shutil.copy(src_cfg, tmp_cfg)
    monkeypatch.setattr(reports, "CONFIG_FILE", str(tmp_cfg))
    reports.apply_config_params(base)
    with open(tmp_cfg) as f:
        written = yaml.safe_load(f)
    w = written["score_weights"]
    written_w = np.array([w["value"], w["quality"], w["income"], w["momentum"]])
    np.testing.assert_allclose(written_w, raw / sum_raw, atol=1e-4)
    assert written["metric_threshold"] == pytest.approx(threshold, abs=1e-4)

    # 3. Gate equivalence on an arbitrary factor-score cross-section: the simulator
    #    re-normalizes the written weights (a no-op), so composite >= threshold
    #    decisions match the validated raw-vector run exactly.
    rng = np.random.default_rng(0)
    scores = rng.normal(loc=threshold, scale=1.0, size=(4, 256))
    relived_w = _regime_tilted_weights(written_w, base[:16], None, 0)
    validated_gate = (sim_w @ scores) >= threshold
    written_gate = (relived_w @ scores) >= threshold
    assert np.array_equal(validated_gate, written_gate)


# ---------------------------------------------------------------------------
# Fix 3 — OOS gating happens at the scope the parameters were tuned at
# ---------------------------------------------------------------------------

def test_run_auto_tune_gates_at_tuned_scope(monkeypatch):
    import tuning.constants as tc
    import tuning.tuner as tt

    captured = {}

    def fake_report(precomp, params, train_sl, val_sl, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(validation_result=_sim(), validation_benchmark_return=0.0)

    p = tc._current_params()
    monkeypatch.setattr(tt, "run_backtest_report", fake_report)
    monkeypatch.setattr(tt, "_run_single", lambda *a, **k: (p.copy(), _sim()))
    monkeypatch.setattr(tt, "run_simulation", lambda *a, **k: _sim())
    # The paired random-window and multi-horizon gates would need real precomps;
    # this test only checks the scope plumbing of the split-gate report calls.
    monkeypatch.setattr(tt, "paired_random_window_gate", lambda *a, **k: (True, [], {}))
    monkeypatch.setattr(tt, "multi_horizon_confirm", lambda *a, **k: (True, [], []))
    fake_precomp = MagicMock()
    fake_precomp.mode = "test"
    fake_precomp.lookahead_bias_level = "low"
    monkeypatch.setattr(tt, "load_and_precompute", lambda *a, **k: fake_precomp)

    tt.run_auto_tune(n_days=90, scope="active_sleeve_compounding", apply=False)
    assert captured.get("scope") == "active_sleeve_compounding"

    captured.clear()
    tt.run_auto_tune(n_days=90, apply=False)  # plain run keeps the default scope
    assert captured.get("scope") == "overall_strategy"


def _patch_auto_tune_revalidation(monkeypatch, validator_result=(True, [])):
    """Patch ParameterTuner.auto_tune's collaborators; returns (tuner_module, captured)."""
    import backtesting.validator as bv
    import tuning.constants as tc
    import tuning.tuner as tt

    p = tc._current_params()
    raw = (p, _sim(), _sim(), _sim(), p, p + 0.01)
    monkeypatch.setattr(tt, "run_auto_tune", lambda **k: raw)
    monkeypatch.setattr(
        tt, "load_and_precompute",
        lambda *a, **k: SimpleNamespace(prices=np.zeros((100, 3))),
    )
    captured = {}

    def fake_report(precomp, params, train_sl, val_sl, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(tt, "run_backtest_report", fake_report)

    class _FakeValidator:
        def validate_report(self, report, bp):
            return validator_result

    monkeypatch.setattr(bv, "WalkForwardValidator", _FakeValidator)
    return tt, captured


def test_parameter_tuner_auto_tune_revalidates_at_tuned_scope(monkeypatch):
    tt, captured = _patch_auto_tune_revalidation(monkeypatch)
    tt.ParameterTuner().auto_tune(
        n_days=90, scope="active_sleeve_compounding", preset="active_exit_floors",
    )
    assert captured.get("scope") == "active_sleeve_compounding"


# ---------------------------------------------------------------------------
# Fix 4 — legacy random tune: shared windows for all candidates + baseline
# ---------------------------------------------------------------------------

def test_legacy_random_tune_shares_windows_across_candidates_and_baseline(monkeypatch):
    import tuning.constants as tc
    import tuning.random_tune as rt
    from backtesting.random_walk import RandomWindowSummary

    seeds = []

    def fake_rwb(precomp, params, **kwargs):
        seeds.append(kwargs["seed"])
        return RandomWindowSummary(
            n_windows=kwargs["n_windows"],
            window_days=kwargs["window_days"],
            params_used=np.asarray(params).copy(),
            median_excess_return=0.01,
            pct_beating_benchmark=0.5,
            robust_score=0.1,
        )

    monkeypatch.setattr(rt, "random_window_backtest", fake_rwb)
    precomp = SimpleNamespace(prices=np.zeros((300, 4)))
    res = rt.run_random_weight_tune(
        precomp,
        base_params=tc._current_params().copy(),
        n_samples=4,
        n_windows=3,
        window_days=30,
        seed=123,
    )
    assert len(seeds) == 5, "4 candidates + the current-config baseline"
    assert set(seeds) == {123}, (
        "all candidates AND the baseline must share one seed (identical windows) so "
        "ranking deltas are paired, not window luck"
    )
    assert res.current_summary is not None


# ---------------------------------------------------------------------------
# Fix 5 — staged tune: leading-segment training + disjoint-seed acceptance,
#          validation on tuning-disjoint windows
# ---------------------------------------------------------------------------

def _staged_tune_scaffold(monkeypatch, cand_check_score):
    import tuning.constants as tc
    import tuning.interaction_screen as iscr
    import tuning.robust_scan as rs
    import tuning.staged_tune as st
    from tuning.interaction_screen import MarginalResult

    train_obj = SimpleNamespace(tag="train")
    slices = []
    monkeypatch.setattr(
        st, "_slice_window_precomp",
        lambda precomp, s: (slices.append(s), train_obj)[1],
    )

    cand = tc._current_params().copy()
    cand[6] = round(float(cand[6]) + 0.05, 4)

    def fake_scan(precomp, params=None, run_matrix=None, scope="active_sleeve_compounding",
                  regime_scope="all", **kwargs):
        assert precomp is train_obj, "tuning + acceptance checks must only see the train segment"
        shifted = int(run_matrix[0]["seed"]) >= st._VALIDATION_SEED_OFFSET
        is_cand = abs(float(params[6]) - float(cand[6])) < 1e-9
        table = {
            (False, False): 0.10,             # baseline on the tuning windows
            (True, False):  0.10,             # baseline on the disjoint-seed windows
            (False, True):  0.50,             # candidate looks great on tuning windows
            (True, True):   cand_check_score,  # candidate on the disjoint-seed windows
        }
        return SimpleNamespace(overall_robust_score=table[(shifted, is_cand)])

    monkeypatch.setattr(rs, "run_robust_scan", fake_scan)

    def fake_tune_subset(precomp, preset, run_matrix, scope, maxiter, popsize,
                         seed=42, baseline=None, regime_scope="all"):
        assert precomp is train_obj
        return MarginalResult(name=preset, score=0.5, params=cand.copy(), active=[6])

    monkeypatch.setattr(iscr, "_tune_subset", fake_tune_subset)
    return st, tc, cand, slices


def test_run_staged_tune_rejects_stage_that_fails_disjoint_seed_check(monkeypatch):
    st, tc, cand, slices = _staged_tune_scaffold(monkeypatch, cand_check_score=0.05)
    precomp = SimpleNamespace(prices=np.zeros((100, 2)))
    run_matrix = [{"horizon_days": 20, "seed": 1, "n_windows": 3}]

    out = st.run_staged_tune(precomp, ["active_exit_ladder"], run_matrix)
    # tuning restricted to the leading train_frac (70%) of the history
    assert slices[0] == slice(0, 70)
    # improvement on the tuning windows alone (0.5 > 0.1) is NOT enough: the
    # disjoint-seed re-evaluation regressed (0.05 < 0.1) → seed noise → rejected.
    assert out.stages and out.stages[0].accepted is False
    assert out.accepted_clusters == []
    np.testing.assert_allclose(out.final_params, tc._current_params())


def test_run_staged_tune_accepts_stage_confirmed_on_disjoint_seeds(monkeypatch):
    st, tc, cand, slices = _staged_tune_scaffold(monkeypatch, cand_check_score=0.40)
    precomp = SimpleNamespace(prices=np.zeros((100, 2)))
    run_matrix = [{"horizon_days": 20, "seed": 1, "n_windows": 3}]

    out = st.run_staged_tune(precomp, ["active_exit_ladder"], run_matrix)
    assert out.stages and out.stages[0].accepted is True
    assert out.accepted_clusters == ["active_exit_ladder"]
    np.testing.assert_allclose(out.final_params, cand)


def _patch_validate_scan(monkeypatch):
    import tuning.robust_scan as rs
    import tuning.staged_tune as st

    holdout_obj = SimpleNamespace(tag="holdout")
    slices = []
    monkeypatch.setattr(
        st, "_slice_window_precomp",
        lambda precomp, s: (slices.append(s), holdout_obj)[1],
    )
    calls = []

    def fake_scan(precomp, params=None, run_matrix=None, scope="active_sleeve_compounding", **kw):
        calls.append((
            precomp,
            [int(c["seed"]) for c in run_matrix],
            [int(c["horizon_days"]) for c in run_matrix],
        ))
        return SimpleNamespace(
            overall_robust_score=0.2,
            overfit_warning_score=lambda: 0.0,
            horizon_heatmap_df=lambda: None,
        )

    monkeypatch.setattr(rs, "run_robust_scan", fake_scan)
    return st, holdout_obj, slices, calls


def test_validate_full_windowed_scans_disjoint_holdout_windows(monkeypatch):
    st, holdout_obj, slices, calls = _patch_validate_scan(monkeypatch)
    precomp = SimpleNamespace(prices=np.zeros((100, 2)))
    run_matrix = [
        {"horizon_days": 20, "seed": 7, "n_windows": 3},
        {"horizon_days": 80, "seed": 8, "n_windows": 3},  # too long for the 30d holdout
    ]
    out = st.validate_full_windowed(precomp, np.zeros(16), run_matrix=run_matrix)
    assert slices == [slice(70, 100)], "scan must run on the terminal holdout segment"
    pre, seeds_used, horizons = calls[0]
    assert pre is holdout_obj
    assert seeds_used == [7 + st._VALIDATION_SEED_OFFSET], "validation seeds must be disjoint"
    assert horizons == [20], "horizons that do not fit the holdout are dropped"
    assert "terminal holdout" in out["validation_note"]
    assert out["robust_score"] == pytest.approx(0.2)


def test_validate_full_windowed_falls_back_with_residual_overlap_note(monkeypatch):
    st, holdout_obj, slices, calls = _patch_validate_scan(monkeypatch)
    precomp = SimpleNamespace(prices=np.zeros((100, 2)))
    run_matrix = [{"horizon_days": 80, "seed": 7, "n_windows": 3}]  # no horizon fits
    out = st.validate_full_windowed(precomp, np.zeros(16), run_matrix=run_matrix)
    assert slices == [], "no holdout slice possible — full history fallback"
    pre, seeds_used, horizons = calls[0]
    assert pre is precomp
    assert seeds_used == [7 + st._VALIDATION_SEED_OFFSET], "seeds still disjoint in fallback"
    assert horizons == [80]
    assert "RESIDUAL OVERLAP" in out["validation_note"]


# ---------------------------------------------------------------------------
# Fix 6a — crashing configs must rank below valid (often negative) scores
# ---------------------------------------------------------------------------

def test_staged_robust_score_penalizes_crashes(monkeypatch):
    import tuning.robust_scan as rs
    import tuning.staged_tune as st

    def boom(*a, **k):
        raise RuntimeError("crash")

    monkeypatch.setattr(rs, "run_robust_scan", boom)
    score = st._robust_score(object(), np.zeros(16), [], "active_sleeve_compounding")
    assert score <= -1e6


def test_interaction_screen_objective_penalizes_crashes(monkeypatch):
    import tuning.robust_scan as rs
    from tuning.interaction_screen import _tune_subset

    def boom(*a, **k):
        raise RuntimeError("crash")

    monkeypatch.setattr(rs, "run_robust_scan", boom)
    # Tiny real DE run over a crashing objective: every eval returns the +1e6 penalty
    # (DE minimizes), so the reported best score is -1e6 — below any valid config.
    m = _tune_subset(None, "active_exit_floors", [], "active_sleeve_compounding", 1, 4)
    assert m is not None
    assert m.score <= -1e6


# ---------------------------------------------------------------------------
# Fix 6b — _effective_bounds clamps config bounds on both sides
# ---------------------------------------------------------------------------

def test_effective_bounds_clamps_config_min_and_max(monkeypatch):
    import tuning.constants as tc

    idx = tc._CONFIG_PATH_TO_PARAM_IDX["metric_threshold"]
    eng_lo, eng_hi = tc.BOUNDS[idx]

    monkeypatch.setattr(tc, "TUNING_PARAMS", {
        "frozen_parameters": [],
        "parameter_bounds": {"metric_threshold": {"min": eng_lo - 5.0, "max": eng_hi + 5.0}},
    })
    lo, hi = tc._effective_bounds()[idx]
    assert lo == eng_lo, "config min may only TIGHTEN, never widen below the engineered floor"
    assert hi == eng_hi, "config max may only TIGHTEN, never widen above the engineered cap"

    width = eng_hi - eng_lo
    monkeypatch.setattr(tc, "TUNING_PARAMS", {
        "frozen_parameters": [],
        "parameter_bounds": {"metric_threshold": {
            "min": eng_lo + width * 0.25, "max": eng_hi - width * 0.25,
        }},
    })
    lo2, hi2 = tc._effective_bounds()[idx]
    assert lo2 == pytest.approx(eng_lo + width * 0.25), "tighter config min is preserved"
    assert hi2 == pytest.approx(eng_hi - width * 0.25), "tighter config max is preserved"


# ---------------------------------------------------------------------------
# Fix 6c — config_written mirrors the predicate the write actually uses
# ---------------------------------------------------------------------------

def test_config_written_flag_matches_write_predicate(monkeypatch):
    from util import BACKTEST_PARAMS

    tt, _ = _patch_auto_tune_revalidation(monkeypatch, validator_result=(True, []))
    auto_apply = bool(BACKTEST_PARAMS.get("auto_apply_if_valid", False))

    res = tt.ParameterTuner().auto_tune(n_days=90, apply=True)
    assert res.validation_passed is True
    assert res.config_written is True, "apply + passing validation writes config"

    res = tt.ParameterTuner().auto_tune(n_days=90, apply=False)
    assert res.config_written is auto_apply, (
        "without --apply the write happens iff backtest.auto_apply_if_valid — the old "
        "flag ignored that branch"
    )


def test_config_written_flag_true_on_force_apply_despite_failed_validation(monkeypatch):
    tt, _ = _patch_auto_tune_revalidation(monkeypatch, validator_result=(False, ["nope"]))
    res = tt.ParameterTuner().auto_tune(n_days=90, apply=False, force_apply=True)
    assert res.validation_passed is False
    assert res.config_written is True, "force_apply writes unconditionally"
