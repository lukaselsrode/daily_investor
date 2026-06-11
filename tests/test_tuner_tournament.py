"""
tests/test_tuner_tournament.py — the tuned-candidate tournament in run_auto_tune.

The blind (sharpe + calmar) / 2 midpoint was replaced by a tournament: the two
optima, three sharpe/calmar blends, and three incumbent blends are evaluated on
the same train/validation split, gated individually (validate_tuned_params),
Pareto-filtered, and ranked by an isolated selection score. The gates remain
the final authority; the random-window gate runs ONLY on the selected candidate.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from backtesting.types import SimResult
from tuning.tuner import (
    _pareto_non_dominated,
    build_tournament_candidates,
    candidate_selection_score,
)
from util import BACKTEST_PARAMS

# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

EXPECTED_IDS = [
    "sharpe_opt", "calmar_opt", "avg_50_50", "avg_25_75", "avg_75_25",
    "incumbent_blend_25", "incumbent_blend_50", "incumbent_blend_75",
]


class TestCandidateConstruction:

    def test_tournament_includes_all_expected_candidates(self):
        s, c, i = np.full(4, 0.2), np.full(4, 0.8), np.full(4, 0.4)
        cands = build_tournament_candidates(s, c, i)
        assert list(cands) == EXPECTED_IDS

    def test_blend_vectors_constructed_correctly(self):
        s, c, i = np.full(4, 0.2), np.full(4, 0.8), np.full(4, 0.4)
        cands = build_tournament_candidates(s, c, i)
        avg = 0.5 * s + 0.5 * c
        assert np.allclose(cands["avg_50_50"], avg)
        assert np.allclose(cands["avg_25_75"], 0.25 * s + 0.75 * c)
        assert np.allclose(cands["avg_75_25"], 0.75 * s + 0.25 * c)
        assert np.allclose(cands["incumbent_blend_25"], 0.75 * i + 0.25 * avg)
        assert np.allclose(cands["incumbent_blend_50"], 0.50 * i + 0.50 * avg)
        assert np.allclose(cands["incumbent_blend_75"], 0.25 * i + 0.75 * avg)

    def test_candidates_are_copies(self):
        s, c, i = np.full(4, 0.2), np.full(4, 0.8), np.full(4, 0.4)
        cands = build_tournament_candidates(s, c, i)
        cands["sharpe_opt"][0] = 99.0
        assert s[0] == 0.2  # input vector untouched


# ---------------------------------------------------------------------------
# Selection score + Pareto filter
# ---------------------------------------------------------------------------

def _metrics(excess=0.05, vs_inc=0.01, sharpe=1.0, calmar=1.0,
             dd=-0.10, turnover=1.0, turn_mult=1.0, dd_worse=0.0):
    return {
        "val_excess": excess, "val_excess_vs_incumbent": vs_inc,
        "val_sharpe": sharpe, "val_calmar": calmar,
        "val_max_drawdown": dd, "val_turnover": turnover,
        "turnover_multiple": turn_mult, "drawdown_worse_than_incumbent": dd_worse,
    }


class TestSelectionScore:

    def test_turnover_penalty_only_above_one(self):
        base = candidate_selection_score(_metrics(turn_mult=1.0))
        assert candidate_selection_score(_metrics(turn_mult=0.5)) == pytest.approx(base)
        assert candidate_selection_score(_metrics(turn_mult=2.0)) < base

    def test_drawdown_penalty(self):
        base = candidate_selection_score(_metrics())
        assert candidate_selection_score(_metrics(dd_worse=0.05)) < base

    def test_excess_dominates(self):
        low = candidate_selection_score(_metrics(excess=0.01, vs_inc=0.0))
        high = candidate_selection_score(_metrics(excess=0.10, vs_inc=0.09))
        assert high > low


class TestParetoFilter:

    def test_dominated_candidate_excluded(self):
        rows = [
            {"candidate_id": "winner", "metrics": _metrics(excess=0.06, calmar=1.2, dd=-0.08, turnover=0.9)},
            {"candidate_id": "dominated", "metrics": _metrics(excess=0.04, calmar=1.0, dd=-0.10, turnover=1.1)},
        ]
        nd = _pareto_non_dominated(rows)
        assert nd == {"winner"}

    def test_tradeoff_candidates_both_survive(self):
        rows = [
            {"candidate_id": "high_excess", "metrics": _metrics(excess=0.08, dd=-0.15)},
            {"candidate_id": "low_dd", "metrics": _metrics(excess=0.04, dd=-0.05)},
        ]
        nd = _pareto_non_dominated(rows)
        assert nd == {"high_excess", "low_dd"}


# ---------------------------------------------------------------------------
# run_auto_tune end-to-end (mocked engine, real selection/gate plumbing)
# ---------------------------------------------------------------------------

def _sim(total_return=0.15, sharpe=1.0, calmar=1.2, max_drawdown=-0.08, turnover=1.0):
    return SimResult(
        final_value=11_000.0, total_return=total_return, sharpe=sharpe,
        calmar=calmar, max_drawdown=max_drawdown, trades_made=10,
        turnover_estimate=turnover,
    )


def _report(val_sim):
    return SimpleNamespace(
        validation_result=val_sim,
        validation_benchmark_return=0.0,
        train_result=_sim(),
        benchmark_return=0.0, benchmark_sharpe=0.5, benchmark_max_drawdown=-0.1,
        excess_return=0.0, notes=[],
    )


def _wire_auto_tune(monkeypatch, metrics_for_params, sharpe_vec, calmar_vec):
    """Patch run_auto_tune's collaborators. `metrics_for_params(params)` returns
    the validation SimResult for any candidate/incumbent vector."""
    import tuning.tuner as tt

    captured = {"rw_params": None, "applied": []}

    monkeypatch.setattr(tt, "_run_single", lambda pc, obj, *a, **k:
                        (sharpe_vec.copy(), _sim()) if obj == "sharpe" else (calmar_vec.copy(), _sim()))
    monkeypatch.setattr(tt, "run_simulation", lambda *a, **k: _sim())
    monkeypatch.setattr(tt, "run_backtest_report",
                        lambda precomp, params, tr, vl, **kw: _report(metrics_for_params(np.asarray(params))))

    def fake_rw(params, incumbent, bp, **kw):
        captured["rw_params"] = np.asarray(params).copy()
        return True, [], {"skipped": True}
    monkeypatch.setattr(tt, "paired_random_window_gate", fake_rw)
    monkeypatch.setattr(tt, "apply_config_params", lambda p: captured["applied"].append(np.asarray(p).copy()))

    fake_precomp = MagicMock()
    fake_precomp.mode = "test"
    fake_precomp.lookahead_bias_level = "low"
    monkeypatch.setattr(tt, "load_and_precompute", lambda *a, **k: fake_precomp)
    return tt, captured


def _passing_sim(turnover=1.0, total_return=0.15, calmar=1.2, max_drawdown=-0.05):
    """A validation SimResult that clears the LIVE absolute gates."""
    bp = BACKTEST_PARAMS
    return _sim(
        total_return=max(total_return, bp["min_validation_excess_return"] + 0.05),
        sharpe=bp["min_validation_sharpe"] + 1.0,
        calmar=calmar,
        max_drawdown=max(max_drawdown, bp["max_validation_drawdown"] + 0.05),
        turnover=turnover,
    )


class TestRunAutoTuneTournament:

    # Candidate vectors are distinguished by slot 0 (distinct for every blend).
    SHARPE = 0.2
    CALMAR = 0.8

    def _vec(self, v):
        from tuning.constants import PARAM_NAMES
        out = np.zeros(len(PARAM_NAMES))
        out[0] = v
        return out

    def test_selected_winner_feeds_random_window_gate_and_return(self, monkeypatch):
        """calmar_opt is engineered to win — the rw gate and the returned tuple
        must carry calmar_opt's vector, NOT the blind average."""
        import tuning.constants as tc
        inc0 = float(tc._current_params()[0])

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - self.CALMAR) < 1e-9:          # calmar_opt → clear winner
                return _passing_sim(total_return=0.40)
            if abs(sig - inc0) < 1e-9:                  # incumbent → modest
                return _passing_sim(total_return=0.10)
            return _passing_sim(total_return=0.12)      # everything else → mid

        tt, cap = _wire_auto_tune(monkeypatch, metrics_for,
                                  self._vec(self.SHARPE), self._vec(self.CALMAR))
        out = tt.run_auto_tune(n_days=90, apply=False)
        assert len(out) == 6  # tuple shape preserved
        selected, _, _, _, sharpe_p, calmar_p = out
        assert selected[0] == pytest.approx(self.CALMAR)
        assert cap["rw_params"] is not None
        assert cap["rw_params"][0] == pytest.approx(self.CALMAR)
        # blind average would have been 0.5 — must NOT be what the rw gate saw
        assert cap["rw_params"][0] != pytest.approx((self.SHARPE + self.CALMAR) / 2)
        assert sharpe_p[0] == pytest.approx(self.SHARPE)
        assert calmar_p[0] == pytest.approx(self.CALMAR)

    def test_failed_candidates_not_selected(self, monkeypatch):
        """The best-scoring candidate fails the turnover gate — a passer with a
        lower score must be selected instead."""
        import tuning.constants as tc
        inc0 = float(tc._current_params()[0])
        turn_mult = BACKTEST_PARAMS.get("max_turnover_multiple", 2.0)

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - inc0) < 1e-9:
                return _passing_sim(total_return=0.10, turnover=1.0)
            if abs(sig - self.CALMAR) < 1e-9:           # huge score but churns
                return _passing_sim(total_return=0.60, turnover=1.0 * (turn_mult + 1.0))
            if abs(sig - self.SHARPE) < 1e-9:           # modest but clean
                return _passing_sim(total_return=0.20, turnover=1.0)
            return _passing_sim(total_return=0.11, turnover=1.0)

        tt, cap = _wire_auto_tune(monkeypatch, metrics_for,
                                  self._vec(self.SHARPE), self._vec(self.CALMAR))
        selected = tt.run_auto_tune(n_days=90, apply=False)[0]
        assert selected[0] == pytest.approx(self.SHARPE)

    def test_all_candidates_failing_means_no_config_write(self, monkeypatch, capsys):
        def metrics_for(params):
            # Every candidate (and the incumbent) below the live Sharpe floor.
            return _sim(total_return=0.30,
                        sharpe=BACKTEST_PARAMS["min_validation_sharpe"] - 1.0)

        tt, cap = _wire_auto_tune(monkeypatch, metrics_for,
                                  self._vec(self.SHARPE), self._vec(self.CALMAR))
        out = tt.run_auto_tune(n_days=90, apply=True)
        assert cap["applied"] == []                  # nothing written
        assert cap["rw_params"] is None              # rw gate never ran
        assert len(out) == 6
        text = capsys.readouterr().out
        assert "Selected candidate: none" in text

    def test_reporting_includes_selected_candidate_id(self, monkeypatch, capsys):
        import tuning.constants as tc
        inc0 = float(tc._current_params()[0])

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - inc0) < 1e-9:
                return _passing_sim(total_return=0.10)
            if abs(sig - self.SHARPE) < 1e-9:
                return _passing_sim(total_return=0.50)
            return _passing_sim(total_return=0.12)

        tt, _ = _wire_auto_tune(monkeypatch, metrics_for,
                                self._vec(self.SHARPE), self._vec(self.CALMAR))
        tt.run_auto_tune(n_days=90, apply=False)
        text = capsys.readouterr().out
        assert "Selected candidate: sharpe_opt" in text
        assert "candidate_id" in text                # tournament table header
        for cid in EXPECTED_IDS:
            assert cid in text

    def test_incumbent_blend_can_win(self, monkeypatch):
        import tuning.constants as tc
        inc = tc._current_params()
        avg0 = (self.SHARPE + self.CALMAR) / 2
        blend50 = 0.5 * float(inc[0]) + 0.5 * avg0

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - blend50) < 1e-9:
                return _passing_sim(total_return=0.50)
            if abs(sig - float(inc[0])) < 1e-9:
                return _passing_sim(total_return=0.10)
            return _passing_sim(total_return=0.12)

        tt, _ = _wire_auto_tune(monkeypatch, metrics_for,
                                self._vec(self.SHARPE), self._vec(self.CALMAR))
        selected = tt.run_auto_tune(n_days=90, apply=False)[0]
        assert selected[0] == pytest.approx(blend50)
