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

    def fake_mh(params, incumbent, bp, **kw):
        captured["mh_params"] = np.asarray(params).copy()
        return True, [], []
    monkeypatch.setattr(tt, "multi_horizon_confirm", fake_mh)

    def fake_sg(params, incumbent, bp, **kw):
        captured["sg_params"] = np.asarray(params).copy()
        return True, [], []
    monkeypatch.setattr(tt, "stress_gauntlet", fake_sg)
    monkeypatch.setattr(tt, "apply_config_params", lambda p: captured["applied"].append(np.asarray(p).copy()))

    fake_precomp = MagicMock()
    fake_precomp.mode = "test"
    fake_precomp.lookahead_bias_level = "low"
    # The tournament's random-search hook needs a REAL train_days from the
    # sliced precomp (window-size arithmetic); a bare MagicMock would raise
    # inside the try/except and silently skip the random candidates.
    fake_precomp._replace.return_value = SimpleNamespace(prices=np.zeros((63, 3)))
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

    def test_selected_vector_persisted_as_lead(self, monkeypatch):
        """The tournament winner is saved as a .npy lead — a gate-blocked
        near-miss from a multi-hour tune must be recoverable via --leads."""
        import os

        import data.cache as dc

        def metrics_for(params):
            return _passing_sim(
                total_return=0.40 if abs(float(params[0]) - self.CALMAR) < 1e-9 else 0.10
            )

        tt, _ = _wire_auto_tune(monkeypatch, metrics_for,
                                self._vec(self.SHARPE), self._vec(self.CALMAR))
        out = tt.run_auto_tune(n_days=90, apply=False)
        lead = os.path.join(dc.DATA_DIRECTORY, "leads_selected_90d.npy")
        assert os.path.exists(lead)
        np.testing.assert_allclose(np.load(lead), out[0])

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


# ---------------------------------------------------------------------------
# Multi-source candidate pool
# ---------------------------------------------------------------------------

class TestMultiSourcePool:

    def test_pool_includes_all_sources(self):
        from tuning.tuner import assemble_candidate_pool
        s, c, i = np.full(4, 0.2), np.full(4, 0.8), np.full(4, 0.4)
        pool = assemble_candidate_pool(
            s, c, i,
            random_topk_vectors={"random_top1": np.full(4, 0.3)},
            lead_vectors={"lead:round_f": np.full(4, 0.6)},
            manual_vectors={"my_vec": np.full(4, 0.7)},
        )
        for cid in EXPECTED_IDS:
            assert cid in pool
        assert pool["random_top1"]["source"] == "random_search"
        assert pool["lead:round_f"]["source"] == "lead"
        assert pool["my_vec"]["source"] == "manual"
        assert np.allclose(pool["my_vec"]["params"], 0.7)

    def test_short_vector_padded_with_incumbent_tail(self):
        from tuning.tuner import assemble_candidate_pool
        s, c, i = np.full(6, 0.2), np.full(6, 0.8), np.arange(6, dtype=float)
        pool = assemble_candidate_pool(s, c, i, lead_vectors={"old_lead": np.array([9.0, 9.0])})
        fitted = pool["old_lead"]["params"]
        assert np.allclose(fitted[:2], 9.0)
        assert np.allclose(fitted[2:], i[2:])  # incumbent tail fills new slots

    def test_overlong_vector_rejected(self):
        from tuning.tuner import assemble_candidate_pool
        s, c, i = np.full(4, 0.2), np.full(4, 0.8), np.full(4, 0.4)
        pool = assemble_candidate_pool(s, c, i, lead_vectors={"future_layout": np.zeros(99)})
        assert "future_layout" not in pool

    def test_random_topk_candidate_can_win(self, monkeypatch):
        """run_random_weight_tune's top vector enters the pool and is selected
        when it dominates on validation."""
        import tuning.constants as tc
        import tuning.random_tune as rt
        inc0 = float(tc._current_params()[0])
        RAND0 = 0.91

        def fake_random_tune(*a, **k):
            from tuning.constants import PARAM_NAMES
            vec = np.zeros(len(PARAM_NAMES))
            vec[0] = RAND0
            cand = SimpleNamespace(full_params=vec)
            return SimpleNamespace(candidates=[cand])
        monkeypatch.setattr(rt, "run_random_weight_tune", fake_random_tune)

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - RAND0) < 1e-9:
                return _passing_sim(total_return=0.60)
            if abs(sig - inc0) < 1e-9:
                return _passing_sim(total_return=0.10)
            return _passing_sim(total_return=0.12)

        tt, cap = _wire_auto_tune(monkeypatch, metrics_for,
                                  np.zeros(len(tc.PARAM_NAMES)) + 0.2,
                                  np.zeros(len(tc.PARAM_NAMES)) + 0.8)
        selected = tt.run_auto_tune(n_days=90, apply=False, random_topk=1)[0]
        assert selected[0] == pytest.approx(RAND0)
        assert cap["rw_params"][0] == pytest.approx(RAND0)

    def test_saved_lead_vector_included_from_file(self, monkeypatch, tmp_path, capsys):
        import tuning.constants as tc
        inc0 = float(tc._current_params()[0])
        LEAD0 = 0.77
        vec = np.zeros(len(tc.PARAM_NAMES))
        vec[0] = LEAD0
        lead_path = tmp_path / "round_f_candidate.npy"
        np.save(lead_path, vec)

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - LEAD0) < 1e-9:
                return _passing_sim(total_return=0.60)
            if abs(sig - inc0) < 1e-9:
                return _passing_sim(total_return=0.10)
            return _passing_sim(total_return=0.12)

        tt, _ = _wire_auto_tune(monkeypatch, metrics_for,
                                np.zeros(len(tc.PARAM_NAMES)) + 0.2,
                                np.zeros(len(tc.PARAM_NAMES)) + 0.8)
        selected = tt.run_auto_tune(n_days=90, apply=False,
                                    lead_vector_paths=[str(lead_path)])[0]
        assert selected[0] == pytest.approx(LEAD0)
        text = capsys.readouterr().out
        assert "lead:round_f_candidate" in text
        assert "lead" in text  # source column

    def test_manual_extra_candidate_included(self, monkeypatch):
        import tuning.constants as tc
        inc0 = float(tc._current_params()[0])
        MAN0 = 0.66
        vec = np.zeros(len(tc.PARAM_NAMES))
        vec[0] = MAN0

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - MAN0) < 1e-9:
                return _passing_sim(total_return=0.60)
            if abs(sig - inc0) < 1e-9:
                return _passing_sim(total_return=0.10)
            return _passing_sim(total_return=0.12)

        tt, _ = _wire_auto_tune(monkeypatch, metrics_for,
                                np.zeros(len(tc.PARAM_NAMES)) + 0.2,
                                np.zeros(len(tc.PARAM_NAMES)) + 0.8)
        selected = tt.run_auto_tune(n_days=90, apply=False,
                                    extra_candidates={"my_lead": vec})[0]
        assert selected[0] == pytest.approx(MAN0)


# ---------------------------------------------------------------------------
# DE turnover penalty
# ---------------------------------------------------------------------------

class TestDeTurnoverPenalty:

    BP = {
        "de_turnover_penalty_enabled": True,
        "de_turnover_penalty_vs_incumbent": True,
        "de_turnover_soft_limit_multiple": 1.5,
        "de_turnover_hard_limit_multiple": 2.5,
        "de_turnover_penalty_weight": 1.0,
    }

    def test_free_below_soft_limit(self):
        from tuning.objective import de_turnover_penalty
        assert de_turnover_penalty(1.0, 1.0, self.BP) == 0.0
        assert de_turnover_penalty(1.5, 1.0, self.BP) == 0.0   # exactly soft
        assert de_turnover_penalty(0.5, 1.0, self.BP) == 0.0   # low-turnover improvement untouched

    def test_linear_ramp_between_soft_and_hard(self):
        from tuning.objective import de_turnover_penalty
        mid = de_turnover_penalty(2.0, 1.0, self.BP)   # multiple 2.0, halfway
        assert mid == pytest.approx(0.5)
        at_hard = de_turnover_penalty(2.5, 1.0, self.BP)
        assert at_hard == pytest.approx(1.0)

    def test_steep_beyond_hard(self):
        from tuning.objective import de_turnover_penalty
        # The 5.5x churn the first tournament produced must now hurt badly.
        p = de_turnover_penalty(5.5, 1.0, self.BP)
        assert p == pytest.approx(1.0 * (1.0 + 3.0))
        assert p > de_turnover_penalty(2.5, 1.0, self.BP) * 3

    def test_disabled_or_no_incumbent_is_zero(self):
        from tuning.objective import de_turnover_penalty
        off = dict(self.BP, de_turnover_penalty_enabled=False)
        assert de_turnover_penalty(9.0, 1.0, off) == 0.0
        assert de_turnover_penalty(9.0, None, self.BP) == 0.0
        assert de_turnover_penalty(9.0, 0.0, self.BP) == 0.0

    def test_weight_scales(self):
        from tuning.objective import de_turnover_penalty
        heavy = dict(self.BP, de_turnover_penalty_weight=2.0)
        assert de_turnover_penalty(2.0, 1.0, heavy) == pytest.approx(1.0)

    def test_live_config_documents_keys(self):
        for key in ("de_turnover_penalty_enabled", "de_turnover_penalty_vs_incumbent",
                    "de_turnover_soft_limit_multiple", "de_turnover_hard_limit_multiple",
                    "de_turnover_penalty_weight"):
            assert key in BACKTEST_PARAMS
        assert BACKTEST_PARAMS["de_turnover_soft_limit_multiple"] < BACKTEST_PARAMS["de_turnover_hard_limit_multiple"]


# ---------------------------------------------------------------------------
# Multi-horizon confirmation
# ---------------------------------------------------------------------------

class TestMultiHorizonConfirm:

    def _wire_mh(self, monkeypatch, sims_by_window):
        """sims_by_window[(window, which)] -> SimResult, which in {sel, inc}."""
        import backtesting.regime_scope as rs
        import tuning.tuner as tt

        full = SimpleNamespace(prices=np.zeros((730, 2)))
        monkeypatch.setattr(tt, "load_and_precompute", lambda *a, **k: full)
        monkeypatch.setattr(rs, "slice_precomp",
                            lambda pc, sl: SimpleNamespace(prices=np.zeros((sl.stop - sl.start, 2))))

        sel_vec = np.full(4, 1.0)

        def fake_sim(pc, params, *a, **k):
            w = pc.prices.shape[0]
            which = "sel" if np.allclose(params, sel_vec) else "inc"
            return sims_by_window[(w, which)]
        monkeypatch.setattr(tt, "run_simulation", fake_sim)
        monkeypatch.setattr(tt, "gate_simulation", fake_sim)
        return tt, sel_vec

    @staticmethod
    def _mh_sim(excess, dd=-0.10, turnover=1.0):
        # total_return - benchmark_twr == excess (benchmark_twr=0 for simplicity)
        return SimResult(final_value=1.0, total_return=excess, sharpe=1.0, calmar=1.0,
                         max_drawdown=dd, trades_made=10, turnover_estimate=turnover,
                         benchmark_twr=0.0)

    def _uniform(self, sel_excess, inc_excess, **sel_kw):
        out = {}
        for w in (90, 180, 365, 730):
            out[(w, "sel")] = self._mh_sim(sel_excess, **sel_kw)
            out[(w, "inc")] = self._mh_sim(inc_excess)
        return out

    def test_passes_when_preserving_everywhere(self, monkeypatch):
        from util import BACKTEST_PARAMS as bp
        tt, sel = self._wire_mh(monkeypatch, self._uniform(0.05, 0.05))
        passed, reasons, rows = tt.multi_horizon_confirm(sel, np.full(4, 2.0), bp)
        assert passed, reasons
        assert [r["window"] for r in rows] == [90, 180, 365, 730]

    def test_regression_beyond_catastrophe_tolerance_blocks(self, monkeypatch):
        from util import BACKTEST_PARAMS as bp
        tol = float(bp["multi_horizon_confirm"].get("regress_tolerance", 0.04))
        sims = self._uniform(0.05, 0.05)
        sims[(90, "sel")] = self._mh_sim(0.05 - tol - 0.01)  # just past tolerance
        tt, sel = self._wire_mh(monkeypatch, sims)
        passed, reasons, _ = tt.multi_horizon_confirm(sel, np.full(4, 2.0), bp)
        assert not passed
        assert any("90d" in r for r in reasons)

    def test_single_window_noise_regression_passes(self, monkeypatch):
        """The regime-sample rule: a sub-tolerance regression on ONE window (the
        old 180d recency-veto scenario) must NOT block a candidate that wins
        everywhere else — no single window gets a fine-grained veto."""
        from util import BACKTEST_PARAMS as bp
        tol = float(bp["multi_horizon_confirm"].get("regress_tolerance", 0.04))
        sims = self._uniform(0.08, 0.05)                       # wins 90/365/730
        sims[(180, "sel")] = self._mh_sim(0.05 - tol + 0.01)   # inside tolerance
        tt, sel = self._wire_mh(monkeypatch, sims)
        passed, reasons, _ = tt.multi_horizon_confirm(sel, np.full(4, 2.0), bp)
        assert passed, reasons

    def test_long_window_catastrophe_blocks(self, monkeypatch):
        from util import BACKTEST_PARAMS as bp
        limit = float(bp["multi_horizon_confirm"].get("long_catastrophe_excess", 0.10))
        sims = self._uniform(0.05, 0.05)
        sims[(730, "sel")] = self._mh_sim(0.05 - limit - 0.02)
        tt, sel = self._wire_mh(monkeypatch, sims)
        passed, reasons, _ = tt.multi_horizon_confirm(sel, np.full(4, 2.0), bp)
        assert not passed
        assert any("catastrophic" in r for r in reasons)

    def test_turnover_blowup_on_any_window_blocks(self, monkeypatch):
        from util import BACKTEST_PARAMS as bp
        mult = float(bp.get("max_turnover_multiple", 2.0))
        sims = self._uniform(0.06, 0.05)
        sims[(180, "sel")] = self._mh_sim(0.06, turnover=1.0 * (mult + 0.5))
        tt, sel = self._wire_mh(monkeypatch, sims)
        passed, reasons, _ = tt.multi_horizon_confirm(sel, np.full(4, 2.0), bp)
        assert not passed
        assert any("turnover" in r for r in reasons)

    def test_confirm_failure_blocks_write_after_earlier_gates_pass(self, monkeypatch, capsys):
        """A candidate that passes the split AND random-window gates must still
        be blocked (no config write) when multi-horizon confirmation fails."""
        import tuning.constants as tc
        inc0 = float(tc._current_params()[0])

        def metrics_for(params):
            sig = float(params[0])
            if abs(sig - inc0) < 1e-9:
                return _passing_sim(total_return=0.10)
            if abs(sig - 0.2) < 1e-9:
                return _passing_sim(total_return=0.50)
            return _passing_sim(total_return=0.12)

        tt, cap = _wire_auto_tune(monkeypatch, metrics_for,
                                  np.zeros(len(tc.PARAM_NAMES)) + 0.2,
                                  np.zeros(len(tc.PARAM_NAMES)) + 0.8)
        # Override the wired stub: rw passes, multi-horizon FAILS.
        monkeypatch.setattr(tt, "multi_horizon_confirm",
                            lambda *a, **k: (False, ["90d: selected excess regresses"], []))
        out = tt.run_auto_tune(n_days=90, apply=True)
        assert cap["applied"] == []          # gates passed earlier tiers, write still blocked
        assert len(out) == 6
        text = capsys.readouterr().out
        assert "90d: selected excess regresses" in text
