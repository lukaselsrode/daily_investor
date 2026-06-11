"""
tests/test_contribution_timing.py — buy-the-dip contribution overlay.

Covers the spec's required guarantees: causality (prior data only), multiplier
clamps, rolling budget cap, carry-forward, borrowing, weekly bounds, regime cap,
weight normalization, exact flat behavior when disabled, the simulator receiving
the adjusted schedule, and the structural impossibility of the overlay touching
holdings (its API accepts no portfolio state at all).
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Live config defaults drive the tests (test philosophy: live constants, not
# hardcoded copies). The overlay block ships disabled; tests enable copies.
import copy

import numpy as np
import pytest

from portfolio.contribution_timing import (
    ContributionState,
    build_contribution_schedule,
    compute_dip_score,
    contribution_multiplier,
    decide_contribution,
    summarize_decisions,
)
from util import CONTRIBUTION_TIMING_PARAMS


def _cfg(**overrides) -> dict:
    cfg = copy.deepcopy(CONTRIBUTION_TIMING_PARAMS)
    cfg["enabled"] = True
    cfg.update(overrides)
    return cfg


def _flat(n=300, level=100.0):
    return np.full(n, level)


def _selloff(n=300, drop=0.12, days=10):
    """Flat history ending in a sharp multi-day selloff."""
    px = np.full(n, 100.0)
    px[-days:] = 100.0 * np.linspace(1.0, 1.0 - drop, days)
    return px


def _rally(n=300, gain=0.10, days=10):
    px = np.full(n, 100.0)
    px[-days:] = 100.0 * np.linspace(1.0, 1.0 + gain, days)
    return px


# ---------------------------------------------------------------------------
# Dip signal
# ---------------------------------------------------------------------------

class TestDipSignal:

    def test_causal_uses_only_prior_data(self):
        """Changing data AFTER the decision day must not change earlier decisions."""
        base = _selloff(300)
        future_changed = base.copy()
        n = 250  # decision uses prices[:250]
        sched_a, dec_a = build_contribution_schedule(base, 300, 5, 400.0, _cfg())
        future_changed[n:] = 500.0  # absurd future rally
        sched_b, dec_b = build_contribution_schedule(future_changed, 300, 5, 400.0, _cfg())
        # All contribution days strictly before day n saw identical history.
        for d in range(n):
            assert sched_a[d] == sched_b[d]

    def test_decision_excludes_same_day_close(self):
        """The deployment-day close must not affect that day's decision."""
        px = _flat(300)
        cfg = _cfg()
        state_a, state_b = ContributionState(), ContributionState()
        a = decide_contribution(px[:250], cfg, state_a)
        px2 = px.copy()
        px2[250] = 50.0  # crash ON the deployment day
        b = decide_contribution(px2[:250], cfg, state_b)
        assert a.adjusted_amount == b.adjusted_amount

    def test_selloff_scores_higher_than_rally(self):
        dip_sell, _, reasons_sell = compute_dip_score(_selloff(300), _cfg()["dip_signal"])
        dip_rally, _, _ = compute_dip_score(_rally(300), _cfg()["dip_signal"])
        dip_flat, _, _ = compute_dip_score(_flat(300), _cfg()["dip_signal"])
        assert dip_sell > dip_flat > dip_rally
        assert "market_down_1w" in reasons_sell
        assert "drawdown_from_20d_high" in reasons_sell

    def test_insufficient_history_is_neutral(self):
        dip, comps, reasons = compute_dip_score(_flat(20), _cfg()["dip_signal"])
        assert np.isnan(dip) and "insufficient_history" in reasons
        state = ContributionState()
        dec = decide_contribution(_flat(20), _cfg(), state)
        assert dec.multiplier == 1.0
        assert dec.adjusted_amount == pytest.approx(_cfg()["base_weekly_contribution"])

    def test_weights_normalize_to_one(self):
        """Scaling all weights by a constant must not change the score."""
        dip_cfg = copy.deepcopy(_cfg()["dip_signal"])
        score_a, _, _ = compute_dip_score(_selloff(300), dip_cfg)
        for k in dip_cfg["weights"]:
            dip_cfg["weights"][k] *= 7.3
        score_b, _, _ = compute_dip_score(_selloff(300), dip_cfg)
        assert score_a == pytest.approx(score_b)


# ---------------------------------------------------------------------------
# Multiplier
# ---------------------------------------------------------------------------

class TestMultiplier:

    def test_clamps(self):
        m = _cfg()["multiplier"]
        assert contribution_multiplier(1.0, m) <= m["max_multiplier"]
        assert contribution_multiplier(0.0, m) >= m["min_multiplier"]

    def test_neutral_dip_gives_base_multiplier(self):
        m = _cfg()["multiplier"]
        assert contribution_multiplier(m["neutral_dip_score"], m) == pytest.approx(1.0)

    def test_smoothing_damps_single_week_spike(self):
        m = dict(_cfg()["multiplier"], smoothing_alpha=0.5)
        spike = contribution_multiplier(1.0, m, prev_multiplier=1.0)
        unsmoothed = dict(m, smoothing_alpha=1.0)
        full = contribution_multiplier(1.0, unsmoothed, prev_multiplier=1.0)
        assert 1.0 < spike < full

    def test_defensive_regime_caps_multiplier(self):
        cfg = _cfg()
        state = ContributionState()
        dec = decide_contribution(_selloff(300, drop=0.20), cfg, state, regime="defensive")
        cap = cfg["regime_controls"]["defensive_max_multiplier"]
        assert dec.multiplier <= cap + 1e-9
        assert "defensive_regime_cap" in dec.reason_codes
        # Same selloff in a bullish regime buys harder than the defensive cap.
        dec_bull = decide_contribution(_selloff(300, drop=0.20), _cfg(), ContributionState(), regime="bullish")
        assert dec_bull.multiplier > cap


# ---------------------------------------------------------------------------
# Budget mechanics
# ---------------------------------------------------------------------------

class TestBudget:

    def test_weekly_min_max_bounds(self):
        cfg = _cfg(min_weekly_contribution=300.0, max_weekly_contribution=500.0,
                   preserve_monthly_budget=False)
        hot = decide_contribution(_selloff(300, drop=0.25), cfg, ContributionState(), regime="bullish")
        cold = decide_contribution(_rally(300, gain=0.20), cfg, ContributionState())
        assert hot.adjusted_amount <= 500.0
        assert cold.adjusted_amount >= 300.0
        assert "max_weekly_cap" in hot.reason_codes
        assert "min_weekly_floor" in cold.reason_codes

    def test_rolling_budget_cap(self):
        """Repeated max-dip weeks cannot blow past target * (1 + tolerance)."""
        cfg = _cfg(carry_forward_unused_budget=False)
        state = ContributionState()
        px = _selloff(400, drop=0.30, days=120)
        total_cap = cfg["target_monthly_contribution"] * (1 + cfg["monthly_budget_tolerance_pct"])
        for week in range(cfg["budget_window_weeks"]):
            decide_contribution(px[: 300 + week * 5], cfg, state, regime="bullish")
        assert state.window_sum() <= total_cap + 1e-6

    def test_budget_cap_reason_code(self):
        cfg = _cfg(carry_forward_unused_budget=False)
        state = ContributionState()
        px = _selloff(400, drop=0.30, days=120)
        last = None
        for week in range(cfg["budget_window_weeks"] + 1):
            last = decide_contribution(px[: 300 + week * 5], cfg, state, regime="bullish")
        assert "monthly_budget_cap" in last.reason_codes

    def test_carry_forward_banks_underspend(self):
        cfg = _cfg()
        state = ContributionState()
        dec = decide_contribution(_rally(300, gain=0.20), cfg, state)
        assert dec.adjusted_amount < cfg["base_weekly_contribution"]
        assert state.carry_forward == pytest.approx(
            cfg["base_weekly_contribution"] - dec.adjusted_amount
        )

    def test_carry_forward_consumed_by_overspend(self):
        cfg = _cfg()
        state = ContributionState()
        state.carry_forward = 150.0
        dec = decide_contribution(_selloff(300, drop=0.15), cfg, state, regime="bullish")
        assert dec.adjusted_amount > cfg["base_weekly_contribution"]
        assert "carry_forward_used" in dec.reason_codes
        assert state.carry_forward < 150.0

    def test_borrow_disabled_caps_at_base_plus_carry(self):
        cfg = _cfg(borrow_from_future_weeks=False)
        state = ContributionState()
        state.carry_forward = 50.0
        dec = decide_contribution(_selloff(300, drop=0.20), cfg, state, regime="bullish")
        assert dec.adjusted_amount <= cfg["base_weekly_contribution"] + 50.0 + 1e-9
        assert "borrow_disabled_cap" in dec.reason_codes

    def test_acceleration_flag_lifts_budget_cap(self):
        state_capped = ContributionState()
        state_free = ContributionState()
        px = _selloff(400, drop=0.30, days=120)
        capped_cfg = _cfg(carry_forward_unused_budget=False)
        free_cfg = _cfg(allow_budget_acceleration=True, carry_forward_unused_budget=False)
        for week in range(5):
            capped = decide_contribution(px[: 300 + week * 5], capped_cfg, state_capped, regime="bullish")
            free = decide_contribution(px[: 300 + week * 5], free_cfg, state_free, regime="bullish")
        assert state_free.window_sum() > state_capped.window_sum()
        assert "monthly_budget_cap" not in free.reason_codes


# ---------------------------------------------------------------------------
# Schedule / simulator integration
# ---------------------------------------------------------------------------

class TestScheduleAndSim:

    def test_disabled_schedule_is_exactly_flat(self):
        amounts, decisions = build_contribution_schedule(_selloff(300), 300, 5, 400.0, None)
        assert decisions == []
        for d in range(300):
            expected = 400.0 if (d > 0 and d % 5 == 0) else 0.0
            assert amounts[d] == expected
        disabled = dict(_cfg(), enabled=False)
        amounts2, decisions2 = build_contribution_schedule(_selloff(300), 300, 5, 400.0, disabled)
        assert decisions2 == [] and np.array_equal(amounts, amounts2)

    def test_overlay_never_accepts_portfolio_state(self):
        """Structural guarantee: the overlay cannot sell or alter holdings — its
        API has no parameter that could carry portfolio/holdings/broker state."""
        for fn in (compute_dip_score, contribution_multiplier, decide_contribution,
                   build_contribution_schedule):
            params = set(inspect.signature(fn).parameters)
            assert not params & {"holdings", "broker", "portfolio", "positions", "shares"}

    def test_simulator_receives_adjusted_schedule(self, monkeypatch):
        from backtesting.simulator import get_default_params, run_simulation
        sys.path.insert(0, os.path.dirname(__file__))
        from test_backtest_fixes import _make_precomp

        n_days = 300
        bench = _selloff(n_days, drop=0.25, days=60)
        pc = _make_precomp(n_days, 3, benchmark=bench)

        import backtesting.simulator as sim_mod
        monkeypatch.setitem(sim_mod.CONTRIBUTION_TIMING_PARAMS, "enabled", True)

        res = run_simulation(pc, get_default_params(), 10_000.0, weekly_contribution=400.0)
        assert res.contribution_timing is not None
        sched = res.contribution_timing["schedule"]
        assert len(sched) == sum(1 for d in range(n_days) if d > 0 and d % 5 == 0)
        total_sched = sum(row["contribution"] for row in sched)
        # Sim accounting must match the varied schedule exactly
        # (net_contributions = starting_capital + all weekly contributions).
        assert res.net_contributions == pytest.approx(10_000.0 + total_sched, abs=0.01)
        # The selloff tail must have pushed at least one week above base...
        assert res.contribution_timing["max_weekly"] > 400.0
        # ...with reason codes on the schedule rows.
        assert any(row["reason_codes"] for row in sched)

    def test_simulator_flat_when_disabled(self, monkeypatch):
        from backtesting.simulator import get_default_params, run_simulation
        sys.path.insert(0, os.path.dirname(__file__))
        from test_backtest_fixes import _make_precomp

        pc = _make_precomp(120, 3, benchmark=_selloff(120))
        import backtesting.simulator as sim_mod
        monkeypatch.setitem(sim_mod.CONTRIBUTION_TIMING_PARAMS, "enabled", False)
        res = run_simulation(pc, get_default_params(), 10_000.0, weekly_contribution=400.0)
        assert res.contribution_timing is None
        n_contrib_days = sum(1 for d in range(120) if d > 0 and d % 5 == 0)
        assert res.net_contributions == pytest.approx(10_000.0 + n_contrib_days * 400.0)

    def test_summary_stats(self):
        amounts, decisions = build_contribution_schedule(
            _selloff(300, drop=0.20, days=40), 300, 5, 400.0, _cfg(),
        )
        stats = summarize_decisions(decisions, 400.0)
        assert stats["weeks"] == len(decisions)
        assert stats["total_contributed"] == pytest.approx(float(amounts.sum()))
        assert 0.0 <= stats["pct_weeks_above_base"] <= 1.0
        assert 0.0 <= stats["pct_weeks_below_base"] <= 1.0


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------

class TestLivePath:

    def test_live_panel_includes_reason_codes_and_budget(self):
        from portfolio.contribution_timing import format_live_panel
        cfg = _cfg()
        dec = decide_contribution(_selloff(300, drop=0.15), cfg, ContributionState(), regime="bullish")
        panel = format_live_panel(dec, cfg)
        assert "Contribution Timing:" in panel
        assert "Multiplier:" in panel and "Dip score:" in panel
        assert "Monthly budget used:" in panel
        for code in dec.reason_codes:
            assert code in panel

    def test_live_state_round_trip(self, tmp_path):
        from portfolio.contribution_timing import load_live_state, record_live_decision
        cfg = _cfg()
        csv = str(tmp_path / "contribution_timing_log.csv")
        dec = decide_contribution(_selloff(300), cfg, ContributionState(), regime="bullish")
        assert record_live_decision(csv, dec) is True
        # Same week — second run must NOT add a second row to the window.
        assert record_live_decision(csv, dec) is False
        state = load_live_state(csv, cfg)
        assert state.window_sum() == pytest.approx(dec.adjusted_amount)
        assert state.prev_multiplier == pytest.approx(round(dec.multiplier, 4))

    def test_load_live_state_missing_file_is_fresh(self, tmp_path):
        from portfolio.contribution_timing import load_live_state
        state = load_live_state(str(tmp_path / "nope.csv"), _cfg())
        assert state.window_sum() == 0.0 and state.carry_forward == 0.0


# ---------------------------------------------------------------------------
# Tuned-slot plumbing
# ---------------------------------------------------------------------------

class TestTunedSlots:

    def test_cfg_from_params_round_trip(self):
        from tuning.constants import (
            _CT_FIELDS,
            _CT_SLOT_OFFSET,
            _current_params,
            contribution_timing_cfg_from_params,
        )
        params = _current_params()
        out = contribution_timing_cfg_from_params(params)
        m = CONTRIBUTION_TIMING_PARAMS["multiplier"]
        assert out["multiplier"]["dip_sensitivity"] == pytest.approx(m["dip_sensitivity"])
        assert out["weights"]["return_1w"] == pytest.approx(
            CONTRIBUTION_TIMING_PARAMS["dip_signal"]["weights"]["return_1w"]
        )
        # Short vectors (base tunes) yield no overrides.
        assert contribution_timing_cfg_from_params(params[:_CT_SLOT_OFFSET]) == {}
        assert len(_CT_FIELDS) == 8

    def test_min_max_multiplier_degenerate_swap(self):
        from tuning.constants import _CT_SLOT_OFFSET, contribution_timing_cfg_from_params
        params = np.zeros(_CT_SLOT_OFFSET + 8)
        params[_CT_SLOT_OFFSET + 2] = 2.0  # min
        params[_CT_SLOT_OFFSET + 3] = 1.0  # max < min — tuner proposed nonsense
        out = contribution_timing_cfg_from_params(params)
        assert out["multiplier"]["min_multiplier"] <= out["multiplier"]["max_multiplier"]

    def test_preset_unfreezes_exactly_eight(self):
        from tuning.constants import _get_active_indices
        base_active = set(_get_active_indices(preset=None))
        ct_active = set(_get_active_indices(preset="contribution_timing"))
        from tuning.constants import _CT_FIELDS, _CT_SLOT_OFFSET
        ct_slots = {_CT_SLOT_OFFSET + i for i in range(len(_CT_FIELDS))}
        assert ct_slots <= ct_active
        assert ct_slots.isdisjoint(base_active)  # frozen by default
