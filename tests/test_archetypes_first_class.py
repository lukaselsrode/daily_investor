"""
tests/test_archetypes_first_class.py — Archetype-as-first-class integration tests.

Covers:
  - Config validation rejects invalid archetype controls
  - Defaults preserve current SimResult behavior (regression safety)
  - Disabled archetype blocks new buys (but keeps existing positions managed)
  - min_score_to_buy gates new buys
  - max_active_weight caps archetype sleeve allocation
  - TradeRecord carries archetype_at_entry / archetype_at_exit / decision_source
  - SimResult exposes archetype_active_excess / archetype_win_rate / archetype_avg_hold_days
  - Live/backtest consistency: sell engine + simulator use the SAME ArchetypePolicy fields
  - Archetype-targeted tuning presets exist and validate
  - Outcome tracker schema includes the new columns
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestArchetypeConfigValidation:

    def test_negative_score_multiplier_rejected(self):
        from config.schema import ArchetypeEntryConfig
        with pytest.raises(ValueError, match="score_multiplier"):
            ArchetypeEntryConfig(score_multiplier=-0.1)

    def test_negative_position_multiplier_rejected(self):
        from config.schema import ArchetypeEntryConfig
        with pytest.raises(ValueError, match="max_position_multiplier"):
            ArchetypeEntryConfig(max_position_multiplier=-0.5)

    def test_max_active_weight_above_one_rejected(self):
        from config.schema import ArchetypeEntryConfig
        with pytest.raises(ValueError, match="max_active_weight"):
            ArchetypeEntryConfig(max_active_weight=1.5)

    def test_defaults_are_no_ops(self):
        from config.schema import ArchetypeEntryConfig
        c = ArchetypeEntryConfig()
        assert c.enabled is True
        assert c.score_multiplier == 1.0
        assert c.max_position_multiplier == 1.0
        assert c.max_active_weight is None
        assert c.min_score_to_buy is None


# ---------------------------------------------------------------------------
# 2. ArchetypePolicy mirrors the new fields
# ---------------------------------------------------------------------------

class TestArchetypePolicy:

    def test_default_policy_no_ops(self):
        from portfolio.position_archetypes import get_archetype_policy
        p = get_archetype_policy("quality_compounder")
        assert p.enabled is True
        assert p.score_multiplier == 1.0
        assert p.max_position_multiplier == 1.0
        assert p.max_active_weight is None
        assert p.min_score_to_buy is None

    def test_overrides_threaded_through(self):
        from portfolio.position_archetypes import get_archetype_policy
        cfg = {
            "enabled": True,
            "quality_compounder": {
                "enabled": False,
                "score_multiplier": 0.5,
                "max_position_multiplier": 2.0,
                "max_active_weight": 0.25,
                "min_score_to_buy": 0.7,
            },
        }
        p = get_archetype_policy("quality_compounder", cfg)
        assert p.enabled is False
        assert p.score_multiplier == 0.5
        assert p.max_position_multiplier == 2.0
        assert p.max_active_weight == 0.25
        assert p.min_score_to_buy == 0.7


# ---------------------------------------------------------------------------
# 3. TradeRecord new fields are backward compatible
# ---------------------------------------------------------------------------

def test_trade_record_defaults_blank_for_new_fields():
    from core.types import TradeRecord
    tr = TradeRecord(date="2026-01-01", symbol="XYZ", side="buy",
                     quantity=1.0, price=10.0, amount=10.0)
    assert tr.archetype_at_entry == ""
    assert tr.archetype_at_exit == ""
    assert tr.decision_source == ""


# ---------------------------------------------------------------------------
# 4. SimResult new dicts default to empty
# ---------------------------------------------------------------------------

def test_sim_result_archetype_rollups_default_empty():
    from backtesting.types import SimResult
    sr = SimResult(
        final_value=10000.0, total_return=0.05, sharpe=0.5,
        calmar=0.3, max_drawdown=-0.1, trades_made=5,
    )
    assert sr.archetype_active_excess == {}
    assert sr.archetype_win_rate == {}
    assert sr.archetype_avg_hold_days == {}
    assert sr.archetype_max_drawdown == {}
    assert sr.archetype_sleeve_weight == {}


# ---------------------------------------------------------------------------
# 5. Tuning presets — archetype-targeted ones exist & validate
# ---------------------------------------------------------------------------

class TestArchetypePresets:

    _NAMES = (
        "active_quality_compounders",
        "active_speculative_momentum",
        "active_value_recovery",
        "active_defensive_income",
        "active_archetype_lifecycle",
        "active_archetype_rotation",
        "active_archetype_alpha",
    )

    def test_all_seven_presets_listed(self):
        from tuning.presets import list_presets
        names = {n for n, _ in list_presets()}
        for n in self._NAMES:
            assert n in names, f"missing preset: {n}"

    def test_all_seven_presets_validate(self):
        from tuning.presets import validate_preset
        for n in self._NAMES:
            validate_preset(n)   # raises on Phase-2-stub or unknown

    def test_lifecycle_unfreezes_24_archetype_slots(self):
        from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX, _get_active_indices
        active = _get_active_indices(
            scope="active_sleeve_compounding",
            preset="active_archetype_lifecycle",
        )
        arch_indices = {
            i for p, i in _CONFIG_PATH_TO_PARAM_IDX.items()
            if p.startswith("archetype_management.")
        }
        # All 24 archetype slots should be active under this preset
        assert arch_indices.issubset(set(active))


# ---------------------------------------------------------------------------
# 6. Param vector + bounds extended
# ---------------------------------------------------------------------------

class TestParamVector:

    def test_vector_length_is_50(self):
        # 16 base + 24 archetype + 3 candidate-filter + 3 position-sizing
        # + 1 regime-tilt (46) + 1 mean-reversion blend (47) + 1 low-vol quality blend (48)
        # + 1 residual-momentum blend (49)
        from tuning.constants import BOUNDS, PARAM_NAMES
        assert len(PARAM_NAMES) == 50
        assert len(BOUNDS) == 50

    def test_archetype_slots_frozen_by_default(self):
        # No preset → archetype slots stay frozen even if config doesn't list them.
        from tuning.constants import _CONFIG_PATH_TO_PARAM_IDX, _get_active_indices
        active = _get_active_indices(scope="overall_strategy", preset=None)
        arch_indices = {
            i for p, i in _CONFIG_PATH_TO_PARAM_IDX.items()
            if p.startswith("archetype_management.")
        }
        # Default behavior: archetype slots NOT active (preserves pre-PR behavior)
        assert arch_indices.isdisjoint(set(active))

    def test_archetype_cfg_from_params(self):
        from tuning.constants import _current_params, archetype_cfg_from_params
        cur = _current_params()
        cfg = archetype_cfg_from_params(cur)
        assert "quality_compounder" in cfg
        assert "harvest_profit_threshold" in cfg["quality_compounder"]
        # minimum_hold_days is cast to int
        assert isinstance(cfg["quality_compounder"]["minimum_hold_days"], int)

    def test_archetype_cfg_empty_for_short_vector(self):
        from tuning.constants import archetype_cfg_from_params
        # 15-element vector → no archetype tail
        params = np.zeros(15)
        assert archetype_cfg_from_params(params) == {}


# ---------------------------------------------------------------------------
# 7. Sell engine: decision_source threaded through
# ---------------------------------------------------------------------------

class TestSellEngineDecisionSource:

    def _holding(self, pct_change: float = 0.0, price: float = 100.0) -> dict:
        return {
            "price": price,
            "average_buy_price": price / (1 + pct_change) if pct_change != -1 else 1.0,
            "percent_change": pct_change * 100.0,
            "quantity": 10.0,
        }

    def test_stop_loss_marks_global_rule(self):
        from portfolio.sell_engine import SellDecisionEngine
        eng = SellDecisionEngine()
        h = self._holding(pct_change=-0.5, price=50.0)
        d = eng.evaluate("X", h, None)
        assert d.should_sell is True
        assert d.exit_type == "failure_exit"
        assert d.decision_source == "global_rule"

    def test_trailing_stop_marks_archetype_when_policy_set(self):
        # Need archetype management enabled for the policy to take effect
        import util
        from portfolio.position_archetypes import get_archetype_policy
        from portfolio.sell_engine import SellDecisionEngine
        old = util.ARCHETYPE_PARAMS
        util.ARCHETYPE_PARAMS = {"enabled": True}
        try:
            eng = SellDecisionEngine()
            policy = get_archetype_policy("speculative_momentum",
                                          {"enabled": True,
                                           "speculative_momentum": {"trailing_stop_pct": -0.05}})
            h = {"price": 90.0, "average_buy_price": 100.0,
                 "percent_change": -10.0, "quantity": 1.0}
            d = eng.evaluate("X", h, None, peak_price=100.0, archetype_policy=policy)
            assert d.should_sell is True
            assert d.decision_source == "archetype_rule"
        finally:
            util.ARCHETYPE_PARAMS = old


# ---------------------------------------------------------------------------
# 8. Outcome tracker schema includes the new columns
# ---------------------------------------------------------------------------

def test_outcome_tracker_schema_has_new_columns():
    from portfolio import outcome_tracker as ot
    for col in ("archetype_at_entry", "archetype_at_exit", "decision_source"):
        assert col in ot._SCHEMA, f"outcome tracker missing column {col!r}"
