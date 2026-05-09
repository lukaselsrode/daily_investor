"""
tests/test_config.py — ConfigManager unit tests.

These tests use ConfigManager.from_dict() — no filesystem access.
"""

import pytest

from config.manager import ConfigManager
from config.schema import ScoreWeightsConfig


class TestConfigManagerScalars:

    def test_metric_threshold(self, cfg):
        assert cfg.metric_threshold == 0.75

    def test_index_pct(self, cfg):
        assert cfg.index_pct == 0.70

    def test_etfs_list(self, cfg):
        assert "SPY" in cfg.etfs
        assert len(cfg.etfs) >= 4

    def test_auto_approve(self, cfg):
        assert cfg.auto_approve is True

    def test_use_sentiment(self, cfg):
        assert cfg.use_sentiment_analysis is True

    def test_confidence_threshold(self, cfg):
        assert cfg.confidence_threshold == 65.0

    def test_max_iterations(self, cfg):
        assert cfg.max_iterations == 10


class TestScoreWeights:

    def test_weights_load(self, cfg):
        w = cfg.score_weights
        assert w.value == pytest.approx(0.08)
        assert w.quality == pytest.approx(0.50)
        assert w.income == pytest.approx(0.08)
        assert w.momentum == pytest.approx(0.34)

    def test_weights_sum_to_one(self, cfg):
        w = cfg.score_weights
        assert w.is_valid

    def test_weights_as_dict(self, cfg):
        d = cfg.score_weights.as_dict()
        assert set(d.keys()) == {"value", "quality", "income", "momentum"}


class TestRiskConfig:

    def test_max_single_position(self, cfg):
        assert cfg.risk.max_single_position_pct == pytest.approx(0.05)

    def test_max_buys_per_rebalance(self, cfg):
        assert cfg.risk.max_buys_per_rebalance == 7

    def test_minimum_hold_days(self, cfg):
        assert cfg.risk.minimum_hold_days == 5

    def test_min_liquidity_volume(self, cfg):
        assert cfg.risk.min_liquidity_volume == 500_000


class TestSellRules:

    def test_stop_loss(self, cfg):
        assert cfg.sell_rules.stop_loss_pct == pytest.approx(-0.20)

    def test_trailing_stop(self, cfg):
        assert cfg.sell_rules.trailing_stop_pct == pytest.approx(-0.08)

    def test_take_profit(self, cfg):
        assert cfg.sell_rules.take_profit_pct == pytest.approx(0.60)

    def test_sell_weak_value(self, cfg):
        assert cfg.sell_rules.sell_weak_value_below == pytest.approx(0.45)

    def test_minimum_days_before_take_profit(self, cfg):
        assert cfg.sell_rules.minimum_days_before_take_profit == 10

    def test_min_days_held_before_value_exit(self, cfg):
        assert cfg.sell_rules.min_days_held_before_value_exit == 21


class TestBacktestConfig:

    def test_turnover_penalty(self, cfg):
        assert cfg.backtest.turnover_penalty_enabled is True
        assert cfg.backtest.turnover_penalty_trade_count == 50
        assert cfg.backtest.turnover_penalty_weight == pytest.approx(0.35)

    def test_starting_capital(self, cfg):
        assert cfg.backtest.starting_capital == pytest.approx(5000.0)


class TestTuningConfig:

    def test_frozen_params(self, cfg):
        frozen = cfg.tuning.frozen_parameters
        assert "score_weights.value" in frozen
        assert "score_weights.income" in frozen
        assert "metric_threshold" in frozen

    def test_is_frozen(self, cfg):
        assert cfg.tuning.is_frozen("score_weights.value")
        assert not cfg.tuning.is_frozen("score_weights.quality")

    def test_bounds(self, cfg):
        lo, hi = cfg.tuning.bounds_for("score_weights.quality")
        assert lo == pytest.approx(0.35)
        assert hi == pytest.approx(0.60)

    def test_bounds_missing(self, cfg):
        assert cfg.tuning.bounds_for("nonexistent.param") is None


class TestRegimeConfig:

    def test_regime_thresholds(self, cfg):
        assert cfg.regime.vix_defensive_threshold == pytest.approx(30.0)
        assert cfg.regime.vix_neutral_threshold == pytest.approx(20.0)

    def test_defensive_overrides(self, cfg):
        assert cfg.regime.defensive.index_pct_override == pytest.approx(0.85)
        assert cfg.regime.defensive.max_buys_override == 3

    def test_neutral_overrides_are_none(self, cfg):
        assert cfg.regime.neutral.index_pct_override is None
        assert cfg.regime.neutral.max_buys_override is None


class TestEffectiveRegime:

    def test_bullish_uses_base(self, cfg):
        assert cfg.effective_index_pct("bullish") == pytest.approx(0.70)
        assert cfg.effective_max_buys("bullish") == 7

    def test_defensive_overrides(self, cfg):
        assert cfg.effective_index_pct("defensive") == pytest.approx(0.85)
        assert cfg.effective_max_buys("defensive") == 3

    def test_neutral_falls_back_to_base(self, cfg):
        assert cfg.effective_index_pct("neutral") == pytest.approx(0.70)
        assert cfg.effective_max_buys("neutral") == 7


class TestSingleton:

    def test_singleton_returns_same_instance(self):
        a = ConfigManager.get()
        b = ConfigManager.get()
        assert a is b

    def test_from_dict_independent(self, base_config):
        cfg1 = ConfigManager.from_dict(base_config)
        cfg2 = ConfigManager.from_dict({**base_config, "metric_threshold": 0.90})
        assert cfg1.metric_threshold != cfg2.metric_threshold

    def test_reload_creates_new_instance(self):
        a = ConfigManager.get()
        b = ConfigManager.get(reload=True)
        assert a is not b


class TestHarvestConfig:

    def test_harvest_enabled(self, cfg):
        assert cfg.harvest.enabled is True

    def test_harvest_etfs(self, cfg):
        assert "SPY" in cfg.harvest.harvest_etfs
        assert "VTI" in cfg.harvest.harvest_etfs

    def test_min_harvest_amount(self, cfg):
        assert cfg.harvest.min_harvest_amount == pytest.approx(25.0)


class TestScoreWeightsImmutability:

    def test_frozen_dataclass_raises_on_mutation(self, cfg):
        with pytest.raises((AttributeError, TypeError)):
            cfg.score_weights.value = 0.99  # type: ignore
