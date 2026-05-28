"""
tests/conftest.py — Shared pytest fixtures.

All tests use ConfigManager.from_dict() so they never touch the filesystem.
The singleton is reset between tests via the config_manager fixture.
"""

import os
import sys

# Ensure src/ is on path so both old and new modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from config.manager import ConfigManager


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Ensure ConfigManager singleton doesn't leak between tests."""
    ConfigManager._reset()
    yield
    ConfigManager._reset()


@pytest.fixture
def base_config() -> dict:
    """Minimal valid config dict matching current production defaults."""
    return {
        "ignore_negative_pe": True,
        "ignore_negative_pb": False,
        "dividend_threshold": 0.03,
        "metric_threshold": 0.75,
        "selloff_threshold": 30,
        "weekly_investment": 400,
        "index_pct": 0.70,
        "auto_approve": True,
        "use_sentiment_analysis": True,
        "confidence_threshold": 65,
        "sell_sentiment_override_confidence": 85,
        "max_iterations": 10,
        "etfs": ["SPY", "VOO", "VTI", "QQQ", "SCHD"],
        "score_weights": {"value": 0.08, "quality": 0.50, "income": 0.08, "momentum": 0.34},
        "valuation_guardrails": {"max_pe_component": 5.0, "max_pb_component": 5.0, "min_pe_ratio": 1.0, "min_pb_ratio": 0.1},
        "risk": {
            "max_single_position_pct": 0.05,
            "max_sector_pct": 0.25,
            "max_order_pct_of_cash": 0.10,
            "min_order_amount": 5.0,
            "min_liquidity_volume": 500000,
            "max_buys_per_rebalance": 7,
            "max_sentiment_candidates": 20,
            "minimum_hold_days": 5,
            "allow_whole_share_fallback": False,
            "max_whole_share_buys_per_run": 2,
            "max_whole_share_allocation_multiplier": 1.25,
            "min_index_pct": 0.60,
        },
        "sell_rules": {
            "stop_loss_pct": -0.20,
            "trailing_stop_pct": -0.08,
            "take_profit_pct": 0.60,
            "take_profit_value_floor_multiplier": 1.20,
            "sell_weak_value_below": 0.45,
            "sell_yield_trap": True,
            "sell_low_quality_below": -0.25,
            "min_days_held_before_value_exit": 21,
            "minimum_days_before_take_profit": 10,
        },
        "harvest": {
            "enabled": True,
            "profit_harvest_pct": 0.40,
            "harvest_to_etfs_pct": 0.80,
            "recycle_to_stocks_pct": 0.20,
            "harvest_only_if_value_metric_below_multiplier": 1.20,
            "min_harvest_amount": 25.0,
            "max_harvest_pct_of_portfolio": 0.02,
            "harvest_etfs": ["SPY", "VTI"],
        },
        "regime": {
            "spy_ma_period": 200,
            "vix_defensive_threshold": 30.0,
            "vix_neutral_threshold": 20.0,
            "defensive": {"index_pct_override": 0.85, "max_buys_override": 3, "stop_loss_tighten": 0.05},
            "neutral": {"index_pct_override": None, "max_buys_override": None},
        },
        "backtest": {
            "turnover_penalty_enabled": True,
            "turnover_penalty_trade_count": 50,
            "turnover_penalty_weight": 0.35,
            "starting_capital": 5000.0,
            "weekly_contribution": 400.0,
            "slippage_bps": 10.0,
            "train_pct": 0.70,
        },
        "tuning": {
            "frozen_parameters": [
                "score_weights.value",
                "score_weights.income",
                "metric_threshold",
            ],
            "parameter_bounds": {
                "score_weights.quality": {"min": 0.35, "max": 0.60},
                "score_weights.momentum": {"min": 0.20, "max": 0.40},
                "index_pct": {"min": 0.60, "max": 0.85},
            },
        },
        "reliability": {"enabled": False, "min_reliability_score": 0.70},
    }


@pytest.fixture
def cfg(base_config) -> ConfigManager:
    return ConfigManager.from_dict(base_config)
