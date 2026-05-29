"""
config/schema.py — Typed, immutable config section dataclasses.

Each dataclass mirrors one YAML top-level section. Frozen=True enforces
immutability — config is loaded once and never mutated at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    max_single_position_pct: float = 0.05
    max_sector_pct: float = 0.25
    max_order_pct_of_cash: float = 0.10
    min_order_amount: float = 5.0
    min_liquidity_volume: float = 500_000
    max_buys_per_rebalance: int = 7
    max_sentiment_candidates: int = 20
    minimum_hold_days: int = 0
    allow_whole_share_fallback: bool = False
    max_whole_share_buys_per_run: int = 2
    max_whole_share_allocation_multiplier: float = 1.25
    min_index_pct: float = 0.60


@dataclass(frozen=True)
class SellRulesConfig:
    stop_loss_pct: float = -0.20
    trailing_stop_pct: float = -0.08
    take_profit_pct: float = 0.60
    take_profit_value_floor_multiplier: float = 1.20
    sell_weak_value_below: float = 0.45
    sell_yield_trap: bool = True
    sell_low_quality_below: float = -0.25
    min_days_held_before_value_exit: int = 21
    minimum_days_before_take_profit: int = 10


@dataclass(frozen=True)
class ScoreWeightsConfig:
    value: float = 0.08
    quality: float = 0.50
    income: float = 0.08
    momentum: float = 0.34

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "quality": self.quality,
                "income": self.income, "momentum": self.momentum}

    @property
    def is_valid(self) -> bool:
        return abs(self.value + self.quality + self.income + self.momentum - 1.0) <= 0.01


@dataclass(frozen=True)
class ValuationGuardrailsConfig:
    max_pe_component: float = 5.0
    max_pb_component: float = 5.0
    min_pe_ratio: float = 1.0
    min_pb_ratio: float = 0.1


@dataclass(frozen=True)
class ScoringConfig:
    value_pe_weight: float = 0.60
    value_pb_weight: float = 0.40
    income_score_cap: float = 1.5
    yield_trap_threshold: float = 0.10
    distress_pe_max: float = 5.0
    quality_volume_high: float = 1_000_000
    quality_volume_low: float = 100_000
    quality_dividend_min: float = 0.02
    quality_dividend_max: float = 0.06
    quality_weight_has_positive_pe: float = 0.5
    quality_weight_distress_pe: float = -0.4
    quality_weight_has_positive_pb: float = 0.2
    quality_weight_high_volume: float = 0.3
    quality_weight_low_volume: float = -0.3
    quality_weight_yield_trap: float = -0.6
    quality_weight_healthy_dividend: float = 0.2


@dataclass(frozen=True)
class MomentumConfig:
    position_bin_boundaries: tuple[float, ...] = (0.15, 0.35, 0.75, 0.95)
    position_bin_scores: tuple[float, ...] = (-0.35, -0.10, 0.55, 0.85, 0.45)
    return_1m_low_position_cutoff: float = 0.40
    return_1m_recovery_threshold: float = 0.05
    return_1m_falling_knife_threshold: float = -0.10
    return_1m_recovery_bonus: float = 0.15
    return_1m_falling_knife_penalty: float = 0.20

    def as_dict(self) -> dict:
        return {
            "position_bin_boundaries": list(self.position_bin_boundaries),
            "position_bin_scores": list(self.position_bin_scores),
            "return_1m_low_position_cutoff": self.return_1m_low_position_cutoff,
            "return_1m_recovery_threshold": self.return_1m_recovery_threshold,
            "return_1m_falling_knife_threshold": self.return_1m_falling_knife_threshold,
            "return_1m_recovery_bonus": self.return_1m_recovery_bonus,
            "return_1m_falling_knife_penalty": self.return_1m_falling_knife_penalty,
        }


@dataclass(frozen=True)
class MomentumV2WeightsConfig:
    rs_3m: float = 0.25
    rs_6m: float = 0.25
    risk_adj_3m: float = 0.20
    trend_structure: float = 0.15
    return_1m: float = 0.10
    return_5d: float = 0.05


@dataclass(frozen=True)
class MomentumV2PenaltiesConfig:
    falling_knife_3m_threshold: float = -0.15
    falling_knife_penalty: float = 0.25
    overextension_52w_threshold: float = 0.97
    overextension_penalty: float = 0.20
    high_vol_annual_threshold: float = 0.50
    high_vol_penalty: float = 0.15


@dataclass(frozen=True)
class MomentumV2Config:
    weights: MomentumV2WeightsConfig = field(default_factory=MomentumV2WeightsConfig)
    penalties: MomentumV2PenaltiesConfig = field(default_factory=MomentumV2PenaltiesConfig)
    clamp_low: float = -1.0
    clamp_high: float = 1.5
    winsorize_pct: float = 0.05

    def weights_dict(self) -> dict[str, float]:
        w = self.weights
        return {
            "rs_3m": w.rs_3m, "rs_6m": w.rs_6m, "risk_adj_3m": w.risk_adj_3m,
            "trend_structure": w.trend_structure, "return_1m": w.return_1m,
            "return_5d": w.return_5d,
        }

    def penalties_dict(self) -> dict[str, float]:
        p = self.penalties
        return {
            "falling_knife_3m_threshold": p.falling_knife_3m_threshold,
            "falling_knife_penalty": p.falling_knife_penalty,
            "overextension_52w_threshold": p.overextension_52w_threshold,
            "overextension_penalty": p.overextension_penalty,
            "high_vol_annual_threshold": p.high_vol_annual_threshold,
            "high_vol_penalty": p.high_vol_penalty,
        }

    def as_dict(self) -> dict:
        return {
            "weights": self.weights_dict(),
            "penalties": self.penalties_dict(),
            "clamp_low": self.clamp_low,
            "clamp_high": self.clamp_high,
            "winsorize_pct": self.winsorize_pct,
        }


@dataclass(frozen=True)
class AnalystConfig:
    strong_buy_ratio: float = 5.0
    net_sell_ratio: float = 1.0
    strong_buy_multiplier: float = 1.05
    net_sell_multiplier: float = 0.95


@dataclass(frozen=True)
class RegimeDefensiveConfig:
    index_pct_override: float | None = 0.85
    max_buys_override: int | None = 3
    stop_loss_tighten: float = 0.05


@dataclass(frozen=True)
class RegimeNeutralConfig:
    index_pct_override: float | None = None
    max_buys_override: int | None = None


@dataclass(frozen=True)
class RegimeConfig:
    spy_ma_period: int = 200
    vix_defensive_threshold: float = 30.0
    vix_neutral_threshold: float = 20.0
    defensive: RegimeDefensiveConfig = field(default_factory=RegimeDefensiveConfig)
    neutral: RegimeNeutralConfig = field(default_factory=RegimeNeutralConfig)


@dataclass(frozen=True)
class HarvestConfig:
    enabled: bool = True
    profit_harvest_pct: float = 0.40
    harvest_to_etfs_pct: float = 0.80
    recycle_to_stocks_pct: float = 0.20
    harvest_only_if_value_metric_below_multiplier: float = 1.20
    min_harvest_amount: float = 25.0
    max_harvest_pct_of_portfolio: float = 0.02
    harvest_etfs: tuple[str, ...] = ("SPY", "VTI")


@dataclass(frozen=True)
class EtfRiskConfig:
    enabled: bool = True
    use_ma_filter: bool = True
    ma_period: int = 200
    defensive_etf_pct: float = 0.85
    defensive_cash_pct: float = 0.10


@dataclass(frozen=True)
class BacktestConfig:
    default_mode: str = "liquid_universe_sanity_test"
    universe_selection: str = "liquid_sample"
    max_symbols: int = 300
    min_volume: float = 500_000
    random_seed: int = 42
    benchmark_symbol: str = "SPY"
    slippage_bps: float = 10.0
    commission_per_trade: float = 0.0
    starting_capital: float = 5_000.0
    weekly_contribution: float = 400.0
    rebalance_frequency_days: int = 5
    deploy_initial_cash: bool = True
    reinvest_sell_proceeds: bool = True
    train_pct: float = 0.70
    use_out_of_sample_validation: bool = True
    auto_apply_if_valid: bool = False
    min_validation_excess_return: float = 0.0
    max_validation_drawdown: float = -0.20
    min_validation_sharpe: float = 0.25
    use_time_weighted_returns: bool = True
    turnover_penalty_enabled: bool = True
    turnover_penalty_trade_count: int = 50
    turnover_penalty_weight: float = 0.35
    llm_review_enabled: bool = False
    llm_review_top_n: int = 5
    llm_review_apply: bool = False
    llm_review_model: str = "claude-sonnet-4-6"
    max_trades_per_week: int = 10
    cooldown_days_after_sell: int = 3
    cooldown_days_after_stopout: int = 7
    vol_slippage_scaling: bool = True
    vol_slippage_multiplier: float = 2.0


@dataclass(frozen=True)
class TuningConfig:
    frozen_parameters: tuple[str, ...] = field(default_factory=tuple)
    parameter_bounds: dict[str, dict[str, float]] = field(default_factory=dict)

    def is_frozen(self, param_path: str) -> bool:
        return param_path in self.frozen_parameters

    def bounds_for(self, param_path: str) -> tuple[float, float] | None:
        b = self.parameter_bounds.get(param_path)
        if b is None:
            return None
        return (float(b["min"]), float(b["max"]))


@dataclass(frozen=True)
class ReliabilityConfig:
    enabled: bool = False
    min_reliability_score: float = 0.70


@dataclass(frozen=True)
class StabilityConfig:
    enabled: bool = True
    windows: tuple[int, ...] = (30, 60, 90, 180, 365)
    objectives: tuple[str, ...] = ("sharpe", "calmar")
    output_dir: str = "reports/stability"
    unstable_spread_threshold: float = 0.15
    unstable_cv_threshold: float = 0.30
    max_unstable_params: int = 5
    scan_maxiter: int = 15
    scan_popsize: int = 6


@dataclass(frozen=True)
class ResearchConfig:
    min_snapshots_for_weight_recommendations: int = 20
    min_snapshots_for_high_confidence: int = 60


@dataclass(frozen=True)
class ArchetypeEntryConfig:
    trim_profit_threshold: float = 0.20
    harvest_profit_threshold: float = 0.30
    trailing_stop_pct: float = -0.08
    minimum_hold_days: int = 10
    thesis_exit_requires_confirmation: bool = False
    allow_deeper_drawdown: bool = False
    # Behavioral controls — defaults are no-ops; live + backtest read these identically.
    enabled: bool = True
    score_multiplier: float = 1.0
    max_position_multiplier: float = 1.0
    max_active_weight: float | None = None
    min_score_to_buy: float | None = None

    def __post_init__(self) -> None:
        if self.score_multiplier < 0.0:
            raise ValueError(
                f"score_multiplier must be >= 0 (got {self.score_multiplier})"
            )
        if self.max_position_multiplier < 0.0:
            raise ValueError(
                f"max_position_multiplier must be >= 0 (got {self.max_position_multiplier})"
            )
        if self.max_active_weight is not None and not (0.0 <= self.max_active_weight <= 1.0):
            raise ValueError(
                f"max_active_weight must be in [0,1] (got {self.max_active_weight})"
            )
        if self.min_score_to_buy is not None and not (-1.0 <= self.min_score_to_buy <= 5.0):
            raise ValueError(
                f"min_score_to_buy must be in [-1,5] (got {self.min_score_to_buy})"
            )


@dataclass(frozen=True)
class ArchetypeManagementConfig:
    enabled: bool = True
    quality_compounder: ArchetypeEntryConfig = field(default_factory=ArchetypeEntryConfig)
    legacy_turnaround: ArchetypeEntryConfig = field(default_factory=ArchetypeEntryConfig)
    speculative_momentum: ArchetypeEntryConfig = field(default_factory=ArchetypeEntryConfig)
    value_recovery: ArchetypeEntryConfig = field(default_factory=ArchetypeEntryConfig)
    defensive_income: ArchetypeEntryConfig = field(default_factory=ArchetypeEntryConfig)
    core_default: ArchetypeEntryConfig = field(default_factory=ArchetypeEntryConfig)

    def as_legacy_dict(self) -> dict:
        """Emit the raw dict format that classify_archetype_from_scores() expects."""
        result: dict = {"enabled": self.enabled}
        for name in ("quality_compounder", "legacy_turnaround", "speculative_momentum",
                     "value_recovery", "defensive_income", "core_default"):
            entry: ArchetypeEntryConfig = getattr(self, name)
            result[name] = {
                "trim_profit_threshold": entry.trim_profit_threshold,
                "harvest_profit_threshold": entry.harvest_profit_threshold,
                "trailing_stop_pct": entry.trailing_stop_pct,
                "minimum_hold_days": entry.minimum_hold_days,
                "thesis_exit_requires_confirmation": entry.thesis_exit_requires_confirmation,
                "allow_deeper_drawdown": entry.allow_deeper_drawdown,
                "enabled": entry.enabled,
                "score_multiplier": entry.score_multiplier,
                "max_position_multiplier": entry.max_position_multiplier,
                "max_active_weight": entry.max_active_weight,
                "min_score_to_buy": entry.min_score_to_buy,
            }
        return result


@dataclass(frozen=True)
class ConcentrationLimitsConfig:
    enabled: bool = True
    max_cluster_weight: float = 0.35
    max_sector_weight: float = 0.40
    cluster_method: str = "pca"
    n_clusters: int = 6
    warn_only: bool = True
