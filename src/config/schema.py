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
class QualityChecklistConfig:
    """Legacy quality-checklist weights, used as small-peer-group fallback."""
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
class MomentumInputsWeightsConfig:
    rs_3m: float = 0.25
    rs_6m: float = 0.25
    risk_adj_3m: float = 0.20
    trend_structure: float = 0.15
    return_1m: float = 0.10
    return_5d: float = 0.05


@dataclass(frozen=True)
class MomentumInputsPenaltiesConfig:
    falling_knife_3m_threshold: float = -0.15
    falling_knife_penalty: float = 0.25
    overextension_52w_threshold: float = 0.97
    overextension_penalty: float = 0.20
    high_vol_annual_threshold: float = 0.50
    high_vol_penalty: float = 0.15


@dataclass(frozen=True)
class MomentumInputsConfig:
    """Cross-sectional momentum input weights + penalties consumed by peer scoring."""
    weights: MomentumInputsWeightsConfig = field(default_factory=MomentumInputsWeightsConfig)
    penalties: MomentumInputsPenaltiesConfig = field(default_factory=MomentumInputsPenaltiesConfig)
    clamp_low: float = -1.0
    clamp_high: float = 1.5
    winsorize_pct: float = 0.05


@dataclass(frozen=True)
class MomentumWarmupConfig:
    """Bin-based momentum scoring used during the simulator's ~63-day warm-up window."""
    position_bin_boundaries: tuple[float, ...] = (0.15, 0.35, 0.75, 0.95)
    position_bin_scores: tuple[float, ...] = (-0.35, -0.10, 0.55, 0.85, 0.45)
    return_1m_low_position_cutoff: float = 0.40
    return_1m_recovery_threshold: float = 0.05
    return_1m_falling_knife_threshold: float = -0.10
    return_1m_recovery_bonus: float = 0.15
    return_1m_falling_knife_penalty: float = 0.20


@dataclass(frozen=True)
class PeerBlendConfig:
    industry_relative: float = 0.60
    sector_relative: float = 0.25
    market_relative: float = 0.15


@dataclass(frozen=True)
class PeerStandardizationConfig:
    group_by: str = "industry"
    fallback_group_by: str = "sector"
    min_group_size: int = 8
    method: str = "percentile"
    winsorize_pct: float = 0.05
    clamp_low: float = -1.0
    clamp_high: float = 1.5
    blend: PeerBlendConfig = field(default_factory=PeerBlendConfig)

    def __post_init__(self) -> None:
        if self.method not in {"percentile", "robust_z"}:
            raise ValueError(f"peer_standardization.method must be 'percentile' or 'robust_z' (got {self.method!r})")
        if self.group_by not in {"industry", "sector", "market"}:
            raise ValueError(f"peer_standardization.group_by invalid: {self.group_by!r}")
        if self.min_group_size < 2:
            raise ValueError(f"peer_standardization.min_group_size must be >= 2 (got {self.min_group_size})")
        if not (0.0 <= self.winsorize_pct <= 0.25):
            raise ValueError(f"peer_standardization.winsorize_pct out of [0, 0.25] (got {self.winsorize_pct})")
        if self.clamp_low >= self.clamp_high:
            raise ValueError(
                f"peer_standardization.clamp_low must be < clamp_high (got {self.clamp_low}, {self.clamp_high})"
            )
        _blend_sum = (
            self.blend.industry_relative + self.blend.sector_relative + self.blend.market_relative
        )
        if abs(_blend_sum - 1.0) > 0.01:
            raise ValueError(
                f"peer_standardization.blend weights must sum to 1.0 (got {_blend_sum:.3f})"
            )


@dataclass(frozen=True)
class FactorDistressConfig:
    pe_threshold: float = 5.0
    pe_penalty: float = 0.30
    negative_eps_penalty: float = 0.25


@dataclass(frozen=True)
class FactorConfig:
    """Per-factor knobs. Not every field applies to every factor — extras are ignored."""
    enabled: bool = True
    peer_relative: bool = True
    pe_weight: float = 0.70
    pb_weight: float = 0.30
    use_legacy_checklist_fallback: bool = True
    safety_aware: bool = True
    anchor_blend: float = 0.0
    distress: FactorDistressConfig = field(default_factory=FactorDistressConfig)


@dataclass(frozen=True)
class ScoringFactorsConfig:
    value: FactorConfig = field(default_factory=lambda: FactorConfig(pe_weight=0.70, pb_weight=0.30))
    quality: FactorConfig = field(default_factory=FactorConfig)
    momentum: FactorConfig = field(default_factory=lambda: FactorConfig(enabled=False))
    income: FactorConfig = field(default_factory=FactorConfig)
    growth_leadership: FactorConfig = field(default_factory=lambda: FactorConfig(enabled=False))


@dataclass(frozen=True)
class ScoringConfig:
    """Single unified scoring engine config — replaces v1 scoring, momentum, momentum_v2,
    value_v2, and scoring_v3 blocks. Hard-cutover; legacy keys raise ConfigError."""
    enabled: bool = True
    peer_standardization: PeerStandardizationConfig = field(default_factory=PeerStandardizationConfig)
    factors: ScoringFactorsConfig = field(default_factory=ScoringFactorsConfig)
    momentum_inputs: MomentumInputsConfig = field(default_factory=MomentumInputsConfig)
    momentum_warmup: MomentumWarmupConfig = field(default_factory=MomentumWarmupConfig)
    quality_checklist: QualityChecklistConfig = field(default_factory=QualityChecklistConfig)


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
    backtest_derisk_frac: float = 0.0
    backtest_derisk_switch_bps: float = 20.0
    backtest_derisk_lag: int = 0


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
class EtfAllocationConstraintsConfig:
    min_weight: float = 0.0
    max_single_etf_weight: float = 0.60
    min_core_market_weight: float = 0.40
    max_growth_weight: float = 0.35
    max_semis_weight: float = 0.25
    max_thematic_combined: float = 0.25
    max_real_estate_weight: float = 0.10
    max_small_cap_weight: float = 0.15
    max_international_weight: float = 0.20
    max_bond_or_cashlike_weight: float = 0.60
    max_gold_commodity_weight: float = 0.15
    max_turnover_per_rebalance: float = 0.35
    rebalance_band: float = 0.03


@dataclass(frozen=True)
class EtfAllocationConfig:
    """ETF/core sleeve allocation. enabled:false AND mode:equal_weight both reproduce
    the historical equal-weight behavior exactly. Weights are bucket-parameterized
    (equal-weight within a bucket) and tuned only via the gated tune-etf-allocation flow."""
    enabled: bool = False
    mode: str = "equal_weight"            # equal_weight | static_weights | regime_weights
    universe_mode: str = "configured_only"  # configured_only | approved_allowlist | curated_exploration
    configured_universe: tuple[str, ...] = field(default_factory=tuple)
    approved_allowlist: tuple[str, ...] = field(default_factory=tuple)
    default_weights: dict[str, float | None] = field(default_factory=dict)
    regime_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    constraints: EtfAllocationConstraintsConfig = field(default_factory=EtfAllocationConstraintsConfig)
    buckets: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    default_mode: str = "liquid_universe_full"
    universe_selection: str = "liquid_sample"
    max_symbols: int = 0   # 0 = full universe (breadth is the edge); >0 caps for smoke-tests only
    min_volume: float = 500_000
    survivorship_free: bool = False
    # Point-in-time fundamentals for survivorship-free backtests/tuning (no static-snapshot
    # look-ahead). Live scoring unchanged. Hard-raises if PIT can't build unless fallback true.
    point_in_time_fundamentals: bool = True
    allow_static_fundamentals_fallback: bool = False
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
class ConcentrationApplyToConfig:
    """Which sleeves are subject to concentration enforcement."""
    active_sleeve: bool = True
    etf_sleeve: bool = False


@dataclass(frozen=True)
class ConcentrationEnforcementConfig:
    """Per-decision rules. Defaults are no-op (block_new_buys gated by warn_only)."""
    block_new_buys: bool = True
    allow_existing_positions: bool = True
    allow_trim_only: bool = True
    allow_sell: bool = True
    allow_if_underweight: bool = True
    downsize_to_fit: bool = True            # try to shrink alloc before blocking
    min_remaining_alloc_multiple: float = 1.0  # x min_order_amount


@dataclass(frozen=True)
class ConcentrationLimitsConfig:
    """Cluster + sector concentration limits. `warn_only: true` (default) keeps
    pre-enforcement behavior — set false to activate `apply_to` + `enforcement`."""
    enabled: bool = True
    warn_only: bool = True
    max_cluster_weight: float = 0.35
    max_sector_weight: float = 0.40
    cluster_method: str = "pca"
    n_clusters: int = 6
    apply_to: ConcentrationApplyToConfig = field(default_factory=ConcentrationApplyToConfig)
    enforcement: ConcentrationEnforcementConfig = field(default_factory=ConcentrationEnforcementConfig)


# ── Archetype classifier v2 — config-driven thresholds + strict defensive_income gate ──

@dataclass(frozen=True)
class ConfidenceBucketsConfig:
    """Thresholds for high/medium/low confidence buckets on classifier results."""
    high_min: float = 0.65
    medium_min: float = 0.45


@dataclass(frozen=True)
class DefensiveIncomeGateConfig:
    """Strict eligibility gate for defensive_income. require_yield=false (default)
    keeps current scoring behavior."""
    require_yield: bool = False
    min_income_score: float = 0.30
    min_quality_score: float = 0.40
    min_momentum_score: float = -0.10
    max_volatility_percentile: float = 0.75
    reject_falling_knife: bool = True
    yield_high: float = 0.80
    yield_moderate: float = 0.50
    yield_minimal: float = 0.05
    sector_defensive: tuple[str, ...] = (
        "Utilities", "Real Estate", "Consumer Non-Durables",
        "Consumer Staples", "Finance",
    )
    industry_defensive: tuple[str, ...] = (
        "Electric Utilities", "Gas Utilities", "Multi-Utilities",
        "Water Utilities", "Real Estate Investment Trusts",
        "Real Estate (Operations & Services)",
    )
    quality_min_label: float = 0.25
    momentum_disqualify_above: float = 0.50


@dataclass(frozen=True)
class QualityCompounderThresholdsConfig:
    market_cap_mega: float = 100_000_000_000
    market_cap_large: float =  10_000_000_000
    market_cap_small: float =     500_000_000
    maintenance_low: float = 0.25
    maintenance_high: float = 0.27
    maintenance_speculative: float = 1.0
    day_trade_normal_max: float = 0.25
    analyst_buy_strong: float = 0.80
    analyst_buy_moderate: float = 0.65
    analyst_buy_weak: float = 0.40
    quality_high: float = 0.60
    quality_moderate: float = 0.35
    quality_low: float = 0.10
    employees_scaled: float = 50_000
    employees_small: float = 2_000


@dataclass(frozen=True)
class LegacyTurnaroundThresholdsConfig:
    maintenance_speculative: float = 1.0
    maintenance_elevated: float = 0.40
    maintenance_above_standard: float = 0.27
    day_trade_elevated: float = 0.25
    market_cap_mid: float = 2_000_000_000
    market_cap_large: float = 10_000_000_000
    market_cap_mega: float = 100_000_000_000
    analyst_buy_weak: float = 0.35
    analyst_buy_moderate: float = 0.55
    analyst_buy_strong: float = 0.80
    momentum_strong: float = 0.30


@dataclass(frozen=True)
class SpeculativeMomentumThresholdsConfig:
    momentum_very_strong: float = 0.60
    momentum_strong: float = 0.35
    quality_very_low: float = 0.10
    quality_low: float = 0.25
    quality_too_high: float = 0.60
    maintenance_high: float = 1.0
    maintenance_elevated: float = 0.40
    day_trade_high: float = 0.40
    market_cap_small: float = 500_000_000
    market_cap_mega: float = 100_000_000_000
    income_minimal: float = 0.05


@dataclass(frozen=True)
class ValueRecoveryThresholdsConfig:
    value_undervalued: float = 0.60
    value_moderate: float = 0.30
    momentum_improving_max: float = 0.40
    momentum_falling_min: float = -0.20
    quality_min: float = 0.15
    quality_max: float = 0.55
    maintenance_distress: float = 1.0


@dataclass(frozen=True)
class ArchetypeClassifierConfig:
    """Config-driven thresholds for each archetype scorer + the defensive_income gate.
    `enabled: false` (default) keeps the v1 hardcoded-threshold scorers' behavior."""
    enabled: bool = False
    confidence_buckets: ConfidenceBucketsConfig = field(default_factory=ConfidenceBucketsConfig)
    defensive_income: DefensiveIncomeGateConfig = field(default_factory=DefensiveIncomeGateConfig)
    quality_compounder: QualityCompounderThresholdsConfig = field(
        default_factory=QualityCompounderThresholdsConfig
    )
    legacy_turnaround: LegacyTurnaroundThresholdsConfig = field(
        default_factory=LegacyTurnaroundThresholdsConfig
    )
    speculative_momentum: SpeculativeMomentumThresholdsConfig = field(
        default_factory=SpeculativeMomentumThresholdsConfig
    )
    value_recovery: ValueRecoveryThresholdsConfig = field(
        default_factory=ValueRecoveryThresholdsConfig
    )
