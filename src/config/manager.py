"""
config/manager.py — ConfigManager: typed, singleton config access.

Usage:
    from config import ConfigManager
    cfg = ConfigManager.get()
    print(cfg.risk.max_single_position_pct)
    print(cfg.sell_rules.trailing_stop_pct)

For tests, use from_dict() to avoid touching the filesystem:
    cfg = ConfigManager.from_dict({"metric_threshold": 0.75, ...})

All section properties are cached after first access.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any, ClassVar, Optional

import yaml

from .schema import (
    AnalystConfig,
    BacktestConfig,
    EtfRiskConfig,
    HarvestConfig,
    MomentumConfig,
    MomentumV2Config,
    MomentumV2PenaltiesConfig,
    MomentumV2WeightsConfig,
    RegimeConfig,
    RegimeDefensiveConfig,
    RegimeNeutralConfig,
    ReliabilityConfig,
    ResearchConfig,
    RiskConfig,
    ScoreWeightsConfig,
    ScoringConfig,
    SellRulesConfig,
    StabilityConfig,
    TuningConfig,
    ValuationGuardrailsConfig,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_ROOT, "cfg", "config.yaml")
_DEFAULT_RATIOS = os.path.join(_ROOT, "cfg", "ratios.yaml")


class ConfigManager:
    """
    Singleton config accessor.

    Load once at startup; expose all YAML sections as typed, immutable
    dataclass instances. Callers never parse YAML directly.
    """

    _instance: ClassVar[Optional["ConfigManager"]] = None

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        ratios_path: str = _DEFAULT_RATIOS,
        _data: dict | None = None,
        _ratios: dict | None = None,
    ) -> None:
        if _data is not None:
            self._raw = _data
        else:
            with open(config_path) as f:
                self._raw: dict[str, Any] = yaml.safe_load(f) or {}

        if _ratios is not None:
            self._ratios_raw = _ratios
        else:
            try:
                with open(ratios_path) as f:
                    self._ratios_raw: dict = yaml.safe_load(f) or {}
            except FileNotFoundError:
                self._ratios_raw = {}

    # ── Singleton lifecycle ───────────────────────────────────────────────────

    @classmethod
    def get(cls, *, reload: bool = False) -> "ConfigManager":
        """Return the singleton, creating it from disk on first call."""
        if cls._instance is None or reload:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def from_dict(
        cls,
        data: dict,
        ratios: dict | None = None,
    ) -> "ConfigManager":
        """Create a ConfigManager from a plain dict — for testing."""
        return cls(_data=data, _ratios=ratios or {})

    @classmethod
    def _reset(cls) -> None:
        """Clear singleton — test teardown only."""
        cls._instance = None

    # ── Scalar top-level values ───────────────────────────────────────────────

    @cached_property
    def etfs(self) -> list[str]:
        return list(self._raw.get("etfs", ["SPY", "VOO", "VTI", "QQQ", "SCHD"]))

    @cached_property
    def index_pct(self) -> float:
        return float(self._raw.get("index_pct", 0.70))

    @cached_property
    def metric_threshold(self) -> float:
        return float(self._raw.get("metric_threshold", 0.75))

    @cached_property
    def weekly_investment(self) -> float:
        return float(self._raw.get("weekly_investment", 400))

    @cached_property
    def auto_approve(self) -> bool:
        return bool(self._raw.get("auto_approve", False))

    @cached_property
    def use_sentiment_analysis(self) -> bool:
        return bool(self._raw.get("use_sentiment_analysis", False))

    @cached_property
    def confidence_threshold(self) -> float:
        return float(self._raw.get("confidence_threshold", 65))

    @cached_property
    def sell_sentiment_override_confidence(self) -> float:
        return float(self._raw.get("sell_sentiment_override_confidence", 85))

    @cached_property
    def max_iterations(self) -> int:
        return int(self._raw.get("max_iterations", 10))

    @cached_property
    def dividend_threshold(self) -> float:
        return float(self._raw.get("dividend_threshold", 0.03))

    @cached_property
    def selloff_threshold(self) -> float:
        return float(self._raw.get("selloff_threshold", 30))

    @cached_property
    def ignore_negative_pe(self) -> bool:
        return bool(self._raw.get("ignore_negative_pe", True))

    @cached_property
    def ignore_negative_pb(self) -> bool:
        return bool(self._raw.get("ignore_negative_pb", False))

    # ── Section properties ────────────────────────────────────────────────────

    @cached_property
    def risk(self) -> RiskConfig:
        r = self._raw.get("risk", {})
        return RiskConfig(
            max_single_position_pct=float(r.get("max_single_position_pct", 0.05)),
            max_sector_pct=float(r.get("max_sector_pct", 0.25)),
            max_order_pct_of_cash=float(r.get("max_order_pct_of_cash", 0.10)),
            min_order_amount=float(r.get("min_order_amount", 5.0)),
            min_liquidity_volume=float(r.get("min_liquidity_volume", 500_000)),
            max_buys_per_rebalance=int(r.get("max_buys_per_rebalance", 7)),
            max_sentiment_candidates=int(r.get("max_sentiment_candidates", 20)),
            minimum_hold_days=int(r.get("minimum_hold_days", 0)),
            allow_whole_share_fallback=bool(r.get("allow_whole_share_fallback", False)),
            max_whole_share_buys_per_run=int(r.get("max_whole_share_buys_per_run", 2)),
            max_whole_share_allocation_multiplier=float(r.get("max_whole_share_allocation_multiplier", 1.25)),
            min_index_pct=float(r.get("min_index_pct", 0.60)),
        )

    @cached_property
    def sell_rules(self) -> SellRulesConfig:
        s = self._raw.get("sell_rules", {})
        return SellRulesConfig(
            stop_loss_pct=float(s.get("stop_loss_pct", -0.20)),
            trailing_stop_pct=float(s.get("trailing_stop_pct", -0.08)),
            take_profit_pct=float(s.get("take_profit_pct", 0.60)),
            take_profit_value_floor_multiplier=float(s.get("take_profit_value_floor_multiplier", 1.20)),
            sell_weak_value_below=float(s.get("sell_weak_value_below", 0.45)),
            sell_yield_trap=bool(s.get("sell_yield_trap", True)),
            sell_low_quality_below=float(s.get("sell_low_quality_below", -0.25)),
            min_days_held_before_value_exit=int(s.get("min_days_held_before_value_exit", 21)),
            minimum_days_before_take_profit=int(s.get("minimum_days_before_take_profit", 10)),
        )

    @cached_property
    def score_weights(self) -> ScoreWeightsConfig:
        w = self._raw.get("score_weights", {})
        v = float(w.get("value", 0.08))
        q = float(w.get("quality", 0.50))
        i = float(w.get("income", 0.08))
        m = float(w.get("momentum", 0.34))
        cfg = ScoreWeightsConfig(value=v, quality=q, income=i, momentum=m)
        if not cfg.is_valid:
            import warnings
            warnings.warn(
                f"score_weights sum to {v+q+i+m:.3f} — should be 1.0",
                stacklevel=2,
            )
        return cfg

    @cached_property
    def valuation_guardrails(self) -> ValuationGuardrailsConfig:
        vg = self._raw.get("valuation_guardrails", {})
        return ValuationGuardrailsConfig(
            max_pe_component=float(vg.get("max_pe_component", 5.0)),
            max_pb_component=float(vg.get("max_pb_component", 5.0)),
            min_pe_ratio=float(vg.get("min_pe_ratio", 1.0)),
            min_pb_ratio=float(vg.get("min_pb_ratio", 0.1)),
        )

    @cached_property
    def scoring(self) -> ScoringConfig:
        sc = self._raw.get("scoring", {})
        return ScoringConfig(
            value_pe_weight=float(sc.get("value_pe_weight", 0.60)),
            value_pb_weight=float(sc.get("value_pb_weight", 0.40)),
            income_score_cap=float(sc.get("income_score_cap", 1.5)),
            yield_trap_threshold=float(sc.get("yield_trap_threshold", 0.10)),
            distress_pe_max=float(sc.get("distress_pe_max", 5.0)),
            quality_volume_high=float(sc.get("quality_volume_high", 1_000_000)),
            quality_volume_low=float(sc.get("quality_volume_low", 100_000)),
            quality_dividend_min=float(sc.get("quality_dividend_min", 0.02)),
            quality_dividend_max=float(sc.get("quality_dividend_max", 0.06)),
            quality_weight_has_positive_pe=float(sc.get("quality_weight_has_positive_pe", 0.5)),
            quality_weight_distress_pe=float(sc.get("quality_weight_distress_pe", -0.4)),
            quality_weight_has_positive_pb=float(sc.get("quality_weight_has_positive_pb", 0.2)),
            quality_weight_high_volume=float(sc.get("quality_weight_high_volume", 0.3)),
            quality_weight_low_volume=float(sc.get("quality_weight_low_volume", -0.3)),
            quality_weight_yield_trap=float(sc.get("quality_weight_yield_trap", -0.6)),
            quality_weight_healthy_dividend=float(sc.get("quality_weight_healthy_dividend", 0.2)),
        )

    @cached_property
    def momentum(self) -> MomentumConfig:
        mo = self._raw.get("momentum", {})
        return MomentumConfig(
            position_bin_boundaries=tuple(mo.get("position_bin_boundaries", [0.15, 0.35, 0.75, 0.95])),
            position_bin_scores=tuple(mo.get("position_bin_scores", [-0.35, -0.10, 0.55, 0.85, 0.45])),
            return_1m_low_position_cutoff=float(mo.get("return_1m_low_position_cutoff", 0.40)),
            return_1m_recovery_threshold=float(mo.get("return_1m_recovery_threshold", 0.05)),
            return_1m_falling_knife_threshold=float(mo.get("return_1m_falling_knife_threshold", -0.10)),
            return_1m_recovery_bonus=float(mo.get("return_1m_recovery_bonus", 0.15)),
            return_1m_falling_knife_penalty=float(mo.get("return_1m_falling_knife_penalty", 0.20)),
        )

    @cached_property
    def momentum_v2(self) -> MomentumV2Config:
        mv2 = self._raw.get("momentum_v2", {})
        w = mv2.get("weights", {})
        p = mv2.get("penalties", {})
        return MomentumV2Config(
            weights=MomentumV2WeightsConfig(
                rs_3m=float(w.get("rs_3m", 0.25)),
                rs_6m=float(w.get("rs_6m", 0.25)),
                risk_adj_3m=float(w.get("risk_adj_3m", 0.20)),
                trend_structure=float(w.get("trend_structure", 0.15)),
                return_1m=float(w.get("return_1m", 0.10)),
                return_5d=float(w.get("return_5d", 0.05)),
            ),
            penalties=MomentumV2PenaltiesConfig(
                falling_knife_3m_threshold=float(p.get("falling_knife_3m_threshold", -0.15)),
                falling_knife_penalty=float(p.get("falling_knife_penalty", 0.25)),
                overextension_52w_threshold=float(p.get("overextension_52w_threshold", 0.97)),
                overextension_penalty=float(p.get("overextension_penalty", 0.20)),
                high_vol_annual_threshold=float(p.get("high_vol_annual_threshold", 0.50)),
                high_vol_penalty=float(p.get("high_vol_penalty", 0.15)),
            ),
            clamp_low=float(mv2.get("clamp_low", -1.0)),
            clamp_high=float(mv2.get("clamp_high", 1.5)),
            winsorize_pct=float(mv2.get("winsorize_pct", 0.05)),
        )

    @cached_property
    def analyst(self) -> AnalystConfig:
        ar = self._raw.get("analyst_ratings", {})
        return AnalystConfig(
            strong_buy_ratio=float(ar.get("strong_buy_ratio", 5.0)),
            net_sell_ratio=float(ar.get("net_sell_ratio", 1.0)),
            strong_buy_multiplier=float(ar.get("strong_buy_multiplier", 1.05)),
            net_sell_multiplier=float(ar.get("net_sell_multiplier", 0.95)),
        )

    @cached_property
    def regime(self) -> RegimeConfig:
        rg = self._raw.get("regime", {})
        d = rg.get("defensive", {})
        n = rg.get("neutral", {})
        return RegimeConfig(
            spy_ma_period=int(rg.get("spy_ma_period", 200)),
            vix_defensive_threshold=float(rg.get("vix_defensive_threshold", 30.0)),
            vix_neutral_threshold=float(rg.get("vix_neutral_threshold", 20.0)),
            defensive=RegimeDefensiveConfig(
                index_pct_override=(float(d["index_pct_override"]) if d.get("index_pct_override") not in (None, "None", "") else None),
                max_buys_override=(int(d["max_buys_override"]) if d.get("max_buys_override") not in (None, "None", "") else None),
                stop_loss_tighten=float(d.get("stop_loss_tighten", 0.05)),
            ),
            neutral=RegimeNeutralConfig(
                index_pct_override=(float(n["index_pct_override"]) if n.get("index_pct_override") not in (None, "None", "") else None),
                max_buys_override=(int(n["max_buys_override"]) if n.get("max_buys_override") not in (None, "None", "") else None),
            ),
        )

    @cached_property
    def harvest(self) -> HarvestConfig:
        hv = self._raw.get("harvest", {})
        return HarvestConfig(
            enabled=bool(hv.get("enabled", True)),
            profit_harvest_pct=float(hv.get("profit_harvest_pct", 0.40)),
            harvest_to_etfs_pct=float(hv.get("harvest_to_etfs_pct", 0.80)),
            recycle_to_stocks_pct=float(hv.get("recycle_to_stocks_pct", 0.20)),
            harvest_only_if_value_metric_below_multiplier=float(
                hv.get("harvest_only_if_value_metric_below_multiplier", 1.20)
            ),
            min_harvest_amount=float(hv.get("min_harvest_amount", 25.0)),
            max_harvest_pct_of_portfolio=float(hv.get("max_harvest_pct_of_portfolio", 0.02)),
            harvest_etfs=tuple(hv.get("harvest_etfs", ["SPY", "VTI"])),
        )

    @cached_property
    def etf_risk(self) -> EtfRiskConfig:
        er = self._raw.get("etf_risk", {})
        return EtfRiskConfig(
            enabled=bool(er.get("enabled", True)),
            use_ma_filter=bool(er.get("use_ma_filter", True)),
            ma_period=int(er.get("ma_period", 200)),
            defensive_etf_pct=float(er.get("defensive_etf_pct", 0.85)),
            defensive_cash_pct=float(er.get("defensive_cash_pct", 0.10)),
        )

    @cached_property
    def backtest(self) -> BacktestConfig:
        bt = self._raw.get("backtest", {})
        return BacktestConfig(
            default_mode=str(bt.get("default_mode", "liquid_universe_sanity_test")),
            universe_selection=str(bt.get("universe_selection", "liquid_sample")),
            max_symbols=int(bt.get("max_symbols", 300)),
            min_volume=float(bt.get("min_volume", 500_000)),
            random_seed=int(bt.get("random_seed", 42)),
            benchmark_symbol=str(bt.get("benchmark_symbol", "SPY")),
            slippage_bps=float(bt.get("slippage_bps", 10.0)),
            commission_per_trade=float(bt.get("commission_per_trade", 0.0)),
            starting_capital=float(bt.get("starting_capital", 5_000.0)),
            weekly_contribution=float(bt.get("weekly_contribution", 400.0)),
            rebalance_frequency_days=int(bt.get("rebalance_frequency_days", 5)),
            deploy_initial_cash=bool(bt.get("deploy_initial_cash", True)),
            reinvest_sell_proceeds=bool(bt.get("reinvest_sell_proceeds", True)),
            train_pct=float(bt.get("train_pct", 0.70)),
            use_out_of_sample_validation=bool(bt.get("use_out_of_sample_validation", True)),
            auto_apply_if_valid=bool(bt.get("auto_apply_if_valid", False)),
            min_validation_excess_return=float(bt.get("min_validation_excess_return", 0.0)),
            max_validation_drawdown=float(bt.get("max_validation_drawdown", -0.20)),
            min_validation_sharpe=float(bt.get("min_validation_sharpe", 0.25)),
            use_time_weighted_returns=bool(bt.get("use_time_weighted_returns", True)),
            turnover_penalty_enabled=bool(bt.get("turnover_penalty_enabled", True)),
            turnover_penalty_trade_count=int(bt.get("turnover_penalty_trade_count", 50)),
            turnover_penalty_weight=float(bt.get("turnover_penalty_weight", 0.35)),
            llm_review_enabled=bool(bt.get("llm_review_enabled", False)),
            llm_review_top_n=int(bt.get("llm_review_top_n", 5)),
            llm_review_apply=bool(bt.get("llm_review_apply", False)),
            llm_review_model=str(bt.get("llm_review_model", "claude-sonnet-4-6")),
            max_trades_per_week=int(bt.get("max_trades_per_week", 10)),
            cooldown_days_after_sell=int(bt.get("cooldown_days_after_sell", 3)),
            cooldown_days_after_stopout=int(bt.get("cooldown_days_after_stopout", 7)),
            vol_slippage_scaling=bool(bt.get("vol_slippage_scaling", True)),
            vol_slippage_multiplier=float(bt.get("vol_slippage_multiplier", 2.0)),
        )

    @cached_property
    def tuning(self) -> TuningConfig:
        tn = self._raw.get("tuning", {})
        return TuningConfig(
            frozen_parameters=tuple(tn.get("frozen_parameters", [])),
            parameter_bounds=dict(tn.get("parameter_bounds", {})),
        )

    @cached_property
    def reliability(self) -> ReliabilityConfig:
        rel = self._raw.get("reliability", {})
        return ReliabilityConfig(
            enabled=bool(rel.get("enabled", False)),
            min_reliability_score=float(rel.get("min_reliability_score", 0.70)),
        )

    @cached_property
    def stability(self) -> StabilityConfig:
        st = self._raw.get("stability", {})
        return StabilityConfig(
            enabled=bool(st.get("enabled", True)),
            windows=tuple(st.get("windows", [30, 60, 90, 180, 365])),
            objectives=tuple(st.get("objectives", ["sharpe", "calmar"])),
            output_dir=str(st.get("output_dir", "reports/stability")),
            unstable_spread_threshold=float(st.get("unstable_spread_threshold", 0.15)),
            unstable_cv_threshold=float(st.get("unstable_cv_threshold", 0.30)),
            max_unstable_params=int(st.get("max_unstable_params", 5)),
            scan_maxiter=int(st.get("scan_maxiter", 15)),
            scan_popsize=int(st.get("scan_popsize", 6)),
        )

    @cached_property
    def research(self) -> ResearchConfig:
        rc = self._raw.get("research", {})
        return ResearchConfig(
            min_snapshots_for_weight_recommendations=int(
                rc.get("min_snapshots_for_weight_recommendations", 20)
            ),
            min_snapshots_for_high_confidence=int(
                rc.get("min_snapshots_for_high_confidence", 60)
            ),
        )

    # ── Raw access (backward compat / tuner writes) ───────────────────────────

    @property
    def raw(self) -> dict:
        """Direct dict access — use only for YAML write-back in tuner."""
        return self._raw

    @property
    def ratios_raw(self) -> dict:
        """Raw ratios.yaml data — used by get_investment_ratios()."""
        return self._ratios_raw

    # ── Convenience helpers ───────────────────────────────────────────────────

    def effective_index_pct(self, regime: str) -> float:
        """Return index_pct adjusted for the current market regime."""
        if regime == "defensive":
            ovr = self.regime.defensive.index_pct_override
            return ovr if ovr is not None else self.index_pct
        if regime == "neutral":
            ovr = self.regime.neutral.index_pct_override
            return ovr if ovr is not None else self.index_pct
        return self.index_pct

    def effective_max_buys(self, regime: str) -> int:
        """Return max_buys_per_rebalance adjusted for the current market regime."""
        if regime == "defensive":
            ovr = self.regime.defensive.max_buys_override
            return ovr if ovr is not None else self.risk.max_buys_per_rebalance
        if regime == "neutral":
            ovr = self.regime.neutral.max_buys_override
            return ovr if ovr is not None else self.risk.max_buys_per_rebalance
        return self.risk.max_buys_per_rebalance

    def __repr__(self) -> str:
        return (
            f"ConfigManager(metric_threshold={self.metric_threshold}, "
            f"index_pct={self.index_pct}, "
            f"score_weights={self.score_weights})"
        )
