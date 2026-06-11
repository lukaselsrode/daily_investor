"""
backtesting/types.py — Shared data types for the backtesting package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np

BacktestScope = Literal["overall_strategy", "active_sleeve_compounding"]


class PrecomputedData(NamedTuple):
    symbols: list[str]
    prices: np.ndarray            # (n_days, n_stocks) float64
    pe_comp: np.ndarray           # (n_stocks,)
    pb_comp: np.ndarray           # (n_stocks,)
    quality_scores: np.ndarray    # (n_stocks,)
    income_scores: np.ndarray     # (n_stocks,)
    yield_trap_mask: np.ndarray   # (n_stocks,) bool
    bin_indices: np.ndarray       # (n_stocks,) int 0-4
    has_position_52w: np.ndarray  # (n_stocks,) bool
    position_52w_arr: np.ndarray  # (n_stocks,) float, NaN where missing
    return_1m_arr: np.ndarray     # (n_stocks,) float, NaN where missing
    etf_symbols: list[str]
    etf_prices: np.ndarray        # (n_days, n_etfs) float64
    baseline_scores: np.ndarray   # (n_stocks,) scored with current config
    sector_labels: list[str]      # (n_stocks,) sector per stock
    volume_arr: np.ndarray        # (n_stocks,) daily avg volume
    mode: str                     # lookahead bias mode
    universe_selection: str       # selection method used
    lookahead_bias_level: str     # HIGH / MEDIUM / LOW
    benchmark_prices: np.ndarray  # (n_days,) benchmark close prices
    benchmark_symbol: str
    # Daily rolling price-derived features for dynamic re-scoring
    position_52w_daily: np.ndarray      # (n_days, n_stocks) float, NaN until window fills
    return_1m_daily: np.ndarray         # (n_days, n_stocks) float, NaN until 21d available
    bin_indices_daily: np.ndarray       # (n_days, n_stocks) int
    has_position_52w_daily: np.ndarray  # (n_days, n_stocks) bool
    # Daily rolling momentum-input features — None when not computed (warm-up bin scorer activates)
    ret_5d_daily: np.ndarray | None = None    # (n_days, n_stocks) 5-day return
    ret_3m_daily: np.ndarray | None = None    # (n_days, n_stocks) 63-day return
    ret_6m_daily: np.ndarray | None = None    # (n_days, n_stocks) 126-day return
    rs_3m_daily: np.ndarray | None = None     # (n_days, n_stocks) relative strength vs SPY
    rs_6m_daily: np.ndarray | None = None     # (n_days, n_stocks) relative strength vs SPY
    vol_3m_daily: np.ndarray | None = None    # (n_days, n_stocks) annualized realized vol
    above_50dma_daily: np.ndarray | None = None   # (n_days, n_stocks) bool
    above_200dma_daily: np.ndarray | None = None  # (n_days, n_stocks) bool
    spy_prices: np.ndarray | None = None      # (n_days,) SPY closes for RS computation
    # Per-stock classification signals (used by archetype classifier in backtest).
    # Defaults are empty/None for backward compat with existing PrecomputedData callers.
    industry_labels: tuple[str, ...] = ()     # (n_stocks,) industry per stock
    market_caps: np.ndarray | None = None     # (n_stocks,) market cap; NaN where unknown
    # Static per-stock momentum score (warmup-bin scoring at day 0). Used by the
    # archetype classifier to seed `momentum_score`; live recomputes momentum daily.
    momentum_scores: np.ndarray | None = None  # (n_stocks,) momentum proxy at sim start
    # (n_days, n_stocks) CAUSAL daily dollar-volume (close × volume) for cap-proxy position
    # sizing without the static-market_caps look-ahead. Populated by the survivorship-free loader.
    dollar_volume_daily: np.ndarray | None = None
    # (n_stocks,) bool — discretionary NEVER-BUY mask (industry/sector exclusions), mirroring
    # the live gate in data/fundamentals.py. True → never enters candidate selection. None →
    # no exclusions applied (full-universe research). Built by load_and_precompute.
    excluded_mask: np.ndarray | None = None
    # Optional point-in-time market regime labels aligned to rows. Used when an
    # experiment slices a regime-specific block and the simulator must not reset
    # the 200DMA warm-up context at day 0.
    regime_labels_daily: np.ndarray | None = None
    # (n_days,) ^VIX close aligned to the benchmark calendar. Lets the backtest use the
    # SAME VIX-primary regime classifier as live (strategy/regimes/classifier.py). None →
    # the simulator falls back to the legacy SPY-vs-200DMA rule (byte-identical to pre-VIX).
    vix_prices: np.ndarray | None = None
    # (n_days, n_stocks) bool — True while a symbol still has NATIVE (non-ffilled) prices.
    # Built by the survivorship-free loader: dead names' prices are forward-filled past their
    # delist date so held positions can mark, but they must never be BOUGHT there. None →
    # all finite-priced symbols are tradeable (default yfinance path).
    tradeable_mask_daily: np.ndarray | None = None


@dataclass
class SimResult:
    final_value: float
    total_return: float       # time-weighted return (excludes contributions)
    sharpe: float             # computed from TWR daily series
    calmar: float
    max_drawdown: float
    trades_made: int
    # extended fields — default to 0 for backward compat with existing tuner calls
    sells_made: int = 0
    skipped_buys: int = 0
    cap_reductions: int = 0
    average_positions: float = 0.0
    max_positions: int = 0
    average_cash_pct: float = 0.0
    turnover_estimate: float = 0.0
    friction_cost: float = 0.0
    net_contributions: float = 0.0  # starting_capital + all weekly contributions
    profit: float = 0.0             # final_value - net_contributions
    # attribution & regime diagnostics
    stopout_count: int = 0          # hard stop-loss triggered
    trim_count: int = 0             # partial exits (trim_exit)
    harvest_count: int = 0          # partial profit-harvest exits
    cooldown_skips: int = 0         # buys skipped due to post-sell cooldown
    regime_days: dict | None = None  # {"bullish": N, "neutral": N, "defensive": N}
    # Regime de-risk overlay telemetry (None when overlay disabled, frac=0.0).
    # {"enabled": bool, "frac": float, "lag": int, "days_active": N,
    #  "rotations": N, "switch_cost": $, "max_overlay_value": $}
    overlay_telemetry: dict | None = None
    benchmark_twr: float = 0.0     # contribution-adjusted benchmark TWR for comparison
    # Contribution-timing overlay diagnostics (None when overlay disabled):
    # summarize_decisions() stats + per-week "schedule" rows
    # (day, dip_score, multiplier, contribution, carry_forward, reason_codes).
    contribution_timing: dict | None = None
    pool_diagnostics: CandidatePoolDiagnostics | None = None  # day-0 candidate pool
    trade_log: list = field(default_factory=list)  # list[TradeRecord] for attribution
    # Daily equity curves for charting (empty when not requested)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    benchmark_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    benchmark_ca_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    # Archetype breakdown (populated when archetype_aware=True)
    archetype_pnl: dict[str, float] = field(default_factory=dict)
    archetype_trade_counts: dict[str, int] = field(default_factory=dict)
    archetype_exit_breakdown: dict[str, dict] = field(default_factory=dict)
    # Extended archetype rollups — empty when no trades in the archetype.
    archetype_active_excess: dict[str, float] = field(default_factory=dict)
    archetype_win_rate: dict[str, float] = field(default_factory=dict)
    archetype_avg_hold_days: dict[str, float] = field(default_factory=dict)
    archetype_max_drawdown: dict[str, float] = field(default_factory=dict)
    archetype_sleeve_weight: dict[str, float] = field(default_factory=dict)
    archetype_realized_pnl: dict[str, float] = field(default_factory=dict)
    archetype_unrealized_pnl: dict[str, float] = field(default_factory=dict)
    archetype_decision_source_counts: dict[str, dict] = field(default_factory=dict)
    # Walk-forward cluster concentration result (None when cluster_tracking=False)
    cluster_result: object | None = None
    # Per-cluster rollups (populated when cluster_tracking=True). Empty otherwise.
    cluster_pnl: dict[str, float] = field(default_factory=dict)
    cluster_trade_counts: dict[str, int] = field(default_factory=dict)
    cluster_win_rate: dict[str, float] = field(default_factory=dict)
    cluster_avg_hold_days: dict[str, float] = field(default_factory=dict)
    cluster_sleeve_weight: dict[str, float] = field(default_factory=dict)   # end-of-sim
    cluster_active_excess: dict[str, float] = field(default_factory=dict)
    cluster_dominant_sectors: dict[str, str] = field(default_factory=dict)
    cluster_dominant_archetypes: dict[str, str] = field(default_factory=dict)
    cluster_violations_count: int = 0
    cluster_decision_counts: dict[str, int] = field(default_factory=dict)   # {"allowed", "downsized", "blocked"}
    # Per-archetype attribution split by confidence bucket (high/medium/low).
    # Keys are "{archetype}|{bucket}" strings.
    archetype_pnl_by_confidence: dict[str, float] = field(default_factory=dict)
    archetype_trade_counts_by_confidence: dict[str, int] = field(default_factory=dict)
    # Backtest scope and active sleeve metrics (None when scope == "overall_strategy")
    scope: BacktestScope = "overall_strategy"
    active_equity_curve: np.ndarray | None = None
    active_total_return: float | None = None
    active_sharpe: float | None = None
    active_calmar: float | None = None
    active_max_drawdown: float | None = None
    active_excess_return: float | None = None
    active_information_ratio: float | None = None


@dataclass
class BacktestReport:
    mode: str
    universe_selection: str
    lookahead_bias_level: str
    n_symbols: int
    n_days: int
    train_result: SimResult
    validation_result: SimResult | None
    benchmark_return: float            # train-window benchmark (simple price return)
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    excess_return: float               # train excess return vs benchmark
    validation_benchmark_return: float # validation-window benchmark (0.0 if no val window)
    notes: list[str]
    # extended reporting
    train_benchmark_twr: float = 0.0   # contribution-adjusted benchmark TWR
    val_benchmark_twr: float = 0.0
    trade_log: list = field(default_factory=list)  # list[TradeRecord] from train window
    config_hash: str = ""              # SHA-256[:12] of config at run time
    run_timestamp: str = ""            # ISO datetime string
    # scoring engine metadata (populated whenever a backtest runs)
    scoring_engine_version: str = "peer-1"  # scoring model identity; see SCORING_MODEL_VERSION
    peer_config: dict = field(default_factory=dict)  # SCORING_PARAMS["peer_standardization"]


@dataclass
class CandidatePoolDiagnostics:
    n_candidates: int
    score_cutoff: float
    avg_quality: float
    avg_momentum: float
    avg_income: float
    avg_value: float
    sector_counts: dict
    n_income_trap_excluded: int
    n_quality_gate_excluded: int
    n_momentum_gate_excluded: int
    n_floor_excluded: int
    excluded_high_income_low_momentum: list   # up to 10 symbol names
    # Peer-rank gate diagnostics — populated by the unified scoring engine.
    n_peer_relative_rank_excluded: int = 0
    avg_peer_rank_pct: float = 0.0
    peer_fallback_usage: dict = field(default_factory=dict)  # {fallback_reason → count}
