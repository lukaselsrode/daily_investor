"""
core/types.py — Canonical domain types shared across all modules.

These are the single source of truth for data shapes. No business logic lives here —
only type definitions, dataclasses, and TypedDicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# Simulation / Backtest
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """
    Output of a single backtest simulation run.
    Moved here from backtest.py to break the import chain.
    """
    final_value: float
    total_return: float
    sharpe: float
    calmar: float
    max_drawdown: float
    trades_made: int

    # extended diagnostics — default 0 for backward compat
    sells_made: int = 0
    skipped_buys: int = 0
    cap_reductions: int = 0
    average_positions: float = 0.0
    max_positions: int = 0
    average_cash_pct: float = 0.0
    turnover_estimate: float = 0.0
    friction_cost: float = 0.0
    net_contributions: float = 0.0
    profit: float = 0.0

    # regime / attribution diagnostics
    stopout_count: int = 0
    trailing_stop_count: int = 0
    take_profit_count: int = 0
    weak_value_exit_count: int = 0
    yield_trap_exit_count: int = 0
    quality_floor_exit_count: int = 0

    # attribution by sleeve
    etf_return: float = 0.0
    stock_return: float = 0.0
    etf_allocation_avg: float = 0.0

    # regime days
    defensive_days: int = 0
    neutral_days: int = 0
    bullish_days: int = 0

    # validation
    is_valid: bool = True
    validation_notes: list[str] = field(default_factory=list)


@dataclass
class BacktestReport:
    """Full report output from a backtest run — wraps SimResult with metadata."""
    train: SimResult
    validation: Optional[SimResult]
    full: SimResult
    n_days: int
    mode: str
    universe_size: int
    benchmark_return: float
    benchmark_symbol: str
    excess_return: float
    passes_validation: bool
    validation_reason: str = ""

    # parameter snapshot used for this run
    params_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trade records
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single trade event, used for attribution and reporting."""
    date: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    amount: float
    reason: str = ""
    exit_type: Literal["stop_loss", "trailing_stop", "take_profit", "weak_value",
                       "yield_trap", "quality_floor", "harvest_exit", "trim_exit",
                       "regime", ""] = ""
    pnl: float = 0.0
    hold_days: int = 0
    is_partial: bool = False


# ---------------------------------------------------------------------------
# Sell decisions
# ---------------------------------------------------------------------------

@dataclass
class SellDecision:
    """
    Output of SellDecisionEngine.evaluate().
    Previously returned as a plain dict from evaluate_sell_candidate() in main.py.
    """
    symbol: str
    should_sell: bool
    reason: str
    severity: Optional[Literal["hard", "soft"]]
    exit_type: Optional[Literal["failure_exit", "harvest_exit", "trim_exit", "thesis_exit"]]
    percent_change: Optional[float]
    value_metric: Optional[float]
    quality_score: Optional[float]
    yield_trap_flag: Optional[bool]
    trim_fraction: Optional[float] = None


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class SentimentResult(TypedDict):
    """Output of a single-stock or batch sentiment call."""
    action: Literal["BUY", "SELL", "HOLD"]
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


# ---------------------------------------------------------------------------
# Portfolio snapshot (for UI / reporting)
# ---------------------------------------------------------------------------

@dataclass
class PositionSnapshot:
    symbol: str
    quantity: float
    current_price: float
    avg_buy_price: float
    equity: float
    pct_change: float
    sector: str = "Unknown"
    is_etf: bool = False


@dataclass
class PortfolioSnapshot:
    """Point-in-time view of the portfolio, suitable for display and reporting."""
    timestamp: str
    total_equity: float
    available_cash: float
    positions: list[PositionSnapshot] = field(default_factory=list)

    @property
    def invested_value(self) -> float:
        return sum(p.equity for p in self.positions)

    @property
    def etf_value(self) -> float:
        return sum(p.equity for p in self.positions if p.is_etf)

    @property
    def stock_value(self) -> float:
        return sum(p.equity for p in self.positions if not p.is_etf)

    @property
    def etf_pct(self) -> float:
        return self.etf_value / max(self.total_equity, 1e-9)
