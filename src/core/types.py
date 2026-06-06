"""
core/types.py — Canonical domain types shared across all modules.

These are the single source of truth for shared data shapes. No business logic lives here.

NOTE: The backtest-specific SimResult and BacktestReport are defined in
backtesting/types.py (canonical, extended schema). Import them from there.
core/types.py owns TradeRecord, SellDecision, and the UI/portfolio snapshot types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

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
                       "opportunity_cost", "regime", ""] = ""
    pnl: float = 0.0
    hold_days: int = 0
    is_partial: bool = False
    archetype: str = ""   # populated when archetype_aware=True
    archetype_at_entry: str = ""    # archetype assigned at buy time
    archetype_at_exit: str = ""     # archetype classification at sell time
    decision_source: str = ""       # "global_rule" | "archetype_rule" | "both" | ""
    # Cluster concentration metadata (populated when concentration cap is enforced).
    cluster_id: str = ""
    cluster_decision: str = ""      # "allowed" | "downsized" | "blocked" | ""
    cluster_block_reason: str = ""


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
    severity: Literal["hard", "soft"] | None
    exit_type: Literal["failure_exit", "harvest_exit", "trim_exit", "thesis_exit",
                       "opportunity_cost"] | None
    percent_change: float | None
    value_metric: float | None
    quality_score: float | None
    yield_trap_flag: bool | None
    trim_fraction: float | None = None
    decision_source: str = ""   # "global_rule" | "archetype_rule" | "both" | ""
    # Updated consecutive-weak-evaluation streak to persist, for the archetype
    # `thesis_exit_requires_confirmation` switch. None → the position is not in a weak
    # streak (caller resets to 0); an int → the new streak count to store.
    weak_streak_next: int | None = None


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
