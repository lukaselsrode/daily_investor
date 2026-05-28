"""core — Shared infrastructure: types, logging."""
# SimResult and BacktestReport moved to backtesting.types (canonical, extended schema)
from .logging import configure_logging, get_logger
from .types import (
    PortfolioSnapshot,
    SellDecision,
    SentimentResult,
    TradeRecord,
)

__all__ = [
    "PortfolioSnapshot",
    "SellDecision",
    "SentimentResult",
    "TradeRecord",
    "configure_logging",
    "get_logger",
]
