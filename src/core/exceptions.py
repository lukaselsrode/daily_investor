"""
core/exceptions.py — Domain exception hierarchy for Daily Investor.

Every module should raise a subclass of DailyInvestorError rather than
bare ValueError / RuntimeError so callers can catch specifically.
"""

from __future__ import annotations


class DailyInvestorError(Exception):
    """Base exception for all application errors."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ConfigError(DailyInvestorError):
    """Invalid or missing configuration."""


class MissingConfigKeyError(ConfigError):
    """A required config key was not found."""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class DataError(DailyInvestorError):
    """Base class for data pipeline errors."""


class MarketDataError(DataError):
    """Error fetching OHLCV / price data from provider."""


class FundamentalsError(DataError):
    """Error fetching fundamental data (PE, PB, volume, etc.)."""


class InsufficientDataError(DataError):
    """Not enough historical data to perform the computation."""

    def __init__(self, msg: str = "", *, required: int = 0, available: int = 0) -> None:
        self.required = required
        self.available = available
        super().__init__(msg or f"Need {required} data points, have {available}")


class SnapshotError(DataError):
    """Error reading or writing snapshot parquet files."""


class UniverseError(DataError):
    """Error building the stock universe."""


# ---------------------------------------------------------------------------
# Scoring / Strategy
# ---------------------------------------------------------------------------

class ScoringError(DailyInvestorError):
    """Error during factor scoring."""


class RegimeError(DailyInvestorError):
    """Error in regime classification or regime data."""


class FactorResearchError(DailyInvestorError):
    """Error during factor IC or decay computation."""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class ExecutionError(DailyInvestorError):
    """Error placing or cancelling an order."""


class RiskViolation(ExecutionError):
    """Order blocked by a portfolio risk constraint."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Risk violation for {symbol}: {reason}")


class BrokerAuthError(ExecutionError):
    """Authentication with broker failed."""


# ---------------------------------------------------------------------------
# Backtesting / Tuning
# ---------------------------------------------------------------------------

class BacktestError(DailyInvestorError):
    """Error during backtesting."""


class ValidationError(DailyInvestorError):
    """Backtest out-of-sample validation failed."""


class OptimizerError(DailyInvestorError):
    """Error during parameter optimization."""


class OverfitWarning(UserWarning):
    """Emitted when a tuned parameter looks suspiciously overfit."""
