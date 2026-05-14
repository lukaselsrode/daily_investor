"""
core/interfaces.py — Typed service protocols for Daily Investor.

Defines the contracts every major service must satisfy.
No business logic — interfaces only.
Concrete implementations live in their respective modules.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Protocol, runtime_checkable

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@runtime_checkable
class IConfigManager(Protocol):
    """Typed config accessor."""

    @property
    def metric_threshold(self) -> float: ...

    @property
    def index_pct(self) -> float: ...

    def effective_index_pct(self, regime: str) -> float: ...

    def effective_max_buys(self, regime: str) -> int: ...


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@runtime_checkable
class IMarketDataService(Protocol):
    """Fetch market prices and derived momentum features."""

    def get_prices(self, symbols: list[str], period: str = "1y") -> pd.DataFrame: ...

    def get_momentum_features(self, symbols: list[str]) -> dict[str, dict]: ...

    def get_spy_returns(self) -> dict[str, Optional[float]]: ...


@runtime_checkable
class IFundamentalDataService(Protocol):
    """Fetch fundamental data (PE, PB, volume, etc.) from broker."""

    def get_fundamentals(self, symbols: list[str]) -> dict[str, dict]: ...

    def get_quotes(self, symbols: list[str]) -> dict[str, float]: ...

    def get_analyst_ratings(self, symbol: str) -> Optional[float]: ...


# ---------------------------------------------------------------------------
# Strategy / Factors
# ---------------------------------------------------------------------------


@runtime_checkable
class IFactorEngine(Protocol):
    """Score stocks using the multi-factor model."""

    def score_universe(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def score_single(self, symbol: str, features: dict) -> dict: ...

    def factor_exposures(self, df: pd.DataFrame) -> dict[str, dict]: ...

    def factor_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class IRegimeDetector(Protocol):
    """Classify current market regime from live market signals."""

    def detect(self) -> Any: ...

    def detect_from_data(
        self,
        spy_price: Optional[float],
        spy_ma200: Optional[float],
        vix: Optional[float],
    ) -> Any: ...

    def classify_history(self, days: int = 365) -> list[Any]: ...


@runtime_checkable
class IFactorResearchEngine(Protocol):
    """Multi-horizon IC analytics and factor decay research."""

    def compute_multi_horizon_ic(
        self,
        factors: Optional[list[str]] = None,
        horizons: Optional[list[int]] = None,
        ic_type: str = "spearman",
    ) -> pd.DataFrame: ...

    def compute_ic_summary(self, ic_df: pd.DataFrame) -> pd.DataFrame: ...

    def compute_factor_decay(self, factors: Optional[list[str]] = None) -> pd.DataFrame: ...

    def compute_decile_spread(
        self,
        factor: str,
        horizon_days: int = 20,
        n_deciles: int = 10,
    ) -> pd.DataFrame: ...

    def compute_rolling_icir(
        self,
        factor: str,
        horizon_days: int = 20,
        window: int = 12,
    ) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@runtime_checkable
class IPortfolioConstructor(Protocol):
    """Construct target portfolio allocations given scored universe."""

    def build(
        self,
        scored_df: pd.DataFrame,
        cash: float,
        regime: str,
    ) -> list[dict]: ...


@runtime_checkable
class IRiskManager(Protocol):
    """Enforce portfolio-level risk constraints."""

    def can_buy(
        self,
        symbol: str,
        amount: float,
        portfolio: dict,
    ) -> tuple[bool, str]: ...

    def can_sell(
        self,
        symbol: str,
        quantity: float,
        portfolio: dict,
    ) -> tuple[bool, str]: ...


@runtime_checkable
class IExposureAnalyzer(Protocol):
    """Compute portfolio factor and sector exposure diagnostics."""

    def analyze(
        self,
        portfolio: dict,
        universe_df: pd.DataFrame,
        total_equity: float,
        cash: float,
    ) -> Any: ...

    def compute_rolling_drift(self, portfolio: dict, days: int = 90) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@runtime_checkable
class IExecutionEngine(Protocol):
    """Place and cancel orders via broker."""

    def buy(self, symbol: str, amount: float, dry_run: bool = True) -> dict: ...

    def sell(self, symbol: str, quantity: float, dry_run: bool = True) -> dict: ...


# ---------------------------------------------------------------------------
# Backtesting / Tuning
# ---------------------------------------------------------------------------


@runtime_checkable
class IBacktestEngine(Protocol):
    """Run portfolio simulations over historical universe data."""

    def run(self, universe_df: pd.DataFrame, params: dict) -> Any: ...


@runtime_checkable
class IOptimizerEngine(Protocol):
    """Optimize strategy parameters against validation objectives."""

    def optimize(self, objective: str, max_iter: int) -> dict: ...


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@runtime_checkable
class IReportingEngine(Protocol):
    """Generate structured run summaries and reports."""

    def generate_summary(self, run_result: dict) -> str: ...


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RegimeLabel = Literal["bullish", "neutral", "defensive"]
