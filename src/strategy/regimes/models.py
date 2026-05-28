"""
strategy/regimes/models.py — Regime data types.

These are plain dataclasses — no business logic, no external imports.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Literal

RegimeLabel = Literal["bullish", "neutral", "defensive"]


@dataclass
class RegimeState:
    """Point-in-time regime classification with supporting market signals."""

    regime: RegimeLabel
    confidence: float                         # 0.0–1.0
    vix: float | None
    spy_price: float | None
    spy_ma200: float | None
    spy_vs_200dma_pct: float | None        # (price/ma200) - 1
    detected_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    previous_regime: RegimeLabel | None = None
    transition_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_defensive(self) -> bool:
        return self.regime == "defensive"

    @property
    def is_bullish(self) -> bool:
        return self.regime == "bullish"

    @property
    def transitioned(self) -> bool:
        return (
            self.previous_regime is not None
            and self.previous_regime != self.regime
        )


@dataclass
class RegimeHistoryEntry:
    """Single row in a historical regime classification series."""

    date: datetime.date
    regime: RegimeLabel
    vix: float
    spy_vs_200dma_pct: float
    confidence: float
