"""
strategy/base.py — Scorer abstract base class.

Every sub-scorer (value, quality, income, momentum) inherits from ScorerBase.
Each scorer:
  - accepts a dict of raw features
  - returns a single normalized score (float)
  - exposes a diagnostic breakdown dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ScoreBreakdown:
    """Factor contribution breakdown for a single stock."""
    symbol: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ScorerBase(ABC):
    """Abstract scorer interface."""

    @abstractmethod
    def score(self, features: dict) -> float:
        """Compute and return the normalized score for a single stock's features."""

    @abstractmethod
    def breakdown(self, symbol: str, features: dict) -> ScoreBreakdown:
        """Return a full diagnostic breakdown for attribution reporting."""
