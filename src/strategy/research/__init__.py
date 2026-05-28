"""
strategy/research — Compatibility re-export only.

FactorResearchEngine has moved to research/ic_engine.py.
Import from there directly: from research.ic_engine import FactorResearchEngine
"""

from research.ic_engine import FactorResearchEngine

__all__ = ["FactorResearchEngine"]
