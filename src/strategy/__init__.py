"""strategy — Composable scoring engines."""
from .base import ScorerBase
from .composite import CompositeScorer

__all__ = ["ScorerBase", "CompositeScorer"]
