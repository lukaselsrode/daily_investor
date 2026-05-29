"""
strategy/scoring — Peer-relative factor scoring engine (sole implementation).

Re-scores each ticker against its industry / sector / market peers. Replaces the
prior v1 (raw checklist) and v2 (cross-sectional) lineages.

Public entry point:
    strategy.scoring.composite.compute_metric(df, score_weights, scoring_config)
"""

from .composite import compute_metric, scoring_config_hash
from .growth import apply_growth
from .income import apply_income
from .momentum import apply_momentum
from .peer import blend_relative, peer_percentile, robust_z
from .quality import apply_quality
from .value import apply_value

__all__ = [
    "apply_growth",
    "apply_income",
    "apply_momentum",
    "apply_quality",
    "apply_value",
    "blend_relative",
    "compute_metric",
    "peer_percentile",
    "robust_z",
    "scoring_config_hash",
]
