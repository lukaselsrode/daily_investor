"""
backtesting/cluster_tracker.py — Walk-forward cluster concentration tracking.

Fits PCA+KMeans independently at each rebalance date using ONLY data available
up to that date — no future leakage. Tracks how much of the active sleeve sits
in each factor cluster over the simulation period.

Usage in run_simulation():
    cluster_labels_by_day = precompute_cluster_labels(precomp, rebalance_days)
    # then inside the loop, call record_cluster_snapshot() at each rebalance day
    # after the loop, call build_cluster_result()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_SKLEARN_AVAILABLE: bool | None = None


def _check_sklearn() -> bool:
    global _SKLEARN_AVAILABLE
    if _SKLEARN_AVAILABLE is None:
        try:
            import sklearn  # noqa: F401
            _SKLEARN_AVAILABLE = True
        except ImportError:
            _SKLEARN_AVAILABLE = False
    return _SKLEARN_AVAILABLE


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClusterSnapshot:
    day: int
    cluster_weights: dict[int, float]    # cluster_id -> fraction of active equity
    sector_weights: dict[str, float]     # sector -> fraction of active equity
    max_cluster_weight: float
    violation: bool                      # exceeded config threshold
    n_held: int                          # number of positions at this snapshot


@dataclass
class ClusterTrackingResult:
    snapshots: list[ClusterSnapshot] = field(default_factory=list)
    n_violation_days: int = 0
    avg_max_cluster_weight: float = 0.0
    worst_max_cluster_weight: float = 0.0
    cluster_timeline: np.ndarray | None = None  # (n_snapshots, n_clusters)
    n_clusters: int = 6

    def max_weight_series(self) -> list[float]:
        return [s.max_cluster_weight for s in self.snapshots]

    def violation_days(self) -> list[int]:
        return [s.day for s in self.snapshots if s.violation]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def precompute_cluster_labels(
    precomp,
    rebalance_days: list[int],
    n_clusters: int = 6,
) -> np.ndarray:
    """
    Fit PCA(2) + KMeans at each rebalance day using only data available at that day.
    Returns (n_rebalances, n_stocks) int32 array. Entries are -1 when unassigned.

    Feature columns used (any that are available):
      quality_scores, income_scores, rs_3m_daily[d], rs_6m_daily[d],
      vol_3m_daily[d], ret_3m_daily[d], ret_6m_daily[d]
    """
    if not _check_sklearn():
        logger.warning("scikit-learn not installed — cluster tracking disabled")
        n = len(recomp_symbols(precomp))
        return np.full((len(rebalance_days), n), -1, dtype=np.int32)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_stocks = precomp.prices.shape[1]
    result   = np.full((len(rebalance_days), n_stocks), -1, dtype=np.int32)

    qs = precomp.quality_scores   # (n_stocks,)
    is_ = precomp.income_scores   # (n_stocks,)

    for ri, day in enumerate(rebalance_days):
        try:
            feat_cols = [qs, is_]
            feat_names = ["quality", "income"]

            def _day_arr(arr, _day=day):
                if arr is not None and arr.ndim == 2 and _day < arr.shape[0]:
                    return arr[_day]
                return None

            for arr, name in [
                (precomp.rs_3m_daily,  "rs_3m"),
                (precomp.rs_6m_daily,  "rs_6m"),
                (precomp.vol_3m_daily, "vol_3m"),
                (precomp.ret_3m_daily, "ret_3m"),
                (precomp.ret_6m_daily, "ret_6m"),
            ]:
                col = _day_arr(arr)
                if col is not None:
                    feat_cols.append(col)
                    feat_names.append(name)

            X = np.column_stack(feat_cols).astype(np.float64)

            # Replace NaN / inf with column medians
            for j in range(X.shape[1]):
                col_j = X[:, j]
                mask   = ~np.isfinite(col_j)
                if mask.all():
                    X[:, j] = 0.0
                elif mask.any():
                    X[mask, j] = float(np.nanmedian(col_j[~mask]))

            # Need at least n_clusters valid rows
            valid_rows = np.all(np.isfinite(X), axis=1)
            if valid_rows.sum() < max(n_clusters, 5):
                continue

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            n_comp = min(2, X.shape[1], int(valid_rows.sum()) - 1)
            pca    = PCA(n_components=n_comp, random_state=42)
            Xp     = pca.fit_transform(Xs)

            km = KMeans(n_clusters=min(n_clusters, int(valid_rows.sum())),
                        n_init=3, random_state=42)
            km.fit(Xp)
            result[ri] = km.labels_
        except Exception as exc:
            logger.debug("Cluster fit failed at day %d: %s", day, exc)

    return result


def recomp_symbols(precomp) -> list[str]:
    return precomp.symbols


def record_cluster_snapshot(
    day: int,
    stock_shares: np.ndarray,
    prices: np.ndarray,
    cluster_labels: np.ndarray,
    sector_labels: list[str],
    max_cluster_weight_threshold: float = 0.35,
    max_sector_weight_threshold: float = 0.40,
) -> ClusterSnapshot:
    """
    Compute equity-weighted cluster/sector weights for the current held positions.
    Called once per rebalance day from within run_simulation().
    """
    held = stock_shares > 0
    n_held = int(held.sum())

    if n_held == 0:
        return ClusterSnapshot(
            day=day,
            cluster_weights={},
            sector_weights={},
            max_cluster_weight=0.0,
            violation=False,
            n_held=0,
        )

    equity = stock_shares * prices
    equity = np.where(np.isfinite(equity), equity, 0.0)
    total_active = float(equity[held].sum())
    if total_active <= 0:
        total_active = 1.0

    # Cluster weights
    cluster_eq: dict[int, float] = {}
    for i in np.where(held)[0]:
        cid = int(cluster_labels[i]) if cluster_labels is not None and i < len(cluster_labels) else -1
        if cid >= 0:
            cluster_eq[cid] = cluster_eq.get(cid, 0.0) + float(equity[i])
    cluster_weights = {cid: v / total_active for cid, v in cluster_eq.items()}

    # Sector weights
    sector_eq: dict[str, float] = {}
    for i in np.where(held)[0]:
        sec = sector_labels[i] if i < len(sector_labels) else "Unknown"
        sector_eq[sec] = sector_eq.get(sec, 0.0) + float(equity[i])
    sector_weights = {s: v / total_active for s, v in sector_eq.items()}

    max_cw = max(cluster_weights.values(), default=0.0)
    violation = max_cw > max_cluster_weight_threshold

    return ClusterSnapshot(
        day=day,
        cluster_weights=cluster_weights,
        sector_weights=sector_weights,
        max_cluster_weight=max_cw,
        violation=violation,
        n_held=n_held,
    )


def build_cluster_result(
    snapshots: list[ClusterSnapshot],
    n_clusters: int = 6,
) -> ClusterTrackingResult:
    """Aggregate per-snapshot data into a summary ClusterTrackingResult."""
    if not snapshots:
        return ClusterTrackingResult(n_clusters=n_clusters)

    n_violations = sum(1 for s in snapshots if s.violation)
    max_weights  = [s.max_cluster_weight for s in snapshots]
    avg_max      = float(np.mean(max_weights))
    worst_max    = float(np.max(max_weights))

    # Build timeline matrix (n_snapshots, n_clusters)
    timeline = np.zeros((len(snapshots), n_clusters), dtype=np.float64)
    for ri, snap in enumerate(snapshots):
        for cid, w in snap.cluster_weights.items():
            if 0 <= cid < n_clusters:
                timeline[ri, cid] = w

    return ClusterTrackingResult(
        snapshots=snapshots,
        n_violation_days=n_violations,
        avg_max_cluster_weight=avg_max,
        worst_max_cluster_weight=worst_max,
        cluster_timeline=timeline,
        n_clusters=n_clusters,
    )
