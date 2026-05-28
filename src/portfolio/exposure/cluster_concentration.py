"""
portfolio/exposure/cluster_concentration.py — Factor-cluster concentration diagnostics.

Answers: "Is the active sleeve accidentally piled into a single factor-space cluster?"

Flow
────
1. Run PCA + KMeans on the full scored universe (agg_data) to assign a cluster
   label to every symbol.
2. Restrict to owned active positions only.
3. Compute equity-weighted cluster concentration and sector concentration for
   the active sleeve.
4. Flag any cluster or sector whose weight exceeds the configured threshold and
   emit a WARNING log entry for each violation.

The result is a ConcentrationReport that can be:
  • logged at run-start in portfolio/manager.py
  • displayed in the allocation_diagnostics UI component
  • queried programmatically: report.has_violations

Thresholds come from config.yaml → concentration_limits:
    max_cluster_weight: 0.35    # warn if any cluster > 35% of active sleeve
    max_sector_weight:  0.40    # warn if any sector  > 40% of active sleeve
    cluster_method:     pca
    n_clusters:         6
    warn_only:          true    # always true for now — never blocks execution

SAFE: read-only.  Never modifies factor scores, weights, or config.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConcentrationViolation:
    """A single threshold breach in the active sleeve."""
    kind:      str         # "cluster" | "sector"
    label:     str         # cluster id (str) or sector name
    weight:    float       # actual fraction of active sleeve equity
    threshold: float       # configured limit
    symbols:   list[str]   # owned positions contributing to this group
    equity:    float       # total equity in this group


@dataclass
class ConcentrationReport:
    """Full concentration diagnostic for one portfolio snapshot."""
    timestamp:           str
    total_active_equity: float
    n_active_positions:  int

    cluster_weights:     dict[str, float]   # cluster_label -> fraction of active sleeve
    sector_weights:      dict[str, float]   # sector        -> fraction of active sleeve
    cluster_symbols:     dict[str, list[str]]  # cluster_label -> [owned symbols]
    sector_symbols:      dict[str, list[str]]  # sector        -> [owned symbols]

    violations:          list[ConcentrationViolation]

    method:              str   # "pca" | "umap"
    n_clusters:          int
    unmatched_symbols:   list[str]   # owned symbols with no cluster assignment

    @property
    def has_violations(self) -> bool:
        return bool(self.violations)

    def log_warnings(self) -> None:
        """Emit a WARNING log entry for every violation."""
        for v in self.violations:
            logger.warning(
                "CONCENTRATION %s — '%s' holds %.1f%% of active sleeve "
                "(limit %.0f%%).  Positions: %s",
                v.kind.upper(),
                v.label,
                v.weight * 100,
                v.threshold * 100,
                ", ".join(v.symbols[:8]) + (" …" if len(v.symbols) > 8 else ""),
            )

    def summary_lines(self) -> list[str]:
        """Human-readable one-liner per violation (for CLI / log digest)."""
        lines: list[str] = []
        for v in self.violations:
            lines.append(
                f"  [{v.kind}] '{v.label}':  {v.weight:.1%} > {v.threshold:.0%} limit"
                f"  ({len(v.symbols)} position(s), ${v.equity:,.0f})"
            )
        return lines


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_concentration(
    holdings_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    etfs: list[str],
    max_cluster_weight: float = 0.35,
    max_sector_weight:  float = 0.40,
    n_clusters:         int   = 6,
    method:             str   = "pca",
) -> ConcentrationReport:
    """
    Compute factor-cluster and sector concentration for the active sleeve.

    Parameters
    ----------
    holdings_df : holdings CSV frame (columns: symbol, equity, [quantity])
    agg_df      : scored universe frame (columns: symbol, factor scores, sector)
    etfs        : list of ETF tickers to exclude from active sleeve
    max_cluster_weight : warn threshold for any single cluster (fraction of active equity)
    max_sector_weight  : warn threshold for any single sector  (fraction of active equity)
    n_clusters  : number of KMeans clusters to use
    method      : "pca" or "umap"

    Returns
    -------
    ConcentrationReport — always returns (never raises); on total failure returns
    an empty report with the error recorded in unmatched_symbols.
    """
    ts = datetime.datetime.utcnow().isoformat()

    try:
        return _compute(
            holdings_df, agg_df, etfs,
            max_cluster_weight, max_sector_weight,
            n_clusters, method, ts,
        )
    except Exception as exc:
        logger.warning("cluster_concentration: computation failed: %s", exc)
        return ConcentrationReport(
            timestamp=ts,
            total_active_equity=0.0,
            n_active_positions=0,
            cluster_weights={},
            sector_weights={},
            cluster_symbols={},
            sector_symbols={},
            violations=[],
            method=method,
            n_clusters=n_clusters,
            unmatched_symbols=[f"ERROR: {exc}"],
        )


def _compute(
    holdings_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    etfs: list[str],
    max_cluster_weight: float,
    max_sector_weight:  float,
    n_clusters: int,
    method: str,
    ts: str,
) -> ConcentrationReport:
    # ── Step 1: isolate active holdings ──────────────────────────────────────
    h = holdings_df.copy()
    h["equity"] = pd.to_numeric(h.get("equity", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    if "quantity" in h.columns:
        h["quantity"] = pd.to_numeric(h["quantity"], errors="coerce").fillna(0.0)
        h = h[h["quantity"] > 0]

    active = h[~h["symbol"].isin(etfs) & (h["equity"] > 0)].copy()

    if active.empty:
        return ConcentrationReport(
            timestamp=ts, total_active_equity=0.0, n_active_positions=0,
            cluster_weights={}, sector_weights={}, cluster_symbols={},
            sector_symbols={}, violations=[], method=method,
            n_clusters=n_clusters, unmatched_symbols=[],
        )

    total_active_equity = float(active["equity"].sum())
    _active_syms = set(active["symbol"].astype(str).tolist())

    # ── Step 2: run factor map on full universe to get cluster labels ─────────
    from portfolio.visualization.factor_map import build_factor_map

    _, df_mapped, _ = build_factor_map(
        agg_df,
        method=method,
        kmeans_clusters=n_clusters,
        output_html=None,
        show=False,
    )

    # df_mapped has "cluster" column (string) for every symbol in the universe
    cluster_lookup: dict[str, str] = {}
    if "cluster" in df_mapped.columns and "symbol" in df_mapped.columns:
        for _, row in df_mapped[["symbol", "cluster"]].iterrows():
            cluster_lookup[str(row["symbol"])] = str(row["cluster"])

    # Bring sector info into active holdings
    sector_lookup: dict[str, str] = {}
    if "sector" in agg_df.columns and "symbol" in agg_df.columns:
        for _, row in agg_df[["symbol", "sector"]].iterrows():
            s = str(row.get("sector") or "Unknown")
            sector_lookup[str(row["symbol"])] = s

    # ── Step 3: assign cluster + sector to each active position ──────────────
    cluster_equity:  dict[str, float]       = {}
    cluster_symbols: dict[str, list[str]]   = {}
    sector_equity:   dict[str, float]       = {}
    sector_symbols:  dict[str, list[str]]   = {}
    unmatched: list[str] = []

    for _, row in active.iterrows():
        sym = str(row["symbol"])
        eq  = float(row["equity"])

        cl  = cluster_lookup.get(sym)
        if cl is None:
            unmatched.append(sym)
            cl = "unassigned"

        sec = sector_lookup.get(sym, "Unknown")

        cluster_equity[cl]  = cluster_equity.get(cl, 0.0) + eq
        cluster_symbols.setdefault(cl, []).append(sym)

        sector_equity[sec]  = sector_equity.get(sec, 0.0) + eq
        sector_symbols.setdefault(sec, []).append(sym)

    if unmatched:
        logger.debug(
            "cluster_concentration: %d active symbol(s) not in universe map: %s",
            len(unmatched), unmatched[:10],
        )

    # ── Step 4: compute weights and flag violations ───────────────────────────
    cluster_weights = {
        cl: round(eq / total_active_equity, 4)
        for cl, eq in sorted(cluster_equity.items(), key=lambda x: -x[1])
    }
    sector_weights = {
        sec: round(eq / total_active_equity, 4)
        for sec, eq in sorted(sector_equity.items(), key=lambda x: -x[1])
    }

    violations: list[ConcentrationViolation] = []

    for cl, w in cluster_weights.items():
        if w > max_cluster_weight:
            violations.append(ConcentrationViolation(
                kind="cluster", label=cl, weight=w,
                threshold=max_cluster_weight,
                symbols=cluster_symbols.get(cl, []),
                equity=cluster_equity.get(cl, 0.0),
            ))

    for sec, w in sector_weights.items():
        if w > max_sector_weight:
            violations.append(ConcentrationViolation(
                kind="sector", label=sec, weight=w,
                threshold=max_sector_weight,
                symbols=sector_symbols.get(sec, []),
                equity=sector_equity.get(sec, 0.0),
            ))

    return ConcentrationReport(
        timestamp=ts,
        total_active_equity=round(total_active_equity, 2),
        n_active_positions=len(active),
        cluster_weights=cluster_weights,
        sector_weights=sector_weights,
        cluster_symbols=cluster_symbols,
        sector_symbols=sector_symbols,
        violations=violations,
        method=method,
        n_clusters=n_clusters,
        unmatched_symbols=unmatched,
    )


# ---------------------------------------------------------------------------
# Convenience loader (used by UI component and manager.py)
# ---------------------------------------------------------------------------

def run_concentration_check(
    holdings_df: pd.DataFrame | None = None,
    agg_df:      pd.DataFrame | None = None,
    etfs:        list[str] | None    = None,
) -> ConcentrationReport | None:
    """
    Load data from disk if not supplied, apply thresholds from config, and return report.

    Returns None if disabled in config or if data is unavailable.
    """
    from util import CONCENTRATION_LIMIT_PARAMS, ETFS

    params = CONCENTRATION_LIMIT_PARAMS
    if not params["enabled"]:
        return None

    if holdings_df is None or agg_df is None:
        try:
            from data.cache import read_data_as_pd
            if holdings_df is None:
                holdings_df = read_data_as_pd("holdings")
            if agg_df is None:
                agg_df = read_data_as_pd("agg_data")
        except Exception as exc:
            logger.warning("run_concentration_check: could not load data: %s", exc)
            return None

    if holdings_df is None or holdings_df.empty:
        return None
    if agg_df is None or agg_df.empty:
        return None

    if etfs is None:
        etfs = ETFS

    return compute_concentration(
        holdings_df=holdings_df,
        agg_df=agg_df,
        etfs=etfs,
        max_cluster_weight=params["max_cluster_weight"],
        max_sector_weight=params["max_sector_weight"],
        n_clusters=params["n_clusters"],
        method=params["cluster_method"],
    )
