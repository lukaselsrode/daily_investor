"""
portfolio/exposure/cluster_enforcement.py — Cluster concentration buy-decision helpers.

Pure functions used by both the live buy-cycle (`portfolio/manager.py:buy_cycle`) and
the backtest simulator (`backtesting/simulator.py:_do_buy`). They take a proposed
buy allocation + the current cluster exposure and decide whether to allow it,
downsize it, or block it.

The existing `compute_concentration()` and `precompute_cluster_labels()` continue
to handle cluster assignment (PCA + KMeans). This module only adds the
buy-decision layer that consumes those labels.

Usage
-----
    decision, adjusted_alloc, reason = cluster_buy_decision(
        symbol="MSFT", cluster_id="cluster_3",
        current_cluster_weight=0.42, alloc=200.0, portfolio_value=10_000,
        cluster_limit=0.60, enforcement_cfg=concentration_limits["enforcement"],
        min_order_amount=30.0, is_underweight=False,
    )
    # decision: "allowed" | "downsized" | "blocked"
    # adjusted_alloc: float (≤ original alloc)
    # reason: human-readable string
"""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def current_cluster_weight(
    holdings: dict,
    cluster_lookup: dict[str, str],
    portfolio_value: float,
    cluster_id: str,
) -> float:
    """Sum the equity of held positions whose cluster_lookup label matches cluster_id."""
    if portfolio_value <= 0:
        return 0.0
    total = 0.0
    for sym, data in (holdings or {}).items():
        if cluster_lookup.get(sym) != cluster_id:
            continue
        eq = data.get("equity") if isinstance(data, dict) else 0.0
        try:
            total += float(eq) if eq is not None else 0.0
        except (TypeError, ValueError):
            continue
    return total / portfolio_value


def projected_cluster_weight(
    current_weight: float,
    alloc: float,
    portfolio_value: float,
) -> float:
    """Projected cluster weight if `alloc` is added to the named cluster.

    Conservative: assumes the new buy is full-weight in the cluster; doesn't
    account for sleeve recycling or simultaneous sells.
    """
    if portfolio_value <= 0:
        return 0.0
    return current_weight + (alloc / portfolio_value)


def cluster_buy_decision(
    symbol: str,
    cluster_id: str | None,
    current_weight: float,
    alloc: float,
    portfolio_value: float,
    cluster_limit: float,
    enforcement_cfg: dict,
    min_order_amount: float,
    *,
    is_underweight: bool = False,
) -> tuple[Literal["allowed", "downsized", "blocked"], float, str]:
    """Decide whether a proposed buy should be allowed at full size, downsized to
    fit the cluster cap, or blocked.

    Parameters
    ----------
    symbol : str — logging only.
    cluster_id : str | None — None means we have no cluster assignment for this
                  symbol. We allow the buy (logged as "no_cluster") so missing
                  data never blocks trading. The caller can choose to skip via
                  config if desired.
    current_weight : current cluster weight as fraction of portfolio_value.
    alloc : proposed buy allocation in dollars.
    portfolio_value : total portfolio equity.
    cluster_limit : max allowed cluster weight (fraction).
    enforcement_cfg : dict mirroring ConcentrationEnforcementConfig.
    min_order_amount : minimum order size; downsized allocs below this become blocks.
    is_underweight : if True and enforcement_cfg["allow_if_underweight"], the
                     allocation is allowed at full size (caller decides what
                     "underweight" means — typically: active sleeve below target).

    Returns
    -------
    (decision, adjusted_alloc, reason)
    """
    # Trivial passes
    if alloc <= 0:
        return ("allowed", 0.0, "alloc=0")
    if cluster_id is None:
        return ("allowed", alloc, "no_cluster_assignment")
    if portfolio_value <= 0:
        return ("allowed", alloc, "no_portfolio_value")
    if cluster_limit <= 0:
        return ("allowed", alloc, "no_cluster_cap_configured")

    # Underweight exemption
    if is_underweight and enforcement_cfg.get("allow_if_underweight", True):
        return ("allowed", alloc, "underweight_exemption")

    projected = projected_cluster_weight(current_weight, alloc, portfolio_value)
    if projected <= cluster_limit:
        return ("allowed", alloc, f"projected={projected:.3f} ≤ cap={cluster_limit:.3f}")

    # Compute the max alloc that fits the cap (with a small safety margin)
    headroom = max(0.0, cluster_limit - current_weight)
    max_fit_alloc = headroom * portfolio_value

    if enforcement_cfg.get("downsize_to_fit", True):
        # Minimum remaining alloc to be a meaningful order
        min_floor = float(enforcement_cfg.get("min_remaining_alloc_multiple", 1.0)) * min_order_amount
        if max_fit_alloc >= min_floor:
            return (
                "downsized", max_fit_alloc,
                f"cluster_cap downsize: {current_weight:.3f}+{alloc/portfolio_value:.3f}"
                f" > cap={cluster_limit:.3f} → fit {max_fit_alloc:.2f}",
            )
        # Downsize alloc falls below the floor — block instead
        return (
            "blocked", 0.0,
            f"cluster_cap: downsize would be ${max_fit_alloc:.2f} < min ${min_floor:.2f}",
        )

    # No downsize — strict block
    if enforcement_cfg.get("block_new_buys", True):
        return (
            "blocked", 0.0,
            f"cluster_cap: cluster={cluster_id} current={current_weight:.3f}"
            f" + alloc={alloc/portfolio_value:.3f} > cap={cluster_limit:.3f}",
        )

    # Fall-through: block_new_buys disabled (warn-only inside enforcement block)
    return ("allowed", alloc, "block_new_buys=false (warn-only-internal)")
