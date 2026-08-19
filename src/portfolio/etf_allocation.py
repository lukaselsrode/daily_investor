"""
portfolio/etf_allocation.py — ETF/core sleeve allocation (single source of truth).

Decides per-ETF weights WITHIN the ETF/core sleeve. The index_pct split between the
ETF and active sleeves is unchanged and lives in PortfolioManager / the simulator;
this module governs only how the ETF sleeve's dollars are distributed across the ETF
universe, and how the sleeve rebalances toward a target.

Hard guarantees (encoded in tests):
  * enabled:false      -> equal weight over the universe (exact historical behavior)
  * mode:equal_weight  -> equal weight over the universe (exact historical behavior)
  * weights are always NORMALIZED at decision time (sum to 1 over held ETFs)
  * never emits a weight for an ETF outside the passed universe
  * static_weights / regime_weights allocations that violate constraints are REJECTED
    loudly (logged) and fall back to equal weight — the equal-weight baseline itself is
    never constraint-checked (it is the historical incumbent, exempt by design)

Used by both live (PortfolioManager / HarvestManager) and backtest (simulator).
The pure helpers (expand_bucket_weights, validate_allocation, rebalance_plan,
split_budget) are consumed directly by the tuner, which feeds bucket weights from the
optimization vector rather than from config.
"""
from __future__ import annotations

import logging

from util import ETF_ALLOCATION_PARAMS

logger = logging.getLogger(__name__)

# Bucket groupings used by the constraint checks. "thematic" = risk-on equity tilts;
# "bond/cashlike" = the (currently empty) fixed-income buckets reserved for curated
# exploration (Milestone B).
THEMATIC_BUCKETS = ("growth", "semis")
BOND_CASHLIKE_BUCKETS = ("cashlike_bonds", "intermediate_bonds", "long_bonds")

# bucket name -> constraint key for the simple single-bucket MAX caps.
_BUCKET_MAX_CAP: dict[str, str] = {
    "growth":           "max_growth_weight",
    "semis":            "max_semis_weight",
    "real_estate":      "max_real_estate_weight",
    "small_cap":        "max_small_cap_weight",
    "international":     "max_international_weight",
    "gold_commodities": "max_gold_commodity_weight",
}

_EPS = 1e-9

# Allocation resolution runs once per rebalance day, so an unchanging condition would
# otherwise emit the same line hundreds of times (one run logged 522 identical errors,
# which is how the silent-fallback bug stayed buried). Log each distinct signature once.
_LOGGED_ONCE: set[str] = set()


def _log_once(log_fn, key: str, msg: str, *args) -> None:
    if key in _LOGGED_ONCE:
        return
    _LOGGED_ONCE.add(key)
    log_fn(msg, *args)


# ---------------------------------------------------------------------------
# Pure weight helpers
# ---------------------------------------------------------------------------

def equal_weights(universe: list[str]) -> dict[str, float]:
    """Equal weight over the universe. This is the historical (and fallback) behavior."""
    n = len(universe)
    if n == 0:
        return {}
    w = 1.0 / n
    return {e: w for e in universe}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    """Drop non-positive weights and renormalize the rest to sum to 1.0."""
    pos = {k: float(v) for k, v in weights.items() if v and v > 0}
    total = sum(pos.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in pos.items()}


def expand_bucket_weights(
    bucket_weights: dict[str, float],
    buckets: dict[str, list[str]],
    universe: list[str],
) -> dict[str, float]:
    """Expand BUCKET weights to per-ETF weights, equal-weight within each bucket and
    restricted to ETFs present in `universe`. Buckets with no in-universe member are
    dropped and their weight renormalized away. Result is normalized to sum to 1.0."""
    uni = set(universe)
    out: dict[str, float] = {}
    for bucket, bw in bucket_weights.items():
        if not bw or bw <= 0:
            continue
        members = [e for e in buckets.get(bucket, []) if e in uni]
        if not members:
            continue
        per = float(bw) / len(members)
        for e in members:
            out[e] = out.get(e, 0.0) + per
    return normalize(out)


def _etf_to_bucket(buckets: dict[str, list[str]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for b, members in buckets.items():
        for e in members:
            m[e] = b
    return m


def bucket_weight_sums(
    weights: dict[str, float],
    buckets: dict[str, list[str]],
) -> dict[str, float]:
    """Sum per-ETF weights into per-bucket weights (uncategorized ETFs bucket = '_other')."""
    etf_bucket = _etf_to_bucket(buckets)
    sums: dict[str, float] = {}
    for e, w in weights.items():
        b = etf_bucket.get(e, "_other")
        sums[b] = sums.get(b, 0.0) + w
    return sums


def validate_allocation(
    weights: dict[str, float],
    buckets: dict[str, list[str]],
    constraints: dict,
    universe: list[str],
) -> list[str]:
    """Return a list of constraint violations (empty == valid). Applied to tuned /
    static / regime allocations and to tuning candidates BEFORE backtest — NOT to the
    equal-weight incumbent baseline."""
    violations: list[str] = []
    if not weights:
        return ["empty allocation"]

    uni = set(universe)
    outside = sorted(e for e in weights if e not in uni)
    if outside:
        violations.append(f"weights for ETFs outside universe: {outside}")

    s = sum(weights.values())
    if abs(s - 1.0) > 1e-6:
        violations.append(f"weights sum to {s:.4f}, not 1.0")

    min_w = float(constraints["min_weight"])
    max_single = float(constraints["max_single_etf_weight"])
    for e, w in weights.items():
        if w < -_EPS:
            violations.append(f"{e} weight {w:.4f} < 0")
        if w > max_single + _EPS:
            violations.append(f"{e} weight {w:.4f} > max_single_etf_weight {max_single}")
        if _EPS < w < min_w - _EPS:
            violations.append(f"{e} weight {w:.4f} < min_weight {min_w}")

    sums = bucket_weight_sums(weights, buckets)

    core = sums.get("core_market", 0.0)
    min_core = float(constraints["min_core_market_weight"])
    if core < min_core - _EPS:
        violations.append(f"core_market {core:.4f} < min_core_market_weight {min_core}")

    for bucket, ckey in _BUCKET_MAX_CAP.items():
        cap = float(constraints[ckey])
        val = sums.get(bucket, 0.0)
        if val > cap + _EPS:
            violations.append(f"{bucket} {val:.4f} > {ckey} {cap}")

    thematic = sum(sums.get(b, 0.0) for b in THEMATIC_BUCKETS)
    cap = float(constraints["max_thematic_combined"])
    if thematic > cap + _EPS:
        violations.append(f"thematic(growth+semis) {thematic:.4f} > max_thematic_combined {cap}")

    bonds = sum(sums.get(b, 0.0) for b in BOND_CASHLIKE_BUCKETS)
    cap = float(constraints["max_bond_or_cashlike_weight"])
    if bonds > cap + _EPS:
        violations.append(f"bonds/cashlike {bonds:.4f} > max_bond_or_cashlike_weight {cap}")

    return violations


# ---------------------------------------------------------------------------
# Config-driven target weights (the single source of truth)
# ---------------------------------------------------------------------------

def repair_allocation(
    weights: dict[str, float],
    buckets: dict[str, list[str]],
    constraints: dict,
    universe: list[str],
    max_passes: int = 8,
) -> dict[str, float] | None:
    """Project an allocation back into the constraint set, or None if unrepairable.

    Why this exists (2026-08-19): the tradeable universe SHRINKS in historical windows —
    VXUS and SCHD have no prices before 2011, VOO before 2010. `expand_bucket_weights`
    renormalizes over whatever survives, which inflates the remaining buckets and can
    breach their caps. A candidate allocating SMH 21.8% became 29.5% semis in a pre-2011
    stress episode, breaching max_semis_weight, and the old code silently substituted
    EQUAL WEIGHT — so the stress gauntlet scored a different allocation than the one on
    trial and reported a pass. Capping the offenders preserves the candidate's intent;
    substituting equal weight destroys it while looking like success.

    Method: cap each over-limit bucket at its ceiling and redistribute the freed weight to
    core_market, which is the natural sink (it carries a MINIMUM, never a maximum, and
    splits across enough ETFs that max_single_etf_weight cannot bind). Repeat until clean,
    since capping one bucket renormalizes the rest.
    """
    if not weights:
        return None
    core_etfs = [e for e in (buckets.get("core_market") or []) if e in set(universe)]
    if not core_etfs:
        return None   # no sink to absorb the excess

    etf_bucket = _etf_to_bucket(buckets)
    w = normalize({e: max(0.0, float(x)) for e, x in weights.items()})

    for _ in range(max_passes):
        if not validate_allocation(w, buckets, constraints, universe):
            return w
        sums = bucket_weight_sums(w, buckets)

        # Per-bucket ceilings, plus the combined thematic ceiling expressed as a
        # proportional haircut across its member buckets.
        excess: dict[str, float] = {}
        for bucket, ckey in _BUCKET_MAX_CAP.items():
            cap = float(constraints[ckey])
            over = sums.get(bucket, 0.0) - cap
            if over > _EPS:
                excess[bucket] = max(excess.get(bucket, 0.0), over)
        thematic = sum(sums.get(b, 0.0) for b in THEMATIC_BUCKETS)
        t_cap = float(constraints["max_thematic_combined"])
        if thematic > t_cap + _EPS and thematic > _EPS:
            over = thematic - t_cap
            for b in THEMATIC_BUCKETS:
                share = sums.get(b, 0.0) / thematic
                excess[b] = max(excess.get(b, 0.0), over * share)

        if not excess:
            return None   # violations we cannot fix by capping (e.g. core below its floor)

        freed = 0.0
        for bucket, over in excess.items():
            members = [e for e in w if etf_bucket.get(e) == bucket and w[e] > 0]
            b_tot = sum(w[e] for e in members)
            if b_tot <= _EPS:
                continue
            scale = max(0.0, (b_tot - over)) / b_tot
            for e in members:
                freed += w[e] * (1.0 - scale)
                w[e] *= scale
        if freed <= _EPS:
            return None
        for e in core_etfs:
            w[e] = w.get(e, 0.0) + freed / len(core_etfs)
        w = normalize(w)

    return w if not validate_allocation(w, buckets, constraints, universe) else None


def etf_target_weights(
    regime: str,
    universe: list[str],
    params: dict | None = None,
    on_violation: str = "repair",
) -> dict[str, float]:
    """Target per-ETF weights for the ETF sleeve in the given regime.

    `universe` is the ACTIVE ETF universe (the ETFs actually held/tradeable). Weights
    are never emitted for ETFs outside it. enabled:false and equal_weight reproduce the
    historical equal-weight behavior exactly and are NOT constraint-checked.
    """
    p = params if params is not None else ETF_ALLOCATION_PARAMS
    universe = list(universe)
    if not universe:
        return {}

    mode = p.get("mode", "equal_weight")
    if not p.get("enabled", False) or mode == "equal_weight":
        return equal_weights(universe)

    if mode == "static_weights":
        defaults = p.get("default_weights", {}) or {}
        present = {e: defaults.get(e) for e in universe if defaults.get(e) is not None}
        if not present:
            logger.warning("etf_allocation static_weights all-null/absent — using equal weight")
            return equal_weights(universe)
        weights = normalize(present)
    elif mode == "regime_weights":
        bw = (p.get("regime_weights", {}) or {}).get(regime, {}) or {}
        if not bw:
            logger.warning(
                "etf_allocation regime_weights empty for regime=%r — using equal weight", regime
            )
            return equal_weights(universe)
        weights = expand_bucket_weights(bw, p.get("buckets", {}) or {}, universe)
        if not weights:
            logger.warning(
                "etf_allocation regime_weights for regime=%r expanded to nothing in universe — "
                "using equal weight", regime
            )
            return equal_weights(universe)
    else:
        logger.error("etf_allocation unknown mode %r — using equal weight", mode)
        return equal_weights(universe)

    buckets = p.get("buckets", {}) or {}
    violations = validate_allocation(weights, buckets, p["constraints"], universe)
    if not violations:
        return weights

    # A violation here usually means the tradeable universe shrank (VXUS/SCHD have no
    # history before 2011) and renormalization pushed a bucket past its cap. Repair
    # preserves the allocation's intent; the old silent equal-weight substitution scored
    # a DIFFERENT allocation than the one under test and reported it as a pass.
    if on_violation == "raise":
        raise ValueError(
            f"etf_allocation {mode} allocation invalid for regime={regime!r} "
            f"over universe {sorted(universe)}: {'; '.join(violations)}"
        )
    if on_violation == "repair":
        repaired = repair_allocation(weights, buckets, p["constraints"], universe)
        if repaired is not None:
            _log_once(
                logger.warning,
                f"repair:{mode}:{regime}:{','.join(sorted(universe))}",
                "etf_allocation %s regime=%s invalid over the available universe (%s) — "
                "REPAIRED into the constraint set (capped offenders, excess to core_market). "
                "Intent preserved; this is not the incumbent.",
                mode, regime, "; ".join(violations),
            )
            return repaired
        _log_once(
            logger.error,
            f"unrepairable:{mode}:{regime}:{','.join(sorted(universe))}",
            "etf_allocation %s regime=%s invalid (%s) and NOT repairable over universe %s "
            "— falling back to equal weight. Any gate scoring this allocation is scoring "
            "the INCUMBENT, not the candidate.",
            mode, regime, "; ".join(violations), sorted(universe),
        )
        return equal_weights(universe)

    _log_once(
        logger.error,
        f"fallback:{mode}:{regime}:{','.join(sorted(universe))}",
        "etf_allocation %s allocation invalid (%s) — falling back to equal weight",
        mode, "; ".join(violations),
    )
    return equal_weights(universe)


# ---------------------------------------------------------------------------
# Execution helpers (shared by live + backtest)
# ---------------------------------------------------------------------------

def split_budget(target_weights: dict[str, float], budget: float) -> dict[str, float]:
    """Distribute a positive dollar `budget` across ETFs by target weights. Used for
    inflows (contributions, end-of-run sweep, harvest/trim routing). No selling."""
    if budget <= 0:
        return {}
    return {e: budget * w for e, w in target_weights.items() if w > 0}


def rebalance_plan(
    current_dollars: dict[str, float],
    target_weights: dict[str, float],
    rebalance_band: float,
    max_turnover_per_rebalance: float,
) -> dict[str, float]:
    """Per-ETF dollar deltas (buy > 0, sell < 0) to move the ETF sleeve toward
    `target_weights`. No-op (empty) when one-way drift <= rebalance_band or the sleeve
    is empty. One-way turnover is capped at max_turnover_per_rebalance of sleeve value
    (the move is scaled down proportionally if it would exceed the cap). This is what
    makes regime_weights a real, turnover-bounded lever — `enabled:false` never calls it.
    """
    total = sum(max(0.0, float(v)) for v in current_dollars.values())
    if total <= 0:
        return {}
    etfs = set(current_dollars) | set(target_weights)
    cur_w = {e: max(0.0, float(current_dollars.get(e, 0.0))) / total for e in etfs}
    tgt_w = {e: float(target_weights.get(e, 0.0)) for e in etfs}
    drift = 0.5 * sum(abs(tgt_w[e] - cur_w[e]) for e in etfs)
    if drift <= rebalance_band:
        return {}
    scale = 1.0
    if drift > max_turnover_per_rebalance > 0:
        scale = max_turnover_per_rebalance / drift
    deltas: dict[str, float] = {}
    for e in etfs:
        d = (tgt_w[e] - cur_w[e]) * total * scale
        if abs(d) > _EPS:
            deltas[e] = d
    return deltas
