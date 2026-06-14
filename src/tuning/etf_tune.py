"""
tuning/etf_tune.py — Gated ETF/core-sleeve allocation tournament.

Brings the active-sleeve research discipline to the ETF sleeve: generate candidate
regime-aware bucket allocations, evaluate each on a train/val split, then run the
winner through the SAME six gates that protect active-stock config writes
(train/val · incumbent-relative · turnover · paired random-window · multi-horizon ·
stress gauntlet). Nothing is written unless a candidate clears every gate.

Candidates are full param vectors with the ETF slots set (enabled=1 for tuned, 0 for
the incumbent). Because run_simulation reads the ETF slots when scope=="etf_allocation",
every existing gate runs an ETF candidate unchanged — no gate plumbing is duplicated.
"""
from __future__ import annotations

import logging

import numpy as np

from backtesting.reports import format_etf_sleeve_diagnostics
from backtesting.simulator import run_backtest_report, split_price_window
from portfolio.etf_allocation import expand_bucket_weights, validate_allocation
from util import BACKTEST_PARAMS, ETF_ALLOCATION_PARAMS

from .constants import (
    _ETF_BUCKETS,
    _ETF_ENABLED_SLOT,
    _ETF_REGIMES,
    _ETF_WEIGHT_SLOT_OFFSET,
    _current_params,
)
from .gauntlet import stress_gauntlet
from .reports import apply_etf_allocation_params
from .tuner import (
    multi_horizon_confirm,
    paired_random_window_gate,
    should_apply_tuned_config,
    validate_tuned_params,
)

logger = logging.getLogger(__name__)
_SCOPE = "etf_allocation"

# Objective weights (Phase 8) — named so they stay tunable/auditable.
W_CALMAR = 0.25
W_SHARPE = 0.10
W_ETF_TURNOVER = 0.25          # penalize one-way ETF turnover (incumbent equal-weight = 0)
W_DRAWDOWN_WORSE = 0.50        # penalize drawdown worse than incumbent
W_CONCENTRATION = 0.10         # penalize bucket concentration (HHI over the regime mix)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def _vector_from_regime_buckets(rbw: dict[str, dict[str, float]], *, enabled: bool = True) -> np.ndarray:
    """Build a full param vector from a {regime: {bucket: weight}} spec (ETF slots only;
    every active-stock slot stays at incumbent)."""
    v = _current_params().copy()
    v[_ETF_ENABLED_SLOT] = 1.0 if enabled else 0.0
    for ri, r in enumerate(_ETF_REGIMES):
        bw = rbw.get(r, {}) or {}
        for bi, b in enumerate(_ETF_BUCKETS):
            v[_ETF_WEIGHT_SLOT_OFFSET + ri * len(_ETF_BUCKETS) + bi] = float(bw.get(b, 0.0))
    return v


def _is_valid_spec(rbw: dict[str, dict[str, float]]) -> bool:
    """A regime spec is valid only if EVERY regime's expanded allocation satisfies the
    constraints (over the configured universe)."""
    buckets = ETF_ALLOCATION_PARAMS["buckets"]
    cons = ETF_ALLOCATION_PARAMS["constraints"]
    uni = ETF_ALLOCATION_PARAMS["configured_universe"]
    for r in _ETF_REGIMES:
        w = expand_bucket_weights(rbw.get(r, {}) or {}, buckets, uni)
        if validate_allocation(w, buckets, cons, uni):
            return False
    return True


def _all_regimes(bw: dict[str, float]) -> dict[str, dict[str, float]]:
    return {r: dict(bw) for r in _ETF_REGIMES}


def _baseline_specs() -> list[tuple[str, dict]]:
    """Interpretable bucket-level baselines (configured_only). Each is constraint-valid
    by construction; invalid ones are filtered before scoring."""
    out: list[tuple[str, dict]] = [
        ("core_only",        _all_regimes({"core_market": 1.0})),
        ("core_dividend",    _all_regimes({"core_market": 0.6, "dividend_defensive": 0.4})),
        ("core_growth",      _all_regimes({"core_market": 0.65, "growth": 0.35})),
        ("core_div_intl",    _all_regimes({"core_market": 0.5, "dividend_defensive": 0.3, "international": 0.2})),
        ("core_reit",        _all_regimes({"core_market": 0.7, "dividend_defensive": 0.2, "real_estate": 0.1})),
        ("risk_on",          _all_regimes({"core_market": 0.45, "growth": 0.2, "semis": 0.05, "small_cap": 0.15, "international": 0.15})),
        ("defensive_heavy",  _all_regimes({"core_market": 0.7, "dividend_defensive": 0.3})),
        ("balanced",         _all_regimes({"core_market": 0.5, "dividend_defensive": 0.2, "growth": 0.1, "international": 0.1, "real_estate": 0.1})),
        # Regime-conditional intuition (validated, not hardcoded): risk-on bull, defensive defensive.
        ("regime_tilted", {
            "bullish":   {"core_market": 0.45, "growth": 0.2, "semis": 0.05, "small_cap": 0.15, "international": 0.15},
            "neutral":   {"core_market": 0.6, "dividend_defensive": 0.2, "growth": 0.1, "international": 0.1},
            "defensive": {"core_market": 0.75, "dividend_defensive": 0.25},
        }),
    ]
    return [(cid, spec) for cid, spec in out if _is_valid_spec(spec)]


def _random_specs(n: int, seed: int) -> list[tuple[str, dict]]:
    """Constrained random bucket-weight candidates. Rejection-sampled: only allocations
    that pass the constraint set are kept (so invalid weights never reach a backtest)."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, dict]] = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        # Sample one shared bucket vector (static across regimes) — interpretable, low-DOF.
        raw = rng.random(len(_ETF_BUCKETS))
        # Bias core_market up so the 0.40 floor is reachable more often.
        raw[0] += 1.0
        bw = {b: float(raw[i]) for i, b in enumerate(_ETF_BUCKETS)}
        s = sum(bw.values())
        bw = {b: w / s for b, w in bw.items()}
        spec = _all_regimes(bw)
        if _is_valid_spec(spec):
            out.append((f"random_{len(out)}", spec))
    return out


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _sim_for_score(report):
    return report.validation_result or report.train_result


def _bench_for_score(report):
    return (report.validation_benchmark_return
            if report.validation_result is not None else report.benchmark_return)


def etf_robust_score(report, incumbent_report, spec: dict | None = None) -> float:
    """Phase-8 objective: reward validation excess vs SPY and vs incumbent, Calmar,
    Sharpe; penalize ETF turnover, worse-than-incumbent drawdown, and concentration."""
    sim = _sim_for_score(report)
    inc = _sim_for_score(incumbent_report)
    if sim is None or inc is None:
        return -1e9
    val_excess = sim.total_return - _bench_for_score(report)
    inc_excess = inc.total_return - _bench_for_score(incumbent_report)
    vs_inc = val_excess - inc_excess
    etf_to = sim.etf_turnover or 0.0
    dd_worse = max(0.0, (inc.max_drawdown - sim.max_drawdown))  # maxdd negative; worse = lower
    conc = 0.0
    if spec is not None:
        # HHI over the (regime-averaged) bucket mix; 1/n_buckets = perfectly diversified.
        avg = {b: 0.0 for b in _ETF_BUCKETS}
        for r in _ETF_REGIMES:
            bw = spec.get(r, {}) or {}
            tot = sum(bw.values()) or 1.0
            for b in _ETF_BUCKETS:
                avg[b] += (bw.get(b, 0.0) / tot) / len(_ETF_REGIMES)
        conc = sum(w * w for w in avg.values())
    return (
        val_excess + vs_inc
        + W_CALMAR * (sim.calmar or 0.0)
        + W_SHARPE * (sim.sharpe or 0.0)
        - W_ETF_TURNOVER * etf_to
        - W_DRAWDOWN_WORSE * dd_worse
        - W_CONCENTRATION * conc
    )


# ---------------------------------------------------------------------------
# Tournament driver
# ---------------------------------------------------------------------------

def run_etf_allocation_tune(
    n_days: int = 1250,
    preset: str = "etf_allocation",
    random_topk: int = 10,
    apply: bool = False,
    force_apply: bool = False,
    seed: int = 42,
    mode: str | None = None,
) -> dict:
    """Generate ETF allocation candidates, gate them, and (optionally) write the winner.

    Returns a result dict with the candidate table, the selected candidate (or None),
    per-gate outcomes, and whether config was written.
    """
    from backtesting.data_loader import load_and_precompute

    bcfg = BACKTEST_PARAMS
    precomp = load_and_precompute(n_days, mode=mode)
    train_sl, val_sl = split_price_window(precomp.prices.shape[0], bcfg.get("train_pct", 0.70))

    incumbent = _current_params()  # enabled=0 → equal-weight (current behavior)
    inc_report = run_backtest_report(precomp, incumbent, train_sl, val_sl, scope=_SCOPE)

    # Candidate pool (configured_only): baselines + constrained random search.
    specs: list[tuple[str, str, dict]] = [("baseline:" + cid, "baseline", spec)
                                          for cid, spec in _baseline_specs()]
    specs += [("random:" + cid, "random", spec) for cid, spec in _random_specs(random_topk, seed)]

    rows: list[dict] = []
    for cid, source, spec in specs:
        vec = _vector_from_regime_buckets(spec, enabled=True)
        rep = run_backtest_report(precomp, vec, train_sl, val_sl, scope=_SCOPE)
        passed_split, reasons = validate_tuned_params(rep, bcfg, incumbent_report=inc_report)
        score = etf_robust_score(rep, inc_report, spec=spec)
        sim = _sim_for_score(rep)
        rows.append({
            "candidate_id": cid, "source": source, "universe_mode": "configured_only",
            "spec": spec, "vector": vec, "report": rep, "score": score,
            "split_passed": passed_split, "split_reasons": reasons,
            "val_excess": (sim.total_return - _bench_for_score(rep)) if sim else None,
            "sharpe": sim.sharpe if sim else None, "calmar": sim.calmar if sim else None,
            "max_dd": sim.max_drawdown if sim else None,
            "etf_turnover": sim.etf_turnover if sim else None,
            "total_turnover": sim.turnover_estimate if sim else None,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    _print_candidate_table(rows, inc_report)

    # Gate cascade on the best split-passing candidate.
    selected = None
    gate_log: dict = {}
    for r in rows:
        if not r["split_passed"]:
            continue
        vec = r["vector"]
        ok_rw, rw_reasons, rw_stats = paired_random_window_gate(
            vec, incumbent, bcfg, mode=mode, scope=_SCOPE, precomp=None)
        gate_log = {"candidate": r["candidate_id"], "random_window": (ok_rw, rw_reasons)}
        if not ok_rw:
            print(f"\n  ✗ {r['candidate_id']} failed random-window gate: {rw_reasons}")
            continue
        ok_mh, mh_reasons, _ = multi_horizon_confirm(vec, incumbent, bcfg, mode=mode, scope=_SCOPE)
        gate_log["multi_horizon"] = (ok_mh, mh_reasons)
        if not ok_mh:
            print(f"\n  ✗ {r['candidate_id']} failed multi-horizon gate: {mh_reasons}")
            continue
        ok_sg, sg_reasons, _ = stress_gauntlet(vec, incumbent, bcfg, mode=mode, scope=_SCOPE, precomp=None)
        gate_log["stress_gauntlet"] = (ok_sg, sg_reasons)
        if not ok_sg:
            print(f"\n  ✗ {r['candidate_id']} failed stress gauntlet: {sg_reasons}")
            continue
        selected = r
        break

    written = False
    if selected is not None:
        print(f"\n  ✓ {selected['candidate_id']} PASSED ALL GATES")
        print(format_etf_sleeve_diagnostics(
            selected["report"].validation_result or selected["report"].train_result,
            label=selected["candidate_id"],
            current_weights=(inc_report.train_result.etf_final_weights or {}),
        ))
        if should_apply_tuned_config(apply, True, bcfg, force_apply):
            apply_etf_allocation_params(selected["vector"], provenance={
                "candidate_id": selected["candidate_id"],
                "source": selected["source"],
                "universe_mode": "configured_only",
                "val_excess_vs_spy": f"{selected['val_excess']:+.4f}",
                "n_days": n_days,
            })
            written = True
        else:
            print("\n  (--apply not set or gates require it — config NOT written)")
    else:
        print("\n  ✗ No candidate passed all gates — config unchanged.")
        _save_etf_leads(rows[:3])

    return {"rows": rows, "selected": selected, "gate_log": gate_log,
            "config_written": written, "incumbent_report": inc_report}


def _print_candidate_table(rows: list[dict], inc_report) -> None:
    print(f"\n{'=' * 96}")
    print("ETF ALLOCATION CANDIDATE TOURNAMENT")
    print(f"{'candidate':<22}{'source':<10}{'val_exc':>9}{'sharpe':>8}{'calmar':>8}"
          f"{'maxDD':>8}{'ETF_to':>8}{'split':>7}{'score':>9}")
    print("-" * 96)
    inc_sim = inc_report.validation_result or inc_report.train_result
    inc_exc = (inc_sim.total_return - (inc_report.validation_benchmark_return
               if inc_report.validation_result else inc_report.benchmark_return)) if inc_sim else 0.0
    print(f"{'INCUMBENT(equal-wt)':<22}{'incumbent':<10}{inc_exc:>+9.2%}"
          f"{(inc_sim.sharpe if inc_sim else 0):>8.2f}{(inc_sim.calmar if inc_sim else 0):>8.2f}"
          f"{(inc_sim.max_drawdown if inc_sim else 0):>8.1%}{(inc_sim.etf_turnover or 0):>8.2f}"
          f"{'—':>7}{'—':>9}")
    for r in rows:
        _ve = "n/a" if r["val_excess"] is None else f"{r['val_excess']:+.2%}"
        print(f"{r['candidate_id']:<22}{r['source']:<10}{_ve:>9}"
              f"{(r['sharpe'] or 0):>8.2f}{(r['calmar'] or 0):>8.2f}{(r['max_dd'] or 0):>8.1%}"
              f"{(r['etf_turnover'] or 0):>8.2f}{('Y' if r['split_passed'] else 'n'):>7}"
              f"{r['score']:>9.3f}")
    print("=" * 96)


def _save_etf_leads(rows: list[dict]) -> None:
    """Persist top rejected candidates' ETF vectors as .npy leads for future runs."""
    try:
        import os
        os.makedirs("reports/etf_leads", exist_ok=True)
        for i, r in enumerate(rows):
            np.save(f"reports/etf_leads/etf_lead_{i}_{r['candidate_id'].replace(':','_')}.npy",
                    r["vector"])
        print(f"  Saved {len(rows)} rejected candidate(s) as ETF leads in reports/etf_leads/")
    except Exception as exc:
        logger.debug("could not save ETF leads: %s", exc)
