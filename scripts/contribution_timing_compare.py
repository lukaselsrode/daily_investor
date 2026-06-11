"""
scripts/contribution_timing_compare.py — A/B/C contribution-timing comparison.

A: flat $400/week baseline
B: contribution-timing overlay, config defaults
C: tuned overlay (MWR-aware random search over the 8 preset slots)

Windows: trailing 90/180/365/730 trading days from one 730d survivorship-free
full-universe load, plus N random 120d windows (validation, unseen by the tune).

Because the overlay changes contribution TIMING, TWR barely moves by design —
the decision metrics are ending portfolio value and money-weighted return (IRR)
at (near-)equal total contributed. The benchmark inside each sim receives the
SAME schedule as the strategy, so excess-vs-SPY isolates stock selection.

The "hit rate" evaluates timing skill with FUTURE data (evaluation only, never
fed back into decisions): the fraction of above-base weeks whose next-20-day
benchmark return beats the all-weeks average.

Guardrails for adoption (spec): budget adherence, ending value >= flat, max
drawdown not materially worse, random-window robustness, cash-drag bound.
"""

import argparse
import copy
import json
import sys

sys.path.insert(0, "src")

import numpy as np

from backtesting.data_loader import load_and_precompute
from backtesting.regime_scope import regime_labels, slice_precomp
from backtesting.simulator import get_default_params, run_simulation
from tuning.constants import _CT_FIELDS, _CT_SLOT_OFFSET, _current_params
from util import CONTRIBUTION_TIMING_PARAMS

BASE_WEEKLY = 400.0
START_CAP = 10_000.0
REBAL = 5
WINDOWS = [90, 180, 365, 730]
N_RANDOM_WINDOWS = 16
RANDOM_WINDOW_DAYS = 120
RANDOM_SEED = 1337           # validation windows — never used for tuning
TUNE_SAMPLES = 48
TUNE_SEED = 7


# ---------------------------------------------------------------------------
# Money-weighted return (annualized IRR on the daily flow schedule)
# ---------------------------------------------------------------------------

def annualized_mwr(flows: list[tuple[int, float]], final_day: int, final_value: float) -> float:
    """Solve the daily IRR for NPV(flows) + final_value discounted = 0 via bisection.
    flows: (day, amount_invested) — investments positive. Returns annualized rate."""
    def npv(daily_r: float) -> float:
        total = -final_value / (1.0 + daily_r) ** final_day
        for day, amt in flows:
            total += amt / (1.0 + daily_r) ** day
        return total

    lo, hi = -0.05, 0.05  # ±~1e4x annualized — far beyond plausible
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    daily = (lo + hi) / 2.0
    return (1.0 + daily) ** 252 - 1.0


def run_variant(pc, params, label: str) -> dict:
    """One sim + the full metric block for a variant on a window."""
    res = run_simulation(
        pc, params, START_CAP,
        slippage_bps=10.0, weekly_contribution=BASE_WEEKLY,
        rebalance_frequency_days=REBAL,
    )
    n_days = pc.prices.shape[0]
    ct = res.contribution_timing
    if ct:
        sched = {row["day"]: row["contribution"] for row in ct["schedule"]}
    else:
        sched = {d: BASE_WEEKLY for d in range(n_days) if d > 0 and d % REBAL == 0}
    flows = [(0, START_CAP)] + sorted(sched.items())
    mwr = annualized_mwr(flows, n_days - 1, res.final_value)

    # Timing hit rate (evaluation-only future data): above-base weeks whose
    # next-20d benchmark return beats the all-weeks average forward return.
    bench = pc.benchmark_prices
    fwd = {}
    for d in sched:
        if d + 20 < n_days and bench[d] > 0:
            fwd[d] = bench[d + 20] / bench[d] - 1.0
    hit_rate = float("nan")
    if fwd:
        avg_fwd = float(np.mean(list(fwd.values())))
        above = [d for d in fwd if sched[d] > BASE_WEEKLY + 0.01]
        if above:
            hit_rate = float(np.mean([fwd[d] > avg_fwd for d in above]))

    amts = np.array(list(sched.values()))
    return {
        "label": label,
        "final_value": res.final_value,
        "total_contributed": float(amts.sum()),
        "twr": res.total_return,
        "bench_twr": res.benchmark_twr,
        "excess_twr": res.total_return - res.benchmark_twr,
        "mwr": mwr,
        "sharpe": res.sharpe,
        "calmar": res.calmar,
        "max_drawdown": res.max_drawdown,
        "avg_weekly": float(amts.mean()),
        "min_weekly": float(amts.min()),
        "max_weekly": float(amts.max()),
        "pct_above": float((amts > BASE_WEEKLY + 0.01).mean()),
        "pct_below": float((amts < BASE_WEEKLY - 0.01).mean()),
        "hit_rate": hit_rate,
        "weeks": len(amts),
    }


def set_overlay(enabled: bool) -> None:
    CONTRIBUTION_TIMING_PARAMS["enabled"] = enabled


def tuned_params(sample: dict) -> np.ndarray:
    """Full param vector with the 8 CT slots set from a tune sample."""
    p = _current_params().copy()
    for i, (_, path, _) in enumerate(_CT_FIELDS):
        field = path.split(".")[-1]
        p[_CT_SLOT_OFFSET + i] = sample[field]
    return p


def sample_ct(rng: np.random.Generator) -> dict:
    """One random tune sample within the preset bounds; weights renormalized
    downstream so their scale is irrelevant — sample uniform then normalize for
    readability."""
    s = {
        "dip_sensitivity":   rng.uniform(0.0, 3.0),
        "neutral_dip_score": rng.uniform(0.1, 0.7),
        "min_multiplier":    rng.uniform(0.25, 1.0),
        "max_multiplier":    rng.uniform(1.0, 3.0),
    }
    w = rng.uniform(0.05, 1.0, 4)
    w = w / w.sum() * 0.85  # leave the configured ma-gap weights (0.15 total) in place
    s.update({
        "return_1w": w[0], "return_1m": w[1],
        "drawdown_20d": w[2], "drawdown_60d": w[3],
    })
    return s


def apply_dip_only_cfg(sensitivity: float) -> None:
    """Mutate the live overlay config into the dip-only acceleration shape:
    always contribute base, ADD up to +50% ($200 on a $400 base) on dips, never
    reduce below baseline. The defensive-regime cap (1.25x) still applies."""
    CONTRIBUTION_TIMING_PARAMS["enabled"] = True
    CONTRIBUTION_TIMING_PARAMS["allow_budget_acceleration"] = True
    CONTRIBUTION_TIMING_PARAMS["borrow_from_future_weeks"] = True
    CONTRIBUTION_TIMING_PARAMS["max_weekly_contribution"] = BASE_WEEKLY * 1.5
    m = CONTRIBUTION_TIMING_PARAMS["multiplier"]
    m["min_multiplier"] = 1.0
    m["max_multiplier"] = 1.5
    m["dip_sensitivity"] = sensitivity
    m["neutral_dip_score"] = 0.35
    m["smoothing_alpha"] = 0.50


def run_dip_only(window_pcs: dict, full, n_full: int, out_path: str) -> None:
    """Variant D (dip-only acceleration) vs A (flat $400) and A' (flat at D's
    realized average weekly — equal total capital, evenly spread). D wins on
    TIMING only if it beats A' at the same total spend; MWR vs A measures
    per-dollar efficiency of the extra deployment."""
    import os

    flat_params = get_default_params()
    saved = copy.deepcopy(CONTRIBUTION_TIMING_PARAMS)
    results: dict = {"mode": "dip_only_acceleration", "sweep": {}, "windows": {}, "random_windows": []}

    # ── Sensitivity sweep on the deterministic windows (select by mean MWR edge vs A) ──
    sweep_scores = {}
    set_overlay(False)
    a_runs = {w: run_variant(pc, flat_params, "A_flat") for w, pc in window_pcs.items()}
    for sens in (0.5, 1.0, 1.5, 2.5):
        apply_dip_only_cfg(sens)
        edges = []
        for w, pc in window_pcs.items():
            r = run_variant(pc, flat_params, f"D_s{sens}")
            if not (np.isnan(r["mwr"]) or np.isnan(a_runs[w]["mwr"])):
                edges.append(r["mwr"] - a_runs[w]["mwr"])
        sweep_scores[sens] = float(np.mean(edges)) if edges else float("-inf")
        print(f"  sensitivity {sens}: mean MWR edge vs A {sweep_scores[sens]:+.3%}", flush=True)
    best_sens = max(sweep_scores, key=sweep_scores.get)
    results["sweep"] = {"scores": sweep_scores, "selected": best_sens}
    print(f"Selected dip_sensitivity={best_sens}", flush=True)

    # ── Full A / A' / D comparison per window ─────────────────────────────────
    for w, pc in window_pcs.items():
        apply_dip_only_cfg(best_sens)
        d = run_variant(pc, flat_params, "D_dip_only")
        set_overlay(False)
        a = a_runs[w]
        # A': flat schedule at D's realized average weekly — same total capital.
        a_prime = run_variant_flat_amount(pc, flat_params, d["avg_weekly"])
        results["windows"][w] = {"A": a, "A_prime": a_prime, "D": d}
        print(
            f"[{w:>3}d] A ${a['final_value']:>9,.0f} (MWR {a['mwr']:+.2%}) | "
            f"A' ${a_prime['final_value']:>9,.0f} @ ${d['avg_weekly']:,.0f}/wk | "
            f"D ${d['final_value']:>9,.0f} (MWR {d['mwr']:+.2%}, extra ${d['total_contributed']-a['total_contributed']:,.0f})",
            flush=True,
        )

    # ── Random-window validation ──────────────────────────────────────────────
    rng_w = np.random.default_rng(RANDOM_SEED)
    eligible = n_full - RANDOM_WINDOW_DAYS - 1
    starts = sorted(rng_w.choice(eligible, size=min(N_RANDOM_WINDOWS, eligible), replace=False).tolist())
    for s0 in starts:
        pc = slice_precomp(full, slice(s0, s0 + RANDOM_WINDOW_DAYS))
        apply_dip_only_cfg(best_sens)
        d = run_variant(pc, flat_params, "D")
        set_overlay(False)
        a = run_variant(pc, flat_params, "A")
        a_prime = run_variant_flat_amount(pc, flat_params, d["avg_weekly"])
        results["random_windows"].append({"start": int(s0), "A": a, "A_prime": a_prime, "D": d})

    # restore process config
    CONTRIBUTION_TIMING_PARAMS.clear()
    CONTRIBUTION_TIMING_PARAMS.update(saved)

    results["verdict"] = evaluate_dip_only_guardrails(results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nFull results → {out_path}")
    print_dip_only_report(results)


def run_variant_flat_amount(pc, params, weekly: float) -> dict:
    """Flat-contribution run at an arbitrary weekly amount (overlay disabled)."""
    res = run_simulation(
        pc, params, START_CAP,
        slippage_bps=10.0, weekly_contribution=weekly,
        rebalance_frequency_days=REBAL,
    )
    n_days = pc.prices.shape[0]
    sched = {d: weekly for d in range(n_days) if d > 0 and d % REBAL == 0}
    flows = [(0, START_CAP)] + sorted(sched.items())
    amts = np.array(list(sched.values()))
    return {
        "label": f"A_prime_{weekly:.0f}",
        "final_value": res.final_value,
        "total_contributed": float(amts.sum()),
        "twr": res.total_return,
        "bench_twr": res.benchmark_twr,
        "excess_twr": res.total_return - res.benchmark_twr,
        "mwr": annualized_mwr(flows, n_days - 1, res.final_value),
        "sharpe": res.sharpe,
        "calmar": res.calmar,
        "max_drawdown": res.max_drawdown,
        "avg_weekly": weekly,
        "min_weekly": weekly,
        "max_weekly": weekly,
        "pct_above": 0.0,
        "pct_below": 0.0,
        "hit_rate": float("nan"),
        "weeks": len(amts),
    }


def evaluate_dip_only_guardrails(results: dict) -> dict:
    """Adoption rules for the dip-only shape: the extra dollars must EARN their
    keep (MWR vs A), the timing must beat equal-capital-spread-evenly (D vs A'),
    drawdown must hold, and the budget overshoot must stay bounded."""
    win = results["windows"]
    checks: dict[str, bool] = {}
    notes: list[str] = []

    # 1. Per-dollar efficiency: MWR(D) >= MWR(A) on a majority of windows.
    mwr_wins = sum(win[w]["D"]["mwr"] >= win[w]["A"]["mwr"] for w in win
                   if not (np.isnan(win[w]["D"]["mwr"]) or np.isnan(win[w]["A"]["mwr"])))
    checks["mwr_vs_flat"] = mwr_wins >= (len(win) + 1) // 2
    notes.append(f"MWR >= flat on {mwr_wins}/{len(win)} deterministic windows")
    # 2. Timing value: D beats A' (same total capital) on ending value, majority of windows.
    ev_wins = sum(win[w]["D"]["final_value"] > win[w]["A_prime"]["final_value"] for w in win)
    checks["timing_vs_equal_capital"] = ev_wins >= (len(win) + 1) // 2
    notes.append(f"ending value > equal-capital control on {ev_wins}/{len(win)} windows")
    # 3. Drawdown: no window worse than flat by more than 1pp.
    checks["drawdown"] = all(win[w]["D"]["max_drawdown"] >= win[w]["A"]["max_drawdown"] - 0.01 for w in win)
    # 4. Budget overshoot bounded: extra spend <= 25% of flat total on every window.
    checks["bounded_overshoot"] = all(
        win[w]["D"]["total_contributed"] - win[w]["A"]["total_contributed"]
        <= 0.25 * win[w]["A"]["total_contributed"] + 1e-6 for w in win
    )
    for w in win:
        extra = win[w]["D"]["total_contributed"] - win[w]["A"]["total_contributed"]
        notes.append(f"{w}d extra spend: ${extra:,.0f} ({extra / max(win[w]['A']['total_contributed'],1):.1%} over budget)")
    # 5. Random-window robustness of the timing claim: D > A' on >= 55% of windows.
    rw = results.get("random_windows", [])
    if rw:
        wins = sum(r["D"]["final_value"] > r["A_prime"]["final_value"] for r in rw)
        rate = wins / len(rw)
        checks["random_window_timing"] = rate >= 0.55
        notes.append(f"random-window D>A' rate: {rate:.0%} ({wins}/{len(rw)})")
    else:
        checks["random_window_timing"] = False

    return {"adopt": all(checks.values()), "checks": checks, "notes": notes}


def print_dip_only_report(results: dict) -> None:
    print("\n" + "=" * 100)
    print("DIP-ONLY ACCELERATION — A (flat $400) vs A' (flat @ D's avg) vs D (base + dip extra, never below)")
    print("=" * 100)
    hdr = (f"{'win':>5} {'var':>7} {'final $':>10} {'contrib $':>10} {'MWR':>8} {'TWR':>8} "
           f"{'maxDD':>7} {'avg wk':>7} {'max wk':>6} {'>base':>6} {'hit':>5}")
    print(hdr)
    for w in sorted(results["windows"]):
        for v in ("A", "A_prime", "D"):
            r = results["windows"][w][v]
            hit = f"{r['hit_rate']:.0%}" if not np.isnan(r["hit_rate"]) else "  n/a"
            print(f"{w:>5} {v:>7} {r['final_value']:>10,.0f} {r['total_contributed']:>10,.0f} "
                  f"{r['mwr']:>+8.2%} {r['twr']:>+8.2%} {r['max_drawdown']:>7.1%} "
                  f"{r['avg_weekly']:>7.0f} {r['max_weekly']:>6.0f} {r['pct_above']:>6.0%} {hit:>5}")
    print(f"\nSweep: {results['sweep']['scores']}  → selected sensitivity {results['sweep']['selected']}")
    v = results["verdict"]
    print("\nGUARDRAILS (D):")
    for k, ok in v["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    for n in v["notes"]:
        print(f"        {n}")
    print(f"\nVERDICT: {'ADOPT dip-only acceleration' if v['adopt'] else 'DO NOT ADOPT — guardrails not met'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--samples", type=int, default=TUNE_SAMPLES)
    ap.add_argument("--mode", choices=["budget_shift", "dip_only"], default="budget_shift")
    ap.add_argument("--out", default="reports/contribution_timing_compare.json")
    args = ap.parse_args()

    print(f"Loading {args.days}d precomp …", flush=True)
    full = load_and_precompute(args.days)
    n_full = full.prices.shape[0]
    # Attach PIT regime labels once so every slice keeps real labels (defensive cap live).
    if full.regime_labels_daily is None:
        full = full._replace(regime_labels_daily=regime_labels(full))

    window_pcs = {}
    for w in WINDOWS:
        if w <= n_full:
            window_pcs[w] = slice_precomp(full, slice(n_full - w, n_full)) if w < n_full else full
    print(f"Windows available: {sorted(window_pcs)}", flush=True)

    if args.mode == "dip_only":
        run_dip_only(window_pcs, full, n_full, args.out)
        return

    flat_params = get_default_params()

    # ── A + B on every deterministic window ──────────────────────────────────
    results: dict = {"windows": {}, "random_windows": {}, "tune": {}}
    for w, pc in window_pcs.items():
        set_overlay(False)
        a = run_variant(pc, flat_params, "A_flat")
        set_overlay(True)
        b = run_variant(pc, flat_params, "B_default_overlay")
        results["windows"][w] = {"A": a, "B": b}
        print(f"[{w:>3}d] A final ${a['final_value']:>9,.0f} (contrib ${a['total_contributed']:,.0f}, MWR {a['mwr']:+.2%}) | "
              f"B final ${b['final_value']:>9,.0f} (contrib ${b['total_contributed']:,.0f}, MWR {b['mwr']:+.2%})", flush=True)

    # ── Tune (variant C): random search, objective = mean MWR edge vs flat ───
    print(f"\nTuning: {args.samples} samples × {len(window_pcs)} windows (objective: mean MWR edge vs A at comparable budget) …", flush=True)
    rng = np.random.default_rng(TUNE_SEED)
    set_overlay(True)
    best = None
    flat_mwr = {w: results["windows"][w]["A"]["mwr"] for w in window_pcs}
    flat_contrib = {w: results["windows"][w]["A"]["total_contributed"] for w in window_pcs}
    for i in range(args.samples):
        s = sample_ct(rng)
        p = tuned_params(s)
        edges, budget_ok = [], True
        for w, pc in window_pcs.items():
            r = run_variant(pc, p, "cand")
            if np.isnan(r["mwr"]) or np.isnan(flat_mwr[w]):
                continue
            edges.append(r["mwr"] - flat_mwr[w])
            # Budget adherence INSIDE the objective: candidates that starve or
            # overshoot the flat budget by >15% are rejected outright.
            if abs(r["total_contributed"] - flat_contrib[w]) > 0.15 * flat_contrib[w]:
                budget_ok = False
        score = float(np.mean(edges)) if edges else float("-inf")
        if budget_ok and (best is None or score > best["score"]):
            best = {"score": score, "sample": s}
            print(f"  [{i+1:>2}/{args.samples}] new best mean MWR edge {score:+.3%}  {json.dumps({k: round(v,3) for k,v in s.items()})}", flush=True)
    results["tune"] = best or {}

    # ── C on every window ─────────────────────────────────────────────────────
    if best:
        p_best = tuned_params(best["sample"])
        for w, pc in window_pcs.items():
            set_overlay(True)
            results["windows"][w]["C"] = run_variant(pc, p_best, "C_tuned_overlay")

    # ── Random-window validation (seed never seen by the tune) ───────────────
    rng_w = np.random.default_rng(RANDOM_SEED)
    eligible = n_full - RANDOM_WINDOW_DAYS - 1
    starts = sorted(rng_w.choice(eligible, size=min(N_RANDOM_WINDOWS, eligible), replace=False).tolist())
    rows = []
    for s0 in starts:
        pc = slice_precomp(full, slice(s0, s0 + RANDOM_WINDOW_DAYS))
        set_overlay(False)
        a = run_variant(pc, flat_params, "A")
        set_overlay(True)
        b = run_variant(pc, flat_params, "B")
        row = {"start": int(s0), "A": a, "B": b}
        if best:
            row["C"] = run_variant(pc, tuned_params(best["sample"]), "C")
        rows.append(row)
    results["random_windows"] = rows
    set_overlay(False)  # leave the process the way we found it

    # ── Guardrail verdict ─────────────────────────────────────────────────────
    verdict = evaluate_guardrails(results)
    results["verdict"] = verdict

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nFull results → {args.out}")
    print_report(results)


def evaluate_guardrails(results: dict) -> dict:
    """The spec's adoption rules, mechanically applied to variant C (fallback B)."""
    checks: dict[str, bool] = {}
    notes: list[str] = []
    variant = "C" if any("C" in v for v in results["windows"].values()) else "B"

    win = results["windows"]
    have = [w for w in win if variant in win[w]]
    if not have:
        return {"adopt": False, "variant": variant, "checks": {}, "notes": ["no variant runs"]}

    # 1. Budget adherence: total contributed within ±15% of flat on every window.
    checks["budget_adherence"] = all(
        abs(win[w][variant]["total_contributed"] - win[w]["A"]["total_contributed"])
        <= 0.15 * win[w]["A"]["total_contributed"] for w in have
    )
    # 2. Ending value: strictly better than flat on a majority of windows, not worse overall.
    ev_edges = [win[w][variant]["final_value"] - win[w]["A"]["final_value"] for w in have]
    checks["ending_value"] = sum(e > 0 for e in ev_edges) >= (len(ev_edges) + 1) // 2 and sum(ev_edges) > 0
    # 3. MWR: mean improvement positive.
    mwr_edges = [win[w][variant]["mwr"] - win[w]["A"]["mwr"] for w in have
                 if not (np.isnan(win[w][variant]["mwr"]) or np.isnan(win[w]["A"]["mwr"]))]
    checks["mwr_improvement"] = bool(mwr_edges) and float(np.mean(mwr_edges)) > 0
    # 4. Drawdown: no window worse by more than 1pp.
    checks["drawdown"] = all(
        win[w][variant]["max_drawdown"] >= win[w]["A"]["max_drawdown"] - 0.01 for w in have
    )
    # 5. Random-window robustness: variant beats flat on ending value in >= 55% of windows.
    rw = results.get("random_windows", [])
    rw_have = [r for r in rw if variant in r]
    if rw_have:
        wins = sum(r[variant]["final_value"] > r["A"]["final_value"] for r in rw_have)
        rate = wins / len(rw_have)
        checks["random_window_robustness"] = rate >= 0.55
        notes.append(f"random-window ending-value win rate: {rate:.0%} ({wins}/{len(rw_have)})")
    else:
        checks["random_window_robustness"] = False
    # 6. Cash drag: average weekly contribution not below 85% of base on any window.
    checks["cash_drag"] = all(win[w][variant]["avg_weekly"] >= 0.85 * BASE_WEEKLY for w in have)

    return {"adopt": all(checks.values()), "variant": variant, "checks": checks, "notes": notes}


def print_report(results: dict) -> None:
    print("\n" + "=" * 100)
    print("CONTRIBUTION TIMING — A (flat) vs B (default overlay) vs C (tuned overlay)")
    print("=" * 100)
    hdr = (f"{'win':>5} {'var':>3} {'final $':>10} {'contrib $':>10} {'MWR':>8} {'TWR':>8} "
           f"{'exc-TWR':>8} {'maxDD':>7} {'avg wk':>7} {'min':>5} {'max':>5} {'>base':>6} {'<base':>6} {'hit':>5}")
    print(hdr)
    for w in sorted(results["windows"]):
        for v in ("A", "B", "C"):
            r = results["windows"][w].get(v)
            if not r:
                continue
            hit = f"{r['hit_rate']:.0%}" if not np.isnan(r["hit_rate"]) else "  n/a"
            print(f"{w:>5} {v:>3} {r['final_value']:>10,.0f} {r['total_contributed']:>10,.0f} "
                  f"{r['mwr']:>+8.2%} {r['twr']:>+8.2%} {r['excess_twr']:>+8.2%} {r['max_drawdown']:>7.1%} "
                  f"{r['avg_weekly']:>7.0f} {r['min_weekly']:>5.0f} {r['max_weekly']:>5.0f} "
                  f"{r['pct_above']:>6.0%} {r['pct_below']:>6.0%} {hit:>5}")
    if results.get("tune"):
        print(f"\nTuned sample (mean MWR edge {results['tune']['score']:+.3%}):")
        print(f"  {json.dumps({k: round(v, 4) for k, v in results['tune']['sample'].items()})}")
    v = results["verdict"]
    print(f"\nGUARDRAILS ({v['variant']}):")
    for k, ok in v["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    for n in v["notes"]:
        print(f"        {n}")
    print(f"\nVERDICT: {'ADOPT — tuned overlay improves MWR/ending value within guardrails' if v['adopt'] else 'DO NOT ADOPT — guardrails not met'}")


if __name__ == "__main__":
    main()
