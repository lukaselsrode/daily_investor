"""
scripts/gauntlet_tune_peer3.py — full-gauntlet staged tune under peer-3, sized to finish.

RESEARCH ONLY — never writes config.

Why this exists: `auto-tune-all --profile standard` uses the SAME run matrix for the
DE search and the final confirmation. On the 730d full universe that matrix is 18
entries × 10 windows = 180 backtests per objective call, so one cluster costs ~59k
full-universe backtests — a 2026-08-06 run burned 39 hours without finishing its
first cluster. The search budget is what explodes, not the confirmation.

So: search with the CHEAP matrix (quick profile, ~15 backtests/call) across all five
interaction clusters, then confirm the winner with the EXPENSIVE standard matrix on
the temporally-disjoint holdout. Search precision is second-order — the confirmation
is what decides adopt/reject, and it keeps full rigor.

Everything else matches the house rules: full survivorship-free universe, multi-window,
active_sleeve_compounding scope, excess-vs-SPY, holdout disjoint from the tuning slice.

Usage:  PYTHONPATH=src python3 scripts/gauntlet_tune_peer3.py [n_days] [clusters_csv]
        RESUME=1 ... to continue after an interruption
        MAX_SECONDS=N ... wall-clock budget per stage
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 730
CLUSTERS_ARG = sys.argv[2] if len(sys.argv) > 2 else ""
CHECKPOINT = os.environ.get("CHECKPOINT", "gauntlet_peer3")
RESUME = os.environ.get("RESUME", "").strip() in ("1", "true", "True")
MAX_SECONDS = float(os.environ["MAX_SECONDS"]) if os.environ.get("MAX_SECONDS") else None

# Cheap search matrix / expensive confirmation matrix.
SEARCH_PROFILE = ("quick", "short")
CONFIRM_PROFILE = ("standard", "mixed")
SEARCH_MAXITER, SEARCH_POPSIZE = 6, 4


def _weights(v: np.ndarray) -> str:
    w = np.asarray(v[:4], dtype=float)
    s = w.sum() or 1.0
    val, qual, inc, mom = w / s
    return f"value {val:.4f} | quality {qual:.4f} | income {inc:.4f} | momentum {mom:.4f}"


def main() -> None:
    from backtesting.data_loader import load_and_precompute
    from strategy.scoring.composite import SCORING_MODEL_VERSION
    from tuning.constants import _current_params
    from tuning.interaction_screen import DEFAULT_CLUSTERS
    from tuning.profiles import expand_run_matrix, projected_tune_cost
    from tuning.staged_tune import run_staged_tune, validate_full_windowed

    t0 = time.time()
    clusters = ([c.strip() for c in CLUSTERS_ARG.split(",") if c.strip()]
                or list(DEFAULT_CLUSTERS))
    search_matrix = expand_run_matrix(*SEARCH_PROFILE)
    confirm_matrix = expand_run_matrix(*CONFIRM_PROFILE)
    cost = projected_tune_cost(
        search_matrix, maxiter=SEARCH_MAXITER, popsize=SEARCH_POPSIZE,
        n_stages=len(clusters) + 1,
    )
    print(
        f"Gauntlet tune — engine={SCORING_MODEL_VERSION}, {N_DAYS}d, "
        f"scope=active_sleeve_compounding\n"
        f"  search matrix:  {len(search_matrix)} entries (profile {SEARCH_PROFILE[0]}), "
        f"maxiter={SEARCH_MAXITER} popsize={SEARCH_POPSIZE}\n"
        f"  confirm matrix: {len(confirm_matrix)} entries (profile {CONFIRM_PROFILE[0]})\n"
        f"  clusters: {clusters}\n"
        f"  projected search cost: {cost['sims_per_objective_call']:,} sims/call × "
        f"~{cost['evals_per_stage']} evals = ~{cost['sims_per_stage']:,}/stage, "
        f"~{cost['sims_total']:,} total\n"
        f"  checkpoint: {CHECKPOINT}{' (resuming)' if RESUME else ''}"
        + (f", max {MAX_SECONDS:.0f}s/stage" if MAX_SECONDS else ""),
        flush=True,
    )

    print("Loading full-universe data …", flush=True)
    precomp = load_and_precompute(N_DAYS, mode=None)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    stage_t = [time.time()]

    def _cb(done: int, total: int, label: str) -> None:
        now = time.time()
        print(
            f"  [{done}/{total}] {label}  (+{now - stage_t[-1]:.0f}s, "
            f"{now - t0:.0f}s total)",
            flush=True,
        )
        stage_t.append(now)

    staged = run_staged_tune(
        precomp, clusters=clusters, run_matrix=search_matrix,
        scope="active_sleeve_compounding", maxiter=SEARCH_MAXITER,
        popsize=SEARCH_POPSIZE, progress_callback=_cb, regime_scope="all",
        checkpoint=CHECKPOINT, resume=RESUME, max_seconds=MAX_SECONDS,
    )

    print("\nStaged trace:")
    print(staged.trace_df().to_string(index=False), flush=True)
    print(
        f"\nrobust score (search matrix): {staged.baseline_score:.4f} (baseline) → "
        f"{staged.final_score:.4f} (final); promoted clusters: "
        f"{staged.accepted_clusters or 'none'}"
    )
    print(f"noise band: {staged.noise_band:.4f} — a promotion must clear the largest gain "
          f"posted by a cluster that did NOT replicate")
    if staged.promotion_blocked_reason:
        print(f"\n🚫 ANTI-RATCHET BLOCKED THE PROMOTION — {staged.promotion_blocked_reason}")
    elif not staged.accepted_clusters:
        print("\nNo cluster cleared replication + the noise band — the incumbent holds.")
    print(f"promotable: {staged.promotable}")

    incumbent = _current_params().astype(float)
    print(f"\nincumbent weights: {_weights(incumbent)}")
    print(f"proposed  weights: {_weights(staged.final_params)}")

    changed = [
        (i, float(incumbent[i]), float(staged.final_params[i]))
        for i in range(min(len(incumbent), len(staged.final_params)))
        if abs(float(incumbent[i]) - float(staged.final_params[i])) > 1e-6
    ]
    if changed:
        from tuning.constants import PARAM_NAMES
        print(f"\nchanged slots ({len(changed)}):")
        for i, a, b in changed:
            name = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"slot{i}"
            print(f"  {name:38s} {a:+.4f} → {b:+.4f}")
    else:
        print("\nNo slot changed — staged tune kept the incumbent everywhere.")

    print("\nConfirming on the STANDARD matrix (holdout, disjoint from tuning) …", flush=True)
    for label, params in (("INCUMBENT", incumbent), ("PROPOSED", staged.final_params)):
        v = validate_full_windowed(
            precomp, params, run_matrix=confirm_matrix,
            scope="active_sleeve_compounding", regime_scope="all",
        )
        print(f"\n=== {label} ===")
        print(
            f"  OOS gate: {'pass' if v.get('oos_passed') else 'FAIL'} "
            f"({'; '.join(v.get('oos_reasons', [])) or 'all gates pass'})"
        )
        print(
            f"  robust={v.get('robust_score', 0):.4f}  "
            f"overfit={v.get('overfit_score', 1):.0%}  "
            f"confirmed={v.get('confirmed')}"
        )
        if v.get("horizon_df") is not None:
            print(v["horizon_df"].to_string(index=False))
        rep = v.get("report")
        if rep is not None:
            for w, r in (("train", rep.train_result), ("val", rep.validation_result)):
                if r is None:
                    continue
                ex = (r.active_excess_return if r.active_excess_return is not None
                      else r.total_return - r.benchmark_twr)
                print(f"  {w}: excess-vs-SPY {ex:+.4f}")

    np.save("data/gauntlet_peer3_proposed.npy", staged.final_params)
    print(f"\nproposed vector saved: data/gauntlet_peer3_proposed.npy  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
