"""
scripts/quality_variant_gauntlet.py — system-level test of peer-3 quality component variants.

RESEARCH ONLY — never writes config.

The 2026-08-14 component study (scripts/quality_component_study.py, 204-date PIT panel) found
that neg_accruals (weight 0.15) and low_leverage (0.12) carry statistically significant
NEGATIVE incremental IC, and that a 4-component blend beat the live 7 by +0.0136 IC on
held-out dates. This script asks the only question that decides shipping: does that survive
the SYSTEM-level test?

That distinction is load-bearing in this repo. quality_low_vol_blend and momentum_residual_blend
both had genuine signal-level edge and both FAILED the multi-seed system backtest — they are
frozen at 0.0 to this day. IC is not P&L.

Design: the params vector is IDENTICAL across arms (same weights, thresholds, exits). ONLY the
quality factor's component set changes, which means the PIT factor panels must be rebuilt per
arm — quality is baked into them at precompute time, so each arm gets its own load. Every arm
is then scored by validate_full_windowed on the standard matrix over the temporally-disjoint
holdout: the same surface that rejected the gauntlet's weight proposal.

Usage:  PYTHONPATH=src python3 scripts/quality_variant_gauntlet.py [n_days]
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 730

# Live weights, minus the components the study indicted. The scorer renormalizes whatever
# it is given, so the surviving components keep their relative proportions.
_LIVE = {
    "roe_ttm": 0.22, "fcf_to_assets": 0.20, "neg_accruals": 0.15,
    "gross_margin_ttm": 0.12, "low_leverage": 0.12,
    "share_count_shrink_yoy": 0.10, "gm_trend_yoy": 0.09,
}
_DROP3 = {k: v for k, v in _LIVE.items()
          if k not in ("neg_accruals", "low_leverage", "gm_trend_yoy")}
_CORE3 = {k: v for k, v in _LIVE.items()
          if k in ("roe_ttm", "fcf_to_assets", "share_count_shrink_yoy")}

ARMS = [
    ("live 7-component (incumbent)", _LIVE,  "what is trading today"),
    ("4-component (drop the 3 dead)", _DROP3, "study H2 IC +0.0584 vs live +0.0448"),
    ("3-component profitability core", _CORE3, "study H2 IC +0.0579 — simplest arm"),
]


def main() -> None:
    from backtesting.data_loader import load_and_precompute
    from strategy.scoring.composite import SCORING_MODEL_VERSION
    from tuning.constants import _current_params
    from tuning.profiles import expand_run_matrix
    from tuning.staged_tune import validate_full_windowed
    from util import SCORING_PARAMS

    t0 = time.time()
    confirm_matrix = expand_run_matrix("standard", "mixed")
    params = _current_params().astype(float)
    print(f"Quality-variant gauntlet — engine={SCORING_MODEL_VERSION}, {N_DAYS}d, "
          f"confirm matrix={len(confirm_matrix)} cells, scope=active_sleeve_compounding")
    print("Params vector is IDENTICAL across arms — only the quality component set differs.\n",
          flush=True)

    results = []
    for label, comps, note in ARMS:
        print("=" * 92)
        print(f"=== {label} ===")
        print(f"  components: {', '.join(comps)}")
        print(f"  [{note}]", flush=True)

        # Mutate in place so every consumer (pit_precompute passes SCORING_PARAMS by
        # reference into build_pit_factor_panels) sees the same dict.
        SCORING_PARAMS["quality_components"].clear()
        SCORING_PARAMS["quality_components"].update(comps)

        t_arm = time.time()
        precomp = load_and_precompute(N_DAYS, mode=None)
        print(f"  precomputed in {time.time() - t_arm:.0f}s (PIT panels rebuilt for this arm)",
              flush=True)

        v = validate_full_windowed(
            precomp, params, run_matrix=confirm_matrix,
            scope="active_sleeve_compounding", regime_scope="all",
        )
        rep = v.get("report")
        train = val = float("nan")
        if rep is not None:
            for which, r in (("train", rep.train_result), ("val", rep.validation_result)):
                if r is None:
                    continue
                ex = (r.active_excess_return if r.active_excess_return is not None
                      else r.total_return - r.benchmark_twr)
                if which == "train":
                    train = ex
                else:
                    val = ex
        print(f"  OOS gate: {'pass' if v.get('oos_passed') else 'FAIL'} "
              f"({'; '.join(v.get('oos_reasons', [])) or 'all gates pass'})")
        print(f"  robust={v.get('robust_score', 0):.4f}  overfit={v.get('overfit_score', 1):.0%}  "
              f"train excess={train:+.4f}  val excess={val:+.4f}")
        if v.get("horizon_df") is not None:
            print(v["horizon_df"].to_string(index=False))
        print(flush=True)
        results.append((label, len(comps), v.get("robust_score", 0.0), train, val,
                        bool(v.get("oos_passed"))))

    # restore the live set so an interrupted run cannot leave the process skewed
    SCORING_PARAMS["quality_components"].clear()
    SCORING_PARAMS["quality_components"].update(_LIVE)

    print("=" * 92)
    print(f"{'arm':34s} {'n':>2s} {'robust':>8s} {'train ex':>9s} {'val ex':>8s}  gate")
    for label, n, robust, train, val, gate in sorted(results, key=lambda r: -r[2]):
        print(f"{label:34s} {n:2d} {robust:8.4f} {train:+9.4f} {val:+8.4f}  "
              f"{'pass' if gate else 'FAIL'}")
    print("=" * 92)
    if results:
        base = next((r for r in results if r[0].startswith("live 7")), None)
        best = max(results, key=lambda r: r[2])
        print(f"Best on the disjoint holdout: {best[0]}")
        if base and best[0] != base[0]:
            print(f"  vs incumbent: robust {base[2]:.4f} -> {best[2]:.4f} "
                  f"({best[2] - base[2]:+.4f}), val excess {base[4]:+.4f} -> {best[4]:+.4f} "
                  f"({best[4] - base[4]:+.4f})")
            print("  ADOPT only if it BEATS the incumbent here — a tie is not a reason to change "
                  "a live factor.")
        elif base:
            print("  The incumbent holds — the IC gain did not survive the system test "
                  "(the quality_low_vol_blend pattern).")
    print(f"\n({time.time() - t0:.0f}s) — nothing written to cfg/config.yaml")


if __name__ == "__main__":
    main()
