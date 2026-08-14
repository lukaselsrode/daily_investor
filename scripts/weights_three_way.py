"""
scripts/weights_three_way.py — head-to-head of the three score-weight regimes on ONE substrate.

RESEARCH ONLY — never writes config.

Context: the live incumbent (0.25/0.40/0.25/0.10) is a MANUAL, user-directed rebalance
from 2026-07-01 whose own config comment says "NOT yet gauntlet-validated". It replaced a
DE-tuned income-led config (2026-06-28, income 0.575) three days after that was applied,
and before it a bear-validated trend-led one (2026-06-11, momentum 0.562). The 2026-08-07
gauntlet only compared its own proposal against the incumbent — it never asked whether the
incumbent beats what it displaced.

Design: hold EVERYTHING else at the current live vector and vary only slots 0-3. The old
configs' thresholds were calibrated on the peer-1/peer-2 score scale and are invalid under
peer-3, so swapping whole historical configs would confound the weighting question with a
threshold-scale mismatch. This isolates the weights.

Caveat the numbers cannot express: peer-3 redefined what income and quality MEAN. Income was
a saturated near-binary payer flag (305 names pinned at the 1.5 cap); it is now a ranked
yield + payout-sustainability factor. Quality was liquidity + "not distressed"; it is now
statement fundamentals. So the 2026-06-28 income weight was fitted to a different factor of
the same name — this test asks whether the RATIO still holds up, not whether that tune was right.

All configs are scored on the standard matrix over the temporally-disjoint holdout, the same
surface that decided the gauntlet.

Usage:  PYTHONPATH=src python3 scripts/weights_three_way.py [n_days]
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 730

# (label, raw weights value/quality/income/momentum, provenance)
REGIMES = [
    ("incumbent (manual 2026-07-01)", (0.25, 0.40, 0.25, 0.10),
     "user-directed quality-led rebalance; never gauntlet-validated"),
    ("income-led (DE tune 2026-06-28)", (0.1354, 0.1263, 0.5747, 0.1636),
     "the deep tune the manual rebalance displaced"),
    ("trend-led (bear-validated 2026-06-11)", (0.064, 0.323, 0.051, 0.562),
     "1250d tournament + stress gauntlet winner"),
    ("gauntlet proposal (2026-08-07)", None,
     "loaded from data/gauntlet_peer3_proposed.npy"),
    # Control: if the intuited weights don't beat naive equal-weighting, the whole
    # weighting question is noise and no re-tune is worth running.
    ("equal weights (control)", (0.25, 0.25, 0.25, 0.25),
     "naive baseline — is ANY weighting choice earning its keep?"),
]


def _fmt(w) -> str:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return f"v{w[0]:.3f} q{w[1]:.3f} i{w[2]:.3f} m{w[3]:.3f}"


def main() -> None:
    from backtesting.data_loader import load_and_precompute
    from strategy.scoring.composite import SCORING_MODEL_VERSION
    from tuning.constants import _current_params
    from tuning.profiles import expand_run_matrix
    from tuning.staged_tune import validate_full_windowed

    t0 = time.time()
    confirm_matrix = expand_run_matrix("standard", "mixed")
    print(f"Three-way weight comparison — engine={SCORING_MODEL_VERSION}, {N_DAYS}d, "
          f"confirm matrix={len(confirm_matrix)} cells, scope=active_sleeve_compounding")
    print("Loading full-universe data …", flush=True)
    precomp = load_and_precompute(N_DAYS, mode=None)
    print(f"  loaded in {time.time() - t0:.0f}s\n", flush=True)

    base = _current_params().astype(float)
    results = []
    for label, weights, note in REGIMES:
        params = base.copy()
        if weights is None:
            try:
                params = np.load("data/gauntlet_peer3_proposed.npy").astype(float)
            except Exception as exc:
                print(f"  (skipping {label}: {exc})")
                continue
        else:
            params[:4] = np.asarray(weights, dtype=float)

        print(f"=== {label} ===")
        print(f"  {_fmt(params[:4])}   [{note}]", flush=True)
        v = validate_full_windowed(
            precomp, params, run_matrix=confirm_matrix,
            scope="active_sleeve_compounding", regime_scope="all",
        )
        rep = v.get("report")
        train = val = float("nan")
        if rep is not None:
            for attr, r in (("train", rep.train_result), ("val", rep.validation_result)):
                if r is None:
                    continue
                ex = (r.active_excess_return if r.active_excess_return is not None
                      else r.total_return - r.benchmark_twr)
                if attr == "train":
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
        results.append((label, _fmt(params[:4]), v.get("robust_score", 0.0), train, val,
                        bool(v.get("oos_passed"))))

    print("=" * 100)
    print(f"{'config':40s} {'weights':26s} {'robust':>8s} {'train ex':>9s} {'val ex':>8s}  gate")
    for label, w, robust, train, val, gate in sorted(results, key=lambda r: -r[2]):
        print(f"{label:40s} {w:26s} {robust:8.4f} {train:+9.4f} {val:+8.4f}  "
              f"{'pass' if gate else 'FAIL'}")
    print("=" * 100)
    best = max(results, key=lambda r: r[2]) if results else None
    if best:
        print(f"Highest robust score on the disjoint holdout: {best[0]}")
    print(f"({time.time() - t0:.0f}s)  — nothing written to cfg/config.yaml")


if __name__ == "__main__":
    main()
