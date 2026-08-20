"""
scripts/etf_seed_stability.py — is the ETF tournament winner structural, or one seed's luck?

RESEARCH ONLY — never writes config (apply=False is hardcoded).

The 2026-08-19 tune picked `random_13` out of 29 candidates: +17.05% validation excess vs
the equal-weight incumbent's +13.65%, and it cleared all six gates including a stress
gauntlet that (after the repair fix) genuinely tests the candidate. But the winner of a
29-way tournament is a MAXIMUM over noisy draws, and it is a random bucket vector with no
thesis behind it — nothing explains why 15% VXUS and 1% VNQ should be right.

So: re-run the whole tournament under different seeds. Each seed draws a different random
candidate pool, so the winning VECTOR will differ by construction. The question is whether
the winning ALLOCATIONS agree in bucket space:

  - converged  → the optimum is a real region of the allocation space; a specific vector
                 from it is defensible.
  - scattered  → each seed just surfaces its own lucky draw, and `random_13` means nothing
                 beyond "this pool's max".

Reports each seed's winner in bucket space plus the cross-seed spread per bucket.

Usage:  PYTHONPATH=src python3 scripts/etf_seed_stability.py [n_days] [seed,seed,...]
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 1250
SEEDS = ([int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [42, 7, 99, 2024])
RANDOM_TOPK = 20


def _buckets_of(vector: np.ndarray) -> dict[str, float]:
    """Normalized bullish-regime bucket weights carried by a param vector."""
    from tuning.constants import _ETF_BUCKETS, _ETF_WEIGHT_SLOT_OFFSET
    n = len(_ETF_BUCKETS)
    raw = {b: float(vector[_ETF_WEIGHT_SLOT_OFFSET + i]) for i, b in enumerate(_ETF_BUCKETS)}
    tot = sum(raw.values()) or 1.0
    return {b: w / tot for b, w in raw.items()}


def main() -> None:
    from tuning.etf_tune import run_etf_allocation_tune

    t0 = time.time()
    print(f"ETF seed-stability — {N_DAYS}d, seeds {SEEDS}, {RANDOM_TOPK} random candidates each")
    print("Each seed runs the FULL tournament + all six gates. RESEARCH ONLY.\n", flush=True)

    winners: list[tuple[int, str, float, dict]] = []
    for seed in SEEDS:
        print("=" * 88)
        print(f"=== seed {seed} ===", flush=True)
        res = run_etf_allocation_tune(
            n_days=N_DAYS, preset="etf_allocation", random_topk=RANDOM_TOPK,
            apply=False, force_apply=False, seed=seed,
        )
        sel = res.get("selected")
        if not sel:
            print(f"  seed {seed}: NO candidate passed all gates")
            winners.append((seed, "none", float("nan"), {}))
            continue
        bw = _buckets_of(sel["vector"])
        winners.append((seed, sel["candidate_id"], float(sel.get("val_excess", float("nan"))), bw))
        print(f"  seed {seed} winner: {sel['candidate_id']} "
              f"val_excess {sel.get('val_excess', float('nan')):+.4f}", flush=True)

    print("\n" + "=" * 88)
    print("CROSS-SEED WINNERS (bullish-regime bucket weights)")
    passed = [w for w in winners if w[3]]
    if not passed:
        print("  no seed produced a gate-passing winner — nothing to adopt")
        return

    buckets = sorted({b for _, _, _, bw in passed for b, w in bw.items() if w > 0.005})
    hdr = f"{'seed':>6} {'winner':<16} {'val_exc':>8}  " + "  ".join(f"{b[:11]:>11}" for b in buckets)
    print(hdr)
    for seed, cid, ve, bw in passed:
        row = f"{seed:>6} {cid:<16} {ve:>+8.4f}  " + "  ".join(f"{bw.get(b,0.0):>11.3f}" for b in buckets)
        print(row)

    print("\nPER-BUCKET SPREAD ACROSS SEEDS (low spread = structural, high = luck)")
    print(f"{'bucket':<20} {'mean':>8} {'min':>8} {'max':>8} {'range':>8}")
    verdict_ranges = []
    for b in buckets:
        vals = [bw.get(b, 0.0) for _, _, _, bw in passed]
        rng = max(vals) - min(vals)
        verdict_ranges.append(rng)
        print(f"{b:<20} {np.mean(vals):>8.3f} {min(vals):>8.3f} {max(vals):>8.3f} {rng:>8.3f}")

    worst = max(verdict_ranges) if verdict_ranges else 0.0
    print()
    if len(passed) < 2:
        print("VERDICT: only one seed produced a winner — cannot assess stability.")
    elif worst <= 0.10:
        print(f"VERDICT: CONVERGED (max bucket range {worst:.3f} <= 0.10) — the winning "
              "region looks structural, not seed luck.")
    else:
        print(f"VERDICT: SCATTERED (max bucket range {worst:.3f} > 0.10) — each seed found "
              "its own draw. A single winning vector is not evidence of an optimum.")
    print(f"\n({time.time() - t0:.0f}s) — nothing written to cfg/config.yaml")


if __name__ == "__main__":
    main()
