"""scripts/gauntlet_compare.py — stress-gauntlet: value-led candidate vs income-led incumbent.

RESEARCH ONLY — never writes config. Runs the codebase's own multi-window robust validator
(tuning.staged_tune.validate_full_windowed — the same one `auto-tune-all` uses) on TWO fixed
configs at active_sleeve_compounding scope, with index_pct PINNED to the user's 0.90 (the 10%
"play money" sleeve is a fixed allocation choice, not a tunable). Goal: does the value-led open
optimum beat the bear-validated income-led incumbent robustly for that 10% sleeve?

Usage:  PYTHONPATH=src python3 scripts/gauntlet_compare.py [profile] [n_days]
        profile in {quick,standard,deep} (default standard); n_days default 730.
"""
from __future__ import annotations

import sys

import numpy as np

from backtesting.data_loader import load_and_precompute
from tuning.constants import PARAM_NAMES, _current_params
from tuning.profiles import expand_run_matrix
from tuning.staged_tune import validate_full_windowed

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "standard"
N_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 730
_HORIZON = {"quick": "short", "standard": "mixed", "deep": "mixed"}[PROFILE]
PINNED_INDEX_PCT = 0.90  # user constraint: keep the active sleeve at 10%, regardless.


def _norm_weights(v: np.ndarray) -> str:
    w = v[:4].astype(float)
    s = w.sum() or 1.0
    val, qual, inc, mom = (w / s)
    return f"value {val:.3f} | quality {qual:.3f} | income {inc:.3f} | momentum {mom:.3f}"


def _run(name: str, params: np.ndarray, run_matrix: list[dict], precomp) -> dict:
    v = validate_full_windowed(
        precomp, params, run_matrix=run_matrix,
        scope="active_sleeve_compounding", regime_scope="all",
    )
    print(f"\n=== {name} ===")
    print(f"  weights:   {_norm_weights(params)}")
    print(f"  index_pct: {params[4]:.4f}  metric_threshold: {params[5]:.4f}")
    print(f"  OOS gate:  {'pass' if v.get('oos_passed') else 'FAIL'} "
          f"({'; '.join(v.get('oos_reasons', [])) or 'all gates pass'})")
    print(f"  robust score: {v.get('robust_score', 0):.4f}   overfit: {v.get('overfit_score', 1):.0%}")
    if v.get("horizon_df") is not None:
        print("  per-horizon:")
        print(v["horizon_df"].to_string(index=False))
    return v


def main() -> None:
    print(f"Gauntlet compare — profile={PROFILE} (horizon={_HORIZON}), {N_DAYS}d, "
          f"scope=active_sleeve_compounding, index_pct pinned {PINNED_INDEX_PCT}")
    print("Loading full-universe data …")
    precomp = load_and_precompute(N_DAYS, mode=None)
    run_matrix = expand_run_matrix(PROFILE, _HORIZON)

    incumbent = _current_params().astype(float)
    incumbent[4] = PINNED_INDEX_PCT

    cand = np.load("data/leads_selected_365d.npy").astype(float)
    cand = cand.reshape(-1, len(PARAM_NAMES))[-1] if cand.ndim > 1 else cand
    cand = cand.copy()
    cand[4] = PINNED_INDEX_PCT  # pin: only the active-sleeve params differ from incumbent

    vi = _run("INCUMBENT (income-led, live config)", incumbent, run_matrix, precomp)
    vc = _run("CANDIDATE (value-led open optimum)", cand, run_matrix, precomp)

    ri, rc = vi.get("robust_score", 0.0), vc.get("robust_score", 0.0)
    print("\n" + "=" * 64)
    print(f"VERDICT (active-sleeve robust score, index_pct=0.90):")
    print(f"  incumbent={ri:.4f}  candidate={rc:.4f}  Δ={rc - ri:+.4f}")
    better = rc > ri and vc.get("oos_passed")
    print(f"  → {'CANDIDATE wins the gauntlet' if better else 'INCUMBENT holds — candidate does NOT clear it'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
