"""scripts/gauntlet_peer3.py — Gate V3: engine swap no-regression (peer-2 vs peer-3).

RESEARCH ONLY — never writes config. Runs the codebase's own multi-window robust
validator (tuning.staged_tune.validate_full_windowed — the same one auto-tune-all
uses) on the INCUMBENT live params at active_sleeve_compounding scope, full
survivorship-free universe. Excess-vs-SPY is the reported metric.

The A/B is selected by PYTHONPATH: run once with the current tree (peer-3 engine)
and once with a git worktree checked out at the pre-peer-3 commit (peer-2 engine),
SAME params, SAME data. Pass bar (engine swap = no-regression, NOT beat):
mean windowed excess >= incumbent − 0.5pp and not worse in a majority of windows.

Usage:  PYTHONPATH=<tree>/src python3 scripts/gauntlet_peer3.py [label] [profile] [n_days]
"""
from __future__ import annotations

import sys

from backtesting.data_loader import load_and_precompute
from tuning.constants import _current_params
from tuning.profiles import expand_run_matrix
from tuning.staged_tune import validate_full_windowed

LABEL = sys.argv[1] if len(sys.argv) > 1 else "peer-3"
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "standard"
N_DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 730
_HORIZON = {"quick": "short", "standard": "mixed", "deep": "mixed"}[PROFILE]


def main() -> None:
    try:
        from strategy.scoring.composite import SCORING_MODEL_VERSION
    except Exception:
        SCORING_MODEL_VERSION = "unknown"
    print(f"Gauntlet peer-3 A/B — engine={SCORING_MODEL_VERSION} label={LABEL} "
          f"profile={PROFILE} horizon={_HORIZON} n_days={N_DAYS}")
    print("Loading full-universe data …", flush=True)
    precomp = load_and_precompute(N_DAYS, mode=None)
    run_matrix = expand_run_matrix(PROFILE, _HORIZON)

    incumbent = _current_params().astype(float)

    v = validate_full_windowed(
        precomp, incumbent, run_matrix=run_matrix,
        scope="active_sleeve_compounding", regime_scope="all",
    )
    print(f"\n=== {LABEL} (engine {SCORING_MODEL_VERSION}) — incumbent params ===")
    print(f"  OOS gate:  {'pass' if v.get('oos_passed') else 'FAIL'} "
          f"({'; '.join(v.get('oos_reasons', [])) or 'all gates pass'})")
    print(f"  robust score: {v.get('robust_score', 0):.4f}   overfit: {v.get('overfit_score', 1):.0%}")
    if v.get("horizon_df") is not None:
        print("  per-horizon:")
        print(v["horizon_df"].to_string(index=False))
    report = v.get("report")
    if report is not None:
        tr, vr = report.train_result, report.validation_result
        def _x(r):
            if r is None:
                return "n/a"
            ex = r.active_excess_return if r.active_excess_return is not None else r.total_return - r.benchmark_twr
            return f"excess-vs-SPY {ex:+.4f} (ret {r.total_return:+.4f}, bench_twr {r.benchmark_twr:+.4f})"
        print(f"  train: {_x(tr)}")
        print(f"  val:   {_x(vr)}")


if __name__ == "__main__":
    main()
