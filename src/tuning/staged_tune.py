"""
tuning/staged_tune.py — staged coordinate-ascent ("Auto-tune All") + windowed validation.

`run_staged_tune` tunes a chosen set of interaction clusters in a fixed leverage order
(scoring/momentum first — they change WHAT you hold — then exits, then breadth). Each
cluster is robust-tuned ON TOP of the evolving best vector and accepted only if it
improves the robust (multi-window) score; a final DOF-bounded joint re-tune of the
accepted clusters captures residual cross-cluster gains. Per-stage DOF stays small, so
this is structurally far less overfit-prone than one giant joint tune.

`validate_full_windowed` is the confirmation step: it runs the candidate through the
out-of-sample train/val gate AND a full robust-scan across horizons/seeds on the full
universe, returning PASS/FAIL + per-window metrics + an overfit score.

Both reuse the robust objective (robust_scan), preset composition, and the OOS gate.
RESEARCH ONLY — neither writes config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fixed leverage order: scoring/momentum first (change which stocks rank high), then
# the quality tilt, the buy gate, the exit ladder, and finally breadth/turnover.
_CLUSTER_ORDER: tuple[str, ...] = (
    "active_momentum_engine",
    "active_quality_stack",
    "active_buy_gate",
    "active_exit_ladder",
    "active_breadth_turnover",
)


@dataclass
class StageResult:
    cluster: str
    score_before: float
    score_after: float
    accepted: bool


@dataclass
class StagedTuneResult:
    stages: list[StageResult] = field(default_factory=list)
    final_params: np.ndarray | None = None
    final_score: float = 0.0
    baseline_score: float = 0.0
    accepted_clusters: list[str] = field(default_factory=list)

    def trace_df(self):
        import pandas as pd
        rows = [
            {
                "stage": i + 1, "cluster": s.cluster,
                "score before": round(s.score_before, 4),
                "score after": round(s.score_after, 4),
                "Δ": round(s.score_after - s.score_before, 4),
                "result": "✅ accepted" if s.accepted else "— kept prior",
            }
            for i, s in enumerate(self.stages)
        ]
        return pd.DataFrame(rows)


def _robust_score(precomp, params, run_matrix, scope) -> float:
    from .robust_scan import run_robust_scan
    try:
        return float(run_robust_scan(
            precomp, params=params, run_matrix=run_matrix, scope=scope,
        ).overall_robust_score)
    except Exception:
        return 0.0


def run_staged_tune(
    precomp,
    clusters,
    run_matrix: list[dict],
    scope: str = "active_sleeve_compounding",
    maxiter: int = 8,
    popsize: int = 6,
    min_improve: float = 0.0,
    progress_callback=None,
) -> StagedTuneResult:
    """Staged coordinate-ascent over the selected clusters. progress_callback(done,total,label)."""
    from .constants import _current_params
    from .interaction_screen import _tune_subset

    # Run selected clusters in the fixed leverage order; unknowns appended at the end.
    ordered = [c for c in _CLUSTER_ORDER if c in clusters]
    ordered += [c for c in clusters if c not in _CLUSTER_ORDER]

    baseline = _current_params().copy()
    orig_score = _robust_score(precomp, baseline, run_matrix, scope)
    cur_score = orig_score
    out = StagedTuneResult(final_params=baseline, final_score=orig_score, baseline_score=orig_score)

    total = len(ordered) + 1  # + final joint pass
    done = 0
    for c in ordered:
        m = _tune_subset(precomp, c, run_matrix, scope, maxiter, popsize, baseline=baseline)
        done += 1
        if progress_callback:
            progress_callback(done, total, f"stage: {c}")
        if m is None:
            continue
        accepted = m.score > cur_score + min_improve
        out.stages.append(StageResult(c, cur_score, m.score, accepted))
        if accepted:
            baseline = m.params.copy()
            cur_score = m.score

    out.accepted_clusters = [s.cluster for s in out.stages if s.accepted]

    # Final DOF-bounded joint re-tune of the accepted clusters (captures cross-cluster gains).
    if len(out.accepted_clusters) >= 2:
        joint = _tune_subset(
            precomp, "+".join(out.accepted_clusters), run_matrix, scope,
            maxiter, popsize, baseline=baseline,
        )
        if joint is not None and joint.score > cur_score + min_improve:
            baseline = joint.params.copy()
            cur_score = joint.score
    done += 1
    if progress_callback:
        progress_callback(done, total, "final joint re-tune")

    out.final_params = baseline
    out.final_score = cur_score
    return out


def validate_full_windowed(
    precomp,
    params: np.ndarray,
    run_matrix: list[dict],
    scope: str = "active_sleeve_compounding",
) -> dict:
    """
    Full windowed confirmation of a candidate config: OOS train/val gate + robust scan
    across horizons/seeds on the full universe. Returns a verdict dict (never writes config).
    """
    from backtesting.simulator import run_backtest_report, split_price_window
    from util import BACKTEST_PARAMS as bp

    from .robust_scan import run_robust_scan

    out: dict = {}

    # 1. Out-of-sample train/val gate (same gate auto-tune uses before writing config).
    try:
        n = int(precomp.prices.shape[0])
        train_sl, val_sl = split_price_window(n, bp.get("train_pct", 0.70))
        report = run_backtest_report(precomp, np.asarray(params, dtype=float), train_sl, val_sl, scope=scope)
        from backtesting.validator import WalkForwardValidator
        passed, reasons = WalkForwardValidator().validate_report(report, bp)
        out["oos_passed"] = bool(passed)
        out["oos_reasons"] = list(reasons)
        out["report"] = report
    except Exception as exc:
        out["oos_passed"] = False
        out["oos_reasons"] = [f"validation error: {exc}"]

    # 2. Robust scan across the full window matrix (the "full windowed backtest confirmation").
    try:
        scan = run_robust_scan(precomp, params=np.asarray(params, dtype=float),
                               run_matrix=run_matrix, scope=scope)
        out["robust_score"] = float(scan.overall_robust_score)
        out["overfit_score"] = float(scan.overfit_warning_score())
        out["horizon_df"] = scan.horizon_heatmap_df()
        out["scan"] = scan
    except Exception as exc:
        out["robust_score"] = 0.0
        out["overfit_score"] = 1.0
        out["scan_error"] = str(exc)

    # Overall confirmation: OOS gate passes AND not strongly overfit across horizons.
    out["confirmed"] = bool(out.get("oos_passed")) and out.get("overfit_score", 1.0) <= 0.5
    return out
