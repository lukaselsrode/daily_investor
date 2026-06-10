"""
tuning/staged_tune.py — staged coordinate-ascent ("Auto-tune All") + windowed validation.

`run_staged_tune` tunes a chosen set of interaction clusters in a fixed leverage order
(scoring/momentum first — they change WHAT you hold — then exits, then breadth). Each
cluster is robust-tuned ON TOP of the evolving best vector and accepted only if it
improves the robust (multi-window) score; a final DOF-bounded joint re-tune of the
accepted clusters captures residual cross-cluster gains. Per-stage DOF stays small, so
this is structurally far less overfit-prone than one giant joint tune.

`validate_full_windowed` is the confirmation step: it runs the candidate through the
out-of-sample train/val gate AND a robust-scan whose windows are DISJOINT from tuning
(terminal holdout segment + offset seeds), returning PASS/FAIL + per-window metrics +
an overfit score.

Both reuse the robust objective (robust_scan), preset composition, and the OOS gate.
RESEARCH ONLY — neither writes config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtesting.random_walk import _slice_precomp as _slice_window_precomp

# Fixed leverage order: scoring/momentum first (change which stocks rank high), then
# the quality tilt, the buy gate, the exit ladder, and finally breadth/turnover.
_CLUSTER_ORDER: tuple[str, ...] = (
    "active_momentum_engine",
    "active_quality_stack",
    "active_buy_gate",
    "active_exit_ladder",
    "active_breadth_turnover",
)

# Seed offset separating the disjoint-seed acceptance check (and the windowed
# validation) from the tuning run-matrix: same horizons/window counts, but the
# random windows are sampled independently of the ones the optimizer fit.
_VALIDATION_SEED_OFFSET = 10_000


def _shifted_run_matrix(run_matrix: list[dict], seed_offset: int = _VALIDATION_SEED_OFFSET) -> list[dict]:
    """Copy of run_matrix with every cell's seed offset — disjoint random windows."""
    return [{**cell, "seed": int(cell["seed"]) + seed_offset} for cell in run_matrix]


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


def _robust_score(precomp, params, run_matrix, scope, regime_scope: str = "all") -> float:
    from .robust_scan import run_robust_scan
    try:
        return float(run_robust_scan(
            precomp, params=params, run_matrix=run_matrix, scope=scope,
            regime_scope=regime_scope,
        ).overall_robust_score)
    except Exception:
        # Score space is higher-is-better and robust scores are routinely NEGATIVE,
        # so returning 0.0 here made a crashing config outrank every valid
        # negative-scoring config. A crash must rank below anything valid.
        return -1e6


def run_staged_tune(
    precomp,
    clusters,
    run_matrix: list[dict],
    scope: str = "active_sleeve_compounding",
    maxiter: int = 8,
    popsize: int = 6,
    min_improve: float = 0.0,
    progress_callback=None,
    regime_scope: str = "all",
    train_frac: float = 0.70,
) -> StagedTuneResult:
    """Staged coordinate-ascent over the selected clusters. progress_callback(done,total,label).

    Overfit guards:
      - Tuning only sees the FIRST `train_frac` of the history; the terminal segment is
        reserved for validate_full_windowed, whose windows are therefore temporally
        disjoint from everything the optimizer touched.
      - A stage (and the final joint pass) is accepted only when it improves the robust
        score on BOTH the tuning run-matrix AND a disjoint-seed re-evaluation of the same
        matrix (different random windows, same data). With min_improve=0.0, requiring the
        improvement to replicate across two independent window draws is the noise floor —
        previously a stage was accepted on a single window set, i.e. on seed noise.
    """
    from .constants import _current_params
    from .interaction_screen import _tune_subset

    # Restrict tuning to the leading train_frac of the history (random_window_backtest
    # has no max-start constraint, so we slice the substrate itself).
    n_days = int(precomp.prices.shape[0])
    split = int(n_days * train_frac)
    if 0 < split < n_days:
        precomp = _slice_window_precomp(precomp, slice(0, split))

    # Run selected clusters in the fixed leverage order; unknowns appended at the end.
    ordered = [c for c in _CLUSTER_ORDER if c in clusters]
    ordered += [c for c in clusters if c not in _CLUSTER_ORDER]

    check_matrix = _shifted_run_matrix(run_matrix)
    baseline = _current_params().copy()
    orig_score = _robust_score(precomp, baseline, run_matrix, scope, regime_scope)
    cur_score = orig_score
    baseline_check = _robust_score(precomp, baseline, check_matrix, scope, regime_scope)
    out = StagedTuneResult(final_params=baseline, final_score=orig_score, baseline_score=orig_score)

    def _confirmed(params) -> tuple[bool, float]:
        """Disjoint-seed re-evaluation: does the candidate also beat the baseline there?"""
        check = _robust_score(precomp, params, check_matrix, scope, regime_scope)
        return check > baseline_check + min_improve, check

    total = len(ordered) + 1  # + final joint pass
    done = 0
    for c in ordered:
        m = _tune_subset(precomp, c, run_matrix, scope, maxiter, popsize, baseline=baseline,
                         regime_scope=regime_scope)
        done += 1
        if progress_callback:
            progress_callback(done, total, f"stage: {c}")
        if m is None:
            continue
        accepted = m.score > cur_score + min_improve
        if accepted:
            accepted, cand_check = _confirmed(m.params)
        out.stages.append(StageResult(c, cur_score, m.score, accepted))
        if accepted:
            baseline = m.params.copy()
            cur_score = m.score
            baseline_check = cand_check

    out.accepted_clusters = [s.cluster for s in out.stages if s.accepted]

    # Final DOF-bounded joint re-tune of the accepted clusters (captures cross-cluster gains).
    if len(out.accepted_clusters) >= 2:
        joint = _tune_subset(
            precomp, "+".join(out.accepted_clusters), run_matrix, scope,
            maxiter, popsize, baseline=baseline, regime_scope=regime_scope,
        )
        if joint is not None and joint.score > cur_score + min_improve:
            joint_ok, joint_check = _confirmed(joint.params)
            if joint_ok:
                baseline = joint.params.copy()
                cur_score = joint.score
                baseline_check = joint_check
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
    regime_scope: str = "all",
    holdout_start_frac: float = 0.70,
) -> dict:
    """
    Full windowed confirmation of a candidate config: OOS train/val gate + robust scan
    across horizons/seeds on the full universe. Returns a verdict dict (never writes config).

    The robust-scan windows are DISJOINT from the tuning windows: they are sampled from
    the terminal (1 - holdout_start_frac) segment of the history — the segment
    run_staged_tune (train_frac=holdout_start_frac) never tunes on — with seeds offset
    from the tuning matrix. Previously this step re-ran the IDENTICAL windows the
    parameters were tuned on, so the robust/overfit verdicts were in-sample.
    """
    from backtesting.simulator import run_backtest_report, split_price_window
    from util import BACKTEST_PARAMS as bp

    from .robust_scan import run_robust_scan
    if regime_scope != "all":
        from backtesting.regime_scope import apply_regime_scope
        precomp, _ = apply_regime_scope(precomp, regime_scope)

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

    # 2. Robust scan confirmation on tuning-disjoint windows: terminal holdout segment
    # + disjoint seeds. Horizons too long for the holdout are dropped; if NONE fit, fall
    # back to full-history windows with disjoint seeds only — still different windows,
    # but drawn from the period the optimizer saw, an honest residual overlap recorded
    # in validation_note.
    n_total = int(precomp.prices.shape[0])
    split = int(n_total * holdout_start_frac)
    holdout_days = n_total - split
    val_matrix = _shifted_run_matrix(run_matrix)
    fitting = [c for c in val_matrix if int(c["horizon_days"]) + 1 <= holdout_days]
    if fitting and 0 < split < n_total:
        scan_precomp = _slice_window_precomp(precomp, slice(split, n_total))
        val_matrix = fitting
        out["validation_note"] = (
            f"validation windows drawn from the terminal holdout segment "
            f"(days {split}-{n_total}, excluded from tuning) with disjoint seeds "
            f"(+{_VALIDATION_SEED_OFFSET}); {len(fitting)}/{len(run_matrix)} matrix cells fit"
        )
    else:
        scan_precomp = precomp
        out["validation_note"] = (
            "RESIDUAL OVERLAP: no run-matrix horizon fits the terminal holdout segment "
            f"({holdout_days} days), so validation windows use the FULL history with "
            f"disjoint seeds (+{_VALIDATION_SEED_OFFSET}) — different windows than tuning, "
            "but drawn from a period the optimizer saw. Treat the verdict as weaker."
        )
    try:
        scan = run_robust_scan(scan_precomp, params=np.asarray(params, dtype=float),
                               run_matrix=val_matrix, scope=scope)
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
