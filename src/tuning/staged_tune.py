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

import logging
from dataclasses import dataclass, field

import numpy as np

from backtesting.random_walk import _slice_precomp as _slice_window_precomp
from backtesting.types import BacktestScope

logger = logging.getLogger(__name__)

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
    # Tier 3 diagnostics: the raw gain this stage proposed, and whether it cleared the
    # noise band measured from the stages that did NOT win.
    delta: float = 0.0
    noise_band: float = 0.0
    cleared_band: bool = True


@dataclass
class StagedTuneResult:
    stages: list[StageResult] = field(default_factory=list)
    final_params: np.ndarray | None = None
    final_score: float = 0.0
    baseline_score: float = 0.0
    accepted_clusters: list[str] = field(default_factory=list)
    # Tier 3: the largest gain posted by a cluster that did NOT replicate — i.e. how big
    # "nothing" measures on this substrate. A promoted cluster must clear it.
    noise_band: float = 0.0
    # Anti-ratchet: the final vector re-scored against the ORIGINAL incumbent on the
    # disjoint-seed matrix. None when the run made no change (nothing to check).
    original_score: float | None = None
    final_vs_original: float | None = None
    beats_original: bool = True
    promotion_blocked_reason: str = ""

    def trace_df(self):
        import pandas as pd
        rows = [
            {
                "stage": i + 1, "cluster": s.cluster,
                "score before": round(s.score_before, 4),
                "score after": round(s.score_after, 4),
                "Δ": round(s.score_after - s.score_before, 4),
                "noise band": round(s.noise_band, 4),
                "result": (
                    "✅ accepted" if s.accepted
                    else ("✗ inside noise band" if not s.cleared_band else "— kept prior")
                ),
            }
            for i, s in enumerate(self.stages)
        ]
        return pd.DataFrame(rows)

    @property
    def promotable(self) -> bool:
        """True when this run produced a change that cleared every gate."""
        return bool(self.accepted_clusters) and self.beats_original


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
    scope: BacktestScope = "active_sleeve_compounding",
    maxiter: int = 8,
    popsize: int = 6,
    min_improve: float = 0.0,
    progress_callback=None,
    regime_scope: str = "all",
    train_frac: float = 0.70,
    checkpoint: str | None = None,
    resume: bool = False,
    max_seconds: float | None = None,
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

    Durability (`checkpoint` = run id under data/tune_checkpoints/):
      - State is written after every stage and after the final joint pass, plus after
        each DE generation WITHIN a stage — a standard-profile cluster can outlast a
        day on its own, so between-stage checkpoints alone would not have saved the
        39-hour run that motivated this.
      - `resume=True` restarts from the last completed stage; the in-flight stage warm
        starts from its saved best-so-far. A checkpoint from a different code revision
        or run configuration RAISES (see tuning.checkpoint) rather than silently
        resuming onto a param layout its indices no longer describe.
      - `max_seconds` bounds each stage's DE, stopping it cleanly with its checkpoint
        intact instead of requiring a kill.
    """
    import time

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

    ckpt = None
    ckpt_mod = None
    identity = None
    if checkpoint:
        from . import checkpoint as ckpt_mod
        identity = ckpt_mod.run_identity(
            run_matrix=run_matrix, clusters=ordered, scope=str(scope),
            regime_scope=regime_scope, maxiter=maxiter, popsize=popsize,
            train_frac=train_frac, baseline=baseline,
        )
        if resume:
            ckpt = ckpt_mod.load(checkpoint, expected_identity=identity)
        if ckpt is None:
            ckpt = ckpt_mod.new_checkpoint(checkpoint, identity, baseline)

    out = StagedTuneResult(final_params=baseline, final_score=0.0, baseline_score=0.0)

    # The ORIGINAL incumbent is frozen for the whole run: every cluster is a marginal
    # against IT, never against an evolving vector (see the docstring's path-bias note),
    # and the anti-ratchet check at the end re-scores against it too.
    original = baseline.copy()
    marginals: dict[str, dict] = {}

    resumed_stages = 0
    if ckpt is not None and ckpt.stage_index > 0:
        orig_score = ckpt.orig_score
        baseline_check = ckpt.baseline_check
        marginals = dict(ckpt.marginals or {})
        resumed_stages = int(ckpt.stage_index)
        logger.info(
            "staged tune: resuming '%s' after %d completed marginal(s) (updated %s)",
            checkpoint, resumed_stages, ckpt.updated_at,
        )
    else:
        orig_score = _robust_score(precomp, original, run_matrix, scope, regime_scope)
        baseline_check = _robust_score(precomp, original, check_matrix, scope, regime_scope)
    out.baseline_score = orig_score
    out.final_score = orig_score

    def _persist(*, stage_index: int, de_state: dict | None = None,
                 joint_done: bool = False) -> None:
        if ckpt is None:
            return
        ckpt.stage_index = stage_index
        ckpt.baseline = [float(v) for v in np.asarray(original).ravel()]
        ckpt.cur_score = float(out.final_score)
        ckpt.baseline_check = float(baseline_check)
        ckpt.orig_score = float(orig_score)
        ckpt.stages = [vars(s).copy() for s in out.stages]
        ckpt.marginals = marginals
        ckpt.de_state = de_state
        ckpt.joint_done = joint_done
        ckpt_mod.save(ckpt)

    def _replicates(params) -> tuple[bool, float]:
        """Disjoint-seed re-evaluation: does the candidate also beat the incumbent there?"""
        check = _robust_score(precomp, params, check_matrix, scope, regime_scope)
        return check > baseline_check + min_improve, check

    def _stage_hooks(cluster: str):
        """(checkpoint_cb, warm-start x0) for one cluster's DE run."""
        if ckpt is None:
            return None, None
        prev = ckpt.de_state or {}
        warm = (
            list(prev.get("best_x") or [])
            if prev.get("cluster") == cluster and prev.get("best_x")
            else None
        )

        def _cb(generation: int, best_x, best_fun) -> None:
            _persist(
                stage_index=ckpt.stage_index,
                de_state={
                    "cluster": cluster,
                    "generation": int(generation),
                    "best_x": [float(v) for v in np.asarray(best_x).ravel()],
                    "best_fun": float(best_fun) if best_fun is not None else float("inf"),
                },
            )

        return _cb, warm

    # ── Phase 1: marginals, every cluster against the FROZEN incumbent ────────────
    total = len(ordered) + 1  # + final joint pass
    done = resumed_stages
    t_stage = time.time()
    for idx, c in enumerate(ordered):
        if idx < resumed_stages:
            continue
        cb, warm = _stage_hooks(c)
        m = _tune_subset(precomp, c, run_matrix, scope, maxiter, popsize, baseline=original,
                         regime_scope=regime_scope, checkpoint_cb=cb, x0=warm,
                         max_seconds=max_seconds)
        done += 1
        if progress_callback:
            progress_callback(done, total, f"stage: {c}")
        if m is not None:
            replicated, check = _replicates(m.params)
            marginals[c] = {
                "score": float(m.score),
                "delta": float(m.score - orig_score),
                "replicated": bool(replicated),
                "check": float(check),
                "params": [float(v) for v in np.asarray(m.params).ravel()],
            }
        _persist(stage_index=idx + 1, de_state=None)
        logger.info("staged tune: marginal %d/%d (%s) done in %.0fs",
                    done, total, c, time.time() - t_stage)
        t_stage = time.time()

    # ── Tier 3: noise band from the clusters that did NOT replicate ───────────────
    # A cluster that improved the tuning windows but failed the disjoint-seed re-draw is,
    # by construction, a measured noise draw. The largest such gain is how big "nothing"
    # looks on this substrate today, so a real winner must clear it. This is the price of
    # taking a max over N clusters — without it, promoting whichever cluster happens to
    # top the list promotes noise nearly every round.
    noise_draws = [v["delta"] for v in marginals.values() if not v["replicated"]]
    band = max([d for d in noise_draws if d > 0.0], default=0.0)
    if noise_draws:
        logger.info(
            "staged tune: noise band %.4f from %d non-replicating cluster(s) %s",
            band, len(noise_draws), [round(d, 4) for d in noise_draws],
        )
    else:
        logger.info("staged tune: no non-replicating clusters — noise band unmeasured (0.0)")
    out.noise_band = band

    for c in ordered:
        v = marginals.get(c)
        if v is None:
            continue
        cleared = v["delta"] > band + min_improve
        accepted = bool(v["replicated"]) and cleared
        out.stages.append(StageResult(
            cluster=c, score_before=orig_score, score_after=v["score"], accepted=accepted,
            delta=v["delta"], noise_band=band, cleared_band=cleared,
        ))
        if v["replicated"] and not cleared:
            logger.info(
                "staged tune: %s replicated but its gain %.4f is inside the %.4f noise "
                "band — not promoted", c, v["delta"], band,
            )

    out.accepted_clusters = [s.cluster for s in out.stages if s.accepted]

    # ── Phase 2: joint re-tune over the promoted clusters (captures interactions) ──
    best = original.copy()
    best_score = orig_score
    if len(out.accepted_clusters) == 1:
        only = marginals[out.accepted_clusters[0]]
        best = np.asarray(only["params"], dtype=float)
        best_score = only["score"]
    elif len(out.accepted_clusters) >= 2 and not (ckpt is not None and ckpt.joint_done):
        joint_name = "+".join(out.accepted_clusters)
        cb, warm = _stage_hooks(joint_name)
        joint = _tune_subset(
            precomp, joint_name, run_matrix, scope, maxiter, popsize, baseline=original,
            regime_scope=regime_scope, checkpoint_cb=cb, x0=warm, max_seconds=max_seconds,
        )
        # Start from the best single marginal, so a failed joint pass never loses ground.
        top = max((marginals[c] for c in out.accepted_clusters), key=lambda v: v["score"])
        best = np.asarray(top["params"], dtype=float)
        best_score = top["score"]
        if joint is not None and joint.score > best_score + min_improve:
            joint_ok, _ = _replicates(joint.params)
            if joint_ok:
                best = joint.params.copy()
                best_score = joint.score
    done += 1
    if progress_callback:
        progress_callback(done, total, "final joint re-tune")

    # ── Anti-ratchet: the result must beat the ORIGINAL incumbent, not a re-baselined
    # one. Without this, running rounds back-to-back walks uphill on noise — each round
    # compares against a vector that was itself selected as a maximum.
    if out.accepted_clusters:
        final_vs_orig = _robust_score(precomp, best, check_matrix, scope, regime_scope)
        out.original_score = baseline_check
        out.final_vs_original = final_vs_orig
        out.beats_original = bool(final_vs_orig > baseline_check + min_improve)
        if not out.beats_original:
            out.promotion_blocked_reason = (
                f"final vector scores {final_vs_orig:.4f} on the disjoint-seed matrix vs the "
                f"ORIGINAL incumbent's {baseline_check:.4f} — promoted stages did not survive "
                "re-scoring against the starting config"
            )
            logger.warning("staged tune: %s", out.promotion_blocked_reason)
            best = original.copy()
            best_score = orig_score
            out.accepted_clusters = []

    out.final_params = best
    out.final_score = best_score
    _persist(stage_index=len(ordered), de_state=None, joint_done=True)
    return out


def _uniform_weight_variant(params: np.ndarray) -> np.ndarray:
    """The same candidate with slots 0-3 set equal — its weighting claim, neutralized.

    Everything else (thresholds, momentum internals, exits) is held identical, so the
    comparison isolates "is this WEIGHTING worth anything?" rather than re-testing the
    whole vector.
    """
    control = np.asarray(params, dtype=float).copy()
    control[:4] = 0.25
    return control


def _weights_are_uniform(params: np.ndarray, tol: float = 0.02) -> bool:
    w = np.asarray(params[:4], dtype=float)
    total = float(w.sum())
    if total <= 0:
        return True
    return bool(np.max(np.abs(w / total - 0.25)) <= tol)


def validate_full_windowed(
    precomp,
    params: np.ndarray,
    run_matrix: list[dict],
    scope: BacktestScope = "active_sleeve_compounding",
    regime_scope: str = "all",
    holdout_start_frac: float = 0.70,
    weight_control: bool = True,
) -> dict:
    """
    Full windowed confirmation of a candidate config: OOS train/val gate + robust scan
    across horizons/seeds on the full universe. Returns a verdict dict (never writes config).

    The robust-scan windows are DISJOINT from the tuning windows: they are sampled from
    the terminal (1 - holdout_start_frac) segment of the history — the segment
    run_staged_tune (train_frac=holdout_start_frac) never tunes on — with seeds offset
    from the tuning matrix. Previously this step re-ran the IDENTICAL windows the
    parameters were tuned on, so the robust/overfit verdicts were in-sample.

    EQUAL-WEIGHT CONTROL (weight_control=True, 2026-08-13). A candidate that proposes a
    non-uniform score weighting is ALSO scored with slots 0-3 flattened to equal weights,
    everything else identical, on the same windows. If the weighting cannot beat uniform
    it fails confirmation. Measured that day: the live incumbent (.25/.40/.25/.10) scored
    0.2387 against equal-weight's 0.2379 — a 0.3% gap, i.e. the weighting bought nothing —
    while every DE-tuned weighting scored strictly BELOW both. Without this control,
    "beats the incumbent" reads as a finding when it is a draw against a naive baseline.
    Candidates already ~uniform skip the control (nothing to justify).
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

    # 3. Equal-weight control — a non-uniform weighting must earn its keep against the
    # naive baseline, scored on the SAME windows so the comparison is paired.
    if not weight_control:
        out["weight_control"] = "skipped (disabled by caller)"
        out["weight_control_passed"] = True
    elif _weights_are_uniform(params):
        out["weight_control"] = "skipped (candidate weights already ~uniform)"
        out["weight_control_passed"] = True
    elif "scan_error" in out:
        out["weight_control"] = "skipped (candidate scan failed)"
        out["weight_control_passed"] = True
    else:
        try:
            ctrl_scan = run_robust_scan(
                scan_precomp, params=_uniform_weight_variant(params),
                run_matrix=val_matrix, scope=scope,
            )
            ctrl = float(ctrl_scan.overall_robust_score)
            cand = float(out.get("robust_score", 0.0))
            out["control_robust_score"] = ctrl
            out["weight_control_passed"] = bool(cand > ctrl)
            out["weight_control"] = (
                f"candidate {cand:.4f} vs equal-weight {ctrl:.4f} "
                f"({'beats' if cand > ctrl else 'LOSES TO'} the naive baseline)"
            )
            if not out["weight_control_passed"]:
                out.setdefault("oos_reasons", []).append(
                    f"weighting does not beat equal weights ({cand:.4f} <= {ctrl:.4f})"
                )
                logger.warning(
                    "equal-weight control: candidate robust %.4f <= uniform %.4f — "
                    "the proposed weighting is not earning its keep", cand, ctrl,
                )
        except Exception as exc:
            out["weight_control"] = f"control scan failed: {exc}"
            out["weight_control_passed"] = True   # never fail a candidate on our own error

    # Overall confirmation: OOS gate passes, not strongly overfit across horizons, AND
    # (when it proposes one) its weighting beats the equal-weight control.
    out["confirmed"] = (
        bool(out.get("oos_passed"))
        and out.get("overfit_score", 1.0) <= 0.5
        and bool(out.get("weight_control_passed", True))
    )
    return out
