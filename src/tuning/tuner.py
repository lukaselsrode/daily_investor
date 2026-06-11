"""
tuning/tuner.py — ParameterTuner and module-level run_tuner / run_auto_tune.

Module-level functions (run_tuner, run_auto_tune, validate_tuned_params,
should_apply_tuned_config) are the canonical implementations. ParameterTuner
is a thin typed wrapper over them.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from backtesting.data_loader import load_and_precompute
from backtesting.simulator import run_backtest_report, run_simulation
from backtesting.types import BacktestReport, SimResult
from util import BACKTEST_PARAMS

from .constants import (
    PARAM_NAMES,
    _current_params,
    _effective_bounds,
    _get_active_indices,
)
from .objective import _run_single
from .reports import (
    apply_config_params,
    build_llm_review_payload,
    merge_llm_recommendation_with_config,
    request_llm_tune_review,
    validate_llm_review_response,
)
from .results import AutoTuneResult, TuneResult

logger = logging.getLogger(__name__)

# Gate outcome of the most recent run_auto_tune() call — (passed, reasons).
# ParameterTuner.auto_tune() consumes this so AutoTuneResult.validation_passed /
# config_written mirror the EXACT predicate the write used (the gates now include
# incumbent-relative and random-window checks that a recompute would have to
# duplicate, including a second multi-day data load).
_LAST_GATE_OUTCOME: tuple[bool, list[str]] | None = None


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def validate_tuned_params(
    report: BacktestReport,
    backtest_cfg: dict,
    incumbent_report: BacktestReport | None = None,
) -> tuple[bool, list[str]]:
    """
    Gate a tuned candidate on its held-out validation window.

    Absolute gates (min excess / max drawdown / min Sharpe) catch outright failures.
    When *incumbent_report* is given (the CURRENT config evaluated on the SAME
    train/val split), two relative gates are added — these exist because a tuned
    config once passed the absolute gates with +0.34% validation excess while the
    incumbent sat at +8.87% with 5x less turnover (overfit signature):

      • tuned validation excess-vs-SPY must beat the incumbent's by at least
        `min_excess_vs_incumbent` (default 0.0 — never apply a config that is
        worse out-of-sample than what we already run);
      • tuned validation turnover must not exceed the incumbent's by more than
        `max_turnover_multiple` (default 2.0 — churn blowups are how overfit
        configs harvest in-sample noise).
    """
    if report.validation_result is None:
        return False, ["No validation window available — cannot validate"]

    vr = report.validation_result
    reasons: list[str] = []

    min_exc = backtest_cfg.get("min_validation_excess_return", 0.0)
    val_excess = vr.total_return - report.validation_benchmark_return
    if val_excess < min_exc:
        reasons.append(f"Validation excess return {val_excess:+.2%} < {min_exc:+.2%}")

    max_dd = backtest_cfg.get("max_validation_drawdown", -0.20)
    if vr.max_drawdown < max_dd:
        reasons.append(f"Validation max drawdown {vr.max_drawdown:.2%} < {max_dd:.2%}")

    min_sh = backtest_cfg.get("min_validation_sharpe", 0.25)
    if vr.sharpe < min_sh:
        reasons.append(f"Validation Sharpe {vr.sharpe:.3f} < {min_sh:.3f}")

    if incumbent_report is not None and incumbent_report.validation_result is not None:
        ivr = incumbent_report.validation_result
        inc_excess = ivr.total_return - incumbent_report.validation_benchmark_return

        margin = backtest_cfg.get("min_excess_vs_incumbent", 0.0)
        if val_excess < inc_excess + margin:
            reasons.append(
                f"Validation excess {val_excess:+.2%} does not beat incumbent config's "
                f"{inc_excess:+.2%} (+{margin:.2%} margin) on the same split"
            )

        turn_mult = backtest_cfg.get("max_turnover_multiple", 2.0)
        inc_turn = max(float(ivr.turnover_estimate), 1e-9)
        if float(vr.turnover_estimate) > inc_turn * turn_mult:
            reasons.append(
                f"Validation turnover {vr.turnover_estimate:.2f} > {turn_mult:.1f}x "
                f"incumbent's {ivr.turnover_estimate:.2f}"
            )

    return len(reasons) == 0, reasons


def paired_random_window_gate(
    tuned_params: np.ndarray,
    incumbent_params: np.ndarray,
    backtest_cfg: dict,
    mode: str | None = None,
    scope: str = "overall_strategy",
    precomp=None,
) -> tuple[bool, list[str], dict]:
    """
    Reproducibility gate: tuned vs incumbent params on the SAME random sub-windows
    of a longer history than the tune saw (paired comparison — shared seed ⇒ shared
    windows). A single train/val split can be won by luck; a paired multi-window
    sweep on unseen history cannot.

    Pass requires (config under backtest.random_window_gate, defaults shown):
      • enabled: true
      • paired per-window win rate ≥ `min_win_rate` (0.5)
      • tuned median excess-vs-SPY > incumbent median excess  (always on)
      • tuned robust_score > incumbent robust_score           (always on)
        (robust_score is excess-dominant and already penalizes turnover/std/drawdown)

    Returns (passed, reasons, stats). `precomp` may be injected for tests; otherwise
    `history_days` (default 730) of data are loaded fresh so windows cover history
    the optimizer never trained on.
    """
    gw = backtest_cfg.get("random_window_gate", {}) or {}
    if not gw.get("enabled", True):
        return True, [], {"skipped": True}

    from backtesting.random_walk import random_window_backtest

    history_days = int(gw.get("history_days", 730))
    n_windows    = int(gw.get("n_windows", 12))
    window_days  = int(gw.get("window_days", 120))
    seed         = int(gw.get("seed", 42))
    min_win_rate = float(gw.get("min_win_rate", 0.5))

    if precomp is None:
        precomp = load_and_precompute(history_days, mode=mode)

    common = dict(
        n_windows=n_windows,
        window_days=window_days,
        seed=seed,
        slippage_bps=backtest_cfg.get("slippage_bps", 10.0),
        commission_per_trade=backtest_cfg.get("commission_per_trade", 0.0),
        rebalance_frequency_days=backtest_cfg.get("rebalance_frequency_days", 5),
        scope=scope,
    )
    tuned_sum     = random_window_backtest(precomp, tuned_params,     **common)
    incumbent_sum = random_window_backtest(precomp, incumbent_params, **common)

    # Shared seed on the same precomp ⇒ identical windows; pair by window_id.
    inc_by_id = {w.window_id: w for w in incumbent_sum.window_results}
    paired = [
        (t, inc_by_id[t.window_id])
        for t in tuned_sum.window_results
        if t.window_id in inc_by_id
    ]
    wins = sum(1 for t, i in paired if t.excess_return > i.excess_return)
    win_rate = wins / len(paired) if paired else 0.0

    reasons: list[str] = []
    if win_rate < min_win_rate:
        reasons.append(
            f"Paired win rate {win_rate:.0%} ({wins}/{len(paired)} windows) < {min_win_rate:.0%}"
        )
    if tuned_sum.median_excess_return <= incumbent_sum.median_excess_return:
        reasons.append(
            f"Median excess-vs-SPY {tuned_sum.median_excess_return:+.2%} <= "
            f"incumbent's {incumbent_sum.median_excess_return:+.2%}"
        )
    if tuned_sum.robust_score <= incumbent_sum.robust_score:
        reasons.append(
            f"Robust score {tuned_sum.robust_score:.4f} <= incumbent's {incumbent_sum.robust_score:.4f}"
        )

    stats = {
        "win_rate": win_rate,
        "wins": wins,
        "n_paired": len(paired),
        "tuned_median_excess": tuned_sum.median_excess_return,
        "incumbent_median_excess": incumbent_sum.median_excess_return,
        "tuned_robust_score": tuned_sum.robust_score,
        "incumbent_robust_score": incumbent_sum.robust_score,
        "tuned_median_turnover": tuned_sum.median_turnover,
        "incumbent_median_turnover": incumbent_sum.median_turnover,
    }
    return len(reasons) == 0, reasons, stats


def multi_horizon_confirm(
    selected_params: np.ndarray,
    incumbent_params: np.ndarray,
    backtest_cfg: dict,
    mode: str | None = None,
    scope: str = "overall_strategy",
) -> tuple[bool, list[str], list[dict]]:
    """
    Final confirmation tier — runs ONLY after the split, incumbent-relative,
    and random-window gates have all passed. The selected candidate and the
    incumbent are simulated on trailing windows (default 90/180/365/730d, same
    mode/scope as the tune) and compared on excess-vs-SPY, max drawdown, and
    turnover. Automates the manual "confirm with independent backtests before
    trusting a tune" step.

    Rules (config: backtest.multi_horizon_confirm):
      • short windows (<=120d): excess may regress at most short_regress_tolerance
      • mid windows (<=400d):   improve-or-preserve (mid_regress_tolerance slack)
      • long windows:           no catastrophic regime failure — excess regression
        beyond long_catastrophe_excess OR drawdown deeper by more than
        long_catastrophe_drawdown fails
      • every window: turnover within max_turnover_multiple of the incumbent

    Returns (passed, reasons, rows). Disabled or no usable windows → passes
    (the prior gates remain in force).
    """
    cfg = backtest_cfg.get("multi_horizon_confirm", {}) or {}
    if not cfg.get("enabled", True):
        return True, [], []

    from backtesting.regime_scope import slice_precomp

    windows = sorted(int(w) for w in cfg.get("windows", [90, 180, 365, 730]))
    short_tol = float(cfg.get("short_regress_tolerance", 0.02))
    mid_tol = float(cfg.get("mid_regress_tolerance", 0.005))
    long_excess = float(cfg.get("long_catastrophe_excess", 0.10))
    long_dd = float(cfg.get("long_catastrophe_drawdown", 0.05))
    turn_mult = float(backtest_cfg.get("max_turnover_multiple", 2.0))

    try:
        full = load_and_precompute(max(windows), mode=mode)
    except Exception as exc:
        return False, [f"multi-horizon confirm: could not load history ({exc})"], []
    n_full = full.prices.shape[0]

    def _sim(pc, params):
        return run_simulation(
            pc, params, backtest_cfg.get("starting_capital", 10_000.0),
            slippage_bps=backtest_cfg.get("slippage_bps", 10.0),
            commission_per_trade=backtest_cfg.get("commission_per_trade", 0.0),
            weekly_contribution=backtest_cfg.get("weekly_contribution", 0.0),
            rebalance_frequency_days=backtest_cfg.get("rebalance_frequency_days", 5),
            scope=scope,
        )

    rows: list[dict] = []
    reasons: list[str] = []
    for w in windows:
        if w > n_full:
            continue
        pc = full if w == n_full else slice_precomp(full, slice(n_full - w, n_full))
        sel = _sim(pc, selected_params)
        inc = _sim(pc, incumbent_params)
        sel_exc = sel.total_return - sel.benchmark_twr
        inc_exc = inc.total_return - inc.benchmark_twr
        delta = sel_exc - inc_exc
        row = {
            "window": w,
            "incumbent_excess": inc_exc, "selected_excess": sel_exc, "delta": delta,
            "incumbent_max_drawdown": inc.max_drawdown, "selected_max_drawdown": sel.max_drawdown,
            "incumbent_turnover": inc.turnover_estimate, "selected_turnover": sel.turnover_estimate,
        }
        rows.append(row)

        tier = "short" if w <= 120 else ("mid" if w <= 400 else "long")
        if tier == "short" and delta < -short_tol:
            reasons.append(
                f"{w}d: selected excess {sel_exc:+.2%} regresses incumbent {inc_exc:+.2%} "
                f"by more than {short_tol:.1%}"
            )
        elif tier == "mid" and delta < -mid_tol:
            reasons.append(
                f"{w}d: selected excess {sel_exc:+.2%} fails improve-or-preserve vs "
                f"incumbent {inc_exc:+.2%} (tolerance {mid_tol:.1%})"
            )
        elif tier == "long":
            if delta < -long_excess:
                reasons.append(
                    f"{w}d: catastrophic excess regression {delta:+.2%} (limit -{long_excess:.0%})"
                )
            if sel.max_drawdown < inc.max_drawdown - long_dd:
                reasons.append(
                    f"{w}d: drawdown {sel.max_drawdown:.1%} deeper than incumbent "
                    f"{inc.max_drawdown:.1%} by more than {long_dd:.0%}"
                )
        inc_turn = max(float(inc.turnover_estimate), 1e-9)
        if float(sel.turnover_estimate) > inc_turn * turn_mult:
            reasons.append(
                f"{w}d: turnover {sel.turnover_estimate:.2f} > {turn_mult:.1f}x "
                f"incumbent's {inc.turnover_estimate:.2f}"
            )
    return len(reasons) == 0, reasons, rows


def _print_multi_horizon_table(rows: list[dict]) -> None:
    if not rows:
        return
    print(
        f"\n{'window':>7} {'inc_exc':>9} {'sel_exc':>9} {'delta':>8} "
        f"{'inc_DD':>8} {'sel_DD':>8} {'turn_Δ':>8}"
    )
    for r in rows:
        print(
            f"{r['window']:>6}d {r['incumbent_excess']:>+9.2%} {r['selected_excess']:>+9.2%} "
            f"{r['delta']:>+8.2%} {r['incumbent_max_drawdown']:>8.1%} "
            f"{r['selected_max_drawdown']:>8.1%} "
            f"{r['selected_turnover'] - r['incumbent_turnover']:>+8.2f}"
        )


# ---------------------------------------------------------------------------
# Candidate tournament — the selection layer between DE optimization and the
# gates. The arithmetic midpoint of the Sharpe- and Calmar-optimal vectors is
# not guaranteed to be a strategy either objective wants; instead, a small
# family of candidates (the optima, three blends, and three incumbent blends)
# is evaluated on the SAME train/validation split and the best gate-passing
# candidate is selected. The gates remain the final authority.
# ---------------------------------------------------------------------------

def build_tournament_candidates(
    sharpe_params: np.ndarray,
    calmar_params: np.ndarray,
    incumbent_params: np.ndarray,
) -> dict[str, np.ndarray]:
    """Ordered candidate_id → full param vector. Blends are convex combinations,
    so candidates stay inside any region the inputs occupy."""
    avg = 0.5 * sharpe_params + 0.5 * calmar_params
    return {
        "sharpe_opt":         sharpe_params.copy(),
        "calmar_opt":         calmar_params.copy(),
        "avg_50_50":          avg,
        "avg_25_75":          0.25 * sharpe_params + 0.75 * calmar_params,
        "avg_75_25":          0.75 * sharpe_params + 0.25 * calmar_params,
        "incumbent_blend_25": 0.75 * incumbent_params + 0.25 * avg,
        "incumbent_blend_50": 0.50 * incumbent_params + 0.50 * avg,
        "incumbent_blend_75": 0.25 * incumbent_params + 0.75 * avg,
    }


def _coerce_vector(vec: np.ndarray, reference: np.ndarray, label: str) -> np.ndarray | None:
    """Fit an externally-supplied vector to the current param layout: shorter
    vectors (saved before newer slot families were appended) are padded with
    the reference (incumbent) tail; longer ones are rejected — a longer vector
    means it was saved under an UNKNOWN future layout and slot meanings can't
    be trusted."""
    vec = np.asarray(vec, dtype=float).ravel()
    if len(vec) == len(reference):
        return vec.copy()
    if len(vec) < len(reference):
        out = reference.copy()
        out[: len(vec)] = vec
        logger.info(
            "Candidate %s padded from %d to %d slots with incumbent values",
            label, len(vec), len(reference),
        )
        return out
    logger.warning(
        "Candidate %s has %d slots but the current layout has %d — skipped",
        label, len(vec), len(reference),
    )
    return None


def assemble_candidate_pool(
    sharpe_params: np.ndarray,
    calmar_params: np.ndarray,
    incumbent_params: np.ndarray,
    random_topk_vectors: dict[str, np.ndarray] | None = None,
    lead_vectors: dict[str, np.ndarray] | None = None,
    manual_vectors: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """
    Full tournament pool: candidate_id → {"params", "source"}.

    Sources: the DE optima + blends ("de"/"blend"/"incumbent_blend"), top-K
    vectors from the robust-score random search ("random_search"), saved lead
    vectors from prior research ("lead"), and manually supplied vectors
    ("manual"). All enter the SAME evaluation, gates, and selection.
    """
    pool: dict[str, dict] = {}
    base_source = {
        "sharpe_opt": "de", "calmar_opt": "de",
        "avg_50_50": "blend", "avg_25_75": "blend", "avg_75_25": "blend",
        "incumbent_blend_25": "incumbent_blend",
        "incumbent_blend_50": "incumbent_blend",
        "incumbent_blend_75": "incumbent_blend",
    }
    for cid, vec in build_tournament_candidates(sharpe_params, calmar_params, incumbent_params).items():
        pool[cid] = {"params": vec, "source": base_source[cid]}
    for group, source in ((random_topk_vectors, "random_search"),
                          (lead_vectors, "lead"),
                          (manual_vectors, "manual")):
        for cid, vec in (group or {}).items():
            fitted = _coerce_vector(vec, incumbent_params, cid)
            if fitted is None:
                continue
            if cid in pool:
                cid = f"{source}:{cid}"
            pool[cid] = {"params": fitted, "source": source}
    return pool


def _select_tournament_winner(rows: list[dict]) -> tuple[dict, bool]:
    """Pick the tournament winner: Pareto-non-dominated gate-passers ranked by
    selection score. With no passer, keep the best-scoring row (legacy-midpoint
    fallback) for DIAGNOSTICS only — nothing is written when gates failed.
    Prints the table + selection. Returns (selected_row, any_passed)."""
    passers = [r for r in rows if r["gates_passed"] and r["metrics"] is not None]
    if passers:
        non_dominated = _pareto_non_dominated(passers)
        pool = [r for r in passers if r["candidate_id"] in non_dominated] or passers
        selected = max(pool, key=lambda r: r["score"])
        note = "highest validation robust score among candidates passing gates."
    else:
        diagnostic = max(rows, key=lambda r: r["score"])
        selected = next(r for r in rows if r["candidate_id"] == "avg_50_50")
        if np.isfinite(diagnostic["score"]):
            selected = diagnostic
        note = "none passed gates — best-scoring candidate kept for diagnostics only."
    _print_tournament_table(rows, selected["candidate_id"] if passers else None)
    print(f"\nSelected candidate: {selected['candidate_id'] if passers else 'none (gates failed)'}")
    print(f"Reason: {note}")
    return selected, bool(passers)


def _post_selection_gates(
    selected: dict,
    selected_params: np.ndarray,
    incumbent_params: np.ndarray,
    backtest_cfg: dict,
    mode: str | None,
    scope: str,
) -> tuple[bool, list[str]]:
    """The gate chain AFTER tournament selection: the candidate's own split-gate
    outcome, then the paired random-window gate, then multi-horizon confirmation.
    Later tiers run only while everything earlier passed — failed candidates
    never consume the expensive gates."""
    validation_passed, reasons = selected["gates_passed"], list(selected["reasons"])

    if validation_passed:
        gw_cfg = backtest_cfg.get("random_window_gate", {}) or {}
        if gw_cfg.get("enabled", True):
            print(
                f"\nRandom-window gate: {gw_cfg.get('n_windows', 12)}x"
                f"{gw_cfg.get('window_days', 120)}d paired windows over "
                f"{gw_cfg.get('history_days', 730)}d history …"
            )
            rw_passed, rw_reasons, rw_stats = paired_random_window_gate(
                selected_params, incumbent_params, backtest_cfg, mode=mode, scope=scope,
            )
            if rw_stats and not rw_stats.get("skipped"):
                print(
                    f"  paired win rate {rw_stats['win_rate']:.0%} "
                    f"({rw_stats['wins']}/{rw_stats['n_paired']})  |  median excess "
                    f"tuned {rw_stats['tuned_median_excess']:+.2%} vs incumbent "
                    f"{rw_stats['incumbent_median_excess']:+.2%}  |  robust score "
                    f"tuned {rw_stats['tuned_robust_score']:.4f} vs incumbent "
                    f"{rw_stats['incumbent_robust_score']:.4f}"
                )
            validation_passed = rw_passed
            reasons.extend(rw_reasons)

    if validation_passed:
        mh_cfg = backtest_cfg.get("multi_horizon_confirm", {}) or {}
        if mh_cfg.get("enabled", True):
            print(
                f"\nMulti-horizon confirm: selected vs incumbent over "
                f"{mh_cfg.get('windows', [90, 180, 365, 730])} trailing days …"
            )
            mh_passed, mh_reasons, mh_rows = multi_horizon_confirm(
                selected_params, incumbent_params, backtest_cfg, mode=mode, scope=scope,
            )
            _print_multi_horizon_table(mh_rows)
            validation_passed = mh_passed
            reasons.extend(mh_reasons)

    return validation_passed, reasons


def _source_random_candidates(
    tune_precomp,
    incumbent_params: np.ndarray,
    random_topk: int,
    backtest_cfg: dict,
    scope: str,
    preset: str | None,
    train_days: int,
) -> dict[str, np.ndarray]:
    """Top-K vectors from the robust-score random search (shared-window ranking
    vs the incumbent baseline) for the tournament. Empty dict when disabled or
    when the search fails — candidate sourcing must never sink the tune."""
    out: dict[str, np.ndarray] = {}
    if random_topk <= 0:
        return out
    try:
        from .random_tune import run_random_weight_tune
        print(f"\nRobust random search for tournament candidates (top-{random_topk}) …")
        rs = run_random_weight_tune(
            tune_precomp,
            base_params=incumbent_params.copy(),
            n_samples=max(24, random_topk * 8),
            n_windows=10,
            window_days=min(120, max(30, train_days - 1)),
            seed=7,
            slippage_bps=backtest_cfg["slippage_bps"],
            rebalance_frequency_days=backtest_cfg["rebalance_frequency_days"],
            scope=scope,
            preset=preset,
        )
        for rank, cand in enumerate(rs.candidates[:random_topk], 1):
            out[f"random_top{rank}"] = cand.full_params
    except Exception as exc:
        print(f"  robust random search unavailable ({exc}) — continuing without")
    return out


def _load_lead_vectors(paths: list[str]) -> dict[str, np.ndarray]:
    """Load saved .npy lead vectors (e.g. prior research candidates); unreadable
    files are skipped with a warning, never fatal."""
    import os

    out: dict[str, np.ndarray] = {}
    for p in paths or []:
        try:
            name = f"lead:{os.path.splitext(os.path.basename(p))[0]}"
            out[name] = np.load(p)
        except Exception as exc:
            logger.warning("Could not load lead vector %s: %s", p, exc)
    return out


# Isolated so the selection trade-offs can be adjusted without touching the
# tournament mechanics. Excess terms dominate (project rule: judge on excess);
# Calmar/Sharpe are tie-breakers; churn and incumbent-relative drawdown
# deterioration are penalized.
_SELECTION_W_CALMAR    = 0.25
_SELECTION_W_SHARPE    = 0.10
_SELECTION_W_TURNOVER  = 0.25
_SELECTION_W_DRAWDOWN  = 0.50


def candidate_selection_score(m: dict) -> float:
    """Validation robust score for tournament ranking (higher is better)."""
    return (
        m["val_excess"]
        + m["val_excess_vs_incumbent"]
        + _SELECTION_W_CALMAR * m["val_calmar"]
        + _SELECTION_W_SHARPE * m["val_sharpe"]
        - _SELECTION_W_TURNOVER * max(0.0, m["turnover_multiple"] - 1.0)
        - _SELECTION_W_DRAWDOWN * max(0.0, m["drawdown_worse_than_incumbent"])
    )


def _candidate_metrics(report: BacktestReport, incumbent_report: BacktestReport) -> dict | None:
    """Validation metrics vs SPY and vs the incumbent. None when either report
    lacks a validation window (the candidate then fails gates anyway)."""
    vr = report.validation_result
    ivr = incumbent_report.validation_result if incumbent_report is not None else None
    if vr is None or ivr is None:
        return None
    val_excess = vr.total_return - report.validation_benchmark_return
    inc_excess = ivr.total_return - incumbent_report.validation_benchmark_return
    inc_turn = max(float(ivr.turnover_estimate), 1e-9)
    return {
        "val_excess":                   float(val_excess),
        "val_excess_vs_incumbent":      float(val_excess - inc_excess),
        "val_sharpe":                   float(vr.sharpe),
        "val_calmar":                   float(vr.calmar),
        "val_max_drawdown":             float(vr.max_drawdown),
        "val_turnover":                 float(vr.turnover_estimate),
        "turnover_multiple":            float(vr.turnover_estimate) / inc_turn,
        # max_drawdown is negative; positive when the candidate draws down DEEPER.
        "drawdown_worse_than_incumbent": max(0.0, float(ivr.max_drawdown) - float(vr.max_drawdown)),
    }


def _pareto_non_dominated(rows: list[dict]) -> set[str]:
    """IDs of candidates not dominated on (excess↑, calmar↑, drawdown↑ [less
    negative], turnover↓). Dominated = another candidate is >= on all four with
    at least one strict improvement."""
    def dominates(a: dict, b: dict) -> bool:
        ge = (
            a["val_excess"] >= b["val_excess"]
            and a["val_calmar"] >= b["val_calmar"]
            and a["val_max_drawdown"] >= b["val_max_drawdown"]
            and a["val_turnover"] <= b["val_turnover"]
        )
        strict = (
            a["val_excess"] > b["val_excess"]
            or a["val_calmar"] > b["val_calmar"]
            or a["val_max_drawdown"] > b["val_max_drawdown"]
            or a["val_turnover"] < b["val_turnover"]
        )
        return ge and strict

    out: set[str] = set()
    for r in rows:
        if r.get("metrics") is None:
            continue
        if not any(
            other is not r and other.get("metrics") is not None
            and dominates(other["metrics"], r["metrics"])
            for other in rows
        ):
            out.add(r["candidate_id"])
    return out


def run_candidate_tournament(
    precomp,
    candidate_pool: dict[str, dict],
    train_sl,
    val_slice,
    scope: str,
    backtest_cfg: dict,
    incumbent_report: BacktestReport,
) -> list[dict]:
    """Evaluate every candidate (id → {params, source}) on the same split and
    gate each with validate_tuned_params (absolute + incumbent-relative +
    turnover). Returns rows of {candidate_id, source, params, report, metrics,
    gates_passed, reasons, score}; score is -inf without validation metrics."""
    rows: list[dict] = []
    for cid, entry in candidate_pool.items():
        params = entry["params"]
        report = run_backtest_report(precomp, params, train_sl, val_slice, scope=scope)
        passed, reasons = validate_tuned_params(report, backtest_cfg, incumbent_report)
        metrics = _candidate_metrics(report, incumbent_report)
        rows.append({
            "candidate_id": cid,
            "source": entry.get("source", "de"),
            "params": params,
            "report": report,
            "metrics": metrics,
            "gates_passed": passed,
            "reasons": reasons,
            "score": candidate_selection_score(metrics) if metrics is not None else float("-inf"),
        })
    return rows


def _print_tournament_table(rows: list[dict], selected_id: str | None) -> None:
    print(
        f"\n{'candidate_id':<22} {'source':<15} {'val_exc':>8} {'vs_inc':>8} {'sharpe':>7} "
        f"{'calmar':>7} {'maxDD':>7} {'turn':>6} {'gates':>6} {'score':>8}"
    )
    for r in rows:
        m = r.get("metrics")
        src = r.get("source", "")
        mark = " ◀ selected" if r["candidate_id"] == selected_id else ""
        if m is None:
            print(f"{r['candidate_id']:<22} {src:<15} {'—':>8} {'—':>8} {'—':>7} {'—':>7} "
                  f"{'—':>7} {'—':>6} {'FAIL':>6} {'—':>8}{mark}")
            continue
        print(
            f"{r['candidate_id']:<22} {src:<15} {m['val_excess']:>+8.2%} {m['val_excess_vs_incumbent']:>+8.2%} "
            f"{m['val_sharpe']:>7.2f} {m['val_calmar']:>7.2f} {m['val_max_drawdown']:>7.1%} "
            f"{m['val_turnover']:>6.2f} {'PASS' if r['gates_passed'] else 'FAIL':>6} "
            f"{r['score']:>+8.4f}{mark}"
        )


def should_apply_tuned_config(
    apply_flag: bool,
    validation_passed: bool,
    backtest_cfg: dict,
    force_apply: bool = False,
) -> bool:
    if force_apply:
        return True
    return validation_passed and (apply_flag or backtest_cfg.get("auto_apply_if_valid", False))


# ---------------------------------------------------------------------------
# Single-objective public entry point
# ---------------------------------------------------------------------------

def run_tuner(
    n_days: int = 90,
    objective: Literal["sharpe", "calmar", "info_ratio"] = "sharpe",
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
    mode: str | None = None,
    scope: str = "overall_strategy",
    preset: str | None = None,
    regime_scope: str = "all",
) -> tuple[np.ndarray, SimResult]:
    """Optimize a single objective. Returns (best_params, SimResult)."""
    if preset is not None:
        from .presets import validate_preset
        validate_preset(preset)

    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scipy is required. Install: pip install scipy") from exc

    precomp = load_and_precompute(n_days, mode=mode)
    if regime_scope != "all":
        from backtesting.regime_scope import apply_regime_scope
        precomp, regime_meta = apply_regime_scope(precomp, regime_scope)
        n_days = precomp.prices.shape[0]
        print(
            f"Regime scope: {regime_meta['requested']} → {regime_meta['effective']} "
            f"({regime_meta['selected_days']}/{regime_meta['total_days']} days, "
            f"slice={regime_meta['start_day']}:{regime_meta['end_day']})"
        )
    _preset_label = f", preset={preset}" if preset else ""
    print(
        f"\nOptimizing {len(PARAM_NAMES)} parameters over {n_days} trading days "
        f"(objective: {objective}, mode={precomp.mode}, scope={scope}{_preset_label})."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")
    print("This may take several minutes …\n")
    return _run_single(precomp, objective, starting_capital, maxiter, popsize, scope=scope, preset=preset)


# ---------------------------------------------------------------------------
# Dual-objective auto-tune
# ---------------------------------------------------------------------------

def run_auto_tune(
    n_days: int = 90,
    starting_capital: float = 10_000.0,
    maxiter: int = 25,
    popsize: int = 8,
    mode: str | None = None,
    apply: bool = False,
    force_apply: bool = False,
    llm_review: bool = False,
    scope: str = "overall_strategy",
    preset: str | None = None,
    regime_scope: str = "all",
    random_topk: int = 0,
    lead_vector_paths: list[str] | None = None,
    extra_candidates: dict[str, np.ndarray] | None = None,
) -> tuple:
    """
    Run Sharpe + Calmar optimizations, then a candidate TOURNAMENT (the optima,
    sharpe/calmar blends, and incumbent blends — see build_tournament_candidates)
    evaluated on the held-out split; the best gate-passing, Pareto-non-dominated
    candidate is selected. Writes config.yaml only when the selected candidate
    passes every gate (split + incumbent-relative + random-window) and
    apply=True or auto_apply_if_valid=True.

    Returns (selected_params, sharpe_result, calmar_result, selected_result,
    sharpe_params, calmar_params) — the first/fourth slots held the blind
    average before the tournament existed; callers' tuple shape is unchanged.
    """
    global _LAST_GATE_OUTCOME
    _LAST_GATE_OUTCOME = None

    if preset is not None:
        from .presets import validate_preset
        validate_preset(preset)

    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scipy is required. Install: pip install scipy") from exc

    bp = BACKTEST_PARAMS
    use_val  = bp.get("use_out_of_sample_validation", True)
    train_pct = bp.get("train_pct", 0.70)

    precomp = load_and_precompute(n_days, mode=mode)
    if regime_scope != "all":
        from backtesting.regime_scope import apply_regime_scope
        precomp, regime_meta = apply_regime_scope(precomp, regime_scope)
        n_days = precomp.prices.shape[0]
        print(
            f"Regime scope: {regime_meta['requested']} → {regime_meta['effective']} "
            f"({regime_meta['selected_days']}/{regime_meta['total_days']} days, "
            f"slice={regime_meta['start_day']}:{regime_meta['end_day']})"
        )

    from backtesting.simulator import split_price_window
    train_sl, val_sl = split_price_window(n_days, train_pct)

    def _opt_sl(arr):
        return arr[train_sl] if arr is not None else None

    tune_precomp = precomp._replace(
        prices=precomp.prices[train_sl],
        etf_prices=precomp.etf_prices[train_sl],
        benchmark_prices=precomp.benchmark_prices[train_sl],
        position_52w_daily=precomp.position_52w_daily[train_sl],
        return_1m_daily=precomp.return_1m_daily[train_sl],
        bin_indices_daily=precomp.bin_indices_daily[train_sl],
        has_position_52w_daily=precomp.has_position_52w_daily[train_sl],
        ret_5d_daily=_opt_sl(precomp.ret_5d_daily),
        ret_3m_daily=_opt_sl(precomp.ret_3m_daily),
        ret_6m_daily=_opt_sl(precomp.ret_6m_daily),
        rs_3m_daily=_opt_sl(precomp.rs_3m_daily),
        rs_6m_daily=_opt_sl(precomp.rs_6m_daily),
        vol_3m_daily=_opt_sl(precomp.vol_3m_daily),
        above_50dma_daily=_opt_sl(precomp.above_50dma_daily),
        above_200dma_daily=_opt_sl(precomp.above_200dma_daily),
    )
    train_days = tune_precomp.prices.shape[0]

    print(
        f"\nAuto-tune: {len(PARAM_NAMES)} params, {n_days} days "
        f"({train_days} train / {n_days - train_days} val), "
        f"mode={precomp.mode}, bias={precomp.lookahead_bias_level}."
    )
    print(f"scipy differential_evolution: popsize={popsize}, maxiter={maxiter}")
    _preset_label = f", preset={preset}" if preset else ""

    print(f"\n[1/2] Optimizing for Sharpe (scope={scope}{_preset_label}) …\n")
    sharpe_params, sharpe_result = _run_single(tune_precomp, "sharpe", starting_capital, maxiter, popsize, scope=scope, preset=preset)

    print(f"\n[2/2] Optimizing for Calmar (scope={scope}{_preset_label}) …\n")
    calmar_params, calmar_result = _run_single(tune_precomp, "calmar", starting_capital, maxiter, popsize, scope=scope, preset=preset)

    val_slice = val_sl if use_val else None

    # Incumbent = the config we're currently running, on the SAME split/scope.
    # Needed FIRST: it seeds the incumbent-blend candidates and anchors the
    # relative gates. Gate at the SAME scope the parameters were optimized at:
    # an active-sleeve preset tune gated at "overall_strategy" is ~77% benchmark
    # (index_pct frozen), so the gates would pass/fail on index noise.
    incumbent_params = _current_params()
    incumbent_report = run_backtest_report(precomp, incumbent_params, train_sl, val_slice, scope=scope)

    # ── Candidate tournament (multi-source) ───────────────────────────────────
    # Replaces the blind (sharpe + calmar) / 2 midpoint: the DE optima, blends,
    # incumbent blends — plus optional robust-search top-K, saved lead vectors,
    # and manually supplied vectors — are each evaluated on the same split,
    # gated individually, Pareto-filtered, and ranked by the selection score.
    random_topk_vectors = _source_random_candidates(
        tune_precomp, incumbent_params, random_topk, bp, scope, preset, train_days,
    )
    lead_vectors = _load_lead_vectors(lead_vector_paths or [])
    candidate_pool = assemble_candidate_pool(
        sharpe_params, calmar_params, incumbent_params,
        random_topk_vectors=random_topk_vectors,
        lead_vectors=lead_vectors,
        manual_vectors=extra_candidates,
    )
    rows = run_candidate_tournament(
        precomp, candidate_pool, train_sl, val_slice, scope, bp, incumbent_report,
    )
    selected, _ = _select_tournament_winner(rows)

    selected_params = selected["params"]
    train_report = selected["report"]
    selected_result = run_simulation(
        tune_precomp, selected_params, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )

    _sel_vr = train_report.validation_result
    _inc_vr = incumbent_report.validation_result
    if use_val and _sel_vr is not None and _inc_vr is not None:
        _t_exc = _sel_vr.total_return - train_report.validation_benchmark_return
        _i_exc = _inc_vr.total_return - incumbent_report.validation_benchmark_return
        print(
            f"\nValidation window (excess vs SPY):  selected {_t_exc:+.2%}  vs  incumbent {_i_exc:+.2%}"
            f"   |  turnover: selected {_sel_vr.turnover_estimate:.2f}"
            f" vs incumbent {_inc_vr.turnover_estimate:.2f}"
        )

    validation_passed, reasons = _post_selection_gates(
        selected, selected_params, incumbent_params, bp, mode, scope,
    )

    if reasons:
        print("\n⚠  Validation gates FAILED:")
        for r in reasons:
            print(f"   • {r}")
    else:
        print("\n✓  Validation gates passed.")

    _LAST_GATE_OUTCOME = (validation_passed, list(reasons))

    use_llm = llm_review or bp.get("llm_review_enabled", False)
    llm_apply = bp.get("llm_review_apply", False)
    final_params = selected_params

    if use_llm:
        print(f"\n[LLM review] Sending candidates to {bp.get('llm_review_model', 'claude-sonnet-4-6')} …")
        try:
            tr = train_report.train_result
            vr = train_report.validation_result

            def _llm_entry(cid: str, params, train_metrics) -> dict:
                """One review-payload row — shared shape for every candidate."""
                return {
                    "candidate_id": cid,
                    "alpha_params": dict(zip(PARAM_NAMES, params.tolist())),
                    "train_return": train_metrics.total_return,
                    "train_sharpe": train_metrics.sharpe,
                    "train_calmar": train_metrics.calmar,
                    "train_max_drawdown": train_metrics.max_drawdown,
                    "train_trades": train_metrics.trades_made,
                    "train_avg_positions": train_metrics.average_positions,
                    "train_max_positions": train_metrics.max_positions,
                    "train_avg_cash_pct": train_metrics.average_cash_pct,
                    "train_turnover": train_metrics.turnover_estimate,
                    "train_friction_cost": train_metrics.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                }

            candidates = [
                _llm_entry("sharpe_opt", sharpe_params, sharpe_result),
                _llm_entry("calmar_opt", calmar_params, calmar_result),
                _llm_entry(f"selected:{selected['candidate_id']}", selected_params, tr),
            ]
            payload = build_llm_review_payload(
                candidates,
                mode=precomp.mode,
                universe_selection=precomp.universe_selection,
                benchmark_symbol=precomp.benchmark_symbol,
                validation_cfg=bp,
            )
            response = request_llm_tune_review(payload)
            valid, errors = validate_llm_review_response(response, candidates)
            if not valid:
                print(f"[LLM review] Response invalid — ignoring: {errors}")
            else:
                print(
                    f"[LLM review] Recommended: {response['recommended_candidate_id']}  "
                    f"confidence={response.get('confidence', '?'):.0%}\n"
                    f"  Rationale: {response.get('rationale', '')}\n"
                )
                if response.get("risk_warnings"):
                    for w in response["risk_warnings"]:
                        print(f"  ⚠ {w}")
                if llm_apply and response.get("apply_candidate_as_is") and response.get("proposed_adjustments"):
                    import yaml as _yaml

                    from util import CONFIG_FILE
                    with open(CONFIG_FILE) as f:
                        cfg = _yaml.safe_load(f)
                    merged = merge_llm_recommendation_with_config(cfg, response)
                    with open(CONFIG_FILE, "w") as f:
                        _yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
                    print("[LLM review] Adjustments applied to config.yaml")
        except Exception as e:
            print(f"[LLM review] Failed ({e}) — continuing without LLM input")

    if should_apply_tuned_config(apply, validation_passed, bp, force_apply=force_apply):
        apply_config_params(final_params)
    elif apply and not validation_passed:
        print("Config NOT written: validation gates failed. Use --force-apply to override.")
    else:
        print("Config NOT written (--apply requires validation to pass; use --force-apply to override).")

    # Tuple shape preserved for callers: slots [0]/[3] carried the blind average
    # before the tournament; they now carry the SELECTED candidate.
    return selected_params, sharpe_result, calmar_result, selected_result, sharpe_params, calmar_params


# ---------------------------------------------------------------------------
# ParameterTuner — typed wrapper
# ---------------------------------------------------------------------------

class ParameterTuner:
    """
    Conservative parameter optimizer backed by scipy differential_evolution.

    Runs in a reduced parameter space defined by config.tuning.frozen_parameters.
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    @property
    def param_names(self) -> list[str]:
        return list(PARAM_NAMES)

    @staticmethod
    def _active_param_names(scope: str = "overall_strategy", preset: str | None = None) -> list[str]:
        """Names of the slots actually tunable for the given scope/preset."""
        active_idx = set(_get_active_indices(scope, preset=preset))
        return [PARAM_NAMES[i] for i in sorted(active_idx)]

    @property
    def active_params(self) -> list[str]:
        return self._active_param_names()

    @property
    def frozen_params(self) -> list[str]:
        active_idx = set(_get_active_indices())
        return [name for i, name in enumerate(PARAM_NAMES) if i not in active_idx]

    @property
    def effective_bounds(self) -> list[tuple[float, float]]:
        return _effective_bounds()

    def current_params(self) -> np.ndarray:
        return _current_params()

    def tune(
        self,
        n_days: int,
        objective: str = "sharpe",
        starting_capital: float = 10_000.0,
        mode: str | None = None,
        scope: str = "overall_strategy",
        preset: str | None = None,
        regime_scope: str = "all",
    ) -> TuneResult:
        params, sim = run_tuner(
            n_days=n_days,
            objective=objective,  # type: ignore[arg-type]
            starting_capital=starting_capital,
            mode=mode,
            scope=scope,
            preset=preset,
            regime_scope=regime_scope,
        )
        return TuneResult(
            params=params,
            sim=sim,
            objective=objective,
            n_days=n_days,
            active_params=self._active_param_names(scope, preset),
        )

    def auto_tune(
        self,
        n_days: int,
        apply: bool = False,
        force_apply: bool = False,
        mode: str | None = None,
        llm_review: bool = False,
        starting_capital: float = 10_000.0,
        scope: str = "overall_strategy",
        preset: str | None = None,
        regime_scope: str = "all",
        random_topk: int = 0,
        lead_vector_paths: list[str] | None = None,
        extra_candidates: dict | None = None,
    ) -> AutoTuneResult:
        # Clear any stale stash so we only consume the outcome of THIS call
        # (run_auto_tune may be replaced in tests; it resets the stash itself
        # when it actually runs).
        global _LAST_GATE_OUTCOME
        _LAST_GATE_OUTCOME = None
        raw = run_auto_tune(
            n_days=n_days,
            starting_capital=starting_capital,
            mode=mode,
            apply=apply,
            force_apply=force_apply,
            llm_review=llm_review,
            scope=scope,
            preset=preset,
            regime_scope=regime_scope,
            random_topk=random_topk,
            lead_vector_paths=lead_vector_paths,
            extra_candidates=extra_candidates,
        )
        avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params = raw

        bp = BACKTEST_PARAMS
        use_val = bp.get("use_out_of_sample_validation", True)

        validation_passed = False
        validation_reasons: list[str] = []

        if _LAST_GATE_OUTCOME is not None:
            # run_auto_tune just executed and stashed the exact gate outcome it
            # used for the write decision (absolute + incumbent-relative +
            # random-window gates). Consume it instead of recomputing a weaker
            # approximation that could disagree with what was actually written.
            validation_passed, validation_reasons = _LAST_GATE_OUTCOME
        elif use_val:
            try:
                from backtesting.simulator import split_price_window
                precomp = load_and_precompute(n_days, mode=mode)
                if regime_scope != "all":
                    from backtesting.regime_scope import apply_regime_scope
                    precomp, _ = apply_regime_scope(precomp, regime_scope)
                actual_n = precomp.prices.shape[0]
                train_pct = bp.get("train_pct", 0.70)
                train_sl, val_sl = split_price_window(actual_n, train_pct)
                # Re-validate at the scope the run was tuned at (see run_auto_tune).
                report = run_backtest_report(precomp, avg_params, train_sl, val_sl, scope=scope)
                from backtesting.validator import WalkForwardValidator
                validation_passed, validation_reasons = WalkForwardValidator().validate_report(report, bp)
            except Exception as e:
                logger.warning("Could not re-run validation for AutoTuneResult: %s", e)
                validation_passed = False
                validation_reasons = [str(e)]
        else:
            validation_passed = True

        # Mirror the EXACT predicate run_auto_tune uses before writing (it also
        # writes on auto_apply_if_valid, which the old flag ignored).
        config_written = should_apply_tuned_config(apply, validation_passed, bp, force_apply=force_apply)

        return AutoTuneResult(
            avg_params=avg_params,
            sharpe_params=sharpe_params,
            calmar_params=calmar_params,
            sharpe_result=sharpe_result,
            calmar_result=calmar_result,
            avg_result=avg_result,
            n_days=n_days,
            validation_passed=validation_passed,
            validation_reasons=validation_reasons,
            config_written=config_written,
            active_params=self._active_param_names(scope, preset),
        )

    def apply_params(self, params: np.ndarray) -> None:
        apply_config_params(params)
