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
    candidate_vectors: dict[str, np.ndarray],
    train_sl,
    val_slice,
    scope: str,
    backtest_cfg: dict,
    incumbent_report: BacktestReport,
) -> list[dict]:
    """Evaluate every candidate on the same split and gate each with
    validate_tuned_params (absolute + incumbent-relative + turnover). Returns
    rows of {candidate_id, params, report, metrics, gates_passed, reasons,
    score}; score is -inf when the candidate has no validation metrics."""
    rows: list[dict] = []
    for cid, params in candidate_vectors.items():
        report = run_backtest_report(precomp, params, train_sl, val_slice, scope=scope)
        passed, reasons = validate_tuned_params(report, backtest_cfg, incumbent_report)
        metrics = _candidate_metrics(report, incumbent_report)
        rows.append({
            "candidate_id": cid,
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
        f"\n{'candidate_id':<20} {'val_exc':>8} {'vs_inc':>8} {'sharpe':>7} "
        f"{'calmar':>7} {'maxDD':>7} {'turn':>6} {'gates':>6} {'score':>8}"
    )
    for r in rows:
        m = r.get("metrics")
        mark = " ◀ selected" if r["candidate_id"] == selected_id else ""
        if m is None:
            print(f"{r['candidate_id']:<20} {'—':>8} {'—':>8} {'—':>7} {'—':>7} "
                  f"{'—':>7} {'—':>6} {'FAIL':>6} {'—':>8}{mark}")
            continue
        print(
            f"{r['candidate_id']:<20} {m['val_excess']:>+8.2%} {m['val_excess_vs_incumbent']:>+8.2%} "
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

    # ── Candidate tournament ──────────────────────────────────────────────────
    # Replaces the blind (sharpe + calmar) / 2 midpoint: the optima, three
    # blends, and three incumbent blends are each evaluated on the same split,
    # gated individually, Pareto-filtered, and ranked by the selection score.
    candidate_vectors = build_tournament_candidates(sharpe_params, calmar_params, incumbent_params)
    rows = run_candidate_tournament(
        precomp, candidate_vectors, train_sl, val_slice, scope, bp, incumbent_report,
    )
    passers = [r for r in rows if r["gates_passed"] and r["metrics"] is not None]
    if passers:
        non_dominated = _pareto_non_dominated(passers)
        pool = [r for r in passers if r["candidate_id"] in non_dominated] or passers
        selected = max(pool, key=lambda r: r["score"])
        selection_note = "highest validation robust score among candidates passing gates."
    else:
        # No passer: keep the legacy midpoint as the returned vector for caller
        # compatibility (nothing is written — gates failed). Prefer the best-
        # scoring row's diagnostics when metrics exist.
        diagnostic = max(rows, key=lambda r: r["score"])
        selected = next(r for r in rows if r["candidate_id"] == "avg_50_50")
        if np.isfinite(diagnostic["score"]):
            selected = diagnostic
        selection_note = "none passed gates — best-scoring candidate kept for diagnostics only."
    selected_id = selected["candidate_id"] if passers else None
    _print_tournament_table(rows, selected_id)
    print(f"\nSelected candidate: {selected['candidate_id'] if passers else 'none (gates failed)'}")
    print(f"Reason: {selection_note}")

    selected_params = selected["params"]
    train_report = selected["report"]
    selected_result = run_simulation(
        tune_precomp, selected_params, starting_capital,
        slippage_bps=bp["slippage_bps"],
        commission_per_trade=bp["commission_per_trade"],
        weekly_contribution=bp["weekly_contribution"],
        rebalance_frequency_days=bp["rebalance_frequency_days"],
    )

    if use_val and train_report.validation_result is not None and incumbent_report.validation_result is not None:
        _t, _i = train_report, incumbent_report
        _t_exc = _t.validation_result.total_return - _t.validation_benchmark_return
        _i_exc = _i.validation_result.total_return - _i.validation_benchmark_return
        print(
            f"\nValidation window (excess vs SPY):  selected {_t_exc:+.2%}  vs  incumbent {_i_exc:+.2%}"
            f"   |  turnover: selected {_t.validation_result.turnover_estimate:.2f}"
            f" vs incumbent {_i.validation_result.turnover_estimate:.2f}"
        )

    validation_passed, reasons = selected["gates_passed"], list(selected["reasons"])

    # Reproducibility gate: paired random windows on longer, mostly-unseen
    # history — run ONLY on the selected candidate (never on failed ones).
    if validation_passed:
        gw_cfg = bp.get("random_window_gate", {}) or {}
        if gw_cfg.get("enabled", True):
            print(
                f"\nRandom-window gate: {gw_cfg.get('n_windows', 12)}x"
                f"{gw_cfg.get('window_days', 120)}d paired windows over "
                f"{gw_cfg.get('history_days', 730)}d history …"
            )
            rw_passed, rw_reasons, rw_stats = paired_random_window_gate(
                selected_params, incumbent_params, bp, mode=mode, scope=scope,
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
            candidates = [
                {
                    "candidate_id": "sharpe_opt",
                    "alpha_params": dict(zip(PARAM_NAMES, sharpe_params.tolist())),
                    "train_return": sharpe_result.total_return,
                    "train_sharpe": sharpe_result.sharpe,
                    "train_calmar": sharpe_result.calmar,
                    "train_max_drawdown": sharpe_result.max_drawdown,
                    "train_trades": sharpe_result.trades_made,
                    "train_avg_positions": sharpe_result.average_positions,
                    "train_max_positions": sharpe_result.max_positions,
                    "train_avg_cash_pct": sharpe_result.average_cash_pct,
                    "train_turnover": sharpe_result.turnover_estimate,
                    "train_friction_cost": sharpe_result.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                },
                {
                    "candidate_id": "calmar_opt",
                    "alpha_params": dict(zip(PARAM_NAMES, calmar_params.tolist())),
                    "train_return": calmar_result.total_return,
                    "train_sharpe": calmar_result.sharpe,
                    "train_calmar": calmar_result.calmar,
                    "train_max_drawdown": calmar_result.max_drawdown,
                    "train_trades": calmar_result.trades_made,
                    "train_avg_positions": calmar_result.average_positions,
                    "train_max_positions": calmar_result.max_positions,
                    "train_avg_cash_pct": calmar_result.average_cash_pct,
                    "train_turnover": calmar_result.turnover_estimate,
                    "train_friction_cost": calmar_result.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                },
                {
                    "candidate_id": f"selected:{selected['candidate_id']}",
                    "alpha_params": dict(zip(PARAM_NAMES, selected_params.tolist())),
                    "train_return": tr.total_return,
                    "train_sharpe": tr.sharpe,
                    "train_calmar": tr.calmar,
                    "train_max_drawdown": tr.max_drawdown,
                    "train_trades": tr.trades_made,
                    "train_avg_positions": tr.average_positions,
                    "train_max_positions": tr.max_positions,
                    "train_avg_cash_pct": tr.average_cash_pct,
                    "train_turnover": tr.turnover_estimate,
                    "train_friction_cost": tr.friction_cost,
                    "val_return": vr.total_return if vr else None,
                    "val_sharpe": vr.sharpe if vr else None,
                    "val_max_drawdown": vr.max_drawdown if vr else None,
                    "bench_return": train_report.benchmark_return,
                    "bench_sharpe": train_report.benchmark_sharpe,
                    "bench_max_drawdown": train_report.benchmark_max_drawdown,
                    "excess_return": train_report.excess_return,
                    "lookahead_bias_level": precomp.lookahead_bias_level,
                    "notes": train_report.notes,
                },
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
