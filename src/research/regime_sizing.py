"""Regime-scoped sizing/exposure research helpers.

The evidence so far says per-regime score weights are mostly noise. This module
keeps the useful part: compare regime-conditional exposure/sizing variants on
random windows that are fully inside a selected regime.

Read-only research module: it never writes config or places orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from backtesting.random_walk import (
    RandomWindowSummary,
    WindowResult,
    compute_robust_score,
    random_window_backtest,
)
from backtesting.random_walk import _slice_precomp as slice_window_precomp
from backtesting.regime_scope import eligible_window_starts
from backtesting.simulator import run_simulation
from backtesting.types import BacktestScope, PrecomputedData


@dataclass(frozen=True)
class SizingVariant:
    """One candidate exposure policy for windows in a selected regime."""

    name: str
    index_pct: float
    max_buys: int | None = None


@dataclass(frozen=True)
class SizingResult:
    """Random-window result for one sizing variant."""

    variant: SizingVariant
    summary: RandomWindowSummary

    @property
    def median_excess(self) -> float:
        return self.summary.median_excess_return

    @property
    def pct_beating(self) -> float:
        return self.summary.pct_beating_benchmark

    @property
    def robust_score(self) -> float:
        return self.summary.robust_score


def build_sizing_params(base_params: np.ndarray, variant: SizingVariant) -> np.ndarray:
    """Return a copied param vector with index_pct and optional max_buys changed.

    ``run_simulation`` reads index_pct from slot 4. It reads max_buys from slot 44
    when the vector is long enough, which ``tuning.constants._current_params`` is.
    A short vector can still test index_pct; max_buys is ignored unless present.
    """
    params = np.asarray(base_params, dtype=float).copy()
    params[4] = float(variant.index_pct)
    if variant.max_buys is not None and len(params) > 44:
        params[44] = float(variant.max_buys)
    return params


def default_neutral_sizing_grid(
    current_index_pct: float,
    current_max_buys: int,
) -> list[SizingVariant]:
    """Candidate neutral-regime exposure grid, with current config first.

    The grid leans active exposure up gradually in neutral, where the sleeve has
    shown edge. Defensive is intentionally not included here; it should stay flat
    unless a separate OOS study says otherwise.
    """
    seen: set[tuple[float, int]] = set()
    raw = [
        (float(current_index_pct), int(current_max_buys), "current"),
        (0.70, max(int(current_max_buys), 4), "neutral_idx70"),
        (0.65, max(int(current_max_buys), 5), "neutral_idx65"),
        (0.60, max(int(current_max_buys), 6), "neutral_idx60"),
        (0.55, max(int(current_max_buys), 6), "neutral_idx55"),
        (0.50, max(int(current_max_buys), 8), "neutral_idx50"),
    ]
    out: list[SizingVariant] = []
    for index_pct, max_buys, name in raw:
        key = (round(index_pct, 6), max_buys)
        if key in seen:
            continue
        seen.add(key)
        out.append(SizingVariant(name=name, index_pct=index_pct, max_buys=max_buys))
    return out


def run_regime_sizing_grid(
    precomp: PrecomputedData,
    base_params: np.ndarray,
    variants: Iterable[SizingVariant],
    regime_scope: str = "neutral",
    n_windows: int = 40,
    window_days: int = 45,
    seed: int = 42,
    scope: BacktestScope = "overall_strategy",
) -> list[SizingResult]:
    """Evaluate sizing variants using random windows fully inside regime_scope."""
    results: list[SizingResult] = []
    for i, variant in enumerate(variants):
        params = build_sizing_params(base_params, variant)
        summary = random_window_backtest(
            precomp,
            params,
            n_windows=n_windows,
            window_days=window_days,
            seed=seed + i,
            scope=scope,
            regime_scope=regime_scope,
        )
        results.append(SizingResult(variant=variant, summary=summary))
    return results


def _summary_from_window_results(
    results: list[WindowResult],
    params: np.ndarray,
    window_days: int,
    scope: BacktestScope,
) -> RandomWindowSummary:
    if not results:
        raise RuntimeError("No window results to summarize")

    returns = np.array([r.strategy_return for r in results])
    excess_arr = np.array([r.excess_return for r in results])
    sharpes = np.array([r.sharpe for r in results])
    dds = np.array([r.max_drawdown for r in results])
    calmars = np.array([r.calmar for r in results])
    turns = np.array([r.turnover for r in results])
    benches = np.array([r.benchmark_return for r in results])

    worst_decile_ret = float(np.percentile(returns, 10))
    worst_decile_dd = float(np.percentile(dds, 10))
    pct_beating = float(np.mean([r.wins_benchmark for r in results]))
    std_excess = float(np.std(excess_arr)) if len(excess_arr) > 1 else 0.0
    med_excess = float(np.median(excess_arr))
    med_sharpe = float(np.median(sharpes))
    med_calmar = float(np.median(calmars))
    med_dd = float(np.median(dds))
    med_turnover = float(np.median(turns))
    robust = compute_robust_score(
        med_excess, med_sharpe, pct_beating, worst_decile_dd, med_turnover, std_excess,
    )

    active_median_excess = None
    active_median_sharpe = None
    active_pct_beating = None
    active_worst_dd = None
    active_robust = None
    if scope == "active_sleeve_compounding":
        a_exc = [r.active_excess_return for r in results if r.active_excess_return is not None]
        a_sh = [r.active_sharpe for r in results if r.active_sharpe is not None]
        a_dd = [r.active_drawdown for r in results if r.active_drawdown is not None]
        a_beat = [r.wins_benchmark_active for r in results if r.wins_benchmark_active is not None]
        if a_exc:
            active_median_excess = float(np.median(a_exc))
            active_median_sharpe = float(np.median(a_sh)) if a_sh else 0.0
            active_pct_beating = float(np.mean(a_beat)) if a_beat else 0.0
            active_worst_dd = float(np.percentile(a_dd, 10)) if a_dd else 0.0
            active_robust = (
                active_median_excess
                + 0.50 * active_median_sharpe
                + 0.25 * active_pct_beating
                - 0.50 * abs(active_worst_dd)
                - 0.20 * med_turnover
                - 0.25 * (float(np.std(a_exc)) if len(a_exc) > 1 else 0.0)
            )

    return RandomWindowSummary(
        n_windows=len(results),
        window_days=window_days,
        params_used=params.copy(),
        scope=scope,
        window_results=results,
        median_strategy_return=float(np.median(returns)),
        median_benchmark_return=float(np.median(benches)),
        median_excess_return=med_excess,
        median_sharpe=med_sharpe,
        median_drawdown=med_dd,
        median_calmar=med_calmar,
        median_turnover=med_turnover,
        pct_beating_benchmark=pct_beating,
        worst_decile_return=worst_decile_ret,
        worst_decile_drawdown=worst_decile_dd,
        std_excess_return=std_excess,
        robust_score=robust,
        median_active_excess_return=active_median_excess,
        median_active_sharpe=active_median_sharpe,
        pct_active_beating_benchmark=active_pct_beating,
        worst_decile_active_drawdown=active_worst_dd,
        active_robust_score=active_robust,
    )


def run_single_window(
    precomp: PrecomputedData,
    params: np.ndarray,
    start: int,
    window_days: int,
    starting_capital: float = 10_000.0,
    weekly_contribution: float = 0.0,
    slippage_bps: float = 10.0,
    commission_per_trade: float = 0.0,
    rebalance_frequency_days: int = 5,
    scope: BacktestScope = "overall_strategy",
) -> WindowResult:
    end = int(start) + int(window_days)
    s = slice(int(start), end)
    win_precomp = slice_window_precomp(precomp, s)
    sim = run_simulation(
        win_precomp,
        params,
        starting_capital=starting_capital,
        slippage_bps=slippage_bps,
        commission_per_trade=commission_per_trade,
        weekly_contribution=weekly_contribution,
        rebalance_frequency_days=rebalance_frequency_days,
        scope=scope,
    )
    bench_arr = precomp.benchmark_prices[s]
    if len(bench_arr) >= 2 and np.isfinite(bench_arr).all() and bench_arr[0] > 0:
        bench_ret = float(bench_arr[-1] / bench_arr[0]) - 1.0
    else:
        bench_ret = 0.0
    excess = sim.total_return - bench_ret
    bench_eq = (bench_arr / max(bench_arr[0], 1e-9)) if len(bench_arr) > 0 else None
    return WindowResult(
        window_id=0,
        start_day=int(start),
        end_day=end,
        strategy_return=sim.total_return,
        benchmark_return=bench_ret,
        excess_return=excess,
        sharpe=sim.sharpe,
        max_drawdown=sim.max_drawdown,
        calmar=sim.calmar,
        turnover=sim.turnover_estimate,
        trades=sim.trades_made,
        avg_positions=sim.average_positions,
        wins_benchmark=excess > 0,
        equity_curve=sim.equity_curve.copy() if len(sim.equity_curve) > 0 else None,
        benchmark_equity=bench_eq,
        active_return=sim.active_total_return,
        active_excess_return=sim.active_excess_return,
        active_sharpe=sim.active_sharpe,
        active_drawdown=sim.active_max_drawdown,
        active_equity_curve=(
            sim.active_equity_curve.copy()
            if sim.active_equity_curve is not None and len(sim.active_equity_curve) > 0
            else None
        ),
        wins_benchmark_active=(sim.active_excess_return > 0 if sim.active_excess_return is not None else None),
    )


def sample_regime_window_starts(
    precomp: PrecomputedData,
    window_days: int,
    regime_scope: str = "neutral",
    n_windows: int = 40,
    seed: int = 42,
    segment: str = "all",
    split_day: int | None = None,
) -> np.ndarray:
    """Sample starts once so every sizing variant uses identical windows.

    segment='train' keeps windows ending at or before split_day; segment='holdout'
    keeps windows starting at/after split_day. This gives a temporal OOS guard.
    """
    eligible, _ = eligible_window_starts(precomp, window_days, regime_scope)
    starts = np.asarray(eligible, dtype=int)
    if segment != "all":
        if split_day is None:
            raise ValueError("split_day is required for train/holdout sampling")
        if segment == "train":
            starts = starts[starts + window_days <= split_day]
        elif segment == "holdout":
            starts = starts[starts >= split_day]
        else:
            raise ValueError("segment must be 'all', 'train', or 'holdout'")
    if len(starts) == 0:
        raise ValueError(f"No eligible {segment} starts for regime_scope={regime_scope!r}")
    actual_n = min(int(n_windows), len(starts))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(starts, size=actual_n, replace=False)
    return np.array(sorted(int(x) for x in chosen), dtype=int)


def run_regime_sizing_grid_on_starts(
    precomp: PrecomputedData,
    base_params: np.ndarray,
    variants: Iterable[SizingVariant],
    starts: np.ndarray,
    window_days: int,
    starting_capital: float = 10_000.0,
    weekly_contribution: float = 0.0,
    slippage_bps: float = 10.0,
    commission_per_trade: float = 0.0,
    rebalance_frequency_days: int = 5,
    scope: BacktestScope = "overall_strategy",
) -> list[SizingResult]:
    """Evaluate all variants on the exact same start indices for paired deltas."""
    results: list[SizingResult] = []
    starts_arr = np.asarray(starts, dtype=int)
    for variant in variants:
        params = build_sizing_params(base_params, variant)
        window_results: list[WindowResult] = []
        for window_id, start in enumerate(starts_arr):
            wr = run_single_window(
                precomp,
                params,
                int(start),
                window_days,
                starting_capital=starting_capital,
                weekly_contribution=weekly_contribution,
                slippage_bps=slippage_bps,
                commission_per_trade=commission_per_trade,
                rebalance_frequency_days=rebalance_frequency_days,
                scope=scope,
            )
            wr.window_id = window_id
            window_results.append(wr)
        results.append(SizingResult(
            variant=variant,
            summary=_summary_from_window_results(window_results, params, window_days, scope),
        ))
    return results


def result_rows(results: Iterable[SizingResult]) -> list[dict]:
    """Serialize results to plain rows for CLI printing / CSV writing."""
    rows: list[dict] = []
    for result in results:
        s = result.summary
        rows.append({
            "variant": result.variant.name,
            "index_pct": result.variant.index_pct,
            "active_pct": 1.0 - result.variant.index_pct,
            "max_buys": result.variant.max_buys,
            "n_windows": s.n_windows,
            "window_days": s.window_days,
            "median_excess": s.median_excess_return,
            "pct_beating": s.pct_beating_benchmark,
            "median_sharpe": s.median_sharpe,
            "median_drawdown": s.median_drawdown,
            "robust_score": s.robust_score,
            "median_strategy_return": s.median_strategy_return,
            "median_benchmark_return": s.median_benchmark_return,
        })
    return rows
