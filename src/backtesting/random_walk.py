"""
backtesting/random_walk.py — Randomized walk-forward backtest engine.

Samples N random windows of M trading days from a pre-loaded PrecomputedData,
runs the strategy simulation on each, and computes a robust_score that rewards
consistency across many market regimes rather than one lucky historical window.

Public API
----------
random_window_backtest(precomp, params, n_windows, window_days, seed, ...) -> RandomWindowSummary
compute_robust_score(summary) -> float

robust_score formula
---------------------
  robust_score =
      median_excess_return
    + W_SHARPE      * median_sharpe
    + W_PCT_BEATING * pct_beating_benchmark
    - W_DRAWDOWN    * abs(worst_decile_drawdown)
    - W_TURNOVER    * median_turnover
    - W_STD_EXCESS  * std_excess_return

  Rewards:  excess return vs SPY (the dominant term), risk-adjusted performance,
  consistency. Penalizes: deep drawdowns, high turnover, return volatility.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .regime_scope import eligible_window_starts, regime_labels, slice_precomp
from .simulator import run_simulation
from .types import BacktestScope, PrecomputedData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Metrics for a single random backtest window."""
    window_id: int
    start_day: int
    end_day: int
    strategy_return: float
    benchmark_return: float
    excess_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    turnover: float
    trades: int
    avg_positions: float
    wins_benchmark: bool
    # Equity curves — stored for fan chart visualization
    equity_curve: np.ndarray | None = None
    benchmark_equity: np.ndarray | None = None
    # Active sleeve metrics — populated when scope == "active_sleeve_compounding"
    active_return: float | None = None
    active_excess_return: float | None = None
    active_sharpe: float | None = None
    active_drawdown: float | None = None
    active_equity_curve: np.ndarray | None = None
    wins_benchmark_active: bool | None = None


@dataclass
class RandomWindowSummary:
    """Aggregated results across all random windows."""
    n_windows: int
    window_days: int
    params_used: np.ndarray
    scope: BacktestScope = "overall_strategy"
    window_results: list[WindowResult] = field(default_factory=list)
    # Per-metric summaries
    median_strategy_return: float = 0.0
    median_benchmark_return: float = 0.0
    median_excess_return: float = 0.0
    median_sharpe: float = 0.0
    median_drawdown: float = 0.0
    median_calmar: float = 0.0
    median_turnover: float = 0.0
    pct_beating_benchmark: float = 0.0
    worst_decile_return: float = 0.0
    worst_decile_drawdown: float = 0.0
    std_excess_return: float = 0.0
    robust_score: float = 0.0
    # Active sleeve aggregate metrics — populated when scope == "active_sleeve_compounding"
    median_active_excess_return: float | None = None
    median_active_sharpe: float | None = None
    pct_active_beating_benchmark: float | None = None
    worst_decile_active_drawdown: float | None = None
    active_robust_score: float | None = None

    def to_dataframe(self) -> pd.DataFrame:
        if not self.window_results:
            return pd.DataFrame()
        rows = [
            {
                "window_id":        w.window_id,
                "start_day":        w.start_day,
                "end_day":          w.end_day,
                "strategy_return":  w.strategy_return,
                "benchmark_return": w.benchmark_return,
                "excess_return":    w.excess_return,
                "sharpe":           w.sharpe,
                "calmar":           w.calmar,
                "max_drawdown":     w.max_drawdown,
                "turnover":         w.turnover,
                "trades":           w.trades,
                "avg_positions":    w.avg_positions,
                "beats_benchmark":  w.wins_benchmark,
            }
            for w in self.window_results
        ]
        return pd.DataFrame(rows)

    def summary_dict(self) -> dict:
        return {
            "n_windows":           self.n_windows,
            "window_days":         self.window_days,
            "median_return":       f"{self.median_strategy_return:+.1%}",
            "median_benchmark":    f"{self.median_benchmark_return:+.1%}",
            "median_excess":       f"{self.median_excess_return:+.1%}",
            "median_sharpe":       f"{self.median_sharpe:.3f}",
            "median_drawdown":     f"{self.median_drawdown:.1%}",
            "pct_beating":         f"{self.pct_beating_benchmark:.0%}",
            "worst_decile_return": f"{self.worst_decile_return:+.1%}",
            "worst_decile_dd":     f"{self.worst_decile_drawdown:.1%}",
            "std_excess":          f"{self.std_excess_return:.3f}",
            "robust_score":        f"{self.robust_score:.4f}",
        }


# ---------------------------------------------------------------------------
# Robust score
# ---------------------------------------------------------------------------

# Term weights — terms must be in COMPARABLE UNITS so median excess-vs-SPY dominates
# the ranking (project rule: results are judged on excess vs SPY, never isolated
# Sharpe/Calmar). On real 45-60d windows median_excess is a per-window return ~±0.05
# while annualized short-window sharpe is ~±2: the old 0.50 sharpe weight made the
# score ~95% Sharpe. 0.05 * sharpe (~±0.10) and 0.10 * pct_beating (~0-0.10) keep
# both as tie-breakers of the same magnitude as the excess term, not its master.
ROBUST_W_SHARPE      = 0.05
ROBUST_W_PCT_BEATING = 0.10
ROBUST_W_DRAWDOWN    = 0.50
ROBUST_W_TURNOVER    = 0.25
ROBUST_W_STD_EXCESS  = 0.25


def compute_robust_score(
    median_excess: float,
    median_sharpe: float,
    pct_beating: float,
    worst_decile_dd: float,
    median_turnover: float,
    std_excess: float,
) -> float:
    """
    Combines multiple robustness signals into a single scalar. Higher is better.

    median excess-vs-SPY is the dominant reward term; sharpe and pct_beating are
    deliberately down-weighted (ROBUST_W_SHARPE / ROBUST_W_PCT_BEATING) so they act
    as same-magnitude tie-breakers rather than swamping a per-window return ~±0.05
    with an annualized Sharpe ~±2 — see the weight-constant comment above.
    """
    return (
        median_excess
        + ROBUST_W_SHARPE      * median_sharpe
        + ROBUST_W_PCT_BEATING * pct_beating
        - ROBUST_W_DRAWDOWN    * abs(worst_decile_dd)
        - ROBUST_W_TURNOVER    * median_turnover
        - ROBUST_W_STD_EXCESS  * std_excess
    )


# ---------------------------------------------------------------------------
# Precomp window slicer — the ONE canonical implementation lives in regime_scope.
# Re-exported under the old private name for callers (research/regime_sizing).
# ---------------------------------------------------------------------------

_slice_precomp = slice_precomp


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def random_window_backtest(
    precomp: PrecomputedData,
    params: np.ndarray,
    n_windows: int = 20,
    window_days: int = 60,
    seed: int = 42,
    starting_capital: float = 10_000.0,
    weekly_contribution: float = 0.0,
    slippage_bps: float = 10.0,
    commission_per_trade: float = 0.0,
    rebalance_frequency_days: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
    scope: BacktestScope = "overall_strategy",
    regime_scope: str = "all",
) -> RandomWindowSummary:
    """
    Sample n_windows random sub-windows of window_days each from precomp and
    run the strategy simulation on each. Returns aggregated metrics and
    per-window results.

    Args:
        precomp:                  Precomputed data (must have >= window_days + 1 rows).
        params:                   15-element strategy parameter vector.
        n_windows:                Number of random windows to evaluate.
        window_days:              Length of each window in trading days.
        seed:                     Random seed for reproducibility.
        starting_capital:         Capital deployed at the start of each window.
        weekly_contribution:      Cash added each rebalance cycle.
        slippage_bps:             Slippage in basis points per trade.
        commission_per_trade:     Fixed commission per trade.
        rebalance_frequency_days: Days between rebalance cycles.
        progress_callback:        Optional callable(current, total) for UI progress.

    Returns:
        RandomWindowSummary with per-window results and aggregate statistics.
    """
    n_total = precomp.prices.shape[0]

    if params is None:
        from backtesting.simulator import get_default_params
        params = get_default_params()

    if n_total < window_days + 1:
        raise ValueError(
            f"PrecomputedData has only {n_total} days; need at least {window_days + 1} "
            f"for window_days={window_days}."
        )

    rng = np.random.default_rng(seed)
    eligible_starts, _regime_meta = eligible_window_starts(precomp, window_days, regime_scope)
    # ALWAYS attach point-in-time regime labels computed on the FULL load before slicing.
    # Short sliced windows have no 200DMA history of their own, so without precomputed
    # labels every day of every window would default to "bullish" — structurally
    # disabling regime tilts/defensive logic in exactly this harness.
    if precomp.regime_labels_daily is None:
        precomp = precomp._replace(regime_labels_daily=regime_labels(precomp))
    n_possible = len(eligible_starts)

    if n_possible <= 0:
        raise ValueError(
            f"No valid start indices for window_days={window_days} with {n_total} total days "
            f"and regime_scope={regime_scope!r}."
        )

    actual_n = min(n_windows, n_possible)
    if actual_n < n_windows:
        logger.warning(
            "Only %d unique start positions available (requested %d windows). Using %d.",
            n_possible, n_windows, actual_n,
        )

    # Sample without replacement from eligible starts; then wrap if more requested.
    replace = actual_n > n_possible
    raw_positions = rng.choice(n_possible, size=actual_n, replace=replace)
    start_indices = sorted(int(eligible_starts[pos]) for pos in set(int(p) for p in raw_positions))[:actual_n]

    results: list[WindowResult] = []

    for wid, start in enumerate(start_indices):
        if progress_callback is not None:
            progress_callback(wid, len(start_indices))

        end = start + window_days
        s   = slice(start, end)

        try:
            win_precomp = _slice_precomp(precomp, s)
            sim = run_simulation(
                win_precomp, params,
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

            # Build indexed benchmark equity for this window
            bench_win = precomp.benchmark_prices[s]
            bench_eq = (bench_win / max(bench_win[0], 1e-9)) if len(bench_win) > 0 else None

            results.append(WindowResult(
                window_id=wid,
                start_day=start,
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
                wins_benchmark_active=(
                    sim.active_excess_return > 0 if sim.active_excess_return is not None else None
                ),
            ))
        except Exception as exc:
            logger.warning("Window %d (days %d–%d) failed: %s", wid, start, end, exc)
            continue

    if not results:
        raise RuntimeError("All random windows failed. Check data quality and window_days vs n_total.")

    returns    = np.array([r.strategy_return for r in results])
    excess_arr = np.array([r.excess_return   for r in results])
    sharpes    = np.array([r.sharpe          for r in results])
    dds        = np.array([r.max_drawdown    for r in results])
    calmars    = np.array([r.calmar          for r in results])
    turns      = np.array([r.turnover        for r in results])
    benches    = np.array([r.benchmark_return for r in results])

    worst_decile_ret = float(np.percentile(returns,    10))
    worst_decile_dd  = float(np.percentile(dds,        10))
    pct_beating      = float(np.mean([r.wins_benchmark for r in results]))
    std_excess       = float(np.std(excess_arr)) if len(excess_arr) > 1 else 0.0

    med_excess   = float(np.median(excess_arr))
    med_sharpe   = float(np.median(sharpes))
    med_calmar   = float(np.median(calmars))
    med_dd       = float(np.median(dds))
    med_turnover = float(np.median(turns))

    robust = compute_robust_score(
        med_excess, med_sharpe, pct_beating,
        worst_decile_dd, med_turnover, std_excess,
    )

    _active_median_excess: float | None = None
    _active_median_sharpe: float | None = None
    _active_pct_beating:   float | None = None
    _active_worst_dd:      float | None = None
    _active_robust:        float | None = None

    if scope == "active_sleeve_compounding":
        a_exc  = [r.active_excess_return for r in results if r.active_excess_return is not None]
        a_sh   = [r.active_sharpe        for r in results if r.active_sharpe        is not None]
        a_dd   = [r.active_drawdown      for r in results if r.active_drawdown      is not None]
        a_beat = [r.wins_benchmark_active for r in results if r.wins_benchmark_active is not None]
        if a_exc:
            _active_median_excess = float(np.median(a_exc))
            _active_median_sharpe = float(np.median(a_sh)) if a_sh else 0.0
            _active_pct_beating   = float(np.mean(a_beat)) if a_beat else 0.0
            _active_worst_dd      = float(np.percentile(a_dd, 10)) if a_dd else 0.0
            # Same excess-dominant weighting as the overall robust score.
            _active_robust = compute_robust_score(
                _active_median_excess,
                _active_median_sharpe,
                _active_pct_beating,
                _active_worst_dd,
                med_turnover,
                float(np.std(a_exc)) if len(a_exc) > 1 else 0.0,
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
        median_active_excess_return=_active_median_excess,
        median_active_sharpe=_active_median_sharpe,
        pct_active_beating_benchmark=_active_pct_beating,
        worst_decile_active_drawdown=_active_worst_dd,
        active_robust_score=_active_robust,
    )
