# backtest.py — compatibility shim. Import from backtesting.* instead.
from backtesting.types import (
    BacktestReport,
    CandidatePoolDiagnostics,
    PrecomputedData,
    SimResult,
)
from backtesting.data_loader import load_and_precompute, select_backtest_universe
from backtesting.simulator import (
    compare_candidate_selection_modes,
    compute_performance_metrics,
    get_default_params,
    print_pool_diagnostics,
    run_backtest_report,
    run_simulation,
    score_stocks,
    score_stocks_at_day,
    select_candidates,
    split_price_window,
)
from backtesting.reports import print_backtest_report, print_comparison_report

# Re-exported for legacy callers (e.g. tests.py does `from backtest import MOMENTUM_PARAMS`)
from util import MOMENTUM_PARAMS

__all__ = [
    "BacktestReport", "CandidatePoolDiagnostics", "PrecomputedData", "SimResult",
    "load_and_precompute", "select_backtest_universe",
    "compare_candidate_selection_modes", "compute_performance_metrics",
    "get_default_params", "print_pool_diagnostics", "run_backtest_report",
    "run_simulation", "score_stocks", "score_stocks_at_day", "select_candidates",
    "split_price_window", "print_backtest_report", "print_comparison_report",
    "MOMENTUM_PARAMS",
]
