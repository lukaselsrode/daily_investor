"""
cli/commands.py — Command handler functions.

Each command is a standalone function that accepts parsed args and
delegates to the appropriate service. No business logic lives here.

Commands:
  run              → main.run_daily_strat()            (live trading)
  backtest DAYS    → backtesting.BacktestEngine.run()
  auto-tune [DAYS] → tuning.ParameterTuner.auto_tune()
  tune DAYS        → tuning.ParameterTuner.tune()
  stability-scan   → tuning.StabilityAnalyzer.scan()
  report           → backtesting.BacktestEngine + stability-scan hint
"""

from __future__ import annotations

import logging
from typing import Optional

import tuner as _t

logger = logging.getLogger(__name__)


def cmd_fetch_data() -> None:
    """
    Fetch all market data and save CSVs + snapshot — no trades placed.

    Pipeline:
      1. Industry valuation benchmarks  (ratios.yaml)
      2. Dividends                       (dividends CSV)
      3. Holdings                        (holdings CSV — current positions + enriched open dates)
      4. Universe + fundamentals + news  (stock_tickers, robinhood_data, news, agg_data CSVs)
      5. Parquet snapshot                (data/snapshots/YYYY_MM_DD.parquet for IC analysis)

    Requires Robinhood login.
    """
    from main import (
        login,
        update_industry_valuations,
        _fetch_and_save_dividends,
        get_current_positions,
        _enrich_holdings_with_created_at,
        save_holdings_csv,
    )
    from source_data import get_data as generate_daily_undervalued_stocks

    login()
    logger.info("=== Fetch-Data run (no trades) ===")

    logger.info("Step 1/4: industry valuations")
    update_industry_valuations(verbose=True)

    logger.info("Step 2/4: dividends")
    _fetch_and_save_dividends()

    logger.info("Step 3/4: holdings")
    try:
        holdings = get_current_positions()
        _enrich_holdings_with_created_at(holdings)
        save_holdings_csv(holdings)
        logger.info("Holdings saved: %d positions", len(holdings))
    except Exception as exc:
        logger.warning("Holdings fetch failed (continuing): %s", exc)

    logger.info("Step 4/4: universe + fundamentals + news + scoring")
    df = generate_daily_undervalued_stocks(refresh=True)
    if df.empty:
        logger.error("Data fetch returned an empty DataFrame — check credentials and connectivity")
    else:
        logger.info("Fetch-Data complete: %d symbols written to agg_data CSV + snapshot", len(df))


def cmd_run(
    skip_data: bool = False,
    op_mode: Optional[str] = None,
) -> None:
    """Live trading run."""
    if op_mode:
        from main import _apply_op_mode
        _apply_op_mode(op_mode)
    from main import login, run_daily_strat
    login()
    run_daily_strat()


def cmd_backtest(
    n_days: int,
    mode: Optional[str] = None,
    params: Optional[dict] = None,
) -> None:
    """Run a single backtest and print results."""
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run(n_days=n_days, params=params, mode=mode)
    print(result)


def cmd_tune(
    n_days: int,
    objective: str = "sharpe",
    mode: Optional[str] = None,
) -> None:
    """Single-objective tune — prints diff, does NOT write config."""
    from tuning.tuner import ParameterTuner
    tuner = ParameterTuner()
    result = tuner.tune(n_days=n_days, objective=objective, mode=mode)
    _t.print_config_diff(result.params, result.sim)


def cmd_auto_tune(
    n_days: int = 90,
    mode: Optional[str] = None,
    apply: bool = False,
    force_apply: bool = False,
    llm_review: bool = False,
) -> None:
    """Dual-objective auto-tune with walk-forward validation."""
    from tuning.tuner import ParameterTuner
    tuner = ParameterTuner()
    result = tuner.auto_tune(
        n_days=n_days,
        apply=apply,
        force_apply=force_apply,
        mode=mode,
        llm_review=llm_review,
    )
    _t._diff_table(
        result.avg_params,
        label=f"mean of Sharpe + Calmar over {n_days}d",
        sharpe_ref=result.sharpe_result,
        calmar_ref=result.calmar_result,
        sharpe_params=result.sharpe_params,
        calmar_params=result.calmar_params,
    )
    print(
        f"\nAveraged result:  ret={result.avg_result.total_return:+.1%}  "
        f"sharpe={result.avg_result.sharpe:+.3f}  "
        f"calmar={result.avg_result.calmar:+.3f}  "
        f"trades={result.avg_result.trades_made}"
    )
    print(result.summary())


def cmd_stability_scan(
    windows: Optional[list[int]] = None,
    mode: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Stability scan — RESEARCH ONLY, never writes config."""
    from tuning.stability import StabilityAnalyzer
    analyzer = StabilityAnalyzer()
    result = analyzer.scan(windows=windows, mode=mode, output_dir=output_dir)
    print(result.summary())


def cmd_report(output_dir: str = "reports") -> None:
    """Run a quick 90-day backtest and print results."""
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run(n_days=90)
    print(result)
    print(f"\nFor full diagnostics, run: stability-scan --output-dir {output_dir}")
