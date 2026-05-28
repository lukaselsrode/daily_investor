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

import tuning.reports as _t

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
      6. Outcome backfill                (decision_outcomes.parquet — if fill_returns_on_run is set)

    Requires Robinhood login.
    """
    from data.market import get_data as generate_daily_undervalued_stocks
    from data.valuation import update_industry_valuations
    from main import (
        _broker,
        _fetch_and_save_dividends,
        _maybe_fill_outcomes,
        login,
        save_holdings_csv,
    )

    login()
    logger.info("=== Fetch-Data run (no trades) ===")

    logger.info("Step 1/5: industry valuations")
    update_industry_valuations(verbose=True)

    logger.info("Step 2/5: dividends")
    _fetch_and_save_dividends()

    logger.info("Step 3/5: holdings")
    try:
        holdings = _broker.get_holdings()
        _broker.enrich_holdings_created_at(holdings)
        save_holdings_csv(holdings)
        logger.info("Holdings saved: %d positions", len(holdings))
    except Exception as exc:
        logger.warning("Holdings fetch failed (continuing): %s", exc)

    logger.info("Step 4/5: universe + fundamentals + news + scoring")
    df = generate_daily_undervalued_stocks(refresh=True)
    if df.empty:
        logger.error("Data fetch returned an empty DataFrame — check credentials and connectivity")
    else:
        logger.info("Fetch-Data complete: %d symbols written to agg_data CSV + snapshot", len(df))

    logger.info("Step 5/5: outcome backfill")
    _maybe_fill_outcomes()


def cmd_run(
    skip_data: bool = False,
    op_mode: str | None = None,
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
    mode: str | None = None,
    params: dict | None = None,
    compare: bool = False,
    archetype_compare: bool = False,
    scope: str = "overall_strategy",
) -> None:
    """Run a single backtest and print results.

    --scope overall_strategy        Tests the full deployed ETF + active portfolio,
                                    including harvest-to-ETF routing and total drawdown.
    --scope active_sleeve_compounding  Tests whether the stock-picking engine compounds
                                    when all active proceeds are recycled into future picks.
    """
    from backtesting.data_loader import load_and_precompute
    from backtesting.engine import BacktestEngine
    from backtesting.reports import print_backtest_report, print_comparison_report
    from backtesting.simulator import (
        compare_archetype_modes,
        compare_candidate_selection_modes,
        get_default_params,
    )

    engine = BacktestEngine()

    if archetype_compare:
        precomp = load_and_precompute(n_days, mode=mode)
        default_params = get_default_params()
        result = compare_archetype_modes(precomp, default_params)
        _print_archetype_comparison(result, n_days)
    elif compare:
        precomp = load_and_precompute(n_days, mode=mode)
        default_params = get_default_params()
        comparison = compare_candidate_selection_modes(precomp, default_params)
        print_comparison_report(comparison)
    else:
        result = engine.run(n_days=n_days, params=params, mode=mode, scope=scope)
        print_backtest_report(result.report)


def _print_archetype_comparison(result: dict, n_days: int) -> None:
    """Print a side-by-side uniform vs archetype-aware comparison table."""
    u   = result["uniform"]
    a   = result["archetype_aware"]
    d   = result["_delta"]
    bench = result["_benchmark_return"]
    actual_days = result["_n_days"]

    def _fmt_pct(v):  return f"{v:+.2%}" if v is not None else "n/a"
    def _fmt_f(v):    return f"{v:+.3f}" if v is not None else "n/a"
    def _fmt_d(v, pct=True): return (_fmt_pct(v) if pct else _fmt_f(v)) if v is not None else "n/a"

    print(f"\n{'='*60}")
    print(f"  Archetype A/B comparison — {n_days}d window ({actual_days} trading days)")
    print(f"  Benchmark return: {bench:+.2%}")
    print(f"{'='*60}")
    print(f"  {'Metric':<22}  {'Uniform':>10}  {'Arch-Aware':>10}  {'Delta':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")

    rows = [
        ("Total return",  _fmt_pct(u.total_return),   _fmt_pct(a.total_return),   _fmt_d(d["total_return"])),
        ("Sharpe",        _fmt_f(u.sharpe),            _fmt_f(a.sharpe),           _fmt_d(d["sharpe"], pct=False)),
        ("Calmar",        _fmt_f(u.calmar),            _fmt_f(a.calmar),           _fmt_d(d["calmar"], pct=False)),
        ("Max drawdown",  _fmt_pct(u.max_drawdown),    _fmt_pct(a.max_drawdown),   _fmt_d(d["max_drawdown"])),
        ("Trades made",   str(u.trades_made),          str(a.trades_made),         str(d["trades_made"] or "")),
    ]
    for label, uv, av, dv in rows:
        print(f"  {label:<22}  {uv:>10}  {av:>10}  {dv:>10}")

    print(f"{'='*60}\n")


def cmd_tune(
    n_days: int,
    objective: str = "sharpe",
    mode: str | None = None,
    scope: str = "overall_strategy",
) -> None:
    """Single-objective tune — prints diff, does NOT write config."""
    from tuning.tuner import ParameterTuner
    tuner = ParameterTuner()
    result = tuner.tune(n_days=n_days, objective=objective, mode=mode, scope=scope)
    _t.print_config_diff(result.params, result.sim)


def cmd_auto_tune(
    n_days: int = 90,
    mode: str | None = None,
    apply: bool = False,
    force_apply: bool = False,
    llm_review: bool = False,
    scope: str = "overall_strategy",
) -> None:
    """Dual-objective auto-tune with walk-forward validation.

    --scope active_sleeve_compounding  freezes index_pct and ETF routing params,
    optimizes only stock-picking parameters, ranks by active sleeve metrics.
    """
    from tuning.tuner import ParameterTuner
    tuner = ParameterTuner()
    result = tuner.auto_tune(
        n_days=n_days,
        apply=apply,
        force_apply=force_apply,
        mode=mode,
        llm_review=llm_review,
        scope=scope,
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
    windows: list[int] | None = None,
    mode: str | None = None,
    output_dir: str | None = None,
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


def cmd_update_outcomes() -> None:
    """
    Backfill future return outcomes for past decisions in decision_outcomes.parquet.

    Fetches current prices + SPY price history via yfinance, then fills:
      future_7d_return / future_30d_return / future_90d_return
      future_7d_vs_spy / future_30d_vs_spy / future_90d_vs_spy
      outperformed_hold / premature_exit / bad_hold / good_trim / good_exit

    Only fills rows where enough calendar time has elapsed.
    NEVER modifies live factor weights or decision logic.
    """
    import datetime
    import logging

    import yfinance as yf

    from portfolio.outcome_tracker import fill_future_returns, load_outcomes

    log = logging.getLogger(__name__)
    log.info("=== update-outcomes: backfilling realized outcomes ===")

    df = load_outcomes()
    if df.empty:
        print("No decision outcomes recorded yet. Run the bot first.")
        return

    # Collect unique symbols that need outcome backfill
    sym_col = "symbol" if "symbol" in df.columns else "ticker"
    symbols = [s for s in df[sym_col].dropna().unique() if str(s).strip()]
    if not symbols:
        print("No symbols found in outcome log.")
        return

    # Add SPY for benchmark comparison
    fetch_syms = sorted(set(symbols) | {"SPY"})
    print(f"Fetching prices for {len(fetch_syms)} symbols...")

    # Download last 120 days of history to cover all horizons
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=125)).isoformat()
    try:
        hist = yf.download(fetch_syms, start=start, auto_adjust=True, progress=False)
        close = hist["Close"] if "Close" in hist.columns else hist
    except Exception as exc:
        print(f"Price download failed: {exc}")
        return

    current_prices: dict[str, float] = {}
    spy_price_history: dict[str, float] = {}
    spy_current_price: float | None = None

    for sym in fetch_syms:
        try:
            col = close[sym] if sym in close.columns else None
            if col is None or col.dropna().empty:
                continue
            current_prices[sym] = float(col.dropna().iloc[-1])
        except Exception:
            continue

    # Build SPY daily history dict for VS-SPY computation
    if "SPY" in close.columns:
        spy_series = close["SPY"].dropna()
        spy_current_price = float(spy_series.iloc[-1])
        for ts, px in spy_series.items():
            date_str = str(ts)[:10]   # YYYY-MM-DD
            spy_price_history[date_str] = float(px)

    n_updated = fill_future_returns(
        current_prices=current_prices,
        spy_current_price=spy_current_price,
        spy_price_history=spy_price_history,
    )

    print(f"update-outcomes complete: {n_updated} outcome cells filled across {len(df)} recorded decisions.")
    log.info("update-outcomes: %d cells updated", n_updated)


def cmd_factor_map(
    method: str = "pca",
    color_by: str | None = None,
    kmeans_clusters: int | None = None,
    output: str | None = None,
    owned_only: bool = False,
    actions: list[str] | None = None,
    sectors: list[str] | None = None,
    show: bool = False,
) -> None:
    """
    Build a 3-D factor-map of the scored universe and save to HTML.

    Loads today's agg_data, merges owned/equity from holdings, then runs
    PCA (or UMAP) dimensionality reduction and optional KMeans clustering.

    SAFE: read-only.  Never modifies config, factor scores, or portfolio state.
    """
    from portfolio.visualization.factor_map import build_factor_map, load_universe_with_holdings

    df = load_universe_with_holdings()
    logger.info("factor-map: loaded %d symbols", len(df))

    out_path = output or "reports/factor_map.html"
    color    = "cluster" if kmeans_clusters else color_by

    fig, df_out, diags = build_factor_map(
        df,
        method=method,
        color_by=color,
        kmeans_clusters=kmeans_clusters,
        owned_only=owned_only,
        actions=actions,
        sectors=sectors,
        output_html=out_path,
        show=show,
    )

    if "cluster_summary" in diags:
        print("\n── Cluster Summary ─────────────────────────────────────────────")
        print(diags["cluster_summary"].to_string(index=False))

    if "sector_exposure" in diags:
        print("\n── Sector Exposure (owned) ──────────────────────────────────────")
        print(diags["sector_exposure"].to_string(index=False))

    print(f"\nFactor map saved → {out_path}")
