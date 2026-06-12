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

    try:
        login()
        authed = True
    except Exception as exc:
        # Unauthenticated mode still refreshes valuations/universe/fundamentals;
        # account-bound steps are SKIPPED so a dead session can't overwrite good
        # snapshots (a failed login once saved a 0-position holdings CSV).
        authed = False
        logger.warning(
            "Robinhood login failed (%s) — running UNAUTHENTICATED: "
            "dividends and holdings steps will be skipped.", exc,
        )
    logger.info("=== Fetch-Data run (no trades) ===")

    logger.info("Step 1/5: industry valuations")
    update_industry_valuations(verbose=True)

    logger.info("Step 2/5: dividends")
    if authed:
        _fetch_and_save_dividends()
    else:
        logger.warning("Skipped (not logged in)")

    logger.info("Step 3/5: holdings")
    if not authed:
        logger.warning("Skipped (not logged in) — existing holdings snapshot preserved")
    else:
        try:
            holdings = _broker.get_holdings()
            _broker.enrich_holdings_created_at(holdings)
            if holdings:
                save_holdings_csv(holdings)
                logger.info("Holdings saved: %d positions", len(holdings))
            else:
                logger.warning("0 holdings returned — snapshot NOT overwritten")
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
    regime_scope: str = "all",
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
        result = engine.run(n_days=n_days, params=params, mode=mode, scope=scope, regime_scope=regime_scope)
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


def cmd_list_presets() -> None:
    """Print available tuning presets and exit."""
    from tuning.presets import list_presets
    print("\nAvailable tuning presets:\n")
    for name, desc in list_presets():
        print(f"  {name:<30}  {desc}")
    print()


def cmd_tune(
    n_days: int,
    objective: str = "sharpe",
    mode: str | None = None,
    scope: str = "overall_strategy",
    preset: str | None = None,
    regime_scope: str = "all",
) -> None:
    """Single-objective tune — prints diff, does NOT write config.

    --preset <name>  Restrict tunable parameters to a named preset.
                     Use --list-presets to see available presets.
    """
    from tuning.tuner import ParameterTuner
    tuner = ParameterTuner()
    result = tuner.tune(
        n_days=n_days, objective=objective, mode=mode, scope=scope,
        preset=preset, regime_scope=regime_scope,
    )
    _t.print_config_diff(result.params, result.sim)


def cmd_auto_tune(
    n_days: int = 90,
    mode: str | None = None,
    apply: bool = False,
    force_apply: bool = False,
    llm_review: bool = False,
    scope: str = "overall_strategy",
    preset: str | None = None,
    regime_scope: str = "all",
    random_topk: int = 0,
    lead_vector_paths: list[str] | None = None,
) -> None:
    """Dual-objective auto-tune with candidate tournament + validation gates.

    --scope active_sleeve_compounding  freezes index_pct and ETF routing params,
    optimizes only stock-picking parameters, ranks by active sleeve metrics.
    --preset <name>  Restrict tunable parameters to a named preset.
                     Use --list-presets to see available presets.
    --random-topk N  Add the top-N robust-random-search candidates to the tournament.
    --leads a.npy,b.npy  Add saved lead param vectors to the tournament.
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
        preset=preset,
        regime_scope=regime_scope,
        random_topk=random_topk,
        lead_vector_paths=lead_vector_paths,
    )
    _t._diff_table(
        result.avg_params,
        label=f"tournament-selected over {n_days}d",
        sharpe_ref=result.sharpe_result,
        calmar_ref=result.calmar_result,
        sharpe_params=result.sharpe_params,
        calmar_params=result.calmar_params,
    )
    print(
        f"\nSelected result:  ret={result.avg_result.total_return:+.1%}  "
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


def cmd_auto_tune_all(
    profile: str = "standard",
    n_days: int = 730,
    mode: str | None = None,
    clusters: list[str] | None = None,
    regime_scope: str = "all",
) -> None:
    """
    Auto-tune All — staged coordinate-ascent over interaction clusters, then a full
    windowed validation confirmation. RESEARCH ONLY — never writes config (review the
    trace + verdict, then apply via the UI or auto-tune --apply on a chosen preset).
    """
    from backtesting.data_loader import load_and_precompute
    from tuning.interaction_screen import DEFAULT_CLUSTERS
    from tuning.profiles import expand_run_matrix
    from tuning.staged_tune import run_staged_tune, validate_full_windowed

    _profiles = {
        "quick":    dict(robustness="quick",    horizon="short", maxiter=6,  popsize=4),
        "standard": dict(robustness="standard", horizon="mixed", maxiter=10, popsize=6),
        "deep":     dict(robustness="deep",     horizon="mixed", maxiter=14, popsize=8),
    }
    cfg = _profiles.get(profile, _profiles["standard"])
    sel = clusters or list(DEFAULT_CLUSTERS)
    run_matrix = expand_run_matrix(cfg["robustness"], cfg["horizon"])

    print(f"\nAuto-tune All — profile={profile}, {n_days}d, regime_scope={regime_scope}, clusters={sel}")
    print("Loading full-universe data …")
    precomp = load_and_precompute(n_days, mode=mode)

    def _cb(done: int, total: int, label: str) -> None:
        print(f"  [{done}/{total}] {label}", flush=True)

    staged = run_staged_tune(
        precomp, clusters=sel, run_matrix=run_matrix, scope="active_sleeve_compounding",
        maxiter=cfg["maxiter"], popsize=cfg["popsize"], progress_callback=_cb,
        regime_scope=regime_scope,
    )
    print("\nStaged trace:")
    print(staged.trace_df().to_string(index=False))
    print(f"\nrobust score: {staged.baseline_score:.4f} (baseline) → {staged.final_score:.4f} "
          f"(final); accepted clusters: {staged.accepted_clusters or 'none'}")

    print("\nValidating (full windowed confirmation) …")
    v = validate_full_windowed(precomp, staged.final_params, run_matrix=run_matrix,
                               scope="active_sleeve_compounding", regime_scope=regime_scope)
    badge = "✅ CONFIRMED" if v["confirmed"] else "❌ NOT CONFIRMED"
    print(f"\n{badge}  —  OOS gate: {'pass' if v.get('oos_passed') else 'FAIL'} "
          f"({'; '.join(v.get('oos_reasons', [])) or 'all gates pass'}); "
          f"robust={v.get('robust_score', 0):.3f}, overfit={v.get('overfit_score', 1):.0%}")
    if v.get("horizon_df") is not None:
        print("\nPer-horizon robustness:")
        print(v["horizon_df"].to_string(index=False))


def cmd_interaction_screen(
    profile: str = "standard",
    n_days: int = 730,
    mode: str | None = None,
    output_dir: str | None = None,
    regime_scope: str = "all",
) -> None:
    """
    Parameter-interaction screener — RESEARCH ONLY, never writes config.

    Measures which interaction-cluster pairs SYNERGIZE vs CLASH when co-tuned, on
    the full universe with the robust multi-window objective. The full screen is an
    overnight job; --profile quick runs a fast smoke version.
    """
    import os

    from backtesting.data_loader import load_and_precompute
    from tuning.interaction_screen import DEFAULT_CLUSTERS, screen_interactions
    from tuning.profiles import expand_run_matrix

    _profiles = {
        "quick":    dict(robustness="quick",    horizon="short",  maxiter=5,  popsize=4),
        "standard": dict(robustness="standard", horizon="mixed",  maxiter=8,  popsize=6),
        "deep":     dict(robustness="deep",     horizon="mixed",  maxiter=12, popsize=8),
    }
    cfg = _profiles.get(profile, _profiles["standard"])
    run_matrix = expand_run_matrix(cfg["robustness"], cfg["horizon"])

    print(f"\nInteraction screen — profile={profile}, {n_days}d, regime_scope={regime_scope}, clusters={len(DEFAULT_CLUSTERS)}")
    print("Loading full-universe data …")
    precomp = load_and_precompute(n_days, mode=mode)

    def _cb(done: int, total: int) -> None:
        print(f"  [{done}/{total}] tunes complete", flush=True)

    result = screen_interactions(
        precomp, run_matrix=run_matrix, scope="active_sleeve_compounding",
        maxiter=cfg["maxiter"], popsize=cfg["popsize"], progress_callback=_cb,
        regime_scope=regime_scope,
    )

    print("\nInteraction matrix (diagonal = marginal robust score; off-diagonal = interaction):")
    print(result.matrix_df().to_string())
    print("\nPair verdicts (sorted by synergy):")
    print(result.pairs_df().to_string(index=False))

    out_dir = output_dir or "reports"
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "interaction_screen.csv")
        result.pairs_df().to_csv(path, index=False)
        print(f"\nWrote {path}")
    except Exception as exc:
        print(f"Could not write CSV: {exc}")


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


def cmd_config(action: str, *, dry_run: bool = False, no_backup: bool = False) -> None:
    """Config maintenance commands.

    Currently supports:
      migrate-scoring — rewrite legacy YAML (top-level scoring_v3/momentum_v2/value_v2)
                        into the unified `scoring:` block expected by the new util.py reader.
                        Idempotent. Creates .bak copies by default.
    """
    if action != "migrate-scoring":
        print(f"Unknown config action: {action!r} (expected: migrate-scoring)")
        return
    from cli.migrate_scoring import cmd_migrate_scoring
    cmd_migrate_scoring(dry_run=dry_run, no_backup=no_backup)


# ---------------------------------------------------------------------------
# Experiment runner — compare control variants across multiple time windows
# ---------------------------------------------------------------------------

# Variants A-G from the spec. Each holds a list of (cfg_path, value) overrides
# applied to live config copies during the experiment.
EXPERIMENT_VARIANTS: dict[str, dict] = {
    "A_baseline": {
        "description": "Current live config — baseline",
        "overrides": {},
    },
    "B_no_defensive_income_buys": {
        "description": "Disable defensive_income archetype management (no new DI buys)",
        "overrides": {"archetype_management.defensive_income.enabled": False},
    },
    "C_defensive_income_strict_gate": {
        "description": "Strict defensive_income gate (require yield, quality, momentum)",
        "overrides": {
            "archetype_classifier.enabled": True,
            "archetype_classifier.defensive_income.require_yield": True,
        },
    },
    "D_quality_compounder_only": {
        "description": "Only allow quality_compounder new buys (other archetypes off)",
        "overrides": {
            "archetype_management.legacy_turnaround.enabled": False,
            "archetype_management.speculative_momentum.enabled": False,
            "archetype_management.value_recovery.enabled": False,
            "archetype_management.defensive_income.enabled": False,
            "archetype_management.core_default.enabled": False,
        },
    },
    "E_cluster_cap_60": {
        "description": "Enforce cluster cap at 60%",
        "overrides": {
            "concentration_limits.warn_only": False,
            "concentration_limits.max_cluster_weight": 0.60,
        },
    },
    "F_cluster_cap_50": {
        "description": "Enforce cluster cap at 50%",
        "overrides": {
            "concentration_limits.warn_only": False,
            "concentration_limits.max_cluster_weight": 0.50,
        },
    },
    "G_cluster_cap_plus_strict_di": {
        "description": "Cluster cap 60% + strict defensive_income gate",
        "overrides": {
            "concentration_limits.warn_only": False,
            "concentration_limits.max_cluster_weight": 0.60,
            "archetype_classifier.enabled": True,
            "archetype_classifier.defensive_income.require_yield": True,
        },
    },
}


def cmd_experiment(
    days: str = "90,180,365",
    scope: str = "active_sleeve_compounding",
    variants: str | None = None,
    mode: str | None = None,
) -> None:
    """Run side-by-side backtest variants A–G across one or more time windows.

    Each variant patches the live config in-memory (the live cfg/config.yaml is
    not modified) and runs `run_simulation()` against the same precomp window.
    Output is a plain-text comparison table.

    --days 90,180,365              Comma-separated trading-day windows.
    --scope ...                    overall_strategy | active_sleeve_compounding (default).
    --variants A,E,F               Comma-separated variant IDs (defaults to all).
    --mode liquid_universe_full   Optional precomp mode override.
    """
    import copy as _copy

    from backtesting.data_loader import load_and_precompute
    from backtesting.simulator import (
        get_default_params,
        run_simulation,
        split_price_window,
    )
    from util import ARCHETYPE_CLASSIFIER_PARAMS, ARCHETYPE_PARAMS, CONCENTRATION_LIMIT_PARAMS

    # Filter variants
    selected_ids = (
        [v.strip() for v in variants.split(",")]
        if variants else list(EXPERIMENT_VARIANTS)
    )
    bad = [v for v in selected_ids if v not in EXPERIMENT_VARIANTS]
    if bad:
        print(f"Unknown variants: {bad}. Available: {list(EXPERIMENT_VARIANTS)}")
        return

    window_days = [int(d.strip()) for d in days.split(",")]

    # Build precomp once per window
    print(f"\nExperiment runner: variants={selected_ids} windows={window_days} scope={scope}\n")
    print(f"{'window':>7}  {'variant':<30}  {'TWR':>7}  {'bench':>7}  {'excess':>7}  "
          f"{'Sharpe':>7}  {'Calmar':>7}  {'maxDD':>7}  {'trades':>6}  {'clust_viol':>10}  "
          f"{'arch_mix':<40}")
    print("-" * 145)

    for n_days in window_days:
        precomp = load_and_precompute(n_days=n_days, mode=mode)
        train_slice, _ = split_price_window(precomp.prices.shape[0], train_pct=0.70)

        def _slice(s, precomp=precomp):
            def _o(a):
                return a[s] if a is not None else None
            return precomp._replace(
                prices=precomp.prices[s],
                etf_prices=precomp.etf_prices[s],
                benchmark_prices=precomp.benchmark_prices[s],
                position_52w_daily=precomp.position_52w_daily[s],
                return_1m_daily=precomp.return_1m_daily[s],
                bin_indices_daily=precomp.bin_indices_daily[s],
                has_position_52w_daily=precomp.has_position_52w_daily[s],
                ret_5d_daily=_o(precomp.ret_5d_daily),
                ret_3m_daily=_o(precomp.ret_3m_daily),
                ret_6m_daily=_o(precomp.ret_6m_daily),
                rs_3m_daily=_o(precomp.rs_3m_daily),
                rs_6m_daily=_o(precomp.rs_6m_daily),
                vol_3m_daily=_o(precomp.vol_3m_daily),
                above_50dma_daily=_o(precomp.above_50dma_daily),
                above_200dma_daily=_o(precomp.above_200dma_daily),
            )
        train_pc = _slice(train_slice)
        params = get_default_params()

        # Snapshot original globals so we can restore between variants
        _orig_arch = _copy.deepcopy(ARCHETYPE_PARAMS)
        _orig_cl = _copy.deepcopy(CONCENTRATION_LIMIT_PARAMS)
        _orig_ac = _copy.deepcopy(ARCHETYPE_CLASSIFIER_PARAMS)

        try:
            for variant_id in selected_ids:
                spec = EXPERIMENT_VARIANTS[variant_id]
                _apply_overrides(spec["overrides"])
                try:
                    sim = run_simulation(
                        train_pc, params,
                        starting_capital=10_000.0,
                        slippage_bps=10.0,
                        weekly_contribution=400.0,
                        rebalance_frequency_days=5,
                        archetype_aware=True,
                        cluster_tracking=True,
                        scope=scope,
                    )
                    _print_experiment_row(n_days, variant_id, sim, precomp)
                except Exception as exc:
                    print(f"{n_days:>5}d  {variant_id:<30}  ERROR: {exc}")
                finally:
                    # Restore globals between variants
                    ARCHETYPE_PARAMS.clear()
                    ARCHETYPE_PARAMS.update(_orig_arch)
                    CONCENTRATION_LIMIT_PARAMS.clear()
                    CONCENTRATION_LIMIT_PARAMS.update(_orig_cl)
                    ARCHETYPE_CLASSIFIER_PARAMS.clear()
                    ARCHETYPE_CLASSIFIER_PARAMS.update(_orig_ac)
        finally:
            ARCHETYPE_PARAMS.clear()
            ARCHETYPE_PARAMS.update(_orig_arch)
            CONCENTRATION_LIMIT_PARAMS.clear()
            CONCENTRATION_LIMIT_PARAMS.update(_orig_cl)
            ARCHETYPE_CLASSIFIER_PARAMS.clear()
            ARCHETYPE_CLASSIFIER_PARAMS.update(_orig_ac)
        print()  # blank line between windows


def _apply_overrides(overrides: dict) -> None:
    """Apply dotted-path overrides to the in-memory config dicts."""
    from util import ARCHETYPE_CLASSIFIER_PARAMS, ARCHETYPE_PARAMS, CONCENTRATION_LIMIT_PARAMS

    for path, value in overrides.items():
        parts = path.split(".")
        root = parts[0]
        if root == "archetype_management":
            d = ARCHETYPE_PARAMS
        elif root == "concentration_limits":
            d = CONCENTRATION_LIMIT_PARAMS
        elif root == "archetype_classifier":
            d = ARCHETYPE_CLASSIFIER_PARAMS
        else:
            continue
        cur = d
        for part in parts[1:-1]:
            cur = cur.setdefault(part, {}) if isinstance(cur, dict) else cur
        if isinstance(cur, dict):
            cur[parts[-1]] = value


def _print_experiment_row(n_days: int, variant_id: str, sim, precomp) -> None:
    """Print one row of the experiment comparison table."""
    import numpy as np

    bench_prices = precomp.benchmark_prices[:int(precomp.prices.shape[0] * 0.70)]
    bench_ret = float(bench_prices[-1] / bench_prices[0] - 1.0) if (
        len(bench_prices) >= 2 and np.isfinite(bench_prices).all() and bench_prices[0] > 0
    ) else 0.0
    excess = sim.total_return - bench_ret
    arch_total = sum(sim.archetype_pnl.values()) or 1.0
    arch_mix_parts = []
    for archetype, pnl in sorted(sim.archetype_pnl.items(), key=lambda kv: -kv[1])[:3]:
        arch_mix_parts.append(f"{archetype[:8]}{pnl/arch_total:+.2f}")
    arch_mix = " ".join(arch_mix_parts) or "—"
    print(
        f"{n_days:>5}d  {variant_id:<30}  "
        f"{sim.total_return:+6.1%}  {bench_ret:+6.1%}  {excess:+6.1%}  "
        f"{sim.sharpe:+6.2f}  {sim.calmar:+6.2f}  {sim.max_drawdown:+6.1%}  "
        f"{sim.trades_made:>6}  {sim.cluster_violations_count:>10}  {arch_mix:<40}"
    )


def cmd_snapshots(
    action: str,
    *,
    input_dir: str | None = None,
    output_dir: str | None = None,
    dry_run: bool = False,
    in_place_with_backup: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """Snapshot maintenance commands.

    Currently supports:
      rescore — rescore historical snapshots under the unified peer engine.
                Canonical column names (value_score, value_metric, …) are
                written and any legacy `*_v3` columns are stripped.
                Use --in-place-with-backup to overwrite source files (with
                .bak.parquet copies created first), or --output PATH to write
                a parallel tree.
    """
    if action != "rescore":
        print(f"Unknown snapshots action: {action!r} (expected: rescore)")
        return

    from strategy.snapshots import rescore_snapshots

    report = rescore_snapshots(
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=dry_run,
        in_place_with_backup=in_place_with_backup,
        overwrite_existing=overwrite_existing,
    )
    print(report.pretty())


def cmd_fmp(
    action: str,
    *,
    symbols_source: str = "current",
    start: str = "2015-01-01",
    end: str = "2030-01-01",
    kinds: list[str] | None = None,
    limit: int = 44,
    max_symbols: int | None = None,
    max_pages: int = 50,
    min_adv: float = 500_000.0,
    force: bool = False,
    allow_fetch_prices: bool = False,
) -> None:
    """FMP cache operations: status, backfill, dead-universe build, validation."""
    from data import fmp_ops

    if action == "status":
        print(fmp_ops.fmp_cache_status().pretty())
        return
    if action == "validate-cache":
        print(fmp_ops.validate_cache().pretty())
        return
    if action == "backfill-delisted":
        print(fmp_ops.backfill_delisted_roster(max_pages=max_pages).pretty())
        return
    if action == "build-dead-universe":
        print(fmp_ops.build_dead_universe(
            min_adv=min_adv,
            start=start,
            end=end,
            max_symbols=max_symbols,
            allow_fetch_prices=allow_fetch_prices,
        ).pretty())
        return

    if action in {"backfill-prices", "backfill-statements"}:
        symbols = fmp_ops.load_symbol_list(symbols_source, max_symbols=max_symbols)
        print(f"Loaded {len(symbols)} symbols from {symbols_source!r}")
        if action == "backfill-prices":
            print(fmp_ops.backfill_prices(symbols, start=start, end=end, force=force).pretty())
        else:
            print(fmp_ops.backfill_statements(
                symbols,
                kinds=kinds,
                limit=limit,
                force=force,
            ).pretty())
        return

    print(f"Unknown fmp action: {action!r}")


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
