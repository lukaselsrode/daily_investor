"""
cli/main.py — Thin CLI dispatcher.

New-style invocation:
  python -m cli run
  python -m cli backtest 365
  python -m cli auto-tune 180 --apply
  python -m cli tune 120 --objective calmar
  python -m cli stability-scan
  python -m cli report

Old-style invocation via src/main.py is preserved for backward compatibility.

Dispatch is table-driven: each command's body lives in a `_cmd_<name>(rest)`
handler and `_COMMANDS` maps command names (including aliases) to handlers.
Handlers import from `cli.commands` lazily (at call time) so tests can patch
`cli.commands.cmd_*` before invoking `main`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from core.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    # Load .env so EVERY subcommand has API keys/creds (FMP_KEY, etc.). Previously only
    # src/main.py (the live-trading entry) loaded it, so `fmp`/`tune`/`backtest` ran
    # without FMP_KEY and silently failed every fetch.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = args[0]
    rest = args[1:]

    # --skip-fetch-news: reuse the most-recent cached news scrape instead of the
    # (slow) full news fetch, even on a fresh-data run. Surfaced as an env var so
    # the data layer (data/news.py) honors it without coupling to sys.argv.
    if "--skip-fetch-news" in rest:
        os.environ["SKIP_FETCH_NEWS"] = "1"

    _apply_config_override(rest)

    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd!r}")
        _print_help()
        sys.exit(1)
    handler(rest)


def _apply_config_override(rest: list[str]) -> None:
    """--config <path> — override which YAML the run reads.

    Must be applied BEFORE importing cli.commands so core/paths.CONFIG_FILE
    picks up the override at import time.
    """
    _cfg_override = _flag_value(rest, "--config")
    if _cfg_override:
        if not os.path.isabs(_cfg_override):
            _cfg_override = os.path.abspath(_cfg_override)
        if not os.path.isfile(_cfg_override):
            print(f"--config: file not found: {_cfg_override}")
            sys.exit(2)
        os.environ["DAILY_INVESTOR_CONFIG"] = _cfg_override
        print(f"[config override] {_cfg_override}")


def _cmd_list_presets(rest: list[str]) -> None:
    from cli.commands import cmd_list_presets
    cmd_list_presets()


def _cmd_fetch_data(rest: list[str]) -> None:
    from cli.commands import cmd_fetch_data
    cmd_fetch_data()


def _cmd_run(rest: list[str]) -> None:
    from cli.commands import cmd_run
    skip_data = "--skip-data" in rest
    op_mode = _flag_value(rest, "--op-mode")
    cmd_run(skip_data=skip_data, op_mode=op_mode)


def _cmd_backtest(rest: list[str]) -> None:
    from cli.commands import cmd_backtest
    n_days = int(rest[0]) if rest and rest[0].isdigit() else 365
    mode = _flag_value(rest, "--mode")
    compare = "--compare" in rest
    archetype_compare = "--archetype-compare" in rest
    scope = _flag_value(rest, "--scope") or "overall_strategy"
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    if "--compare-etf-allocation" in rest:
        # ETF sleeve before/after: forced equal-weight vs the current config allocation.
        from backtesting.data_loader import load_and_precompute
        from backtesting.reports import format_etf_sleeve_diagnostics
        from backtesting.simulator import run_backtest_report, split_price_window
        from tuning.constants import _ETF_ENABLED_SLOT, _current_params
        _pc = load_and_precompute(n_days, mode=mode)
        _tr, _vl = split_price_window(_pc.prices.shape[0], 0.70)
        _eq = _current_params().copy()
        _eq[_ETF_ENABLED_SLOT] = 0.0
        _cfg = _current_params().copy()  # reflects config etf_allocation.enabled
        _r_eq = run_backtest_report(_pc, _eq, _tr, _vl, scope="etf_allocation")
        _r_cfg = run_backtest_report(_pc, _cfg, _tr, _vl, scope="etf_allocation")
        print(format_etf_sleeve_diagnostics(
            _r_eq.validation_result or _r_eq.train_result, label="BEFORE: equal-weight"))
        print(format_etf_sleeve_diagnostics(
            _r_cfg.validation_result or _r_cfg.train_result,
            label="AFTER: current config",
            current_weights=(_r_eq.train_result.etf_final_weights or {})))
        return
    cmd_backtest(n_days=n_days, mode=mode, compare=compare,
                 archetype_compare=archetype_compare, scope=scope,
                 regime_scope=regime_scope)


def _cmd_tune(rest: list[str]) -> None:
    from cli.commands import cmd_tune
    if not rest or not rest[0].isdigit():
        print("Usage: tune DAYS [--objective sharpe|calmar|info_ratio] [--scope ...] [--preset ...] [--regime-scope all|bullish|neutral|defensive]")
        sys.exit(1)
    n_days = int(rest[0])
    objective = _flag_value(rest, "--objective") or "sharpe"
    mode = _flag_value(rest, "--mode")
    scope = _flag_value(rest, "--scope") or "overall_strategy"
    preset = _flag_value(rest, "--preset")
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    cmd_tune(n_days=n_days, objective=objective, mode=mode, scope=scope, preset=preset, regime_scope=regime_scope)


def _cmd_auto_tune(rest: list[str]) -> None:
    from cli.commands import cmd_auto_tune
    n_days = int(rest[0]) if rest and rest[0].isdigit() else 90
    mode = _flag_value(rest, "--mode")
    apply = "--apply" in rest
    force_apply = "--force-apply" in rest
    llm_review = "--llm-review" in rest
    scope = _flag_value(rest, "--scope") or "overall_strategy"
    preset = _flag_value(rest, "--preset")
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    random_topk = int(_flag_value(rest, "--random-topk") or 0)
    _leads_raw = _flag_value(rest, "--leads")
    lead_vector_paths = [p for p in (_leads_raw or "").split(",") if p] or None
    cmd_auto_tune(n_days=n_days, mode=mode, apply=apply, force_apply=force_apply, llm_review=llm_review, scope=scope, preset=preset, regime_scope=regime_scope, random_topk=random_topk, lead_vector_paths=lead_vector_paths)


def _cmd_tune_etf_allocation(rest: list[str]) -> None:
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 1250
    universe = _flag_value(rest, "--universe") or "configured_only"
    _mode = _flag_value(rest, "--mode") or "regime"
    if universe == "curated_exploration":
        print("curated_exploration is Milestone B — not yet available. "
              "Use --universe configured_only.")
        return
    preset = "etf_defensive_only" if _mode == "defensive" else "etf_allocation"
    random_topk = int(_flag_value(rest, "--random-topk") or 10)
    apply = "--apply" in rest
    force_apply = "--force-apply" in rest
    from tuning.etf_tune import run_etf_allocation_tune
    run_etf_allocation_tune(n_days=n_days, preset=preset, random_topk=random_topk,
                            apply=apply, force_apply=force_apply)


def _cmd_report_etf_allocation(rest: list[str]) -> None:
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 1250
    from backtesting.data_loader import load_and_precompute
    from backtesting.reports import format_etf_sleeve_diagnostics
    from backtesting.simulator import run_backtest_report, split_price_window
    from tuning.constants import _current_params
    _pc = load_and_precompute(n_days, mode=None)
    _tr, _vl = split_price_window(_pc.prices.shape[0], 0.70)
    _rep = run_backtest_report(_pc, _current_params(), _tr, _vl, scope="etf_allocation")
    print(format_etf_sleeve_diagnostics(
        _rep.validation_result or _rep.train_result, label=f"current config ({n_days}d)"))


def _cmd_odte_social_report(rest: list[str]) -> None:
    # 0DTE social-sentiment watchlist — ANALYSIS/PAPER ONLY, places no orders.
    # --no-fetch runs offline (cache-only/empty) for a safe dry run.
    # --reddit-bearer-token TOKEN: OPTIONAL ephemeral read-only bearer for the WSB daily-thread
    # comments fetch. Passed straight through as a RUNTIME arg — never stored, logged, or
    # echoed; obtain it manually from your browser/devtools. The tool never reads cookies or
    # mints tokens. Omit it for the default fail-closed OAuth/public behavior.
    # --daily-thread-id / --daily-thread-url: explicit override for the WSB daily-thread when
    # listing/search discovery can't find it. Overlaid into a COPY of the params at runtime —
    # never written back to config.
    from data.social_sentiment import (
        build_odte_social_report,
        format_report,
        format_report_json,
        load_0dte_runtime_config,
    )
    from util import OPTIONS_SOCIAL_PARAMS
    # --json is a machine contract: stdout must be ONLY the JSON. Logs go to stdout in this CLI, so
    # silence them for the run (the diagnostics an agent needs are in the JSON's source statuses).
    if "--json" in rest:
        import logging
        logging.disable(logging.ERROR)
    _bearer = _flag_value(rest, "--reddit-bearer-token")
    _dt_id = _flag_value(rest, "--daily-thread-id")
    _dt_url = _flag_value(rest, "--daily-thread-url")
    _dt_limit = _flag_value(rest, "--daily-thread-limit")
    # Auto-config for hands-off runs (e.g. the Hermes agent): pull creds/thread-id from ~/0dte/
    # (config.json / reddit_token.json / daily_thread_id.txt). Explicit flags always win. Also export
    # REDDIT_CLIENT_ID/SECRET (the robust, non-expiring app-OAuth path) so the comments fetch works
    # without a daily bearer-token refresh. Offline dry runs (--no-fetch) skip all of this.
    if "--no-fetch" not in rest:
        _auto = load_0dte_runtime_config()
        for _env, _key in (("REDDIT_CLIENT_ID", "reddit_client_id"),
                           ("REDDIT_CLIENT_SECRET", "reddit_client_secret")):
            if _auto.get(_key) and not os.environ.get(_env):
                os.environ[_env] = _auto[_key]
        _bearer = _bearer or _auto.get("reddit_bearer_token")
        if not (_dt_id or _dt_url):
            _dt_id = _dt_id or _auto.get("daily_thread_id")
    _params = None
    if _dt_id or _dt_url or _dt_limit:
        _params = {**OPTIONS_SOCIAL_PARAMS}   # shallow copy; global config left untouched
        if _dt_id:
            _params["daily_thread_id"] = _dt_id
        if _dt_url:
            _params["daily_thread_url"] = _dt_url
        if _dt_limit:
            try:
                _params["daily_thread_limit"] = int(_dt_limit)
            except ValueError:
                print(f"--daily-thread-limit: not an integer: {_dt_limit}")
                sys.exit(2)
    rep = build_odte_social_report(allow_fetch="--no-fetch" not in rest,
                                   params=_params, reddit_bearer_token=_bearer)
    # --json: clean, machine-ingestible payload for an agent (signal only, no paper/disclaimer prose).
    print(format_report_json(rep) if "--json" in rest else format_report(rep))


def _cmd_odte_watchdog(rest: list[str]) -> None:
    # Script-only 0DTE watchdog — NO LLM, NO Robinhood, places no orders. Runs the LOCAL report,
    # diffs the actionable candidate vs the prior run, and writes ~/0dte/{watchdog_state,triggers}.json.
    # stdout contract for a no_agent cron: EMPTY when nothing actionable; compact one-line JSON when a
    # trigger fires. --json always prints the compact state. --no-fetch runs offline (cache-only).
    import json
    import logging
    logging.disable(logging.ERROR)   # keep stdout a clean machine contract
    from data.odte_watchdog import run_watchdog
    state_dir = _flag_value(rest, "--state-dir")
    policy = _flag_value(rest, "--policy")
    payload = run_watchdog(state_dir=state_dir or os.path.expanduser("~/0dte"),
                           policy_path=policy, allow_fetch="--no-fetch" not in rest)
    if "--json" in rest or payload.get("alert"):
        print(json.dumps(payload, separators=(",", ":"), default=str))


def _cmd_stability_scan(rest: list[str]) -> None:
    from cli.commands import cmd_stability_scan
    mode = _flag_value(rest, "--mode")
    out_dir = _flag_value(rest, "--output-dir")
    cmd_stability_scan(mode=mode, output_dir=out_dir)


def _cmd_interaction_screen(rest: list[str]) -> None:
    from cli.commands import cmd_interaction_screen
    mode = _flag_value(rest, "--mode")
    out_dir = _flag_value(rest, "--output-dir")
    profile = _flag_value(rest, "--profile") or "standard"
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 730
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    cmd_interaction_screen(profile=profile, n_days=n_days, mode=mode, output_dir=out_dir,
                           regime_scope=regime_scope)


def _cmd_auto_tune_all(rest: list[str]) -> None:
    from cli.commands import cmd_auto_tune_all
    mode = _flag_value(rest, "--mode")
    profile = _flag_value(rest, "--profile") or "standard"
    _nd = _flag_value(rest, "--days")
    n_days = int(_nd) if _nd else 730
    _cl = _flag_value(rest, "--clusters")
    clusters = [c.strip() for c in _cl.split(",") if c.strip()] if _cl else None
    regime_scope = _flag_value(rest, "--regime-scope") or "all"
    cmd_auto_tune_all(profile=profile, n_days=n_days, mode=mode, clusters=clusters,
                      regime_scope=regime_scope)


def _cmd_report(rest: list[str]) -> None:
    from cli.commands import cmd_report
    out_dir = _flag_value(rest, "--output-dir") or "reports"
    cmd_report(output_dir=out_dir)


def _cmd_update_outcomes(rest: list[str]) -> None:
    from cli.commands import cmd_update_outcomes
    cmd_update_outcomes()


def _cmd_experiment(rest: list[str]) -> None:
    from cli.commands import cmd_experiment
    days = _flag_value(rest, "--days") or "90,180,365"
    scope = _flag_value(rest, "--scope") or "active_sleeve_compounding"
    variants = _flag_value(rest, "--variants")
    ex_mode = _flag_value(rest, "--mode")
    cmd_experiment(days=days, scope=scope, variants=variants, mode=ex_mode)


def _cmd_config(rest: list[str]) -> None:
    from cli.commands import cmd_config
    sub = rest[0] if rest else ""
    sub_rest = rest[1:] if len(rest) > 1 else []
    if sub == "migrate-scoring":
        dry_run = "--dry-run" in sub_rest
        no_backup = "--no-backup" in sub_rest
        cmd_config(action="migrate-scoring", dry_run=dry_run, no_backup=no_backup)
    else:
        print(f"Unknown config action: {sub!r}")
        sys.exit(1)


def _cmd_snapshots(rest: list[str]) -> None:
    from cli.commands import cmd_snapshots
    sub = rest[0] if rest else ""
    sub_rest = rest[1:] if len(rest) > 1 else []
    if sub == "rescore":
        input_dir   = _flag_value(sub_rest, "--input")
        output_dir  = _flag_value(sub_rest, "--output")
        dry_run     = "--dry-run" in sub_rest
        in_place    = "--in-place-with-backup" in sub_rest
        overwrite   = "--overwrite-existing" in sub_rest
        cmd_snapshots(
            action="rescore",
            input_dir=input_dir,
            output_dir=output_dir,
            dry_run=dry_run,
            in_place_with_backup=in_place,
            overwrite_existing=overwrite,
        )
    else:
        print("Usage: snapshots rescore "
              "[--dry-run] [--input PATH] [--output PATH] [--in-place-with-backup] [--overwrite-existing]")
        sys.exit(2)


def _cmd_fmp(rest: list[str]) -> None:
    from cli.commands import cmd_fmp
    sub = rest[0] if rest else "status"
    sub_rest = rest[1:] if len(rest) > 1 else []
    if sub == "status":
        cmd_fmp(action="status")
    elif sub == "validate-cache":
        cmd_fmp(action="validate-cache")
    elif sub == "backfill-prices":
        source = _flag_value(sub_rest, "--symbols") or "current"
        start = _flag_value(sub_rest, "--start") or "2015-01-01"
        end = _flag_value(sub_rest, "--end") or "2030-01-01"
        max_symbols = _int_flag(sub_rest, "--max-symbols")
        cmd_fmp(action="backfill-prices", symbols_source=source, start=start, end=end,
                max_symbols=max_symbols, force="--force" in sub_rest)
    elif sub == "backfill-statements":
        source = _flag_value(sub_rest, "--symbols") or "current"
        kinds_s = _flag_value(sub_rest, "--kinds")
        kinds = [k.strip() for k in kinds_s.split(",") if k.strip()] if kinds_s else None
        max_symbols = _int_flag(sub_rest, "--max-symbols")
        limit = _int_flag(sub_rest, "--limit") or 44
        cmd_fmp(action="backfill-statements", symbols_source=source, kinds=kinds,
                max_symbols=max_symbols, limit=limit, force="--force" in sub_rest)
    elif sub == "backfill-delisted":
        cmd_fmp(action="backfill-delisted", max_pages=_int_flag(sub_rest, "--max-pages") or 50)
    elif sub == "build-dead-universe":
        cmd_fmp(
            action="build-dead-universe",
            start=_flag_value(sub_rest, "--start") or "2015-01-01",
            end=_flag_value(sub_rest, "--end") or "2030-01-01",
            min_adv=float(_flag_value(sub_rest, "--min-adv") or 500_000.0),
            max_symbols=_int_flag(sub_rest, "--max-symbols"),
            allow_fetch_prices="--fetch-prices" in sub_rest,
        )
    else:
        print("Usage: fmp status | validate-cache | backfill-prices | backfill-statements | "
              "backfill-delisted | build-dead-universe")
        sys.exit(2)


def _cmd_factor_map(rest: list[str]) -> None:
    from cli.commands import cmd_factor_map
    method   = _flag_value(rest, "--method") or "pca"
    color    = _flag_value(rest, "--color")
    clusters_str = _flag_value(rest, "--clusters")
    clusters = int(clusters_str) if clusters_str and clusters_str.isdigit() else None
    out      = _flag_value(rest, "--output")
    owned    = "--owned-only" in rest
    show     = "--show" in rest
    cmd_factor_map(
        method=method,
        color_by=color,
        kmeans_clusters=clusters,
        output=out,
        owned_only=owned,
        show=show,
    )


# Command-name → handler. Aliases (e.g. options-social) map to the same handler.
_COMMANDS: dict[str, Callable[[list[str]], None]] = {
    "list-presets": _cmd_list_presets,
    "fetch-data": _cmd_fetch_data,
    "run": _cmd_run,
    "backtest": _cmd_backtest,
    "tune": _cmd_tune,
    "auto-tune": _cmd_auto_tune,
    "tune-etf-allocation": _cmd_tune_etf_allocation,
    "report-etf-allocation": _cmd_report_etf_allocation,
    "odte-social-report": _cmd_odte_social_report,
    "options-social": _cmd_odte_social_report,
    "odte-watchdog": _cmd_odte_watchdog,
    "stability-scan": _cmd_stability_scan,
    "interaction-screen": _cmd_interaction_screen,
    "auto-tune-all": _cmd_auto_tune_all,
    "report": _cmd_report,
    "update-outcomes": _cmd_update_outcomes,
    "experiment": _cmd_experiment,
    "config": _cmd_config,
    "snapshots": _cmd_snapshots,
    "fmp": _cmd_fmp,
    "factor-map": _cmd_factor_map,
}


def _flag_value(args: list[str], flag: str) -> str | None:
    try:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    except ValueError:
        pass
    return None


def _int_flag(args: list[str], flag: str) -> int | None:
    value = _flag_value(args, flag)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"{flag} requires an integer, got {value!r}")
        sys.exit(2)


def _print_help() -> None:
    print("""
daily-investor CLI

COMMANDS
  fetch-data               Fetch fresh data only — no trades (fundamentals, news, snapshot)
  run                      Live trading run (requires Robinhood credentials)
  backtest DAYS            Run backtest simulation (--archetype-compare for A/B vs uniform)
  tune DAYS                Single-objective parameter tune (prints diff, no write)
  auto-tune [DAYS]         Dual-objective tune with walk-forward validation (default: 90d)
  auto-tune-all            Staged coordinate-ascent over interaction clusters + windowed
                           validation (--profile quick|standard|deep, --clusters a,b; research only)
  interaction-screen       Screen which param clusters synergize/clash when co-tuned
                           (--profile quick|standard|deep; research only)
  list-presets             Print available tuning presets and exit (presets compose with '+')
  stability-scan           Parameter stability scan (research only, no writes)
  report                   Generate diagnostics report
  update-outcomes          Backfill future returns for past decisions (calibration only)
  factor-map               3-D PCA/UMAP factor-space scatter of the scored universe
  fmp <SUB>                FMP cache operations (status, backfill, validate)
  config <SUB>             config maintenance (sub: migrate-scoring)
  snapshots <SUB>          snapshot maintenance (sub: rescore)
  tune-etf-allocation      Gated ETF/core sleeve allocation tournament (--days, --mode regime|defensive,
                           --universe configured_only, --random-topk N, --apply)
  report-etf-allocation    Print ETF/core sleeve diagnostics for the current config (--days)
  odte-social-report       0DTE social-sentiment watchlist — ANALYSIS/PAPER ONLY, places NO orders.
                           Reddit (JSON+Atom fallback) + X (official API only if X_BEARER_TOKEN);
                           DAY-OF posts only; attaches a PAPER same-day option idea (yfinance);
                           --no-fetch = offline (no network, no options lookup);
                           --json = clean signal-only JSON for an agent (no prose; logs silenced);
                           --reddit-bearer-token TOKEN = optional ephemeral read-only token for WSB
                           daily-thread comments (never stored/logged; not cookies);
                           --daily-thread-id ID / --daily-thread-url URL = override daily-thread
                           discovery (runtime only; not persisted to config);
                           --daily-thread-limit N = cap on comments read (default: auto-paginate
                           the WHOLE thread, up to a runaway safety ceiling).
  options-social           Alias for odte-social-report (identical behavior and options).
  odte-watchdog            Script-only 0DTE watchdog — NO LLM, NO Robinhood, places NO orders.
                           Runs the LOCAL report, diffs the actionable candidate vs the prior run,
                           writes ~/0dte/{watchdog_state,triggers}.json. For a no_agent cron:
                           EMPTY stdout when nothing actionable, compact one-line JSON on a trigger
                           (new/changed non-restricted candidate, or missing/invalid policy).
                           --json always prints state; --no-fetch runs offline (cache-only);
                           --policy PATH / --state-dir DIR override the defaults (~/0dte/).

OPTIONS (run)
  --skip-data              Reuse existing CSV data
  --op-mode safe|automated|no-sentiment

OPTIONS (run / fetch-data)
  --skip-fetch-news        Refresh all other data but reuse the latest cached news
                           scrape (skips the slowest stage; any age). 0DTE options
                           sentiment is unaffected. Env NEWS_FORCE_REFETCH=1 overrides
                           the default 8h news-freshness reuse window.

OPTIONS (tune / auto-tune)
  --apply                  Write config.yaml if validation passes
  --force-apply            Write config.yaml unconditionally
  --llm-review             Add Claude second-opinion review
  --scope SCOPE            overall_strategy (default) or active_sleeve_compounding
  --regime-scope SCOPE     all (default), bullish, neutral, or defensive (bearish accepted as an alias for defensive)
  --preset NAME[+NAME...]  Restrict tunable params to a preset; compose several with '+'
                           to co-tune their union (e.g. active_exits+active_exit_floors)
  --random-topk N          auto-tune only: add the top-N robust-random-search candidates
                           to the selection tournament (default 0 = off)
  --leads a.npy,b.npy      auto-tune only: add saved lead param vectors to the tournament

OPTIONS (auto-tune-all / interaction-screen)
  --profile P              quick | standard | deep  (default: standard)
  --days N                 history window to load (default: 730)
  --clusters a,b,c         auto-tune-all only: which interaction clusters to co-tune

OPTIONS (fmp)
  fmp status
  fmp validate-cache
  fmp backfill-prices --symbols current|AAPL,MSFT|path.csv [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                      [--max-symbols N] [--force]
  fmp backfill-statements --symbols current|AAPL,MSFT|path.csv [--kinds income-statement,...]
                          [--limit N] [--max-symbols N] [--force]
  fmp backfill-delisted [--max-pages N]
  fmp build-dead-universe [--min-adv N] [--fetch-prices] [--max-symbols N]

OPTIONS (any command)
  --config PATH            Use a different YAML config (default: cfg/config.yaml).
                           Useful for cfg/config_<name>.yaml A/B comparisons.

OPTIONS (tune only)
  --objective sharpe|calmar|info_ratio  Optimization target (default: sharpe). info_ratio = excess-vs-SPY
                             / tracking-error (active scope). NOTE: `auto-tune` is dual-objective
                             (averages sharpe+calmar) and ignores --objective.

OPTIONS (all)
  --mode MODE                Backtest universe selection mode. One of:
                               liquid_universe_full          (default; full liquid universe)
                               walk_forward_price_only_test  (low lookahead; price/momentum only)
                               current_universe_stress_test  (high lookahead; current-score ranking)
  --output-dir PATH          Report output directory
""")


if __name__ == "__main__":
    main()
