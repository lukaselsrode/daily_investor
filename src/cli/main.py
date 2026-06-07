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
"""

from __future__ import annotations

import os
import sys

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

    # --config <path>  — override which YAML the run reads.
    # Must be applied BEFORE importing cli.commands so core/paths.CONFIG_FILE
    # picks up the override at import time.
    _cfg_override = _flag_value(rest, "--config")
    if _cfg_override:
        if not os.path.isabs(_cfg_override):
            _cfg_override = os.path.abspath(_cfg_override)
        if not os.path.isfile(_cfg_override):
            print(f"--config: file not found: {_cfg_override}")
            sys.exit(2)
        os.environ["DAILY_INVESTOR_CONFIG"] = _cfg_override
        print(f"[config override] {_cfg_override}")

    from cli.commands import (
        cmd_auto_tune,
        cmd_auto_tune_all,
        cmd_backtest,
        cmd_config,
        cmd_factor_map,
        cmd_fetch_data,
        cmd_fmp,
        cmd_interaction_screen,
        cmd_list_presets,
        cmd_report,
        cmd_run,
        cmd_snapshots,
        cmd_stability_scan,
        cmd_tune,
        cmd_update_outcomes,
    )

    if cmd == "list-presets":
        cmd_list_presets()

    elif cmd == "fetch-data":
        cmd_fetch_data()

    elif cmd == "run":
        skip_data = "--skip-data" in rest
        op_mode = _flag_value(rest, "--op-mode")
        cmd_run(skip_data=skip_data, op_mode=op_mode)

    elif cmd == "backtest":
        n_days = int(rest[0]) if rest and rest[0].isdigit() else 365
        mode = _flag_value(rest, "--mode")
        compare = "--compare" in rest
        archetype_compare = "--archetype-compare" in rest
        scope = _flag_value(rest, "--scope") or "overall_strategy"
        regime_scope = _flag_value(rest, "--regime-scope") or "all"
        cmd_backtest(n_days=n_days, mode=mode, compare=compare,
                     archetype_compare=archetype_compare, scope=scope,
                     regime_scope=regime_scope)

    elif cmd == "tune":
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

    elif cmd == "auto-tune":
        n_days = int(rest[0]) if rest and rest[0].isdigit() else 90
        mode = _flag_value(rest, "--mode")
        apply = "--apply" in rest
        force_apply = "--force-apply" in rest
        llm_review = "--llm-review" in rest
        scope = _flag_value(rest, "--scope") or "overall_strategy"
        preset = _flag_value(rest, "--preset")
        regime_scope = _flag_value(rest, "--regime-scope") or "all"
        cmd_auto_tune(n_days=n_days, mode=mode, apply=apply, force_apply=force_apply, llm_review=llm_review, scope=scope, preset=preset, regime_scope=regime_scope)

    elif cmd == "stability-scan":
        mode = _flag_value(rest, "--mode")
        out_dir = _flag_value(rest, "--output-dir")
        cmd_stability_scan(mode=mode, output_dir=out_dir)

    elif cmd == "interaction-screen":
        mode = _flag_value(rest, "--mode")
        out_dir = _flag_value(rest, "--output-dir")
        profile = _flag_value(rest, "--profile") or "standard"
        _nd = _flag_value(rest, "--days")
        n_days = int(_nd) if _nd else 730
        regime_scope = _flag_value(rest, "--regime-scope") or "all"
        cmd_interaction_screen(profile=profile, n_days=n_days, mode=mode, output_dir=out_dir,
                               regime_scope=regime_scope)

    elif cmd == "auto-tune-all":
        mode = _flag_value(rest, "--mode")
        profile = _flag_value(rest, "--profile") or "standard"
        _nd = _flag_value(rest, "--days")
        n_days = int(_nd) if _nd else 730
        _cl = _flag_value(rest, "--clusters")
        clusters = [c.strip() for c in _cl.split(",") if c.strip()] if _cl else None
        regime_scope = _flag_value(rest, "--regime-scope") or "all"
        cmd_auto_tune_all(profile=profile, n_days=n_days, mode=mode, clusters=clusters,
                          regime_scope=regime_scope)

    elif cmd == "report":
        out_dir = _flag_value(rest, "--output-dir") or "reports"
        cmd_report(output_dir=out_dir)

    elif cmd == "update-outcomes":
        cmd_update_outcomes()

    elif cmd == "experiment":
        from cli.commands import cmd_experiment
        days = _flag_value(rest, "--days") or "90,180,365"
        scope = _flag_value(rest, "--scope") or "active_sleeve_compounding"
        variants = _flag_value(rest, "--variants")
        ex_mode = _flag_value(rest, "--mode")
        cmd_experiment(days=days, scope=scope, variants=variants, mode=ex_mode)

    elif cmd == "config":
        sub = rest[0] if rest else ""
        sub_rest = rest[1:] if len(rest) > 1 else []
        if sub == "migrate-scoring":
            dry_run = "--dry-run" in sub_rest
            no_backup = "--no-backup" in sub_rest
            cmd_config(action="migrate-scoring", dry_run=dry_run, no_backup=no_backup)
        else:
            print(f"Unknown config action: {sub!r}")
            sys.exit(1)

    elif cmd == "snapshots":
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

    elif cmd == "fmp":
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

    elif cmd == "factor-map":
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

    else:
        print(f"Unknown command: {cmd!r}")
        _print_help()
        sys.exit(1)


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

OPTIONS (run)
  --skip-data              Reuse existing CSV data
  --op-mode safe|automated|no-sentiment

OPTIONS (tune / auto-tune)
  --apply                  Write config.yaml if validation passes
  --force-apply            Write config.yaml unconditionally
  --llm-review             Add Claude second-opinion review
  --scope SCOPE            overall_strategy (default) or active_sleeve_compounding
  --regime-scope SCOPE     all (default), bullish, neutral, or defensive (bearish accepted as an alias for defensive)
  --preset NAME[+NAME...]  Restrict tunable params to a preset; compose several with '+'
                           to co-tune their union (e.g. active_exits+active_exit_floors)

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
  --mode MODE                Backtest universe selection mode
  --output-dir PATH          Report output directory
""")


if __name__ == "__main__":
    main()
