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

import sys

from core.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = args[0]
    rest = args[1:]

    from cli.commands import (
        cmd_auto_tune,
        cmd_backtest,
        cmd_factor_map,
        cmd_fetch_data,
        cmd_list_presets,
        cmd_report,
        cmd_run,
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
        cmd_backtest(n_days=n_days, mode=mode, compare=compare,
                     archetype_compare=archetype_compare, scope=scope)

    elif cmd == "tune":
        if not rest or not rest[0].isdigit():
            print("Usage: tune DAYS [--objective sharpe|calmar] [--scope ...] [--preset ...]")
            sys.exit(1)
        n_days = int(rest[0])
        objective = _flag_value(rest, "--objective") or "sharpe"
        mode = _flag_value(rest, "--mode")
        scope = _flag_value(rest, "--scope") or "overall_strategy"
        preset = _flag_value(rest, "--preset")
        cmd_tune(n_days=n_days, objective=objective, mode=mode, scope=scope, preset=preset)

    elif cmd == "auto-tune":
        n_days = int(rest[0]) if rest and rest[0].isdigit() else 90
        mode = _flag_value(rest, "--mode")
        apply = "--apply" in rest
        force_apply = "--force-apply" in rest
        llm_review = "--llm-review" in rest
        scope = _flag_value(rest, "--scope") or "overall_strategy"
        preset = _flag_value(rest, "--preset")
        cmd_auto_tune(n_days=n_days, mode=mode, apply=apply, force_apply=force_apply, llm_review=llm_review, scope=scope, preset=preset)

    elif cmd == "stability-scan":
        mode = _flag_value(rest, "--mode")
        out_dir = _flag_value(rest, "--output-dir")
        cmd_stability_scan(mode=mode, output_dir=out_dir)

    elif cmd == "report":
        out_dir = _flag_value(rest, "--output-dir") or "reports"
        cmd_report(output_dir=out_dir)

    elif cmd == "update-outcomes":
        cmd_update_outcomes()

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


def _print_help() -> None:
    print("""
daily-investor CLI

COMMANDS
  fetch-data               Fetch fresh data only — no trades (fundamentals, news, snapshot)
  run                      Live trading run (requires Robinhood credentials)
  backtest DAYS            Run backtest simulation (--archetype-compare for A/B vs uniform)
  tune DAYS                Single-objective parameter tune (prints diff, no write)
  auto-tune [DAYS]         Dual-objective tune with walk-forward validation (default: 90d)
  list-presets             Print available tuning presets and exit
  stability-scan           Parameter stability scan (research only, no writes)
  report                   Generate diagnostics report
  update-outcomes          Backfill future returns for past decisions (calibration only)
  factor-map               3-D PCA/UMAP factor-space scatter of the scored universe

OPTIONS (run)
  --skip-data              Reuse existing CSV data
  --op-mode safe|automated|no-sentiment

OPTIONS (tune / auto-tune)
  --apply                  Write config.yaml if validation passes
  --force-apply            Write config.yaml unconditionally
  --llm-review             Add Claude second-opinion review
  --scope SCOPE            overall_strategy (default) or active_sleeve_compounding
  --preset NAME            Restrict tunable params to a named preset (see list-presets)

OPTIONS (all)
  --objective sharpe|calmar  Optimization target (default: sharpe)
  --mode MODE                Backtest universe selection mode
  --output-dir PATH          Report output directory
""")


if __name__ == "__main__":
    main()
