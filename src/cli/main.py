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
        cmd_fetch_data,
        cmd_report,
        cmd_run,
        cmd_stability_scan,
        cmd_tune,
    )

    if cmd == "fetch-data":
        cmd_fetch_data()

    elif cmd == "run":
        skip_data = "--skip-data" in rest
        op_mode = _flag_value(rest, "--op-mode")
        cmd_run(skip_data=skip_data, op_mode=op_mode)

    elif cmd == "backtest":
        n_days = int(rest[0]) if rest and rest[0].isdigit() else 365
        mode = _flag_value(rest, "--mode")
        cmd_backtest(n_days=n_days, mode=mode)

    elif cmd == "tune":
        if not rest or not rest[0].isdigit():
            print("Usage: tune DAYS [--objective sharpe|calmar]")
            sys.exit(1)
        n_days = int(rest[0])
        objective = _flag_value(rest, "--objective") or "sharpe"
        mode = _flag_value(rest, "--mode")
        cmd_tune(n_days=n_days, objective=objective, mode=mode)

    elif cmd == "auto-tune":
        n_days = int(rest[0]) if rest and rest[0].isdigit() else 90
        mode = _flag_value(rest, "--mode")
        apply = "--apply" in rest
        force_apply = "--force-apply" in rest
        llm_review = "--llm-review" in rest
        cmd_auto_tune(n_days=n_days, mode=mode, apply=apply, force_apply=force_apply, llm_review=llm_review)

    elif cmd == "stability-scan":
        mode = _flag_value(rest, "--mode")
        out_dir = _flag_value(rest, "--output-dir")
        cmd_stability_scan(mode=mode, output_dir=out_dir)

    elif cmd == "report":
        out_dir = _flag_value(rest, "--output-dir") or "reports"
        cmd_report(output_dir=out_dir)

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
  backtest DAYS            Run backtest simulation
  tune DAYS                Single-objective parameter tune (prints diff, no write)
  auto-tune [DAYS]         Dual-objective tune with walk-forward validation (default: 90d)
  stability-scan           Parameter stability scan (research only, no writes)
  report                   Generate diagnostics report

OPTIONS (run)
  --skip-data              Reuse existing CSV data
  --op-mode safe|automated|no-sentiment

OPTIONS (auto-tune)
  --apply                  Write config.yaml if validation passes
  --force-apply            Write config.yaml unconditionally
  --llm-review             Add Claude second-opinion review
  --mode MODE              Backtest universe mode

OPTIONS (all)
  --objective sharpe|calmar  Optimization target (default: sharpe)
  --mode MODE                Backtest universe selection mode
  --output-dir PATH          Report output directory
""")


if __name__ == "__main__":
    main()
