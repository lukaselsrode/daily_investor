#!/usr/bin/env python3
"""Random-window regime sizing experiment.

This is the canonical follow-up to the per-regime weight-tuning result: do not
retune score weights; test whether neutral-regime active exposure should be
larger. The script is read-only and writes only an optional CSV report.

Example:
  PYTHONPATH=src .venv/bin/python scripts/regime_sizing_random_window.py \
      --regime neutral --n-days 5000 --window-days 45 --n-windows 40 \
      --output reports/regime_sizing_neutral.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="neutral", choices=["bullish", "neutral", "defensive", "bearish"])
    parser.add_argument("--n-days", type=int, default=5000)
    parser.add_argument("--n-windows", type=int, default=40)
    parser.add_argument("--window-days", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default=None, help="Backtest mode; default reads cfg/config.yaml")
    parser.add_argument("--scope", default="overall_strategy", choices=["overall_strategy", "active_sleeve_compounding"])
    parser.add_argument("--index-pcts", default="", help="Comma-separated index_pct values; default uses neutral exposure grid")
    parser.add_argument("--max-buys", default="", help="Comma-separated max_buys values paired with --index-pcts")
    parser.add_argument("--output", default="", help="Optional CSV output path")
    parser.add_argument("--paired", action="store_true", help="Evaluate all variants on identical sampled windows")
    parser.add_argument("--segment", default="all", choices=["all", "train", "holdout"], help="Temporal segment for paired starts")
    parser.add_argument("--split-pct", type=float, default=0.70, help="Train/holdout split day fraction for paired segment mode")
    args = parser.parse_args()

    import pandas as pd

    from backtesting.data_loader import load_and_precompute
    from backtesting.regime_scope import eligible_window_starts
    from research.regime_sizing import (
        SizingVariant,
        default_neutral_sizing_grid,
        result_rows,
        run_regime_sizing_grid,
        run_regime_sizing_grid_on_starts,
        sample_regime_window_starts,
    )
    from tuning.constants import _current_params
    from util import REGIME_PARAMS, RISK_LIMITS

    print(">>> loading precomputed data ...", flush=True)
    precomp = load_and_precompute(args.n_days, mode=args.mode)
    print(f">>> precomp: {precomp.prices.shape[0]} days x {precomp.prices.shape[1]} stocks", flush=True)

    for wd in sorted({30, 45, 60, int(args.window_days)}):
        parts = []
        for regime in ("bullish", "neutral", "defensive"):
            try:
                starts, _ = eligible_window_starts(precomp, wd, regime)
                parts.append(f"{regime}={len(starts)}")
            except Exception as exc:
                parts.append(f"{regime}=0({type(exc).__name__})")
        print(f"eligible windows @ {wd:>3}d: " + "  ".join(parts), flush=True)

    base_params = _current_params()
    current_index_pct = float(REGIME_PARAMS.get(args.regime, {}).get("index_pct_override") or base_params[4])
    current_max_buys = int(REGIME_PARAMS.get(args.regime, {}).get("max_buys_override") or RISK_LIMITS["max_buys_per_rebalance"])

    if args.index_pcts:
        index_pcts = [float(x.strip()) for x in args.index_pcts.split(",") if x.strip()]
        max_buys_vals = [current_max_buys] * len(index_pcts)
        if args.max_buys:
            max_buys_vals = [int(x.strip()) for x in args.max_buys.split(",") if x.strip()]
            if len(max_buys_vals) != len(index_pcts):
                raise SystemExit("--max-buys must have the same number of values as --index-pcts")
        variants = [
            SizingVariant(name=f"idx{idx:.2f}_buys{mb}", index_pct=idx, max_buys=mb)
            for idx, mb in zip(index_pcts, max_buys_vals, strict=True)
        ]
    else:
        variants = default_neutral_sizing_grid(current_index_pct, current_max_buys)

    print(
        f"\n>>> running regime sizing grid: regime={args.regime}, scope={args.scope}, "
        f"window_days={args.window_days}, n_windows={args.n_windows}\n",
        flush=True,
    )
    if args.paired:
        split_day = int(precomp.prices.shape[0] * args.split_pct)
        starts = sample_regime_window_starts(
            precomp,
            window_days=args.window_days,
            regime_scope=args.regime,
            n_windows=args.n_windows,
            seed=args.seed,
            segment=args.segment,
            split_day=split_day,
        )
        print(
            f">>> paired mode: using {len(starts)} identical {args.segment} windows "
            f"(split_day={split_day}, starts={starts[:8].tolist()}{'...' if len(starts) > 8 else ''})",
            flush=True,
        )
        results = run_regime_sizing_grid_on_starts(
            precomp=precomp,
            base_params=base_params,
            variants=variants,
            starts=starts,
            window_days=args.window_days,
            scope=args.scope,  # type: ignore[arg-type] argparse restricts choices
        )
    else:
        results = run_regime_sizing_grid(
            precomp=precomp,
            base_params=base_params,
            variants=variants,
            regime_scope=args.regime,
            n_windows=args.n_windows,
            window_days=args.window_days,
            seed=args.seed,
            scope=args.scope,  # type: ignore[arg-type] argparse restricts choices
        )

    rows = result_rows(results)
    df = pd.DataFrame(rows).sort_values("robust_score", ascending=False)
    display = df.copy()
    for col in [
        "index_pct", "active_pct", "median_excess", "pct_beating",
        "median_drawdown", "median_strategy_return", "median_benchmark_return",
    ]:
        display[col] = display[col].map(_fmt_pct)
    display["median_sharpe"] = display["median_sharpe"].map(lambda x: f"{x:+.2f}")
    display["robust_score"] = display["robust_score"].map(lambda x: f"{x:+.4f}")
    print(display.to_string(index=False), flush=True)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n>>> wrote {path}", flush=True)
    print("\n>>> DONE", flush=True)


if __name__ == "__main__":
    main()
