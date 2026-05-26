"""
tuning/stability.py — StabilityAnalyzer and module-level run_stability_scan.

RESEARCH / DIAGNOSTIC ONLY — never writes config.yaml.
Runs the optimizer across multiple windows and objectives to detect
unstable or overfit parameters.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backtesting.data_loader import load_and_precompute
from backtesting.simulator import run_backtest_report, run_simulation, split_price_window
from util import BACKTEST_PARAMS, STABILITY_PARAMS

from .constants import PARAM_NAMES
from .objective import _run_single
from .results import StabilityReport
from .tuner import validate_tuned_params

logger = logging.getLogger(__name__)


def run_stability_scan(
    windows: "list[int] | None" = None,
    objectives: "list[str] | None" = None,
    mode: "str | None" = None,
    starting_capital: float = 10_000.0,
    output_dir: "str | None" = None,
) -> dict:
    """
    Run the optimizer across multiple time windows and objectives to assess
    parameter stability and strategy robustness.

    DIAGNOSTIC ONLY — results are never applied to config.yaml.

    Returns a dict with:
        window_results:  list of per-window result dicts
        stability_df:    pd.DataFrame of per-parameter stability metrics
        output_paths:    dict of file paths written to output_dir
    """
    try:
        from scipy.optimize import differential_evolution  # noqa: F401
    except ImportError:
        raise RuntimeError("scipy is required. Install: pip install scipy")

    from reporting import compute_parameter_stability, generate_all_reports

    sp         = STABILITY_PARAMS
    bp         = BACKTEST_PARAMS
    windows    = windows    or sp["windows"]
    objectives = objectives or sp["objectives"]
    output_dir = output_dir or sp["output_dir"]
    maxiter    = sp["scan_maxiter"]
    popsize    = sp["scan_popsize"]
    cv_thr     = sp["unstable_cv_threshold"]
    spread_thr = sp["unstable_spread_threshold"]

    print(
        f"\n{'='*64}\n"
        f"STABILITY SCAN — RESEARCH / DIAGNOSTIC ONLY\n"
        f"Never modifies config.yaml.\n"
        f"Windows: {windows}\n"
        f"Objectives: {objectives}\n"
        f"Output: {output_dir}\n"
        f"{'='*64}\n"
    )

    window_results: list[dict] = []

    for w_idx, n_days in enumerate(windows):
        print(f"\n[{w_idx + 1}/{len(windows)}] Window: {n_days}d …")

        try:
            precomp = load_and_precompute(n_days, mode=mode)
        except Exception as e:
            print(f"  ⚠ Skipping {n_days}d — load failed: {e}")
            continue

        train_sl, val_sl = split_price_window(n_days, bp.get("train_pct", 0.70))

        def _opt_sl(arr):
            return arr[train_sl] if arr is not None else None

        tune_precomp = precomp._replace(
            prices=precomp.prices[train_sl],
            etf_prices=precomp.etf_prices[train_sl],
            benchmark_prices=precomp.benchmark_prices[train_sl],
            position_52w_daily=precomp.position_52w_daily[train_sl],
            return_1m_daily=precomp.return_1m_daily[train_sl],
            bin_indices_daily=precomp.bin_indices_daily[train_sl],
            has_position_52w_daily=precomp.has_position_52w_daily[train_sl],
            ret_5d_daily=_opt_sl(precomp.ret_5d_daily),
            ret_3m_daily=_opt_sl(precomp.ret_3m_daily),
            ret_6m_daily=_opt_sl(precomp.ret_6m_daily),
            rs_3m_daily=_opt_sl(precomp.rs_3m_daily),
            rs_6m_daily=_opt_sl(precomp.rs_6m_daily),
            vol_3m_daily=_opt_sl(precomp.vol_3m_daily),
            above_50dma_daily=_opt_sl(precomp.above_50dma_daily),
            above_200dma_daily=_opt_sl(precomp.above_200dma_daily),
        )

        per_obj: dict = {}
        for obj in objectives:
            print(f"  Optimizing {obj} …", end=" ", flush=True)
            try:
                params, result = _run_single(
                    tune_precomp, obj, starting_capital, maxiter, popsize
                )
                per_obj[obj] = {"params": params, "result": result}
                print(
                    f"ret={result.total_return:+.1%}  "
                    f"{obj}={result.sharpe if obj == 'sharpe' else result.calmar:+.3f}  "
                    f"trades={result.trades_made}"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                per_obj[obj] = None

        valid_params = [v["params"] for v in per_obj.values() if v is not None]
        if not valid_params:
            print(f"  ⚠ All objectives failed for {n_days}d — skipping")
            continue

        avg_params = np.mean(valid_params, axis=0)

        try:
            report = run_backtest_report(precomp, avg_params, train_sl, val_sl)
            vr = report.validation_result
            val_passed, _ = validate_tuned_params(report, bp)
            val_excess   = (vr.total_return - report.validation_benchmark_return) if vr else 0.0
            val_sharpe   = vr.sharpe if vr else 0.0
            val_drawdown = vr.max_drawdown if vr else 0.0
            avg_result   = report.train_result
        except Exception as e:
            print(f"  ⚠ Report failed for {n_days}d: {e}")
            val_passed = False
            val_excess = val_sharpe = val_drawdown = 0.0
            avg_result = run_simulation(
                tune_precomp, avg_params, starting_capital,
                slippage_bps=bp["slippage_bps"],
                commission_per_trade=bp["commission_per_trade"],
                weekly_contribution=bp["weekly_contribution"],
                rebalance_frequency_days=bp["rebalance_frequency_days"],
            )

        sharpe_p = per_obj.get("sharpe", {})
        calmar_p = per_obj.get("calmar", {})
        sc_spread_vec = []
        if sharpe_p and calmar_p and sharpe_p.get("params") is not None and calmar_p.get("params") is not None:
            sc_spread_vec = np.abs(sharpe_p["params"] - calmar_p["params"])
        unstable_count = int(np.sum(sc_spread_vec > spread_thr)) if len(sc_spread_vec) else 0

        window_results.append({
            "window":           n_days,
            "params_avg":       avg_params,
            "params_sharpe":    sharpe_p["params"] if sharpe_p else None,
            "params_calmar":    calmar_p["params"] if calmar_p else None,
            "result_sharpe":    sharpe_p["result"] if sharpe_p else None,
            "result_calmar":    calmar_p["result"] if calmar_p else None,
            "val_excess_return": val_excess,
            "val_sharpe":       val_sharpe,
            "val_drawdown":     val_drawdown,
            "turnover":         avg_result.turnover_estimate,
            "trades":           avg_result.trades_made,
            "unstable_params":  unstable_count,
            "validation_passed": val_passed,
        })

        status = "✓ passed" if val_passed else "✗ failed"
        print(
            f"  Summary: val_excess={val_excess:+.2%}  "
            f"val_sharpe={val_sharpe:+.3f}  "
            f"unstable_params={unstable_count}  "
            f"validation={status}"
        )

    if not window_results:
        import pandas as pd
        print("\n⚠ No windows completed successfully — no reports generated.")
        return {"window_results": [], "stability_df": pd.DataFrame(), "output_paths": {}}

    stability_df = compute_parameter_stability(
        window_results, PARAM_NAMES, cv_threshold=cv_thr, spread_threshold=spread_thr
    )

    n_unstable = int((stability_df["stability"] == "UNSTABLE").sum()) if not stability_df.empty else 0
    max_allow  = sp["max_unstable_params"]

    print(f"\n{'='*64}")
    print(f"Stability scan complete: {len(window_results)} windows")
    print(f"Unstable parameters: {n_unstable} (threshold: {max_allow})")
    if n_unstable > max_allow:
        print(f"⚠ {n_unstable} unstable params exceed max_unstable_params={max_allow}")
        print("  Consider fixing the most volatile params and re-tuning.")
    else:
        print("✓ Parameter count within stability budget.")
    print(f"{'='*64}")

    output_paths = generate_all_reports(
        window_results, stability_df, PARAM_NAMES, output_dir
    )

    print("\nOutputs written:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")

    return {
        "window_results": window_results,
        "stability_df":   stability_df,
        "output_paths":   output_paths,
    }


class StabilityAnalyzer:
    """
    Multi-window, multi-objective parameter stability scanner.

    Output:
      - StabilityReport with per-window results and stability_df
      - CSV summary per parameter (mean, stddev, CV across windows)
      - Heatmap PNGs (requires matplotlib)
      - Human-readable robustness report
      - Instability flags: STABLE / MODERATELY_STABLE / UNSTABLE
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    def scan(
        self,
        windows: Optional[list[int]] = None,
        mode: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> StabilityReport:
        raw = run_stability_scan(
            windows=windows,
            mode=mode,
            output_dir=output_dir,
        )
        return StabilityReport(
            window_results=raw.get("window_results", []),
            stability_df=raw.get("stability_df"),
            output_paths=raw.get("output_paths", {}),
        )

    def param_names(self) -> list[str]:
        return list(PARAM_NAMES)
