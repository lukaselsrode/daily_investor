"""
backtesting/artifacts.py — Persist backtest run results to reports/backtests/.

Saves four files per run:
  *_metrics.csv   — all scalar metrics in one row
  *_equity.csv    — daily strategy + benchmark equity curves
  *_trades.csv    — trade_log serialized to rows
  *_meta.json     — run metadata (mode, config_hash, timestamp, etc.)
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path

from core.paths import ROOT_DIR

BACKTESTS_DIR = os.path.join(ROOT_DIR, "reports", "backtests")


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def compute_config_hash(cfg: dict) -> str:
    """SHA-256[:12] of the canonical JSON representation of a config dict."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def save_backtest_result(
    report,
    output_dir: str = BACKTESTS_DIR,
    run_label: str | None = None,
    scope: str = "overall_strategy",
) -> dict[str, str]:
    """
    Persist backtest artifacts. Returns dict of {label: file_path}.

    report is a BacktestReport from backtesting.types.
    """
    import pandas as pd

    scope_dir = os.path.join(output_dir, scope)
    _ensure_dir(scope_dir)
    output_dir = scope_dir

    ts    = report.run_timestamp or datetime.datetime.utcnow().isoformat(timespec="seconds")
    ts_fs = ts.replace(":", "-").replace("T", "_")[:16]
    chash = report.config_hash or "nohash"
    label = f"backtest_{ts_fs}_{chash}"
    if run_label:
        label = f"{label}_{run_label[:20].replace(' ', '_')}"

    paths: dict[str, str] = {}

    # ── metrics.csv ──────────────────────────────────────────────────────
    train = report.train_result
    val   = report.validation_result
    row: dict = {
        "run_label":       label,
        "config_hash":     chash,
        "run_timestamp":   ts,
        "mode":            report.mode,
        "universe":        report.universe_selection,
        "lookahead_bias":  report.lookahead_bias_level,
        "n_symbols":       report.n_symbols,
        "n_days":          report.n_days,
        # train metrics
        "train_return":         train.total_return,
        "train_sharpe":         train.sharpe,
        "train_calmar":         train.calmar,
        "train_max_drawdown":   train.max_drawdown,
        "train_trades":         train.trades_made,
        "train_turnover":       train.turnover_estimate,
        "train_friction":       train.friction_cost,
        # benchmark
        "benchmark_return":     report.benchmark_return,
        "benchmark_sharpe":     report.benchmark_sharpe,
        "excess_return":        report.excess_return,
    }
    if val is not None:
        row.update({
            "val_return":       val.total_return,
            "val_sharpe":       val.sharpe,
            "val_max_drawdown": val.max_drawdown,
            "val_excess":       val.total_return - report.validation_benchmark_return,
        })
    metrics_path = os.path.join(output_dir, f"{label}_metrics.csv")
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    paths["metrics"] = metrics_path

    # ── equity.csv ───────────────────────────────────────────────────────
    eq    = train.equity_curve
    bench = train.benchmark_equity
    if len(eq) > 0:
        n = len(eq)
        rows_eq = {"day": list(range(n)), "strategy": eq.tolist()}
        if len(bench) == n:
            rows_eq["benchmark"] = bench.tolist()
        equity_path = os.path.join(output_dir, f"{label}_equity.csv")
        pd.DataFrame(rows_eq).to_csv(equity_path, index=False)
        paths["equity"] = equity_path

    # ── trades.csv ───────────────────────────────────────────────────────
    tlog = report.trade_log or train.trade_log
    if tlog:
        trade_rows = []
        for t in tlog:
            if dataclasses.is_dataclass(t):
                trade_rows.append(dataclasses.asdict(t))  # type: ignore[arg-type]
            elif hasattr(t, "__dict__"):
                trade_rows.append(dict(t.__dict__))
        if trade_rows:
            trades_path = os.path.join(output_dir, f"{label}_trades.csv")
            pd.DataFrame(trade_rows).to_csv(trades_path, index=False)
            paths["trades"] = trades_path

    # ── meta.json ────────────────────────────────────────────────────────
    meta: dict = {
        "run_label":       label,
        "config_hash":     chash,
        "run_timestamp":   ts,
        "scope":           scope,
        "mode":            report.mode,
        "universe":        report.universe_selection,
        "lookahead_bias":  report.lookahead_bias_level,
        "n_symbols":       report.n_symbols,
        "n_days":          report.n_days,
        "notes":           report.notes,
        "train_return":    train.total_return,
        "benchmark_return": report.benchmark_return,
        "excess_return":   report.excess_return,
        "files":           {k: os.path.basename(v) for k, v in paths.items()},
    }
    if scope == "active_sleeve_compounding":
        meta.update({
            "active_total_return":  train.active_total_return,
            "active_sharpe":        train.active_sharpe,
            "active_max_drawdown":  train.active_max_drawdown,
            "active_excess_return": train.active_excess_return,
        })
    meta_path = os.path.join(output_dir, f"{label}_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    paths["meta"] = meta_path

    return paths


def list_saved_runs(output_dir: str = BACKTESTS_DIR) -> list[dict]:
    """Scan for *_meta.json files across all scope subdirs, return sorted list (newest first)."""
    if not os.path.isdir(output_dir):
        return []
    # Search both top-level (legacy) and scope subdirectories
    base = Path(output_dir)
    metas = sorted(
        list(base.glob("*_meta.json")) + list(base.glob("*/*_meta.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result = []
    for p in metas:
        try:
            with open(p) as fh:
                m = json.load(fh)
            row: dict = {
                "Timestamp":    m.get("run_timestamp", "")[:16],
                "Hash":         m.get("config_hash", ""),
                "Scope":        m.get("scope", "overall_strategy"),
                "Mode":         m.get("mode", ""),
                "N days":       m.get("n_days", ""),
                "Train return": f"{m.get('train_return', 0):+.1%}" if isinstance(m.get("train_return"), (int, float)) else "",
                "Benchmark":    f"{m.get('benchmark_return', 0):+.1%}" if isinstance(m.get("benchmark_return"), (int, float)) else "",
                "Excess":       f"{m.get('excess_return', 0):+.1%}" if isinstance(m.get("excess_return"), (int, float)) else "",
            }
            if m.get("scope") == "active_sleeve_compounding":
                a_exc = m.get("active_excess_return")
                row["Active excess"] = f"{a_exc:+.1%}" if isinstance(a_exc, (int, float)) else ""
            result.append(row)
        except Exception:
            pass
    return result
