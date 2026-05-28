"""
ui/services — Thin orchestration layer between UI/CLI and core engines.

Available services:
  backtest_service  — run_single_backtest, run_random_windows, list_saved_runs
  tuning_service    — run_weight_tune, run_stability_scan

UI components and CLI should call these instead of importing core engines
directly (exception: ablation_runner.py, which needs raw run_simulation access).
"""
