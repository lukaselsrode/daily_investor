"""
ui/services — Thin orchestration layer between UI/CLI and core engines.

Available services:
  backtest_service  — run_single_backtest, run_random_windows, list_saved_runs
  tuning_service    — run_weight_tune, run_stability_scan
  odte_service      — cockpit_state, budget_now, safety_state, funnel, orb_near_misses,
                      trade_ledger, latency_rows, lease_timeline, day_index/day_packet,
                      day_score_series, shadow_state (the 0DTE pages' only data seam)

UI components and CLI should call these instead of importing core engines
directly (exception: ablation_runner.py, which needs raw run_simulation access).
"""
