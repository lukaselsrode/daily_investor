# Codebase Cleanup / Refactor Report

**Date:** 2026-05-25  
**Branch:** main

---

## Summary

This pass focused on consolidation, maintainability, and UI simplification
without changing live execution or strategy behavior.

---

## Changes Made

### New files

| File | Purpose |
|------|---------|
| `src/ui/components/common.py` | Shared UI utilities: metric_row, status_badge, warn_banner, empty_state, df_download, yaml_diff_viewer, cmd_preview |
| `src/ui/components/decision_journal.py` | Decision Journal tab (Portfolio section): displays decision_outcomes.parquet with filters + outcome rates |
| `src/ui/components/attribution.py` | Attribution tab (Portfolio section): parameter stability + factor attribution stub |
| `src/ui/components/snapshot_health.py` | Snapshot/Data Health tab (System section): CSV inventory, snapshot count, outcomes fill rates |
| `src/ui/services/__init__.py` | Service layer package |
| `src/ui/services/data_service.py` | Data loading: agg_data, snapshots, CSVs, decision outcomes |
| `src/ui/services/config_service.py` | Config loading, audit, config file list |
| `src/ui/services/portfolio_service.py` | Holdings, decision outcomes, position journal |
| `src/ui/services/validation_service.py` | Backtest and walk-forward wrappers for UI |
| `src/ui/services/research_service.py` | IC computation and snapshot loading |
| `src/config/constants.py` | Domain-labelled re-exports of util.py constants for new code |

### Modified files

| File | Change |
|------|--------|
| `src/ui/sections/portfolio.py` | Added Decision Journal + Attribution tabs (3 → 5 tabs) |
| `src/ui/sections/system.py` | Added Data Explorer, Reliability, Snapshot Health tabs (2 → 5 tabs) |
| `src/ui/sections/operations.py` | Added Logs tab (4 → 5 tabs) |
| `src/portfolio/sell_engine.py` | Added module-level `evaluate_sell_candidate()` as dict-returning backward-compat wrapper |
| `src/main.py` | Removed 153-line duplicate `evaluate_sell_candidate`; now imports from portfolio.sell_engine |

---

## UI Navigation After Refactor

### Operations (5 tabs)
- Dashboard · Run Control · Order Intents · Execute · **Logs** ← new

### Portfolio (5 tabs)
- Holdings · Exposure · Regime · **Decision Journal** · **Attribution** ← 2 new

### Research (11 tabs — unchanged)
- Overview · Factors · IC Analysis · Rank & Deciles · Correlations · Regime
- Conditional Features · Distribution · Model Calibration · Candidate Pool · Experimental

### Validation (7 tabs — unchanged from previous session)
- Backtests · Stability · Reliability · Tuning · Config Diagnostics · Config Compare · Ablation Runner

### System (5 tabs)
- Config · Logs & Audit · **Data Explorer** · **Reliability** · **Snapshot Health** ← 3 new

---

## Code Removed / Consolidated

- `evaluate_sell_candidate` (153 lines) removed from `main.py` — canonical version in `portfolio/sell_engine.py`

---

## Remaining Cleanup Tasks (follow-up PRs)

### High impact

| Task | File | Notes |
|------|------|-------|
| Move `can_buy_symbol` | main.py → portfolio/risk.py | Requires `get_position_value` + `get_sector_exposure` to move too |
| Move `allocate_harvest_proceeds_to_etfs` | main.py → portfolio/harvest.py | Requires Robinhood API abstraction |
| Move `make_sales` | main.py → portfolio/sell_engine.py | Large, touches rb API; needs careful testing |
| Move `make_buys` | main.py → portfolio/manager.py | Large, touches rb API; needs careful testing |
| Move `get_market_regime` | main.py → strategy/regimes/detector.py | Pure computation, safe to move |
| Move `get_sector_exposure` | main.py → portfolio/exposure/analyzer.py | Pure computation |
| Move `_log_all_holding_decisions` / `_log_candidate` | main.py → portfolio/decision_logger.py | Already has related logic |

### Medium impact

| Task | File | Notes |
|------|------|-------|
| Retire `util.py` | util.py | Replace with `config/constants.py` re-exports after all callers updated |
| Merge `sentiment_analysis.py` | → `data/sentiment.py` | data/sentiment.py already wraps it |
| Merge `sentiments.py` | → `data/news.py` | data/news.py already wraps it |
| Retire `src/tests.py` | src/tests.py | Superseded by `tests/` directory |
| Move `_reporting_legacy.py` functions | → reporting/ modules | reporting/* already wraps them |

### Low impact

| Task | Notes |
|------|-------|
| Research section: collapse to ~6 tabs | Move Candidate Pool to Validation; move Experimental to System (already in System now) |
| Add `position_journal.csv` display to Decision Journal | portfolio_service.get_position_journal() already loads it |
| Factor attribution | Needs BacktestReport.trade_log instrumentation (stub in attribution.py) |
| common.py adoption | Gradually replace duplicate metric/banner patterns across existing components |
| services/ adoption | Gradually update existing components to call services instead of loading data directly |

---

## Architecture Status

```
src/
  cli/          ✅ complete
  config/       ✅ complete (ConfigManager + schema + new constants.py)
  core/         ✅ complete
  data/         ✅ complete (wraps legacy source_data/sentiments)
  strategy/     ✅ complete
  portfolio/    ✅ complete (evaluate_sell_candidate migrated)
  execution/    ✅ complete
  backtesting/  ✅ complete (wraps backtest.py)
  tuning/       ✅ complete (wraps tuner.py)
  research/     ✅ complete
  reporting/    ✅ complete (wraps _reporting_legacy.py)
  ui/
    sections/   ✅ complete — 5 sections match spec
    services/   ✅ new — service layer created
    components/
      common.py ✅ new
      decision_journal.py ✅ new
      attribution.py      ✅ new
      snapshot_health.py  ✅ new

Legacy (still active, not yet retired):
  src/backtest.py          — real engine, wrapped by backtesting/
  src/tuner.py             — real engine, wrapped by tuning/
  src/source_data.py       — active, partially wrapped by data/
  src/util.py              — active, re-exported by config/constants.py
  src/_reporting_legacy.py — active, wrapped by reporting/
  src/sentiment_analysis.py — active, wrapped by data/sentiment.py
  src/sentiments.py        — active, wrapped by data/news.py
  src/main.py              — active orchestrator, still 1970 lines
```
