VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
DI        := $(VENV)/bin/daily-investor
STREAMLIT := $(VENV)/bin/streamlit
SRC       := src

# ── Dashboard ─────────────────────────────────────────────────────────────────

.PHONY: ui
ui:                          ## Launch the Streamlit dashboard
	$(STREAMLIT) run $(SRC)/ui/streamlit_app.py

# ── Data ──────────────────────────────────────────────────────────────────────

.PHONY: fetch-data
fetch-data:                  ## Fetch all data: valuations, dividends, holdings, fundamentals, news, snapshot (no trades)
	$(DI) fetch-data

# ── Live trading ──────────────────────────────────────────────────────────────

.PHONY: run
run:                         ## Live trading run  (safe mode — manual confirmation)
	$(DI) run --op-mode safe

.PHONY: run-auto
run-auto:                    ## Live trading run  (automated mode)
	$(DI) run --op-mode automated

.PHONY: run-skip
run-skip:                    ## Live trading run, reuse existing CSV data (faster)
	$(DI) run --op-mode safe --skip-data

.PHONY: run-dry
run-dry:                     ## Dry-run: skip data + sentiment  (scoring + logic preview only)
	$(DI) run --op-mode no-sentiment --skip-data

# ── Backtesting ───────────────────────────────────────────────────────────────

DAYS    ?= 365
BT_MODE ?= liquid_universe_sanity_test

.PHONY: backtest
backtest:                    ## Backtest  (DAYS=N  BT_MODE=...)
	$(DI) backtest $(DAYS) --mode $(BT_MODE)

.PHONY: backtest-wf
backtest-wf:                 ## Walk-forward backtest  (low lookahead, DAYS=N)
	$(DI) backtest $(DAYS) --mode walk_forward_price_only_test

# ── Parameter tuning ──────────────────────────────────────────────────────────

OBJ       ?= sharpe
TUNE_DAYS ?= 120
AUTO_DAYS ?= 90

.PHONY: tune
tune:                        ## Single-objective tune, no write  (TUNE_DAYS=N  OBJ=sharpe|calmar)
	$(DI) tune $(TUNE_DAYS) --objective $(OBJ)

.PHONY: auto-tune
auto-tune:                   ## Dual-objective tune, walk-forward validation, no write
	$(DI) auto-tune $(AUTO_DAYS) --objective $(OBJ)

.PHONY: auto-tune-apply
auto-tune-apply:             ## auto-tune + write config.yaml if validation passes
	$(DI) auto-tune $(AUTO_DAYS) --objective $(OBJ) --apply

.PHONY: auto-tune-llm
auto-tune-llm:               ## auto-tune + Claude second-opinion + apply
	$(DI) auto-tune $(AUTO_DAYS) --objective $(OBJ) --apply --llm-review

# ── Research / diagnostics ────────────────────────────────────────────────────

.PHONY: stability
stability:                   ## Parameter stability scan across multiple windows (research only)
	$(DI) stability-scan

.PHONY: report
report:                      ## Generate diagnostics report → reports/
	$(DI) report --output-dir reports

.PHONY: regime
regime:                      ## Print current market regime (live SPY + VIX fetch)
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.regimes import RegimeDetector; s = RegimeDetector().detect(); dma = f'{s.spy_vs_200dma_pct:+.2%}' if s.spy_vs_200dma_pct is not None else 'N/A'; print(f'Regime: {s.regime.upper()}  |  Confidence: {s.confidence:.0%}  |  VIX: {s.vix}  |  SPY vs 200DMA: {dma}'); print('Notes:', '  '.join(s.notes) if s.notes else 'none')"

.PHONY: snapshot-info
snapshot-info:               ## Show snapshot store status (count, date range)
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.snapshots import list_snapshots; snaps = list_snapshots(); print(f'{len(snaps)} snapshots  |  {snaps[0][0]}  →  {snaps[-1][0]}' if snaps else 'No snapshots found in data/snapshots/')"

.PHONY: snapshot-backfill
snapshot-backfill:           ## Backfill parquet snapshots from existing agg_data CSVs
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.snapshots import backfill_from_csvs; n = backfill_from_csvs(); print(f'Backfilled {n} snapshot(s)')"

.PHONY: ic
ic:                          ## Print quick IC summary across default horizons  (needs ≥2 snapshots)
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.research import FactorResearchEngine; engine = FactorResearchEngine(); ic = engine.compute_multi_horizon_ic(); summ = engine.compute_ic_summary(ic); print(summ.sort_values(['factor','horizon_days']).to_string(index=False) if not summ.empty else 'Not enough snapshots — need ≥ 2')"

# ── Development ───────────────────────────────────────────────────────────────

.PHONY: install
install:                     ## Install / reinstall package in editable mode
	$(PIP) install -e ".[ui,dev]" -q

.PHONY: test
test:                        ## Run full test suite
	/opt/homebrew/bin/pytest tests/ -q

.PHONY: test-watch
test-watch:                  ## Re-run tests on file changes  (requires pytest-watch)
	/opt/homebrew/bin/ptw tests/ -- -q

.PHONY: lint
lint:                        ## Run ruff linter over src/
	$(PYTHON) -m ruff check $(SRC)/ --select E,W,F --ignore E501

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:                        ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
