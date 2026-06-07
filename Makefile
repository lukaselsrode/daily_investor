# Auto-detect venv; fall back to system tools if no venv present
ifneq ($(wildcard .venv/bin/python),)
  PYTHON       := .venv/bin/python
  DI           := .venv/bin/daily-investor
  STREAMLIT    := .venv/bin/streamlit
  LINT_IMPORTS := PYTHONPATH=$(SRC) .venv/bin/lint-imports
  RADON        := .venv/bin/radon
else
  PYTHON       := python3
  DI           := daily-investor
  STREAMLIT    := streamlit
  LINT_IMPORTS := PYTHONPATH=$(SRC) lint-imports
  RADON        := radon
endif

SRC := src

# ── Dashboard ─────────────────────────────────────────────────────────────────

.PHONY: ui
ui:                          ## Launch the Streamlit dashboard
	$(STREAMLIT) run $(SRC)/ui/streamlit_app.py

# ── Data ──────────────────────────────────────────────────────────────────────

.PHONY: fetch-data
fetch-data:                  ## Fetch all data: valuations, dividends, holdings, fundamentals, news, snapshot (no trades)
	$(DI) fetch-data

.PHONY: update-outcomes
update-outcomes:             ## Backfill future return outcomes for past decisions (calibration only — never touches live scoring)
	$(DI) update-outcomes

FMP_SYMBOLS ?= current
FMP_START   ?= 2006-01-01
FMP_END     ?= 2030-01-01
FMP_MAX     ?=
FMP_KINDS   ?= income-statement,balance-sheet-statement,cash-flow-statement
FMP_PAGES   ?= 50
FMP_MIN_ADV ?= 500000

.PHONY: fmp-status
fmp-status:                  ## Show FMP cache/key/quota/coverage status
	$(DI) fmp status

.PHONY: fmp-validate-cache
fmp-validate-cache:          ## Read-only sanity check of FMP price/statement/dead-universe cache
	$(DI) fmp validate-cache

.PHONY: fmp-backfill-prices
fmp-backfill-prices:         ## Backfill FMP adjusted prices (FMP_SYMBOLS=current|AAPL,MSFT|path.csv)
	$(DI) fmp backfill-prices --symbols $(FMP_SYMBOLS) --start $(FMP_START) --end $(FMP_END) $(if $(FMP_MAX),--max-symbols $(FMP_MAX),)

.PHONY: fmp-backfill-statements
fmp-backfill-statements:     ## Backfill FMP statements (FMP_SYMBOLS=current|AAPL,MSFT|path.csv  FMP_KINDS=a,b)
	$(DI) fmp backfill-statements --symbols $(FMP_SYMBOLS) --kinds $(FMP_KINDS) $(if $(FMP_MAX),--max-symbols $(FMP_MAX),)

.PHONY: fmp-backfill-delisted
fmp-backfill-delisted:       ## Backfill FMP delisted-company roster (FMP_PAGES=N)
	$(DI) fmp backfill-delisted --max-pages $(FMP_PAGES)

.PHONY: fmp-build-dead-universe
fmp-build-dead-universe:     ## Build dead_universe.parquet from delisted roster + cached prices
	$(DI) fmp build-dead-universe --start $(FMP_START) --end $(FMP_END) --min-adv $(FMP_MIN_ADV) $(if $(FMP_MAX),--max-symbols $(FMP_MAX),)

# ── Live trading ──────────────────────────────────────────────────────────────

.PHONY: run
run:                         ## Live trading run  (safe mode — manual confirmation)
	$(DI) run --op-mode safe

.PHONY: run-auto
run-auto:                    ## Live trading run  (automated mode — no prompts)
	$(DI) run --op-mode automated

.PHONY: run-skip
run-skip:                    ## Live trading run, reuse existing CSV data  (faster)
	$(DI) run --op-mode safe --skip-data

.PHONY: run-dry
run-dry:                     ## Dry-run: skip data + no sentiment  (scoring + logic preview only)
	$(DI) run --op-mode no-sentiment --skip-data

# ── Backtesting ───────────────────────────────────────────────────────────────

DAYS    ?= 365
BT_MODE ?= liquid_universe_full

.PHONY: backtest
backtest:                    ## Backtest  (DAYS=N  BT_MODE=...)
	$(DI) backtest $(DAYS) --mode $(BT_MODE)

.PHONY: backtest-wf
backtest-wf:                 ## Walk-forward backtest  (low lookahead bias, DAYS=N)
	$(DI) backtest $(DAYS) --mode walk_forward_price_only_test

.PHONY: backtest-compare
backtest-compare:            ## A/B/C candidate selection mode comparison  (DAYS=N  BT_MODE=...)
	$(DI) backtest $(DAYS) --mode $(BT_MODE) --compare

# ── Parameter tuning ──────────────────────────────────────────────────────────

OBJ       ?= sharpe
TUNE_DAYS ?= 120
AUTO_DAYS ?= 90
MODE      ?=

.PHONY: tune
tune:                        ## Single-objective tune, no write  (TUNE_DAYS=N  OBJ=sharpe|calmar)
	$(DI) tune $(TUNE_DAYS) --objective $(OBJ) $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune
auto-tune:                   ## Dual-objective tune, walk-forward validation, no write  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-apply
auto-tune-apply:             ## auto-tune + write config.yaml if validation gates pass
	$(DI) auto-tune $(AUTO_DAYS) $(if $(MODE),--mode $(MODE),) --apply

.PHONY: auto-tune-llm
auto-tune-llm:               ## auto-tune + Claude second-opinion + apply
	$(DI) auto-tune $(AUTO_DAYS) $(if $(MODE),--mode $(MODE),) --apply --llm-review

.PHONY: auto-tune-active
auto-tune-active:            ## Active sleeve tune — score weights preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_core_weights $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-exits
auto-tune-active-exits:      ## Active sleeve tune — exit rules preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_exits $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-full
auto-tune-active-full:       ## Active sleeve tune — weights + exits preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_full_safe $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-factors
auto-tune-active-factors:    ## Active sleeve tune — factor internals preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_factor_internals $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-exit-floors
auto-tune-active-exit-floors: ## Active sleeve tune — DAE soft-exit floors preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_exit_floors $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-opportunity-cost
auto-tune-active-opportunity-cost: ## Active sleeve tune — opportunity-cost stall exit (needs opportunity_cost.enabled)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_opportunity_cost $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-candidate-filters
auto-tune-active-candidate-filters: ## Active sleeve tune — candidate selection filters preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_candidate_filters $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-sizing
auto-tune-active-sizing:     ## Active sleeve tune — position sizing / breadth preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_position_sizing $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-cooldown
auto-tune-active-cooldown:   ## Active sleeve tune — rebalance cadence + cooldowns preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_rebalance_cooldown $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-regime
auto-tune-active-regime:     ## Active sleeve tune — regime bull momentum tilt preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_regime_tilt $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-alpha
auto-tune-active-alpha:      ## Active sleeve tune — full alpha engine (high DOF) preset  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_alpha_engine $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-blends
auto-tune-active-blends:     ## Active sleeve tune — low-vol + residual-momentum blends (re-test)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_scoring_blends $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-active-archetypes
auto-tune-active-archetypes: ## Active sleeve tune — all 24 archetype lifecycle thresholds  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_archetype_lifecycle $(if $(MODE),--mode $(MODE),)

# ── Interaction-cluster joint tunes (co-tune params that share a decision surface) ─
.PHONY: auto-tune-cluster-buy-gate
auto-tune-cluster-buy-gate:  ## Joint tune — buy-gate cluster (weights+threshold+filters)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_buy_gate $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-cluster-momentum
auto-tune-cluster-momentum:  ## Joint tune — momentum-engine cluster (weight+subweights+tilt)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_momentum_engine $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-cluster-exit-ladder
auto-tune-cluster-exit-ladder: ## Joint tune — exit-ladder cluster (exits+floors+opp-cost)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_exit_ladder $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-cluster-breadth
auto-tune-cluster-breadth:   ## Joint tune — breadth/turnover cluster (sizing+filters+cadence)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_breadth_turnover $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune-cluster-quality
auto-tune-cluster-quality:   ## Joint tune — quality-stack cluster (weight+low_vol+min_q+floor)  (AUTO_DAYS=N)
	$(DI) auto-tune $(AUTO_DAYS) --scope active_sleeve_compounding --preset active_quality_stack $(if $(MODE),--mode $(MODE),)

# ── Research / diagnostics ────────────────────────────────────────────────────

OUTPUT_DIR ?= reports

.PHONY: stability
stability:                   ## Parameter stability scan across multiple windows  (research only, no writes)
	$(DI) stability-scan $(if $(MODE),--mode $(MODE),) --output-dir $(OUTPUT_DIR)

.PHONY: interaction-screen
interaction-screen:          ## Screen which param clusters synergize/clash when co-tuned  (PROFILE=quick|standard|deep, research only)
	$(DI) interaction-screen --profile $(if $(PROFILE),$(PROFILE),standard) $(if $(MODE),--mode $(MODE),) --output-dir $(OUTPUT_DIR)

.PHONY: auto-tune-all
auto-tune-all:               ## Staged coordinate-ascent over interaction clusters + full windowed validation  (PROFILE=quick|standard|deep, research only)
	$(DI) auto-tune-all --profile $(if $(PROFILE),$(PROFILE),standard) $(if $(MODE),--mode $(MODE),) $(if $(CLUSTERS),--clusters $(CLUSTERS),)

.PHONY: report
report:                      ## Quick 90-day backtest → print results + stability hint
	$(DI) report --output-dir $(OUTPUT_DIR)

.PHONY: regime
regime:                      ## Print current market regime  (live SPY + VIX fetch)
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.regimes import RegimeDetector; s = RegimeDetector().detect(); dma = f'{s.spy_vs_200dma_pct:+.2%}' if s.spy_vs_200dma_pct is not None else 'N/A'; print(f'Regime: {s.regime.upper()}  |  Confidence: {s.confidence:.0%}  |  VIX: {s.vix}  |  SPY vs 200DMA: {dma}'); print('Notes:', '  '.join(s.notes) if s.notes else 'none')"

.PHONY: snapshot-info
snapshot-info:               ## Show snapshot store status  (count + date range)
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.snapshots import list_snapshots; snaps = list_snapshots(); print(f'{len(snaps)} snapshots  |  {snaps[0][0]}  →  {snaps[-1][0]}' if snaps else 'No snapshots found in data/snapshots/')"

.PHONY: snapshot-backfill
snapshot-backfill:           ## Backfill parquet snapshots from existing agg_data CSVs
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.snapshots import backfill_from_csvs; n = backfill_from_csvs(); print(f'Backfilled {n} snapshot(s)')"

.PHONY: ic
ic:                          ## Print IC summary across default horizons  (needs ≥ 2 snapshots)
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC)'); from strategy.research import FactorResearchEngine; engine = FactorResearchEngine(); ic = engine.compute_multi_horizon_ic(); summ = engine.compute_ic_summary(ic); print(summ.sort_values(['factor','horizon_days']).to_string(index=False) if not summ.empty else 'Not enough snapshots — need ≥ 2')"

# ── Development ───────────────────────────────────────────────────────────────

.PHONY: install
install:                     ## Install / reinstall package in editable mode
	$(PYTHON) -m pip install -e ".[ui,dev]" -q

.PHONY: install-system
install-system:              ## Install editable, bypassing Homebrew protection  (macOS Homebrew Python)
	$(PYTHON) -m pip install -e ".[ui,dev]" --break-system-packages -q

.PHONY: test
test:                        ## Run full test suite
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-watch
test-watch:                  ## Re-run tests on file changes  (requires pytest-watch)
	$(PYTHON) -m ptw tests/ -- -q

.PHONY: lint
lint:                        ## Ruff lint over src/ and tests/  (config from pyproject.toml)
	$(PYTHON) -m ruff check $(SRC)/ tests/

.PHONY: format
format:                      ## Auto-format src/ with ruff
	$(PYTHON) -m ruff format $(SRC)/

.PHONY: type-check
type-check:                  ## MyPy type check  (non-strict; excludes ui/ and util.py)
	$(PYTHON) -m mypy src/core src/backtesting src/strategy src/portfolio src/reporting src/tuning src/config src/execution src/research

.PHONY: dead-code
dead-code:                   ## Vulture dead-code scan  (advisory — review before deleting)
	$(PYTHON) -m vulture src/ vulture_whitelist.py --min-confidence 80

.PHONY: complexity
complexity:                  ## Radon cyclomatic complexity + maintainability index
	$(RADON) cc $(SRC)/ -a -nb --total-average
	$(RADON) mi $(SRC)/ -nb

.PHONY: arch-check
arch-check:                  ## Import-linter layer boundary contracts
	$(LINT_IMPORTS)

.PHONY: pre-commit-install
pre-commit-install:          ## Install pre-commit hooks into .git/hooks
	$(PYTHON) -m pre_commit install

.PHONY: hygiene
hygiene: lint arch-check             ## Blocking hygiene suite  (lint + architecture; type-check is separate)
	@echo "Hygiene checks passed."

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:                        ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
