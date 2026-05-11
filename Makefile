VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
DI      := $(VENV)/bin/daily-investor
STREAMLIT := $(VENV)/bin/streamlit
SRC     := src

# ── UI ────────────────────────────────────────────────────────────────────────

.PHONY: ui
ui:                          ## Launch the Streamlit dashboard
	$(STREAMLIT) run $(SRC)/ui/streamlit_app.py

# ── Live trading ──────────────────────────────────────────────────────────────

.PHONY: run
run:                         ## Live trading run  (safe mode)
	$(DI) run --op-mode safe

.PHONY: run-auto
run-auto:                    ## Live trading run  (automated mode)
	$(DI) run --op-mode automated

.PHONY: run-skip
run-skip:                    ## Live trading run, reuse existing CSV data
	$(DI) run --op-mode safe --skip-data

# ── Research / analysis ───────────────────────────────────────────────────────

DAYS    ?= 365
BT_MODE ?= liquid_universe_sanity_test

.PHONY: backtest
backtest:                    ## Backtest  (DAYS=N  BT_MODE=...)
	$(DI) backtest $(DAYS) --mode $(BT_MODE)

OBJ     ?= sharpe
TUNE_DAYS ?= 120

.PHONY: tune
tune:                        ## Single-objective tune, no write  (TUNE_DAYS=N  OBJ=sharpe|calmar)
	$(DI) tune $(TUNE_DAYS) --objective $(OBJ)

AUTO_DAYS ?= 90

.PHONY: auto-tune
auto-tune:                   ## Dual-objective tune, walk-forward validation, no write
	$(DI) auto-tune $(AUTO_DAYS) --objective $(OBJ)

.PHONY: auto-tune-apply
auto-tune-apply:             ## auto-tune + write config.yaml if validation passes
	$(DI) auto-tune $(AUTO_DAYS) --objective $(OBJ) --apply

.PHONY: auto-tune-llm
auto-tune-llm:               ## auto-tune + Claude second-opinion + apply
	$(DI) auto-tune $(AUTO_DAYS) --objective $(OBJ) --apply --llm-review

.PHONY: stability
stability:                   ## Parameter stability scan  (research only)
	$(DI) stability-scan

.PHONY: report
report:                      ## Generate diagnostics report → reports/
	$(DI) report --output-dir reports

# ── Dev ───────────────────────────────────────────────────────────────────────

.PHONY: install
install:                     ## Install / reinstall package in editable mode
	$(PIP) install -e ".[ui,dev]" -q

.PHONY: test
test:                        ## Run full test suite
	/opt/homebrew/bin/pytest tests/ -q

.PHONY: test-watch
test-watch:                  ## Re-run tests on file changes (requires pytest-watch)
	/opt/homebrew/bin/ptw tests/ -- -q

.PHONY: help
help:                        ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
