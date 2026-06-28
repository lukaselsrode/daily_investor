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
##@ Dashboard

.PHONY: ui
ui:                          ## Launch the Streamlit dashboard
	$(STREAMLIT) run $(SRC)/ui/streamlit_app.py

# ── Data ──────────────────────────────────────────────────────────────────────
##@ Data

SKIP_NEWS ?=

.PHONY: fetch-data
fetch-data:                  ## Fetch all market data + snapshot, no trades  (SKIP_NEWS=1 reuses cached news)
	$(DI) fetch-data $(if $(SKIP_NEWS),--skip-fetch-news,)

.PHONY: update-outcomes
update-outcomes:             ## Backfill future return outcomes for past decisions (calibration only — never touches live scoring)
	$(DI) update-outcomes

FMP_SYMBOLS      ?= current
FMP_START        ?= 2006-01-01
FMP_END          ?= 2030-01-01
FMP_MAX          ?=
FMP_KINDS        ?= income-statement,balance-sheet-statement,cash-flow-statement
FMP_PAGES        ?= 50
FMP_MIN_ADV      ?= 500000

.PHONY: prepare-data
prepare-data:                ## One-shot fetch + deep-backfill of ALL survivorship-free data  (resumable)
	-$(DI) fetch-data
	$(DI) fmp backfill-delisted --max-pages $(FMP_PAGES)
	$(DI) fmp backfill-prices --symbols $(FMP_SYMBOLS) --start $(FMP_START) --end $(FMP_END) $(if $(FMP_MAX),--max-symbols $(FMP_MAX),)
	$(DI) fmp build-dead-universe --start $(FMP_START) --end $(FMP_END) --min-adv $(FMP_MIN_ADV) --fetch-prices
	$(DI) fmp backfill-statements --symbols $(FMP_SYMBOLS) --kinds $(FMP_KINDS)
	$(MAKE) snapshot-backfill
	$(DI) fmp validate-cache
	@echo "prepare-data complete. (^VIX auto-fetched on first backtest/precomp build.)"

.PHONY: fmp-status
fmp-status:                  ## FMP cache coverage / key status  (granular backfills: daily-investor fmp <action>)
	$(DI) fmp status

# ── Live trading ──────────────────────────────────────────────────────────────
##@ Live trading

OP_MODE   ?= safe
SKIP_DATA ?=

.PHONY: run
run:                         ## Live trading run (OP_MODE=safe|automated|no-sentiment  SKIP_DATA=1  SKIP_NEWS=1)
	$(DI) run --op-mode $(OP_MODE) $(if $(SKIP_DATA),--skip-data,) $(if $(SKIP_NEWS),--skip-fetch-news,)

# ── Backtesting ───────────────────────────────────────────────────────────────
##@ Backtesting
##: BT_MODE / MODE values: liquid_universe_full (default) · walk_forward_price_only_test · current_universe_stress_test

DAYS    ?= 365
BT_MODE ?= liquid_universe_full
COMPARE ?=

.PHONY: backtest
backtest:                    ## Backtest (DAYS=N  BT_MODE=...  COMPARE=1). Walk-forward: BT_MODE=walk_forward_price_only_test
	$(DI) backtest $(DAYS) --mode $(BT_MODE) $(if $(COMPARE),--compare,)

# ── Parameter tuning ──────────────────────────────────────────────────────────
##@ Parameter tuning
##: MODE = backtest universe mode (see Backtesting above; empty = engine default)

OBJ       ?= sharpe
TUNE_DAYS ?= 120
AUTO_DAYS ?= 90
MODE      ?=
PRESET    ?=
APPLY     ?=
LLM       ?=

.PHONY: tune
tune:                        ## Single-objective tune, no write  (TUNE_DAYS=N  OBJ=sharpe|calmar  MODE=<universe>)
	$(DI) tune $(TUNE_DAYS) --objective $(OBJ) $(if $(MODE),--mode $(MODE),)

.PHONY: auto-tune
auto-tune:                   ## Dual-objective tune + tournament + gate tiers (AUTO_DAYS=N  APPLY=1  LLM=1  PRESET=name  MODE=<universe>)
	$(DI) auto-tune $(AUTO_DAYS) $(if $(MODE),--mode $(MODE),) \
	  $(if $(PRESET),--scope active_sleeve_compounding --preset $(PRESET),) \
	  $(if $(LLM),--apply --llm-review,$(if $(APPLY),--apply,))

.PHONY: list-presets
list-presets:                ## List tunable presets (use a name with auto-tune PRESET=...)
	$(DI) list-presets

# ── Research / diagnostics ────────────────────────────────────────────────────
##@ Research / diagnostics

OUTPUT_DIR ?= reports
REGIME ?= neutral

.PHONY: stability
stability:                   ## Parameter stability scan across multiple windows  (research only, no writes)
	$(DI) stability-scan $(if $(MODE),--mode $(MODE),) --output-dir $(OUTPUT_DIR)

.PHONY: interaction-screen
interaction-screen:          ## Screen which param clusters synergize/clash when co-tuned  (PROFILE=quick|standard|deep, research only)
	$(DI) interaction-screen --profile $(if $(PROFILE),$(PROFILE),standard) $(if $(MODE),--mode $(MODE),) --output-dir $(OUTPUT_DIR)

.PHONY: auto-tune-all
auto-tune-all:               ## Staged coordinate-ascent over interaction clusters + full windowed validation  (PROFILE=quick|standard|deep, research only)
	$(DI) auto-tune-all --profile $(if $(PROFILE),$(PROFILE),standard) $(if $(MODE),--mode $(MODE),) $(if $(CLUSTERS),--clusters $(CLUSTERS),)

.PHONY: regime-sizing
regime-sizing:               ## Random-window regime sizing/exposure grid (REGIME=neutral; research only, no writes)
	PYTHONPATH=$(SRC) $(PYTHON) scripts/regime_sizing_random_window.py --regime $(REGIME) --output $(OUTPUT_DIR)/regime_sizing_$(REGIME).csv

.PHONY: report
report:                      ## Quick 90-day backtest → print results + stability hint
	$(DI) report --output-dir $(OUTPUT_DIR)

OFFLINE             ?=
REDDIT_BEARER_TOKEN ?=
DAILY_THREAD_ID     ?=
DAILY_THREAD_URL    ?=
DAILY_THREAD_LIMIT  ?=

# Easy live daily-thread run (paste your token + thread id):
#   make odte-report REDDIT_BEARER_TOKEN="$REDDIT_TOKEN" DAILY_THREAD_ID=1u9240r
# Hands-off run (for the Hermes agent): just `make odte-report` — when those vars are absent it
# auto-loads the bearer token + daily-thread-id from ~/0dte/ (reddit_token.json/{"token","expires"}
# or legacy ~/.reddit_token.json; daily_thread_id.txt or config.json). Explicit vars always win.
# Optional: DAILY_THREAD_LIMIT=200 to cap comments read (default: auto-paginate the WHOLE thread).
# You never set comment depth/nesting — that's handled with sane defaults.
# Each run also dumps analyzed texts to data/odte/scrape/{reddit,x}_text.txt plus timestamped snapshots.
# Agent-friendly: `make odte-report JSON=1` emits clean signal-only JSON (no paper/disclaimer prose);
# pair with 2>/dev/null to drop log lines, e.g. `make odte-report JSON=1 2>/dev/null`.
.PHONY: odte-report
odte-report:                 ## 0DTE social watchlist — PAPER ONLY (live: REDDIT_BEARER_TOKEN="..." DAILY_THREAD_ID=...; OFFLINE=1 dry run; JSON=1 agent output)
	@$(DI) odte-social-report $(if $(OFFLINE),--no-fetch,) $(if $(JSON),--json,) \
	  $(if $(REDDIT_BEARER_TOKEN),--reddit-bearer-token $(REDDIT_BEARER_TOKEN),) \
	  $(if $(DAILY_THREAD_ID),--daily-thread-id $(DAILY_THREAD_ID),) \
	  $(if $(DAILY_THREAD_URL),--daily-thread-url $(DAILY_THREAD_URL),) \
	  $(if $(DAILY_THREAD_LIMIT),--daily-thread-limit $(DAILY_THREAD_LIMIT),)

# Script-only 0DTE watchdog — NO LLM, NO Robinhood, places NO orders. Runs the LOCAL report,
# diffs the actionable candidate vs the prior run, and writes data/odte/{watchdog_state,triggers}.json.
# Empty stdout when nothing actionable; compact one-line JSON on a trigger. For a no_agent cron.
#   make odte-watchdog            # cron form: empty unless a trigger fires
#   make odte-watchdog JSON=1     # always print compact state
#   make odte-watchdog OFFLINE=1  # offline dry run (cache-only, no network)
.PHONY: odte-watchdog
odte-watchdog:               ## 0DTE script-only watchdog — NO LLM/Robinhood (JSON=1 state; OFFLINE=1 dry run)
	@$(DI) odte-watchdog $(if $(OFFLINE),--no-fetch,) $(if $(JSON),--json,)

# Broker-AWARE, DECISION-ONLY live-position watchdog — places NO orders, NO broker/LLM calls.
# Reads data/odte/active_trade.json + a caller-supplied snapshot (Hermes feeds real MCP broker values;
# never faked) and writes data/odte/{position_state,position_decision}.json. Empty stdout on HOLD/
# NO_POSITION; compact JSON on an actionable decision.
#   make odte-position JSON=1                 # always print the decision
#   make odte-position SNAPSHOT=data/odte/snap.json JSON=1   # feed a live snapshot file
.PHONY: odte-position
odte-position:               ## 0DTE live-position decision watchdog — NO orders/broker (SNAPSHOT=path; JSON=1)
	@$(DI) odte-position $(if $(SNAPSHOT),--snapshot $(SNAPSHOT),) $(if $(PLAN),--plan $(PLAN),) $(if $(JSON),--json,)

# 0DTE decision journal — local/offline, NO broker/LLM/secrets. Append events, then report.
#   make odte-journal EVENT='{"event_type":"postmortem","trade_id":"t1","mode":"scalp",...}'
#   make odte-journal EVENT_FILE=data/odte/event.json
#   make odte-journal-report JSON=1          # metrics JSON
#   make odte-journal-report WRITE=1         # writes data/odte/reports/{md,csv}
.PHONY: odte-journal
odte-journal:                ## Append a 0DTE journal event (EVENT='{...}' or EVENT_FILE=path; JSON=1)
	@$(DI) odte-journal $(if $(EVENT),--event-json '$(EVENT)',) $(if $(EVENT_FILE),--event $(EVENT_FILE),) $(if $(JSON),--json,)

.PHONY: odte-ingest-artifacts
odte-ingest-artifacts:       ## Fold loose data/odte/*.json controller artifacts into the journal — idempotent (DATE=YYYY-MM-DD; DRYRUN=1; DAYPACKET=1; JSON=1)
	@$(DI) odte-ingest-artifacts $(if $(DATE),--date $(DATE),) $(if $(DRYRUN),--dry-run,) $(if $(DAYPACKET),--day-packet,) $(if $(DATA_DIR),--data-dir $(DATA_DIR),) $(if $(JOURNAL),--journal $(JOURNAL),) $(if $(JSON),--json,)

.PHONY: odte-journal-report
odte-journal-report:         ## Summarize the 0DTE journal (JSON=1 metrics; WRITE=1 md/csv artifacts)
	@$(DI) odte-journal-report $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

# 0DTE option-chain gamma / pin map — PURE/OFFLINE (no broker/LLM/network). Reads option-quote rows
# Hermes/RH exported to INPUT=path; honest concentration only (NOT dealer GEX).
#   make odte-gamma-map INPUT=data/odte/spy_chain.json SPOT=734.8 UNDERLYING=SPY JSON=1
#   make odte-gamma-map INPUT=data/odte/spy_chain.json WRITE=1   # writes data/odte/reports/ artifacts
.PHONY: odte-gamma-map
odte-gamma-map:              ## 0DTE gamma/pin map from exported quote rows — NO broker (INPUT=path; SPOT=; JSON=1; WRITE=1)
	@$(DI) odte-gamma-map $(if $(INPUT),--input $(INPUT),) $(if $(SPOT),--spot $(SPOT),) $(if $(UNDERLYING),--underlying $(UNDERLYING),) $(if $(EXPIRATION),--expiration $(EXPIRATION),) $(if $(JSON),--json,) $(if $(WRITE),--write,)

# Pair the two SEPARATE arrays Robinhood returns (option quotes/market-data + option instruments)
# into flat rows odte-gamma-map consumes — PURE/OFFLINE (no broker/LLM/network). HONEST: ABSOLUTE
# gamma/OI rows only, never dealer GEX. Pipe the output into odte-gamma-map via INPUT=.
#   make odte-rh-rows QUOTES=data/odte/spy_quotes.json INSTRUMENTS=data/odte/spy_instruments.json OUT=data/odte/spy_chain.json
#   make odte-gamma-map INPUT=data/odte/spy_chain.json SPOT=734.8 UNDERLYING=SPY JSON=1
.PHONY: odte-rh-rows
odte-rh-rows:                ## Pair RH option quotes+instruments into gamma-map rows — NO broker (QUOTES=path; INSTRUMENTS=path; OUT=path)
	@$(DI) odte-rh-rows $(if $(QUOTES),--quotes $(QUOTES),) $(if $(INSTRUMENTS),--instruments $(INSTRUMENTS),) $(if $(OUT),--out $(OUT),)

# 0DTE candidate vehicle/contract score — PURE/OFFLINE, NO broker/network/LLM. This is the
# non-sentiment "is this a good or bad bet for the day?" layer. Feed it a candidate contract plus
# optional market/gamma JSON gathered by the controller; it returns GOOD_BET / WATCH / BAD_BET.
#   make odte-vehicle-score CONTRACT=data/odte/candidate.json MARKET=data/odte/market.json GAMMA=data/odte/reports/odte_gamma_map_qqq.json BP=108 JSON=1
.PHONY: odte-vehicle-score
odte-vehicle-score:          ## 0DTE non-sentiment vehicle score — NO broker (CONTRACT=path; MARKET=path; GAMMA=path; BP=; JSON=1)
	@$(DI) odte-vehicle-score $(if $(CONTRACT),--contract $(CONTRACT),) $(if $(MARKET),--market $(MARKET),) $(if $(GAMMA),--gamma $(GAMMA),) $(if $(DIRECTION),--direction $(DIRECTION),) $(if $(BP),--buying-power $(BP),) $(if $(JSON),--json,) $(if $(WRITE),--write,)

# 0DTE day-regime score — PURE/OFFLINE, NO broker/network/LLM. The "is today a GOOD_DAY to press
# directional 0DTE, a CHOP day to scalp, or an AVOID day to stay flat?" layer, scored from a market
# snapshot (VIX/VIXY, gap, per-index ORB/VWAP, expected move, minutes-to-close). Companion to
# odte-vehicle-score (which scores one contract).
#   make odte-day-score MARKET=data/odte/market.json GAMMA=data/odte/reports/odte_gamma_map_qqq.json JSON=1
.PHONY: odte-day-score
odte-day-score:              ## 0DTE non-sentiment day score — NO broker (MARKET=path; GAMMA=path; JSON=1; WRITE=1)
	@$(DI) odte-day-score $(if $(MARKET),--market $(MARKET),) $(if $(GAMMA),--gamma $(GAMMA),) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

# PURE/OFFLINE thesis->entry gate. Assembles a journalable entry-gate decision (enter/deny/veto/
# observe) from the upstream artifacts. Records intent ONLY — places NO orders, NO broker/network/LLM.
# execution_allowed is True only when every required gate is explicitly true and not scan_only/restricted.
#   make odte-entry-gate TRIGGER=data/odte/triggers.json DAY_SCORE=data/odte/reports/odte_day_score.json VEHICLE=data/odte/reports/odte_vehicle_score_qqq.json BROKER=data/odte/broker.json JSON=1
#   make odte-entry-gate TRIGGER=data/odte/triggers.json ... JOURNAL=1   # also append an entry_decision event
.PHONY: odte-entry-gate
odte-entry-gate:             ## 0DTE thesis->entry gate — NO orders/broker (TRIGGER=; DAY_SCORE=; VEHICLE=; GAMMA=; BROKER=; SCAN_ONLY=1; PROMOTE=1; JOURNAL=1; JSON=1)
	@$(DI) odte-entry-gate $(if $(TRIGGER),--trigger $(TRIGGER),) $(if $(CANDIDATE),--candidate $(CANDIDATE),) $(if $(DAY_SCORE),--day-score $(DAY_SCORE),) $(if $(VEHICLE),--vehicle-score $(VEHICLE),) $(if $(GAMMA),--gamma $(GAMMA),) $(if $(BROKER),--broker $(BROKER),) $(if $(SCAN_ONLY),--scan-only,) $(if $(PROMOTE),--promote-to-execution,) $(if $(JOURNAL),--journal,) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

# FMP single-name context for 0DTE meme/squeeze SANITY — read-only, NO orders, NO options/gamma.
# Cheap FMP stable fundamentals (profile/quote/shares-float/key-metrics-ttm/news) + squeeze profile.
# FMP options are unavailable; Robinhood stays the gamma source. Fail-closed without FMP_KEY.
#   make odte-fmp-context SYMBOL=WEN JSON=1
#   make odte-fmp-context SYMBOL=WEN WRITE=1   # writes data/odte/reports/ artifacts
.PHONY: odte-fmp-context
odte-fmp-context:            ## FMP meme/squeeze sanity context — NO orders/options (SYMBOL=WEN; JSON=1; WRITE=1)
	@$(DI) odte-fmp-context $(SYMBOL) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

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
##@ Development

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
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		/^##:/ { printf "    \033[2m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
		END { print "" }' $(MAKEFILE_LIST)
	@printf "\nUsage: \033[36mmake <target> [VAR=val ...]\033[0m   e.g. make run OP_MODE=automated\n\n"

.DEFAULT_GOAL := help
