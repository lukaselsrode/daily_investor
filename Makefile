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
# Machine-readable output: gates + failing_gates/unknown_gates + next_action/next_command — a passed
# gate says "mint an execution lease, then broker review/place, then the order guard"; a BP/vehicle-fit
# fail says scan the other ETF vehicles (QQQ/SPY/IWM) before declaring no-trade; a missing input names
# the exact command that produces it. PROMOTE=1 (--promote-to-execution) is DEPRECATED + fail-closed
# (2026-07-23 incident): it answers with execution_lease_required — and a CONFIRM_ENTRY
# candidate decision is refused outright (use_odte_convert): conversion runs ONLY via odte-convert.
#   make odte-entry-gate TRIGGER=data/odte/triggers.json DAY_SCORE=data/odte/reports/odte_day_score.json VEHICLE=data/odte/reports/odte_vehicle_score_qqq.json BROKER=data/odte/broker.json JSON=1
#   make odte-entry-gate CANDIDATE=data/odte/active_candidate.json CANDIDATE_DECISION=data/odte/candidate_decision.json DAY_SCORE=<fresh> VEHICLE=<fresh> BROKER=<fresh> CONFIRMATIONS=<fresh-live-confirmations.json> JSON=1 WRITE=1 JOURNAL=1
#   make odte-entry-gate TRIGGER=data/odte/triggers.json ... JOURNAL=1   # also append an entry_decision event
.PHONY: odte-entry-gate
odte-entry-gate:             ## 0DTE thesis->entry gate — NO orders/broker (TRIGGER=; CANDIDATE=; CANDIDATE_DECISION=; DAY_SCORE=; VEHICLE=; GAMMA=; BROKER=; CONFIRMATIONS=; SCAN_ONLY=1; JOURNAL=1; JSON=1; PROMOTE=1 deprecated/fail-closed)
	@$(DI) odte-entry-gate $(if $(TRIGGER),--trigger $(TRIGGER),) $(if $(CANDIDATE),--candidate $(CANDIDATE),) $(if $(CANDIDATE_DECISION),--candidate-decision $(CANDIDATE_DECISION),) $(if $(DAY_SCORE),--day-score $(DAY_SCORE),) $(if $(VEHICLE),--vehicle-score $(VEHICLE),) $(if $(GAMMA),--gamma $(GAMMA),) $(if $(BROKER),--broker $(BROKER),) $(if $(CONFIRMATIONS),--confirmations $(CONFIRMATIONS),) $(if $(SCAN_ONLY),--scan-only,) $(if $(PROMOTE),--promote-to-execution,) $(if $(JOURNAL),--journal,) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

# 0DTE execution-safety layer (2026-07-23 delayed-fill remediation). The ONE tier that mints
# execution authority — as a SINGLE-USE, short-lived lease (default 30s TTL, hard cap 60s) bound to
# one exact symbol/direction/contract/quantity/price ceiling — plus the pending-order cancel-first
# guard. Decision/record tooling only: places NO orders; the Hermes MCP lane owns broker review/
# place/cancel. Runtime ladder: SCAN_ONLY → CANDIDATE_CONFIRMED → EXECUTION_LEASE_READY →
# BROKER_REVIEW → PENDING_ORDER_GUARD → FILLED_POSITION_MANAGEMENT → EXIT/FLAT.
#   make odte-execution-authorize GATE=data/odte/reports/odte_entry_gate_spy.json CANDIDATE_DECISION=data/odte/candidate_decision.json VEHICLE=data/odte/reports/odte_vehicle_score_spy.json BROKER=data/odte/broker.json MARKET=data/odte/market.json JSON=1 WRITE=1
#   make odte-order-guard ORDER=data/odte/order_truth.json MARKET=data/odte/market.json JSON=1 WRITE=1 JOURNAL=1
.PHONY: odte-execution-authorize
odte-execution-authorize:    ## Mint/refuse a single-use 0DTE execution lease — NO orders (GATE=; CANDIDATE_DECISION=; VEHICLE=; BROKER=; MARKET=; POLICY=; JSON=1; WRITE=1; JOURNAL=1)
	@$(DI) odte-execution-authorize $(if $(GATE),--gate $(GATE),) $(if $(CANDIDATE_DECISION),--candidate-decision $(CANDIDATE_DECISION),) $(if $(VEHICLE),--vehicle-score $(VEHICLE),) $(if $(BROKER),--broker $(BROKER),) $(if $(MARKET),--market $(MARKET),) $(if $(POLICY),--policy $(POLICY),) $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(JOURNAL),--journal,)

# Atomic conversion (2026-08-02 retune): fresh tape re-check → computed confirmations → entry gate
# → execution lease in ONE process under ONE clock, so the in-process freshness TTLs pass by
# construction. Caller fetches FRESH market/broker/contract snapshots immediately before the call.
# Non-converting stages journal an identity-bound terminal no_trade_decision. Places NO orders.
#   make odte-convert MARKET=data/odte/market.json BROKER=data/odte/broker.json CONTRACT=data/odte/contract.json JSON=1
.PHONY: odte-convert
odte-convert:                ## Atomic confirm→gate→lease conversion in ONE process — NO orders (CANDIDATE=; MARKET=; BROKER=; CONTRACT=; GAMMA=; POLICY=; NO_WRITE=1; NO_JOURNAL=1; JSON=1)
	@$(DI) odte-convert $(if $(CANDIDATE),--candidate $(CANDIDATE),) $(if $(MARKET),--market $(MARKET),) $(if $(BROKER),--broker $(BROKER),) $(if $(CONTRACT),--contract $(CONTRACT),) $(if $(GAMMA),--gamma $(GAMMA),) $(if $(POLICY),--policy $(POLICY),) $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(JOURNAL_PATH),--journal-path $(JOURNAL_PATH),) $(if $(NO_WRITE),--no-write,) $(if $(NO_JOURNAL),--no-journal,) $(if $(JSON),--json,)

# Deterministic data/odte sweep — the ONLY sanctioned cleanup (hardcoded keep-list protects the
# canonical loop state; ad-hoc mv sweeps are forbidden). Dry-run unless APPLY=1.
#   make odte-cleanup            # dry-run: list what would be swept
#   make odte-cleanup APPLY=1 PRUNE_SCRAPE=1
.PHONY: odte-cleanup
odte-cleanup:                ## Sweep non-canonical data/odte artifacts — dry-run by default (APPLY=1; PRUNE_SCRAPE=1; SCRAPE_KEEP=N; JSON=1)
	@$(DI) odte-cleanup $(if $(APPLY),--apply,) $(if $(PRUNE_SCRAPE),--prune-scrape,) $(if $(SCRAPE_KEEP),--scrape-keep $(SCRAPE_KEEP),) $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(JSON),--json,)

.PHONY: odte-order-guard
odte-order-guard:            ## 0DTE pending-order cancel-first guard — NO orders (ORDER=; LEASE=; MARKET=; JSON=1; WRITE=1; JOURNAL=1)
	@$(DI) odte-order-guard $(if $(ORDER),--order $(ORDER),) $(if $(LEASE),--lease $(LEASE),) $(if $(MARKET),--market $(MARKET),) $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(JOURNAL),--journal,)

.PHONY: odte-candidate-watch
odte-candidate-watch:        ## 0DTE pre-entry candidate HAWK — NO orders/broker (CANDIDATE=; MARKET=; DAY_SCORE=; VEHICLE=; GAMMA=; BROKER_HEALTH=; WRITE=1; JSON=1)
	@$(DI) odte-candidate-watch $(if $(CANDIDATE),--candidate $(CANDIDATE),) $(if $(MARKET),--market $(MARKET),) $(if $(DAY_SCORE),--day-score $(DAY_SCORE),) $(if $(VEHICLE),--vehicle-score $(VEHICLE),) $(if $(GAMMA),--gamma $(GAMMA),) $(if $(BROKER_HEALTH),--broker-health $(BROKER_HEALTH),) $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(WRITE),--write,) $(if $(JSON),--json,)

# FMP single-name context for 0DTE meme/squeeze SANITY — read-only, NO orders, NO options/gamma.
# Cheap FMP stable fundamentals (profile/quote/shares-float/key-metrics-ttm/news) + squeeze profile.
# FMP options are unavailable; Robinhood stays the gamma source. Fail-closed without FMP_KEY.
#   make odte-fmp-context SYMBOL=WEN JSON=1
#   make odte-fmp-context SYMBOL=WEN WRITE=1   # writes data/odte/reports/ artifacts
.PHONY: odte-fmp-context
odte-fmp-context:            ## FMP meme/squeeze sanity context — NO orders/options (SYMBOL=WEN; JSON=1; WRITE=1)
	@$(DI) odte-fmp-context $(SYMBOL) $(if $(JSON),--json,) $(if $(WRITE),--write,) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

# One read-only surface for the live loop: summarizes data/odte artifacts (active_trade /
# position_decision / triggers / decision_journal) into the current state + coarse cron POSTURE
# (MANAGE_POSITION / EXECUTION_READY / SCOUT_FRESH_SETUP / WAIT_FRESH_CONFIRMATION / FLAT_NO_TRADE /
# STALE_DATA_BLOCKED / BROKER_DEGRADED) + per-artifact freshness + next command. The posture ladder
# drives action, not nudges: MANAGE_POSITION = quote & manage the open position NOW; EXECUTION_READY =
# every gate passed — broker review/place under standing auth, then odte-position; SCOUT_FRESH_SETUP =
# candidate confirmed — build/promote a fresh entry gate; WAIT_FRESH_CONFIRMATION = keep polling, the
# payload names the exact trigger needed; FLAT_NO_TRADE = normal idle tick, never "stale".
# PURE/OFFLINE — makes NO broker call. BROKER_HEALTH=path (else data/odte/broker_health.json) folds a
# SUPPLIED/PROBED broker-health JSON (Hermes writes it from an MCP/CLI probe) so the lane reads
# ok/down/stale/read-only-fallback. BROKER TRUTH FIRST: a STALE broker_health.json is an outdated probe
# FILE, not a confirmed fault — the payload carries broker_lane.refresh_command (re-probe parent MCP,
# rewrite the file) instead of pretending live orders are impossible; only a confirmed down/read-only
# lane reads BROKER_DEGRADED. Fails closed by default (live mode): unknown/missing broker can't
# authorize a live order. OFFLINE=1 relaxes to pure decision-support.
#   make odte-loop-status            # Markdown: posture + where in the loop + what runs next
#   make odte-loop-status JSON=1     # compact machine payload
#   make odte-loop-status BROKER_HEALTH=data/odte/broker_health.json JSON=1
.PHONY: odte-loop-status
odte-loop-status:            ## 0DTE loop state machine — posture + where in scan→exit→review + next command (BROKER_HEALTH=path; OFFLINE=1; JSON=1)
	@$(DI) odte-loop-status $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(BROKER_HEALTH),--broker-health $(BROKER_HEALTH),) $(if $(OFFLINE),--offline,) $(if $(JSON),--json,)

# ── 0DTE fast lane (two-lane architecture, 2026-08-05) ──────────────────────────────────────
# The slow lane (Hermes) ARMs machine-readable intents; the deterministic daemon fires them
# with zero LLM turns. Stage file (data/odte/fast_lane_stage.json) is the mode authority:
# shadow → exits_live → entries_live. Flags can only be MORE conservative than the stage.
#   make odte-arm INTENT=intent.json            # validate + arm (fail-closed)
#   make odte-arm DISARM=all                    # kill switch #2
#   make odte-fast-lane MODE=shadow             # run the daemon (or ONCE=1 for one tick)
#   make odte-fast-lane-pause / -resume         # kill switch #1 (halts placements next tick)
#   make odte-fast-lane-stage STAGE=shadow      # rollout stage control / kill switch #4
#   make odte-shadow-report                     # the rollout-gate adjudication surface
.PHONY: odte-arm
odte-arm:                    ## Arm/disarm/list fast-lane intents — NO orders (INTENT=path; INTENT_JSON='{...}'; DISARM=id|all; LIST=1; JSON=1)
	@$(DI) odte-arm $(if $(INTENT),--intent $(INTENT),) $(if $(INTENT_JSON),--intent-json '$(INTENT_JSON)',) $(if $(DISARM),--disarm $(DISARM),) $(if $(LIST),--list,) $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(JSON),--json,)

.PHONY: odte-fast-lane
odte-fast-lane:              ## Run the fast-lane daemon (MODE=shadow|exits|live; ONCE=1; ACCOUNT=; STATE_DIR=)
	@$(DI) odte-fast-lane $(if $(filter shadow,$(MODE)),--shadow,) $(if $(filter exits,$(MODE)),--exits-only,) $(if $(filter live,$(MODE)),--live,) $(if $(ONCE),--once,) $(if $(ACCOUNT),--account $(ACCOUNT),) $(if $(STATE_DIR),--state-dir $(STATE_DIR),)

.PHONY: odte-fast-lane-pause
odte-fast-lane-pause:        ## KILL SWITCH: halt all fast-lane placements on the next tick
	@touch data/odte/fast_lane_pause && echo "fast lane PAUSED (data/odte/fast_lane_pause)"

.PHONY: odte-fast-lane-resume
odte-fast-lane-resume:       ## Remove the fast-lane pause file
	@rm -f data/odte/fast_lane_pause && echo "fast lane resumed"

.PHONY: odte-fast-lane-stage
odte-fast-lane-stage:        ## Set the rollout stage (STAGE=shadow|exits_live|entries_live)
	@test -n "$(STAGE)" || { echo "usage: make odte-fast-lane-stage STAGE=shadow|exits_live|entries_live"; exit 2; }
	@echo '{"stage": "$(STAGE)", "set_at": "'$$(date -u +%Y-%m-%dT%H:%M:%SZ)'", "set_by": "make odte-fast-lane-stage"}' > data/odte/fast_lane_stage.json
	@cat data/odte/fast_lane_stage.json

.PHONY: odte-shadow-report
odte-shadow-report:          ## Shadow-vs-live divergence report — the rollout-gate evidence (JSON=1)
	@$(DI) odte-shadow-report $(if $(STATE_DIR),--state-dir $(STATE_DIR),) $(if $(JSON),--json,)

.PHONY: odte-fast-lane-install
odte-fast-lane-install:      ## Install the launchd supervisor (Mon-Fri 09:25 ET start)
	@cp scripts/launchd/com.dailyinvestor.odte-fastlane.plist ~/Library/LaunchAgents/ && launchctl unload ~/Library/LaunchAgents/com.dailyinvestor.odte-fastlane.plist 2>/dev/null; launchctl load ~/Library/LaunchAgents/com.dailyinvestor.odte-fastlane.plist && echo "installed + loaded"

.PHONY: odte-fast-lane-uninstall
odte-fast-lane-uninstall:    ## KILL SWITCH: unload + remove the launchd supervisor
	@launchctl unload ~/Library/LaunchAgents/com.dailyinvestor.odte-fastlane.plist 2>/dev/null; rm -f ~/Library/LaunchAgents/com.dailyinvestor.odte-fastlane.plist && echo "uninstalled"

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
