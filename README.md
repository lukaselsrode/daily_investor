# Daily Investor

A systematic investment platform built on Robinhood. Combines multi-factor fundamental scoring, AI-powered sentiment, and a full quantitative research stack — backtesting, walk-forward validation, parameter stability, regime detection, factor IC analytics, distribution regime analysis, and exposure diagnostics.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/lukaselsrode/daily_investor.git
cd daily_investor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"

# 2. Add credentials (.env at project root)
cat > .env <<EOF
RB_ACCT=your_robinhood_email
RB_CREDS=your_robinhood_password
RB_MFA_SECRET=your_totp_secret        # optional — skips interactive MFA prompt
ANTHROPIC_API_KEY=your_anthropic_key  # required for sentiment + LLM tune review
FMP_KEY=your_fmp_key                  # optional — enables FMP backfills / survivorship-free cache refresh
EOF

# 3. Verify setup — fetch data and save first snapshot (no trades placed)
make fetch-data

# 4. Run a quick backtest to confirm everything works
make backtest DAYS=90

# 5. Launch the dashboard
make ui
```

That's it. The dashboard opens at `http://localhost:8501`.

**No Robinhood account?** The backtester and research tools work on historical CSV data — skip steps 2–3 and run `make backtest` or `make ui` directly after seeding `data/` with your own CSV files.

---

## Key Features

- **Multi-Factor Scoring** — Value (P/E + P/B, sector-relative), quality, income, and momentum → single `value_metric`
- **Momentum** — Relative strength vs SPY (3m/6m), risk-adjusted return, DMA trend structure, short-term momentum — all cross-sectionally percentile-ranked per day, causal
- **Value** — Sector-relative winsorized percentile ranking with distress penalties; replaces ratio-based value scoring
- **Three-Tier Market Regime** — Bullish / Neutral / Defensive via SPY 200DMA + VIX; raises ETF allocation, limits active buys, tightens stops in defensive mode
- **Factor Research Platform** — `FactorResearchEngine`: multi-horizon IC (5/20/60/120/252d), factor decay curves, decile monotonicity, rolling ICIR, cumulative IC
- **Distribution Regime Analysis** — Bimodality detection (GMM + BC test), local IC, cluster analysis, conditional alpha, threshold simulation — investigates whether alpha concentrates in score tails
- **Parquet Snapshot Store** — Each scoring run saves `data/snapshots/YYYY_MM_DD.parquet` for rolling IC and forward-return validation
- **Exposure Analytics** — Factor tilts (z-score vs universe), sector weights, HHI concentration, rolling exposure drift
- **Disciplined Sell Engine** — Hard sells (stop-loss, yield trap, quality floor) execute immediately; soft sells (take-profit, weak value) require Claude confirmation
- **Anti-Lookahead Backtest** — All rolling features computed causally; contribution-adjusted TWR (chain-link) for portfolio and benchmark
- **Validation-Aware Auto-Tune** — `scipy.differential_evolution` across Sharpe and Calmar feeds a multi-source candidate tournament (optima, blends, incumbent blends, `--random-topk` robust-search candidates, `--leads` saved vectors); the winner must pass held-out split gates, incumbent-relative excess/turnover gates, a paired random-window gate, a regime-symmetric 90/180/365/730d multi-horizon confirmation (catastrophe-scale tolerances — no single window holds a fine-grained veto), and a stress gauntlet through named historical bear/stress episodes (GFC '08 → 2022) before config.yaml is written. The tournament winner is also saved to `data/leads_selected_<days>d.npy` so a gate-blocked near-miss is re-testable via `--leads`. A churn penalty inside the DE objective steers the search away from turnover regions the gates reject
- **Optional LLM Tune Review** — Routes optimizer candidates through Claude for a second-opinion before applying
- **Batch AI Sentiment** — Async concurrent Claude calls with exponential backoff
- **Contribution-Timing Overlay** *(config-gated, ships disabled)* — Buy-the-dip weekly contribution sizing: a causal dip score (1w/1m returns, 20/60d drawdowns, 50/200DMA gaps) maps to a contribution multiplier under a rolling monthly budget with carry-forward; defensive-regime cap prevents knife-catching. `contribution_timing` tuning preset; compare via `python scripts/contribution_timing_compare.py` (flat vs default vs tuned across 90/180/365/730d + random windows, reporting MWR/ending value)
- **Streamlit Dashboard** — Five-section interactive UI covering operations, portfolio, research, validation, and system config

---

## Project Structure

```
daily_investor/
├── cfg/
│   ├── config.yaml                   # All tunable parameters (never commit credentials)
│   └── ratios.yaml                   # Industry valuation benchmarks (auto-updated)
├── data/
│   ├── agg_data_YYYY_MM_DD.csv       # Scored universe (newest always used)
│   ├── robinhood_data_YYYY_MM_DD.csv # Raw Robinhood fundamentals cache
│   ├── news_YYYY_MM_DD.csv           # News sentiment cache
│   ├── stock_tickers_YYYY_MM_DD.csv  # Universe ticker list cache
│   ├── holdings_YYYY_MM_DD.csv       # Portfolio snapshots
│   ├── buy_history.csv               # All-time buy log (wash-sale tracking)
│   ├── sell_history.csv              # All-time sell log
│   ├── peak_prices.json              # Per-symbol all-time-high tracker (trailing stop)
│   └── snapshots/
│       └── YYYY_MM_DD.parquet        # Daily scored universe snapshots (IC store)
├── src/
│   ├── cli/
│   │   ├── main.py                   # Argument parsing + command dispatch
│   │   └── commands.py               # Per-command handlers (fetch-data, run, backtest, …)
│   ├── core/
│   │   ├── types.py                  # Shared dataclasses: SimResult, TradeRecord, SellDecision
│   │   ├── logging.py                # Structured JSON logging
│   │   ├── paths.py                  # Canonical path constants
│   │   └── utils.py                  # safe_float, run_async
│   ├── config/
│   │   ├── schema.py                 # Frozen dataclasses for all YAML sections
│   │   └── manager.py                # Singleton ConfigManager with cached_property sections
│   ├── data/
│   │   ├── cache.py                  # CSV read/write helpers
│   │   ├── universe.py               # Universe builder (scrapes Wikipedia + Robinhood sources)
│   │   ├── fundamentals.py           # Fundamentals fetch + scoring (yfinance + Robinhood)
│   │   ├── market.py                 # get_data(): full scored universe pipeline
│   │   ├── valuation.py              # Industry ratio fetching (FinViz)
│   │   └── sentiment.py              # Async Claude batch sentiment
│   ├── strategy/
│   │   ├── momentum.py               # Momentum engine (multi-factor + warmup fallback)
│   │   ├── factor_interactions.py    # Cross-factor interaction adjustments
│   │   ├── snapshots.py              # Parquet snapshot store: save, load, prune, backfill, rescore
│   │   ├── scoring/                  # Unified peer-relative factor scoring engine (peer-1)
│   │   │   ├── composite.py          # compute_metric: blends factors → value_metric
│   │   │   ├── value.py              # Sector-relative winsorized percentile value scoring
│   │   │   ├── quality.py            # Quality scoring (peer-relative + legacy checklist fallback)
│   │   │   ├── income.py             # Income/yield scoring with trap detection
│   │   │   ├── momentum.py           # Momentum factor scoring
│   │   │   ├── growth.py             # Growth factor scoring
│   │   │   ├── peer.py               # Peer-relative ranking + anchor blending
│   │   │   └── _legacy_checklist.py  # Private legacy checklist scorers (fallback)
│   │   ├── regimes/
│   │   │   ├── models.py             # RegimeState, RegimeHistoryEntry, RegimeLabel
│   │   │   ├── detector.py           # RegimeDetector: live detect + historical replay
│   │   │   └── __init__.py
│   │   └── research/                 # Compat re-export only → research/ic_engine.py
│   │       └── __init__.py
│   ├── research/
│   │   ├── ic_engine.py              # FactorResearchEngine: multi-horizon IC, decay, decile
│   │   └── distribution_regime_analysis.py  # DistributionAnalyzer: bimodality, tail IC, clusters
│   ├── portfolio/
│   │   ├── risk.py                   # RiskManager.can_buy() — position/sector/order gates
│   │   ├── sell_engine.py            # SellDecisionEngine.evaluate() — hard/soft sell logic
│   │   ├── manager.py                # PortfolioManager: sell_cycle, buy_cycle, rebalance
│   │   ├── harvest.py                # Profit harvesting + ETF routing
│   │   ├── decision_logger.py        # Structured decision audit log
│   │   ├── outcome_tracker.py        # Forward-return outcome backfill
│   │   ├── position_rationale.py     # Deterministic position rationale engine
│   │   ├── exit_analysis.py          # Exit signal analysis helpers
│   │   ├── decision_adjustment_engine.py  # HARVEST/TRIM/REVIEW downgrade logic
│   │   └── exposure/
│   │       └── analyzer.py           # ExposureAnalyzer: factor tilts, sector, HHI, drift
│   ├── execution/
│   │   ├── base.py                   # BrokerAdapter ABC
│   │   ├── paper.py                  # PaperBroker — in-memory, no API
│   │   └── robinhood.py              # RobinhoodBroker — live orders with retry backoff
│   ├── backtesting/
│   │   ├── types.py                  # PrecomputedData, SimResult, BacktestReport, TradeRecord
│   │   ├── data_loader.py            # load_and_precompute(), select_backtest_universe()
│   │   ├── simulator.py              # run_simulation(), score_stocks_at_day(), select_candidates()
│   │   ├── reports.py                # print_backtest_report(), compare_candidate_selection_modes()
│   │   ├── engine.py                 # BacktestEngine: simulate, run, walk_forward
│   │   ├── validator.py              # WalkForwardValidator: train/val split, gate checks
│   │   └── results.py                # BacktestResult, ValidationResult typed wrappers
│   ├── tuning/
│   │   ├── constants.py              # PARAM_NAMES, PARAM_BOUNDS, _CONFIG_PATH_TO_PARAM_IDX
│   │   ├── objective.py              # _objective(), run_simulation_for_objective()
│   │   ├── reports.py                # print_config_diff(), _diff_table()
│   │   ├── tuner.py                  # ParameterTuner: tune, auto_tune, apply_params
│   │   ├── stability.py              # StabilityAnalyzer: multi-window parameter scan
│   │   └── results.py                # TuneResult, AutoTuneResult, StabilityReport
│   ├── reporting/
│   │   ├── attribution.py            # AttributionReporter: factor/sleeve/exit-type attribution
│   │   ├── diagnostics.py            # DiagnosticsReporter: CSV + robustness TXT
│   │   └── plots.py                  # PlotManager: heatmaps and validation charts
│   ├── research/
│   │   └── distribution_regime_analysis.py  # DistributionAnalyzer: bimodality, tail IC, clusters
│   ├── ui/
│   │   ├── streamlit_app.py          # Dashboard entry point
│   │   ├── utils.py                  # Shared UI helpers, path constants, CSV loaders
│   │   ├── layout/sidebar.py         # Navigation sidebar
│   │   ├── sections/                 # Top-level page sections (one per sidebar entry)
│   │   │   ├── operations.py
│   │   │   ├── portfolio.py
│   │   │   ├── research.py
│   │   │   ├── validation.py
│   │   │   └── system.py
│   │   └── components/               # Reusable tab components
│   │       ├── home.py               # System dashboard / status
│   │       ├── run_control.py        # CLI command builder + subprocess runner
│   │       ├── intents.py            # Order intent dry-run preview
│   │       ├── execution.py          # Live execution panel
│   │       ├── portfolio.py          # Holdings display
│   │       ├── exposure.py           # Factor tilts and sector exposure
│   │       ├── regime.py             # Regime inspector + effective config by regime
│   │       ├── scoring.py            # Scored universe explorer
│   │       ├── value_diagnostics.py  # Value distribution and decile analysis
│   │       ├── factor_analysis.py    # Factor correlation and orthogonalization
│   │       ├── rolling_ic.py         # Single-horizon rolling IC time series
│   │       ├── factor_lab.py         # Multi-horizon IC, decay curves, decile spread
│   │       ├── distribution_intelligence.py  # Bimodality, tail IC, clusters, threshold sim
│   │       ├── data_explorer.py      # Raw CSV/parquet explorer
│   │       ├── backtests.py          # Backtest runner + results
│   │       ├── stability.py          # Stability scan runner + heatmaps
│   │       ├── reliability.py        # Data pipeline integrity diagnostics
│   │       ├── tuning.py             # Auto-tune UI
│   │       ├── config_viewer.py      # Config viewer + live editor (gated write)
│   │       └── logs.py               # Log tail + audit CSVs
│   ├── main.py                       # Live trading loop
│   └── util.py                       # Config constants, schema, CSV helpers
├── tests/                            # pytest test suite (no API credentials required)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_scoring.py
│   ├── test_risk.py
│   ├── test_sell_engine.py
│   ├── test_execution.py
│   ├── test_backtesting.py
│   ├── test_tuning.py
│   ├── test_reporting.py
│   └── test_cli.py
├── Makefile
├── pyproject.toml
└── .env                              # Credentials (never commit)
```

---

## CLI

```
daily-investor COMMAND [OPTIONS]
```

| Command | Description |
|---------|-------------|
| `fetch-data` | Fetch all data (valuations, dividends, holdings, fundamentals, news, snapshot) — **no trades placed** |
| `run` | Live trading run (sell + buy cycle) |
| `backtest DAYS` | Run backtest simulation |
| `tune DAYS` | Single-objective parameter tune — prints diff, no write |
| `auto-tune [DAYS]` | Dual-objective tune + multi-source candidate tournament (`--random-topk N`, `--leads a.npy`) gated by split, incumbent-relative, random-window, multi-horizon, and stress-gauntlet tiers (default: 90d) |
| `auto-tune-all` | Staged coordinate-ascent over interaction clusters + full windowed validation (`--profile`, `--clusters`) — research only |
| `interaction-screen` | Screen which param clusters synergize/clash when co-tuned (`--profile quick\|standard\|deep`) — research only |
| `list-presets` | Print available tuning presets and exit (presets compose with `+`) |
| `stability-scan` | Parameter stability scan across multiple windows — research only, no writes |
| `report` | Run a quick 90-day backtest and print results |
| `update-outcomes` | Backfill realized future returns for past decisions — calibration only, never touches live scoring |
| `factor-map` | 3-D PCA/UMAP factor-space scatter of the scored universe |
| `fmp <SUB>` | FMP cache operations: `status`, `validate-cache`, `backfill-prices`, `backfill-statements`, `backfill-delisted`, `build-dead-universe` |
| `config <SUB>` | Config maintenance — sub: `migrate-scoring` (rewrite legacy YAML to unified scoring) |
| `snapshots <SUB>` | Snapshot maintenance — sub: `rescore` (re-score on-disk snapshots to current model) |
| `tune-etf-allocation` | Gated ETF/core sleeve allocation tournament (`--days`, `--mode regime\|defensive`, `--universe configured_only`, `--random-topk N`, `--apply`) — writes only the `etf_allocation` config section if all gates pass |
| `report-etf-allocation` | Print ETF/core sleeve diagnostics for the current config (`--days`) |
| `odte-social-report` (alias `options-social`) | **ANALYSIS / PAPER ONLY — places NO orders.** 0DTE social-sentiment watchlist from Reddit (official OAuth when `REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` set → public JSON → Atom-feed fallback; no scraping) + X official API (only if `X_BEARER_TOKEN` set; no scraping). Counts posts as fresh by **market session** (`freshness_mode: market_window`, America/New_York): weekend → since the last Friday 16:00 ET close; weekday pre-open → since the previous close; weekday at/after open → since today 09:30 ET (so weekend-accumulated sentiment is retained for Monday prep), floored by `max_lookback_hours` (default 96). Applies **transparent spam/quality filtering** (no ML): drops promo/scam (Telegram/VIP/100X/“free signals”/WhatsApp), off-topic crypto, class-action/legal blasts, shotgun-cashtag spam, and near-duplicates; ODTE evidence additionally requires an allowed ticker **plus** an options/day-trading context token (0DTE, call/put, strike, scalp, FOMC…) so generic SPY/QQQ chatter doesn’t inflate mention counts. News enrichment applies the same spam/dedupe pass but **not** the options-context requirement. Attaches a **paper-only same-day option idea** (yfinance; bullish→calls / bearish→puts; budget-capped, liquidity-sorted; fails closed when market closed / no chain / `--no-fetch`). The CLI runs **on demand regardless of config** (gated by neither `enabled` nor network). Separately, `fetch-data`/`force-refresh` **always** enriches the news-sentiment substrate with social items for the active-sleeve LLM — **fail-closed** and independent of `options_social.enabled`; opt out with `options_social.disable_social_news_enrichment: true`. Social items are merged as **ordinary news articles** (title, source/`api_source`, link, date, raw text, engagement counts) so the LLM judges news and social **uniformly** — no precomputed bullish/bearish/net social score is injected into the active-sleeve prompt (the report keeps its own transparent heuristics, separately). Optional bounded **comments enrichment** (`reddit_comments_enrich: true`, default off) folds top comments of the top posts into the post text (OAuth → public JSON; cached). **Employer/compliance-restricted underlyings (NVDA by default; add more via `options_social.restricted_underlyings`) are hard-blocked in code** — never a candidate, contracts stripped, surfaced only as read-only `RESTRICTED_EMPLOYER` context (`restricted: true`, `restricted_reason: "employer"`). |
| `odte-watchdog` | **Script-only watchdog — NO LLM, NO Robinhood, places NO orders.** Runs the LOCAL `odte-social-report` (zero model calls — the OpenAI/model-429 avoidance), diffs the actionable candidate vs the prior run, and writes `data/odte/watchdog_state.json` + `data/odte/triggers.json`. Reads `~/0dte/controller_policy.json` (a **secret**, kept in the home dir) for presence/validity only (never echoes its contents). Designed for a `no_agent` cron: **empty stdout** when nothing is actionable, a **compact one-line JSON alert** when a conservative trigger fires (a new/changed non-restricted candidate, or a missing/invalid policy). `--json` always prints the compact state; `--no-fetch` (or `make odte-watchdog OFFLINE=1`) runs offline cache-only; `--state-dir DIR` overrides the `data/odte/` data dir and `--policy PATH` the `~/0dte/` policy. |
| `odte-position` | **Broker-AWARE, DECISION-ONLY live-position watchdog — places NO orders, makes NO broker/LLM calls.** The discipline layer for an already-open 0DTE option: reads the active trade plan (`data/odte/active_trade.json`) + a **caller-supplied** live snapshot (Hermes feeds real broker/market values from its MCP tools — this command never fabricates broker data) and emits structured triggers `TAKE_PROFIT` (scale at +35–50%, strong full exit at +60%) / `THESIS_DEAD` (underlying/SPY/QQQ/VIX/VIXY stop levels) / `BID_FLOOR` (near-worthless) / `TIME_RISK` (`tighten_after` / `flat_before` ET) / `MONITORING_DEGRADED` (can't value the position) / `HOLD` / `NO_POSITION`. Employer-restricted underlyings (NVDA) return `RESTRICTED` with no management triggers. Writes `data/odte/{position_state,position_decision}.json`; **empty stdout** on `HOLD`/`NO_POSITION`, compact JSON on an actionable decision. Supply live values with `--snapshot PATH` or `--snapshot-json '{...}'`; `--plan` / `--state-dir` override defaults; `--json` always prints. The decision core (`evaluate_position`) is pure and unit-tested without Robinhood/network. |
| `odte-journal` | **Local/offline decision journal — NO broker, NO LLM, NO secrets.** Appends one event to `data/odte/decision_journal.jsonl` (JSONL). Events: `pre_trade_thesis` / `entry_decision` / `order_filled` / `management_check` / `exit_decision` / `order_closed` / `postmortem` / `experiment`. Each carries free-form `thesis` (direction, catalyst, social pulse, market read, key levels, invalidation, profit plan, time rules), `decision` (action, confidence, reasons, alternatives, changed-since-prior), `outcome` (entry/exit, MFE/MAE, realized P/L, rule violations, lessons), or `experiment` (hypothesis, metric, promote/kill) fields. Supply with `--event-json '{...}'` or `--event PATH`. NVDA/employer-restricted underlyings are tagged `restricted` on store and excluded from forward experiments/metrics. `--json` prints the stored event. |
| `odte-journal-report` | Summarize the decision journal into **deterministic metrics + visual artifacts** (no heavy deps): trades by mode, hit rate, average realized P/L, MFE capture (realized/MFE), rule violations, decision timing, experiments backlog, lessons. Also rolls up any **`sentiment`** snapshots (verdict/direction/confidence/score distributions + latest read) and **`gamma`** snapshots (pin-risk level distribution + latest max-gamma strike/walls) attached to events into `sentiment_status` / `gamma_status` — restricted-underlying reads are tagged and kept out of the latest read/bias, and the gamma rollup carries `includes_dealer_positioning: false` + the honest `pin_risk_only_not_dealer_gex` regime so it is **never** mistaken for dealer net GEX. Renders a **Markdown** report with text bars/sparklines (Telegram/terminal friendly) and a **CSV** by-mode summary for plotting. `--json` prints the metrics payload; default prints Markdown; `--write` (or `--out-dir DIR`) writes `data/odte/reports/odte_journal_report.md` + `odte_journal_summary.csv`. Pure/offline; `data.odte_journal.event_from_position_decision()` converts an `odte-position` payload into a `management_check` event. |
| `odte-gamma-map` | **0DTE option-chain gamma / pin map — PURE/OFFLINE, NO broker, NO LLM, NO network.** Reads option-quote rows that Hermes/Robinhood MCP exported to a JSON file (`--input PATH`) or string (`--input-json '{...}'`) and computes **absolute** gamma + open-interest concentration: per-strike call/put OI & volume, `gamma_notional_1pct` (`γ·OI·100·spot²·0.01`), **call wall / put wall / max-gamma strike** (gamma-weighted, OI fallback), ATM-straddle **expected move** band, **pin risk** (high/medium/low/stale), and **quote freshness**. **Honest by construction:** every output is labeled `gamma_regime: pin_risk_only_not_dealer_gex` and carries a disclaimer — RH exposes per-contract greeks but **not** dealer positioning, so this is a concentration heuristic, **not dealer net GEX / gamma flip / sign**. `--spot`/`--underlying`/`--expiration` refine the read; `--json` prints the map; `--write` (or `--out-dir DIR`) writes `data/odte/reports/odte_gamma_map_<sym>.{md,json}`. **FMP note:** FMP options endpoints (`/stable/option-chain`, `/stable/options-chain`, etc.) are 404/legacy-403 on the current key, so FMP is **not** a gamma source; Robinhood-exported quotes are the only input. |
| `odte-rh-rows` | **PURE/OFFLINE, NO broker, NO LLM, NO network.** Pairs the two **separate** arrays Robinhood returns — option quotes/market-data (`get_option_market_data[_by_id]`, with per-contract greeks/OI/volume/mark) and option instruments (`get_option_instrument_data`, with strike/type/expiration/chain_symbol) — into flat rows that `odte-gamma-map` consumes directly. Each quote is joined to its instrument **by id/url**; instrument contract fields fill the gaps the quote lacks while the quote's live greeks win on overlap; rows that resolve no usable strike/side (no instrument match) are **dropped, not guessed**. `--quotes PATH` / `--quotes-json '[...]'` supply the quote array; `--instruments PATH` / `--instruments-json '[...]'` supply the companion instruments (optional if the quotes already carry strike/type). Prints a JSON row list (feed straight into `odte-gamma-map --input`), or `--out PATH` writes it. **Honest by construction:** emits **absolute** per-contract gamma/OI rows only — it neither needs nor invents dealer positioning, so the downstream map is pin-risk concentration, **never dealer net GEX**. (`data.odte_gamma_map.rh_rows_from_quotes()` is the pure helper.) |
| `odte-vehicle-score` | **PURE/OFFLINE non-sentiment vehicle score — NO broker, NO LLM, NO network.** Given a candidate contract (`--contract` / `--contract-json`) plus optional market snapshot and gamma map, returns `GOOD_BET` / `WATCH` / `BAD_BET` from tape/VWAP alignment, VIXY confirmation, gamma/pin/expected-move fit, liquidity, and buying-power fit. This is the simple contract/vehicle "good or bad bet for the day" layer outside pure sentiment. |
| `odte-day-score` | **PURE/OFFLINE non-sentiment day-regime score — NO broker, NO LLM, NO network.** Companion to `odte-vehicle-score` (which scores one contract); this scores the *whole day* before you pick a vehicle. Given a market/regime snapshot (`--market` / `--market-json`: VIX, VIX/VIXY change, opening `gap_pct`, per-index `{spy,qqq,iwm}_above_vwap` + `{sym}_orb_state`, `expected_move_pct`, `minutes_to_close`) plus an optional gamma map (`--gamma` / `--gamma-json`, which lets the expected move be derived from the ATM-straddle band), returns `GOOD_DAY` / `CHOP` / `AVOID` from trend alignment, volatility regime, gap, expected-move room, and a late-day theta gate. Hard `AVOID` on a very elevated/spiking VIX or `minutes_to_close ≤ 30`. `--json` prints the payload; default prints Markdown; `--write` (or `--out-dir DIR`) writes `data/odte/reports/odte_day_score.json`. No orders, no network, no sentiment. |
| `odte-fmp-context SYMBOL` | **FMP single-name context for meme/squeeze SANITY — read-only, NO orders, NO options/gamma.** Fetches cheap FMP *stable* fundamentals (`profile`, `quote`, `shares-float`, `key-metrics-ttm`, a few `news` headlines) and classifies a `squeeze_profile` — `tiny_float_squeeze_candidate` / `small_float_momentum` / `mid_float_meme_momentum` / `large_float_meme_momentum_not_tiny_float` / `no_float_data` — with a plain-English `trade_implication`. Output includes price, market cap, beta, 52w range, volume / average / **relative volume**, float / outstanding shares / free-float %, net-debt/EBITDA, news count + titles, and warnings. **It is a sanity check, not an entry signal — no orders are placed.** FMP options endpoints are unavailable, so every output carries `fmp_options_available: false` and **Robinhood remains the option-chain / gamma source**. Fail-closed when `FMP_KEY` is missing (the key is never printed). `--json` prints the context; `--write` (or `--out-dir DIR`) writes `data/odte/reports/odte_fmp_context_<sym>.{md,json}`; `--no-fetch` runs offline. Deliberately **not** called by `odte-watchdog` (kept cheap / no-network) — the controller enriches a candidate only on a trigger. NVDA stays employer-restricted (tagged context-only). |
| `odte-loop-status` | **PURE/OFFLINE loop state machine — NO broker, NO LLM, places NO orders.** One read-only surface that tells the live controller (Hermes/MCP) **where in the loop it is and which command runs next**, so `scan → thesis → entry → watch → exit → review` reads as one obvious cycle instead of a pile of independent tools. Summarizes the canonical `data/odte/` artifacts (`active_trade.json`, `position_decision.json`, `triggers.json`, and the latest `entry_decision` / `postmortem` in `decision_journal.jsonl`) into the current **state** — `SCAN` (keep scanning → `odte-watchdog`) / `CANDIDATE` (non-restricted candidate on the board, scan_only → `odte-entry-gate`) / `GATED` (entry-gate record, not execution-allowed → `odte-entry-gate --promote-to-execution`) / `PROMOTED` (gate execution-allowed — manager may enter → `odte-position`) / `ENTERED` (plan open, no live decision yet) / `MANAGING` (live position + current decision → `odte-position`) / `EXITED` (trade closed, no postmortem → `odte-journal` then `odte-ingest-artifacts`) / `REVIEWED` (closed + reviewed → `odte-journal-report`) / `DEGRADED` (live position can't be valued, or a live artifact is malformed/stale) — plus the **next command**, an `executable` flag, and a short `reasons` list. **Re-derives NO gate or decision** — it only reads what the other tools already wrote. A **live position always outranks** the scan/candidate/gate lane (managing/exiting an open trade beats chasing a new one); missing artifacts read as `SCAN` and a malformed/stale live artifact **degrades** rather than crashing. `--json` prints the compact machine payload (clean stdout contract); default prints Markdown; `--state-dir DIR` overrides `data/odte/`. (`data.odte_loop_status.derive_loop_state()` is the pure, unit-tested core.) |

### 0DTE storage layout

All 0DTE **data** lives under the app's data tree (`data/odte/`, gitignored) so the Streamlit
dashboard reads it alongside the rest of the app:

- `data/odte/decision_journal.jsonl` — append-only decision journal
- `data/odte/active_trade.json` — the active trade plan
- `data/odte/{watchdog_state,triggers,position_state,position_decision}.json` — watchdog/position state
- `data/odte/reports/` — gamma-map / fmp-context / journal Markdown+CSV artifacts
- `data/odte/scrape/` — **timestamped** analyzed-text snapshots (`{reddit,x}_text_YYYY_MM_DD_HH_MM.txt`,
  plus a stable `{reddit,x}_text.txt` latest pointer) so scraped social text accumulates over time

Only **secrets/config** stay in `~/0dte/` (so Hermes/MCP's hands-off auth is untouched):
`config.json` (Reddit OAuth + `daily_thread_id`), `reddit_token.json`, `daily_thread_id.txt`, and
`controller_policy.json`. The full 0DTE workflow is also surfaced in the dashboard's **0DTE**
section (`make ui`).

> **Migrating from a pre-existing `~/0dte/`?** Move your data files into `data/odte/` and leave the
> secrets behind: `mkdir -p data/odte/reports data/odte/scrape && mv ~/0dte/decision_journal.jsonl
> ~/0dte/active_trade.json ~/0dte/*_state.json ~/0dte/*_decision.json ~/0dte/triggers.json
> ~/0dte/reports/* data/odte/ 2>/dev/null` (keep `config.json`, `reddit_token.json`,
> `daily_thread_id.txt`, `controller_policy.json` in `~/0dte/`).

Research scripts:

```bash
make regime-sizing REGIME=neutral
# runs scripts/regime_sizing_random_window.py; read-only random-window exposure grid
```

**Key options:**

```
run:
  --op-mode safe|automated|no-sentiment
  --skip-data              Reuse existing CSVs (skip fetch)

auto-tune:
  --apply                  Write config.yaml if validation passes
  --force-apply            Write config.yaml unconditionally (debugging only)
  --llm-review             Claude second-opinion before applying
  --mode MODE              Backtest universe mode

all:
  --mode MODE              liquid_universe_full | walk_forward_price_only_test | current_universe_stress_test
  --objective sharpe|calmar|info_ratio   (info_ratio = excess-vs-SPY / tracking-error; active scope)
  --output-dir PATH
```

**Survivorship-free backtesting.** Set `backtest.survivorship_free: true` in `cfg/config.yaml`
(or tick the "🧬 Survivorship-free data" box in the UI Validation tab) to run every backtest and
tune against split-adjusted prices for the current universe **plus the delisted names** from the
FMP cache (`data/fmp_cache_adj/`), removing the ~35% survivorship inflation. Requires the cache to
be populated; falls back to yfinance with a warning if it is absent. Use the first-class FMP cache
commands to maintain it:

```bash
make fmp-status
make fmp-backfill-prices FMP_SYMBOLS=current FMP_START=2015-01-01
make fmp-backfill-statements FMP_SYMBOLS=current FMP_MAX=500
make fmp-backfill-delisted
make fmp-build-dead-universe
make fmp-validate-cache
```

All backfill commands are cache-first and quota-aware. Reads inside backtests are cache-only; only
these explicit `fmp backfill-*` commands spend FMP calls.

---

## Makefile Targets

```bash
# Data
make fetch-data            # Fetch fresh fundamentals + news, save CSVs + snapshot (no trades)
make fetch-data SKIP_NEWS=1 # Same, but reuse cached news (skip the slow news scrape)
make update-outcomes       # Backfill future return labels for past decisions (calibration only)
make fmp-status            # FMP cache/key/quota/coverage status
make fmp-validate-cache    # Read-only FMP cache sanity check
make fmp-backfill-prices FMP_SYMBOLS=current FMP_MAX=100
make fmp-backfill-statements FMP_SYMBOLS=current FMP_MAX=100
make fmp-backfill-delisted
make fmp-build-dead-universe

# Live trading
make run                                    # Safe mode — manual confirmation at each step
make run OP_MODE=automated                  # Automated mode — no prompts
make run SKIP_DATA=1                        # Safe mode, reuse cached CSVs (faster)
make run OP_MODE=no-sentiment SKIP_DATA=1   # No sentiment, no trades — scoring + logic preview only
make run SKIP_NEWS=1                        # Refresh data but reuse cached news (skip slow news scrape)

# Backtesting
make backtest              # 365-day backtest (default mode)
make backtest DAYS=180
make backtest BT_MODE=walk_forward_price_only_test   # Walk-forward mode (low lookahead)
make backtest COMPARE=1    # A/B/C candidate selection mode comparison

# Parameter tuning
make tune                  # Single-objective tune, no write  (TUNE_DAYS=120  OBJ=sharpe)
make auto-tune             # Dual-objective tune, walk-forward validation, no write
make auto-tune APPLY=1     # auto-tune + write config.yaml if validation passes
make auto-tune LLM=1       # auto-tune + Claude second-opinion + apply
make auto-tune PRESET=active_core_weights  # Tune a single active-sleeve preset (names: make list-presets)

# Research & diagnostics
make stability             # Parameter stability scan across multiple windows
make report                # Quick 90-day backtest → reports/
make regime                # Print current market regime (live SPY + VIX)
make regime-sizing REGIME=neutral  # Random-window sizing/exposure grid; writes reports/regime_sizing_neutral.csv
make ic                    # Print IC summary across default horizons (needs ≥ 2 snapshots)
make snapshot-info         # Show snapshot store status (count, date range)
make snapshot-backfill     # Backfill parquet snapshots from existing agg_data CSVs

# Dashboard
make ui                    # Launch Streamlit dashboard

# Development
make install               # Install / reinstall package in editable mode
make install-system        # Install editable, bypassing Homebrew protection (macOS Homebrew Python)
make test                  # Run full pytest suite
make test-watch            # Re-run tests on file changes (requires pytest-watch)
make lint                  # Run ruff linter over src/
make format                # Auto-format src/ with ruff
```

---

## Architecture

```
ui/          renders
ui/services/ orchestrates
cli/         dispatches
backtesting/ simulates
tuning/      searches parameters
portfolio/   decides buys/sells
strategy/    scores stocks
research/    evaluates offline (read-only)
reporting/   summarizes results
config/      loads and validates config
core/        shared types, paths, utils
execution/   broker adapters
```

Import rules: no `streamlit` in core packages; no `ui/` imports in core packages.
See `AGENTS.md` for the full architecture contract.

---

## Scoring Model

### Factor Scores

| Score | What it measures |
|-------|-----------------|
| `value_score` | Sector-relative P/E and P/B cheapness (winsorized percentile ranking) |
| `income_score` | Dividend yield quality (capped; 0 if yield trap or no yield) |
| `quality_score` | Liquidity, earnings existence, dividend health |
| `momentum_score` | Multi-factor: relative strength, risk-adjusted return, DMA trend, short-term momentum |

### Composite Score

```
value_metric = sw_value    × value_score
             + sw_quality  × quality_score
             + sw_income   × income_score
             + sw_momentum × momentum_score
```

Current weights (`score_weights` in `config.yaml`):

```yaml
score_weights:
  value:    0.05
  quality:  0.45
  income:   0.05
  momentum: 0.45
```

### Momentum Score

All sub-scores are **cross-sectionally percentile-ranked** across the live universe each day. Causal — no lookahead.

| Sub-factor | Default weight | What it captures |
|---|---|---|
| `rs_3m` | 0.25 | Return_3m − SPY_3m (relative strength, 3-month) |
| `rs_6m` | 0.25 | Return_6m − SPY_6m (relative strength, 6-month) |
| `risk_adj_3m` | 0.20 | return_3m / realized_vol_3m (Sharpe-like, 63-day) |
| `trend_structure` | 0.15 | Price vs 50 DMA and 200 DMA |
| `return_1m` | 0.10 | Raw 21-day return, percentile-ranked |
| `return_5d` | 0.05 | 5-day short-term check |

Penalties after weighting: falling-knife (3m return < −15%), overextension (52w position > 97%), high volatility (annualized vol > 50%). Final score clamped to [−1.0, 1.5].

### Value Score

1. Within each sector (min 5 stocks), winsorize P/E and P/B at 5th/95th percentile
2. Percentile-rank each stock against its sector peers (low PE → high rank)
3. Blend: `0.60 × pe_rank + 0.40 × pb_rank`, scaled to [−1.0, 1.5]
4. Distress penalties: PE ≤ 5 → −0.30; negative EPS → −0.25
5. Falls back to global ranking for sectors with fewer than 5 stocks

---

## Market Regime

| Regime | Trigger | ETF alloc | Max stock buys | Stop adjustment |
|--------|---------|-----------|---------------|-----------------|
| **Bullish** | SPY above 200DMA AND VIX < 20 | 72% (base) | 4 (base) | none |
| **Neutral** | SPY below 200DMA OR VIX 20–30 | 77% | 4 | none |
| **Defensive** | VIX ≥ 30 | 85% | 3 | +0.05 tighter; ETF MA filter on |

All thresholds configurable under `regime:` in `config.yaml`. `RegimeDetector` also exposes confidence scores calibrated by distance from VIX thresholds.

```python
from strategy.regimes import RegimeDetector

state = RegimeDetector().detect()             # live fetch
state = RegimeDetector().detect_from_data(    # pure computation, testable
    spy_price=580.0, spy_ma200=540.0, vix=18.0
)
history = RegimeDetector().classify_history(days=365)   # historical replay
```

---

## Factor Research Platform

### Snapshot Store

Every scoring run saves a dated Parquet file to `data/snapshots/YYYY_MM_DD.parquet`. These are the foundation for all rolling IC and forward-return validation.

```bash
make snapshot-info          # count + date range
make snapshot-backfill      # migrate existing agg_data CSVs to parquets
make ic                     # quick IC summary (needs ≥ 2 snapshots)
```

### FactorResearchEngine

```python
from strategy.research import FactorResearchEngine

engine = FactorResearchEngine()
ic_df  = engine.compute_multi_horizon_ic(horizons=[5, 20, 60, 120])
summ   = engine.compute_ic_summary(ic_df)
decay  = engine.compute_factor_decay()
spread = engine.compute_decile_spread("momentum_score", horizon_days=20)
```

| Method | Returns |
|---|---|
| `compute_multi_horizon_ic(factors, horizons, ic_type)` | `[date, factor, horizon_days, ic, n_stocks, p_value]` |
| `compute_ic_summary(ic_df)` | `[factor, horizon_days, mean_ic, icir, hit_rate, t_stat]` |
| `compute_factor_decay(factors)` | IC vs horizon per factor — the decay curve |
| `compute_decile_spread(factor, horizon_days)` | Mean forward return by score decile |
| `compute_rolling_icir(factor, horizon_days, window)` | Trailing ICIR over time |
| `compute_regime_conditioned_ic(factors, horizon_days)` | IC per factor × market regime |

IC > 0.05 = moderate signal. ICIR > 0.5 = actionable.

### Distribution Regime Analysis

Tests whether the bimodal peer-relative value score distribution contains predictive information.

```python
from research.distribution_regime_analysis import DistributionAnalyzer

ana = DistributionAnalyzer(agg_df)
bm  = ana.test_bimodality("value_metric")        # BC coefficient + GMM BIC k=1 vs k=2
buckets = ana.compute_tail_buckets(               # returns by top/mid/bottom percentile
    "value_metric", "return_1m"
)
local_ic = ana.compute_local_ic("value_metric")  # IC in sliding windows — reveals nonlinearity
clusters = ana.compute_clusters(n_clusters=2)    # GMM regime clustering
cond_ic  = ana.compute_conditional_ic(           # IC of value within momentum quartiles
    "value_score", "momentum_score"
)
sim = ana.simulate_threshold_modes(              # rank-based vs threshold-gated selection
    "value_metric", "return_1m"
)
```

---

## Portfolio Risk Controls

| Rule | Config key | Default | Behaviour |
|------|-----------|---------|-----------|
| Liquidity gate | `min_liquidity_volume` | 500,000 | Skip if avg volume below threshold |
| Order size cap | `max_order_pct_of_cash` | 10% | Cap single order to 10% of available cash |
| Position cap | `max_single_position_pct` | 5% | Reduce buy so total position ≤ 5% of portfolio |
| Sector cap | `max_sector_pct` | 25% | Reduce buy so sector exposure ≤ 25% |
| Minimum order | `min_order_amount` | $5.00 | Skip if reduced amount falls below this |
| ETF floor | `min_index_pct` | 65% | ETF allocation floor; optimizer cannot breach |

When a cap is hit the allocation is **reduced** to the maximum allowed. The buy is skipped only if the reduced amount falls below `min_order_amount`.

---

## Sell Decision Engine

### Hard Sells — execute immediately, sentiment cannot override

| Trigger | Condition |
|---------|-----------|
| Stop loss | `percent_change ≤ stop_loss_pct` (default −20%) |
| Trailing stop | `price / peak_price − 1 ≤ trailing_stop_pct` (default −8%) |
| Yield trap | `yield_trap_flag=True` and `value_metric < sell_weak_value_below` |
| Quality floor | `quality_score < sell_low_quality_below` (default −0.25) |

### Soft Sells — Claude can override with HOLD

| Trigger | Condition |
|---------|-----------|
| Take profit | `percent_change ≥ take_profit_pct` (default +60%) and `value_metric` below floor |
| Weak value | `value_metric < sell_weak_value_below` and held ≥ `min_days_held_before_value_exit` days |

Soft sells are sent to Claude. `HOLD` with confidence ≥ `sell_sentiment_override_confidence` (default 85%) keeps the position. Profit harvesting routes take-profit proceeds to core ETFs rather than idle cash.

---

## Backtest Engine

### Modes

| Mode | Lookahead | Universe |
|------|-----------|----------|
| `liquid_universe_full` | MEDIUM | Full liquid universe, deterministic (default; `max_symbols: 0`) |
| `walk_forward_price_only_test` | LOW | Full liquid universe; fundamental arrays zeroed, momentum only |
| `current_universe_stress_test` | HIGH | Current `value_metric` top-N — **not predictive**, stress test only |

### Causal Rolling Features

| Feature | Lookback |
|---------|---------|
| `return_1m` | 21 bars |
| `return_5d` | 5 bars |
| `return_3m / rs_3m` | 63 bars |
| `return_6m / rs_6m` | 126 bars |
| `realized_vol_3m` | 63 bars |
| `above_50dma` | 50 bars |
| `above_200dma / position_52w` | 200 / 252 bars |

Cross-sectional percentile ranking across stocks at one point in time is **not** a lookahead bias.

---

## Parameter Tuner

`scipy.optimize.differential_evolution` maximizes Sharpe or Calmar ratio over a back-simulation window.

### Tunable Parameters

```
score_weights (quality, momentum)
index_pct                   — ETF allocation fraction
trailing_stop_pct           — trailing stop distance from peak
scoring.momentum_inputs.weights sub-weights:
  rs_3m, rs_6m, risk_adj_3m, trend_structure, return_1m
```

Safety parameters (hard stop-loss, position caps, order caps) are never touched by the optimizer. `score_weights.value` and `score_weights.income` are frozen by default.

### Validation Gates (auto-tune)

Before writing any changes to `config.yaml`, the selected tournament candidate must clear every tier in order:

| Tier | Gate | Config keys (defaults) |
|------|------|------------------------|
| 1. Absolute floors | excess vs benchmark / max drawdown / Sharpe on the held-out split | `min_validation_excess_return` (0.0%), `max_validation_drawdown` (−20%), `min_validation_sharpe` (0.25) |
| 2. Incumbent-relative | must beat the current config's validation excess; turnover capped | `min_excess_vs_incumbent` (0.0), `max_turnover_multiple` (2.0) |
| 3. Random-window | paired win rate + median excess + robust score vs incumbent on shared random sub-windows | `random_window_gate.*` (12×120d over 730d, `min_win_rate` 0.5) |
| 4. Multi-horizon confirm | regime-symmetric trailing windows; catastrophe-scale tolerances only — no single window holds a fine-grained veto | `multi_horizon_confirm.*` (`regress_tolerance` 0.04 sub-400d; `long_catastrophe_excess` 0.10 / `long_catastrophe_drawdown` 0.05) |
| 5. Stress gauntlet | must SURVIVE named historical stress regimes vs the incumbent (falsification, not win-requirements); skipped episodes are reported, never silent | `stress_gauntlet.*` (episodes GFC '08, 2011, 2015, Q4 '18, COVID '20, 2022; pre-2021 episodes are survivor-biased — dead-name coverage starts 2021) |

---

## Streamlit Dashboard

Five top-level sections, each with tabs. Launch: `make ui`

### ⚡ Operations

| Tab | Description |
|-----|-------------|
| 🏠 Dashboard | System status, config summary, data freshness, log tail |
| 🚀 Run Control | Build any CLI command with full options; streams subprocess output live |
| 🎯 Order Intents | Dry-run preview of proposed buys/sells/harvests without placing orders |
| ⚡ Execute | Direct execution panel for live order placement |

### 💼 Portfolio

| Tab | Description |
|-----|-------------|
| 📊 Holdings | Current positions from Robinhood with cost basis and P&L |
| ⚖️ Exposure | Factor tilts, sector allocation, HHI concentration, rolling exposure drift |
| 🌡️ Regime | Current regime thresholds; effective config (ETF %, max buys, stops) under each regime |

### 🔬 Research

| Tab | Description |
|-----|-------------|
| 📊 Overview | Synthesized IC conclusions; factor signal strength by horizon |
| 🔍 Factors | Scoring universe explorer + Value diagnostics (PE/PB ranks, sector comparison) |
| 📡 IC Analysis | Multi-horizon IC, factor decay curves, cumulative IC, ICIR; rolling IC time series |
| 📊 Rank & Deciles | Decile monotonicity — does higher score predict better returns? |
| 🔗 Correlations | Pairwise factor IC, VIF, OLS residualization, variance decomposition |
| 🌡️ Regime | IC conditioned on market regime (bull/bear/high-vol/sideways) |
| 🧬 Distribution | Bimodality test, tail analysis, local IC, GMM clusters, conditional alpha, threshold simulation |
| 🔎 Single Stock | **Decision-support only — places NO orders.** Single-name deep dive (Universe tab): latest holdings exposure, cached factor scores, yfinance price/trend/fundamentals/news + options surface, Reddit/X social evidence (spam-filtered, with provenance), leveraged-ETF diagnostics (realized beta/correlation + cumulative vs daily-reset 2× with risk notes), and a hypothetical position-structure helper. Fails closed if yfinance/network is unavailable. |
| 🧪 Experimental | Raw CSV/parquet explorer; prototype analyses |

### ✅ Validation

| Tab | Description |
|-----|-------------|
| 📈 Backtests | Run `BacktestEngine` interactively; full results, benchmark comparison, validation split |
| 🔭 Stability & Robustness | Multi-window parameter stability scan; heatmaps; Sharpe vs Calmar spread |
| 🩺 Reliability | Data pipeline integrity: NaN rates, zero-score coverage, liquidity failures |
| ⚙️ Tuning | Dual-objective auto-tune; view diff and validation status; apply with gate |

### ⚙️ System

| Tab | Description |
|-----|-------------|
| 🛠️ Config | Interactive `config.yaml` viewer with section-by-section display; live edit mode (gated) |
| 📋 Logs & Audit | Application log tail; order history; audit CSVs |

### Safety

- The UI starts in **read-only mode** by default
- The live-execution toggle is locked unless `ui.allow_live_execution: true` in config
- All paths go through the same `RiskManager` and audit logging as the CLI

```yaml
# cfg/config.yaml — enable controlled live execution from the UI
ui:
  allow_live_execution: true
  allow_config_writes: true      # enables the Config tab edit mode
  allow_force_apply: false
```

---

## Setup

**Requirements:** Python 3.10+, Robinhood account, Anthropic API key

```bash
git clone https://github.com/lukaselsrode/daily_investor.git
cd daily_investor

python -m venv .venv
source .venv/bin/activate

pip install -e ".[ui,dev]"     # installs CLI + Streamlit + dev tools
```

`.env` file (at project root):
```
RB_ACCT=your_robinhood_email
RB_CREDS=your_robinhood_password
RB_MFA_SECRET=your_totp_secret         # optional: skips interactive MFA prompt
ANTHROPIC_API_KEY=your_anthropic_key   # required for sentiment and LLM tune review

# Optional — social-sentiment enrichment (Reddit/X). All optional; absence just
# downgrades the Reddit fetch to public JSON, then the Atom/RSS feed.
REDDIT_CLIENT_ID=your_reddit_app_id        # official Reddit app-only OAuth (preferred)
REDDIT_CLIENT_SECRET=your_reddit_app_secret
REDDIT_USER_AGENT=your-app/1.0 by u/you    # optional; Reddit asks for a descriptive UA
X_BEARER_TOKEN=your_x_api_bearer_token     # optional; enables X via the official API only
```

Reddit OAuth uses the **app-only `client_credentials`** grant (no user login) against
`oauth.reddit.com` — it is ToS-clean and avoids the 403s/rate-limits anonymous server
requests hit. No browser automation or HTML scraping is used anywhere. (This is the same
read-only application-only OAuth that PRAW would use under the hood, done directly with
`requests` — so **PRAW is not a dependency**.)

**WSB daily-discussion-thread comments** (the real intraday chatter) need the same
`REDDIT_CLIENT_ID`/`REDDIT_CLIENT_SECRET` — without them the public-JSON comments endpoint
returns 403 and the report shows `daily thread comments: unavailable: auth needed` instead of a
silently low count. Create a Reddit app at <https://www.reddit.com/prefs/apps> (type **script**
or **web app**), copy its id/secret into `.env`. The daily thread is usually stickied and may be
absent from the `hot` listing; in that case set an explicit override in `cfg/config.yaml` under
`options_social`: `daily_thread_id: <base36 id>` or `daily_thread_url: <full /comments/ URL>`.

If you don't have app OAuth creds, you may pass an **ephemeral read-only bearer token** as a
one-off CLI argument: `daily-investor odte-social-report --reddit-bearer-token <TOKEN>`. The token
is used only as an `Authorization: Bearer` header against `oauth.reddit.com` for that run — it is
**never stored, never logged, never echoed**, and is **not** read from `.env` or config. You can
obtain such a token manually from your browser/devtools; it typically expires in ~24h. The tool
**never reads cookies, never sends a `Cookie` header, and never mints tokens** (it does not call
`/svc/shreddit/token`). App OAuth (`REDDIT_CLIENT_ID/SECRET`) remains the preferred, durable path.

```bash
daily-investor --help
make fetch-data                        # verify credentials + save first snapshot
make ui                                # launch dashboard
pytest                                 # runs all tests — no credentials needed
```

---

## Typical Workflow

```bash
# Day-to-day: fetch data, review in dashboard, then run
make fetch-data            # pulls fresh fundamentals + news, saves snapshot
make ui                    # review scored universe, regime, and portfolio

make run                   # execute live strategy (safe mode)

# Research (after accumulating ≥ 2 snapshots)
make ic                    # quick IC check
make ui                    # Research → IC Analysis, Distribution Intelligence tabs

# Periodic maintenance
make auto-tune             # review tuning diff
make auto-tune APPLY=1     # apply if satisfied
make stability             # check parameter sensitivity
```

---

## Configuration Reference

Key sections in `cfg/config.yaml`:

```yaml
# Factor weights (quality + momentum drive most of the signal)
score_weights:
  value:    0.05
  quality:  0.45
  income:   0.05
  momentum: 0.45

# ETF allocation by regime
index_pct: 0.72     # bullish baseline

regime:
  spy_ma_period: 200
  vix_neutral_threshold: 20.0
  vix_defensive_threshold: 30.0
  neutral:
    index_pct_override: 0.77    # slight ETF tilt vs bullish
    max_buys_override: 4        # unchanged from base
  defensive:
    index_pct_override: 0.85
    max_buys_override: 3
    stop_loss_tighten: 0.05

# Peer-relative factor scoring (unified `scoring` block — replaces the
# legacy value_v2/momentum_v2/scoring_v3 top-level keys)
scoring:
  factors:
    value:
      enabled: true
      peer_relative: true
      pe_weight: 0.7
      pb_weight: 0.3
      anchor_blend: 0.5

# Backtest
backtest:
  default_mode: liquid_universe_full
  starting_capital: 5000.0
  weekly_contribution: 400.0
  slippage_bps: 10.0
  train_pct: 0.70

# Snapshot store
snapshots:
  enabled: true
  retention_days: 365

# Optimizer control — these parameters are never tuned
tuning:
  frozen_parameters:
    - score_weights.value
    - score_weights.income
    - metric_threshold
```

### Operating Modes

| Mode | `--op-mode` | `auto_approve` | `use_sentiment` |
|------|-------------|---------------|-----------------|
| Safe | `safe` | false | true |
| Automated | `automated` | true | true |
| No Sentiment | `no-sentiment` | false | false |

---

## Troubleshooting

**Portfolio page crashes with `Unknown format code '%' for object of type 'str'`** — A config value under `regime.neutral` was saved as the string `"None"` (YAML quirk) instead of a number. Use YAML `null` or a real float. The `regime.neutral` block should have numeric overrides.

**Inflated `value_metric` from stale CSVs** — The bot always loads the most-recently dated CSV. Delete stale files from `data/` or run without `--skip-data` to regenerate.

**"All stocks show NEUTRAL"** — Batch Claude call failed. Check `investment_bot.log`. Common causes: missing `ANTHROPIC_API_KEY`, or Python < 3.10 event loop. Use `--op-mode no-sentiment` to bypass.

**"Config NOT written: validation gates failed"** — Tuned parameters didn't pass the held-out validation window. Use `--force-apply` only for manual inspection — not production.

**"⚠ unstable" in the diff table** — A parameter's Sharpe-opt and Calmar-opt values differ by > 5%. The averaged value may not be robust; review manually before applying.

**Not enough snapshots for IC computation** — `FactorResearchEngine` needs ≥ 2 dated parquets in `data/snapshots/`. Run the bot (or `make fetch-data`) on at least two separate days, or backfill: `make snapshot-backfill`.

**GMM test unavailable** — Install `scikit-learn`: `pip install -e ".[ui]"` (it's included in the `ui` extras).

---

## Security

- Never commit `.env` or any file containing credentials
- All sensitive values are read from environment variables at runtime
- The LLM review payload contains only performance metrics and parameter candidates — never account IDs, balances, or PII
- Safety parameters (stop-loss, position caps, order caps) are excluded from the optimizer
- `--op-mode` affects only the current process — it never writes to `config.yaml`

---

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses. Always conduct your own research and consider consulting a licensed financial advisor before making investment decisions.

## License

MIT License — see [LICENSE](LICENSE) for details.
