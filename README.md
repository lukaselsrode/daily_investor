# Daily Investor

A systematic investment platform built on Robinhood. Combines multi-factor fundamental scoring, AI-powered sentiment, and a full quantitative research stack — backtesting, walk-forward validation, parameter stability, regime detection, factor IC analytics, distribution regime analysis, and exposure diagnostics.

---

## Key Features

- **Multi-Factor Scoring** — Value (P/E + P/B, sector-relative v2), quality, income, and momentum → single `value_metric`
- **Momentum v2** — Relative strength vs SPY (3m/6m), risk-adjusted return, DMA trend structure, short-term momentum — all cross-sectionally percentile-ranked per day, causal
- **Value v2** — Sector-relative winsorized percentile ranking with distress penalties; replaces ratio-based value scoring
- **Three-Tier Market Regime** — Bullish / Neutral / Defensive via SPY 200DMA + VIX; raises ETF allocation, limits active buys, tightens stops in defensive mode
- **Factor Research Platform** — `FactorResearchEngine`: multi-horizon IC (5/20/60/120/252d), factor decay curves, decile monotonicity, rolling ICIR, cumulative IC
- **Distribution Regime Analysis** — Bimodality detection (GMM + BC test), local IC, cluster analysis, conditional alpha, threshold simulation — investigates whether alpha concentrates in score tails
- **Parquet Snapshot Store** — Each scoring run saves `data/snapshots/YYYY_MM_DD.parquet` for rolling IC and forward-return validation
- **Exposure Analytics** — Factor tilts (z-score vs universe), sector weights, HHI concentration, rolling exposure drift
- **Disciplined Sell Engine** — Hard sells (stop-loss, yield trap, quality floor) execute immediately; soft sells (take-profit, weak value) require Claude confirmation
- **Anti-Lookahead Backtest** — All rolling features computed causally; contribution-adjusted TWR (chain-link) for portfolio and benchmark
- **Validation-Aware Auto-Tune** — `scipy.differential_evolution` across Sharpe and Calmar; parameters written only if held-out validation gates pass
- **Optional LLM Tune Review** — Routes optimizer candidates through Claude for a second-opinion before applying
- **Batch AI Sentiment** — Async concurrent Claude calls with exponential backoff
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
│   │   ├── exceptions.py             # Domain exception hierarchy
│   │   └── interfaces.py             # Typed Protocol contracts for all services
│   ├── config/
│   │   ├── schema.py                 # Frozen dataclasses for all YAML sections
│   │   └── manager.py                # Singleton ConfigManager with cached_property sections
│   ├── data/
│   │   ├── base.py                   # ABCs: MarketDataProvider, SentimentProvider, etc.
│   │   ├── cache.py                  # CSV read/write helpers
│   │   ├── universe.py               # UniverseBuilder (scrapes Wikipedia + Robinhood sources)
│   │   ├── fundamentals.py           # FundamentalsProvider (yfinance + Robinhood)
│   │   ├── market.py                 # MarketDataProvider (yfinance wrapper)
│   │   └── sentiment.py              # SentimentProvider wrapping sentiment_analysis.py
│   ├── strategy/
│   │   ├── base.py                   # ScorerBase ABC, ScoreBreakdown
│   │   ├── value.py                  # ValueScorer: P/E + P/B with guardrails (legacy)
│   │   ├── value_v2.py               # ValueScorerV2: sector-relative winsorized percentile
│   │   ├── quality.py                # QualityScorer: liquidity, earnings, dividend health
│   │   ├── income.py                 # IncomeScorer: yield with trap detection
│   │   ├── momentum.py               # MomentumEngine: v2 multi-factor + v1 fallback
│   │   ├── composite.py              # CompositeScorer: weighted combination → value_metric
│   │   ├── snapshots.py              # Parquet snapshot store: save, load, prune, backfill
│   │   ├── factors/
│   │   │   └── engine.py             # FactorEngine: score_single, score_universe, exposures
│   │   ├── regimes/
│   │   │   ├── models.py             # RegimeState, RegimeHistoryEntry, RegimeLabel
│   │   │   ├── detector.py           # RegimeDetector: live detect + historical replay
│   │   │   └── __init__.py
│   │   └── research/
│   │       ├── ic_engine.py          # FactorResearchEngine: multi-horizon IC, decay, decile
│   │       └── __init__.py
│   ├── research/
│   │   └── distribution_regime_analysis.py  # DistributionAnalyzer: bimodality, tail IC, clusters
│   ├── portfolio/
│   │   ├── risk.py                   # RiskManager.can_buy() — position/sector/order gates
│   │   ├── sell_engine.py            # SellDecisionEngine.evaluate() — hard/soft sell logic
│   │   ├── manager.py                # PortfolioManager
│   │   ├── sizing.py                 # Order sizing helpers
│   │   ├── harvest.py                # Profit harvesting logic
│   │   └── exposure/
│   │       └── analyzer.py           # ExposureAnalyzer: factor tilts, sector, HHI, drift
│   ├── execution/
│   │   ├── base.py                   # BrokerAdapter ABC
│   │   ├── paper.py                  # PaperBroker — in-memory, no API
│   │   └── robinhood.py              # RobinhoodBroker — live orders with retry backoff
│   ├── backtesting/
│   │   ├── engine.py                 # BacktestEngine: simulate, run, walk_forward
│   │   ├── validator.py              # WalkForwardValidator: train/val split, gate checks
│   │   └── results.py                # BacktestResult, ValidationResult typed wrappers
│   ├── tuning/
│   │   ├── tuner.py                  # ParameterTuner: tune, auto_tune, apply_params
│   │   ├── stability.py              # StabilityAnalyzer: multi-window parameter scan
│   │   └── results.py                # TuneResult, AutoTuneResult, StabilityReport
│   ├── reporting/
│   │   ├── attribution.py            # AttributionReporter
│   │   ├── diagnostics.py            # DiagnosticsReporter: CSV + robustness TXT
│   │   └── plots.py                  # PlotManager: heatmaps and validation charts
│   ├── ui/
│   │   ├── streamlit_app.py          # Dashboard entry point
│   │   ├── utils.py                  # Shared UI helpers, path constants, CSV loaders
│   │   ├── layout/sidebar.py         # Navigation sidebar
│   │   ├── sections/                 # Top-level page sections (one per sidebar entry)
│   │   │   ├── operations.py         # ⚡ Operations section
│   │   │   ├── portfolio.py          # 💼 Portfolio section
│   │   │   ├── research.py           # 🔬 Research section
│   │   │   ├── validation.py         # ✅ Validation section
│   │   │   └── system.py             # ⚙️ System section
│   │   └── components/               # Reusable tab components
│   │       ├── home.py               # System dashboard / status
│   │       ├── run_control.py        # CLI command builder + subprocess runner
│   │       ├── intents.py            # Order intent dry-run preview
│   │       ├── execution.py          # Live execution panel
│   │       ├── portfolio.py          # Holdings display
│   │       ├── exposure.py           # Factor tilts and sector exposure
│   │       ├── regime.py             # Regime inspector + effective config by regime
│   │       ├── scoring.py            # Scored universe explorer
│   │       ├── value_diagnostics.py  # Value v2 distribution and decile analysis
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
│   ├── backtest.py                   # Simulation core (PrecomputedData, SimResult)
│   ├── tuner.py                      # Optimizer core
│   ├── source_data.py                # Universe + fundamentals + scoring pipeline
│   ├── sentiment_analysis.py         # Batch async + single-stock Claude sentiment
│   ├── sentiments.py                 # News data collection (yfinance + Robinhood fallback)
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
| `fetch-data` | Fetch fresh fundamentals + news, save CSVs + snapshot — **no trades placed** |
| `run` | Live trading run (sell + buy cycle) |
| `backtest DAYS` | Run backtest simulation |
| `tune DAYS` | Single-objective parameter tune — prints diff, no write |
| `auto-tune [DAYS]` | Dual-objective tune with walk-forward validation (default: 90d) |
| `stability-scan` | Parameter stability scan across multiple windows — research only, no writes |
| `report` | Run a quick 90-day backtest and print results |

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
  --mode MODE              liquid_universe_sanity_test | walk_forward_price_only_test | current_universe_stress_test
  --objective sharpe|calmar
  --output-dir PATH
```

---

## Makefile Targets

```bash
# Data
make fetch-data            # Fetch fresh fundamentals + news, save CSVs + snapshot (no trades)

# Live trading
make run                   # Safe mode — manual confirmation at each step
make run-auto              # Automated mode — no prompts
make run-skip              # Safe mode, reuse cached CSVs (faster)
make run-dry               # No sentiment, no trades — scoring + logic preview only

# Backtesting
make backtest              # 365-day backtest (default mode)
make backtest DAYS=180
make backtest BT_MODE=walk_forward_price_only_test
make backtest-wf           # Walk-forward mode (low lookahead)

# Parameter tuning
make tune                  # Single-objective tune, no write  (TUNE_DAYS=120  OBJ=sharpe)
make auto-tune             # Dual-objective tune, walk-forward validation, no write
make auto-tune-apply       # auto-tune + write config.yaml if validation passes
make auto-tune-llm         # auto-tune + Claude second-opinion + apply

# Research & diagnostics
make stability             # Parameter stability scan across multiple windows
make report                # Quick 90-day backtest → reports/
make regime                # Print current market regime (live SPY + VIX)
make ic                    # Print IC summary across default horizons (needs ≥ 2 snapshots)
make snapshot-info         # Show snapshot store status (count, date range)
make snapshot-backfill     # Backfill parquet snapshots from existing agg_data CSVs

# Dashboard
make ui                    # Launch Streamlit dashboard

# Development
make install               # Install / reinstall package in editable mode
make test                  # Run full pytest suite
make test-watch            # Re-run tests on file changes (requires pytest-watch)
make lint                  # Run ruff linter over src/
```

---

## Scoring Model

### Factor Scores

| Score | What it measures |
|-------|-----------------|
| `value_score` | Sector-relative P/E and P/B cheapness (v2: winsorized percentile ranking) |
| `income_score` | Dividend yield quality (capped; 0 if yield trap or no yield) |
| `quality_score` | Liquidity, earnings existence, dividend health |
| `momentum_score` | Multi-factor v2: relative strength, risk-adjusted return, DMA trend, short-term momentum |

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

### Momentum Score v2

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

### Value Score v2

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

Tests whether the post-value_v2 bimodal score distribution contains predictive information.

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
| `liquid_universe_sanity_test` | MEDIUM | Random sample of liquid stocks (default) |
| `walk_forward_price_only_test` | LOW | Liquid sample; fundamental arrays zeroed, momentum only |
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
momentum_v2 sub-weights:
  rs_3m, rs_6m, risk_adj_3m, trend_structure, return_1m
```

Safety parameters (hard stop-loss, position caps, order caps) are never touched by the optimizer. `score_weights.value` and `score_weights.income` are frozen by default.

### Validation Gates (auto-tune)

Before writing any changes to `config.yaml`:

| Gate | Config key | Default |
|------|-----------|---------|
| Excess return vs benchmark | `min_validation_excess_return` | 0.0% |
| Max drawdown | `max_validation_drawdown` | −20% |
| Sharpe ratio | `min_validation_sharpe` | 0.25 |

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
| 🔍 Factors | Scoring universe explorer + Value v2 diagnostics (PE/PB ranks, sector comparison) |
| 📡 IC Analysis | Multi-horizon IC, factor decay curves, cumulative IC, ICIR; rolling IC time series |
| 📊 Rank & Deciles | Decile monotonicity — does higher score predict better returns? |
| 🔗 Correlations | Pairwise factor IC, VIF, OLS residualization, variance decomposition |
| 🌡️ Regime | IC conditioned on market regime (bull/bear/high-vol/sideways) |
| 🧬 Distribution | Bimodality test, tail analysis, local IC, GMM clusters, conditional alpha, threshold simulation |
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
git clone https://github.com/yourusername/daily_investor.git
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
```

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
make auto-tune-apply       # apply if satisfied
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

# Sector-relative value scoring (v2)
value_v2:
  enabled: true
  winsorize_pct: 0.05
  sector_relative: true
  clamp_low: -1.0
  clamp_high: 1.5

# Backtest
backtest:
  default_mode: liquid_universe_sanity_test
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
