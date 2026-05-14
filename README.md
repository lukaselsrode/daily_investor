# Daily Investor

An automated investment strategy and quantitative research platform. Combines multi-factor fundamental analysis, AI-powered sentiment, and a full portfolio intelligence stack — backtesting, walk-forward validation, parameter stability, regime detection, factor IC analytics, and exposure diagnostics — all executing via Robinhood.

## Key Features

- **Multi-Factor Scoring**: Value (P/E, P/B sector-relative v2), quality, income, and momentum → single `value_metric`
- **Momentum v2**: Relative strength vs SPY (3m/6m), risk-adjusted return, DMA trend structure, short-term momentum — all cross-sectionally ranked per day, no lookahead
- **Value v2**: Sector-relative winsorized percentile ranking with distress penalties — replaces ratio-based value scoring
- **Three-Tier Market Regime**: Bullish / Neutral / Defensive via SPY 200DMA + VIX. Raises ETF allocation, limits active buys, and tightens stops in defensive mode
- **Regime Detector Service**: `RegimeDetector.detect()` gives live regime + confidence; `classify_history()` replays N days for backtesting
- **Factor Research Platform**: `FactorResearchEngine` — multi-horizon IC (5/20/60/120/252d), factor decay curves, decile spread / monotonicity, rolling ICIR, cumulative IC
- **Parquet Snapshot Store**: Each scoring run saves `data/snapshots/YYYY_MM_DD.parquet` for rolling IC and forward-return validation
- **Exposure Analytics**: `ExposureAnalyzer` — factor tilts (z-score vs. universe), sector weights, HHI concentration, rolling exposure drift
- **Factor Orthogonalization**: Correlation matrix, VIF analysis, and variance decomposition across factor pairs
- **Disciplined Sell Engine**: Hard sells (stop-loss, yield trap, quality floor) execute immediately; soft sells (take-profit, weak value) require Claude confirmation
- **Portfolio Risk Controls**: Per-position, per-sector, per-order caps enforced before every buy
- **Profit Harvesting**: Take-profit proceeds route to core ETFs rather than sitting as idle cash
- **Anti-Lookahead Backtest**: All rolling features computed causally. Contribution-adjusted TWR (chain-link) for both portfolio and benchmark
- **Validation-Aware Auto-Tune**: `scipy.differential_evolution` across Sharpe and Calmar; parameters written only if held-out validation gates pass
- **Parameter Stability Reporting**: Spread between Sharpe-opt and Calmar-opt parameter sets flags unstable dimensions
- **Optional LLM Tune Review**: Routes optimizer candidates through Claude for a second-opinion before applying
- **Batch AI Sentiment**: Async concurrent Claude calls with exponential backoff

## Project Structure

```
daily_investor/
├── cfg/
│   └── config.yaml                   # All tunable parameters (never commit credentials)
├── data/
│   ├── agg_data_YYYY_MM_DD.csv       # Scored universe (newest always used)
│   └── snapshots/
│       └── YYYY_MM_DD.parquet        # Daily scored universe snapshots (IC store)
├── src/
│   ├── cli/                          # CLI dispatcher
│   │   ├── main.py                   # Argument parsing, command dispatch
│   │   └── commands.py               # Per-command handlers
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
│   │   ├── sentiment.py              # SentimentProvider wrapping sentiment_analysis.py
│   │   ├── universe.py               # UniverseBuilder wrapping gen_symbols_list
│   │   ├── fundamentals.py           # FundamentalsProvider
│   │   └── market.py                 # MarketDataProvider (yfinance wrapper)
│   ├── strategy/
│   │   ├── base.py                   # ScorerBase ABC, ScoreBreakdown
│   │   ├── value.py                  # ValueScorer: P/E + P/B with guardrails
│   │   ├── value_v2.py               # ValueScorerV2: sector-relative winsorized percentile
│   │   ├── quality.py                # QualityScorer: liquidity, earnings, dividend health
│   │   ├── income.py                 # IncomeScorer: yield with trap detection
│   │   ├── momentum.py               # MomentumEngine: v2 multi-factor + v1 fallback
│   │   ├── composite.py              # CompositeScorer: weighted combination → value_metric
│   │   ├── snapshots.py              # Parquet snapshot store: save, load, prune, backfill
│   │   ├── factors/
│   │   │   ├── engine.py             # FactorEngine: score_single, score_universe, exposures
│   │   │   └── __init__.py
│   │   ├── regimes/
│   │   │   ├── models.py             # RegimeState, RegimeHistoryEntry, RegimeLabel
│   │   │   ├── detector.py           # RegimeDetector: live detect + historical replay
│   │   │   └── __init__.py
│   │   └── research/
│   │       ├── ic_engine.py          # FactorResearchEngine: multi-horizon IC, decay, decile
│   │       └── __init__.py
│   ├── portfolio/
│   │   ├── risk.py                   # RiskManager.can_buy() — position/sector/order gates
│   │   ├── sell_engine.py            # SellDecisionEngine.evaluate() — hard/soft sell logic
│   │   ├── manager.py                # PortfolioManager
│   │   ├── sizing.py                 # Order sizing helpers
│   │   ├── harvest.py                # Profit harvesting logic
│   │   └── exposure/
│   │       ├── analyzer.py           # ExposureAnalyzer: factor tilts, sector, HHI, drift
│   │       └── __init__.py
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
│   │   ├── utils.py                  # Shared UI helpers
│   │   └── components/               # One file per page (see UI section)
│   ├── main.py                       # Legacy live-trading loop
│   ├── backtest.py                   # Legacy simulation core
│   ├── tuner.py                      # Legacy optimizer core
│   ├── source_data.py                # Universe + fundamentals + scoring pipeline
│   ├── sentiment_analysis.py         # Batch async + single-stock Claude sentiment
│   ├── sentiments.py                 # News data collection
│   ├── util.py                       # Config constants, schema, CSV helpers
│   └── tests.py                      # Legacy pure-function unit tests
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
└── .env                              # Credentials (never commit)
```

The modular layer (`core/`, `config/`, `strategy/`, `portfolio/`, `execution/`, `backtesting/`, `tuning/`, `reporting/`) provides a typed, testable API over the legacy engine files. Legacy `.py` files remain as computation backends and are being gradually hollowed out.

---

## Scoring Model

### Factor Scores

| Score | What it measures |
|-------|-----------------|
| `value_score` | Sector-relative P/E and P/B cheapness (v2: winsorized percentile ranking) |
| `income_score` | Dividend yield quality (capped; 0 if yield trap or no yield) |
| `quality_score` | Liquidity, earnings existence, dividend health signal |
| `momentum_score` | Multi-factor v2: relative strength, risk-adjusted return, DMA trend, short-term momentum |

### Momentum Score v2

All sub-scores are **cross-sectionally percentile-ranked** across the live universe on each day. Causal — no lookahead.

| Sub-factor | Default weight | What it captures |
|---|---|---|
| `rs_3m` | 0.25 | Return_3m − SPY_3m (relative strength, 3-month) |
| `rs_6m` | 0.25 | Return_6m − SPY_6m (relative strength, 6-month) |
| `risk_adj_3m` | 0.20 | return_3m / realized_vol_3m (Sharpe-like, 63-day) |
| `trend_structure` | 0.15 | Price vs 50 DMA and 200 DMA (deterministic, not ranked) |
| `return_1m` | 0.10 | Raw 21-day return, percentile-ranked |
| `return_5d` | 0.05 | 5-day short-term check |

**Penalties applied after weighting:**

| Penalty | Trigger |
|---|---|
| Falling-knife | 3m return < −15% |
| Overextension | 52-week position > 97% |
| High volatility | Annualized realized vol > 50% |

Final momentum score clamped to [−1.0, 1.5].

### Value Score v2

Replaces the ratio-based value scorer with sector-relative winsorized percentile ranking:

1. Within each sector (minimum 5 stocks), winsorize P/E and P/B at 5th/95th percentile
2. Percentile-rank each stock against its sector peers
3. Blend: `0.60 × pe_rank + 0.40 × pb_rank`, scaled to [−1.0, 1.5]
4. Apply distress penalties: `PE ≤ 5 → −0.30`, negative EPS → `−0.25`

Falls back to global ranking for small sectors (< 5 stocks).

### Composite Score

```
value_metric = sw_value    × value_score
             + sw_quality  × quality_score
             + sw_income   × income_score
             + sw_momentum × momentum_score
```

Default weights:

```yaml
score_weights:
  value:    0.08
  quality:  0.50
  income:   0.08
  momentum: 0.34
```

---

## Factor Research Platform

### Snapshot Store

Every scoring run saves a dated Parquet file to `data/snapshots/YYYY_MM_DD.parquet`. These snapshots are the foundation for all rolling IC and forward-return validation.

```python
from strategy.snapshots import save_snapshot, load_snapshots, list_snapshots
from strategy.research import FactorResearchEngine

# Snapshots are saved automatically on each run.
# Backfill from existing CSVs:
from strategy.snapshots import backfill_from_csvs
backfill_from_csvs()

# Multi-horizon IC across all snapshot pairs:
engine = FactorResearchEngine()
ic_df  = engine.compute_multi_horizon_ic(horizons=[5, 20, 60, 120])
summ   = engine.compute_ic_summary(ic_df)
decay  = engine.compute_factor_decay()
spread = engine.compute_decile_spread("momentum_score", horizon_days=20)
```

### FactorResearchEngine

| Method | Returns |
|---|---|
| `compute_multi_horizon_ic(factors, horizons, ic_type)` | `[date, factor, horizon_days, ic, n_stocks, p_value]` |
| `compute_ic_summary(ic_df)` | `[factor, horizon_days, mean_ic, icir, hit_rate, t_stat, n_periods]` |
| `compute_factor_decay(factors)` | IC vs. horizon per factor — the decay curve |
| `compute_decile_spread(factor, horizon_days, n_deciles)` | Mean forward return by score decile |
| `compute_rolling_icir(factor, horizon_days, window)` | Trailing ICIR over time |
| `compute_cumulative_ic(factors, horizon_days)` | Cumulative IC over time per factor |

**IC interpretation:**

| IC range | Interpretation |
|---|---|
| > 0.10 | Strong signal |
| 0.05 – 0.10 | Moderate |
| 0.00 – 0.05 | Weak / noise |
| < 0.00 | Negative (contrarian) |

ICIR > 0.5 is generally considered actionable.

### RegimeDetector

```python
from strategy.regimes import RegimeDetector

det   = RegimeDetector()
state = det.detect()            # fetches live SPY + VIX
print(state.regime, state.confidence, state.notes)

# Pure computation (testable, no network):
state = det.detect_from_data(spy_price=580.0, spy_ma200=540.0, vix=18.0)

# Historical replay:
history = det.classify_history(days=365)
```

### ExposureAnalyzer

```python
from portfolio.exposure import ExposureAnalyzer

analyzer = ExposureAnalyzer()
report   = analyzer.analyze(portfolio, universe_df, total_equity=50000, cash=5000)

print(report.momentum_tilt)    # z-score vs. universe median
print(report.hhi)              # Herfindahl-Hirschman concentration
print(report.sector_weights)   # {sector: weight}

drift_df = analyzer.compute_rolling_drift(portfolio, days=90)
```

---

## Portfolio Risk Controls

Applied to every buy before an order is placed:

| Rule | Config key | Default | Behaviour |
|------|-----------|---------|-----------|
| Liquidity gate | `min_liquidity_volume` | 500,000 | Skip if avg volume below threshold |
| Order size cap | `max_order_pct_of_cash` | 10% | Cap single order to 10% of available cash |
| Position cap | `max_single_position_pct` | 5% | Reduce buy so total position ≤ 5% of portfolio |
| Sector cap | `max_sector_pct` | 25% | Reduce buy so sector exposure ≤ 25% of portfolio |
| Minimum order | `min_order_amount` | $5.00 | Skip if reduced amount falls below this |
| ETF floor | `min_index_pct` | 60% | ETF allocation floor; optimizer cannot reduce below this |

When a cap is hit the allocation is **reduced** to the maximum allowed. The buy is skipped only if the reduced amount falls below `min_order_amount`.

---

## Sell Decision Engine

Each non-ETF holding is evaluated by `SellDecisionEngine.evaluate()` which classifies sells as **hard** or **soft**.

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

Soft sells are sent to Claude. `HOLD` with confidence ≥ `sell_sentiment_override_confidence` (default 85%) keeps the position.

### Profit Harvesting

Take-profit proceeds are classified as `harvest_exit`. Proceeds exceeding `min_harvest_amount` ($25) route to `harvest_etfs` (default: SPY, VTI) rather than sitting as idle cash. Split: 80% to ETFs, 20% recycled for continued stock exposure.

---

## Market Regime

| Regime | SPY vs 200DMA | VIX | Effect |
|--------|---------------|-----|--------|
| **Bullish** | Above | < 20 | Normal operation |
| **Neutral** | Below, or VIX 20–30 | 20–30 | Base config |
| **Defensive** | — | ≥ 30 | ETF allocation → 85%, max stock buys → 3, stop-loss tightened by 0.05, ETF MA filter active |

All thresholds configurable under `regime:` in `config.yaml`. The `RegimeDetector` also exposes confidence scores calibrated by distance from the VIX thresholds.

---

## Backtest Engine

Simulates the full strategy over historical price data from `yfinance`.

### Anti-Lookahead Causal Features

All rolling features use only price data up to day D on day D:

| Feature | Lookback | Notes |
|---------|---------|-------|
| `position_52w` | 252 bars | 52-week high/low range position |
| `return_1m` | 21 bars | 1-month price return |
| `return_5d` | 5 bars | 5-day price return |
| `return_3m` | 63 bars | 3-month price return |
| `return_6m` | 126 bars | 6-month price return |
| `realized_vol_3m` | 63 bars | Annualized std dev |
| `rs_3m`, `rs_6m` | 63/126 bars | Return minus benchmark return |
| `above_50dma` | 50 bars | Price vs 50-bar rolling mean |
| `above_200dma` | 200 bars | Price vs 200-bar rolling mean |

Cross-sectional percentile ranking across stocks at one point in time is **not** a lookahead bias.

### Backtest Modes

| Mode | Lookahead bias | Universe |
|------|---------------|----------|
| `liquid_universe_sanity_test` | MEDIUM | Random sample from liquid stocks (default) |
| `current_universe_stress_test` | HIGH | Top-N by current `value_metric` — **not predictive** |
| `walk_forward_price_only_test` | LOW | Liquid sample; fundamental arrays zeroed, momentum only |

### Realism Controls

| Feature | Config key | Default |
|---------|-----------|---------|
| Volatility-scaled slippage | `vol_slippage_scaling` | `true` — effective_bps = base × (1 + 2.0 × annualized_vol) |
| Max trades per week | `max_trades_per_week` | 10 |
| Post-sell cooldown | `cooldown_days_after_sell` | 3 days |
| Post-stopout cooldown | `cooldown_days_after_stopout` | 7 days |
| Regime classification | SPY vs 200DMA + VIX | tracked daily |

---

## Parameter Tuner

`scipy.optimize.differential_evolution` maximizes Sharpe or Calmar ratio over a back-simulation window. Tunable parameters:

```
score_weights (value, quality, income, momentum)
index_pct                   — ETF allocation fraction
metric_threshold            — minimum score to qualify as a buy
trailing_stop_pct           — trailing stop distance from peak
momentum_v2 sub-weights:
  rs_3m, rs_6m, risk_adj_3m, trend_structure, return_1m
```

Safety parameters (stop-loss, position caps, order caps) are never touched by the optimizer.

### Validation Gates

Before writing any changes to `config.yaml`:

| Gate | Config key | Default |
|------|-----------|---------|
| Excess return vs benchmark | `min_validation_excess_return` | 0.0% |
| Max drawdown | `max_validation_drawdown` | −20% |
| Sharpe ratio | `min_validation_sharpe` | 0.25 |

---

## Streamlit Dashboard

An interactive research and control panel. Runs separately from the CLI — does not replace it.

### Pages

| Page | Description |
|---|---|
| 📊 Home | System status, config overview, data freshness, log tail |
| 🚀 Run Control | Build and execute any CLI command with full options; streams output |
| 💼 Portfolio | Holdings from cached CSV; live broker data on demand |
| 🎯 Order Intents | Dry-run preview of proposed buys/sells/harvests |
| 🔬 Scoring Explorer | Filter, sort, and drill into the scored universe |
| 📐 Value Diagnostics | Value score breakdown: PE/PB ranks, sector-relative comparisons |
| 🔗 Factor Orthogonalization | Correlation matrix, VIF, variance decomposition across factors |
| 📡 Rolling IC | Single-horizon Spearman IC over time with ICIR and distribution |
| 🧪 Factor Lab | Multi-horizon IC, factor decay curves, cumulative IC, decile spread, rolling ICIR |
| 📈 Backtests | Run `BacktestEngine` interactively; view full results and attribution |
| ⚙️ Auto-Tune | Dual-objective tune; view diff and validation status; apply with gate |
| 🔭 Stability & Robustness | Multi-window stability scan; heatmaps and robustness report |
| 🌡️ Regime & Risk | Current regime thresholds; effective config under each regime |
| ⚖️ Exposure | Factor tilts, sector allocation, HHI concentration, rolling exposure drift |
| 🩺 Reliability Diag. | NaN coverage, score distributions, yield traps, liquidity signals |
| 🗂️ Data Explorer | Explore any CSV/parquet with interactive charts and filters |
| 📋 Logs / Audit | Tail application log; browse order intents and order results |
| 🛠️ Config Viewer | Parsed config.yaml in readable sections; download button |

### Launch

```bash
make ui
# or:
streamlit run src/ui/streamlit_app.py
```

### Safety

- The UI starts in **read-only mode** by default
- The live-execution toggle is locked unless `ui.allow_live_execution: true` in config
- All paths go through the same `RiskManager` and audit logging as the CLI

Add to `cfg/config.yaml` to enable controlled live execution from the UI:

```yaml
ui:
  allow_live_execution: true
  allow_config_writes: true
  allow_force_apply: false
  require_confirmation_phrase: true
  confirmation_phrase: EXECUTE
  intent_ttl_minutes: 5
```

---

## Running the Application

```bash
# Full run — refresh data, analyze, trade
make run           # safe mode (manual confirmation)
make run-auto      # automated mode

# Skip data generation — reuse today's cached CSVs
make run-skip

# Backtesting
make backtest                        # 365-day default
make backtest DAYS=180
make backtest BT_MODE=walk_forward_price_only_test

# Parameter tuning
make tune                            # single-objective, no write
make auto-tune                       # dual-objective, walk-forward validation
make auto-tune-apply                 # write config if validation passes
make auto-tune-llm                   # + Claude second-opinion

# Research
make stability                       # parameter stability scan
make report                          # diagnostics report → reports/
make snapshot-info                   # show snapshot store status
make regime                          # print current market regime

# Dashboard
make ui                              # launch Streamlit

# Tests
make test
```

Or use the CLI directly:

```bash
daily-investor run --op-mode safe
daily-investor backtest 90 --mode liquid_universe_sanity_test
daily-investor auto-tune 180 --apply --llm-review
daily-investor stability-scan
```

---

## Configuration

All settings live in `cfg/config.yaml`. Key sections:

```yaml
# Factor weights (optimizer-tunable)
score_weights:
  value:    0.08
  quality:  0.50
  income:   0.08
  momentum: 0.34

# Three-tier market regime
regime:
  spy_ma_period: 200
  vix_defensive_threshold: 30.0
  vix_neutral_threshold: 20.0
  defensive:
    index_pct_override: 0.85
    max_buys_override: 3
    stop_loss_tighten: 0.05

# Snapshot store (for rolling IC)
snapshots:
  enabled: true
  retention_days: 365
  compression: snappy

# Sector-relative value scoring (v2)
value_v2:
  enabled: true
  winsorize_pct: 0.05
  sector_relative: true

# Backtest
backtest:
  default_mode: liquid_universe_sanity_test
  starting_capital: 5000.0
  weekly_contribution: 400.0
  slippage_bps: 10.0
  train_pct: 0.70
  use_time_weighted_returns: true
  vol_slippage_scaling: true

# Optimizer control
tuning:
  frozen_parameters:
    - score_weights.value
    - score_weights.income
    - momentum_v2.weights.rs_3m
    ...
```

### Operating Modes

| Mode | `--op-mode` arg | `auto_approve` | `use_sentiment_analysis` |
|------|-----------------|---------------|--------------------------|
| Safe | `safe` | `false` | `true` |
| Automated | `automated` | `true` | `true` |
| No Sentiment | `no-sentiment` | `false` | `false` |

---

## Setup

**Requirements:** Python 3.10+, Robinhood account, Anthropic API key

```bash
git clone https://github.com/yourusername/daily_investor.git
cd daily_investor

python -m venv .venv
source .venv/bin/activate

pip install -e .               # installs daily-investor CLI + all dependencies
pip install -e ".[ui,dev]"     # also installs Streamlit and test tools
```

`.env` file (at project root):
```
RB_ACCT=your_robinhood_email
RB_CREDS=your_robinhood_password
RB_MFA_SECRET=your_totp_secret         # optional: skip interactive MFA prompt
ANTHROPIC_API_KEY=your_anthropic_key   # required for sentiment and LLM tune review
```

```bash
daily-investor --help
pytest                                 # runs all tests — no credentials needed
```

---

## Troubleshooting

**Inflated value_metrics from old cached CSVs** — The bot always loads the most-recently dated CSV. Delete stale files from `data/` or run without `--skip-data` to regenerate.

**"All stocks show NEUTRAL"** — Batch Claude call failed. Check `investment_bot.log`. Common causes: missing `ANTHROPIC_API_KEY`, or Python < 3.10 event loop. Use `--op-mode no-sentiment` to bypass.

**"37 columns passed, passed data had 33 columns"** — Stale `source_data.py` with incorrect `_BASE_AGG_COLUMNS`. The value_v2 diagnostic columns (`value_score_raw`, `sector_value_score`, `relative_pe`, `relative_pb`) are injected post-construction and must be excluded from `_BASE_AGG_COLUMNS`.

**"Config NOT written: validation gates failed"** — Tuned parameters didn't pass the held-out validation window. Use `--force-apply` to override for manual inspection only.

**"⚠ unstable" in the diff table** — A parameter's Sharpe-opt and Calmar-opt values differ by > 5%. The averaged value may not be robust; review manually before applying.

**Not enough snapshots for IC computation** — `FactorResearchEngine` needs at least 2 dated parquets in `data/snapshots/`. Run the bot on at least two separate days, or use `backfill_from_csvs()` to migrate existing CSVs.

---

## Security

- Never commit `.env` or any file containing credentials
- All sensitive values are read from environment variables at runtime
- The LLM review payload contains only performance metrics and parameter candidates — never account IDs, balances, or PII
- Safety parameters (stop-loss, position caps, order caps) are excluded from the LLM-adjustable set
- `--op-mode` affects only the current process — it never writes to `config.yaml`

---

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this tool. Always conduct your own research and consider consulting with a licensed financial advisor before making investment decisions.

## License

MIT License — see [LICENSE](LICENSE) for details.
