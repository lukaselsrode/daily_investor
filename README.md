# Daily Investor

An automated investment strategy tool that combines fundamental analysis with AI-powered sentiment analysis to make informed investment decisions. The system evaluates stocks based on financial metrics, multi-factor momentum, market regime, and news analysis, then executes trades via Robinhood.

## Key Features

- **Multi-Factor Momentum (v2)**: Relative strength vs SPY (3m/6m), risk-adjusted momentum (return/vol), trend structure (50/200 DMA), 5d/1m returns — all cross-sectionally ranked per day, no lookahead
- **Factor-Based Scoring**: Combines value (P/E, P/B), income (dividend yield), quality, and momentum into a single `value_metric`
- **Valuation Guardrails**: Caps P/E and P/B components to prevent extreme scores from thin or stale fundamental data
- **Portfolio Risk Controls**: Per-position cap, per-sector cap, and per-order size cap enforced before every buy
- **Disciplined Sell Engine**: Separates hard sells (stop-loss, yield trap, quality floor) from soft sells (take-profit, weak value) with sentiment override only on soft sells
- **Three-Tier Market Regime**: Bullish / Neutral / Defensive classification using SPY 200DMA + VIX. Defensive regime raises ETF allocation, limits active buys, tightens stops, and activates ETF MA filter
- **ETF Core Protection**: ETF positions are exempt from stock stop-loss logic. The optional ETF MA filter only activates in defensive regime
- **Profit Harvesting**: Take-profit proceeds are automatically reinvested into core ETFs rather than sitting as idle cash
- **Anti-Lookahead Backtest**: All rolling features computed causally — only price data up to day D used on day D. Contribution-adjusted TWR (chain-link) for both portfolio and benchmark
- **Backtest Realism**: Per-trade cooldown after sells/stopouts, weekly trade budget, volatility-scaled slippage, regime tracking, stopout attribution
- **Validation-Aware Auto-Tune**: Optimizes score weights and sell thresholds via `scipy.differential_evolution`; tuned parameters are only written to config if they pass held-out validation gates
- **Parameter Stability Reporting**: Shows spread between Sharpe-optimized and Calmar-optimized parameter sets to flag unstable dimensions
- **Optional LLM Tune Review**: Routes optimizer candidates through Claude for a second-opinion review before applying
- **Batch AI Sentiment Analysis**: Analyzes multiple stocks per Claude API call using async concurrency
- **Fractional + Whole Share Fallback**: Attempts fractional orders first; falls back to whole-share market order with a final risk re-check
- **Centralized Rate-Limit Backoff**: All Robinhood API calls retry with exponential backoff on 429 / throttle errors

## Project Structure

```
daily_investor/
├── cfg/
│   └── config.yaml                  # All tunable parameters (never commit credentials here)
├── data/                            # CSV cache (dated filenames, newest always used)
├── src/
│   ├── cli/                         # CLI dispatcher — new modular entry point
│   │   ├── main.py                  # Argument parsing, command dispatch
│   │   └── commands.py              # Per-command handlers (run, backtest, tune, …)
│   ├── core/
│   │   ├── types.py                 # Shared dataclasses: SimResult, TradeRecord, SellDecision, …
│   │   └── logging.py               # Structured JSON logging, configure_logging()
│   ├── config/
│   │   ├── schema.py                # 19 frozen dataclasses for all YAML sections
│   │   └── manager.py               # Singleton ConfigManager with cached_property sections
│   ├── data/
│   │   ├── base.py                  # ABCs: MarketDataProvider, SentimentProvider
│   │   ├── cache.py                 # CSV read/write helpers
│   │   ├── sentiment.py             # SentimentProvider wrapping sentiment_analysis.py
│   │   ├── universe.py              # UniverseBuilder wrapping gen_symbols_list
│   │   ├── fundamentals.py          # FundamentalsProvider (stub — Phase 2)
│   │   └── market.py                # MarketDataProvider (stub — Phase 2)
│   ├── strategy/
│   │   ├── value.py                 # ValueScorer: P/E + P/B with guardrails
│   │   ├── quality.py               # QualityScorer: liquidity, earnings, dividend health
│   │   ├── income.py                # IncomeScorer: yield with trap detection
│   │   ├── momentum.py              # MomentumEngine: v2 multi-factor + v1 fallback
│   │   └── composite.py             # CompositeScorer: weighted combination → value_metric
│   ├── portfolio/
│   │   ├── risk.py                  # RiskManager.can_buy() — all position/sector/order gates
│   │   └── sell_engine.py           # SellDecisionEngine.evaluate() — hard/soft sell logic
│   ├── execution/
│   │   ├── base.py                  # BrokerAdapter ABC
│   │   ├── paper.py                 # PaperBroker — in-memory, no API
│   │   └── robinhood.py             # RobinhoodBroker — live orders with retry backoff
│   ├── backtesting/
│   │   ├── engine.py                # BacktestEngine: simulate(), run(), run_walk_forward()
│   │   ├── validator.py             # WalkForwardValidator: train/val split, gate checks
│   │   └── results.py               # BacktestResult, ValidationResult typed wrappers
│   ├── tuning/
│   │   ├── tuner.py                 # ParameterTuner: tune(), auto_tune(), apply_params()
│   │   ├── stability.py             # StabilityAnalyzer: multi-window scan()
│   │   └── results.py               # TuneResult, AutoTuneResult, StabilityReport
│   ├── reporting/
│   │   ├── attribution.py           # AttributionReporter: stability classification
│   │   ├── diagnostics.py           # DiagnosticsReporter: CSV + robustness TXT
│   │   └── plots.py                 # PlotManager: param/objective/validation heatmaps
│   ├── main.py                      # Legacy live-trading loop (still the execution engine)
│   ├── backtest.py                  # Legacy simulation core (still the computation engine)
│   ├── tuner.py                     # Legacy optimizer core (scipy DE, LLM review)
│   ├── source_data.py               # Universe + fundamentals + momentum scoring
│   ├── sentiment_analysis.py        # Batch async + single-stock Claude sentiment
│   ├── sentiments.py                # News/Reddit data collection
│   ├── util.py                      # Config constants, schema, CSV helpers
│   └── tests.py                     # Legacy pure-function unit tests
├── tests/                           # pytest test suite (271 tests, no API required)
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
└── .env                             # Credentials (never commit)
```

The new modular layer (`cli/`, `core/`, `config/`, `strategy/`, `portfolio/`, `execution/`, `backtesting/`, `tuning/`, `reporting/`) provides a typed, testable API over the legacy engine modules. The legacy `.py` files remain as the computation backends and are gradually being hollowed out.

## Scoring Model

### Factor Scores

| Score | What it measures |
|-------|-----------------|
| `value_score` | P/E and P/B cheapness relative to sector thresholds |
| `income_score` | Dividend yield quality (capped at 1.5×; 0 if no yield or yield trap) |
| `quality_score` | Liquidity, earnings existence, dividend health signal |
| `momentum_score` | Multi-factor v2: relative strength, risk-adjusted return, trend structure, short-term momentum |

### Momentum Score v2 — Multi-Factor Continuous Model

All sub-scores are **cross-sectionally percentile-ranked** across the live universe on each day. This is causal (ranking across stocks at one point in time, not across time) and removes the need for asset-specific normalization.

| Sub-factor | Default weight | What it captures |
|---|---|---|
| `rs_3m` | 0.25 | Return_3m − SPY_3m (relative strength, 3-month) |
| `rs_6m` | 0.25 | Return_6m − SPY_6m (relative strength, 6-month) |
| `risk_adj_3m` | 0.20 | return_3m / realized_vol_3m (Sharpe-like, 63-day) |
| `trend_structure` | 0.15 | Price vs 50 DMA and 200 DMA (deterministic signal, not ranked) |
| `return_1m` | 0.10 | Raw 21-day return, percentile-ranked |
| `return_5d` | 0.05 | 5-day short-term check (fixed, not optimizer-tunable) |

**Penalties applied after weighting:**

| Penalty | Trigger |
|---|---|
| Falling-knife | 3m return < −15% |
| Overextension | 52-week position > 97% |
| High volatility | Annualized realized vol > 50% |

**Trend structure scoring** (not ranked — deterministic):

| Signal | Score |
|---|---|
| Above 50 DMA and 200 DMA | +0.50 |
| Above 50 DMA only | +0.10 |
| Above 200 DMA only | −0.10 |
| Below both | −0.50 |

Final momentum score is clamped to [−1.0, 1.5]. All weights are optimizer-tunable under `momentum_v2.weights` in `config.yaml`.

> **Backward compatibility**: The original 5-bin bucket system (`momentum.position_bin_scores`) is retained in config and unit tests. The backtest engine automatically routes to v2 when multi-factor price arrays are available, and falls back to v1 for legacy test fixtures.

### Final Metric Formula

```
value_metric = sw_value    × value_score
             + sw_quality  × quality_score
             + sw_income   × income_score
             + sw_momentum × momentum_score
```

Default weights (from `cfg/config.yaml`):

```yaml
score_weights:
  value:    0.10
  quality:  0.45
  income:   0.10
  momentum: 0.35
```

Weights are YAML-configurable. The optimizer normalizes them internally so they do not need to sum to 1.0 in config, but they are written normalized after a tune run.

### Valuation Guardrails

```
pe_comp = sector_PE / pe_ratio   (only if min_pe_ratio ≤ pe_ratio < sector_PE)
pe_comp = min(pe_comp, max_pe_component)   # default 5.0

pb_comp = sector_PB / pb_ratio   (only if min_pb_ratio ≤ pb_ratio < sector_PB)
pb_comp = min(pb_comp, max_pb_component)   # default 5.0

value_score = value_pe_weight × pe_comp + (1 − value_pe_weight) × pb_comp
              (default: 0.60 × pe_comp + 0.40 × pb_comp)
```

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

When a cap is hit the allocation is **reduced** to the maximum allowed rather than skipped outright. Only if the reduced amount falls below `min_order_amount` is the buy skipped.

## Sell Decision Engine

Each non-ETF holding is evaluated by `evaluate_sell_candidate()` which classifies sells as **hard** or **soft**. ETF positions are explicitly excluded from stock stop-loss logic.

### Hard Sells — execute immediately, sentiment cannot override

| Trigger | Condition |
|---------|-----------|
| Stop loss | `percent_change ≤ stop_loss_pct` (default −20%) |
| Trailing stop | `price / peak_price − 1 ≤ trailing_stop_pct` (default −8%) |
| Yield trap | `yield_trap_flag=True` and `value_metric < sell_weak_value_below` |
| Quality floor | `quality_score < sell_low_quality_below` (default −0.25) |

### Soft Sells — sentiment can hold

| Trigger | Condition |
|---------|-----------|
| Take profit | `percent_change ≥ take_profit_pct` (default +60%) and `value_metric` below floor |
| Weak value | `value_metric < sell_weak_value_below` (default 0.45) and held ≥ `min_days_held_before_value_exit` days |

Soft sells are sent to Claude. A `HOLD` response with confidence ≥ `sell_sentiment_override_confidence` (default 85%, separate from the buy confidence threshold of 65%) keeps the position. Hard sells always execute regardless of sentiment.

### Pending Order Awareness

Before the sell scan, all open sell orders on Robinhood are fetched. Symbols with an existing open sell order are skipped entirely — no double-sell risk.

### Profit Harvesting

Take-profit proceeds are classified as `harvest_exit`. If the run total exceeds `min_harvest_amount` ($25), proceeds are routed to `harvest_etfs` (default: SPY, VTI) rather than sitting as idle cash. Split: `harvest_to_etfs_pct` (80%) to ETFs, remainder recycled for continued stock exposure.

### ETF MA Filter (defensive regime only)

In defensive regime, ETF positions below their MA are also exited (`etf_risk.use_ma_filter: true`, `ma_period: 200`). This is the only mechanism by which ETF positions are sold — they are never touched by the stock stop-loss or trailing-stop logic.

## Market Regime

On each run, `get_market_regime()` classifies the environment into one of three tiers:

| Regime | SPY vs 200 DMA | VIX | Effect |
|--------|---------------|-----|--------|
| **Bullish** | Above | < 20 | Normal operation |
| **Neutral** | Below or VIX ≥ 20 | 20–30 | Base config (no override) |
| **Defensive** | Below | ≥ 30 | ETF allocation raised to 85%, max stock buys = 3, stop-loss tightened by 0.05, ETF MA filter active |

All thresholds are configurable under `regime:` in `config.yaml`.

## Backtest Engine

`backtest.py` simulates the full strategy over historical price data from `yfinance`. Key design choices:

### Anti-Lookahead Causal Features

All rolling features are computed strictly causally — only price data up to day D is used on day D:

| Feature | Lookback | Notes |
|---------|---------|-------|
| `position_52w` | 252 bars | 52-week high/low range position |
| `return_1m` | 21 bars | 1-month price return |
| `return_5d` | 5 bars | 5-day price return |
| `return_3m` | 63 bars | 3-month price return |
| `return_6m` | 126 bars | 6-month price return |
| `realized_vol_3m` | 63 bar daily returns | Annualized std dev |
| `rs_3m`, `rs_6m` | 63 / 126 bars | Return minus benchmark return |
| `above_50dma` | 50 bars | Price vs 50-bar rolling mean |
| `above_200dma` | 200 bars | Price vs 200-bar rolling mean |

Cross-sectional percentile ranking across stocks on a single day is **not** a lookahead bias (it ranks contemporaneous values, not future ones).

### Contribution-Adjusted TWR

Both portfolio and benchmark returns use chain-link time-weighted return methodology. External cash flows (weekly contributions) are stripped from the return computation so reported figures reflect market performance only, not capital growth from deposits. The benchmark invests each weekly contribution into SPY on the same schedule.

### Backtest Realism Controls

| Feature | Config key | Default |
|---------|-----------|---------|
| Volatility-scaled slippage | `vol_slippage_scaling` | `true` — effective_bps = base × (1 + 2.0 × annualized_vol) |
| Max trades per week | `max_trades_per_week` | 10 |
| Post-sell cooldown | `cooldown_days_after_sell` | 3 days |
| Post-stopout cooldown | `cooldown_days_after_stopout` | 7 days |
| Regime classification | uses `benchmark_prices` vs 200 DMA + threshold | tracked daily |

### Backtest Modes

| Mode | Lookahead bias | Universe |
|------|---------------|----------|
| `liquid_universe_sanity_test` | MEDIUM | Random sample from liquid stocks (default) |
| `current_universe_stress_test` | HIGH | Top-N by current `value_metric` — **not predictive** |
| `walk_forward_price_only_test` | LOW | Liquid sample; fundamental arrays zeroed, momentum only |

### Backtest Report Fields

```
Return (TWR):       contribution-adjusted time-weighted return
Bench TWR:          same methodology applied to benchmark (SPY buy-and-hold + contributions)
Benchmark (buy-hold): simple price return over the window
Excess return:      strategy TWR − benchmark price return
Sharpe / Calmar / Max drawdown: from TWR daily series
Stopouts:           hard stop-loss or trailing-stop exits
Cooldown skips:     buys blocked by post-sell cooldown
Regime days:        breakdown of bullish / neutral / defensive days
```

## Parameter Tuner

`tuner.py` uses `scipy.optimize.differential_evolution` to maximize Sharpe or Calmar ratio over a back-simulation window. Tunable parameters:

```
score_weights (value, quality, income, momentum)
index_pct                   — ETF allocation fraction (floor: min_index_pct)
metric_threshold            — minimum score to qualify as a buy candidate
take_profit_pct             — when to harvest gains
sell_weak_value_below       — when to exit on thesis degradation
trailing_stop_pct           — trailing stop distance from peak
value_pe_weight             — PE vs PB split within value score
momentum_v2 sub-weights:
  rs_3m, rs_6m              — relative strength weight (3m, 6m)
  risk_adj_3m               — risk-adjusted momentum weight
  trend_structure           — DMA signal weight
  return_1m                 — raw 1m return weight
```

Safety parameters (stop_loss_pct, position caps, order caps) are never touched by the optimizer.

### Diversification Penalty

The optimizer applies a graduated penalty when `average_positions < 5`, discouraging parameter sets that cherry-pick 1–2 lucky stocks. Combined with the hard trade floor (`_MIN_TRADES_HARD = 20`), this pushes toward genuinely diversified strategies.

### Parameter Stability Reporting

After averaging Sharpe and Calmar runs, the diff table flags any parameter dimension where `|sharpe_opt − calmar_opt| > 0.05`. Unstable dimensions are marked `⚠ unstable` — if a parameter swings wildly between objective functions, the averaged value may be unreliable.

### Validation Gates

Before writing any changes to `config.yaml`:

| Gate | Config key | Default |
|------|-----------|---------|
| Excess return vs benchmark | `min_validation_excess_return` | 0.0% |
| Max drawdown | `max_validation_drawdown` | −20% |
| Sharpe ratio | `min_validation_sharpe` | 0.25 |

All gates must pass. `--apply` with a failed validation prints a warning and does not write config. `--force-apply` bypasses validation for debugging.

### LLM Review (optional)

When `--llm-review` is passed (or `llm_review_enabled: true` in config), all three candidates — Sharpe-optimized, Calmar-optimized, and averaged — are sent to Claude for a second-opinion review. The model returns a recommendation, rationale, and optional alpha-parameter adjustments. Safety parameters are explicitly excluded from the adjustable set. If `llm_review_apply: true`, Claude's adjustments are merged into `config.yaml`.

## Running the Application

After installation (see **Setup** below), all commands are available as `daily-investor`:

```bash
# Full run — refresh data, fetch news, analyze, trade
daily-investor run

# Override operating mode for this run only (does not write config.yaml)
daily-investor run --op-mode safe          # manual confirmation before every trade
daily-investor run --op-mode automated     # fully hands-off
daily-investor run --op-mode no-sentiment  # value_metric weight only, no Claude calls

# Skip data generation — reuse today's cached CSVs (much faster)
daily-investor run --skip-data

# Run a backtest (prints BacktestResult summary)
daily-investor backtest 90
daily-investor backtest 365 --mode walk_forward_price_only_test

# Single-objective tune: print suggested config diff (no file changes)
daily-investor tune 90
daily-investor tune 90 --objective calmar

# Auto-tune: Sharpe + Calmar, train/val split, validate, print diff
daily-investor auto-tune
daily-investor auto-tune 180

# Auto-tune with backtest mode override
daily-investor auto-tune --mode walk_forward_price_only_test

# Write config only if validation passes
daily-investor auto-tune --apply

# Write config regardless of validation (debugging only)
daily-investor auto-tune --force-apply

# Auto-tune + LLM second-opinion review
daily-investor auto-tune --llm-review
daily-investor auto-tune --llm-review --apply

# Parameter stability scan across multiple windows (research only, never writes config)
daily-investor stability-scan
daily-investor stability-scan --mode walk_forward_price_only_test --output-dir reports/

# Quick diagnostics report
daily-investor report
```

The live strategy runs in a loop (up to `max_iterations` runs, default 10). Stocks that were skipped, failed, or already bought are excluded from subsequent iterations.

## Configuration

All settings live in `cfg/config.yaml`.

```yaml
# Fundamental screening
ignore_negative_pe: true
ignore_negative_pb: false
dividend_threshold: 0.03       # Minimum dividend yield (3%)
metric_threshold: 0.75         # Minimum value_metric to qualify as a buy candidate

# Capital allocation
weekly_investment: 400
index_pct: 0.65                # Fraction of investable cash allocated to ETFs
auto_approve: true
use_sentiment_analysis: true
confidence_threshold: 65       # Buy sentiment: minimum Claude confidence (0–100)
sell_sentiment_override_confidence: 85  # Sell override: higher bar to hold vs. sell

# Factor weights
score_weights:
  value:    0.10
  quality:  0.45
  income:   0.10
  momentum: 0.35

# Momentum v2 sub-weights (raw, normalized internally by scorer)
momentum_v2:
  weights:
    rs_3m: 0.25
    rs_6m: 0.25
    risk_adj_3m: 0.20
    trend_structure: 0.15
    return_1m: 0.10
    return_5d: 0.05          # fixed — not optimizer-tunable
  penalties:
    falling_knife_3m_threshold: -0.15
    falling_knife_penalty: 0.25
    overextension_52w_threshold: 0.97
    overextension_penalty: 0.20
    high_vol_annual_threshold: 0.50
    high_vol_penalty: 0.15
  clamp_low: -1.0
  clamp_high: 1.5

# Three-tier market regime
regime:
  spy_ma_period: 200
  vix_defensive_threshold: 30.0   # VIX ≥ this → defensive
  vix_neutral_threshold: 20.0     # VIX ≥ this (but < defensive) → neutral
  defensive:
    index_pct_override: 0.85      # raise ETF allocation
    max_buys_override: 3          # limit active sleeve buys
    stop_loss_tighten: 0.05       # tighten stop_loss by this amount
  neutral:
    index_pct_override: null      # use base config
    max_buys_override: null

# ETF core protection
etf_risk:
  enabled: true
  use_ma_filter: true
  ma_period: 200                  # exit ETF position if price < MA (defensive only)
  defensive_etf_pct: 0.85

# Portfolio risk limits
risk:
  max_single_position_pct: 0.05
  max_sector_pct: 0.25
  max_order_pct_of_cash: 0.10
  min_order_amount: 5.0
  min_liquidity_volume: 500000
  min_index_pct: 0.60             # optimizer floor for ETF allocation
  max_buys_per_rebalance: 10

# Sell rules (all percentages as decimals)
sell_rules:
  stop_loss_pct: -0.20
  trailing_stop_pct: -0.08
  take_profit_pct: 0.60
  sell_weak_value_below: 0.45
  sell_yield_trap: true
  sell_low_quality_below: -0.25
  min_days_held_before_value_exit: 21

# Backtest / tuner settings
backtest:
  default_mode: liquid_universe_sanity_test
  starting_capital: 5000.0
  weekly_contribution: 400.0
  slippage_bps: 10.0
  train_pct: 0.70
  use_out_of_sample_validation: true
  auto_apply_if_valid: false
  min_validation_excess_return: 0.0
  max_validation_drawdown: -0.20
  min_validation_sharpe: 0.25
  use_time_weighted_returns: true
  max_trades_per_week: 10
  cooldown_days_after_sell: 3
  cooldown_days_after_stopout: 7
  vol_slippage_scaling: true
  vol_slippage_multiplier: 2.0
  llm_review_enabled: false      # set true to always run LLM review after auto-tune
  llm_review_apply: false        # set true to let Claude's adjustments write to config
  llm_review_model: claude-sonnet-4-6
```

### Operating Modes

Use `--op-mode` on the CLI to override `auto_approve` and `use_sentiment_analysis` for a single run without touching `config.yaml`. Alternatively, set the values directly in config for a permanent change.

| Mode | `--op-mode` arg | `auto_approve` | `use_sentiment_analysis` | Notes |
|------|-----------------|---------------|--------------------------|-------|
| Safe | `safe` | `false` | `true` | Manual confirmation before each trade |
| Automated | `automated` | `true` | `true` | Executes high-confidence trades automatically |
| No Sentiment | `no-sentiment` | `false` | `false` | Buys by `value_metric` weight only, no Claude API calls |

```bash
daily-investor run --op-mode safe          # one-off safe run
daily-investor run --op-mode no-sentiment  # no API key needed, pure quantitative
```

## Sentiment Analysis Architecture

### Buy Path — Batch Async
1. Pre-filter candidates by `metric_threshold`
2. Cap to `max_sentiment_candidates` (score + buy-to-sell ratio ranked)
3. Dispatch all batches concurrently via `asyncio.gather()` with `Semaphore(MAX_CONCURRENT=5)`
4. Exponential backoff (`2^attempt × (1 + jitter)`) on 429s and transient errors
5. Parse per-symbol results and run through risk controls before placing orders

### Sell Path — Hard/Soft Engine
1. Load all holdings and `agg_data` once; fetch open sell orders (skip symbols with pending sells)
2. Call `evaluate_sell_candidate()` for each non-ETF holding
3. Execute hard sells immediately (no sentiment check)
4. Run ETF MA filter (defensive regime only)
5. Batch soft sell candidates through Claude as a hold-check (confidence ≥ 85% bullish = hold)
6. Aggregate `harvest_exit` proceeds and route to `harvest_etfs`

## Setup

**Requirements:** Python 3.10+, Robinhood account, Anthropic API key

```bash
git clone https://github.com/yourusername/daily_investor.git
cd daily_investor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # installs daily-investor CLI + all dependencies
pip install -e ".[heatmaps]"     # also install matplotlib for stability-scan heatmaps
```

`.env` file (at the project root):
```
RB_ACCT=your_robinhood_email
RB_CREDS=your_robinhood_password
RB_MFA_SECRET=your_totp_secret        # Optional: skip interactive MFA prompt
ANTHROPIC_API_KEY=your_anthropic_key  # Required for sentiment analysis and LLM tune review
```

After installation, `daily-investor` is available on your `PATH`:

```bash
daily-investor --help
```

To run the test suite (no Robinhood or Anthropic credentials needed):

```bash
pytest                           # runs all 271 tests
pytest tests/test_backtesting.py # single module
```

## Troubleshooting

**Inflated value_metrics from old cached CSVs**
The bot always loads the most-recently dated CSV for each dataset. Delete stale files from `data/` or run without `--skip-data` to regenerate.

**"All stocks show NEUTRAL"**
Batch Claude call failed. Check `investment_bot.log` for the stack trace. Common causes: missing `ANTHROPIC_API_KEY`, network issue, or Python < 3.10 event loop incompatibility. Use `--op-mode no-sentiment` to bypass sentiment entirely.

**"Fractional order unavailable, retrying as market order"**
Some tickers (foreign ADRs, low-liquidity stocks) don't support fractional shares on Robinhood. The bot retries with `order_buy_market(symbol, 1)` automatically, with a final risk re-check before placing.

**"Skipping SYMBOL: position cap reached"**
The stock already fills its allowed slice of the portfolio (default 5%). Adjust `max_single_position_pct` in `cfg/config.yaml` if needed.

**"Skipping SYMBOL: sector cap reached"**
A single sector would exceed 25% of portfolio value. Adjust `max_sector_pct` to change the limit.

**Auto-tune returns impossibly high Sharpe**
Check that `use_time_weighted_returns: true` is set in the `backtest` section. The `current_universe_stress_test` mode uses current fundamental scores throughout history (HIGH lookahead bias) and is not predictive — prefer `walk_forward_price_only_test` for the most conservative evaluation.

**"Config NOT written: validation gates failed"**
The tuned parameters did not outperform SPY on the held-out validation window, exceed the Sharpe floor, or stay within the drawdown limit. Use `--force-apply` to override for manual inspection.

**"⚠ unstable" in the diff table**
A parameter's Sharpe-optimized and Calmar-optimized values differ by more than 5%. The averaged value may not be robust. Consider re-running with a longer window or reviewing the flagged parameter manually before applying.

## Security

- Never commit `.env` or any file containing credentials
- All sensitive values are read from environment variables at runtime
- The LLM review payload is sanitized: it never contains account IDs, balances, credentials, or PII — only performance metrics and alpha parameter candidates
- Safety parameters (stop-loss, position caps, order caps) are excluded from the LLM-adjustable parameter set
- `--op-mode` only affects the current process — it never writes to `config.yaml`

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this tool. Always conduct your own research and consider consulting with a licensed financial advisor before making investment decisions.

## License

MIT License — see [LICENSE](LICENSE) for details.
