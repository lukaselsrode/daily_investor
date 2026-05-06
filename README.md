# Daily Investor

An automated investment strategy tool that combines fundamental analysis with AI-powered sentiment analysis to make informed investment decisions. The system evaluates stocks based on financial metrics, momentum, market sentiment, and news analysis, then executes trades via Robinhood.

## Key Features

- **Factor-Based Scoring**: Combines value (P/E, P/B), income (dividend yield), quality, and 52-week momentum into a single `value_metric`
- **Valuation Guardrails**: Caps P/E and P/B components to prevent extreme scores from thin or stale fundamental data
- **Portfolio Risk Controls**: Per-position cap, per-sector cap, and per-order size cap enforced before every buy
- **Disciplined Sell Engine**: Separates hard sells (stop-loss, yield trap, quality floor) from soft sells (take-profit, weak value) with sentiment override only on soft sells
- **Bear Market Regime Detection**: Suspends new buys and tightens sector exposure when SPY is below its 200-day MA or VIX is elevated
- **Profit Harvesting**: Take-profit proceeds are automatically reinvested into core ETFs rather than sitting as idle cash
- **Backtest Engine**: Simulate the strategy over historical price data with realistic slippage, weekly contributions, and time-weighted returns
- **Validation-Aware Auto-Tune**: Optimizes score weights and sell thresholds via `scipy.differential_evolution`; tuned parameters are only written to config if they pass held-out validation gates
- **Optional LLM Tune Review**: Routes top optimizer candidates through the Claude API for a second-opinion review before applying
- **Batch AI Sentiment Analysis**: Analyzes multiple stocks per Claude API call using async concurrency
- **ETF Dollar-Cost Averaging**: Periodic allocation to a configurable set of ETFs (default: SPY, VOO, VTI, QQQ, SCHD, SMH, VXUS, VNQ, IWM)
- **Async + Exponential Backoff**: Concurrent Claude calls with automatic retry on rate limits
- **Fractional + Whole Share Fallback**: Attempts fractional orders first; falls back to a whole-share market order

## Project Structure

```
daily_investor/
├── cfg/
│   └── config.yaml            # All tunable parameters (never commit credentials here)
├── data/                      # CSV cache (dated filenames, newest always used)
├── src/
│   ├── main.py                # Entry point: login, buy/sell loops, CLI dispatcher
│   ├── backtest.py            # Simulation engine: price history, TWR metrics, reports
│   ├── tuner.py               # Parameter optimizer: scipy DE, validation gating, LLM review
│   ├── sentiment_analysis.py  # Batch async + single-stock Claude sentiment
│   ├── sentiments.py          # News/Reddit data collection
│   ├── source_data.py         # Universe generation, fundamentals, scoring
│   ├── util.py                # Config constants, schema, CSV helpers
│   └── tests.py               # Pure-function unit tests (no API required)
└── .env                       # Credentials (never commit)
```

## Scoring Model

### Factor Scores

| Score | What it measures |
|-------|-----------------|
| `value_score` | P/E and P/B cheapness relative to sector thresholds |
| `income_score` | Dividend yield quality (capped at 1.5×; 0 if no yield or yield trap) |
| `quality_score` | Liquidity, earnings existence, dividend health signal |
| `momentum_score` | 52-week price-location bin + 1-month return recovery/falling-knife adjustments |

### Momentum Score — 52-Week Position Bins

`position_52w = (current_price − 52w_low) / (52w_high − 52w_low)`, clamped to [0, 1].

| position_52w | Default bin score | Signal |
|---|---|---|
| < 0.15 | −0.35 | Possible falling knife |
| 0.15 – 0.35 | −0.10 | Beaten down, not dead |
| 0.35 – 0.75 | +0.55 | Healthy mid/upper range |
| 0.75 – 0.95 | +0.85 | Strong momentum |
| > 0.95 | +0.45 | Near 52w high, possible extension |
| missing data | 0.00 | No signal |

A 1-month return recovery bonus (+0.15) applies when the stock is in the bottom quartile but rebounding. A falling-knife penalty (−0.20) applies when it is in the bottom quartile and still declining. Bin boundaries and scores are optimizer-tunable parameters.

### Final Metric Formula

```
value_metric = sw_value    × value_score
             + sw_quality  × quality_score
             + sw_income   × income_score
             + sw_momentum × momentum_score
```

Default weights (from `cfg/config.yaml`):

```
value:    0.10
quality:  0.45
income:   0.10
momentum: 0.35
```

Weights are YAML-configurable under `score_weights`. The optimizer normalizes them internally so they do not need to sum to 1.0 in config, but they are written normalized after a tune run.

### Valuation Guardrails

Before computing `value_score`, each component is gated and capped:

```
pe_comp = sector_PE / pe_ratio   (only if min_pe_ratio ≤ pe_ratio < sector_PE)
pe_comp = min(pe_comp, max_pe_component)   # default 5.0

pb_comp = sector_PB / pb_ratio   (only if min_pb_ratio ≤ pb_ratio < sector_PB)
pb_comp = min(pb_comp, max_pb_component)   # default 5.0

value_score = value_pe_weight × pe_comp + (1 − value_pe_weight) × pb_comp
              (default: 0.60 × pe_comp + 0.40 × pb_comp)
```

This prevents extreme `value_metric` values from thin or suspicious fundamentals (e.g. PE=0.05 would otherwise produce scores in the hundreds).

### Agg Data Schema

Every scored stock row contains:

```
symbol, industry, sector, volume,
pe_ratio, pb_ratio, dividend_yield,
current_price, low_52w, high_52w, position_52w, return_1m,
pe_comp, pb_comp,
value_score, income_score, quality_score, momentum_score,
yield_trap_flag, value_metric, buy_to_sell_ratio
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

When a cap is hit, the allocation is **reduced** to the maximum allowed rather than skipped outright. Only if the reduced amount falls below `min_order_amount` is the buy skipped. Every cap decision is logged.

## Sell Decision Engine

Each non-ETF holding is evaluated by `evaluate_sell_candidate()` which classifies sells as **hard** or **soft**:

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

If sentiment returns `YES` with confidence ≥ `confidence_threshold`, a soft sell is held. Hard sells always execute regardless of sentiment.

### Profit Harvesting

When a take-profit sell is executed, proceeds are classified as a `harvest_exit`. If the total harvest amount for the run exceeds `min_harvest_amount` ($25), the proceeds are routed to `harvest_etfs` (default: SPY, VTI) rather than sitting as idle cash. The split is controlled by `harvest_to_etfs_pct` (default 80%) with the remainder recycled for continued stock exposure.

## Bear Market Regime

On each run, `_is_bear_market_regime()` checks:
1. SPY closing price vs its 200-day moving average
2. VIX spot level vs `vix_threshold` (default 25.0)

If either condition triggers, the buy loop uses a reduced candidate pool and tighter sector limits. No new stock buys are made in a confirmed bear regime; only ETF contributions continue.

## Backtest Engine

`backtest.py` simulates the full strategy over historical price data downloaded via `yfinance`. Key design choices:

- **Time-Weighted Return (TWR)**: Metrics are computed on a contribution-adjusted daily value series that strips external cash flows. This means reported returns reflect market performance, not the raw growth of deposited capital.
- **Dynamic per-rebalance scoring**: 52-week position and 1-month return features are precomputed as `(n_days, n_stocks)` rolling arrays. Scores are refreshed at every rebalance using only price history available up to that day — no lookahead bias from fundamentals.
- **Realistic friction**: Configurable slippage in basis points and per-trade commission are applied on every buy and sell.
- **Train / validation split**: The price window is split 70/30 (configurable). The optimizer sees only the train window; tuned parameters are evaluated on the held-out validation window before any config changes are written.

### Backtest Modes

| Mode | Lookahead bias | Universe |
|------|---------------|----------|
| `liquid_universe_sanity_test` | MEDIUM | Random sample from liquid stocks |
| `current_universe_stress_test` | HIGH | Top-N by current `value_metric` (not predictive) |
| `walk_forward_price_only_test` | LOW | Liquid sample; fundamental arrays zeroed, momentum only |

### Backtest Report Fields

```
Return (TWR):   time-weighted return (excludes contributions)
Benchmark:      SPY buy-and-hold over the same window
Excess return:  strategy TWR − benchmark return
Sharpe / Calmar / Max drawdown: from TWR daily series
Final value:    total portfolio value (includes all contributions)
net_contributions: starting_capital + all weekly deposits
profit:         final_value − net_contributions
```

## Parameter Tuner

`tuner.py` uses `scipy.optimize.differential_evolution` to maximize Sharpe or Calmar ratio over a back-simulation window. Tunable parameters:

```
score_weights (value, quality, income, momentum)
index_pct              — ETF allocation fraction (floor: min_index_pct)
metric_threshold       — minimum score to qualify as a buy candidate
take_profit_pct        — when to harvest gains
sell_weak_value_below  — when to exit on thesis degradation
trailing_stop_pct      — trailing stop distance from peak
value_pe_weight        — PE vs PB split within value score
momentum bin scores[0..4]
```

Safety parameters (stop_loss_pct, position caps, order caps) are never touched by the optimizer.

### Validation Gates

Before writing any changes to `config.yaml`, the optimizer checks the held-out validation window:

| Gate | Config key | Default |
|------|-----------|---------|
| Excess return vs benchmark | `min_validation_excess_return` | 0.0% |
| Max drawdown | `max_validation_drawdown` | −20% |
| Sharpe ratio | `min_validation_sharpe` | 0.25 |

All gates must pass. `--apply` with a failed validation prints a warning and does not write config. `--force-apply` bypasses validation for debugging.

### Turnover Penalty

The optimizer penalizes parameter sets that produce more than 80 new position entries per window: `penalty = max(0, trades − 80) / 80`. This discourages aggressive churn strategies that would not survive realistic transaction costs.

### LLM Review (optional)

When `llm_review_enabled: true`, the top-N tuning candidates are sent to the Claude API for a second-opinion review. The LLM may recommend one candidate as-is or propose minor adjustments to alpha parameters. Safety parameters are explicitly excluded from the allowed adjustment set and are validated before any merge.

## Running the Application

```bash
# Full run — refresh data, fetch news, analyze, trade
python src/main.py

# Skip data generation — reuse today's cached CSVs (much faster)
python src/main.py --skip-data

# Single-objective tune: print suggested config diff (no file changes)
python src/main.py --tune 90
python src/main.py --tune 90 --objective calmar

# Auto-tune: Sharpe + Calmar, train/val split, validate, print diff
python src/main.py --auto-tune
python src/main.py --auto-tune 180

# Auto-tune with backtest mode override
python src/main.py --auto-tune --mode walk_forward_price_only_test

# Write config only if validation passes
python src/main.py --auto-tune --apply

# Write config regardless of validation (debugging only)
python src/main.py --auto-tune --force-apply

# Run pure-function unit tests (no Robinhood or Claude API required)
python src/tests.py
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
confidence_threshold: 65       # Minimum Claude confidence (0–100) to hold a soft sell

# Factor weights
score_weights:
  value:    0.10
  quality:  0.45
  income:   0.10
  momentum: 0.35

# Portfolio risk limits
risk:
  max_single_position_pct: 0.05
  max_sector_pct: 0.25
  max_order_pct_of_cash: 0.10
  min_order_amount: 5.0
  min_liquidity_volume: 500000
  min_index_pct: 0.60           # Optimizer floor for ETF allocation

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
  llm_review_enabled: false
  llm_review_model: claude-sonnet-4-6
```

### Operating Modes

| Mode | `auto_approve` | `use_sentiment_analysis` | Notes |
|------|---------------|--------------------------|-------|
| Safe | `false` | `true` | Manual confirmation before each trade |
| Automated | `true` | `true` | Executes high-confidence trades automatically |
| No Sentiment | `false` | `false` | Buys by `value_metric` weight only |

## Sentiment Analysis Architecture

### Buy Path — Batch Async
1. Pre-filter candidates by `metric_threshold`
2. Build batches of stocks from cached CSV data
3. Dispatch all batches concurrently via `asyncio.gather()` with `Semaphore(MAX_CONCURRENT=5)`
4. Exponential backoff (`2^attempt × (1 + jitter)`) on 429s and transient errors
5. Parse per-symbol results and run through risk controls before placing orders

### Sell Path — Hard/Soft Engine
1. Load all holdings and `agg_data` once
2. Call `evaluate_sell_candidate()` for each non-ETF holding
3. Execute hard sells immediately (no sentiment check)
4. Batch soft sell candidates through Claude as a hold-check
5. Sentiment `YES` with sufficient confidence holds the position; otherwise sell executes
6. Aggregate `harvest_exit` proceeds and route to `harvest_etfs`

## Setup

**Requirements:** Python 3.10+, Robinhood account, Anthropic API key

```bash
git clone https://github.com/yourusername/daily_investor.git
cd daily_investor
pip install -r requirements.txt
```

`.env` file:
```
RB_ACCT=your_robinhood_email
RB_CREDS=your_robinhood_password
RB_MFA_SECRET=your_totp_secret        # Optional: skip interactive MFA prompt
ANTHROPIC_API_KEY=your_anthropic_key  # Required for sentiment analysis and LLM tune review
```

**Additional dependency for the optimizer:**
```bash
pip install scipy
```

## Troubleshooting

**Inflated value_metrics from old cached CSVs**
The bot always loads the most-recently dated CSV for each dataset. Delete stale files from `data/` or run without `--skip-data` to regenerate.

**"All stocks show NEUTRAL"**
Batch Claude call failed. Check `investment_bot.log` for the stack trace. Common causes: missing `ANTHROPIC_API_KEY`, network issue, or Python < 3.10 event loop incompatibility.

**"Fractional order unavailable, retrying as market order"**
Some tickers (foreign ADRs, low-liquidity stocks) don't support fractional shares on Robinhood. The bot retries with `order_buy_market(symbol, 1)` automatically.

**"Skipping SYMBOL: position cap reached"**
The stock already fills its allowed slice of the portfolio (default 5%). Adjust `max_single_position_pct` in `cfg/config.yaml` if needed.

**"Skipping SYMBOL: sector cap reached"**
A single sector would exceed 25% of portfolio value. Adjust `max_sector_pct` to change the limit.

**Auto-tune returns impossibly high Sharpe**
Check that `use_time_weighted_returns: true` is set in the `backtest` section and that the mode is `walk_forward_price_only_test` for the most conservative (lowest lookahead bias) evaluation. The `current_universe_stress_test` mode uses current fundamental scores throughout history (HIGH lookahead bias) and is not predictive.

**"Config NOT written: validation gates failed"**
The tuned parameters did not outperform SPY on the held-out validation window, exceed the Sharpe floor, or stay within the drawdown limit. Use `--force-apply` to override for manual inspection.

## Security

- Never commit `.env` or any file containing credentials
- All sensitive values are read from environment variables at runtime
- The LLM review payload is sanitized: it never contains account IDs, balances, credentials, or PII — only performance metrics and alpha parameter candidates
- Safety parameters (stop-loss, position caps, order caps) are excluded from the LLM-adjustable parameter set

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this tool. Always conduct your own research and consider consulting with a licensed financial advisor before making investment decisions.

## License

MIT License — see [LICENSE](LICENSE) for details.
