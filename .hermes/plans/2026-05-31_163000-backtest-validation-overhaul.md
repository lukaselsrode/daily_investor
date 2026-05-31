# Backtest Validation Overhaul
**Date:** 2026-05-31  
**Status:** PLAN — not yet executed

---

## The honest answer to "did we crack alpha?"

**Maybe — but we can't tell yet, because the validation infrastructure has a
fundamental flaw that makes results non-repeatable and non-comparable.**

Here is exactly what happened:

| Substrate | Universe | SPY total | pct_beating |
|-----------|----------|-----------|-------------|
| Pinned 730d (May 29, liquid_sample) | 300 random stocks | +76.5% | **72%** |
| Fresh 730d today (liquid_all) | 300 top-volume stocks | +76.5% | **22%** |

Same SPY path. Same date range. Same params. 50-percentage-point swing from
changing which 300 stocks are in the test. The "66% / 72% yesterday" result
was real — for that specific May 29 random draw. It tells us nothing about
whether the strategy is alpha-generating in general, because:

1. **Universe selection was random and unstable.** `liquid_sample` + `seed=42`
   calls `df.sample()` on an unsorted pool that changes with every bot run.
   247/300 stocks changed between the May29 and May31 runs.

2. **The pinned substrates are biased.** They were created during the research
   sessions on that random draw. All of the "beats SPY 3/5, 4/5" results are
   measured on a hand-optimized universe that was never validated out-of-sample
   on a different universe draw.

3. **The `liquid_all` universe (now the default) genuinely underperforms.**
   22% beating / -1.8% median excess on today's top-300-by-volume is the
   honest current picture. This may mean the alpha was universe-specific, or
   it may mean the liquid_all universe is harder to beat and the strategy needs
   re-tuning on it.

**The strategy may still have real alpha. We just haven't proven it yet on a
universe that can't be cherry-picked.**

---

## Goal

Build a validation infrastructure that can actually answer the question:
*"Does this strategy beat SPY reproducibly, across multiple universes and time
periods, with no cherry-picking?"*

This means:

- A fixed, stable, reproducible test harness that doesn't change with agg_data
- Multi-universe validation (not just one draw)
- Honest separation of in-sample (tuning) and out-of-sample (evaluation)
- Results that are comparable across sessions

---

## Current state / assumptions

- `cfg/config.yaml` `universe_selection: liquid_all` (deterministic, set today)
- Pinned substrates (`.session_tmp/substrate_{730,1400,1550}.pkl`) are all
  `liquid_sample` draws from May 29 — they are frozen artifacts useful for
  reproducing prior results but should not be used as the primary benchmark
- `best_alpha.npy` (48-element vector) was tuned on the 730d pinned substrate
- Gates: `make test` (611 passing), `make hygiene` (4 contracts clean)
- Python path: `PYTHONPATH=src .venv/bin/python`

---

## Proposed approach: three-phase overhaul

### Phase 1 — Build stable, multi-universe OOS substrates (the foundation)

Create new pinned substrates using `liquid_all` (deterministic) across
multiple time spans and re-run the honest baseline. These become the canonical
ground truth for all future validation.

**Why multiple universes?** A strategy that beats SPY on one set of 300 stocks
but not another has universe-specific alpha, not general alpha. We need 3-5
independent universe draws to make a statistical claim.

### Phase 2 — Proper in-sample / out-of-sample separation

The current tuning setup optimizes on a training slice of the same substrate
it validates on. This is soft lookahead — not data leakage, but it means the
"validation" result is correlated with the training data. True OOS requires:
- Tune on one time period
- Evaluate on a *later* time period that was never seen during tuning
- Or tune on universe A, evaluate on universe B

### Phase 3 — Multi-universe robustness score as the real success metric

Replace "pct_beating on one pinned substrate" with a metric that averages
over multiple universe draws. A candidate config that beats SPY on 3+ of 5
independent `liquid_all` universe draws (different random seeds selecting
from the sorted eligible pool) is the honest bar.

---

## Step-by-step plan

### Step 1: Build canonical `liquid_all` substrates and baseline
**Files:** new script `.session_tmp/build_canonical_substrates.py`

1. For seeds [42, 99, 7, 21, 137] and `universe_selection=liquid_all`:
   - Load 730d, 1095d (3yr), 1825d (5yr — if data available) substrates
   - Each seed picks a different 300-stock draw from the sorted eligible pool
   - Pickle to `.session_tmp/canonical/substrate_{n}d_s{seed}.pkl`
2. Run baseline (default params) on each substrate via the standard
   `run_robust_scan(standard, mixed)` harness
3. Report the distribution: median pct_beating, 25th/75th percentile
   — this is the honest baseline the strategy must beat, not the May29 fluke

**Validation:** `# DONE` footer, all 5 seeds complete, results table printed.

### Step 2: Re-tune on the `liquid_all` universe
**Files:** `.session_tmp/liquid_all_tune.py`

The `best_alpha.npy` was tuned on the `liquid_sample` May29 substrate. It
needs to be re-tuned on the new canonical substrates.

1. Load the canonical 730d seed=42 substrate (the new "training universe")
2. Run the `active_alpha_engine` preset via the existing tuning harness
   (same wave structure as prior sessions but on the new universe)
3. Validate on the OTHER 4 seeds (different universe draws of the same period)
   — this is the real OOS test: does a config tuned on one universe draw work
   on a different draw of the same time period?
4. Save the best multi-universe candidate to `.session_tmp/best_alpha_v2.npy`

**Success criterion:** beats SPY on 3+ of 5 universe draws at the standard
rolling-window test. Not just 1 draw.

### Step 3: Extend the temporal horizon
**Files:** `.session_tmp/temporal_oos_test.py`

Even if a config beats SPY on 5 universe draws of the same time period, it
might be period-specific (the bull run 2022-2025 is unusual). We need temporal
OOS:

1. Tune on the oldest 2 years of available data
2. Evaluate on the most recent 1 year (true temporal holdout)
3. Evaluate on a mix of bull/bear periods if data covers them

This is the test the prior sessions could not do because the substrates only
went back to what yfinance had. Need to assess: how far back does the price
history actually go for the current 300-stock universe?

**Files to check:** `backtesting/data_loader.py` — the yfinance call, maximum
look-back.

### Step 4: Fix the UI to reflect the honest situation
**Files:** `src/ui/components/backtests.py`, `src/ui/components/random_windows.py`

1. Add a "Universe info" expander to backtest results showing:
   - Which universe_selection was used
   - How many stocks, first 5 tickers
   - A note that `liquid_all` results are comparable across sessions but the
     universe pool still drifts ~10% between bot runs
2. Add a "Multi-universe robustness" button to the Robust Window Scan page
   that runs the same scan on 3 different universe seeds and shows the
   distribution — this is the "is this real alpha?" button
3. Surface `best_alpha_v2.npy` (once created) as a preset in the UI

---

## Files likely to change

| File | Change |
|------|--------|
| `.session_tmp/build_canonical_substrates.py` | new — build canonical substrates |
| `.session_tmp/liquid_all_tune.py` | new — re-tune on liquid_all |
| `.session_tmp/temporal_oos_test.py` | new — temporal holdout evaluation |
| `.session_tmp/best_alpha_v2.npy` | new — re-tuned params |
| `.session_tmp/canonical/substrate_*.pkl` | new — 5x universe draws |
| `src/ui/components/random_windows.py` | add multi-universe scan button |
| `src/ui/components/backtests.py` | add universe info to results |
| `cfg/config.yaml` | already updated (`liquid_all`) |

**Do NOT touch:**
- `cfg/config.yaml` tuning params (don't apply new params until validated)
- `src/backtesting/simulator.py` (no behavior changes)
- The existing `.session_tmp/substrate_*.pkl` pinned files (keep for reference)

---

## Tests / validation gates

Every script must:
- Print `# DONE` as its final line (footer gate)
- Produce a results table with median/25th/75th over 5 universe draws
- Assert accounting trace reconciles to 0 on at least one substrate

Final bar to call alpha "found":
- `best_alpha_v2` beats SPY (positive excess return AND positive Sharpe delta)
  on 3/5 independent liquid_all universe draws in the standard rolling-window
  test — where "independent" means different 300-stock seeds, same time period
- PLUS beats on a temporal holdout (tune on years 1-2, evaluate on year 3)
- PLUS the improvement survives after subtracting realistic transaction costs
  (already in the simulator at 10bps slippage)

---

## Risks and tradeoffs

| Risk | Mitigation |
|------|-----------|
| liquid_all universe pool drifts ~10% per run (volume rankings change) | Accept as honest market churn; rebuild canonical substrates fresh at start of each research session |
| yfinance data only goes back ~5yr for many tickers; survivorship bias if using current liquid_all for historical runs | Acknowledge limitation; use what we have; don't claim robustness before 2021 |
| Re-tuning on liquid_all may show alpha is universe-specific and doesn't exist | That's the honest answer we need — better to know now |
| The 5-universe-draw bar (3/5 beating) may be too strict given the +76% bull substrate | Agree to also evaluate Sharpe/Calmar vs SPY as the metric (don't just chase raw excess return) |
| Multi-universe scan adds ~5x run time | Use Quick profile for exploration, Standard for conviction |

---

## Open questions

1. **How far back does yfinance price data go for the current liquid_all 300?**
   If we can get 7+ years, we span multiple market regimes including 2022 bear.
   Need to probe this before designing the temporal holdout.

2. **Should we pin the eligible pool (the full sorted list before sampling)
   rather than the final 300?** This would let us reproduce any seed draw from
   a single stored file and reduce the build time for canonical substrates.

3. **Is the active sleeve's underperformance on liquid_all structural or
   tunable?** The liquid_all universe is top-300 by volume — that skews toward
   mega-caps with tight spreads. The prior research was on a random sample that
   included more mid-caps where the scoring factors might have more edge. Worth
   testing: does the strategy beat SPY better on a mid-cap-biased universe?

4. **Should the "multi-universe robustness scan" button be the primary
   validation surface in the UI**, replacing the current single-substrate
   random window scan? That would make the honest test the default, not an
   advanced option.
