# Portfolio-Construction & Exit/Holding Techniques for a Long-Only Equity Sleeve
## Literature synthesis — turning a FIXED ranking into better realized P&L

Scope: selection signal held constant. Everything below is testable on a daily
(n_days, n_stocks) price panel vs SPY, no options, no intraday, no lookahead.
Context: concentrated (10-30 name) long-only Robinhood sleeve with regime
detection (SPY vs 200DMA), trailing stops, take-profit, and a defensive cash/index
sleeve. Known prior finding: the TRIM mechanism cuts winners and is a hidden
alpha-killer (consistent with the Kaminski-Lo result below).

KEY TAKEAWAY UP FRONT — the "HELP Sharpe / HURT raw return" flag:
  Techniques 3 (vol-targeting), 4c (inverse-vol weighting), 5 (crash protection),
  and 6 (gradual scaling) all tend to RAISE Sharpe/Calmar and CUT drawdown while
  LOWERING raw CAGR in sustained bull markets — because they pull net exposure
  below 100% exactly when the market keeps rising. They are the right product for
  a defensive book that cannot beat SPY on raw return in a bull run.
  Technique 2 (stops/take-profit/trims) is the dangerous one: on a momentum/trend
  signal it usually HURTS BOTH return AND Sharpe by truncating the right tail.

================================================================================
(1) HOLDING PERIOD / REBALANCE FREQUENCY — turnover vs alpha-decay
================================================================================

Core tradeoff: alpha decays over the holding horizon; trading to refresh the
portfolio toward the freshest signal recaptures decayed alpha but pays cost +
spread + (for you) the risk of churning winners out. The optimal rebalance
frequency is where MARGINAL recaptured alpha = MARGINAL cost.

Empirical anchors:
- Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs"
  (RFS). Net-of-cost survival depends almost entirely on turnover. Low-turnover
  signals (value, profitability, ~annual) survive costs comfortably; high-turnover
  signals (short-term reversal, monthly momentum legs) can have their entire
  premium eaten. The practical lever is not the signal but the REBALANCE CADENCE.
- Jegadeesh & Titman (1993): classic 12-1 momentum is formed on 12m and held
  3-12 months. The 6-month formation / 6-month hold (6-6) is the canonical sweet
  spot. Holding < 1 month invites reversal; the well-known 1-month skip avoids
  short-term reversal contamination.
- Frazzini, Israel & Moskowitz (2012/2018), "Trading Costs of Asset Pricing
  Anomalies": real-world AQR execution costs are far lower than academic
  stylized costs, and strategies stay profitable net of cost AT INSTITUTIONAL
  scale precisely because they slow trading and trade patiently. At Robinhood
  retail scale spread/impact is ~0 for liquid large-caps, so your dominant
  "cost" of rebalancing is ALPHA TRUNCATION (churning winners), not commission.

Precise rules to test:
- Sweep rebalance interval R ∈ {5, 10, 21, 42, 63 trading days} (weekly→quarterly).
  Plot net CAGR, Sharpe, turnover, and avg holding period vs R.
- "No-trade band" / hysteresis: only replace a held name when a candidate
  out-ranks it by a margin δ (e.g. must beat the held name's rank by >K slots or
  >x% signal). This is the single highest-ROI turnover control — it kills churn
  without changing the signal. Sweep δ.
- Staggered/overlapping tranches (Jegadeesh-Titman construction): hold N
  overlapping cohorts each rebalanced 1/N of the way. Smooths turnover and
  timing luck; lowers variance of realized return without lowering mean.

Effect: return ↑ then ↓ as R falls (too-frequent churns winners and recaptures
little); Sharpe peaks at a moderate R; drawdown largely unaffected by R alone.
Horizon: momentum signals want 1-6 month holds; reversal/short-term want days.
Regime: alpha decays FASTER in high-vol/turbulent regimes — shorter holds help
there, longer holds in calm trends. Most impactful for CONCENTRATED books: a
no-trade band matters MORE for 10-30 names because each forced swap is a large
fraction of the book and a single churned winner is a big realized-P&L hit.

================================================================================
(2) TRAILING-STOP / TAKE-PROFIT / TIME-STOP EXITS — help or hurt?
================================================================================

This is the most important section for your repo given the TRIM finding.

Headline evidence — stops CUT WINNERS on trend/momentum:
- Kaminski & Lo (2014), "When Do Stop-Loss Rules Stop Losses?" (J. Financial
  Markets). A stop-loss adds value ONLY when returns have NEGATIVE serial
  correlation at the stop horizon (mean-reversion / momentum-in-losses), i.e.
  when being out avoids a continuation DOWN. The "stopping premium" is positive
  for momentum-type return processes at the index/asset-class level (where a
  down move predicts further down moves — fat left tail). It is NEGATIVE or zero
  for random-walk / positively-mean-reverting single names, where a stop locks
  in a loss right before the bounce. CRUCIAL NUANCE for you: their positive
  result is for stopping into cash during PERSISTENT DOWN-trends (regime exit),
  NOT for trimming individual WINNERS. A take-profit/trim that sells a name that
  is trending UP truncates the exact right tail that momentum strategies live on.
- This is precisely your TRIM finding: ~110 trims/window cutting winners is a
  textbook right-tail amputation. Momentum P&L is driven by a few big winners
  (positive skew of the WINNER leg); systematically harvesting them destroys the
  skew that pays for the strategy. Disabling it lifting win-rate but being mixed
  OOS is also textbook: stops/trims RAISE win-rate (more small wins) while
  LOWERING expectancy (you give up the rare huge winners). Win-rate is the wrong
  objective; expectancy/skew is the right one.

Where stops/exits DO help:
- As a PORTFOLIO/REGIME-LEVEL trailing stop (de-risk the whole book or the index
  sleeve when SPY breaks its trend), consistent with Kaminski-Lo's asset-level
  stopping premium and with time-series momentum (Moskowitz, Ooi & Pedersen
  2012, "Time Series Momentum", JFE). A 200DMA / trailing-peak rule on SPY that
  scales the defensive sleeve is the GOOD use of a stop. Your existing regime
  detection is the right place for stop logic.
- Time-stops (exit after H days regardless) are gentler than price-stops: they
  don't condition on the drawdown, so they don't systematically sell into
  weakness. A time-stop is really just a holding-period cap (see section 1).

Precise rules to test (and what to expect):
- Per-name trailing stop at x% off peak: expect win-rate ↑, CAGR ↓, Sharpe flat
  or ↓, drawdown slightly ↓. Net usually NEGATIVE for a momentum signal. TEST IT
  as the null to beat, not as an improvement.
- Take-profit / harvest at +y%: expect strong win-rate ↑ but CAGR ↓↓ via right-
  tail truncation. Almost always HURTS a trend book. This is your TRIM.
- Time-stop at H days: roughly neutral; equivalent to holding-period tuning.
- Portfolio trailing stop / regime exit (SPY vs 200DMA or trailing peak): this
  is the exit that HELPS — cuts drawdown materially, lowers raw CAGR in bulls,
  raises Calmar. Keep exits at the BOOK level, not the name level.

Regime: name-level stops are least harmful in mean-reverting/choppy regimes and
most harmful in strong trends. Book-level stops help most around regime breaks.
Concentrated books: name-level stops are MORE damaging here — fewer names means
each truncated winner is a larger share of total P&L; the positive skew you rely
on is carried by 2-3 names. Keep the concentrated book's winners running; do
de-risking at the sleeve level.

================================================================================
(3) VOLATILITY TARGETING / VOL-SCALING OF OVERALL EXPOSURE
================================================================================

- Moreira & Muir (2017), "Volatility-Managed Portfolios" (J. Finance). Scale
  exposure INVERSELY to recent realized variance: w_t = c * σ_target² / σ̂_t²
  (or σ_target/σ̂_t for vol rather than variance scaling). Applied to the market
  and to factor portfolios, this RAISES Sharpe (~by 15-25% for the market
  factor) and produces large positive alphas against the unmanaged factor,
  because volatility is persistent (forecastable) while expected returns do NOT
  rise one-for-one with vol in the short run. So you cut exposure when vol is
  high (and forward returns are not proportionally higher) and lever up when vol
  is low.
- IMPORTANT CAVEATS / the raw-return flag:
  * Without leverage (your case — long-only, no margin), you can only scale DOWN
    from 100%, parking the rest in cash/index. That CAPS upside: in a low-vol
    bull you stay ~100% at best, in high-vol you sit in cash. Net effect for an
    unlevered book: Sharpe ↑, max drawdown ↓↓, but raw CAGR ↓ in bull markets.
    THIS IS A "HELPS SHARPE, LOWERS RAW RETURN" technique — squarely your product.
  * Cederburg, O'Doherty, Wang & Yan (2020, "On the Performance of Volatility-
    Managed Portfolios", JFE) show the Moreira-Muir gains are concentrated in the
    MARKET factor and weaken/disappear out-of-sample for many other factors and
    under realistic constraints — so treat the magnitude as regime/sample
    dependent, but the DRAWDOWN reduction is robust.
  * Harvey et al. (2018), "The Impact of Volatility Targeting" (J. Portfolio
    Management): vol-targeting reliably improves Sharpe and tames drawdowns for
    RISK-ASSET / equity exposures (where vol and returns are negatively related —
    the "leverage effect" and vol clustering around crashes), but does little for
    assets without that vol-return asymmetry. Equities have it → vol-targeting
    helps. The mechanism is mostly DRAWDOWN/tail control, not return enhancement.

Precise rules to test:
- Estimate σ̂_t from trailing 20-60d realized vol (daily returns, annualized).
  Set net equity exposure E_t = clip(σ_target/σ̂_t, 0, 1); remainder → cash or
  short-duration index sleeve. Sweep σ_target (e.g. 10-20% annualized) and the
  vol window. Use only data up to t-1 (no lookahead — lag the vol estimate 1 day).
- Variance vs vol scaling: variance (σ²) scaling is more aggressive (cuts harder
  in high vol) → bigger drawdown reduction, bigger CAGR give-up.
- Apply at the BOOK level (overall exposure) — this is cleaner and better-
  supported than per-name vol scaling, and it's exactly your "index/cash sleeve
  scales up defensively" mechanism. Vol-target IS the principled rule for sizing
  that sleeve.

Effect: Sharpe ↑ (robust for equity/market exposure), max drawdown ↓↓, raw CAGR
↓ in bulls (unlevered cap). Horizon: works daily; vol is forecastable at 1-20d.
Regime: biggest benefit entering high-vol/crash regimes; a drag in calm bulls.
Concentrated vs broad: applies identically at book level either way; if anything
a concentrated book has higher idiosyncratic vol, so book-level vol-targeting is
even more stabilizing.

================================================================================
(4) POSITION SIZING — equal vs rank vs inverse-vol (vol-parity)
================================================================================

(4a) EQUAL WEIGHT (1/N).
- DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive Diversification"
  (RFS): 1/N beats sample-based mean-variance optimization out-of-sample because
  optimizers amplify estimation error. For a fixed selected set, 1/N is a strong,
  hard-to-beat baseline. It implicitly tilts to smaller/higher-vol names (vs
  cap-weight) and needs periodic rebalancing back to equal (a small mean-reversion
  bonus). Best default for a concentrated book.

(4b) RANK / SIGNAL WEIGHT.
- Weight ∝ rank or ∝ signal strength concentrates capital in the highest-
  conviction names. RAISES raw return IF the signal's top decile genuinely
  outperforms monotonically, but RAISES concentration risk and drawdown and is
  more exposed to single-name blowups. In a 10-30 name book this can tip you into
  dangerous concentration. Expect CAGR ↑ (if signal is good), Sharpe ambiguous,
  drawdown ↑.

(4c) INVERSE-VOL / VOLATILITY PARITY (risk parity-lite).
- Weight_i ∝ 1/σ̂_i (each name contributes ~equal risk; ignores correlations =
  "naive risk parity"). Full risk parity equalizes marginal risk contributions
  using the covariance matrix, but the covariance estimate is noisy for 10-30
  names — naive inverse-vol is the robust, estimation-light version and usually
  the better choice at this breadth.
- Evidence: inverse-vol / risk-parity weighting reliably RAISES Sharpe and CUTS
  drawdown vs equal-weight by down-weighting the most volatile (often the most
  crash-prone) names; it tends to LOWER raw return because high-vol names carry
  some of the return premium (and momentum winners are often high-vol). Asness,
  Frazzini & Pedersen (2012), "Leverage Aversion and Risk Parity"; the general
  low-vol/betting-against-beta literature (Frazzini-Pedersen 2014) supports the
  risk-adjusted-return improvement from tilting away from high-vol/high-beta.
- ANOTHER "HELPS SHARPE, LOWERS RAW RETURN" lever in many samples.

Precise rules to test:
- Compare four weightings on the SAME selected set each rebalance: 1/N,
  rank-weight, signal-weight, inverse-vol (σ̂ from trailing 20-60d, lagged 1d).
  Add a vol cap per name (max weight) to prevent any single high-vol name
  dominating. Report CAGR/Sharpe/Calmar/maxDD/turnover for each.
- Hybrid that often wins: inverse-vol WITHIN the selected set + book-level
  vol-target on top (section 3). Risk-balances names AND the whole sleeve.

Effect summary:
  1/N        : strong baseline; balanced.
  rank/signal: CAGR ↑, drawdown ↑, Sharpe ambiguous, concentration risk ↑.
  inverse-vol: Sharpe ↑, drawdown ↓, CAGR often ↓.
Concentrated books: weighting matters MUCH MORE here than in a broad book —
with 500 names weights wash out; with 15 names the weighting scheme is a primary
driver of risk. Inverse-vol + a per-name cap is the highest-value change for a
concentrated, drawdown-sensitive book.

================================================================================
(5) MOMENTUM-CRASH PROTECTION / DYNAMIC EXPOSURE
================================================================================

The single most valuable risk-adjusted-return literature for a defensive book.

- Daniel & Moskowitz (2016), "Momentum Crashes" (JFE). Momentum has rare,
  severe crashes — in PANIC states (after large market declines, when market vol
  is high AND the market is rebounding), the past-loser leg roars back and
  momentum suffers huge, predictable losses with strongly negative skew. The
  crashes are partially FORECASTABLE from (a) bear-market state (market below its
  past-2yr level / down regime) and (b) high ex-ante volatility. They build a
  "dynamic momentum" that scales exposure by forecast return/variance and
  roughly DOUBLES the Sharpe of static momentum by avoiding crashes. For a
  LONG-ONLY book the crash is milder (it's mostly the short leg that explodes)
  but the long winners still draw down hard in the rebound — so de-risking in
  panic states still helps drawdown/Calmar.
- Barroso & Santa-Clara (2015), "Momentum Has Its Moments" (JFE). Constant-
  volatility scaling of the momentum factor: w_t = σ_target/σ̂_t with σ̂ from
  trailing 6-month DAILY realized vol of the strategy. This "risk-managed
  momentum" nearly DOUBLES the Sharpe (≈0.5 → ≈1.0+), roughly HALVES the worst
  drawdowns, and removes the crash-driven negative skew/excess kurtosis — and it
  does so WITHOUT reducing average return much because momentum's own vol spikes
  precede the crashes. This is the cleanest, most testable rule here and the most
  directly relevant: scale the WHOLE sleeve by its own trailing realized vol.
  Barroso-Santa-Clara is essentially section-3 vol-targeting applied to the
  strategy's own returns and is often PREFERRED to forecasting because it needs
  only the strategy's past vol (no lookahead, trivial on a daily panel).

Precise rules to test:
- Barroso-Santa-Clara on YOUR sleeve: compute trailing ~126d realized vol of the
  sleeve's daily returns (lagged 1d), scale net exposure to a constant target,
  park remainder in cash/index. Expect: Sharpe ↑↑, maxDD ↓↓, skew less negative,
  CAGR roughly flat-to-down (down in bulls because you cap at 100% unlevered).
- Daniel-Moskowitz regime gate: when SPY is in a down/panic state (below 200DMA
  AND realized vol elevated), cut equity exposure toward the cash/index sleeve.
  This is your existing regime detector used as a crash filter. Expect big
  drawdown/Calmar improvement, lower CAGR in V-shaped recoveries (the cost: you
  de-risk and miss the snap-back — a known weakness; soften with gradual re-entry,
  section 6).
- Combine: vol-scaling (continuous) + regime gate (discrete) tends to dominate
  either alone.

Regime: ALL the benefit is concentrated in bear/panic/high-vol regimes; in calm
bulls these rules are a pure drag on raw return (the flag again). Concentrated
books: long-only concentrated books crash via correlated drawdowns of high-beta
winners — book-level vol-scaling + regime gating is the main defense; per-name
stops are NOT (section 2).

================================================================================
(6) ENTRY TIMING — gradual scaling vs all-at-once
================================================================================

- All-at-once (target weights immediately) is unbiased and avoids being out of
  the market, but maximizes timing luck and entry-point variance, and (for a
  defensive book re-entering after a regime gate) can buy right before a relapse.
- Gradual scaling / dollar-cost-averaging into target weights over k days reduces
  the VARIANCE of the entry price and timing-luck, smooths turnover, and lowers
  realized drawdown of the entry, at the cost of expected return (you're under-
  invested while scaling, and markets drift up on average → DCA underperforms
  lump-sum in expectation; Vanguard 2012/2023 "Cost averaging: invest now or
  temporarily hold your cash?" finds lump-sum beats DCA ~2/3 of the time on raw
  return, but DCA cuts entry drawdown/regret). Classic "HELPS drawdown/variance,
  LOWERS expected return."
- Most useful as RE-ENTRY logic after a regime gate / vol-scale-down: scale back
  IN gradually as vol normalizes / SPY reclaims its trend, to avoid whipsaw of
  de-risk→relapse→re-risk. Pairs with section 5 to soften the "miss the snap-back"
  cost.
- Overlapping tranches (section 1) are the portfolio analog of gradual entry and
  give the variance-reduction with less expected-return drag than pure DCA.

Precise rules to test:
- Scale to target over k ∈ {1(=lump), 3, 5, 10} days; sweep k. Report entry-window
  drawdown, turnover, CAGR, Sharpe.
- Asymmetric speed: re-enter SLOW (gradual, k large) after de-risking, but
  de-risk FAST (k=1) on regime breaks. This asymmetry is the practically best
  config for a defensive book — quick to protect, patient to redeploy.

Effect: variance/drawdown of entries ↓, expected/raw return slightly ↓, Sharpe ~
flat-to-up. Regime: gradual re-entry matters most after vol spikes / regime
flips. Concentrated books: gradual entry matters more (each name is a big slug),
so scaling reduces single-entry timing risk meaningfully.

================================================================================
CONSOLIDATED "HELPS SHARPE / LOWERS RAW RETURN IN BULLS" FLAG TABLE
================================================================================
Technique                         CAGR(bull)  Sharpe  MaxDD/Calmar  Priority(concentrated)
(1) no-trade band / slower rebal    ~flat/↑     ↑       ~flat         HIGH (stops churn)
(2a) per-name trailing stop          ↓          ↓/flat   slight ↓     AVOID (cuts winners)
(2b) take-profit / TRIM harvest      ↓↓         ↓       slight ↓      AVOID (your alpha-killer)
(2c) book-level/regime stop          ↓          ↑       ↓↓           HIGH (right level for exits)
(3) vol-targeting (book)             ↓          ↑       ↓↓           HIGH ★ flag
(4a) equal-weight 1/N                base       base    base         baseline
(4b) rank/signal weight              ↑          ?       ↑            optional, risk ↑
(4c) inverse-vol weighting           ↓          ↑       ↓            HIGH ★ flag
(5) Barroso-Santa-Clara vol-scale    ↓/flat     ↑↑      ↓↓           HIGHEST ★ flag
(5) Daniel-Moskowitz regime gate     ↓          ↑       ↓↓           HIGH ★ flag
(6) gradual re-entry / DCA           ↓          flat/↑  ↓            MEDIUM ★ flag
★ = "helps Sharpe/Calmar/drawdown, lowers raw return in a bull" — the user's product.

================================================================================
RECOMMENDED TEST ORDER (highest expected value for THIS repo)
================================================================================
1. Quantify the TRIM/take-profit damage explicitly: ablate (2b) and report the
   change in right-tail (top-5 winner contribution), skew, expectancy — confirm
   it cuts winners (Kaminski-Lo / Daniel-Moskowitz skew logic). Make it default-OFF.
2. Barroso-Santa-Clara sleeve vol-scaling (5) — biggest, cleanest Sharpe/Calmar
   win, trivial to compute lookahead-free, and it IS the principled rule for your
   existing defensive cash/index sleeve. This is the headline product feature.
3. Inverse-vol weighting (4c) within the selected names + per-name cap — matters
   a lot at 10-30 names; pairs with #2.
4. No-trade band / hysteresis on rebalancing (1) — stops winner-churn at the
   selection→holding boundary without touching the signal.
5. Daniel-Moskowitz regime gate (5) wired to your SPY-vs-200DMA detector, with
   ASYMMETRIC gradual re-entry (6) to soften V-recovery cost.
6. Treat per-name stops/take-profits (2a/2b) as the NULL to beat, expecting them
   to lose on expectancy even when they raise win-rate.

================================================================================
PRIMARY CITATIONS
================================================================================
Kaminski & Lo (2014) "When Do Stop-Loss Rules Stop Losses?" J. Financial Markets 18.
Moreira & Muir (2017) "Volatility-Managed Portfolios" J. Finance 72(4).
Cederburg, O'Doherty, Wang & Yan (2020) "On the Performance of Volatility-Managed
  Portfolios" J. Financial Economics 138(1).
Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & van Hemert (2018) "The Impact of
  Volatility Targeting" J. Portfolio Management.
Daniel & Moskowitz (2016) "Momentum Crashes" J. Financial Economics 122(2).
Barroso & Santa-Clara (2015) "Momentum Has Its Moments" J. Financial Economics 116(1).
Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" J. Financial Economics 104(2).
Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers" J. Finance 48(1).
Novy-Marx & Velikov (2016) "A Taxonomy of Anomalies and Their Trading Costs" RFS 29(1).
Frazzini, Israel & Moskowitz (2012/2018) "Trading Costs of Asset Pricing Anomalies".
DeMiguel, Garlappi & Uppal (2009) "Optimal Versus Naive Diversification" RFS 22(5).
Asness, Frazzini & Pedersen (2012) "Leverage Aversion and Risk Parity" FAJ.
Frazzini & Pedersen (2014) "Betting Against Beta" J. Financial Economics 111(1).
Vanguard (2012, upd. 2023) "Cost averaging: invest now or temporarily hold your cash?"
