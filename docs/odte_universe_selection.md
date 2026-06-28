# 0DTE / short-dated options universe — selection criteria & sources

**Scope.** This note documents *why* the `options_social.core_universe` watchlist (see
`cfg/config.yaml`) contains the names it does. The list is a **scanning watchlist for the
analysis/paper 0DTE report — NOT an auto-trade list.** The tool places no orders; it filters
social chatter to liquid, options-suitable underlyings so the report ranks tradable names instead
of arbitrary all-caps noise. Tradability of any specific contract is still re-checked live
(price/chain fetch, spread caps) before a name is surfaced as an idea.

## Evidence-backed criteria for an options underlying

Across the practitioner and academic literature the same liquidity primitives recur:

1. **Open interest (per strike/expiry).** Higher OI ⇒ more resting size, more market makers
   quoting, tighter fills. A common rule of thumb is to avoid strikes under ~100 contracts of OI.
2. **Bid-ask spread.** A tight (ideally penny-wide ATM) spread is the single most direct cost of
   trading; it is the empirical proxy for option-market liquidity. Far-OTM strikes or thinly
   quoted underlyings can show decent OI yet still trade wide.
3. **Underlying volume / turnover.** Heavy share volume in the underlying is what *makes* the
   option chain active in the first place — major-index ETFs sit at the top on this axis.
4. **Transaction costs & event risk.** Realized-vs-implied vol and scheduled events (earnings,
   FOMC) drive the cost and tail risk of short-dated bets; index/sector ETFs diffuse single-name
   event risk relative to single stocks.

These map to the report's own guardrails: `_WIDE_SPREAD` caps confidence on wide chains, and the
universe below is biased toward names that clear the OI/volume/spread bar by construction.

## Daily (true 0DTE) vs. weekly/frequent expirations

Only six products carry **0DTE options that expire every trading day**: **SPX, SPY, XSP, NDX,
QQQ, IWM** (Cboe). A broader set (e.g. AAPL, MSFT, AMZN, META, TSLA, GLD) carries
**Mon/Wed/Fri — and, for the most liquid single names, Tue/Thu in 2026 — short-dated** chains, plus
standard weeklies. The watchlist therefore separates *true-daily index ETFs* from *frequent/weekly
single names & sector ETFs*; the latter are valid short-dated-scan targets but are not literally
0DTE every session.

0DTE represented **roughly 43% of SPX average daily volume in 2023** (Cboe), up sharply from a
few years earlier — the liquidity is concentrated in exactly these index/mega-cap names.

## Account-fit & compliance constraints

- **Tiny account.** Prefer ETFs and mega-caps with penny-ish ATM spreads so slippage doesn't
  dominate a small position; avoid illiquid/meme single names.
- **XSP (mini-SPX) is SCAN-ONLY.** Cash-settled, true-daily 0DTE at 1/10 the SPX notional, so it is
  the most tiny-account-friendly index 0DTE *if the broker's quoted liquidity/spreads are
  acceptable* (XSP can quote wider than SPY despite identical underlying — verify the live chain
  before trading). It is in the broad social-scan `core_universe`, **not** the live execution core:
  the scan list is intentionally broader than the names actually traded.
- **NVDA is hard-excluded as a trade vehicle** (employer restriction). It is *not* in the tradable
  universe; if it appears in chatter the code surfaces it only as a read-only `RESTRICTED_EMPLOYER`
  context card with all contracts stripped (`RESTRICTED_EMPLOYER_TICKERS` in
  `src/data/social_sentiment.py`). This is enforced in code, not just config.

## Sources

**Live-verified (URL fetched and loaded, 2026-06-27):**

- **Cboe, "The Evolution of Same-Day Options Trading"** —
  https://www.cboe.com/insights/posts/the-evolution-of-same-day-options-trading/ — confirms the six
  daily-expiry products (SPX, SPY, XSP, NDX, QQQ, IWM) and that SPX 0DTE was ~43% of average daily
  volume in 2023.
- **TradingBlock, "Options Trading Liquidity: Volume, Open Interest, Size"** —
  https://www.tradingblock.com/blog/options-liquidity — explains the four liquidity metrics (volume,
  open interest, spread, bid/ask size).

**Corroborated by web search / practitioner references (not individually URL-verified this
session — automated fetch was blocked or pages move):**

- Options Hawk, "Why Volume and Open Interest Matter to Liquidity" — more OI ⇒ more market makers
  quoting ⇒ tighter spreads; SPY/QQQ ATM spreads can be ~$0.01. (Page 403s an automated fetch;
  content confirmed via live search snippet.)
- Charles Schwab, "What Are 0DTE Options? Learn the Basics" and "Wide Bid/Ask Options Spreads in
  Volatile Markets" — basics + spread behavior. (Schwab learn pages auth-block automated fetching;
  cited by title, link intentionally omitted to avoid a dead-link claim.)
- Option Samurai, "Options Liquidity"; my0dteoptions.com / bitget wiki — per-underlying daily vs.
  Mon/Wed/Fri expiration coverage.

**Academic / prior reference (not re-fetched this session):**

- M. Nemes, "Option Market Liquidity — an empirical study of option market bid-ask spreads" (ETH
  Zürich MAS thesis) — bid-ask spread as the empirical liquidity measure.
