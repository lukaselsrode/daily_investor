"""
source_data.py — Stock universe generation and fundamental data collection.

Key fixes vs original:
  - Removed dead get_reddit_data() and get_portfolio_data() stubs
  - Ticker validation now allows dots (BRK.B, BF.B) — was incorrectly using isalpha()
  - safe_float and DATA_DIRECTORY imported from util (no more duplication)
  - AGG_DATA_COLUMNS / METRIC_KEYS imported from util (single definition)
  - Reddit merge in get_data() removed — reddit data was never reliably populated
    and added unnecessary complexity; news is the active sentiment signal

Changes in this revision:
  - Added 52-week price fields: current_price, low_52w, high_52w, position_52w
  - Added momentum_score based on 52-week price location
  - Valuation guardrails: MIN_PE_RATIO, MIN_PB_RATIO, MAX_PE_COMPONENT, MAX_PB_COMPONENT
  - Score weights driven by SCORE_WEIGHTS from util (YAML-configurable)
  - Quote enrichment step adds current_price to fundamental data
"""

import json
import logging
import os
import time

import pandas as pd
import requests
import robin_stocks.robinhood as rb
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from sentiments import get_news_for_tickers_by_symbol
from util import (
    AGG_DATA_COLUMNS,
    ANALYST_PARAMS,
    DATA_DIRECTORY,
    DIVIDEND_THRESHOLD,
    EXCLUDED_STOCK_INDUSTRIES,
    EXCLUDED_STOCK_SECTORS,
    IGNORE_NEGATIVE_PB,
    IGNORE_NEGATIVE_PE,
    MAX_PE_COMPONENT,
    MAX_PB_COMPONENT,
    MIN_PE_RATIO,
    MIN_PB_RATIO,
    METRIC_KEYS,
    METRIC_THRESHOLD,
    MOMENTUM_PARAMS,
    MOMENTUM_V2_PARAMS,
    RELIABILITY_PARAMS,
    SCORE_WEIGHTS,
    SCORING_PARAMS,
    get_investment_ratios,
    read_data_as_pd,
    safe_float,
    store_data_as_csv,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stock universe
# ---------------------------------------------------------------------------

_INDEX_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/Nasdaq-100",
    "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "https://en.wikipedia.org/wiki/Russell_2000_Index",
]

_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

_ROBINHOOD_TAGS = [
    "100-most-popular",
    "upcoming-earnings",
    "new-on-robinhood",
    "technology",
    "finance",
    "healthcare",
    "energy",
]

_VALID_TICKER_RE = __import__("re").compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")


def _is_valid_ticker(symbol: str) -> bool:
    """Accept standard US equity tickers including dot-suffixed ones (BRK.B, BF.B)."""
    return bool(symbol and isinstance(symbol, str) and _VALID_TICKER_RE.match(symbol))


def _scrape_wikipedia_tickers(url: str) -> set[str]:
    try:
        resp = requests.get(url, headers=_WIKI_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")
        table = soup.find("table", {"class": "wikitable sortable"})
        if not table:
            return set()
        symbols: set[str] = set()
        for row in table.find_all("tr")[1:]:
            for cell in row.find_all("td"):
                text = cell.text.strip()
                if _is_valid_ticker(text):
                    symbols.add(text)
        return symbols
    except Exception as e:
        print(f"Wikipedia scrape failed for {url}: {e}")
        return set()


def gen_symbols_list(force_refresh: bool = False) -> list[str]:
    if not force_refresh:
        cached = read_data_as_pd("stock_tickers")
        if cached is not None and not cached.empty and "symbol" in cached.columns:
            return cached["symbol"].tolist()

    # Wikipedia indices
    all_symbols: set[str] = set()
    for url in _INDEX_URLS:
        print(f"Scraping {url}")
        all_symbols.update(_scrape_wikipedia_tickers(url))

    # Robinhood sources — each call wrapped individually so a reset on one doesn't abort all
    rb_sources: list = []
    for fn, args, label in [
        (rb.get_top_movers_sp500, ("down",), "top_movers_sp500(down)"),
        (rb.get_top_movers,       (),         "top_movers"),
        (rb.get_top_100,          (),         "top_100"),
        (rb.get_top_movers_sp500, ("up",),    "top_movers_sp500(up)"),
    ]:
        try:
            result = fn(*args)
            if result:
                rb_sources.append(result)
        except Exception as e:
            print(f"  {label} failed: {str(e)[:60]}")
        time.sleep(0.5)

    for tag in _ROBINHOOD_TAGS:
        try:
            stocks = rb.get_all_stocks_from_market_tag(tag)
            if stocks:
                rb_sources.append(stocks)
                print(f"  Tag '{tag}': {len(stocks)} stocks")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Tag '{tag}' failed: {str(e)[:50]}")

    invalid = 0
    for source in rb_sources:
        for item in (source or []):
            sym = item.get("symbol", "")
            if _is_valid_ticker(sym):
                all_symbols.add(sym)
            else:
                invalid += 1

    print(f"Universe: {len(all_symbols)} valid tickers ({invalid} invalid skipped)")
    store_data_as_csv("stock_tickers", ["symbol"], [[s] for s in sorted(all_symbols)])
    return sorted(all_symbols)


# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------

def _dividend_income_score(dividend_yield: float) -> tuple[float, bool]:
    """Return (income_score, yield_trap_flag)."""
    if not dividend_yield or dividend_yield <= 0:
        return 0.0, False
    if dividend_yield >= SCORING_PARAMS["yield_trap_threshold"]:
        return 0.0, True
    if dividend_yield >= DIVIDEND_THRESHOLD:
        return min(dividend_yield / DIVIDEND_THRESHOLD, SCORING_PARAMS["income_score_cap"]), False
    return 0.0, False


def _quality_score(
    pe_ratio: float | None,
    pb_ratio: float | None,
    volume: float,
    dividend_yield: float,
) -> float:
    sp = SCORING_PARAMS
    score = 0.0
    if pe_ratio is not None and pe_ratio > 0:
        score += sp["quality_weight_has_positive_pe"]
    if pe_ratio is not None and 0 < pe_ratio < sp["distress_pe_max"]:
        score += sp["quality_weight_distress_pe"]
    if pb_ratio is not None and pb_ratio > 0:
        score += sp["quality_weight_has_positive_pb"]
    if volume >= sp["quality_volume_high"]:
        score += sp["quality_weight_high_volume"]
    elif volume < sp["quality_volume_low"]:
        score += sp["quality_weight_low_volume"]
    if dividend_yield >= sp["yield_trap_threshold"]:
        score += sp["quality_weight_yield_trap"]
    elif sp["quality_dividend_min"] <= dividend_yield <= sp["quality_dividend_max"]:
        score += sp["quality_weight_healthy_dividend"]
    return round(score, 3)


def _position_52w(
    current_price: float | None,
    low_52w: float | None,
    high_52w: float | None,
) -> float | None:
    """Return price location within 52-week range, clamped to [0.0, 1.0]."""
    if current_price is None or low_52w is None or high_52w is None:
        return None
    if high_52w <= low_52w:
        return None
    raw = (current_price - low_52w) / (high_52w - low_52w)
    return max(0.0, min(1.0, raw))


def get_momentum_score(position_52w: float | None, return_1m: float | None = None) -> float:
    """Map 52-week position and 1-month return to a momentum score."""
    if position_52w is None:
        return 0.0

    mp = MOMENTUM_PARAMS
    bins   = mp["position_bin_boundaries"]
    scores = mp["position_bin_scores"]

    base = scores[-1]
    for i, boundary in enumerate(bins):
        if position_52w < boundary:
            base = scores[i]
            break

    cutoff = mp["return_1m_low_position_cutoff"]
    if return_1m is not None and position_52w < cutoff:
        if return_1m >= mp["return_1m_recovery_threshold"]:
            base += mp["return_1m_recovery_bonus"]
        elif return_1m <= mp["return_1m_falling_knife_threshold"]:
            base -= mp["return_1m_falling_knife_penalty"]

    return round(base, 3)


def _get_buy_to_sell_ratio(symbol: str) -> float | None:
    try:
        ratings = rb.stocks.get_ratings(symbol)
        if not isinstance(ratings, dict):
            return None
        summary = ratings.get("summary") or {}
        buys = summary.get("num_buy_ratings") or 0
        sells = summary.get("num_sell_ratings") or 0
        return buys / (sells or 1)
    except Exception as e:
        if "404" not in str(e) and "None" not in str(e):
            print(f"Ratings fetch failed for {symbol}: {str(e)[:50]}")
        return None


def _evaluate_stock(symbol: str, stock: dict) -> list | None:
    """Compute all metric columns for one stock. Returns None if stock is unscoreable."""
    if not isinstance(stock, dict):
        return None

    required = ["industry", "sector", "volume", "pe_ratio", "pb_ratio"]
    if not all(k in stock for k in required):
        return None
    if not stock.get("industry") and not stock.get("sector"):
        return None

    # Exclude investment trusts and mutual funds — they don't score like operating companies
    # and accounted for 23% of the universe (400 rows) in analysis.
    # Intentional ETF exposure is handled separately via the explicit ETFS config list.
    _sector   = stock.get("sector", "") or ""
    _industry = stock.get("industry", "") or ""
    if _sector in EXCLUDED_STOCK_SECTORS or _industry in EXCLUDED_STOCK_INDUSTRIES:
        return None

    volume = safe_float(stock.get("volume"), 0)
    if not volume:
        return None

    pe_ratio = safe_float(stock.get("pe_ratio"))
    pb_ratio = safe_float(stock.get("pb_ratio"))
    dividend_yield_raw = safe_float(stock.get("dividend_yield"), 0.0)
    dividend_yield = dividend_yield_raw / 100 if dividend_yield_raw else 0.0

    if pe_ratio is not None and pe_ratio < 0 and IGNORE_NEGATIVE_PE:
        return None
    if pb_ratio is not None and pb_ratio < 0 and IGNORE_NEGATIVE_PB:
        return None

    # 52-week / momentum fields — try multiple field name conventions
    current_price = safe_float(
        stock.get("current_price") or stock.get("last_trade_price") or stock.get("adjusted_open_price")
    )
    low_52w  = safe_float(stock.get("low_52w")  or stock.get("low_52_weeks"))
    high_52w = safe_float(stock.get("high_52w") or stock.get("high_52_weeks"))
    pos_52w   = _position_52w(current_price, low_52w, high_52w)
    return_1m = safe_float(stock.get("return_1m"))
    momentum  = get_momentum_score(pos_52w, return_1m)

    pe_threshold, pb_threshold = get_investment_ratios(stock.get("sector"), stock.get("industry"))

    # PE component with guardrails
    pe_comp_raw = 0.0
    if pe_ratio is not None and MIN_PE_RATIO <= pe_ratio < pe_threshold:
        pe_comp_raw = pe_threshold / pe_ratio
    pe_comp = min(pe_comp_raw, MAX_PE_COMPONENT)

    # PB component with guardrails
    pb_comp_raw = 0.0
    if pb_ratio is not None and MIN_PB_RATIO <= pb_ratio < pb_threshold:
        pb_comp_raw = pb_threshold / pb_ratio
    pb_comp = min(pb_comp_raw, MAX_PB_COMPONENT)

    if pe_comp_raw > MAX_PE_COMPONENT or pb_comp_raw > MAX_PB_COMPONENT:
        logger.debug(
            f"{symbol}: capped PE/PB component "
            f"pe_raw={pe_comp_raw:.3f}, pe_capped={pe_comp:.3f}, "
            f"pb_raw={pb_comp_raw:.3f}, pb_capped={pb_comp:.3f}"
        )

    # Penalise stocks with no valuation evidence at all.  A zero score (the
    # previous behaviour) treated missing data as neutral; -0.25 makes it a
    # mild negative so these names can only win on quality + momentum.
    missing_value_flag = pe_ratio is None and pb_ratio is None
    if missing_value_flag:
        value_score = -0.25
    else:
        value_score = round(
            SCORING_PARAMS["value_pe_weight"] * pe_comp
            + SCORING_PARAMS["value_pb_weight"] * pb_comp,
            3,
        )

    income_score, yield_trap_flag = _dividend_income_score(dividend_yield)
    quality      = _quality_score(pe_ratio, pb_ratio, volume, dividend_yield)

    final_metric = round(
        SCORE_WEIGHTS["value"]    * value_score
        + SCORE_WEIGHTS["quality"]  * quality
        + SCORE_WEIGHTS["income"]   * income_score
        + SCORE_WEIGHTS["momentum"] * momentum,
        3,
    )
    buy_to_sell = None  # fetched post-filter for shortlisted candidates only

    # High-quality businesses near their 52w low: tag for monitoring but don't auto-buy.
    # Criteria: quality>=1.0, negative momentum, position in bottom quarter of 52w range.
    if quality >= 1.0 and momentum < 0 and pos_52w is not None and pos_52w < 0.25:
        strategy_bucket = "contrarian_watchlist"
    else:
        strategy_bucket = "core_candidate"

    return [
        stock.get("industry"),
        stock.get("sector"),
        volume,
        pe_ratio,
        pb_ratio,
        dividend_yield,
        current_price,
        low_52w,
        high_52w,
        pos_52w,
        return_1m,
        pe_comp,
        pb_comp,
        value_score,
        income_score,
        quality,
        momentum,   # placeholder — overwritten by _apply_cross_sectional_momentum_scores
        yield_trap_flag,
        final_metric,
        buy_to_sell,
        missing_value_flag,
        strategy_bucket,
        # momentum v2 raw features (populated by _enrich_with_momentum)
        stock.get("return_5d"),
        stock.get("return_3m"),
        stock.get("return_6m"),
        stock.get("rs_1m"),
        stock.get("rs_3m"),
        stock.get("rs_6m"),
        stock.get("realized_vol_3m"),
        stock.get("risk_adj_momentum_3m"),
        stock.get("above_50dma"),
        stock.get("above_200dma"),
    ]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _enrich_with_quotes(symbols: list[str], fundamentals: dict[str, dict]) -> None:
    """Batch-fetch current prices from Robinhood quotes and merge into fundamentals. Non-fatal."""
    batch_size = 50
    enriched = 0
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i: i + batch_size]
        try:
            quotes = rb.stocks.get_quotes(batch)
            if not quotes or not isinstance(quotes, list):
                continue
            for q in quotes:
                if not q or not isinstance(q, dict):
                    continue
                sym   = q.get("symbol")
                price = q.get("last_trade_price") or q.get("last_extended_hours_trade_price")
                if sym and sym in fundamentals and price:
                    fundamentals[sym]["current_price"] = price
                    enriched += 1
        except Exception as e:
            logger.warning(f"Quote batch {i//batch_size} failed: {str(e)[:60]}")
    logger.info(f"Quote enrichment: {enriched}/{len(symbols)} symbols have current_price")


def _enrich_with_momentum(symbols: list[str], fundamentals: dict[str, dict]) -> None:
    """
    Batch-fetch multi-timeframe returns and momentum features via yfinance.

    Computes for each stock (if data available):
      return_5d, return_1m, return_3m, return_6m
      realized_vol_3m (annualized from 63-day daily stddev)
      above_50dma, above_200dma
      rs_1m, rs_3m, rs_6m (vs SPY)
      risk_adj_momentum_3m = return_3m / realized_vol_3m

    Threads=False + small batches prevents EMFILE on 2500+ symbol universes.
    Period=220d covers: 5d, 21d, 63d, 126d lookbacks + 200dma buffer.
    """
    import requests as _requests

    # Fetch SPY reference returns once — used for relative-strength computation
    spy_returns: dict[str, float | None] = {"5d": None, "1m": None, "3m": None, "6m": None}
    try:
        spy_raw = yf.download("SPY", period="230d", progress=False, auto_adjust=True, threads=False)
        if not spy_raw.empty:
            spy_closes = spy_raw["Close"].squeeze().dropna()
            n = len(spy_closes)
            if n >= 5:
                spy_returns["5d"] = round(float(spy_closes.iloc[-1] / spy_closes.iloc[-5]) - 1.0, 4)
            if n >= 21:
                spy_returns["1m"] = round(float(spy_closes.iloc[-1] / spy_closes.iloc[-21]) - 1.0, 4)
            if n >= 63:
                spy_returns["3m"] = round(float(spy_closes.iloc[-1] / spy_closes.iloc[-63]) - 1.0, 4)
            if n >= 126:
                spy_returns["6m"] = round(float(spy_closes.iloc[-1] / spy_closes.iloc[-126]) - 1.0, 4)
    except Exception as e:
        logger.warning(f"SPY reference fetch failed: {e}")

    logger.info(
        "SPY reference returns: 5d=%s 1m=%s 3m=%s 6m=%s",
        spy_returns["5d"], spy_returns["1m"], spy_returns["3m"], spy_returns["6m"],
    )

    batch_size = 50
    coverage: dict[str, int] = {k: 0 for k in ["return_5d", "return_1m", "return_3m", "return_6m",
                                                  "realized_vol_3m", "above_50dma", "above_200dma",
                                                  "rs_1m", "rs_3m", "rs_6m", "risk_adj_momentum_3m"]}
    n_sym = len(symbols)

    for i in range(0, n_sym, batch_size):
        batch = symbols[i: i + batch_size]
        session = _requests.Session()
        try:
            raw = yf.download(
                batch,
                period="230d",
                progress=False,
                auto_adjust=True,
                threads=False,
                session=session,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                try:
                    closes_df = raw["Close"]
                except KeyError:
                    continue
            else:
                closes_df = raw[["Close"]].rename(columns={"Close": batch[0]})
            if isinstance(closes_df, pd.Series):
                closes_df = closes_df.to_frame(name=batch[0])

            for sym in batch:
                if sym not in closes_df.columns:
                    continue
                col = closes_df[sym].dropna()
                n = len(col)
                if n < 5:
                    continue

                last = float(col.iloc[-1])

                # Multi-timeframe returns
                def _ret(lookback: int) -> float | None:
                    if n >= lookback:
                        prev = float(col.iloc[-lookback])
                        return round(last / prev - 1.0, 4) if prev > 0 else None
                    return None

                r5d  = _ret(5)
                r1m  = _ret(21)
                r3m  = _ret(63)
                r6m  = _ret(126)

                if r5d  is not None: fundamentals[sym]["return_5d"]  = r5d;  coverage["return_5d"]  += 1
                if r1m  is not None: fundamentals[sym]["return_1m"]  = r1m;  coverage["return_1m"]  += 1
                if r3m  is not None: fundamentals[sym]["return_3m"]  = r3m;  coverage["return_3m"]  += 1
                if r6m  is not None: fundamentals[sym]["return_6m"]  = r6m;  coverage["return_6m"]  += 1

                # Realized volatility (3-month window, annualized)
                if n >= 63:
                    daily_rets = col.pct_change().dropna()
                    if len(daily_rets) >= 20:
                        vol_3m = round(float(daily_rets.iloc[-63:].std() * (252 ** 0.5)), 4)
                        fundamentals[sym]["realized_vol_3m"] = vol_3m
                        coverage["realized_vol_3m"] += 1

                        # Risk-adjusted 3m momentum
                        if r3m is not None and vol_3m > 0:
                            fundamentals[sym]["risk_adj_momentum_3m"] = round(r3m / vol_3m, 4)
                            coverage["risk_adj_momentum_3m"] += 1

                # 50/200 DMA signals
                if n >= 50:
                    ma50 = float(col.iloc[-50:].mean())
                    fundamentals[sym]["above_50dma"] = last > ma50
                    coverage["above_50dma"] += 1
                if n >= 200:
                    ma200 = float(col.iloc[-200:].mean())
                    fundamentals[sym]["above_200dma"] = last > ma200
                    coverage["above_200dma"] += 1

                # Relative strength vs SPY
                if r1m is not None and spy_returns["1m"] is not None:
                    fundamentals[sym]["rs_1m"] = round(r1m - spy_returns["1m"], 4)
                    coverage["rs_1m"] += 1
                if r3m is not None and spy_returns["3m"] is not None:
                    fundamentals[sym]["rs_3m"] = round(r3m - spy_returns["3m"], 4)
                    coverage["rs_3m"] += 1
                if r6m is not None and spy_returns["6m"] is not None:
                    fundamentals[sym]["rs_6m"] = round(r6m - spy_returns["6m"], 4)
                    coverage["rs_6m"] += 1

        except Exception as e:
            logger.warning(f"Momentum batch {i // batch_size + 1} failed: {str(e)[:80]}")
        finally:
            session.close()

    for feat, cnt in coverage.items():
        pct = cnt / max(n_sym, 1) * 100
        level = logging.WARNING if pct < 50.0 else logging.INFO
        logger.log(level, "Momentum feature %-25s  coverage: %5.1f%% (%d/%d)", feat, pct, cnt, n_sym)


def _pct_rank_series(s: pd.Series, winsorize_pct: float = 0.05) -> pd.Series:
    """
    Cross-sectional percentile rank, winsorized and scaled to [-1, 1].
    Missing values → 0.0 (neutral mid-rank).
    This is NOT a lookahead bias: we rank contemporaneous values across stocks.
    """
    finite = s.notna()
    if finite.sum() < 2:
        return pd.Series(0.0, index=s.index)
    vals = s[finite].copy()
    if winsorize_pct > 0:
        lo = vals.quantile(winsorize_pct)
        hi = vals.quantile(1.0 - winsorize_pct)
        vals = vals.clip(lo, hi)
    ranks = vals.rank(method="average") / (len(vals) + 1)  # (0, 1)
    result = pd.Series(0.0, index=s.index)
    result[finite] = ranks * 2 - 1  # scale to (-1, 1)
    return result


def _apply_cross_sectional_momentum_scores(df: pd.DataFrame) -> None:
    """
    Replace momentum_score with the v2 continuous cross-sectional formula.
    Called once after all stocks are evaluated, so ranking is across the full
    daily universe — no lookahead into future dates.
    """
    import numpy as np

    cfg = MOMENTUM_V2_PARAMS
    wp  = cfg["weights"]
    pen = cfg["penalties"]
    wp_total = sum(wp.values())
    if wp_total < 1e-9:
        return

    # Normalize weights
    w_rs3m    = wp["rs_3m"]          / wp_total
    w_rs6m    = wp["rs_6m"]          / wp_total
    w_radj    = wp["risk_adj_3m"]    / wp_total
    w_trend   = wp["trend_structure"] / wp_total
    w_r1m     = wp["return_1m"]      / wp_total
    w_r5d     = wp["return_5d"]      / wp_total

    wz = cfg["winsorize_pct"]

    def _col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(float("nan"), index=df.index)

    # Percentile-rank each feature across the universe
    n_rs3m  = _pct_rank_series(_col("rs_3m"),                 wz)
    n_rs6m  = _pct_rank_series(_col("rs_6m"),                 wz)
    n_radj  = _pct_rank_series(_col("risk_adj_momentum_3m"),  wz)
    n_r1m   = _pct_rank_series(_col("return_1m"),             wz)
    n_r5d   = _pct_rank_series(_col("return_5d"),             wz)

    # Trend structure: deterministic signal, not percentile-ranked
    above50  = df["above_50dma"].astype(bool)  if "above_50dma"  in df.columns else pd.Series(False, index=df.index)
    above200 = df["above_200dma"].astype(bool) if "above_200dma" in df.columns else pd.Series(False, index=df.index)
    trend = pd.Series(np.select(
        [above50 & above200, above50 & ~above200, ~above50 & above200],
        [0.5,                 0.1,                 -0.1],
        default=-0.5,
    ), index=df.index)

    # Composite score
    score = (
        w_rs3m  * n_rs3m  +
        w_rs6m  * n_rs6m  +
        w_radj  * n_radj  +
        w_trend * trend    +
        w_r1m   * n_r1m   +
        w_r5d   * n_r5d
    )

    # Penalties
    ret3m   = _col("return_3m")
    vol3m   = _col("realized_vol_3m")
    pos52   = _col("position_52w")

    falling_knife  = ret3m.fillna(0.0) < pen["falling_knife_3m_threshold"]
    overextended   = pos52.fillna(0.0) > pen["overextension_52w_threshold"]
    high_vol       = vol3m.fillna(0.0) > pen["high_vol_annual_threshold"]

    score = score - falling_knife.astype(float) * pen["falling_knife_penalty"]
    score = score - overextended.astype(float)  * pen["overextension_penalty"]
    score = score - high_vol.astype(float)       * pen["high_vol_penalty"]

    score = score.clip(cfg["clamp_low"], cfg["clamp_high"]).round(3)
    df["momentum_score"] = score

    # Log distribution summary
    logger.info(
        "Momentum v2 score distribution: min=%.3f p25=%.3f med=%.3f p75=%.3f max=%.3f unique=%d",
        float(score.min()), float(score.quantile(0.25)),
        float(score.median()), float(score.quantile(0.75)),
        float(score.max()), int(score.nunique()),
    )


def _compute_reliability_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute four component reliability scores (0–1) and combine into reliability_score.

    DATA / SIGNAL QUALITY indicators only — NOT alpha factors.
    Measures how much we can trust a stock's computed value_metric,
    not whether the stock will outperform.
    """
    import numpy as np

    n = len(df)

    def _num(col: str) -> "pd.Series":
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([float("nan")] * n)

    # ── 1. Data Quality Score ─────────────────────────────────────────────────
    # Rewards presence of clean fundamental inputs; penalizes extreme/suspicious values.
    dq = np.full(n, 0.40)  # base: partial credit even with thin data

    pe = _num("pe_ratio")
    pb = _num("pb_ratio")
    dy = _num("dividend_yield")
    bsr = _num("buy_to_sell_ratio")
    pe_c = _num("pe_comp")

    dq += 0.20 * np.where(pe.notna() & (pe > 0), 1.0, 0.0)
    dq += 0.20 * np.where(pb.notna() & (pb > 0), 1.0, 0.0)
    dq += 0.10 * np.where(dy.notna(), 1.0, 0.0)
    dq += 0.10 * np.where(bsr.notna(), 1.0, 0.0)
    # Penalty for implausible valuation components
    dq -= 0.15 * np.where(pe_c.notna() & (pe_c <= 0), 1.0, 0.0)

    dq = np.clip(dq, 0.0, 1.0)

    # ── 2. Feature Coverage Score ─────────────────────────────────────────────
    # Fraction of momentum v2 inputs that are non-NaN. More features → more reliable score.
    momentum_cols = [
        "return_1m", "return_3m", "return_6m", "return_5d",
        "rs_3m", "rs_6m", "realized_vol_3m", "above_50dma", "above_200dma",
    ]
    present = [c for c in momentum_cols if c in df.columns]
    if present:
        fc = np.mean(
            [np.where(_num(c).notna(), 1.0, 0.0) for c in present], axis=0
        )
    else:
        fc = np.zeros(n)

    # ── 3. Liquidity Reliability Score ────────────────────────────────────────
    # Higher volume and lower realized volatility → more reliable pricing.
    lr = np.full(n, 0.50)  # neutral base

    vol_ser = _num("volume").fillna(0)
    # Map log volume to [-0.30, +0.30] bonus/penalty around the min_liquidity threshold
    log_vol = np.log1p(vol_ser.values)
    log_low  = float(np.log1p(500_000))
    log_high = float(np.log1p(5_000_000))
    lr += 0.30 * np.clip(
        (log_vol - log_low) / max(log_high - log_low, 1e-9) - 0.5, -0.5, 0.5
    )

    rv = _num("realized_vol_3m")
    lr -= 0.15 * np.where(rv.notna() & (rv > 0.60), 1.0, 0.0)
    lr -= 0.10 * np.where(rv.notna() & (rv > 0.40), 1.0, 0.0)

    lr = np.clip(lr, 0.0, 1.0)

    # ── 4. Signal Stability Score ─────────────────────────────────────────────
    # Measures internal consistency of momentum signals; high agreement → stable signal.
    ss = np.full(n, 0.50)

    # High realized vol → noisier momentum signal
    rv_fill = rv.fillna(0.30).values
    ss -= 0.25 * np.clip(rv_fill / 0.50, 0.0, 1.0)

    # 3m and 6m return direction agreement → signal is trending, not reversing
    r3 = _num("return_3m").fillna(0)
    r6 = _num("return_6m").fillna(0)
    ss += 0.20 * np.where(np.sign(r3) == np.sign(r6), 1.0, 0.0)

    # RS 3m and RS 6m direction agreement → relative strength is persistent
    rs3 = _num("rs_3m").fillna(0)
    rs6 = _num("rs_6m").fillna(0)
    ss += 0.20 * np.where(np.sign(rs3) == np.sign(rs6), 1.0, 0.0)

    ss = np.clip(ss, 0.0, 1.0)

    # ── Composite ─────────────────────────────────────────────────────────────
    reliability = 0.30 * dq + 0.30 * fc + 0.20 * lr + 0.20 * ss

    df = df.copy()
    df["data_quality_score"]          = np.round(dq,          3)
    df["feature_coverage_score"]      = np.round(fc,          3)
    df["liquidity_reliability_score"] = np.round(lr,          3)
    df["signal_stability_score"]      = np.round(ss,          3)
    df["reliability_score"]           = np.round(reliability, 3)

    rel = df["reliability_score"]
    logger.info(
        "Reliability scores: mean=%.3f  high(≥0.70): %.0f%%  "
        "feature_coverage mean=%.2f",
        rel.mean(), (rel >= 0.70).mean() * 100, float(fc.mean()),
    )
    return df


def _get_robinhood_fundamentals(tickers: list[str], force_refresh: bool) -> pd.DataFrame | None:
    if not force_refresh:
        return read_data_as_pd("robinhood_data")

    # Raise the fd soft limit before bulk yfinance downloads to avoid EMFILE
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 4096)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

    batch_size = 50
    fundamentals: dict[str, dict] = {}

    print(f"Fetching fundamentals for {len(tickers)} stocks in batches of {batch_size}...")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        batch_num = i // batch_size + 1
        total = (len(tickers) + batch_size - 1) // batch_size
        try:
            print(f"Batch {batch_num}/{total} ({len(batch)} stocks)...")
            result = rb.get_fundamentals(batch)
            if result and isinstance(result, list):
                for item in result:
                    if item and isinstance(item, dict) and "symbol" in item:
                        fundamentals[item["symbol"]] = item
        except Exception as e:
            print(f"Batch {batch_num} failed: {str(e)[:60]}")

    print(f"Fetched fundamentals for {len(fundamentals)} stocks")

    # Enrich with current prices from quotes
    _enrich_with_quotes(list(fundamentals.keys()), fundamentals)

    # Enrich with 1-month price returns for directional momentum scoring
    _enrich_with_momentum(list(fundamentals.keys()), fundamentals)
    r1m_have = sum(1 for d in fundamentals.values() if d.get("return_1m") is not None)
    r1m_pct  = r1m_have / max(len(fundamentals), 1) * 100
    logger.info("return_1m coverage: %.1f%% (%d/%d)", r1m_pct, r1m_have, len(fundamentals))
    if r1m_pct < 50.0:
        logger.warning(
            "return_1m coverage is low (%.1f%%) — momentum scoring will rely on position_52w only",
            r1m_pct,
        )

    rows = []
    for symbol, data in fundamentals.items():
        metrics = _evaluate_stock(symbol, data)
        if metrics:
            rows.append([symbol] + metrics)

    df_raw = pd.DataFrame(rows, columns=AGG_DATA_COLUMNS)

    # Cross-sectional momentum v2 normalization — must run before saving
    _apply_cross_sectional_momentum_scores(df_raw)

    # Recompute value_metric with updated momentum_score
    for col in ["value_score", "income_score", "quality_score", "momentum_score"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0.0)
    df_raw["value_metric"] = (
        SCORE_WEIGHTS["value"]    * df_raw["value_score"]
        + SCORE_WEIGHTS["quality"]  * df_raw["quality_score"]
        + SCORE_WEIGHTS["income"]   * df_raw["income_score"]
        + SCORE_WEIGHTS["momentum"] * df_raw["momentum_score"]
    ).round(3)

    # Reliability scoring — data/signal quality, not alpha
    df_raw = _compute_reliability_scores(df_raw)

    # Universe composition diagnostics
    if "sector" in df_raw.columns:
        sector_counts = df_raw["sector"].value_counts()
        logger.info("Universe composition by sector:\n%s", sector_counts.to_string())
    nan_rates = {
        c: round(df_raw[c].isna().mean() * 100, 1)
        for c in ["pe_ratio", "pb_ratio", "return_1m", "return_3m", "rs_3m", "realized_vol_3m"]
        if c in df_raw.columns
    }
    logger.info("NaN rates %%: %s", nan_rates)

    store_data_as_csv("robinhood_data", AGG_DATA_COLUMNS, df_raw)
    time.sleep(1)
    df = read_data_as_pd("robinhood_data")

    # Fetch analyst buy/sell ratings only for shortlisted candidates — avoids 1000+
    # sequential API calls during bulk collection while still giving Claude a signal
    # and incorporating consensus into the score (±5% multiplier).
    if df is not None and not df.empty:
        df["value_metric"] = pd.to_numeric(df["value_metric"], errors="coerce")
        candidates = df[df["value_metric"] >= METRIC_THRESHOLD]["symbol"].tolist()
        if candidates:
            print(f"Fetching analyst ratings for {len(candidates)} shortlisted stocks...")
            for sym in candidates:
                ratio = _get_buy_to_sell_ratio(sym)
                df.loc[df["symbol"] == sym, "buy_to_sell_ratio"] = ratio
                # BTR is stored for tie-breaking in make_buys but does NOT mutate
                # value_metric — coverage is too sparse (~17%) for a reliable multiplier.
                time.sleep(0.3)
            store_data_as_csv("robinhood_data", AGG_DATA_COLUMNS, df)
            time.sleep(1)
            df = read_data_as_pd("robinhood_data")

    return df


def _get_news(tickers: list[str], force_refresh: bool) -> pd.DataFrame | None:
    if not force_refresh:
        return read_data_as_pd("news")

    # Only fetch news for liquid stocks (volume already in robinhood_data)
    rb_data = read_data_as_pd("robinhood_data")
    if rb_data is not None and not rb_data.empty and "volume" in rb_data.columns:
        liquid = rb_data[rb_data["volume"] >= 500_000]["symbol"].tolist()
        print(f"News filter: {len(tickers)} total → {len(liquid)} liquid tickers")
    else:
        liquid = tickers

    news_by_symbol = get_news_for_tickers_by_symbol(liquid, max_articles=3)

    # Ensure every ticker has an entry (empty list for low-volume ones)
    for t in tickers:
        news_by_symbol.setdefault(t, [])

    news_df = pd.DataFrame([
        {"symbol": sym, "news": json.dumps(articles)}
        for sym, articles in news_by_symbol.items()
    ])
    store_data_as_csv("news", ["symbol", "news"], news_df)
    return read_data_as_pd("news")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_data(refresh: bool = False) -> pd.DataFrame:
    tickers = gen_symbols_list(refresh)
    metrics = _get_robinhood_fundamentals(tickers, refresh)
    news_df = _get_news(tickers, refresh)

    if metrics is None or metrics.empty:
        print("Warning: No fundamental data available")
        return pd.DataFrame()

    if news_df is not None and not news_df.empty:
        result = metrics.merge(news_df, on="symbol", how="left")
    else:
        result = metrics.copy()

    if not result.empty:
        store_data_as_csv("agg_data", "", result)
        time.sleep(1)

    return result

if __name__ == "__main__":
    df = get_data(refresh=False)
    print(df)
