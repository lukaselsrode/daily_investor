"""
data/fundamentals.py — Fundamental data fetching, enrichment, and scoring.

Functions:
    _enrich_with_quotes()        — fetch current prices from Robinhood quotes
    _enrich_with_momentum()      — fetch multi-timeframe returns via yfinance
    _position_52w()              — price location within 52-week range
    _get_buy_to_sell_ratio()     — analyst buy/sell ratio from Robinhood
    _get_earnings_bonus()        — EPS surprise quality adjustment
    _diagnose_stock_filter()     — human-readable filter failure reason
    _evaluate_stock()            — per-stock scoring → row vector
    _compute_reliability_scores() — data/signal quality scores (not alpha)
    get_fundamentals_df()        — main pipeline: fetch → score → persist
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import robin_stocks.robinhood as rb
import yfinance as yf

from core.utils import safe_float
from data.cache import read_data_as_pd, store_data_as_csv
from data.valuation import get_investment_ratios
from strategy.scoring.composite import compute_metric
from strategy.snapshots import backfill_from_csvs as _backfill_snapshots  # noqa: F401
from strategy.snapshots import save_snapshot as _save_snapshot
from util import (
    AGG_DATA_COLUMNS,
    EARNINGS_PARAMS,
    EXCLUDED_STOCK_INDUSTRIES,
    EXCLUDED_STOCK_SECTORS,
    IGNORE_NEGATIVE_PB,
    IGNORE_NEGATIVE_PE,
    MAX_PB_COMPONENT,
    MAX_PE_COMPONENT,
    METRIC_THRESHOLD,
    MIN_PB_RATIO,
    MIN_PE_RATIO,
    SCORE_WEIGHTS,
)

logger = logging.getLogger(__name__)

_RELIABILITY_COLS = {
    "data_quality_score", "feature_coverage_score",
    "liquidity_reliability_score", "signal_stability_score", "reliability_score",
}
# Peer-relative diagnostic columns are populated by strategy.scoring.compute_metric()
# at the batch step — not by the per-stock _evaluate_stock loop. Exclude them from
# the per-stock column list so the DataFrame width matches the row width.
_PEER_DIAGNOSTIC_COLS = {
    "value_industry_rank", "value_sector_rank", "value_market_rank",
    "value_fallback_reason", "value_distress_flag",
    "quality_industry_rank", "quality_sector_rank", "quality_market_rank",
    "quality_fallback_reason",
    "momentum_industry_rank", "momentum_sector_rank", "momentum_market_rank",
    "momentum_fallback_reason", "momentum_penalties_applied",
    "income_industry_rank", "income_sector_rank", "income_fallback_reason",
    "scoring_model_version",
}
# instrument_type is fetched from Robinhood and merged onto the universe in
# data.market.get_data — not produced by the per-stock _evaluate_stock loop.
_MARKET_STRUCTURE_COLS = {"instrument_type"}
_BASE_AGG_COLUMNS = [
    c for c in AGG_DATA_COLUMNS
    if c not in _RELIABILITY_COLS
    and c not in _PEER_DIAGNOSTIC_COLS
    and c not in _MARKET_STRUCTURE_COLS
]


# ---------------------------------------------------------------------------
# 52-week position helper
# ---------------------------------------------------------------------------

def _position_52w(
    current_price: float | None,
    low_52w: float | None,
    high_52w: float | None,
) -> float | None:
    if current_price is None or low_52w is None or high_52w is None:
        return None
    if high_52w <= low_52w:
        return None
    raw = (current_price - low_52w) / (high_52w - low_52w)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

def _enrich_with_quotes(symbols: list[str], fundamentals: dict[str, dict]) -> None:
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
            logger.warning("Quote batch %d failed: %s", i // batch_size, str(e)[:60])
    logger.info("Quote enrichment: %d/%d symbols have current_price", enriched, len(symbols))


def _enrich_with_momentum(symbols: list[str], fundamentals: dict[str, dict]) -> None:
    """
    Batch-fetch multi-timeframe returns and momentum features via yfinance.

    Computes for each stock (if data available):
      return_5d, return_1m, return_3m, return_6m
      realized_vol_3m (annualized from 63-day daily stddev)
      above_50dma, above_200dma
      rs_1m, rs_3m, rs_6m (vs SPY)
      risk_adj_momentum_3m = return_3m / realized_vol_3m
    """
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
        logger.warning("SPY reference fetch failed: %s", e)

    logger.info(
        "SPY reference returns: 5d=%s 1m=%s 3m=%s 6m=%s",
        spy_returns["5d"], spy_returns["1m"], spy_returns["3m"], spy_returns["6m"],
    )

    batch_size = 50
    coverage: dict[str, int] = {k: 0 for k in [
        "return_5d", "return_1m", "return_3m", "return_6m",
        "realized_vol_3m", "above_50dma", "above_200dma",
        "rs_1m", "rs_3m", "rs_6m", "risk_adj_momentum_3m",
    ]}
    n_sym = len(symbols)

    for i in range(0, n_sym, batch_size):
        batch = symbols[i: i + batch_size]
        try:
            raw = yf.download(
                batch, period="230d", progress=False, auto_adjust=True, threads=False,
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

                def _ret(lookback: int, n=n, col=col, last=last) -> float | None:
                    if n >= lookback:
                        prev = float(col.iloc[-lookback])
                        return round(last / prev - 1.0, 4) if prev > 0 else None
                    return None

                r5d = _ret(5)
                r1m = _ret(21)
                r3m = _ret(63)
                r6m = _ret(126)

                if r5d is not None:
                    fundamentals[sym]["return_5d"] = r5d
                    coverage["return_5d"] += 1
                if r1m is not None:
                    fundamentals[sym]["return_1m"] = r1m
                    coverage["return_1m"] += 1
                if r3m is not None:
                    fundamentals[sym]["return_3m"] = r3m
                    coverage["return_3m"] += 1
                if r6m is not None:
                    fundamentals[sym]["return_6m"] = r6m
                    coverage["return_6m"] += 1

                if n >= 63:
                    daily_rets = col.pct_change().dropna()
                    if len(daily_rets) >= 20:
                        vol_3m = round(float(daily_rets.iloc[-63:].std() * (252 ** 0.5)), 4)
                        fundamentals[sym]["realized_vol_3m"] = vol_3m
                        coverage["realized_vol_3m"] += 1
                        if r3m is not None and vol_3m > 0:
                            fundamentals[sym]["risk_adj_momentum_3m"] = round(r3m / vol_3m, 4)
                            coverage["risk_adj_momentum_3m"] += 1

                if n >= 50:
                    ma50 = float(col.iloc[-50:].mean())
                    fundamentals[sym]["above_50dma"] = last > ma50
                    coverage["above_50dma"] += 1
                if n >= 200:
                    ma200 = float(col.iloc[-200:].mean())
                    fundamentals[sym]["above_200dma"] = last > ma200
                    coverage["above_200dma"] += 1

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
            logger.warning("Momentum batch %d failed: %s", i // batch_size + 1, str(e)[:80])

    for feat, cnt in coverage.items():
        pct = cnt / max(n_sym, 1) * 100
        level = logging.WARNING if pct < 50.0 else logging.INFO
        logger.log(level, "Momentum feature %-25s  coverage: %5.1f%% (%d/%d)", feat, pct, cnt, n_sym)


# ---------------------------------------------------------------------------
# Analyst / earnings helpers
# ---------------------------------------------------------------------------

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


def _get_earnings_bonus(symbol: str) -> float:
    if not EARNINGS_PARAMS.get("enabled"):
        return 0.0
    try:
        reports = rb.get_earnings(symbol) or []
        if not reports:
            return 0.0

        pos_bonus   = EARNINGS_PARAMS["positive_surprise_bonus"]
        neg_penalty = EARNINGS_PARAMS["negative_surprise_penalty"]
        max_bonus   = EARNINGS_PARAMS["max_quality_bonus"]

        bonus = 0.0
        quarters_used = 0
        for report in reports:
            if quarters_used >= 3:
                break
            eps = report.get("eps") or {}
            estimate = safe_float(eps.get("estimate"))
            actual   = safe_float(eps.get("actual"))
            if estimate is None or actual is None:
                continue
            quarters_used += 1
            if actual > estimate:
                bonus += pos_bonus
            elif actual < estimate:
                bonus -= neg_penalty

        return max(-max_bonus, min(max_bonus, bonus))
    except Exception as e:
        logger.debug("Earnings bonus fetch failed for %s: %s", symbol, e)
        return 0.0


# ---------------------------------------------------------------------------
# Per-stock scoring
# ---------------------------------------------------------------------------

def _diagnose_stock_filter(symbol: str, stock: dict) -> str:
    required = ["industry", "sector", "volume", "pe_ratio", "pb_ratio"]
    missing = [k for k in required if k not in stock]
    if missing:
        return f"missing required fields: {missing}"
    if not stock.get("industry") and not stock.get("sector"):
        return "no industry/sector returned by API"
    _sector   = stock.get("sector", "") or ""
    _industry = stock.get("industry", "") or ""
    if _sector in EXCLUDED_STOCK_SECTORS:
        return f"excluded sector: '{_sector}' (EXCLUDED_STOCK_SECTORS)"
    if _industry in EXCLUDED_STOCK_INDUSTRIES:
        return f"excluded industry: '{_industry}' (EXCLUDED_STOCK_INDUSTRIES)"
    if not safe_float(stock.get("volume"), 0):
        return "volume=0 or None"
    pe = safe_float(stock.get("pe_ratio"))
    if pe is not None and pe < 0 and IGNORE_NEGATIVE_PE:
        return f"negative PE ({pe:.2f}) — set ignore_negative_pe: false in config to include"
    pb = safe_float(stock.get("pb_ratio"))
    if pb is not None and pb < 0 and IGNORE_NEGATIVE_PB:
        return f"negative PB ({pb:.2f}) — set ignore_negative_pb: false in config to include"
    return "passed all filter checks (scored successfully)"


def _evaluate_stock(symbol: str, stock: dict) -> list | None:
    if not isinstance(stock, dict):
        return None

    required = ["industry", "sector", "volume", "pe_ratio", "pb_ratio"]
    if not all(k in stock for k in required):
        return None
    if not stock.get("industry") and not stock.get("sector"):
        return None

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

    current_price = safe_float(
        stock.get("current_price") or stock.get("last_trade_price") or stock.get("adjusted_open_price")
    )
    low_52w  = safe_float(stock.get("low_52w")  or stock.get("low_52_weeks"))
    high_52w = safe_float(stock.get("high_52w") or stock.get("high_52_weeks"))
    pos_52w   = _position_52w(current_price, low_52w, high_52w)
    return_1m = safe_float(stock.get("return_1m"))

    pe_threshold, pb_threshold = get_investment_ratios(stock.get("sector"), stock.get("industry"))

    pe_comp_raw = 0.0
    if pe_ratio is not None and MIN_PE_RATIO <= pe_ratio < pe_threshold:
        pe_comp_raw = pe_threshold / pe_ratio
    pe_comp = min(pe_comp_raw, MAX_PE_COMPONENT)

    pb_comp_raw = 0.0
    if pb_ratio is not None and MIN_PB_RATIO <= pb_ratio < pb_threshold:
        pb_comp_raw = pb_threshold / pb_ratio
    pb_comp = min(pb_comp_raw, MAX_PB_COMPONENT)

    if pe_comp_raw > MAX_PE_COMPONENT or pb_comp_raw > MAX_PB_COMPONENT:
        logger.debug(
            "%s: capped PE/PB component pe_raw=%.3f pe_capped=%.3f pb_raw=%.3f pb_capped=%.3f",
            symbol, pe_comp_raw, pe_comp, pb_comp_raw, pb_comp,
        )

    missing_value_flag = pe_ratio is None and pb_ratio is None
    # Per-stock component scores default to 0.0 — the batch compute_metric() call at
    # the end of get_fundamentals_df overwrites them with peer-relative values.
    value_score = 0.0
    income_score = 0.0
    yield_trap_flag = False
    quality = 0.0
    momentum = 0.0
    final_metric = 0.0
    buy_to_sell = None
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
        momentum,
        yield_trap_flag,
        final_metric,
        buy_to_sell,
        missing_value_flag,
        strategy_bucket,
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


def _compute_reliability_scores(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([float("nan")] * n)

    dq = np.full(n, 0.40)
    pe = _num("pe_ratio")
    pb = _num("pb_ratio")
    dy = _num("dividend_yield")
    bsr = _num("buy_to_sell_ratio")
    pe_c = _num("pe_comp")

    dq += 0.20 * np.where(pe.notna() & (pe > 0), 1.0, 0.0)
    dq += 0.20 * np.where(pb.notna() & (pb > 0), 1.0, 0.0)
    dq += 0.10 * np.where(dy.notna(), 1.0, 0.0)
    dq += 0.10 * np.where(bsr.notna(), 1.0, 0.0)
    dq -= 0.15 * np.where(pe_c.notna() & (pe_c <= 0), 1.0, 0.0)
    dq = np.clip(dq, 0.0, 1.0)

    momentum_cols = [
        "return_1m", "return_3m", "return_6m", "return_5d",
        "rs_3m", "rs_6m", "realized_vol_3m", "above_50dma", "above_200dma",
    ]
    present = [c for c in momentum_cols if c in df.columns]
    fc = np.mean([np.where(_num(c).notna(), 1.0, 0.0) for c in present], axis=0) if present else np.zeros(n)

    lr = np.full(n, 0.50)
    vol_ser = _num("volume").fillna(0)
    log_vol  = np.log1p(vol_ser.values)
    log_low  = float(np.log1p(500_000))
    log_high = float(np.log1p(5_000_000))
    lr += 0.30 * np.clip(
        (log_vol - log_low) / max(log_high - log_low, 1e-9) - 0.5, -0.5, 0.5
    )
    rv = _num("realized_vol_3m")
    lr -= 0.15 * np.where(rv.notna() & (rv > 0.60), 1.0, 0.0)
    lr -= 0.10 * np.where(rv.notna() & (rv > 0.40), 1.0, 0.0)
    lr = np.clip(lr, 0.0, 1.0)

    ss = np.full(n, 0.50)
    rv_fill = rv.fillna(0.30).values
    ss -= 0.25 * np.clip(rv_fill / 0.50, 0.0, 1.0)
    r3 = _num("return_3m").fillna(0)
    r6 = _num("return_6m").fillna(0)
    ss += 0.20 * np.where(np.sign(r3) == np.sign(r6), 1.0, 0.0)
    rs3 = _num("rs_3m").fillna(0)
    rs6 = _num("rs_6m").fillna(0)
    ss += 0.20 * np.where(np.sign(rs3) == np.sign(rs6), 1.0, 0.0)
    ss = np.clip(ss, 0.0, 1.0)

    reliability = 0.30 * dq + 0.30 * fc + 0.20 * lr + 0.20 * ss

    df = df.copy()
    df["data_quality_score"]          = np.round(dq,          3)
    df["feature_coverage_score"]      = np.round(fc,          3)
    df["liquidity_reliability_score"] = np.round(lr,          3)
    df["signal_stability_score"]      = np.round(ss,          3)
    df["reliability_score"]           = np.round(reliability, 3)

    rel = df["reliability_score"]
    logger.info(
        "Reliability scores: mean=%.3f  high(≥0.70): %.0f%%  feature_coverage mean=%.2f",
        rel.mean(), (rel >= 0.70).mean() * 100, float(fc.mean()),
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def get_fundamentals_df(
    tickers: list[str],
    force_refresh: bool,
    portfolio_symbols: set[str] | None = None,
) -> pd.DataFrame | None:
    """Fetch, score, and persist fundamental data for all tickers."""
    if not force_refresh:
        return read_data_as_pd("robinhood_data")

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

    _enrich_with_quotes(list(fundamentals.keys()), fundamentals)
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
    filtered_stocks: dict[str, str] = {}
    for symbol, data in fundamentals.items():
        metrics = _evaluate_stock(symbol, data)
        if metrics:
            rows.append([symbol] + metrics)
        else:
            filtered_stocks[symbol] = _diagnose_stock_filter(symbol, data)

    df_raw = pd.DataFrame(rows, columns=_BASE_AGG_COLUMNS)
    logger.info(
        "Scoring: %d scored, %d filtered out of %d fetched symbols",
        len(rows), len(filtered_stocks), len(fundamentals),
    )

    # Regime-aware scoring: in confirmed-bull regime the composite tilts toward
    # momentum (alpha engine). Best-effort — if regime detection fails, score
    # regime-neutral (no tilt) rather than block the pipeline.
    _regime = None
    try:
        from strategy.regimes.detector import get_current_regime
        _regime = get_current_regime()
        logger.info("scoring under regime=%s", _regime)
    except Exception as exc:
        logger.warning("regime detection failed; scoring regime-neutral: %s", exc)

    compute_metric(df_raw, regime=_regime)

    # Strategy-bucket classification. Every row is seeded "core_candidate" in
    # _compute_stock_row(); reclassify confirmed DOWNTREND names (below their 200-day
    # moving average) as "contrarian_watchlist". This is the only place the two buckets
    # are distinguished — without it strategy_bucket is a constant, which (a) makes the
    # factor-map "candidate centroid" the centroid of the whole cheap universe rather
    # than the genuine momentum-confirmed candidates, and (b) leaves the contrarian
    # soft-penalty + position cap in PortfolioManager permanently inert. Names with no
    # 200DMA coverage (NaN) stay "core_candidate" — we only down-weight a confirmed downtrend.
    if "above_200dma" in df_raw.columns:
        _below_200dma = df_raw["above_200dma"].map(
            lambda v: v is False or v == 0 or str(v).strip().lower() == "false"
        ).fillna(False).astype(bool)
        df_raw.loc[_below_200dma, "strategy_bucket"] = "contrarian_watchlist"
        logger.info(
            "strategy_bucket: %d core_candidate / %d contrarian_watchlist (below 200DMA)",
            int((df_raw["strategy_bucket"] == "core_candidate").sum()),
            int(_below_200dma.sum()),
        )

    vm = df_raw["value_metric"].dropna()
    logger.info(
        "value_metric: mean=%.3f  p25=%.3f  p50=%.3f  p75=%.3f  "
        "≥metric_threshold(%s): %d/%d",
        float(vm.mean()), float(vm.quantile(0.25)),
        float(vm.quantile(0.50)), float(vm.quantile(0.75)),
        METRIC_THRESHOLD, int((vm >= METRIC_THRESHOLD).sum()), len(vm),
    )

    df_raw = _compute_reliability_scores(df_raw)

    try:
        _save_snapshot(df_raw)
    except Exception as _snap_err:
        logger.warning("Snapshot save failed (non-fatal): %s", _snap_err)

    if "sector" in df_raw.columns:
        sector_counts = df_raw["sector"].value_counts()
        logger.info("Universe composition by sector:\n%s", sector_counts.to_string())
    nan_rates = {
        c: round(df_raw[c].isna().mean() * 100, 1)
        for c in ["pe_ratio", "pb_ratio", "return_1m", "return_3m", "rs_3m", "realized_vol_3m"]
        if c in df_raw.columns
    }
    logger.info("NaN rates %%: %s", nan_rates)

    if portfolio_symbols:
        scored = set(df_raw["symbol"].tolist())
        for sym in sorted(portfolio_symbols):
            if sym in scored:
                continue
            if sym not in fundamentals:
                logger.warning(
                    "HELD %s: Robinhood fundamentals API returned no data — "
                    "stock may be delisted, suspended, or unsupported",
                    sym,
                )
            else:
                reason = filtered_stocks.get(sym) or _diagnose_stock_filter(sym, fundamentals[sym])
                logger.warning("HELD %s: scored data missing — %s", sym, reason)

    df_raw["value_metric"] = pd.to_numeric(df_raw["value_metric"], errors="coerce")
    candidates = df_raw[df_raw["value_metric"] >= METRIC_THRESHOLD]["symbol"].tolist()
    if candidates:
        fetch_earnings = EARNINGS_PARAMS.get("enabled", False)
        label = "analyst ratings + earnings" if fetch_earnings else "analyst ratings"
        print(f"Fetching {label} for {len(candidates)} shortlisted stocks...")

        for col in ("value_score", "quality_score", "income_score", "momentum_score"):
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

        def _fetch_one(sym: str) -> tuple[str, float | None, float]:
            ratio = _get_buy_to_sell_ratio(sym)
            bonus = _get_earnings_bonus(sym) if fetch_earnings else 0.0
            time.sleep(0.3)
            return sym, ratio, bonus

        with ThreadPoolExecutor(max_workers=6) as _pool:
            prefetched = list(_pool.map(_fetch_one, candidates))

        sw = SCORE_WEIGHTS
        has_q = "quality_score" in df_raw.columns
        df_raw = df_raw.set_index("symbol", drop=False)

        for sym, ratio, bonus in prefetched:
            if sym not in df_raw.index:
                continue
            df_raw.at[sym, "buy_to_sell_ratio"] = ratio

            if fetch_earnings and has_q and bonus != 0.0:
                old_q = df_raw.at[sym, "quality_score"]
                new_q = (old_q + bonus) if pd.notna(old_q) else bonus
                df_raw.at[sym, "quality_score"] = new_q
                new_vm = (
                    sw.get("value",    0.08) * df_raw.at[sym, "value_score"]
                    + sw.get("quality",  0.50) * new_q
                    + sw.get("income",   0.08) * df_raw.at[sym, "income_score"]
                    + sw.get("momentum", 0.34) * df_raw.at[sym, "momentum_score"]
                )
                df_raw.at[sym, "value_metric"] = new_vm
                logger.debug(
                    "%s: earnings bonus %+.3f → quality=%.3f  value_metric=%.3f",
                    sym, bonus, new_q, new_vm,
                )

        df_raw = df_raw.reset_index(drop=True)

    # Append filtered (unscored) stocks so they remain visible in research UI
    df_raw["is_scored"] = True
    df_raw["filter_reason"] = ""

    if filtered_stocks:
        _frows = []
        for sym, reason in filtered_stocks.items():
            d = fundamentals[sym]
            price = safe_float(
                d.get("current_price") or d.get("last_trade_price") or d.get("adjusted_open_price")
            )
            _frows.append({
                "symbol":        sym,
                "industry":      d.get("industry"),
                "sector":        d.get("sector"),
                "volume":        safe_float(d.get("volume"), 0),
                "pe_ratio":      safe_float(d.get("pe_ratio")),
                "pb_ratio":      safe_float(d.get("pb_ratio")),
                "current_price": price,
                "low_52w":       safe_float(d.get("low_52w")  or d.get("low_52_weeks")),
                "high_52w":      safe_float(d.get("high_52w") or d.get("high_52_weeks")),
                "is_scored":     False,
                "filter_reason": reason,
            })
        df_filtered = (
            pd.DataFrame(_frows)
            .reindex(columns=list(_BASE_AGG_COLUMNS) + ["is_scored", "filter_reason"])
        )
        df_raw = pd.concat([df_raw, df_filtered], ignore_index=True)
        logger.info(
            "Output: %d scored + %d unscored = %d total rows in agg_data",
            len(rows), len(filtered_stocks), len(df_raw),
        )

    _save_cols = list(AGG_DATA_COLUMNS) + ["is_scored", "filter_reason"]
    store_data_as_csv("robinhood_data", _save_cols, df_raw)
    return df_raw


