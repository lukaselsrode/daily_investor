"""data/fundamental_features.py — point-in-time fundamental features from the FMP statement cache.

Computes the peer-3 quality inputs (profitability, cash generation, accruals, margins,
leverage, share-count discipline) and income sustainability inputs (dividend coverage,
growth, streak) strictly as-of a given date: a statement row is usable only if its
`filingDate` was public STRICTLY BEFORE the as-of date (no same-day look-ahead), with
restatements deduped by (fiscalYear, period) keeping the latest filing. The same call
therefore serves today's ETL (`asof=today`) and historical snapshot rescoring.

Also reconstructs valuation ratios (pe/pb/dividend_yield/market_cap) for snapshot
vintages that never stored them, filling ONLY missing cells so Robinhood-sourced
values on healthy vintages are never overwritten.

Read-only over data/fmp_cache_adj/ (never networks). Symbols absent from the cache
stay NaN and the peer machinery drops/renormalizes around them.
"""

from __future__ import annotations

import datetime as _dt
import logging

import numpy as np
import pandas as pd

from .fmp_client import _price_path, statement

logger = logging.getLogger(__name__)

# Quality inputs (statement-derived) + income sustainability inputs (dividend-history).
FUND_FEATURE_COLS = (
    "roe_ttm",
    "gross_margin_ttm",
    "gm_trend_yoy",
    "debt_to_assets",
    "neg_accruals",
    "fcf_to_assets",
    "share_count_shrink_yoy",
    "div_fcf_coverage_ttm",
    "div_growth_1y",
    "div_streak_quarters",
)
_STATEMENT_FEATURES = FUND_FEATURE_COLS[:8]
_DIVIDEND_FEATURES = FUND_FEATURE_COLS[8:]

VALUATION_COLS = ("pe_ratio", "pb_ratio", "dividend_yield", "market_cap")

# A filing older than this at the as-of date is stale (delisted/acquired name still in
# the frame) — treat as uncovered rather than scoring dead fundamentals.
_MAX_FILING_AGE_DAYS = 400

# Payment gaps up to this many days count as "consecutive" (quarterly ~91d with slack;
# semiannual payers still register as consistent).
_DIV_STREAK_MAX_GAP_DAYS = 190
_DIV_STREAK_CAP = 20

_DIV_COVERAGE_CLIP = 5.0

COVERAGE_WARN_THRESHOLD = 0.60


def _asof_iso(asof: _dt.date | str) -> str:
    if isinstance(asof, str):
        return asof[:10]
    return asof.isoformat()


def _dedup_statement(symbol: str, kind: str) -> pd.DataFrame | None:
    """Statement rows with parseable filingDate, restatement-deduped, ascending by filing."""
    try:
        df = statement(symbol, kind, allow_fetch=False)
    except Exception:
        return None
    if df is None or "filingDate" not in df:
        return None
    df = df.copy()
    df["_fd"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df = df[df["_fd"].notna()]
    if df.empty:
        return None
    if "fiscalYear" in df.columns and "period" in df.columns:
        df = df.sort_values("_fd").drop_duplicates(subset=["fiscalYear", "period"], keep="last")
    return df.sort_values("_fd").reset_index(drop=True)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def _ttm(series: pd.Series) -> pd.Series:
    return series.rolling(4).sum()


def _feature_timeline(symbol: str) -> pd.DataFrame | None:
    """Per-filing PIT feature rows for one symbol, ascending by filing date (_fd).

    Carries the eight statement-derived FUND_FEATURE_COLS plus ttm_eps/shares/book
    for valuation reconstruction. None when fewer than 4 distinct quarters exist.
    """
    inc = _dedup_statement(symbol, "income-statement")
    if inc is None or len(inc) < 4:
        return None

    ni = _num(inc, "netIncome")
    rev = _num(inc, "revenue")
    gp = _num(inc, "grossProfit")
    eps = _num(inc, "epsDiluted")
    shs = _num(inc, "weightedAverageShsOutDil")

    ttm_ni = _ttm(ni)
    ttm_rev = _ttm(rev)
    gm_ttm = (_ttm(gp) / ttm_rev.where(ttm_rev > 0)).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame({
        "_fd": inc["_fd"].to_numpy(),
        "ttm_ni": ttm_ni.to_numpy(),
        "ttm_eps": _ttm(eps).to_numpy(),
        "shares": shs.to_numpy(),
        "gross_margin_ttm": gm_ttm.to_numpy(),
        "gm_trend_yoy": (gm_ttm - gm_ttm.shift(4)).to_numpy(),
        "share_count_shrink_yoy": (-(shs - shs.shift(4)) / shs.shift(4).abs()).to_numpy(),
    })

    cf = _dedup_statement(symbol, "cash-flow-statement")
    if cf is not None:
        # commonDividendsPaid is a negative outflow → negate to get +$ paid.
        cser = pd.DataFrame({
            "_cfd": cf["_fd"].to_numpy(),
            "ttm_cfo": _ttm(_num(cf, "operatingCashFlow")).to_numpy(),
            "ttm_fcf": _ttm(_num(cf, "freeCashFlow")).to_numpy(),
            "ttm_divpaid": _ttm(-_num(cf, "commonDividendsPaid")).to_numpy(),
        })
        out = pd.merge_asof(
            out.sort_values("_fd"), cser.sort_values("_cfd"),
            left_on="_fd", right_on="_cfd", direction="backward",
        ).drop(columns=["_cfd"])
    else:
        out["ttm_cfo"] = np.nan
        out["ttm_fcf"] = np.nan
        out["ttm_divpaid"] = np.nan

    bal = _dedup_statement(symbol, "balance-sheet-statement")
    if bal is not None:
        bser = pd.DataFrame({
            "_bfd": bal["_fd"].to_numpy(),
            "assets": _num(bal, "totalAssets").to_numpy(),
            "book": _num(bal, "totalStockholdersEquity").to_numpy(),
            "debt": _num(bal, "totalDebt").to_numpy(),
        })
        out = pd.merge_asof(
            out.sort_values("_fd"), bser.sort_values("_bfd"),
            left_on="_fd", right_on="_bfd", direction="backward",
        ).drop(columns=["_bfd"])
    else:
        out["assets"] = np.nan
        out["book"] = np.nan
        out["debt"] = np.nan

    assets = out["assets"].where(out["assets"] > 0)
    book = out["book"].where(out["book"] > 0)
    out["roe_ttm"] = out["ttm_ni"] / book
    out["debt_to_assets"] = out["debt"] / assets
    out["neg_accruals"] = -((out["ttm_ni"] - out["ttm_cfo"]) / assets)
    out["fcf_to_assets"] = out["ttm_fcf"] / assets
    out["div_fcf_coverage_ttm"] = (
        out["ttm_fcf"] / out["ttm_divpaid"].where(out["ttm_divpaid"] > 0)
    ).clip(lower=0.0, upper=_DIV_COVERAGE_CLIP)
    return out


class FundamentalsCache:
    """Lazy per-symbol loader of PIT feature timelines, dividend records, and closes.

    One instance should be shared across a whole migration run — repeated
    construction rereads thousands of statement JSONs.
    """

    def __init__(self) -> None:
        self._timelines: dict[str, pd.DataFrame | None] = {}
        self._dividends: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._closes: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def timeline(self, symbol: str) -> pd.DataFrame | None:
        if not symbol or not isinstance(symbol, str):
            return None
        if symbol not in self._timelines:
            self._timelines[symbol] = _feature_timeline(symbol)
        return self._timelines[symbol]

    def dividends(self, symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(ex_dates datetime64 ndarray ascending, split-adjusted amounts) or None."""
        if not symbol or not isinstance(symbol, str):
            return None
        if symbol not in self._dividends:
            self._dividends[symbol] = self._load_dividends(symbol)
        return self._dividends[symbol]

    @staticmethod
    def _load_dividends(symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            dv = statement(symbol, "dividends", allow_fetch=False)
        except Exception:
            return None
        if dv is None or "date" not in dv:
            return None
        d = dv.copy()
        d["_d"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["_d"]).sort_values("_d")
        if d.empty:
            return None
        amt_col = "adjDividend" if "adjDividend" in d.columns else "dividend"
        amounts = pd.to_numeric(d[amt_col], errors="coerce").fillna(0.0).to_numpy()
        return d["_d"].to_numpy(), amounts

    def close_asof(self, symbol: str, asof_iso: str) -> float:
        """Last cached close dated <= asof, NaN when uncached/empty."""
        if not symbol or not isinstance(symbol, str):
            return float("nan")
        if symbol not in self._closes:
            self._closes[symbol] = self._load_closes(symbol)
        loaded = self._closes[symbol]
        if loaded is None:
            return float("nan")
        dates, close = loaded
        end = int(np.searchsorted(dates, asof_iso, side="right"))
        return float(close[end - 1]) if end > 0 else float("nan")

    @staticmethod
    def _load_closes(symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            df = pd.read_parquet(_price_path(symbol), columns=["close"])
        except (FileNotFoundError, OSError):
            return None
        except Exception:
            logger.warning("fundamental_features: unreadable price cache for %s", symbol)
            return None
        if df.empty:
            return None
        return df.index.to_numpy(dtype=str), df["close"].to_numpy(dtype=float)


def dividend_streak_at_ex(ex_dates: np.ndarray) -> np.ndarray:
    """Streak value AS OF each ex-date (for vectorized panel lookups).

    streak[i] = 1 when the gap from the previous ex-date exceeds the max gap,
    else min(streak[i-1] + 1, cap). A caller resolving "streak as of date a"
    takes streak[last ex-date < a], zeroed when a - last_ex exceeds the gap.
    """
    n = len(ex_dates)
    streak = np.ones(n)
    gap = np.timedelta64(_DIV_STREAK_MAX_GAP_DAYS, "D")
    for i in range(1, n):
        streak[i] = 1.0 if (ex_dates[i] - ex_dates[i - 1]) > gap else min(
            streak[i - 1] + 1.0, float(_DIV_STREAK_CAP)
        )
    return streak


def _dividend_history_features(
    records: tuple[np.ndarray, np.ndarray], asof: pd.Timestamp
) -> tuple[float, float]:
    """(div_growth_1y, div_streak_quarters) from ex-dates strictly before asof."""
    ex, amt = records
    a64 = np.datetime64(asof)
    y1 = a64 - np.timedelta64(365, "D")
    y2 = a64 - np.timedelta64(730, "D")

    t1 = float(amt[(ex >= y1) & (ex < a64)].sum())
    t0 = float(amt[(ex >= y2) & (ex < y1)].sum())
    growth = (t1 / t0 - 1.0) if (t0 > 0 and t1 > 0) else np.nan

    before = ex[ex < a64]
    if len(before) == 0:
        return growth, np.nan
    gap_limit = np.timedelta64(_DIV_STREAK_MAX_GAP_DAYS, "D")
    if a64 - before[-1] > gap_limit:
        return growth, 0.0  # payer that stopped — consistency is zero, not unknown
    streak = 1
    for i in range(len(before) - 1, 0, -1):
        if streak >= _DIV_STREAK_CAP:
            break
        if before[i] - before[i - 1] > gap_limit:
            break
        streak += 1
    return growth, float(min(streak, _DIV_STREAK_CAP))


def _timeline_row_asof(
    tl: pd.DataFrame | None, a_ts: pd.Timestamp, max_age_days: int = _MAX_FILING_AGE_DAYS
) -> pd.Series | None:
    """Latest timeline row filed STRICTLY before a_ts and not staler than max_age_days."""
    if tl is None or tl.empty:
        return None
    fd = tl["_fd"].to_numpy()
    idx = int(np.searchsorted(fd, np.datetime64(a_ts), side="left"))
    if idx == 0:
        return None
    row = tl.iloc[idx - 1]
    if (a_ts - pd.Timestamp(row["_fd"])).days > max_age_days:
        return None
    return row


def compute_fundamental_features_asof(
    symbols: list[str],
    asof: _dt.date | str,
    cache: FundamentalsCache,
) -> pd.DataFrame:
    """Per-symbol FUND_FEATURE_COLS using only filings/ex-dates strictly before asof.

    Index = symbol, columns = FUND_FEATURE_COLS. All-NaN row when the symbol is
    uncached, has < 4 distinct quarters, or its latest filing is stale.
    """
    a_ts = pd.Timestamp(_asof_iso(asof))
    rows: dict[str, list[float]] = {}
    for sym in symbols:
        vals = {col: np.nan for col in FUND_FEATURE_COLS}
        row = _timeline_row_asof(cache.timeline(sym), a_ts)
        if row is not None:
            for col in _STATEMENT_FEATURES:
                v = row[col]
                vals[col] = float(v) if pd.notna(v) else np.nan
        records = cache.dividends(sym)
        if records is not None:
            vals["div_growth_1y"], vals["div_streak_quarters"] = _dividend_history_features(
                records, a_ts
            )
        rows[sym] = [vals[c] for c in FUND_FEATURE_COLS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(FUND_FEATURE_COLS))


def add_fundamental_features(
    df: pd.DataFrame,
    asof: _dt.date | str,
    cache: FundamentalsCache | None = None,
) -> float:
    """Merge FUND_FEATURE_COLS into df (in place, keyed on df['symbol']); returns coverage.

    Coverage = fraction of rows with a non-null roe_ttm (the headline profitability
    input); the dividend-history columns are payer-only by construction and do not
    count toward coverage.
    """
    if "symbol" not in df.columns or df.empty:
        for col in FUND_FEATURE_COLS:
            df[col] = np.nan
        return 0.0

    cache = cache if cache is not None else FundamentalsCache()
    symbols = ["" if pd.isna(s) else str(s) for s in df["symbol"].tolist()]
    feats = compute_fundamental_features_asof(symbols, asof, cache)
    aligned = feats.reindex(symbols)
    for col in FUND_FEATURE_COLS:
        df[col] = aligned[col].to_numpy()

    coverage = float(df["roe_ttm"].notna().mean())
    log = logger.warning if coverage < COVERAGE_WARN_THRESHOLD else logger.info
    log(
        "fundamental_features: asof=%s coverage=%.1f%% (n=%d)",
        _asof_iso(asof), coverage * 100, len(df),
    )
    return coverage


def add_valuation_ratios_asof(
    df: pd.DataFrame,
    asof: _dt.date | str,
    cache: FundamentalsCache | None = None,
    fill_missing_only: bool = True,
) -> float:
    """Reconstruct pe_ratio/pb_ratio/dividend_yield/market_cap from FMP statements + closes.

    For snapshot vintages that never stored broker ratios. With fill_missing_only=True
    (default) only NaN cells are written, so Robinhood-sourced values on healthy
    vintages are never overwritten. Price = the frame's current_price when positive,
    else the cached close as-of. Returns the post-fill pe_ratio coverage fraction.
    """
    if "symbol" not in df.columns or df.empty:
        return 0.0

    cache = cache if cache is not None else FundamentalsCache()
    iso = _asof_iso(asof)
    a_ts = pd.Timestamp(iso)

    for col in VALUATION_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    px_frame = (
        pd.to_numeric(df["current_price"], errors="coerce")
        if "current_price" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    new_vals: dict[str, list[float]] = {col: [] for col in VALUATION_COLS}
    for i in df.index:
        sym = df.at[i, "symbol"]
        sym = "" if pd.isna(sym) else str(sym)
        price = float(px_frame.at[i]) if pd.notna(px_frame.at[i]) and px_frame.at[i] > 0 else (
            cache.close_asof(sym, iso)
        )
        pe = pb = mcap = dy = np.nan
        if np.isfinite(price) and price > 0:
            row = _timeline_row_asof(cache.timeline(sym), a_ts)
            if row is not None:
                ttm_eps = float(row["ttm_eps"]) if pd.notna(row["ttm_eps"]) else np.nan
                shares = float(row["shares"]) if pd.notna(row["shares"]) else np.nan
                book = float(row["book"]) if pd.notna(row["book"]) else np.nan
                if np.isfinite(ttm_eps) and ttm_eps > 0:
                    pe = price / ttm_eps
                if np.isfinite(shares) and shares > 0:
                    mcap = price * shares
                    if np.isfinite(book) and book > 0:
                        pb = mcap / book
            records = cache.dividends(sym)
            if records is not None:
                ex, amt = records
                a64 = np.datetime64(a_ts)
                ttm_div = float(amt[(ex >= a64 - np.timedelta64(365, "D")) & (ex < a64)].sum())
                dy = ttm_div / price if ttm_div > 0 else 0.0
            else:
                dy = 0.0  # no dividend history in cache → non-payer
        new_vals["pe_ratio"].append(pe)
        new_vals["pb_ratio"].append(pb)
        new_vals["dividend_yield"].append(dy)
        new_vals["market_cap"].append(mcap)

    for col in VALUATION_COLS:
        computed = pd.Series(new_vals[col], index=df.index)
        if fill_missing_only:
            df[col] = df[col].where(df[col].notna(), computed)
        else:
            df[col] = computed

    coverage = float(df["pe_ratio"].notna().mean())
    logger.info(
        "valuation_ratios: asof=%s pe coverage=%.1f%% (n=%d, fill_missing_only=%s)",
        iso, coverage * 100, len(df), fill_missing_only,
    )
    return coverage
