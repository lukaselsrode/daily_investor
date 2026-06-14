"""
data/pit_fundamentals.py — Point-in-time fundamentals from raw FMP statements.

Reconstructs CAUSAL valuation/quality/income factors for any symbol as-of any historical
date, using only company filings whose `filingDate` was public STRICTLY BEFORE `asof`
(no same-day look-ahead) and the survivorship-free split-adjusted price. This is the
production home (the `data/` layer owns FMP access); `research/pit_fundamentals.py` is a
thin re-export so research/offline code keeps working without production depending on it.

Derivations (all causal):
  TTM EPS   = Σ epsDiluted over the last 4 DISTINCT fiscal quarters filed before `asof`
              (deduped by (fiscalYear, period) so restatements/amendments don't double-count)
  shares    = netIncome / epsDiluted of the latest filed quarter      → market_cap = price × shares
  book      = totalStockholdersEquity of the latest balance filed     → PB = market_cap / book
  quality   = TTM ROE (NI/equity), net & gross margin, leverage (debt/assets)
  income    = TTM cash dividends (ex-date strictly before `asof`) / price   → dividend yield

Cache-only (allow_fetch=False) — never hits the network. Returns None / 0.0 on missing data
so the caller can neutral-score exactly as the live path does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import fmp_client as fmp


def _causal_income(symbol: str, asof: str) -> pd.DataFrame | None:
    """Income-statement rows filed STRICTLY before `asof`, deduped to the most recent
    filing per (fiscalYear, period), newest first. None if fewer than 4 distinct quarters."""
    inc = fmp.statement(symbol, "income-statement", allow_fetch=False)
    if inc is None or "filingDate" not in inc:
        return None
    inc = inc.copy()
    inc["_fd"] = pd.to_datetime(inc["filingDate"], errors="coerce")
    a = pd.Timestamp(asof)
    inc = inc[inc["_fd"].notna() & (inc["_fd"] < a)]
    if inc.empty:
        return None
    # Dedup by fiscal period (keep the latest-filed version → handles restatements).
    if "fiscalYear" in inc.columns and "period" in inc.columns:
        inc = inc.sort_values("_fd").drop_duplicates(subset=["fiscalYear", "period"], keep="last")
    inc = inc.sort_values("_fd", ascending=False)
    if len(inc) < 4:
        return None
    return inc


def fundamentals_asof(symbol: str, asof: str, price: float) -> dict | None:
    """Causal PE/PB/market_cap/ROE/margins/leverage for `symbol` as-of `asof` (YYYY-MM-DD).

    Returns None when statements are unavailable or fewer than 4 distinct quarters were
    filed strictly before `asof`. Cache-only.
    """
    inc = _causal_income(symbol, asof)
    if inc is None:
        return None
    l4 = inc.head(4)

    def _sum(col):
        return float(l4[col].astype(float).sum()) if col in l4 else np.nan

    ttm_eps = _sum("epsDiluted")
    ttm_ni  = _sum("netIncome")
    ttm_rev = _sum("revenue")
    ttm_gp  = _sum("grossProfit")

    q = l4.iloc[0]
    eps_q = float(q.get("epsDiluted") or 0.0)
    shares = (float(q.get("netIncome") or 0.0) / eps_q) if eps_q else np.nan
    mcap = price * shares if np.isfinite(shares) else np.nan

    equity = debt = assets = np.nan
    bal = fmp.statement(symbol, "balance-sheet-statement", allow_fetch=False)
    if bal is not None and "filingDate" in bal:
        bal = bal.copy()
        bal["_fd"] = pd.to_datetime(bal["filingDate"], errors="coerce")
        b = bal[bal["_fd"].notna() & (bal["_fd"] < pd.Timestamp(asof))].sort_values(
            "_fd", ascending=False
        )
        if len(b):
            equity = float(b.iloc[0].get("totalStockholdersEquity") or np.nan)
            debt   = float(b.iloc[0].get("totalDebt") or np.nan)
            assets = float(b.iloc[0].get("totalAssets") or np.nan)

    pos = lambda x: x if (np.isfinite(x) and x > 0) else np.nan  # noqa: E731
    return {
        "pe":           price / ttm_eps if ttm_eps > 0 else np.nan,
        "pb":           mcap / pos(equity) if np.isfinite(mcap) else np.nan,
        "market_cap":   mcap,
        "roe":          ttm_ni / pos(equity),
        "net_margin":   ttm_ni / pos(ttm_rev),
        "gross_margin": ttm_gp / pos(ttm_rev),
        "leverage":     debt / pos(assets) if np.isfinite(debt) else np.nan,
        "ttm_eps":      ttm_eps,
    }


def causal_ttm_series(symbol: str) -> pd.DataFrame | None:
    """Per-filing causal TTM aggregates for `symbol` (cache-only), for vectorized PIT panels.

    Returns a DataFrame sorted ascending by filing date with columns:
      _fd       : filing date (datetime) — the date this row's data became public
      ttm_eps   : Σ epsDiluted over the trailing 4 distinct quarters up to that filing
      shares    : netIncome/epsDiluted of that filing's quarter (diluted share count)
      book      : totalStockholdersEquity from the latest balance filed by that date
    The caller looks up the row with `_fd` STRICTLY before an as-of date (no same-day
    look-ahead). None if fewer than 4 distinct quarters were ever filed.
    """
    inc = fmp.statement(symbol, "income-statement", allow_fetch=False)
    if inc is None or "filingDate" not in inc:
        return None
    inc = inc.copy()
    inc["_fd"] = pd.to_datetime(inc["filingDate"], errors="coerce")
    inc = inc[inc["_fd"].notna()]
    if "fiscalYear" in inc.columns and "period" in inc.columns:
        inc = inc.sort_values("_fd").drop_duplicates(subset=["fiscalYear", "period"], keep="last")
    inc = inc.sort_values("_fd")
    if len(inc) < 4:
        return None
    eps = pd.to_numeric(inc.get("epsDiluted"), errors="coerce")
    ni  = pd.to_numeric(inc.get("netIncome"), errors="coerce")
    out = pd.DataFrame({
        "_fd": inc["_fd"].to_numpy(),
        "ttm_eps": eps.rolling(4).sum().to_numpy(),       # trailing 4 quarters incl. this filing
        "shares": (ni / eps.replace(0, np.nan)).to_numpy(),
    })
    bal = fmp.statement(symbol, "balance-sheet-statement", allow_fetch=False)
    if bal is not None and "filingDate" in bal:
        bal = bal.copy()
        bal["_bfd"] = pd.to_datetime(bal["filingDate"], errors="coerce")
        bal = bal.dropna(subset=["_bfd"]).sort_values("_bfd")
        bser = pd.DataFrame({
            "_bfd": bal["_bfd"].to_numpy(),
            "book": pd.to_numeric(bal.get("totalStockholdersEquity"), errors="coerce").to_numpy(),
        }).dropna()
        if len(bser):
            out = pd.merge_asof(
                out.sort_values("_fd"), bser.sort_values("_bfd"),
                left_on="_fd", right_on="_bfd", direction="backward",
            ).drop(columns=["_bfd"])
    if "book" not in out.columns:
        out["book"] = np.nan
    out = out.dropna(subset=["ttm_eps"]).reset_index(drop=True)
    return out if len(out) else None


def dividend_records(symbol: str):
    """Sorted (ex_dates ndarray[datetime64], amounts ndarray[float]) for `symbol`, or None.
    Cache-only — for vectorized causal TTM-dividend lookups (ex-date strictly before as-of)."""
    dv = fmp.statement(symbol, "dividends", allow_fetch=False)
    if dv is None or "date" not in dv or "dividend" not in dv:
        return None
    d = dv.copy()
    d["_d"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["_d"]).sort_values("_d")
    if d.empty:
        return None
    return d["_d"].to_numpy(), pd.to_numeric(d["dividend"], errors="coerce").fillna(0.0).to_numpy()


def dividend_yield_asof(symbol: str, asof: str, price: float) -> float:
    """Causal TTM cash-dividend yield as-of `asof`: Σ dividends with ex-date in
    [asof-365d, asof) divided by `price`. Strictly before `asof` (no same-day look-ahead).
    Returns 0.0 for non-payers / missing data / non-positive price — matching the live
    income factor's treatment of zero-yield names. Cache-only.
    """
    if price is None or not np.isfinite(price) or price <= 0:
        return 0.0
    dv = fmp.statement(symbol, "dividends", allow_fetch=False)
    if dv is None or "date" not in dv or "dividend" not in dv:
        return 0.0
    dv = dv.copy()
    dv["_d"] = pd.to_datetime(dv["date"], errors="coerce")
    a = pd.Timestamp(asof)
    win = dv[dv["_d"].notna() & (dv["_d"] < a) & (dv["_d"] >= a - pd.Timedelta(days=365))]
    if win.empty:
        return 0.0
    ttm_div = float(pd.to_numeric(win["dividend"], errors="coerce").fillna(0.0).sum())
    return ttm_div / price if ttm_div > 0 else 0.0
