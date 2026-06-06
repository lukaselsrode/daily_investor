"""
research/pit_fundamentals.py — Point-in-time fundamentals from raw FMP statements.

Reconstructs CAUSAL valuation/quality factors for any symbol as-of any historical date, using
only company filings whose `filingDate` was public by then (no look-ahead) and the
survivorship-free split-adjusted price. This is what lets us rebuild honest historical snapshots
to feed the IC engine years of data instead of the ~5 weeks the live snapshot store holds.

Derivations (all causal):
  TTM EPS   = Σ epsDiluted over the last 4 quarters filed by `asof`   → PE = price / TTM_EPS
  shares    = netIncome / epsDiluted of the latest filed quarter      → market_cap = price × shares
  book      = totalStockholdersEquity of the latest balance filed     → PB = market_cap / book
  quality   = TTM ROE (NI/equity), net & gross margin, leverage (debt/assets)

Validated on AAPL: PE 31→20→29→31 across 2022-2025 with market caps matching reality.
Requires the statement cache (data/fmp_cache_adj/statements/) — see fmp_client.statement().
"""
from __future__ import annotations

import numpy as np

from data import fmp_client as fmp


def fundamentals_asof(symbol: str, asof: str, price: float) -> dict | None:
    """Causal PE/PB/market_cap/ROE/margins/leverage for `symbol` as-of `asof` (YYYY-MM-DD).

    Returns None when statements are unavailable or fewer than 4 quarters were filed by `asof`.
    Cache-only (allow_fetch=False) — backfill statements first with fmp_client.statement().
    """
    inc = fmp.statement(symbol, "income-statement", allow_fetch=False)
    if inc is None or "filingDate" not in inc:
        return None
    inc = inc[inc["filingDate"] <= asof].sort_values("filingDate", ascending=False)
    if len(inc) < 4:
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

    equity = np.nan
    bal = fmp.statement(symbol, "balance-sheet-statement", allow_fetch=False)
    if bal is not None and "filingDate" in bal:
        b = bal[bal["filingDate"] <= asof].sort_values("filingDate", ascending=False)
        if len(b):
            equity = float(b.iloc[0].get("totalStockholdersEquity") or np.nan)
            debt   = float(b.iloc[0].get("totalDebt") or np.nan)
            assets = float(b.iloc[0].get("totalAssets") or np.nan)
        else:
            debt = assets = np.nan
    else:
        debt = assets = np.nan

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
