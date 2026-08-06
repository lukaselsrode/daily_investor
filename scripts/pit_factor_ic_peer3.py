"""
scripts/pit_factor_ic_peer3.py — Gate V1: PIT factor IC, peer-2 vs peer-3 quality/income.

Survivorship-free, point-in-time comparison over the alpha-discovery monthly panel
(.session_tmp/alpha_discovery/out/poc_panel.parquet: ~6.7k symbols × 204 business
month-ends, 2009→2025, forward SPY-excess returns at 21/63/126d).

For every (symbol, date) row the peer-3 fundamental features are reconstructed from
the FMP statement cache via data.fundamental_features timelines (filingDate strictly
before the as-of date, live staleness guard). Factor scores are per-date
cross-sectional weighted rank composites (market tier — the peer-industry machinery
is invisible to IC direction at this granularity):

  quality_new : FUND components at live _COMPONENT_WEIGHTS
  quality_old : peer-2 reconstruction — 30% weighted ranks of {has_positive_pe,
                has_positive_pb, no_distress_pe, healthy_yield, log market cap}
                (volume/analyst components unavailable on this panel → renormalized,
                exactly the peer-2 sparse-frame behavior) + 70% legacy checklist
                (volume terms skipped).
  income_new  : payers-only weighted ranks of yield/coverage/growth/streak
  income_old  : max(payer yield rank scaled to [-1,1.5], dy/0.03 capped 1.5)

PASS criteria (per the peer-3 plan):
  - quality_new mean Spearman IC >= quality_old at 63d, full sample AND both halves
  - income_new IC >= income_old, and the max single-value mass among nonzero payer
    scores collapses vs the old cap-pinning

Research-only: reads caches, writes nothing.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")

PANEL = ".session_tmp/alpha_discovery/out/poc_panel.parquet"
HORIZONS = ("fwd_exc_21d", "fwd_exc_63d", "fwd_exc_126d")
MIN_NAMES_PER_DATE = 300

QUALITY_NEW_WEIGHTS = None  # loaded from live module below
INCOME_NEW_WEIGHTS = None


def _rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1], NaN preserved."""
    return s.rank(pct=True)


def _weighted_rank_composite(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Per-row NaN-renormalized weighted mean of column pct-ranks."""
    numer = pd.Series(0.0, index=frame.index)
    denom = pd.Series(0.0, index=frame.index)
    for col, w in weights.items():
        if col not in frame.columns:
            continue
        r = _rank(frame[col])
        valid = r.notna()
        numer += w * r.fillna(0.0)
        denom += w * valid.astype(float)
    return numer / denom.where(denom > 0)


def build_feature_matrix(symbols: list[str], grid: np.ndarray) -> dict[str, np.ndarray]:
    """(n_dates, n_symbols) matrices of the peer-3 features + dy, via one timeline
    read per symbol (vectorized searchsorted over the date grid)."""
    from data.fundamental_features import (
        _DIV_STREAK_MAX_GAP_DAYS,
        _MAX_FILING_AGE_DAYS,
        FundamentalsCache,
        dividend_streak_at_ex,
    )

    feats = ("roe_ttm", "gross_margin_ttm", "gm_trend_yoy", "debt_to_assets",
             "neg_accruals", "fcf_to_assets", "share_count_shrink_yoy",
             "div_fcf_coverage_ttm")
    n_d, n_s = len(grid), len(symbols)
    out = {c: np.full((n_d, n_s), np.nan) for c in feats}
    out["ttm_div"] = np.zeros((n_d, n_s))
    out["div_growth_1y"] = np.full((n_d, n_s), np.nan)
    out["div_streak_quarters"] = np.full((n_d, n_s), np.nan)

    cache = FundamentalsCache()
    max_age = np.timedelta64(_MAX_FILING_AGE_DAYS, "D")
    gap = np.timedelta64(_DIV_STREAK_MAX_GAP_DAYS, "D")
    t0 = time.time()
    for j, sym in enumerate(symbols):
        tl = cache.timeline(sym)
        if tl is not None and len(tl):
            fd = pd.to_datetime(tl["_fd"]).to_numpy()
            pos = np.searchsorted(fd, grid, side="left") - 1
            valid = pos >= 0
            safe = np.clip(pos, 0, len(fd) - 1)
            fresh = valid & ((grid - fd[safe]) <= max_age)
            for c in feats:
                if c in tl.columns:
                    out[c][fresh, j] = tl[c].to_numpy()[safe[fresh]]
        rec = cache.dividends(sym)
        if rec is not None:
            ex, amt = rec
            cum = np.concatenate([[0.0], np.cumsum(amt)])
            hi = np.searchsorted(ex, grid, side="left")
            lo1 = np.searchsorted(ex, grid - np.timedelta64(365, "D"), side="left")
            lo2 = np.searchsorted(ex, grid - np.timedelta64(730, "D"), side="left")
            t1 = cum[hi] - cum[lo1]
            t0_ = cum[lo1] - cum[lo2]
            out["ttm_div"][:, j] = t1
            with np.errstate(divide="ignore", invalid="ignore"):
                out["div_growth_1y"][:, j] = np.where(
                    (t0_ > 0) & (t1 > 0), t1 / t0_ - 1.0, np.nan
                )
            streaks = dividend_streak_at_ex(ex)
            last = hi - 1
            has = last >= 0
            sl = np.clip(last, 0, len(ex) - 1)
            recent = has & ((grid - ex[sl]) <= gap)
            out["div_streak_quarters"][recent, j] = streaks[sl[recent]]
            out["div_streak_quarters"][has & ~recent, j] = 0.0
        if (j + 1) % 1000 == 0:
            print(f"  ...{j + 1}/{n_s} symbols, {time.time() - t0:.0f}s", flush=True)
    # Free cached timelines before the big merge.
    return out


def per_date_scores(g: pd.DataFrame) -> pd.DataFrame:
    """Score one date's cross-section: quality/income, old and new."""
    from strategy.scoring.income import _DEFAULT_WEIGHTS as INC_W
    from strategy.scoring.quality import _COMPONENT_WEIGHTS as Q_W

    out = pd.DataFrame(index=g.index)

    # ── quality NEW: fundamentals composite at live weights ────────────────────
    frame = g.copy()
    frame["low_leverage"] = -frame["debt_to_assets"]
    out["quality_new"] = _weighted_rank_composite(frame, dict(Q_W))

    # ── quality OLD (peer-2 reconstruction) ────────────────────────────────────
    pe_pos = g["earnings_yield"] > 0
    pb_pos = g["book"] > 0
    # PE < 5  ⇔  earnings_yield > 0.2
    distress = pe_pos & (g["earnings_yield"] > 0.2)
    dy = g["dy"]
    healthy = (dy >= 0.02) & (dy <= 0.06)
    trap = dy >= 0.10
    comp = pd.DataFrame({
        "has_positive_pe": pe_pos.astype(float),
        "has_positive_pb": pb_pos.astype(float),
        "no_distress_pe": (~distress).astype(float),
        "healthy_yield": healthy.astype(float),
        "market_cap": g["log_mcap"],
    })
    peer_part = _weighted_rank_composite(comp, {
        "has_positive_pe": 0.20, "has_positive_pb": 0.10, "no_distress_pe": 0.20,
        "healthy_yield": 0.10, "market_cap": 0.05,
    })
    checklist = (
        0.5 * pe_pos.astype(float)
        - 0.4 * distress.astype(float)
        + 0.2 * pb_pos.astype(float)
        - 0.6 * trap.astype(float)
        + 0.2 * (healthy & ~trap).astype(float)
    )
    out["quality_old"] = 0.7 * checklist + 0.3 * peer_part

    # ── income NEW: payers-only weighted ranks ─────────────────────────────────
    payer = dy > 0
    rankable = payer & ~trap
    inc_frame = pd.DataFrame({
        "dividend_yield": dy.where(rankable),
        "div_fcf_coverage": g["div_fcf_coverage_ttm"].where(rankable),
        "div_growth": g["div_growth_1y"].where(rankable),
        "div_streak": g["div_streak_quarters"].where(rankable),
    })
    inc_new = _weighted_rank_composite(inc_frame, dict(INC_W))
    out["income_new"] = inc_new.fillna(0.0).where(rankable, 0.0)

    # ── income OLD: max(rank, capped yield/threshold) ──────────────────────────
    yr = _rank(dy.where(rankable)) * 2.5 - 1.0        # [-1, 1.5] scale
    capped = (dy / 0.03).clip(upper=1.5)
    inc_old = pd.concat([yr.fillna(0.0), capped.where(payer, 0.0)], axis=1).max(axis=1)
    out["income_old"] = inc_old.where(~trap, 0.0).where(payer, 0.0)
    return out


def ic_table(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    dates = np.sort(panel["date"].unique())
    half = dates[len(dates) // 2]
    for factor in factors:
        for hz in HORIZONS:
            ics, ics_h1, ics_h2 = [], [], []
            for d, g in panel.groupby("date"):
                ok = g[factor].notna() & g[hz].notna()
                if ok.sum() < MIN_NAMES_PER_DATE:
                    continue
                ic = spearmanr(g.loc[ok, factor], g.loc[ok, hz])[0]
                ics.append(ic)
                (ics_h1 if d < half else ics_h2).append(ic)
            arr = np.array(ics)
            rows.append({
                "factor": factor, "horizon": hz, "n_dates": len(arr),
                "mean_ic": arr.mean(), "t_stat": arr.mean() / (arr.std() / np.sqrt(len(arr))),
                "mean_ic_h1": np.mean(ics_h1) if ics_h1 else np.nan,
                "mean_ic_h2": np.mean(ics_h2) if ics_h2 else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    symbols = sorted(panel["symbol"].unique())
    grid = np.sort(panel["date"].unique())
    print(f"panel: {len(panel):,} rows, {len(symbols)} symbols, {len(grid)} dates")

    mats = build_feature_matrix(symbols, grid)
    sym_idx = {s: j for j, s in enumerate(symbols)}
    date_idx = {d: i for i, d in enumerate(grid)}
    ii = panel["date"].map(date_idx).to_numpy()
    jj = panel["symbol"].map(sym_idx).to_numpy()
    for c, mat in mats.items():
        panel[c] = mat[ii, jj]
    panel["dy"] = np.where(panel["entry_close"] > 0, panel["ttm_div"] / panel["entry_close"], 0.0)
    print(f"features merged ({time.time() - t0:.0f}s)")

    scored = []
    for _d, g in panel.groupby("date"):
        s = per_date_scores(g)
        s["date"] = _d
        for hz in HORIZONS:
            s[hz] = g[hz]
        scored.append(s)
    scored = pd.concat(scored)
    print(f"scored ({time.time() - t0:.0f}s)")

    table = ic_table(scored, ["quality_new", "quality_old", "income_new", "income_old"])
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print("\n=== PIT Spearman IC vs forward SPY-excess (per-date mean) ===")
    print(table.to_string(index=False))

    # degeneracy check
    for name in ("income_new", "income_old"):
        nz = scored[name][scored[name] != 0.0]
        mass = nz.round(4).value_counts().iloc[0] / max(len(nz), 1)
        print(f"\n{name}: max single-value mass among NONZERO scores = {mass:.2%}")

    q63 = {r["factor"]: r for _, r in table[table.horizon == "fwd_exc_63d"].iterrows()}
    ok_q = (
        q63["quality_new"]["mean_ic"] >= q63["quality_old"]["mean_ic"]
        and q63["quality_new"]["mean_ic_h1"] >= q63["quality_old"]["mean_ic_h1"]
        and q63["quality_new"]["mean_ic_h2"] >= q63["quality_old"]["mean_ic_h2"]
    )
    ok_i = q63["income_new"]["mean_ic"] >= q63["income_old"]["mean_ic"]
    print(f"\nGATE V1 quality (63d, full + both halves): {'PASS' if ok_q else 'FAIL'}")
    print(f"GATE V1 income  (63d):                      {'PASS' if ok_i else 'FAIL'}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
