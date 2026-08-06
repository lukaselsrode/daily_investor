"""
backtesting/pit_precompute.py — Point-in-time factor panels for the survivorship-free path.

Builds per-day 2D arrays (n_days × n_symbols) of pe_comp / pb_comp / value_penalty /
quality_score / income_score scored CAUSALLY: at each rebalance date the value/quality/
income factors are computed from fundamentals filed STRICTLY BEFORE that date
(data.fundamental_features timelines) and the as-of price, ranked cross-sectionally over
the names tradeable that day using the production peer scorers (strategy.scoring) — never
the current ratios.yaml sector baselines. Results are forward-filled to every day until
the next rebalance.

Value parity with live (strategy.scoring.value.apply_value): the pe_comp/pb_comp panels
carry the per-ratio anchor blend (anchor_blend × cross-sectional rank + (1−a) × peer
rank), the missing-ratio cross-fill (a name with only PE present scores its PE component
at any pe_weight mix), and the neither-ratio floor; distress/negative-EPS penalties are
emitted as a separate ADDITIVE panel so the simulator's `pe_w·pe + (1−pe_w)·pb − penalty`
reproduces the live factor for any tuned pe_weight. One intentional residual divergence:
value.benchmark_blend (the cfg/ratios.yaml sector anchor) stays live-only — forward-IC is
that knob's validation surface. Live's inner clamp before anchor blending is folded into
a single final clamp (differs only for deeply-distressed names past the clamp boundary).

peer-3 quality/income read the fundamental feature columns (ROE, FCF/assets, accruals,
margins, leverage, share discipline, dividend sustainability), reconstructed as-of each
rebalance date from the same FundamentalsCache timelines. Missing fundamentals
neutral-score exactly as the live path treats missing data. Cache-only — no network.

Cost model: O(n_symbols) statement-cache reads (once each), then vectorized panels across
rebalance dates, then one cross-sectional scoring per rebalance date.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Statement-derived feature columns pulled from FundamentalsCache timelines into
# each per-rebalance frame (peer-3 quality + income coverage inputs).
_TL_FEATURES = (
    "roe_ttm", "gross_margin_ttm", "gm_trend_yoy", "debt_to_assets",
    "neg_accruals", "fcf_to_assets", "share_count_shrink_yoy", "div_fcf_coverage_ttm",
)
# Valuation reconstruction columns (no staleness guard — pre-peer-3 behavior kept).
_TL_VALUATION = ("ttm_eps", "shares", "book")


def _value_components(
    fr: pd.DataFrame, scoring_cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Live-parity value sub-components for one cross-sectional frame.

    Returns (pe_c, pb_c, penalty) such that for any pe_w (pb_w = 1−pe_w):
        clip(pe_w·pe_c + pb_w·pb_c − penalty, clamp) ≈ live apply_value score
    (exact except live's inner clamp before anchor blending; benchmark_blend excluded).
    """
    from strategy.scoring.peer import _pct_rank_series, compute_peer_relative, safe_col

    factor = scoring_cfg.get("factors", {}).get("value", {})
    anchor_blend = float(factor.get("anchor_blend", 0.0))
    dist = factor.get("distress", {})
    dist_thr = float(dist.get("pe_threshold", 5.0))
    dist_pen = float(dist.get("pe_penalty", 0.30))
    neg_pen = float(dist.get("negative_eps_penalty", 0.25))

    pe_raw = safe_col(fr, "pe_ratio")
    pb_raw = safe_col(fr, "pb_ratio")
    pe_in = pe_raw.where(pe_raw > 0)
    pb_in = pb_raw.where(pb_raw > 0)

    pe_peer, *_ = compute_peer_relative(pe_in, fr, scoring_cfg, higher_is_better=False)
    pb_peer, *_ = compute_peer_relative(pb_in, fr, scoring_cfg, higher_is_better=False)

    has_pe = pe_in.notna().to_numpy()
    has_pb = pb_in.notna().to_numpy()
    pe_p = pe_peer.fillna(0.0).to_numpy()
    pb_p = pb_peer.fillna(0.0).to_numpy()

    # Missing-ratio handling at the PEER level, mirroring apply_value's composite
    # masks for ANY pe_w mix: single-ratio names carry that ratio's rank in both
    # components; neither-ratio names carry the explicit floor.
    pe_eff = np.select(
        [has_pe & has_pb, has_pe & ~has_pb, ~has_pe & has_pb], [pe_p, pe_p, pb_p], -0.25
    )
    pb_eff = np.select(
        [has_pe & has_pb, has_pe & ~has_pb, ~has_pe & has_pb], [pb_p, pe_p, pb_p], -0.25
    )

    if anchor_blend > 0.0:
        # Live anchor: cross-sectional rank of the raw ratio, inverted (cheap = high),
        # missing → 0.0. No cross-fill on the anchor (mirrors apply_value).
        pe_anchor = (-_pct_rank_series(pe_in)).to_numpy()
        pb_anchor = (-_pct_rank_series(pb_in)).to_numpy()
        pe_c = anchor_blend * pe_anchor + (1.0 - anchor_blend) * pe_eff
        pb_c = anchor_blend * pb_anchor + (1.0 - anchor_blend) * pb_eff
    else:
        pe_c, pb_c = pe_eff, pb_eff

    # Additive penalties: live subtracts them inside both the peer composite and the
    # anchor, which (weights summing to 1) totals exactly one application.
    pe_np = pe_raw.to_numpy(dtype=float)
    distress = np.isfinite(pe_np) & (pe_np > 0) & (pe_np <= dist_thr)
    neg_eps = np.isfinite(pe_np) & (pe_np < 0)
    penalty = dist_pen * distress.astype(float) + neg_pen * neg_eps.astype(float)

    return pe_c, pb_c, penalty


def build_pit_factor_panels(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    sectors: list[str],
    industries: list[str] | tuple[str, ...],
    position_52w_daily: np.ndarray | None,
    volume: np.ndarray | None,
    rebalance_freq: int,
    scoring_cfg: dict,
    dollar_volume_daily: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return {pe_comp_daily, pb_comp_daily, value_penalty_daily, quality_scores_daily,
    income_scores_daily}, each float64 (n_days, n_symbols). Raises RuntimeError if NO
    symbol has usable PIT data (so the loader can hard-raise rather than silently degrade)."""
    from data.fundamental_features import (
        _DIV_STREAK_MAX_GAP_DAYS,
        _MAX_FILING_AGE_DAYS,
        FundamentalsCache,
        dividend_streak_at_ex,
    )
    from strategy.scoring.income import apply_income
    from strategy.scoring.quality import apply_quality

    n_days, n_sym = prices.shape
    # Coerce the calendar to datetime64 (the loader's index may be object/string dtype) so
    # all searchsorted comparisons against filing/ex dates are datetime-vs-datetime.
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    rebal = list(range(0, n_days, max(1, rebalance_freq)))
    rebal_ts = dates[rebal].to_numpy()  # datetime64 at each rebalance date
    n_rebal = len(rebal)

    cache = FundamentalsCache()

    # ── Per-symbol causal step series (one cache read per symbol) ──────────────
    ttm_eps_panel = np.full((n_rebal, n_sym), np.nan)   # TTM EPS as-of each rebalance
    shares_panel  = np.full((n_rebal, n_sym), np.nan)
    book_panel    = np.full((n_rebal, n_sym), np.nan)
    feat_panels = {col: np.full((n_rebal, n_sym), np.nan) for col in _TL_FEATURES}
    div_panel    = np.zeros((n_rebal, n_sym))           # TTM cash dividends as-of each rebalance
    div_growth_panel = np.full((n_rebal, n_sym), np.nan)
    div_streak_panel = np.full((n_rebal, n_sym), np.nan)
    max_age = np.timedelta64(_MAX_FILING_AGE_DAYS, "D")
    streak_gap = np.timedelta64(_DIV_STREAK_MAX_GAP_DAYS, "D")

    n_with_fund = 0
    for j, sym in enumerate(symbols):
        tl = cache.timeline(sym)
        if tl is not None and len(tl):
            n_with_fund += 1
            fd = pd.to_datetime(tl["_fd"]).to_numpy()  # ensure datetime64 for searchsorted
            # index of the LATEST filing strictly before each rebalance date (side="left")
            pos = np.searchsorted(fd, rebal_ts, side="left") - 1
            valid = pos >= 0
            safe_pos = np.clip(pos, 0, len(fd) - 1)
            for col, panel in (
                ("ttm_eps", ttm_eps_panel), ("shares", shares_panel), ("book", book_panel),
            ):
                if col in tl.columns:
                    arr = tl[col].to_numpy()
                    panel[valid, j] = arr[safe_pos[valid]]
            # peer-3 statement features carry the live staleness guard: a filing older
            # than the max age at the as-of date is uncovered (neutral-scored).
            fresh = valid & ((rebal_ts - fd[safe_pos]) <= max_age)
            for col in _TL_FEATURES:
                if col in tl.columns:
                    arr = tl[col].to_numpy()
                    feat_panels[col][fresh, j] = arr[safe_pos[fresh]]

        records = cache.dividends(sym)
        if records is not None:
            ddates, damts = records
            cum = np.concatenate([[0.0], np.cumsum(damts)])
            hi = np.searchsorted(ddates, rebal_ts, side="left")           # ex-date strictly < asof
            lo1 = np.searchsorted(ddates, rebal_ts - np.timedelta64(365, "D"), side="left")
            lo2 = np.searchsorted(ddates, rebal_ts - np.timedelta64(730, "D"), side="left")
            t1 = cum[hi] - cum[lo1]
            t0 = cum[lo1] - cum[lo2]
            div_panel[:, j] = t1
            with np.errstate(divide="ignore", invalid="ignore"):
                div_growth_panel[:, j] = np.where((t0 > 0) & (t1 > 0), t1 / t0 - 1.0, np.nan)
            # Streak as-of: streak at the last ex-date strictly before the rebalance,
            # zeroed when the payer has gone quiet past the gap.
            streaks = dividend_streak_at_ex(ddates)
            last = hi - 1
            has_hist = last >= 0
            safe_last = np.clip(last, 0, len(ddates) - 1)
            recent = has_hist & ((rebal_ts - ddates[safe_last]) <= streak_gap)
            div_streak_panel[recent, j] = streaks[safe_last[recent]]
            div_streak_panel[has_hist & ~recent, j] = 0.0

    if n_with_fund == 0:
        raise RuntimeError(
            "PIT precompute: no symbol had >=4 quarters of cached statements — cannot build "
            "point-in-time factor panels (check data/fmp_cache_adj/statements coverage)."
        )

    # ── PE / PB / dividend-yield at each rebalance date (daily price × step fundamentals) ──
    px_rebal = prices[rebal, :]                       # (n_rebal, n_sym)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe_rebal = np.where(ttm_eps_panel > 0, px_rebal / ttm_eps_panel, np.nan)
        mcap = px_rebal * shares_panel
        pb_rebal = np.where((shares_panel > 0) & (book_panel > 0), mcap / book_panel, np.nan)
        dy_rebal = np.where(px_rebal > 0, div_panel / px_rebal, 0.0)

    sectors = list(sectors)
    industries = list(industries)
    vol_col = (np.asarray(volume, dtype=np.float64) if volume is not None
               else np.zeros(n_sym))

    # ── Cross-sectional scoring per rebalance date (production peer scorers) ───
    pe_comp_r = np.zeros((n_rebal, n_sym))
    pb_comp_r = np.zeros((n_rebal, n_sym))
    val_pen_r = np.zeros((n_rebal, n_sym))
    qual_r    = np.zeros((n_rebal, n_sym))
    inc_r     = np.zeros((n_rebal, n_sym))
    for ri, d in enumerate(rebal):
        px = prices[d, :]
        tradeable = np.isfinite(px) & (px > 0)
        if not tradeable.any():
            continue
        idx = np.where(tradeable)[0]
        pos52 = (position_52w_daily[d, idx] if position_52w_daily is not None
                 else np.full(idx.size, np.nan))
        fr = pd.DataFrame({
            "symbol":         [symbols[k] for k in idx],
            "sector":         [sectors[k] for k in idx],
            "industry":       [industries[k] for k in idx],
            "pe_ratio":       pe_rebal[ri, idx],
            "pb_ratio":       pb_rebal[ri, idx],
            "dividend_yield": dy_rebal[ri, idx],
            "volume":         vol_col[idx],
            "position_52w":   pos52,
            "div_growth_1y":  div_growth_panel[ri, idx],
            "div_streak_quarters": div_streak_panel[ri, idx],
        })
        for col in _TL_FEATURES:
            fr[col] = feat_panels[col][ri, idx]

        pe_c, pb_c, pen = _value_components(fr, scoring_cfg)
        apply_quality(fr, scoring_cfg)
        apply_income(fr, scoring_cfg)
        pe_comp_r[ri, idx] = np.nan_to_num(pe_c, nan=0.0)
        pb_comp_r[ri, idx] = np.nan_to_num(pb_c, nan=0.0)
        val_pen_r[ri, idx] = pen
        qual_r[ri, idx]    = np.nan_to_num(fr["quality_score"].to_numpy(), nan=0.0)
        inc_r[ri, idx]     = np.nan_to_num(fr["income_score"].to_numpy(), nan=0.0)

    # ── Forward-fill rebalance scores to every day ────────────────────────────
    def _ffill_daily(rebal_vals: np.ndarray) -> np.ndarray:
        out = np.zeros((n_days, n_sym))
        for ri, d in enumerate(rebal):
            end = rebal[ri + 1] if ri + 1 < len(rebal) else n_days
            out[d:end, :] = rebal_vals[ri, :]
        return out

    logger.info(
        "PIT precompute: %d/%d symbols with >=4q statements; %d rebalance dates scored",
        n_with_fund, n_sym, n_rebal,
    )
    return {
        "pe_comp_daily":        _ffill_daily(pe_comp_r),
        "pb_comp_daily":        _ffill_daily(pb_comp_r),
        "value_penalty_daily":  _ffill_daily(val_pen_r),
        "quality_scores_daily": _ffill_daily(qual_r),
        "income_scores_daily":  _ffill_daily(inc_r),
    }
