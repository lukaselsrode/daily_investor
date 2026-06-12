"""
tuning/gauntlet.py — named stress-episode falsification gate.

At this project's sample size (~5-20y of usable history) excess-return point
estimates cannot rank strategies: the noise floor over a window is roughly the
strategy's tracking error scaled by sqrt(t), so a few percent per year of
claimed edge needs decades to resolve. What a small sample CAN do is falsify
fragility. The gauntlet simulates the selected candidate and the incumbent
through history's named stress regimes and demands the candidate SURVIVE each
one relative to the incumbent (catastrophe-scale floors) — not win them.

Survivorship caveat: dead-name coverage in the FMP cache starts 2021 (the
current plan serves only page 0 of /delisted-companies). Pre-2021 episodes
therefore run on a survivor-biased cross-section. Checks stay incumbent-
RELATIVE — both configs share the bias — and every episode row carries its
dead-name count so the bias is visible in every report.
"""

from __future__ import annotations

import logging

import numpy as np

from backtesting.data_loader import load_and_precompute
from backtesting.simulator import run_simulation

logger = logging.getLogger(__name__)

# Major drawdown/stress regimes within the cache's reach (benchmark axis starts
# 2006-07). Calendar dates; rows are matched on the precomp's trading calendar.
DEFAULT_EPISODES: dict[str, dict[str, str]] = {
    "gfc_2008":       {"start": "2007-10-01", "end": "2009-06-30"},
    "euro_debt_2011": {"start": "2011-05-01", "end": "2011-12-31"},
    "china_2015":     {"start": "2015-06-01", "end": "2016-03-31"},
    "q4_2018":        {"start": "2018-09-01", "end": "2018-12-31"},
    "covid_2020":     {"start": "2020-02-01", "end": "2020-08-31"},
    "inflation_2022": {"start": "2022-01-01", "end": "2022-12-31"},
}


def _dead_names_in_window(start: str, end: str) -> int:
    """How many known-delisted names existed inside [start, end] — the episode's
    survivorship-honesty signal (0 for pre-2021 episodes on the current cache)."""
    try:
        from backtesting.survivorship import dead_universe

        du = dead_universe()
        if du.empty:
            return 0
        first = du["first_date"].astype(str).str[:10]
        delist = du["delist_date"].astype(str).str[:10]
        return int(((first <= end) & (delist >= start)).sum())
    except Exception as exc:
        logger.debug("dead-name window count unavailable: %s", exc)
        return 0


def stress_gauntlet(
    selected_params: np.ndarray,
    incumbent_params: np.ndarray,
    backtest_cfg: dict,
    mode: str | None = None,
    scope: str = "overall_strategy",
    precomp=None,
) -> tuple[bool, list[str], list[dict]]:
    """Run selected vs incumbent through each configured stress episode.

    Config: backtest.stress_gauntlet {enabled, history_days, catastrophe_excess,
    catastrophe_drawdown, min_symbols, episodes{name: {start, end}}}.

    Per episode the candidate fails on: excess regression vs the incumbent
    beyond catastrophe_excess, drawdown deeper by more than
    catastrophe_drawdown, or turnover beyond max_turnover_multiple. An episode
    outside the data axis or below min_symbols coverage is SKIPPED with a
    visible row — never silently counted as evidence.

    Returns (passed, reasons, rows). Disabled → passes (prior gates remain).
    `precomp` lets callers inject an existing deep load instead of re-loading.
    """
    cfg = backtest_cfg.get("stress_gauntlet", {}) or {}
    if not cfg.get("enabled", False):
        return True, [], []

    from backtesting.regime_scope import slice_precomp

    episodes: dict = cfg.get("episodes") or DEFAULT_EPISODES
    cat_excess = float(cfg.get("catastrophe_excess", 0.10))
    cat_dd = float(cfg.get("catastrophe_drawdown", 0.05))
    min_symbols = int(cfg.get("min_symbols", 300))
    turn_mult = float(backtest_cfg.get("max_turnover_multiple", 2.0))

    if precomp is None:
        try:
            precomp = load_and_precompute(int(cfg.get("history_days", 5000)), mode=mode)
        except Exception as exc:
            return False, [f"gauntlet: could not load deep history ({exc})"], []
    if precomp.dates is None:
        return False, ["gauntlet: precomp carries no dates (re-load with the current loader)"], []

    dates = np.asarray(precomp.dates, dtype=str)

    def _sim(pc, params):
        return run_simulation(
            pc, params, backtest_cfg.get("starting_capital", 10_000.0),
            slippage_bps=backtest_cfg.get("slippage_bps", 10.0),
            commission_per_trade=backtest_cfg.get("commission_per_trade", 0.0),
            weekly_contribution=backtest_cfg.get("weekly_contribution", 0.0),
            rebalance_frequency_days=backtest_cfg.get("rebalance_frequency_days", 5),
            scope=scope,
        )

    rows: list[dict] = []
    reasons: list[str] = []
    for name, ep in episodes.items():
        start, end = str(ep["start"]), str(ep["end"])
        i0 = int(np.searchsorted(dates, start))
        i1 = int(np.searchsorted(dates, end, side="right"))
        row: dict = {"episode": name, "start": start, "end": end,
                     "status": "ok", "n_days": i1 - i0}
        if i0 == 0 and len(dates) and str(dates[0]) > start:
            row.update(status="skipped: predates data axis", n_days=0)
            rows.append(row)
            continue
        if i1 - i0 < 30:
            row.update(status="skipped: <30 trading days in axis")
            rows.append(row)
            continue

        # Listed = finite price at episode start; the tradeable mask alone is
        # True from axis start (it only encodes the post-delist tail), so it
        # must be ANDed with price availability or pre-listing names count.
        listed = np.isfinite(np.asarray(precomp.prices[i0], dtype=float))
        if precomp.tradeable_mask_daily is not None:
            listed &= np.asarray(precomp.tradeable_mask_daily[i0], dtype=bool)
        n_avail = int(listed.sum())
        row["n_symbols"] = n_avail
        row["n_dead"] = _dead_names_in_window(start, end)
        if n_avail < min_symbols:
            row.update(status=f"skipped: only {n_avail} symbols (<{min_symbols})")
            rows.append(row)
            continue

        pc = slice_precomp(precomp, slice(i0, i1))
        sel = _sim(pc, selected_params)
        inc = _sim(pc, incumbent_params)
        sel_exc = sel.total_return - sel.benchmark_twr
        inc_exc = inc.total_return - inc.benchmark_twr
        delta = sel_exc - inc_exc
        row.update(
            incumbent_excess=inc_exc, selected_excess=sel_exc, delta=delta,
            incumbent_max_drawdown=inc.max_drawdown, selected_max_drawdown=sel.max_drawdown,
            incumbent_turnover=inc.turnover_estimate, selected_turnover=sel.turnover_estimate,
        )
        rows.append(row)

        if delta < -cat_excess:
            reasons.append(
                f"gauntlet {name}: excess {sel_exc:+.2%} vs incumbent {inc_exc:+.2%} — "
                f"regression {delta:+.2%} beyond catastrophe limit -{cat_excess:.0%}"
            )
        if sel.max_drawdown < inc.max_drawdown - cat_dd:
            reasons.append(
                f"gauntlet {name}: drawdown {sel.max_drawdown:.1%} deeper than incumbent "
                f"{inc.max_drawdown:.1%} by more than {cat_dd:.0%}"
            )
        inc_turn = max(float(inc.turnover_estimate), 1e-9)
        if float(sel.turnover_estimate) > inc_turn * turn_mult:
            reasons.append(
                f"gauntlet {name}: turnover {sel.turnover_estimate:.2f} > {turn_mult:.1f}x "
                f"incumbent's {inc.turnover_estimate:.2f}"
            )
    return len(reasons) == 0, reasons, rows


def print_gauntlet_table(rows: list[dict]) -> None:
    if not rows:
        return
    print(
        f"\n{'episode':<16} {'window':<22} {'n_sym':>6} {'dead':>5} "
        f"{'inc_exc':>9} {'sel_exc':>9} {'delta':>8} {'inc_DD':>8} {'sel_DD':>8}"
    )
    for r in rows:
        win = f"{r['start']}..{r['end']}"
        if r.get("status", "ok") != "ok":
            print(f"{r['episode']:<16} {win:<22} → {r['status']}")
            continue
        print(
            f"{r['episode']:<16} {win:<22} {r.get('n_symbols', 0):>6} {r.get('n_dead', 0):>5} "
            f"{r['incumbent_excess']:>+9.1%} {r['selected_excess']:>+9.1%} {r['delta']:>+8.1%} "
            f"{r['incumbent_max_drawdown']:>8.1%} {r['selected_max_drawdown']:>8.1%}"
        )
