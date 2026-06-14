"""
scripts/active_sleeve_retire_diag.py — should production retire the active sleeve (index_pct->1.0)?

NO config writes, NO tuning. One full survivorship-free PIT load (1250d). Compares the CURRENT
production split (index_pct ~0.9486, ~5.14% active) vs 100% ETF/core (index_pct=1.0 -> zero
active stock buys, ETF sleeve = current equal-weight incumbent, NO ETF tilt). "100% ETF" is
defined in simulator terms as param slot 4 (index_pct) = 1.0; the backtest uses a single
index_pct for all regimes (it ignores regime index_pct_override, simulator.py:1358), so this
truly zeroes stock buys. Verified by trades_made ~0.

Reports portfolio-level (excess vs SPY, win rate, drawdown, turnover/trades) AND ETF/active
sleeve metrics, full-window + paired random windows (10x90 and 20x120, seed 7).
"""
import os
import sys
import warnings

os.chdir("/Users/lukaselsrode/dev_work/daily_investor")
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import util  # noqa: E402

util.BACKTEST_PARAMS["point_in_time_fundamentals"] = True
util.BACKTEST_PARAMS["allow_static_fundamentals_fallback"] = False

from backtesting.data_loader import load_and_precompute  # noqa: E402
from backtesting.random_walk import random_window_backtest  # noqa: E402
from backtesting.simulator import run_simulation  # noqa: E402
from tuning.constants import PARAM_NAMES, _current_params  # noqa: E402

assert PARAM_NAMES[4] == "index_pct", PARAM_NAMES[4]
inc = _current_params()
etf100 = inc.copy()
etf100[4] = 1.0  # 100% ETF/core: no active stock budget

pc = load_and_precompute(1250, mode=None)
print(f"PIT present={pc.pe_comp_daily is not None} | surv-free {pc.prices.shape[1]} stocks x {pc.prices.shape[0]}d")
print(f"index_pct: CURRENT={float(inc[4]):.4f} (active {1-float(inc[4]):.2%})  vs  100%-ETF={float(etf100[4]):.1f}")


def _num(x):
    return None if x is None else round(float(x), 4)


def full(label, vec):
    r = run_simulation(pc, vec, scope="active_sleeve_compounding")
    pexc = r.total_return - r.benchmark_twr
    print(f"{label:<26} total={r.total_return:>+7.2%} bench={r.benchmark_twr:>+7.2%} "
          f"port_exc={pexc:>+7.2%} maxDD={r.max_drawdown:>+7.2%} trades={r.trades_made:>4d} "
          f"turn={r.turnover_estimate:>5.2f} | etf_ret={(r.etf_sleeve_return or 0):>+7.2%} "
          f"etf_exc={(r.etf_excess_return or 0):>+7.2%} act_exc={_num(r.active_excess_return)}")
    return r


print("\n=== FULL-WINDOW 1250d (PIT) ===")
full("CURRENT (~5.14% active)", inc)
r0 = full("100% ETF (index_pct=1.0)", etf100)
print(f"verify 100%-ETF zeroes active: trades_made={r0.trades_made} (expect ~0)")


def paired(nw, wd):
    a = random_window_backtest(pc, inc, n_windows=nw, window_days=wd, seed=7, slippage_bps=10.0,
                               scope="active_sleeve_compounding")
    b = random_window_backtest(pc, etf100, n_windows=nw, window_days=wd, seed=7, slippage_bps=10.0,
                               scope="active_sleeve_compounding")
    print(f"\n=== PAIRED {nw}x{wd}d seed 7 (portfolio excess vs SPY) ===")
    print(f"  CURRENT  : med_exc={a.median_excess_return:>+.4f}  win%={a.pct_beating_benchmark:>.0%}  std={a.std_excess_return:.4f}")
    print(f"  100% ETF : med_exc={b.median_excess_return:>+.4f}  win%={b.pct_beating_benchmark:>.0%}  std={b.std_excess_return:.4f}")
    print(f"  delta (CURRENT - 100%ETF) med_exc = {a.median_excess_return - b.median_excess_return:>+.4f}  (noise std ~{a.std_excess_return:.4f})")


paired(10, 90)
paired(20, 120)
print("\nNo config written.")
