"""
scripts/pit_incumbent_vs_momentum_diag.py — PIT incumbent-vs-momentum active-sleeve diagnostic.

NO tuning, NO config edits (mutates util.BACKTEST_PARAMS in-process only). Loads the full
survivorship-free universe twice — once PIT-on (honest), once PIT-off (lookahead-contaminated
static reference) — and runs incumbent vs momentum-only on the SAME 10x90d paired random
windows at scope=active_sleeve_compounding. Active-only and portfolio-level reported separately.

Gate note for the momentum-only variant (B/D): only the composite WEIGHTS are momentum-only
([0,0,0,1]); the candidate gates (min_quality 0.38, income-trap) STILL apply and use the same
daily factor arrays as the run (PIT daily when PIT-on, static when off). So B isolates the
WEIGHTING contribution of fundamentals while holding the gates constant.
"""
import os
import sys
import warnings

os.chdir("/Users/lukaselsrode/dev_work/daily_investor")
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")


import util  # noqa: E402
from tuning.constants import _current_params  # noqa: E402

NW, WD, SEED = 10, 90, 7
inc = _current_params()
mom = inc.copy()
mom[0] = mom[1] = mom[2] = 0.0
mom[3] = 1.0


def _g(s, a):
    v = getattr(s, a, None)
    return round(v, 4) if isinstance(v, float) else v


def _load(pit_on):
    util.BACKTEST_PARAMS["point_in_time_fundamentals"] = pit_on
    util.BACKTEST_PARAMS["allow_static_fundamentals_fallback"] = not pit_on
    from backtesting.data_loader import load_and_precompute
    return load_and_precompute(1250, mode=None)


def _run(pc, params):
    from backtesting.random_walk import random_window_backtest
    return random_window_backtest(
        pc, params, n_windows=NW, window_days=WD, seed=SEED,
        slippage_bps=10.0, scope="active_sleeve_compounding",
    )


def _row(label, s):
    print(f"{label:<34} "
          f"act_excess={_g(s,'median_active_excess_return')!s:>8} "
          f"act_beat%={_g(s,'pct_active_beating_benchmark')!s:>6} "
          f"act_wdDD={_g(s,'worst_decile_active_drawdown')!s:>8} | "
          f"portf_excess={_g(s,'median_excess_return')!s:>8} "
          f"noise_std={_g(s,'std_excess_return')!s:>7}")


print(f"=== {NW}x{WD}d paired (seed {SEED}), scope=active_sleeve_compounding, 10bps ===", flush=True)

pc_pit = _load(True)
print(f"PIT-ON load: pe_comp_daily present={pc_pit.pe_comp_daily is not None}, "
      f"surv-free {pc_pit.prices.shape[1]} stocks x {pc_pit.prices.shape[0]}d", flush=True)
a = _run(pc_pit, inc)
b = _run(pc_pit, mom)

pc_static = _load(False)
print(f"STATIC load (LOOKAHEAD-CONTAMINATED reference): pe_comp_daily present="
      f"{pc_static.pe_comp_daily is not None}", flush=True)
c = _run(pc_static, inc)
d = _run(pc_static, mom)

print("\n--- RESULTS (active-only excess vs SPY + portfolio-level) ---")
_row("A incumbent  (PIT, honest)", a)
_row("B momentum-only (PIT)", b)
_row("C incumbent  (STATIC, lookahead)", c)
_row("D momentum-only (STATIC, lookahead)", d)
print("\nKEY DELTAS (active median excess):")
print(f"  PIT  : incumbent - momentum_only = {(a.median_active_excess_return or 0) - (b.median_active_excess_return or 0):+.4f}")
print(f"  STAT : incumbent - momentum_only = {(c.median_active_excess_return or 0) - (d.median_active_excess_return or 0):+.4f}")
print(f"  incumbent PIT - STATIC (lookahead inflation) = {(a.median_active_excess_return or 0) - (c.median_active_excess_return or 0):+.4f}")
