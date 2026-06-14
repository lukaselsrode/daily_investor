"""
scripts/pit_gates_off_diag.py — does a clean PRICE-based active edge survive once fundamental
gates are removed? No tuning, no config edits (in-process mutation only). ONE PIT-on
survivorship-free 1250d load; 10x90d paired (seed 7); active_sleeve_compounding scope;
active-only and portfolio-level reported separately.

Variants (same windows):
  A  PIT incumbent              — full weights, full gates (baseline).
  B  PIT momentum-only          — weights [0,0,0,1], fundamental gates STILL ON (PIT daily).
  E  PIT momentum, GATES OFF    — weights [0,0,0,1]; min_quality (slot41) & min_momentum
                                  (slot42) set very low; income-trap neutralized
                                  (min_conditional_momentum_score = -1e9). disc/tradeable
                                  gates remain (universe gates, not fundamental).
Index-only (variant 2) is the definitional baseline: active_excess_vs_SPY = 0 (putting the
active 5% into SPY). A "buys-disabled" run would be cash-drag, not index, so it is NOT run.
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
util.BACKTEST_PARAMS["point_in_time_fundamentals"] = True
util.BACKTEST_PARAMS["allow_static_fundamentals_fallback"] = False


def _g(s, a):
    v = getattr(s, a, None)
    return round(v, 4) if isinstance(v, float) else v


def _run(pc, params):
    from backtesting.random_walk import random_window_backtest
    return random_window_backtest(
        pc, params, n_windows=NW, window_days=WD, seed=SEED,
        slippage_bps=10.0, scope="active_sleeve_compounding",
    )


def _row(label, s):
    print(f"{label:<32} "
          f"act_excess={_g(s,'median_active_excess_return')!s:>8} "
          f"act_beat%={_g(s,'pct_active_beating_benchmark')!s:>6} "
          f"act_wdDD={_g(s,'worst_decile_active_drawdown')!s:>8} | "
          f"portf_excess={_g(s,'median_excess_return')!s:>8} "
          f"noise_std={_g(s,'std_excess_return')!s:>7}")


from backtesting.data_loader import load_and_precompute  # noqa: E402

pc = load_and_precompute(1250, mode=None)
print(f"PIT load: pe_comp_daily present={pc.pe_comp_daily is not None}, "
      f"surv-free {pc.prices.shape[1]} stocks x {pc.prices.shape[0]}d", flush=True)

inc = _current_params()
mom = inc.copy()
mom[0] = mom[1] = mom[2] = 0.0
mom[3] = 1.0

# A and B use the live gates (run BEFORE any gate mutation).
a = _run(pc, inc)
b = _run(pc, mom)

# E: neutralize fundamental gates. slots 41/42 -> very low; income-trap off via the
# CANDIDATE_SELECTION_PARAMS field (no param slot). disc/tradeable gates stay on.
mom_gatesoff = mom.copy()
mom_gatesoff[41] = -1e9   # min_quality_score
mom_gatesoff[42] = -1e9   # min_momentum_score
util.CANDIDATE_SELECTION_PARAMS["min_conditional_momentum_score"] = -1e9  # income-trap off
util.CANDIDATE_SELECTION_PARAMS["min_quality_score"] = -1e9
util.CANDIDATE_SELECTION_PARAMS["min_momentum_score"] = -1e9
e = _run(pc, mom_gatesoff)

print("\n--- RESULTS (PIT-on; active-only excess vs SPY + portfolio-level) ---")
_row("A incumbent (full gates)", a)
_row("B momentum-only (gates ON)", b)
_row("E momentum-only (GATES OFF)", e)
print("\nINDEX-ONLY baseline (active 5% in SPY): active_excess_vs_SPY = 0.0000 by definition")
print("\nVERDICT INPUTS:")
print(f"  best PIT active median excess = {max((a.median_active_excess_return or -9), (b.median_active_excess_return or -9), (e.median_active_excess_return or -9)):+.4f}")
print(f"  noise band (portfolio std_excess, A) = {a.std_excess_return:.4f}")
