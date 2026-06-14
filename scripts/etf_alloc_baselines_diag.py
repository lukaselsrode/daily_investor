"""
scripts/etf_alloc_baselines_diag.py — honest ETF/core allocation diagnostic. NO config writes,
NO tuning. One full survivorship-free PIT load (1250d); equal-weight incumbent vs a few
interpretable bucket/regime allocations via the etf_allocation scope (active-stock params are
frozen and identical across variants, so active-sleeve effects cancel — the deltas are pure
ETF-sleeve). Reports ETF-sleeve return/excess/turnover AND portfolio-level excess separately,
both full-window (cumulative) and 10x90d paired (robustness).
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
from tuning.constants import _current_params  # noqa: E402
from tuning.etf_tune import _all_regimes, _vector_from_regime_buckets  # noqa: E402

NW, WD, SEED = 10, 90, 7

ALLOCS = [
    ("equal_weight (INCUMBENT)", None),
    ("core_only", _all_regimes({"core_market": 1.0})),
    ("defensive core+div", _all_regimes({"core_market": 0.7, "dividend_defensive": 0.3})),
    ("risk_on growth/semis", _all_regimes(
        {"core_market": 0.45, "growth": 0.2, "semis": 0.05, "small_cap": 0.15, "international": 0.15})),
    ("regime_aware", {
        "bullish":   {"core_market": 0.45, "growth": 0.2, "semis": 0.05, "small_cap": 0.15, "international": 0.15},
        "neutral":   {"core_market": 0.6, "dividend_defensive": 0.2, "growth": 0.1, "international": 0.1},
        "defensive": {"core_market": 0.75, "dividend_defensive": 0.25},
    }),
]


def _vec(spec):
    return _current_params() if spec is None else _vector_from_regime_buckets(spec, enabled=True)


pc = load_and_precompute(1250, mode=None)
print(f"PIT load: pe_comp_daily present={pc.pe_comp_daily is not None}, "
      f"surv-free {pc.prices.shape[1]} stocks x {pc.prices.shape[0]}d", flush=True)

print("\n=== FULL-WINDOW (1250d) — ETF sleeve decomposition + portfolio ===")
print(f"{'allocation':<26}{'etf_ret':>9}{'etf_exc':>9}{'etf_turn':>9}{'total':>9}{'bench':>9}{'port_exc':>9}")
full = {}
for name, spec in ALLOCS:
    r = run_simulation(pc, _vec(spec), scope="etf_allocation")
    pexc = r.total_return - r.benchmark_twr
    full[name] = pexc
    print(f"{name:<26}{(r.etf_sleeve_return or 0):>+9.2%}{(r.etf_excess_return or 0):>+9.2%}"
          f"{(r.etf_turnover or 0):>9.2f}{r.total_return:>+9.2%}{r.benchmark_twr:>+9.2%}{pexc:>+9.2%}", flush=True)

print(f"\n=== 10x90d PAIRED (seed {SEED}) — portfolio excess vs SPY (robustness) ===")
print(f"{'allocation':<26}{'med_port_exc':>13}{'noise_std':>11}{'etf_turn(full)':>15}")
for name, spec in ALLOCS:
    s = random_window_backtest(pc, _vec(spec), n_windows=NW, window_days=WD, seed=SEED,
                               slippage_bps=10.0, scope="etf_allocation")
    print(f"{name:<26}{(s.median_excess_return or 0):>+13.4f}{(s.std_excess_return or 0):>11.4f}", flush=True)

print("\nNOTE: active-stock params identical across all rows (etf_allocation scope) -> "
      "cross-row deltas are pure ETF-sleeve effect. No config written.")
