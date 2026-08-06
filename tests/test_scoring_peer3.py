"""
tests/test_scoring_peer3.py — peer-3 income de-saturation + sustainability contract.

The peer-2 income factor was max(peer_rank, dy/DIVIDEND_THRESHOLD) capped at 1.5,
which pinned every yield ≥ 4.5% at exactly 1.500 (305 live names) and made income a
near-binary payer indicator (the root of the quality↔income correlation).

Covers:
  1. No pinning: high-yield payers spread across distinct score values.
  2. Sustainability moves the score: same-yield twins split on coverage/growth/streak.
  3. Missing sustainability columns renormalize to yield-only (vintage frames).
  4. Yield traps and non-payers still score exactly 0.
  5. Config weights come from scoring.income_inputs (live constants, not copies).

Quality-side orthogonality (dividends/PE/PB/liquidity zero-influence) is covered in
tests/test_peer2_scoring.py::test_quality_ignores_dividends_pe_pb_and_liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.scoring.income import apply_income
from util import SCORING_PARAMS


def _cfg(**overrides) -> dict:
    cfg = {
        **{k: v for k, v in SCORING_PARAMS.items()},
        "peer_standardization": {**SCORING_PARAMS["peer_standardization"], "min_group_size": 5},
    }
    cfg["factors"] = {
        name: {**f, "anchor_blend": 0.0, "benchmark_blend": 0.0}
        for name, f in SCORING_PARAMS["factors"].items()
    }
    cfg.update(overrides)
    return cfg


def _payer_universe(n: int = 60, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trap = float(SCORING_PARAMS["income_inputs"]["yield_trap_threshold"])
    return pd.DataFrame({
        "symbol":   [f"P{i:03d}" for i in range(n)],
        "industry": ["banks"] * (n // 2) + ["utilities"] * (n - n // 2),
        "sector":   ["financials"] * (n // 2) + ["utilities"] * (n - n // 2),
        # High-yield but below-trap payers — the zone the old cap pinned at 1.5.
        "dividend_yield":       rng.uniform(0.045, trap - 0.005, n),
        "div_fcf_coverage_ttm": rng.uniform(0.5, 5.0, n),
        "div_growth_1y":        rng.uniform(-0.1, 0.2, n),
        "div_streak_quarters":  rng.integers(1, 20, n).astype(float),
    })


def test_income_no_pinning_mass():
    df = _payer_universe()
    apply_income(df, _cfg())
    counts = df["income_score"].value_counts()
    # Old engine: every one of these rows scored exactly 1.500. New engine: the
    # largest single-value mass among these high-yield payers stays small.
    assert counts.iloc[0] / len(df) < 0.10
    assert df["income_score"].nunique() > len(df) * 0.5


def test_income_sustainability_moves_score():
    df = _payer_universe(seed=5)
    # Twins with identical yield, opposite sustainability.
    df.loc[0, "dividend_yield"] = df.loc[1, "dividend_yield"] = 0.05
    df.loc[0, ["div_fcf_coverage_ttm", "div_growth_1y", "div_streak_quarters"]] = [5.0, 0.15, 20.0]
    df.loc[1, ["div_fcf_coverage_ttm", "div_growth_1y", "div_streak_quarters"]] = [0.2, -0.30, 1.0]
    apply_income(df, _cfg())
    assert df.loc[0, "income_score"] > df.loc[1, "income_score"]


def test_income_renormalizes_to_yield_only_on_vintage_frames():
    base = _payer_universe(seed=9)
    vintage = base.drop(columns=["div_fcf_coverage_ttm", "div_growth_1y", "div_streak_quarters"])
    yield_only = base.copy()

    apply_income(vintage, _cfg())

    cfg_yield_only = _cfg()
    cfg_yield_only["income_inputs"] = {
        **cfg_yield_only["income_inputs"],
        "weights": {"dividend_yield": 1.0},
    }
    apply_income(yield_only, cfg_yield_only)

    pd.testing.assert_series_equal(vintage["income_score"], yield_only["income_score"])


def test_income_traps_and_non_payers_score_zero():
    df = _payer_universe(seed=13)
    trap = float(SCORING_PARAMS["income_inputs"]["yield_trap_threshold"])
    df.loc[0, "dividend_yield"] = trap + 0.02   # yield trap
    df.loc[1, "dividend_yield"] = 0.0           # non-payer
    apply_income(df, _cfg())
    assert df.loc[0, "income_score"] == 0.0
    assert bool(df.loc[0, "yield_trap_flag"])
    assert df.loc[1, "income_score"] == 0.0
    assert df.loc[1, "income_fallback_reason"] == "no_dividend"


def test_income_weights_come_from_config():
    df_default = _payer_universe(seed=17)
    df_custom = _payer_universe(seed=17)
    apply_income(df_default, _cfg())

    cfg = _cfg()
    cfg["income_inputs"] = {
        **cfg["income_inputs"],
        "weights": {"dividend_yield": 0.10, "div_streak": 0.90},
    }
    apply_income(df_custom, cfg)
    assert not df_default["income_score"].equals(df_custom["income_score"])
