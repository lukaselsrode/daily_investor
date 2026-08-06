"""
tests/test_peer2_scoring.py — peer-2/peer-3 scoring mechanics (fundamentals
quality, rel_volume momentum input, value sector-benchmark blend).

Covers:
  1. Quality graceful degradation: frames without fundamental columns (old
     snapshot vintages) still score — low-coverage components are dropped and
     the remaining weights renormalize (exactly equivalent to a config that
     never listed them).
  2. Quality direction: the fundamentally stronger of two same-industry twins
     (higher ROE/FCF, lower leverage) ranks higher.
  3. Momentum rel_volume: weight 0.0 is byte-identical to a frame without the
     columns; weight > 0 lifts volume-confirmed names.
  4. Value benchmark_blend: 0.0 is a no-op; > 0 with pe_comp/pb_comp absent is a
     no-op; > 0 with the columns present moves scores.
  5. compute_metric stamps SCORING_MODEL_VERSION.

Weights/defaults are read from the live module constants (_COMPONENT_WEIGHTS,
SCORING_MODEL_VERSION), never hardcoded copies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.scoring.composite import SCORING_MODEL_VERSION, compute_metric
from strategy.scoring.momentum import apply_momentum
from strategy.scoring.quality import _COMPONENT_WEIGHTS, apply_quality
from strategy.scoring.value import apply_value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUND_COLS = ("roe_ttm", "fcf_to_assets", "neg_accruals", "gross_margin_ttm",
              "debt_to_assets", "share_count_shrink_yoy", "gm_trend_yoy")


def _universe(n: int = 60, with_dollar_vol: bool = False, seed: int = 42,
              with_fundamentals: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    industries = (["banks"] * 20 + ["software"] * 20 + ["utilities"] * 20)
    sectors = (["financials"] * 20 + ["technology"] * 20 + ["utilities"] * 20)
    df = pd.DataFrame({
        "symbol":   [f"T{i:03d}" for i in range(n)],
        "industry": industries[:n],
        "sector":   sectors[:n],
        "pe_ratio": rng.uniform(5, 40, n),
        "pb_ratio": rng.uniform(0.8, 8.0, n),
        "volume":   rng.uniform(5e5, 5e7, n),
        "dividend_yield": rng.uniform(0.0, 0.08, n),
        "current_price":  rng.uniform(20, 200, n),
        "position_52w":   rng.uniform(0.1, 0.95, n),
        "return_1m":  rng.uniform(-0.1, 0.1, n),
        "return_3m":  rng.uniform(-0.2, 0.2, n),
        "return_5d":  rng.uniform(-0.05, 0.05, n),
        "return_6m":  rng.uniform(-0.3, 0.3, n),
        "rs_3m":      rng.uniform(-0.2, 0.2, n),
        "rs_6m":      rng.uniform(-0.2, 0.2, n),
        "risk_adj_momentum_3m": rng.uniform(-0.3, 0.3, n),
        "realized_vol_3m": rng.uniform(0.15, 0.40, n),
        "above_50dma":  rng.choice([True, False], n),
        "above_200dma": rng.choice([True, False], n),
    })
    if with_dollar_vol:
        df["dollar_vol_5d"] = rng.uniform(1e6, 5e9, n)
        df["dollar_vol_21d"] = df["dollar_vol_5d"] * rng.uniform(0.8, 1.2, n)
        df["dollar_vol_63d"] = df["dollar_vol_5d"] * rng.uniform(0.8, 1.2, n)
        df["dollar_vol_cv_63d"] = rng.uniform(0.1, 2.0, n)
    if with_fundamentals:
        df["roe_ttm"] = rng.uniform(-0.10, 0.35, n)
        df["fcf_to_assets"] = rng.uniform(-0.05, 0.20, n)
        df["neg_accruals"] = rng.uniform(-0.10, 0.10, n)
        df["gross_margin_ttm"] = rng.uniform(0.10, 0.80, n)
        df["debt_to_assets"] = rng.uniform(0.0, 0.6, n)
        df["share_count_shrink_yoy"] = rng.uniform(-0.05, 0.05, n)
        df["gm_trend_yoy"] = rng.uniform(-0.05, 0.05, n)
    return df


def _cfg(**overrides) -> dict:
    from util import SCORING_PARAMS
    cfg = {
        **{k: v for k, v in SCORING_PARAMS.items()},
        "peer_standardization": {**SCORING_PARAMS["peer_standardization"], "min_group_size": 5},
    }
    # Pure peer scores by default (no anchors) so tests isolate the new mechanics.
    cfg["factors"] = {
        name: {**f, "anchor_blend": 0.0, "benchmark_blend": 0.0}
        for name, f in SCORING_PARAMS["factors"].items()
    }
    cfg["factors"]["momentum"]["enabled"] = True
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1-2. Quality
# ---------------------------------------------------------------------------

def test_quality_scores_frame_without_new_columns():
    df = _universe(with_dollar_vol=False)
    apply_quality(df, _cfg())
    assert df["quality_score"].notna().all()
    assert np.isfinite(df["quality_score"]).all()


def test_quality_coverage_drop_equals_restricted_config():
    """Dropping absent components must equal a config that never listed them."""
    present = ("roe_ttm", "low_leverage")
    base_components = {k: v for k, v in _COMPONENT_WEIGHTS.items() if k in present}

    def _partial_universe(seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = _universe(with_fundamentals=False)
        df["roe_ttm"] = rng.uniform(-0.10, 0.35, len(df))
        df["debt_to_assets"] = rng.uniform(0.0, 0.6, len(df))
        return df

    df_a = _partial_universe()
    apply_quality(df_a, _cfg())

    df_b = _partial_universe()
    apply_quality(df_b, _cfg(quality_components=base_components))

    pd.testing.assert_series_equal(df_a["quality_score"], df_b["quality_score"])


def test_quality_prefers_strong_fundamentals():
    df = _universe(seed=3, with_fundamentals=True)
    # Two same-industry twins split ONLY on fundamentals strength.
    df.loc[0, ["roe_ttm", "fcf_to_assets", "neg_accruals", "gross_margin_ttm",
               "share_count_shrink_yoy", "gm_trend_yoy"]] = [0.35, 0.20, 0.08, 0.75, 0.04, 0.04]
    df.loc[0, "debt_to_assets"] = 0.05
    df.loc[1, ["roe_ttm", "fcf_to_assets", "neg_accruals", "gross_margin_ttm",
               "share_count_shrink_yoy", "gm_trend_yoy"]] = [-0.08, -0.04, -0.08, 0.12, -0.04, -0.04]
    df.loc[1, "debt_to_assets"] = 0.58
    apply_quality(df, _cfg())
    assert df.loc[0, "quality_score"] > df.loc[1, "quality_score"]


def test_quality_ignores_dividends_pe_pb_and_liquidity():
    """peer-3 orthogonality contract: dividend/PE/PB/volume changes must not move quality."""
    df_a = _universe(seed=11, with_fundamentals=True, with_dollar_vol=True)
    df_b = df_a.copy()
    rng = np.random.default_rng(99)
    df_b["dividend_yield"] = rng.uniform(0.0, 0.12, len(df_b))     # incl. trap zone
    df_b["pe_ratio"] = rng.uniform(1.0, 60.0, len(df_b))           # incl. distress zone
    df_b["pb_ratio"] = rng.uniform(0.2, 12.0, len(df_b))
    df_b["dollar_vol_5d"] = rng.uniform(1e5, 1e10, len(df_b))
    df_b["dollar_vol_63d"] = rng.uniform(1e5, 1e10, len(df_b))
    df_b["position_52w"] = rng.uniform(0.0, 1.0, len(df_b))
    apply_quality(df_a, _cfg())
    apply_quality(df_b, _cfg())
    pd.testing.assert_series_equal(df_a["quality_score"], df_b["quality_score"])


# ---------------------------------------------------------------------------
# 3. Momentum rel_volume
# ---------------------------------------------------------------------------

def test_momentum_rel_volume_zero_weight_is_noop():
    cfg = _cfg()
    assert cfg["momentum_inputs"]["weights"].get("rel_volume", 0.0) == 0.0

    df_no_cols = _universe(with_dollar_vol=False)
    df_with_cols = _universe(with_dollar_vol=True)
    apply_momentum(df_no_cols, cfg)
    apply_momentum(df_with_cols, cfg)
    pd.testing.assert_series_equal(
        df_no_cols["momentum_score"], df_with_cols["momentum_score"]
    )


def test_momentum_rel_volume_lifts_volume_confirmed_names():
    cfg = _cfg()
    cfg["momentum_inputs"] = {
        **cfg["momentum_inputs"],
        "weights": {**cfg["momentum_inputs"]["weights"], "rel_volume": 0.30},
    }
    df = _universe(with_dollar_vol=True, seed=9)
    baseline = _universe(with_dollar_vol=True, seed=9)
    # Symbol 0: volume surge (5d ≫ 63d). Symbol 1: volume collapse.
    for frame in (df, baseline):
        frame.loc[0, "dollar_vol_63d"] = 1e8
        frame.loc[1, "dollar_vol_63d"] = 1e8
    df.loc[0, "dollar_vol_5d"] = 5e8
    df.loc[1, "dollar_vol_5d"] = 2e7
    baseline.loc[[0, 1], "dollar_vol_5d"] = 1e8  # neutral ratio

    apply_momentum(df, cfg)
    apply_momentum(baseline, cfg)
    assert df.loc[0, "momentum_score"] >= baseline.loc[0, "momentum_score"]
    assert df.loc[1, "momentum_score"] <= baseline.loc[1, "momentum_score"]


# ---------------------------------------------------------------------------
# 4. Value benchmark blend
# ---------------------------------------------------------------------------

def _with_comps(df: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["pe_comp"] = rng.uniform(0.2, 5.0, len(df))
    out["pb_comp"] = rng.uniform(0.2, 5.0, len(df))
    return out


def test_value_benchmark_blend_zero_is_noop():
    cfg = _cfg()
    df_plain = _universe()
    df_comps = _with_comps(_universe())
    apply_value(df_plain, cfg)
    apply_value(df_comps, cfg)
    pd.testing.assert_series_equal(df_plain["value_score"], df_comps["value_score"])


def test_value_benchmark_blend_missing_columns_is_noop():
    cfg = _cfg()
    cfg["factors"]["value"] = {**cfg["factors"]["value"], "benchmark_blend": 0.4}
    df_on = _universe()
    apply_value(df_on, cfg)

    cfg_off = _cfg()
    df_off = _universe()
    apply_value(df_off, cfg_off)
    pd.testing.assert_series_equal(df_on["value_score"], df_off["value_score"])


def test_value_benchmark_blend_moves_scores_when_columns_present():
    cfg = _cfg()
    cfg["factors"]["value"] = {**cfg["factors"]["value"], "benchmark_blend": 0.4}
    df = _with_comps(_universe())
    apply_value(df, cfg)

    cfg_off = _cfg()
    df_off = _with_comps(_universe())
    apply_value(df_off, cfg_off)
    assert not df["value_score"].equals(df_off["value_score"])


# ---------------------------------------------------------------------------
# 5. Version stamp
# ---------------------------------------------------------------------------

def test_compute_metric_stamps_current_version():
    assert SCORING_MODEL_VERSION == "peer-3"
    df = _universe(with_dollar_vol=True, with_fundamentals=True)
    from util import SCORE_WEIGHTS
    compute_metric(df, SCORE_WEIGHTS, _cfg(), None)
    assert (df["scoring_model_version"] == SCORING_MODEL_VERSION).all()
    assert df["value_metric"].notna().all()
