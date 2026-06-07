"""
tests/test_regime_unification.py — one regime definition across live + backtest.

The live detector (strategy/regimes/detector.py) and the backtest (simulator._detect_regime)
now share `strategy/regimes/classifier.classify_regime`, so a regime means the same thing
whether you tune/validate or deploy. Covers:
  1. classify_regime reproduces the live detector's label across a (spy, ma200, vix) battery
     — the single-source-of-truth guarantee (catches any future drift).
  2. The VIX-primary thresholds + the vix=None fallback.
  3. The backtest uses VIX when present (defensive ⇔ VIX≥30, even with SPY above 200DMA).
  4. Behavior gate: with no VIX, _detect_regime is the legacy SPY-vs-200DMA rule (unchanged).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtesting.simulator import _detect_regime
from backtesting.types import PrecomputedData
from strategy.regimes.classifier import classify_regime
from strategy.regimes.detector import RegimeDetector


def test_classify_regime_matches_live_detector():
    """classify_regime's label == RegimeDetector.detect_from_data().regime for every case."""
    det = RegimeDetector()
    rc = det._rc
    battery = [
        (450, 400, 35), (450, 400, 25), (450, 400, 12), (450, 400, 14),
        (380, 400, 12), (380, 400, 25), (380, 400, 35), (380, 400, 19),
        (450, 400, None), (380, 400, None), (None, None, 40), (None, None, None),
    ]
    for spy, ma, vix in battery:
        assert classify_regime(spy, ma, vix, rc) == det.detect_from_data(spy, ma, vix).regime, (spy, ma, vix)


def test_vix_primary_thresholds():
    det = RegimeDetector()
    rc = det._rc
    assert classify_regime(450, 400, 35, rc) == "defensive"   # VIX >= 30
    assert classify_regime(450, 400, 25, rc) == "neutral"     # 20 <= VIX < 30
    assert classify_regime(450, 400, 12, rc) == "bullish"     # VIX < 20 and SPY > 200DMA
    assert classify_regime(380, 400, 12, rc) == "neutral"     # VIX < 20 and SPY < 200DMA


def test_vix_none_fallback_never_defensive():
    det = RegimeDetector()
    rc = det._rc
    assert classify_regime(450, 400, None, rc) == "bullish"   # SPY > 200DMA
    assert classify_regime(380, 400, None, rc) == "neutral"   # SPY < 200DMA (NOT defensive)


def _bench_precomp(bench: np.ndarray, vix: np.ndarray | None) -> PrecomputedData:
    """Minimal precomp carrying only what _detect_regime reads."""
    n = len(bench)
    z = np.zeros(1)
    return PrecomputedData(
        symbols=["X"], prices=np.ones((n, 1)), pe_comp=z, pb_comp=z, quality_scores=z,
        income_scores=z, yield_trap_mask=np.zeros(1, bool), bin_indices=np.zeros(1, np.int32),
        has_position_52w=np.ones(1, bool), position_52w_arr=z, return_1m_arr=z,
        etf_symbols=[], etf_prices=np.zeros((n, 0)), baseline_scores=z, sector_labels=["X"],
        volume_arr=np.ones(1), mode="t", universe_selection="t", lookahead_bias_level="LOW",
        benchmark_prices=bench.astype(float), benchmark_symbol="SPY",
        position_52w_daily=np.zeros((n, 1)), return_1m_daily=np.zeros((n, 1)),
        bin_indices_daily=np.zeros((n, 1), np.int32), has_position_52w_daily=np.ones((n, 1), bool),
        vix_prices=(vix.astype(float) if vix is not None else None),
    )


def test_backtest_uses_vix_when_present():
    """With VIX present, a calm uptrend (SPY well above 200DMA) is 'defensive' iff VIX>=30."""
    bench = np.linspace(100, 200, 260)          # steady uptrend → SPY >> 200DMA
    pc_spike = _bench_precomp(bench, np.full(260, 35.0))   # VIX spike
    pc_calm = _bench_precomp(bench, np.full(260, 12.0))    # low VIX
    assert _detect_regime(pc_spike, 259) == "defensive"   # VIX-driven, despite SPY uptrend
    assert _detect_regime(pc_calm, 259) == "bullish"


def test_backtest_no_vix_is_legacy_spy_rule():
    """Behavior gate: vix_prices=None → legacy SPY-vs-200DMA (incl. the 0.95 defensive band)."""
    # Sharp drop so the last day sits >5% below its trailing 200DMA.
    bench = np.concatenate([np.linspace(100, 150, 200), np.linspace(150, 95, 60)])
    pc = _bench_precomp(bench, None)
    assert _detect_regime(pc, 259) == "defensive"         # SPY < 0.95×MA200, legacy rule
    # Steady uptrend with no VIX → bullish.
    pc_up = _bench_precomp(np.linspace(100, 200, 260), None)
    assert _detect_regime(pc_up, 259) == "bullish"
