"""
tests/test_pit_wiring.py — point-in-time fundamentals wiring (P2-P5).

Covers: config defaults; the simulator's _pit_or_static routing (daily when present,
static 1D otherwise = byte-identical PIT-off); that the candidate GATE (select_candidates)
uses PIT daily quality, not just the composite (the look-ahead-via-gate fix); the
precompute's neutral-missing + causal behavior (monkeypatched, cache-independent); and
that no production/core package imports from research/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from backtesting.types import PrecomputedData

# ---------------------------------------------------------------------------
# 1. Config defaults: PIT on, static fallback off (honest survivorship-free modes)
# ---------------------------------------------------------------------------

def test_pit_config_defaults():
    from util import BACKTEST_PARAMS
    assert BACKTEST_PARAMS["point_in_time_fundamentals"] is True
    assert BACKTEST_PARAMS["allow_static_fundamentals_fallback"] is False


def test_config_manager_backtest_exposes_pit_flags():
    """config/manager.backtest must pass the PIT + survivorship flags through (typed parity)."""
    from config.manager import ConfigManager
    bt = ConfigManager.from_dict({"backtest": {
        "survivorship_free": True,
        "point_in_time_fundamentals": True,
        "allow_static_fundamentals_fallback": False,
    }}).backtest
    assert bt.survivorship_free is True
    assert bt.point_in_time_fundamentals is True
    assert bt.allow_static_fundamentals_fallback is False


def test_pit_disabled_for_walk_forward_price_only():
    """Regression: walk_forward_price_only_test stays price-only — PIT must NOT rebuild the
    fundamentals it zeroes. PIT stays on for the normal survivorship-free path."""
    from backtesting.data_loader import _pit_enabled_for
    assert _pit_enabled_for("walk_forward_price_only_test", True) is False
    assert _pit_enabled_for("liquid_universe_full", True) is True
    assert _pit_enabled_for("liquid_universe_full", False) is False  # not surv-free → no PIT


# ---------------------------------------------------------------------------
# Minimal precomp builder (n_days x n_stocks) for the simulator-facing tests
# ---------------------------------------------------------------------------

def _mk_precomp(n_days=30, n_stocks=4, *, with_pit=False,
                static_quality=0.9, daily_quality=0.1):
    z = np.zeros(n_stocks)
    prices = np.cumprod(1 + np.full((n_days, n_stocks), 0.001), axis=0) * 100
    kw = dict(
        symbols=[f"S{i}" for i in range(n_stocks)],
        prices=prices,
        pe_comp=z.copy(), pb_comp=z.copy(),
        quality_scores=np.full(n_stocks, static_quality),
        income_scores=np.full(n_stocks, 0.5),
        yield_trap_mask=np.zeros(n_stocks, bool),
        bin_indices=np.zeros(n_stocks, np.int32),
        has_position_52w=np.ones(n_stocks, bool),
        position_52w_arr=z.copy(), return_1m_arr=z.copy(),
        etf_symbols=[], etf_prices=np.zeros((n_days, 0)),
        baseline_scores=z.copy(), sector_labels=["X"] * n_stocks,
        volume_arr=np.ones(n_stocks) * 1e6,
        mode="liquid_universe_full", universe_selection="liquid_all",
        lookahead_bias_level="MEDIUM",
        benchmark_prices=np.linspace(100, 110, n_days),
        benchmark_symbol="SPY",
        position_52w_daily=np.zeros((n_days, n_stocks)),
        return_1m_daily=np.zeros((n_days, n_stocks)),
        bin_indices_daily=np.zeros((n_days, n_stocks), np.int32),
        has_position_52w_daily=np.ones((n_days, n_stocks), bool),
    )
    if with_pit:
        kw.update(
            pe_comp_daily=np.zeros((n_days, n_stocks)),
            pb_comp_daily=np.zeros((n_days, n_stocks)),
            quality_scores_daily=np.full((n_days, n_stocks), daily_quality),
            income_scores_daily=np.full((n_days, n_stocks), 0.5),
        )
    return PrecomputedData(**kw)


# ---------------------------------------------------------------------------
# 2. _pit_or_static routes to daily when present, static otherwise
# ---------------------------------------------------------------------------

def test_pit_or_static_routing():
    from backtesting.simulator import _pit_or_static
    static = _mk_precomp(with_pit=False)
    pit = _mk_precomp(with_pit=True, static_quality=0.9, daily_quality=0.1)
    # static path -> 1D static quality (0.9)
    assert _pit_or_static(static, 5)[2][0] == 0.9
    # PIT path -> day slice (0.1), NOT the static 0.9
    assert _pit_or_static(pit, 5)[2][0] == 0.1


# ---------------------------------------------------------------------------
# 3. The candidate GATE uses PIT daily quality (look-ahead-via-gate fix)
# ---------------------------------------------------------------------------

def _loose_cs(min_quality):
    return {
        "mode": "percentile", "top_percentile": 1.0, "max_candidates": 10,
        "min_candidates": 1, "use_absolute_score_floor": False,
        "absolute_score_floor": -99.0, "min_quality_score": min_quality,
        "min_momentum_score": -99.0, "min_conditional_momentum_score": -99.0,
        "allow_income_defensive_exception": False,
    }


def test_select_candidates_gate_uses_pit_daily_quality():
    from backtesting.simulator import select_candidates
    params = np.zeros(16)            # len<40 -> no slot override; uses cs_params
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    cs = _loose_cs(min_quality=0.5)

    # Static path: static quality 0.9 >= 0.5 -> names pass the quality gate.
    mask_static, _ = select_candidates(5, scores, _mk_precomp(with_pit=False), params, cs)
    assert mask_static.sum() > 0

    # PIT path: daily quality 0.1 < 0.5 -> gate must EXCLUDE all (uses PIT daily, not static 0.9).
    mask_pit, _ = select_candidates(
        5, scores, _mk_precomp(with_pit=True, static_quality=0.9, daily_quality=0.1), params, cs
    )
    assert mask_pit.sum() == 0


# ---------------------------------------------------------------------------
# 4. Precompute: neutral missing + causal (monkeypatched, no cache needed)
# ---------------------------------------------------------------------------

def test_pit_precompute_causal_cross_sectional(monkeypatch):
    """Two priced names, both with fundamentals: S0's EPS steps up mid-window (its PE drops
    from 20 to 10, matching S1's), so S0's value sub-score must IMPROVE after the filing —
    a causal, cross-sectional change driven only by data filed before each rebalance date."""
    import data.pit_fundamentals as pf
    from backtesting.pit_precompute import build_pit_factor_panels
    from util import SCORING_PARAMS

    series = {
        "S0": pd.DataFrame({
            "_fd": pd.to_datetime(["2023-01-15", "2023-07-15"]),
            "ttm_eps": [5.0, 10.0], "shares": [1e6, 1e6], "book": [1e7, 1e7],
        }),
        "S1": pd.DataFrame({
            "_fd": pd.to_datetime(["2023-01-15"]),
            "ttm_eps": [10.0], "shares": [1e6], "book": [1e7],
        }),
    }
    monkeypatch.setattr(pf, "causal_ttm_series", lambda s: series.get(s))
    monkeypatch.setattr(pf, "dividend_records", lambda s: None)

    dates = pd.bdate_range("2023-02-01", "2023-12-29")  # spans the 2nd S0 filing
    nd = len(dates)
    prices = np.full((nd, 2), 100.0)
    out = build_pit_factor_panels(
        ["S0", "S1"], dates, prices, ["X", "X"], ["I", "I"],
        None, np.ones(2) * 1e6, 5, SCORING_PARAMS,
    )
    for a in out.values():
        assert a.shape == (nd, 2)
    pe = out["pe_comp_daily"]
    before = pe[dates.get_indexer([pd.Timestamp("2023-06-01")], method="ffill")[0], 0]
    after = pe[dates.get_indexer([pd.Timestamp("2023-09-01")], method="ffill")[0], 0]
    # Before the 2nd filing S0 (PE 20) is pricier than S1 (PE 10) -> lower value score;
    # after, S0's PE drops to 10 (tie) -> value score must rise. Strictly causal.
    assert after > before


def test_pit_precompute_missing_symbol_is_neutral_constant(monkeypatch):
    """A symbol with no cached fundamentals carries NO time-varying value signal (its value
    sub-score is constant across the window — the scorer's missing-data neutral, not leakage)."""
    import data.pit_fundamentals as pf
    from backtesting.pit_precompute import build_pit_factor_panels
    from util import SCORING_PARAMS

    series = {"S0": pd.DataFrame({
        "_fd": pd.to_datetime(["2023-01-15", "2023-07-15"]),
        "ttm_eps": [5.0, 10.0], "shares": [1e6, 1e6], "book": [1e7, 1e7],
    })}
    monkeypatch.setattr(pf, "causal_ttm_series", lambda s: series.get(s))
    monkeypatch.setattr(pf, "dividend_records", lambda s: None)
    dates = pd.bdate_range("2023-02-01", "2023-12-29")
    prices = np.full((len(dates), 2), 100.0)
    out = build_pit_factor_panels(
        ["S0", "S1"], dates, prices, ["X", "X"], ["I", "I"],
        None, np.ones(2) * 1e6, 5, SCORING_PARAMS,
    )
    # S1 (index 1) has no fundamentals -> constant value sub-score over time (no info leak).
    assert np.ptp(out["pe_comp_daily"][:, 1]) == 0.0
    assert np.ptp(out["pb_comp_daily"][:, 1]) == 0.0


def test_pit_precompute_accepts_object_dtype_dates(monkeypatch):
    """Regression: the real loader passes an object/string-dtype date index; the precompute
    must coerce it to datetime64 so searchsorted against filing dates doesn't TypeError."""
    import data.pit_fundamentals as pf
    from backtesting.pit_precompute import build_pit_factor_panels
    from util import SCORING_PARAMS

    series = {"S0": pd.DataFrame({
        "_fd": pd.to_datetime(["2023-01-15", "2023-07-15"]),
        "ttm_eps": [5.0, 10.0], "shares": [1e6, 1e6], "book": [1e7, 1e7],
    })}
    monkeypatch.setattr(pf, "causal_ttm_series", lambda s: series.get(s))
    monkeypatch.setattr(
        pf, "dividend_records",
        lambda s: (np.array(["2023-03-01", "2023-09-01"], dtype="datetime64[ns]"),
                   np.array([0.5, 0.5])) if s == "S0" else None,
    )
    dates = pd.Index([str(d.date()) for d in pd.bdate_range("2023-02-01", "2023-12-29")])
    prices = np.full((len(dates), 2), 100.0)
    out = build_pit_factor_panels(["S0", "S1"], dates, prices, ["X", "X"], ["I", "I"],
                                  None, np.ones(2) * 1e6, 5, SCORING_PARAMS)
    assert out["pe_comp_daily"].shape == (len(dates), 2)


def test_pit_precompute_raises_when_no_fundamentals(monkeypatch):
    import data.pit_fundamentals as pf
    from backtesting.pit_precompute import build_pit_factor_panels
    from util import SCORING_PARAMS

    monkeypatch.setattr(pf, "causal_ttm_series", lambda s: None)
    monkeypatch.setattr(pf, "dividend_records", lambda s: None)
    dates = pd.bdate_range("2023-01-02", "2023-06-30")
    prices = np.full((len(dates), 2), 100.0)
    import pytest
    with pytest.raises(RuntimeError):
        build_pit_factor_panels(["A", "B"], dates, prices, ["X", "X"], ["I", "I"],
                                None, None, 5, SCORING_PARAMS)


# ---------------------------------------------------------------------------
# 5. No production/core package imports from research/
# ---------------------------------------------------------------------------

def test_no_research_imports_in_core_packages():
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    core_pkgs = ["backtesting", "strategy", "portfolio", "tuning", "data", "execution",
                 "config", "core", "reporting"]
    # Pre-existing compatibility shim (FactorResearchEngine moved to research/ic_engine);
    # NOT introduced by the PIT work. Everything else (incl. all PIT code) must be clean.
    allow = {"strategy/research/__init__.py"}
    pat = re.compile(r"^\s*(from|import)\s+research(\.|\s|$)", re.M)
    offenders = []
    for pkg in core_pkgs:
        for p in (src / pkg).rglob("*.py"):
            rel = str(p.relative_to(src))
            if rel in allow:
                continue
            if pat.search(p.read_text()):
                offenders.append(rel)
    assert not offenders, f"core packages import research/: {offenders}"
    # The PIT modules specifically must never import research/.
    for f in ("data/pit_fundamentals.py", "backtesting/pit_precompute.py"):
        assert not pat.search((src / f).read_text())
