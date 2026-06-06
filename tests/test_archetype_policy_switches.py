"""
tests/test_archetype_policy_switches.py — the two per-archetype policy switches
(`thesis_exit_requires_confirmation`, `allow_deeper_drawdown`).

Covers:
  1. Round-trip: config -> _current_params() -> archetype_cfg_from_params() -> policy.
     Legacy 60-slot vectors drop the appended booleans (classifier falls back to defaults).
  2. _build_archetype_thresholds populates per-stock confirm/deepdd arrays from the policy.
  3. Behavior: allow_deeper_drawdown widens the catastrophic hard stop so a flagged
     archetype survives a drawdown that stops out an unflagged one.

Backtest-only by design — these flags are wired into the simulator, NOT the live
sell engine (promotion to live is gated on a survivorship-free A/B showing an effect).
"""
from __future__ import annotations

import datetime
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtesting.simulator import (
    _DEEPER_DD_FACTOR,
    _STOP_LOSS_PCT,
    _build_archetype_thresholds,
    run_simulation,
)
from backtesting.types import PrecomputedData
from tuning.constants import (
    _ARCH_BOOL_FIELDS,
    _ARCH_BOOL_SLOT_OFFSET,
    _ARCH_FIELDS,
    _ARCH_KEYS,
    _ARCH_SLOT_OFFSET,
    _current_params,
    archetype_cfg_from_params,
)

_OPEN_CS = {
    "mode": "percentile",
    "top_percentile": 1.0,
    "max_candidates": 15,
    "min_candidates": 1,
    "use_absolute_score_floor": False,
    "absolute_score_floor": 0.0,
    "min_quality_score": 0.0,
    "min_momentum_score": -999.0,
    "min_conditional_momentum_score": -999.0,
    "allow_income_defensive_exception": False,
    "fallback_thresholds": [],
    "min_post_cooldown_candidates": 1,
}

_NB = len(_ARCH_BOOL_FIELDS)
_NF = len(_ARCH_FIELDS)


def _bool_slot(arch_i: int, field: str) -> int:
    fi = [f[0] for f in _ARCH_BOOL_FIELDS].index(field)
    return _ARCH_BOOL_SLOT_OFFSET + arch_i * _NB + fi


def _num_slot(arch_i: int, field: str) -> int:
    fi = [f[0] for f in _ARCH_FIELDS].index(field)
    return _ARCH_SLOT_OFFSET + arch_i * _NF + fi


def _mk_precomp(stock_prices: np.ndarray) -> PrecomputedData:
    n_days, n = stock_prices.shape
    n_etfs = 2
    return PrecomputedData(
        symbols          = [f"STK{i}" for i in range(n)],
        prices           = stock_prices.astype(np.float64),
        pe_comp          = np.full(n, 0.5),
        pb_comp          = np.full(n, 0.5),
        quality_scores   = np.array([0.80, 0.75, 0.70, 0.65])[:n],
        income_scores    = np.full(n, 0.05),
        yield_trap_mask  = np.zeros(n, dtype=bool),
        bin_indices      = np.full(n, 2, dtype=np.int32),
        has_position_52w = np.ones(n, dtype=bool),
        position_52w_arr = np.full(n, 0.50),
        return_1m_arr    = np.zeros(n),
        etf_symbols      = [f"ETF{j}" for j in range(n_etfs)],
        etf_prices       = np.full((n_days, n_etfs), 200.0),
        baseline_scores  = np.full(n, 0.60),
        sector_labels    = ["Tech", "Health", "Finance", "Energy"][:n],
        volume_arr       = np.full(n, 2_000_000.0),
        mode             = "test",
        universe_selection = "test",
        lookahead_bias_level = "LOW",
        benchmark_prices = np.full(n_days, 300.0),
        benchmark_symbol = "SPY",
        position_52w_daily     = np.full((n_days, n), 0.50),
        return_1m_daily        = np.zeros((n_days, n)),
        bin_indices_daily      = np.full((n_days, n), 2, dtype=np.int32),
        has_position_52w_daily = np.ones((n_days, n), dtype=bool),
        ret_5d_daily  = None, ret_3m_daily = None, ret_6m_daily = None,
        rs_3m_daily   = None, rs_6m_daily  = None, vol_3m_daily = None,
        above_50dma_daily = None, above_200dma_daily = None, spy_prices = None,
    )


def test_policy_switch_booleans_roundtrip():
    """config -> vector -> archetype_cfg_from_params -> per-archetype entry."""
    p = np.asarray(_current_params(), float)
    cfg = archetype_cfg_from_params(p)
    # Shipped config: quality_compounder is the conviction archetype (both flags on).
    assert cfg["quality_compounder"]["thesis_exit_requires_confirmation"] is True
    assert cfg["quality_compounder"]["allow_deeper_drawdown"] is True
    # core_default keeps both off.
    assert cfg["core_default"]["thesis_exit_requires_confirmation"] is False
    assert cfg["core_default"]["allow_deeper_drawdown"] is False
    # Legacy 60-slot vector predates the appended group → booleans absent → classifier
    # falls back to its hardcoded per-archetype defaults.
    cfg_legacy = archetype_cfg_from_params(p[:60])
    assert "thesis_exit_requires_confirmation" not in cfg_legacy["quality_compounder"]


def test_build_thresholds_populates_switch_arrays():
    """_build_archetype_thresholds returns per-stock confirm/deepdd arrays from policy."""
    precomp = _mk_precomp(np.full((10, 4), 100.0))
    p_on = np.asarray(_current_params(), float).copy()
    p_off = p_on.copy()
    for ai in range(len(_ARCH_KEYS)):
        for field in ("thesis_exit_requires_confirmation", "allow_deeper_drawdown"):
            p_on[_bool_slot(ai, field)] = 1.0
            p_off[_bool_slot(ai, field)] = 0.0
    arr_on = _build_archetype_thresholds(
        precomp, 0.5, -0.1, 5, arch_cfg_override=archetype_cfg_from_params(p_on))
    arr_off = _build_archetype_thresholds(
        precomp, 0.5, -0.1, 5, arch_cfg_override=archetype_cfg_from_params(p_off))
    assert "confirm" in arr_on and "deepdd" in arr_on
    assert arr_on["confirm"].any() and arr_on["deepdd"].any()
    assert not arr_off["confirm"].any() and not arr_off["deepdd"].any()


def test_allow_deeper_drawdown_widens_hard_stop():
    """A drawdown between the normal stop and stop×factor stops out an unflagged
    archetype but is survived by a flagged (allow_deeper_drawdown) one."""
    # Drawdown midway between the hard stop and the widened (×factor) stop.
    target = _STOP_LOSS_PCT * ((1.0 + _DEEPER_DD_FACTOR) / 2.0)
    assert _STOP_LOSS_PCT * _DEEPER_DD_FACTOR < target < _STOP_LOSS_PCT  # (more neg ... less neg)
    low = 100.0 * (1.0 + target)

    n_days, n = 40, 4
    prices = np.full((n_days, n), 100.0)
    prices[8:, :] = low          # crash on day 8, never recovers, never rose above cost

    precomp = _mk_precomp(prices)

    def _params(deepdd: bool) -> np.ndarray:
        p = np.asarray(_current_params(), float).copy()
        for ai in range(len(_ARCH_KEYS)):
            # Gate the trailing stop off entirely (hold > horizon) so ONLY the hard stop
            # can fire — isolating the allow_deeper_drawdown effect.
            p[_num_slot(ai, "minimum_hold_days")] = 60.0
            p[_bool_slot(ai, "allow_deeper_drawdown")] = 1.0 if deepdd else 0.0
            p[_bool_slot(ai, "thesis_exit_requires_confirmation")] = 0.0
        return p

    common = dict(
        starting_capital=2000.0, weekly_contribution=200.0,
        rebalance_frequency_days=5, cs_params=_OPEN_CS,
        scope="active_sleeve_compounding", archetype_aware=True,
    )
    res_off = run_simulation(precomp, _params(deepdd=False), **common)
    res_on  = run_simulation(precomp, _params(deepdd=True), **common)

    # Normal hard stop catches the drawdown; the widened stop lets it ride.
    assert res_off.stopout_count > 0
    assert res_on.stopout_count < res_off.stopout_count


# ===========================================================================
# Live sell engine (promoted from backtest — sell_engine.py / manager.py)
# ===========================================================================

from portfolio.position_archetypes import get_archetype_policy  # noqa: E402
from portfolio.sell_engine import _DEEPER_DD_FACTOR as _LIVE_DD_FACTOR  # noqa: E402
from portfolio.sell_engine import _THESIS_CONFIRM_EVALS as _LIVE_CONFIRM_EVALS  # noqa: E402
from portfolio.sell_engine import SellDecisionEngine  # noqa: E402
from util import SELL_RULES  # noqa: E402


def _policy(archetype: str, **flags):
    return get_archetype_policy(archetype, {"enabled": True, archetype: flags})


def test_live_allow_deeper_drawdown_widens_hard_stop():
    """sell_engine: a drawdown between the normal stop and stop×factor stops out an
    unflagged position but is held by a flagged (allow_deeper_drawdown) one."""
    stop = float(SELL_RULES["stop_loss_pct"])
    target = stop * ((1.0 + _LIVE_DD_FACTOR) / 2.0)   # between stop and stop×factor
    holding = {
        "quantity": 1, "price": 100.0 * (1.0 + target),
        "average_buy_price": 100.0, "percent_change": str(target * 100.0),
    }
    metrics = pd.Series({"value_metric": 0.5, "quality_score": 0.9, "yield_trap_flag": False})
    eng = SellDecisionEngine()

    d_off = eng.evaluate("AAA", holding, metrics,
                         archetype_policy=_policy("quality_compounder", allow_deeper_drawdown=False))
    d_on = eng.evaluate("AAA", holding, metrics,
                        archetype_policy=_policy("quality_compounder", allow_deeper_drawdown=True))
    assert d_off.should_sell and d_off.exit_type == "failure_exit"
    assert not d_on.should_sell


def test_live_thesis_exit_requires_confirmation():
    """sell_engine: a flagged archetype holds on the first weak eval and exits only once
    the weak streak reaches the confirmation count; an unflagged one exits immediately."""
    sell_weak = float(SELL_RULES["sell_weak_value_below"])
    created = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.timedelta(days=90)).isoformat()
    holding = {"quantity": 1, "price": 95.0, "average_buy_price": 100.0,
               "percent_change": "-5.0", "created_at": created}
    metrics = pd.Series({"value_metric": sell_weak - 0.1, "quality_score": 0.9,
                         "yield_trap_flag": False})
    eng = SellDecisionEngine()
    pol_confirm = _policy("quality_compounder", thesis_exit_requires_confirmation=True)

    # First weak eval: held, streak advances to 1.
    d1 = eng.evaluate("AAA", holding, metrics, archetype_policy=pol_confirm, weak_streak=0)
    assert not d1.should_sell
    assert d1.weak_streak_next == 1
    # Streak reaches the confirmation count → thesis exit fires.
    d2 = eng.evaluate("AAA", holding, metrics, archetype_policy=pol_confirm,
                      weak_streak=_LIVE_CONFIRM_EVALS - 1)
    assert d2.should_sell and d2.exit_type == "thesis_exit"
    assert d2.weak_streak_next == _LIVE_CONFIRM_EVALS
    # Unflagged archetype exits on the first weak eval (no confirmation required).
    d_now = eng.evaluate("AAA", holding, metrics,
                         archetype_policy=_policy("core_default",
                                                  thesis_exit_requires_confirmation=False),
                         weak_streak=0)
    assert d_now.should_sell and d_now.exit_type == "thesis_exit"


def test_apply_config_params_persists_archetype_slots(tmp_path, monkeypatch):
    """apply_config_params now writes the tuned archetype lifecycle slots back to config
    (previously dropped), including the appended policy-switch booleans."""
    from tuning import reports
    src_cfg = os.path.join(os.path.dirname(__file__), "..", "cfg", "config.yaml")
    tmp_cfg = tmp_path / "config.yaml"
    shutil.copy(src_cfg, tmp_cfg)
    monkeypatch.setattr(reports, "CONFIG_FILE", str(tmp_cfg))

    params = np.asarray(_current_params(), float).copy()
    qc = _ARCH_KEYS.index("quality_compounder")
    params[_num_slot(qc, "harvest_profit_threshold")] = 0.42
    params[_bool_slot(qc, "thesis_exit_requires_confirmation")] = 0.0   # flip True -> False
    reports.apply_config_params(params)

    written = yaml.safe_load(open(tmp_cfg))["archetype_management"]["quality_compounder"]
    assert written["harvest_profit_threshold"] == 0.42
    assert written["thesis_exit_requires_confirmation"] is False
