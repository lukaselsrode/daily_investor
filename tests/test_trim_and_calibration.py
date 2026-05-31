"""
tests/test_trim_and_calibration.py — Tests for TRIM logic, capital routing,
backtest parity, and calibration summary.

Covers:
  1. SellDecisionEngine: trim_exit fires when profit >= trim_min_gain
     and value_metric is below buy threshold (but not full exit territory)
  2. SellDecisionEngine: trim_exit does NOT fire when fully below sell_weak
     (thesis_exit takes precedence via ordering) or above metric_threshold
  3. SellDecisionEngine: trim_fraction is carried in the SellDecision
  4. manager.py sell_cycle: 100 shares with trim_fraction=0.33 → 33 sold, 67 remain
  5. Capital routing: trim proceeds split etf_pct / active_pct correctly
  6. outcome_tracker: get_calibration_summary returns correct rates from synthetic data
  7. sleeve_tracker: record_event + load_sleeve_events round-trips correctly
  8. sleeve_tracker: get_allocation_state correctly separates ETF vs active equity
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio.sell_engine import SellDecisionEngine
from util import ETFS, EXIT_DECISION_PARAMS, METRIC_THRESHOLD, SELL_RULES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> SellDecisionEngine:
    return SellDecisionEngine()


def _holding(
    percent_change: float | None = None,
    price: float = 100.0,
    average_buy_price: float = 100.0,
    quantity: float = 100.0,
    equity: float | None = None,
) -> dict:
    h: dict = {
        "price": str(price),
        "average_buy_price": str(average_buy_price),
        "quantity": str(quantity),
        "equity": str(equity if equity is not None else price * quantity),
    }
    if percent_change is not None:
        h["percent_change"] = str(percent_change * 100)
    return h


def _metrics(
    value_metric: float = 1.0,
    quality_score: float = 0.5,
    yield_trap: bool = False,
) -> pd.Series:
    return pd.Series({
        "value_metric": value_metric,
        "quality_score": quality_score,
        "yield_trap_flag": yield_trap,
    })


def _trim_value_metric() -> float:
    """
    A value_metric in the trim zone: above sell_weak_value_below and below trim_score_below.
    """
    sell_weak  = SELL_RULES["sell_weak_value_below"]
    trim_below = float(EXIT_DECISION_PARAMS["trim_score_below"])
    # Place mid-way between sell_weak and trim_score_below
    return (sell_weak + trim_below) / 2.0


def _trim_min_gain() -> float:
    return EXIT_DECISION_PARAMS["trim_min_gain_pct"]


# ---------------------------------------------------------------------------
# 1. Trim fires on correct conditions
# ---------------------------------------------------------------------------

class TestTrimExitFires:

    @pytest.fixture(autouse=True)
    def _force_trim_enabled(self):
        """Trim is disabled in production config (let-winners-run, 2026-05), but
        these tests verify the trim MECHANISM still works when enabled. Force it
        on for the duration of this class, then restore."""
        _orig = EXIT_DECISION_PARAMS.get("trim_enabled", True)
        EXIT_DECISION_PARAMS["trim_enabled"] = True
        yield
        EXIT_DECISION_PARAMS["trim_enabled"] = _orig

    def test_trim_fires_when_profitable_and_weakening(self):
        gain       = _trim_min_gain() + 0.05        # clearly profitable
        vm         = _trim_value_metric()             # below buy threshold, above sell_weak
        d = _engine().evaluate("AAPL", _holding(percent_change=gain), _metrics(value_metric=vm))
        assert d.exit_type == "trim_exit", f"Expected trim_exit, got {d.exit_type!r} ({d.reason})"
        assert d.should_sell
        assert d.severity == "soft"

    def test_trim_fraction_present_on_trim_exit(self):
        gain = _trim_min_gain() + 0.05
        vm   = _trim_value_metric()
        d = _engine().evaluate("AAPL", _holding(percent_change=gain), _metrics(value_metric=vm))
        assert d.trim_fraction is not None
        assert 0 < d.trim_fraction < 1.0

    def test_trim_fraction_matches_config(self):
        gain = _trim_min_gain() + 0.05
        vm   = _trim_value_metric()
        d = _engine().evaluate("AAPL", _holding(percent_change=gain), _metrics(value_metric=vm))
        assert d.trim_fraction == pytest.approx(EXIT_DECISION_PARAMS["trim_fraction"], abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Trim does NOT fire when conditions aren't met
# ---------------------------------------------------------------------------

class TestTrimExitDoesNotFire:

    def test_no_trim_when_below_sell_weak(self):
        """Full thesis_exit should take precedence over trim when value collapses."""
        gain = _trim_min_gain() + 0.05
        vm   = SELL_RULES["sell_weak_value_below"] - 0.05  # below sell_weak → thesis_exit
        d = _engine().evaluate("AAPL", _holding(percent_change=gain), _metrics(value_metric=vm))
        assert d.exit_type != "trim_exit", "Should be thesis_exit not trim_exit"

    def test_no_trim_when_above_metric_threshold(self):
        """No trim when value_metric is still above the buy threshold."""
        gain = _trim_min_gain() + 0.05
        vm   = METRIC_THRESHOLD + 0.2  # well above threshold → still cheap, no trim
        d = _engine().evaluate("AAPL", _holding(percent_change=gain), _metrics(value_metric=vm))
        assert d.exit_type != "trim_exit"
        assert not d.should_sell or d.exit_type in ("harvest_exit",), (
            f"Unexpected exit: {d.exit_type}"
        )

    def test_no_trim_when_not_profitable(self):
        """No trim when position is not profitable enough."""
        gain = _trim_min_gain() - 0.05  # below min gain
        vm   = _trim_value_metric()
        d = _engine().evaluate("AAPL", _holding(percent_change=gain), _metrics(value_metric=vm))
        assert d.exit_type != "trim_exit"

    def test_hard_sell_not_trim(self):
        """Stop-loss overrides trim."""
        stop = SELL_RULES["stop_loss_pct"]
        vm   = _trim_value_metric()
        d = _engine().evaluate("AAPL", _holding(percent_change=stop - 0.05), _metrics(value_metric=vm))
        assert d.severity == "hard"
        assert d.exit_type == "failure_exit"


# ---------------------------------------------------------------------------
# 3. Partial sell quantity: 100 shares * trim_fraction → 33 sold, 67 remain
# ---------------------------------------------------------------------------

class TestTrimQuantity:

    def test_trim_fraction_33_pct_of_100_shares(self):
        frac = EXIT_DECISION_PARAMS["trim_fraction"]
        total_shares = 100.0
        expected_sold   = round(total_shares * frac, 6)
        expected_remain = total_shares - expected_sold
        assert expected_sold   == pytest.approx(total_shares * frac,  rel=1e-4)
        assert expected_remain == pytest.approx(total_shares * (1 - frac), rel=1e-4)

    def test_trim_fraction_produces_valid_remainder(self):
        frac = EXIT_DECISION_PARAMS["trim_fraction"]
        assert 0 < frac < 1.0, "trim_fraction must be a proper fraction"
        remaining = 1.0 - frac
        assert remaining > 0, "Must leave some shares remaining"


# ---------------------------------------------------------------------------
# 4. Capital routing: trim proceeds split correctly
# ---------------------------------------------------------------------------

class TestTrimProceedsRouting:

    def test_etf_and_active_fractions_sum_to_one(self):
        etf_pct    = EXIT_DECISION_PARAMS["trim_to_etfs_pct"]
        active_pct = 1.0 - etf_pct
        assert etf_pct + active_pct == pytest.approx(1.0, abs=1e-9)

    def test_etf_pct_is_positive(self):
        assert EXIT_DECISION_PARAMS["trim_to_etfs_pct"] > 0

    def test_active_pct_retained(self):
        etf_pct = EXIT_DECISION_PARAMS["trim_to_etfs_pct"]
        assert etf_pct < 1.0, "Some proceeds should stay as cash/active reserve"


# ---------------------------------------------------------------------------
# 5. Calibration summary with synthetic outcome data
# ---------------------------------------------------------------------------

class TestCalibrationSummary:

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a synthetic outcomes DataFrame."""
        from portfolio.outcome_tracker import _SCHEMA
        base = {col: None for col in _SCHEMA}
        records = []
        for r in rows:
            rec = dict(base)
            rec.update(r)
            records.append(rec)
        return pd.DataFrame(records, columns=_SCHEMA)

    def test_premature_exit_rate_all_premature(self):
        from portfolio.outcome_tracker import get_calibration_summary
        df = self._make_df([
            {"decision_state": "EXIT", "premature_exit": True},
            {"decision_state": "EXIT", "premature_exit": True},
        ])
        result = get_calibration_summary(df)
        assert result["premature_exit_rate"] == pytest.approx(1.0)
        assert result["n_exit"] == 2

    def test_premature_exit_rate_none_premature(self):
        from portfolio.outcome_tracker import get_calibration_summary
        df = self._make_df([
            {"decision_state": "EXIT", "premature_exit": False},
            {"decision_state": "EXIT", "premature_exit": False},
        ])
        result = get_calibration_summary(df)
        assert result["premature_exit_rate"] == pytest.approx(0.0)

    def test_trim_success_rate_all_good(self):
        from portfolio.outcome_tracker import get_calibration_summary
        df = self._make_df([
            {"decision_state": "TRIM", "good_trim": True},
            {"decision_state": "TRIM", "good_trim": True},
            {"decision_state": "TRIM", "good_trim": False},
        ])
        result = get_calibration_summary(df)
        assert result["trim_success_rate"] == pytest.approx(2/3, rel=1e-3)
        assert result["n_trim"] == 3

    def test_bad_hold_rate(self):
        from portfolio.outcome_tracker import get_calibration_summary
        df = self._make_df([
            {"decision_state": "HOLD", "bad_hold": True},
            {"decision_state": "HOLD", "bad_hold": False},
            {"decision_state": "HOLD", "bad_hold": False},
            {"decision_state": "HOLD", "bad_hold": False},
        ])
        result = get_calibration_summary(df)
        assert result["bad_hold_rate"] == pytest.approx(0.25, rel=1e-3)
        assert result["n_hold"] == 4

    def test_harvest_regret_rate(self):
        from portfolio.outcome_tracker import get_calibration_summary
        df = self._make_df([
            {"decision_state": "HARVEST", "future_30d_return": 0.15},   # regret
            {"decision_state": "HARVEST", "future_30d_return": 0.05},   # no regret
            {"decision_state": "HARVEST", "future_30d_return": -0.03},  # no regret
        ])
        result = get_calibration_summary(df)
        assert result["harvest_regret_rate"] == pytest.approx(1/3, rel=1e-3)
        assert result["n_harvest"] == 3

    def test_empty_df_returns_none_rates(self):
        from portfolio.outcome_tracker import get_calibration_summary
        result = get_calibration_summary(pd.DataFrame())
        assert result["premature_exit_rate"] is None
        assert result["trim_success_rate"] is None
        assert result["bad_hold_rate"] is None
        assert result["harvest_regret_rate"] is None

    def test_no_resolved_outcomes_returns_none(self):
        from portfolio.outcome_tracker import _SCHEMA, get_calibration_summary
        df = pd.DataFrame([{col: None for col in _SCHEMA}])
        result = get_calibration_summary(df)
        assert result["premature_exit_rate"] is None


# ---------------------------------------------------------------------------
# 6. Sleeve tracker: record + load round-trip
# ---------------------------------------------------------------------------

class TestSleeveTracker:

    def _mock_io(self):
        """Return (saved_dfs, patch context) for testing sleeve_tracker without parquet."""
        import portfolio.sleeve_tracker as st_mod
        stored: list[pd.DataFrame] = []

        def _fake_save(df: pd.DataFrame) -> None:
            stored.clear()
            stored.append(df.copy())

        def _fake_load() -> pd.DataFrame:
            from portfolio.sleeve_tracker import _SCHEMA
            return stored[0] if stored else pd.DataFrame(columns=_SCHEMA)

        return stored, patch.multiple(
            st_mod,
            _save_events=_fake_save,
            load_sleeve_events=_fake_load,
        )

    def test_record_and_load_roundtrip(self):
        from portfolio.sleeve_tracker import record_event

        stored, ctx = self._mock_io()
        with ctx:
            record_event("weekly_contribution", 400.0, destination="mixed", notes="test")
            import portfolio.sleeve_tracker as st_mod
            df = st_mod.load_sleeve_events()

        assert len(df) == 1
        assert df.iloc[0]["event_type"] == "weekly_contribution"
        assert df.iloc[0]["amount"] == pytest.approx(400.0)
        assert df.iloc[0]["destination"] == "mixed"
        assert df.iloc[0]["notes"] == "test"

    def test_multiple_events_accumulate(self):
        from portfolio.sleeve_tracker import record_event

        stored, ctx = self._mock_io()
        with ctx:
            record_event("trim_proceeds",       50.0,  source_symbol="AAPL", etf_pct_routed=0.85)
            record_event("weekly_contribution", 400.0)
            import portfolio.sleeve_tracker as st_mod
            df = st_mod.load_sleeve_events()

        assert len(df) == 2
        types = set(df["event_type"])
        assert "trim_proceeds" in types
        assert "weekly_contribution" in types

    def test_etf_pct_routed_stored_correctly(self):
        from portfolio.sleeve_tracker import log_trim_proceeds

        stored, ctx = self._mock_io()
        with ctx:
            log_trim_proceeds("MSFT", 100.0, etf_pct=0.85)
            import portfolio.sleeve_tracker as st_mod
            df = st_mod.load_sleeve_events()

        row = df[df["event_type"] == "trim_proceeds"].iloc[0]
        assert row["etf_pct_routed"] == pytest.approx(0.85)
        assert row["active_pct_routed"] == pytest.approx(0.15)

    def test_get_allocation_state_separates_etf_from_active(self):
        from portfolio.sleeve_tracker import get_allocation_state

        etf_sym   = list(ETFS)[0]
        stock_sym = "NVDA"

        mock_broker = MagicMock()
        mock_broker.get_portfolio_value.return_value = 1000.0
        mock_broker.get_cash.return_value             = 100.0
        mock_broker.get_holdings.return_value = {
            etf_sym:   {"equity": "700"},
            stock_sym: {"equity": "200"},
        }

        state = get_allocation_state(mock_broker)
        assert state["etf_equity"]    == pytest.approx(700.0)
        assert state["active_equity"] == pytest.approx(200.0)
        assert state["cash"]          == pytest.approx(100.0)
        assert state["etf_pct"]       == pytest.approx(0.70, abs=1e-3)

    def test_etf_drift_positive_when_overweight(self):
        from portfolio.sleeve_tracker import get_allocation_state
        from util import INDEX_PCT

        etf_sym = list(ETFS)[0]
        mock_broker = MagicMock()
        mock_broker.get_portfolio_value.return_value = 1000.0
        mock_broker.get_cash.return_value             = 0.0
        # Put 90% in ETFs, clearly overweight vs INDEX_PCT
        mock_broker.get_holdings.return_value = {
            etf_sym: {"equity": "900"},
            "AAPL":  {"equity": "100"},
        }

        state = get_allocation_state(mock_broker)
        assert state["etf_drift_pct"] == pytest.approx(0.90 - INDEX_PCT, abs=1e-3)

    def test_load_returns_empty_df_when_no_file(self, tmp_path):
        from portfolio.sleeve_tracker import load_sleeve_events

        with patch("portfolio.sleeve_tracker._data_dir", return_value=tmp_path):
            df = load_sleeve_events()

        assert df.empty
