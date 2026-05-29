"""
tests/test_robust_profiles.py — Profile expansion and RobustScanResult aggregation.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from tuning.profiles import (
    HORIZON_PROFILES,
    ROBUSTNESS_PROFILES,
    effort_caption,
    expand_run_matrix,
    total_simulations,
)

# ---------------------------------------------------------------------------
# Profile expansion
# ---------------------------------------------------------------------------

class TestExpandRunMatrix:

    def test_quick_mixed_expands_to_6_cells(self):
        matrix = expand_run_matrix("quick", "mixed")
        assert len(matrix) == len(HORIZON_PROFILES["mixed"]) * len(ROBUSTNESS_PROFILES["quick"]["seeds"])
        assert len(matrix) == 6  # 6 horizons × 1 seed

    def test_standard_medium_expands_to_9_cells(self):
        matrix = expand_run_matrix("standard", "medium")
        assert len(matrix) == 3 * 3  # 3 horizons × 3 seeds
        assert len(matrix) == 9

    def test_exhaustive_short_expands_to_24_cells(self):
        matrix = expand_run_matrix("exhaustive", "short")
        n_seeds    = len(ROBUSTNESS_PROFILES["exhaustive"]["seeds"])
        n_horizons = len(HORIZON_PROFILES["short"])
        assert len(matrix) == n_horizons * n_seeds  # 3 × 8 = 24

    def test_each_cell_has_required_keys(self):
        matrix = expand_run_matrix("quick", "short")
        for cell in matrix:
            assert "horizon_days" in cell
            assert "seed" in cell
            assert "n_windows" in cell
            assert "weight_samples" in cell

    def test_n_windows_matches_profile(self):
        matrix = expand_run_matrix("standard", "short")
        expected_nw = ROBUSTNESS_PROFILES["standard"]["windows_per_horizon"]
        for cell in matrix:
            assert cell["n_windows"] == expected_nw

    def test_weight_samples_matches_profile(self):
        matrix = expand_run_matrix("deep", "medium")
        expected_ws = ROBUSTNESS_PROFILES["deep"]["weight_samples"]
        for cell in matrix:
            assert cell["weight_samples"] == expected_ws

    def test_custom_horizons_override_profile(self):
        custom = [45, 90]
        matrix = expand_run_matrix("quick", "mixed", custom_horizons=custom)
        horizons_in_matrix = [c["horizon_days"] for c in matrix]
        assert set(horizons_in_matrix) == set(custom)
        assert len(horizons_in_matrix) == 2  # 2 horizons × 1 seed

    def test_custom_seeds_override_profile(self):
        custom_seeds = [7, 13, 99]
        matrix = expand_run_matrix("quick", "short", custom_seeds=custom_seeds)
        seeds_in_matrix = [c["seed"] for c in matrix]
        assert set(seeds_in_matrix) == set(custom_seeds)

    def test_windows_override_replaces_profile_value(self):
        matrix = expand_run_matrix("standard", "short", windows_override=7)
        for cell in matrix:
            assert cell["n_windows"] == 7

    def test_unknown_robustness_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown robustness profile"):
            expand_run_matrix("ultrafast", "short")

    def test_unknown_horizon_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown horizon profile"):
            expand_run_matrix("quick", "very_long")


# ---------------------------------------------------------------------------
# Total simulations count
# ---------------------------------------------------------------------------

class TestTotalSimulations:

    def test_total_simulations_is_cells_times_windows(self):
        matrix = expand_run_matrix("standard", "medium")
        expected = sum(c["n_windows"] for c in matrix)
        assert total_simulations(matrix) == expected

    def test_quick_short_total(self):
        matrix = expand_run_matrix("quick", "short")
        nw = ROBUSTNESS_PROFILES["quick"]["windows_per_horizon"]
        n_cells = len(HORIZON_PROFILES["short"])  # 3 horizons × 1 seed
        assert total_simulations(matrix) == nw * n_cells

    def test_override_windows_changes_total(self):
        matrix1 = expand_run_matrix("standard", "short")
        matrix2 = expand_run_matrix("standard", "short", windows_override=3)
        assert total_simulations(matrix2) < total_simulations(matrix1)


# ---------------------------------------------------------------------------
# Effort caption smoke test
# ---------------------------------------------------------------------------

class TestEffortCaption:

    def test_caption_is_non_empty_string(self):
        cap = effort_caption("standard", "mixed")
        assert isinstance(cap, str)
        assert len(cap) > 10

    def test_caption_contains_simulation_count(self):
        cap = effort_caption("quick", "short")
        matrix = expand_run_matrix("quick", "short")
        expected_sims = total_simulations(matrix)
        assert str(expected_sims) in cap


# ---------------------------------------------------------------------------
# RobustScanResult aggregation
# ---------------------------------------------------------------------------

def _make_mock_summary(pct_beating: float, excess: float = 0.02,
                        robust_score: float = 0.05, scope: str = "overall_strategy"):
    """Build a minimal mock RandomWindowSummary-like object."""
    import types
    return types.SimpleNamespace(
        n_windows=5,
        window_days=60,
        window_results=[],
        median_excess_return=excess,
        median_sharpe=0.5,
        median_drawdown=-0.08,
        median_turnover=1.0,
        std_excess_return=0.02,
        pct_beating_benchmark=pct_beating,
        worst_decile_drawdown=-0.12,
        robust_score=robust_score,
        median_benchmark_return=0.01,
        median_strategy_return=excess + 0.01,
        median_calmar=0.3,
        median_active_excess_return=None,
        median_active_sharpe=None,
        pct_active_beating_benchmark=None,
        worst_decile_active_drawdown=None,
        active_robust_score=None,
    )


class TestRobustScanResult:

    def _make_result(self, horizon_excesses: dict[int, float]) -> RobustScanResult:
        """Build a RobustScanResult with one cell per horizon (single seed=42)."""
        from tuning.robust_scan import ScanCell, _aggregate
        cells = [
            ScanCell(horizon_days=h, seed=42, summary=_make_mock_summary(
                pct_beating=1.0 if excess > 0 else 0.0, excess=excess))
            for h, excess in horizon_excesses.items()
        ]
        matrix = [{"horizon_days": h, "seed": 42, "n_windows": 5, "weight_samples": 20}
                  for h in horizon_excesses]
        return _aggregate(matrix, cells, "overall_strategy")

    def test_overfit_score_high_when_only_one_horizon_beats(self):
        result = self._make_result({30: 0.05, 60: -0.02, 90: -0.01, 120: -0.03, 180: -0.02})
        assert result.overfit_warning_score() >= 0.7

    def test_overfit_score_low_when_all_horizons_beat(self):
        result = self._make_result({30: 0.05, 60: 0.03, 90: 0.02, 120: 0.04, 180: 0.01})
        assert result.overfit_warning_score() <= 0.2

    def test_overfit_score_zero_when_all_beat(self):
        result = self._make_result({60: 0.03, 90: 0.02, 180: 0.01})
        assert result.overfit_warning_score() == pytest.approx(0.0)

    def test_overfit_score_one_when_none_beat(self):
        result = self._make_result({60: -0.03, 90: -0.02, 180: -0.01})
        assert result.overfit_warning_score() == pytest.approx(1.0)

    def test_horizon_heatmap_df_has_correct_shape(self):
        from tuning.robust_scan import ScanCell, _aggregate
        horizons = [60, 90, 180]
        cells = [ScanCell(horizon_days=h, seed=42, summary=_make_mock_summary(0.6))
                 for h in horizons]
        matrix = [{"horizon_days": h, "seed": 42, "n_windows": 5, "weight_samples": 20}
                  for h in horizons]
        result = _aggregate(matrix, cells, "overall_strategy")
        df = result.horizon_heatmap_df()
        assert len(df) == 3
        expected_cols = {"horizon (days)", "median excess", "median Sharpe", "% beating",
                         "median DD", "robust score"}
        assert expected_cols.issubset(set(df.columns))

    def test_seed_stability_df_has_correct_shape(self):
        from tuning.robust_scan import ScanCell, _aggregate
        seeds    = [7, 42, 99]
        horizons = [60, 90]
        cells = [ScanCell(horizon_days=h, seed=s, summary=_make_mock_summary(0.5))
                 for h in horizons for s in seeds]
        matrix = [{"horizon_days": h, "seed": s, "n_windows": 5, "weight_samples": 20}
                  for h in horizons for s in seeds]
        result = _aggregate(matrix, cells, "overall_strategy")
        df = result.seed_stability_df()
        assert len(df) == 3  # 3 seeds
        assert "seed" in df.columns
        assert "overall" in df.columns
