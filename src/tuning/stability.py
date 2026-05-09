"""
tuning/stability.py — StabilityAnalyzer.

RESEARCH / DIAGNOSTIC ONLY — never writes config.yaml.
Runs the optimizer across multiple windows and objectives to detect
unstable or overfit parameters.

Wraps tuner.run_stability_scan with a typed StabilityReport output.
"""

from __future__ import annotations

import logging
from typing import Optional

import tuner as _t

from .results import StabilityReport

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    Multi-window, multi-objective parameter stability scanner.

    Output:
      - StabilityReport with per-window results and stability_df
      - CSV summary per parameter (mean, stddev, CV across windows)
      - Heatmap PNGs (requires matplotlib)
      - Human-readable robustness report
      - Instability flags: STABLE / MODERATELY_STABLE / UNSTABLE
    """

    def __init__(self, config=None) -> None:
        self._cfg = config

    def scan(
        self,
        windows: Optional[list[int]] = None,
        mode: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> StabilityReport:
        """
        Run the optimizer across multiple time windows.

        Args:
            windows:    List of look-back window sizes in trading days.
                        Defaults to stability.windows from config.
            mode:       Universe selection mode (see backtest.select_backtest_universe).
            output_dir: Directory for CSV / PNG output.
                        Defaults to stability.output_dir from config.

        Returns:
            StabilityReport with window_results, stability_df, output_paths.
        """
        raw = _t.run_stability_scan(
            windows=windows,
            mode=mode,
            output_dir=output_dir,
        )
        return StabilityReport(
            window_results=raw.get("window_results", []),
            stability_df=raw.get("stability_df"),
            output_paths=raw.get("output_paths", {}),
        )

    def param_names(self) -> list[str]:
        """Return the full parameter name list."""
        return list(_t.PARAM_NAMES)
