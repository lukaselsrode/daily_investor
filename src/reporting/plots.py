"""
reporting/plots.py — PlotManager.

Wraps _reporting_legacy heatmap generators.
All methods return the path (str) to the saved PNG file.
Requires matplotlib — raises RuntimeError if not installed.
"""

from __future__ import annotations

from typing import Optional

import _reporting_legacy as _rl


class PlotManager:
    """Generates and saves heatmap PNGs for stability analysis reports."""

    def param_heatmap(
        self,
        window_results: list[dict],
        param_names: list[str],
        output_dir: str,
        date: Optional[str] = None,
    ) -> str:
        """
        Heatmap of averaged param values across optimization windows.
        Column-normalized so each param's range fills [0, 1].
        Returns PNG path.
        """
        return _rl.generate_param_heatmap(window_results, param_names, output_dir, date)

    def objective_heatmap(
        self,
        window_results: list[dict],
        param_names: list[str],
        output_dir: str,
        date: Optional[str] = None,
    ) -> str:
        """
        Heatmap comparing Sharpe-opt, Calmar-opt, and averaged params.
        Highlights objective disagreement per parameter.
        Returns PNG path.
        """
        return _rl.generate_objective_heatmap(window_results, param_names, output_dir, date)

    def validation_heatmap(
        self,
        window_results: list[dict],
        output_dir: str,
        date: Optional[str] = None,
    ) -> str:
        """
        Heatmap of out-of-sample validation metrics across windows.
        Returns PNG path.
        """
        return _rl.generate_validation_heatmap(window_results, output_dir, date)
