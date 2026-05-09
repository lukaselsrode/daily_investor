"""
reporting/diagnostics.py — DiagnosticsReporter.

Wraps _reporting_legacy functions for stability diagnostics output:
CSV summaries, text robustness reports, and the all-in-one generate_all.

RESEARCH / DIAGNOSTIC ONLY — never modifies config.yaml.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import _reporting_legacy as _rl

if TYPE_CHECKING:
    import pandas as pd


class DiagnosticsReporter:
    """
    Generates stability diagnostic output: CSV, robustness text report,
    and all-in-one report bundle (CSV + TXT + heatmaps).

    All methods return output file paths so callers can log or display them.
    """

    def write_stability_csv(
        self,
        stability_df: "pd.DataFrame",
        window_results: list[dict],
        output_dir: str,
        date: Optional[str] = None,
    ) -> str:
        """Write per-parameter and per-window CSV files. Returns path to param CSV."""
        return _rl.write_stability_summary_csv(
            stability_df, window_results, output_dir, date
        )

    def write_robustness_txt(
        self,
        stability_df: "pd.DataFrame",
        window_results: list[dict],
        param_names: list[str],
        output_dir: str,
        date: Optional[str] = None,
    ) -> str:
        """Write human-readable robustness report. Returns path to .txt file."""
        return _rl.write_robustness_report_txt(
            stability_df, window_results, param_names, output_dir, date
        )

    def generate_all(
        self,
        window_results: list[dict],
        stability_df: "pd.DataFrame",
        param_names: list[str],
        output_dir: str,
    ) -> dict[str, str]:
        """
        Generate all reports (CSV, txt, heatmaps if matplotlib available).

        Returns dict of output paths keyed by report type:
            stability_csv, robustness_txt, param_heatmap,
            objective_heatmap, validation_heatmap.
        Heatmaps are silently skipped if matplotlib is not installed.
        """
        return _rl.generate_all_reports(
            window_results, stability_df, param_names, output_dir
        )
