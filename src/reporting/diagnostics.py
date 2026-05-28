"""
reporting/diagnostics.py — DiagnosticsReporter and stability report writers.

RESEARCH / DIAGNOSTIC ONLY — never modifies config.yaml.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

from .attribution import (
    _MODERATELY_STABLE,
    _UNSTABLE,
    _date_str,
    _ensure_dir,
)
from .plots import (
    generate_objective_heatmap,
    generate_param_heatmap,
    generate_validation_heatmap,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def write_stability_summary_csv(
    stability_df: pd.DataFrame,
    window_results: list[dict],
    output_dir: str,
    date: str | None = None,
) -> str:
    """Write per-parameter and per-window CSV files. Returns path to param CSV."""
    date = date or _date_str()

    param_out = os.path.join(_ensure_dir(output_dir), f"stability_summary_{date}.csv")
    stability_df.to_csv(param_out, index=False)
    logger.info("Saved stability summary CSV: %s", param_out)

    val_rows = []
    for r in window_results:
        val_rows.append({
            "window_days":       r.get("window"),
            "val_excess_return": r.get("val_excess_return"),
            "val_sharpe":        r.get("val_sharpe"),
            "val_drawdown":      r.get("val_drawdown"),
            "turnover":          r.get("turnover"),
            "trades":            r.get("trades"),
            "unstable_params":   r.get("unstable_params"),
            "validation_passed": r.get("validation_passed"),
        })
    val_out = os.path.join(output_dir, f"validation_summary_{date}.csv")
    pd.DataFrame(val_rows).to_csv(val_out, index=False)
    logger.info("Saved validation summary CSV: %s", val_out)

    return param_out


def write_robustness_report_txt(
    stability_df: pd.DataFrame,
    window_results: list[dict],
    param_names: list[str],
    output_dir: str,
    date: str | None = None,
) -> str:
    """Write a human-readable robustness report. Returns path to .txt file."""
    date = date or _date_str()
    lines: list[str] = []

    def h(text: str, char: str = "=") -> None:
        lines.append(char * 64)
        lines.append(text)
        lines.append(char * 64)

    h(f"ROBUSTNESS REPORT — {date}")
    lines.append(
        "RESEARCH / DIAGNOSTIC ONLY — not a trading signal.\n"
        "Purpose: understand which parameters are stable vs overfit.\n"
    )

    h("WINDOWS ANALYZED", "-")
    for r in window_results:
        vp = "PASS" if r.get("validation_passed") else "FAIL"
        lines.append(
            f"  {r['window']:>4}d  val_excess={r.get('val_excess_return', 0):+.2%}  "
            f"val_sharpe={r.get('val_sharpe', 0):+.3f}  "
            f"drawdown={r.get('val_drawdown', 0):.2%}  "
            f"trades={r.get('trades', 0)}  validation={vp}"
        )
    lines.append("")

    h("PARAMETER STABILITY SUMMARY", "-")
    if not stability_df.empty:
        for _, row in stability_df.sort_values("instability_score", ascending=False).iterrows():
            flag = " ⚠" if row["stability"] == _UNSTABLE else \
                   " ~" if row["stability"] == _MODERATELY_STABLE else "  "
            lines.append(
                f"{flag} {row['param']:<36}  "
                f"mean={row['mean']:+.4f}  stddev={row['stddev']:.4f}  "
                f"cv={row['cv']:.3f}  spread={row['sharpe_calmar_spread']:.4f}  "
                f"conv={row['convergence_frequency']:.0%}  "
                f"[{row['stability']}]"
            )
    lines.append("")

    unstable = stability_df[stability_df["stability"] == _UNSTABLE]["param"].tolist() \
        if not stability_df.empty else []
    mod_stable = stability_df[stability_df["stability"] == _MODERATELY_STABLE]["param"].tolist() \
        if not stability_df.empty else []

    h("FINDINGS", "-")
    if not unstable:
        lines.append("  ✓ No UNSTABLE parameters detected.")
    else:
        lines.append(f"  ⚠ {len(unstable)} UNSTABLE parameter(s):")
        for p in unstable:
            lines.append(f"      • {p}")
        lines.append(
            "\n  Unstable parameters have high cross-window variance or large\n"
            "  disagreement between Sharpe and Calmar objectives. Their\n"
            "  averaged values may not generalize. Consider:\n"
            "    - Fixing them to config defaults and not tuning\n"
            "    - Using a longer window to reduce overfitting\n"
            "    - Investigating whether the signal is genuinely informative"
        )
    lines.append("")

    if mod_stable:
        lines.append(f"  ~ {len(mod_stable)} MODERATELY_STABLE parameter(s) — monitor:")
        for p in mod_stable:
            lines.append(f"      • {p}")
    lines.append("")

    consistent_wins = sum(1 for r in window_results if r.get("validation_passed"))
    lines.append(f"  Validation passed in {consistent_wins}/{len(window_results)} windows.")
    if consistent_wins < len(window_results) // 2:
        lines.append(
            "  ⚠ Majority of windows failed validation — "
            "strategy may not generalize across horizons."
        )
    elif consistent_wins == len(window_results):
        lines.append(
            "  ✓ Validation passed in all windows — "
            "strategy shows consistent out-of-sample performance."
        )
    lines.append("")

    h("INTERPRETATION GUIDE", "-")
    lines.append(
        "  cv (coefficient of variation) = stddev / |mean| across windows.\n"
        "  High cv → parameter value changes substantially with window length.\n\n"
        "  sharpe_calmar_spread = mean |sharpe_opt - calmar_opt| per window.\n"
        "  High spread → objectives fundamentally disagree on this parameter.\n\n"
        "  convergence_frequency = fraction of windows within 1 stddev of mean.\n"
        "  Low frequency → optimizer rarely lands in the same region.\n\n"
        "  instability_score = composite (0=stable, 1=unstable).\n"
        "  Use this to rank parameters by robustness concern.\n"
    )

    h("END OF REPORT")

    out = os.path.join(_ensure_dir(output_dir), f"robustness_report_{date}.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved robustness report: %s", out)
    return out


def generate_all_reports(
    window_results: list[dict],
    stability_df: pd.DataFrame,
    param_names: list[str],
    output_dir: str,
) -> dict[str, str]:
    """
    Generate all reports for a stability scan. Returns dict of output paths.
    Heatmaps are skipped gracefully if matplotlib is not installed.
    """
    date = _date_str()
    outputs: dict[str, str] = {}

    outputs["stability_csv"] = write_stability_summary_csv(
        stability_df, window_results, output_dir, date
    )
    outputs["robustness_txt"] = write_robustness_report_txt(
        stability_df, window_results, param_names, output_dir, date
    )

    try:
        outputs["param_heatmap"] = generate_param_heatmap(
            window_results, param_names, output_dir, date
        )
        outputs["objective_heatmap"] = generate_objective_heatmap(
            window_results, param_names, output_dir, date
        )
        outputs["validation_heatmap"] = generate_validation_heatmap(
            window_results, output_dir, date
        )
    except RuntimeError as e:
        logger.warning("Heatmaps skipped: %s", e)

    return outputs


class DiagnosticsReporter:
    """Generates stability diagnostic output: CSV, robustness text report, and all-in-one bundle."""

    def write_stability_csv(
        self,
        stability_df: pd.DataFrame,
        window_results: list[dict],
        output_dir: str,
        date: str | None = None,
    ) -> str:
        return write_stability_summary_csv(stability_df, window_results, output_dir, date)

    def write_robustness_txt(
        self,
        stability_df: pd.DataFrame,
        window_results: list[dict],
        param_names: list[str],
        output_dir: str,
        date: str | None = None,
    ) -> str:
        return write_robustness_report_txt(
            stability_df, window_results, param_names, output_dir, date
        )

    def generate_all(
        self,
        window_results: list[dict],
        stability_df: pd.DataFrame,
        param_names: list[str],
        output_dir: str,
    ) -> dict[str, str]:
        return generate_all_reports(window_results, stability_df, param_names, output_dir)
