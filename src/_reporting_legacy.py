"""
reporting.py — Parameter stability analysis, heatmaps, and robustness reports.

RESEARCH / DIAGNOSTIC ONLY.
This module NEVER modifies config.yaml, NEVER affects live execution,
and NEVER introduces lookahead bias.

Outputs written to reports/stability/ (configurable via stability.output_dir).
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_STABLE            = "STABLE"
_MODERATELY_STABLE = "MODERATELY_STABLE"
_UNSTABLE          = "UNSTABLE"

_STABILITY_PALETTE = {
    _STABLE:            "green",
    _MODERATELY_STABLE: "orange",
    _UNSTABLE:          "red",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _date_str() -> str:
    return datetime.date.today().isoformat()


def _try_matplotlib():
    """Import matplotlib with headless backend. Returns (plt, sns_or_None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            sns = None
        return plt, sns
    except ImportError:
        raise RuntimeError(
            "matplotlib is required for heatmaps. Install: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------

def classify_stability(
    cv: float,
    spread: float,
    cv_threshold: float = 0.30,
    spread_threshold: float = 0.15,
) -> str:
    """
    Classify a parameter's stability based on cross-window coefficient of
    variation and sharpe/calmar objective spread.
    """
    if cv > cv_threshold or spread > spread_threshold:
        return _UNSTABLE
    if cv > cv_threshold * 0.60 or spread > spread_threshold * 0.60:
        return _MODERATELY_STABLE
    return _STABLE


# ---------------------------------------------------------------------------
# Core stability analysis
# ---------------------------------------------------------------------------

def compute_parameter_stability(
    window_results: list[dict],
    param_names: list[str],
    cv_threshold: float = 0.30,
    spread_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Compute per-parameter stability metrics across all window runs.

    window_results entries must have:
        window (int), params_avg (np.ndarray),
        params_sharpe (np.ndarray), params_calmar (np.ndarray).

    Returns a DataFrame with columns:
        param, mean, stddev, cv, sharpe_calmar_spread,
        convergence_frequency, instability_score, stability.
    """
    rows = []
    n_windows = len(window_results)

    for i, name in enumerate(param_names):
        avg_vals = np.array([
            r["params_avg"][i] for r in window_results
            if r.get("params_avg") is not None
        ], dtype=float)

        if len(avg_vals) == 0:
            continue

        mean   = float(avg_vals.mean())
        std    = float(avg_vals.std())
        cv     = float(std / abs(mean)) if abs(mean) > 1e-9 else 0.0

        # Sharpe vs Calmar spread per window, then averaged
        sc_spreads = []
        for r in window_results:
            ps = r.get("params_sharpe")
            pc = r.get("params_calmar")
            if ps is not None and pc is not None:
                sc_spreads.append(abs(float(ps[i]) - float(pc[i])))
        sc_spread = float(np.mean(sc_spreads)) if sc_spreads else 0.0

        # Convergence frequency: fraction of windows where value is within
        # one stddev of the cross-window mean (higher = more consistent)
        if std > 1e-9 and n_windows > 1:
            within_1std = float(np.mean(np.abs(avg_vals - mean) <= std))
        else:
            within_1std = 1.0

        # Instability score (0 = perfectly stable, 1 = maximally unstable)
        norm_cv     = min(cv     / max(cv_threshold,     1e-9), 1.0)
        norm_spread = min(sc_spread / max(spread_threshold, 1e-9), 1.0)
        instability = 0.50 * norm_cv + 0.50 * norm_spread

        stability = classify_stability(cv, sc_spread, cv_threshold, spread_threshold)

        rows.append({
            "param":                 name,
            "mean":                  round(mean,          4),
            "stddev":                round(std,           4),
            "cv":                    round(cv,            4),
            "sharpe_calmar_spread":  round(sc_spread,     4),
            "convergence_frequency": round(within_1std,   3),
            "instability_score":     round(instability,   3),
            "stability":             stability,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heatmap 1 — Window stability: rows=windows, cols=params
# ---------------------------------------------------------------------------

def generate_param_heatmap(
    window_results: list[dict],
    param_names: list[str],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Heatmap of averaged param values across optimization windows.
    Color scale is column-normalized so each param's range fills [0, 1].
    This makes convergence/divergence visible regardless of param magnitude.
    """
    plt, _ = _try_matplotlib()
    date = date or _date_str()

    windows = [r["window"] for r in window_results]
    matrix  = np.array([
        [r["params_avg"][i] for i in range(len(param_names))]
        for r in window_results
    ], dtype=float)

    # Column-normalize: each param independently scaled to [0, 1]
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_rng = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
    norm    = (matrix - col_min) / col_rng

    n_rows, n_cols = norm.shape
    fig_w = max(12, n_cols * 0.85)
    fig_h = max(3, n_rows * 0.70 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{w}d" for w in windows], fontsize=9)
    ax.set_title(
        f"Parameter Values by Window — {date}\n"
        "(column-normalized; green=high, red=low within each param's range)",
        fontsize=10,
    )

    # Raw value annotations
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if 0.2 < norm[i, j] < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Relative value (column-scaled)")
    plt.tight_layout()

    out = os.path.join(_ensure_dir(output_dir), f"param_heatmap_{date}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved param heatmap: %s", out)
    return out


# ---------------------------------------------------------------------------
# Heatmap 2 — Objective stability: rows=objective, cols=params
# ---------------------------------------------------------------------------

def generate_objective_heatmap(
    window_results: list[dict],
    param_names: list[str],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Heatmap comparing Sharpe-optimized, Calmar-optimized, and averaged params.
    Averaged across all windows for each objective.
    Highlights where the two objectives fundamentally disagree.
    """
    plt, _ = _try_matplotlib()
    date = date or _date_str()

    def _mean_params(key: str) -> np.ndarray:
        vals = [r[key] for r in window_results if r.get(key) is not None]
        return np.mean(vals, axis=0) if vals else np.zeros(len(param_names))

    sharpe_row = _mean_params("params_sharpe")
    calmar_row = _mean_params("params_calmar")
    avg_row    = _mean_params("params_avg")

    matrix = np.array([sharpe_row, calmar_row, avg_row], dtype=float)
    row_labels = ["Sharpe-opt", "Calmar-opt", "Averaged"]

    # Column-normalize
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_rng = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
    norm    = (matrix - col_min) / col_rng

    n_rows, n_cols = norm.shape
    fig_w = max(12, n_cols * 0.85)
    fig_h = max(3, n_rows * 0.90 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(
        f"Objective Disagreement Heatmap — {date}\n"
        "(column-normalized; blue=low, red=high within each param; divergence = objective conflict)",
        fontsize=10,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=6.5)

    plt.colorbar(im, ax=ax, label="Relative value (column-scaled)")
    plt.tight_layout()

    out = os.path.join(_ensure_dir(output_dir), f"objective_heatmap_{date}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved objective heatmap: %s", out)
    return out


# ---------------------------------------------------------------------------
# Heatmap 3 — Validation metrics: rows=windows, cols=metrics
# ---------------------------------------------------------------------------

def generate_validation_heatmap(
    window_results: list[dict],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Heatmap of out-of-sample validation metrics across windows.
    Each cell is column-normalized. Red = poor, green = good.
    """
    plt, _ = _try_matplotlib()
    date = date or _date_str()

    metric_keys = [
        "val_excess_return",
        "val_sharpe",
        "val_drawdown",
        "turnover",
        "trades",
        "unstable_params",
    ]
    # Higher is better for most metrics; drawdown and turnover are exceptions
    higher_is_better = {
        "val_excess_return": True,
        "val_sharpe":        True,
        "val_drawdown":      False,   # more negative = worse
        "turnover":          False,   # higher = more churn
        "trades":            True,    # more = more diversified (within reason)
        "unstable_params":   False,   # fewer = more robust
    }

    windows = [r["window"] for r in window_results]
    matrix  = np.array([
        [float(r.get(k, 0) or 0) for k in metric_keys]
        for r in window_results
    ], dtype=float)

    # Normalize each column to [0, 1] respecting sign convention
    norm = np.zeros_like(matrix)
    for j, key in enumerate(metric_keys):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        rng = hi - lo if (hi - lo) > 1e-9 else 1.0
        scaled = (col - lo) / rng   # 0 = min in window, 1 = max
        norm[:, j] = scaled if higher_is_better[key] else (1.0 - scaled)

    n_rows, n_cols = norm.shape
    fig, ax = plt.subplots(figsize=(max(9, n_cols * 1.4), max(3, n_rows * 0.70 + 1.5)))

    im = ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        ["Val excess ret", "Val Sharpe", "Val drawdown",
         "Turnover", "Trades", "Unstable params"],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{w}d" for w in windows], fontsize=9)
    ax.set_title(
        f"Validation Diagnostics by Window — {date}\n"
        "(green = better outcome in each column's context)",
        fontsize=10,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            fmt = f"{val:.1%}" if j in (0, 1, 2) else f"{val:.1f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Relative quality (column-scaled)")
    plt.tight_layout()

    out = os.path.join(_ensure_dir(output_dir), f"validation_heatmap_{date}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved validation heatmap: %s", out)
    return out


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def write_stability_summary_csv(
    stability_df: pd.DataFrame,
    window_results: list[dict],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Write per-parameter stability metrics and per-window validation metrics to CSV.
    """
    date = date or _date_str()

    # Per-parameter table
    param_out = os.path.join(_ensure_dir(output_dir), f"stability_summary_{date}.csv")
    stability_df.to_csv(param_out, index=False)
    logger.info("Saved stability summary CSV: %s", param_out)

    # Per-window validation table
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


# ---------------------------------------------------------------------------
# Text robustness report
# ---------------------------------------------------------------------------

def write_robustness_report_txt(
    stability_df: pd.DataFrame,
    window_results: list[dict],
    param_names: list[str],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Write a human-readable robustness report summarizing stability findings.
    """
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

    # Window summary
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

    # Stability summary
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

    # Unstable params
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

    # Convergence across windows
    consistent_wins = sum(
        1 for r in window_results if r.get("validation_passed")
    )
    lines.append(
        f"  Validation passed in {consistent_wins}/{len(window_results)} windows."
    )
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


# ---------------------------------------------------------------------------
# Master entry point for stability scan output
# ---------------------------------------------------------------------------

def generate_all_reports(
    window_results: list[dict],
    stability_df: pd.DataFrame,
    param_names: list[str],
    output_dir: str,
) -> dict[str, str]:
    """
    Generate all reports for a stability scan. Returns dict of output paths.
    Safe to call even if matplotlib is not installed — heatmaps are skipped
    with a warning and CSV / TXT are always produced.
    """
    date = _date_str()
    outputs: dict[str, str] = {}

    # Always produce text + CSV
    outputs["stability_csv"] = write_stability_summary_csv(
        stability_df, window_results, output_dir, date
    )
    outputs["robustness_txt"] = write_robustness_report_txt(
        stability_df, window_results, param_names, output_dir, date
    )

    # Heatmaps — skip gracefully if matplotlib missing
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
