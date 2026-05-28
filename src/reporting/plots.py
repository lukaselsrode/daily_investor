"""
reporting/plots.py — PlotManager and heatmap generators.

RESEARCH / DIAGNOSTIC ONLY — never modifies config.yaml.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .attribution import _date_str, _ensure_dir, _try_matplotlib

logger = logging.getLogger(__name__)


def generate_param_heatmap(
    window_results: list[dict],
    param_names: list[str],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Heatmap of averaged param values across optimization windows.
    Column-normalized so each param's range fills [0, 1].
    """
    plt, _ = _try_matplotlib()
    date = date or _date_str()

    windows = [r["window"] for r in window_results]
    matrix  = np.array([
        [r["params_avg"][i] for i in range(len(param_names))]
        for r in window_results
    ], dtype=float)

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


def generate_objective_heatmap(
    window_results: list[dict],
    param_names: list[str],
    output_dir: str,
    date: str | None = None,
) -> str:
    """
    Heatmap comparing Sharpe-optimized, Calmar-optimized, and averaged params.
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
        "(column-normalized; blue=low, red=high; divergence = objective conflict)",
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


def generate_validation_heatmap(
    window_results: list[dict],
    output_dir: str,
    date: str | None = None,
) -> str:
    """Heatmap of out-of-sample validation metrics across windows."""
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
    higher_is_better = {
        "val_excess_return": True,
        "val_sharpe":        True,
        "val_drawdown":      False,
        "turnover":          False,
        "trades":            True,
        "unstable_params":   False,
    }

    windows = [r["window"] for r in window_results]
    matrix  = np.array([
        [float(r.get(k, 0) or 0) for k in metric_keys]
        for r in window_results
    ], dtype=float)

    norm = np.zeros_like(matrix)
    for j, key in enumerate(metric_keys):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        rng = hi - lo if (hi - lo) > 1e-9 else 1.0
        scaled = (col - lo) / rng
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


class PlotManager:
    """Generates and saves heatmap PNGs for stability analysis reports."""

    def param_heatmap(
        self,
        window_results: list[dict],
        param_names: list[str],
        output_dir: str,
        date: str | None = None,
    ) -> str:
        return generate_param_heatmap(window_results, param_names, output_dir, date)

    def objective_heatmap(
        self,
        window_results: list[dict],
        param_names: list[str],
        output_dir: str,
        date: str | None = None,
    ) -> str:
        return generate_objective_heatmap(window_results, param_names, output_dir, date)

    def validation_heatmap(
        self,
        window_results: list[dict],
        output_dir: str,
        date: str | None = None,
    ) -> str:
        return generate_validation_heatmap(window_results, output_dir, date)
