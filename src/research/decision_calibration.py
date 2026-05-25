"""
research/decision_calibration.py — Decision Calibration Engine.

Reads decision_outcomes.parquet (recorded by outcome_tracker.py) and
computes accuracy metrics for each decision state.

ARCHITECTURE CONTRACT
─────────────────────
This module:
  • Reads realized outcomes — NEVER reads factor scores or alpha signals
  • May adjust decision THRESHOLDS only (exit_threshold, watch_threshold, etc.)
  • NEVER adjusts factor weights, momentum formula, or composite computation
  • Writes suggestions to data/calibration_state.json for the DAE to read

Metrics computed
────────────────
exit_accuracy            : % of EXITs where future_30d_return < 0 (exit was right)
premature_exit_rate      : % of EXITs where future_30d_return > 0.02 (missed gains)
false_exit_rate          : % of EXITs that outperformed SPY after exit
watch_recovery_rate      : % of WATCHes where price rose 30d later
review_precision         : % of REVIEWs where holding longer was better
hold_outperformance_rate : % of HOLDs that outperformed SPY in 30d
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MIN_SAMPLE = 10   # minimum decisions before calibration is meaningful


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    n_total: int
    n_with_outcomes: int

    # Per-state accuracy
    exit_accuracy: float             # 0-1: % EXITs that subsequently went down
    premature_exit_rate: float       # 0-1: % EXITs that went up ≥ 2% afterward
    false_exit_rate: float           # 0-1: % EXITs that outperformed SPY
    watch_recovery_rate: float       # 0-1: % WATCHes that recovered within 30d
    review_precision: float          # 0-1: % REVIEWs eventually justified
    hold_outperformance_rate: float  # 0-1: % HOLDs that beat SPY

    # Threshold suggestions (these feed the DAE, not the factor engine)
    suggested_premature_exit_threshold: float   # default 0.45
    suggested_review_confidence_threshold: float  # default 0.50

    # Raw DataFrames for UI display
    by_state: Optional[pd.DataFrame] = None     # accuracy broken down by state
    confusion_matrix: Optional[pd.DataFrame] = None
    calibration_curve: Optional[pd.DataFrame] = None  # confidence vs actual accuracy

    @property
    def is_reliable(self) -> bool:
        return self.n_with_outcomes >= _MIN_SAMPLE

    def to_threshold_adjustments(self) -> dict:
        """
        Return threshold overrides for the Decision Adjustment Engine.
        Only decision thresholds — never factor weights.
        """
        return {
            "premature_exit_threshold":      self.suggested_premature_exit_threshold,
            "review_confidence_threshold":   self.suggested_review_confidence_threshold,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    try:
        from ui.utils import DATA_DIR
        return DATA_DIR
    except Exception:
        return Path(__file__).parent.parent.parent / "data"


def load_calibration_state() -> dict:
    path = _data_dir() / "calibration_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_calibration_state(result: CalibrationResult) -> None:
    """Persist threshold adjustments for the DAE to read on next run."""
    path = _data_dir() / "calibration_state.json"
    state = result.to_threshold_adjustments()
    state["last_updated"] = pd.Timestamp.now().isoformat()
    state["n_decisions"]  = result.n_total
    state["premature_exit_rate"] = result.premature_exit_rate
    state["exit_accuracy"]       = result.exit_accuracy
    try:
        path.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        logger.warning("Could not write calibration_state.json: %s", exc)


# ---------------------------------------------------------------------------
# Main calibration computation
# ---------------------------------------------------------------------------

def compute_calibration(min_outcomes: int = _MIN_SAMPLE) -> CalibrationResult:
    """
    Load decision_outcomes.parquet and compute accuracy metrics.
    Returns a CalibrationResult regardless of data availability.
    """
    from portfolio.outcome_tracker import load_outcomes

    df = load_outcomes()
    n_total = len(df)

    # Only rows with realized 30d returns can be evaluated
    has_30d = df[df["future_30d_return"].notna()].copy() if not df.empty else pd.DataFrame()
    n_outcomes = len(has_30d)

    if n_outcomes < min_outcomes:
        return CalibrationResult(
            n_total=n_total,
            n_with_outcomes=n_outcomes,
            exit_accuracy=0.0,
            premature_exit_rate=0.0,
            false_exit_rate=0.0,
            watch_recovery_rate=0.0,
            review_precision=0.0,
            hold_outperformance_rate=0.0,
            suggested_premature_exit_threshold=0.45,
            suggested_review_confidence_threshold=0.50,
            by_state=None,
            confusion_matrix=None,
            calibration_curve=None,
        )

    exits   = has_30d[has_30d["decision_state"] == "EXIT"]
    watches = has_30d[has_30d["decision_state"] == "WATCH"]
    reviews = has_30d[has_30d["decision_state"] == "REVIEW"]
    holds   = has_30d[has_30d["decision_state"] == "HOLD"]

    def _safe_mean(series, condition_fn) -> float:
        if series.empty:
            return 0.0
        try:
            mask = condition_fn(pd.to_numeric(series, errors="coerce").fillna(0.0))
            return round(float(mask.mean()), 4)
        except Exception:
            return 0.0

    # ── EXIT accuracy ─────────────────────────────────────────────────────────
    exit_30d = pd.to_numeric(exits["future_30d_return"], errors="coerce").fillna(0.0)
    exit_accuracy       = _safe_mean(exit_30d, lambda x: x < 0.0)
    premature_exit_rate = _safe_mean(exit_30d, lambda x: x > 0.02)
    false_exit_rate_raw = exits["outperformed_spy_30d"].dropna()
    false_exit_rate     = round(float(false_exit_rate_raw.mean()), 4) if not false_exit_rate_raw.empty else 0.0

    # ── WATCH recovery ────────────────────────────────────────────────────────
    watch_30d = pd.to_numeric(watches["future_30d_return"], errors="coerce").fillna(0.0)
    watch_recovery_rate = _safe_mean(watch_30d, lambda x: x > 0.01)

    # ── REVIEW precision ──────────────────────────────────────────────────────
    # A REVIEW is "precise" if the stock subsequently declined (exit would have been right)
    # OR if it recovered significantly (hold was right and the REVIEW flagged correctly)
    review_30d = pd.to_numeric(reviews["future_30d_return"], errors="coerce").fillna(0.0)
    review_precision = round(
        float((abs(review_30d) > 0.05).mean()), 4
    ) if not reviews.empty else 0.0

    # ── HOLD outperformance ───────────────────────────────────────────────────
    hold_spy = holds["outperformed_spy_30d"].dropna()
    hold_outperformance_rate = round(float(hold_spy.mean()), 4) if not hold_spy.empty else 0.0

    # ── By-state summary table ────────────────────────────────────────────────
    by_state_rows = []
    for state, grp in has_30d.groupby("decision_state"):
        r30 = pd.to_numeric(grp["future_30d_return"], errors="coerce")
        by_state_rows.append({
            "state":      state,
            "n":          len(grp),
            "mean_30d_return":   round(float(r30.mean()), 4) if not r30.empty else None,
            "pct_positive":      round(float((r30 > 0).mean()), 4) if not r30.empty else None,
            "pct_negative":      round(float((r30 < 0).mean()), 4) if not r30.empty else None,
        })
    by_state = pd.DataFrame(by_state_rows) if by_state_rows else None

    # ── Calibration curve — confidence bin vs actual accuracy ────────────────
    calibration_curve = _compute_calibration_curve(has_30d)

    # ── Threshold suggestions ─────────────────────────────────────────────────
    suggested_pet = _suggest_premature_exit_threshold(premature_exit_rate, false_exit_rate)
    suggested_rct = _suggest_review_confidence_threshold(review_precision)

    return CalibrationResult(
        n_total=n_total,
        n_with_outcomes=n_outcomes,
        exit_accuracy=exit_accuracy,
        premature_exit_rate=premature_exit_rate,
        false_exit_rate=false_exit_rate,
        watch_recovery_rate=watch_recovery_rate,
        review_precision=review_precision,
        hold_outperformance_rate=hold_outperformance_rate,
        suggested_premature_exit_threshold=suggested_pet,
        suggested_review_confidence_threshold=suggested_rct,
        by_state=by_state,
        confusion_matrix=_compute_confusion_matrix(has_30d),
        calibration_curve=calibration_curve,
    )


# ---------------------------------------------------------------------------
# Threshold suggestion helpers
# ---------------------------------------------------------------------------

def _suggest_premature_exit_threshold(
    premature_exit_rate: float,
    false_exit_rate: float,
) -> float:
    """
    Adjust the premature_exit_threshold based on realized outcomes.

    High premature_exit_rate → lower threshold (be more cautious about exiting).
    Low premature_exit_rate  → raise threshold (exits are generally correct).
    Clipped to [0.30, 0.65] to prevent runaway.
    """
    base = 0.45
    # If > 40% of exits were premature, lower the threshold (require stronger exit evidence)
    if premature_exit_rate > 0.40:
        base = max(0.30, base - (premature_exit_rate - 0.40) * 0.50)
    # If < 15% of exits were premature, exits are accurate — can raise threshold
    elif premature_exit_rate < 0.15 and false_exit_rate < 0.20:
        base = min(0.60, base + 0.05)
    return round(base, 3)


def _suggest_review_confidence_threshold(review_precision: float) -> float:
    """
    Adjust the review_confidence_threshold based on REVIEW decision quality.

    High precision → keep threshold; low precision → tighten.
    """
    base = 0.50
    if review_precision < 0.30:
        base = min(0.65, base + 0.08)   # tighten — too many unhelpful reviews
    elif review_precision > 0.65:
        base = max(0.38, base - 0.05)   # loosen — reviews are useful
    return round(base, 3)


# ---------------------------------------------------------------------------
# Confusion matrix and calibration curve
# ---------------------------------------------------------------------------

def _compute_confusion_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Predicted state vs actual outcome (positive = stock rose, negative = fell).
    Simple 2-class: "up" (>0) vs "down" (≤0) in 30d.
    """
    if df.empty:
        return None
    try:
        d = df[["decision_state", "future_30d_return"]].dropna()
        d = d.copy()
        d["actual"] = pd.to_numeric(d["future_30d_return"], errors="coerce").apply(
            lambda x: "up" if x > 0 else "down"
        )
        return d.groupby(["decision_state", "actual"]).size().unstack(fill_value=0)
    except Exception:
        return None


def _compute_calibration_curve(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Bin decisions by confidence level (HIGH/MEDIUM/LOW) and compute
    mean realized 30d return per bin.
    """
    if df.empty or "decision_confidence" not in df.columns:
        return None
    try:
        rows = []
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            grp = df[df["decision_confidence"] == conf]
            r30 = pd.to_numeric(grp["future_30d_return"], errors="coerce").dropna()
            if not r30.empty:
                rows.append({
                    "confidence":      conf,
                    "n":               len(r30),
                    "mean_30d_return": round(float(r30.mean()), 4),
                    "pct_correct":     round(float((r30 < 0).mean()), 4),  # for EXIT decisions
                })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None
