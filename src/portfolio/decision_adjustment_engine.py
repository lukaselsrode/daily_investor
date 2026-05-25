"""
portfolio/decision_adjustment_engine.py — Decision Adjustment Engine (DAE).

Takes factor engine outputs + holding context + diagnostic features and
produces the final portfolio action.

ARCHITECTURE CONTRACT
─────────────────────
Input  : factor scores (read-only snapshot), holding context, diagnostic features
Output : action (BUY / HOLD / WATCH / REVIEW / TRIM / HARVEST / EXIT), confidence, badges
NEVER  : modifies factor scores, factor weights, or the composite formula

Decision hierarchy for soft exits
──────────────────────────────────
1. HARD EXIT   — stop_loss / trailing_stop / severe score collapse / thesis truly broken
2. HARVEST     — large gain (≥15%) + score weakening + momentum still positive / thesis partial
3. TRIM        — moderate gain (≥8%) + score weakening + momentum positive
4. REVIEW      — score below threshold BUT positive P/L / momentum / quality / intact thesis
5. WATCH       — mild deterioration, score below threshold, insufficient confirmation
6. EXIT        — hard score collapse (score < 0.20) + no positive signals + thesis broken

Key rule: a score-below-threshold condition alone DOES NOT produce EXIT.
  At least one hard deterioration signal is required for confirmation.

Calibration feedback loop (safe):
  • decision_calibration.py writes data/calibration_state.json
  • DAE reads threshold overrides from that file at startup
  • Only decision thresholds are adjusted — never factor weights
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    try:
        from ui.utils import DATA_DIR
        return DATA_DIR
    except Exception:
        return Path(__file__).parent.parent.parent / "data"


def _load_calibration_state() -> dict:
    path = _data_dir() / "calibration_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _dae_config() -> dict:
    try:
        from ui.utils import load_config_raw
        raw = load_config_raw()
        return raw.get("decision_adjustment", {})
    except Exception:
        return {}


def _exit_config() -> dict:
    try:
        from ui.utils import load_config_raw
        raw = load_config_raw()
        return raw.get("exit_decision", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecisionInput:
    """
    All inputs to the DAE.
    Factor scores are read-only — the DAE never writes back to them.
    """

    # ── Factor scores (alpha engine outputs — read-only) ─────────────────────
    composite_score: Optional[float] = None
    value_score: Optional[float]     = None
    quality_score: Optional[float]   = None
    momentum_score: Optional[float]  = None
    income_score: Optional[float]    = None
    percentile_rank: Optional[float] = None

    # ── Raw action from classify_state / SellDecisionEngine ──────────────────
    raw_action: str              = "HOLD"
    raw_reason: str              = ""
    exit_severity: Optional[str] = None   # "hard" / "soft" / None
    exit_type: Optional[str]     = None   # "stop_loss" / "trailing_stop" / "soft_thesis" / etc.

    # ── Holding context ───────────────────────────────────────────────────────
    pct_change: Optional[float]  = None
    holding_days: Optional[int]  = None

    # ── Diagnostic features (from ExitAnalysis — observations, not alpha) ────
    thesis_intact_score: Optional[float]       = None
    exit_confidence: Optional[str]             = None
    is_premature: bool                         = False
    confidence_collapse_weight: Optional[float] = None

    # ── Score trajectory ──────────────────────────────────────────────────────
    score_at_buy: Optional[float]   = None
    score_now: Optional[float]      = None
    rank_pct_at_buy: Optional[float] = None
    rank_pct_now: Optional[float]    = None


@dataclass
class DecisionOutput:
    """
    Final portfolio action + rich diagnostics.
    `action` is the only thing that changes portfolio behavior.
    All other fields are observational — they never feed back into factor scores.
    """
    action: str              # BUY / HOLD / WATCH / REVIEW / TRIM / HARVEST / EXIT
    confidence: str          # HIGH / MEDIUM / LOW
    adjustment_applied: bool
    adjustment_reason: str
    raw_sell_trigger: str            = ""      # original trigger before downgrades
    is_hard_exit: bool               = False
    is_review: bool                  = False
    is_trim_harvest: bool            = False
    premature_exit_probability: float          = 0.0
    thesis_intact_score: float                 = 0.5
    signal_deterioration_velocity: Optional[float] = None
    rank_decay: Optional[float]                    = None
    badges: list[str]                = field(default_factory=list)
    diagnostic_summary: str          = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DecisionAdjustmentEngine:
    """
    Produces final portfolio actions using diagnostic evidence.

    Invariants:
      1. Factor scores are read-only input — never written back to.
      2. Hard stops (stop_loss, trailing_stop) are never moderated.
      3. Diagnostics can soften exits — never create alpha.
      4. Calibration adjusts thresholds only — not factor params.
    """

    def __init__(self, calibration_overrides: Optional[dict] = None) -> None:
        self._cfg  = _dae_config()
        self._ecfg = _exit_config()
        self._cal  = calibration_overrides if calibration_overrides is not None else _load_calibration_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def adjust(self, inp: DecisionInput) -> DecisionOutput:
        tis  = inp.thesis_intact_score if inp.thesis_intact_score is not None else 0.5
        pep  = self._premature_exit_probability(inp)
        sdv  = self._signal_deterioration_velocity(inp)
        rdec = self._rank_decay(inp)

        if inp.raw_action in ("HOLD", "BUY"):
            return DecisionOutput(
                action=inp.raw_action,
                confidence="MEDIUM" if inp.raw_action == "HOLD" else "HIGH",
                adjustment_applied=False,
                adjustment_reason="",
                raw_sell_trigger="",
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                diagnostic_summary="Thesis intact" if inp.raw_action == "HOLD" else "Buy signal",
            )

        # Determine whether the raw exit is hard (mechanical safety rule)
        is_hard = (
            inp.exit_severity == "hard"
            or inp.exit_type in ("stop_loss", "trailing_stop")
            or inp.raw_reason.lower().startswith(("stop loss", "trailing stop"))
        )

        if inp.raw_action == "EXIT" and is_hard:
            return DecisionOutput(
                action="EXIT",
                confidence="HIGH",
                adjustment_applied=False,
                adjustment_reason="Hard mechanical stop — never moderated",
                raw_sell_trigger=inp.raw_reason,
                is_hard_exit=True,
                premature_exit_probability=0.0,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=["SAFE EXIT"],
                diagnostic_summary=f"Hard stop: {inp.raw_reason}",
            )

        if inp.raw_action == "EXIT":
            return self._evaluate_soft_exit(inp, pep, tis, sdv, rdec)

        if inp.raw_action == "WATCH":
            return self._evaluate_watch(inp, pep, tis, sdv, rdec)

        # Fallthrough
        return DecisionOutput(
            action=inp.raw_action,
            confidence="LOW",
            adjustment_applied=False,
            adjustment_reason="",
            raw_sell_trigger=inp.raw_reason,
            thesis_intact_score=tis,
            signal_deterioration_velocity=sdv,
            rank_decay=rdec,
        )

    # ── Soft EXIT — full 4-level hierarchy ────────────────────────────────────

    def _evaluate_soft_exit(
        self,
        inp: DecisionInput,
        pep: float,
        tis: float,
        sdv: Optional[float],
        rdec: Optional[float],
    ) -> DecisionOutput:
        ec = self._ecfg
        enabled = ec.get("enabled", True)
        requires_confirmation = ec.get("score_below_threshold_requires_confirmation", True)

        # Is this a score-below-threshold (soft thesis) exit?
        is_score_exit = (
            inp.exit_type in ("soft_thesis",)
            or "score below" in inp.raw_reason.lower()
            or "score below exit threshold" in inp.raw_reason.lower()
        )

        if not enabled or not is_score_exit or not requires_confirmation:
            return self._legacy_soft_exit(inp, pep, tis, sdv, rdec)

        pnl  = inp.pct_change      if inp.pct_change      is not None else 0.0
        mom  = inp.momentum_score  if inp.momentum_score  is not None else 0.0
        qual = inp.quality_score   if inp.quality_score   is not None else 0.0
        snw  = inp.score_now       if inp.score_now       is not None else 0.3

        # Thresholds from config (with calibration overrides)
        pet  = float(self._cal.get("premature_exit_threshold",
                     ec.get("premature_exit_threshold", 0.45)))
        rct  = float(self._cal.get("review_confidence_threshold",
                     ec.get("review_confidence_threshold", 0.50)))

        hard_score_floor = float(ec.get("hard_exit_score_below",  0.20))
        tis_hard_floor   = float(ec.get("thesis_intact_hard_exit_below", 0.35))
        harvest_pct      = float(ec.get("harvest_profit_threshold",  0.15))
        trim_pct         = float(ec.get("trim_profit_threshold",     0.08))
        pnl_floor        = float(ec.get("positive_pnl_review_floor",       0.00))
        mom_floor        = float(ec.get("positive_momentum_review_floor",   0.10))
        qual_floor       = float(ec.get("strong_quality_review_floor",      0.70))
        tis_floor        = float(ec.get("thesis_intact_review_floor",       0.60))

        # ── Positive signal flags ─────────────────────────────────────────────
        positive_pnl  = pnl >= pnl_floor   and ec.get("positive_pnl_exit_downgrade",        True)
        positive_mom  = mom >= mom_floor   and ec.get("positive_momentum_exit_downgrade",    True)
        strong_qual   = qual >= qual_floor  and ec.get("strong_quality_exit_downgrade",       True)
        thesis_ok     = tis >= tis_floor

        # ── Hard collapse check ───────────────────────────────────────────────
        # Both momentum AND quality materially negative = confirmed breakdown
        mom_bad  = mom  < -0.20
        qual_bad = qual < -0.20
        score_collapsed = snw < hard_score_floor
        thesis_broken   = tis < tis_hard_floor
        confirmed_breakdown = score_collapsed and thesis_broken and mom_bad and qual_bad

        if confirmed_breakdown:
            return DecisionOutput(
                action="EXIT",
                confidence="HIGH",
                adjustment_applied=False,
                adjustment_reason="Thesis confirmed broken — score collapse + quality and momentum both deteriorated",
                raw_sell_trigger=inp.raw_reason,
                is_hard_exit=True,
                premature_exit_probability=pep,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=["SAFE EXIT"],
                diagnostic_summary=f"Confirmed exit: score={snw:.3f}, tis={tis:.2f}, mom={mom:.2f}, qual={qual:.2f}",
            )

        # ── HARVEST — large gain + thesis partially alive ─────────────────────
        if pnl >= harvest_pct and ec.get("positive_pnl_exit_downgrade", True):
            if positive_mom or thesis_ok:
                parts = [f"gain={pnl:+.1%}"]
                if positive_mom: parts.append(f"momentum={mom:.2f}")
                if thesis_ok:    parts.append(f"thesis_intact={tis:.2f}")
                reason = f"Large gain with active signals: {', '.join(parts)}"
                return DecisionOutput(
                    action="HARVEST",
                    confidence="MEDIUM",
                    adjustment_applied=True,
                    adjustment_reason=reason,
                    raw_sell_trigger=inp.raw_reason,
                    is_trim_harvest=True,
                    premature_exit_probability=pep,
                    thesis_intact_score=tis,
                    signal_deterioration_velocity=sdv,
                    rank_decay=rdec,
                    badges=["HARVEST"],
                    diagnostic_summary=f"Harvest profits: {pnl:+.1%} gain, score weakening",
                )

        # ── TRIM — moderate gain + score weakening + momentum positive ────────
        if pnl >= trim_pct and ec.get("positive_pnl_exit_downgrade", True):
            if positive_mom:
                reason = f"Profitable ({pnl:+.1%}) with positive momentum ({mom:.2f}) — consider trimming"
                return DecisionOutput(
                    action="TRIM",
                    confidence="MEDIUM",
                    adjustment_applied=True,
                    adjustment_reason=reason,
                    raw_sell_trigger=inp.raw_reason,
                    is_trim_harvest=True,
                    premature_exit_probability=pep,
                    thesis_intact_score=tis,
                    signal_deterioration_velocity=sdv,
                    rank_decay=rdec,
                    badges=["TRIM"],
                    diagnostic_summary=f"Trim: {pnl:+.1%} gain, momentum={mom:.2f}",
                )

        # ── REVIEW — any positive signal OR premature exit flag ───────────────
        has_any_positive = positive_pnl or positive_mom or strong_qual or thesis_ok or inp.is_premature or pep >= pet

        if has_any_positive:
            parts = []
            if pnl > 0:       parts.append(f"P/L={pnl:+.1%}")
            if positive_mom:  parts.append(f"momentum={mom:.2f}")
            if strong_qual:   parts.append(f"quality={qual:.2f}")
            if thesis_ok:     parts.append(f"thesis_intact={tis:.2f}")
            if inp.is_premature: parts.append("premature_exit_flag")
            reason = (
                f"Score below threshold but conflicting evidence — "
                f"{', '.join(parts) if parts else 'positive signals present'}. "
                f"Confirmation required before exit."
            )
            return DecisionOutput(
                action="REVIEW",
                confidence="LOW",
                adjustment_applied=True,
                adjustment_reason=reason,
                raw_sell_trigger=inp.raw_reason,
                is_review=True,
                premature_exit_probability=pep,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=["REVIEW NEEDED"],
                diagnostic_summary=f"Score weak but {', '.join(parts[:2]) if parts else 'signals positive'} — review before acting",
            )

        # ── Multi-signal deterioration — all signals negative, no positives ─────
        # Exit even without full collapse when momentum AND quality are both bad
        # and the loss is meaningful (prevents hanging on clear deterioration)
        pnl_bad_enough = pnl < -0.05
        all_negative = mom_bad and qual_bad and pnl_bad_enough
        if all_negative:
            return DecisionOutput(
                action="EXIT",
                confidence="MEDIUM",
                adjustment_applied=False,
                adjustment_reason="",
                raw_sell_trigger=inp.raw_reason,
                is_hard_exit=False,
                premature_exit_probability=pep,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=["SAFE EXIT"],
                diagnostic_summary=f"Confirmed deterioration: pnl={pnl:+.1%}, mom={mom:.2f}, qual={qual:.2f}",
            )

        # ── WATCH — score weak but no strong positive or negative signals ─────
        if not score_collapsed:
            return DecisionOutput(
                action="WATCH",
                confidence="LOW",
                adjustment_applied=True,
                adjustment_reason="Score below threshold without confirmation — insufficient evidence for exit",
                raw_sell_trigger=inp.raw_reason,
                premature_exit_probability=pep,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=[],
                diagnostic_summary="Score weak — monitoring for confirmation signal",
            )

        # ── EXIT — score collapsed, no positive signals ───────────────────────
        return DecisionOutput(
            action="EXIT",
            confidence="MEDIUM",
            adjustment_applied=False,
            adjustment_reason="",
            raw_sell_trigger=inp.raw_reason,
            is_hard_exit=False,
            premature_exit_probability=pep,
            thesis_intact_score=tis,
            signal_deterioration_velocity=sdv,
            rank_decay=rdec,
            badges=["SAFE EXIT"],
            diagnostic_summary=f"Score collapsed (score={snw:.3f}) + no positive signals",
        )

    # ── Legacy path for non-score-threshold exits ─────────────────────────────

    def _legacy_soft_exit(
        self,
        inp: DecisionInput,
        pep: float,
        tis: float,
        sdv: Optional[float],
        rdec: Optional[float],
    ) -> DecisionOutput:
        """Original premature-exit-based logic for non-score-threshold exits."""
        pet = float(self._cal.get("premature_exit_threshold",
                    self._ecfg.get("premature_exit_threshold", 0.45)))
        rct = float(self._cal.get("review_confidence_threshold",
                    self._cfg.get("review_confidence_threshold", 0.50)))

        if inp.is_premature or pep >= pet:
            reason = (
                f"Premature exit flag (pep={pep:.0%})"
                if inp.is_premature
                else f"Premature exit probability {pep:.0%} ≥ {pet:.0%}"
            )
            return DecisionOutput(
                action="REVIEW",
                confidence="LOW",
                adjustment_applied=True,
                adjustment_reason=reason,
                raw_sell_trigger=inp.raw_reason,
                is_review=True,
                premature_exit_probability=pep,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=["REVIEW NEEDED"],
                diagnostic_summary=f"Exit conflicted — {pep:.0%} premature risk",
            )

        if inp.exit_confidence == "LOW" and tis >= rct:
            return DecisionOutput(
                action="REVIEW",
                confidence="LOW",
                adjustment_applied=True,
                adjustment_reason=f"Low exit confidence + thesis_intact_score={tis:.2f}",
                raw_sell_trigger=inp.raw_reason,
                is_review=True,
                premature_exit_probability=pep,
                thesis_intact_score=tis,
                signal_deterioration_velocity=sdv,
                rank_decay=rdec,
                badges=["REVIEW NEEDED"],
                diagnostic_summary="Uncertain exit on intact thesis",
            )

        badge = "RISKY EXIT" if tis > 0.50 else "SAFE EXIT"
        return DecisionOutput(
            action="EXIT",
            confidence=inp.exit_confidence or "MEDIUM",
            adjustment_applied=False,
            adjustment_reason="",
            raw_sell_trigger=inp.raw_reason,
            premature_exit_probability=pep,
            thesis_intact_score=tis,
            signal_deterioration_velocity=sdv,
            rank_decay=rdec,
            badges=[badge],
            diagnostic_summary=f"Exit confirmed (tis={tis:.2f}, pep={pep:.0%})",
        )

    # ── WATCH escalation ──────────────────────────────────────────────────────

    def _evaluate_watch(
        self,
        inp: DecisionInput,
        pep: float,
        tis: float,
        sdv: Optional[float],
        rdec: Optional[float],
    ) -> DecisionOutput:
        escalate = bool(self._cfg.get("escalate_watch_to_review", False))
        if escalate:
            fast_decay    = sdv is not None and sdv < -0.005
            rank_collapse = rdec is not None and rdec > 0.30
            if fast_decay and rank_collapse:
                return DecisionOutput(
                    action="REVIEW",
                    confidence="MEDIUM",
                    adjustment_applied=True,
                    adjustment_reason="WATCH escalated: fast decay + rank collapse",
                    raw_sell_trigger=inp.raw_reason,
                    is_review=True,
                    premature_exit_probability=pep,
                    thesis_intact_score=tis,
                    signal_deterioration_velocity=sdv,
                    rank_decay=rdec,
                    badges=["REVIEW NEEDED"],
                    diagnostic_summary="WATCH escalated to REVIEW",
                )

        return DecisionOutput(
            action="WATCH",
            confidence="MEDIUM",
            adjustment_applied=False,
            adjustment_reason="",
            raw_sell_trigger=inp.raw_reason,
            premature_exit_probability=pep,
            thesis_intact_score=tis,
            signal_deterioration_velocity=sdv,
            rank_decay=rdec,
            diagnostic_summary=inp.raw_reason,
        )

    # ── Diagnostic helpers (read-only — never write factor scores) ────────────

    def _premature_exit_probability(self, inp: DecisionInput) -> float:
        if inp.raw_action not in ("EXIT", "WATCH"):
            return 0.0

        if inp.is_premature:
            base = 0.75
        elif inp.thesis_intact_score is not None and inp.thesis_intact_score > 0.70:
            base = 0.50
        elif inp.exit_confidence == "LOW":
            base = 0.40
        elif inp.thesis_intact_score is not None and inp.thesis_intact_score > 0.55:
            base = 0.28
        else:
            base = 0.10

        pnl = inp.pct_change or 0.0
        if pnl > 0.08:
            base = min(1.0, base + 0.12)
        elif pnl > 0.02:
            base = min(1.0, base + 0.05)
        elif pnl < -0.15:
            base = max(0.0, base - 0.20)
        elif pnl < -0.08:
            base = max(0.0, base - 0.10)

        q = inp.quality_score or 0.0
        if q > 0.40:
            base = min(1.0, base + 0.05)
        elif q < -0.20:
            base = max(0.0, base - 0.15)

        m = inp.momentum_score or 0.0
        if m > 0.20:
            base = min(1.0, base + 0.05)
        elif m < -0.30:
            base = max(0.0, base - 0.10)

        return round(max(0.0, min(1.0, base)), 3)

    def _signal_deterioration_velocity(self, inp: DecisionInput) -> Optional[float]:
        if (
            inp.score_at_buy is not None
            and inp.score_now is not None
            and inp.holding_days is not None
            and inp.holding_days > 0
        ):
            return round((inp.score_now - inp.score_at_buy) / inp.holding_days, 5)
        return None

    def _rank_decay(self, inp: DecisionInput) -> Optional[float]:
        if inp.rank_pct_at_buy is not None and inp.rank_pct_now is not None:
            return round(inp.rank_pct_at_buy - inp.rank_pct_now, 3)
        return None


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_decision_input(
    raw_action: str,
    raw_reason: str,
    holding: dict,
    metrics,             # pd.Series or None
    buy_context: dict | None,
    exit_analysis,       # ExitAnalysis or None
    exit_severity: Optional[str] = None,
    exit_type: Optional[str]     = None,
) -> DecisionInput:
    """
    Build a DecisionInput from the objects already produced by position_rationale.py.
    One-way adapter — reads factor outputs, never writes back.
    """
    def _sf(v) -> Optional[float]:
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    pct_raw    = _sf(holding.get("percent_change"))
    pct_change = pct_raw / 100.0 if pct_raw is not None else None
    if pct_change is None:
        avg = _sf(holding.get("average_buy_price"))
        cur = _sf(holding.get("current_price"))
        if avg and avg > 0 and cur:
            pct_change = (cur / avg) - 1.0

    import datetime
    holding_days: Optional[int] = None
    if buy_context:
        try:
            bd = datetime.date.fromisoformat(str(buy_context.get("buy_date", "")).strip())
            holding_days = (datetime.date.today() - bd).days
        except Exception:
            pass

    cs  = _sf(metrics.get("composite_score") if metrics is not None else None) or \
          _sf(metrics.get("value_metric")     if metrics is not None else None)
    qs  = _sf(metrics.get("quality_score")   if metrics is not None else None)
    ms  = _sf(metrics.get("momentum_score")  if metrics is not None else None)
    vs  = _sf(metrics.get("value_score")     if metrics is not None else None)
    ins = _sf(metrics.get("income_score")    if metrics is not None else None)
    snw = _sf(metrics.get("value_metric")    if metrics is not None else None)

    thesis_intact_score = _sf(getattr(exit_analysis, "thesis_intact_score",  None))
    exit_confidence     = getattr(exit_analysis, "confidence",   None)
    is_premature        = bool(getattr(exit_analysis, "is_premature", False))
    conf_collapse_w     = _sf(
        getattr(exit_analysis, "reason_weights", {}).get("confidence_loss") if exit_analysis else None
    )

    score_at_buy = _sf(buy_context.get("composite_score_at_buy")    if buy_context else None)
    rank_at_buy  = _sf(buy_context.get("universe_rank_pct_at_buy")  if buy_context else None)

    return DecisionInput(
        composite_score=cs,
        value_score=vs,
        quality_score=qs,
        momentum_score=ms,
        income_score=ins,
        percentile_rank=rank_at_buy,
        raw_action=raw_action,
        raw_reason=raw_reason,
        exit_severity=exit_severity,
        exit_type=exit_type,
        pct_change=pct_change,
        holding_days=holding_days,
        thesis_intact_score=thesis_intact_score,
        exit_confidence=exit_confidence,
        is_premature=is_premature,
        confidence_collapse_weight=conf_collapse_w,
        score_at_buy=score_at_buy,
        score_now=snw,
        rank_pct_at_buy=rank_at_buy,
        rank_pct_now=None,   # populated by caller
    )
