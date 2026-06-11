"""
portfolio/contribution_timing.py — buy-the-dip weekly contribution overlay.

Sizes each week's NEW cash contribution from short-term benchmark conditions:
contribute more during short-term drawdowns, less after strong rallies, while a
rolling budget window keeps the ~monthly total near target. This is a
contribution-sizing overlay ONLY — it never sells holdings, never alters
positions, and changes nothing except how much new cash arrives each week.

Causality: the dip score for a contribution at day *d* uses benchmark closes
strictly BEFORE d (``bench_prices[:d]``) — the signal is computable the night
before the cash is deployed. No future data.

Component normalization (each in [0, 1]; 0 = extended market, 1 = meaningful dip):
  • 1w / 1m returns and MA gaps are CENTERED — a flat market scores 0.5.
  • Drawdown-from-high components ramp from 0 (at the high) to 1 (deep dip).
  With the default weights this puts a flat, at-the-high market at a dip score
  ≈ 0.35 — exactly the default ``neutral_dip_score`` — so a normal week yields
  multiplier ≈ 1.0 and the base $400 contribution.

The whole schedule depends only on benchmark prices + config, so backtests can
precompute it for every day before the sim loop (`build_contribution_schedule`)
and feed the SAME varied schedule to the benchmark comparator.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Fixed normalization scales (full-score points for each component). These are
# deliberately NOT config knobs — the tunable surface is the component WEIGHTS
# and the multiplier mapping; adding per-component scales would over-knob the
# overlay (12 ways to overfit instead of 8).
_SCALE_RET_1W   = 0.06   # ±3% week spans the [1, 0] range around 0.5
_SCALE_RET_1M   = 0.12   # ±6% month
_SCALE_DD_20    = 0.06   # 6% below the 20d high → full score
_SCALE_DD_60    = 0.12   # 12% below the 60d high → full score
_SCALE_MA50_GAP = 0.08   # ±4% around the 50DMA
_SCALE_MA200_GAP = 0.16  # ±8% around the 200DMA

# Minimum prior closes before the signal is trusted at all. Below this the
# overlay is inert for the week (multiplier 1.0, reason insufficient_history).
_MIN_HISTORY_DAYS = 60


@dataclass
class ContributionDecision:
    """One week's contribution decision with full diagnostics."""
    base_amount: float
    adjusted_amount: float
    multiplier: float
    dip_score: float
    budget_window_contributed: float   # including this week's adjusted amount
    remaining_budget: float            # window headroom left AFTER this week
    carry_forward: float               # unused budget banked AFTER this week
    reason_codes: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    day: int | None = None


@dataclass
class ContributionState:
    """Rolling state carried between weekly decisions (in-sim or persisted live)."""
    window_amounts: deque = field(default_factory=deque)  # last N weekly adjusted amounts
    carry_forward: float = 0.0
    prev_multiplier: float | None = None

    def window_sum(self) -> float:
        return float(sum(self.window_amounts))


def _centered(value: float, scale: float) -> float:
    """0.5 at value==0; 1.0 at value == -scale/2 (dip); 0.0 at +scale/2 (extended)."""
    return float(np.clip(0.5 - value / scale, 0.0, 1.0))


def _ramp(value: float, scale: float) -> float:
    """0.0 at value<=0 rising linearly to 1.0 at value>=scale."""
    return float(np.clip(value / scale, 0.0, 1.0))


def compute_dip_score(
    bench_history: np.ndarray,
    dip_cfg: dict,
) -> tuple[float, dict[str, float], list[str]]:
    """
    Causal dip score from PRIOR benchmark closes (the caller passes history that
    excludes the deployment day). Returns (score, component_scores, reason_codes).

    Components whose lookback exceeds the available history are dropped and the
    remaining weights renormalized, so the score stays meaningful early in a
    window (e.g. before 200 days exist for the MA200 gap).
    """
    px = np.asarray(bench_history, dtype=float)
    px = px[np.isfinite(px) & (px > 0)]
    n = len(px)
    reasons: list[str] = []
    if n < _MIN_HISTORY_DAYS:
        return float("nan"), {}, ["insufficient_history"]

    w_cfg = dip_cfg.get("weights", {}) or {}
    lb_1w   = int(dip_cfg.get("lookback_1w_days", 5))
    lb_1m   = int(dip_cfg.get("lookback_1m_days", 21))
    hi_s    = int(dip_cfg.get("high_lookback_short", 20))
    hi_m    = int(dip_cfg.get("high_lookback_medium", 60))
    ma_s    = int(dip_cfg.get("ma_short", 50))
    ma_l    = int(dip_cfg.get("ma_long", 200))

    last = px[-1]
    comp: dict[str, float] = {}
    weights: dict[str, float] = {}

    def _add(name: str, score: float, weight: float) -> None:
        if weight > 0:
            comp[name] = score
            weights[name] = weight

    if n > lb_1w:
        r1w = last / px[-1 - lb_1w] - 1.0
        _add("return_1w", _centered(r1w, _SCALE_RET_1W), float(w_cfg.get("return_1w", 0.25)))
        if r1w < -0.01:
            reasons.append("market_down_1w")
    if n > lb_1m:
        r1m = last / px[-1 - lb_1m] - 1.0
        _add("return_1m", _centered(r1m, _SCALE_RET_1M), float(w_cfg.get("return_1m", 0.25)))
    if n >= hi_s:
        dd20 = max(0.0, 1.0 - last / px[-hi_s:].max())
        _add("drawdown_20d", _ramp(dd20, _SCALE_DD_20), float(w_cfg.get("drawdown_20d", 0.20)))
        if dd20 > 0.02:
            reasons.append("drawdown_from_20d_high")
    if n >= hi_m:
        dd60 = max(0.0, 1.0 - last / px[-hi_m:].max())
        _add("drawdown_60d", _ramp(dd60, _SCALE_DD_60), float(w_cfg.get("drawdown_60d", 0.15)))
    if n >= ma_s:
        gap50 = last / float(px[-ma_s:].mean()) - 1.0
        _add("ma50_gap", _centered(gap50, _SCALE_MA50_GAP), float(w_cfg.get("ma50_gap", 0.10)))
        if gap50 < -0.005:
            reasons.append("market_below_50dma")
    if n >= ma_l:
        gap200 = last / float(px[-ma_l:].mean()) - 1.0
        _add("ma200_gap", _centered(gap200, _SCALE_MA200_GAP), float(w_cfg.get("ma200_gap", 0.05)))

    total_w = sum(weights.values())
    if not comp or total_w <= 0:
        return float("nan"), {}, ["insufficient_history"]
    # Normalize weights over the computable components so they sum to 1.0.
    score = sum(comp[k] * weights[k] for k in comp) / total_w
    return float(score), comp, reasons


def contribution_multiplier(
    dip_score: float,
    mult_cfg: dict,
    prev_multiplier: float | None = None,
) -> float:
    """base + sensitivity * (dip - neutral), EMA-smoothed against last week, clamped."""
    neutral = float(mult_cfg.get("neutral_dip_score", 0.35))
    sens    = float(mult_cfg.get("dip_sensitivity", 1.25))
    lo      = float(mult_cfg.get("min_multiplier", 0.50))
    hi      = float(mult_cfg.get("max_multiplier", 2.00))
    alpha   = float(mult_cfg.get("smoothing_alpha", 0.50))

    raw = 1.0 + sens * (dip_score - neutral)
    if prev_multiplier is not None and 0.0 < alpha < 1.0:
        raw = alpha * raw + (1.0 - alpha) * prev_multiplier
    return float(np.clip(raw, lo, hi))


def _apply_budget_and_bounds(
    raw: float,
    base: float,
    cfg: dict,
    state: ContributionState,
    reasons: list[str],
) -> float:
    """Clamp a raw weekly amount through the budget mechanics, appending reason
    codes and updating ``state.carry_forward`` in place: weekly min/max bounds →
    carry/borrow rules for above-base spending → rolling window cap →
    carry-forward bookkeeping. Returns the final adjusted amount."""
    target  = float(cfg.get("target_monthly_contribution", 1600.0))
    wk_min  = float(cfg.get("min_weekly_contribution", 100.0))
    wk_max  = float(cfg.get("max_weekly_contribution", 800.0))
    preserve = bool(cfg.get("preserve_monthly_budget", True))
    accelerate = bool(cfg.get("allow_budget_acceleration", False))
    tol     = float(cfg.get("monthly_budget_tolerance_pct", 0.15))
    carry_ok = bool(cfg.get("carry_forward_unused_budget", True))
    borrow_ok = bool(cfg.get("borrow_from_future_weeks", True))

    # Weekly hard bounds.
    adjusted = raw
    if adjusted > wk_max:
        adjusted = wk_max
        reasons.append("max_weekly_cap")
    if adjusted < wk_min:
        adjusted = wk_min
        reasons.append("min_weekly_floor")

    # Above-base spending draws on banked carry-forward first; without
    # borrow_from_future_weeks it may not exceed base + banked carry.
    if adjusted > base:
        available_carry = state.carry_forward if carry_ok else 0.0
        if not borrow_ok and adjusted > base + available_carry:
            adjusted = base + available_carry
            reasons.append("borrow_disabled_cap")
        if available_carry > 0:
            reasons.append("carry_forward_used")

    # Rolling budget cap across the window (this week counts toward it).
    if preserve and not accelerate:
        budget_cap = target * (1.0 + tol) + (state.carry_forward if carry_ok else 0.0)
        headroom = budget_cap - state.window_sum()
        if adjusted > headroom:
            adjusted = max(0.0, headroom)
            reasons.append("monthly_budget_cap")

    adjusted = float(round(adjusted, 2))

    # Carry-forward bookkeeping (banked under-spend, consumed by over-spend).
    if carry_ok:
        if adjusted < base:
            state.carry_forward = min(state.carry_forward + (base - adjusted), target)
        elif adjusted > base:
            state.carry_forward = max(0.0, state.carry_forward - (adjusted - base))
    return adjusted


def decide_contribution(
    bench_history: np.ndarray,
    cfg: dict,
    state: ContributionState,
    regime: str | None = None,
    day: int | None = None,
) -> ContributionDecision:
    """
    Full weekly pipeline: dip score → multiplier (regime-capped) → raw amount →
    weekly min/max clamps → rolling budget constraint → carry-forward update.

    MUTATES ``state`` (window deque, carry_forward, prev_multiplier) so the
    caller can run it week after week. ``bench_history`` must contain only
    closes available BEFORE the contribution is deployed.
    """
    base   = float(cfg.get("base_weekly_contribution", 400.0))
    target = float(cfg.get("target_monthly_contribution", 1600.0))
    window = int(cfg.get("budget_window_weeks", 4))

    dip, comps, reasons = compute_dip_score(bench_history, cfg.get("dip_signal", {}) or {})

    if np.isnan(dip):
        mult = 1.0
    else:
        mult = contribution_multiplier(dip, cfg.get("multiplier", {}) or {}, state.prev_multiplier)
        neutral = float((cfg.get("multiplier", {}) or {}).get("neutral_dip_score", 0.35))
        if dip < neutral - 0.10:
            reasons.append("market_extended")

    # Regime cap: optionally limit dip-buying aggression in a defensive tape —
    # downturns are not automatically good buys in bear regimes.
    rc = cfg.get("regime_controls", {}) or {}
    if (
        regime == "defensive"
        and bool(rc.get("cap_multiplier_in_defensive", True))
        and mult > float(rc.get("defensive_max_multiplier", 1.25))
    ):
        mult = float(rc.get("defensive_max_multiplier", 1.25))
        reasons.append("defensive_regime_cap")

    adjusted = _apply_budget_and_bounds(base * mult, base, cfg, state, reasons)

    state.window_amounts.append(adjusted)
    while len(state.window_amounts) > window:
        state.window_amounts.popleft()
    if not np.isnan(dip):
        state.prev_multiplier = mult

    window_sum_after = state.window_sum()
    return ContributionDecision(
        base_amount=base,
        adjusted_amount=adjusted,
        multiplier=float(mult),
        dip_score=float(dip) if not np.isnan(dip) else float("nan"),
        budget_window_contributed=window_sum_after,
        remaining_budget=max(0.0, target - window_sum_after),
        carry_forward=state.carry_forward,
        reason_codes=reasons,
        components=comps,
        day=day,
    )


def build_contribution_schedule(
    bench_prices: np.ndarray,
    n_days: int,
    rebalance_frequency_days: int,
    flat_weekly_contribution: float,
    cfg: dict | None,
    regime_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, list[ContributionDecision]]:
    """
    Precompute the per-day contribution amounts for a backtest window.

    Returns (amounts[n_days], decisions). When the overlay is disabled (or cfg
    is None) the schedule is EXACTLY the flat legacy behavior: ``flat`` on every
    contribution day — byte-identical results to the pre-overlay simulator.

    The overlay's base/target come from the overlay config; the sim's
    ``weekly_contribution`` argument remains the flat fallback so disabled runs
    and tuner calls are unchanged.
    """
    amounts = np.zeros(n_days)
    contrib_days = [
        d for d in range(n_days) if d > 0 and d % rebalance_frequency_days == 0
    ]
    if not cfg or not cfg.get("enabled", False):
        for d in contrib_days:
            amounts[d] = flat_weekly_contribution
        return amounts, []

    state = ContributionState()
    decisions: list[ContributionDecision] = []
    for d in contrib_days:
        regime = None
        if regime_labels is not None and d < len(regime_labels):
            regime = str(regime_labels[d])
        decision = decide_contribution(
            bench_prices[:d],  # strictly prior closes — no same-day data
            cfg,
            state,
            regime=regime,
            day=d,
        )
        decisions.append(decision)
        amounts[d] = decision.adjusted_amount
    return amounts, decisions


# ---------------------------------------------------------------------------
# Live path — persisted weekly state + display panel
# ---------------------------------------------------------------------------

def load_live_state(csv_path: str, cfg: dict) -> ContributionState:
    """Rebuild ContributionState from the persisted weekly log (date, amount,
    multiplier, carry_forward). Window = entries within budget_window_weeks * 7
    calendar days. Missing/empty log → fresh state."""
    import datetime

    import pandas as pd

    state = ContributionState()
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except FileNotFoundError:
        return state
    except Exception as exc:
        logger.warning("Could not load contribution log %s: %s", csv_path, exc)
        return state
    if df.empty:
        return state
    df = df.sort_values("date")
    window_days = int(cfg.get("budget_window_weeks", 4)) * 7
    cutoff = datetime.datetime.now() - datetime.timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]
    for amt in recent["amount"].tolist():
        state.window_amounts.append(float(amt))
    last = df.iloc[-1]
    state.carry_forward = float(last.get("carry_forward", 0.0) or 0.0)
    if "multiplier" in df.columns and np.isfinite(last.get("multiplier", np.nan)):
        state.prev_multiplier = float(last["multiplier"])
    return state


def record_live_decision(csv_path: str, decision: ContributionDecision) -> bool:
    """Append this week's decision to the log — at most one row per 5 calendar
    days, so multiple runs in the same week don't pollute the budget window.
    Returns True when a row was written."""
    import datetime

    import pandas as pd

    now = datetime.datetime.now()
    try:
        existing = pd.read_csv(csv_path, parse_dates=["date"])
        if not existing.empty:
            last = existing["date"].max()
            if (now - last).days < 5:
                return False
    except FileNotFoundError:
        existing = None
    except Exception as exc:
        logger.warning("Could not read contribution log %s: %s", csv_path, exc)
        existing = None

    row = pd.DataFrame([{
        "date": now.strftime("%Y-%m-%d"),
        "amount": decision.adjusted_amount,
        "base": decision.base_amount,
        "multiplier": round(decision.multiplier, 4),
        "dip_score": round(decision.dip_score, 4) if np.isfinite(decision.dip_score) else "",
        "carry_forward": round(decision.carry_forward, 2),
        "reason_codes": "|".join(decision.reason_codes),
    }])
    try:
        import os
        row.to_csv(csv_path, mode="a" if os.path.exists(csv_path) else "w",
                   header=not os.path.exists(csv_path), index=False)
        return True
    except Exception as exc:
        logger.warning("Could not record contribution decision: %s", exc)
        return False


def format_live_panel(decision: ContributionDecision, cfg: dict) -> str:
    """Human-readable weekly recommendation block for the live run log."""
    target = float(cfg.get("target_monthly_contribution", 1600.0))
    used = decision.budget_window_contributed
    dip_txt = f"{decision.dip_score:.2f}" if np.isfinite(decision.dip_score) else "n/a"
    stance = (
        "ABOVE normal — buying the dip" if decision.adjusted_amount > decision.base_amount + 0.01
        else "BELOW normal — market extended" if decision.adjusted_amount < decision.base_amount - 0.01
        else "AT normal"
    )
    lines = [
        "Contribution Timing:",
        f"  Base weekly contribution: ${decision.base_amount:,.0f}",
        f"  Recommended this week:    ${decision.adjusted_amount:,.2f}  ({stance})",
        f"  Multiplier: {decision.multiplier:.2f}x   Dip score: {dip_txt}",
        f"  Reasons: {', '.join(decision.reason_codes) if decision.reason_codes else 'none'}",
        f"  Monthly budget used: ${used:,.2f} / ${target:,.0f}"
        f"   Remaining: ${decision.remaining_budget:,.2f}"
        f"   Carry-forward: ${decision.carry_forward:,.2f}",
    ]
    return "\n".join(lines)


def summarize_decisions(decisions: list[ContributionDecision], base: float) -> dict:
    """Compact stats block for reports/SimResult."""
    if not decisions:
        return {}
    amts = np.array([d.adjusted_amount for d in decisions])
    dips = np.array([d.dip_score for d in decisions])
    finite_dips = dips[np.isfinite(dips)]
    return {
        "weeks": len(decisions),
        "total_contributed": float(amts.sum()),
        "avg_weekly": float(amts.mean()),
        "min_weekly": float(amts.min()),
        "max_weekly": float(amts.max()),
        "pct_weeks_above_base": float((amts > base + 0.01).mean()),
        "pct_weeks_below_base": float((amts < base - 0.01).mean()),
        "avg_dip_score": float(finite_dips.mean()) if len(finite_dips) else float("nan"),
        "final_carry_forward": float(decisions[-1].carry_forward),
    }
