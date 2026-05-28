"""
ui/components/config_diagnostics.py — Config sanity audit and live-vs-backtest diagnostics.

Reads config.yaml and compares against known invariants, tuner bounds,
backtest plumbing assumptions, and exit/harvest simulation gaps.

SAFE: read-only except for the optional "Apply patch" flow which requires
      ui.allow_config_writes: true and explicit user confirmation.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[3]
CFG_DIR = ROOT / "cfg"
CFG_PATH = CFG_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Audit finding data structure
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    category: str
    field: str
    current_value: Any
    problem: str
    severity: str          # CRITICAL / MODERATE / LOW
    recommendation: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Audit logic — pure Python, no imports from util so it works on any dict
# ---------------------------------------------------------------------------

def audit_config(cfg: dict) -> list[Finding]:
    findings: list[Finding] = []

    # ── 0. Deprecated keys ────────────────────────────────────────────────
    if "bear_market" in cfg:
        findings.append(Finding(
            category="Deprecated keys",
            field="bear_market",
            current_value=str(cfg["bear_market"]),
            problem="The 'bear_market' section is deprecated and ignored at runtime. "
                    "All regime detection now uses the 'regime' section.",
            severity="LOW",
            recommendation="Remove bear_market from config.yaml to reduce confusion.",
        ))

    index_pct     = float(cfg.get("index_pct", 0.85))
    sw            = cfg.get("score_weights", {})
    sw_income     = float(sw.get("income", 0.15))
    sw_quality    = float(sw.get("quality", 0.25))
    sw_momentum   = float(sw.get("momentum", 0.15))
    sw_value      = float(sw.get("value", 0.45))
    risk          = cfg.get("risk", {})
    min_idx_pct   = float(risk.get("min_index_pct", 0.60))
    cs            = cfg.get("candidate_selection", {})
    tn            = cfg.get("tuning", {})
    bt            = cfg.get("backtest", {})
    exit_d        = cfg.get("exit_decision", {})
    sell          = cfg.get("sell_rules", {})
    regime        = cfg.get("regime", {})
    neutral_rg    = regime.get("neutral", {})

    # ── 1. index_pct below min_index_pct ─────────────────────────────────
    if index_pct < min_idx_pct:
        findings.append(Finding(
            category="Allocation",
            field="index_pct vs risk.min_index_pct",
            current_value=f"index_pct={index_pct:.2f}, min_index_pct={min_idx_pct:.2f}",
            problem=(
                f"index_pct ({index_pct:.0%}) is BELOW the tuner floor min_index_pct ({min_idx_pct:.0%}). "
                "The optimizer's BOUNDS start at min_index_pct, so it has never tested the live allocation. "
                "Backtest validation metrics refer to a different strategy than what runs live."
            ),
            severity="CRITICAL",
            recommendation=f"Align index_pct to {min_idx_pct:.2f} or lower min_index_pct to {index_pct:.2f}. "
                           "Do not tune until these agree.",
        ))

    # ── 2. Backtest weekly contribution split ─────────────────────────────
    findings.append(Finding(
        category="Backtest plumbing",
        field="Weekly contribution ETF split",
        current_value="Fixed in backtest.py (2026-05-25)",
        problem=(
            "Before the fix, 100% of each weekly $400 contribution went to stocks in the backtest. "
            "Live bot splits each contribution by index_pct (e.g. 50% ETFs, 50% stocks). "
            "Over 365 days this was ~$10,400 extra stock exposure in backtest vs live."
        ),
        severity="LOW",
        recommendation="Verify the fix is present in backtest.py (line ~895 should split contribution by index_pct).",
        detail="FIXED — backtest now splits weekly contributions by index_pct.",
    ))

    # ── 3. income weight vs tuner bound ──────────────────────────────────
    tn_bounds = tn.get("parameter_bounds", {})
    income_bound_max = tn_bounds.get("score_weights.income", {}).get("max", None)
    if income_bound_max is not None and sw_income > income_bound_max:
        findings.append(Finding(
            category="Tuner bounds",
            field="score_weights.income vs tuning.parameter_bounds.score_weights.income.max",
            current_value=f"income={sw_income:.2f}, bound max={income_bound_max:.2f}",
            problem=(
                f"Live income weight ({sw_income:.2f}) exceeds the tuner bound max ({income_bound_max:.2f}). "
                "If income is unfrozen, the optimizer silently clamps it to the bound. "
                "The live config is in a region the optimizer can never validate or tune toward."
            ),
            severity="MODERATE",
            recommendation=f"Either raise the bound max to {sw_income:.2f} or reduce income weight to {income_bound_max:.2f}.",
        ))

    # ── 4. income frozen with self-canceling income trap ─────────────────
    frozen = set(tn.get("frozen_parameters", []))
    min_cond_mom = float(cs.get("min_conditional_momentum_score", 0.00))
    if sw_income >= 0.15 and "score_weights.income" in frozen and min_cond_mom >= 0.0:
        findings.append(Finding(
            category="Candidate selection",
            field="score_weights.income + income_trap interaction",
            current_value=f"income={sw_income:.2f}, min_conditional_momentum_score={min_cond_mom:.2f}",
            problem=(
                f"Income weight {sw_income:.2f} attracts dividend stocks, but income trap protection "
                f"(min_conditional_momentum_score={min_cond_mom:.2f}) excludes any income-paying stock with "
                "momentum < 0. At income=0.20, the weight primarily functions to admit high-income/high-momentum "
                "names, giving little benefit over a lower weight. The 0.20 weight is mostly self-canceling "
                "for defensive dividend names."
            ),
            severity="MODERATE",
            recommendation="Reduce income to 0.10 and set min_conditional_momentum_score: -0.05 "
                           "to allow defensive income names during corrections.",
        ))

    # ── 5. Absolute score floor + percentile double-filter ────────────────
    use_floor = bool(cs.get("use_absolute_score_floor", False))
    abs_floor = float(cs.get("absolute_score_floor", 0.45))
    cs_mode   = cs.get("mode", "percentile")
    if use_floor and cs_mode == "percentile" and abs_floor >= 0.40:
        findings.append(Finding(
            category="Candidate selection",
            field="use_absolute_score_floor + percentile mode",
            current_value=f"mode=percentile, use_absolute_score_floor=true, floor={abs_floor:.2f}",
            problem=(
                f"Percentile mode already selects the top {cs.get('top_percentile', 0.15):.0%} by score. "
                f"The absolute floor ({abs_floor:.2f}) then removes any top-percentile stock below the floor. "
                "When score distributions compress (correction/defensive regime), more candidates fail the floor "
                "and pool size becomes unpredictable. In MEDIUM-bias backtests scores are inflated, so the floor "
                "rarely fires, creating a live/backtest divergence in pool composition."
            ),
            severity="MODERATE",
            recommendation="Disable use_absolute_score_floor in percentile mode, or drop the floor to 0.30 "
                           "as a pure emergency backstop only.",
        ))

    # ── 6. Exit threshold vs backtest take_profit mismatch ────────────────
    trim_threshold    = float(exit_d.get("trim_profit_threshold", 0.08))
    harvest_threshold = float(exit_d.get("harvest_profit_threshold", 0.15))
    take_profit_bt    = float(sell.get("take_profit_pct", 0.60))

    if trim_threshold < 0.12:
        findings.append(Finding(
            category="Exit/harvest vs backtest",
            field="exit_decision.trim_profit_threshold",
            current_value=f"trim={trim_threshold:.0%}, backtest take_profit={take_profit_bt:.0%}",
            problem=(
                f"Live bot trims winners at +{trim_threshold:.0%}. "
                f"Backtest holds positions until +{take_profit_bt:.0%} or weak-value exit. "
                "These are categorically different strategies. Backtest performance measures "
                "a patient buy-and-hold-to-60% strategy; live performance will differ systematically. "
                f"Winners clipped at {trim_threshold:.0%} cannot compound to backtest levels."
            ),
            severity="CRITICAL",
            recommendation="Raise trim_profit_threshold to 0.15 and harvest_profit_threshold to 0.25. "
                           "Or add trim/harvest simulation to the backtest engine so results are comparable.",
        ))

    if harvest_threshold < 0.20 and trim_threshold >= 0.12:
        findings.append(Finding(
            category="Exit/harvest vs backtest",
            field="exit_decision.harvest_profit_threshold",
            current_value=f"harvest={harvest_threshold:.0%}, backtest take_profit={take_profit_bt:.0%}",
            problem=(
                f"Harvest at +{harvest_threshold:.0%} still clips winners well before the backtest's "
                f"+{take_profit_bt:.0%} take-profit. Performance comparisons remain misleading."
            ),
            severity="MODERATE",
            recommendation="Raise harvest_profit_threshold to at least 0.25.",
        ))

    # ── 7. Turnover penalty cliff at same threshold as diversification floor ─
    tp_count  = int(bt.get("turnover_penalty_trade_count", 80))
    tp_weight = float(bt.get("turnover_penalty_weight", 1.0))
    _MIN_TRADES_SOFT = 40  # hardcoded in tuner.py
    if tp_count == _MIN_TRADES_SOFT:
        findings.append(Finding(
            category="Backtest plumbing",
            field="backtest.turnover_penalty_trade_count",
            current_value=f"turnover_penalty_trade_count={tp_count}, internal MIN_TRADES_SOFT={_MIN_TRADES_SOFT}",
            problem=(
                f"Turnover penalty starts at {tp_count} trades, which equals the internal soft "
                f"diversification floor ({_MIN_TRADES_SOFT} trades). Exactly {tp_count} trades passes "
                "the diversification check while incurring zero turnover penalty — one more trade starts "
                "the penalty. This creates contradictory optimizer pressure right at the boundary."
            ),
            severity="MODERATE",
            recommendation="Raise turnover_penalty_trade_count to 60 (above the 40-trade floor) "
                           "and reduce turnover_penalty_weight to 0.15 to soften the penalty.",
        ))
    elif tp_weight > 0.3:
        findings.append(Finding(
            category="Backtest plumbing",
            field="backtest.turnover_penalty_weight",
            current_value=f"turnover_penalty_weight={tp_weight:.2f}",
            problem=(
                f"Turnover penalty weight {tp_weight:.2f} is aggressive. For a 90-day run with 72 possible "
                f"trades, penalty at full activity = ({tp_count}-{_MIN_TRADES_SOFT})/{tp_count} × {tp_weight:.1f} "
                "= significant. This strongly biases the optimizer toward low-churn strategies which may "
                "not be optimal."
            ),
            severity="MODERATE",
            recommendation="Reduce turnover_penalty_weight to 0.15.",
        ))

    # ── 8. Neutral regime not differentiated from bullish ────────────────
    neutral_idx_ovr = neutral_rg.get("index_pct_override")
    if neutral_idx_ovr is None:
        findings.append(Finding(
            category="Regime",
            field="regime.neutral.index_pct_override",
            current_value="null (falls back to base index_pct)",
            problem=(
                "Neutral regime (VIX 20–30) uses the same index_pct as bullish. "
                "No defensive tilt applied during elevated volatility."
            ),
            severity="LOW",
            recommendation="Set regime.neutral.index_pct_override: 0.72 for a light defensive tilt.",
        ))

    # ── 9. stop_loss hardcoded in backtest.py ────────────────────────────
    stop_loss_cfg = float(sell.get("stop_loss_pct", -0.20))
    _BT_HARDCODED_STOP = -0.20
    if abs(stop_loss_cfg - _BT_HARDCODED_STOP) > 0.01:
        findings.append(Finding(
            category="Backtest plumbing",
            field="sell_rules.stop_loss_pct vs backtest hardcoded",
            current_value=f"config={stop_loss_cfg:.0%}, backtest hardcoded={_BT_HARDCODED_STOP:.0%}",
            problem=(
                f"Config has stop_loss_pct={stop_loss_cfg:.0%} but backtest.py hardcodes "
                f"_STOP_LOSS_PCT={_BT_HARDCODED_STOP:.0%}. If you change the config value, live behavior "
                "changes but backtest is not affected."
            ),
            severity="LOW",
            recommendation="Pipe SELL_RULES['stop_loss_pct'] into the backtest simulation rather than hardcoding.",
        ))

    # ── 10. score_weights sum ─────────────────────────────────────────────
    total_sw = sw_value + sw_quality + sw_income + sw_momentum
    if abs(total_sw - 1.0) > 0.01:
        findings.append(Finding(
            category="Config validity",
            field="score_weights sum",
            current_value=f"{total_sw:.3f}",
            problem=f"Score weights sum to {total_sw:.3f}, not 1.0. util.py will fall back to defaults.",
            severity="CRITICAL",
            recommendation="Ensure value + quality + income + momentum = 1.0 exactly.",
        ))

    # ── 11. allow_income_defensive_exception in non-defensive regimes ─────
    allow_def_exc = bool(cs.get("allow_income_defensive_exception", False))
    if not allow_def_exc:
        findings.append(Finding(
            category="Candidate selection",
            field="candidate_selection.allow_income_defensive_exception",
            current_value="false",
            problem=(
                "Income trap protection removes income-paying stocks with negative momentum even in "
                "defensive regime. This blocks exactly the defensive dividend names that provide "
                "stability during drawdowns (utilities, REITs, staples)."
            ),
            severity="LOW",
            recommendation="Set allow_income_defensive_exception: true.",
        ))

    return findings


# ---------------------------------------------------------------------------
# Recommended minimal patch (critical + moderate issues only)
# ---------------------------------------------------------------------------

def _recommended_patch_lines(cfg: dict) -> list[str]:
    """Return the set of config keys that should change, with values."""
    patches: list[str] = []
    index_pct   = float(cfg.get("index_pct", 0.85))
    risk        = cfg.get("risk", {})
    min_idx_pct = float(risk.get("min_index_pct", 0.60))
    sw          = cfg.get("score_weights", {})
    sw_income   = float(sw.get("income", 0.15))
    exit_d      = cfg.get("exit_decision", {})
    trim        = float(exit_d.get("trim_profit_threshold", 0.08))
    harvest_t   = float(exit_d.get("harvest_profit_threshold", 0.15))
    cs          = cfg.get("candidate_selection", {})
    use_floor   = bool(cs.get("use_absolute_score_floor", False))
    bt          = cfg.get("backtest", {})
    tp_count    = int(bt.get("turnover_penalty_trade_count", 80))
    tp_weight   = float(bt.get("turnover_penalty_weight", 1.0))

    if index_pct < min_idx_pct:
        patches.append(f"index_pct: {index_pct:.2f}  →  index_pct: {min_idx_pct:.2f}")
    if trim < 0.12:
        patches.append(f"exit_decision.trim_profit_threshold: {trim:.2f}  →  0.15")
    if harvest_t < 0.20:
        patches.append(f"exit_decision.harvest_profit_threshold: {harvest_t:.2f}  →  0.25")
    if sw_income > 0.12:
        patches.append(f"score_weights.income: {sw_income:.2f}  →  0.10  (adjust momentum to 0.40)")
    if use_floor and cs.get("mode") == "percentile":
        patches.append("candidate_selection.use_absolute_score_floor: true  →  false")
    if tp_count <= 40:
        patches.append(f"backtest.turnover_penalty_trade_count: {tp_count}  →  60")
    if tp_weight > 0.3:
        patches.append(f"backtest.turnover_penalty_weight: {tp_weight:.2f}  →  0.15")

    return patches


def _yaml_diff(old_cfg: dict, new_cfg: dict) -> str:
    old_lines = yaml.dump(old_cfg, default_flow_style=False, sort_keys=True).splitlines(keepends=True)
    new_lines = yaml.dump(new_cfg, default_flow_style=False, sort_keys=True).splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile="config.yaml (current)", tofile="config.yaml (patched)"))
    return "".join(diff)


def _apply_recommended_patch(cfg: dict) -> dict:
    """Return a copy of cfg with critical+moderate issues fixed."""
    import copy
    c = copy.deepcopy(cfg)

    risk = c.setdefault("risk", {})
    min_idx = float(risk.get("min_index_pct", 0.60))
    if float(c.get("index_pct", 0.85)) < min_idx:
        c["index_pct"] = min_idx

    sw = c.setdefault("score_weights", {})
    if float(sw.get("income", 0.15)) > 0.12:
        sw["income"] = 0.10
        sw["momentum"] = round(float(sw.get("momentum", 0.30)) + (float(sw.get("income", 0.20)) - 0.10), 4)
        # renormalize
        total = sum(float(sw.get(k, 0)) for k in ["value", "quality", "income", "momentum"])
        if abs(total - 1.0) > 0.01:
            for k in ["value", "quality", "income", "momentum"]:
                sw[k] = round(float(sw.get(k, 0)) / total, 4)

    exit_d = c.setdefault("exit_decision", {})
    if float(exit_d.get("trim_profit_threshold", 0.08)) < 0.12:
        exit_d["trim_profit_threshold"] = 0.15
    if float(exit_d.get("harvest_profit_threshold", 0.15)) < 0.20:
        exit_d["harvest_profit_threshold"] = 0.25

    cs = c.setdefault("candidate_selection", {})
    if cs.get("mode") == "percentile" and cs.get("use_absolute_score_floor"):
        cs["use_absolute_score_floor"] = False
        cs["absolute_score_floor"] = 0.30

    bt = c.setdefault("backtest", {})
    if int(bt.get("turnover_penalty_trade_count", 80)) <= 40:
        bt["turnover_penalty_trade_count"] = 60
    if float(bt.get("turnover_penalty_weight", 1.0)) > 0.3:
        bt["turnover_penalty_weight"] = 0.15

    return c


# ---------------------------------------------------------------------------
# Severity badge helper
# ---------------------------------------------------------------------------

def _severity_badge(s: str) -> str:
    return {"CRITICAL": "🔴 CRITICAL", "MODERATE": "🟡 MODERATE", "LOW": "🟢 LOW"}.get(s, s)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.subheader("Config Diagnostics")
    st.caption(
        "Automated audit of config.yaml: contradictions, tuner-bound conflicts, "
        "backtest plumbing mismatches, and exit/harvest simulation gaps."
    )

    try:
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        st.error(f"Could not load config.yaml: {exc}")
        return

    findings = audit_config(cfg)

    # ── Summary row ──────────────────────────────────────────────────────
    n_crit = sum(1 for f in findings if f.severity == "CRITICAL")
    n_mod  = sum(1 for f in findings if f.severity == "MODERATE")
    n_low  = sum(1 for f in findings if f.severity == "LOW")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total findings", len(findings))
    c2.metric("🔴 Critical", n_crit, delta=None if n_crit == 0 else f"{n_crit} require action", delta_color="inverse")
    c3.metric("🟡 Moderate", n_mod)
    c4.metric("🟢 Low / Info", n_low)

    if n_crit == 0:
        st.success("No critical issues found.")
    else:
        st.error(f"{n_crit} critical issue(s) require attention before trusting auto-tune results.")

    st.divider()

    # ── Filter controls ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.multiselect(
            "Filter by severity",
            ["CRITICAL", "MODERATE", "LOW"],
            default=["CRITICAL", "MODERATE", "LOW"],
            key="diag_sev_filter",
        )
    with col2:
        cat_filter = st.multiselect(
            "Filter by category",
            sorted({f.category for f in findings}),
            default=sorted({f.category for f in findings}),
            key="diag_cat_filter",
        )

    visible = [f for f in findings if f.severity in sev_filter and f.category in cat_filter]

    # ── Findings table ────────────────────────────────────────────────────
    if not visible:
        st.info("No findings match current filters.")
    else:
        rows = []
        for f in visible:
            rows.append({
                "Severity":       _severity_badge(f.severity),
                "Category":       f.category,
                "Field":          f.field,
                "Current value":  str(f.current_value),
                "Problem summary": f.problem[:120] + ("…" if len(f.problem) > 120 else ""),
                "Recommendation": f.recommendation[:100] + ("…" if len(f.recommendation) > 100 else ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Detailed findings ─────────────────────────────────────────────────
    st.subheader("Finding Details")
    for f in visible:
        badge = _severity_badge(f.severity)
        with st.expander(f"{badge} — {f.category}: {f.field}", expanded=(f.severity == "CRITICAL")):
            st.markdown(f"**Current value:** `{f.current_value}`")
            st.markdown(f"**Problem:** {f.problem}")
            st.markdown(f"**Recommendation:** {f.recommendation}")
            if f.detail:
                st.info(f.detail)

    # ── Live vs backtest mismatch summary ─────────────────────────────────
    st.divider()
    st.subheader("Live vs Backtest Strategy Comparison")
    st.caption("These differences cause live performance to diverge from backtest even with identical config.")

    exit_d    = cfg.get("exit_decision", {})
    sell      = cfg.get("sell_rules", {})
    index_pct = float(cfg.get("index_pct", 0.85))

    rows = [
        {
            "Dimension":         "Take-profit",
            "Live behavior":     f"Trim at +{exit_d.get('trim_profit_threshold', 0.08):.0%}, harvest at +{exit_d.get('harvest_profit_threshold', 0.15):.0%}",
            "Backtest behavior": f"Full exit at +{sell.get('take_profit_pct', 0.60):.0%}",
            "Gap":               "CRITICAL — different strategies",
        },
        {
            "Dimension":         "Weekly contributions",
            "Live behavior":     f"Split by index_pct ({index_pct:.0%} ETF / {1-index_pct:.0%} stocks)",
            "Backtest behavior": "Now fixed — split by index_pct since 2026-05-25 patch",
            "Gap":               "FIXED",
        },
        {
            "Dimension":         "REVIEW / WATCH decisions",
            "Live behavior":     "Decision engine may hold, watch, or escalate to exit",
            "Backtest behavior": "No REVIEW/WATCH logic — binary buy/sell only",
            "Gap":               "MODERATE — live holds more positions than backtest",
        },
        {
            "Dimension":         "Harvest partial exit",
            "Live behavior":     f"Harvest {cfg.get('harvest', {}).get('profit_harvest_pct', 0.4):.0%} of position at threshold",
            "Backtest behavior": "No partial exits — only full exits",
            "Gap":               "MODERATE",
        },
        {
            "Dimension":         "Sentiment override",
            "Live behavior":     "High-confidence sell sentiment can force exits",
            "Backtest behavior": "No sentiment — quant signals only",
            "Gap":               "LOW — sentiment rarely overrides",
        },
        {
            "Dimension":         "Stop-loss value",
            "Live behavior":     f"sell_rules.stop_loss_pct = {sell.get('stop_loss_pct', -0.20):.0%}",
            "Backtest behavior": "Hardcoded _STOP_LOSS_PCT = -20% in backtest.py",
            "Gap":               "LOW — currently match, fragile if config changes",
        },
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tuner bound conflicts ─────────────────────────────────────────────
    st.divider()
    st.subheader("Tuner Bound Conflicts")
    st.caption("Parameters whose live values fall outside optimizer bounds — the tuner can never validate or suggest these values.")

    tn = cfg.get("tuning", {})
    tn_bounds = tn.get("parameter_bounds", {})
    sw = cfg.get("score_weights", {})
    frozen = set(tn.get("frozen_parameters", []))

    bound_rows = []
    param_map = {
        "score_weights.value":    float(sw.get("value", 0.05)),
        "score_weights.quality":  float(sw.get("quality", 0.45)),
        "score_weights.income":   float(sw.get("income", 0.20)),
        "score_weights.momentum": float(sw.get("momentum", 0.30)),
        "index_pct":              float(cfg.get("index_pct", 0.85)),
    }

    # Default tuner bounds (from tuner.py BOUNDS)
    default_bounds = {
        "score_weights.value":    (0.05, 0.80),
        "score_weights.quality":  (0.05, 0.60),
        "score_weights.income":   (0.00, 0.40),
        "score_weights.momentum": (0.00, 0.40),
        "index_pct":              (float(cfg.get("risk", {}).get("min_index_pct", 0.60)), 0.95),
    }

    for path, live_val in param_map.items():
        config_bound = tn_bounds.get(path, {})
        default_lo, default_hi = default_bounds.get(path, (0, 1))
        lo = float(config_bound.get("min", default_lo))
        hi = float(config_bound.get("max", default_hi))
        in_range = lo <= live_val <= hi
        is_frozen = path in frozen
        flag = "✅ OK" if in_range else ("🔒 Frozen (moot)" if is_frozen else "❌ Out of bounds")
        bound_rows.append({
            "Parameter":   path,
            "Live value":  f"{live_val:.3f}",
            "Bound min":   f"{lo:.3f}",
            "Bound max":   f"{hi:.3f}",
            "In range":    flag,
            "Frozen":      "Yes" if is_frozen else "No",
        })

    st.dataframe(pd.DataFrame(bound_rows), use_container_width=True, hide_index=True)

    # ── Recommended patch ────────────────────────────────────────────────
    st.divider()
    st.subheader("Recommended Patch")
    st.caption(
        "Minimal diff to resolve CRITICAL and MODERATE issues. "
        "Review carefully — this does not auto-apply without your confirmation."
    )

    patches = _recommended_patch_lines(cfg)
    if not patches:
        st.success("No critical/moderate patches needed — config is in good shape.")
        return

    for p in patches:
        st.code(p, language="yaml")

    patched_cfg = _apply_recommended_patch(cfg)
    diff_text   = _yaml_diff(cfg, patched_cfg)

    with st.expander("Full YAML diff"):
        st.code(diff_text, language="diff")

    copy_col, apply_col = st.columns(2)
    with copy_col:
        st.download_button(
            "⬇ Download patched config.yaml",
            data=yaml.dump(patched_cfg, default_flow_style=False, sort_keys=False),
            file_name="config_patched.yaml",
            mime="text/yaml",
        )

    ui_cfg = cfg.get("ui", {})
    allow_write = bool(ui_cfg.get("allow_config_writes", False))

    with apply_col:
        if not allow_write:
            st.info("Set `ui.allow_config_writes: true` in config.yaml to enable direct apply.")
        else:
            confirm = st.text_input(
                "Type APPLY to write patched config.yaml",
                key="diag_apply_confirm",
                placeholder="APPLY",
            )
            if st.button("Apply patch to config.yaml", key="diag_apply_btn", type="primary"):
                if confirm.strip().upper() != "APPLY":
                    st.error("Type APPLY in the text box to confirm.")
                else:
                    try:
                        with open(CFG_PATH, "w") as fh:
                            yaml.dump(patched_cfg, fh, default_flow_style=False, sort_keys=False)
                        st.success("config.yaml updated. Restart the app to reload util.py constants.")
                    except Exception as exc:
                        st.error(f"Write failed: {exc}")
