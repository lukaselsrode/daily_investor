"""
ui/components/decision_diagnostics.py — Per-position Decision Diagnostics widget.

Shows for every active position:
  - Raw sell trigger (what fired before any adjustment)
  - Final decision after downgrades
  - Downgrade reason
  - Hard exit / Review / Trim-Harvest flags
  - Thesis intact score, premature exit probability, exit confidence
  - Signal deterioration velocity, rank decay
  - Decision badge (SAFE EXIT / RISKY EXIT / REVIEW NEEDED / TRIM / HARVEST)

Pure display — no feedback to factor engine.
"""

from __future__ import annotations

from typing import Optional
import streamlit as st


# ---------------------------------------------------------------------------
# Badge config
# ---------------------------------------------------------------------------

_BADGE_STYLE: dict[str, tuple[str, str]] = {
    "SAFE EXIT":     ("#27ae60", "✓ SAFE EXIT"),
    "RISKY EXIT":    ("#e67e22", "⚠ RISKY EXIT"),
    "REVIEW NEEDED": ("#8e44ad", "🔎 REVIEW NEEDED"),
    "TRIM":          ("#e67e22", "🔶 TRIM"),
    "HARVEST":       ("#1abc9c", "💰 HARVEST"),
}


def _badge(text: str) -> None:
    color, label = _BADGE_STYLE.get(text, ("#7f8c8d", text))
    st.markdown(
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">{label}</span>',
        unsafe_allow_html=True,
    )


def _flag_pill(label: str, active: bool, active_color: str) -> str:
    color = active_color if active else "#bdc3c7"
    check = "✓" if active else "✗"
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:11px;margin-right:4px;">{check} {label}</span>'
    )


def _mini_bar(label: str, value: float, low_good: bool = False) -> None:
    clamped = max(0.0, min(1.0, value))
    if low_good:
        color = "#27ae60" if clamped < 0.35 else ("#e67e22" if clamped < 0.60 else "#e74c3c")
    else:
        color = "#e74c3c" if clamped < 0.35 else ("#f39c12" if clamped < 0.65 else "#27ae60")

    st.markdown(
        f'<div style="margin:4px 0;">'
        f'<div style="font-size:11px;color:#7f8c8d;">{label}</div>'
        f'<div style="background:#ecf0f1;border-radius:6px;height:8px;">'
        f'<div style="background:{color};border-radius:6px;height:8px;width:{clamped*100:.0f}%;"></div></div>'
        f'<div style="font-size:11px;text-align:right;color:{color};font-weight:600;">{clamped:.0%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_decision_diagnostics(
    decision_output,
    key_prefix: str = "",
) -> None:
    """
    Render Decision Diagnostics for one position.

    Parameters
    ----------
    decision_output : DecisionOutput or None
    key_prefix      : unique key prefix (use ticker symbol)
    """
    if decision_output is None:
        st.caption("No decision diagnostics available.")
        return

    action   = getattr(decision_output, "action",   "—")
    conf     = getattr(decision_output, "confidence", "—")
    pep      = getattr(decision_output, "premature_exit_probability", None)
    tis      = getattr(decision_output, "thesis_intact_score", None)
    sdv      = getattr(decision_output, "signal_deterioration_velocity", None)
    rdec     = getattr(decision_output, "rank_decay", None)
    badges   = getattr(decision_output, "badges", [])
    adj      = getattr(decision_output, "adjustment_applied", False)
    adj_rsn  = getattr(decision_output, "adjustment_reason", "")
    raw_trig = getattr(decision_output, "raw_sell_trigger", "")
    is_hard  = getattr(decision_output, "is_hard_exit", False)
    is_rev   = getattr(decision_output, "is_review", False)
    is_th    = getattr(decision_output, "is_trim_harvest", False)
    diag_sum = getattr(decision_output, "diagnostic_summary", "")

    # ── Required output: raw trigger + final decision + flags ─────────────────
    st.markdown("**Decision Diagnostics**")

    # Raw trigger
    if raw_trig:
        st.markdown(
            f'<div style="font-size:12px;color:#7f8c8d;">Raw sell trigger: '
            f'<span style="color:#e74c3c;">{raw_trig}</span></div>',
            unsafe_allow_html=True,
        )

    # Final decision line
    state_color = {
        "EXIT": "#e74c3c", "REVIEW": "#9b59b6", "TRIM": "#e67e22",
        "HARVEST": "#1abc9c", "WATCH": "#f39c12", "HOLD": "#3498db",
    }.get(action, "#7f8c8d")
    st.markdown(
        f'<div style="font-size:13px;margin:4px 0;">Final decision: '
        f'<strong style="color:{state_color};">{action}</strong>'
        f'{"  →  adjusted from raw EXIT" if adj else ""}</div>',
        unsafe_allow_html=True,
    )

    # Flags row
    flags_html = (
        _flag_pill("HARD EXIT",    is_hard, "#e74c3c") +
        _flag_pill("REVIEW",       is_rev,  "#9b59b6") +
        _flag_pill("TRIM/HARVEST", is_th,   "#1abc9c")
    )
    st.markdown(f'<div style="margin:6px 0;">{flags_html}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Badges ────────────────────────────────────────────────────────────────
    if badges:
        cols = st.columns(len(badges))
        for col, b in zip(cols, badges):
            with col:
                _badge(b)
        st.markdown("")

    # ── Key metrics ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        if tis is not None:
            _mini_bar("Thesis Intact Score", tis, low_good=False)
    with c2:
        if pep is not None:
            _mini_bar("Premature Exit Risk", pep, low_good=True)
    with c3:
        conf_color = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}.get(conf, "#7f8c8d")
        st.markdown(
            f'<div style="font-size:11px;color:#7f8c8d;">Exit Confidence</div>'
            f'<div style="font-size:20px;font-weight:700;color:{conf_color};">{conf}</div>',
            unsafe_allow_html=True,
        )

    # ── Velocity + rank decay ─────────────────────────────────────────────────
    c4, c5 = st.columns(2)
    with c4:
        if sdv is not None:
            color = "#e74c3c" if sdv < -0.003 else ("#f39c12" if sdv < 0 else "#27ae60")
            sign  = "+" if sdv > 0 else ""
            st.markdown(
                f'<div style="font-size:11px;color:#7f8c8d;">Signal Deterioration Velocity</div>'
                f'<div style="font-size:13px;font-weight:600;color:{color};">{sign}{sdv:.5f} / day</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Signal velocity: N/A")
    with c5:
        if rdec is not None:
            color = "#e74c3c" if rdec > 0.20 else ("#f39c12" if rdec > 0.05 else "#27ae60")
            sign  = "+" if rdec > 0 else ""
            st.markdown(
                f'<div style="font-size:11px;color:#7f8c8d;">Rank Decay (since buy)</div>'
                f'<div style="font-size:13px;font-weight:600;color:{color};">{sign}{rdec:.1%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Rank decay: N/A")

    # ── Downgrade reason ──────────────────────────────────────────────────────
    if adj and adj_rsn:
        st.info(f"**Downgrade reason:** {adj_rsn}", icon="🔄")
    elif diag_sum:
        st.caption(diag_sum)
