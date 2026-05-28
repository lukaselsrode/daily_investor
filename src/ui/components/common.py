"""
ui/components/common.py — Shared UI utilities.

Import these helpers instead of repeating display patterns in every component.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Metric display
# ---------------------------------------------------------------------------

def metric_row(metrics: list[tuple[str, Any, str | None]], cols: int = 4) -> None:
    """Render a row of st.metric cards. Each tuple: (label, value, delta)."""
    chunks = [metrics[i : i + cols] for i in range(0, len(metrics), cols)]
    for chunk in chunks:
        columns = st.columns(len(chunk))
        for col, (label, value, delta) in zip(columns, chunk):
            col.metric(label, value, delta)


def pct_metric(label: str, val: float, ref: float = 0.0, col=None) -> None:
    """Show a percentage metric with delta vs ref."""
    target = col or st
    target.metric(label, f"{val:+.1%}", f"{val - ref:+.1%}" if ref else None)


# ---------------------------------------------------------------------------
# Status badges
# ---------------------------------------------------------------------------

_BADGE_COLOURS = {
    "ok":      ("#27ae60", "✓"),
    "warn":    ("#e67e22", "⚠"),
    "error":   ("#e74c3c", "✗"),
    "info":    ("#2980b9", "ℹ"),
    "neutral": ("#7f8c8d", "·"),
}


def status_badge(text: str, level: str = "info") -> None:
    colour, icon = _BADGE_COLOURS.get(level, _BADGE_COLOURS["neutral"])
    st.markdown(
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">{icon} {text}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Warning / info banners
# ---------------------------------------------------------------------------

def warn_banner(msg: str) -> None:
    st.warning(f"⚠️ {msg}")


def info_banner(msg: str) -> None:
    st.info(f"ℹ️ {msg}")


def error_banner(msg: str) -> None:
    st.error(f"🚫 {msg}")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def empty_state(title: str, hint: str = "") -> None:
    st.markdown(
        f"<div style='text-align:center;padding:2rem;color:#888'>"
        f"<div style='font-size:2rem'>📭</div>"
        f"<div style='font-weight:600;margin-top:0.5rem'>{title}</div>"
        f"<div style='font-size:0.85rem;margin-top:0.25rem'>{hint}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def df_download(df: pd.DataFrame, filename: str, label: str = "⬇ Download CSV") -> None:
    """Add a download button for a DataFrame."""
    st.download_button(label, df.to_csv(index=False), file_name=filename, mime="text/csv")


def styled_df(
    df: pd.DataFrame,
    pct_cols: list[str] | None = None,
    int_cols: list[str] | None = None,
    highlight_neg: list[str] | None = None,
) -> Any:
    """Return a Styler with common formatting rules applied."""
    fmt: dict[str, str] = {}
    if pct_cols:
        for c in pct_cols:
            if c in df.columns:
                fmt[c] = "{:+.1%}"
    if int_cols:
        for c in int_cols:
            if c in df.columns:
                fmt[c] = "{:,.0f}"

    styler = df.style.format(fmt, na_rep="—")
    if highlight_neg:
        for c in highlight_neg:
            if c in df.columns:
                styler = styler.applymap(
                    lambda v: "color:#e74c3c" if isinstance(v, float) and v < 0 else "",
                    subset=[c],
                )
    return styler


# ---------------------------------------------------------------------------
# Config diff viewer
# ---------------------------------------------------------------------------

def yaml_diff_viewer(diff_text: str) -> None:
    """Render a unified diff with syntax-like colouring."""
    lines = diff_text.splitlines()
    rendered = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            rendered.append(f'<span style="color:#27ae60">{line}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            rendered.append(f'<span style="color:#e74c3c">{line}</span>')
        elif line.startswith("@@"):
            rendered.append(f'<span style="color:#2980b9">{line}</span>')
        else:
            rendered.append(line)
    st.markdown(
        "<pre style='background:#1e1e1e;color:#d4d4d4;padding:1rem;"
        "border-radius:4px;overflow:auto;font-size:12px'>"
        + "\n".join(rendered)
        + "</pre>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Run command preview
# ---------------------------------------------------------------------------

def cmd_preview(cmd: str, note: str = "") -> None:
    st.code(cmd, language="bash")
    if note:
        st.caption(note)


# ---------------------------------------------------------------------------
# Section divider with label
# ---------------------------------------------------------------------------

def section(title: str, caption: str = "") -> None:
    st.divider()
    st.subheader(title)
    if caption:
        st.caption(caption)


# ---------------------------------------------------------------------------
# Freshness label
# ---------------------------------------------------------------------------

def freshness_label(path: Path | None) -> str:
    """Return a human-readable freshness string from a dated filename."""
    if path is None:
        return "no data"
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 4:
        try:
            return f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
        except Exception:
            pass
    return stem
