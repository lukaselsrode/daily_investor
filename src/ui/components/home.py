"""
ui/components/home.py — System status dashboard.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui.utils import (
    DATA_DIR, CFG_PATH, LOG_PATH, ROOT,
    data_date, load_config_raw, load_latest_csv, ui_config,
)


def _file_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600


def _freshness_color(hours: float | None) -> str:
    if hours is None:
        return "red"
    if hours < 25:
        return "green"
    if hours < 50:
        return "orange"
    return "red"


def _metric_card(col, label: str, value: str, color: str = "green") -> None:
    with col:
        icon = {"green": "✅", "orange": "⚠️", "red": "❌"}.get(color, "ℹ️")
        st.metric(label, f"{icon} {value}")


def render() -> None:
    st.title("📊 System Status")
    cfg = load_config_raw()
    ui_cfg = ui_config()

    # ----- Config status --------------------------------------------------
    st.subheader("Configuration")
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "Config file", "Found" if CFG_PATH.exists() else "MISSING",
                 "green" if CFG_PATH.exists() else "red")
    auto_approve = cfg.get("auto_approve", False)
    _metric_card(c2, "Auto-approve", "ON" if auto_approve else "off",
                 "orange" if auto_approve else "green")
    sentiment = cfg.get("use_sentiment_analysis", True)
    _metric_card(c3, "Sentiment", "enabled" if sentiment else "disabled",
                 "green" if sentiment else "orange")
    live = st.session_state.get("live_enabled", False)
    _metric_card(c4, "Live execution", "ENABLED" if live else "locked off",
                 "orange" if live else "green")

    st.divider()

    # ----- Data freshness --------------------------------------------------
    st.subheader("Data freshness")
    prefixes = ["agg_data", "robinhood_data", "news"]
    cols = st.columns(len(prefixes))
    for col, prefix in zip(cols, prefixes):
        latest = sorted(DATA_DIR.glob(f"{prefix}_*.csv"))
        if latest:
            p = latest[-1]
            age = _file_age_hours(p)
            color = _freshness_color(age)
            age_str = f"{age:.0f}h ago" if age is not None else "?"
            _metric_card(col, prefix, f"{p.stem.split('_',1)[1]} ({age_str})", color)
        else:
            _metric_card(col, prefix, "No data", "red")

    st.divider()

    # ----- Key config values -----------------------------------------------
    st.subheader("Active strategy config")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("metric_threshold", cfg.get("metric_threshold", "—"))
        st.metric("index_pct", cfg.get("index_pct", "—"))
    with c2:
        sw = cfg.get("score_weights", {})
        st.metric("quality weight", sw.get("quality", "—"))
        st.metric("momentum weight", sw.get("momentum", "—"))
    with c3:
        sr = cfg.get("sell_rules", {})
        st.metric("stop_loss_pct", sr.get("stop_loss_pct", "—"))
        st.metric("trailing_stop_pct", sr.get("trailing_stop_pct", "—"))
    with c4:
        st.metric("weekly_investment", f"${cfg.get('weekly_investment', '—')}")
        regime_cfg = cfg.get("regime", {})
        st.metric("VIX defensive threshold", regime_cfg.get("vix_defensive_threshold", "—"))

    st.divider()

    # ----- Scored universe snapshot ----------------------------------------
    st.subheader("Latest scored universe snapshot")
    df = load_latest_csv("agg_data")
    if df is None:
        st.info("No agg_data found. Run `daily-investor run` or `daily-investor run --skip-data` first.")
    else:
        thresh = cfg.get("metric_threshold", 0.0)
        buy_candidates = df[df["value_metric"] >= thresh] if "value_metric" in df.columns else df.head(0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe size", len(df))
        c2.metric("Buy candidates", len(buy_candidates))
        c3.metric("Yield traps", int(df["yield_trap_flag"].sum()) if "yield_trap_flag" in df.columns else "—")
        if "value_metric" in df.columns:
            c4.metric("Avg value_metric", f"{df['value_metric'].mean():.3f}")

        st.caption(f"Source: {data_date('agg_data')}")

    st.divider()

    # ----- Log tail --------------------------------------------------------
    st.subheader("Recent log activity")
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-30:])
        st.code(tail, language="text")
    else:
        st.info("No log file found.")
