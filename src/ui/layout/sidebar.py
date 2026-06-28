"""ui/layout/sidebar.py — Section navigation and sidebar status panel."""
from __future__ import annotations

import streamlit as st

SECTIONS = [
    ("⚡", "Operations",  "operations"),
    ("💼", "Portfolio",   "portfolio"),
    ("🔬", "Research",    "research"),
    ("✅", "Validation",  "validation"),
    ("🎰", "0DTE",        "odte"),
    ("⚙️",  "System",     "system"),
]

def render_sidebar() -> str:
    """Render sidebar; return selected section key."""
    with st.sidebar:
        st.title("📈 Daily Investor")
        st.caption("Quant research & execution platform")
        st.divider()

        labels = [f"{icon} {name}" for icon, name, _ in SECTIONS]
        key_map = {f"{icon} {name}": key for icon, name, key in SECTIONS}

        selected_label = st.radio(
            "Section", labels, label_visibility="collapsed", key="nav_section"
        )
        selected = key_map.get(selected_label, "operations")

        st.divider()
        _render_status_panel()
        st.divider()
        _render_live_toggle()

    return selected


def _render_status_panel() -> None:
    """Lightweight status strip — no heavy computation."""
    from ui.utils import DATA_DIR, latest_csv_path, load_config_raw, load_latest_csv

    # Data freshness
    snap_path = latest_csv_path("agg_data")
    if snap_path:
        date_str = snap_path.stem.split("_", 1)[-1].replace("_", "-")
        st.caption(f"📊 Data: {date_str}")
    else:
        st.caption("📊 Data: none")

    # Universe size
    agg = load_latest_csv("agg_data")
    if agg is not None and not agg.empty:
        st.caption(f"📦 Universe: {len(agg):,} stocks")

    # Snapshot count
    snap_dir = DATA_DIR / "snapshots"
    if snap_dir.exists():
        n_snaps = len(list(snap_dir.glob("*.parquet")))
        st.caption(f"🗂️ Snapshots: {n_snaps}")

    # Regime
    cfg = load_config_raw()
    vix_def = cfg.get("regime", {}).get("vix_defensive_threshold", 30)
    vix_neu = cfg.get("regime", {}).get("vix_neutral_threshold", 20)
    st.caption(f"🌡️ VIX gate: {vix_neu}↔{vix_def}")


def _render_live_toggle() -> None:
    from ui.utils import ui_config
    if "live_enabled" not in st.session_state:
        st.session_state.live_enabled = False
    _ui_cfg = ui_config()
    if _ui_cfg.get("allow_live_execution"):
        st.session_state.live_enabled = st.toggle(
            "🔴 Live execution",
            value=st.session_state.live_enabled,
            help="Enable live order placement. Disabled by default.",
        )
    else:
        st.session_state.live_enabled = False
        st.caption("🔒 Live execution locked (config)")
    st.caption(f"Live: {'✅ ON' if st.session_state.live_enabled else '🔒 OFF'}")
