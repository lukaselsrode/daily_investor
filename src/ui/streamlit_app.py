"""
ui/streamlit_app.py — Daily Investor dashboard entry point.

Launch:
    streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

st.set_page_config(
    page_title="Daily Investor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    ("📊", "Home",                  "home"),
    ("🚀", "Run Control",           "run_control"),
    ("💼", "Portfolio",             "portfolio"),
    ("🎯", "Order Intents",         "intents"),
    ("🔬", "Scoring Explorer",      "scoring"),
    ("📐", "Value Diagnostics",    "value_diagnostics"),
    ("🔗", "Factor Orthogonalization", "factor_analysis"),
    ("📡", "Rolling IC",             "rolling_ic"),
    ("🧪", "Factor Lab",            "factor_lab"),
    ("📈", "Backtests",             "backtests"),
    ("⚙️",  "Auto-Tune",            "tuning"),
    ("🔭", "Stability & Robustness","stability"),
    ("🌡️",  "Regime & Risk",        "regime"),
    ("⚖️",  "Exposure",            "exposure"),
    ("🩺", "Reliability Diag.",     "reliability"),
    ("🗂️",  "Data Explorer",        "data_explorer"),
    ("📋", "Logs / Audit",          "logs"),
    ("🛠️",  "Config Viewer",        "config_viewer"),
]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 Daily Investor")
    st.caption("Portfolio control panel")
    st.divider()

    labels   = [f"{icon} {name}" for icon, name, _ in PAGES]
    modules  = {f"{icon} {name}": mod for icon, name, mod in PAGES}
    selected = st.radio("Navigate", labels, label_visibility="collapsed")
    st.divider()

    # Global safety toggle
    if "live_enabled" not in st.session_state:
        st.session_state.live_enabled = False

    from ui.utils import ui_config
    _ui_cfg = ui_config()
    if _ui_cfg.get("allow_live_execution"):
        st.session_state.live_enabled = st.toggle(
            "🔴 Live execution",
            value=st.session_state.live_enabled,
            help="Enable live order placement. Disabled by default.",
        )
    else:
        st.session_state.live_enabled = False
        st.caption("🔒 Live execution locked off in config")

    st.caption(
        f"Live: {'✅ ON' if st.session_state.live_enabled else '🔒 OFF'}"
    )

# ---------------------------------------------------------------------------
# Render selected page
# ---------------------------------------------------------------------------

mod_name = modules.get(selected, "home")
try:
    mod = importlib.import_module(f"ui.components.{mod_name}")
    mod.render()
except Exception as exc:
    st.error(f"Failed to load page `{mod_name}`: {exc}")
    st.exception(exc)
