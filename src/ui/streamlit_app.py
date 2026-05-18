"""
ui/streamlit_app.py — Daily Investor dashboard entry point.

Launch:
    streamlit run src/ui/streamlit_app.py
"""
from __future__ import annotations

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

from ui.layout.sidebar import render_sidebar
from ui.sections import operations, portfolio, research, validation, system as system_page

_SECTION_MAP = {
    "operations": operations.render,
    "portfolio":  portfolio.render,
    "research":   research.render,
    "validation": validation.render,
    "system":     system_page.render,
}

section = render_sidebar()
try:
    _SECTION_MAP[section]()
except Exception as exc:
    st.error(f"Failed to load section `{section}`: {exc}")
    st.exception(exc)
