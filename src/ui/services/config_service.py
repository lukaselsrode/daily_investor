"""
ui/services/config_service.py — Config access and audit for UI.

Thin wrapper so components never parse YAML directly or import from util.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st
import yaml

_ROOT = Path(__file__).resolve().parents[4]
_CFG_DIR = _ROOT / "cfg"


@st.cache_data(ttl=60)
def load_config(filename: str = "config.yaml") -> Optional[dict]:
    p = _CFG_DIR / filename
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def get_section(cfg: dict, *keys: str, default: Any = None) -> Any:
    d = cfg
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def list_config_files() -> list[tuple[str, str]]:
    """Return [(label, filename)] for all named config files that exist."""
    named = [
        ("Current (config.yaml)",     "config.yaml"),
        ("Baseline snapshot",         "config_baseline_current.yaml"),
        ("Research safe",             "config_research_safe.yaml"),
        ("Momentum anchor",           "config_momentum_anchor.yaml"),
        ("Quality anchor",            "config_quality_anchor.yaml"),
    ]
    return [(lbl, fname) for lbl, fname in named if (_CFG_DIR / fname).exists()]


def run_audit(cfg: Optional[dict] = None) -> list:
    """Run config audit and return list of Finding objects."""
    try:
        from ui.components.config_diagnostics import audit_config
        return audit_config(cfg or load_config() or {})
    except Exception:
        return []
