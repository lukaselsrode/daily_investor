"""
ui/components/config_viewer.py — Parsed config.yaml viewer.
Read-only unless ui.allow_config_writes is true.
"""

from __future__ import annotations

import streamlit as st
import yaml

from ui.utils import CFG_PATH, load_config_raw, ui_config


_SECTIONS = [
    ("scoring",           "Scoring & weights"),
    ("score_weights",     "Score weights"),
    ("momentum_v2",       "Momentum v2 sub-weights"),
    ("sell_rules",        "Sell rules"),
    ("risk",              "Risk limits"),
    ("harvest",           "Profit harvesting"),
    ("regime",            "Market regime"),
    ("etf_risk",          "ETF risk"),
    ("backtest",          "Backtest / tuner"),
    ("tuning",            "Tuning frozen params"),
    ("reliability",       "Reliability gating"),
    ("ui",                "UI settings"),
]


def render() -> None:
    st.title("🛠️ Config Viewer")
    st.caption(f"Source: `{CFG_PATH}`")

    if not CFG_PATH.exists():
        st.error(f"Config file not found: `{CFG_PATH}`")
        return

    cfg = load_config_raw()
    ui_cfg = ui_config()

    # ---- Top-level scalars -----------------------------------------------
    st.subheader("Top-level settings")
    scalars = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    if scalars:
        c1, c2, c3, c4 = st.columns(4)
        items = list(scalars.items())
        per_col = max(1, (len(items) + 3) // 4)
        for i, col in enumerate([c1, c2, c3, c4]):
            for k, v in items[i * per_col:(i + 1) * per_col]:
                col.metric(k, str(v))

    # ETFs
    if "etfs" in cfg:
        st.metric("etfs", ", ".join(cfg["etfs"]))

    st.divider()

    # ---- Sections --------------------------------------------------------
    for section_key, section_label in _SECTIONS:
        data = cfg.get(section_key)
        if data is None:
            continue
        with st.expander(section_label):
            if isinstance(data, dict):
                _render_dict(data)
            else:
                st.code(yaml.dump({section_key: data}, default_flow_style=False), language="yaml")

    # ---- Raw YAML --------------------------------------------------------
    with st.expander("Full raw config.yaml"):
        st.code(CFG_PATH.read_text(), language="yaml")

    # ---- Download --------------------------------------------------------
    st.download_button(
        "⬇ Download config.yaml",
        data=CFG_PATH.read_text(),
        file_name="config.yaml",
        mime="text/yaml",
    )

    # ---- Optional write --------------------------------------------------
    if ui_cfg.get("allow_config_writes"):
        st.divider()
        st.subheader("Edit config")
        st.warning("Config editing is available but not implemented in this version. Use the auto-tune page to write optimized parameters.")


def _render_dict(d: dict, prefix: str = "") -> None:
    flat = {}
    nested = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested[k] = v
        else:
            flat[k] = v

    if flat:
        rows = list(flat.items())
        cols = st.columns(min(4, len(rows)))
        for i, (k, v) in enumerate(rows):
            cols[i % len(cols)].metric(f"{prefix}{k}", str(v))

    for k, v in nested.items():
        st.markdown(f"**{k}**")
        _render_dict(v, prefix=f"{k}.")
