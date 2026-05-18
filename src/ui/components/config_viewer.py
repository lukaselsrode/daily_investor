"""
ui/components/config_viewer.py — Interactive config.yaml viewer and editor.
"""

from __future__ import annotations

import streamlit as st
import yaml

from ui.utils import CFG_PATH, load_config_raw, ui_config


_SECTIONS = [
    ("scoring",              "Scoring & Weights"),
    ("score_weights",        "Score Weights"),
    ("valuation_guardrails", "Valuation Guardrails"),
    ("momentum",             "Momentum v1"),
    ("momentum_v2",          "Momentum v2 Sub-Weights"),
    ("sell_rules",           "Sell Rules"),
    ("risk",                 "Risk Limits"),
    ("harvest",              "Profit Harvesting"),
    ("regime",               "Market Regime"),
    ("bear_market",          "Bear Market"),
    ("etf_risk",             "ETF Risk"),
    ("analyst_ratings",      "Analyst Ratings"),
    ("backtest",             "Backtest / Tuner"),
    ("tuning",               "Tuning Frozen Params"),
    ("reliability",          "Reliability Gating"),
    ("stability",            "Stability"),
    ("candidate_rotation",   "Candidate Rotation"),
    ("value_v2",             "Value v2"),
    ("snapshots",            "Snapshots"),
    ("dividends",            "Dividends"),
    ("research",             "Research"),
    ("earnings",             "Earnings"),
    ("ui",                   "UI Settings"),
]

_PILL_TRUE  = '<span style="background:#145a32;color:#abebc6;padding:2px 11px;border-radius:10px;font-size:0.8em;font-weight:700;letter-spacing:.03em">YES</span>'
_PILL_FALSE = '<span style="background:#641e16;color:#f1948a;padding:2px 11px;border-radius:10px;font-size:0.8em;font-weight:700;letter-spacing:.03em">NO</span>'
_TAG_STYLE  = "background:#1c2b3a;color:#7fb3d3;padding:2px 8px;border-radius:6px;font-size:0.78em;margin:2px 3px 2px 0;display:inline-block"
_LABEL_STYLE = "color:#6c757d;font-size:0.76em;display:block;margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em"
_SUB_HEADER = "color:#8e9aaa;font-size:0.8em;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin:14px 0 6px 0"


def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return _PILL_TRUE if v else _PILL_FALSE
    if isinstance(v, float):
        return f'<code style="font-size:0.88em;color:#a3c4e8">{v:g}</code>'
    if isinstance(v, int):
        return f'<code style="font-size:0.88em;color:#a3c4e8">{v}</code>'
    if isinstance(v, list):
        if not v:
            return '<em style="color:#555;font-size:0.85em">empty</em>'
        return "".join(f'<span style="{_TAG_STYLE}">{item}</span>' for item in v)
    return f'<code style="font-size:0.88em;color:#c8d6e5">{v}</code>'


def _kv_card(label: str, value) -> str:
    return (
        f'<div style="padding:8px 0 4px 0">'
        f'<span style="{_LABEL_STYLE}">{label.replace("_", " ")}</span>'
        f'{_fmt_value(value)}'
        f'</div>'
    )


def _render_readonly(data: dict, prefix: str = "") -> None:
    flat   = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in data.items() if isinstance(v, dict)}

    if flat:
        items = list(flat.items())
        ncols = min(4, len(items))
        cols  = st.columns(ncols)
        for i, (k, v) in enumerate(items):
            cols[i % ncols].markdown(_kv_card(k, v), unsafe_allow_html=True)

    for k, v in nested.items():
        st.markdown(f'<div style="{_SUB_HEADER}">{k}</div>', unsafe_allow_html=True)
        _render_readonly(v, prefix=f"{prefix}{k}.")


def _safe_cast(s: str, typ):
    try:
        return typ(s)
    except (ValueError, TypeError):
        return s


def _detect_numeric_type(lst: list):
    """Return float, int, or None by inspecting the first element — handles string-encoded numbers."""
    if not lst:
        return None
    first = lst[0]
    if isinstance(first, bool):
        return None
    if isinstance(first, float):
        return float
    if isinstance(first, int):
        return int
    if isinstance(first, str):
        s = first.strip()
        try:
            if "." not in s:
                int(s)
                return int
        except ValueError:
            pass
        try:
            float(s)
            return float
        except ValueError:
            pass
    return None


def _edit_widget(section_key: str, path: str, label: str, value):
    key = f"cv_{section_key}_{path}"
    if value is None:
        raw = st.text_input(label, value="", key=key)
        return None if not raw.strip() else raw.strip()
    if isinstance(value, bool):
        return st.toggle(label, value=value, key=key)
    if isinstance(value, int):
        return st.number_input(label, value=value, step=1, key=key)
    if isinstance(value, float):
        step = 0.001 if abs(value) < 1 else (0.01 if abs(value) < 10 else 1.0)
        return st.number_input(label, value=value, step=step, format="%.4g", key=key)
    if isinstance(value, list):
        raw = st.text_input(label, value=", ".join(str(x) for x in value), key=key)
        items = [x.strip() for x in raw.split(",") if x.strip()]
        numeric_type = _detect_numeric_type(value)
        if numeric_type is not None:
            return [_safe_cast(x, numeric_type) for x in items]
        return items
    return st.text_input(label, value=str(value), key=key)


def _render_editable(section_key: str, data: dict, edits: dict, prefix: str = "") -> None:
    flat   = {k: v for k, v in data.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in data.items() if isinstance(v, dict)}

    if flat:
        items = list(flat.items())
        ncols = min(3, len(items))
        cols  = st.columns(ncols)
        for i, (k, v) in enumerate(items):
            path = f"{prefix}{k}"
            label = k.replace("_", " ").title()
            with cols[i % ncols]:
                edits[path] = _edit_widget(section_key, path, label, v)

    for k, v in nested.items():
        st.markdown(f'<div style="{_SUB_HEADER}">{k}</div>', unsafe_allow_html=True)
        _render_editable(section_key, v, edits, prefix=f"{prefix}{k}.")


def _apply_edits(data: dict, edits: dict, prefix: str = "") -> dict:
    result = {}
    for k, v in data.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            result[k] = _apply_edits(v, edits, prefix=f"{path}.")
        else:
            result[k] = edits.get(path, v)
    return result


def render() -> None:
    st.title("🛠️ Config")
    st.caption(f"`{CFG_PATH}`")

    if not CFG_PATH.exists():
        st.error(f"Config file not found: `{CFG_PATH}`")
        return

    cfg       = load_config_raw()
    ui_cfg    = ui_config()
    allow_write = ui_cfg.get("allow_config_writes", False)

    # ---- Mode toggle -------------------------------------------------------
    if allow_write:
        col_a, col_b = st.columns([3, 1])
        with col_b:
            edit_mode = st.toggle("Edit mode", value=False, key="cv_edit_mode")
    else:
        edit_mode = False
        st.info("🔒 Read-only. Set `ui.allow_config_writes: true` in `config.yaml` to enable editing.")

    st.divider()

    # ---- Top-level scalars -------------------------------------------------
    scalars = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    top_level_edits: dict = {}
    if scalars:
        st.markdown("**Top-level**")
        if edit_mode:
            items = list(scalars.items())
            ncols = min(3, len(items))
            cols  = st.columns(ncols)
            for i, (k, v) in enumerate(items):
                with cols[i % ncols]:
                    top_level_edits[k] = _edit_widget("toplevel", k, k.replace("_", " ").title(), v)
        else:
            items = list(scalars.items())
            cols  = st.columns(min(5, len(items)))
            for i, (k, v) in enumerate(items):
                cols[i % len(cols)].markdown(_kv_card(k, v), unsafe_allow_html=True)

    # ---- ETFs list ---------------------------------------------------------
    etfs_edit = None
    if "etfs" in cfg:
        if edit_mode:
            raw = st.text_input(
                "ETFs (comma-separated)",
                value=", ".join(str(x) for x in cfg["etfs"]),
                key="cv_toplevel_etfs",
            )
            etfs_edit = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            st.markdown(
                f'<div style="margin-top:8px"><span style="{_LABEL_STYLE}">etfs</span>'
                + _fmt_value(cfg["etfs"])
                + "</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ---- Sections ----------------------------------------------------------
    all_edits: dict[str, dict] = {}

    for section_key, section_label in _SECTIONS:
        data = cfg.get(section_key)
        if data is None:
            continue

        with st.expander(section_label, expanded=False):
            if not isinstance(data, dict):
                st.code(yaml.dump({section_key: data}, default_flow_style=False), language="yaml")
                continue

            if edit_mode:
                edits: dict = {}
                _render_editable(section_key, data, edits)
                all_edits[section_key] = edits
            else:
                _render_readonly(data)

    # ---- Save --------------------------------------------------------------
    if edit_mode:
        st.divider()
        if st.button("💾 Save to config.yaml", type="primary", key="cv_save"):
            new_cfg = dict(cfg)
            # Apply top-level scalar edits
            for k, v in top_level_edits.items():
                new_cfg[k] = v
            # Apply etfs edit
            if etfs_edit is not None:
                new_cfg["etfs"] = etfs_edit
            # Apply section edits
            for section_key, edits in all_edits.items():
                if section_key in new_cfg and isinstance(new_cfg[section_key], dict):
                    new_cfg[section_key] = _apply_edits(new_cfg[section_key], edits)
            try:
                CFG_PATH.write_text(
                    yaml.dump(new_cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
                )
                st.success("✅ Saved. Reloading…")
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    # ---- Raw / download ----------------------------------------------------
    with st.expander("Raw config.yaml"):
        st.code(CFG_PATH.read_text(), language="yaml")

    st.download_button(
        "⬇ Download config.yaml",
        data=CFG_PATH.read_text(),
        file_name="config.yaml",
        mime="text/yaml",
        key="cv_download",
    )
