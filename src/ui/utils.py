"""
ui/utils.py — Shared path constants and helpers for all UI components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_UI_DIR  = Path(__file__).resolve().parent
_SRC_DIR = _UI_DIR.parent
ROOT     = _SRC_DIR.parent
DATA_DIR = ROOT / "data"
CFG_PATH = ROOT / "cfg" / "config.yaml"
LOG_PATH = ROOT / "investment_bot.log"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def latest_csv_path(prefix: str) -> Path | None:
    files = sorted(DATA_DIR.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None


def load_latest_csv(prefix: str) -> pd.DataFrame | None:
    p = latest_csv_path(prefix)
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def list_csv_files() -> dict[str, Path]:
    """Return {display_name: path} for all CSV files in data/, newest first."""
    out: dict[str, Path] = {}
    for p in sorted(DATA_DIR.glob("*.csv"), reverse=True):
        out[p.name] = p
    return out


def load_config_raw() -> dict:
    if not CFG_PATH.exists():
        return {}
    try:
        with open(CFG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def render_overlay_banner(cfg: dict | None = None) -> None:
    """Show an st.info banner when the regime de-risk overlay is active (frac>0).

    Shared by every backtest surface so the user always knows the overlay is on
    before running — it silently changes downturn behavior. No-op when disabled.
    """
    import streamlit as st
    if cfg is None:
        cfg = load_config_raw()
    ro = (cfg.get("regime", {}) or {}).get("defensive", {}) or {}
    frac = float(ro.get("backtest_derisk_frac", 0.0) or 0.0)
    if frac <= 0:
        return
    st.info(
        f"🛡️ **Regime de-risk overlay ACTIVE** (frac={frac:.2f}, "
        f"lag={int(ro.get('backtest_derisk_lag', 1))}d, "
        f"{float(ro.get('backtest_derisk_switch_bps', 20.0)):.0f}bps switch). "
        "On defensive-regime entry (SPY >5% below 200DMA) this fraction of the "
        "held stock book rotates into the benchmark until the regime clears. "
        "No-op in bull/neutral windows."
    )


def ui_config() -> dict:
    """Read ui: section from config, falling back to safe defaults."""
    cfg = load_config_raw()
    defaults = {
        "allow_live_execution": False,
        "allow_config_writes": False,
        "allow_force_apply": False,
        "require_confirmation_phrase": True,
        "confirmation_phrase": "EXECUTE",
        "require_preview_before_execute": True,
        "intent_ttl_minutes": 5,
        "default_select_hard_sells": True,
        "default_select_soft_sells": False,
        "default_select_buys": False,
        "default_select_harvests": False,
    }
    defaults.update(cfg.get("ui", {}))
    return defaults


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def data_date(prefix: str) -> str:
    p = latest_csv_path(prefix)
    return p.stem.split("_", maxsplit=1)[1] if p else "—"


def no_data_msg(prefix: str) -> str:
    return f"No `{prefix}_*.csv` found in `data/`. Run the bot first to generate data."


def pct(val: float) -> str:
    return f"{val:+.1%}"


def dollars(val: float) -> str:
    return f"${val:,.2f}"


MODES = {
    "Default (from config)": None,
    "Safe (manual confirm)": "safe",
    "Automated (hands-off)": "automated",
    "No-sentiment (quant only)": "no-sentiment",
}

BACKTEST_MODES = [
    "liquid_universe_sanity_test",
    "walk_forward_price_only_test",
    "current_universe_stress_test",
]

LOOKAHEAD_LABELS = {
    "liquid_universe_sanity_test":   "MEDIUM — top-300 by volume (liquid_all, deterministic). Fundamental scores used.",
    "walk_forward_price_only_test":  "LOW — top-300 by volume, price-only momentum. No fundamental scores (active sleeve gets 0 trades).",
    "current_universe_stress_test":  "HIGH — top-300 by current score, forward-looking selection bias. Not predictive.",
}

# (level_str, emoji) pairs — used where compact display is needed
LOOKAHEAD_LEVELS = {
    "liquid_universe_sanity_test":  ("MEDIUM", "🟡"),
    "walk_forward_price_only_test": ("LOW",    "🟢"),
    "current_universe_stress_test": ("HIGH",   "🔴"),
}


def fmt_bin_index(counts: pd.Series) -> pd.Series:
    """Replace pd.IntervalIndex labels with 'left–right' strings for st.bar_chart."""
    import pandas as pd
    if isinstance(counts.index, pd.IntervalIndex):
        mag = max(abs(counts.index.left.max()), abs(counts.index.right.max()))
        dec = 0 if mag >= 100 else (1 if mag >= 10 else 2)
        fmt = f"{{:.{dec}f}}"
        counts = counts.copy()
        counts.index = [f"{fmt.format(i.left)}–{fmt.format(i.right)}" for i in counts.index]
    return counts
