"""
ui/components/intents.py — Order Intent Preview.
Shows what the bot would do without executing. Reads from order_intents CSV
if available, or shows instructions for generating one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR, load_config_raw, ui_config, MODES, _SRC_DIR
import sys


_INTENT_COLS = [
    "symbol", "side", "sleeve", "amount", "quantity",
    "order_type", "reason", "exit_type", "confidence",
    "sentiment_override", "reliability_score",
]


def render() -> None:
    st.title("🎯 Order Intent Preview")
    st.caption("Preview what the bot plans to do. No orders are placed here.")

    ui_cfg = ui_config()
    cfg = load_config_raw()

    # Try loading from CSV first
    intent_files = sorted(DATA_DIR.glob("order_intents_*.csv"))
    intents_df: pd.DataFrame | None = None
    if intent_files:
        try:
            intents_df = pd.read_csv(intent_files[-1])
        except Exception:
            pass

    # ---- Generate dry-run intents -----------------------------------------
    st.subheader("Generate intents (dry run)")
    mode_label = st.selectbox("Operating mode", list(MODES.keys()))
    op_mode = MODES[mode_label]
    skip = st.checkbox("Skip data refresh")

    cmd_parts = ["daily-investor", "run"]
    if op_mode:
        cmd_parts += ["--op-mode", op_mode]
    if skip:
        cmd_parts.append("--skip-data")
    st.code(" ".join(cmd_parts) + "  # (dry-run preview)", language="bash")

    if st.button("▶ Generate intent preview (dry run)"):
        import subprocess, threading, time
        cmd = [sys.executable, "-m", "cli", "run"]
        if op_mode:
            cmd += ["--op-mode", op_mode]
        if skip:
            cmd.append("--skip-data")
        # For safety: dry-run mode means we call the real pipeline but
        # the UI's live_enabled=False prevents actual order placement via
        # the confirmation gate in main.py (auto_approve=False)
        lines: list[str] = []
        out = st.empty()
        with st.spinner("Running dry-run…"):
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(_SRC_DIR),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                def _r():
                    for l in proc.stdout:
                        lines.append(l.rstrip())
                t = threading.Thread(target=_r, daemon=True)
                t.start()
                while proc.poll() is None or t.is_alive():
                    out.code("\n".join(lines[-60:]) or "(waiting…)", language="text")
                    time.sleep(0.4)
                t.join()
                out.code("\n".join(lines), language="text")
                rc = proc.returncode
                if rc == 0:
                    st.success("✅ Dry run complete.")
                    # Try to reload intents CSV
                    intent_files2 = sorted(DATA_DIR.glob("order_intents_*.csv"))
                    if intent_files2:
                        intents_df = pd.read_csv(intent_files2[-1])
                else:
                    st.warning(f"Exited with code {rc}.")
            except Exception as exc:
                st.error(f"Dry run failed: {exc}")

    st.divider()

    # ---- Show intents table -----------------------------------------------
    if intents_df is None:
        st.info("No order_intents CSV found in data/. Run a dry-run above to generate one.")
        st.markdown("""
        To generate intent data, run the bot in safe mode which will log intents
        without placing orders (when `auto_approve: false`):
        ```bash
        daily-investor run --op-mode safe
        ```
        Or look for `order_intents_*.csv` files in the `data/` directory.
        """)
        return

    st.subheader(f"Order intents ({len(intents_df)} rows)")
    disp_cols = [c for c in _INTENT_COLS if c in intents_df.columns]

    # Group by side
    for side_label, side_val in [
        ("🔴 Hard sells", "hard_sell"), ("🟠 Soft sells", "soft_sell"),
        ("🟢 Buys", "buy"), ("🌾 Harvests", "harvest"), ("📊 ETF actions", "etf"),
    ]:
        if "side" in intents_df.columns:
            subset = intents_df[intents_df["side"] == side_val]
        elif side_val == "buy":
            subset = intents_df
        else:
            continue
        if subset.empty:
            continue
        with st.expander(f"{side_label} ({len(subset)})"):
            st.dataframe(subset[disp_cols] if disp_cols else subset, use_container_width=True)
