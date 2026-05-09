"""
ui/components/run_control.py — Command center.

Builds the equivalent CLI command from UI selections, previews it, and
executes via subprocess. Never rewrites strategy logic — delegates entirely
to cli.commands and the existing pipeline.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS, MODES, _SRC_DIR, ui_config, load_config_raw


def _build_command(run_type: str, op_mode: str | None, skip_data: bool,
                   n_days: int, bt_mode: str, objective: str,
                   apply: bool, force_apply: bool, llm_review: bool,
                   windows: list[int] | None) -> list[str]:
    base = [sys.executable, "-m", "cli"]

    if run_type == "Full Strategy Run":
        base.append("run")
        if op_mode:
            base += ["--op-mode", op_mode]
        if skip_data:
            base.append("--skip-data")

    elif run_type == "Backtest":
        base += ["backtest", str(n_days), "--mode", bt_mode]

    elif run_type == "Tune":
        base += ["tune", str(n_days), "--objective", objective, "--mode", bt_mode]

    elif run_type == "Auto-Tune":
        base += ["auto-tune", str(n_days), "--mode", bt_mode]
        if apply:
            base.append("--apply")
        if force_apply:
            base.append("--force-apply")
        if llm_review:
            base.append("--llm-review")

    elif run_type == "Stability Scan":
        base.append("stability-scan")
        base += ["--mode", bt_mode]

    elif run_type == "Report":
        base.append("report")

    return base


def _cmd_str(cmd: list[str]) -> str:
    parts = cmd[:]
    # Replace 'python -m cli' with 'daily-investor' for display
    if parts[:3] == [sys.executable, "-m", "cli"]:
        parts = ["daily-investor"] + parts[3:]
    return " ".join(parts)


def _stream_subprocess(cmd: list[str], out_container) -> int:
    """Run command, stream stdout+stderr to out_container. Returns exit code."""
    lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        cwd=str(_SRC_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader():
        for line in proc.stdout:
            lines.append(line.rstrip())

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    placeholder = out_container.empty()
    while proc.poll() is None or t.is_alive():
        placeholder.code("\n".join(lines[-80:]) or "(waiting…)", language="text")
        time.sleep(0.4)

    t.join()
    placeholder.code("\n".join(lines) or "(no output)", language="text")
    return proc.returncode


def render() -> None:
    st.title("🚀 Run Control")
    st.caption("Build and execute CLI commands. The equivalent shell command is shown before every run.")

    ui_cfg = ui_config()
    cfg = load_config_raw()
    live = st.session_state.get("live_enabled", False)

    # ---- Run type ---------------------------------------------------------
    st.subheader("A. Run type")
    run_type = st.selectbox("What do you want to run?", [
        "Full Strategy Run", "Backtest", "Tune", "Auto-Tune", "Stability Scan", "Report",
    ])

    # ---- Operating mode (only relevant for Full Strategy Run) -------------
    op_mode_key = None
    if run_type == "Full Strategy Run":
        st.subheader("B. Operating mode")
        mode_label = st.radio(
            "Mode",
            list(MODES.keys()),
            horizontal=True,
            help="Overrides config for this run only. Does NOT write config.yaml.",
        )
        op_mode_key = MODES[mode_label]

        c1, c2 = st.columns(2)
        with c1:
            skip_data = st.checkbox("Skip data refresh (use cached CSVs)", value=False)
        with c2:
            if not live:
                st.warning("🔒 Live execution is OFF. Run will execute in dry-run / read-only observation mode unless enabled in sidebar.")
    else:
        skip_data = False

    # ---- Backtest / tune settings -----------------------------------------
    n_days, bt_mode, objective, apply_cfg, force_apply, llm_review, windows = 90, BACKTEST_MODES[0], "sharpe", False, False, False, None

    if run_type in ("Backtest", "Tune", "Auto-Tune", "Stability Scan"):
        st.subheader("C. Simulation settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_days = st.number_input("Look-back days", min_value=30, max_value=1000, value=90, step=30)
        with c2:
            bt_mode = st.selectbox("Backtest mode", BACKTEST_MODES)
            st.caption(LOOKAHEAD_LABELS[bt_mode])
        with c3:
            if run_type == "Tune":
                objective = st.selectbox("Objective", ["sharpe", "calmar"])

        if run_type == "Auto-Tune":
            st.subheader("D. Tune options")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                apply_cfg = st.checkbox("Apply if validation passes", value=False,
                                        disabled=not ui_cfg.get("allow_config_writes"))
            with cc2:
                force_apply = st.checkbox("Force apply (skip validation)",
                                          disabled=not ui_cfg.get("allow_force_apply"))
            with cc3:
                llm_review = st.checkbox("LLM second-opinion review")

            if not ui_cfg.get("allow_config_writes"):
                st.info("🔒 Config writes are disabled (`ui.allow_config_writes: false`). Results will be shown but not written.")

    # ---- Command preview --------------------------------------------------
    st.divider()
    st.subheader("Equivalent CLI command")
    cmd = _build_command(run_type, op_mode_key, skip_data, n_days, bt_mode,
                         objective, apply_cfg, force_apply, llm_review, windows)
    st.code(_cmd_str(cmd), language="bash")

    # ---- Safety summary ---------------------------------------------------
    with st.expander("Run summary & safety checklist", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"- **Run type:** {run_type}")
            st.markdown(f"- **Live execution:** {'✅ ON' if live else '🔒 OFF'}")
            if run_type == "Full Strategy Run":
                st.markdown(f"- **Operating mode:** {mode_label}")
        with c2:
            if run_type == "Auto-Tune":
                st.markdown(f"- **Apply config:** {'YES' if apply_cfg else 'no'}")
                st.markdown(f"- **Force apply:** {'YES ⚠️' if force_apply else 'no'}")
                st.markdown(f"- **LLM review:** {'yes' if llm_review else 'no'}")
            if run_type in ("Backtest", "Tune", "Auto-Tune", "Stability Scan"):
                st.markdown(f"- **Days:** {n_days}  |  **Mode:** {bt_mode}")

    if run_type == "current_universe_stress_test" or bt_mode == "current_universe_stress_test":
        st.warning("⚠️ `current_universe_stress_test` uses current fundamental scores throughout history — this is a stress test with HIGH lookahead bias. Results are not predictive.")

    if force_apply:
        st.error("⚠️ Force-apply bypasses validation gates. Config will be written regardless of out-of-sample performance.")

    # ---- Run button -------------------------------------------------------
    st.divider()
    if run_type == "Full Strategy Run" and not live:
        st.button("▶ Run", disabled=True, help="Enable live execution in the sidebar first.")
        st.info("Live execution is OFF. Enable it in the sidebar to place real orders.")
    else:
        if st.button(f"▶ Run: {_cmd_str(cmd)}", type="primary"):
            st.session_state["last_cmd"] = _cmd_str(cmd)
            out = st.empty()
            with st.spinner(f"Running: {_cmd_str(cmd)}"):
                rc = _stream_subprocess(cmd, out)
            if rc == 0:
                st.success("✅ Command completed successfully.")
            else:
                st.error(f"❌ Command exited with code {rc}.")
            st.session_state["last_run_rc"] = rc

    # Show last run result if available
    if "last_cmd" in st.session_state:
        st.caption(f"Last command: `{st.session_state['last_cmd']}`  |  "
                   f"Exit: {st.session_state.get('last_run_rc', '?')}")
