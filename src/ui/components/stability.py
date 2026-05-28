"""
ui/components/stability.py — Stability scan runner and report viewer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS, ROOT


def _load_stability_csv(output_dir: str) -> pd.DataFrame | None:
    import glob
    files = sorted(glob.glob(f"{output_dir}/stability_summary_*.csv"))
    if not files:
        return None
    try:
        return pd.read_csv(files[-1])
    except Exception:
        return None


def _load_robustness_txt(output_dir: str) -> str | None:
    import glob
    files = sorted(glob.glob(f"{output_dir}/robustness_report_*.txt"))
    if not files:
        return None
    return Path(files[-1]).read_text(errors="replace")


def render() -> None:
    st.title("🔭 Stability & Robustness")
    st.caption("Multi-window parameter stability scan. RESEARCH ONLY — never writes config.")

    # ---- Controls ---------------------------------------------------------
    st.subheader("Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="stab_mode")
        st.caption(LOOKAHEAD_LABELS[mode])
    with c2:
        output_dir = st.text_input("Output directory", value=str(ROOT / "reports" / "stability"), key="stab_output_dir")
    with c3:
        windows_str = st.text_input("Windows (days, comma-separated)", value="30,60,90,120", key="stab_windows")

    try:
        windows = [int(w.strip()) for w in windows_str.split(",") if w.strip()]
    except ValueError:
        st.error("Invalid windows format. Use comma-separated integers.")
        windows = [30, 60, 90]

    st.code(f"daily-investor stability-scan --mode {mode} --output-dir {output_dir}", language="bash")
    st.warning("⚠️ Stability scans are long-running (one optimization per window). Allow several minutes.")

    # ---- Run --------------------------------------------------------------
    if st.button("▶ Run stability scan", type="primary"):
        from tuning.stability import StabilityAnalyzer
        with st.spinner(f"Running stability scan over {windows} windows…"):
            try:
                analyzer = StabilityAnalyzer()
                result = analyzer.scan(windows=windows, mode=mode, output_dir=output_dir)
                st.session_state["stability_result"] = result
                st.session_state["stability_output_dir"] = output_dir
                st.success(f"✅ Scan complete. {result.n_windows} windows. Outputs in {output_dir}")
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                st.exception(exc)
                return

    # ---- Results ----------------------------------------------------------
    result = st.session_state.get("stability_result")
    out_dir = st.session_state.get("stability_output_dir", output_dir)

    if result is None:
        # Try loading from disk
        st.info("No scan run this session. Showing reports from disk if available.")
    else:
        st.divider()
        st.subheader("Scan summary")
        st.caption(result.summary())

    # Load stability CSV
    stability_df = result.stability_df if result and result.stability_df is not None else _load_stability_csv(out_dir)

    if stability_df is not None and not stability_df.empty:
        st.subheader("Parameter stability table")
        stable_counts = stability_df["stability"].value_counts() if "stability" in stability_df.columns else {}
        cc = st.columns(3)
        for i, label in enumerate(["STABLE", "MODERATELY_STABLE", "UNSTABLE"]):
            count = stable_counts.get(label, 0)
            cc[i].metric(label, count)

        def _color_row(row):
            if row.get("stability") == "UNSTABLE":
                return ["background-color: #ffe0e0"] * len(row)
            if row.get("stability") == "MODERATELY_STABLE":
                return ["background-color: #fff8e0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            stability_df.sort_values("instability_score", ascending=False)
                         .style.apply(_color_row, axis=1),
            use_container_width=True,
        )

    # Heatmaps
    heatmap_dir = Path(out_dir)
    heatmaps_found = list(heatmap_dir.glob("*.png")) if heatmap_dir.exists() else []
    if heatmaps_found:
        st.subheader("Heatmaps")
        for img_path in sorted(heatmaps_found):
            st.image(str(img_path), caption=img_path.name, use_container_width=True)

    # Robustness report text
    txt = _load_robustness_txt(out_dir)
    if txt:
        with st.expander("Robustness report"):
            st.code(txt, language="text")
