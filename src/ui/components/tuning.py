"""
ui/components/tuning.py — Parameter tuner UI.
Calls ParameterTuner / auto_tune directly, same as CLI cmd_tune / cmd_auto_tune.
"""

from __future__ import annotations

import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS, ui_config


def render() -> None:
    st.title("⚙️ Auto-Tune")
    st.caption("Optimize parameters via scipy differential evolution. Validation-gated before any config write.")

    ui_cfg = ui_config()
    allow_write = ui_cfg.get("allow_config_writes", False)
    allow_force = ui_cfg.get("allow_force_apply", False)

    # ---- Mode selector ----------------------------------------------------
    tune_type = st.radio("Tune type", ["Auto-Tune (Sharpe+Calmar)", "Single-Objective Tune"],
                          horizontal=True)

    # ---- Settings ---------------------------------------------------------
    st.subheader("Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_days = st.number_input("Look-back days", min_value=30, max_value=1000, value=90, step=30, key="tune_n_days")
    with c2:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="tune_mode")
        st.caption(LOOKAHEAD_LABELS[mode])
    with c3:
        if tune_type == "Single-Objective Tune":
            objective = st.selectbox("Objective", ["sharpe", "calmar"])
        else:
            objective = "sharpe"

    apply_cfg = False
    force_apply = False
    llm_review = False

    if tune_type == "Auto-Tune (Sharpe+Calmar)":
        st.subheader("Apply options")
        c1, c2, c3 = st.columns(3)
        with c1:
            apply_cfg = st.checkbox(
                "Apply if validation passes",
                disabled=not allow_write,
                help="Writes config.yaml only when all validation gates pass." if allow_write
                     else "Disabled: set ui.allow_config_writes: true in config.yaml",
            )
        with c2:
            force_apply = st.checkbox(
                "Force apply (skip validation) ⚠️",
                disabled=not allow_force,
                help="Bypasses validation gates. Debugging only." if allow_force
                     else "Disabled: set ui.allow_force_apply: true",
            )
        with c3:
            llm_review = st.checkbox("LLM second-opinion review")

    # ---- CLI preview ------------------------------------------------------
    if tune_type == "Auto-Tune (Sharpe+Calmar)":
        flags = f" --mode {mode}"
        if apply_cfg:
            flags += " --apply"
        if force_apply:
            flags += " --force-apply"
        if llm_review:
            flags += " --llm-review"
        st.code(f"daily-investor auto-tune {n_days}{flags}", language="bash")
    else:
        st.code(f"daily-investor tune {n_days} --objective {objective} --mode {mode}", language="bash")

    if not allow_write:
        st.info("🔒 Config writes disabled. Results will be shown but not written to config.yaml.")

    # ---- Run --------------------------------------------------------------
    if st.button("▶ Run", type="primary"):
        from tuning.tuner import ParameterTuner
        tuner = ParameterTuner()

        if tune_type == "Single-Objective Tune":
            with st.spinner(f"Running {n_days}-day {objective} tune…"):
                try:
                    result = tuner.tune(n_days=n_days, objective=objective, mode=mode)
                    st.session_state["tune_result"] = result
                    st.session_state["tune_type"] = "single"
                    st.success("✅ Tune complete.")
                except Exception as exc:
                    st.error(f"Tune failed: {exc}")
                    st.exception(exc)
                    return
        else:
            with st.spinner(f"Running {n_days}-day auto-tune…"):
                try:
                    result = tuner.auto_tune(
                        n_days=n_days, mode=mode,
                        apply=apply_cfg, force_apply=force_apply,
                        llm_review=llm_review,
                    )
                    st.session_state["tune_result"] = result
                    st.session_state["tune_type"] = "auto"
                    st.success("✅ Auto-tune complete.")
                except Exception as exc:
                    st.error(f"Auto-tune failed: {exc}")
                    st.exception(exc)
                    return

    # ---- Results ----------------------------------------------------------
    result = st.session_state.get("tune_result")
    if result is None:
        st.info("No tune run yet.")
        return

    st.divider()
    st.subheader("Results")
    tune_type_done = st.session_state.get("tune_type", "single")

    if tune_type_done == "single":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Objective", result.objective)
        c2.metric("Score", f"{result.score:+.3f}")
        c3.metric("Total return", f"{result.sim.total_return:+.1%}")
        c4.metric("Max drawdown", f"{result.sim.max_drawdown:.1%}")
        c1.metric("Sharpe", f"{result.sim.sharpe:+.3f}")
        c2.metric("Calmar", f"{result.sim.calmar:+.3f}")
        c3.metric("Trades", result.sim.trades_made)
        c4.metric("Active params", len(result.active_params))
        st.caption(f"Active params: {', '.join(result.active_params)}")

    else:  # auto
        # Validation status
        if result.validation_passed:
            st.success("✅ Validation PASSED" + (" — config written" if result.config_written else ""))
        else:
            st.error(f"❌ Validation FAILED: {'; '.join(result.validation_reasons)}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Sharpe-optimized run**")
            st.metric("Return", f"{result.sharpe_result.total_return:+.1%}")
            st.metric("Sharpe", f"{result.sharpe_result.sharpe:+.3f}")
        with c2:
            st.markdown("**Calmar-optimized run**")
            st.metric("Return", f"{result.calmar_result.total_return:+.1%}")
            st.metric("Calmar", f"{result.calmar_result.calmar:+.3f}")
        with c3:
            st.markdown("**Averaged result**")
            st.metric("Return", f"{result.avg_result.total_return:+.1%}")
            st.metric("Sharpe", f"{result.avg_result.sharpe:+.3f}")
            st.metric("Calmar", f"{result.avg_result.calmar:+.3f}")

        with st.expander("Parameter stability (Sharpe vs Calmar spread)"):
            spread = result.param_spread
            import pandas as pd
            sp_df = pd.DataFrame({"parameter": list(spread.keys()), "spread": list(spread.values())})
            sp_df = sp_df.sort_values("spread", ascending=False)
            sp_df["flag"] = sp_df["spread"].apply(lambda x: "⚠️ unstable" if x > 0.05 else "✅")
            st.dataframe(sp_df, use_container_width=True)

        st.caption(result.summary())
