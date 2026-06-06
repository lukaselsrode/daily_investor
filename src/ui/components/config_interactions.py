"""
ui/components/config_interactions.py — parameter-interaction screener panel.

Runs the empirical interaction screener (tuning.interaction_screen) from the UI and
renders the cluster×cluster synergy/conflict heatmap + pair verdicts. RESEARCH ONLY —
never writes config. The full 5-cluster screen is an overnight job; the panel defaults
to the `quick` profile and points at the CLI (`make interaction-screen`) for full runs.
"""
from __future__ import annotations

import streamlit as st

from ui.utils import BACKTEST_MODES, LOOKAHEAD_LABELS

_PROFILES = {
    "quick":    dict(robustness="quick",    horizon="short", maxiter=5,  popsize=4,
                     note="fast smoke — directional only"),
    "standard": dict(robustness="standard", horizon="mixed", maxiter=8,  popsize=6,
                     note="normal research"),
    "deep":     dict(robustness="deep",     horizon="mixed", maxiter=12, popsize=8,
                     note="overnight — strongest signal"),
}


def render() -> None:
    st.title("🔗 Parameter Interactions")
    st.caption(
        "Which interaction-cluster pairs SYNERGIZE vs CLASH when co-tuned, on the full "
        "universe with the robust multi-window objective. RESEARCH ONLY — never writes config."
    )

    from tuning.interaction_screen import DEFAULT_CLUSTERS

    c1, c2, c3 = st.columns(3)
    with c1:
        profile = st.selectbox(
            "Profile", list(_PROFILES),
            format_func=lambda k: f"{k} — {_PROFILES[k]['note']}",
            key="isc_profile",
        )
    with c2:
        mode = st.selectbox("Backtest mode", BACKTEST_MODES, key="isc_mode")
        st.caption(LOOKAHEAD_LABELS.get(mode, ""))
    with c3:
        n_days = int(st.number_input("History (days)", min_value=180, max_value=1500,
                                     value=730, step=60, key="isc_days"))

    clusters = st.multiselect(
        "Clusters to screen", list(DEFAULT_CLUSTERS), default=list(DEFAULT_CLUSTERS),
        key="isc_clusters",
        help="Each cluster is a group of params that co-determine one decision surface. "
             "The screen tunes each alone and every PAIR jointly, then reports "
             "interaction = joint − best-marginal.",
    )

    n = len(clusters)
    n_pairs = n * (n - 1) // 2
    n_tunes = n + n_pairs
    st.caption(
        f"{n} marginals + {n_pairs} joint pairs = **{n_tunes} robust tunes**. "
        "Each tune runs differential evolution over the multi-window robust objective."
    )
    st.code(f"make interaction-screen PROFILE={profile}", language="bash")
    if profile != "quick":
        st.warning("⚠️ Non-quick profiles are long-running (many full-universe robust tunes). "
                   "For the full overnight screen, prefer the CLI: `make interaction-screen PROFILE=deep`.")

    if st.button("▶ Run interaction screen", type="primary", key="isc_run"):
        if n < 2:
            st.error("Select at least 2 clusters.")
            return
        bar = st.progress(0, text="Loading full-universe data…")
        try:
            from ui.services.backtest_service import load_precomp
            precomp = load_precomp(n_days, mode=mode)
        except Exception as exc:
            bar.empty()
            st.error(f"Failed to load data: {exc}")
            return

        from tuning.interaction_screen import screen_interactions
        from tuning.profiles import expand_run_matrix
        cfg = _PROFILES[profile]
        run_matrix = expand_run_matrix(cfg["robustness"], cfg["horizon"])

        def _cb(done: int, total: int) -> None:
            bar.progress(min(int(done / max(total, 1) * 100), 99), text=f"Tune {done}/{total}…")

        try:
            with st.spinner("Screening interactions…"):
                result = screen_interactions(
                    precomp, run_matrix=run_matrix, cluster_names=clusters,
                    scope="active_sleeve_compounding",
                    maxiter=cfg["maxiter"], popsize=cfg["popsize"], progress_callback=_cb,
                )
            bar.empty()
            st.session_state["isc_result"] = result
            st.success(f"✅ Screened {n_tunes} tunes across {n} clusters.")
        except Exception as exc:
            bar.empty()
            st.error(f"Screen failed: {exc}")
            st.exception(exc)
            return

    result = st.session_state.get("isc_result")
    if result is None:
        st.info("No screen run yet. Configure above and click Run (start with the quick profile).")
        return

    st.divider()
    st.subheader("Synergy / conflict matrix")
    st.caption("Diagonal = each cluster's marginal robust score. Off-diagonal = interaction "
               "(joint − best-marginal): 🟢 green > 0 synergize (co-tune), 🔴 red < 0 clash.")
    mat = result.matrix_df()
    try:
        import numpy as np
        import plotly.express as px
        # Heatmap of interaction only (blank the diagonal so marginals don't skew the scale).
        inter = mat.copy().astype(float)
        for nme in inter.index:
            inter.loc[nme, nme] = np.nan
        fig = px.imshow(
            inter, text_auto=".3f", color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0.0, aspect="auto",
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_colorbar=dict(title="interaction"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(mat, use_container_width=True)

    st.subheader("Pair verdicts (sorted by synergy)")
    st.dataframe(result.pairs_df(), use_container_width=True, hide_index=True)
    st.caption(
        "🟢 synergy → co-tune the pair (the join beats either alone) · "
        "🔴 clash → tuning one undermines the other · "
        "↔ compromise → params move a lot but net score is flat · "
        "⚪ ~independent → tune separately."
    )
