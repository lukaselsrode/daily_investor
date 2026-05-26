"""
ui/components/logs.py — Log viewer and audit trail.
"""

from __future__ import annotations


import pandas as pd
import streamlit as st

from ui.utils import DATA_DIR, LOG_PATH


def render() -> None:
    st.title("📋 Logs / Audit Trail")
    st.caption("Read-only log viewer for debugging and auditability.")

    # ---- Main application log --------------------------------------------
    st.subheader("Application log")
    if not LOG_PATH.exists():
        st.info(f"No log file found at `{LOG_PATH}`.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            n_lines = st.slider("Lines to show", 20, 500, 100, 20)
        with c2:
            filter_text = st.text_input("Filter (keyword)")

        text = LOG_PATH.read_text(errors="replace")
        lines = text.splitlines()

        if filter_text:
            lines = [ln for ln in lines if filter_text.lower() in ln.lower()]

        tail = "\n".join(lines[-n_lines:])
        st.code(tail or "(no matching lines)", language="text")
        st.caption(f"Showing last {min(n_lines, len(lines))} of {len(lines)} matching lines.")

    st.divider()

    # ---- Audit CSVs ------------------------------------------------------
    st.subheader("Audit files")
    audit_prefixes = [
        ("order_intents",  "Order intents"),
        ("order_results",  "Order results"),
    ]

    for prefix, label in audit_prefixes:
        files = sorted(DATA_DIR.glob(f"{prefix}_*.csv"))
        if not files:
            st.info(f"No `{prefix}_*.csv` files found.")
            continue

        with st.expander(f"{label} ({len(files)} files)"):
            chosen = st.selectbox(f"File — {label}", [f.name for f in reversed(files)],
                                   key=f"sel_{prefix}")
            chosen_path = DATA_DIR / chosen
            try:
                df = pd.read_csv(chosen_path)
                st.caption(f"{len(df)} rows")

                # Filter by symbol if available
                if "symbol" in df.columns:
                    syms = ["All"] + sorted(df["symbol"].dropna().unique().tolist())
                    sym_filter = st.selectbox("Symbol", syms, key=f"sym_{prefix}")
                    if sym_filter != "All":
                        df = df[df["symbol"] == sym_filter]

                st.dataframe(df, use_container_width=True)
                st.download_button("⬇ Download", df.to_csv(index=False), file_name=chosen,
                                   key=f"dl_{prefix}_{chosen}")
            except Exception as exc:
                st.error(f"Failed to load {chosen}: {exc}")

    st.divider()

    # ---- News / sentiment CSV --------------------------------------------
    st.subheader("News & sentiment")
    news_files = sorted(DATA_DIR.glob("news_*.csv"))
    if news_files:
        nf = news_files[-1]
        with st.expander(f"Latest news CSV: {nf.name}"):
            try:
                ndf = pd.read_csv(nf)
                sym_filter = st.selectbox("Symbol", ["All"] + sorted(ndf["symbol"].dropna().unique().tolist())
                                           if "symbol" in ndf.columns else ["All"], key="news_sym")
                if sym_filter != "All" and "symbol" in ndf.columns:
                    ndf = ndf[ndf["symbol"] == sym_filter]
                display_cols = [c for c in ["symbol", "title", "pub_date", "publisher"] if c in ndf.columns]
                st.dataframe(ndf[display_cols] if display_cols else ndf, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("No news CSV files found.")
