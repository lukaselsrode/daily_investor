"""
ui/components/execution.py — Live execution with safety interlocks.
Only reachable when ui.allow_live_execution is true AND sidebar toggle is on.
"""

from __future__ import annotations

import streamlit as st

from ui.utils import ui_config


def render() -> None:
    st.title("⚡ Live Execution")

    ui_cfg = ui_config()
    live = st.session_state.get("live_enabled", False)

    if not ui_cfg.get("allow_live_execution"):
        st.error("🔒 Live execution is permanently disabled in config (`ui.allow_live_execution: false`).")
        st.info("To enable, add to `cfg/config.yaml`:\n```yaml\nui:\n  allow_live_execution: true\n```")
        return

    if not live:
        st.warning("🔒 Live execution toggle is OFF. Enable it in the sidebar first.")
        return

    # ---- Safety interlock summary ----------------------------------------
    st.warning("⚠️ **LIVE EXECUTION MODE** — Real orders will be placed with real money. Proceed carefully.")

    with st.expander("Safety checklist (required reading)", expanded=True):
        st.markdown("""
        - Orders placed here call the same `RobinhoodBroker` and `RiskManager` as the CLI.
        - `RiskManager.can_buy()` is always enforced — position caps, sector caps, and order caps apply.
        - Hard sells execute immediately. Soft sells respect the sentiment gate.
        - All intents are logged to `order_intents_*.csv` before execution.
        - All results are logged to `order_results_*.csv` after execution.
        - You must preview intents **before** executing.
        - Intents older than the TTL require a re-preview.
        """)

    # ---- Pre-flight: require intent preview first -------------------------
    import time
    ttl_minutes = ui_cfg.get("intent_ttl_minutes", 5)
    intents_ts  = st.session_state.get("intents_ts")
    intents     = st.session_state.get("last_intents")

    if intents is None:
        st.error("No intents generated. Go to **Order Intents** page first and run a preview.")
        return

    age_minutes = (time.time() - intents_ts) / 60 if intents_ts else 999
    if age_minutes > ttl_minutes:
        st.error(f"Intents are {age_minutes:.0f}m old (TTL = {ttl_minutes}m). Re-run the intent preview first.")
        return

    st.success(f"✅ Intents are fresh ({age_minutes:.1f}m old).")

    # ---- Intent selection -------------------------------------------------
    st.subheader("Select intents to execute")
    hard_sells  = [i for i in intents if i.get("side") == "hard_sell"]
    soft_sells  = [i for i in intents if i.get("side") == "soft_sell"]
    buys        = [i for i in intents if i.get("side") == "buy"]
    harvests    = [i for i in intents if i.get("side") == "harvest"]

    sel_hard  = st.checkbox(f"Execute {len(hard_sells)} hard sell(s)",  value=ui_cfg.get("default_select_hard_sells", True))
    sel_soft  = st.checkbox(f"Execute {len(soft_sells)} soft sell(s)",  value=False)
    sel_buys  = st.checkbox(f"Execute {len(buys)} buy(s)",              value=False)
    sel_harv  = st.checkbox(f"Execute {len(harvests)} harvest ETF buy(s)", value=False)

    selected = []
    if sel_hard:
        selected += hard_sells
    if sel_soft:
        selected += soft_sells
    if sel_buys:
        selected += buys
    if sel_harv:
        selected += harvests

    if not selected:
        st.info("No intents selected.")
        return

    # ---- Final confirmation -----------------------------------------------
    st.divider()
    st.subheader("Final confirmation")
    buy_total  = sum(i.get("amount", 0) for i in selected if i.get("side") not in ("hard_sell", "soft_sell"))
    sell_count = sum(1 for i in selected if i.get("side") in ("hard_sell", "soft_sell"))
    buy_syms   = [i["symbol"] for i in selected if i.get("side") == "buy"]
    sell_syms  = [i["symbol"] for i in selected if "sell" in i.get("side", "")]

    st.markdown(f"- **Sells:** {sell_count} ({', '.join(sell_syms) or 'none'})")
    st.markdown(f"- **Buys:** {len(buy_syms)} — ${buy_total:,.2f} ({', '.join(buy_syms) or 'none'})")

    confirmed = st.checkbox("I have reviewed the intents above and accept responsibility for live orders.")

    phrase = ui_cfg.get("confirmation_phrase", "EXECUTE")
    if ui_cfg.get("require_confirmation_phrase"):
        typed = st.text_input(f'Type "{phrase}" to confirm:')
        phrase_ok = typed.strip() == phrase
    else:
        phrase_ok = True

    can_execute = confirmed and phrase_ok

    if st.button("🔴 EXECUTE SELECTED INTENTS", type="primary", disabled=not can_execute):
        st.error("⚠️ Live order dispatch is not yet wired to the broker adapter in this UI version. Use `daily-investor run --op-mode safe` from the CLI for controlled live execution.")
        st.info("This interlock is intentional. The run_control page's subprocess approach (daily-investor run) is the supported live execution path.")
