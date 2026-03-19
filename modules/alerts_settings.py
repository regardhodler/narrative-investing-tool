"""Alerts Settings — configure Telegram alerts and manage triggers."""

import streamlit as st
from utils.theme import COLORS
from utils.alerts_config import load_config, save_config
from services.alerts_service import send_telegram


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">ALERTS</div>',
        unsafe_allow_html=True,
    )

    cfg = load_config()

    # --- Telegram Setup ---
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">TELEGRAM SETUP</div>', unsafe_allow_html=True)

    with st.expander("How to set up Telegram alerts", expanded=False):
        st.markdown("""
**Step 1:** Open Telegram, search for **@BotFather**

**Step 2:** Send `/newbot` and follow prompts to create your bot

**Step 3:** Copy the **bot token** (looks like `123456:ABC-DEF1234`)

**Step 4:** Start a chat with your new bot, send any message

**Step 5:** Get your chat ID:
- Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
- Find `"chat":{"id": 123456}` — that number is your chat ID

**Step 6:** Paste both values below and click TEST
        """)

    col1, col2 = st.columns(2)
    with col1:
        bot_token = st.text_input("Bot Token", value=cfg.get("telegram_bot_token", ""),
                                   type="password", key="alert_token")
    with col2:
        chat_id = st.text_input("Chat ID", value=cfg.get("telegram_chat_id", ""), key="alert_chatid")

    col_save, col_test = st.columns(2)
    with col_save:
        if st.button("SAVE CONFIG", key="alert_save"):
            cfg["telegram_bot_token"] = bot_token
            cfg["telegram_chat_id"] = chat_id
            save_config(cfg)
            st.success("Config saved.")

    with col_test:
        if st.button("SEND TEST", key="alert_test"):
            if bot_token and chat_id:
                ok = send_telegram(bot_token, chat_id,
                                   "📡 <b>HRT Test</b>\n\nTelegram alerts are working!")
                if ok:
                    st.success("Test message sent!")
                else:
                    st.error("Failed to send. Check your token and chat ID.")
            else:
                st.warning("Enter bot token and chat ID first.")

    # --- Triggers ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">ALERT TRIGGERS</div>', unsafe_allow_html=True)

    triggers = cfg.get("triggers", {})
    thresholds = cfg.get("thresholds", {})

    col1, col2 = st.columns(2)
    with col1:
        regime_flip = st.checkbox("Risk Regime Flip", value=triggers.get("regime_flip", True),
                                   key="trig_regime", help="Alert when regime changes (Risk-On / Risk-Off)")
        insider_cluster = st.checkbox("Insider Buying Cluster", value=triggers.get("insider_cluster", True),
                                       key="trig_insider", help="Alert when 3+ insider buys in 30 days on watchlist")
    with col2:
        pc_ratio = st.checkbox("Options P/C Ratio", value=triggers.get("options_pc_ratio", False),
                                key="trig_pc", help="Alert when P/C ratio exceeds threshold")
        stress = st.checkbox("Stress Score Threshold", value=triggers.get("stress_threshold", False),
                              key="trig_stress", help="Alert when stress score crosses threshold")

    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin:8px 0 4px 0;">'
                f'THRESHOLDS</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        pc_thresh = st.number_input("P/C Ratio Threshold", value=thresholds.get("pc_ratio", 1.5),
                                     min_value=0.5, max_value=5.0, step=0.1, key="thresh_pc")
    with col2:
        stress_thresh = st.number_input("Stress Score Threshold", value=thresholds.get("stress_score", 70),
                                         min_value=0, max_value=100, step=5, key="thresh_stress")

    if st.button("SAVE TRIGGERS", type="primary", key="alert_save_triggers"):
        cfg["triggers"] = {
            "regime_flip": regime_flip,
            "insider_cluster": insider_cluster,
            "options_pc_ratio": pc_ratio,
            "stress_threshold": stress,
        }
        cfg["thresholds"] = {
            "pc_ratio": pc_thresh,
            "stress_score": stress_thresh,
        }
        save_config(cfg)
        st.success("Triggers saved.")

    # --- Alert History ---
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:8px 0 4px 0;">ALERT HISTORY</div>', unsafe_allow_html=True)

    history = cfg.get("alert_history", [])
    if not history:
        st.info("No alerts sent yet.")
    else:
        for entry in history[:20]:
            time_str = entry.get("time", "")[:16].replace("T", " ")
            msg = entry.get("message", "")
            st.markdown(
                f'<div style="font-size:12px;padding:4px 0;border-bottom:1px solid {COLORS["border"]};">'
                f'<span style="color:{COLORS["text_dim"]};">{time_str}</span> '
                f'<span style="color:{COLORS["text"]};">{msg}</span></div>',
                unsafe_allow_html=True,
            )

    # Note about limitations
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0;"></div>',
                unsafe_allow_html=True)
    st.caption("Alerts only fire when the app is open (Streamlit has no background scheduler). "
               "For always-on alerts, set up Windows Task Scheduler to hit http://localhost:8501 hourly.")
