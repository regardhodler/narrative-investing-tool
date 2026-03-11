from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from utils.session import get_narrative, get_ticker, is_ibkr_connected

st.set_page_config(
    page_title="Narrative Investing Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Authentication gate ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.markdown("## Narrative Investing Intelligence")
    pwd = st.text_input("Enter password to continue", type="password")
    if st.button("Login", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## NARRATIVE INTEL")
    st.markdown("---")

    narrative = get_narrative()
    ticker = get_ticker()
    ibkr = is_ibkr_connected()

    st.markdown(f"**Narrative:** {narrative or '—'}")
    st.markdown(f"**Ticker:** {ticker or '—'}")
    st.markdown(f"**IBKR:** {'🟢 Connected' if ibkr else '🔴 Disconnected'}")
    st.markdown("---")

    page = st.radio(
        "Module",
        [
            "0 · Risk Regime",
            "1 · Discovery",
            "2 · Price Action",
            "3 · EDGAR Scanner",
            "4 · Institutional (13F)",
            "5 · Insider & Congress",
            "6 · Options Activity",
            "7 · Valuation",
            "8 · Whale Buyers",
            "9 · Stress Signals",
        ],
    )

# Route to module
if page.startswith("0"):
    from modules.risk_regime import render
    render()
elif page.startswith("1"):
    from modules.narrative_discovery import render
    render()
elif page.startswith("2"):
    from modules.narrative_pulse import render
    render()
elif page.startswith("3"):
    from modules.edgar_scanner import render
    render()
elif page.startswith("4"):
    from modules.institutional import render
    render()
elif page.startswith("5"):
    from modules.insider_congress import render
    render()
elif page.startswith("6"):
    from modules.options_activity import render
    render()
elif page.startswith("7"):
    from modules.valuation import render
    render()
elif page.startswith("8"):
    from modules.whale_buyers import render
    render()
elif page.startswith("9"):
    from modules.stress_signals import render
    render()
