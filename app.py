from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from utils.session import get_narrative, get_ticker, is_ibkr_connected

st.set_page_config(
    page_title="Narrative Investing Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
            "1 · Discovery",
            "2 · Narrative Pulse",
            "3 · EDGAR Scanner",
            "4 · Smart Money",
            "5 · Options Activity",
        ],
    )

# Route to module
if page.startswith("1"):
    from modules.narrative_discovery import render
    render()
elif page.startswith("2"):
    from modules.narrative_pulse import render
    render()
elif page.startswith("3"):
    from modules.edgar_scanner import render
    render()
elif page.startswith("4"):
    from modules.smart_money import render
    render()
elif page.startswith("5"):
    from modules.options_activity import render
    render()
