from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
import streamlit as st
from utils.session import get_narrative, get_ticker, set_ticker
from utils.watchlist import load_watchlist, remove_from_watchlist
from utils.theme import COLORS

st.set_page_config(
    page_title="Highly Regarded Terminals",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Bloomberg Terminal Global CSS ---
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
    font-size: 14px;
}}
.stApp {{
    background-color: {COLORS["bg"]};
    background-image: none;
}}
.block-container {{
    padding: 1rem 2rem;
}}
[data-testid="stSidebar"] {{
    background-color: {COLORS["sidebar_bg"]};
    border-right: 1px solid {COLORS["border"]};
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 13px;
    padding: 4px 8px;
    border-radius: 4px;
    transition: background-color 0.15s;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background-color: {COLORS["hover"]};
}}
.stButton > button {{
    background-color: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    text-transform: uppercase;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
    color: {COLORS["text"]};
    transition: all 0.15s;
}}
.stButton > button:hover {{
    border-color: {COLORS["accent"]};
    background-color: {COLORS["hover"]};
}}
.stButton > button[kind="primary"] {{
    background-color: {COLORS["bloomberg_orange"]};
    border-color: {COLORS["bloomberg_orange"]};
    color: #000;
    font-weight: 600;
}}
.stTextInput input, .stSelectbox [data-baseweb="select"] {{
    background-color: {COLORS["input_bg"]};
    border: 1px solid {COLORS["border"]};
    font-family: 'JetBrains Mono', Consolas, monospace;
    color: {COLORS["text"]};
}}
.stTextInput input:focus {{
    border-color: {COLORS["accent"]};
    box-shadow: 0 0 0 1px {COLORS["accent"]};
}}
.stTabs [data-baseweb="tab"] {{
    text-transform: uppercase;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 13px;
    letter-spacing: 0.05em;
}}
.stTabs [aria-selected="true"] {{
    border-bottom: 2px solid {COLORS["bloomberg_orange"]};
}}
[data-testid="stMetric"] {{
    background-color: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    padding: 12px;
    border-radius: 6px;
}}
[data-testid="stExpander"] {{
    background-color: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
}}
.stDataFrame {{
    font-family: 'JetBrains Mono', Consolas, monospace;
}}
.stDataFrame [data-testid="stDataFrameResizable"] {{
    background-color: {COLORS["surface_dark"]};
}}
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: {COLORS["bg"]};
}}
::-webkit-scrollbar-thumb {{
    background: {COLORS["border"]};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {COLORS["accent"]};
}}
code {{
    background-color: {COLORS["surface_dark"]};
    color: {COLORS["accent"]};
}}
</style>""", unsafe_allow_html=True)

# --- Bloomberg header bar ---
_now = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"""<div style="
    background:{COLORS["header_bg"]};
    border-left:4px solid {COLORS["bloomberg_orange"]};
    border-bottom:1px solid {COLORS["bloomberg_orange"]}33;
    padding:10px 20px;
    margin:-1rem -2rem 1rem -2rem;
    display:flex;
    align-items:center;
    justify-content:space-between;
">
    <span style="
        font-family:'JetBrains Mono',Consolas,monospace;
        font-size:16px;
        font-weight:700;
        color:{COLORS["bloomberg_orange"]};
        letter-spacing:0.1em;
    ">HIGHLY REGARDED TERMINALS</span>
    <span style="
        font-family:'JetBrains Mono',Consolas,monospace;
        font-size:12px;
        color:{COLORS["text_dim"]};
    ">{_now}</span>
</div>""", unsafe_allow_html=True)

# --- Authentication gate ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.markdown(f"""<div style="
        display:flex;flex-direction:column;align-items:center;justify-content:center;
        margin:8vh auto 0 auto;max-width:420px;
    ">
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:32px;font-weight:700;
            color:{COLORS['bloomberg_orange']};letter-spacing:0.12em;margin-bottom:4px;
        ">HIGHLY REGARDED</div>
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:32px;font-weight:300;
            color:{COLORS['text']};letter-spacing:0.18em;margin-bottom:8px;
        ">TERMINALS</div>
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:14px;font-weight:400;
            color:{COLORS['text_dim']};letter-spacing:0.10em;margin-bottom:8px;
        ">by Regardhodler</div>
        <div style="
            width:60px;height:2px;background:{COLORS['bloomberg_orange']};margin-bottom:16px;
        "></div>
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;
            color:{COLORS['text_dim']};letter-spacing:0.15em;text-transform:uppercase;
        ">Investment Intelligence System</div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        pwd = st.text_input("ACCESS CODE", type="password", label_visibility="visible")
        if st.button("AUTHENTICATE", type="primary", use_container_width=True):
            if pwd == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Access denied.")
        st.markdown(f"""<div style="
            text-align:center;margin-top:24px;font-family:'JetBrains Mono',Consolas,monospace;
            font-size:10px;color:{COLORS['text_dim']};letter-spacing:0.08em;
        ">v1.0 &middot; {_now}</div>""", unsafe_allow_html=True)
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:15px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.08em;margin-bottom:8px;">HIGHLY REGARDED TERMINALS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:4px 0 12px 0;"></div>', unsafe_allow_html=True)

    narrative = get_narrative()
    ticker = get_ticker()

    _status_style = (
        f"font-family:'JetBrains Mono',Consolas,monospace;font-size:12px;"
        f"color:{COLORS['text_dim']};margin:2px 0;"
    )
    _val_style = f"color:{COLORS['text']};font-weight:600;"
    st.markdown(
        f'<div style="{_status_style}">NARRATIVE <span style="{_val_style}">{narrative or "—"}</span></div>'
        f'<div style="{_status_style}">TICKER <span style="{_val_style}">{ticker or "—"}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0 8px 0;"></div>', unsafe_allow_html=True)

    # --- Watchlist widget ---
    watchlist = load_watchlist()
    if watchlist:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["bloomberg_orange"]};font-weight:700;letter-spacing:0.06em;margin-bottom:4px;">'
            f'WATCHLIST <span style="background:{COLORS["surface"]};padding:1px 6px;border-radius:3px;'
            f'font-size:11px;color:{COLORS["text"]};">{len(watchlist)}</span></div>',
            unsafe_allow_html=True,
        )
        wl_tickers = [f"{w['ticker']} — {w.get('narrative', '')}" for w in watchlist]
        selected_wl = st.selectbox("Switch to", wl_tickers, key="wl_select", label_visibility="collapsed")
        col_go, col_rm = st.columns(2)
        with col_go:
            if st.button("GO", key="wl_go", type="primary"):
                wl_ticker = selected_wl.split(" — ")[0]
                set_ticker(wl_ticker)
                st.session_state["top_module"] = "Discovery"
                st.session_state["sub_module"] = "Narrative Discovery"
                st.rerun()
        with col_rm:
            if st.button("DEL", key="wl_rm"):
                wl_ticker = selected_wl.split(" — ")[0]
                remove_from_watchlist(wl_ticker)
                st.rerun()
        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:8px 0;"></div>', unsafe_allow_html=True)

    top_level = st.radio(
        "Module",
        ["Discovery", "Risk Regime", "Whale Movement", "Stress Signals"],
        key="top_module",
    )

    sub_module = None
    if top_level == "Discovery":
        st.markdown(
            f'<style>[data-testid="stSidebar"] .discovery-indent label {{'
            f'padding-left: 20px !important; font-size: 12px !important;'
            f'}}</style>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="margin-left:12px;border-left:2px solid {COLORS["bloomberg_orange"]}33;padding-left:0px;">', unsafe_allow_html=True)
        sub_module = st.radio(
            "Discovery Modules",
            [
                "Narrative Discovery",
                "Options Activity",
                "Price Action",
                "EDGAR Scanner",
                "Institutional (13F)",
                "Insider & Congress",
                "Valuation",
            ],
            label_visibility="collapsed",
            key="sub_module",
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Route to module
if top_level == "Risk Regime":
    from modules.risk_regime import render
    render()
elif top_level == "Whale Movement":
    from modules.whale_buyers import render
    render()
elif top_level == "Stress Signals":
    from modules.stress_signals import render
    render()
elif top_level == "Discovery":
    _ticker = get_ticker()
    if _ticker:
        st.markdown(
            f'<div style="text-align:right;margin:-8px 0 8px 0;">'
            f'<span style="background:{COLORS["surface"]};border:1px solid {COLORS["bloomberg_orange"]};'
            f'border-radius:4px;padding:4px 12px;font-family:\'JetBrains Mono\',Consolas,monospace;'
            f'font-size:13px;">'
            f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{_ticker}</span>'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    if sub_module == "Narrative Discovery":
        from modules.narrative_discovery import render
        render()
    elif sub_module == "Price Action":
        from modules.narrative_pulse import render
        render()
    elif sub_module == "EDGAR Scanner":
        from modules.edgar_scanner import render
        render()
    elif sub_module == "Institutional (13F)":
        from modules.institutional import render
        render()
    elif sub_module == "Insider & Congress":
        from modules.insider_congress import render
        render()
    elif sub_module == "Options Activity":
        from modules.options_activity import render
        render()
    elif sub_module == "Valuation":
        from modules.valuation import render
        render()
