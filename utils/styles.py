"""Global Bloomberg terminal CSS — injected once at app startup."""

import streamlit as st
from utils.theme import COLORS


def apply_global_css() -> None:
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
    font-size: 15px;
    font-weight: 600;
    padding: 5px 8px;
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
