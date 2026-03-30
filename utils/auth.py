"""Authentication gate — shows login screen and halts if APP_PASSWORD is set."""

import os
from datetime import datetime

import streamlit as st
from utils.theme import COLORS


def auth_gate() -> None:
    """If APP_PASSWORD is set and user is not authenticated, show login and st.stop()."""
    password = os.getenv("APP_PASSWORD", "")
    if not password or st.session_state.get("authenticated"):
        return

    _now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""<div style="
        display:flex;flex-direction:column;align-items:center;justify-content:center;
        margin:8vh auto 0 auto;max-width:420px;
    ">
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:42px;font-weight:700;
            color:{COLORS['bloomberg_orange']};letter-spacing:0.12em;margin-bottom:4px;
        ">REGARD</div>
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:42px;font-weight:300;
            color:{COLORS['text']};letter-spacing:0.18em;margin-bottom:8px;
        ">TERMINALS</div>
            font-family:'JetBrains Mono',Consolas,monospace;font-size:42px;font-weight:300;
            color:{COLORS['text']};letter-spacing:0.18em;margin-bottom:8px;
        ">TERMINALS</div>
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:13px;font-weight:400;
            color:#475569;letter-spacing:0.06em;margin-bottom:2px;
        ">providing excellent analytics, one losing trade at a time</div>
        <div style="
            font-family:'JetBrains Mono',Consolas,monospace;font-size:13px;font-weight:400;
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
            if pwd == password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Access denied.")
        st.markdown(f"""<div style="
            text-align:center;margin-top:24px;font-family:'JetBrains Mono',Consolas,monospace;
            font-size:10px;color:{COLORS['text_dim']};letter-spacing:0.08em;
        ">v1.0 &middot; {_now}</div>""", unsafe_allow_html=True)
    st.stop()
