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
    _date = datetime.now().strftime("%a %d %b %Y")
    _or = COLORS['bloomberg_orange']

    st.markdown(f"""
    <style>
        .block-container {{ padding-top: 0 !important; }}
        div[data-testid="stTextInput"] input {{
            background: #0a0a0a !important;
            border: 1px solid {_or} !important;
            border-radius: 2px !important;
            color: #E0E0E0 !important;
            font-family: 'JetBrains Mono', Consolas, monospace !important;
            font-size: 13px !important;
            letter-spacing: 0.12em !important;
        }}
        div[data-testid="stTextInput"] label {{
            font-family: 'JetBrains Mono', Consolas, monospace !important;
            font-size: 10px !important;
            letter-spacing: 0.18em !important;
            color: #64748b !important;
        }}
    </style>
    <div style="min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0D0D0D;margin:-4rem -4rem 0 -4rem;padding:4rem;">
        <div style="position:fixed;top:0;left:0;right:0;background:#111;border-bottom:1px solid #1e293b;padding:6px 24px;display:flex;justify-content:space-between;align-items:center;font-family:'JetBrains Mono',Consolas,monospace;font-size:10px;color:#334155;letter-spacing:0.12em;z-index:9999;">
            <span>REGARD TERMINALS &nbsp;&middot;&nbsp; INTELLIGENCE SYSTEM v2.0</span>
            <span>{_date} &nbsp;&middot;&nbsp; {_now}</span>
        </div>
        <div style="width:100%;max-width:480px;border:1px solid #1e293b;border-top:3px solid {_or};background:#111;padding:48px 48px 40px 48px;box-shadow:0 0 60px rgba(255,136,17,0.06);">
            <div style="margin-bottom:32px;">
                <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:72px;font-weight:800;line-height:0.9;color:{_or};letter-spacing:0.06em;">REGARD</div>
                <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:28px;font-weight:300;color:#E0E0E0;letter-spacing:0.32em;margin-left:4px;">TERMINALS</div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:28px;">
                <div style="flex:1;height:1px;background:#1e293b;"></div>
                <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:9px;color:#334155;letter-spacing:0.2em;">MARKET INTELLIGENCE</div>
                <div style="flex:1;height:1px;background:#1e293b;"></div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:#1e293b;margin-bottom:36px;">
                <div style="background:#111;padding:10px 12px;">
                    <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:9px;color:#334155;letter-spacing:0.15em;margin-bottom:3px;">MODULES</div>
                    <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:18px;font-weight:700;color:{_or};">8</div>
                </div>
                <div style="background:#111;padding:10px 12px;">
                    <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:9px;color:#334155;letter-spacing:0.15em;margin-bottom:3px;">SIGNALS</div>
                    <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:18px;font-weight:700;color:{_or};">17+</div>
                </div>
                <div style="background:#111;padding:10px 12px;">
                    <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:9px;color:#334155;letter-spacing:0.15em;margin-bottom:3px;">ACCURACY</div>
                    <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:18px;font-weight:700;color:#64748b;">TBD</div>
                </div>
            </div>
            <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:11px;color:#334155;letter-spacing:0.08em;margin-bottom:36px;border-left:2px solid {_or};padding-left:12px;line-height:1.8;">
                Providing excellent analytics,<br>one losing trade at a time.<br>
                <span style="color:#1e293b;">&#8212; Regardhodler</span>
            </div>
            <div style="font-family:'JetBrains Mono',Consolas,monospace;font-size:9px;color:#334155;letter-spacing:0.2em;margin-bottom:12px;">SECURE ACCESS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        pwd = st.text_input("ACCESS CODE", type="password", label_visibility="visible")
        if st.button("AUTHENTICATE", type="primary", use_container_width=True):
            if pwd == password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Access denied.")
        st.markdown(
            f'<div style="text-align:center;margin-top:20px;font-family:\'JetBrains Mono\',Consolas,monospace;'
            f'font-size:9px;color:#1e293b;letter-spacing:0.12em;">'
            f'REGARD TERMINALS &nbsp;&middot;&nbsp; {_now} &nbsp;&middot;&nbsp; AUTHORIZED USE ONLY</div>',
            unsafe_allow_html=True,
        )
    st.stop()
