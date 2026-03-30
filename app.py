from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from utils.styles import apply_global_css
from utils.auth import auth_gate
from utils.components import render_header, render_sidebar_header, render_macro_events, render_regime_strip, render_watchlist_widget
from utils.session import get_narrative, get_ticker
from utils.theme import COLORS

st.set_page_config(
    page_title="Regarded Terminals",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_css()
render_header()
auth_gate()

# --- Alert check on page load (1hr cooldown) ---
try:
    from services.alerts_service import check_and_send_alerts
    _fired = check_and_send_alerts()
    for _alert_msg in _fired:
        st.toast(_alert_msg, icon="📡")
except Exception:
    pass

# Load persisted AI signals on first run of each session
if not st.session_state.get("_signals_cache_loaded"):
    try:
        from services.signals_cache import load_signals
        load_signals()
    except Exception:
        pass
    st.session_state["_signals_cache_loaded"] = True

# Warm FRED cache once per session so Fed Forecaster + Risk Regime load fast
if not st.session_state.get("_fred_cache_warmed"):
    try:
        from services.market_data import warm_fred_cache
        warm_fred_cache()
    except Exception:
        pass
    st.session_state["_fred_cache_warmed"] = True

# --- Sidebar ---
with st.sidebar:
    render_sidebar_header(get_narrative(), get_ticker())
    render_macro_events()
    render_regime_strip()
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:4px 0 8px 0;"></div>', unsafe_allow_html=True)
    render_watchlist_widget()

    top_level = st.radio(
        "Module",
        [
            "⚡ Quick Intel Run", "Risk Regime", "Fed Forecaster", "Current Events",
            "Discovery", "Technical Analysis", "Whale Movement", "Stress Signals",
            "Short Squeeze Radar", "Backtesting", "My Regarded Portfolio",
            "Signal Audit", "Forecast Tracker", "Export Hub", "Alerts",
        ],
        key="top_module",
    )

    sub_module = None
    if top_level == "Discovery":
        _oc = COLORS["bloomberg_orange"]
        st.markdown(f"""<style>
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] {{
    border-left: 2px solid {_oc}55;
    margin-left: 18px !important;
    padding-left: 8px !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
}}
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] label {{
    font-size: 11px !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: 0.02em !important;
    padding: 2px 6px 2px 0 !important;
    border-radius: 0 4px 4px 0 !important;
    margin: 1px 0 !important;
    color: {COLORS["text_dim"]} !important;
    display: flex !important;
    align-items: center !important;
    transition: color 0.1s, background 0.1s;
}}
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] label::before {{
    content: '├·';
    margin-right: 6px;
    color: {_oc};
    opacity: 0.4;
    font-family: 'JetBrains Mono', Consolas, monospace;
    flex-shrink: 0;
}}
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] label:has(input:checked) {{
    color: {_oc} !important;
    background: {_oc}18 !important;
}}
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] label:has(input:checked)::before {{
    opacity: 1 !important;
    content: '├●';
}}
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] label:hover {{
    background: {_oc}0D !important;
    color: {_oc}BB !important;
}}
section[data-testid="stSidebar"] div:has(#disc-sub-anchor) ~ div [data-testid="stRadio"] label:hover::before {{
    opacity: 0.7 !important;
}}
</style>""", unsafe_allow_html=True)

        st.markdown(
            f'<div style="margin:0 0 0 20px;border-left:2px solid {_oc}55;height:6px;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div id="disc-sub-anchor" style="height:0;overflow:hidden;margin:0;padding:0;"></div>', unsafe_allow_html=True)

        sub_module = st.radio(
            "Discovery Modules",
            [
                "Narrative Discovery", "Options Activity", "Price Action",
                "EDGAR Scanner", "Institutional (13F)", "Insider & Congress", "Valuation",
            ],
            label_visibility="collapsed",
            key="sub_module",
        )

# --- Module routing ---
if top_level == "⚡ Quick Intel Run":
    from modules.quick_run import render; render()
elif top_level == "Risk Regime":
    from modules.risk_regime import render; render()
elif top_level == "Fed Forecaster":
    from modules.fed_forecaster import render; render()
elif top_level == "Current Events":
    from modules.current_events import render; render()
elif top_level == "Technical Analysis":
    _ew_tab, _wy_tab = st.tabs(["📈 Elliott Wave", "🔬 Wyckoff"])
    with _ew_tab:
        from modules.elliott_wave import render as _ew_render; _ew_render()
    with _wy_tab:
        from modules.wyckoff import render as _wy_render; _wy_render()
elif top_level == "Whale Movement":
    from modules.whale_buyers import render; render()
elif top_level == "Stress Signals":
    from modules.stress_signals import render; render()
elif top_level == "Discovery":
    _sub_icons = {
        "Narrative Discovery": "📡", "Options Activity": "📊", "Price Action": "📈",
        "EDGAR Scanner": "📋", "Institutional (13F)": "🐋",
        "Insider & Congress": "🏛", "Valuation": "💹",
    }
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:11px;letter-spacing:0.06em;margin-bottom:2px;'
        f'display:flex;align-items:center;gap:6px;">'
        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">◉</span>'
        f'<span style="color:{COLORS["text_dim"]}">DISCOVERY</span>'
        f'<span style="color:{COLORS["bloomberg_orange"]}">›</span>'
        f'<span style="color:{COLORS["text"]};font-weight:700;">'
        f'{_sub_icons.get(sub_module, "›")} {(sub_module or "").upper()}</span>'
        f'</div>'
        f'<div style="height:2px;margin-bottom:14px;'
        f'background:linear-gradient(90deg,{COLORS["bloomberg_orange"]},'
        f'{COLORS["bloomberg_orange"]}44,transparent);border-radius:1px;"></div>',
        unsafe_allow_html=True,
    )
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
        from modules.narrative_discovery import render; render()
    elif sub_module == "Price Action":
        from modules.narrative_pulse import render; render()
    elif sub_module == "EDGAR Scanner":
        from modules.edgar_scanner import render; render()
    elif sub_module == "Institutional (13F)":
        from modules.institutional import render; render()
    elif sub_module == "Insider & Congress":
        from modules.insider_congress import render; render()
    elif sub_module == "Options Activity":
        from modules.options_activity import render; render()
    elif sub_module == "Valuation":
        from modules.valuation import render; render()
elif top_level == "Short Squeeze Radar":
    from modules.signal_scorecard import render; render()
elif top_level == "Backtesting":
    from modules.backtesting import render; render()
elif top_level == "My Regarded Portfolio":
    from modules.trade_journal import render; render()
elif top_level == "Signal Audit":
    from modules.signal_audit import render; render()
elif top_level == "Forecast Tracker":
    from modules.forecast_accuracy import render; render()
elif top_level == "Export Hub":
    from modules.export_hub import render; render()
elif top_level == "Alerts":
    from modules.alerts_settings import render; render()

# Persist AI signals to disk on every rerun
try:
    from services.signals_cache import save_signals
    save_signals()
except Exception:
    pass
