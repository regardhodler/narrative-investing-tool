"""Reusable sidebar and header UI components."""

from datetime import datetime

import streamlit as st
from utils.theme import COLORS


def render_header() -> None:
    """Bloomberg-style top header bar with app name and current timestamp."""
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


def render_sidebar_header(narrative: str, ticker: str) -> None:
    """App title + active narrative/ticker status lines."""
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:15px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.08em;margin-bottom:8px;">HIGHLY REGARDED TERMINALS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:4px 0 12px 0;"></div>', unsafe_allow_html=True)

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


def render_macro_events() -> None:
    """FOMC / CPI / NFP countdown cells."""
    try:
        from services.fed_forecaster import get_next_fomc, get_next_cpi, get_next_nfp
        _events = [
            ("FOMC", get_next_fomc()),
            ("CPI",  get_next_cpi()),
            ("NFP",  get_next_nfp()),
        ]
        _cells = ""
        for _label, _ev in _events:
            _days = _ev.get("days_away", 99)
            _date_str = _ev.get("date", "")[:6]
            if _days == 0:
                _c, _bg = "#ef4444", "#2d0a0a"
                _d = "TODAY"
            elif _days == 1:
                _c, _bg = "#f97316", "#1f0d00"
                _d = "TMRW"
            elif _days <= 5:
                _c, _bg = "#f59e0b", "#1a1200"
                _d = f"{_days}d"
            else:
                _c, _bg = "#475569", "#0d1117"
                _d = f"{_days}d"
            _cells += (
                f'<div style="background:{_bg};border:1px solid {_c}44;border-radius:4px;'
                f'padding:4px 6px;text-align:center;">'
                f'<div style="font-size:9px;color:#64748b;letter-spacing:0.08em;">{_label}</div>'
                f'<div style="font-size:11px;font-weight:700;color:{_c};">{_d}</div>'
                f'<div style="font-size:9px;color:#475569;">{_date_str}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin:8px 0;">'
            f'{_cells}</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def render_regime_strip() -> None:
    """30-day regime history dot strip."""
    try:
        from modules.risk_regime import _load_history
        _hist = _load_history()
        if not _hist:
            return
        _recent = sorted(_hist, key=lambda x: x["date"])[-30:]
        _dots = ""
        for _h in _recent:
            _s = _h.get("macro_score", 50)
            _r = _h.get("regime", "")
            if _s >= 60 or "Risk-On" in _r:
                _dc = "#22c55e"
            elif _s <= 40 or "Risk-Off" in _r:
                _dc = "#ef4444"
            else:
                _dc = "#f59e0b"
            _dots += (
                f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
                f'background:{_dc};margin:1px;" title="{_h["date"]}"></span>'
            )
        _last = _recent[-1]
        _last_score = _last.get("macro_score", 50)
        _rc = "#22c55e" if _last_score >= 60 else ("#ef4444" if _last_score <= 40 else "#f59e0b")
        st.markdown(
            f'<div style="margin:6px 0 4px 0;">'
            f'<div style="font-size:9px;color:#475569;letter-spacing:0.08em;margin-bottom:3px;">REGIME · 30D</div>'
            f'<div style="line-height:1;">{_dots}</div>'
            f'<div style="font-size:10px;color:{_rc};font-weight:700;margin-top:4px;">'
            f'{_last.get("regime", "")} · {_last_score:.0f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def render_watchlist_widget() -> None:
    """Watchlist selectbox with GO and DEL actions."""
    from utils.watchlist import load_watchlist, remove_from_watchlist
    from utils.session import set_ticker

    watchlist = load_watchlist()
    if not watchlist:
        return

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
            set_ticker(selected_wl.split(" — ")[0])
            st.session_state["top_module"] = "Discovery"
            st.session_state["sub_module"] = "Narrative Discovery"
            st.rerun()
    with col_rm:
        if st.button("DEL", key="wl_rm"):
            remove_from_watchlist(selected_wl.split(" — ")[0])
            st.rerun()
    st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:8px 0;"></div>', unsafe_allow_html=True)
