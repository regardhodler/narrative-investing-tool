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
        padding:10px 20px 10px 20px;
        margin:2.5rem 0 1rem 0;
        display:flex;
        align-items:center;
        justify-content:flex-start;
        gap:16px;
    ">
        <span style="
            font-family:'JetBrains Mono',Consolas,monospace;
            font-size:11px;
            color:#2d2d2d;
            letter-spacing:0.08em;
        ">Jude · Wincyl · Elijah · Eloise · Pakaps</span>
        <span style="
            font-family:'JetBrains Mono',Consolas,monospace;
            font-size:11px;
            color:{COLORS["text_dim"]};
            margin-left:auto;
            padding-right:160px;
            white-space:nowrap;
        ">{_now}</span>
    </div>""", unsafe_allow_html=True)


def render_sidebar_header(narrative: str, ticker: str) -> None:
    """App title + active narrative/ticker status lines."""
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:22px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.08em;margin-bottom:2px;">REGARD TERMINALS</div>'
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
        f'color:#475569;font-style:italic;line-height:1.5;margin-bottom:8px;">'
        f'providing excellent analytics,<br>one losing trade at a time<br>'
        f'<span style="color:#374151;">— by Regardhodler</span></div>',
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
            f'<div style="font-size:9px;color:#334155;margin-top:3px;line-height:1.4;">'
            f'Each dot = 1 day &nbsp;'
            f'<span style="color:#22c55e;">●</span> Risk-On &nbsp;'
            f'<span style="color:#f59e0b;">●</span> Neutral &nbsp;'
            f'<span style="color:#ef4444;">●</span> Risk-Off'
            f'</div>'
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


def render_signal_coverage() -> None:
    """Prompt Context panel — shows which Quick Intel signals are loaded + freshness badges.
    Reusable across Quick Intel Run, Valuation, Discovery, and Portfolio Intelligence."""
    from datetime import datetime as _dt2
    _coverage_signals = [
        ("Regime",          "_regime_context",          "_regime_context_ts"),
        ("Tactical Regime", "_tactical_context",        "_tactical_context_ts"),
        ("Fed Rate Path",   "_dominant_rate_path",      "_rate_path_probs_ts"),
        ("Fed Funds Rate",  "_fed_funds_rate",           None),
        ("Rate-Path Plays", "_fed_plays_result",         "_fed_plays_result_ts"),
        ("Regime Plays",    "_rp_plays_result",          None),
        ("Doom Briefing",   "_doom_briefing",            "_doom_briefing_ts"),
        ("Policy Trans.",   "_chain_narration",          "_chain_narration_ts"),
        ("Black Swans",     "_custom_swans",             "_custom_swans_ts"),
        ("Whale Activity",  "_whale_summary",            "_whale_summary_ts"),
        ("Current Events",  "_current_events_digest",   "_current_events_digest_ts"),
        ("Risk Snapshot",   "_portfolio_risk_snapshot", "_portfolio_risk_snapshot_ts"),
        ("Social Sentiment","_stocktwits_digest",        "_stocktwits_digest_ts"),
    ]
    _now2 = _dt2.now()
    _n_loaded = sum(1 for _, k, _ in _coverage_signals if st.session_state.get(k))
    _bar_pct = int(_n_loaded / len(_coverage_signals) * 100)
    _bar_color = "#22c55e" if _n_loaded == len(_coverage_signals) else ("#f59e0b" if _n_loaded >= 7 else "#ef4444")

    _left = _coverage_signals[:6]
    _right = _coverage_signals[6:]
    _rows_html = ""
    _pad = (None, None, None)
    for (lbl_l, k_l, ts_l), (lbl_r, k_r, ts_r) in zip(_left, _right + [_pad] * (len(_left) - len(_right))):
        def _sig_cell(lbl, k, ts_k):
            if lbl is None:
                return '<td></td>'
            ok = bool(st.session_state.get(k))
            icon = f'<span style="color:#22c55e;">✓</span>' if ok else '<span style="color:#ef4444;">✗</span>'
            age = ""
            if ok and ts_k:
                _ts = st.session_state.get(ts_k)
                if _ts:
                    _m = int((_now2 - _ts).total_seconds() / 60)
                    age = f' <span style="color:#555;font-size:10px;">· {"just now" if _m < 1 else (f"{_m}m ago" if _m < 60 else f"{_m//60}h ago")}</span>'
            color = "#e2e8f0" if ok else "#475569"
            return f'<td style="padding:2px 12px 2px 0;white-space:nowrap;">{icon} <span style="color:{color};">{lbl}</span>{age}</td>'
        _rows_html += f"<tr>{_sig_cell(lbl_l, k_l, ts_l)}{_sig_cell(lbl_r, k_r, ts_r)}</tr>"

    st.markdown(
        f'<div style="border:1px solid #334155;border-radius:6px;padding:10px 14px;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
        f'<span style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase;">Prompt Context</span>'
        f'<span style="font-size:10px;color:{_bar_color};font-weight:600;">{_n_loaded}/{len(_coverage_signals)} signals loaded</span>'
        f'</div>'
        f'<div style="height:2px;background:#1e293b;border-radius:1px;margin-bottom:8px;">'
        f'<div style="height:2px;width:{_bar_pct}%;background:{_bar_color};border-radius:1px;"></div>'
        f'</div>'
        f'<table style="width:100%;font-size:11px;font-family:monospace;border-collapse:collapse;">'
        f'{_rows_html}</table>'
        f'</div>',
        unsafe_allow_html=True,
    )
