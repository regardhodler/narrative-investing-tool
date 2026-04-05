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


def render_rr_score_mode_toggle(
    *,
    key: str,
    help_text: str = "Controls Risk Regime scoring preference for subsequent runs.",
    compact: bool = False,
) -> str:
    """Render the shared Risk Regime scoring-mode toggle and persist selection.

    Returns the internal mode string: "normal" or "ewma_fast".
    """
    _label = st.radio(
        "Mode" if compact else "Core signal scoring mode",
        ["Normal", "Coke Mode"],
        horizontal=True,
        key=key,
        help=help_text,
    )
    st.markdown(
        (
            '<div style="color:#475569;font-size:10px;margin-top:-8px;margin-bottom:2px;">EWMA - fast react</div>'
            if compact else
            '<div style="color:#475569;font-size:10px;margin-top:-4px;">EWMA - fast react</div>'
        ),
        unsafe_allow_html=True,
    )
    _mode = "ewma_fast" if _label == "Coke Mode" else "normal"
    st.session_state["_rr_score_mode"] = _mode
    return _mode


def _validate_core_contracts() -> list[str]:
    """Validate critical cross-module session-state contracts."""
    issues: list[str] = []

    rc = st.session_state.get("_regime_context")
    if rc is not None:
        if not isinstance(rc, dict):
            issues.append("_regime_context is not a dict")
        else:
            for k in ("regime", "score"):
                if k not in rc:
                    issues.append(f"_regime_context missing '{k}'")

    drp = st.session_state.get("_dominant_rate_path")
    if drp is not None and not isinstance(drp, (dict, str)):
        issues.append("_dominant_rate_path has invalid type")

    pa = st.session_state.get("_portfolio_analysis")
    if pa is not None:
        if not isinstance(pa, dict):
            issues.append("_portfolio_analysis is not a dict")
        else:
            for k in ("verdict", "risk_score"):
                if k not in pa:
                    issues.append(f"_portfolio_analysis missing '{k}'")

    db = st.session_state.get("_adversarial_debate")
    if db is not None:
        if not isinstance(db, dict):
            issues.append("_adversarial_debate is not a dict")
        else:
            for k in ("verdict", "confidence"):
                if k not in db:
                    issues.append(f"_adversarial_debate missing '{k}'")

    rs = st.session_state.get("_portfolio_risk_snapshot")
    if rs is not None and not isinstance(rs, dict):
        issues.append("_portfolio_risk_snapshot is not a dict")

    return issues


def _freshness_status() -> tuple[str, str, float]:
    """Return (label, color, confidence_penalty) based on key signal freshness."""
    now = datetime.now()
    ts_keys = [
        "_regime_context_ts", "_rate_path_probs_ts", "_fed_plays_result_ts",
        "_current_events_digest_ts", "_doom_briefing_ts", "_whale_summary_ts",
        "_portfolio_risk_snapshot_ts", "_tactical_context_ts",
    ]
    ages_h = []
    for k in ts_keys:
        ts = st.session_state.get(k)
        if isinstance(ts, datetime):
            ages_h.append((now - ts).total_seconds() / 3600)

    if not ages_h:
        return "UNKNOWN", "#64748b", 0.90

    max_age = max(ages_h)
    if max_age <= 2:
        return "FRESH", "#22c55e", 1.00
    if max_age <= 8:
        return "AGING", "#f59e0b", 0.90
    return "STALE", "#ef4444", 0.80


def apply_confidence_penalty(confidence: int | float) -> int:
    """Apply current freshness penalty to an AI confidence score (1-10)."""
    try:
        raw = float(confidence)
    except Exception:
        raw = 5.0
    penalty = float(st.session_state.get("_intel_conf_penalty", 1.0))
    adjusted = max(1, min(10, int(round(raw * penalty))))
    return adjusted


def render_intel_health_bar(*, compact: bool = False) -> None:
    """Render data freshness + contract health and persist confidence penalty."""
    status, color, penalty = _freshness_status()
    st.session_state["_intel_conf_penalty"] = penalty
    issues = _validate_core_contracts()
    contracts = "OK" if not issues else f"{len(issues)} issue(s)"
    contracts_color = "#22c55e" if not issues else "#ef4444"
    mt = "-6px" if compact else "0"

    st.markdown(
        f'<div style="border:1px solid #1e293b;border-radius:6px;padding:6px 10px;margin:{mt} 0 8px 0;">'
        f'<span style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;">INTEL HEALTH</span>'
        f'<span style="margin-left:10px;font-size:11px;color:{color};font-weight:700;">{status}</span>'
        f'<span style="margin-left:8px;font-size:10px;color:#475569;">conf x{penalty:.2f}</span>'
        f'<span style="margin-left:12px;font-size:10px;color:{contracts_color};">contracts: {contracts}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if issues and not compact:
        st.caption("Contract guardrails: " + " | ".join(issues[:3]))


def render_action_queue(*, max_items: int = 5) -> None:
    """Render a simple top action queue from current cross-module signals."""
    actions: list[tuple[int, str, str]] = []

    rc = st.session_state.get("_regime_context") or {}
    regime = str(rc.get("regime", ""))
    score = float(rc.get("score", 0.0) or 0.0)
    if "Risk-Off" in regime or score < -0.3:
        actions.append((95, "Raise cash by 10-20%", f"Regime {regime or 'risk-off'} ({score:+.2f})"))
        actions.append((90, "Add downside hedge", "Macro trend and stress sensitivity elevated"))
    elif "Risk-On" in regime or score > 0.3:
        actions.append((88, "Buy quality leaders on dips", f"Regime {regime or 'risk-on'} ({score:+.2f})"))

    drp = st.session_state.get("_dominant_rate_path") or {}
    if isinstance(drp, dict):
        scen = str(drp.get("scenario", "")).lower()
        if "hike" in scen:
            actions.append((82, "Reduce duration / rate-sensitive exposure", f"Dominant path: {drp.get('scenario', '')}"))
        elif "cut" in scen:
            actions.append((80, "Lean into cyclicals and duration", f"Dominant path: {drp.get('scenario', '')}"))

    pa = st.session_state.get("_portfolio_analysis") or {}
    for i, act in enumerate((pa.get("priority_actions") or [])[:3]):
        actions.append((85 - i, str(act), "From Portfolio AI priority actions"))

    debate = st.session_state.get("_adversarial_debate") or {}
    if debate.get("verdict") == "CONTESTED" and debate.get("contested_bias"):
        actions.append((78, f"Bias small: {debate.get('contested_bias')}", "Debate contested; reduce size and wait for confirmation"))

    rs = st.session_state.get("_portfolio_risk_snapshot") or {}
    beta = rs.get("beta")
    if isinstance(beta, (int, float)) and beta > 1.25:
        actions.append((84, "Trim high-beta names", f"Portfolio beta {beta:.2f} is elevated"))

    # Deduplicate by action text, keep highest score.
    dedup: dict[str, tuple[int, str]] = {}
    for score_i, action, why in actions:
        prev = dedup.get(action)
        if prev is None or score_i > prev[0]:
            dedup[action] = (score_i, why)

    ranked = sorted(((s, a, w) for a, (s, w) in dedup.items()), reverse=True)[:max_items]
    if not ranked:
        return

    st.markdown(
        '<div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:0.08em;margin:6px 0 4px 0;">TOP ACTION QUEUE</div>',
        unsafe_allow_html=True,
    )
    for idx, (score_i, action, why) in enumerate(ranked, 1):
        st.markdown(
            f'<div style="border-left:3px solid #f97316;background:#0f172a;padding:6px 10px;margin:4px 0;border-radius:0 4px 4px 0;">'
            f'<span style="font-size:11px;color:#e2e8f0;"><b>{idx}. {action}</b></span>'
            f'<span style="font-size:10px;color:#64748b;"> · score {score_i}</span><br>'
            f'<span style="font-size:10px;color:#94a3b8;">{why}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_signal_scorecard() -> None:
    """Bloomberg-style compact signal scorecard — shown at the top of every page.

    Displays all quantified signals + fear composite in a single dark strip.
    Shows greyed placeholders if QIR hasn't run yet.
    """
    rc  = st.session_state.get("_regime_context") or {}
    tac = st.session_state.get("_tactical_context") or {}
    of  = st.session_state.get("_options_flow_context") or {}
    sz  = st.session_state.get("_stress_zscore") or {}
    wf  = st.session_state.get("_whale_flow_score") or {}
    ev  = st.session_state.get("_events_sentiment_score") or {}
    ca  = st.session_state.get("_canary_score") or {}
    fc  = st.session_state.get("_fear_composite") or {}

    qir_run = bool(rc)
    dim = "#334155"
    orange = COLORS.get("bloomberg_orange", "#f59e0b")

    def _cell(label: str, value: str, color: str, sub: str = "") -> str:
        sub_html = f'<div style="font-size:9px;color:#475569;margin-top:1px;">{sub}</div>' if sub else ""
        return (
            f'<div style="padding:6px 10px;border-right:1px solid #1e293b;min-width:90px;">'
            f'<div style="font-size:8px;color:{orange};font-weight:700;letter-spacing:0.08em;">{label}</div>'
            f'<div style="font-size:12px;font-weight:700;color:{color};">{value}</div>'
            f'{sub_html}'
            f'</div>'
        )

    if not qir_run:
        # Pre-run placeholder strip
        cells = "".join([
            _cell("REGIME",   "— run QIR", dim),
            _cell("TACTICAL", "— run QIR", dim),
            _cell("OPT FLOW", "— run QIR", dim),
            _cell("STRESS",   "— run QIR", dim),
            _cell("WHALE",    "— run QIR", dim),
            _cell("EVENTS",   "— run QIR", dim),
            _cell("CANARY",   "— run QIR", dim),
        ])
        fear_html = (
            f'<div style="padding:6px 14px;background:#0f172a;border-left:2px solid #1e293b;">'
            f'<div style="font-size:8px;color:{dim};font-weight:700;letter-spacing:0.08em;">FEAR INDEX</div>'
            f'<div style="font-size:13px;font-weight:800;color:{dim};">— / 100</div>'
            f'</div>'
        )
    else:
        # Regime cell
        macro_s  = int(rc.get("macro_score", 50) or 50)
        regime   = rc.get("regime", "—")
        r_arrow  = "▲" if "Risk-On" in regime else ("▼" if "Risk-Off" in regime else "◆")
        r_color  = "#22c55e" if "Risk-On" in regime else ("#ef4444" if "Risk-Off" in regime else "#f59e0b")
        div_pts  = int(rc.get("leading_divergence", 0) or 0)
        div_col  = "#22c55e" if div_pts > 5 else ("#ef4444" if div_pts < -5 else "#64748b")
        div_sub  = f'{div_pts:+d}pts leading' if div_pts != 0 else "aligned"

        # Tactical cell
        tac_s   = int(tac.get("tactical_score", 50) or 50)
        tac_lbl = tac.get("label", "—")
        t_arrow = "▲" if tac_s >= 55 else ("▼" if tac_s < 45 else "◆")
        t_color = "#22c55e" if tac_s >= 55 else ("#ef4444" if tac_s < 45 else "#f59e0b")

        # Options flow cell
        of_s    = int(of.get("options_score", 50) or 50) if of else 50
        of_lbl  = of.get("label", "—") if of else "—"
        o_color = "#22c55e" if of_s >= 55 else ("#ef4444" if of_s < 45 else "#f59e0b")

        # Stress z-score cell
        sz_z    = sz.get("z", "—")
        sz_pct  = sz.get("pct", "—")
        sz_str  = f'{sz_z:+.1f}σ' if isinstance(sz_z, (int, float)) else "—"
        sz_col  = "#ef4444" if isinstance(sz_z, (int, float)) and sz_z > 1.5 else (
                  "#f59e0b" if isinstance(sz_z, (int, float)) and sz_z > 0.5 else "#22c55e")

        # Whale flow cell
        wf_b    = wf.get("bull_pct", "—")
        wf_lbl  = wf.get("label", "—")
        wf_str  = f'{wf_b:.0f}% bull' if isinstance(wf_b, (int, float)) else "—"
        wf_col  = "#22c55e" if isinstance(wf_b, (int, float)) and wf_b > 55 else (
                  "#ef4444" if isinstance(wf_b, (int, float)) and wf_b < 45 else "#f59e0b")

        # Events sentiment cell
        ev_s    = ev.get("sentiment", "—")
        ev_lbl  = ev.get("label", "—")
        ev_str  = f'{ev_s:+.2f}' if isinstance(ev_s, (int, float)) else "—"
        ev_col  = "#22c55e" if isinstance(ev_s, (int, float)) and ev_s > 0.15 else (
                  "#ef4444" if isinstance(ev_s, (int, float)) and ev_s < -0.15 else "#f59e0b")
        ev_src  = "ai" if ev.get("source") == "ai" else "kw"

        # Canary cell
        ca_s    = ca.get("composite", "—")
        ca_str  = f'{ca_s:.0f}/100' if isinstance(ca_s, (int, float)) else "—"
        ca_col  = "#22c55e" if isinstance(ca_s, (int, float)) and ca_s >= 60 else (
                  "#ef4444" if isinstance(ca_s, (int, float)) and ca_s < 40 else "#f59e0b")

        cells = "".join([
            _cell("REGIME",   f'{r_arrow} {macro_s}', r_color, div_sub),
            _cell("TACTICAL", f'{t_arrow} {tac_s}', t_color, tac_lbl[:12]),
            _cell("OPT FLOW", f'{of_s}', o_color, of_lbl[:12]),
            _cell("STRESS",   sz_str, sz_col, f'{sz_pct}th pct'),
            _cell("WHALE 13F", wf_str, wf_col, wf_lbl[:14]),
            _cell("EVENTS",   ev_str, ev_col, f'{ev_lbl[:12]} [{ev_src}]'),
            _cell("CANARY",   ca_str, ca_col, f'breadth {ca.get("breadth_pct","—")}%'),
        ])

        # Fear composite badge
        fc_s    = fc.get("score", "—")
        fc_lbl  = fc.get("label", "—")
        fc_col  = ("#ef4444" if isinstance(fc_s, (int, float)) and fc_s >= 60 else
                   "#f59e0b" if isinstance(fc_s, (int, float)) and fc_s >= 40 else "#22c55e")
        fear_html = (
            f'<div style="padding:6px 14px;background:#0f172a;border-left:3px solid {fc_col};">'
            f'<div style="font-size:8px;color:{orange};font-weight:700;letter-spacing:0.08em;">FEAR INDEX</div>'
            f'<div style="font-size:15px;font-weight:800;color:{fc_col};">{fc_s:.0f}</div>'
            f'<div style="font-size:9px;color:{fc_col};margin-top:1px;">{fc_lbl}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="display:flex;align-items:stretch;background:#0f172a;'
        f'border:1px solid #1e293b;border-radius:6px;margin-bottom:12px;overflow:hidden;">'
        f'{cells}'
        f'{fear_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
