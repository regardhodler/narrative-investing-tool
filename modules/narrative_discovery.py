import plotly.graph_objects as go
import streamlit as st

from services.trends_client import (
    get_interest_over_time,
    get_interest_over_time_multi,
    get_trending_searches,
    get_yahoo_trending_tickers,
)
from services.claude_client import classify_narrative, describe_company, group_tickers_by_narrative
from services.market_data import fetch_batch_safe
from services.sec_client import get_company_info, search_ticker_by_name
from utils.session import get_ticker, set_narrative, set_ticker
from utils.watchlist import add_to_watchlist, is_in_watchlist
from utils.theme import COLORS, apply_dark_layout
from utils.ai_tier import render_ai_tier_selector, TIER_OPTS, TIER_MAP, MODEL_HINT_HTML

ASSET_CLASSES = {
    "Equities": None,  # sentinel — use Yahoo trending
    "Commodities": {
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "WTI Crude",
        "NG=F": "Natural Gas", "HG=F": "Copper",
        "ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans",
    },
    "Bonds": {
        "TLT": "20Y+ Treasury", "IEF": "10Y Treasury", "SHY": "2Y Treasury",
        "LQD": "IG Corporate", "HYG": "High Yield Corp",
        "TIP": "TIPS", "AGG": "US Agg Bond",
    },
    "Currencies": {
        "UUP": "USD Bull", "FXE": "Euro", "FXY": "Yen",
        "FXB": "GBP", "FXA": "AUD", "FXC": "CAD",
    },
}

_ASSET_BADGE = {
    "Commodities": ("CMDTY", COLORS["orange"]),
    "Bonds": ("BOND", COLORS["blue"]),
    "Currencies": ("FX", COLORS["yellow"]),
}

# Lookup: ticker → asset class name (for non-equity classes)
_TICKER_TO_CLASS = {}
for _cls, _tickers in ASSET_CLASSES.items():
    if _tickers is not None:
        for _t in _tickers:
            _TICKER_TO_CLASS[_t] = _cls

_NON_FINANCIAL_KEYWORDS = {
    "nfl", "nba", "mlb", "nhl", "ufc", "wwe", "espn", "super bowl",
    "playoff", "draft pick", "touchdown", "goalkeeper", "premier league",
    "champions league", "world cup", "olympics", "quarterback",
    "kardashian", "taylor swift", "beyonce", "drake", "kanye",
    "bachelor", "bachelorette", "survivor", "big brother",
    "grammys", "oscars", "emmys", "golden globes",
    "tiktok trend", "meme", "viral video", "celebrity",
    "wordle", "fortnite", "minecraft",
}


@st.cache_data(ttl=3600)
def _get_macro_context_for_plays() -> dict:
    """Collect enriched macro regime + stress + credit signals for the sector plays expander."""
    from services.market_data import fetch_fred_series_safe, fetch_batch_safe

    # HY credit spread
    try:
        hy_s = fetch_fred_series_safe("BAMLH0A0HYM2")
        hy_v = float(hy_s.iloc[-1]) if hy_s is not None and len(hy_s) else 0.0
    except Exception:
        hy_v = 0.0

    # Yield curve (10Y-2Y)
    try:
        yc_s = fetch_fred_series_safe("T10Y2Y")
        yc_v = float(yc_s.iloc[-1]) if yc_s is not None and len(yc_s) else 0.0
    except Exception:
        yc_v = 0.0

    # NFCI — National Financial Conditions Index
    try:
        nfci_s = fetch_fred_series_safe("NFCI")
        nfci_v = float(nfci_s.iloc[-1]) if nfci_s is not None and len(nfci_s) else 0.0
    except Exception:
        nfci_v = 0.0

    # Core PCE — for inflation direction (3-month trend)
    try:
        pce_s = fetch_fred_series_safe("PCEPILFE")
        if pce_s is not None and len(pce_s) >= 4:
            pce_3m = float(pce_s.iloc[-1]) - float(pce_s.iloc[-4])
        else:
            pce_3m = 0.0
    except Exception:
        pce_3m = 0.0

    # VIX, SPY, HYG (HY bond ETF), LQD (IG bond ETF)
    try:
        snaps = fetch_batch_safe({"^VIX": "VIX", "SPY": "S&P 500", "HYG": "HY Bond", "LQD": "IG Bond"}, "5d", "1d")
        vix_v = snaps["^VIX"].latest_price or 20.0
        spy_1d = snaps["SPY"].pct_change_1d or 0.0
        hyg_price = snaps["HYG"].latest_price
        lqd_price = snaps["LQD"].latest_price
        hyg_lqd = round(hyg_price / lqd_price, 4) if hyg_price and lqd_price and lqd_price > 0 else None
    except Exception:
        vix_v, spy_1d, hyg_lqd = 20.0, 0.0, None

    # Derived labels
    stress = "HIGH" if (hy_v > 500 or vix_v > 35) else "ELEVATED" if (hy_v > 300 or vix_v > 25) else "CALM"
    regime = (
        "Risk-On" if (yc_v > 0 and vix_v < 20 and hy_v < 300) else
        "Risk-Off" if (vix_v > 25 or hy_v > 400) else "Neutral"
    )
    score = round(max(-1.0, min(1.0,
        (1 - vix_v / 40) * 0.5 + (yc_v / 2) * 0.3 + (1 - hy_v / 600) * 0.2
    )), 2)

    # Dalio quadrant: growth direction × inflation direction
    growth_dir = "Rising" if (yc_v > 0.2 and vix_v < 22) else "Falling"
    inflation_dir = "Rising" if pce_3m > 0 else "Falling"
    if growth_dir == "Rising" and inflation_dir == "Falling":
        quadrant = "Goldilocks"
    elif growth_dir == "Rising" and inflation_dir == "Rising":
        quadrant = "Reflation"
    elif growth_dir == "Falling" and inflation_dir == "Rising":
        quadrant = "Stagflation"
    else:
        quadrant = "Deflation"

    # Credit risk composite
    hyg_lqd_low = hyg_lqd is not None and hyg_lqd < 0.95
    credit_risk = (
        "HIGH" if (hy_v > 400 or nfci_v > 0.5 or hyg_lqd_low) else
        "MODERATE" if (hy_v > 300 or nfci_v > 0.0) else "LOW"
    )

    return {
        "regime": regime, "score": score, "stress": stress,
        "quadrant": quadrant, "credit_risk": credit_risk,
        "vix": round(vix_v, 1), "hy_spread": round(hy_v, 1),
        "yield_curve": round(yc_v, 2), "spy_1d": round(spy_1d, 2),
        "nfci": round(nfci_v, 2), "hyg_lqd": hyg_lqd,
        "pce_3m": round(pce_3m, 4),
    }


def _conviction_stars(n: int) -> str:
    n = max(1, min(3, int(n)))
    return "★" * n + "☆" * (3 - n)


def _render_trending_narratives():
    """AI-powered trending narrative discovery — Google Trends + news frequency synthesis."""
    import os
    from services.claude_client import discover_trending_narratives
    from services.news_feed import fetch_financial_headlines
    from services.trends_client import get_trending_searches

    _oc = COLORS["bloomberg_orange"]
    st.markdown(
        f'<div style="font-size:13px;color:{_oc};font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:4px;">🔥 TRENDING NARRATIVES</div>',
        unsafe_allow_html=True,
    )

    _has_xai = bool(os.getenv("XAI_API_KEY"))
    _has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

    col_e, col_t = st.columns([2, 1])
    with col_e:
        _use_claude, _model = render_ai_tier_selector(
            key="disc_trend_engine",
            label="Engine",
            recommendation="⚡ Freeloader for daily trend scanning · 🧠 Regard for deeper narrative grouping",
        )
    with col_t:
        _tf = st.radio("Timeframe", ["1W", "1M", "3M"], horizontal=True, key="disc_trend_tf")
    # Show cached result if available
    _cached = st.session_state.get("_trending_narratives")
    _cached_ts = st.session_state.get("_trending_narratives_ts")
    _cached_tf = st.session_state.get("_trending_narratives_tf")

    _auto_refresh_fired = False
    if _cached_ts:
        from datetime import datetime as _dt2
        _age_m = int((_dt2.now() - _cached_ts).total_seconds() / 60)
        _age_str = f"{_age_m}m ago" if _age_m < 60 else f"{_age_m // 60}h ago"
        st.caption(f"Last scan {_age_str} · {_cached_tf or _tf}")
        # Auto-refresh if cache is ≥4h old
        if _age_m >= 240 and not st.session_state.get("_narr_auto_refresh_running"):
            st.session_state["_narr_auto_refresh_running"] = True
            _auto_refresh_fired = True
            st.info("♻️ Narratives are 4h+ old — auto-refreshing...")

    if st.button("🔥 Scan Trending Narratives", key="disc_trend_scan", type="primary") or _auto_refresh_fired:
        with st.spinner("Fetching headlines + Google Trends..."):
            _headlines_raw = fetch_financial_headlines()
            _headline_strs = [f"[{h['source']}] {h['title']}" for h in _headlines_raw[:40]]
            try:
                if _tf == "1W":
                    # Real-time trending searches for short-term
                    _trends = get_trending_searches()
                else:
                    # Interest-over-time for key finance themes over the chosen window
                    from services.trends_client import get_interest_over_time_multi
                    _finance_keywords = (
                        "gold", "oil price", "interest rates", "AI stocks", "inflation",
                    )
                    _trends_df = get_interest_over_time_multi(_finance_keywords, timeframe=_tf)
                    if not _trends_df.empty:
                        # Return keywords sorted by average interest descending
                        _avgs = {k: _trends_df[k].mean() for k in _finance_keywords if k in _trends_df.columns}
                        _trends = [f"{k} (avg interest: {v:.0f}/100)" for k, v in sorted(_avgs.items(), key=lambda x: -x[1])]
                    else:
                        _trends = get_trending_searches()
                # Filter out obvious non-financial terms
                _trends = [t for t in _trends if not any(
                    kw in t.lower() for kw in _NON_FINANCIAL_KEYWORDS
                )][:25]
            except Exception:
                _trends = []

        _macro = dict(st.session_state.get("_regime_context") or {})
        # Inject tactical context so AI can prioritize narratives suited to current entry conditions
        from services.claude_client import _fmt_tactical_ctx as _fmt_tac_nd
        _tac_nd = _fmt_tac_nd(st.session_state.get("_tactical_context"))
        if _tac_nd:
            _macro["tactical_context"] = _tac_nd

        with st.spinner("AI synthesizing top narratives..."):
            _result = discover_trending_narratives(
                headlines=_headline_strs,
                trends=_trends,
                macro_context=_macro,
                timeframe=_tf,
                use_claude=_use_claude,
                model=_model,
            )

        if _result:
            from datetime import datetime as _dt2
            st.session_state["_trending_narratives"] = _result
            st.session_state["_trending_narratives_ts"] = _dt2.now()
            st.session_state["_trending_narratives_tf"] = _tf
            st.session_state.pop("_narr_auto_refresh_running", None)
            _cached = _result
        else:
            st.session_state.pop("_narr_auto_refresh_running", None)
            st.error("Narrative scan failed — check API keys or try Groq.")

    if _cached:
        _conv_colors = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#64748b"}
        _cat_colors = {
            "macro": "#3b82f6", "sector": _oc, "commodity": "#f59e0b",
            "geopolitical": "#ef4444", "tech": "#8b5cf6",
        }
        # Read tactical + options flow context once for all cards
        _tac_nd = st.session_state.get("_tactical_context") or {}
        _tac_badge_html = ""
        if _tac_nd:
            _tac_score = _tac_nd.get("tactical_score", 50)
            _tac_lbl = _tac_nd.get("label", "")
            _tac_icon = "⚡" if _tac_score >= 65 else ("⏸" if _tac_score >= 52 else ("⚠" if _tac_score >= 38 else "🔴"))
            _tac_color = "#22c55e" if _tac_score >= 65 else ("#f59e0b" if _tac_score >= 38 else "#ef4444")
            _tac_badge_html = (
                f'<span style="background:{_tac_color}18;border:1px solid {_tac_color}44;border-radius:3px;'
                f'padding:1px 7px;font-size:10px;color:{_tac_color};">{_tac_icon} {_tac_lbl}</span>'
            )
        _of_nd = st.session_state.get("_options_flow_context") or {}
        _of_badge_html = ""
        if _of_nd:
            _of_score = _of_nd.get("options_score", 50)
            _of_lbl = _of_nd.get("label", "")
            _of_icon = "📈" if _of_score >= 65 else ("➡" if _of_score >= 52 else ("📉" if _of_score >= 38 else "🔴"))
            _of_color = "#22c55e" if _of_score >= 65 else ("#f59e0b" if _of_score >= 38 else "#ef4444")
            _of_badge_html = (
                f'<span style="background:{_of_color}18;border:1px solid {_of_color}44;border-radius:3px;'
                f'padding:1px 7px;font-size:10px;color:{_of_color};">{_of_icon} {_of_lbl}</span>'
            )
        # Load open position tickers once for "You hold" badge on each card
        try:
            from utils.journal import load_journal as _load_j_nd
            _open_tickers_nd = {t["ticker"].upper() for t in _load_j_nd() if t.get("status") == "open"}
        except Exception:
            _open_tickers_nd = set()

        for _n in _cached:
            _name = _n.get("narrative", "")
            _ev = _n.get("evidence", "")
            _tks = _n.get("tickers", [])
            _conv = _n.get("conviction", "MEDIUM")
            _tf_label = "Short-term" if _n.get("timeframe") == "short" else "Medium-term"
            _cat = _n.get("category", "macro")
            _cc = _conv_colors.get(_conv, "#888")
            _catc = _cat_colors.get(_cat, "#888")
            # "You hold" badge — blue if any open position tickers match this narrative
            _held_tickers = [t for t in _tks if t.upper() in _open_tickers_nd]
            _held_badge_html = ""
            if _held_tickers:
                _held_str = ", ".join(_held_tickers[:3]) + ("…" if len(_held_tickers) > 3 else "")
                _held_badge_html = (
                    f'<span style="background:#1e3a5f;border:1px solid #3b82f644;border-radius:3px;'
                    f'padding:1px 7px;font-size:10px;color:#60a5fa;">📂 {_held_str}</span>'
                )

            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #1e293b;border-left:3px solid {_catc};'
                f'border-radius:4px;padding:10px 14px;margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
                f'<span style="color:{_oc};font-weight:700;font-size:13px;">{_name}</span>'
                f'<div style="display:flex;gap:6px;flex-wrap:wrap;">'
                f'<span style="background:#1e293b;border-radius:3px;padding:1px 7px;font-size:10px;color:{_catc};">{_cat}</span>'
                f'<span style="background:#1e293b;border-radius:3px;padding:1px 7px;font-size:10px;color:{_cc};">{_conv}</span>'
                f'<span style="background:#1e293b;border-radius:3px;padding:1px 7px;font-size:10px;color:#475569;">{_tf_label}</span>'
                + (_tac_badge_html if _tac_badge_html else "")
                + (_of_badge_html if _of_badge_html else "")
                + (_held_badge_html if _held_badge_html else "")
                + f'</div></div>'
                f'<div style="color:#94a3b8;font-size:11px;margin-bottom:6px;">{_ev}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:4px;">'
                + "".join(
                    f'<span style="background:#1e293b;border:1px solid {_oc}44;border-radius:3px;'
                    f'padding:2px 8px;font-size:11px;font-weight:700;color:{_oc};cursor:pointer;">{t}</span>'
                    for t in _tks
                )
                + f'</div></div>',
                unsafe_allow_html=True,
            )
            # Clickable ticker buttons
            if _tks:
                _btn_cols = st.columns(len(_tks[:5]))
                for _bi, _bt in enumerate(_tks[:5]):
                    with _btn_cols[_bi]:
                        if st.button(_bt, key=f"tn_tk_{_name[:8]}_{_bi}"):
                            set_ticker(_bt)
                            set_narrative(_name)
                            st.rerun()

    st.markdown(
        f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>',
        unsafe_allow_html=True,
    )


def render():
    st.header("NARRATIVE DISCOVERY")

    # ── Signal Coverage ────────────────────────────────────────────────────────
    from utils.components import render_signal_coverage as _render_sig_cov_disc
    _render_sig_cov_disc()

    # ── Sector Rotation Strip ──────────────────────────────────────────────
    try:
        from services.sector_rotation import get_sector_momentum, QUADRANT_ALIGNMENT
        from utils.theme import COLORS as _DISC_COLORS
        _quadrant_disc = st.session_state.get("_regime_context", {}).get("quadrant", "")
        _sectors_disc  = get_sector_momentum()
        if _sectors_disc and _quadrant_disc:
            _aln_disc = set(QUADRANT_ALIGNMENT.get(_quadrant_disc, []))
            _top_disc  = _sectors_disc[:5]
            _cells_disc = []
            for _s in _top_disc:
                _ret = _s.get("ret_4w")
                _ret_str = f"{_ret:+.1f}%" if _ret is not None else "—"
                _is_aln = _s["ticker"] in _aln_disc
                _ret_col = "#22c55e" if (_ret or 0) >= 0 else "#ef4444"
                _border  = "1px solid #22c55e66" if _is_aln else "1px solid #1e293b"
                _cells_disc.append(
                    f'<div style="display:inline-block;border:{_border};border-radius:4px;'
                    f'padding:4px 10px;margin-right:6px;background:#0f172a;">'
                    f'<span style="font-size:11px;font-weight:700;color:#f97316;">{_s["ticker"]}</span>'
                    f'<span style="font-size:10px;color:#64748b;"> {_s["name"]}</span>'
                    f'<span style="font-size:11px;color:{_ret_col};font-weight:600;margin-left:4px;">{_ret_str}</span>'
                    + (' <span style="font-size:9px;color:#22c55e;">✓</span>' if _is_aln else '')
                    + '</div>'
                )
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<span style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
                f'margin-right:8px;">🔄 TOP SECTORS</span>'
                + "".join(_cells_disc)
                + f'<span style="font-size:10px;color:#334155;margin-left:6px;">'
                f'{_quadrant_disc} · ✓ regime-aligned</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # ── Trending Narratives (AI-powered) ──────────────────────────────────
    _render_trending_narratives()

    # ── Cross-Signal Macro Plays ───────────────────────────────────────────
    import os
    _has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    _PLAY_MODEL_OPTIONS = ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"] if _has_api_key else ["⚡ Freeloader Mode"]
    _PLAY_MODEL_MAP = {
        "⚡ Freeloader Mode": (False, None),
        "🧠 Regard Mode": (True, "grok-4-1-fast-reasoning"),
        "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
    }

    with st.expander("📡 Cross-Signal Macro Plays", expanded=False):
        _macro_ctx = _get_macro_context_for_plays()
        _regime = _macro_ctx["regime"]
        _stress = _macro_ctx["stress"]
        _quadrant = _macro_ctx["quadrant"]
        _credit_risk = _macro_ctx["credit_risk"]
        _vix = _macro_ctx["vix"]
        _hy = _macro_ctx["hy_spread"]
        _yc = _macro_ctx["yield_curve"]
        _nfci = _macro_ctx["nfci"]
        _hyg_lqd = _macro_ctx["hyg_lqd"]

        # Macro snapshot bar — enriched with quadrant + credit risk + rate path + fed rate
        _stress_color = {"HIGH": "#ef4444", "ELEVATED": "#f59e0b", "CALM": "#22c55e"}.get(_stress, "#888")
        _regime_color = {"Risk-On": "#22c55e", "Risk-Off": "#ef4444"}.get(_regime, "#f59e0b")
        _quadrant_color = {
            "Goldilocks": "#22c55e", "Reflation": "#f59e0b",
            "Stagflation": "#ef4444", "Deflation": "#60a5fa",
        }.get(_quadrant, "#888")
        _credit_color = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#22c55e"}.get(_credit_risk, "#888")
        _hyg_lqd_str = f"{_hyg_lqd:.3f}" if _hyg_lqd is not None else "N/A"

        # Rate path + Fed rate from session state (populated when Fed Forecaster runs)
        _dom_rp = st.session_state.get("_dominant_rate_path", {})
        _fed_rate_snap = st.session_state.get("_fed_funds_rate")
        _snap_pill_labels = {
            "cut_25": "25bp cut", "cut_50": "50bp cut",
            "hold": "Hold", "hike_25": "25bp hike",
        }
        _rp_scenario_snap = _dom_rp.get("scenario", "")
        _rp_prob_snap = _dom_rp.get("prob_pct", 0)
        _rp_display_snap = f"{_snap_pill_labels.get(_rp_scenario_snap, _rp_scenario_snap)} {_rp_prob_snap:.0f}%" if _rp_scenario_snap else "—"
        _rp_color_snap = {"cut_25": "#22c55e", "cut_50": "#22c55e", "hold": "#f59e0b", "hike_25": "#ef4444"}.get(_rp_scenario_snap, "#888")
        _fed_rate_display = f"{_fed_rate_snap:.2f}%" if _fed_rate_snap is not None else "—"

        st.markdown(
            f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;font-size:13px;">'
            f'<span>Regime: <b style="color:{_regime_color}">{_regime}</b></span>'
            f'<span>Quadrant: <b style="color:{_quadrant_color}">{_quadrant}</b></span>'
            f'<span>Stress: <b style="color:{_stress_color}">{_stress}</b></span>'
            f'<span>Credit Risk: <b style="color:{_credit_color}">{_credit_risk}</b></span>'
            f'<span>Rate Path: <b style="color:{_rp_color_snap}">{_rp_display_snap}</b></span>'
            f'</div>'
            f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px;font-size:12px;color:#94a3b8;">'
            f'<span>VIX: <b style="color:#e2e8f0">{_vix}</b></span>'
            f'<span>HY Spread: <b style="color:#e2e8f0">{_hy}bps</b></span>'
            f'<span>Yield Curve: <b style="color:#e2e8f0">{_yc:+.2f}%</b></span>'
            f'<span>NFCI: <b style="color:#e2e8f0">{_nfci:+.2f}</b></span>'
            f'<span>HYG/LQD: <b style="color:#e2e8f0">{_hyg_lqd_str}</b></span>'
            f'<span>Fed Rate: <b style="color:#e2e8f0">{_fed_rate_display}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Engine selector
        _use_cl_play, _cl_model_play = render_ai_tier_selector(
            key="play_engine_radio",
            label="Engine",
            recommendation="🧠 Regard recommended for trade play generation — Grok 4.1 reasoning improves entry/exit logic",
        )

        # ── Stress-Test Overlay (optional scenario input) ─────────────────
        st.markdown(
            '<p style="font-size:13px;font-weight:600;margin:8px 0 2px 0;">🎯 Stress-Test Overlay</p>',
            unsafe_allow_html=True,
        )
        st.caption("Optional — layer a macro shock on current signals to stress-test your plays")
        # Pick up any quick-fill value BEFORE the widget is instantiated
        _qs_pending = st.session_state.pop("_pending_overlay_input", None)
        if _qs_pending is not None:
            st.session_state["overlay_scenario_input"] = _qs_pending
        _overlay_scenario = st.text_input(
            "Stress-Test Overlay",
            placeholder="e.g. Reverse yen carry trade, Strait of Hormuz closure, US credit downgrade",
            label_visibility="collapsed",
            key="overlay_scenario_input",
        )
        _pre_swans = st.session_state.get("_custom_swans", {})
        if _pre_swans:
            st.caption("Quick-fill from analyzed Black Swan events:")
            _qs_cols = st.columns(min(len(_pre_swans), 3))
            for _qsi, _qslabel in enumerate(list(_pre_swans.keys())[:3]):
                with _qs_cols[_qsi]:
                    _short = (_qslabel[:26] + "…") if len(_qslabel) > 26 else _qslabel
                    _qsprob = _pre_swans[_qslabel].get("probability_pct", 0)
                    if st.button(f"{_short} ({_qsprob:.0f}%)", key=f"qs_swan_{_qsi}"):
                        st.session_state["_pending_overlay_input"] = _qslabel
                        st.rerun()

        # ── Upstream context checklist ────────────────────────────────────
        def _age_label(ts_key: str) -> str:
            from datetime import datetime as _dt
            _ts = st.session_state.get(ts_key)
            if not _ts:
                return ""
            _mins = int((_dt.now() - _ts).total_seconds() / 60)
            if _mins < 1: return " · just now"
            if _mins < 60: return f" · {_mins}m ago"
            return f" · {_mins // 60}h ago"

        _dom_rp_scenario = st.session_state.get("_dominant_rate_path", {}).get("scenario", "")
        _dom_rp_prob = st.session_state.get("_dominant_rate_path", {}).get("prob_pct", 0)
        _dp_labels = {"cut_25": "25bp Cut", "cut_50": "50bp Cut", "hold": "Hold", "hike_25": "25bp Hike"}
        _dom_rp_label = _dp_labels.get(_dom_rp_scenario, _dom_rp_scenario)
        _bs_count = len(st.session_state.get("_custom_swans", {}))
        _ff_rate = st.session_state.get("_fed_funds_rate")

        _of_ctx_nd = st.session_state.get("_options_flow_context") or {}
        _tac_ctx_nd = st.session_state.get("_tactical_context") or {}
        _ctx_signals = [
            (
                bool(st.session_state.get("_regime_context")),
                "Regime",
                f"{st.session_state.get('_regime_context', {}).get('regime', '')} · "
                f"score {st.session_state.get('_regime_context', {}).get('score', 0):+.2f} · "
                f"{st.session_state.get('_regime_context', {}).get('quadrant', '')}",
                "_regime_context_ts",
            ),
            (
                bool(_tac_ctx_nd),
                "Tactical Regime",
                f"{_tac_ctx_nd.get('tactical_score', '')}/100 · {_tac_ctx_nd.get('label', '')}" if _tac_ctx_nd else "",
                "_tactical_context_ts",
            ),
            (
                bool(st.session_state.get("_dominant_rate_path")),
                "Fed Rate Path",
                f"{_dom_rp_label} ({_dom_rp_prob:.0f}%)" if _dom_rp_label else "",
                "_rate_path_probs_ts",
            ),
            (
                _ff_rate is not None,
                "Fed Funds Rate",
                f"{_ff_rate:.2f}%" if _ff_rate is not None else "",
                None,
            ),
            (
                bool(st.session_state.get("_fed_plays_result")),
                "Rate-Path Plays",
                st.session_state.get("_fed_plays_engine", ""),
                "_fed_plays_result_ts",
            ),
            (
                bool(st.session_state.get("_rp_plays_result")),
                "Regime Plays",
                st.session_state.get("_rp_plays_last_tier", ""),
                None,
            ),
            (
                bool(st.session_state.get("_doom_briefing")),
                "Doom Briefing",
                st.session_state.get("_doom_briefing_engine", ""),
                "_doom_briefing_ts",
            ),
            (
                bool(st.session_state.get("_chain_narration")),
                "Policy Trans.",
                "",
                None,
            ),
            (
                bool(st.session_state.get("_custom_swans")),
                "Black Swans",
                f"{_bs_count} event(s)" if _bs_count else "",
                "_custom_swans_ts",
            ),
            (
                bool(st.session_state.get("_whale_summary")),
                "Whale Activity",
                "",
                "_whale_summary_ts",
            ),
            (
                bool(st.session_state.get("_activism_digest")),
                "Activism (13D)",
                st.session_state.get("_activism_digest_engine", ""),
                "_activism_digest_ts",
            ),
            (
                bool(st.session_state.get("_sector_regime_digest")),
                "Sector×Regime",
                st.session_state.get("_sector_regime_digest_engine", ""),
                "_sector_regime_digest_ts",
            ),
            (
                bool(st.session_state.get("_current_events_digest")),
                "Current Events",
                st.session_state.get("_current_events_engine", ""),
                "_current_events_digest_ts",
            ),
            (
                bool(_of_ctx_nd),
                "Opt Flow",
                f"score {_of_ctx_nd.get('options_score', 0)}/100 · {_of_ctx_nd.get('label', '')}" if _of_ctx_nd else "",
                "_options_flow_context_ts",
            ),
            (
                bool(st.session_state.get("_portfolio_risk_snapshot")),
                "Risk Snapshot",
                "",
                "_portfolio_risk_snapshot_ts",
            ),
        ]

        _n_loaded = sum(1 for ok, *_ in _ctx_signals if ok)
        _total = len(_ctx_signals)
        _bar_pct = int(_n_loaded / _total * 100)
        _bar_color = "#22c55e" if _n_loaded >= 7 else ("#f59e0b" if _n_loaded >= 4 else "#ef4444")

        # Build 2-column checklist rows (dynamic mid-split)
        _rows_html = ""
        _mid = len(_ctx_signals) // 2
        _left = _ctx_signals[:_mid]
        _right = _ctx_signals[_mid:]
        for (ok_l, label_l, detail_l, ts_l), (ok_r, label_r, detail_r, ts_r) in zip(_left, _right):
            _icon_l = f'<span style="color:#22c55e;">✓</span>' if ok_l else '<span style="color:#ef4444;">✗</span>'
            _icon_r = f'<span style="color:#22c55e;">✓</span>' if ok_r else '<span style="color:#ef4444;">✗</span>'
            _detail_l = f'<span style="color:#555;font-size:10px;"> {detail_l}{_age_label(ts_l) if ts_l else ""}</span>' if (ok_l and detail_l) else (f'<span style="color:#555;font-size:10px;">{_age_label(ts_l)}</span>' if (ok_l and ts_l) else "")
            _detail_r = f'<span style="color:#555;font-size:10px;"> {detail_r}{_age_label(ts_r) if ts_r else ""}</span>' if (ok_r and detail_r) else (f'<span style="color:#555;font-size:10px;">{_age_label(ts_r)}</span>' if (ok_r and ts_r) else "")
            _col_l = f'<td style="padding:2px 12px 2px 0;white-space:nowrap;">{_icon_l} <span style="color:{"#e2e8f0" if ok_l else "#475569"};">{label_l}</span>{_detail_l}</td>'
            _col_r = f'<td style="padding:2px 0;">{_icon_r} <span style="color:{"#e2e8f0" if ok_r else "#475569"};">{label_r}</span>{_detail_r}</td>'
            _rows_html += f"<tr>{_col_l}{_col_r}</tr>"

        # Compact amber banner for missing critical signals only
        _dq_miss = [l for k, l in [("_regime_context", "Risk Regime"), ("_dominant_rate_path", "Fed Rate Path"), ("_rate_path_probs", "Rate Probs")] if not st.session_state.get(k)]
        if _dq_miss:
            st.markdown(
                f'<div style="background:#1a0d00;border:1px solid #f59e0b55;border-radius:6px;'
                f'padding:8px 14px;margin-bottom:8px;font-size:11px;">'
                f'<span style="color:#f59e0b;font-weight:700;">⚠ Data Quality</span>'
                f'<span style="color:#94a3b8;margin-left:8px;"><b>Missing:</b> {", ".join(_dq_miss)}</span>'
                f'<span style="color:#64748b;margin-left:8px;">— run ⚡ Quick Intel Run to refresh</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="border:1px solid #334155;border-radius:6px;padding:10px 14px;margin-bottom:8px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#64748b;text-transform:uppercase;">Prompt Context</span>'
            f'<span style="font-size:10px;color:{_bar_color};font-weight:600;">{_n_loaded}/{_total} signals loaded</span>'
            f'</div>'
            f'<div style="height:2px;background:#1e293b;border-radius:1px;margin-bottom:8px;">'
            f'<div style="height:2px;width:{_bar_pct}%;background:{_bar_color};border-radius:1px;"></div>'
            f'</div>'
            f'<table style="width:100%;font-size:11px;font-family:monospace;border-collapse:collapse;">'
            f'{_rows_html}</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Generate Plays ────────────────────────────────────────────────
        _gen_plays = st.button("Generate Plays", type="primary", key="gen_plays_btn")

        if _gen_plays or st.session_state.get("_plays_result"):
            if _gen_plays:
                _use_cl, _cl_model = _use_cl_play, _cl_model_play
                _signal_summary = (
                    f"VIX: {_vix}, HY Spread: {_hy}bps, Yield Curve: {_yc:+.2f}%, "
                    f"System Stress: {_stress}, SPY 1-day: {_macro_ctx['spy_1d']:+.2f}%, "
                    f"Dalio Quadrant: {_quadrant}, Credit Risk: {_credit_risk}, "
                    f"NFCI: {_nfci:+.2f}"
                    + (f", HYG/LQD Ratio: {_hyg_lqd:.3f}" if _hyg_lqd else "")
                )

                # ── Enrich with cached Regime + Fed Forecaster context ────────
                _cached_regime_ctx  = st.session_state.get("_regime_context")
                _cached_fed_plays   = st.session_state.get("_fed_plays_result")
                _cached_rp_tier     = st.session_state.get("_regime_plays_tier", "")
                _enrichment_parts   = []

                if _cached_regime_ctx:
                    _enrichment_parts.append(
                        f"[Regime AI ({_cached_rp_tier}): "
                        f"regime={_cached_regime_ctx['regime']}, "
                        f"score={_cached_regime_ctx.get('score', 0.0):+.2f}, "
                        f"quadrant={_cached_regime_ctx.get('quadrant', 'Unknown')}, "
                        f"context={_cached_regime_ctx['signal_summary']}]"
                    )

                if _cached_fed_plays:
                    _fp_sectors   = ", ".join(s.get("name", "") for s in _cached_fed_plays.get("sectors", [])[:4])
                    _fp_stocks    = ", ".join(s.get("ticker", "") for s in _cached_fed_plays.get("stocks", [])[:4])
                    _fp_bonds     = ", ".join(s.get("ticker", "") for s in _cached_fed_plays.get("bonds", [])[:3])
                    _fp_rationale = _cached_fed_plays.get("rationale", "")
                    _enrichment_parts.append(
                        f"[Rate-Path AI Plays: "
                        f"Favored sectors: {_fp_sectors}; "
                        f"Stocks: {_fp_stocks}; "
                        f"Bonds/Macro: {_fp_bonds}; "
                        f"Rationale: {_fp_rationale}]"
                    )

                _rp_cached = st.session_state.get("_dominant_rate_path")
                _rp_all_cached = st.session_state.get("_rate_path_probs", [])
                if _rp_cached and _rp_cached.get("scenario"):
                    _scenario_labels_enrich = {
                        "cut_25": "25bp cut", "cut_50": "50bp cut",
                        "hold": "Hold", "hike_25": "25bp hike",
                    }
                    _rp_str = _scenario_labels_enrich.get(_rp_cached["scenario"], _rp_cached["scenario"])
                    _rp_prob = _rp_cached.get("prob_pct", 0)
                    _rp_all_str = ", ".join(
                        f"{_scenario_labels_enrich.get(r['scenario'], r['scenario'])} {round(r.get('prob', 0) * 100)}%"
                        for r in sorted(_rp_all_cached, key=lambda r: r.get("prob", 0), reverse=True)
                    )
                    _enrichment_parts.append(
                        f"[Fed Rate Path: dominant={_rp_str} ({_rp_prob:.0f}% prob) | all scenarios: {_rp_all_str}]"
                    )

                _custom_swans_cached = st.session_state.get("_custom_swans", {})
                if _custom_swans_cached:
                    _swan_parts = []
                    for _slabel, _sdata in list(_custom_swans_cached.items())[:3]:
                        _sprob = _sdata.get("probability_pct", 0)
                        _simpacts = _sdata.get("asset_impacts", {})
                        _simpact_str = ", ".join(
                            f"{k}={v}" for k, v in list(_simpacts.items())[:4]
                        )
                        _swan_parts.append(f"{_slabel} ({_sprob:.0f}% prob): {_simpact_str}")
                    _enrichment_parts.append(
                        f"[Black Swan Tail Risks: {'; '.join(_swan_parts)}]"
                    )

                # Policy Transmission narration
                _narration_cached = st.session_state.get("_chain_narration")
                if _narration_cached:
                    _enrichment_parts.append(f"[Policy Transmission: {_narration_cached[:400]}]")

                # Doom Briefing risk assessment
                _doom_cached = st.session_state.get("_doom_briefing")
                if _doom_cached:
                    _enrichment_parts.append(f"[Risk Intelligence Briefing: {_doom_cached[:400]}]")

                # Prior Cross-Signal Macro Plays (previous Discovery run — enriches next)
                _prev_plays = st.session_state.get("_plays_result")
                if _prev_plays:
                    _pp_sectors = ", ".join(s.get("name", "") for s in _prev_plays.get("sectors", [])[:3])
                    _enrichment_parts.append(
                        f"[Prior Discovery Plays: Sectors: {_pp_sectors} | {_prev_plays.get('rationale', '')}]"
                    )

                # Institutional Whale Activity Summary
                _whale_cached = st.session_state.get("_whale_summary")
                if _whale_cached:
                    _enrichment_parts.append(f"[Whale Activity: {_whale_cached[:400]}]")

                # Activism Campaigns (13D digest)
                _activism_cached = st.session_state.get("_activism_digest")
                if _activism_cached:
                    _enrichment_parts.append(f"[Activism Campaigns (13D): {_activism_cached[:300]}]")

                # Sector×Regime digest
                _srd_disc = st.session_state.get("_sector_regime_digest")
                if _srd_disc:
                    _enrichment_parts.append(f"[Sector×Regime: {_srd_disc[:300]}]")

                # AI Regime Plays (sector/stock picks for current regime)
                _rp_plays_cached = st.session_state.get("_rp_plays_result")
                if _rp_plays_cached:
                    _rp_sectors = ", ".join(s.get("name", "") for s in _rp_plays_cached.get("sectors", [])[:3])
                    _rp_stocks = ", ".join(s.get("ticker", "") for s in _rp_plays_cached.get("stocks", [])[:4])
                    _enrichment_parts.append(
                        f"[AI Regime Plays: Sectors: {_rp_sectors}; Stocks: {_rp_stocks}; "
                        f"{_rp_plays_cached.get('rationale', '')[:200]}]"
                    )

                # Fed Funds Rate
                _ff_rate_disc = st.session_state.get("_fed_funds_rate")
                if _ff_rate_disc is not None:
                    _enrichment_parts.append(f"[Fed Funds Rate: {_ff_rate_disc:.2f}%]")

                # Regime signal_summary (17-signal breakdown)
                _rc_disc = st.session_state.get("_regime_context")
                if _rc_disc and _rc_disc.get("signal_summary"):
                    _enrichment_parts.append(f"[Regime Signal Detail: {_rc_disc['signal_summary'][:400]}]")

                # Current Events digest (from Current Events module)
                _ce_disc = st.session_state.get("_current_events_digest", "")
                if _ce_disc:
                    _enrichment_parts.append(f"[Current Events: {_ce_disc[:400]}]")

                # Trending Narratives (from AI trending scanner)
                _tn_disc = st.session_state.get("_trending_narratives")
                if _tn_disc:
                    _tn_lines = [
                        f"{n['narrative']} ({n.get('conviction','')}) [{n.get('category','')}]"
                        f" — {', '.join(n.get('tickers', []))}"
                        for n in _tn_disc[:3]
                    ]
                    _enrichment_parts.append("[Trending Narratives: " + " | ".join(_tn_lines) + "]")

                # Auto-Trending ticker groups (Yahoo Finance price movers)
                _atg_disc = st.session_state.get("_auto_trending_groups")
                if _atg_disc:
                    _atg_lines = [
                        f"{g['narrative']} ({g.get('conviction','')}, {g.get('regime_alignment','')})"
                        f" — {', '.join(g.get('tickers', []))}"
                        for g in _atg_disc[:3]
                    ]
                    _enrichment_parts.append("[Trending Price Movers: " + " | ".join(_atg_lines) + "]")

                # Macro Options Flow (SPY-level P/C, gamma, put wall)
                _of_nd = st.session_state.get("_options_flow_context") or {}
                if _of_nd:
                    _enrichment_parts.append(
                        f"[Macro Options Flow (SPY): {_of_nd.get('label', '')} "
                        f"(score {_of_nd.get('options_score', 50)}/100) — {_of_nd.get('action_bias', '')[:150]}]"
                    )

                # Portfolio Risk Snapshot (from Quick Intel Run or Trade Journal)
                _pr_disc = st.session_state.get("_portfolio_risk_snapshot") or {}
                if _pr_disc:
                    _pr_disc_parts = []
                    if _pr_disc.get("beta") is not None:
                        _pr_disc_parts.append(f"Beta {_pr_disc['beta']} | VaR95 {_pr_disc.get('var_95_pct')}%")
                    _sw_disc = _pr_disc.get("sector_weights") or {}
                    if _sw_disc:
                        _top_sectors = sorted(_sw_disc.items(), key=lambda x: -x[1])[:3]
                        _pr_disc_parts.append("Heavy in: " + ", ".join(f"{s} {w}%" for s, w in _top_sectors))
                    _rf_disc = _pr_disc.get("risk_flags") or []
                    if _rf_disc:
                        _pr_disc_parts.append("Flags: " + "; ".join(f.replace("⚠ ", "") for f in _rf_disc[:2]))
                    if _pr_disc_parts:
                        _enrichment_parts.append("[Portfolio Risk: " + " | ".join(_pr_disc_parts) + "]")

                if _enrichment_parts:
                    _signal_summary += " || UPSTREAM AI CONTEXT: " + " | ".join(_enrichment_parts)
                _scenario_text = _overlay_scenario.strip()
                if _scenario_text:
                    # Overlay active — run scenario-aware play generation
                    from services.claude_client import suggest_scenario_plays
                    with st.spinner(f"Generating plays + stress-testing '{_scenario_text}'..."):
                        _plays = suggest_scenario_plays(
                            scenario=_scenario_text,
                            regime=_regime,
                            quadrant=_quadrant,
                            signal_summary=_signal_summary,
                            use_claude=_use_cl,
                            model=_cl_model,
                        )
                else:
                    # Base play — no overlay
                    from services.claude_client import suggest_regime_plays
                    with st.spinner("Generating macro plays..."):
                        _plays = suggest_regime_plays(
                            _regime, _macro_ctx["score"], _signal_summary,
                            use_claude=_use_cl, model=_cl_model,
                        )
                st.session_state["_plays_result"] = _plays
                st.session_state["_plays_engine"] = _selected_play_model
                st.session_state["_plays_overlay_text"] = _scenario_text
                st.session_state["_discovery_tier"] = _selected_play_model
                from services.play_log import append_play as _append_play
                _append_play("Discovery Plays", _selected_play_model, _plays,
                             meta={"overlay": _scenario_text or None, "regime": _regime})
            else:
                _plays = st.session_state["_plays_result"]
                _cached_plays_engine = st.session_state.get("_plays_engine", "⚡ Freeloader Mode")

            if _plays and (_plays.get("sectors") or _plays.get("stocks") or _plays.get("bonds")):
                _display_engine = _cached_plays_engine if not _gen_plays else _selected_play_model
                _overlay_used = st.session_state.get("_plays_overlay_text", "")
                if _overlay_used:
                    st.caption(f"*{_display_engine} · Stress-Test: \"{_overlay_used}\"*")
                else:
                    st.caption(f"*{_display_engine} · Base Macro Regime*")

                _s_col, _st_col, _b_col = st.columns(3)

                with _s_col:
                    st.markdown("**Sectors**")
                    for _item in _plays.get("sectors", []):
                        _stars = _conviction_stars(_item.get("conviction", 1))
                        st.markdown(f"{_stars} {_item.get('name', '')}")

                with _st_col:
                    st.markdown("**Stocks**")
                    for _item in _plays.get("stocks", []):
                        _stars = _conviction_stars(_item.get("conviction", 1))
                        _reason = _item.get("reason", "")
                        st.markdown(f"{_stars} **{_item.get('ticker', '')}** — {_reason}")

                with _b_col:
                    st.markdown("**Bonds**")
                    for _item in _plays.get("bonds", []):
                        _stars = _conviction_stars(_item.get("conviction", 1))
                        _reason = _item.get("reason", "")
                        st.markdown(f"{_stars} **{_item.get('ticker', '')}** — {_reason}")

                if _plays.get("avoid"):
                    _avoid_str = ", ".join(_plays["avoid"]) if isinstance(_plays["avoid"], list) else str(_plays["avoid"])
                    st.markdown(f"🚫 **Avoid:** {_avoid_str}")

                if _plays.get("rationale"):
                    st.caption(_plays["rationale"])

        # ── Macro Fit Check ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**🎯 Macro Fit Check** — analyze any ticker against current regime")
        _fit_cols = st.columns([3, 1, 1])
        with _fit_cols[0]:
            _fit_ticker = st.text_input(
                "Ticker", placeholder="e.g. AAPL, XLE, TLT",
                label_visibility="collapsed", key="macro_fit_ticker_input",
            )
        with _fit_cols[1]:
            _fit_engine = st.selectbox("Engine", _PLAY_MODEL_OPTIONS, key="macro_fit_engine")
        with _fit_cols[2]:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            _run_fit = st.button("Assess Fit", type="primary", key="run_macro_fit_btn")

        # ── Ticker preview (fires on Enter / focus-out, no button needed) ──
        _fit_t_input = _fit_ticker.strip().upper() if _fit_ticker.strip() else ""
        _prev_preview_t = st.session_state.get("_fit_preview_ticker", "")
        if _fit_t_input and _fit_t_input != _prev_preview_t:
            import yfinance as _yf_prev
            _prev_raw = _yf_prev.Ticker(_fit_t_input).info or {}
            st.session_state["_fit_preview_info"] = {
                "ticker": _fit_t_input,
                "name": _prev_raw.get("longName", _fit_t_input),
                "sector": _prev_raw.get("sector", _prev_raw.get("industry", "")),
                "price": _prev_raw.get("currentPrice") or _prev_raw.get("regularMarketPrice"),
                "summary": (_prev_raw.get("longBusinessSummary") or "")[:220],
            }
            st.session_state["_fit_preview_ticker"] = _fit_t_input

        _preview_info = st.session_state.get("_fit_preview_info")
        if _preview_info and _preview_info.get("ticker") == _fit_t_input and _fit_t_input:
            _pv_price = f" · ${_preview_info['price']:.2f}" if _preview_info.get("price") else ""
            _pv_sector = f" · {_preview_info['sector']}" if _preview_info.get("sector") else ""
            st.markdown(
                f'<div style="font-size:11px;color:#94a3b8;border:1px solid #1e293b;border-radius:6px;'
                f'padding:8px 12px;margin:4px 0 8px 0;">'
                f'<span style="font-weight:700;color:#cbd5e1;">{_preview_info["name"]}</span>'
                f'<span style="color:#64748b;">{_pv_sector}{_pv_price}</span>'
                + (f'<br><span style="color:#64748b;">{_preview_info["summary"]}…</span>' if _preview_info.get("summary") else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

        if _run_fit and _fit_t_input:
            _fit_use_cl, _fit_model = _PLAY_MODEL_MAP[_fit_engine]
            # Reuse preview info if available (avoids double yfinance fetch)
            if _preview_info and _preview_info.get("ticker") == _fit_t_input:
                _fit_name = _preview_info["name"]
                _fit_sector = _preview_info.get("sector", "")
                _fit_price_str = f"Current ${_preview_info['price']:.2f}" if _preview_info.get("price") else ""
            else:
                import yfinance as _yf_fit
                _fit_info = _yf_fit.Ticker(_fit_t_input).info or {}
                _fit_name = _fit_info.get("longName", _fit_t_input)
                _fit_sector = _fit_info.get("sector", _fit_info.get("industry", ""))
                _fit_price = _fit_info.get("currentPrice") or _fit_info.get("regularMarketPrice")
                _fit_price_str = f"Current ${_fit_price:.2f}" if _fit_price else ""

            _fit_regime_ctx = st.session_state.get("_regime_context", {}).get("signal_summary", "Not loaded")
            _fit_rp_all = st.session_state.get("_rate_path_probs", [])
            _fit_rp_labels = {"cut_25": "25bp cut", "cut_50": "50bp cut", "hold": "Hold", "hike_25": "25bp hike"}
            _fit_rp_str = ", ".join(
                f"{_fit_rp_labels.get(r['scenario'], r['scenario'])} {round(r.get('prob', 0) * 100)}%"
                for r in sorted(_fit_rp_all, key=lambda r: r.get("prob", 0), reverse=True)
            ) if _fit_rp_all else "Not loaded"
            _fit_swans = st.session_state.get("_custom_swans", {})
            _fit_swan_str = "; ".join(
                f"{k} ({v.get('probability_pct', 0):.0f}%): " +
                ", ".join(f"{a}={b}" for a, b in list(v.get("asset_impacts", {}).items())[:3])
                for k, v in list(_fit_swans.items())[:2]
            ) if _fit_swans else ""

            from services.claude_client import assess_macro_fit as _assess_macro_fit
            with st.spinner(f"Assessing macro fit for {_fit_t_input}..."):
                from services.claude_client import _fmt_tactical_ctx as _fmt_tac
                _fit_result = _assess_macro_fit(
                    ticker=_fit_t_input, company_name=_fit_name, sector=_fit_sector,
                    price_summary=_fit_price_str,
                    regime_context=_fit_regime_ctx,
                    rate_path_context=_fit_rp_str,
                    black_swan_context=_fit_swan_str,
                    tactical_context=_fmt_tac(st.session_state.get("_tactical_context")),
                    use_claude=_fit_use_cl, model=_fit_model,
                )
            if _fit_result and "_error" in _fit_result:
                st.session_state["_macro_fit_error"] = _fit_result["_error"]
                _fit_result = None
            else:
                st.session_state.pop("_macro_fit_error", None)
            st.session_state["_macro_fit_result"] = _fit_result
            st.session_state["_macro_fit_ticker"] = _fit_t_input
            # Accumulate in per-ticker dict for Portfolio Intelligence
            if _fit_result and "_error" not in (_fit_result or {}):
                _mf_dict = st.session_state.get("_macro_fit_results", {})
                _mf_dict[_fit_t_input.upper()] = {
                    "fit_stars": _fit_result.get("fit_stars", 0),
                    "verdict": _fit_result.get("verdict", ""),
                    "rationale": _fit_result.get("rationale", ""),
                }
                st.session_state["_macro_fit_results"] = _mf_dict

        _fit_err = st.session_state.get("_macro_fit_error")
        if _fit_err:
            st.error(f"Macro Fit failed: {_fit_err}")

        _cached_fit = st.session_state.get("_macro_fit_result")
        _cached_fit_t = st.session_state.get("_macro_fit_ticker", "")
        if _cached_fit and _cached_fit_t:
            _fit_stars_n = _cached_fit.get("fit_stars", 0)
            _fit_stars_str = "★" * _fit_stars_n + "☆" * (5 - _fit_stars_n)
            _fit_verdict = _cached_fit.get("verdict", "")
            _fit_verdict_color = {
                "Strong Fit": "#22c55e", "Moderate Fit": "#86efac",
                "Neutral": "#f59e0b", "Caution": "#f97316", "Avoid": "#ef4444",
            }.get(_fit_verdict, "#94a3b8")
            _fit_tac = st.session_state.get("_tactical_context") or {}
            _fit_tac_html = ""
            if _fit_tac:
                _fts = _fit_tac.get("tactical_score", 50)
                _ftl = _fit_tac.get("label", "")
                _fta = _fit_tac.get("action_bias", "")
                _ftc = "#22c55e" if _fts >= 65 else ("#f59e0b" if _fts >= 38 else "#ef4444")
                _fit_tac_html = (
                    f'<div style="margin-top:8px;padding:6px 10px;background:{_ftc}12;'
                    f'border-left:2px solid {_ftc};border-radius:0 4px 4px 0;">'
                    f'<span style="font-size:10px;color:{_ftc};font-weight:700;letter-spacing:0.05em;">TIMING</span>'
                    f'&nbsp;&nbsp;<span style="font-size:11px;color:{_ftc};">{_ftl} ({_fts}/100)</span>'
                    f'<span style="font-size:11px;color:#64748b;"> — {_fta}</span>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="border:1px solid #1e293b;border-radius:8px;padding:12px 16px;margin-top:8px;">'
                f'<span style="font-size:13px;font-weight:700;color:#e2e8f0;">{_cached_fit_t}</span>'
                f'&nbsp;&nbsp;<span style="color:#f59e0b;font-size:14px;">{_fit_stars_str}</span>'
                f'&nbsp;&nbsp;<span style="font-size:12px;font-weight:700;color:{_fit_verdict_color};">{_fit_verdict}</span>'
                f'<p style="font-size:12px;color:#94a3b8;margin:8px 0 6px 0;">{_cached_fit.get("rationale", "")}</p>'
                + "".join(f'<span style="font-size:11px;color:#22c55e;">↑ {t}&nbsp;&nbsp;</span>' for t in _cached_fit.get("tailwinds", []))
                + "".join(f'<span style="font-size:11px;color:#ef4444;">↓ {h}&nbsp;&nbsp;</span>' for h in _cached_fit.get("headwinds", []))
                + _fit_tac_html
                + '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### Ticker & Narrative Research")
    st.caption("Look up a specific ticker or explore a named narrative")

    mode = st.radio("Mode", ["Manual", "Auto — Trending"], horizontal=True)

    if mode == "Auto — Trending":
        _render_auto()
    else:
        _render_manual()


def _fetch_curated_assets(ticker_dict: dict[str, str], asset_class: str) -> list[dict]:
    """Fetch price data for curated non-equity tickers and return in trending-compatible shape."""
    snapshots = fetch_batch_safe(ticker_dict, period="5d")
    items = []
    for idx, (ticker, label) in enumerate(ticker_dict.items()):
        snap = snapshots.get(ticker)
        pct = snap.pct_change_1d if snap and snap.pct_change_1d is not None else 0.0
        items.append({
            "symbol": ticker,
            "name": label,
            "pct_change": pct,
            "buzz_rank": idx,
            "asset_class": asset_class,
        })
    items.sort(key=lambda x: abs(x.get("pct_change", 0)), reverse=True)
    return items


def _render_auto():
    from utils.ai_tier import render_ai_tier_selector
    _auto_use_claude, _auto_model = render_ai_tier_selector(
        key="auto_trend_engine",
        label="Engine",
        recommendation="⚡ Freeloader sufficient for trend discovery · 🧠 Regard for richer narrative grouping",
    )
    _auto_engine_sel = st.session_state.get("auto_trend_engine", "⚡ Freeloader Mode")

    # --- Asset class filter ---
    selected_classes = st.multiselect(
        "Asset Classes", list(ASSET_CLASSES.keys()),
        default=["Equities"], key="asset_class_filter",
    )

    if not selected_classes:
        st.info("Select at least one asset class to discover trending assets.")
        return

    all_trending = []

    # --- Equities: Yahoo Finance trending tickers ---
    if "Equities" in selected_classes:
        with st.spinner("Fetching trending tickers from Yahoo Finance + StockTwits..."):
            yf_trending = get_yahoo_trending_tickers()
        if yf_trending:
            for item in yf_trending:
                item["asset_class"] = "Equities"
            all_trending.extend(yf_trending)

        # Merge StockTwits trending (from cached QIR digest or fresh fetch)
        _st_digest = st.session_state.get("_stocktwits_digest") or {}
        _st_tickers = _st_digest.get("all_trending_symbols", [])
        if not _st_tickers:
            # QIR hasn't run yet — try a quick live fetch (cached for 1h)
            try:
                from services.stocktwits_client import get_trending_symbols as _st_trending
                _st_raw = _st_trending(limit=15)
                _st_tickers = [t["symbol"] for t in _st_raw]
            except Exception:
                _st_tickers = []
        # Add StockTwits symbols not already in the pool
        _existing_symbols = {item["symbol"].upper() for item in all_trending}
        for _stk in _st_tickers:
            if _stk.upper() not in _existing_symbols:
                all_trending.append({
                    "symbol": _stk,
                    "name": _stk,
                    "asset_class": "Equities",
                    "source": "StockTwits",
                })
                _existing_symbols.add(_stk.upper())

    # --- Non-equity asset classes ---
    for cls in selected_classes:
        if cls == "Equities":
            continue
        ticker_dict = ASSET_CLASSES.get(cls)
        if ticker_dict:
            with st.spinner(f"Fetching {cls} data..."):
                all_trending.extend(_fetch_curated_assets(ticker_dict, cls))

    if not all_trending:
        st.warning("No data available for the selected asset classes.")
        return

    yf_trending = all_trending  # keep variable name for downstream compat

    if yf_trending:
        # Build a lookup from symbol → item
        ticker_lookup = {item["symbol"]: item for item in yf_trending}

        # Group tickers by narrative theme via AI (with regime context if available)
        import json as _json
        tickers_for_grouping = _json.dumps(
            [{"symbol": t["symbol"], "name": t["name"]} for t in yf_trending]
        )
        _rc_auto = st.session_state.get("_regime_context") or {}
        _regime_ctx_str = ""
        if _rc_auto:
            _regime_ctx_str = (
                f"Regime: {_rc_auto.get('regime','')} (score {_rc_auto.get('score', 0.0):+.2f})"
                f" | Quadrant: {_rc_auto.get('quadrant','')}"
            )
        with st.spinner("Grouping tickers by narrative + scoring regime fit..."):
            narrative_groups = group_tickers_by_narrative(
                tickers_for_grouping, _regime_ctx_str, _auto_use_claude, _auto_model
            )

        if narrative_groups:
            from datetime import datetime as _dt2
            st.session_state["_auto_trending_groups"] = narrative_groups
            st.session_state["_auto_trending_groups_ts"] = _dt2.now()
            st.session_state["_auto_trending_groups_engine"] = _auto_engine_sel
            from services.signals_cache import save_signals
            save_signals()

        from datetime import datetime
        _engine_badge_color = {"⚡ Freeloader Mode": "#f59e0b", "🧠 Regard Mode": "#3b82f6", "👑 Highly Regarded Mode": "#a855f7"}.get(_auto_engine_sel, "#64748b")
        st.markdown(
            f'<span style="color:#475569;font-size:11px;">LAST UPDATE {datetime.now().strftime("%Y-%m-%d %H:%M")} · CACHE 1H</span>'
            f'&nbsp;&nbsp;<span style="background:{_engine_badge_color}22;color:{_engine_badge_color};'
            f'font-size:10px;font-weight:700;padding:2px 8px;border-radius:3px;letter-spacing:0.06em;">'
            f'{_auto_engine_sel}</span>',
            unsafe_allow_html=True,
        )

        if narrative_groups:
            st.subheader(f"{len(yf_trending)} Trending Tickers · {len(narrative_groups)} Narratives")

            for g_idx, group in enumerate(narrative_groups):
                narrative_title = group.get("narrative", "Market Movers")
                description = group.get("description", "")
                group_tickers = group.get("tickers", [])

                # Conviction + regime alignment badges
                _conv = group.get("conviction", "")
                _align = group.get("regime_alignment", "")
                _rationale = group.get("rationale", "")
                _conv_color = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(_conv, "#64748b")
                _align_color = {"aligned": "#22c55e", "contrarian": "#ef4444", "neutral": "#94a3b8"}.get(_align, "#64748b")
                _align_icon = {"aligned": "✓", "contrarian": "✗", "neutral": "~"}.get(_align, "")
                _badges_html = ""
                if _conv:
                    _badges_html += (
                        f'<span style="background:{_conv_color}22;color:{_conv_color};'
                        f'font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;'
                        f'letter-spacing:0.08em;margin-right:6px;">{_conv}</span>'
                    )
                if _align:
                    _badges_html += (
                        f'<span style="background:{_align_color}22;color:{_align_color};'
                        f'font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;'
                        f'letter-spacing:0.08em;">{_align_icon} {_align.upper()}</span>'
                    )

                # Narrative header
                st.markdown(
                    f'<div style="background:{COLORS["surface"]}; padding:12px 16px; '
                    f'border-radius:8px; border-left:4px solid {COLORS["accent"]}; '
                    f'margin:16px 0 8px 0;">'
                    f'<div style="color:{COLORS["accent"]}; font-size:18px; font-weight:700;">'
                    f'{narrative_title}</div>'
                    f'<div style="margin:4px 0;">{_badges_html}</div>'
                    f'<div style="color:{COLORS["text_dim"]}; font-size:13px; margin-top:4px;">'
                    f'{description}</div>'
                    + (f'<div style="color:#64748b; font-size:11px; margin-top:3px; font-style:italic;">{_rationale}</div>' if _rationale else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )

                # Ticker cards in columns
                group_items = [
                    ticker_lookup[sym] for sym in group_tickers
                    if sym in ticker_lookup
                ]
                if not group_items:
                    continue

                cols = st.columns(min(3, len(group_items)))
                for i, item in enumerate(group_items):
                    col = cols[i % len(cols)]
                    with col:
                        with st.container(border=True):
                            # Asset class badge
                            _badge = _ASSET_BADGE.get(item.get("asset_class"))
                            if _badge:
                                st.markdown(
                                    f'<span style="background:{_badge[1]}22;color:{_badge[1]};'
                                    f'font-size:10px;font-weight:700;padding:2px 6px;border-radius:3px;'
                                    f'letter-spacing:0.08em;font-family:\'JetBrains Mono\',monospace;">'
                                    f'{_badge[0]}</span>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**{item['name']}**")
                            st.code(item["symbol"], language=None)
                            # Intraday % change
                            pct = item.get("pct_change")
                            if pct is not None:
                                pct_color = COLORS.get("green", "#00d4aa") if pct >= 0 else COLORS.get("red", "#ff4d4d")
                                arrow = "▲" if pct >= 0 else "▼"
                                st.markdown(
                                    f'<span style="color:{pct_color};font-weight:700;font-size:15px;">'
                                    f'{arrow} {pct:+.2f}%</span>',
                                    unsafe_allow_html=True,
                                )
                            # Buzz star rating (top of trending list = 5 stars)
                            buzz_rank = item.get("buzz_rank", 99)
                            total = len(yf_trending)
                            stars = max(1, min(5, 5 - int((buzz_rank - 1) / max(1, total / 5))))
                            st.markdown(
                                f'<span style="color:#FFD700;font-size:16px;" title="Buzz: {stars}/5">'
                                f'{"★" * stars}{"☆" * (5 - stars)}</span>',
                                unsafe_allow_html=True,
                            )
                            if st.button("Select", key=f"yf_select_{g_idx}_{i}", type="primary"):
                                set_narrative(narrative_title)
                                set_ticker(item["symbol"])
                                st.rerun()
                            if st.button("Watch", key=f"wl_add_{g_idx}_{i}"):
                                from utils.watchlist import add_to_watchlist
                                add_to_watchlist(item["symbol"], narrative_title)
                                st.rerun()
        else:
            # Fallback: flat list if grouping fails
            st.subheader(f"{len(yf_trending)} Trending Tickers")
            cols = st.columns(3)
            for i, item in enumerate(yf_trending):
                col = cols[i % 3]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{item['name']}**")
                        st.code(item["symbol"], language=None)
                        pct = item.get("pct_change")
                        if pct is not None:
                            pct_color = COLORS.get("green", "#00d4aa") if pct >= 0 else COLORS.get("red", "#ff4d4d")
                            arrow = "▲" if pct >= 0 else "▼"
                            st.markdown(
                                f'<span style="color:{pct_color};font-weight:700;font-size:15px;">'
                                f'{arrow} {pct:+.2f}%</span>',
                                unsafe_allow_html=True,
                            )
                        buzz_rank = item.get("buzz_rank", 99)
                        total = len(yf_trending)
                        stars = max(1, min(5, 5 - int((buzz_rank - 1) / max(1, total / 5))))
                        st.markdown(
                            f'<span style="color:#FFD700;font-size:16px;" title="Buzz: {stars}/5">'
                            f'{"★" * stars}{"☆" * (5 - stars)}</span>',
                            unsafe_allow_html=True,
                        )
                        if st.button("Select", key=f"yf_select_{i}", type="primary"):
                            set_narrative(item["name"])
                            set_ticker(item["symbol"])
                            st.rerun()
                        if st.button("Watch", key=f"wl_add_flat_{i}"):
                            from utils.watchlist import add_to_watchlist
                            add_to_watchlist(item["symbol"], item["name"])
                            st.rerun()

    # --- Show overview for selected ticker ---
    active = get_ticker()
    if active:
        _render_company_overview(active)

    # --- Trending search interest (auto, no click needed) ---
    if yf_trending:
        _render_trending_interest(yf_trending)


def _render_manual():
    tab_ticker, tab_narrative = st.tabs(["Ticker Symbol", "Narrative Keyword"])

    with tab_ticker:
        search_input = st.text_input(
            "Search by ticker or company name",
            placeholder="e.g. AAPL, Apple, TSLA, Tesla Inc",
        ).strip()

        if st.button("Search", type="primary", key="set_ticker_btn") and search_input:
            st.session_state["ticker_search_query"] = search_input

        query = st.session_state.get("ticker_search_query", "")
        if query:
            # Quick-set if it looks like a ticker (short alpha string)
            if query.isalpha() and len(query) <= 5:
                if st.button(f"Set **{query.upper()}** as active ticker", key="direct_set_ticker"):
                    set_ticker(query.upper())
                    st.rerun()

            # Company name search results
            results = search_ticker_by_name(query)
            if results:
                st.caption(f"Matches ({len(results)}):")
                for r in results:
                    c_name, c_btn = st.columns([3, 1])
                    c_name.markdown(f"**{r['ticker']}** — {r['name']}")
                    if c_btn.button("Select", key=f"co_sel_{r['ticker']}"):
                        set_ticker(r["ticker"])
                        st.rerun()
            elif not (query.isalpha() and len(query) <= 5):
                st.caption("No matching companies found.")

        # Show confirmed ticker info + watchlist + overview
        active = get_ticker()
        if active:
            st.success(f"Active ticker: **{active}**")
            if is_in_watchlist(active):
                st.caption(f"**{active}** is on your watchlist")
            elif st.button("+ Watch", key="watch_ticker_manual"):
                add_to_watchlist(active, st.session_state.get("active_narrative", ""))
                st.rerun()
            _render_company_overview(active)
            try:
                _render_interest_chart(active, key_suffix="ticker")
            except Exception:
                st.caption("Search interest data unavailable.")

    with tab_narrative:
        keyword = st.text_input(
            "Enter a narrative keyword",
            placeholder="e.g. nuclear energy, weight loss drugs, AI chips",
        )
        if st.button("Analyze", type="primary") and keyword:
            set_narrative(keyword)
            result = classify_narrative(keyword)
            st.session_state["narrative_result"] = result
            st.session_state["narrative_keyword"] = keyword

        # Show results persistently after analysis
        result = st.session_state.get("narrative_result")
        kw = st.session_state.get("narrative_keyword", "")
        if result:
            if result.get("market_relevant"):
                st.success(f"Narrative set: **{kw}**")
                st.markdown(f"**Sector:** {result.get('sector', 'N/A')}")
                st.markdown(f"**Thesis:** {result.get('thesis', '')}")
                tickers = result.get("suggested_tickers", [])
                if tickers:
                    st.markdown(f"**Suggested tickers:** {', '.join(tickers)}")
                    selected = st.selectbox("Set active ticker", tickers, key="narrative_ticker_select")
                    if st.button("Confirm Ticker", key="confirm_narrative_ticker") and selected:
                        set_ticker(selected)
                        st.rerun()
            else:
                st.info(
                    "Topic classified as not directly market-relevant, but narrative is set."
                )

        # Show company overview and interest chart for confirmed ticker
        active = get_ticker()
        if active and result:
            if is_in_watchlist(active):
                st.caption(f"**{active}** is on your watchlist")
            elif st.button("+ Watch", key="watch_narrative_manual"):
                add_to_watchlist(active, kw)
                st.rerun()
            _render_company_overview(active)
            try:
                _render_interest_chart(active, key_suffix="narrative")
            except Exception:
                st.caption("Search interest data unavailable.")


def _render_trending_interest(trending: list[dict]):
    """Auto-render a multi-line interest chart for the top trending tickers."""
    # Take first 5 (pytrends limit)
    top = trending[:5]
    symbols = tuple(item["symbol"] for item in top)

    st.subheader("Trending Search Interest")
    timeframe = st.select_slider(
        "Timeframe",
        options=["1M", "3M", "6M", "1Y", "YTD"],
        value="3M",
        key="interest_tf_trending",
    )

    with st.spinner("Fetching search interest for trending tickers..."):
        df = get_interest_over_time_multi(symbols, timeframe)

    if df.empty:
        st.caption("No interest data available for trending tickers.")
        return

    line_colors = [COLORS["accent"], COLORS["blue"], COLORS["yellow"], COLORS["red"], "#AB47BC"]

    fig = go.Figure()
    for i, sym in enumerate(symbols):
        if sym not in df.columns:
            continue
        color = line_colors[i % len(line_colors)]
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[sym],
                mode="lines",
                name=sym,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{sym}</b><br>%{{x|%b %d}}: %{{y}}<extra></extra>",
            )
        )

    apply_dark_layout(
        fig,
        title="Search Interest: Top Trending Tickers",
        yaxis_title="Relative Interest (0–100)",
        xaxis_title="",
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Source: Google Trends · comparing relative search interest across tickers")


def _render_company_overview(ticker: str):
    """Show company description and narrative for a ticker."""
    # Route non-equity assets to simplified overview
    if ticker in _TICKER_TO_CLASS:
        _render_asset_overview(ticker)
        return

    with st.spinner(f"Looking up {ticker}..."):
        company_info = get_company_info(ticker)

    if not company_info:
        st.warning(f"Could not fetch company info for {ticker} from SEC. The SEC API may be temporarily unavailable.")
        return

    with st.spinner("Generating company overview..."):
        overview = describe_company(
            company_info["name"],
            ticker,
            company_info["sic_description"],
        )

    with st.container(border=True):
        st.subheader(f"{company_info['name']}  ·  {ticker}")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            if overview.get("description"):
                st.markdown(overview["description"])
            else:
                st.caption(company_info["sic_description"])
        with col_b:
            st.markdown(f"**Sector:** {overview.get('sector', company_info['sic_description'])}")
            if company_info["state"]:
                st.markdown(f"**Incorporated:** {company_info['state']}")
            if company_info["exchanges"]:
                st.markdown(f"**Exchange:** {', '.join(company_info['exchanges'])}")

        if overview.get("narrative"):
            st.info(f"**Narrative:** {overview['narrative']}")


def _render_asset_overview(ticker: str):
    """Simplified overview for non-equity assets (commodities, bonds, currencies)."""
    asset_class = _TICKER_TO_CLASS.get(ticker, "Unknown")
    # Find label from ASSET_CLASSES
    label = ticker
    class_dict = ASSET_CLASSES.get(asset_class, {})
    if class_dict:
        label = class_dict.get(ticker, ticker)

    with st.spinner(f"Fetching {label} data..."):
        snapshots = fetch_batch_safe({ticker: label}, period="3mo")

    snap = snapshots.get(ticker)
    badge_info = _ASSET_BADGE.get(asset_class, (asset_class.upper(), COLORS["accent"]))

    with st.container(border=True):
        st.markdown(
            f'<span style="background:{badge_info[1]}22;color:{badge_info[1]};'
            f'font-size:11px;font-weight:700;padding:2px 8px;border-radius:3px;'
            f'letter-spacing:0.08em;">{badge_info[0]}</span>',
            unsafe_allow_html=True,
        )
        st.subheader(f"{label}  ·  {ticker}")

        if snap and snap.latest_price is not None:
            col_price, col_1d, col_5d, col_30d = st.columns(4)
            with col_price:
                st.metric("Price", f"${snap.latest_price:,.2f}")
            with col_1d:
                val = snap.pct_change_1d
                st.metric("1D", f"{val:+.2f}%" if val is not None else "N/A",
                           delta=f"{val:+.2f}%" if val is not None else None)
            with col_5d:
                val = snap.pct_change_5d
                st.metric("5D", f"{val:+.2f}%" if val is not None else "N/A",
                           delta=f"{val:+.2f}%" if val is not None else None)
            with col_30d:
                val = snap.pct_change_30d
                st.metric("30D", f"{val:+.2f}%" if val is not None else "N/A",
                           delta=f"{val:+.2f}%" if val is not None else None)

            # Mini sparkline
            if snap.series is not None and not snap.series.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=snap.series.index,
                    y=snap.series.values,
                    mode="lines",
                    line=dict(color=COLORS["accent"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 212, 170, 0.1)",
                    hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.2f}<extra></extra>",
                ))
                apply_dark_layout(fig, title=f"{label} — 3 Month",
                                  margin=dict(l=40, r=20, t=40, b=30))
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch price data for {ticker}.")


def _render_interest_chart(keyword: str, key_suffix: str = "default"):
    """Render a Google Trends interest-over-time area chart for a keyword."""
    st.subheader("Search Interest")
    timeframe = st.select_slider(
        "Timeframe",
        options=["1M", "3M", "6M", "1Y", "YTD"],
        value="3M",
        key=f"interest_tf_{key_suffix}",
    )

    with st.spinner("Fetching search interest..."):
        df = get_interest_over_time(keyword, timeframe)

    if df.empty:
        st.caption("No interest data available for this keyword.")
        return

    peak = df["interest"].max()
    peak_date = df.loc[df["interest"].idxmax(), "date"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["interest"],
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.15)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Interest: %{y}<extra></extra>",
        )
    )

    # Peak annotation
    fig.add_annotation(
        x=peak_date,
        y=peak,
        text=f"Peak: {peak}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS["accent"],
        font=dict(color=COLORS["accent"], size=11),
        ax=0,
        ay=-30,
    )

    apply_dark_layout(
        fig,
        title=f"Google Trends Interest: {keyword}",
        yaxis_title="Relative Interest (0–100)",
        xaxis_title="",
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Source: Google Trends · 100 = peak search interest in the selected period")
