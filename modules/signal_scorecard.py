"""Signal Scorecard — Short Squeeze Screener + Composite Scorecard."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.theme import COLORS, apply_dark_layout
from utils.watchlist import load_watchlist
from services.scoring import score_ticker, scan_short_interest

# Curated list of historically high short-interest names across sectors
_CURATED = [
    "GME", "AMC", "MSTR", "BYND", "UPST", "RIVN", "LCID", "PLUG", "SPCE",
    "SOFI", "HOOD", "AFRM", "OPEN", "RKT",
    "HIMS", "TDOC", "SEER", "NNOX",
    "CHWY", "W", "CVNA",
    "CHPT", "NKLA", "BLNK",
]

_TIER_MAP = {
    "⚡ Freeloader Mode":           (False, None),
    "🧠 Regard Mode (Grok 4.1)": (True,  "grok-4-1-fast-reasoning"),
    "👑 Highly Regarded (Claude)": (True,  "claude-sonnet-4-6"),
}


def _score_color(score: int) -> str:
    if score >= 70:
        return COLORS["positive"]
    elif score >= 40:
        return COLORS["yellow"]
    return COLORS["negative"]


def _short_color(pct: float) -> str:
    if pct >= 0.20:
        return "#ef4444"
    elif pct >= 0.10:
        return "#f59e0b"
    elif pct >= 0.05:
        return "#94a3b8"
    return "#475569"


def render():
    _render_squeeze_screen()


# ── Short Squeeze Screen ───────────────────────────────────────────────────────

def _render_squeeze_screen():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">SHORT SQUEEZE RADAR</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Hunts for high short-interest setups. Scan → click any row → full signal detail + AI thesis."
    )

    # Universe controls
    watchlist = load_watchlist()
    wl_tickers = [w["ticker"] for w in watchlist] if watchlist else []

    c1, c2 = st.columns([2, 1])
    with c1:
        _src = st.radio(
            "Universe", ["Watchlist", "Custom", "Curated (High-SI Names)"],
            horizontal=True, key="sq_src",
        )
        if _src == "Watchlist" and wl_tickers:
            _universe = wl_tickers
            st.caption(f"{len(_universe)} tickers from watchlist")
        elif _src == "Curated (High-SI Names)":
            _universe = _CURATED
            st.caption(f"{len(_universe)} curated high-SI names: meme, fintech, EV, biotech, retail")
        else:
            _raw = st.text_input(
                "Tickers (comma-separated, max 30)",
                value="GME, MSTR, BYND, UPST, RIVN, SOFI, HOOD, AFRM, CVNA, PLUG",
                key="sq_tickers",
            )
            _universe = [t.strip().upper() for t in _raw.split(",") if t.strip()][:30]

    with c2:
        _min_short = st.slider("Min Short % Float", 0, 40, 5, key="sq_min_short")
        _min_dtc = st.slider("Min Days-to-Cover", 0, 10, 0, key="sq_min_dtc")

    if not _universe:
        st.warning("Add at least one ticker.")
        return

    col_scan, col_clear = st.columns([1, 5])
    with col_scan:
        _do_scan = st.button("SCAN UNIVERSE", type="primary", key="sq_scan")
    with col_clear:
        if st.button("Clear", key="sq_clear"):
            for _k in ("sq_scan_results", "sq_drill_result", "sq_selected_ticker", "sq_thesis"):
                st.session_state.pop(_k, None)
            st.rerun()

    if _do_scan:
        with st.spinner(f"Scanning {len(_universe)} tickers for short interest data..."):
            _raw_results = scan_short_interest(tuple(_universe))
        st.session_state["sq_scan_results"] = _raw_results
        # Clear previous drill on fresh scan
        for _k in ("sq_drill_result", "sq_selected_ticker", "sq_thesis"):
            st.session_state.pop(_k, None)

    _all_results = st.session_state.get("sq_scan_results")
    if not _all_results:
        st.info("Click **SCAN UNIVERSE** to screen for squeeze setups.")
        return

    # Apply filters
    _results = [
        r for r in _all_results
        if r["short_pct"] * 100 >= _min_short and r["days_to_cover"] >= _min_dtc
    ]

    if not _results:
        st.warning("No tickers match the current filters. Try lowering the thresholds.")
        return

    # Summary metrics
    _extreme = sum(1 for r in _results if r["short_pct"] >= 0.20)
    _elevated = sum(1 for r in _results if 0.10 <= r["short_pct"] < 0.20)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scanned", len(_all_results))
    m2.metric("Showing", len(_results))
    m3.metric("🔴 Extreme (>20%)", _extreme)
    m4.metric("🟡 Elevated (10-20%)", _elevated)

    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin:14px 0 4px 0;">RANKED BY SQUEEZE SCORE — click any row to analyze</div>',
        unsafe_allow_html=True,
    )

    # Build dataframe for interactive selection
    _df_rows = []
    for r in _results:
        _chk = r.get("checks", {})
        _setup_count = sum([
            _chk.get("short_pct", False),
            _chk.get("days_cover", False),
            _chk.get("inst_buying", False),
        ])
        _setup_str = ("✅✅✅" if _setup_count == 3
                      else "✅✅⬜" if _setup_count == 2
                      else "✅⬜⬜" if _setup_count == 1
                      else "⬜⬜⬜")
        _df_rows.append({
            "Ticker": r["ticker"],
            "Name": r["name"][:22],
            "Short % Float": round(r["short_pct"] * 100, 1),
            "Days-to-Cover": round(r["days_to_cover"], 1),
            "Squeeze Score": r["squeeze_score"],
            "Inst. Own%": round(r["inst_pct"] * 100, 0),
            "Setup": _setup_str,
        })

    _df = pd.DataFrame(_df_rows)

    _event = st.dataframe(
        _df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Short % Float": st.column_config.NumberColumn(format="%.1f%%"),
            "Days-to-Cover": st.column_config.NumberColumn(format="%.1fd"),
            "Inst. Own%": st.column_config.NumberColumn(format="%.0f%%"),
            "Squeeze Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d"
            ),
        },
        hide_index=True,
        key="sq_table",
    )

    st.caption(
        "Setup: ✅ Short % ≥ 10% &nbsp;·&nbsp; "
        "✅ Days-to-cover ≥ 3 &nbsp;·&nbsp; "
        "✅ Institutional ownership ≥ 30%"
    )

    # ── Auto-analyze on row click ──────────────────────────────────────────────
    _sel_rows = _event.selection.rows if _event and hasattr(_event, "selection") else []
    if _sel_rows:
        _sel_idx = _sel_rows[0]
        if 0 <= _sel_idx < len(_results):
            _clicked_ticker = _results[_sel_idx]["ticker"]
            _prev_ticker = st.session_state.get("sq_selected_ticker")
            if _clicked_ticker != _prev_ticker:
                st.session_state["sq_selected_ticker"] = _clicked_ticker
                st.session_state.pop("sq_drill_result", None)
                st.session_state.pop("sq_thesis", None)
            _selected = _clicked_ticker
        else:
            _selected = st.session_state.get("sq_selected_ticker")
    else:
        _selected = st.session_state.get("sq_selected_ticker")

    if not _selected:
        return

    # Auto-run full score if not cached for this ticker
    _drill = st.session_state.get("sq_drill_result")
    if not _drill or _drill.get("ticker") != _selected:
        with st.spinner(f"Running full 6-category analysis on {_selected}..."):
            _full = score_ticker(_selected)
        st.session_state["sq_drill_result"] = _full
        _drill = _full

    st.markdown("---")
    _scan_row = next((r for r in _results if r["ticker"] == _selected), {})
    _render_squeeze_detail(_drill, _scan_row)


def _render_squeeze_detail(r: dict, scan_row: dict) -> None:
    """Full detail panel — radar + squeeze checklist + AI thesis."""
    _ticker = r["ticker"]
    _short_pct = scan_row.get("short_pct", 0.0)
    _dtc = scan_row.get("days_to_cover", 0.0)
    _price = scan_row.get("price", 0.0)

    # Header bar
    _sc_col = _short_color(_short_pct)
    st.markdown(
        f'<div style="border:1px solid #1e293b;border-radius:8px;padding:10px 18px;'
        f'background:#0f172a;display:flex;align-items:baseline;gap:14px;'
        f'flex-wrap:wrap;margin:10px 0;">'
        f'<span style="font-size:18px;font-weight:700;color:#f1f5f9;">{_ticker}</span>'
        + (f'<span style="font-size:14px;color:#94a3b8;">${_price:,.2f}</span>' if _price else "")
        + f'<span style="color:#334155;">·</span>'
        f'<span style="font-size:13px;font-weight:700;color:{_sc_col};">'
        f'Short: {_short_pct * 100:.1f}% float</span>'
        f'<span style="font-size:12px;color:#94a3b8;">DTC: {_dtc:.1f}d</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Squeeze setup checklist — 4 checks including price vs 50-SMA
    _chk = dict(scan_row.get("checks", {}))
    _tech_det = r.get("details", {}).get("technicals", {})
    _sma50 = _tech_det.get("sma50")
    _curr_p = _tech_det.get("price")
    _above_50sma = bool(_curr_p and _sma50 and _curr_p > _sma50)
    _chk["above_50sma"] = _above_50sma

    _check_defs = [
        ("short_pct",   "Short % ≥ 10%",       f"{_short_pct * 100:.1f}% of float shorted"),
        ("days_cover",  "Days-to-Cover ≥ 3",    f"{_dtc:.1f}d to exit shorts"),
        ("inst_buying", "Institutional ≥ 30%",  f"{scan_row.get('inst_pct', 0) * 100:.0f}% inst. owned"),
        ("above_50sma", "Price above 50-SMA",   "Uptrend" if _above_50sma else "Below 50-SMA"),
    ]

    _ck_cols = st.columns(4)
    for i, (key, label, detail) in enumerate(_check_defs):
        _pass = _chk.get(key, False)
        _ck_col = "#22c55e" if _pass else "#ef4444"
        _ck_bg = "#0a2218" if _pass else "#1f0a0a"
        _ck_cols[i].markdown(
            f'<div style="border:1px solid {_ck_col}44;border-radius:6px;'
            f'padding:10px 12px;background:{_ck_bg};text-align:center;">'
            f'<div style="font-size:20px;">{"✅" if _pass else "❌"}</div>'
            f'<div style="font-size:10px;color:{_ck_col};font-weight:700;margin:4px 0 2px 0;">'
            f'{label}</div>'
            f'<div style="font-size:10px;color:#64748b;">{detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Radar chart — red theme for squeeze
    _categories = ["Technicals", "Fundamentals", "Insider", "Options", "Congress", "Short Int"]
    _values = [
        r.get("technicals", 50), r.get("fundamentals", 50), r.get("insider", 50),
        r.get("options", 50), r.get("congress", 50), r.get("short_interest", 50),
    ]
    _composite = r.get("composite", 0)

    _fig = go.Figure()
    _fig.add_trace(go.Scatterpolar(
        r=_values + [_values[0]],
        theta=_categories + [_categories[0]],
        fill="toself",
        fillcolor="rgba(239,68,68,0.12)",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=6, color="#ef4444"),
        name=_ticker,
    ))
    apply_dark_layout(
        _fig,
        title=f"{_ticker} — Full Signal Breakdown  (Composite: {_composite})",
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(range=[0, 100], gridcolor=COLORS["grid"],
                            tickfont=dict(color=COLORS["text_dim"])),
            angularaxis=dict(gridcolor=COLORS["grid"],
                             tickfont=dict(color=COLORS["text"])),
        ),
    )
    st.plotly_chart(_fig, use_container_width=True)

    # Category detail cards
    _details = r.get("details", {})
    if _details:
        _labels = ["Technicals", "Fundamentals", "Insider", "Options", "Congress", "Short Int"]
        _keys = ["technicals", "fundamentals", "insider", "options", "congress", "short_interest"]
        _dcols = st.columns(6)
        for i, key in enumerate(_keys):
            _dat = _details.get(key, {})
            _cat_score = r.get(key, 50)
            _cat_col = _score_color(_cat_score)
            with _dcols[i]:
                st.markdown(
                    f'<div style="font-size:11px;font-weight:700;color:{_cat_col};">'
                    f'{_labels[i].upper()}'
                    f'<div style="font-size:20px;margin:2px 0;">{_cat_score}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if isinstance(_dat, dict):
                    for k, v in _dat.items():
                        if v is not None:
                            st.markdown(
                                f'<div style="font-size:10px;color:#64748b;">'
                                f'{k}: <span style="color:#94a3b8;">{v}</span></div>',
                                unsafe_allow_html=True,
                            )

    # ── AI Squeeze Thesis ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin-bottom:6px;">AI SQUEEZE THESIS</div>',
        unsafe_allow_html=True,
    )

    _tier_sel = st.radio(
        "Engine", list(_TIER_MAP.keys()), horizontal=True, key="sq_thesis_engine"
    )
    st.markdown(
        '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
        'margin-top:-10px;margin-bottom:6px;">'
        '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast-reasoning &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
        '</div>',
        unsafe_allow_html=True,
    )
    _use_cl, _model = _TIER_MAP[_tier_sel]

    if st.button("GENERATE SQUEEZE THESIS", type="primary", key="sq_gen_thesis"):
        from services.claude_client import generate_squeeze_thesis
        with st.spinner(f"Generating squeeze thesis for {_ticker}..."):
            _thesis = generate_squeeze_thesis(
                _ticker, scan_row, r, use_claude=_use_cl, model=_model
            )
        st.session_state["sq_thesis"] = {"ticker": _ticker, "text": _thesis}

    _cached_thesis = st.session_state.get("sq_thesis")
    if _cached_thesis and _cached_thesis.get("ticker") == _ticker:
        _border = f"1px solid {COLORS['bloomberg_orange']}44"
        st.markdown(
            f'<div style="border:{_border};border-radius:6px;padding:12px 16px;'
            f'background:#1A1F2E;margin-top:8px;white-space:pre-line;line-height:1.9;">'
            f'{_cached_thesis["text"]}</div>',
            unsafe_allow_html=True,
        )
