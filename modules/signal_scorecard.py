"""Signal Scorecard — Short Squeeze Screener + Composite Scorecard."""

import streamlit as st
import plotly.graph_objects as go
from utils.theme import COLORS, apply_dark_layout
from utils.watchlist import load_watchlist
from services.scoring import score_multiple, score_ticker, scan_short_interest

# Curated list of historically high short-interest names across sectors
_CURATED = [
    "GME", "AMC", "MSTR", "BYND", "UPST", "RIVN", "LCID", "PLUG", "SPCE",
    "SOFI", "HOOD", "AFRM", "OPEN", "RKT",
    "HIMS", "TDOC", "SEER", "NNOX",
    "CHWY", "W", "CVNA",
    "CHPT", "NKLA", "BLNK",
]


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
    tab_squeeze, tab_composite = st.tabs(["🎯 Short Squeeze Screen", "📊 Composite Scorecard"])
    with tab_squeeze:
        _render_squeeze_screen()
    with tab_composite:
        _render_composite_scorecard()


# ── Short Squeeze Screen ───────────────────────────────────────────────────────

def _render_squeeze_screen():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">SHORT SQUEEZE SCREEN</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Hunts for high short-interest setups across a universe of tickers. "
        "Scan → rank by squeeze potential → click any ticker for full signal detail."
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
            st.session_state.pop("sq_scan_results", None)
            st.session_state.pop("sq_drill_result", None)
            st.rerun()

    if _do_scan:
        with st.spinner(f"Scanning {len(_universe)} tickers for short interest data..."):
            _raw_results = scan_short_interest(tuple(_universe))
        st.session_state["sq_scan_results"] = _raw_results

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
        f'margin:14px 0 6px 0;">RANKED BY SQUEEZE SCORE</div>',
        unsafe_allow_html=True,
    )

    # Ranked table
    _tbl = (
        f'<table style="width:100%;border-collapse:collapse;'
        f'font-family:JetBrains Mono,monospace;font-size:12px;">'
        f'<tr style="border-bottom:2px solid {COLORS["bloomberg_orange"]};">'
    )
    for _h in ["#", "Ticker", "Name", "Short % Float", "Days-to-Cover",
               "Squeeze Score", "Inst. Own%", "Setup"]:
        _tbl += (
            f'<th style="padding:5px 10px;text-align:left;'
            f'color:{COLORS["bloomberg_orange"]};">{_h}</th>'
        )
    _tbl += "</tr>"

    for i, r in enumerate(_results):
        _bg = COLORS["surface"] if i % 2 == 0 else COLORS["bg"]
        _sc_col = _short_color(r["short_pct"])
        _sq_col = _score_color(r["squeeze_score"])
        _chk = r.get("checks", {})
        _setup = "".join([
            f'<span title="Short % ≥ 10%">{"✅" if _chk.get("short_pct") else "⬜"}</span>',
            f'<span title="Days-to-cover ≥ 3">{"✅" if _chk.get("days_cover") else "⬜"}</span>',
            f'<span title="Institutional ≥ 30%">{"✅" if _chk.get("inst_buying") else "⬜"}</span>',
        ])
        _tbl += (
            f'<tr style="background:{_bg};">'
            f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};">{i + 1}</td>'
            f'<td style="padding:5px 10px;font-weight:700;color:{COLORS["text"]};">'
            f'{r["ticker"]}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};font-size:11px;">'
            f'{r["name"][:22]}</td>'
            f'<td style="padding:5px 10px;color:{_sc_col};font-weight:700;">'
            f'{r["short_pct"] * 100:.1f}%</td>'
            f'<td style="padding:5px 10px;color:{COLORS["text"]};">'
            f'{r["days_to_cover"]:.1f}d</td>'
            f'<td style="padding:5px 10px;color:{_sq_col};font-weight:800;font-size:16px;">'
            f'{r["squeeze_score"]}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};">'
            f'{r["inst_pct"] * 100:.0f}%</td>'
            f'<td style="padding:5px 10px;letter-spacing:3px;">{_setup}</td>'
            f'</tr>'
        )
    _tbl += "</table>"
    st.markdown(_tbl, unsafe_allow_html=True)

    st.caption(
        "Setup: ✅ Short % ≥ 10% &nbsp;·&nbsp; "
        "✅ Days-to-cover ≥ 3 &nbsp;·&nbsp; "
        "✅ Institutional ownership ≥ 30%"
    )

    # ── Drill-down ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin-bottom:6px;">FULL SIGNAL DETAIL</div>',
        unsafe_allow_html=True,
    )

    _drill_opts = [r["ticker"] for r in _results]
    _selected = st.selectbox("Select ticker", _drill_opts, key="sq_drill")

    if st.button("ANALYZE", type="primary", key="sq_analyze"):
        with st.spinner(f"Running full 6-category analysis on {_selected}..."):
            _full = score_ticker(_selected)
        st.session_state["sq_drill_result"] = _full

    _drill = st.session_state.get("sq_drill_result")
    if _drill and _drill.get("ticker") == _selected:
        _scan_row = next((r for r in _results if r["ticker"] == _selected), {})
        _render_squeeze_detail(_drill, _scan_row)


def _render_squeeze_detail(r: dict, scan_row: dict) -> None:
    """Full detail panel — radar + squeeze checklist + category breakdowns."""
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


# ── Composite Scorecard (existing, unchanged) ──────────────────────────────────

def _render_composite_scorecard():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">COMPOSITE SCORECARD</div>',
        unsafe_allow_html=True,
    )

    col_input, col_weights = st.columns([2, 1])

    with col_input:
        watchlist = load_watchlist()
        wl_tickers = [w["ticker"] for w in watchlist] if watchlist else []
        input_mode = st.radio("Ticker Source", ["Watchlist", "Custom"], horizontal=True, key="sc_mode")
        if input_mode == "Watchlist" and wl_tickers:
            tickers = wl_tickers
            st.caption(f"Scanning: {', '.join(tickers)}")
        else:
            raw = st.text_input(
                "Tickers (comma-separated, max 20)",
                value="AAPL, NVDA, MSFT, TSLA, SPY",
                key="sc_tickers",
            )
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()][:20]

    with col_weights:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin-bottom:4px;">'
            f'CATEGORY WEIGHTS</div>',
            unsafe_allow_html=True,
        )
        w_tech  = st.slider("Technicals",     0, 100, 25, key="w_tech")
        w_fund  = st.slider("Fundamentals",   0, 100, 20, key="w_fund")
        w_ins   = st.slider("Insider",        0, 100, 15, key="w_ins")
        w_opt   = st.slider("Options",        0, 100, 15, key="w_opt")
        w_cong  = st.slider("Congress",       0, 100, 15, key="w_cong")
        w_short = st.slider("Short Interest", 0, 100, 10, key="w_short")

    weights = {
        "technicals": w_tech, "fundamentals": w_fund, "insider": w_ins,
        "options": w_opt, "congress": w_cong, "short_interest": w_short,
    }

    if not tickers:
        st.warning("Enter at least one ticker.")
        return

    if st.button("SCAN", type="primary", key="sc_scan"):
        progress = st.progress(0, text="Scanning...")

        def _upd(pct):
            progress.progress(pct, text=f"Scanning... {int(pct * 100)}%")

        results = score_multiple(tickers, weights, progress_callback=_upd)
        progress.empty()
        st.session_state["sc_results"] = results

    results = st.session_state.get("sc_results")
    if not results:
        st.info("Click SCAN to score tickers.")
        return

    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin:16px 0 8px 0;">RANKED RESULTS</div>',
        unsafe_allow_html=True,
    )

    html = (
        f'<table style="width:100%;border-collapse:collapse;'
        f'font-family:JetBrains Mono,monospace;font-size:13px;">'
        f'<tr style="border-bottom:2px solid {COLORS["bloomberg_orange"]};">'
    )
    for h in ["#", "Ticker", "Composite", "Technicals", "Fundamentals",
              "Insider", "Options", "Congress", "Short Int"]:
        html += (
            f'<th style="padding:6px 10px;text-align:left;'
            f'color:{COLORS["bloomberg_orange"]};">{h}</th>'
        )
    html += "</tr>"
    for i, r in enumerate(results):
        bg = COLORS["surface"] if i % 2 == 0 else COLORS["bg"]
        html += f'<tr style="background:{bg};">'
        html += f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};">{i + 1}</td>'
        html += (
            f'<td style="padding:5px 10px;font-weight:700;color:{COLORS["text"]};">'
            f'{r["ticker"]}</td>'
        )
        for key in ["composite", "technicals", "fundamentals", "insider",
                    "options", "congress", "short_interest"]:
            val = r.get(key, 50)
            html += (
                f'<td style="padding:5px 10px;color:{_score_color(val)};'
                f'font-weight:600;">{val}</td>'
            )
        html += "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("How to Read These Scores"):
        st.markdown(f"""
**Score Ranges**
- <span style="color:{COLORS['positive']};font-weight:600;">70–100</span> — Strong / Bullish
- <span style="color:{COLORS['yellow']};font-weight:600;">40–69</span> — Neutral / Mixed
- <span style="color:{COLORS['negative']};font-weight:600;">0–39</span> — Weak / Bearish

| Category | High Score Means |
|---|---|
| **Technicals** | Price above SMAs, positive momentum |
| **Fundamentals** | Low P/E, strong growth, healthy margins |
| **Insider** | Insiders are net buyers, recent cluster |
| **Options** | High P/C ratio = elevated fear = contrarian bullish |
| **Congress** | Congress members net buyers recently |
| **Short Int** | High short % = squeeze fuel = contrarian bullish |
""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin:20px 0 8px 0;">DRILL-DOWN</div>',
        unsafe_allow_html=True,
    )

    selected = st.selectbox("Select ticker", [r["ticker"] for r in results], key="sc_drill")
    sel_data = next((r for r in results if r["ticker"] == selected), None)

    if sel_data:
        categories = ["Technicals", "Fundamentals", "Insider", "Options", "Congress", "Short Int"]
        values = [
            sel_data["technicals"], sel_data["fundamentals"], sel_data["insider"],
            sel_data["options"], sel_data["congress"], sel_data.get("short_interest", 50),
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(0,212,170,0.15)",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6, color=COLORS["accent"]),
            name=selected,
        ))
        apply_dark_layout(
            fig,
            title=f"{selected} — Score Breakdown (Composite: {sel_data['composite']})",
            polar=dict(
                bgcolor=COLORS["bg"],
                radialaxis=dict(range=[0, 100], gridcolor=COLORS["grid"],
                                tickfont=dict(color=COLORS["text_dim"])),
                angularaxis=dict(gridcolor=COLORS["grid"],
                                 tickfont=dict(color=COLORS["text"])),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        details = sel_data.get("details", {})
        if details:
            cols = st.columns(6)
            for i, (cat, data) in enumerate(details.items()):
                with cols[i]:
                    st.markdown(f"**{cat.upper()}**")
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if v is not None:
                                st.markdown(f"{k}: `{v}`")
