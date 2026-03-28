"""Signal Scorecard — multi-ticker scoring and ranking dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import COLORS, apply_dark_layout
from utils.watchlist import load_watchlist
from services.scoring import score_multiple


def _score_color(score: int) -> str:
    """Return hex color for score 0-100 (red → yellow → green)."""
    if score >= 70:
        return COLORS["positive"]
    elif score >= 40:
        return COLORS["yellow"]
    return COLORS["negative"]


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">SIGNAL SCORECARD</div>',
        unsafe_allow_html=True,
    )

    # --- Input ---
    col_input, col_weights = st.columns([2, 1])

    with col_input:
        watchlist = load_watchlist()
        wl_tickers = [w["ticker"] for w in watchlist] if watchlist else []

        input_mode = st.radio("Ticker Source", ["Watchlist", "Custom"], horizontal=True, key="sc_mode")
        if input_mode == "Watchlist" and wl_tickers:
            tickers = wl_tickers
            st.caption(f"Scanning: {', '.join(tickers)}")
        else:
            raw = st.text_input("Tickers (comma-separated, max 20)", value="AAPL, NVDA, MSFT, TSLA, SPY",
                                key="sc_tickers")
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()][:20]

    with col_weights:
        st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin-bottom:4px;">'
                    f'CATEGORY WEIGHTS</div>', unsafe_allow_html=True)
        w_tech = st.slider("Technicals", 0, 100, 25, key="w_tech")
        w_fund = st.slider("Fundamentals", 0, 100, 20, key="w_fund")
        w_ins = st.slider("Insider", 0, 100, 15, key="w_ins")
        w_opt = st.slider("Options", 0, 100, 15, key="w_opt")
        w_cong = st.slider("Congress", 0, 100, 15, key="w_cong")
        w_short = st.slider("Short Interest", 0, 100, 10, key="w_short")

    weights = {"technicals": w_tech, "fundamentals": w_fund, "insider": w_ins, "options": w_opt, "congress": w_cong, "short_interest": w_short}

    if not tickers:
        st.warning("Enter at least one ticker.")
        return

    # --- Scan ---
    if st.button("SCAN", type="primary", key="sc_scan"):
        progress = st.progress(0, text="Scanning...")

        def update_progress(pct):
            progress.progress(pct, text=f"Scanning... {int(pct * 100)}%")

        results = score_multiple(tickers, weights, progress_callback=update_progress)
        progress.empty()
        st.session_state["sc_results"] = results

    results = st.session_state.get("sc_results")
    if not results:
        st.info("Click SCAN to score tickers.")
        return

    # --- Ranked Table ---
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:16px 0 8px 0;">RANKED RESULTS</div>', unsafe_allow_html=True)

    # Build colored HTML table
    html = '<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:13px;">'
    html += '<tr style="border-bottom:2px solid ' + COLORS["bloomberg_orange"] + ';">'
    for h in ["#", "Ticker", "Composite", "Technicals", "Fundamentals", "Insider", "Options", "Congress", "Short Int"]:
        html += f'<th style="padding:6px 10px;text-align:left;color:{COLORS["bloomberg_orange"]};">{h}</th>'
    html += '</tr>'

    for i, r in enumerate(results):
        bg = COLORS["surface"] if i % 2 == 0 else COLORS["bg"]
        html += f'<tr style="background:{bg};">'
        html += f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};">{i+1}</td>'
        html += f'<td style="padding:5px 10px;font-weight:700;color:{COLORS["text"]};">{r["ticker"]}</td>'
        for key in ["composite", "technicals", "fundamentals", "insider", "options", "congress", "short_interest"]:
            val = r.get(key, 50)
            color = _score_color(val)
            html += f'<td style="padding:5px 10px;color:{color};font-weight:600;">{val}</td>'
        html += '</tr>'
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)

    # --- Interpretive Tips ---
    with st.expander("How to Read These Scores"):
        st.markdown(f"""
**Score Ranges**
- <span style="color:{COLORS['positive']};font-weight:600;">70 – 100</span> — **Strong / Bullish.** Multiple signals align positively.
- <span style="color:{COLORS['yellow']};font-weight:600;">40 – 69</span> — **Neutral / Mixed.** Conflicting signals; proceed with caution.
- <span style="color:{COLORS['negative']};font-weight:600;">0 – 39</span> — **Weak / Bearish.** Most signals point to caution or downside risk.

---

**What Each Category Measures**

| Category | Key Drivers | High Score Means |
|---|---|---|
| **Technicals** | Price vs SMA-20/50/200, RSI zone, 1-month & 3-month momentum | Price trending above moving averages with positive momentum; oversold RSI can also boost the score (buy-the-dip opportunity) |
| **Fundamentals** | Forward P/E ratio, revenue growth, profit margins | Reasonable valuation combined with strong growth and healthy margins |
| **Insider** | Buy/sell ratio from SEC Form 4 filings, cluster detection (3+ buys in 30 days) | Insiders are net buyers — especially meaningful when purchases cluster in a short window |
| **Options** | Put/Call open-interest ratio (contrarian) | High P/C ratio = elevated fear = contrarian bullish signal; low P/C = complacency = contrarian bearish |
| **Congress** | Congressional buy/sell ratio, recency of trades | Members of Congress are net buyers — recent cluster of purchases is especially noteworthy |
| **Short Int** | Short % of float + days-to-cover (contrarian) | High short interest = more squeeze fuel = contrarian bullish; days-to-cover > 5 signals harder exits for shorts |

---

**How to Use the Composite Score**
- The composite is the weighted average of all five categories (adjust weights with the sliders above).
- **No single category should drive a decision alone** — the composite rewards tickers where multiple signals converge.
- Cross-reference high-scoring tickers with the **Backtesting** module to see which signal categories historically predicted returns.
- Log trades in the **Trade Journal** with the signal source so you can track which scores translate to real P&L over time.
""", unsafe_allow_html=True)

    # --- Drill-down radar chart ---
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:20px 0 8px 0;">DRILL-DOWN</div>', unsafe_allow_html=True)

    selected = st.selectbox("Select ticker", [r["ticker"] for r in results], key="sc_drill")
    sel_data = next((r for r in results if r["ticker"] == selected), None)

    if sel_data:
        categories = ["Technicals", "Fundamentals", "Insider", "Options", "Congress", "Short Int"]
        values = [sel_data["technicals"], sel_data["fundamentals"], sel_data["insider"], sel_data["options"], sel_data["congress"], sel_data.get("short_interest", 50)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=f"rgba(0, 212, 170, 0.15)",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6, color=COLORS["accent"]),
            name=selected,
        ))
        apply_dark_layout(fig, title=f"{selected} — Score Breakdown (Composite: {sel_data['composite']})",
                          polar=dict(
                              bgcolor=COLORS["bg"],
                              radialaxis=dict(range=[0, 100], gridcolor=COLORS["grid"],
                                              tickfont=dict(color=COLORS["text_dim"])),
                              angularaxis=dict(gridcolor=COLORS["grid"],
                                               tickfont=dict(color=COLORS["text"])),
                          ))
        st.plotly_chart(fig, use_container_width=True)

        # Show details
        details = sel_data.get("details", {})
        if details:
            cols = st.columns(6)
            for i, (cat, data) in enumerate(details.items()):
                with cols[i]:
                    st.markdown(f'**{cat.upper()}**')
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if v is not None:
                                st.markdown(f'{k}: `{v}`')
