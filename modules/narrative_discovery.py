import plotly.graph_objects as go
import streamlit as st

from services.trends_client import (
    get_interest_over_time,
    get_interest_over_time_multi,
    get_trending_searches,
    get_yahoo_trending_tickers,
)
from services.claude_client import classify_narrative, describe_company, group_tickers_by_narrative
from services.sec_client import get_company_info
from utils.session import get_ticker, set_narrative, set_ticker
from utils.theme import COLORS, apply_dark_layout

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


def render():
    st.header("NARRATIVE DISCOVERY")

    mode = st.radio("Mode", ["Auto — Trending", "Manual"], horizontal=True)

    if mode == "Auto — Trending":
        _render_auto()
    else:
        _render_manual()


def _render_auto():
    # --- Yahoo Finance trending tickers (primary, always financial) ---
    with st.spinner("Fetching trending tickers from Yahoo Finance..."):
        yf_trending = get_yahoo_trending_tickers()

    if yf_trending:
        # Build a lookup from symbol → item
        ticker_lookup = {item["symbol"]: item for item in yf_trending}

        # Group tickers by narrative theme via AI
        import json as _json
        tickers_for_grouping = _json.dumps(
            [{"symbol": t["symbol"], "name": t["name"]} for t in yf_trending]
        )
        with st.spinner("Grouping tickers by narrative..."):
            narrative_groups = group_tickers_by_narrative(tickers_for_grouping)

        if narrative_groups:
            st.subheader(f"{len(yf_trending)} Trending Tickers · {len(narrative_groups)} Narratives")

            for g_idx, group in enumerate(narrative_groups):
                narrative_title = group.get("narrative", "Market Movers")
                description = group.get("description", "")
                group_tickers = group.get("tickers", [])

                # Narrative header
                st.markdown(
                    f'<div style="background:{COLORS["surface"]}; padding:12px 16px; '
                    f'border-radius:8px; border-left:4px solid {COLORS["accent"]}; '
                    f'margin:16px 0 8px 0;">'
                    f'<div style="color:{COLORS["accent"]}; font-size:18px; font-weight:700;">'
                    f'{narrative_title}</div>'
                    f'<div style="color:{COLORS["text_dim"]}; font-size:13px; margin-top:2px;">'
                    f'{description}</div></div>',
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
    else:
        st.warning("Could not fetch Yahoo Finance trending tickers.")

    # --- Show overview for selected ticker ---
    active = get_ticker()
    if active:
        _render_company_overview(active)

    # --- Trending search interest (auto, no click needed) ---
    if yf_trending:
        _render_trending_interest(yf_trending)

    # --- Google Trends supplement (filtered for financial topics) ---
    with st.expander("Google Trends (filtered)", expanded=False):
        with st.spinner("Fetching Google Trends..."):
            topics = get_trending_searches()

        if not topics:
            st.caption("No trending topics from Google Trends.")
        else:
            st.caption(
                f"{len(topics)} trending topics · filtering for financial relevance..."
            )
            classified = []
            for topic in topics[:20]:
                if any(kw in topic.lower() for kw in _NON_FINANCIAL_KEYWORDS):
                    continue
                result = classify_narrative(topic)
                if result.get("market_relevant"):
                    result["topic"] = topic
                    classified.append(result)

            if not classified:
                st.info("No market-relevant narratives in current Google Trends.")
            else:
                st.subheader(f"{len(classified)} Market-Relevant Narratives")
                cols = st.columns(3)
                for i, item in enumerate(classified):
                    col = cols[i % 3]
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{item['topic']}**")
                            st.caption(item.get("sector", ""))
                            st.markdown(
                                item.get("thesis", ""), unsafe_allow_html=False
                            )
                            tickers = item.get("suggested_tickers", [])
                            if tickers:
                                st.code(", ".join(tickers), language=None)
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("Track", key=f"gt_track_{i}"):
                                    set_narrative(item["topic"])
                                    st.rerun()
                            with col_b:
                                if tickers and st.button(
                                    tickers[0], key=f"gt_ticker_{i}"
                                ):
                                    set_narrative(item["topic"])
                                    set_ticker(tickers[0])
                                    st.rerun()


def _render_manual():
    tab_narrative, tab_ticker = st.tabs(["Narrative Keyword", "Ticker Symbol"])

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
            _render_company_overview(active)
            _render_interest_chart(active, key_suffix="narrative")

    with tab_ticker:
        ticker_input = st.text_input(
            "Enter a ticker symbol",
            placeholder="e.g. AAPL, TSLA, NVDA",
        ).strip().upper()
        if st.button("Set Ticker", type="primary", key="set_ticker_btn") and ticker_input:
            set_ticker(ticker_input)
            st.rerun()

        # Show company overview and interest chart for the active ticker
        active = get_ticker()
        if active:
            st.success(f"Active ticker: **{active}**")
            _render_company_overview(active)
            _render_interest_chart(active, key_suffix="ticker")


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
    with st.spinner(f"Looking up {ticker}..."):
        company_info = get_company_info(ticker)

    if not company_info:
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
