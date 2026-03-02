import streamlit as st
from services.trends_client import get_trending_searches, get_yahoo_trending_tickers
from services.claude_client import classify_narrative, describe_company
from services.sec_client import get_company_info
from utils.session import get_ticker, set_narrative, set_ticker

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
        st.subheader(f"{len(yf_trending)} Trending Tickers")
        cols = st.columns(3)
        for i, item in enumerate(yf_trending):
            col = cols[i % 3]
            with col:
                with st.container(border=True):
                    st.markdown(f"**{item['name']}**")
                    st.code(item["symbol"], language=None)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Track", key=f"yf_track_{i}"):
                            set_narrative(item["name"])
                            set_ticker(item["symbol"])
                            st.rerun()
                    with col_b:
                        if st.button("Analyze", key=f"yf_analyze_{i}"):
                            set_narrative(item["name"])
                            set_ticker(item["symbol"])
                            st.rerun()
    else:
        st.warning("Could not fetch Yahoo Finance trending tickers.")

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

        # Show company overview for confirmed ticker
        active = get_ticker()
        if active and result:
            _render_company_overview(active)

    with tab_ticker:
        ticker_input = st.text_input(
            "Enter a ticker symbol",
            placeholder="e.g. AAPL, TSLA, NVDA",
        ).strip().upper()
        if st.button("Set Ticker", type="primary", key="set_ticker_btn") and ticker_input:
            set_ticker(ticker_input)
            st.rerun()

        # Show company overview for the active ticker
        active = get_ticker()
        if active:
            st.success(f"Active ticker: **{active}**")
            _render_company_overview(active)


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
