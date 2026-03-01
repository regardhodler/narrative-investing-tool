import streamlit as st
from services.trends_client import get_trending_searches
from services.claude_client import classify_narrative, describe_company
from services.sec_client import get_company_info
from utils.session import get_ticker, set_narrative, set_ticker


def render():
    st.header("NARRATIVE DISCOVERY")

    mode = st.radio("Mode", ["Auto — Trending", "Manual"], horizontal=True)

    if mode == "Auto — Trending":
        _render_auto()
    else:
        _render_manual()


def _render_auto():
    with st.spinner("Fetching trending searches..."):
        topics = get_trending_searches()

    if not topics:
        st.warning("No trending topics found. Try Manual mode.")
        return

    st.caption(f"{len(topics)} trending topics · classifying market relevance...")

    # Classify all topics
    classified = []
    for topic in topics[:20]:  # Limit to top 20
        result = classify_narrative(topic)
        if result.get("market_relevant"):
            result["topic"] = topic
            classified.append(result)

    if not classified:
        st.info("No market-relevant narratives found in current trends.")
        return

    st.subheader(f"{len(classified)} Market-Relevant Narratives")

    # 3-column grid of cards
    cols = st.columns(3)
    for i, item in enumerate(classified):
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                st.markdown(f"**{item['topic']}**")
                st.caption(item.get("sector", ""))
                st.markdown(item.get("thesis", ""), unsafe_allow_html=False)

                tickers = item.get("suggested_tickers", [])
                if tickers:
                    st.code(", ".join(tickers), language=None)

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Track", key=f"track_{i}"):
                        set_narrative(item["topic"])
                        st.rerun()
                with col_b:
                    if tickers and st.button(tickers[0], key=f"ticker_{i}"):
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

            if result.get("market_relevant"):
                st.success(f"Narrative set: **{keyword}**")
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

            with st.spinner(f"Looking up {active}..."):
                company_info = get_company_info(active)

            if company_info:
                with st.spinner("Generating company overview..."):
                    overview = describe_company(
                        company_info["name"],
                        active,
                        company_info["sic_description"],
                    )

                with st.container(border=True):
                    st.subheader(f"{company_info['name']}  ·  {active}")
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
