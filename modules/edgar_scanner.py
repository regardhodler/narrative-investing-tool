import streamlit as st
import pandas as pd
from services.sec_client import search_filings
from utils.session import get_narrative, set_ticker


def render():
    st.header("EDGAR SIGNAL SCANNER")

    narrative = get_narrative()
    if not narrative:
        st.info("Set an active narrative in Discovery to scan EDGAR filings.")
        return

    st.caption(f'Scanning SEC filings for: **"{narrative}"** · Last 90 days')

    with st.spinner("Searching EDGAR full-text index..."):
        df = search_filings(narrative)

    if df.empty:
        st.warning("No filings found mentioning this keyword.")
        return

    st.success(f"Found {len(df)} filing mentions")

    # Aggregate by company
    agg = (
        df.groupby(["company", "ticker", "cik"])
        .agg(mentions=("form_type", "count"), forms=("form_type", lambda x: ", ".join(x.unique())), latest=("date", "max"))
        .reset_index()
        .sort_values("mentions", ascending=False)
    )

    st.subheader("Companies by Mention Frequency")
    st.dataframe(
        agg[["company", "ticker", "mentions", "forms", "latest"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "company": "Company",
            "ticker": "Ticker",
            "mentions": st.column_config.NumberColumn("Mentions", format="%d"),
            "forms": "Form Types",
            "latest": "Latest Filing",
        },
    )

    # Ticker selection
    tickers_with_data = agg[agg["ticker"] != ""]["ticker"].tolist()
    if tickers_with_data:
        st.subheader("Select Ticker for Downstream Analysis")
        selected = st.selectbox(
            "Ticker",
            tickers_with_data,
            index=0,
            label_visibility="collapsed",
        )
        if st.button("Set Active Ticker", type="primary"):
            set_ticker(selected)
            st.success(f"Active ticker set to **{selected}**")
            st.rerun()

    # Raw filings table (expandable)
    with st.expander("Raw Filing Results"):
        st.dataframe(df, use_container_width=True, hide_index=True)
