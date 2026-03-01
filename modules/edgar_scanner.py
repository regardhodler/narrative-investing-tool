import streamlit as st
from services.sec_client import get_filings_by_ticker
from utils.session import get_ticker, set_ticker


def render():
    st.header("EDGAR FILING SCANNER")

    # Pre-fill with active ticker if set
    default_ticker = get_ticker() or ""

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            value=default_ticker,
            placeholder="e.g. AAPL",
        ).strip().upper()
    with col2:
        st.write("")  # spacing
        search = st.button("Look Up Filings", type="primary")

    if not search and not ticker_input:
        st.info("Enter a ticker symbol to view its recent SEC filings.")
        return

    if not ticker_input:
        st.warning("Please enter a ticker symbol.")
        return

    # Set active ticker button
    if ticker_input and ticker_input != get_ticker():
        if st.button(f"Set **{ticker_input}** as Active Ticker"):
            set_ticker(ticker_input)
            st.success(f"Active ticker set to **{ticker_input}**")
            st.rerun()

    with st.spinner(f"Fetching SEC filings for {ticker_input}..."):
        df = get_filings_by_ticker(ticker_input)

    if df.empty:
        st.warning(f"No filings found for ticker **{ticker_input}**. Check the symbol and try again.")
        return

    st.success(f"Found {len(df)} recent filings for **{ticker_input}**")

    # Form type filter
    form_types = sorted(df["form_type"].unique())
    selected_forms = st.multiselect(
        "Filter by Form Type",
        options=form_types,
        default=None,
        placeholder="All form types",
    )

    filtered = df[df["form_type"].isin(selected_forms)] if selected_forms else df

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "form_type": "Form Type",
            "date": "Filing Date",
            "description": "Description",
            "accession_number": "Accession #",
            "url": st.column_config.LinkColumn("Link", display_text="View"),
        },
    )
