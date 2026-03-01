import streamlit as st
from services.sec_client import get_filings_by_ticker, get_company_info, fetch_filing_text
from services.claude_client import describe_company, summarize_filing
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

    # --- Company Overview ---
    with st.spinner(f"Looking up {ticker_input}..."):
        company_info = get_company_info(ticker_input)

    if not company_info:
        st.warning(f"Could not find company info for **{ticker_input}**.")
        return

    # AI-generated description and narrative
    with st.spinner("Generating company overview..."):
        overview = describe_company(
            company_info["name"],
            ticker_input,
            company_info["sic_description"],
        )

    with st.container(border=True):
        st.subheader(f"{company_info['name']}  ·  {ticker_input}")
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

    st.markdown("---")

    # --- Filings Table ---
    with st.spinner(f"Fetching SEC filings..."):
        df = get_filings_by_ticker(ticker_input)

    if df.empty:
        st.warning(f"No filings found for **{ticker_input}**.")
        return

    st.success(f"Found {len(df)} recent filings")

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

    # --- Filing Summary ---
    st.markdown("---")
    st.subheader("AI Filing Summary")

    summarizable = filtered[filtered["form_type"].isin(["8-K", "8-K/A", "10-K", "10-Q"])].reset_index(drop=True)

    if summarizable.empty:
        st.caption("No 8-K, 10-K, or 10-Q filings to summarize. Adjust the form type filter above.")
        return

    # Build display labels for selectbox
    options = [
        f"{row['form_type']}  ·  {row['date']}  ·  {row['description']}"
        for _, row in summarizable.iterrows()
    ]

    selected_idx = st.selectbox(
        "Select a filing to summarize",
        range(len(options)),
        format_func=lambda i: options[i],
    )

    if st.button("Summarize Filing", type="primary", key="summarize_btn"):
        row = summarizable.iloc[selected_idx]
        url = row["url"]

        if not url:
            st.error("No filing URL available.")
            return

        with st.spinner(f"Fetching {row['form_type']} filing text..."):
            text = fetch_filing_text(url)

        if not text:
            st.error("Could not fetch filing content from SEC.")
            return

        with st.spinner("Generating AI summary..."):
            summary = summarize_filing(text, row["form_type"], company_info["name"])

        with st.container(border=True):
            st.markdown(f"**{row['form_type']}** — {row['date']}")
            st.markdown(summary)
