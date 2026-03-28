import streamlit as st
from services.sec_client import get_filings_by_ticker, get_company_info, fetch_filing_text
from services.claude_client import summarize_filing
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

    # Look up company info (needed for filing summary context)
    with st.spinner(f"Looking up {ticker_input}..."):
        company_info = get_company_info(ticker_input)

    if not company_info:
        st.warning(f"Could not find company info for **{ticker_input}**.")
        return

    # --- Filings Table ---
    with st.spinner(f"Fetching SEC filings..."):
        df = get_filings_by_ticker(ticker_input)

    if df.empty:
        st.warning(f"No filings found for **{ticker_input}**.")
        return

    from datetime import datetime
    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 1H")
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
    import os as _os
    _edgar_has_xai = bool(_os.getenv("XAI_API_KEY"))

    _edgar_has_claude = bool(_os.getenv("ANTHROPIC_API_KEY"))
    _edgar_tier_opts = ["⚡ Groq"] + (["🧠 Regard Mode"] if _edgar_has_xai else []) + (["👑 Highly Regarded Mode"] if _edgar_has_claude else [])
    _edgar_tier_map = {
        "⚡ Groq": (False, None),
        "🧠 Regard Mode": (True, "grok-4-1-fast-reasoning"),
        "👑 Highly Regarded Mode": (True, "claude-sonnet-4-6"),
    }
    st.markdown("---")
    _sel_edgar_tier = st.radio(
        "Summary Engine", _edgar_tier_opts, horizontal=True, key="edgar_summary_engine",
        help="Standard = Groq (fast) · Regard Mode = Grok 4.1 · Highly Regarded = Claude Sonnet"
    )
    _use_claude_edgar, _edgar_model = _edgar_tier_map[_sel_edgar_tier]
    _edgar_badge = f" {_sel_edgar_tier.split()[0]}"
    st.subheader(f"AI Filing Summary{_edgar_badge}")

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
            summary = summarize_filing(text, row["form_type"], company_info["name"], use_claude=_use_claude_edgar, model=_edgar_model)

        # Persist filing digest for downstream use (valuation, portfolio intelligence)
        from datetime import datetime as _dt_edgar
        st.session_state["_filing_digest"] = {
            "ticker": ticker_input,
            "company": company_info["name"],
            "form_type": row["form_type"],
            "date": str(row["date"]),
            "summary": summary,
            "ts": _dt_edgar.now().isoformat(),
        }
        try:
            from services.signals_cache import save_signals
            save_signals()
        except Exception:
            pass

        _border = "2px solid #FF8811" if _use_claude_edgar else "1px solid #2A3040"
        st.markdown(
            f'<div style="border:{_border};border-left-width:4px;border-radius:6px;'
            f'padding:14px 18px;background:#1A1F2E;margin-bottom:8px;">'
            f'<strong>{row["form_type"]}</strong> — {row["date"]}'
            f'<div style="margin-top:10px;font-size:13px;line-height:1.6;">{summary}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
