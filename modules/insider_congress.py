from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.congress_client import get_congress_trades
from services.sec_client import get_insider_trades
from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


_TIMEFRAME_DAYS = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}


def render():
    st.header("INSIDER & CONGRESS TRADING")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view insider & congress data.")
        return

    tab_insider, tab_congress = st.tabs(["Insider Trading", "Congress Trading"])

    with tab_insider:
        _render_insider(ticker)

    with tab_congress:
        _render_congress(ticker)


def _render_insider(ticker: str):
    st.caption(f"Insider Trading (Form 4): **{ticker}**")

    timeframe = st.radio(
        "Timeframe", list(_TIMEFRAME_DAYS.keys()), index=2, horizontal=True, key="insider_tf"
    )

    with st.spinner("Fetching insider trades..."):
        df = get_insider_trades(ticker)

    if df.empty:
        st.info("No recent insider trades found for this ticker.")
        return

    # Filter by timeframe
    days = _TIMEFRAME_DAYS[timeframe]
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    df["date"] = df["date"].astype(str)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    if df.empty:
        st.info(f"No insider trades in the last {timeframe}.")
        return

    # Metrics
    buys = df[df["type"].isin(["Purchase"])]
    sells = df[df["type"].isin(["Sale"])]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Buys", len(buys), f"${buys['value'].sum():,.0f}" if not buys.empty else "$0")
    c3.metric("Sells", len(sells), f"${sells['value'].sum():,.0f}" if not sells.empty else "$0")

    # Monthly activity bar chart
    _render_monthly_chart(df, ticker)

    # Color-coded table
    def _highlight_type(val):
        if val == "Purchase":
            return f"color: {COLORS['green']}"
        elif val == "Sale":
            return f"color: {COLORS['red']}"
        return ""

    display_df = df.copy()
    display_df["value"] = display_df["value"].apply(lambda v: f"${v:,.0f}" if v else "")
    display_df["shares"] = display_df["shares"].apply(lambda s: f"{s:,.0f}" if s else "")
    display_df["price"] = display_df["price"].apply(lambda p: f"${p:,.2f}" if p else "")

    st.dataframe(
        display_df.style.map(_highlight_type, subset=["type"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "insider_name": "Insider",
            "title": "Title",
            "date": "Date",
            "type": "Type",
            "shares": "Shares",
            "price": "Price",
            "value": "Value",
        },
    )


def _render_monthly_chart(df: pd.DataFrame, ticker: str):
    """Render a monthly bar chart of insider purchase vs sale shares."""
    chart_df = df[df["type"].isin(["Purchase", "Sale"])].copy()
    if chart_df.empty:
        return

    chart_df["month"] = pd.to_datetime(chart_df["date"]).dt.to_period("M").astype(str)

    purchases = chart_df[chart_df["type"] == "Purchase"].groupby("month")["shares"].sum()
    sales = chart_df[chart_df["type"] == "Sale"].groupby("month")["shares"].sum()

    all_months = sorted(set(purchases.index) | set(sales.index))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_months,
            y=[purchases.get(m, 0) for m in all_months],
            name="Purchase",
            marker_color=COLORS["green"],
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_months,
            y=[-sales.get(m, 0) for m in all_months],
            name="Sale",
            marker_color=COLORS["red"],
            opacity=0.8,
        )
    )

    apply_dark_layout(
        fig,
        title=f"Insider Monthly Activity: {ticker}",
        xaxis_title="Month",
        yaxis_title="Shares",
        barmode="relative",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_congress(ticker: str):
    st.caption(f"Congress Trading: **{ticker}**")

    with st.spinner("Fetching congress trades..."):
        try:
            df = get_congress_trades(ticker)
        except Exception:
            st.warning("Capitol Trades data unavailable — site format may have changed.")
            return

    if df.empty:
        st.info("No recent congress trades found for this ticker.")
        return

    st.success(f"Found {len(df)} congress trades for {ticker}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "politician": "Politician",
            "party": "Party",
            "date": "Date",
            "type": "Type",
            "size": "Size",
            "price": "Price",
        },
    )
