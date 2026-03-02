import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("INSTITUTIONAL HOLDINGS (13F)")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view institutional data.")
        return

    with st.spinner("Fetching institutional data..."):
        holders, major, error = _get_institutional_data(ticker)

    if error:
        st.error(error)
        return

    # --- Major holders summary ---
    if major is not None and not major.empty:
        _render_major_holders(major, ticker)

    # --- Top institutional holders ---
    if holders is not None and not holders.empty:
        _render_holders_chart(holders, ticker)
        _render_holders_table(holders)
    else:
        st.info("No institutional holder details available for this ticker.")


def _render_major_holders(major: pd.DataFrame, ticker: str):
    """Render summary metrics from major_holders breakdown."""
    # major_holders has rows like: insidersPercentHeld, institutionsPercentHeld, etc.
    data = {}
    for _, row in major.iterrows():
        data[row.iloc[0]] = row.iloc[1]

    inst_pct = data.get("institutionsPercentHeld", data.get("% Held by Institutions", 0))
    insider_pct = data.get("insidersPercentHeld", data.get("% Held by Insiders", 0))
    inst_count = data.get("institutionsCount", data.get("Number of Institutions Holding Shares", 0))
    inst_float_pct = data.get("institutionsFloatPercentHeld", data.get("% of Float Held by Institutions", 0))

    # Convert to percentages if they're decimals
    if isinstance(inst_pct, (int, float)) and inst_pct <= 1:
        inst_pct *= 100
    if isinstance(insider_pct, (int, float)) and insider_pct <= 1:
        insider_pct *= 100
    if isinstance(inst_float_pct, (int, float)) and inst_float_pct <= 1:
        inst_float_pct *= 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Institutional Ownership", f"{inst_pct:.1f}%")
    c2.metric("Insider Ownership", f"{insider_pct:.1f}%")
    c3.metric("Institutions", f"{int(inst_count):,}")
    c4.metric("% of Float (Inst.)", f"{inst_float_pct:.1f}%")

    # Bias indicator
    if inst_pct > 60:
        bias = "HIGH INSTITUTIONAL CONVICTION"
        color = COLORS["green"]
    elif inst_pct > 30:
        bias = "MODERATE INSTITUTIONAL INTEREST"
        color = COLORS["yellow"]
    else:
        bias = "LOW INSTITUTIONAL PRESENCE"
        color = COLORS["red"]

    st.markdown(
        f"<div style='text-align:center; padding:8px; border:1px solid {color}; "
        f"border-radius:8px; margin:10px 0;'>"
        f"<span style='color:{color}; font-size:1.1em; font-weight:bold;'>{bias}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_holders_chart(holders: pd.DataFrame, ticker: str):
    """Horizontal bar chart of top institutional holders by value."""
    df = holders.head(10).copy()
    df = df.sort_values("Value", ascending=True)

    # Color bars by position change
    colors = []
    for _, row in df.iterrows():
        pct_change = row.get("pctChange", 0) or 0
        if pct_change > 0.01:
            colors.append(COLORS["green"])
        elif pct_change < -0.01:
            colors.append(COLORS["red"])
        else:
            colors.append(COLORS["accent"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Holder"],
            x=df["Value"],
            orientation="h",
            marker_color=colors,
            text=df["Value"].apply(lambda v: f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Value: $%{x:,.0f}<extra></extra>",
        )
    )

    apply_dark_layout(
        fig,
        title=f"Top Institutional Holders: {ticker}",
        xaxis_title="Position Value ($)",
        margin=dict(l=200, r=80, t=50, b=40),
        xaxis=dict(showticklabels=False),
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bar color: green = increasing position, red = decreasing, teal = stable")


def _render_holders_table(holders: pd.DataFrame):
    """Color-coded table of institutional holders."""
    st.subheader("Top Institutional Holders")

    display = holders.copy()
    display["% Held"] = display["pctHeld"].apply(
        lambda p: f"{p * 100:.2f}%" if pd.notna(p) else "—"
    )
    display["Shares"] = display["Shares"].apply(
        lambda s: f"{int(s):,}" if pd.notna(s) else "—"
    )
    display["Position Value"] = display["Value"].apply(
        lambda v: f"${v / 1e9:.2f}B" if pd.notna(v) and v >= 1e9
        else f"${v / 1e6:.0f}M" if pd.notna(v) and v >= 1e6
        else f"${v:,.0f}" if pd.notna(v) else "—"
    )
    display["Change"] = display["pctChange"].apply(
        lambda p: f"{p * 100:+.1f}%" if pd.notna(p) else "—"
    )
    display["Date"] = display["Date Reported"].astype(str).str[:10]

    show = display[["Holder", "Shares", "Position Value", "% Held", "Change", "Date"]]

    def _highlight_change(val):
        if "+" in str(val) and val != "—":
            return f"color: {COLORS['green']}"
        elif "-" in str(val) and val != "—":
            return f"color: {COLORS['red']}"
        return ""

    st.dataframe(
        show.style.map(_highlight_change, subset=["Change"]),
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data(ttl=3600)
def _get_institutional_data(ticker: str) -> tuple:
    """Fetch institutional holder data via yfinance.

    Returns (institutional_holders_df, major_holders_df, error_message).
    """
    try:
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders
        major = stock.major_holders
    except Exception as e:
        return None, None, f"Failed to fetch institutional data: {e}"

    if (holders is None or holders.empty) and (major is None or major.empty):
        return None, None, ""

    return holders, major, ""
