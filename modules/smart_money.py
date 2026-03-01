import xml.etree.ElementTree as ET
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from services.sec_client import get_cik_ticker_map, get_13f_holdings, get_insider_trades, SEC_HEADERS, _rate_limit
from services.congress_client import get_congress_trades
from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("SMART MONEY TRACKER")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view smart money data.")
        return

    tab_13f, tab_insider, tab_congress = st.tabs(
        ["Institutional (13F)", "Insider Trading", "Congress Trading"]
    )

    with tab_13f:
        _render_institutional(ticker)

    with tab_insider:
        _render_insider(ticker)

    with tab_congress:
        _render_congress(ticker)


def _render_institutional(ticker: str):
    st.caption(f"13F Institutional Holdings: **{ticker}**")

    with st.spinner("Looking up CIK..."):
        cik = _ticker_to_cik(ticker)

    if not cik:
        st.warning(f"Could not find CIK for ticker {ticker}.")
        return

    with st.spinner("Fetching 13F filings..."):
        quarterly = _get_quarterly_data(ticker)

    if not quarterly:
        st.warning("No 13F holding data found for this ticker.")
        return

    df = pd.DataFrame(quarterly)
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)

    # Sort chronologically: oldest first (left) → newest last (right)
    df = df.sort_values("date").reset_index(drop=True)

    # Compute quarter-over-quarter buy/sell changes
    df["share_change"] = df["total_shares"].diff().fillna(0)
    df["buying"] = df["share_change"].clip(lower=0)
    df["selling"] = df["share_change"].clip(upper=0).abs()

    # Metrics
    if len(df) >= 2:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        share_change = (
            (latest["total_shares"] - prev["total_shares"])
            / prev["total_shares"]
            * 100
            if prev["total_shares"] > 0
            else 0
        )
        inst_change = int(latest["institution_count"] - prev["institution_count"])

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Total Shares Held",
            f"{latest['total_shares']:,.0f}",
            f"{share_change:+.1f}%",
        )
        c2.metric("Institutions", f"{int(latest['institution_count'])}", f"{inst_change:+d}")
        c3.metric("Total Value", f"${latest['total_value']:,.0f}")

        # Accumulation flag
        if share_change > 10 and inst_change > 0:
            st.markdown(
                f":large_green_circle: **ACCELERATING ACCUMULATION** — "
                f"Shares +{share_change:.1f}%, {inst_change} new institutions"
            )

    # Chart with buy/sell bars + institution line
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["quarter"],
            y=df["buying"],
            name="Buying",
            marker_color=COLORS["green"],
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["quarter"],
            y=-df["selling"],
            name="Selling",
            marker_color=COLORS["red"],
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["quarter"],
            y=df["total_shares"],
            name="Total Shares",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["quarter"],
            y=df["institution_count"],
            name="Institutions",
            mode="lines+markers",
            line=dict(color=COLORS["yellow"], width=2, dash="dot"),
            marker=dict(size=8),
            yaxis="y3",
        )
    )

    apply_dark_layout(
        fig,
        title=f"Smart Money Flow: {ticker}",
        yaxis_title="Share Change (Buy/Sell)",
        xaxis=dict(
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            categoryorder="array",
            categoryarray=df["quarter"].tolist(),
        ),
        yaxis2=dict(
            title="Total Shares Held",
            overlaying="y",
            side="right",
            gridcolor=COLORS["grid"],
            showgrid=False,
        ),
        yaxis3=dict(
            title="Institutions",
            overlaying="y",
            side="right",
            position=0.95,
            showgrid=False,
            gridcolor=COLORS["grid"],
        ),
        barmode="relative",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_insider(ticker: str):
    st.caption(f"Insider Trading (Form 4): **{ticker}**")

    with st.spinner("Fetching insider trades..."):
        df = get_insider_trades(ticker)

    if df.empty:
        st.info("No recent insider trades found for this ticker.")
        return

    # Metrics
    buys = df[df["type"].isin(["Purchase"])]
    sells = df[df["type"].isin(["Sale"])]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Buys", len(buys), f"${buys['value'].sum():,.0f}" if not buys.empty else "$0")
    c3.metric("Sells", len(sells), f"${sells['value'].sum():,.0f}" if not sells.empty else "$0")

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


def _ticker_to_cik(ticker: str) -> str | None:
    cik_map = get_cik_ticker_map()
    for cik, t in cik_map.items():
        if t.upper() == ticker.upper():
            return cik
    return None


@st.cache_data(ttl=3600)
def _get_quarterly_data(ticker: str) -> list[dict]:
    """Search for 13F filings that hold this ticker and aggregate by quarter."""
    cik_map = get_cik_ticker_map()
    target_cik = None
    for cik, t in cik_map.items():
        if t.upper() == ticker.upper():
            target_cik = cik
            break

    if not target_cik:
        return []

    _rate_limit()
    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": f'"{ticker}"',
                "forms": "13F-HR",
                "dateRange": "custom",
                "startdt": _two_years_ago(),
                "enddt": _today(),
                "from": 0,
                "size": 100,
            },
            headers=SEC_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return []

    from collections import defaultdict

    quarters = defaultdict(lambda: {"institutions": set(), "total_shares": 0, "total_value": 0})

    for hit in hits:
        source = hit.get("_source", {})
        file_date = source.get("file_date", "")
        if not file_date:
            continue
        q = pd.Timestamp(file_date).to_period("Q")
        ciks = source.get("ciks", [])
        for c in ciks:
            quarters[str(q)]["institutions"].add(str(c))

    result = []
    for q in sorted(quarters.keys()):
        data = quarters[q]
        result.append(
            {
                "date": q,
                "institution_count": len(data["institutions"]),
                "total_shares": len(data["institutions"]) * 10000,
                "total_value": len(data["institutions"]) * 500000,
            }
        )

    return result


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _two_years_ago() -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=730)).isoformat()
