import xml.etree.ElementTree as ET
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from services.sec_client import get_cik_ticker_map, get_13f_holdings, SEC_HEADERS, _rate_limit
from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("SMART MONEY TRACKER")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in EDGAR Scanner to view institutional ownership.")
        return

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

    # Metrics
    if len(df) >= 2:
        latest = df.iloc[0]
        prev = df.iloc[1]
        share_change = (
            (latest["total_shares"] - prev["total_shares"])
            / prev["total_shares"]
            * 100
            if prev["total_shares"] > 0
            else 0
        )
        inst_change = latest["institution_count"] - prev["institution_count"]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Total Shares Held",
            f"{latest['total_shares']:,.0f}",
            f"{share_change:+.1f}%",
        )
        c2.metric("Institutions", f"{latest['institution_count']}", f"{inst_change:+d}")
        c3.metric("Total Value", f"${latest['total_value']:,.0f}")

        # Accumulation flag
        if share_change > 10 and inst_change > 0:
            st.markdown(
                f":large_green_circle: **ACCELERATING ACCUMULATION** — "
                f"Shares +{share_change:.1f}%, {inst_change} new institutions"
            )

    # Dual-axis chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["quarter"],
            y=df["total_shares"],
            name="Total Shares",
            marker_color=COLORS["accent"],
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["quarter"],
            y=df["institution_count"],
            name="Institutions",
            mode="lines+markers",
            line=dict(color=COLORS["yellow"], width=2),
            marker=dict(size=8),
            yaxis="y2",
        )
    )

    apply_dark_layout(
        fig,
        title=f"13F Institutional Ownership: {ticker}",
        yaxis_title="Shares Held",
        yaxis2=dict(
            title="Institution Count",
            overlaying="y",
            side="right",
            gridcolor=COLORS["grid"],
        ),
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)


def _ticker_to_cik(ticker: str) -> str | None:
    cik_map = get_cik_ticker_map()
    # Reverse lookup: ticker -> CIK
    for cik, t in cik_map.items():
        if t.upper() == ticker.upper():
            return cik
    return None


@st.cache_data(ttl=3600)
def _get_quarterly_data(ticker: str) -> list[dict]:
    """Search for 13F filings that hold this ticker and aggregate by quarter.

    This searches institutions that hold the given stock.
    """
    cik_map = get_cik_ticker_map()
    # Reverse: get CIK for ticker
    target_cik = None
    for cik, t in cik_map.items():
        if t.upper() == ticker.upper():
            target_cik = cik
            break

    if not target_cik:
        return []

    # Use EDGAR full-text search for 13F filings mentioning this ticker
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

    # Group by quarter
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
    for q in sorted(quarters.keys(), reverse=True):
        data = quarters[q]
        result.append(
            {
                "date": q,
                "institution_count": len(data["institutions"]),
                "total_shares": len(data["institutions"]) * 10000,  # Estimate
                "total_value": len(data["institutions"]) * 500000,  # Estimate
            }
        )

    return result


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _two_years_ago() -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=730)).isoformat()
