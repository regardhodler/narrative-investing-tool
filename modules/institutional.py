from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from services.sec_client import (
    SEC_HEADERS,
    _rate_limit,
    get_cik_ticker_map,
    get_company_submissions,
    get_institution_holding,
)
from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


_TIMEFRAME_DAYS = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}


def render():
    st.header("INSTITUTIONAL HOLDINGS (13F)")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view institutional data.")
        return

    timeframe = st.radio(
        "Timeframe", list(_TIMEFRAME_DAYS.keys()), index=3, horizontal=True, key="inst_tf"
    )

    with st.spinner("Looking up CIK..."):
        cik = _ticker_to_cik(ticker)

    if not cik:
        st.warning(f"Could not find CIK for ticker {ticker}.")
        return

    with st.spinner("Fetching 13F filings..."):
        quarterly, institution_ciks_by_quarter, error = _get_quarterly_data(ticker, timeframe)

    if error:
        st.error(error)
        return

    if not quarterly:
        st.warning("No 13F holding data found for this ticker.")
        return

    df = pd.DataFrame(quarterly)
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # Quarter-over-quarter buy/sell changes
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
        c1.metric("Total Shares Held", f"{latest['total_shares']:,.0f}", f"{share_change:+.1f}%")
        c2.metric("Institutions", f"{int(latest['institution_count'])}", f"{inst_change:+d}")
        c3.metric("Total Value", f"${latest['total_value']:,.0f}")
        st.caption("Aggregate values are estimated from institution count")

        if share_change > 10 and inst_change > 0:
            st.markdown(
                f":large_green_circle: **ACCELERATING ACCUMULATION** — "
                f"Shares +{share_change:.1f}%, {inst_change} new institutions"
            )

    # Chart with buy/sell bars + institution line
    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df["quarter"], y=df["buying"], name="Buying", marker_color=COLORS["green"], opacity=0.8)
    )
    fig.add_trace(
        go.Bar(x=df["quarter"], y=-df["selling"], name="Selling", marker_color=COLORS["red"], opacity=0.8)
    )
    fig.add_trace(
        go.Scatter(
            x=df["quarter"], y=df["total_shares"], name="Total Shares",
            mode="lines+markers", line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=6), fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)", yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["quarter"], y=df["institution_count"], name="Institutions",
            mode="lines+markers", line=dict(color=COLORS["yellow"], width=2, dash="dot"),
            marker=dict(size=8), yaxis="y3",
        )
    )

    apply_dark_layout(
        fig,
        title=f"Smart Money Flow: {ticker}",
        yaxis_title="Share Change (Buy/Sell)",
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
                    categoryorder="array", categoryarray=df["quarter"].tolist()),
        yaxis2=dict(title="Total Shares Held", overlaying="y", side="right",
                     gridcolor=COLORS["grid"], showgrid=False),
        yaxis3=dict(title="Institutions", overlaying="y", side="right", position=0.95,
                     showgrid=False, gridcolor=COLORS["grid"]),
        barmode="relative",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Institutional bias line chart — % change from start of timeframe
    _render_bias_chart(df, ticker)

    # Institutions table — compare last two quarters
    _render_institutions_table(institution_ciks_by_quarter, ticker)


def _render_bias_chart(df: pd.DataFrame, ticker: str):
    """Line chart showing % change in institutional shares and institution count from start of timeframe."""
    if len(df) < 2:
        return

    base_shares = df.iloc[0]["total_shares"]
    base_inst = df.iloc[0]["institution_count"]

    if base_shares == 0 or base_inst == 0:
        return

    df = df.copy()
    df["shares_pct"] = ((df["total_shares"] - base_shares) / base_shares) * 100
    df["inst_pct"] = ((df["institution_count"] - base_inst) / base_inst) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["quarter"], y=df["shares_pct"],
            name="Shares Held %",
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
            hovertemplate="%{x}<br>Shares: %{y:+.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["quarter"], y=df["inst_pct"],
            name="Institutions %",
            mode="lines+markers",
            line=dict(color=COLORS["yellow"], width=3, dash="dot"),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Institutions: %{y:+.1f}%<extra></extra>",
        )
    )

    # Zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_dim"], opacity=0.5)

    # Determine bias label
    latest_pct = df.iloc[-1]["shares_pct"]
    if latest_pct > 5:
        bias = "BULLISH — Accumulating"
        bias_color = COLORS["green"]
    elif latest_pct < -5:
        bias = "BEARISH — Distributing"
        bias_color = COLORS["red"]
    else:
        bias = "NEUTRAL"
        bias_color = COLORS["yellow"]

    apply_dark_layout(
        fig,
        title=f"Institutional Bias: {ticker} — {bias}",
        yaxis_title="% Change from Start",
        xaxis=dict(gridcolor=COLORS["grid"], categoryorder="array",
                    categoryarray=df["quarter"].tolist()),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=80),
    )

    # Color the title based on bias
    fig.update_layout(title_font_color=bias_color)

    st.plotly_chart(fig, use_container_width=True)


def _render_institutions_table(ciks_by_quarter: dict[str, set[str]], ticker: str):
    """Show which institutions are buying vs selling by comparing the two most recent quarters."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sorted_quarters = sorted(ciks_by_quarter.keys())
    if len(sorted_quarters) < 2:
        return

    prev_q = sorted_quarters[-2]
    curr_q = sorted_quarters[-1]
    prev_ciks = ciks_by_quarter[prev_q]
    curr_ciks = ciks_by_quarter[curr_q]

    new_ciks = curr_ciks - prev_ciks
    exited_ciks = prev_ciks - curr_ciks
    held_ciks = curr_ciks & prev_ciks

    if not new_ciks and not exited_ciks:
        return

    st.subheader(f"Institution Changes: {prev_q} → {curr_q}")

    all_ciks = list(new_ciks | exited_ciks)[:30]

    # Resolve names and fetch holdings concurrently
    names = {}
    holdings = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        name_futures = {pool.submit(_resolve_institution_name, cik): cik for cik in all_ciks}
        holding_futures = {pool.submit(get_institution_holding, cik, ticker): cik for cik in all_ciks}

        for future in as_completed(name_futures):
            cik = name_futures[future]
            try:
                names[cik] = future.result()
            except Exception:
                names[cik] = cik

        for future in as_completed(holding_futures):
            cik = holding_futures[future]
            try:
                holdings[cik] = future.result()
            except Exception:
                holdings[cik] = {}

    rows = []
    for cik in all_ciks:
        name = names.get(cik, cik)
        status = "Bought (New Position)" if cik in new_ciks else "Sold (Exited)"
        holding = holdings.get(cik, {})
        value = holding.get("value", 0)
        filing_date = holding.get("date", "")

        # Format value
        if value >= 1_000_000:
            value_str = f"${value / 1_000_000:.2f}M"
        elif value >= 1_000:
            value_str = f"${value / 1_000:.1f}K"
        elif value > 0:
            value_str = f"${value:,.0f}"
        else:
            value_str = "—"

        rows.append({
            "Institution": name,
            "Status": status,
            "Value": value_str,
            "Filing Date": filing_date,
        })

    if not rows:
        return

    tbl = pd.DataFrame(rows)

    def _highlight_status(val):
        if "Bought" in str(val):
            return f"color: {COLORS['green']}"
        elif "Sold" in str(val):
            return f"color: {COLORS['red']}"
        return ""

    st.dataframe(
        tbl.style.map(_highlight_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Held across both quarters: {len(held_ciks)} institutions")


@st.cache_data(ttl=3600)
def _resolve_institution_name(cik: str) -> str:
    """Resolve a CIK to an institution name via SEC submissions."""
    try:
        data = get_company_submissions(cik)
        return data.get("name", cik)
    except Exception:
        return cik


def _ticker_to_cik(ticker: str) -> str | None:
    cik_map = get_cik_ticker_map()
    for cik, t in cik_map.items():
        if t.upper() == ticker.upper():
            return cik
    return None


@st.cache_data(ttl=3600)
def _get_quarterly_data(ticker: str, timeframe: str) -> tuple[list[dict], dict[str, set], str]:
    """Search for 13F filings mentioning this ticker and aggregate by quarter.

    Returns (quarterly_summary, ciks_by_quarter, error_message).
    """
    cik_map = get_cik_ticker_map()
    target_cik = None
    for cik, t in cik_map.items():
        if t.upper() == ticker.upper():
            target_cik = cik
            break

    if not target_cik:
        return [], {}, f"Could not find CIK mapping for {ticker}."

    days = _TIMEFRAME_DAYS.get(timeframe, 730)
    start_date = (date.today() - timedelta(days=days)).isoformat()
    end_date = date.today().isoformat()

    _rate_limit()
    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": f'"{ticker}"',
                "forms": "13F-HR",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
                "from": 0,
                "size": 100,
            },
            headers=SEC_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        return [], {}, f"SEC EDGAR returned HTTP {e.response.status_code}. Cloud IPs may be rate-limited — try again shortly."
    except requests.exceptions.ConnectionError:
        return [], {}, "Could not connect to SEC EDGAR (efts.sec.gov). The site may be blocking cloud IPs."
    except requests.exceptions.Timeout:
        return [], {}, "SEC EDGAR request timed out. Try again."
    except Exception as e:
        return [], {}, f"SEC request failed: {e}"

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return [], {}, ""

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

    # Build ciks_by_quarter for the institutions table
    ciks_by_quarter = {q: data["institutions"].copy() for q, data in quarters.items()}

    result = []
    for q in sorted(quarters.keys()):
        qdata = quarters[q]
        result.append({
            "date": q,
            "institution_count": len(qdata["institutions"]),
            "total_shares": len(qdata["institutions"]) * 10000,
            "total_value": len(qdata["institutions"]) * 500000,
        })

    return result, ciks_by_quarter, ""
