import re
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.congress_client import get_congress_trades
from services.sec_client import get_insider_trades
from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


_TIMEFRAME_DAYS = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}


def _parse_size_midpoint(size_str: str) -> float:
    """Parse a congress trade size range like '$1,001 - $15,000' into its midpoint dollar value."""
    try:
        amounts = [float(v.replace(",", "")) for v in re.findall(r"\$([\d,]+)", str(size_str))]
        if len(amounts) == 2:
            return (amounts[0] + amounts[1]) / 2
        if len(amounts) == 1:
            return amounts[0]
    except (ValueError, TypeError):
        pass
    return 0.0


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

    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 1H")

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

    buy_value = buys["value"].sum() if not buys.empty else 0
    sell_value = sells["value"].sum() if not sells.empty else 0
    total_value = buy_value + sell_value
    insider_buy_pct = (buy_value / total_value * 100) if total_value > 0 else 50

    if insider_buy_pct > 55:
        ins_bias = "BULLISH"
        ins_color = COLORS["green"]
    elif insider_buy_pct < 45:
        ins_bias = "BEARISH"
        ins_color = COLORS["red"]
    else:
        ins_bias = "NEUTRAL"
        ins_color = COLORS["yellow"]

    # Persist for downstream use (valuation, portfolio intelligence)
    st.session_state["_insider_net_flow"] = {
        "ticker": ticker,
        "bias": ins_bias,
        "buy_pct": round(insider_buy_pct, 1),
        "buy_value": int(buy_value),
        "sell_value": int(sell_value),
        "n_trades": len(df),
    }

    st.markdown(
        f"<div style='text-align:center; padding:10px; border:1px solid {ins_color}; border-radius:8px; margin-bottom:10px;'>"
        f"<span style='font-size:2em; color:{ins_color}; font-weight:bold;'>{insider_buy_pct:.1f}%</span>"
        f"<br><span style='color:{ins_color}; font-size:1.2em;'>{ins_bias}</span>"
        f"<br><span style='color:{COLORS['text_dim']}; font-size:0.85em;'>50% = Neutral · Above = Bullish · Below = Bearish · Weighted by $ value</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Buys", len(buys), f"${buys['value'].sum():,.0f}" if not buys.empty else "$0")
    c3.metric("Sells", len(sells), f"${sells['value'].sum():,.0f}" if not sells.empty else "$0")

    # Cumulative buy vs sell flow
    _render_insider_flow(df, ticker)

    # Insider treemap
    _render_insider_treemap(df, ticker)

    # Monthly activity bar chart
    _render_monthly_chart(df, ticker)

    # Trade value timeline
    _render_insider_timeline(df, ticker)

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

    csv = df.to_csv(index=False)
    st.download_button("Export CSV", csv, f"{ticker}_insider_trades.csv", "text/csv", key="dl_insider")


def _render_insider_flow(df: pd.DataFrame, ticker: str):
    """Cumulative net insider flow area chart — running total of buy $ minus sell $."""
    chart_df = df[df["type"].isin(["Purchase", "Sale"]) & (df["value"] > 0)].copy()
    if chart_df.empty:
        return

    chart_df["date_dt"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.sort_values("date_dt")

    # Signed value: positive for buys, negative for sells
    chart_df["signed"] = chart_df.apply(
        lambda r: r["value"] if r["type"] == "Purchase" else -r["value"], axis=1
    )

    # Group by date, sum, then cumulate
    daily = chart_df.groupby("date_dt")["signed"].sum().cumsum().reset_index()
    daily.columns = ["date", "cumulative"]

    last_val = daily["cumulative"].iloc[-1]
    if last_val > 0:
        flow_label, flow_color = "NET BUYING", COLORS["green"]
    elif last_val < 0:
        flow_label, flow_color = "NET SELLING", COLORS["red"]
    else:
        flow_label, flow_color = "NEUTRAL", COLORS["yellow"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["cumulative"],
            mode="lines",
            line=dict(color=flow_color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(flow_color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.12)",
            hovertemplate="%{x|%b %d, %Y}<br>Net Flow: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_dim"], line_width=1)

    apply_dark_layout(
        fig,
        title=dict(
            text=f"Insider Cumulative Flow: {ticker} — {flow_label} (${last_val:,.0f})",
            font=dict(color=flow_color),
        ),
        yaxis_title="Cumulative Net Flow ($)",
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Above zero = net insider buying · Below zero = net insider selling")


def _render_insider_treemap(df: pd.DataFrame, ticker: str):
    """Treemap of insider trades: size = trade value, color = buy/sell."""
    chart_df = df[df["type"].isin(["Purchase", "Sale"]) & (df["value"] > 0)].copy()
    if chart_df.empty:
        return

    # Aggregate by insider name and type
    grouped = chart_df.groupby(["insider_name", "type"]).agg(
        total_value=("value", "sum"),
        trade_count=("value", "count"),
    ).reset_index()

    if grouped.empty:
        return

    short_names = grouped["insider_name"].apply(
        lambda n: n[:18] + "…" if len(str(n)) > 18 else str(n)
    )
    labels = [
        f"{name}<br>{typ}: ${val:,.0f}"
        for name, typ, val in zip(short_names, grouped["type"], grouped["total_value"])
    ]
    colors = [COLORS["green"] if t == "Purchase" else COLORS["red"] for t in grouped["type"]]

    fig = go.Figure()
    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=[""] * len(grouped),
            values=grouped["total_value"].tolist(),
            marker=dict(
                colors=colors,
                line=dict(color=COLORS["bg"], width=2),
            ),
            textinfo="label",
            textfont=dict(size=13, color="white", family="Courier New, monospace"),
            hovertemplate="<b>%{label}</b><br>Trades: %{customdata}<extra></extra>",
            customdata=grouped["trade_count"].tolist(),
        )
    )

    fig.update_layout(
        title=dict(text=f"Insider Trade Map: {ticker}", font=dict(color=COLORS["text"])),
        paper_bgcolor=COLORS["bg"],
        font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Block size = total trade value · Green = purchase · Red = sale")


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
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.15),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_insider_timeline(df: pd.DataFrame, ticker: str):
    """Scatter plot of individual insider trades by date and dollar value."""
    chart_df = df[df["type"].isin(["Purchase", "Sale"]) & (df["value"] > 0)].copy()
    if chart_df.empty:
        return

    fig = go.Figure()
    for trade_type, color, symbol in [
        ("Purchase", COLORS["green"], "triangle-up"),
        ("Sale", COLORS["red"], "triangle-down"),
    ]:
        subset = chart_df[chart_df["type"] == trade_type]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["value"],
                mode="markers",
                name=trade_type,
                marker=dict(color=color, size=10, symbol=symbol),
                text=subset["insider_name"],
                hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
            )
        )

    apply_dark_layout(
        fig,
        title=f"Insider Trade Values: {ticker}",
        xaxis_title="Date",
        yaxis_title="Trade Value ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_congress_charts(df: pd.DataFrame, ticker: str):
    """Render bias line chart, monthly bar chart, and trade timeline for congress trades."""
    chart_df = df.copy()
    chart_df["month"] = pd.to_datetime(chart_df["date"]).dt.to_period("M").astype(str)

    # --- Bias line chart: cumulative net buy % over time (dollar-weighted) ---
    chart_df["est_value"] = chart_df["size"].apply(_parse_size_midpoint)
    all_months = sorted(chart_df["month"].unique())
    purchases_by_month = chart_df[chart_df["type"] == "Purchase"].groupby("month")["est_value"].sum()
    sales_by_month = chart_df[chart_df["type"] == "Sale"].groupby("month")["est_value"].sum()

    cumulative_buys = 0.0
    cumulative_sells = 0.0
    bias_data = []
    for m in all_months:
        cumulative_buys += purchases_by_month.get(m, 0)
        cumulative_sells += sales_by_month.get(m, 0)
        total = cumulative_buys + cumulative_sells
        buy_pct = (cumulative_buys / total * 100) if total > 0 else 50
        sell_pct = (cumulative_sells / total * 100) if total > 0 else 50
        net_pct = buy_pct - sell_pct  # positive = buying bias, negative = selling bias
        bias_data.append({"month": m, "buy_pct": buy_pct, "sell_pct": sell_pct, "net_bias": net_pct})

    bias_df = pd.DataFrame(bias_data)

    fig_bias = go.Figure()
    fig_bias.add_trace(
        go.Scatter(
            x=bias_df["month"], y=bias_df["buy_pct"],
            name="Buy %",
            mode="lines+markers",
            line=dict(color=COLORS["green"], width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.08)",
            hovertemplate="%{x}<br>Buy: %{y:.1f}%<extra></extra>",
        )
    )
    fig_bias.add_trace(
        go.Scatter(
            x=bias_df["month"], y=bias_df["sell_pct"],
            name="Sell %",
            mode="lines+markers",
            line=dict(color=COLORS["red"], width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Sell: %{y:.1f}%<extra></extra>",
        )
    )

    fig_bias.add_hline(y=50, line_dash="dash", line_color=COLORS["text_dim"], opacity=0.5)

    # Determine bias — buy_pct is the bias score (50% = neutral)
    latest_buy_pct = bias_df.iloc[-1]["buy_pct"] if not bias_df.empty else 50
    if latest_buy_pct > 55:
        bias_label = "BULLISH"
        bias_color = COLORS["green"]
    elif latest_buy_pct < 45:
        bias_label = "BEARISH"
        bias_color = COLORS["red"]
    else:
        bias_label = "NEUTRAL"
        bias_color = COLORS["yellow"]

    # Persist for downstream use
    st.session_state["_congress_bias"] = {
        "ticker": ticker,
        "bias": bias_label,
        "buy_pct": round(latest_buy_pct, 1),
    }

    st.markdown(
        f"<div style='text-align:center; padding:10px; border:1px solid {bias_color}; border-radius:8px; margin-bottom:10px;'>"
        f"<span style='font-size:2em; color:{bias_color}; font-weight:bold;'>{latest_buy_pct:.1f}%</span>"
        f"<br><span style='color:{bias_color}; font-size:1.2em;'>{bias_label}</span>"
        f"<br><span style='color:{COLORS['text_dim']}; font-size:0.85em;'>50% = Neutral · Above = Bullish · Below = Bearish · Weighted by $ value</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    apply_dark_layout(
        fig_bias,
        title=f"Congress Bias: {ticker} — {bias_label} ({latest_buy_pct:.1f}%)",
        yaxis_title="Cumulative %",
        yaxis=dict(range=[0, 100], gridcolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=80),
    )
    fig_bias.update_layout(title_font_color=bias_color)
    st.plotly_chart(fig_bias, use_container_width=True)

    # --- Monthly activity bar chart ---
    purchases = chart_df[chart_df["type"] == "Purchase"].groupby("month").size()
    sales = chart_df[chart_df["type"] == "Sale"].groupby("month").size()

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
        title=f"Congress Monthly Activity: {ticker}",
        xaxis_title="Month",
        yaxis_title="Number of Trades",
        barmode="relative",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Trade timeline scatter ---
    fig2 = go.Figure()
    for trade_type, color, symbol in [
        ("Purchase", COLORS["green"], "triangle-up"),
        ("Sale", COLORS["red"], "triangle-down"),
    ]:
        subset = chart_df[chart_df["type"] == trade_type]
        if subset.empty:
            continue
        fig2.add_trace(
            go.Scatter(
                x=subset["date"],
                y=[trade_type] * len(subset),
                mode="markers",
                name=trade_type,
                marker=dict(color=color, size=10, symbol=symbol),
                text=subset["politician"],
                customdata=subset["size"],
                hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Size: %{customdata}<extra></extra>",
            )
        )

    apply_dark_layout(
        fig2,
        title=f"Congress Trade Timeline: {ticker}",
        xaxis_title="Date",
        yaxis_title="",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=80),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Congress treemap ---
    _render_congress_treemap(chart_df, ticker)


def _render_congress_treemap(df: pd.DataFrame, ticker: str):
    """Treemap of congress trades: size = estimated value, color = buy/sell."""
    tree_df = df[df["type"].isin(["Purchase", "Sale"])].copy()
    tree_df["est_value"] = tree_df["size"].apply(_parse_size_midpoint)
    tree_df = tree_df[tree_df["est_value"] > 0]

    if tree_df.empty:
        return

    # Aggregate by politician and type
    grouped = tree_df.groupby(["politician", "type"]).agg(
        total_value=("est_value", "sum"),
        trade_count=("est_value", "count"),
    ).reset_index()

    short_names = grouped["politician"].apply(
        lambda n: n[:18] + "…" if len(str(n)) > 18 else str(n)
    )
    labels = [
        f"{name}<br>{typ}: ${val:,.0f}"
        for name, typ, val in zip(short_names, grouped["type"], grouped["total_value"])
    ]
    colors = [COLORS["green"] if t == "Purchase" else COLORS["red"] for t in grouped["type"]]

    fig = go.Figure()
    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=[""] * len(grouped),
            values=grouped["total_value"].tolist(),
            marker=dict(
                colors=colors,
                line=dict(color=COLORS["bg"], width=2),
            ),
            textinfo="label",
            textfont=dict(size=13, color="white", family="Courier New, monospace"),
            hovertemplate="<b>%{label}</b><br>Trades: %{customdata}<extra></extra>",
            customdata=grouped["trade_count"].tolist(),
        )
    )

    fig.update_layout(
        title=dict(text=f"Congress Trade Map: {ticker}", font=dict(color=COLORS["text"])),
        paper_bgcolor=COLORS["bg"],
        font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Block size = estimated trade value · Green = purchase · Red = sale")


def _render_congress(ticker: str):
    st.caption(f"Congress Trading: **{ticker}**")

    timeframe = st.radio(
        "Timeframe", list(_TIMEFRAME_DAYS.keys()), index=2, horizontal=True, key="congress_tf"
    )

    with st.spinner("Fetching congress trades..."):
        try:
            df, error = get_congress_trades(ticker)
        except Exception as e:
            st.error(f"Congress data fetch failed: {e}")
            return

    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 1H")

    if error:
        st.error(error)
        return

    if df.empty:
        st.info("No recent congress trades found for this ticker.")
        return

    # Filter by timeframe — dates are ISO format (YYYY-MM-DD) from Quiver Quant
    days = _TIMEFRAME_DAYS[timeframe]
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    df["date"] = df["date"].astype(str)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    if df.empty:
        st.info(f"No congress trades in the last {timeframe}.")
        return

    st.success(f"Found {len(df)} congress trades for {ticker} ({timeframe})")

    buys = df[df["type"] == "Purchase"]
    sells = df[df["type"] == "Sale"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades", len(df))
    c2.metric("Purchases", len(buys))
    c3.metric("Sales", len(sells))

    # Charts
    _render_congress_charts(df, ticker)

    def _highlight_type(val):
        if val == "Purchase":
            return f"color: {COLORS['green']}"
        elif val == "Sale":
            return f"color: {COLORS['red']}"
        return ""

    st.dataframe(
        df.style.map(_highlight_type, subset=["type"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "politician": "Politician",
            "date": "Date",
            "type": "Buy/Sell",
            "size": "Size",
            "price": "Price",
        },
    )

    csv = df.to_csv(index=False)
    st.download_button("Export CSV", csv, f"{ticker}_congress_trades.csv", "text/csv", key="dl_congress")
