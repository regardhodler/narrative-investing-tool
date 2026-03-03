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

    # --- Major holders summary + ownership donut ---
    if major is not None and not major.empty:
        _render_major_holders(major, ticker)
        _render_ownership_donut(major, ticker)

    # --- Net institutional flow bias chart ---
    if holders is not None and not holders.empty:
        _render_bias_chart(holders, ticker)

    # --- Holdings treemap ---
    if holders is not None and not holders.empty:
        _render_holdings_treemap(holders, ticker)

    # --- Top institutional holders ---
    if holders is not None and not holders.empty:
        _render_holders_chart(holders, ticker)
        _render_holders_table(holders)
    else:
        st.info("No institutional holder details available for this ticker.")


def _render_major_holders(major: pd.DataFrame, ticker: str):
    """Render summary metrics from major_holders breakdown."""
    # major_holders: index = breakdown names, single 'Value' column
    data = {}
    for idx, row in major.iterrows():
        data[str(idx)] = row.iloc[0]

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


def _render_ownership_donut(major: pd.DataFrame, ticker: str):
    """Donut chart showing institutional vs insider vs retail ownership."""
    data = {}
    for idx, row in major.iterrows():
        data[str(idx)] = row.iloc[0]

    inst = data.get("institutionsPercentHeld", data.get("% Held by Institutions", 0))
    insider = data.get("insidersPercentHeld", data.get("% Held by Insiders", 0))

    if isinstance(inst, (int, float)) and inst <= 1:
        inst *= 100
    if isinstance(insider, (int, float)) and insider <= 1:
        insider *= 100

    retail = max(0, 100 - inst - insider)

    labels = ["Institutional", "Insider", "Retail"]
    values = [inst, insider, retail]
    colors = [COLORS["accent"], COLORS["yellow"], COLORS["blue"]]

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="label+percent",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Ownership Breakdown: {ticker}", font=dict(color=COLORS["text"])),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
        height=350,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=-0.05,
            x=0.5,
            xanchor="center",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_bias_chart(holders: pd.DataFrame, ticker: str):
    """Vertical bar chart showing net institutional flow bias (pctChange per holder)."""
    df = holders.head(10).copy()
    df = df.dropna(subset=["pctChange", "Value"])

    if df.empty:
        return

    # Value-weighted average pctChange
    total_value = df["Value"].sum()
    if total_value == 0:
        return
    weighted_avg = (df["pctChange"] * df["Value"]).sum() / total_value
    weighted_avg_pct = weighted_avg * 100

    # Bias label
    if weighted_avg_pct > 0.5:
        bias_label, bias_color = "BULLISH", COLORS["green"]
    elif weighted_avg_pct < -0.5:
        bias_label, bias_color = "BEARISH", COLORS["red"]
    else:
        bias_label, bias_color = "NEUTRAL", COLORS["yellow"]

    # Abbreviate holder names
    short_names = df["Holder"].apply(lambda n: n[:15] + "…" if len(str(n)) > 15 else str(n))
    pct_values = df["pctChange"] * 100

    bar_colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in pct_values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=short_names,
            y=pct_values,
            marker_color=bar_colors,
            text=pct_values.apply(lambda v: f"{v:+.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>",
        )
    )

    # Weighted average dashed line
    fig.add_hline(
        y=weighted_avg_pct,
        line_dash="dash",
        line_color=bias_color,
        annotation_text=f"Wt. Avg: {weighted_avg_pct:+.1f}%",
        annotation_position="top right",
        annotation_font_color=bias_color,
    )

    title = f"Institutional Bias: {ticker} — {bias_label} ({weighted_avg_pct:+.1f}%)"
    apply_dark_layout(
        fig,
        title=dict(text=title, font=dict(color=bias_color)),
        yaxis_title="Position Change (%)",
        margin=dict(l=60, r=40, t=60, b=100),
    )
    fig.update_layout(height=420, xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)


def _render_holdings_treemap(holders: pd.DataFrame, ticker: str):
    """Treemap: block size = position value, color = pctChange direction."""
    df = holders.head(10).copy()
    df = df.dropna(subset=["Value"])
    df = df[df["Value"] > 0]

    if df.empty:
        return

    # Fill missing pctChange with 0
    df["pctChange"] = df["pctChange"].fillna(0)
    pct_vals = df["pctChange"] * 100

    # Map pctChange to a continuous color scale
    # Clamp to [-5, 5] for color range
    color_vals = pct_vals.clip(-5, 5)

    short_names = df["Holder"].apply(lambda n: n[:20] + "…" if len(str(n)) > 20 else str(n))
    labels = [
        f"{name}<br>{pct:+.1f}%"
        for name, pct in zip(short_names, pct_vals)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=[""] * len(df),
            values=df["Value"].tolist(),
            marker=dict(
                colors=color_vals.tolist(),
                colorscale=[[0, COLORS["red"]], [0.5, COLORS["yellow"]], [1, COLORS["green"]]],
                cmid=0,
                line=dict(color=COLORS["bg"], width=2),
            ),
            textinfo="label",
            textfont=dict(size=13, color="white"),
            hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Holdings Concentration: {ticker}",
            font=dict(color=COLORS["text"]),
        ),
        paper_bgcolor=COLORS["bg"],
        font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Block size = position value · Color: green = buying, red = selling, yellow = flat")


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
