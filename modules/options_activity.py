import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("OPTIONS ACTIVITY")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view options activity.")
        return

    with st.spinner("Fetching options data..."):
        df, expirations = _get_options_data(ticker)

    if df is None or df.empty:
        st.warning("No options data available for this ticker.")
        return

    # Expiration filter
    selected_exps = st.multiselect(
        "Expirations",
        options=expirations,
        default=expirations[:3],
        key="options_exp_filter",
    )
    if selected_exps:
        df = df[df["expiration"].isin(selected_exps)]

    if df.empty:
        st.info("No data for selected expirations.")
        return

    # --- Metrics ---
    calls = df[df["right"] == "Call"]
    puts = df[df["right"] == "Put"]

    call_vol = calls["volume"].sum()
    put_vol = puts["volume"].sum()
    call_oi = calls["openInterest"].sum()
    put_oi = puts["openInterest"].sum()
    pc_ratio = put_vol / call_vol if call_vol > 0 else 0

    if pc_ratio < 0.7:
        sentiment, sent_color = "BULLISH", COLORS["green"]
    elif pc_ratio > 1.0:
        sentiment, sent_color = "BEARISH", COLORS["red"]
    else:
        sentiment, sent_color = "NEUTRAL", COLORS["yellow"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Call Volume", f"{call_vol:,.0f}")
    c2.metric("Put Volume", f"{put_vol:,.0f}")
    c3.metric("P/C Ratio", f"{pc_ratio:.2f}")
    c4.metric("Call OI", f"{call_oi:,.0f}")
    c5.metric("Put OI", f"{put_oi:,.0f}")

    st.markdown(
        f"<div style='text-align:center; padding:8px; border:1px solid {sent_color}; "
        f"border-radius:8px; margin:10px 0;'>"
        f"<span style='color:{sent_color}; font-size:1.1em; font-weight:bold;'>"
        f"Options Sentiment: {sentiment} (P/C {pc_ratio:.2f})</span></div>",
        unsafe_allow_html=True,
    )

    # --- Volume by expiration ---
    _render_volume_by_expiration(df)

    # --- OI by strike ---
    _render_oi_by_strike(df, ticker)

    # --- IV smile ---
    _render_iv_smile(df, ticker)

    # --- Treemap ---
    _render_options_treemap(df, ticker)

    # --- Unusual activity ---
    _render_unusual_activity(df)

    # --- Full chain ---
    with st.expander("Full Options Chain"):
        display = df.copy()
        display["volume"] = display["volume"].apply(lambda v: f"{v:,.0f}")
        display["openInterest"] = display["openInterest"].apply(lambda v: f"{v:,.0f}")
        display["impliedVolatility"] = display["impliedVolatility"].apply(lambda v: f"{v:.1%}")
        display["strike"] = display["strike"].apply(lambda v: f"${v:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_volume_by_expiration(df: pd.DataFrame):
    """Grouped bar chart of call vs put volume by expiration."""
    vol_by_exp = df.groupby(["expiration", "right"])["volume"].sum().reset_index()

    fig = go.Figure()
    for right, color, name in [("Call", COLORS["green"], "Calls"), ("Put", COLORS["red"], "Puts")]:
        subset = vol_by_exp[vol_by_exp["right"] == right]
        fig.add_trace(
            go.Bar(x=subset["expiration"], y=subset["volume"], name=name, marker_color=color)
        )

    apply_dark_layout(
        fig, title="Volume by Expiration", barmode="group",
        xaxis_title="Expiration", yaxis_title="Volume",
    )
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


def _render_oi_by_strike(df: pd.DataFrame, ticker: str):
    """Mirrored bar chart showing call OI vs put OI by strike price."""
    oi = df.groupby(["strike", "right"])["openInterest"].sum().reset_index()
    call_oi = oi[oi["right"] == "Call"].set_index("strike")["openInterest"]
    put_oi = oi[oi["right"] == "Put"].set_index("strike")["openInterest"]

    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_strikes,
            y=[call_oi.get(s, 0) for s in all_strikes],
            name="Call OI",
            marker_color=COLORS["green"],
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_strikes,
            y=[-put_oi.get(s, 0) for s in all_strikes],
            name="Put OI",
            marker_color=COLORS["red"],
            opacity=0.8,
        )
    )

    apply_dark_layout(
        fig, title=f"Open Interest by Strike: {ticker}",
        xaxis_title="Strike ($)", yaxis_title="Open Interest",
        barmode="relative",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Above = Call OI · Below = Put OI · Peaks show key support/resistance levels")


def _render_iv_smile(df: pd.DataFrame, ticker: str):
    """IV smile/skew chart for the nearest expiration."""
    nearest_exp = df["expiration"].min()
    exp_df = df[df["expiration"] == nearest_exp]

    fig = go.Figure()
    for right, color, name in [("Call", COLORS["green"], "Calls"), ("Put", COLORS["red"], "Puts")]:
        subset = exp_df[exp_df["right"] == right].sort_values("strike")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["strike"], y=subset["impliedVolatility"] * 100,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate="Strike: $%{x:.0f}<br>IV: %{y:.1f}%<extra></extra>",
            )
        )

    apply_dark_layout(
        fig, title=f"IV Smile: {ticker} ({nearest_exp})",
        xaxis_title="Strike ($)", yaxis_title="Implied Volatility (%)",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def _render_options_treemap(df: pd.DataFrame, ticker: str):
    """Treemap of options volume: size = volume, color = calls (green) / puts (red)."""
    chart_df = df[df["volume"] > 0].copy()
    if chart_df.empty:
        return

    grouped = chart_df.groupby(["strike", "expiration", "right"]).agg(
        total_vol=("volume", "sum"),
        total_oi=("openInterest", "sum"),
    ).reset_index()

    if grouped.empty:
        return

    labels = [
        f"${strike:.0f} {right}<br>{exp}<br>Vol: {vol:,.0f}"
        for strike, exp, right, vol in zip(
            grouped["strike"], grouped["expiration"], grouped["right"], grouped["total_vol"]
        )
    ]
    colors = [COLORS["green"] if r == "Call" else COLORS["red"] for r in grouped["right"]]

    fig = go.Figure()
    fig.add_trace(
        go.Treemap(
            labels=labels,
            parents=[""] * len(grouped),
            values=grouped["total_vol"].tolist(),
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="label",
            textfont=dict(size=12, color="white", family="Courier New, monospace"),
            hovertemplate="<b>%{label}</b><br>OI: %{customdata:,.0f}<extra></extra>",
            customdata=grouped["total_oi"].tolist(),
        )
    )

    fig.update_layout(
        title=dict(text=f"Options Volume Map: {ticker}", font=dict(color=COLORS["text"])),
        paper_bgcolor=COLORS["bg"],
        font=dict(family="Courier New, monospace", color=COLORS["text"], size=12),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Block size = volume · Green = calls · Red = puts")


def _render_unusual_activity(df: pd.DataFrame):
    """Table of contracts with unusually high volume relative to open interest."""
    active = df[(df["volume"] > 0) & (df["openInterest"] > 0)].copy()
    if active.empty:
        st.info("No options with both volume and OI to detect unusual activity.")
        return

    active["vol_oi"] = active["volume"] / active["openInterest"]
    unusual = active[active["vol_oi"] > 2.0].sort_values("vol_oi", ascending=False).head(10)

    if unusual.empty:
        st.info("No unusual activity detected (Vol/OI > 2.0)")
        return

    st.subheader("Unusual Activity (Vol/OI > 2.0)")

    def _highlight_type(val):
        if val == "Call":
            return f"color: {COLORS['green']}"
        elif val == "Put":
            return f"color: {COLORS['red']}"
        return ""

    show = unusual[["expiration", "strike", "right", "volume", "openInterest", "impliedVolatility", "vol_oi"]].copy()
    show.columns = ["Expiry", "Strike", "Type", "Volume", "OI", "IV", "Vol/OI"]
    show["Strike"] = show["Strike"].apply(lambda v: f"${v:.2f}")
    show["IV"] = show["IV"].apply(lambda v: f"{v:.1%}")
    show["Vol/OI"] = show["Vol/OI"].apply(lambda v: f"{v:.2f}")

    st.dataframe(
        show.style.map(_highlight_type, subset=["Type"]),
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data(ttl=300)
def _get_options_data(ticker: str) -> tuple:
    """Fetch options chain from yfinance. Returns (DataFrame, list of expirations)."""
    try:
        stock = yf.Ticker(ticker)
        expirations = list(stock.options)

        if not expirations:
            return None, []

        all_rows = []
        for exp in expirations:
            chain = stock.option_chain(exp)

            for side, label in [(chain.calls, "Call"), (chain.puts, "Put")]:
                side_df = side.copy()
                side_df["expiration"] = exp
                side_df["right"] = label
                all_rows.append(side_df)

        df = pd.concat(all_rows, ignore_index=True)

        # Keep relevant columns, fill NaN
        keep = ["expiration", "strike", "right", "lastPrice", "bid", "ask",
                "volume", "openInterest", "impliedVolatility"]
        for col in keep:
            if col not in df.columns:
                df[col] = 0
        df = df[keep]
        df["volume"] = df["volume"].fillna(0).astype(int)
        df["openInterest"] = df["openInterest"].fillna(0).astype(int)
        df["impliedVolatility"] = df["impliedVolatility"].fillna(0)

        return df, expirations
    except Exception:
        return None, []
