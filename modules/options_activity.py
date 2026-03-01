import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from services.ibkr_client import connect_ibkr, disconnect_ibkr, get_options_chain
from utils.session import get_ticker, is_ibkr_connected
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("OPTIONS ACTIVITY")

    # Connection controls
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        port = st.selectbox("Port", [7497, 7496], format_func=lambda p: f"{p} ({'Paper' if p == 7497 else 'Live'})")
    with col2:
        if st.button("Connect IBKR"):
            connect_ibkr(port)
            st.rerun()
    with col3:
        if st.button("Disconnect"):
            disconnect_ibkr()
            st.rerun()

    if not is_ibkr_connected():
        st.warning("Not connected to IBKR. Start TWS/Gateway and click Connect.")
        st.markdown(
            """
            **Setup checklist:**
            - TWS or IB Gateway running
            - API enabled: File → Global Config → API → Settings
            - Socket port matches selection above
            - "Read-Only API" unchecked
            """
        )
        return

    st.success("Connected to IBKR")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in EDGAR Scanner to view options activity.")
        return

    st.caption(f"Options chain: **{ticker}** · ±10% ATM · 3 nearest expirations")

    with st.spinner("Fetching options chain..."):
        chain = get_options_chain(ticker)

    if not chain:
        st.warning("No options data returned. Check ticker or market hours.")
        return

    df = pd.DataFrame(chain)

    # Flag unusual activity
    df["unusual"] = df["vol_oi_ratio"].apply(
        lambda x: x is not None and x > 2.0
    )

    # Calls vs Puts volume bar chart
    calls = df[df["right"] == "C"]
    puts = df[df["right"] == "P"]

    call_vol = calls["volume"].sum()
    put_vol = puts["volume"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Call Volume", f"{call_vol:,.0f}")
    c2.metric("Put Volume", f"{put_vol:,.0f}")
    c3.metric("Put/Call Ratio", f"{put_vol / call_vol:.2f}" if call_vol > 0 else "N/A")

    # Volume by expiration chart
    vol_by_exp = df.groupby(["expiration", "right"])["volume"].sum().reset_index()
    fig = go.Figure()
    for right, color, name in [("C", COLORS["green"], "Calls"), ("P", COLORS["red"], "Puts")]:
        subset = vol_by_exp[vol_by_exp["right"] == right]
        fig.add_trace(
            go.Bar(
                x=subset["expiration"],
                y=subset["volume"],
                name=name,
                marker_color=color,
            )
        )
    apply_dark_layout(fig, title="Volume by Expiration", barmode="group", xaxis_title="Expiration", yaxis_title="Volume")
    st.plotly_chart(fig, use_container_width=True)

    # Unusual activity table
    unusual = df[df["unusual"]].sort_values("vol_oi_ratio", ascending=False).head(10)
    if not unusual.empty:
        st.subheader("Unusual Activity (Vol/OI > 2.0)")
        st.dataframe(
            unusual[
                ["expiration", "strike", "right", "volume", "open_interest", "iv", "delta", "vol_oi_ratio"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "expiration": "Expiry",
                "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                "right": "Type",
                "volume": st.column_config.NumberColumn("Volume", format="%d"),
                "open_interest": st.column_config.NumberColumn("OI", format="%d"),
                "iv": st.column_config.NumberColumn("IV", format="%.2f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                "vol_oi_ratio": st.column_config.NumberColumn("Vol/OI", format="%.2f"),
            },
        )
    else:
        st.info("No unusual activity detected (Vol/OI > 2.0)")

    # Full chain (expandable)
    with st.expander("Full Options Chain"):
        st.dataframe(df, use_container_width=True, hide_index=True)
