import numpy as np
import plotly.graph_objects as go
import streamlit as st
from services.trends_client import get_interest_over_time
from utils.session import get_narrative
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("NARRATIVE PULSE")

    narrative = get_narrative()
    if not narrative:
        st.info("Set an active narrative in Discovery to view pulse data.")
        return

    st.caption(f'Tracking: **"{narrative}"** · 90-day Google Trends')

    with st.spinner("Fetching trend data..."):
        df = get_interest_over_time(narrative)

    if df.empty:
        st.warning("No trend data available for this keyword.")
        return

    # Calculate metrics
    interest = df["interest"].values
    mean = np.mean(interest)
    std = np.std(interest)
    spike_threshold = mean + 2 * std

    current = interest[-1] if len(interest) > 0 else 0
    recent_7d = np.mean(interest[-7:]) if len(interest) >= 7 else current
    prior_7d = (
        np.mean(interest[-14:-7]) if len(interest) >= 14 else np.mean(interest)
    )
    momentum = ((recent_7d - prior_7d) / prior_7d * 100) if prior_7d > 0 else 0
    spike_count = int(np.sum(interest > spike_threshold))

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Interest", f"{current}")
    c2.metric("7-Day Momentum", f"{momentum:+.1f}%")
    c3.metric("Spike Count", f"{spike_count}")

    # Line chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["interest"],
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            name="Interest",
        )
    )

    # Spike markers
    spike_mask = df["interest"] > spike_threshold
    if spike_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[spike_mask, "date"],
                y=df.loc[spike_mask, "interest"],
                mode="markers",
                marker=dict(color=COLORS["yellow"], size=10, symbol="triangle-up"),
                name="Spike (>2σ)",
            )
        )

    # Threshold line
    fig.add_hline(
        y=spike_threshold,
        line_dash="dot",
        line_color=COLORS["text_dim"],
        annotation_text="2σ threshold",
    )

    apply_dark_layout(fig, title=f'Google Trends: "{narrative}"', yaxis_title="Interest")
    st.plotly_chart(fig, use_container_width=True)
