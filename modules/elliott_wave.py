"""
Module: Elliott Wave Analysis (SPY)

Counts primary-degree Elliott Waves on SPY using the elliott_wave_engine,
then generates a Groq LLaMA narrative interpretation.

Layout:
  - SPY price chart with wave overlay
  - Warning banner (if no count or low confidence)
  - Metrics row: Wave Position | Confidence | Invalidation Level
  - Fibonacci Hits list
  - AI Narrative expander
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from services.market_data import fetch_batch_safe
from services.elliott_wave_engine import get_best_wave_count, BestCount
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@st.cache_data(ttl=3600)
def _fetch_spy_data() -> pd.Series | None:
    """Fetch 1 year of SPY daily closes. Returns close price Series or None."""
    snaps = fetch_batch_safe({"SPY": "S&P 500"}, period="1y", interval="1d")
    snap = snaps.get("SPY")
    if snap is None or snap.series is None or snap.series.empty:
        return None
    return snap.series.dropna()


@st.cache_data(ttl=3600)
def _build_groq_narrative(
    wave_position: str,
    current_wave_label: str,
    confidence: int,
    invalidation_level: float,
    fibonacci_hits: tuple,
    current_price: float = 0.0,
) -> str:
    """
    Call Groq LLaMA to generate a 3-5 sentence Elliott Wave narrative.
    fibonacci_hits is a tuple (not list) so it is hashable for @st.cache_data.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "_GROQ_API_KEY not set — narrative unavailable._"

    fib_text = "\n".join(f"  - {h}" for h in fibonacci_hits) if fibonacci_hits else "  None detected"

    # Price context
    price_ctx = ""
    if current_price > 0:
        dist = current_price - invalidation_level
        dist_pct = (dist / invalidation_level) * 100 if invalidation_level else 0
        price_ctx = f"""- Current SPY Price: ${current_price:.2f}
- Distance to Invalidation: ${dist:+.2f} ({dist_pct:+.1f}%)
"""

    prompt = f"""You are an expert Elliott Wave analyst. Interpret the following automated wave count for SPY (S&P 500 ETF) and write a concise 3-5 sentence market commentary.

Current Wave Count:
- Position: {wave_position}
- Active Wave: {current_wave_label}
- Count Confidence: {confidence}/100
- Invalidation Level: ${invalidation_level:.2f}
{price_ctx}- Fibonacci Confirmations:
{fib_text}

Write your commentary covering:
1. What the current wave position means for near-term price action
2. What Elliott Wave theory predicts should happen next
3. The key invalidation level, how far price is from it, and what a breach would signal

Be direct and specific. Do not hedge excessively. Do not repeat the input data verbatim."""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"_Narrative generation failed: {e}_"


def _make_wave_chart(series: pd.Series, best_count: BestCount | None) -> go.Figure:
    """SPY price line with optional wave overlay."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.5),
        name="SPY",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    if best_count is not None:
        pivots = best_count.sequence.pivots
        labels = best_count.sequence.labels
        wave_color = COLORS["green"] if best_count.sequence.wave_type == "impulse" else COLORS["bloomberg_orange"]

        fig.add_trace(go.Scatter(
            x=[p.date for p in pivots],
            y=[p.price for p in pivots],
            mode="lines+markers+text",
            line=dict(color=wave_color, width=2, dash="dot"),
            marker=dict(size=8, color=wave_color),
            text=labels,
            textposition="top center",
            textfont=dict(color=wave_color, size=13, family="JetBrains Mono"),
            name="Wave Count",
            hovertemplate="Wave %{text}<br>$%{y:.2f}<extra></extra>",
        ))

        fig.add_hline(
            y=best_count.invalidation_level,
            line_dash="dash",
            line_color=COLORS["red"],
            line_width=1.5,
            annotation_text=f"Invalidation ${best_count.invalidation_level:.2f}",
            annotation_font_color=COLORS["red"],
            annotation_font_size=11,
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(title="Price ($)"),
    )
    apply_dark_layout(fig, title="SPY — Elliott Wave Primary Count")
    return fig


def render():
    st.title("Elliott Wave Analysis")
    st.caption("SPY primary-degree wave count · ATR pivot detection · Rule-engine validation · Groq AI narrative")

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching SPY data and computing wave count..."):
        series = _fetch_spy_data()

    if series is None:
        st.error("SPY price data unavailable. Please try again later.")
        return

    if len(series) < 60:
        st.warning("Insufficient price history for Elliott Wave analysis (need >= 60 bars).")
        return

    best_count = get_best_wave_count(series)

    # Chart
    fig = _make_wave_chart(series, best_count)
    st.plotly_chart(fig, use_container_width=True)

    # Warning Banner
    if best_count is None:
        st.info(
            "No clean Elliott Wave count detected in the current data window. "
            "Market may be in a complex correction or transitional phase."
        )
        return

    if best_count.confidence < 40:
        st.warning(
            f"Low-confidence wave count ({best_count.confidence}/100) — "
            "structural rules pass but Fibonacci confirmations are weak. Treat as speculative."
        )

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.markdown(bloomberg_metric("Wave Position", best_count.wave_position), unsafe_allow_html=True)
    m2.markdown(bloomberg_metric("Confidence", f"{best_count.confidence}/100"), unsafe_allow_html=True)
    m3.markdown(bloomberg_metric("Invalidation", f"${best_count.invalidation_level:.2f}", COLORS["red"]), unsafe_allow_html=True)

    # Fibonacci Hits
    if best_count.fibonacci_hits:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};margin:8px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em;">'
            f'Fibonacci Confirmations</div>',
            unsafe_allow_html=True,
        )
        for hit in best_count.fibonacci_hits:
            st.markdown(f"- {hit}")
    else:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["text_dim"]};">No Fibonacci ratios confirmed</div>',
            unsafe_allow_html=True,
        )

    # AI Narrative
    with st.expander("Elliott Wave AI Narrative", expanded=False):
        current_price = float(series.iloc[-1])
        narrative_args = (
            best_count.wave_position,
            best_count.current_wave_label,
            best_count.confidence,
            best_count.invalidation_level,
            tuple(best_count.fibonacci_hits),
            current_price,
        )
        try:
            narrative = _build_groq_narrative(*narrative_args)
            if narrative.startswith("_Narrative generation failed") or narrative.startswith("_GROQ"):
                st.warning("AI narrative unavailable.")
                if st.button("Retry Narrative", key="retry_narrative"):
                    _build_groq_narrative.clear()
                    st.rerun()
            else:
                st.markdown(narrative)
                st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Model: {GROQ_MODEL}")
        except Exception:
            st.warning("AI narrative unavailable.")
            if st.button("Retry Narrative", key="retry_narrative_err"):
                st.cache_data.clear()
                st.rerun()
