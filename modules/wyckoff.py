"""
Module: Wyckoff Method Analysis (SPY)

Detects Accumulation, Distribution, Markup, and Markdown phases on SPY
using volume-confirmed price structure, then generates a Groq LLaMA narrative.

Layout:
  - SPY candlestick chart with phase shading and event annotations
  - Volume, RSI, and OBV sub-panels
  - Metrics row: Current Phase | Confidence | Support/Resistance
  - Events list
  - AI Narrative expander
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from services.market_data import fetch_ohlcv_single
from services.wyckoff_engine import analyze_wyckoff, WyckoffAnalysis
from services.indicators import rsi, obv
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PHASE_COLORS = {
    "Accumulation": "rgba(0,212,170,0.08)",
    "Distribution": "rgba(255,75,75,0.08)",
    "Markup": "rgba(75,159,255,0.05)",
    "Markdown": "rgba(255,140,0,0.05)",
}

PHASE_BORDER_COLORS = {
    "Accumulation": "rgba(0,212,170,0.3)",
    "Distribution": "rgba(255,75,75,0.3)",
    "Markup": "rgba(75,159,255,0.2)",
    "Markdown": "rgba(255,140,0,0.2)",
}


@st.cache_data(ttl=3600)
def _fetch_spy_data() -> pd.DataFrame | None:
    """Fetch 1 year of SPY daily OHLCV."""
    return fetch_ohlcv_single("SPY", period="1y", interval="1d")


@st.cache_data(ttl=3600)
def _build_groq_narrative(
    phase: str,
    confidence: int,
    support: float,
    resistance: float,
    events: tuple,
    current_price: float = 0.0,
) -> str:
    """Call Groq LLaMA to generate a Wyckoff narrative."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "_GROQ_API_KEY not set — narrative unavailable._"

    events_text = "\n".join(f"  - {e}" for e in events) if events else "  None detected"

    price_ctx = ""
    if current_price > 0:
        dist_s = current_price - support
        dist_r = resistance - current_price
        price_ctx = f"""- Current SPY Price: ${current_price:.2f}
- Distance to Support: ${dist_s:+.2f}
- Distance to Resistance: ${dist_r:+.2f}
"""

    prompt = f"""You are an expert Wyckoff Method analyst. Interpret the following automated phase detection for SPY (S&P 500 ETF) and write a concise 3-5 sentence market commentary.

Current Wyckoff Phase:
- Phase: {phase}
- Confidence: {confidence}/100
- Support Level: ${support:.2f}
- Resistance Level: ${resistance:.2f}
{price_ctx}- Detected Events:
{events_text}

Write your commentary covering:
1. What the current Wyckoff phase means for market structure
2. Key events detected and their significance
3. What the Wyckoff Method predicts should happen next
4. Critical support/resistance levels to watch

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


def _make_wyckoff_chart(
    ohlcv: pd.DataFrame,
    analysis: WyckoffAnalysis,
    rsi_series: pd.Series,
    obv_series: pd.Series,
    ticker: str = "SPY",
) -> go.Figure:
    """Candlestick chart with Wyckoff phase shading, volume, RSI, and OBV."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.15, 0.20],
        vertical_spacing=0.03,
    )

    close = ohlcv["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"],
        high=ohlcv["High"],
        low=ohlcv["Low"],
        close=ohlcv["Close"],
        name=ticker,
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
    ), row=1, col=1)

    # Phase shading
    for phase in analysis.all_phases:
        color = PHASE_COLORS.get(phase.phase, "rgba(128,128,128,0.05)")
        border = PHASE_BORDER_COLORS.get(phase.phase, "rgba(128,128,128,0.2)")
        fig.add_vrect(
            x0=phase.start_date,
            x1=phase.end_date,
            fillcolor=color,
            line=dict(color=border, width=0.5),
            annotation_text=phase.phase,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=COLORS["text_dim"],
            row=1, col=1,
        )

    # Event annotations
    for phase in analysis.all_phases:
        for evt in phase.events:
            fig.add_annotation(
                x=evt.date,
                y=evt.price,
                text=evt.event_type,
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowcolor=COLORS["bloomberg_orange"],
                font=dict(size=10, color=COLORS["bloomberg_orange"],
                          family="JetBrains Mono"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=COLORS["bloomberg_orange"],
                borderwidth=1,
                row=1, col=1,
            )

    # Sub-phase labels on each phase
    for phase in analysis.all_phases:
        if phase.sub_phase:
            mid_date = phase.start_date + (phase.end_date - phase.start_date) / 2
            phase_color = PHASE_BORDER_COLORS.get(phase.phase, "rgba(128,128,128,0.5)")
            fig.add_annotation(
                x=mid_date, y=phase.key_levels["resistance"],
                text=f"Ph.{phase.sub_phase}",
                showarrow=False,
                font=dict(size=9, color=phase_color.replace("0.3", "0.9"), family="JetBrains Mono"),
                bgcolor="rgba(14,17,23,0.7)",
                row=1, col=1,
            )

    # Demand lines (green dashed) and Supply lines (red dashed)
    for phase in analysis.all_phases:
        if phase.demand_line:
            d1, p1, d2, p2 = phase.demand_line
            fig.add_trace(go.Scatter(
                x=[d1, d2], y=[p1, p2],
                mode="lines",
                line=dict(color=COLORS["green"], width=1.5, dash="dash"),
                name="Demand Line",
                showlegend=False,
                hovertemplate="Demand Line $%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        if phase.supply_line:
            d1, p1, d2, p2 = phase.supply_line
            fig.add_trace(go.Scatter(
                x=[d1, d2], y=[p1, p2],
                mode="lines",
                line=dict(color=COLORS["red"], width=1.5, dash="dash"),
                name="Supply Line",
                showlegend=False,
                hovertemplate="Supply Line $%{y:.2f}<extra></extra>",
            ), row=1, col=1)

    # Cause & Effect target lines
    for phase in analysis.all_phases:
        if phase.cause_target and phase.cause_target > 0:
            target_color = COLORS["green"] if phase.phase == "Accumulation" else COLORS["red"]
            fig.add_hline(
                y=phase.cause_target,
                line_dash="dot",
                line_color=target_color,
                line_width=1,
                annotation_text=f"C&E Target ${phase.cause_target:.0f}",
                annotation_font_color=target_color,
                annotation_font_size=9,
                row=1, col=1,
            )

    # VSA markers on price chart
    vsa_colors = {
        "Strength": COLORS["green"],
        "Weakness": COLORS["red"],
        "No Supply": "#88DDCC",
        "No Demand": "#FF8C00",
        "Effort No Result": "#FFD700",
    }
    for vsa in analysis.vsa_bars:
        vc = vsa_colors.get(vsa.signal, COLORS["text_dim"])
        fig.add_annotation(
            x=vsa.date,
            y=ohlcv["Low"].get(vsa.date, ohlcv["Low"].iloc[-1]) if vsa.signal in ("Weakness", "No Demand", "Effort No Result") else ohlcv["High"].get(vsa.date, ohlcv["High"].iloc[-1]),
            text=vsa.signal[:3],
            showarrow=True,
            arrowhead=1,
            arrowsize=0.6,
            arrowcolor=vc,
            font=dict(size=8, color=vc, family="JetBrains Mono"),
            bgcolor="rgba(0,0,0,0.5)",
            row=1, col=1,
        )

    # Row 2: Volume bars
    vol_colors = [
        COLORS["green"] if close.iloc[i] >= close.iloc[max(0, i - 1)] else COLORS["red"]
        for i in range(len(close))
    ]
    fig.add_trace(go.Bar(
        x=ohlcv.index,
        y=ohlcv["Volume"],
        marker_color=vol_colors,
        name="Volume",
        showlegend=False,
    ), row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(
        x=rsi_series.index,
        y=rsi_series.values,
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.2),
        name="RSI(14)",
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], line_width=0.8, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], line_width=0.8, row=3, col=1)

    # Row 4: OBV
    fig.add_trace(go.Scatter(
        x=obv_series.index,
        y=obv_series.values,
        mode="lines",
        line=dict(color=COLORS["bloomberg_orange"], width=1.2),
        name="OBV",
    ), row=4, col=1)

    # Layout
    fig.update_layout(
        height=800,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI(14)", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="OBV", row=4, col=1)

    apply_dark_layout(fig, title=f"{ticker} — Wyckoff Phase Detection")
    return fig


def render():
    st.title("Wyckoff Method Analysis")

    # Controls row
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption("Phase detection · VSA · Cause & Effect targets · Demand/Supply lines · Groq AI narrative")
    with c2:
        ticker = st.text_input("Ticker", value="SPY", max_chars=10).upper().strip()
    with c3:
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0, label_visibility="collapsed")

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    period_map = {"1d": "2y", "1wk": "10y", "1mo": "max"}
    period = period_map.get(interval, "2y")

    with st.spinner(f"Fetching {ticker} data and analyzing Wyckoff phases..."):
        ohlcv = fetch_ohlcv_single(ticker, period=period, interval=interval)

    if ohlcv is None or ohlcv.empty:
        st.error(f"{ticker} price data unavailable. Check the ticker and try again.")
        return

    # Extract series with DataFrame→Series guards
    close = ohlcv["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    high = ohlcv["High"]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    high = high.dropna()

    low = ohlcv["Low"]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    low = low.dropna()

    volume = ohlcv["Volume"]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    volume = volume.dropna()

    if len(close) < 60:
        st.warning(f"Insufficient price history for {ticker} Wyckoff analysis (need >= 60 bars).")
        return

    analysis = analyze_wyckoff(close, high, low, volume)

    # Compute indicators
    rsi_series = rsi(close)
    obv_series = obv(close, volume)

    if analysis is None:
        st.info(
            "No Wyckoff phases detected in the current data window. "
            "Market may lack sufficient consolidation structure for phase identification."
        )
        # Still show chart without phase overlays
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           row_heights=[0.50, 0.15, 0.15, 0.20], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(
            x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"], close=ohlcv["Close"], name=ticker,
            increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"],
        ), row=1, col=1)
        vol_colors = [
            COLORS["green"] if close.iloc[i] >= close.iloc[max(0, i - 1)] else COLORS["red"]
            for i in range(len(close))
        ]
        fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"], marker_color=vol_colors,
                             name="Volume", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, mode="lines",
                                 line=dict(color=COLORS["blue"], width=1.2), name="RSI(14)"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], line_width=0.8, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], line_width=0.8, row=3, col=1)
        fig.add_trace(go.Scatter(x=obv_series.index, y=obv_series.values, mode="lines",
                                 line=dict(color=COLORS["bloomberg_orange"], width=1.2), name="OBV"), row=4, col=1)
        fig.update_layout(height=800, margin=dict(l=20, r=20, t=40, b=20), showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          xaxis_rangeslider_visible=False)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI(14)", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="OBV", row=4, col=1)
        apply_dark_layout(fig, title=f"{ticker} — Wyckoff Phase Detection")
        st.plotly_chart(fig, use_container_width=True)
        return

    # Chart with phase overlays
    fig = _make_wyckoff_chart(ohlcv, analysis, rsi_series, obv_series, ticker)
    st.plotly_chart(fig, use_container_width=True)

    current = analysis.current_phase

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    phase_color = {
        "Accumulation": COLORS["green"], "Distribution": COLORS["red"],
        "Markup": COLORS["blue"], "Markdown": COLORS["orange"],
    }.get(current.phase, COLORS["text"])
    m1.markdown(bloomberg_metric("Current Phase", current.phase, phase_color), unsafe_allow_html=True)
    m2.markdown(bloomberg_metric("Confidence", f"{current.confidence}/100"), unsafe_allow_html=True)
    m3.markdown(
        bloomberg_metric(
            "Key Levels",
            f"S: ${current.key_levels['support']:.2f}  R: ${current.key_levels['resistance']:.2f}",
        ),
        unsafe_allow_html=True,
    )

    # ── Trade Setup Panel ─────────────────────────────────────────────────────
    support = current.key_levels["support"]
    resistance = current.key_levels["resistance"]
    current_price_val = float(close.iloc[-1])

    if current.cause_target:
        st.markdown("### 🎯 Trade Setup")
        is_long = current.phase in ("Accumulation", "Markup")
        setup_color = COLORS["green"] if is_long else COLORS["red"]
        direction_label = "LONG" if is_long else "SHORT"

        if is_long:
            entry_zone_low = support * 1.005
            entry_zone_high = support * 1.02
            stop = support * 0.985
            target = current.cause_target
        else:
            entry_zone_low = resistance * 0.98
            entry_zone_high = resistance * 0.995
            stop = resistance * 1.015
            target = current.cause_target

        risk = abs(current_price_val - stop)
        reward = abs(target - current_price_val)
        rr = reward / risk if risk > 0 else 0

        ts1, ts2, ts3, ts4, ts5 = st.columns(5)
        ts1.markdown(bloomberg_metric("Setup", direction_label, setup_color), unsafe_allow_html=True)
        ts2.markdown(bloomberg_metric("Entry Zone", f"${entry_zone_low:.2f}–${entry_zone_high:.2f}", setup_color), unsafe_allow_html=True)
        ts3.markdown(bloomberg_metric("Stop Loss", f"${stop:.2f}", COLORS["red"]), unsafe_allow_html=True)
        ts4.markdown(bloomberg_metric("C&E Target", f"${target:.2f}", setup_color), unsafe_allow_html=True)
        ts5.markdown(bloomberg_metric("Risk:Reward", f"1 : {rr:.1f}", COLORS["green"] if rr >= 2 else COLORS["yellow"]), unsafe_allow_html=True)

        sub = current.sub_phase
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["text_dim"]};margin-top:6px;">'
            f'Phase: <b style="color:{setup_color}">{current.phase} — Sub-phase {sub}</b> · '
            f'Confidence: <b>{current.confidence}/100</b> · '
            f'Expected move: <b style="color:{setup_color}">{((target - current_price_val) / current_price_val * 100):+.1f}%</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Events list
    all_events = []
    for phase in analysis.all_phases:
        all_events.extend(phase.events)
    if all_events:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};margin:8px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em;">'
            f'Wyckoff Events Detected</div>',
            unsafe_allow_html=True,
        )
        for evt in sorted(all_events, key=lambda e: e.date):
            st.markdown(f"- **{evt.event_type}** ({evt.date.strftime('%Y-%m-%d')}): {evt.description}")

    # ── VSA Summary ───────────────────────────────────────────────────────────
    if analysis.vsa_bars:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em;">'
            f'Volume Spread Analysis · Recent Signals</div>',
            unsafe_allow_html=True,
        )
        vsa_signal_colors = {
            "Strength": COLORS["green"], "Weakness": COLORS["red"],
            "No Supply": "#88DDCC", "No Demand": "#FF8C00", "Effort No Result": "#FFD700",
        }
        for vsa in sorted(analysis.vsa_bars, key=lambda v: v.date, reverse=True)[:8]:
            vc = vsa_signal_colors.get(vsa.signal, COLORS["text_dim"])
            st.markdown(
                f'<div style="font-size:12px;font-family:\'JetBrains Mono\',monospace;color:{vc};padding:2px 0;">'
                f'{"▲" if vsa.signal in ("Strength","No Supply") else "▼"} '
                f'<b>{vsa.signal}</b> ({vsa.date.strftime("%Y-%m-%d")}) — {vsa.description}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Effort vs Result ──────────────────────────────────────────────────────
    if analysis.effort_vs_result:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em;">'
            f'Effort vs Result Divergences</div>',
            unsafe_allow_html=True,
        )
        for date, desc in analysis.effort_vs_result:
            st.markdown(
                f'<div style="font-size:12px;color:{COLORS["yellow"]};padding:2px 0;">'
                f'⚠ {date.strftime("%Y-%m-%d")} — {desc}</div>',
                unsafe_allow_html=True,
            )

    # AI Narrative
    with st.expander("Wyckoff AI Narrative", expanded=False):
        current_price = float(close.iloc[-1])
        event_strs = tuple(f"{e.event_type}: {e.description}" for e in current.events)
        try:
            narrative = _build_groq_narrative(
                current.phase,
                current.confidence,
                current.key_levels["support"],
                current.key_levels["resistance"],
                event_strs,
                current_price,
            )
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
