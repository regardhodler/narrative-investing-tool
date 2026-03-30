# -*- coding: utf-8 -*-
"""
Module: Elliott Wave Analysis (SPY) — Multi-Degree

Shows all 7 Elliott Wave degrees on a single line chart:
  Grand Supercycle [[I]]-[[V]]  ·  Supercycle (I)-(V)  ·  Cycle I-V
  Primary [1]-[5]  ·  Intermediate (1)-(5)  ·  Minor 1-5  ·  Minute i-v

Layout:
  - SPY line chart with all degree wave overlays (toggle via legend)
  - Volume and RSI sub-panels
  - Metrics: Primary wave | Confidence | Invalidation
  - Per-degree confidence table
  - Fibonacci hits
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
from services.elliott_wave_engine import (
    DEGREE_CONFIGS,
    DEGREE_WAVE_LABELS,
    DEGREE_CORRECTIVE_LABELS,
    get_degree_wave_count,
    get_degree_corrective_count,
    BestCount,
    WaveForecast,
    build_wave_forecast,
    backtest_wave_counts,
)
from services.indicators import rsi
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Degree visual styling ─────────────────────────────────────────────────────

DEGREE_COLOR = {
    "Grand Supercycle": "#FFD700",   # Gold
    "Supercycle":       "#FF8C00",   # Deep orange
    "Cycle":            "#FF4B4B",   # Red
    "Primary":          "#C060FF",   # Violet
    "Intermediate":     "#4B9FFF",   # Blue
    "Minor":            "#00D4AA",   # Teal
    "Minute":           "#88DDCC",   # Pale teal
}

DEGREE_LINE_WIDTH = {
    "Grand Supercycle": 3.5,
    "Supercycle":       3.0,
    "Cycle":            2.5,
    "Primary":          2.2,
    "Intermediate":     1.8,
    "Minor":            1.5,
    "Minute":           1.2,
}

DEGREE_DASH = {
    "Grand Supercycle": "solid",
    "Supercycle":       "solid",
    "Cycle":            "solid",
    "Primary":          "solid",
    "Intermediate":     "dot",
    "Minor":            "dot",
    "Minute":           "dash",
}

DEGREE_MARKER_SIZE = {
    "Grand Supercycle": 14,
    "Supercycle":       12,
    "Cycle":            10,
    "Primary":          9,
    "Intermediate":     7,
    "Minor":            6,
    "Minute":           5,
}

DEGREE_FONT_SIZE = {
    "Grand Supercycle": 14,
    "Supercycle":       13,
    "Cycle":            12,
    "Primary":          11,
    "Intermediate":     10,
    "Minor":            10,
    "Minute":           9,
}

# Which degrees are visible by default (others are toggleable via legend)
DEGREE_VISIBLE = {
    "Grand Supercycle": "legendonly",
    "Supercycle":       "legendonly",
    "Cycle":            True,
    "Primary":          True,
    "Intermediate":     True,
    "Minor":            True,
    "Minute":           "legendonly",
}

# Corrective ABC uses same degree color but lighter (alpha) and dashdot
CORRECTIVE_DASH = "dashdot"
CORRECTIVE_OPACITY = 0.75


# ── Data & analysis ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _fetch_ticker_data(ticker: str, interval: str = "1d") -> pd.DataFrame | None:
    """
    Fetch price history for any ticker for multi-degree EW analysis.
    Adjusts lookback period based on interval to stay within yfinance limits.
    Supports US equities, TSX (.TO), futures (=F), ETFs, crypto (-USD), indices (^).
    """
    # Map interval to max valid period
    period_map = {
        "1d":  "max",
        "1wk": "max",
        "1mo": "max",
        "1h":  "730d",  # ~2 years (yfinance max for 1h)
        "15m": "60d",   # ~2 months
        "5m":  "60d",   # ~2 months
    }
    period = period_map.get(interval, "1y")

    # Try fetching with mapped period
    df = fetch_ohlcv_single(ticker, period=period, interval=interval)

    # Fallback for daily/weekly/monthly if max fails
    if (df is None or df.empty) and interval in ("1d", "1wk", "1mo"):
        df = fetch_ohlcv_single(ticker, period="10y", interval=interval)

    return df


@st.cache_data(ttl=3600)
def _build_claude_narrative(prompt: str, model: str) -> str:
    """Call xAI or Claude to generate a narrative."""
    if model and model.startswith("grok-"):
        _xai_key = os.getenv("XAI_API_KEY", "")
        if not _xai_key:
            return "_XAI_API_KEY not set — Grok narrative unavailable._"
        try:
            from services.claude_client import _call_xai
            return _call_xai([{"role": "user", "content": prompt}], model, 600, 0.3)
        except Exception as e:
            return f"_Grok narrative failed: {e}_"
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "_ANTHROPIC_API_KEY not set — Claude narrative unavailable._"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"_Claude narrative failed: {e}_"


@st.cache_data(ttl=3600)
def _claude_wave_analysis(
    ticker: str,
    close_t: tuple,
    high_t: tuple,
    low_t: tuple,
    dates_t: tuple,
    degree_summary: tuple,
    algo_wave: str,
    algo_confidence: int,
    algo_invalidation: float,
    model: str,
) -> dict:
    """Ask Claude or xAI to independently perform Elliott Wave count and return structured JSON."""
    import json
    _is_grok = model and model.startswith("grok-")
    if _is_grok:
        api_key = os.getenv("XAI_API_KEY", "")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}
    # Compact OHLCV — last 30 bars for token efficiency
    n = min(30, len(close_t))
    ohlcv_lines = [
        f"{dates_t[-(n-i)]}: H={high_t[-(n-i)]:.2f} L={low_t[-(n-i)]:.2f} C={close_t[-(n-i)]:.2f}"
        for i in range(n)
    ]
    ohlcv_text = "\n".join(ohlcv_lines)
    degree_text = "\n".join(f"  - {deg}: Wave {lbl} ({conf}%)" for deg, lbl, conf in degree_summary)
    prompt = f"""You are an expert Elliott Wave analyst performing an independent wave count on {ticker}.

OHLCV Data (last {n} bars, oldest first):
{ohlcv_text}

Algorithm preliminary count (use as a starting reference only):
- Primary wave label: {algo_wave}
- Algorithm confidence: {algo_confidence}/100
- Algorithm invalidation: ${algo_invalidation:.2f}

Multi-degree structure from algorithm:
{degree_text}

Independently evaluate the wave structure. Return ONLY valid JSON, no other text:
{{
  "primary_count": "wave label e.g. [3] or C or (5)",
  "confidence": integer 0-100,
  "alternative_count": "alt wave label",
  "alt_confidence": integer 0-100,
  "invalidation": price level as float where this count is invalidated,
  "next_target": nearest price target as float,
  "rationale": "1-2 sentence explanation"
}}"""
    try:
        if _is_grok:
            from services.claude_client import _call_xai
            text = _call_xai([{"role": "user", "content": prompt}], model, 400, 0.2)
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception:
        return {}


def _build_groq_narrative(
    primary_position: str,
    primary_label: str,
    primary_confidence: int,
    primary_invalidation: float,
    fibonacci_hits: tuple,
    degree_summary: tuple,   # tuple of (degree, label, confidence) for hashability
    current_price: float = 0.0,
    ticker_label: str = "SPY",
) -> str:
    """Call Groq LLaMA to generate a multi-degree Elliott Wave narrative."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "_GROQ_API_KEY not set — narrative unavailable._"

    fib_text = "\n".join(f"  - {h}" for h in fibonacci_hits) if fibonacci_hits else "  None detected"
    degree_text = "\n".join(
        f"  - {deg}: Wave {lbl} ({conf}% confidence)"
        for deg, lbl, conf in degree_summary
    )

    price_ctx = ""
    if current_price > 0:
        dist = current_price - primary_invalidation
        dist_pct = (dist / primary_invalidation) * 100 if primary_invalidation else 0
        price_ctx = f"""- Current {ticker_label} Price: ${current_price:.2f}
- Distance to Primary Invalidation: ${dist:+.2f} ({dist_pct:+.1f}%)
"""

    prompt = f"""You are an expert Elliott Wave analyst. Interpret the following automated multi-degree wave count for {ticker_label} and write a concise 4-6 sentence market commentary.

Primary Degree (most actionable):
- Position: {primary_position}
- Active Wave: {primary_label}
- Confidence: {primary_confidence}/100
- Invalidation Level: ${primary_invalidation:.2f}
{price_ctx}
Multi-Degree Structure:
{degree_text}

Fibonacci Confirmations (Primary):
{fib_text}

Write your commentary covering:
1. What the Primary degree wave position means for near-term price action
2. How the larger-degree counts (Cycle, Supercycle) provide context and direction
3. The key invalidation level and what a breach would signal
4. What the fractal wave structure suggests about the current market phase

Be direct and specific. Reference degree notation properly ([[I]] for Grand Supercycle, (I) for Supercycle, I for Cycle, [1] for Primary, (1) for Intermediate, 1 for Minor, i for Minute). Do not hedge excessively."""

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
                "max_tokens": 500,
                "temperature": 0.3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"_Narrative generation failed: {e}_"


# ── Chart ─────────────────────────────────────────────────────────────────────

def _make_wave_chart(
    ohlcv: pd.DataFrame,
    degree_counts: dict[str, BestCount],
    corrective_counts: dict[str, BestCount],
    rsi_series: pd.Series,
    chart_height: int = 860,
    forecast: "WaveForecast | None" = None,
    ticker: str = "SPY",
) -> go.Figure:
    """
    Multi-instrument line chart with all 7 EW degree overlays, volume, and RSI.

    Price is a clean line (not candlestick) to let wave structure shine through.
    Each degree is a colored line connecting its pivot points with text labels.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.15, 0.20],
        vertical_spacing=0.02,
    )

    close = ohlcv["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # ── Price line (neutral, forms the canvas) ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ohlcv.index,
        y=close,
        mode="lines",
        line=dict(color="#3D4F6B", width=1.2),
        name=f"{ticker} Close",
        showlegend=True,
        hovertemplate=f"$%{{y:.2f}}  %{{x|%Y-%m-%d}}<extra>{ticker}</extra>",
    ), row=1, col=1)

    # ── Wave degree overlays ──────────────────────────────────────────────────
    # Plot largest degrees first so smaller ones render on top
    for degree in list(DEGREE_CONFIGS.keys()):
        count = degree_counts.get(degree)
        if count is None:
            continue

        pivots = count.sequence.pivots
        labels_list = count.sequence.labels
        color = DEGREE_COLOR[degree]

        # Per-pivot text positioning: labels above high pivots, below low pivots
        text_pos = []
        for p in pivots:
            text_pos.append("top center" if p.type == "high" else "bottom center")

        fig.add_trace(go.Scatter(
            x=[p.date for p in pivots],
            y=[p.price for p in pivots],
            mode="lines+markers+text",
            line=dict(
                color=color,
                width=DEGREE_LINE_WIDTH[degree],
                dash=DEGREE_DASH[degree],
            ),
            marker=dict(
                size=DEGREE_MARKER_SIZE[degree],
                color=color,
                symbol="circle",
                line=dict(color=COLORS["bg"], width=1.5),
            ),
            text=labels_list,
            textposition=text_pos,
            textfont=dict(
                color=color,
                size=DEGREE_FONT_SIZE[degree],
                family="JetBrains Mono, monospace",
            ),
            name=f"{degree} ({count.confidence}%)",
            visible=DEGREE_VISIBLE[degree],
            hovertemplate=(
                f"<b>{degree}</b><br>"
                "Wave %{text}<br>"
                "$%{y:.2f}  %{x|%Y-%m-%d}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # ── Corrective ABC overlays ───────────────────────────────────────────────
    for degree in list(DEGREE_CONFIGS.keys()):
        count = corrective_counts.get(degree)
        if count is None:
            continue

        pivots = count.sequence.pivots
        labels_list = count.sequence.labels
        color = DEGREE_COLOR[degree]

        text_pos = []
        for p in pivots:
            text_pos.append("top center" if p.type == "high" else "bottom center")

        fig.add_trace(go.Scatter(
            x=[p.date for p in pivots],
            y=[p.price for p in pivots],
            mode="lines+markers+text",
            line=dict(
                color=color,
                width=DEGREE_LINE_WIDTH[degree],
                dash=CORRECTIVE_DASH,
            ),
            marker=dict(
                size=DEGREE_MARKER_SIZE[degree],
                color=color,
                symbol="diamond",
                opacity=CORRECTIVE_OPACITY,
                line=dict(color=COLORS["bg"], width=1),
            ),
            text=labels_list,
            textposition=text_pos,
            textfont=dict(
                color=color,
                size=DEGREE_FONT_SIZE[degree],
                family="JetBrains Mono, monospace",
            ),
            name=f"{degree} ABC ({count.confidence}%)",
            visible=DEGREE_VISIBLE[degree],
            opacity=CORRECTIVE_OPACITY,
            hovertemplate=(
                f"<b>{degree} Corrective</b><br>"
                "Wave %{text}<br>"
                "$%{y:.2f}  %{x|%Y-%m-%d}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────────
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
        opacity=0.55,
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=rsi_series.index,
        y=rsi_series.values,
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.2),
        name="RSI(14)",
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],   line_width=0.8, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot",  line_color=COLORS["text_dim"], line_width=0.6, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], line_width=0.8, row=3, col=1)

    # ── Forecast zigzag overlay ───────────────────────────────────────────────
    if forecast is not None:
        last_date = ohlcv.index[-1]
        # Estimate forward bar spacing (use median bar spacing)
        if len(ohlcv) >= 2:
            bar_spacing = pd.Timedelta(
                seconds=(ohlcv.index[-1] - ohlcv.index[-2]).total_seconds()
            )
        else:
            bar_spacing = pd.Timedelta(days=1)

        # Build zigzag path: current → waypoint (if any) → primary target
        fwd_dates = []
        fwd_prices = []
        fwd_labels_fc = []

        fwd_dates.append(last_date)
        fwd_prices.append(forecast.current_price)
        fwd_labels_fc.append("Now")

        step = 20  # bars ahead for waypoint
        if forecast.waypoint_target is not None:
            fwd_dates.append(last_date + bar_spacing * step)
            fwd_prices.append(forecast.waypoint_target)
            fwd_labels_fc.append(f"{'W4' if forecast.wave_label in ['3','[3]','(3)','iii'] else 'B'}")

        fwd_dates.append(last_date + bar_spacing * step * 2)
        fwd_prices.append(forecast.primary_target)
        fwd_labels_fc.append(f"Target\n${forecast.primary_target:.0f}")

        # Primary scenario (bright)
        fc_color = COLORS["green"] if forecast.direction == "Bullish" else COLORS["red"]
        fig.add_trace(go.Scatter(
            x=fwd_dates,
            y=fwd_prices,
            mode="lines+markers+text",
            line=dict(color=fc_color, width=2, dash="dash"),
            marker=dict(size=10, color=fc_color, symbol="arrow-right"),
            text=fwd_labels_fc,
            textposition="top right",
            textfont=dict(color=fc_color, size=10, family="JetBrains Mono"),
            name=f"Forecast Primary ({forecast.primary_probability}%)",
            hovertemplate="<b>Forecast</b><br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # Alternate scenario (dim)
        alt_color = COLORS["red"] if forecast.direction == "Bullish" else COLORS["green"]
        fig.add_trace(go.Scatter(
            x=[last_date, last_date + bar_spacing * 15],
            y=[forecast.current_price, forecast.alternate_target],
            mode="lines+markers+text",
            line=dict(color=alt_color, width=1.5, dash="dot"),
            marker=dict(size=8, color=alt_color, symbol="arrow-right", opacity=0.6),
            text=["", f"Alt\n${forecast.alternate_target:.0f}"],
            textposition="bottom right",
            textfont=dict(color=alt_color, size=9, family="JetBrains Mono"),
            name=f"Forecast Alt ({forecast.alternate_probability}%)",
            opacity=0.6,
            hovertemplate="<b>Alt Scenario</b><br>$%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # Invalidation horizontal line
        fig.add_hline(
            y=forecast.invalidation,
            line_dash="dot",
            line_color=COLORS["red"],
            line_width=1,
            annotation_text=f"Invalidation ${forecast.invalidation:.0f}",
            annotation_font_color=COLORS["red"],
            annotation_font_size=10,
            row=1, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=chart_height,
        margin=dict(l=20, r=60, t=120, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10, family="JetBrains Mono"),
            bgcolor="rgba(14,17,23,0.85)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            tracegroupgap=0,
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        dragmode="pan",  # Default to pan (hold left-click to drag)
    )
    # Price axis on right, free vertical zoom/drag on all subplots
    fig.update_yaxes(
        title_text="Price ($)", row=1, col=1,
        gridcolor=COLORS["grid"],
        side="right",
        fixedrange=False,
    )
    fig.update_yaxes(
        title_text="Volume", row=2, col=1,
        gridcolor=COLORS["grid"],
        side="right",
        fixedrange=False,
    )
    fig.update_yaxes(
        title_text="RSI(14)", row=3, col=1, range=[0, 100],
        gridcolor=COLORS["grid"],
        side="right",
        fixedrange=False,
    )
    # X-axes: free horizontal zoom/pan
    fig.update_xaxes(fixedrange=False)

    apply_dark_layout(fig, title="")
    # Title centered at top, above the legend row
    fig.add_annotation(
        text=f"{ticker} — Elliott Wave · All Degrees",
        xref="paper", yref="paper",
        x=0.5, y=1.22,
        showarrow=False,
        font=dict(size=13, color=COLORS["text_dim"], family="JetBrains Mono"),
        xanchor="center",
    )
    return fig


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    st.title("Elliott Wave Analysis")
    
    # ── Ticker Selector ───────────────────────────────────────────────────────
    # Use a plain session state key (not a widget key) so quick-pick buttons
    # can update it freely without triggering the "widget key locked" error.
    if "ew_ticker" not in st.session_state:
        st.session_state["ew_ticker"] = ""

    # Human-readable labels for cryptic futures/forex tickers
    _TICKER_LABELS: dict[str, str] = {
        # US Equity Futures
        "ES=F": "ES=F  S&P 500",  "NQ=F": "NQ=F  Nasdaq",
        "YM=F": "YM=F  Dow",     "RTY=F": "RTY=F  Russell",
        # Commodities
        "GC=F": "GC=F  Gold",    "SI=F": "SI=F  Silver",
        "CL=F": "CL=F  Crude Oil","NG=F": "NG=F  Nat Gas",
        "HG=F": "HG=F  Copper",  "PL=F": "PL=F  Platinum",
        "PA=F": "PA=F  Palladium","ZC=F": "ZC=F  Corn",
        "ZW=F": "ZW=F  Wheat",   "ZS=F": "ZS=F  Soybeans",
        "ZL=F": "ZL=F  Soy Oil", "CC=F": "CC=F  Cocoa",
        "KC=F": "KC=F  Coffee",  "CT=F": "CT=F  Cotton",
        "LBS=F":"LBS=F Lumber",
        # Bonds / Rates
        "ZB=F": "ZB=F  30yr T-Bond","ZN=F": "ZN=F  10yr T-Note",
        "ZF=F": "ZF=F  5yr T-Note", "ZT=F": "ZT=F  2yr T-Note",
        "^TNX": "^TNX  10yr Yield","^TYX": "^TYX  30yr Yield",
        # TSX Index
        "^GSPTSE": "^GSPTSE  TSX",
        # Forex
        "EURUSD=X": "EUR/USD","GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY","USDCAD=X": "USD/CAD",
        "AUDUSD=X": "AUD/USD","USDCHF=X": "USD/CHF",
        # Crypto
        "BTC-USD": "BTC-USD  Bitcoin",  "ETH-USD": "ETH-USD  Ethereum",
        "SOL-USD": "SOL-USD  Solana",   "XRP-USD": "XRP-USD  XRP/Ripple",
        "BNB-USD": "BNB-USD  BNB",      "DOGE-USD": "DOGE-USD  Dogecoin",
        # Global Indices
        "^N225":   "^N225    Nikkei 225 (Japan)",
        "000001.SS": "000001.SS  Shanghai Composite (China)",
        "^HSI":    "^HSI     Hang Seng (Hong Kong)",
        "^NSEI":   "^NSEI    Nifty 50 (India)",
        "^BSESN":  "^BSESN   Sensex (India)",
        "^AXJO":   "^AXJO    ASX 200 (Australia)",
        "^FTSE":   "^FTSE    FTSE 100 (UK)",
        "^GDAXI":  "^GDAXI   DAX (Germany)",
        "^FCHI":   "^FCHI    CAC 40 (France)",
        "^STOXX50E": "^STOXX50E  Euro Stoxx 50",
        "^KS11":   "^KS11    KOSPI (South Korea)",
        "^TWII":   "^TWII    Taiwan Weighted",
    }

    # Full descriptions shown in the ticker info bar
    _TICKER_DESC: dict[str, tuple[str, str]] = {
        # (full name, asset class / detail)
        # US ETFs & Indices
        "SPY":  ("SPDR S&P 500 ETF",         "US Large-Cap Equity ETF"),
        "QQQ":  ("Invesco Nasdaq-100 ETF",    "US Tech/Growth Equity ETF"),
        "IWM":  ("iShares Russell 2000 ETF",  "US Small-Cap Equity ETF"),
        "DIA":  ("SPDR Dow Jones ETF",        "US Blue-Chip Equity ETF"),
        "^SPX": ("S&P 500 Index",             "US Large-Cap Index"),
        "^NDX": ("Nasdaq-100 Index",          "US Tech Index"),
        "^DJI": ("Dow Jones Industrial Avg",  "US Blue-Chip Index"),
        # Popular US Stocks
        "AAPL": ("Apple Inc.",                "Technology — Consumer Electronics"),
        "MSFT": ("Microsoft Corp.",           "Technology — Cloud & Software"),
        "NVDA": ("NVIDIA Corp.",              "Technology — Semiconductors / AI"),
        "TSLA": ("Tesla Inc.",               "Automotive / Clean Energy"),
        "AMZN": ("Amazon.com Inc.",           "E-Commerce & Cloud (AWS)"),
        "META": ("Meta Platforms Inc.",       "Social Media / VR"),
        "GOOGL":("Alphabet Inc.",             "Technology — Search & Advertising"),
        "GOOG": ("Alphabet Inc. (C Shares)",  "Technology — Search & Advertising"),
        "BRK-B":("Berkshire Hathaway B",      "Conglomerate / Financial"),
        "JPM":  ("JPMorgan Chase & Co.",      "Banking & Financial Services"),
        "GS":   ("Goldman Sachs Group",       "Investment Banking"),
        "XOM":  ("Exxon Mobil Corp.",         "Energy — Integrated Oil & Gas"),
        # TSX
        "^GSPTSE":("S&P/TSX Composite Index", "Canadian Broad Market Index"),
        "XIU.TO":("iShares S&P/TSX 60 ETF",  "Canadian Large-Cap ETF"),
        "RY.TO": ("Royal Bank of Canada",     "Canadian Banking"),
        "TD.TO": ("TD Bank Group",            "Canadian Banking"),
        "ENB.TO":("Enbridge Inc.",            "Canadian Energy — Pipelines"),
        "SHOP.TO":("Shopify Inc.",            "E-Commerce Platform"),
        "CNR.TO":("Canadian Nat. Railway",    "Transportation — Rail"),
        "ABX.TO":("Barrick Gold Corp.",       "Gold Mining"),
        # Equity Futures
        "ES=F":  ("E-mini S&P 500 Futures",   "US Large-Cap Equity Futures"),
        "NQ=F":  ("E-mini Nasdaq-100 Futures", "US Tech Equity Futures"),
        "YM=F":  ("E-mini Dow Futures",        "US Blue-Chip Equity Futures"),
        "RTY=F": ("E-mini Russell 2000 Futures","US Small-Cap Equity Futures"),
        # Commodities
        "GC=F":  ("Gold Futures",              "Precious Metal — Safe Haven"),
        "SI=F":  ("Silver Futures",            "Precious Metal — Industrial Use"),
        "CL=F":  ("WTI Crude Oil Futures",     "Energy — Benchmark Crude"),
        "NG=F":  ("Natural Gas Futures",       "Energy — Utility/Heating Fuel"),
        "HG=F":  ("Copper Futures",            "Industrial Metal — Economic Indicator"),
        "PL=F":  ("Platinum Futures",          "Precious Metal — Industrial/Auto"),
        "PA=F":  ("Palladium Futures",         "Precious Metal — Auto Catalysts"),
        "ZC=F":  ("Corn Futures",              "Agriculture — Feed & Ethanol"),
        "ZW=F":  ("Wheat Futures",             "Agriculture — Food Staple"),
        "ZS=F":  ("Soybean Futures",           "Agriculture — Food & Biofuel"),
        "CC=F":  ("Cocoa Futures",             "Soft Commodity — Food"),
        "KC=F":  ("Coffee Futures",            "Soft Commodity — Beverage"),
        # Bonds / Rates
        "ZB=F":  ("30-Year T-Bond Futures",    "US Government Long Bond"),
        "ZN=F":  ("10-Year T-Note Futures",    "US Government Benchmark Bond"),
        "ZF=F":  ("5-Year T-Note Futures",     "US Government Medium Bond"),
        "ZT=F":  ("2-Year T-Note Futures",     "US Government Short Bond"),
        "TLT":   ("iShares 20+ Yr Treasury ETF","Long-Duration Bond ETF"),
        "IEF":   ("iShares 7-10 Yr Treasury ETF","Medium-Duration Bond ETF"),
        "^TNX":  ("10-Year Treasury Yield",    "US Benchmark Interest Rate"),
        "^TYX":  ("30-Year Treasury Yield",    "US Long-Term Interest Rate"),
        # Forex
        "EURUSD=X":("Euro / US Dollar",        "Forex — Major Pair"),
        "GBPUSD=X":("British Pound / USD",     "Forex — Major Pair"),
        "USDJPY=X":("USD / Japanese Yen",      "Forex — Major Pair"),
        "USDCAD=X":("USD / Canadian Dollar",   "Forex — Major Pair"),
        "AUDUSD=X":("Australian Dollar / USD", "Forex — Major Pair"),
        "USDCHF=X":("USD / Swiss Franc",       "Forex — Safe-Haven Pair"),
        # Crypto
        "BTC-USD": ("Bitcoin",                 "Cryptocurrency — Store of Value"),
        "ETH-USD": ("Ethereum",                "Cryptocurrency — Smart Contract Platform"),
        "SOL-USD": ("Solana",                  "Cryptocurrency — High-Speed L1"),
        "XRP-USD": ("XRP (Ripple)",            "Cryptocurrency — Payments Network"),
        "BNB-USD": ("BNB (Binance Coin)",      "Cryptocurrency — Exchange Token"),
        "DOGE-USD":("Dogecoin",                "Cryptocurrency — Meme Coin"),
        # Global Indices
        "^N225":    ("Nikkei 225",             "Japan — Top 225 Companies"),
        "000001.SS":("Shanghai Composite",     "China — All SSE-Listed Stocks"),
        "^HSI":     ("Hang Seng Index",        "Hong Kong — Top 50 Companies"),
        "^NSEI":    ("Nifty 50",               "India — NSE Top 50 Companies"),
        "^BSESN":   ("BSE Sensex",             "India — BSE Top 30 Companies"),
        "^AXJO":    ("S&P/ASX 200",            "Australia — Top 200 Companies"),
        "^FTSE":    ("FTSE 100",               "UK — London Stock Exchange Top 100"),
        "^GDAXI":   ("DAX 40",                 "Germany — Frankfurt Top 40"),
        "^FCHI":    ("CAC 40",                 "France — Paris Top 40"),
        "^STOXX50E":("Euro Stoxx 50",          "Eurozone — Top 50 Blue Chips"),
        "^KS11":    ("KOSPI",                  "South Korea — Broad Market Index"),
        "^TWII":    ("Taiwan Weighted Index",   "Taiwan — TWSE Broad Market"),
    }

    # Quick-pick data: {tab_label: [(ticker, button_label), ...]}
    _QUICK_PICKS: dict[str, list[tuple[str, str]]] = {
        "🇺🇸 US Equities": [
            ("SPY","SPY"),("QQQ","QQQ"),("IWM","IWM"),("DIA","DIA"),
            ("AAPL","AAPL"),("NVDA","NVDA"),("TSLA","TSLA"),
            ("MSFT","MSFT"),("AMZN","AMZN"),("META","META"),
        ],
        "🇨🇦 TSX": [
            ("^GSPTSE","TSX Index"),("XIU.TO","XIU"),("RY.TO","RY"),
            ("TD.TO","TD"),("ENB.TO","ENB"),("SHOP.TO","SHOP"),
            ("CNR.TO","CNR"),("ABX.TO","ABX"),
        ],
        "📊 Eq. Futures": [
            ("ES=F","ES — S&P 500"),("NQ=F","NQ — Nasdaq"),
            ("YM=F","YM — Dow"),("RTY=F","RTY — Russell"),
        ],
        "🛢 Commodities": [
            ("GC=F","Gold"),("SI=F","Silver"),("CL=F","Crude Oil"),
            ("NG=F","Nat Gas"),("HG=F","Copper"),("PL=F","Platinum"),
            ("ZC=F","Corn"),("ZW=F","Wheat"),("ZS=F","Soybeans"),
            ("CC=F","Cocoa"),("KC=F","Coffee"),
        ],
        "🏛 Bonds/Rates": [
            ("ZB=F","30yr Bond"),("ZN=F","10yr Note"),
            ("ZF=F","5yr Note"),("ZT=F","2yr Note"),
            ("TLT","TLT"),("IEF","IEF"),
            ("^TNX","10yr Yield"),("^TYX","30yr Yield"),
        ],
        "💱 Forex & Crypto": [
            ("EURUSD=X","EUR/USD"),("GBPUSD=X","GBP/USD"),
            ("USDJPY=X","USD/JPY"),("USDCAD=X","USD/CAD"),
            ("AUDUSD=X","AUD/USD"),("USDCHF=X","USD/CHF"),
            ("BTC-USD","Bitcoin"),("ETH-USD","Ethereum"),
            ("SOL-USD","Solana"),("XRP-USD","XRP"),
            ("BNB-USD","BNB"),("DOGE-USD","Dogecoin"),
        ],
        "🌏 Global Indices": [
            ("^N225","Nikkei 225"),
            ("000001.SS","Shanghai"),
            ("^HSI","Hang Seng"),
            ("^NSEI","Nifty 50"),
            ("^BSESN","Sensex"),
            ("^AXJO","ASX 200"),
            ("^FTSE","FTSE 100"),
            ("^GDAXI","DAX"),
            ("^FCHI","CAC 40"),
            ("^STOXX50E","Euro Stoxx 50"),
            ("^KS11","KOSPI"),
            ("^TWII","Taiwan"),
        ],
    }

    tk_col, iv_col = st.columns([3, 1])
    with tk_col:
        raw_ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state["ew_ticker"],
            placeholder="SPY · GC=F · RY.TO · BTC-USD · ZB=F · ^TNX",
            key="ew_ticker_input",
            help=(
                "Any yfinance-compatible symbol. "
                "Append .TO for TSX (e.g. RY.TO), =F for futures (GC=F), "
                "-USD for crypto (BTC-USD), ^ for indices (^TNX). "
                "Note: indices (^) have no volume data."
            ),
        )
    with iv_col:
        interval = st.selectbox(
            "Interval",
            options=["1d", "1wk", "1mo", "1h", "15m", "5m"],
            index=0,
            key="ew_interval",
            label_visibility="collapsed",
        )

    ticker = raw_ticker.strip().upper() if raw_ticker else ""
    # Sync back so quick-pick rerun picks up the latest typed value
    st.session_state["ew_ticker"] = ticker

    with st.expander("⚡ Quick Pick — Asset Classes", expanded=False):
        _qp_tabs = st.tabs(list(_QUICK_PICKS.keys()))
        for _tab, (_cat, _pairs) in zip(_qp_tabs, _QUICK_PICKS.items()):
            with _tab:
                _nc = min(len(_pairs), 5)
                _qcols = st.columns(_nc)
                for _i, (_t, _lbl) in enumerate(_pairs):
                    if _qcols[_i % _nc].button(
                        _lbl, key=f"ew_qp_{_t}", use_container_width=True,
                        help=_TICKER_LABELS.get(_t, _t),
                    ):
                        st.session_state["ew_ticker"] = _t
                        st.session_state["ew_ticker_input"] = _t
                        st.rerun()

    # ── AI Engine Tier (top-level — controls wave count override + narrative) ──
    _has_xai_ew = bool(os.getenv("XAI_API_KEY"))
    _has_anthropic_ew = bool(os.getenv("ANTHROPIC_API_KEY"))
    _ew_tier_options_top = ["⚡ Standard", "🧠 Regard Mode", "👑 Highly Regarded Mode"]
    _ew_default_idx = 0
    _ew_saved = st.session_state.get("ew_narrative_tier", "⚡ Standard")
    if _ew_saved in _ew_tier_options_top:
        _ew_default_idx = _ew_tier_options_top.index(_ew_saved)
    st.radio(
        "AI Engine",
        _ew_tier_options_top,
        index=_ew_default_idx,
        horizontal=True,
        key="ew_narrative_tier",
        disabled=not _has_anthropic_ew,
        help="Standard = Groq LLaMA  ·  Regard = Grok 4.1 (overrides wave count)  ·  Highly Regarded = Claude Sonnet",
    )
    st.markdown(
        '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
        'margin-top:-10px;margin-bottom:2px;">'
        '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
        '</div>',
        unsafe_allow_html=True,
    )
    if not _has_anthropic_ew:
        st.caption("Set XAI_API_KEY for Regard Mode · ANTHROPIC_API_KEY for Highly Regarded.")

    # ── Guard: nothing entered yet ────────────────────────────────────────────
    if not ticker:
        st.info("Enter a ticker symbol above or pick one from ⚡ Quick Pick to begin analysis.")
        return

    # ── Ticker info bar ───────────────────────────────────────────────────
    _desc_entry = _TICKER_DESC.get(ticker)
    if _desc_entry:
        _full_name, _asset_class = _desc_entry
    else:
        _full_name = ticker
        _asset_class = "Custom symbol — enter any yfinance-compatible ticker"
    st.markdown(
        f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            background:#0E1E2E;border:1px solid #1E3A4A;border-radius:6px;
            padding:10px 18px;margin:4px 0 14px 0;">
  <div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:18px;
                 font-weight:700;color:#C8D8E8;">{_full_name}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
                 color:#5A7A8A;margin-left:12px;">{_asset_class}</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
              font-weight:800;color:#4B9FFF;letter-spacing:0.05em;">{ticker}</div>
</div>""",
        unsafe_allow_html=True,
    )

    chart_height = st.slider(
        "Chart Height", min_value=500, max_value=1400, value=860, step=50,
        key="ew_chart_height",
        help="Drag to expand/compress the chart. You can also drag the price axis on the right to zoom Y."
    )

    if st.button("Refresh Data", key="ew_refresh_data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner(f"Fetching {ticker} data ({interval})..."):
        ohlcv = _fetch_ticker_data(ticker, interval=interval)

    if ohlcv is None or ohlcv.empty:
        st.error(f"{ticker} price data unavailable for {interval}. Please try again later.")
        return

    # For intraday, we likely have much less data than daily "max".
    # Warn if really short, but proceed if >= 60 bars.
    min_bars = 60
    if len(ohlcv) < min_bars:
        st.warning(f"Insufficient price history for Elliott Wave analysis (need ≥ {min_bars} bars, got {len(ohlcv)}).")
        return

    close = ohlcv["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    # ── Analyze all 7 degrees ─────────────────────────────────────────────────
    with st.spinner("Computing multi-degree wave structure..."):
        degree_counts: dict[str, BestCount] = {}
        for degree in DEGREE_CONFIGS:
            count = get_degree_wave_count(close, degree)
            if count is not None:
                degree_counts[degree] = count

        corrective_counts: dict[str, BestCount] = {}
        for degree in DEGREE_CONFIGS:
            count = get_degree_corrective_count(close, degree)
            if count is not None:
                corrective_counts[degree] = count

    # Build Primary-degree forecast
    forecast = None
    _fc_src = degree_counts.get("Primary") or corrective_counts.get("Primary")
    if _fc_src:
        forecast = build_wave_forecast(_fc_src, float(close.iloc[-1]))

    # ── Claude AI wave count override (Regard / Highly Regarded) ──────────────
    _ew_tier_now = st.session_state.get("ew_narrative_tier", "⚡ Standard")
    _claude_wa = {}
    if _ew_tier_now in ("🧠 Regard Mode", "👑 Highly Regarded Mode") and (bool(os.getenv("XAI_API_KEY")) or bool(os.getenv("ANTHROPIC_API_KEY"))):
        _ca_model = "grok-4-1-fast-reasoning" if _ew_tier_now == "🧠 Regard Mode" else "claude-sonnet-4-6"
        _primary_for_ca = degree_counts.get("Primary") or corrective_counts.get("Primary")
        if _primary_for_ca:
            _deg_sum = tuple((deg, cnt.current_wave_label, cnt.confidence) for deg, cnt in degree_counts.items())
            _ew_high = ohlcv["High"]
            if isinstance(_ew_high, pd.DataFrame):
                _ew_high = _ew_high.iloc[:, 0]
            _ew_low = ohlcv["Low"]
            if isinstance(_ew_low, pd.DataFrame):
                _ew_low = _ew_low.iloc[:, 0]
            with st.spinner(f"✦ Claude ({_ca_model.split('-')[1].capitalize()}) analyzing wave structure..."):
                _claude_wa = _claude_wave_analysis(
                    ticker,
                    tuple(close.iloc[-60:].round(2).tolist()),
                    tuple(_ew_high.dropna().iloc[-60:].round(2).tolist()),
                    tuple(_ew_low.dropna().iloc[-60:].round(2).tolist()),
                    tuple(str(d)[:10] for d in ohlcv.index[-60:]),
                    _deg_sum,
                    _primary_for_ca.current_wave_label,
                    _primary_for_ca.confidence,
                    _primary_for_ca.invalidation_level,
                    _ca_model,
                )

    rsi_series = rsi(close)

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = _make_wave_chart(ohlcv, degree_counts, corrective_counts, rsi_series, chart_height, forecast, ticker=ticker)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,         # scroll wheel = zoom
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
            "modeBarButtonsToAdd": ["pan2d", "resetScale2d"],
        },
    )

    # ── Forecast Panel ────────────────────────────────────────────────────────
    if forecast:
        st.markdown("### 📈 Wave Forecast")
        fc_color_hex = "#00D4AA" if forecast.direction == "Bullish" else "#FF4B4B"
        alt_color_hex = "#FF4B4B" if forecast.direction == "Bullish" else "#00D4AA"

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.markdown(
            bloomberg_metric("Direction", f"{'▲ ' if forecast.direction == 'Bullish' else '▼ '}{forecast.direction}", fc_color_hex),
            unsafe_allow_html=True,
        )
        fc2.markdown(
            bloomberg_metric("Primary Target", f"${forecast.primary_target:.2f}", fc_color_hex),
            unsafe_allow_html=True,
        )
        fc3.markdown(
            bloomberg_metric("Probability", f"{forecast.primary_probability}%", fc_color_hex),
            unsafe_allow_html=True,
        )
        fc4.markdown(
            bloomberg_metric("Alt Target", f"${forecast.alternate_target:.2f}", alt_color_hex),
            unsafe_allow_html=True,
        )

        # Probability bar
        st.markdown(
            f"""<div style="margin:12px 0 4px 0;font-family:'JetBrains Mono',monospace;font-size:11px;color:#8899AA;text-transform:uppercase;letter-spacing:0.08em;">Scenario Probability</div>
<div style="display:flex;height:18px;border-radius:4px;overflow:hidden;border:1px solid #2A3A4A;">
  <div style="width:{forecast.primary_probability}%;background:{fc_color_hex};display:flex;align-items:center;justify-content:center;font-size:10px;font-family:monospace;color:#0E1117;font-weight:700;">{forecast.primary_probability}%</div>
  <div style="width:{forecast.alternate_probability}%;background:{alt_color_hex};display:flex;align-items:center;justify-content:center;font-size:10px;font-family:monospace;color:#0E1117;font-weight:700;">{forecast.alternate_probability}%</div>
</div>""",
            unsafe_allow_html=True,
        )

        # Rationale + waypoint
        st.markdown(
            f'<div style="margin:10px 0 4px 0;font-size:13px;color:#C8D8E8;">'
            f'💡 {forecast.rationale}</div>',
            unsafe_allow_html=True,
        )
        if forecast.waypoint_target:
            st.markdown(
                f'<div style="font-size:12px;color:#8899AA;">⤷ Intermediate waypoint: {forecast.waypoint_label}</div>',
                unsafe_allow_html=True,
            )

        col_a, col_b = st.columns(2)
        col_a.markdown(
            bloomberg_metric("Primary Scenario", forecast.primary_label, fc_color_hex),
            unsafe_allow_html=True,
        )
        col_b.markdown(
            bloomberg_metric("Alt Scenario", forecast.alternate_label, alt_color_hex),
            unsafe_allow_html=True,
        )

    # ── Warning if no counts found ────────────────────────────────────────────
    if not degree_counts:
        st.info(
            "No valid Elliott Wave impulse counts detected. "
            "Market may be in a complex correction or transitional phase."
        )
        return

    # ── Primary degree metrics ────────────────────────────────────────────────
    primary = degree_counts.get("Primary")
    if primary:
        if primary.confidence < 35 and not _claude_wa:
            st.warning(
                f"Primary degree count is low-confidence ({primary.confidence}/100) — "
                "structural rules pass but Fibonacci confirmation is weak."
            )

        # Use Claude's override when available, otherwise fall back to algorithm
        _wave_label = _claude_wa.get("primary_count", primary.current_wave_label)
        _confidence = _claude_wa.get("confidence", primary.confidence)
        _invalidation = _claude_wa.get("invalidation", primary.invalidation_level)
        _next_target = _claude_wa.get("next_target")
        _alt_count = _claude_wa.get("alternative_count")
        _alt_conf = _claude_wa.get("alt_confidence")

        if _claude_wa:
            st.markdown(
                '<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);color:#fff;'
                'font-size:10px;font-weight:700;letter-spacing:0.06em;padding:2px 10px;'
                'border-radius:3px;margin-bottom:10px;display:inline-block;">'
                '✦ CLAUDE WAVE OVERRIDE</span>',
                unsafe_allow_html=True,
            )

        if _claude_wa and _next_target:
            m1, m2, m3, m4, m5 = st.columns(5)
        else:
            m1, m2, m3 = st.columns(3)

        m1.markdown(
            bloomberg_metric("Primary Wave", _wave_label, DEGREE_COLOR["Primary"]),
            unsafe_allow_html=True,
        )
        m2.markdown(
            bloomberg_metric("Confidence", f"{_confidence}/100"),
            unsafe_allow_html=True,
        )
        m3.markdown(
            bloomberg_metric("Invalidation", f"${_invalidation:.2f}", COLORS["red"]),
            unsafe_allow_html=True,
        )
        if _claude_wa and _next_target:
            m4.markdown(
                bloomberg_metric("Next Target", f"${_next_target:.2f}", COLORS["green"]),
                unsafe_allow_html=True,
            )
            m5.markdown(
                bloomberg_metric("Alt Count", f"{_alt_count} ({_alt_conf}%)" if _alt_count else "—", COLORS["yellow"]),
                unsafe_allow_html=True,
            )

        if _claude_wa:
            _rationale = _claude_wa.get("rationale", "")
            if _rationale:
                st.markdown(
                    f'<div style="margin:8px 0 4px 0;font-size:13px;color:#C8D8E8;">💬 {_rationale}</div>',
                    unsafe_allow_html=True,
                )
            # Show algorithm's original count as reference
            st.markdown(
                f'<div style="font-size:11px;color:#5A7A8A;margin-top:4px;">'
                f'Algorithm count: Wave {primary.current_wave_label} · {primary.confidence}% · '
                f'Invalidation ${primary.invalidation_level:.2f}</div>',
                unsafe_allow_html=True,
            )

    # ── All-degree confidence table ───────────────────────────────────────────
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
        f'color:{COLORS["text_dim"]};margin:14px 0 6px 0;text-transform:uppercase;'
        f'letter-spacing:0.08em;">Wave Count by Degree</div>',
        unsafe_allow_html=True,
    )
    degree_cols = st.columns(len(DEGREE_CONFIGS))
    for i, degree in enumerate(DEGREE_CONFIGS):
        count = degree_counts.get(degree)
        color = DEGREE_COLOR.get(degree, COLORS["text_dim"])
        if count:
            degree_cols[i].markdown(
                bloomberg_metric(
                    degree.replace(" ", "<br>"),
                    f"Wave {count.current_wave_label}<br><span style='font-size:13px;'>"
                    f"{count.confidence}%</span>",
                    color,
                ),
                unsafe_allow_html=True,
            )
        else:
            degree_cols[i].markdown(
                bloomberg_metric(degree.replace(" ", "<br>"), "—", COLORS["text_dim"]),
                unsafe_allow_html=True,
            )

    # ── Fibonacci hits (Primary) ──────────────────────────────────────────────
    if primary and primary.fibonacci_hits:
        # Calculate total score from hits
        total_pts = sum(
            int(h.split("(+")[1].replace("pts)", ""))
            for h in primary.fibonacci_hits if "(+" in h
        )
        max_pts = 75  # impulse max (15+25+15+12+8 alternation)
        score_pct = min(int(round(total_pts / max_pts * 100)), 100)
        score_color = COLORS["green"] if score_pct >= 60 else (COLORS["yellow"] if score_pct >= 30 else COLORS["red"])

        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:12px;'
            f'color:{COLORS["text_dim"]};margin:12px 0 4px 0;text-transform:uppercase;'
            f'letter-spacing:0.06em;">Fibonacci Confirmations · Primary</div>',
            unsafe_allow_html=True,
        )

        # Score bar + context
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
  <div style="flex:1;background:#1A2535;border-radius:4px;height:14px;overflow:hidden;border:1px solid #2A3A4A;">
    <div style="width:{score_pct}%;background:{score_color};height:100%;border-radius:4px;"></div>
  </div>
  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{score_color};white-space:nowrap;">
    {total_pts} / {max_pts} pts &nbsp;({score_pct}%)
  </span>
</div>
<div style="font-size:11px;color:{COLORS['text_dim']};margin-bottom:8px;">
  Points reflect how closely wave ratios match EW Fibonacci theory. 
  Max 67 pts = perfect impulse (W2: 15pts · W3: 25pts · W4: 15pts · W5: 12pts).
  Higher = stronger structural confirmation.
</div>""",
            unsafe_allow_html=True,
        )

        for hit in primary.fibonacci_hits:
            # Parse pts for color coding
            try:
                pts = int(hit.split("(+")[1].replace("pts)", ""))
                h_color = COLORS["green"] if pts >= 20 else (COLORS["yellow"] if pts >= 12 else COLORS["text_dim"])
            except Exception:
                h_color = COLORS["text_dim"]
                pts = 0
            st.markdown(
                f'<div style="font-size:12px;font-family:\'JetBrains Mono\',monospace;'
                f'color:{h_color};padding:2px 0;">✦ {hit}</div>',
                unsafe_allow_html=True,
            )
    elif primary:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["text_dim"]};">'
            f'No Fibonacci ratios confirmed at Primary degree — structural rules pass but Fibonacci alignment is weak.</div>',
            unsafe_allow_html=True,
        )

    # ── AI Narrative ──────────────────────────────────────────────────────────
    with st.expander("Elliott Wave AI Narrative", expanded=False):
        _has_xai = bool(os.getenv("XAI_API_KEY"))
        _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        _ew_tier_map = {
            "⚡ Standard":            (False, None),
            "🧠 Regard Mode":         (True,  "grok-4-1-fast-reasoning"),
            "👑 Highly Regarded Mode": (True,  "claude-sonnet-4-6"),
        }
        # Tier is set at top of page — show status here
        _ew_tier = st.session_state.get("ew_narrative_tier", "⚡ Standard")
        st.caption(f"Engine: {_ew_tier} — change at top of page")
        _ew_use_claude, _ew_model = _ew_tier_map.get(_ew_tier, (False, None))

        if primary is None:
            st.warning("Primary degree count unavailable — AI narrative requires Primary degree data.")
        else:
            current_price = float(close.iloc[-1])
            degree_summary = tuple(
                (deg, cnt.current_wave_label, cnt.confidence)
                for deg, cnt in degree_counts.items()
            )

            # Build prompt (shared between Groq and Claude)
            fib_text = "\n".join(f"  - {h}" for h in primary.fibonacci_hits) if primary.fibonacci_hits else "  None detected"
            degree_text = "\n".join(
                f"  - {deg}: Wave {lbl} ({conf}% confidence)"
                for deg, lbl, conf in degree_summary
            )
            dist = current_price - primary.invalidation_level
            dist_pct = (dist / primary.invalidation_level * 100) if primary.invalidation_level else 0
            _ew_prompt = f"""You are an expert Elliott Wave analyst. Interpret the following automated multi-degree wave count for {ticker} and write a concise 4-6 sentence market commentary.

Primary Degree (most actionable):
- Position: {primary.wave_position}
- Active Wave: {primary.current_wave_label}
- Confidence: {primary.confidence}/100
- Invalidation Level: ${primary.invalidation_level:.2f}
- Current {ticker} Price: ${current_price:.2f}
- Distance to Primary Invalidation: ${dist:+.2f} ({dist_pct:+.1f}%)

Multi-Degree Structure:
{degree_text}

Fibonacci Confirmations (Primary):
{fib_text}

Write your commentary covering:
1. What the Primary degree wave position means for near-term price action
2. How the larger-degree counts (Cycle, Supercycle) provide context and direction
3. The key invalidation level and what a breach would signal
4. What the fractal wave structure suggests about the current market phase

Be direct and specific. Reference degree notation properly ([[I]] for Grand Supercycle, (I) for Supercycle, I for Cycle, [1] for Primary, (1) for Intermediate, 1 for Minor, i for Minute). Do not hedge excessively."""

            try:
                if _ew_use_claude and _has_anthropic:
                    narrative = _build_claude_narrative(_ew_prompt, _ew_model)
                    _model_label = _ew_model
                else:
                    narrative = _build_groq_narrative(
                        primary.wave_position,
                        primary.current_wave_label,
                        primary.confidence,
                        primary.invalidation_level,
                        tuple(primary.fibonacci_hits),
                        degree_summary,
                        current_price,
                        ticker_label=ticker,
                    )
                    _model_label = GROQ_MODEL

                if narrative.startswith("_Narrative generation failed") or narrative.startswith("_GROQ") or narrative.startswith("_Claude") or narrative.startswith("_ANTHROPIC"):
                    st.warning("AI narrative unavailable.")
                    if st.button("Retry Narrative", key="ew_retry_narrative"):
                        _build_groq_narrative.clear()
                        st.rerun()
                else:
                    if _ew_use_claude and _has_anthropic:
                        st.markdown(
                            '<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);'
                            'color:#fff;font-size:10px;font-weight:700;letter-spacing:0.06em;'
                            'padding:2px 8px;border-radius:3px;margin-bottom:8px;display:inline-block;">'
                            '✦ POWERED BY CLAUDE</span>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(narrative)
                    st.caption(
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
                        f"Model: {_model_label}"
                    )
            except Exception:
                st.warning("AI narrative unavailable.")
                if st.button("Retry Narrative", key="ew_retry_narrative_err"):
                    st.cache_data.clear()
                    st.rerun()

    # ── Backtest Module ───────────────────────────────────────────────────────
    with st.expander("Backtest Wave Accuracy", expanded=False):
        st.markdown(
            "Run a historical simulation to test how often Elliott Wave signals "
            "(Waves 3 & 5) hit their Fibonacci targets before invalidation."
        )

        # ── Controls row ──────────────────────────────────────────────────
        b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
        with b_col1:
            test_degree = st.selectbox(
                "Degree to Test",
                options=["Primary", "Intermediate", "Minor"],
                index=0,
                key="bt_ew_degree",
            )
        _today = datetime.today().date()
        _one_year_ago = _today.replace(year=_today.year - 1)
        with b_col2:
            bt_start = st.date_input(
                "Start Date",
                value=_one_year_ago,
                min_value=datetime(2000, 1, 1).date(),
                max_value=_today,
                key="bt_ew_start",
            )
        with b_col3:
            bt_end = st.date_input(
                "End Date",
                value=_today,
                min_value=datetime(2000, 1, 1).date(),
                max_value=_today,
                key="bt_ew_end",
            )

        if st.button(f"Run Backtest ({test_degree})", key="bt_ew_run"):
            if bt_start >= bt_end:
                st.warning("Start date must be before end date.")
            else:
                with st.spinner("Simulating trades..."):
                    results = backtest_wave_counts(
                        close,
                        degree=test_degree,
                        start_date=str(bt_start),
                        end_date=str(bt_end),
                    )

                if not results:
                    st.warning("No completed wave setups found in the selected date range.")
                else:
                    st.session_state["bt_ew_results"] = results
                    st.session_state["bt_ew_range"] = (str(bt_start), str(bt_end))

        results = st.session_state.get("bt_ew_results")
        if results:
            date_range = st.session_state.get("bt_ew_range", ("", ""))

            # ── Accuracy metrics ──────────────────────────────────────────
            hits   = [r for r in results if r.outcome == "Target Hit"]
            stops  = [r for r in results if r.outcome == "Invalidated"]
            open_t = [r for r in results if r.outcome == "Indeterminate"]
            closed = hits + stops
            win_rate = (len(hits) / len(closed) * 100) if closed else 0.0

            # Per-wave-type accuracy
            w3 = [r for r in closed if r.wave_label == "Wave 3"]
            w5 = [r for r in closed if r.wave_label == "Wave 5"]
            w3_wr = (sum(1 for r in w3 if r.outcome == "Target Hit") / len(w3) * 100) if w3 else None
            w5_wr = (sum(1 for r in w5 if r.outcome == "Target Hit") / len(w5) * 100) if w5 else None

            avg_win  = (sum(r.pnl_pct for r in hits if r.pnl_pct)  / len(hits)  if hits  else 0.0)
            avg_loss = (sum(r.pnl_pct for r in stops if r.pnl_pct) / len(stops) if stops else 0.0)
            rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

            st.markdown(
                f'<div style="font-size:11px;color:#888;margin-bottom:6px;">'
                f'Period: {date_range[0]} → {date_range[1]}</div>',
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Signals", len(results))
            m2.metric("Win Rate", f"{win_rate:.1f}%")
            m3.metric("Hits / Stops", f"{len(hits)} / {len(stops)}")
            m4.metric("Open", len(open_t))
            m5.metric("Avg Win", f"{avg_win:+.1f}%")
            m6.metric("R:R", f"{rr:.2f}x" if rr != float('inf') else "∞")

            # Wave-type split
            wt_col1, wt_col2 = st.columns(2)
            wt_col1.metric("Wave 3 Accuracy", f"{w3_wr:.1f}%" if w3_wr is not None else "N/A",
                           delta=f"{len(w3)} trades")
            wt_col2.metric("Wave 5 Accuracy", f"{w5_wr:.1f}%" if w5_wr is not None else "N/A",
                           delta=f"{len(w5)} trades")

            # ── Outcome distribution chart ────────────────────────────────
            import plotly.graph_objects as go
            from utils.theme import apply_dark_layout, COLORS as _C

            outcome_counts = {
                "Target Hit": len(hits),
                "Invalidated": len(stops),
                "Open": len(open_t),
            }
            bar_colors = [_C.get("green", "#00c853"), _C.get("red", "#ff1744"), _C.get("bloomberg_orange", "#ff9800")]
            fig_out = go.Figure(go.Bar(
                x=list(outcome_counts.keys()),
                y=list(outcome_counts.values()),
                marker_color=bar_colors,
                text=list(outcome_counts.values()),
                textposition="outside",
            ))
            apply_dark_layout(fig_out)
            fig_out.update_layout(
                title="Outcome Distribution",
                height=280,
                showlegend=False,
                margin=dict(t=36, b=24, l=24, r=24),
            )
            st.plotly_chart(fig_out, use_container_width=True, key="bt_ew_outcome_chart")

            # ── PnL over time (equity-style scatter) ──────────────────────
            sorted_res = sorted(results, key=lambda r: r.date)
            cum_pnl, running = [], 0.0
            for r in sorted_res:
                if r.pnl_pct is not None:
                    running += r.pnl_pct
                cum_pnl.append(running)

            fig_eq = go.Figure(go.Scatter(
                x=[r.date for r in sorted_res],
                y=cum_pnl,
                mode="lines+markers",
                line=dict(color=_C.get("bloomberg_orange", "#ff9800"), width=2),
                marker=dict(
                    color=[
                        _C.get("green", "#00c853") if r.outcome == "Target Hit"
                        else (_C.get("red", "#ff1744") if r.outcome == "Invalidated"
                              else "#888")
                        for r in sorted_res
                    ],
                    size=7,
                ),
            ))
            apply_dark_layout(fig_eq)
            fig_eq.update_layout(
                title="Cumulative PnL % (per signal)",
                height=280,
                margin=dict(t=36, b=24, l=24, r=24),
            )
            fig_eq.add_hline(y=0, line_dash="dot", line_color="#555")
            st.plotly_chart(fig_eq, use_container_width=True, key="bt_ew_equity_chart")

            # ── Detailed Trade Log ────────────────────────────────────────
            st.markdown("#### Trade Log")
            data = []
            for r in sorted_res:
                pnl = f"{r.pnl_pct:+.1f}%" if r.pnl_pct is not None else "—"
                outcome_label = {
                    "Target Hit": "✅ Hit",
                    "Invalidated": "❌ Stop",
                    "Indeterminate": "⏳ Open",
                }.get(r.outcome, r.outcome)
                data.append({
                    "Date": r.date.strftime("%Y-%m-%d"),
                    "Wave": r.wave_label,
                    "Dir": r.direction,
                    "Entry": f"${r.entry_price:.2f}",
                    "Target": f"${r.target_price:.2f}",
                    "Stop": f"${r.invalidation_price:.2f}",
                    "Outcome": outcome_label,
                    "PnL": pnl,
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
