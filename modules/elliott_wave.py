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
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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
        st.session_state["ew_ticker"] = "SPY"

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
            label_visibility="collapsed",
        )

    ticker = (raw_ticker or "SPY").strip().upper()
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
                        st.rerun()

    # ── Controls ─────────────────────────────────────────────────────────────
    st.caption(
        f"{ticker} · 7 degrees · Impulse 1-2-3-4-5 only · Toggle degrees via legend · Groq AI narrative"
    )

    chart_height = st.slider(
        "Chart Height", min_value=500, max_value=1400, value=860, step=50,
        help="Drag to expand/compress the chart. You can also drag the price axis on the right to zoom Y."
    )

    if st.button("Refresh Data"):
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
        if primary.confidence < 35:
            st.warning(
                f"Primary degree count is low-confidence ({primary.confidence}/100) — "
                "structural rules pass but Fibonacci confirmation is weak."
            )
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            bloomberg_metric("Primary Wave", primary.current_wave_label, DEGREE_COLOR["Primary"]),
            unsafe_allow_html=True,
        )
        m2.markdown(
            bloomberg_metric("Confidence", f"{primary.confidence}/100"),
            unsafe_allow_html=True,
        )
        m3.markdown(
            bloomberg_metric("Invalidation", f"${primary.invalidation_level:.2f}", COLORS["red"]),
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
        max_pts = 67  # impulse max
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
        if primary is None:
            st.warning("Primary degree count unavailable — AI narrative requires Primary degree data.")
        else:
            current_price = float(close.iloc[-1])
            degree_summary = tuple(
                (deg, cnt.current_wave_label, cnt.confidence)
                for deg, cnt in degree_counts.items()
            )
            try:
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
                if narrative.startswith("_Narrative generation failed") or narrative.startswith("_GROQ"):
                    st.warning("AI narrative unavailable.")
                    if st.button("Retry Narrative", key="retry_narrative"):
                        _build_groq_narrative.clear()
                        st.rerun()
                else:
                    st.markdown(narrative)
                    st.caption(
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
                        f"Model: {GROQ_MODEL}"
                    )
            except Exception:
                st.warning("AI narrative unavailable.")
                if st.button("Retry Narrative", key="retry_narrative_err"):
                    st.cache_data.clear()
                    st.rerun()

    # ── Backtest Module ───────────────────────────────────────────────────────
    with st.expander("Backtest Wave Accuracy", expanded=False):
        st.markdown(
            "Run a historical simulation to test how often Elliott Wave signals "
            "(Waves 3 & 5) hit their Fibonacci targets before invalidation."
        )
        
        # Controls
        b_col1, b_col2 = st.columns([1, 3])
        with b_col1:
            test_degree = st.selectbox(
                "Degree to Test",
                options=["Primary", "Intermediate", "Minor"],
                index=0
            )
        
        if st.button(f"Run Backtest ({test_degree})"):
            with st.spinner(f"Simulating trades..."):
                test_len = len(close) - 100 # Leave some buffer
                if test_len < 100:
                    st.warning("Insufficient history for backtest.")
                else:
                    results = backtest_wave_counts(close, degree=test_degree, test_period_bars=test_len)
            
                    if not results:
                        st.warning("No completed wave setups found.")
                    else:
                        # Metrics
                        hits = sum(1 for r in results if r.outcome == "Target Hit")
                        invalid = sum(1 for r in results if r.outcome == "Invalidated")
                        total = hits + invalid # Ignore open trades for win rate calculation? Or include?
                        # Let's include only closed trades for win rate
                        win_rate = (hits / total * 100) if total > 0 else 0
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Signals", len(results))
                        m2.metric("Win Rate", f"{win_rate:.1f}%")
                        m3.metric("Hits", hits)
                        m4.metric("Stops", invalid)
                        
                        # Detailed Table
                        st.markdown("### Trade Log")
                        
                        # Convert to DataFrame for display
                        data = []
                        for r in results:
                            pnl = f"{r.pnl_pct:+.1f}%" if r.pnl_pct is not None else "—"
                            outcome_emoji = {
                                "Target Hit": "✅ Hit",
                                "Invalidated": "❌ Stop",
                                "Indeterminate": "⏳ Open"
                            }.get(r.outcome, r.outcome)
                            
                            data.append({
                                "Date": r.date.strftime("%Y-%m-%d"),
                                "Wave": r.wave_label,
                                "Dir": r.direction,
                                "Entry": f"${r.entry_price:.2f}",
                                "Target": f"${r.target_price:.2f}",
                                "Stop": f"${r.invalidation_price:.2f}",
                                "Outcome": outcome_emoji,
                                "PnL": pnl
                            })
                            
                        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
