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

_PERIOD_MAP: dict[str, list[str]] = {
    "5m":  ["5d", "10d", "20d", "30d", "60d"],
    "15m": ["5d",  "10d", "20d", "30d", "45d", "60d"],
    "30m": ["10d", "20d", "30d", "45d", "60d"],
    "1h":  ["30d", "60d", "90d", "180d", "365d"],
    "1d":  ["6mo", "1y",  "2y",  "5y"],
    "1wk": ["2y",  "5y",  "10y", "max"],
    "1mo": ["5y",  "10y", "max"],
}

# Maps each interval to the next-higher timeframe for MTF confirmation
_MTF_HIGHER: dict[str, tuple[str, str]] = {
    "5m":  ("15m", "1h"),
    "15m": ("1h",  "1d"),
    "30m": ("1h",  "1d"),
    "1h":  ("1d",  "1wk"),
    "1d":  ("1wk", "1mo"),
    "1wk": ("1mo", "1mo"),
}


@st.cache_data(ttl=3600)
def _fetch_mtf_phase(ticker: str, interval: str) -> str:
    """Return the dominant phase label for a single ticker/interval (for MTF panel)."""
    period_map = {"5m": "30d", "15m": "60d", "30m": "60d", "1h": "365d", "1d": "2y", "1wk": "10y", "1mo": "max"}
    period = period_map.get(interval, "2y")
    from services.wyckoff_engine import analyze_wyckoff as _aw
    try:
        df = fetch_ohlcv_single(ticker, period=period, interval=interval)
        if df is None or df.empty:
            return "—"
        c = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        h = df["High"].iloc[:,  0] if isinstance(df["High"],  pd.DataFrame) else df["High"]
        lo= df["Low"].iloc[:,  0] if isinstance(df["Low"],   pd.DataFrame) else df["Low"]
        v = df["Volume"].iloc[:,0] if isinstance(df["Volume"],pd.DataFrame) else df["Volume"]
        result = _aw(c.dropna(), h.dropna(), lo.dropna(), v.dropna(), interval=interval)
        return result.current_phase.phase if result else "—"
    except Exception:
        return "—"


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
def _fetch_wyckoff_data(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    """Fetch OHLCV for any ticker/period/interval."""
    return fetch_ohlcv_single(ticker, period=period, interval=interval)


@st.cache_data(ttl=3600)
def _build_groq_narrative(
    phase: str,
    confidence: int,
    support: float,
    resistance: float,
    events: tuple,
    current_price: float = 0.0,
    ticker: str = "SPY",
    interval: str = "1d",
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
        price_ctx = f"""- Current {ticker} Price: ${current_price:.2f}
- Distance to Support: ${dist_s:+.2f}
- Distance to Resistance: ${dist_r:+.2f}
"""

    prompt = f"""You are an expert Wyckoff Method analyst. Interpret the following automated phase detection for {ticker} on the {interval} timeframe and write a concise 3-5 sentence market commentary.

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

    # ── Session state ticker (same pattern as Elliott Wave) ───────────────────
    if "wy_ticker" not in st.session_state:
        st.session_state["wy_ticker"] = ""

    # Human-readable labels for quick-pick button tooltips
    _TICKER_LABELS: dict[str, str] = {
        "ES=F": "ES=F  S&P 500",   "NQ=F": "NQ=F  Nasdaq",
        "YM=F": "YM=F  Dow",       "RTY=F": "RTY=F  Russell",
        "GC=F": "GC=F  Gold",      "SI=F":  "SI=F  Silver",
        "CL=F": "CL=F  Crude Oil", "NG=F":  "NG=F  Nat Gas",
        "HG=F": "HG=F  Copper",    "PL=F":  "PL=F  Platinum",
        "PA=F": "PA=F  Palladium", "ZC=F":  "ZC=F  Corn",
        "ZW=F": "ZW=F  Wheat",     "ZS=F":  "ZS=F  Soybeans",
        "CC=F": "CC=F  Cocoa",     "KC=F":  "KC=F  Coffee",
        "LBS=F":"LBS=F Lumber",
        "ZB=F": "ZB=F  30yr T-Bond","ZN=F": "ZN=F  10yr T-Note",
        "ZF=F": "ZF=F  5yr T-Note", "ZT=F": "ZT=F  2yr T-Note",
        "^TNX": "^TNX  10yr Yield","^TYX":  "^TYX  30yr Yield",
        "^GSPTSE": "^GSPTSE  TSX",
        "EURUSD=X":"EUR/USD","GBPUSD=X":"GBP/USD",
        "USDJPY=X":"USD/JPY","USDCAD=X":"USD/CAD",
        "AUDUSD=X":"AUD/USD","USDCHF=X":"USD/CHF",
        "BTC-USD":"BTC-USD  Bitcoin",   "ETH-USD":"ETH-USD  Ethereum",
        "SOL-USD":"SOL-USD  Solana",    "XRP-USD":"XRP-USD  XRP/Ripple",
        "BNB-USD":"BNB-USD  BNB",       "DOGE-USD":"DOGE-USD  Dogecoin",
        "^N225":     "^N225    Nikkei 225 (Japan)",
        "000001.SS": "000001.SS  Shanghai Composite (China)",
        "^HSI":      "^HSI     Hang Seng (Hong Kong)",
        "^NSEI":     "^NSEI    Nifty 50 (India)",
        "^BSESN":    "^BSESN   Sensex (India)",
        "^AXJO":     "^AXJO    ASX 200 (Australia)",
        "^FTSE":     "^FTSE    FTSE 100 (UK)",
        "^GDAXI":    "^GDAXI   DAX (Germany)",
        "^FCHI":     "^FCHI    CAC 40 (France)",
        "^STOXX50E": "^STOXX50E  Euro Stoxx 50",
        "^KS11":     "^KS11    KOSPI (South Korea)",
        "^TWII":     "^TWII    Taiwan Weighted",
        "^MXX":      "^MXX     IPC (Mexico)",
        "^BVSP":     "^BVSP    Bovespa (Brazil)",
        "^JKSE":     "^JKSE    IDX Composite (Indonesia)",
    }

    # Full name + asset class shown in the ticker info bar below input
    _TICKER_DESC: dict[str, tuple[str, str]] = {
        "SPY":  ("SPDR S&P 500 ETF",           "US Large-Cap Equity ETF"),
        "QQQ":  ("Invesco Nasdaq-100 ETF",      "US Tech/Growth Equity ETF"),
        "IWM":  ("iShares Russell 2000 ETF",    "US Small-Cap Equity ETF"),
        "DIA":  ("SPDR Dow Jones ETF",          "US Blue-Chip Equity ETF"),
        "^SPX": ("S&P 500 Index",               "US Large-Cap Index"),
        "^NDX": ("Nasdaq-100 Index",            "US Tech Index"),
        "^DJI": ("Dow Jones Industrial Avg",    "US Blue-Chip Index"),
        "AAPL": ("Apple Inc.",                  "Technology — Consumer Electronics"),
        "MSFT": ("Microsoft Corp.",             "Technology — Cloud & Software"),
        "NVDA": ("NVIDIA Corp.",                "Technology — Semiconductors / AI"),
        "TSLA": ("Tesla Inc.",                  "Automotive / Clean Energy"),
        "AMZN": ("Amazon.com Inc.",             "E-Commerce & Cloud (AWS)"),
        "META": ("Meta Platforms Inc.",         "Social Media / VR"),
        "GOOGL":("Alphabet Inc.",               "Technology — Search & Advertising"),
        "GOOG": ("Alphabet Inc. (C Shares)",    "Technology — Search & Advertising"),
        "BRK-B":("Berkshire Hathaway B",        "Conglomerate / Financial"),
        "JPM":  ("JPMorgan Chase & Co.",        "Banking & Financial Services"),
        "GS":   ("Goldman Sachs Group",         "Investment Banking"),
        "XOM":  ("Exxon Mobil Corp.",           "Energy — Integrated Oil & Gas"),
        "^GSPTSE":("S&P/TSX Composite Index",   "Canadian Broad Market Index"),
        "XIU.TO":("iShares S&P/TSX 60 ETF",    "Canadian Large-Cap ETF"),
        "RY.TO": ("Royal Bank of Canada",       "Canadian Banking"),
        "TD.TO": ("TD Bank Group",              "Canadian Banking"),
        "ENB.TO":("Enbridge Inc.",              "Canadian Energy — Pipelines"),
        "SHOP.TO":("Shopify Inc.",              "E-Commerce Platform"),
        "CNR.TO":("Canadian Nat. Railway",      "Transportation — Rail"),
        "ABX.TO":("Barrick Gold Corp.",         "Gold Mining"),
        "ES=F":  ("E-mini S&P 500 Futures",     "US Large-Cap Equity Futures"),
        "NQ=F":  ("E-mini Nasdaq-100 Futures",  "US Tech Equity Futures"),
        "YM=F":  ("E-mini Dow Futures",         "US Blue-Chip Equity Futures"),
        "RTY=F": ("E-mini Russell 2000 Futures","US Small-Cap Equity Futures"),
        "GC=F":  ("Gold Futures",               "Precious Metal — Safe Haven"),
        "SI=F":  ("Silver Futures",             "Precious Metal — Industrial Use"),
        "CL=F":  ("WTI Crude Oil Futures",      "Energy — Benchmark Crude"),
        "NG=F":  ("Natural Gas Futures",        "Energy — Utility/Heating Fuel"),
        "HG=F":  ("Copper Futures",             "Industrial Metal — Economic Indicator"),
        "PL=F":  ("Platinum Futures",           "Precious Metal — Industrial/Auto"),
        "PA=F":  ("Palladium Futures",          "Precious Metal — Auto Catalysts"),
        "ZC=F":  ("Corn Futures",               "Agriculture — Feed & Ethanol"),
        "ZW=F":  ("Wheat Futures",              "Agriculture — Food Staple"),
        "ZS=F":  ("Soybean Futures",            "Agriculture — Food & Biofuel"),
        "CC=F":  ("Cocoa Futures",              "Soft Commodity — Food"),
        "KC=F":  ("Coffee Futures",             "Soft Commodity — Beverage"),
        "LBS=F": ("Lumber Futures",             "Building Material — Housing Indicator"),
        "ZB=F":  ("30-Year T-Bond Futures",     "US Government Long Bond"),
        "ZN=F":  ("10-Year T-Note Futures",     "US Government Benchmark Bond"),
        "ZF=F":  ("5-Year T-Note Futures",      "US Government Medium Bond"),
        "ZT=F":  ("2-Year T-Note Futures",      "US Government Short Bond"),
        "TLT":   ("iShares 20+ Yr Treasury ETF","Long-Duration Bond ETF"),
        "IEF":   ("iShares 7-10 Yr Treasury ETF","Medium-Duration Bond ETF"),
        "^TNX":  ("10-Year Treasury Yield",     "US Benchmark Interest Rate"),
        "^TYX":  ("30-Year Treasury Yield",     "US Long-Term Interest Rate"),
        "EURUSD=X":("Euro / US Dollar",         "Forex — Major Pair"),
        "GBPUSD=X":("British Pound / USD",      "Forex — Major Pair"),
        "USDJPY=X":("USD / Japanese Yen",       "Forex — Major Pair"),
        "USDCAD=X":("USD / Canadian Dollar",    "Forex — Major Pair"),
        "AUDUSD=X":("Australian Dollar / USD",  "Forex — Major Pair"),
        "USDCHF=X":("USD / Swiss Franc",        "Forex — Safe-Haven Pair"),
        "BTC-USD": ("Bitcoin",                  "Cryptocurrency — Store of Value"),
        "ETH-USD": ("Ethereum",                 "Cryptocurrency — Smart Contract Platform"),
        "SOL-USD": ("Solana",                   "Cryptocurrency — High-Speed L1"),
        "XRP-USD": ("XRP (Ripple)",             "Cryptocurrency — Payments Network"),
        "BNB-USD": ("BNB (Binance Coin)",       "Cryptocurrency — Exchange Token"),
        "DOGE-USD":("Dogecoin",                 "Cryptocurrency — Meme Coin"),
        "^N225":     ("Nikkei 225",             "Japan — Top 225 Companies"),
        "000001.SS": ("Shanghai Composite",     "China — All SSE-Listed Stocks"),
        "^HSI":      ("Hang Seng Index",        "Hong Kong — Top 50 Companies"),
        "^NSEI":     ("Nifty 50",               "India — NSE Top 50 Companies"),
        "^BSESN":    ("BSE Sensex",             "India — BSE Top 30 Companies"),
        "^AXJO":     ("S&P/ASX 200",            "Australia — Top 200 Companies"),
        "^FTSE":     ("FTSE 100",               "UK — London Stock Exchange Top 100"),
        "^GDAXI":    ("DAX 40",                 "Germany — Frankfurt Top 40"),
        "^FCHI":     ("CAC 40",                 "France — Paris Top 40"),
        "^STOXX50E": ("Euro Stoxx 50",          "Eurozone — Top 50 Blue Chips"),
        "^KS11":     ("KOSPI",                  "South Korea — Broad Market Index"),
        "^TWII":     ("Taiwan Weighted Index",  "Taiwan — TWSE Broad Market"),
        "^MXX":      ("S&P/BMV IPC",            "Mexico — Largest Companies"),
        "^BVSP":     ("Ibovespa",               "Brazil — B3 Exchange Top Stocks"),
        "^JKSE":     ("IDX Composite",          "Indonesia — Jakarta Stock Exchange"),
    }

    # Quick-pick tabs
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
            ("CC=F","Cocoa"),("KC=F","Coffee"),("LBS=F","Lumber"),
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
            ("^N225","Nikkei 225"),("000001.SS","Shanghai"),
            ("^HSI","Hang Seng"),("^NSEI","Nifty 50"),
            ("^BSESN","Sensex"),("^AXJO","ASX 200"),
            ("^FTSE","FTSE 100"),("^GDAXI","DAX"),
            ("^FCHI","CAC 40"),("^STOXX50E","Euro Stoxx 50"),
            ("^KS11","KOSPI"),("^TWII","Taiwan"),
            ("^MXX","IPC Mexico"),("^BVSP","Bovespa"),("^JKSE","IDX"),
        ],
    }

    # ── Controls ──────────────────────────────────────────────────────────────
    st.caption("Phase detection · VSA · Cause & Effect targets · Demand/Supply lines · Groq AI narrative")
    tk_col, iv_col, lk_col = st.columns([3, 1, 1])
    with tk_col:
        raw_ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state["wy_ticker"],
            placeholder="SPY · GC=F · RY.TO · BTC-USD · ^FTSE · ZN=F",
            help=(
                "Any yfinance-compatible symbol. "
                "Append .TO for TSX, =F for futures, -USD for crypto, ^ for indices. "
                "Note: some index symbols (^) have no volume — VSA bars will be skipped."
            ),
        )
    with iv_col:
        interval = st.selectbox(
            "Interval",
            ["5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
            index=4,
            help="'5m'/'15m'/'30m'/'1h' = intraday  ·  '1d' = daily (default)  ·  '1wk'/'1mo' = macro",
        )
    with lk_col:
        period_choices = _PERIOD_MAP.get(interval, ["2y"])
        default_idx = len(period_choices) // 2
        period = st.selectbox("Lookback", period_choices, index=default_idx,
                              help="Amount of history to fetch")

    ticker = raw_ticker.strip().upper() if raw_ticker else ""
    st.session_state["wy_ticker"] = ticker

    with st.expander("⚡ Quick Pick — Asset Classes", expanded=False):
        _qp_tabs = st.tabs(list(_QUICK_PICKS.keys()))
        for _tab, (_cat, _pairs) in zip(_qp_tabs, _QUICK_PICKS.items()):
            with _tab:
                _nc = min(len(_pairs), 5)
                _qcols = st.columns(_nc)
                for _i, (_t, _lbl) in enumerate(_pairs):
                    if _qcols[_i % _nc].button(
                        _lbl, key=f"wy_qp_{_t}", use_container_width=True,
                        help=_TICKER_LABELS.get(_t, _t),
                    ):
                        st.session_state["wy_ticker"] = _t
                        st.rerun()

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # ── Guard: nothing entered ────────────────────────────────────────────────
    if not ticker:
        st.info("Enter a ticker symbol above or pick one from ⚡ Quick Pick to begin analysis.")
        return

    # ── Ticker info bar ───────────────────────────────────────────────────────
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

    with st.spinner(f"Fetching {ticker} {interval} data and analyzing Wyckoff phases..."):
        ohlcv = _fetch_wyckoff_data(ticker, period=period, interval=interval)

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

    analysis = analyze_wyckoff(close, high, low, volume, interval=interval)

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

    # ── Multi-Timeframe Confirmation ──────────────────────────────────────────
    if interval in _MTF_HIGHER:
        htf1, htf2 = _MTF_HIGHER[interval]
        with st.spinner("Loading higher-TF context…"):
            htf1_phase = _fetch_mtf_phase(ticker, htf1)
            htf2_phase = _fetch_mtf_phase(ticker, htf2) if htf2 != htf1 else None

        _pc = lambda p: {"Accumulation": COLORS["green"], "Distribution": COLORS["red"],
                         "Markup": COLORS["blue"], "Markdown": COLORS["orange"]}.get(p, COLORS["text_dim"])

        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
            f'color:{COLORS["text_dim"]};margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em;">'
            f'Multi-Timeframe Confirmation</div>',
            unsafe_allow_html=True,
        )
        mtf_cols = st.columns(3)
        mtf_cols[0].markdown(bloomberg_metric(f"{interval.upper()} (selected)", current.phase, phase_color), unsafe_allow_html=True)
        mtf_cols[1].markdown(bloomberg_metric(f"{htf1.upper()}", htf1_phase, _pc(htf1_phase)), unsafe_allow_html=True)
        if htf2_phase is not None:
            mtf_cols[2].markdown(bloomberg_metric(f"{htf2.upper()}", htf2_phase, _pc(htf2_phase)), unsafe_allow_html=True)

        # Confluence note
        phases_seen = {current.phase, htf1_phase} | ({htf2_phase} if htf2_phase else set())
        if all(p in ("Accumulation", "Markup") for p in phases_seen if p not in ("—", "")):
            st.success("✅ Bullish confluence across all timeframes — strengthens long bias.")
        elif all(p in ("Distribution", "Markdown") for p in phases_seen if p not in ("—", "")):
            st.error("🔴 Bearish confluence across all timeframes — strengthens short bias.")
        elif current.phase in ("Accumulation", "Markup") and htf1_phase in ("Markup", "Accumulation"):
            st.info("ℹ️ Selected TF bullish, higher TF agrees — trade in direction of trend.")
        elif current.phase in ("Distribution", "Markdown") and htf1_phase in ("Markdown", "Distribution"):
            st.info("ℹ️ Selected TF bearish, higher TF agrees — trade in direction of trend.")
        else:
            st.warning("⚠️ Timeframe divergence — wait for alignment before committing.")


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
                ticker=ticker,
                interval=interval,
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
