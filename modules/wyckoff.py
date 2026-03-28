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
GROQ_MODEL = "llama-3.3-70b-versatile"

_PERIOD_MAP: dict[str, list[str]] = {
    "5m":  ["5d", "10d", "20d", "30d", "60d"],
    "15m": ["5d",  "10d", "20d", "30d", "45d", "60d"],
    "30m": ["10d", "20d", "30d", "45d", "60d"],
    "1h":  ["30d", "60d", "90d", "180d", "365d", "730d"],
    "1d":  ["6mo", "1y",  "2y",  "5y"],
    "1wk": ["2y",  "5y",  "10y", "max"],
    "1mo": ["3y",  "5y",  "10y", "max"],
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

# Maps each interval to lower timeframes for full-spectrum MTF view (ascending order)
_MTF_LOWER: dict[str, tuple[str, ...]] = {
    "15m": ("5m",),
    "30m": ("5m",),
    "1h":  ("5m", "30m"),
    "1d":  ("30m", "1h"),
    "1wk": ("1h",  "1d"),
    "1mo": ("1d",  "1wk"),
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
def _build_claude_narrative(prompt: str, model: str) -> str:
    """Call xAI or Claude to generate a narrative."""
    if model and model.startswith("grok-"):
        _xai_key = os.getenv("XAI_API_KEY", "")
        if not _xai_key:
            return "_XAI_API_KEY not set — Grok narrative unavailable._"
        try:
            from services.claude_client import _call_xai
            return _call_xai([{"role": "user", "content": prompt}], model, 500, 0.3)
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
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"_Claude narrative failed: {e}_"


@st.cache_data(ttl=3600)
def _claude_wyckoff_analysis(
    ticker: str,
    close_t: tuple,
    high_t: tuple,
    low_t: tuple,
    volume_t: tuple,
    dates_t: tuple,
    algo_phase: str,
    algo_sub_phase: str,
    algo_confidence: int,
    algo_support: float,
    algo_resistance: float,
    algo_cause_target: float,
    event_strs: tuple,
    interval: str,
    model: str,
) -> dict:
    """Ask Claude or xAI to independently detect Wyckoff phase and return structured JSON."""
    import json
    _is_grok = model and model.startswith("grok-")
    if _is_grok:
        api_key = os.getenv("XAI_API_KEY", "")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}
    n = min(30, len(close_t))
    ohlcv_lines = [
        f"{dates_t[-(n-i)]}: H={high_t[-(n-i)]:.2f} L={low_t[-(n-i)]:.2f} C={close_t[-(n-i)]:.2f} V={volume_t[-(n-i)]:.0f}"
        for i in range(n)
    ]
    ohlcv_text = "\n".join(ohlcv_lines)
    events_text = "\n".join(f"  - {e}" for e in event_strs) if event_strs else "  None"
    prompt = f"""You are an expert Wyckoff Method analyst performing independent phase detection on {ticker} ({interval} timeframe).

OHLCV + Volume Data (last {n} bars, oldest first):
{ohlcv_text}

Algorithm preliminary detection (use as reference only):
- Phase: {algo_phase}
- Sub-phase: {algo_sub_phase}
- Confidence: {algo_confidence}/100
- Support: ${algo_support:.2f}
- Resistance: ${algo_resistance:.2f}
- Cause & Effect Target: ${algo_cause_target:.2f}
- Detected events:
{events_text}

Independently evaluate the Wyckoff phase using price, volume, and spread analysis.
Return ONLY valid JSON, no other text:
{{
  "phase": "Accumulation or Distribution or Markup or Markdown",
  "sub_phase": "A, B, C, D, or E",
  "confidence": integer 0-100,
  "support": price level as float,
  "resistance": price level as float,
  "cause_target": price target as float based on Wyckoff Point & Figure cause,
  "next_expected": "1-sentence description of what should happen next",
  "rationale": "1-2 sentence explanation referencing specific price/volume evidence"
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
    raw_ticker = st.text_input(
        "Ticker Symbol",
        value=st.session_state["wy_ticker"],
        placeholder="SPY · GC=F · RY.TO · BTC-USD · ^FTSE · ZN=F",
        key="wy_ticker_input",
        help=(
            "Any yfinance-compatible symbol. "
            "Append .TO for TSX, =F for futures, -USD for crypto, ^ for indices. "
            "Note: some index symbols (^) have no volume — VSA bars will be skipped."
        ),
    )

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
                        st.session_state["wy_ticker_input"] = _t
                        st.rerun()

    if st.button("Refresh Data", key="wy_refresh_data"):
        st.cache_data.clear()
        st.rerun()

    # ── AI Engine tier selector (top-level) ───────────────────────────────────
    _has_xai_wy = bool(os.getenv("XAI_API_KEY"))
    _has_anthropic_wy = bool(os.getenv("ANTHROPIC_API_KEY"))
    _wy_tier_options_top = ["⚡ Standard"] + (["🧠 Regard Mode"] if _has_xai_wy else []) + (["👑 Highly Regarded Mode"] if _has_anthropic_wy else [])
    _wy_default_idx = 0
    _wy_saved = st.session_state.get("wy_narrative_tier", "⚡ Standard")
    if _wy_saved in _wy_tier_options_top:
        _wy_default_idx = _wy_tier_options_top.index(_wy_saved)
    st.radio(
        "AI Engine",
        _wy_tier_options_top,
        index=_wy_default_idx,
        horizontal=True,
        key="wy_narrative_tier",
        help="Standard = Groq LLaMA  ·  Regard = Grok 4.1 (overrides phase)  ·  Highly Regarded = Claude Sonnet",
    )
    st.markdown(
        '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
        'margin-top:-10px;margin-bottom:2px;">'
        '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast-reasoning &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
        '</div>',
        unsafe_allow_html=True,
    )

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

    # ── MTF Snapshot (automatic — no timeframe selection needed) ──────────────
    _ALL_TFS = ["5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
    _TF_LABELS = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
                  "1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}
    _pc = lambda p: {"Accumulation": COLORS["green"], "Distribution": COLORS["red"],
                     "Markup": COLORS["blue"], "Markdown": COLORS["orange"]}.get(p, COLORS["text_dim"])

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
        f'color:{COLORS["text_dim"]};margin:4px 0 6px 0;text-transform:uppercase;letter-spacing:0.06em;">'
        f'Wyckoff Phase — All Timeframes (auto-detected)</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Detecting Wyckoff phases across all timeframes…"):
        _mtf_snap = {tf: _fetch_mtf_phase(ticker, tf) for tf in _ALL_TFS}
    _snap_cols = st.columns(len(_ALL_TFS))
    for _i, _tf in enumerate(_ALL_TFS):
        _ph = _mtf_snap[_tf]
        _snap_cols[_i].markdown(bloomberg_metric(_TF_LABELS[_tf], _ph, _pc(_ph)), unsafe_allow_html=True)
    _valid_snap = [p for p in _mtf_snap.values() if p not in ("—", "")]
    _bullish_set = {"Accumulation", "Markup"}
    _bearish_set = {"Distribution", "Markdown"}
    if _valid_snap:
        if all(p in _bullish_set for p in _valid_snap):
            st.success("✅ Bullish confluence across ALL timeframes — high-conviction long bias.")
        elif all(p in _bearish_set for p in _valid_snap):
            st.error("🔴 Bearish confluence across ALL timeframes — high-conviction short bias.")
        elif sum(p in _bullish_set for p in _valid_snap) >= len(_valid_snap) * 0.75:
            st.info("ℹ️ Predominantly bullish — majority of timeframes in Accumulation/Markup.")
        elif sum(p in _bearish_set for p in _valid_snap) >= len(_valid_snap) * 0.75:
            st.info("ℹ️ Predominantly bearish — majority of timeframes in Distribution/Markdown.")
        else:
            st.warning("⚠️ Mixed signals across timeframes — wait for higher-TF alignment before committing.")

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;font-size:11px;'
        f'color:{COLORS["text_dim"]};margin:16px 0 6px 0;text-transform:uppercase;letter-spacing:0.06em;">'
        f'Detail Chart — select interval &amp; lookback below</div>',
        unsafe_allow_html=True,
    )
    _iv_col, _lk_col = st.columns([1, 1])
    with _iv_col:
        interval = st.selectbox(
            "Chart Interval",
            ["5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
            index=4,
            key="wy_interval",
            help="Drives the detail chart and trade setup below. The MTF snapshot above always shows all TFs.",
        )
        if interval == "15m":
            st.caption("💡 **15m** works well for intraday structure. ⚠️ Limited to last 60 days by data providers; switch to **1h** for longer history.")
    with _lk_col:
        period_choices = _PERIOD_MAP.get(interval, ["2y"])
        default_idx = len(period_choices) // 2
        period = st.selectbox("Lookback", period_choices, index=default_idx, key="wy_period",
                              help="Amount of history to fetch for the detail chart")

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

    # ── Claude AI Wyckoff override (Regard / Highly Regarded) ─────────────────
    _wy_tier_now = st.session_state.get("wy_narrative_tier", "⚡ Standard")
    _claude_wa = {}
    _wy_xai_ok = bool(os.getenv("XAI_API_KEY"))
    _wy_ant_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    if _wy_tier_now in ("🧠 Regard Mode", "👑 Highly Regarded Mode") and (_wy_xai_ok or _wy_ant_ok) and analysis is not None:
        _ca_model = "grok-4-1-fast-reasoning" if _wy_tier_now == "🧠 Regard Mode" else "claude-sonnet-4-6"
        _cur = analysis.current_phase
        _event_strs = tuple(f"{e.event_type}: {e.description}" for e in _cur.events)
        with st.spinner(f"✦ AI ({_ca_model.split('-')[0].capitalize()}) analyzing Wyckoff structure..."):
            _claude_wa = _claude_wyckoff_analysis(
                ticker,
                tuple(close.iloc[-60:].round(2).tolist()),
                tuple(high.iloc[-60:].round(2).tolist()),
                tuple(low.iloc[-60:].round(2).tolist()),
                tuple(volume.iloc[-60:].round(0).tolist()),
                tuple(str(d)[:10] for d in ohlcv.index[-60:]),
                _cur.phase,
                _cur.sub_phase or "",
                _cur.confidence,
                _cur.key_levels.get("support", 0.0),
                _cur.key_levels.get("resistance", 0.0),
                _cur.cause_target or 0.0,
                _event_strs,
                interval,
                _ca_model,
            )

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
        fig.update_layout(dragmode="pan")
        apply_dark_layout(fig, title=f"{ticker} — Wyckoff Phase Detection")
        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        })
        return

    # Chart with phase overlays
    fig = _make_wyckoff_chart(ohlcv, analysis, rsi_series, obv_series, ticker)
    fig.update_layout(dragmode="pan")
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    })

    current = analysis.current_phase

    # Use Claude overrides if available, otherwise fall back to algorithm
    _phase = _claude_wa.get("phase", current.phase)
    _sub_phase = _claude_wa.get("sub_phase", current.sub_phase or "")
    _confidence = _claude_wa.get("confidence", current.confidence)
    _support = _claude_wa.get("support", current.key_levels["support"])
    _resistance = _claude_wa.get("resistance", current.key_levels["resistance"])
    _cause_target = _claude_wa.get("cause_target", current.cause_target)
    _next_expected = _claude_wa.get("next_expected")
    _rationale = _claude_wa.get("rationale")

    # Metrics Row
    if _claude_wa:
        st.markdown(
            '<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);color:#fff;'
            'font-size:10px;font-weight:700;letter-spacing:0.06em;padding:2px 10px;'
            'border-radius:3px;margin-bottom:10px;display:inline-block;">'
            '✦ CLAUDE WYCKOFF OVERRIDE</span>',
            unsafe_allow_html=True,
        )

    phase_color = {
        "Accumulation": COLORS["green"], "Distribution": COLORS["red"],
        "Markup": COLORS["blue"], "Markdown": COLORS["orange"],
    }.get(_phase, COLORS["text"])

    _phase_label = f"{_phase} · {_sub_phase}" if _sub_phase else _phase
    m1, m2, m3 = st.columns(3)
    m1.markdown(bloomberg_metric("Current Phase", _phase_label, phase_color), unsafe_allow_html=True)
    m2.markdown(bloomberg_metric("Confidence", f"{_confidence}/100"), unsafe_allow_html=True)
    m3.markdown(
        bloomberg_metric(
            "Key Levels",
            f"S: ${_support:.2f}  R: ${_resistance:.2f}",
        ),
        unsafe_allow_html=True,
    )

    if _claude_wa:
        if _next_expected:
            st.markdown(
                f'<div style="margin:8px 0 2px 0;font-size:13px;color:#C8D8E8;">⤷ {_next_expected}</div>',
                unsafe_allow_html=True,
            )
        if _rationale:
            st.markdown(
                f'<div style="font-size:13px;color:#C8D8E8;margin-top:4px;">💬 {_rationale}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="font-size:11px;color:#5A7A8A;margin-top:4px;">'
            f'Algorithm: {current.phase} · {current.confidence}% · '
            f'S: ${current.key_levels["support"]:.2f}  R: ${current.key_levels["resistance"]:.2f}</div>',
            unsafe_allow_html=True,
        )

    support = _support
    resistance = _resistance
    current_price_val = float(close.iloc[-1])

    if _cause_target:
        st.markdown("### 🎯 Trade Setup")
        is_long = _phase in ("Accumulation", "Markup")
        setup_color = COLORS["green"] if is_long else COLORS["red"]
        direction_label = "LONG" if is_long else "SHORT"

        if is_long:
            entry_zone_low = support * 1.005
            entry_zone_high = support * 1.02
            stop = support * 0.985
            target = _cause_target
        else:
            entry_zone_low = resistance * 0.98
            entry_zone_high = resistance * 0.995
            stop = resistance * 1.015
            target = _cause_target

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
        _has_xai = bool(os.getenv("XAI_API_KEY"))
        _has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        _wy_tier_map = {
            "⚡ Standard":            (False, None),
            "🧠 Regard Mode":         (True,  "grok-4-1-fast-reasoning"),
            "👑 Highly Regarded Mode": (True,  "claude-sonnet-4-6"),
        }
        _wy_tier = st.session_state.get("wy_narrative_tier", "⚡ Standard")
        st.caption(f"Engine: {_wy_tier} — change at top of page")
        _wy_use_claude, _wy_model = _wy_tier_map.get(_wy_tier, (False, None))
        if not _has_anthropic and _wy_tier != "⚡ Standard":
            st.caption("Set XAI_API_KEY to unlock Regard Mode · ANTHROPIC_API_KEY for Highly Regarded.")

        current_price = float(close.iloc[-1])
        event_strs = tuple(f"{e.event_type}: {e.description}" for e in current.events)

        # Build shared prompt
        events_text = "\n".join(f"  - {e}" for e in event_strs) if event_strs else "  None detected"
        dist_s = current_price - current.key_levels["support"]
        dist_r = current.key_levels["resistance"] - current_price
        _wy_prompt = f"""You are an expert Wyckoff Method analyst. Interpret the following automated phase detection for {ticker} on the {interval} timeframe and write a concise 3-5 sentence market commentary.

Current Wyckoff Phase:
- Phase: {current.phase}
- Confidence: {current.confidence}/100
- Support Level: ${current.key_levels["support"]:.2f}
- Resistance Level: ${current.key_levels["resistance"]:.2f}
- Current {ticker} Price: ${current_price:.2f}
- Distance to Support: ${dist_s:+.2f}
- Distance to Resistance: ${dist_r:+.2f}
- Detected Events:
{events_text}

Write your commentary covering:
1. What the current Wyckoff phase means for market structure
2. Key events detected and their significance
3. What the Wyckoff Method predicts should happen next
4. Critical support/resistance levels to watch

Be direct and specific. Do not hedge excessively. Do not repeat the input data verbatim."""

        try:
            if _wy_use_claude and (_has_xai or _has_anthropic):
                narrative = _build_claude_narrative(_wy_prompt, _wy_model)
                _model_label = _wy_model
            else:
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
                _model_label = GROQ_MODEL

            if narrative.startswith("_Narrative generation failed") or narrative.startswith("_GROQ") or narrative.startswith("_Claude") or narrative.startswith("_ANTHROPIC"):
                st.warning("AI narrative unavailable.")
                if st.button("Retry Narrative", key="wy_retry_narrative"):
                    _build_groq_narrative.clear()
                    st.rerun()
            else:
                if _wy_use_claude and (_has_xai or _has_anthropic):
                    st.markdown(
                        '<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);'
                        'color:#fff;font-size:10px;font-weight:700;letter-spacing:0.06em;'
                        'padding:2px 8px;border-radius:3px;margin-bottom:8px;display:inline-block;">'
                        '✦ POWERED BY CLAUDE</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(narrative)
                st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Model: {_model_label}")
        except Exception:
            st.warning("AI narrative unavailable.")
            if st.button("Retry Narrative", key="wy_retry_narrative_err"):
                st.cache_data.clear()
                st.rerun()
