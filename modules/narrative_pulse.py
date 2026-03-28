import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout

TIMEFRAME_MAP = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "YTD": "ytd",
}

# yfinance interval for each candle size; 4H requires resampling from 1h
INTERVAL_MAP = {
    "4H": "1h",
    "Daily": "1d",
    "Weekly": "1wk",
    "Monthly": "1mo",
}

# yfinance limits intraday data to ~730 days; 4H needs 1h data which is capped at 730 days
_MAX_PERIOD_FOR_4H = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "YTD": "ytd"}


def render():
    st.header("PRICE ACTION")

    ticker = get_ticker()
    if not ticker:
        st.info("Set an active ticker in Discovery to view price action.")
        return

    col_tf, col_iv = st.columns(2)
    with col_tf:
        timeframe = st.radio(
            "Timeframe",
            list(TIMEFRAME_MAP.keys()),
            index=2,
            horizontal=True,
            key="price_action_tf",
        )
    with col_iv:
        interval = st.radio(
            "Candle Interval",
            list(INTERVAL_MAP.keys()),
            index=1,
            horizontal=True,
            key="price_action_iv",
        )

    # 4H uses 1h data capped at 730 days by yfinance
    yf_interval = INTERVAL_MAP[interval]
    yf_period = TIMEFRAME_MAP[timeframe]

    with st.spinner("Fetching price data..."):
        df = _get_price_data(ticker, yf_period, yf_interval)

    from datetime import datetime
    st.caption(f"LAST UPDATE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CACHE TTL 15M")

    # Resample 1h → 4H if needed
    if df is not None and not df.empty and interval == "4H":
        df = _resample_4h(df)

    if df is None or df.empty:
        st.warning("No price data available for this ticker.")
        return

    # --- Metrics ---
    _render_metrics(df, ticker)

    # --- Candlestick + Volume ---
    _render_candlestick(df, ticker)

    # --- RSI ---
    _render_rsi(df, ticker)

    # --- Persist price momentum signal for downstream use ---
    if len(df) >= 14:
        _rsi_vals = _calc_rsi(df["Close"], 14)
        _cur_rsi = _rsi_vals.iloc[-1]
        _rsi_lbl = "OVERBOUGHT" if _cur_rsi > 70 else ("OVERSOLD" if _cur_rsi < 30 else "NEUTRAL")

        _price_now = df["Close"].iloc[-1]
        _ma_pos = {}
        for _p in [20, 50, 200]:
            if len(df) >= _p:
                _ma_val = df["Close"].rolling(_p).mean().iloc[-1]
                _ma_pos[f"sma_{_p}"] = {"value": round(_ma_val, 2), "above": bool(_price_now > _ma_val)}

        _avg_vol = df["Volume"].mean()
        _last_vol = df["Volume"].iloc[-1]
        _vol_ratio = round(_last_vol / _avg_vol, 2) if _avg_vol > 0 else 1.0

        # MA trend summary: count how many MAs price is above
        _above_count = sum(1 for v in _ma_pos.values() if v["above"])
        _ma_trend = "STRONG UPTREND" if _above_count == 3 else (
            "UPTREND" if _above_count == 2 else (
            "DOWNTREND" if _above_count == 1 else (
            "STRONG DOWNTREND" if _above_count == 0 and _ma_pos else "NEUTRAL")))

        st.session_state["_price_momentum"] = {
            "ticker": ticker,
            "rsi": round(_cur_rsi, 1),
            "rsi_label": _rsi_lbl,
            "ma_trend": _ma_trend,
            "ma_signals": _ma_pos,
            "vol_ratio": _vol_ratio,
            "price": round(_price_now, 2),
        }
        try:
            from services.signals_cache import save_signals
            save_signals()
        except Exception:
            pass


def _render_metrics(df: pd.DataFrame, ticker: str):
    """Key price metrics row."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    first = df.iloc[0]

    price = last["Close"]
    daily_chg = (last["Close"] - prev["Close"]) / prev["Close"] * 100
    period_chg = (last["Close"] - first["Close"]) / first["Close"] * 100
    high = df["High"].max()
    low = df["Low"].min()
    avg_vol = df["Volume"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${price:.2f}", f"{daily_chg:+.2f}%")
    c2.metric("Period Change", f"{period_chg:+.1f}%")
    c3.metric("Period High", f"${high:.2f}")
    c4.metric("Period Low", f"${low:.2f}")
    c5.metric("Avg Volume", _fmt_volume(avg_vol))


def _render_candlestick(df: pd.DataFrame, ticker: str):
    """Candlestick chart with MAs and volume subplot."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            increasing_fillcolor=COLORS["green"],
            decreasing_fillcolor=COLORS["red"],
            name="Price",
        ),
        row=1, col=1,
    )

    # Moving averages
    ma_configs = [
        (20, COLORS["blue"], "SMA 20"),
        (50, COLORS["yellow"], "SMA 50"),
        (200, "#AB47BC", "SMA 200"),
    ]
    for period, color, name in ma_configs:
        if len(df) >= period:
            ma = df["Close"].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma, mode="lines",
                    line=dict(color=color, width=1.5),
                    name=name,
                ),
                row=1, col=1,
            )

    # Volume bars
    vol_colors = [
        COLORS["green"] if c >= o else COLORS["red"]
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=vol_colors, opacity=0.5,
            name="Volume", showlegend=False,
        ),
        row=2, col=1,
    )

    apply_dark_layout(
        fig,
        title=f"{ticker} — Price Action",
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis_rangeslider_visible=False,
    )
    fig.update_layout(
        height=550,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
    )
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_rsi(df: pd.DataFrame, ticker: str):
    """RSI chart with overbought/oversold zones."""
    if len(df) < 14:
        return

    rsi = _calc_rsi(df["Close"], 14)
    current_rsi = rsi.iloc[-1]

    if current_rsi > 70:
        rsi_label, rsi_color = "OVERBOUGHT", COLORS["red"]
    elif current_rsi < 30:
        rsi_label, rsi_color = "OVERSOLD", COLORS["green"]
    else:
        rsi_label, rsi_color = "NEUTRAL", COLORS["text_dim"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rsi, mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            name="RSI (14)",
            hovertemplate="%{x|%b %d}<br>RSI: %{y:.1f}<extra></extra>",
        )
    )

    # Overbought / oversold zones
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["red"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["green"], opacity=0.08, line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color=COLORS["red"], line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color=COLORS["green"], line_width=1)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["text_dim"], line_width=1)

    apply_dark_layout(
        fig,
        title=dict(
            text=f"RSI (14): {current_rsi:.1f} — {rsi_label}",
            font=dict(color=rsi_color),
        ),
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    fig.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using exponential moving average."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _fmt_volume(vol: float) -> str:
    if vol >= 1e9:
        return f"{vol / 1e9:.1f}B"
    if vol >= 1e6:
        return f"{vol / 1e6:.1f}M"
    if vol >= 1e3:
        return f"{vol / 1e3:.0f}K"
    return f"{vol:.0f}"


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-hour OHLCV data into 4-hour candles."""
    return df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


@st.cache_data(ttl=900)
def _get_price_data(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None
        return df
    except Exception:
        return None
