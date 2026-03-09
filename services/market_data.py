"""
Shared market data fetching layer.

Uses yfinance as primary provider with caching.
Other modules should import from here instead of calling yfinance directly.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class AssetSnapshot:
    """Standardized snapshot for a single asset."""
    ticker: str
    label: str
    latest_price: float | None = None
    pct_change_30d: float | None = None
    series: pd.Series | None = None  # close price series for sparklines
    stale: bool = False  # True if data may be outdated (weekend/holiday)


def _fetch_single(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV via yfinance for a single ticker."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            # yfinance returns MultiIndex columns for single ticker downloads
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            return df
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def fetch_batch(tickers: dict[str, str], period: str = "1y", interval: str = "1d") -> dict[str, AssetSnapshot]:
    """
    Fetch multiple tickers concurrently.

    Args:
        tickers: {ticker_symbol: human_label} e.g. {"SPY": "S&P 500"}
        period: yfinance period string
        interval: yfinance interval string

    Returns:
        {ticker: AssetSnapshot}
    """
    results: dict[str, AssetSnapshot] = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_single, t, period, interval): (t, label)
            for t, label in tickers.items()
        }
        for future in as_completed(futures):
            ticker, label = futures[future]
            df = future.result()
            snap = AssetSnapshot(ticker=ticker, label=label)

            if df is not None and not df.empty and "Close" in df.columns:
                close = df["Close"].squeeze()  # ensure Series, not DataFrame
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                snap.latest_price = float(close.iloc[-1])
                snap.series = close.copy()

                # 30-day momentum (approx 22 trading days)
                if len(close) >= 22:
                    snap.pct_change_30d = float((close.iloc[-1] / close.iloc[-22] - 1) * 100)

                # Staleness check: if last data point is >2 days old
                last_date = df.index[-1]
                if hasattr(last_date, 'tz_localize'):
                    pass  # already tz-aware or naive, fine either way
                days_old = (pd.Timestamp.now(tz=last_date.tzinfo if hasattr(last_date, 'tzinfo') and last_date.tzinfo else None) - last_date).days
                snap.stale = days_old > 3

            results[ticker] = snap

    return results


def zscore(series: pd.Series, lookback: int = 252) -> float | None:
    """Z-score of latest value vs trailing lookback window."""
    if series is None or len(series) < max(lookback, 20):
        return None
    window = series.iloc[-lookback:]
    std = window.std()
    if std == 0:
        return 0.0
    return float((series.iloc[-1] - window.mean()) / std)


def ratio_latest(snaps: dict[str, AssetSnapshot], t1: str, t2: str) -> float | None:
    """Ratio of latest prices for two tickers."""
    s1, s2 = snaps.get(t1), snaps.get(t2)
    if s1 and s2 and s1.latest_price and s2.latest_price and s2.latest_price != 0:
        return s1.latest_price / s2.latest_price
    return None


@st.cache_data(ttl=3600)
def fetch_truflation() -> dict | None:
    """Fetch Truflation current inflation from their public API."""
    try:
        r = requests.get(
            "https://api.truflation.com/inflation",
            timeout=8,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
