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
import os
from io import StringIO
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


@st.cache_data(ttl=21600)
def fetch_fred_series(series_id: str) -> pd.Series | None:
    """Fetch a single FRED series as a pandas Series indexed by date."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fred_cache")
    cache_file = os.path.join(cache_dir, f"{series_id}.csv")

    def _parse_csv_text(csv_text: str) -> pd.Series | None:
        df = pd.read_csv(StringIO(csv_text))
        if "DATE" not in df.columns or series_id not in df.columns:
            return None

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna(subset=["DATE", series_id])
        if df.empty:
            return None

        series = pd.Series(df[series_id].values, index=df["DATE"], name=series_id)
        return series.sort_index()

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    for timeout in (12, 25):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            parsed = _parse_csv_text(resp.text)
            if parsed is not None:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                return parsed
        except Exception:
            continue

    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_text = f.read()
            return _parse_csv_text(cached_text)
    except Exception:
        pass

    return None


@st.cache_data(ttl=10800)
def fetch_options_chain_snapshot(ticker: str = "SPY", max_expiries: int = 3) -> dict | None:
    """
    Fetch options chain snapshot and build strike-level aggregates.

    Returns:
        {
            "ticker": str,
            "price": float,
            "asof": str,
            "strikes": [float],
            "call_oi": [float],
            "put_oi": [float],
            "net_gamma_proxy": [float],
            "expiries": [str],
        }
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])
        expiries = list(tk.options or [])
        if not expiries:
            return None

        selected = expiries[:max(1, min(max_expiries, len(expiries)))]
        strike_map: dict[float, dict[str, float]] = {}

        for exp in selected:
            chain = tk.option_chain(exp)
            for side, sign in ((chain.calls, 1.0), (chain.puts, -1.0)):
                if side is None or side.empty:
                    continue
                subset = side[["strike", "openInterest", "impliedVolatility"]].copy()
                subset["openInterest"] = pd.to_numeric(subset["openInterest"], errors="coerce").fillna(0)
                subset["impliedVolatility"] = pd.to_numeric(subset["impliedVolatility"], errors="coerce").fillna(0.2)

                for _, row in subset.iterrows():
                    strike = float(row["strike"])
                    oi = float(row["openInterest"])
                    iv = float(row["impliedVolatility"])

                    if strike not in strike_map:
                        strike_map[strike] = {"call_oi": 0.0, "put_oi": 0.0, "net_gamma_proxy": 0.0}

                    if sign > 0:
                        strike_map[strike]["call_oi"] += oi
                    else:
                        strike_map[strike]["put_oi"] += oi

                    distance = abs(strike - price) / max(price, 1e-6)
                    weight = np.exp(-((distance / 0.1) ** 2))
                    gamma_proxy = sign * oi * iv * weight
                    strike_map[strike]["net_gamma_proxy"] += gamma_proxy

        if not strike_map:
            return None

        strikes = sorted(strike_map.keys())
        return {
            "ticker": ticker,
            "price": price,
            "asof": str(pd.Timestamp.utcnow()),
            "strikes": strikes,
            "call_oi": [float(strike_map[s]["call_oi"]) for s in strikes],
            "put_oi": [float(strike_map[s]["put_oi"]) for s in strikes],
            "net_gamma_proxy": [float(strike_map[s]["net_gamma_proxy"]) for s in strikes],
            "expiries": selected,
        }
    except Exception:
        return None
