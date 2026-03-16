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
from dataclasses import dataclass, field


class _FredFetchError(Exception):
    """Raised when a FRED series fetch fails, preventing st.cache_data from caching None."""
    pass


class _BatchFetchError(Exception):
    """Raised when all tickers fail to fetch, preventing st.cache_data from caching empty snapshots."""
    pass


class _OptionsFetchError(Exception):
    """Raised when options chain fetch fails, preventing st.cache_data from caching None."""
    pass


@dataclass
class AssetSnapshot:
    """Standardized snapshot for a single asset."""
    ticker: str
    label: str
    latest_price: float | None = None
    pct_change_1d: float | None = None
    pct_change_5d: float | None = None
    pct_change_30d: float | None = None
    pct_change_ytd: float | None = None
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


@st.cache_data(ttl=14400)
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
    ticker_list = list(tickers.keys())

    try:
        raw = yf.download(ticker_list, period=period, interval=interval,
                          progress=False, auto_adjust=True, threads=True)
    except Exception:
        raw = pd.DataFrame()

    for ticker, label in tickers.items():
        snap = AssetSnapshot(ticker=ticker, label=label)

        try:
            if raw is not None and not raw.empty:
                # Single ticker: flat columns; multiple tickers: MultiIndex (Price, Ticker)
                if len(ticker_list) == 1:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close = raw["Close"].iloc[:, 0].dropna()
                    else:
                        close = raw["Close"].dropna()
                else:
                    close = raw["Close"][ticker].dropna()

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                if len(close) > 0:
                    snap.latest_price = float(close.iloc[-1])
                    snap.series = close.copy()

                    if len(close) >= 2:
                        snap.pct_change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
                    if len(close) >= 6:
                        snap.pct_change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
                    if len(close) >= 22:
                        snap.pct_change_30d = float((close.iloc[-1] / close.iloc[-22] - 1) * 100)

                    year_start = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
                    ytd_data = close[close.index >= year_start]
                    if len(ytd_data) >= 2:
                        snap.pct_change_ytd = float((close.iloc[-1] / ytd_data.iloc[0] - 1) * 100)

                    last_date = close.index[-1]
                    days_old = (pd.Timestamp.now(tz=last_date.tzinfo if hasattr(last_date, 'tzinfo') and last_date.tzinfo else None) - last_date).days
                    snap.stale = days_old > 3
        except Exception:
            pass

        results[ticker] = snap

    if not any(s.latest_price is not None for s in results.values()):
        raise _BatchFetchError("All tickers failed to fetch")
    return results


def fetch_batch_safe(tickers: dict[str, str], period: str = "1y", interval: str = "1d") -> dict[str, AssetSnapshot]:
    """Wrapper around fetch_batch that returns empty snapshots on total failure instead of raising.

    Use this at call sites so that total failures aren't cached by st.cache_data.
    """
    try:
        return fetch_batch(tickers, period, interval)
    except _BatchFetchError:
        return {t: AssetSnapshot(ticker=t, label=label) for t, label in tickers.items()}


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


@st.cache_data(ttl=43200)
def fetch_fred_series(series_id: str) -> pd.Series:
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

    for timeout in (8, 15):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "NarrativeInvestingTool/1.0"})
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
            result = _parse_csv_text(cached_text)
            if result is not None:
                return result
    except Exception:
        pass

    raise _FredFetchError(f"Failed to fetch FRED series {series_id}")


def fetch_fred_series_safe(series_id: str) -> pd.Series | None:
    """Wrapper around fetch_fred_series that returns None on failure instead of raising.

    Use this at call sites so that transient failures aren't cached by st.cache_data.
    """
    try:
        return fetch_fred_series(series_id)
    except _FredFetchError:
        return None


def warm_fred_cache(series_ids: list[str]):
    """Pre-fetch FRED series in parallel to warm disk cache.

    Call at app startup so the dashboard build doesn't serialize FRED fetches.
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(fetch_fred_series_safe, series_ids))


@st.cache_data(ttl=14400)
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
            raise _OptionsFetchError(f"No price history for {ticker}")

        price = float(hist["Close"].iloc[-1])
        expiries = list(tk.options or [])
        if not expiries:
            raise _OptionsFetchError(f"No options expiries for {ticker}")

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
            raise _OptionsFetchError(f"No strike data for {ticker}")

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
    except _OptionsFetchError:
        raise
    except Exception as e:
        raise _OptionsFetchError(f"Options fetch failed for {ticker}: {e}")


def fetch_options_chain_snapshot_safe(ticker: str = "SPY", max_expiries: int = 3) -> dict | None:
    """Wrapper around fetch_options_chain_snapshot that returns None on failure instead of raising.

    Use this at call sites so that transient failures aren't cached by st.cache_data.
    """
    try:
        return fetch_options_chain_snapshot(ticker, max_expiries)
    except _OptionsFetchError:
        return None
