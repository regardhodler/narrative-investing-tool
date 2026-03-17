import pandas as pd
import requests as _requests
import streamlit as st

# Patch urllib3 Retry to fix pytrends using deprecated 'method_whitelist' kwarg
# (renamed to 'allowed_methods' in urllib3 2.x)
import urllib3.util.retry as _retry_module
_OrigRetry = _retry_module.Retry

class _PatchedRetry(_OrigRetry):
    def __init__(self, *args, **kwargs):
        if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        elif "method_whitelist" in kwargs:
            kwargs.pop("method_whitelist")
        super().__init__(*args, **kwargs)

_retry_module.Retry = _PatchedRetry

from pytrends.request import TrendReq


def _get_pytrends() -> TrendReq:
    return TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=1.0)


@st.cache_data(ttl=900)
def get_yahoo_trending_tickers() -> list[dict]:
    """Fetch trending tickers from Yahoo Finance.

    Returns list of dicts with keys: symbol, name
    """
    url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = _requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes = data["finance"]["result"][0]["quotes"]
        results = []
        for q in quotes:
            symbol = q.get("symbol", "")
            results.append({"symbol": symbol, "name": symbol})
        # Enrich with company names and intraday price change via yfinance
        try:
            import yfinance as yf

            symbols = [r["symbol"] for r in results]
            tickers = yf.Tickers(" ".join(symbols))
            for r in results:
                info = tickers.tickers.get(r["symbol"])
                if info:
                    try:
                        fast = info.fast_info
                        r["name"] = info.info.get("shortName", r["symbol"])
                        r["pct_change"] = round(
                            ((fast.last_price / fast.previous_close) - 1) * 100, 2
                        ) if fast.previous_close else None
                    except Exception:
                        r["pct_change"] = None
        except Exception:
            pass
        # Assign buzz rank (1 = most trending) for star rating
        for idx, r in enumerate(results):
            r["buzz_rank"] = idx + 1
            r.setdefault("pct_change", None)
        return results
    except Exception:
        return []


@st.cache_data(ttl=900)
def get_trending_searches() -> list[str]:
    """Fetch real-time trending searches for the US. Returns list of topic strings."""
    pytrends = _get_pytrends()
    try:
        df = pytrends.trending_searches(pn="united_states")
        return df[0].tolist()
    except (ConnectionError, TimeoutError):
        return []
    except Exception:
        # Fallback: try realtime trending
        try:
            df = pytrends.realtime_trending_searches(pn="US")
            if "title" in df.columns:
                return df["title"].tolist()
            return df.iloc[:, 0].tolist()
        except Exception:
            return []


TIMEFRAME_MAP = {
    "1M": "today 1-m",
    "3M": "today 3-m",
    "1Y": "today 12-m",
}


def _resolve_timeframe(label: str) -> str:
    """Convert a user-friendly timeframe label to a pytrends timeframe string."""
    if label in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[label]

    from datetime import date, timedelta

    today = date.today()

    if label == "6M":
        start = today - timedelta(days=182)
    elif label == "2Y":
        start = today - timedelta(days=730)
    elif label == "YTD":
        start = date(today.year, 1, 1)
    else:
        return "today 3-m"

    return f"{start.strftime('%Y-%m-%d')} {today.strftime('%Y-%m-%d')}"


@st.cache_data(ttl=900)
def get_interest_over_time_multi(keywords: tuple, timeframe: str = "3M") -> pd.DataFrame:
    """Fetch interest over time for up to 5 keywords.

    Returns DataFrame with 'date' column and one column per keyword.
    """
    pytrends = _get_pytrends()
    pytrends.build_payload(list(keywords), timeframe=_resolve_timeframe(timeframe))
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df


@st.cache_data(ttl=900)
def get_interest_over_time(keyword: str, timeframe: str = "3M") -> pd.DataFrame:
    """Fetch interest over time for a keyword.

    Returns DataFrame with 'date' and 'interest' columns.
    """
    pytrends = _get_pytrends()
    pytrends.build_payload([keyword], timeframe=_resolve_timeframe(timeframe))
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.DataFrame(columns=["date", "interest"])
    df = df.reset_index()
    df = df.rename(columns={"date": "date", keyword: "interest"})
    df = df[["date", "interest"]]
    return df
