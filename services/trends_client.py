import pandas as pd
import streamlit as st
from pytrends.request import TrendReq


def _get_pytrends() -> TrendReq:
    return TrendReq(hl="en-US", tz=360)


@st.cache_data(ttl=900)
def get_trending_searches() -> list[str]:
    """Fetch real-time trending searches for the US. Returns list of topic strings."""
    pytrends = _get_pytrends()
    try:
        df = pytrends.trending_searches(pn="united_states")
        return df[0].tolist()
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
