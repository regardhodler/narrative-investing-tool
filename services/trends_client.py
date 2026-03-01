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


@st.cache_data(ttl=900)
def get_interest_over_time(keyword: str) -> pd.DataFrame:
    """Fetch 90-day interest over time for a keyword.

    Returns DataFrame with 'date' and 'interest' columns.
    """
    pytrends = _get_pytrends()
    pytrends.build_payload([keyword], timeframe="today 3-m")
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.DataFrame(columns=["date", "interest"])
    df = df.reset_index()
    df = df.rename(columns={"date": "date", keyword: "interest"})
    df = df[["date", "interest"]]
    return df
