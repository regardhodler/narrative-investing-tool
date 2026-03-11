"""
Stress Signals / Doomsday Monitor — Data Layer

Fetches credit stress indicators from FRED, canary watchlist via yfinance,
distress filings from SEC EDGAR, and whale exit data from 13F bulk files.
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from services.sec_client import SEC_HEADERS, _rate_limit, get_cik_ticker_map

# ---------------------------------------------------------------------------
# FRED API
# ---------------------------------------------------------------------------

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "BAMLC0A0CM": "ICE BofA US Corporate IG OAS",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "TEDRATE": "TED Spread",
    "DRTSCILM": "Banks Tightening C&I Loan Standards",
    "DRTSCLCC": "Banks Tightening Consumer Credit Card Standards",
}


@st.cache_data(ttl=21600)
def fetch_fred_series(series_id: str, observation_start: str | None = None) -> pd.DataFrame:
    """Fetch a single FRED series. Returns DataFrame with 'date' and 'value' columns."""
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return pd.DataFrame(columns=["date", "value"])

    if observation_start is None:
        observation_start = (date.today() - timedelta(days=365)).isoformat()

    try:
        resp = requests.get(
            FRED_BASE_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": observation_start,
                "sort_order": "asc",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

    observations = data.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["date", "value"])

    rows = []
    for obs in observations:
        val = obs.get("value", ".")
        if val == ".":
            continue
        try:
            rows.append({"date": obs["date"], "value": float(val)})
        except (ValueError, KeyError):
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=21600)
def get_credit_spreads() -> dict:
    """Fetch all FRED stress series. Returns dict of {series_id: DataFrame}."""
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return {sid: pd.DataFrame(columns=["date", "value"]) for sid in FRED_SERIES}

    result = {}
    for sid in FRED_SERIES:
        result[sid] = fetch_fred_series(sid)
    return result


# ---------------------------------------------------------------------------
# CANARY WATCHLIST
# ---------------------------------------------------------------------------

CANARY_TICKERS = {
    "Regional Banks": ["KRE", "PACW", "WAL", "NYCB"],
    "Commercial Real Estate": ["VNQ", "MORT", "ABR", "BXMT"],
    "Private Equity Exposure": ["BX", "KKR", "APO", "ARES"],
    "Subprime / Consumer Credit": ["CACC", "SC", "SLM", "NAVI", "OMF"],
    "High Yield / Distressed": ["HYG", "JNK", "BKLN", "SJNK"],
    "Volatility / Fear": ["^VIX", "TLT"],
}


@st.cache_data(ttl=3600)
def get_canary_signals() -> pd.DataFrame:
    """Fetch canary watchlist data via yfinance.

    Returns DataFrame with: ticker, category, price, 1w_ret, 1m_ret, 3m_ret,
    drawdown_52w, volume_ratio.
    """
    all_tickers = []
    ticker_to_cat = {}
    for cat, tickers in CANARY_TICKERS.items():
        for t in tickers:
            all_tickers.append(t)
            ticker_to_cat[t] = cat

    rows = []
    for ticker in all_tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1y")
            if hist.empty or len(hist) < 5:
                continue

            current_price = float(hist["Close"].iloc[-1])
            high_52w = float(hist["Close"].max())
            drawdown = ((current_price - high_52w) / high_52w) * 100 if high_52w > 0 else 0

            # Returns
            def _ret(n_days):
                if len(hist) > n_days:
                    old = float(hist["Close"].iloc[-min(n_days, len(hist))])
                    return ((current_price - old) / old) * 100 if old > 0 else 0
                return 0

            ret_1w = _ret(5)
            ret_1m = _ret(21)
            ret_3m = _ret(63)

            # Volume ratio
            vol_20d_avg = float(hist["Volume"].iloc[-20:].mean()) if len(hist) >= 20 else 0
            vol_today = float(hist["Volume"].iloc[-1])
            vol_ratio = vol_today / vol_20d_avg if vol_20d_avg > 0 else 0

            rows.append({
                "ticker": ticker,
                "category": ticker_to_cat[ticker],
                "price": round(current_price, 2),
                "1w_ret": round(ret_1w, 2),
                "1m_ret": round(ret_1m, 2),
                "3m_ret": round(ret_3m, 2),
                "drawdown_52w": round(drawdown, 2),
                "volume_ratio": round(vol_ratio, 2),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "category", "price", "1w_ret", "1m_ret", "3m_ret",
            "drawdown_52w", "volume_ratio",
        ])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DISTRESS FILING SCANNER (SEC EDGAR EFTS)
# ---------------------------------------------------------------------------

DISTRESS_KEYWORDS = [
    "going concern",
    "material weakness",
    "covenant violation",
    "default",
    "bankruptcy",
    "credit facility termination",
    "liquidity crisis",
]


@st.cache_data(ttl=21600)
def scan_distress_filings(days_back: int = 30) -> list[dict]:
    """Search SEC EDGAR full-text search for filings containing stress keywords.

    Returns list of dicts: company, ticker, form_type, date, keyword, url
    """
    cik_map = get_cik_ticker_map()
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days_back)).isoformat()

    all_results = []
    seen_keys = set()  # deduplicate by (company, date, keyword)

    for keyword in DISTRESS_KEYWORDS:
        _rate_limit()
        try:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f'"{keyword}"',
                    "forms": "8-K,10-K,10-Q",
                    "dateRange": "custom",
                    "startdt": start_date,
                    "enddt": end_date,
                    "from": 0,
                    "size": 20,
                },
                headers=SEC_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        hits = data.get("hits", {}).get("hits", [])
        for hit in hits:
            source = hit.get("_source", {})
            company = ""
            if source.get("display_names"):
                company = source["display_names"][0]
            elif source.get("entity_name"):
                company = source["entity_name"]

            filing_date = source.get("file_date", "")
            form_type = source.get("form_type", "")
            cik = str(source.get("ciks", [""])[0]) if source.get("ciks") else ""
            ticker = cik_map.get(cik, "")

            # Build filing URL
            file_num = source.get("file_num", "")
            accession = source.get("accession_no", "")
            url = ""
            if accession:
                acc_no_dash = accession.replace("-", "")
                padded_cik = cik.zfill(10) if cik else ""
                url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{acc_no_dash}/"

            dedup_key = (company, filing_date, keyword)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            all_results.append({
                "company": company,
                "ticker": ticker,
                "form_type": form_type,
                "date": filing_date,
                "keyword": keyword,
                "url": url,
            })

    # Sort by date descending
    all_results.sort(key=lambda x: x.get("date", ""), reverse=True)
    return all_results


# ---------------------------------------------------------------------------
# DTCC TOP CDS REFERENCE ENTITIES (hardcoded sample data)
# ---------------------------------------------------------------------------

# Sample data — update from DTCC quarterly reports
# Source pattern: https://www.dtcc.com/repository-otc-data
DTCC_TOP_CDS = {
    "as_of": "Q3 2025",
    "sovereign": [
        {"entity": "Federative Republic of Brazil", "net_notional_bn": 15.2, "gross_notional_bn": 45.6, "qoq_change_pct": 5.3, "contracts": 12500},
        {"entity": "Republic of Turkey", "net_notional_bn": 11.8, "gross_notional_bn": 38.4, "qoq_change_pct": 8.7, "contracts": 9800},
        {"entity": "United Mexican States", "net_notional_bn": 10.5, "gross_notional_bn": 31.2, "qoq_change_pct": -2.1, "contracts": 8900},
        {"entity": "Republic of Italy", "net_notional_bn": 9.8, "gross_notional_bn": 28.7, "qoq_change_pct": 3.4, "contracts": 8200},
        {"entity": "Republic of South Africa", "net_notional_bn": 8.4, "gross_notional_bn": 25.1, "qoq_change_pct": 12.5, "contracts": 7100},
        {"entity": "Republic of Colombia", "net_notional_bn": 7.2, "gross_notional_bn": 21.8, "qoq_change_pct": 6.8, "contracts": 6300},
        {"entity": "People's Republic of China", "net_notional_bn": 6.9, "gross_notional_bn": 19.4, "qoq_change_pct": 15.2, "contracts": 5800},
        {"entity": "Republic of Indonesia", "net_notional_bn": 5.6, "gross_notional_bn": 16.7, "qoq_change_pct": -1.3, "contracts": 4900},
        {"entity": "Republic of the Philippines", "net_notional_bn": 4.8, "gross_notional_bn": 14.2, "qoq_change_pct": 2.7, "contracts": 4200},
        {"entity": "Kingdom of Saudi Arabia", "net_notional_bn": 4.1, "gross_notional_bn": 12.5, "qoq_change_pct": 9.1, "contracts": 3600},
    ],
    "corporate": [
        {"entity": "Ford Motor Company", "net_notional_bn": 4.8, "gross_notional_bn": 18.2, "qoq_change_pct": 12.1, "contracts": 8900},
        {"entity": "General Motors Company", "net_notional_bn": 4.2, "gross_notional_bn": 15.6, "qoq_change_pct": 7.3, "contracts": 7600},
        {"entity": "AT&T Inc.", "net_notional_bn": 3.9, "gross_notional_bn": 14.1, "qoq_change_pct": -3.2, "contracts": 7100},
        {"entity": "Teva Pharmaceutical", "net_notional_bn": 3.5, "gross_notional_bn": 12.8, "qoq_change_pct": 18.4, "contracts": 6500},
        {"entity": "Carnival Corporation", "net_notional_bn": 3.1, "gross_notional_bn": 11.4, "qoq_change_pct": 9.6, "contracts": 5800},
        {"entity": "Deutsche Bank AG", "net_notional_bn": 2.8, "gross_notional_bn": 10.2, "qoq_change_pct": 14.7, "contracts": 5200},
        {"entity": "Macy's Inc.", "net_notional_bn": 2.4, "gross_notional_bn": 8.9, "qoq_change_pct": 22.3, "contracts": 4600},
        {"entity": "Dish Network Corp", "net_notional_bn": 2.1, "gross_notional_bn": 7.8, "qoq_change_pct": 31.5, "contracts": 4100},
        {"entity": "Occidental Petroleum", "net_notional_bn": 1.9, "gross_notional_bn": 7.1, "qoq_change_pct": 5.8, "contracts": 3700},
        {"entity": "American Airlines Group", "net_notional_bn": 1.7, "gross_notional_bn": 6.4, "qoq_change_pct": 11.2, "contracts": 3300},
    ],
}


# ---------------------------------------------------------------------------
# WHALE EXITS (reuse whale_screener)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_whale_exits(top_n: int = 20) -> pd.DataFrame:
    """Get biggest whale position closures/decreases from 13F data.

    Returns top N exits sorted by absolute value change (negative = biggest exit).
    """
    try:
        from services.whale_screener import screen_whale_buyers
        df = screen_whale_buyers(top_n=200)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Filter for exits: CLOSED or DECREASED positions
    exits = df[df["status"].isin(["CLOSED", "DECREASED"])].copy()
    if exits.empty:
        return pd.DataFrame()

    # Sort by value_change ascending (most negative first = biggest exits)
    exits = exits.sort_values("value_change", ascending=True).head(top_n)
    return exits.reset_index(drop=True)
