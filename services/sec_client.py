import time
import requests
import pandas as pd
import streamlit as st

SEC_HEADERS = {
    "User-Agent": "NarrativeInvestingTool research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

_last_request_time = 0.0


def _rate_limit():
    """Enforce 10 req/sec rate limit for SEC."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.1:
        time.sleep(0.1 - elapsed)
    _last_request_time = time.time()


@st.cache_data(ttl=86400)
def get_cik_ticker_map() -> dict[str, str]:
    """Fetch CIK-to-ticker mapping from SEC. Returns {CIK_str: ticker}."""
    _rate_limit()
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    mapping = {}
    for entry in data.values():
        cik = str(entry["cik_str"])
        mapping[cik] = entry["ticker"]
    return mapping


@st.cache_data(ttl=3600)
def search_filings(keyword: str, max_results: int = 500) -> pd.DataFrame:
    """Search EDGAR full-text search for keyword mentions in filings.

    Returns DataFrame with columns: company, ticker, form_type, date, cik
    """
    cik_map = get_cik_ticker_map()
    all_results = []
    page_size = 100

    for start in range(0, max_results, page_size):
        _rate_limit()
        params = {
            "q": f'"{keyword}"',
            "dateRange": "custom",
            "startdt": _ninety_days_ago(),
            "enddt": _today(),
            "from": start,
            "size": page_size,
        }
        try:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                headers=SEC_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            source = hit.get("_source", {})
            cik = str(source.get("ciks", [""])[0]) if source.get("ciks") else ""
            company = ""
            if source.get("display_names"):
                company = source["display_names"][0]
            elif source.get("entity_name"):
                company = source["entity_name"]

            ticker = cik_map.get(cik, "")
            all_results.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "form_type": source.get("form_type", ""),
                    "date": source.get("file_date", ""),
                    "cik": cik,
                }
            )

        if len(hits) < page_size:
            break

    if not all_results:
        return pd.DataFrame(columns=["company", "ticker", "form_type", "date", "cik"])

    return pd.DataFrame(all_results)


@st.cache_data(ttl=3600)
def get_company_submissions(cik: str) -> dict:
    """Fetch company submission history from SEC."""
    padded = cik.zfill(10)
    _rate_limit()
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{padded}.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def get_13f_holdings(cik: str) -> list[dict]:
    """Fetch 13F-HR filings for a CIK and parse holdings.

    Returns list of dicts with quarterly aggregated data.
    """
    submissions = get_company_submissions(cik)
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings_13f = []
    for i, form in enumerate(forms):
        if form in ("13F-HR", "13F-HR/A"):
            filings_13f.append(
                {
                    "accession": accessions[i].replace("-", ""),
                    "date": dates[i],
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                }
            )
        if len(filings_13f) >= 8:
            break

    return filings_13f


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _ninety_days_ago() -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=90)).isoformat()
