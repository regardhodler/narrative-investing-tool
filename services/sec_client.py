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


@st.cache_data(ttl=3600)
def get_company_info(ticker: str) -> dict | None:
    """Look up basic company info from SEC submissions for a ticker.

    Returns dict with: name, cik, sic, sic_description, state, fiscal_year_end, tickers, exchanges
    """
    cik_map = get_cik_ticker_map()
    ticker_upper = ticker.strip().upper()
    cik = None
    for c, t in cik_map.items():
        if t == ticker_upper:
            cik = c
            break

    if cik is None:
        return None

    submissions = get_company_submissions(cik)
    return {
        "name": submissions.get("name", ""),
        "cik": cik,
        "sic": submissions.get("sic", ""),
        "sic_description": submissions.get("sicDescription", ""),
        "state": submissions.get("stateOfIncorporation", ""),
        "fiscal_year_end": submissions.get("fiscalYearEnd", ""),
        "tickers": submissions.get("tickers", []),
        "exchanges": submissions.get("exchanges", []),
    }


@st.cache_data(ttl=3600)
def fetch_filing_text(url: str, max_chars: int = 50000) -> str:
    """Fetch the text content of a SEC filing, truncated to max_chars.

    Strips HTML tags for a rough plain-text extraction.
    """
    import re

    _rate_limit()
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated]"

    return text


@st.cache_data(ttl=3600)
def get_filings_by_ticker(ticker: str) -> pd.DataFrame:
    """Look up a ticker's recent SEC filings via CIK.

    Returns DataFrame with columns: form_type, date, description, accession_number, url
    """
    # Reverse lookup: ticker → CIK
    cik_map = get_cik_ticker_map()
    ticker_upper = ticker.strip().upper()
    cik = None
    for c, t in cik_map.items():
        if t == ticker_upper:
            cik = c
            break

    if cik is None:
        return pd.DataFrame(
            columns=["form_type", "date", "description", "accession_number", "url"]
        )

    submissions = get_company_submissions(cik)
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame(
            columns=["form_type", "date", "description", "accession_number", "url"]
        )

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    padded_cik = cik.zfill(10)
    rows = []
    for i in range(len(forms)):
        acc = accessions[i] if i < len(accessions) else ""
        doc = primary_docs[i] if i < len(primary_docs) else ""
        acc_no_dash = acc.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{acc_no_dash}/{doc}"
            if acc and doc
            else ""
        )
        rows.append(
            {
                "form_type": forms[i] if i < len(forms) else "",
                "date": dates[i] if i < len(dates) else "",
                "description": descriptions[i] if i < len(descriptions) else "",
                "accession_number": acc,
                "url": url,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["form_type", "date", "description", "accession_number", "url"]
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def get_insider_trades(ticker: str) -> pd.DataFrame:
    """Fetch recent Form 4 insider trades from SEC EDGAR for a ticker.

    Returns DataFrame: insider_name, title, date, type, shares, price, value
    """
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor, as_completed

    empty = pd.DataFrame(columns=["insider_name", "title", "date", "type", "shares", "price", "value"])

    # Resolve ticker → CIK
    cik_map = get_cik_ticker_map()
    ticker_upper = ticker.strip().upper()
    cik = None
    for c, t in cik_map.items():
        if t == ticker_upper:
            cik = c
            break

    if cik is None:
        return empty

    # Get Form 4 filings from submissions
    submissions = get_company_submissions(cik)
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return empty

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    padded_cik = cik.zfill(10)
    form4_filings = []
    for i, form in enumerate(forms):
        if form == "4":
            form4_filings.append({
                "accession": accessions[i],
                "acc_no_dash": accessions[i].replace("-", ""),
                "date": dates[i],
            })
        if len(form4_filings) >= 20:
            break

    def _fetch_filing(filing):
        """Fetch and parse a single Form 4 filing. Returns list of trade dicts."""
        trades = []
        try:
            index_url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{filing['acc_no_dash']}/index.json"
            resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
            resp.raise_for_status()
            index_data = resp.json()

            xml_filename = None
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml") and "primary_doc" not in name.lower():
                    xml_filename = name
                    break

            if not xml_filename:
                return trades

            xml_url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{filing['acc_no_dash']}/{xml_filename}"
            resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=10)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            owner_el = root.find(f".//{ns}reportingOwner")
            owner_name = ""
            owner_title = ""
            if owner_el is not None:
                name_el = owner_el.find(f".//{ns}rptOwnerName")
                if name_el is not None and name_el.text:
                    owner_name = name_el.text.strip()
                title_el = owner_el.find(f".//{ns}officerTitle")
                if title_el is not None and title_el.text:
                    owner_title = title_el.text.strip()

            for txn in root.findall(f".//{ns}nonDerivativeTransaction"):
                date_el = txn.find(f".//{ns}transactionDate/{ns}value")
                code_el = txn.find(f".//{ns}transactionCoding/{ns}transactionCode")
                shares_el = txn.find(f".//{ns}transactionAmounts/{ns}transactionShares/{ns}value")
                price_el = txn.find(f".//{ns}transactionAmounts/{ns}transactionPricePerShare/{ns}value")
                acq_disp_el = txn.find(f".//{ns}transactionAmounts/{ns}transactionAcquiredDisposedCode/{ns}value")

                txn_date = date_el.text if date_el is not None and date_el.text else filing["date"]
                txn_code = code_el.text if code_el is not None and code_el.text else ""
                txn_shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0
                txn_price = float(price_el.text) if price_el is not None and price_el.text else 0
                acq_disp = acq_disp_el.text if acq_disp_el is not None and acq_disp_el.text else ""

                code_map = {"P": "Purchase", "S": "Sale", "M": "Exercise", "A": "Grant"}
                txn_type = code_map.get(txn_code, txn_code)
                if not txn_type and acq_disp:
                    txn_type = "Purchase" if acq_disp == "A" else "Sale"

                trades.append({
                    "insider_name": owner_name,
                    "title": owner_title,
                    "date": txn_date,
                    "type": txn_type,
                    "shares": txn_shares,
                    "price": txn_price,
                    "value": txn_shares * txn_price,
                })
        except Exception:
            pass
        return trades

    # Fetch all filings concurrently (5 workers stays under SEC 10 req/sec limit)
    all_trades = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_filing, f): f for f in form4_filings}
        for future in as_completed(futures):
            all_trades.extend(future.result())

    if not all_trades:
        return empty

    return pd.DataFrame(all_trades).sort_values("date", ascending=False).reset_index(drop=True)


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _ninety_days_ago() -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=90)).isoformat()
