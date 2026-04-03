import time
import threading
import requests
import pandas as pd
import streamlit as st

SEC_HEADERS = {
    "User-Agent": "NarrativeInvestingTool jud_rabs@yahoo.com",
    "Accept-Encoding": "gzip, deflate",
}

_last_request_time = 0.0
_rate_limit_lock = threading.Lock()


def _rate_limit():
    """Enforce 10 req/sec rate limit for SEC (thread-safe)."""
    global _last_request_time
    with _rate_limit_lock:
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


@st.cache_data(ttl=86400)
def get_ticker_company_map() -> list[tuple[str, str]]:
    """Return (ticker, company_name) pairs from SEC company_tickers.json."""
    _rate_limit()
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [(entry["ticker"], entry["title"]) for entry in data.values()]


def search_ticker_by_name(query: str, max_results: int = 10) -> list[dict]:
    """Search for tickers by company name or ticker substring."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    all_companies = get_ticker_company_map()
    exact = []
    partial = []
    for ticker, name in all_companies:
        if query_lower == ticker.lower():
            exact.append({"ticker": ticker, "name": name})
        elif query_lower in name.lower() or query_lower in ticker.lower():
            partial.append({"ticker": ticker, "name": name})
        if len(exact) + len(partial) >= max_results:
            break
    return (exact + partial)[:max_results]


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

    Uses BeautifulSoup to properly strip HTML/iXBRL tags, preserving readable text.
    """
    import re
    from bs4 import BeautifulSoup

    _rate_limit()
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        raw = resp.text
    except Exception:
        return ""

    # Use BeautifulSoup for proper HTML/iXBRL parsing
    try:
        soup = BeautifulSoup(raw, "html.parser")
        # Remove script/style/hidden elements
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        # Remove XBRL inline tags but keep their text content
        for tag in soup.find_all(True):
            if tag.name and tag.name.startswith("ix:"):
                tag.unwrap()
        text = soup.get_text(separator=" ")
    except Exception:
        # Fallback: naive regex strip
        text = re.sub(r"<[^>]+>", " ", raw)

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

                code_map = {
                    "P": "Purchase", "S": "Sale", "M": "Exercise", "A": "Grant",
                    "D": "Disposition", "J": "Other Acquisition", "G": "Gift", "C": "Conversion",
                    "F": "Tax Withholding", "X": "Option Exercise",
                }
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


@st.cache_data(ttl=3600)
def get_institution_holding(institution_cik: str, target_ticker: str) -> dict:
    """Fetch the latest 13F-HR filing for an institution and find the holding for a target ticker.

    Parses the 13F XML information table to find the holding matching the target company name.
    Returns {"value": dollar_amount, "shares": share_count, "date": filing_date} or empty dict.
    """
    import xml.etree.ElementTree as ET

    # Get the company name for fuzzy matching against 13F nameOfIssuer
    company_info = get_company_info(target_ticker)
    if not company_info or not company_info.get("name"):
        return {}
    target_name = company_info["name"].upper()
    # Build match tokens: split company name into words for flexible matching
    target_tokens = [w for w in target_name.replace(",", "").replace(".", "").split() if len(w) > 1]

    # Find the latest 13F-HR filing
    filings = get_13f_holdings(institution_cik)
    if not filings:
        return {}

    filing = filings[0]  # most recent
    padded_cik = institution_cik.zfill(10)

    try:
        # Fetch filing index to find the XML information table
        _rate_limit()
        index_url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{filing['accession']}/index.json"
        resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        index_data = resp.json()

        xml_filename = None
        for item in index_data.get("directory", {}).get("item", []):
            name = item.get("name", "").lower()
            if name.endswith(".xml") and ("infotable" in name or "information" in name):
                xml_filename = item.get("name", "")
                break

        # Fallback: any .xml that isn't the primary doc or R-file
        if not xml_filename:
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                lower = name.lower()
                if lower.endswith(".xml") and "primary_doc" not in lower and not lower.startswith("r"):
                    xml_filename = name
                    break

        if not xml_filename:
            return {}

        # Fetch and parse the XML
        _rate_limit()
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{filing['accession']}/{xml_filename}"
        resp = requests.get(xml_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)

        # Handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Search all infoTable entries for a match
        for entry in root.iter():
            name_el = entry.find(f"{ns}nameOfIssuer")
            if name_el is None:
                continue

            issuer_name = (name_el.text or "").upper().strip()
            # Match if enough target tokens appear in the issuer name or vice versa
            if not issuer_name:
                continue

            issuer_tokens = [w for w in issuer_name.replace(",", "").replace(".", "").split() if len(w) > 1]
            # Check overlap: at least 2 tokens match, or the shorter name is fully contained
            common = set(target_tokens) & set(issuer_tokens)
            match = (
                len(common) >= 2
                or (len(target_tokens) == 1 and target_tokens[0] in issuer_tokens)
                or (len(issuer_tokens) == 1 and issuer_tokens[0] in target_tokens)
                or target_ticker.upper() == issuer_name
            )
            if not match:
                continue

            value_el = entry.find(f"{ns}value")
            shares_el = entry.find(f"{ns}sshPrnamt") or entry.find(f"{ns}shrsOrPrnAmt/{ns}sshPrnamt")
            val = int(value_el.text) * 1000 if value_el is not None and value_el.text else 0  # 13F values in thousands
            shares = int(shares_el.text) if shares_el is not None and shares_el.text else 0

            return {"value": val, "shares": shares, "date": filing["date"]}

    except Exception:
        pass

    return {}


@st.cache_data(ttl=86400)
def get_latest_annual_filing(ticker: str) -> dict | None:
    """Return the primary document URL + metadata for the most recent 10-K filing.

    Returns dict with: url, accession_number, date, cik  — or None if not found.
    """
    ticker_upper = ticker.strip().upper()
    cik_map = get_cik_ticker_map()
    cik = next((c for c, t in cik_map.items() if t == ticker_upper), None)
    if cik is None:
        return None

    submissions = get_company_submissions(cik)
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return None

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    padded_cik = cik.zfill(10)
    for i, form in enumerate(forms):
        if form == "10-K":
            acc = accessions[i] if i < len(accessions) else ""
            doc = primary_docs[i] if i < len(primary_docs) else ""
            if not acc or not doc:
                continue
            acc_nodash = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{acc_nodash}/{doc}"
            return {
                "url": url,
                "accession_number": acc,
                "date": dates[i] if i < len(dates) else "",
                "cik": cik,
            }
    return None


def extract_mda_text(full_text: str, max_chars: int = 15000) -> str:
    """Extract the MD&A section from a 10-K filing's plain text.

    Searches for the 'Management's Discussion' header and returns text until
    the next major section. Falls back to a middle slice of the document.
    """
    import re

    # Common MD&A section header patterns
    start_patterns = [
        r"MANAGEMENT.{0,10}S DISCUSSION AND ANALYSIS",
        r"Management.{0,10}s Discussion and Analysis",
        r"ITEM\s*7[\.\s]+MANAGEMENT",
        r"Item\s*7[\.\s]+Management",
        r"Item\s*7\.",
        r"ITEM\s*7\.",
    ]
    # Common next-section patterns that signal MD&A end
    end_patterns = [
        r"QUANTITATIVE AND QUALITATIVE",
        r"Quantitative and Qualitative",
        r"ITEM\s*7A",
        r"Item\s*7A",
        r"ITEM\s*8[\.\s]+FINANCIAL STATEMENTS",
        r"Item\s*8[\.\s]+Financial Statements",
        r"CONTROLS AND PROCEDURES",
        r"Item\s*8\.",
        r"ITEM\s*8\.",
    ]

    start_idx = None
    for pat in start_patterns:
        m = re.search(pat, full_text)
        if m:
            start_idx = m.start()
            break

    if start_idx is None:
        # Fallback: grab middle third of the document
        third = len(full_text) // 3
        return full_text[third: third + max_chars]

    end_idx = None
    search_from = start_idx + 200  # skip past the header itself
    for pat in end_patterns:
        m = re.search(pat, full_text[search_from:])
        if m:
            candidate = search_from + m.start()
            # Must be at least 1000 chars of content
            if candidate - start_idx > 1000:
                end_idx = candidate
                break

    section = full_text[start_idx: end_idx] if end_idx else full_text[start_idx: start_idx + max_chars * 2]
    return section[:max_chars]


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _ninety_days_ago() -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=90)).isoformat()


def extract_debt_schedule(filing_text: str) -> list[dict]:
    """Parse a 10-K filing text to extract the debt maturity schedule.

    Looks for the contractual obligations table or long-term debt footnote.
    Returns a list of {year: str, amount: str} dicts, sorted by year.
    e.g. [{"year": "2025", "amount": "$1.2B"}, {"year": "2026", "amount": "$800M"}]
    """
    import re

    lines = filing_text.split("\n")
    results = []
    
    # Pattern: lines like "2025 ... $1,200" or "2026 ... 800,000" near "maturities" or "obligations"
    year_pattern = re.compile(
        r'\b(202[5-9]|203[0-9])\b.*?\$([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?',
        re.IGNORECASE
    )
    # Also capture "Thereafter" lines
    thereafter_pattern = re.compile(
        r'(thereafter|beyond 20[3-9]\d).*?\$([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)?',
        re.IGNORECASE
    )

    seen_years: set = set()
    # Find relevant section
    in_section = False
    for line in lines:
        lower = line.lower()
        if any(k in lower for k in ("contractual obligation", "debt maturity", "long-term debt", "future minimum", "maturities of")):
            in_section = True
        if in_section:
            m = year_pattern.search(line)
            if m:
                year = m.group(1)
                amount_raw = m.group(2).replace(",", "")
                unit = (m.group(3) or "").lower()
                try:
                    amount = float(amount_raw)
                    if unit in ("billion", "b"):
                        display = f"${amount:.1f}B"
                    elif unit in ("million", "m"):
                        display = f"${amount:.0f}M"
                    elif amount >= 1_000_000:
                        display = f"${amount/1e9:.2f}B"
                    elif amount >= 1_000:
                        display = f"${amount/1e6:.0f}M"
                    else:
                        display = f"${amount:,.0f}"
                    if year not in seen_years:
                        results.append({"year": year, "amount": display})
                        seen_years.add(year)
                except ValueError:
                    pass
            tm = thereafter_pattern.search(line)
            if tm:
                amount_raw = tm.group(2).replace(",", "")
                unit = (tm.group(3) or "").lower()
                try:
                    amount = float(amount_raw)
                    if unit in ("billion", "b"):
                        display = f"${amount:.1f}B"
                    else:
                        display = f"${amount/1e9:.2f}B" if amount >= 1e9 else f"${amount/1e6:.0f}M"
                    if "Thereafter" not in seen_years:
                        results.append({"year": "Thereafter", "amount": display})
                        seen_years.add("Thereafter")
                except ValueError:
                    pass
            # Stop after 50 lines past section start to avoid false positives
            if len(results) >= 8 or (in_section and lower.strip() == "" and len(results) > 0):
                if len(results) >= 3:
                    break

    return sorted(results, key=lambda x: (x["year"] != "Thereafter", x["year"]))
