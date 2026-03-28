"""Schedule 13D/13G activism screener — SEC EDGAR EFTS API.

SC 13D = activist stake (>5% + intent to influence management).
SC 13G = passive stake (>5%, no activist intent).

Filed within 10 days of crossing the 5% threshold.
Activism campaigns historically drive 10–30% price moves.
"""

import time
import requests
import streamlit as st

_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_VIEWER = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"

_SEC_HEADERS = {
    "User-Agent": "NarrativeInvestingTool research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

_last_call: float = 0.0


def _rate_limit() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 0.15:
        time.sleep(0.15 - elapsed)
    _last_call = time.time()


@st.cache_data(ttl=3600)
def get_activism_filings(days_back: int = 30, include_13g: bool = False) -> list[dict]:
    """Fetch recent SC 13D (and optionally SC 13G) filings from SEC EDGAR.

    Returns list of dicts with keys:
        filer       — entity filing the report (the large shareholder)
        subject     — target company (when available from display_names)
        form_type   — SC 13D or SC 13G
        file_date   — ISO date string
        accession_no — SEC accession number
        filing_url  — direct link to EDGAR filing index
    """
    from datetime import datetime, timedelta

    start_dt = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    forms = "SC 13D,SC 13G" if include_13g else "SC 13D"

    _rate_limit()
    try:
        resp = requests.get(
            _EFTS_URL,
            params={
                "q": "",
                "forms": forms,
                "dateRange": "custom",
                "startdt": start_dt,
                "from": 0,
                "size": 50,
            },
            headers=_SEC_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    rows = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        display_names = src.get("display_names", [])

        # For 13D/G filings, display_names typically has [filer_name, subject_company]
        filer = display_names[0] if display_names else src.get("entity_name", "Unknown")
        subject = display_names[1] if len(display_names) > 1 else "—"

        accession = src.get("accession_no", "")
        ciks = src.get("ciks", [])
        cik = ciks[0] if ciks else ""

        # Build direct EDGAR filing URL
        if accession and cik:
            acc_path = accession.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/{accession}-index.htm"
        elif accession:
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&accession={accession}"
        else:
            filing_url = ""

        rows.append({
            "filer": filer,
            "subject": subject,
            "form_type": src.get("form_type", "SC 13D"),
            "file_date": src.get("file_date", ""),
            "accession_no": accession,
            "filing_url": filing_url,
        })

    # Sort most recent first
    rows.sort(key=lambda x: x["file_date"], reverse=True)
    return rows
