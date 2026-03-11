"""
13F Whale Screener — Fetch and diff whale filer holdings via SEC EDGAR.

Uses per-filer 13F XML parsing to build a complete holdings picture,
then diffs the two most recent filings to surface new/increased positions.
"""

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st

import services.sec_client as _sec

# ---------------------------------------------------------------------------
# Curated whale filers — CIK -> metadata
# ---------------------------------------------------------------------------
WHALE_FILERS = {
    "1067983": {"name": "Berkshire Hathaway", "category": "fundamental"},
    "1350694": {"name": "Bridgewater Associates", "category": "macro"},
    "1364742": {"name": "BlackRock", "category": "fundamental"},
    "102909": {"name": "Vanguard Group", "category": "fundamental"},
    "1037389": {"name": "Renaissance Technologies", "category": "quant"},
    "1423053": {"name": "Citadel Advisors", "category": "quant"},
    "1145549": {"name": "Jane Street Group", "category": "quant"},
    "1336528": {"name": "Pershing Square Capital", "category": "activist"},
    "1040273": {"name": "Third Point", "category": "activist"},
    "1345471": {"name": "ValueAct Capital", "category": "activist"},
    "1048445": {"name": "Elliott Investment Management", "category": "activist"},
    "1006438": {"name": "Appaloosa Management", "category": "fundamental"},
    "1061768": {"name": "Baupost Group", "category": "fundamental"},
    "1079114": {"name": "Greenlight Capital", "category": "fundamental"},
    "921669": {"name": "Icahn Enterprises", "category": "activist"},
    "1029160": {"name": "Soros Fund Management", "category": "macro"},
    "1167483": {"name": "Tiger Global Management", "category": "fundamental"},
    "1535392": {"name": "Coatue Management", "category": "fundamental"},
    "1649339": {"name": "D1 Capital Partners", "category": "fundamental"},
    "1061165": {"name": "Lone Pine Capital", "category": "fundamental"},
    "1103804": {"name": "Viking Global Investors", "category": "fundamental"},
    "1336326": {"name": "Millennium Management", "category": "quant"},
    "1159159": {"name": "Point72 Asset Management", "category": "quant"},
    "1061219": {"name": "Two Sigma Investments", "category": "quant"},
    "1167557": {"name": "DE Shaw & Co", "category": "quant"},
    "1544863": {"name": "Dragoneer Investment Group", "category": "fundamental"},
    "1603466": {"name": "Durable Capital Partners", "category": "fundamental"},
    "1510342": {"name": "Whale Rock Capital", "category": "fundamental"},
    "1044316": {"name": "Maverick Capital", "category": "fundamental"},
    "1484148": {"name": "Glenview Capital", "category": "fundamental"},
    "1279708": {"name": "Starboard Value", "category": "activist"},
    "1027451": {"name": "Jana Partners", "category": "activist"},
    "1113148": {"name": "Trian Fund Management", "category": "activist"},
    "1159235": {"name": "Paulson & Co", "category": "macro"},
    "1273087": {"name": "Ares Management", "category": "fundamental"},
    "1135730": {"name": "AQR Capital Management", "category": "quant"},
    "1582754": {"name": "Alkeon Capital Management", "category": "fundamental"},
    "1666048": {"name": "TCI Fund Management", "category": "activist"},
    "1009207": {"name": "Capital Research Global Investors", "category": "fundamental"},
    "315066": {"name": "State Street Corp", "category": "fundamental"},
    "1599901": {"name": "Altimeter Capital", "category": "fundamental"},
}

# Common ETF tickers to filter out
FILTERED_ISSUERS = {
    "SPDR S&P 500", "ISHARES TRUST", "INVESCO QQQ", "ISHARES RUSSELL",
    "VANGUARD INDEX", "VANGUARD TOTAL", "VANGUARD S&P", "SPDR GOLD",
    "SELECT SECTOR", "PROSHARES", "DIREXION",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_13f_all_holdings(cik: str, num_filings: int = 2) -> list[dict]:
    """Fetch ALL holdings from recent 13F filings for a filer.

    Returns list of dicts: [{filing_idx, issuer, cusip, value, shares}, ...]
    filing_idx 0 = most recent, 1 = previous quarter, etc.
    """
    filings = _sec.get_13f_holdings(cik)
    if not filings:
        return []

    padded_cik = cik.zfill(10)
    all_holdings = []

    # Grab up to num_filings most recent filings
    for filing_idx, filing in enumerate(filings[:num_filings]):
        try:
            _sec._rate_limit()
            index_url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{filing['accession']}/index.json"
            resp = requests.get(index_url, headers=_sec.SEC_HEADERS, timeout=10)
            resp.raise_for_status()
            index_data = resp.json()

            xml_filename = None
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "").lower()
                if name.endswith(".xml") and ("infotable" in name or "information" in name):
                    xml_filename = item.get("name", "")
                    break
            if not xml_filename:
                for item in index_data.get("directory", {}).get("item", []):
                    name = item.get("name", "")
                    lower = name.lower()
                    if lower.endswith(".xml") and "primary_doc" not in lower and not lower.startswith("r"):
                        xml_filename = name
                        break

            if not xml_filename:
                continue

            _sec._rate_limit()
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{padded_cik}/{filing['accession']}/{xml_filename}"
            resp = requests.get(xml_url, headers=_sec.SEC_HEADERS, timeout=15)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            for entry in root.iter():
                name_el = entry.find(f"{ns}nameOfIssuer")
                if name_el is None:
                    continue

                issuer = (name_el.text or "").strip()
                cusip_el = entry.find(f"{ns}cusip")
                cusip = (cusip_el.text or "").strip() if cusip_el is not None else ""
                value_el = entry.find(f"{ns}value")
                value = int(value_el.text) if value_el is not None and value_el.text else 0  # in thousands
                shares_el = entry.find(f"{ns}sshPrnamt") or entry.find(f"{ns}shrsOrPrnAmt/{ns}sshPrnamt")
                shares = int(shares_el.text) if shares_el is not None and shares_el.text else 0

                all_holdings.append({
                    "filing_idx": filing_idx,
                    "filing_date": filing["date"],
                    "issuer": issuer,
                    "cusip": cusip,
                    "value": value,
                    "shares": shares,
                })

        except Exception:
            continue

    return all_holdings


def _is_etf(issuer_name: str) -> bool:
    """Check if an issuer name looks like a common ETF/index fund."""
    upper = issuer_name.upper()
    return any(kw in upper for kw in FILTERED_ISSUERS)


def get_available_quarters() -> list[str]:
    """Return placeholder quarter labels based on typical 13F filing schedule."""
    from datetime import date
    today = date.today()
    current_q = (today.month - 1) // 3 + 1
    y = today.year

    labels = []
    q = current_q - 1
    if q == 0:
        q = 4
        y -= 1
    for _ in range(4):
        labels.append(f"Q{q} {y}")
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return labels


def screen_whale_buyers(
    top_n: int = 50,
    whale_only: bool = True,
    exclude_etfs: bool = True,
    min_value: float = 0,
    categories: list[str] | None = None,
    progress_callback=None,
    lookback_quarters: int = 1,
) -> pd.DataFrame:
    """Screen for the biggest whale position changes across 13F filings.

    lookback_quarters: 1 = compare latest vs previous (default),
                       2 = compare latest vs 2 quarters ago, etc.
    Returns top N positions by absolute value change, sorted descending.
    min_value is in millions.
    """
    filers = dict(WHALE_FILERS)

    # Filter by category
    if categories and "all" not in [c.lower() for c in categories]:
        filers = {k: v for k, v in filers.items() if v["category"] in categories}

    if not filers:
        return pd.DataFrame()

    num_filings = lookback_quarters + 1  # need current + the one N quarters back
    compare_idx = lookback_quarters  # filing_idx to compare against

    # Fetch holdings for all whale filers concurrently (max 3 workers to respect rate limits)
    all_rows = []
    filer_items = list(filers.items())

    def _fetch_one(cik_meta):
        cik, meta = cik_meta
        holdings = _fetch_13f_all_holdings(cik, num_filings=num_filings)
        return cik, meta, holdings

    completed = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_fetch_one, item): item for item in filer_items}
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed, len(filer_items))
            try:
                cik, meta, holdings = future.result()
            except Exception:
                continue

            if not holdings:
                continue

            # Split into current (filing_idx=0) and comparison (filing_idx=compare_idx)
            current = [h for h in holdings if h["filing_idx"] == 0]
            previous = [h for h in holdings if h["filing_idx"] == compare_idx]
            if not previous:
                # Fall back to the oldest available filing
                max_idx = max((h["filing_idx"] for h in holdings), default=0)
                if max_idx > 0:
                    previous = [h for h in holdings if h["filing_idx"] == max_idx]

            # Build lookup for previous quarter by CUSIP
            prev_map = {}
            for h in previous:
                key = h["cusip"]
                if key in prev_map:
                    prev_map[key]["value"] += h["value"]
                    prev_map[key]["shares"] += h["shares"]
                else:
                    prev_map[key] = {"value": h["value"], "shares": h["shares"], "issuer": h["issuer"]}

            # Build lookup for current quarter by CUSIP
            curr_map = {}
            for h in current:
                key = h["cusip"]
                if key in curr_map:
                    curr_map[key]["value"] += h["value"]
                    curr_map[key]["shares"] += h["shares"]
                else:
                    curr_map[key] = {
                        "value": h["value"], "shares": h["shares"],
                        "issuer": h["issuer"], "filing_date": h["filing_date"],
                    }

            # Compute deltas
            all_cusips = set(list(curr_map.keys()) + list(prev_map.keys()))
            for cusip in all_cusips:
                curr = curr_map.get(cusip, {"value": 0, "shares": 0, "issuer": "", "filing_date": ""})
                prev = prev_map.get(cusip, {"value": 0, "shares": 0, "issuer": ""})

                value_change = curr["value"] - prev["value"]
                shares_change = curr["shares"] - prev["shares"]
                issuer = curr["issuer"] or prev["issuer"]

                if curr["value"] == 0 and prev["value"] == 0:
                    continue

                is_new = prev["value"] == 0 and curr["value"] > 0
                is_closed = curr["value"] == 0 and prev["value"] > 0

                if is_new:
                    status = "NEW"
                elif is_closed:
                    status = "CLOSED"
                elif value_change > 0:
                    status = "INCREASED"
                elif value_change < 0:
                    status = "DECREASED"
                else:
                    status = "UNCHANGED"

                pct_change = 0.0
                if prev["value"] > 0:
                    pct_change = (value_change / prev["value"]) * 100

                all_rows.append({
                    "cik": cik,
                    "filer": meta["name"],
                    "whale_category": meta["category"],
                    "cusip": cusip,
                    "issuer": issuer,
                    "value_curr": curr["value"],
                    "value_prev": prev["value"],
                    "value_change": value_change,
                    "shares_change": shares_change,
                    "pct_change": round(pct_change, 1),
                    "status": status,
                    "filing_date": curr.get("filing_date", ""),
                })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Filter out ETFs
    if exclude_etfs:
        df = df[~df["issuer"].apply(_is_etf)]

    # Filter by min value (min_value in millions, data in thousands)
    if min_value > 0:
        min_val_thousands = min_value * 1000
        df = df[df["value_curr"] >= min_val_thousands]

    # Filter out UNCHANGED
    df = df[df["status"] != "UNCHANGED"]

    # Sort by absolute value change
    df["abs_value_change"] = df["value_change"].abs()
    df = df.sort_values("abs_value_change", ascending=False).head(top_n)
    df.drop(columns=["abs_value_change"], inplace=True)

    return df.reset_index(drop=True)
