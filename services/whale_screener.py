import io
import zipfile
import time

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
    "1656456": {"name": "Elliott Associates", "category": "activist"},
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
    "1037390": {"name": "Duquesne Family Office", "category": "macro"},
    "1273087": {"name": "Ares Management", "category": "fundamental"},
    "1135730": {"name": "AQR Capital Management", "category": "quant"},
    "1044064": {"name": "Och-Ziff Capital Management", "category": "fundamental"},
    "1582754": {"name": "Alkeon Capital Management", "category": "fundamental"},
    "1666048": {"name": "TCI Fund Management", "category": "activist"},
    "1544784": {"name": "Light Street Capital", "category": "fundamental"},
    "1050470": {"name": "Baupost Group LLC", "category": "fundamental"},
    "1345471": {"name": "ValueAct Capital", "category": "activist"},
    "1009207": {"name": "Capital Research Global Investors", "category": "fundamental"},
    "1067983": {"name": "Berkshire Hathaway", "category": "fundamental"},
    "315066": {"name": "State Street Corp", "category": "fundamental"},
    "1166559": {"name": "Matrix Capital Management", "category": "fundamental"},
    "1056831": {"name": "Tiger Management", "category": "fundamental"},
    "1599901": {"name": "Altimeter Capital", "category": "fundamental"},
}

# ---------------------------------------------------------------------------
# Common ETF CUSIPs to filter out
# ---------------------------------------------------------------------------
FILTERED_CUSIPS = {
    "78462F103",  # SPY
    "46090E103",  # IVV
    "922908363",  # VOO
    "46137V621",  # QQQ
    "464287200",  # IWM
    "464287622",  # IWF
    "464287507",  # IWD
    "78464A870",  # SLV
    "464285204",  # EFA
    "464286400",  # EEM
    "464287879",  # IWB
    "464287713",  # IWN
    "464287788",  # IWO
    "464287481",  # IWP
    "78468R663",  # XLF
    "78468R689",  # XLK
    "78468R622",  # XLE
    "78468R846",  # XLV
    "78468R705",  # XLI
    "78468R812",  # XLP
    "78468R838",  # XLU
    "78468R770",  # XLY
    "78468R648",  # XLB
    "922042858",  # VTI
    "922908769",  # VEA
    "922908504",  # VWO
    "260549208",  # DIA
    "233051432",  # GLD
    "46138E776",  # TLT
    "46138E735",  # LQD
    "46138E131",  # HYG
    "25459W540",  # DGRW
}


def _quarter_date_range(year: int, quarter: int) -> str:
    """Build the SEC bulk data file date-range string for a given quarter."""
    ranges = {
        1: (f"{year}q1", f"01jan{year}-31mar{year}"),
        2: (f"{year}q2", f"01apr{year}-30jun{year}"),
        3: (f"{year}q3", f"01jul{year}-30sep{year}"),
        4: (f"{year}q4", f"01oct{year}-31dec{year}"),
    }
    return ranges[quarter][1]


def get_available_quarters() -> list[tuple[int, int]]:
    """Return the 4 most recent quarters as (year, quarter) tuples.

    We go back from the current date; 13F data is typically available
    ~45 days after quarter end, so we skip the current quarter.
    """
    from datetime import date

    today = date.today()
    current_q = (today.month - 1) // 3 + 1
    current_y = today.year

    # Start from the previous quarter
    quarters = []
    y, q = current_y, current_q - 1
    if q == 0:
        q = 4
        y -= 1

    for _ in range(4):
        quarters.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1

    return quarters


@st.cache_data(ttl=86400, show_spinner=False)
def get_13f_bulk_quarter(year: int, quarter: int) -> pd.DataFrame:
    """Download and parse the SEC quarterly bulk 13F ZIP file.

    Returns a DataFrame with columns from INFOTABLE.tsv merged with SUBMISSION.tsv.
    """
    date_range = _quarter_date_range(year, quarter)
    url = f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{date_range}_form13f.zip"

    _sec._rate_limit()
    try:
        resp = requests.get(url, headers=_sec.SEC_HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            return pd.DataFrame()
        raise
    except Exception:
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()

            # Find INFOTABLE and SUBMISSION files (case-insensitive)
            info_file = None
            sub_file = None
            for n in names:
                lower = n.lower()
                if "infotable" in lower and lower.endswith(".tsv"):
                    info_file = n
                elif "submission" in lower and lower.endswith(".tsv"):
                    sub_file = n

            if info_file is None:
                return pd.DataFrame()

            info_df = pd.read_csv(
                zf.open(info_file),
                sep="\t",
                dtype=str,
                on_bad_lines="skip",
            )

            if sub_file:
                sub_df = pd.read_csv(
                    zf.open(sub_file),
                    sep="\t",
                    dtype=str,
                    on_bad_lines="skip",
                )
            else:
                sub_df = pd.DataFrame()

    except Exception:
        return pd.DataFrame()

    # Normalize column names to lowercase
    info_df.columns = [c.strip().lower() for c in info_df.columns]
    if not sub_df.empty:
        sub_df.columns = [c.strip().lower() for c in sub_df.columns]

    # Identify the join key — typically 'accession_number' or 'accessionnumber'
    info_acc_col = None
    for candidate in ["accession_number", "accessionnumber", "accession"]:
        if candidate in info_df.columns:
            info_acc_col = candidate
            break

    if not sub_df.empty and info_acc_col:
        sub_acc_col = None
        for candidate in ["accession_number", "accessionnumber", "accession"]:
            if candidate in sub_df.columns:
                sub_acc_col = candidate
                break

        if sub_acc_col:
            # Keep useful columns from submission
            sub_cols = [sub_acc_col]
            for c in ["cik", "filingmanager_name", "filing_manager_name", "managername"]:
                if c in sub_df.columns:
                    sub_cols.append(c)
            sub_df = sub_df[sub_cols].drop_duplicates(subset=[sub_acc_col])

            info_df = info_df.merge(
                sub_df,
                left_on=info_acc_col,
                right_on=sub_acc_col,
                how="left",
                suffixes=("", "_sub"),
            )

    # Normalize value and shares columns to numeric
    for col in ["value", "sshprnamt", "shares"]:
        if col in info_df.columns:
            info_df[col] = pd.to_numeric(info_df[col], errors="coerce")

    # Standardize CIK column
    if "cik" not in info_df.columns:
        for c in info_df.columns:
            if "cik" in c.lower():
                info_df.rename(columns={c: "cik"}, inplace=True)
                break

    if "cik" in info_df.columns:
        info_df["cik"] = info_df["cik"].astype(str).str.strip()

    return info_df


def compute_quarter_deltas(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    whale_only: bool = True,
    exclude_etfs: bool = True,
) -> pd.DataFrame:
    """Compute position changes between two quarters.

    Returns DataFrame with value_change, shares_change, pct_change, is_new_position, is_closed.
    """
    if current_df.empty or previous_df.empty:
        return pd.DataFrame()

    # Identify CUSIP column
    cusip_col = None
    for c in ["cusip", "cusip_number"]:
        if c in current_df.columns:
            cusip_col = c
            break
    if cusip_col is None:
        return pd.DataFrame()

    # Identify value column
    val_col = "value" if "value" in current_df.columns else None
    if val_col is None:
        return pd.DataFrame()

    # Identify shares column
    shares_col = None
    for c in ["sshprnamt", "shares"]:
        if c in current_df.columns:
            shares_col = c
            break

    # Identify issuer name column
    issuer_col = None
    for c in ["nameofissuer", "issuer_name", "issuername"]:
        if c in current_df.columns:
            issuer_col = c
            break

    # Identify filer name column
    filer_col = None
    for c in ["filingmanager_name", "filing_manager_name", "managername"]:
        if c in current_df.columns:
            filer_col = c
            break

    # Build grouping keys
    group_cols = ["cik", cusip_col]

    # Aggregate current quarter
    agg_dict = {val_col: "sum"}
    if shares_col:
        agg_dict[shares_col] = "sum"

    # Keep first issuer name and filer name
    extra_cols = {}
    if issuer_col:
        extra_cols["issuer"] = current_df.groupby(group_cols)[issuer_col].first()
    if filer_col:
        extra_cols["filer"] = current_df.groupby(group_cols)[filer_col].first()

    curr_agg = current_df.groupby(group_cols).agg(agg_dict).reset_index()
    curr_agg.rename(columns={val_col: "value_curr"}, inplace=True)
    if shares_col:
        curr_agg.rename(columns={shares_col: "shares_curr"}, inplace=True)

    # Same for previous quarter — use same column detection for prev
    prev_cusip_col = None
    for c in ["cusip", "cusip_number"]:
        if c in previous_df.columns:
            prev_cusip_col = c
            break

    prev_val_col = "value" if "value" in previous_df.columns else None
    prev_shares_col = None
    for c in ["sshprnamt", "shares"]:
        if c in previous_df.columns:
            prev_shares_col = c
            break

    if prev_cusip_col is None or prev_val_col is None:
        return pd.DataFrame()

    prev_group = ["cik", prev_cusip_col]
    prev_agg_dict = {prev_val_col: "sum"}
    if prev_shares_col:
        prev_agg_dict[prev_shares_col] = "sum"

    prev_agg = previous_df.groupby(prev_group).agg(prev_agg_dict).reset_index()
    prev_agg.rename(columns={prev_val_col: "value_prev", prev_cusip_col: cusip_col}, inplace=True)
    if prev_shares_col:
        prev_agg.rename(columns={prev_shares_col: "shares_prev"}, inplace=True)

    # Outer join
    merged = curr_agg.merge(prev_agg, on=["cik", cusip_col], how="outer")

    # Fill NaN with 0 for value/shares
    for c in ["value_curr", "value_prev"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    for c in ["shares_curr", "shares_prev"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    # Compute deltas
    merged["value_change"] = merged["value_curr"] - merged["value_prev"]
    if "shares_curr" in merged.columns and "shares_prev" in merged.columns:
        merged["shares_change"] = merged["shares_curr"] - merged["shares_prev"]
    else:
        merged["shares_change"] = 0

    merged["pct_change"] = 0.0
    mask = merged["value_prev"] > 0
    merged.loc[mask, "pct_change"] = (
        merged.loc[mask, "value_change"] / merged.loc[mask, "value_prev"] * 100
    )

    merged["is_new_position"] = (merged["value_prev"] == 0) & (merged["value_curr"] > 0)
    merged["is_closed"] = (merged["value_curr"] == 0) & (merged["value_prev"] > 0)

    # Status label
    def _status(row):
        if row["is_new_position"]:
            return "NEW"
        if row["is_closed"]:
            return "CLOSED"
        if row["value_change"] > 0:
            return "INCREASED"
        if row["value_change"] < 0:
            return "DECREASED"
        return "UNCHANGED"

    merged["status"] = merged.apply(_status, axis=1)

    # Attach issuer name from current (or previous if closed position)
    if issuer_col and issuer_col in current_df.columns:
        issuer_map = current_df.drop_duplicates(subset=[cusip_col])[[cusip_col, issuer_col]]
        issuer_map = issuer_map.rename(columns={issuer_col: "issuer"})
        merged = merged.merge(issuer_map, on=cusip_col, how="left")

        # Fill from previous for closed positions
        if issuer_col in previous_df.columns:
            prev_issuer_col_name = issuer_col
            prev_issuer = previous_df.drop_duplicates(subset=[prev_cusip_col])
            if prev_cusip_col != cusip_col:
                prev_issuer = prev_issuer.rename(columns={prev_cusip_col: cusip_col})
            prev_issuer = prev_issuer[[cusip_col, prev_issuer_col_name]]
            prev_issuer = prev_issuer.rename(columns={prev_issuer_col_name: "issuer_prev"})
            merged = merged.merge(prev_issuer, on=cusip_col, how="left")
            merged["issuer"] = merged["issuer"].fillna(merged.get("issuer_prev", ""))
            if "issuer_prev" in merged.columns:
                merged.drop(columns=["issuer_prev"], inplace=True)

    # Attach filer name
    if filer_col and filer_col in current_df.columns:
        filer_map = current_df.drop_duplicates(subset=["cik"])[["cik", filer_col]]
        filer_map = filer_map.rename(columns={filer_col: "filer"})
        merged = merged.merge(filer_map, on="cik", how="left")
    elif "filer" not in merged.columns:
        merged["filer"] = merged["cik"]

    # Also try mapping filer from previous for closed positions
    if "filer" in merged.columns:
        prev_filer_col = None
        for c in ["filingmanager_name", "filing_manager_name", "managername"]:
            if c in previous_df.columns:
                prev_filer_col = c
                break
        if prev_filer_col:
            prev_filer_map = previous_df.drop_duplicates(subset=["cik"])[["cik", prev_filer_col]]
            prev_filer_map = prev_filer_map.rename(columns={prev_filer_col: "filer_prev"})
            merged = merged.merge(prev_filer_map, on="cik", how="left")
            merged["filer"] = merged["filer"].fillna(merged.get("filer_prev", ""))
            if "filer_prev" in merged.columns:
                merged.drop(columns=["filer_prev"], inplace=True)

    # Map whale names from our dict
    merged["whale_name"] = merged["cik"].map(
        lambda c: WHALE_FILERS.get(str(c), {}).get("name", "")
    )
    merged["whale_category"] = merged["cik"].map(
        lambda c: WHALE_FILERS.get(str(c), {}).get("category", "")
    )
    # Prefer whale_name over filer if available
    mask_whale = merged["whale_name"] != ""
    merged.loc[mask_whale, "filer"] = merged.loc[mask_whale, "whale_name"]

    # Filter to whale filers
    if whale_only:
        whale_ciks = set(WHALE_FILERS.keys())
        merged = merged[merged["cik"].isin(whale_ciks)]

    # Filter out ETF CUSIPs
    if exclude_etfs:
        merged = merged[~merged[cusip_col].isin(FILTERED_CUSIPS)]

    # Rename cusip column for consistency
    if cusip_col != "cusip":
        merged.rename(columns={cusip_col: "cusip"}, inplace=True)

    # Ensure required columns exist
    for col in ["issuer", "filer", "cusip"]:
        if col not in merged.columns:
            merged[col] = ""

    return merged


def screen_whale_buyers(
    year: int | None = None,
    quarter: int | None = None,
    top_n: int = 50,
    whale_only: bool = True,
    exclude_etfs: bool = True,
    min_value: float = 0,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Screen for the biggest whale position changes this quarter.

    Returns top N positions by absolute value change, sorted descending.
    """
    quarters = get_available_quarters()
    if year is not None and quarter is not None:
        curr_q = (year, quarter)
        # Find previous quarter
        idx = None
        for i, q in enumerate(quarters):
            if q == curr_q:
                idx = i
                break
        if idx is not None and idx + 1 < len(quarters):
            prev_q = quarters[idx + 1]
        else:
            # Manually compute previous
            py, pq = year, quarter - 1
            if pq == 0:
                pq = 4
                py -= 1
            prev_q = (py, pq)
    else:
        curr_q = quarters[0]
        prev_q = quarters[1]

    current_df = get_13f_bulk_quarter(curr_q[0], curr_q[1])
    previous_df = get_13f_bulk_quarter(prev_q[0], prev_q[1])

    if current_df.empty:
        return pd.DataFrame()

    deltas = compute_quarter_deltas(
        current_df, previous_df,
        whale_only=whale_only,
        exclude_etfs=exclude_etfs,
    )

    if deltas.empty:
        return pd.DataFrame()

    # Apply min value filter (value in thousands in 13F data)
    if min_value > 0:
        # min_value is in millions from the UI, value in the data is in $thousands
        min_val_thousands = min_value * 1000
        deltas = deltas[deltas["value_curr"] >= min_val_thousands]

    # Filter by category
    if categories and "all" not in [c.lower() for c in categories]:
        deltas = deltas[deltas["whale_category"].isin(categories)]

    # Sort by absolute value change
    deltas["abs_value_change"] = deltas["value_change"].abs()
    deltas = deltas.sort_values("abs_value_change", ascending=False).head(top_n)
    deltas.drop(columns=["abs_value_change"], inplace=True)

    return deltas.reset_index(drop=True)
