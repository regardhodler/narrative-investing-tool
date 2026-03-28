"""Shared sector rotation data layer — 11 SPDR ETF momentum + regime alignment.

Used by: risk_regime, narrative_discovery, valuation, trade_journal (portfolio intel).
"""

import streamlit as st
import pandas as pd
import yfinance as yf

# ── ETF registry: ticker → (display name, 4-word description) ─────────────────
SECTOR_ETFS: dict[str, tuple[str, str]] = {
    "XLK":  ("Technology",        "Software, chips, hardware"),
    "XLV":  ("Health Care",       "Pharma, biotech, devices"),
    "XLE":  ("Energy",            "Oil, gas, pipelines"),
    "XLF":  ("Financials",        "Banks, insurance, asset mgmt"),
    "XLI":  ("Industrials",       "Defense, machinery, transport"),
    "XLC":  ("Communication",     "Media, telecom, social"),
    "XLP":  ("Consumer Staples",  "Food, beverage, household"),
    "XLY":  ("Consumer Discr.",   "Retail, auto, leisure"),
    "XLU":  ("Utilities",         "Electric, water, gas utilities"),
    "XLB":  ("Materials",         "Chemicals, mining, metals"),
    "XLRE": ("Real Estate",       "REITs, property management"),
}

# Sectors historically favored in each Dalio quadrant
QUADRANT_ALIGNMENT: dict[str, list[str]] = {
    "Goldilocks":  ["XLK", "XLC", "XLY", "XLI", "XLF"],
    "Reflation":   ["XLE", "XLB", "XLF", "XLI", "XLY"],
    "Stagflation": ["XLE", "XLB", "XLU", "XLP", "XLV"],
    "Deflation":   ["XLU", "XLV", "XLP", "XLRE"],
}

# Map yfinance sector strings → SPDR ETF ticker
_YFINANCE_SECTOR_MAP: dict[str, str] = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",
    "Energy":                 "XLE",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Utilities":              "XLU",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
}


@st.cache_data(ttl=3600)
def get_sector_momentum() -> list[dict]:
    """Download 6-month weekly closes for all 11 SPDR sector ETFs.

    Returns list of dicts sorted by 4W momentum descending. Each dict has:
        ticker, name, desc, price, ret_4w, ret_12w, ret_26w, rank_4w, rank_12w
    """
    tickers = list(SECTOR_ETFS.keys())
    try:
        raw = yf.download(tickers, period="6mo", interval="1wk",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return []
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        results = []
        for t in tickers:
            if t not in close.columns:
                continue
            s = close[t].dropna()
            if len(s) < 5:
                continue
            price   = float(s.iloc[-1])
            ret_4w  = (price / float(s.iloc[-5])  - 1) * 100 if len(s) >= 5  else None
            ret_12w = (price / float(s.iloc[-13]) - 1) * 100 if len(s) >= 13 else None
            ret_26w = (price / float(s.iloc[0])   - 1) * 100
            name, desc = SECTOR_ETFS[t]
            results.append({
                "ticker": t,
                "name":   name,
                "desc":   desc,
                "price":  round(price, 2),
                "ret_4w":  round(ret_4w,  2) if ret_4w  is not None else None,
                "ret_12w": round(ret_12w, 2) if ret_12w is not None else None,
                "ret_26w": round(ret_26w, 2),
            })
        # Add momentum ranks (1 = strongest)
        v4  = sorted([r for r in results if r["ret_4w"]  is not None], key=lambda x: x["ret_4w"],  reverse=True)
        v12 = sorted([r for r in results if r["ret_12w"] is not None], key=lambda x: x["ret_12w"], reverse=True)
        r4m  = {r["ticker"]: i + 1 for i, r in enumerate(v4)}
        r12m = {r["ticker"]: i + 1 for i, r in enumerate(v12)}
        for r in results:
            r["rank_4w"]  = r4m.get(r["ticker"])
            r["rank_12w"] = r12m.get(r["ticker"])
        results.sort(key=lambda x: (x["ret_4w"] or -999), reverse=True)
        return results
    except Exception:
        return []


def get_top_sectors(quadrant: str, sectors: list[dict] | None = None, n: int = 3) -> list[dict]:
    """Top N sectors confirmed by BOTH positive 4W momentum AND quadrant alignment."""
    if sectors is None:
        sectors = get_sector_momentum()
    aligned = set(QUADRANT_ALIGNMENT.get(quadrant, []))
    confirmed = [s for s in sectors if s["ticker"] in aligned and (s.get("ret_4w") or 0) > 0]
    return confirmed[:n]


def get_sector_context_str(ticker_sector: str, quadrant: str) -> str:
    """Brief sector rotation context for injection into AI prompts.

    Args:
        ticker_sector: The yfinance sector string for the ticker being analyzed
        quadrant: Current Dalio quadrant (Goldilocks / Reflation / Stagflation / Deflation)
    Returns:
        Single-line context string, empty string if data unavailable.
    """
    sectors = get_sector_momentum()
    if not sectors:
        return ""

    aligned = set(QUADRANT_ALIGNMENT.get(quadrant, []))
    top3    = [s for s in sectors[:3]  if s.get("ret_4w") is not None]
    bottom3 = [s for s in sectors[-3:] if s.get("ret_4w") is not None]

    top_str = ", ".join(f"{s['ticker']} {s['name']} ({s['ret_4w']:+.1f}% 4W)" for s in top3)
    bot_str = ", ".join(f"{s['ticker']} {s['name']} ({s['ret_4w']:+.1f}% 4W)" for s in bottom3)

    alignment_note = ""
    if ticker_sector:
        etf = _YFINANCE_SECTOR_MAP.get(ticker_sector)
        if etf:
            etf_data = next((s for s in sectors if s["ticker"] == etf), None)
            if etf_data:
                rank = etf_data.get("rank_4w", "?")
                ret  = etf_data.get("ret_4w")
                ret_str = f"{ret:+.1f}%" if ret is not None else "N/A"
                is_aligned = etf in aligned
                alignment_note = (
                    f" | Ticker sector {etf} ({etf_data['name']}) ranks #{rank} (4W: {ret_str})"
                    + (" — REGIME ALIGNED ✓" if is_aligned else " — not favored by current quadrant")
                )

    return (
        f"Sector Rotation ({quadrant}): "
        f"Leaders: {top_str} | Laggards: {bot_str}"
        + alignment_note
    )
